#!/usr/bin/env python3
"""CPU contract gate for P11-v4 Flow-R1 posterior-noise inference.

This is a structural test, not a quality benchmark.  It verifies that the
Flow arm exposes reproducible per-sample draws without consuming global RNG,
that explicit tensors reproduce the same solve, and that Direct-MSE cannot be
misrepresented as a stochastic posterior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.scene_sketch_v1 import (  # noqa: E402
    EXECUTION_FEATURE_DIM,
    EXECUTION_SLOT_COUNT,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import P11Task  # noqa: E402
from stable_audio_tools.models.scene_thought_p11_v4 import (  # noqa: E402
    P11_V4_DIRECT_MSE_ARM,
    P11_V4_FLOW_ARM,
    P11_V4_THOUGHT_CONTRACT,
    SketchFirstExecutionReasoner,
)


DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/p11_v4_noise_api_contract_20260901.json"
)


def _config(arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "contract": P11_V4_THOUGHT_CONTRACT,
        "backbone": "shared_qwen_causal_v1",
        "continuous_objective": (
            "rectified_flow" if arm == P11_V4_FLOW_ARM else "direct_mse"
        ),
        "editing_continuous_objective": "direct_mse",
        "editing_output_head": "dedicated_delta_v1",
        "slot_count": EXECUTION_SLOT_COUNT,
        "core_dim": EXECUTION_FEATURE_DIM,
        "dim": 32,
        "inference_steps": 2,
        "training_inference_steps": 2,
        "inference_noise_seed": 41,
    }


def _sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _run_slots(tokens: torch.Tensor) -> torch.Tensor:
    # Deterministic, row-separable stand-in for the shared Qwen graph.
    return torch.tanh(tokens)


def _signature() -> dict[str, Any]:
    torch.manual_seed(20260901)
    model = SketchFirstExecutionReasoner(_config(P11_V4_FLOW_ARM), output_dim=32)
    model.eval()
    task_ids = torch.tensor(
        [
            list(P11Task).index(P11Task.GENERATION),
            list(P11Task).index(P11Task.UNDERSTANDING),
        ],
        dtype=torch.long,
    )
    previous = torch.zeros(2, EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM)
    result = model.infer(
        run_slots=_run_slots,
        task_ids=task_ids,
        previous_core=previous,
        noise_seed=[101, 202],
    )
    return {
        "core": result.core.detach().float().cpu().tolist(),
        "core_sha256": _sha256(result.core),
        "noise_sha256": _sha256(result.inference_noise),
        "noise_seeds": list(result.inference_noise_seeds or ()),
        "noise_source": result.inference_noise_source,
    }


def _expect_error(function, contains: str) -> str:
    try:
        function()
    except Exception as error:  # noqa: BLE001 - contract checks exact failure surface.
        message = f"{type(error).__name__}: {error}"
        if contains not in message:
            raise RuntimeError(
                f"expected error containing {contains!r}, observed {message!r}"
            ) from error
        return message
    raise RuntimeError(f"expected failure containing {contains!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--emit-signature", action="store_true")
    args = parser.parse_args()
    if args.emit_signature:
        print(json.dumps(_signature(), sort_keys=True))
        return

    signature = _signature()
    torch.manual_seed(20260901)
    flow = SketchFirstExecutionReasoner(_config(P11_V4_FLOW_ARM), output_dim=32)
    flow.eval()
    task_ids = torch.tensor(
        [
            list(P11Task).index(P11Task.GENERATION),
            list(P11Task).index(P11Task.UNDERSTANDING),
        ],
        dtype=torch.long,
    )
    previous = torch.zeros(2, EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM)

    rng_before = torch.random.get_rng_state().clone()
    batched = flow.infer(
        run_slots=_run_slots,
        task_ids=task_ids,
        previous_core=previous,
        noise_seed=[101, 202],
    )
    rng_after = torch.random.get_rng_state().clone()
    repeated = flow.infer(
        run_slots=_run_slots,
        task_ids=task_ids,
        previous_core=previous,
        noise_seed=[101, 202],
    )
    changed_seed = flow.infer(
        run_slots=_run_slots,
        task_ids=task_ids,
        previous_core=previous,
        noise_seed=[303, 404],
    )
    tensor_replay = flow.infer(
        run_slots=_run_slots,
        task_ids=task_ids,
        previous_core=previous,
        noise=batched.inference_noise,
    )
    default_draw = flow.infer(
        run_slots=_run_slots,
        task_ids=task_ids[:1],
        previous_core=previous[:1],
    )

    per_sample_max_abs = []
    for index, seed in enumerate((101, 202)):
        single = flow.infer(
            run_slots=_run_slots,
            task_ids=task_ids[index : index + 1],
            previous_core=previous[index : index + 1],
            noise_seed=seed,
        )
        per_sample_max_abs.append(
            float((single.core[0] - batched.core[index]).abs().max())
        )

    direct = SketchFirstExecutionReasoner(
        _config(P11_V4_DIRECT_MSE_ARM), output_dim=32
    ).eval()
    rejected = {
        "seed_and_tensor": _expect_error(
            lambda: flow.infer(
                run_slots=_run_slots,
                task_ids=task_ids[:1],
                noise_seed=1,
                noise=torch.zeros(EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM),
            ),
            "either noise_seed or noise",
        ),
        "wrong_seed_count": _expect_error(
            lambda: flow.infer(
                run_slots=_run_slots,
                task_ids=task_ids,
                noise_seed=[1],
            ),
            "one noise seed per sample",
        ),
        "nonfinite_tensor": _expect_error(
            lambda: flow.infer(
                run_slots=_run_slots,
                task_ids=task_ids[:1],
                noise=torch.full(
                    (EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM), float("nan")
                ),
            ),
            "must be finite",
        ),
        "direct_seed": _expect_error(
            lambda: direct.infer(
                run_slots=_run_slots,
                task_ids=task_ids[:1],
                noise_seed=1,
            ),
            "Direct-MSE has no stochastic flow state",
        ),
    }

    child_command = [sys.executable, str(Path(__file__).resolve()), "--emit-signature"]
    child_signatures = [
        json.loads(
            subprocess.run(
                child_command,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        for _ in range(2)
    ]
    child_cores = [torch.tensor(value["core"], dtype=torch.float32) for value in child_signatures]
    cross_process_max_abs = float((child_cores[0] - child_cores[1]).abs().max())

    checks = {
        "local_rng_does_not_advance_global_rng": torch.equal(rng_before, rng_after),
        "same_seed_noise_exact": torch.equal(
            batched.inference_noise, repeated.inference_noise
        ),
        "same_seed_core_exact": torch.equal(batched.core, repeated.core),
        "different_seed_noise_changes": not torch.equal(
            batched.inference_noise, changed_seed.inference_noise
        ),
        "different_seed_core_changes": not torch.equal(
            batched.core, changed_seed.core
        ),
        "explicit_tensor_replays_core": torch.equal(batched.core, tensor_replay.core),
        "explicit_tensor_marked": tensor_replay.inference_noise_source
        == "explicit_tensor",
        "default_seed_replays_legacy_buffer": torch.equal(
            default_draw.inference_noise[0], flow.inference_noise
        ),
        "batch_isolation_within_tolerance": max(per_sample_max_abs) <= 1.0e-6,
        "cross_process_exact": child_signatures[0]["core_sha256"]
        == child_signatures[1]["core_sha256"],
        "cross_process_within_tolerance": cross_process_max_abs <= 1.0e-7,
        "failure_surfaces_rejected": len(rejected) == 4,
        "all_outputs_finite": all(
            bool(torch.isfinite(value.core).all())
            for value in (batched, repeated, changed_seed, tensor_replay, default_draw)
        ),
    }
    report = {
        "schema": "stable_audio_tools.p11_v4_noise_api_contract",
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "CPU structural gate; no learned-quality or posterior-calibration claim",
        "checks": checks,
        "signature": {key: value for key, value in signature.items() if key != "core"},
        "per_sample_batch_vs_single_max_abs": per_sample_max_abs,
        "cross_process_max_abs": cross_process_max_abs,
        "rejected_invalid_calls": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
