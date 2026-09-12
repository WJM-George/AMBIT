#!/usr/bin/env python3
"""Causal intervention gate for the canonical P11-v4 planner.

This evaluator follows the current authority boundary instead of the retired
Flow-E endpoint classifier:

* G/U: SceneSketch owns semantics and continuous ExecutionState owns P10
  duration/activity/trajectory controls;
* E rotate/distance: DeltaSketch owns the categorical endpoint;
* E retime: owner-local continuous DeltaThought owns the interval delta.

Every intervention reuses the same explicit discrete and continuous seeds.
It must preserve discrete/semantic authority while changing the corresponding
numeric P10 controls.  This is a mechanistic dependency gate, not a claim of
held-out quality superiority or rendered-FOA closure.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.evaluate_sceneplan_p11_v4_challenge import (  # noqa: E402
    CANONICAL_QWEN_KERNEL_MODE,
    DEFAULT_CHALLENGE,
    DEFAULT_DATASET,
    MODEL_CONFIGS,
    _dataset,
    _ids,
    _json_sha256,
    _load_model,
    _prepare_inputs,
    _resolved_config_provenance,
    _rng_equal,
    _rng_state,
    _row_seed_key,
    _runtime_source_provenance,
    _score_draw,
    _seed,
    _select_rows,
    _semantic_signature,
    _sha256_file,
)


DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/evals/"
    "p11_v4_flow_r1_causal_interventions_20260902.json"
)
GU_INTERVENTIONS = (
    "zero_thought",
    "shuffle_source_slots",
    "swap_source_1_2",
    "replace_one_control_field",
)


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _retime_ordinal(challenge_path: Path) -> int:
    connection = sqlite3.connect(
        f"file:{challenge_path}?mode=ro&immutable=1", uri=True
    )
    rows = connection.execute(
        "SELECT ordinal,edit_spec_json FROM rows "
        "WHERE task='editing' AND edit_spec_json IS NOT NULL ORDER BY ordinal"
    ).fetchall()
    connection.close()
    for ordinal, raw_spec in rows:
        if str(json.loads(raw_spec).get("operation") or "") == "retime_source":
            return int(ordinal)
    raise RuntimeError("immutable challenge has no retime row")


def _decode_once(
    planner: Any,
    metadata: Mapping[str, Any],
    *,
    arm: str,
    root_seed: int,
    discrete_decode_mode: str,
    intervention: str | None,
) -> tuple[dict[str, Any], bool]:
    inputs = _prepare_inputs(metadata)
    row_key = _row_seed_key(metadata)
    discrete_seed = _seed(root_seed, row_key, 0, "discrete")
    device = planner.plan_embedding.weight.device
    before = _rng_state(device)
    if arm == "flow" and inputs["task"] != "editing":
        output = planner.decode_transfusion_cot_samples(
            inputs["prompt"],
            task=inputs["task"],
            noise_seeds=[_seed(root_seed, row_key, 0, "continuous")],
            input_foa=inputs["input_foa"],
            input_valid_mask=inputs["input_valid_mask"],
            input_semantic=inputs["input_semantic"],
            input_lexical=inputs["input_lexical"],
            input_sceneplan=inputs["input_sceneplan"],
            duration_sec=inputs["duration_sec"],
            sample_id=inputs["sample_id"],
            temperature=0.0,
            discrete_seed=discrete_seed,
            discrete_decode_mode=discrete_decode_mode,
            scene_thought_intervention=intervention,
        )[0]
    else:
        output = planner.decode_transfusion_cot(
            inputs["prompt"],
            task=inputs["task"],
            input_foa=inputs["input_foa"],
            input_valid_mask=inputs["input_valid_mask"],
            input_semantic=inputs["input_semantic"],
            input_lexical=inputs["input_lexical"],
            input_sceneplan=inputs["input_sceneplan"],
            duration_sec=inputs["duration_sec"],
            sample_id=inputs["sample_id"],
            temperature=0.0,
            discrete_seed=discrete_seed,
            discrete_decode_mode=discrete_decode_mode,
            scene_thought_intervention=intervention,
        )
    after = _rng_state(device)
    return output, _rng_equal(before, after)


def _interventions_for(
    task: str, expected_operation: str | None, decoded_operation: str | None
) -> tuple[str, ...]:
    if task in {"generation", "understanding"}:
        return GU_INTERVENTIONS
    if expected_operation != decoded_operation:
        return ()
    if decoded_operation in {"rotate_source", "distance_source"}:
        return ("flip_control_direction",)
    if decoded_operation == "retime_source":
        return ("zero_thought", "flip_retime_delta")
    return ()


def _core(value: Mapping[str, Any]) -> torch.Tensor:
    from stable_audio_tools.data.scene_sketch_v1 import execution_state_core

    return torch.from_numpy(execution_state_core(value["execution_state"])).float()


def _changed_fields(
    baseline: torch.Tensor, changed: torch.Tensor
) -> tuple[list[str], dict[str, list[str]]]:
    from stable_audio_tools.data.scene_sketch_v1 import (
        EXECUTION_FEATURE_NAMES,
        EXECUTION_SLOT_ROLES,
    )

    mask = (changed - baseline).abs().gt(1.0e-6)
    slots = []
    fields: dict[str, list[str]] = {}
    for slot_index, slot_name in enumerate(EXECUTION_SLOT_ROLES):
        indices = torch.nonzero(mask[slot_index], as_tuple=False).flatten().tolist()
        if not indices:
            continue
        slots.append(slot_name)
        fields[slot_name] = [EXECUTION_FEATURE_NAMES[index] for index in indices]
    return slots, fields


def _public_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in score.items() if key != "core"}


def _compare(
    *,
    planner: Any,
    metadata: Mapping[str, Any],
    baseline: Mapping[str, Any],
    changed: Mapping[str, Any],
    baseline_score: Mapping[str, Any],
    changed_score: Mapping[str, Any],
    intervention: str,
    rng_isolation_pass: bool,
) -> dict[str, Any]:
    baseline_core = _core(baseline)
    changed_core = _core(changed)
    changed_slots, changed_fields = _changed_fields(baseline_core, changed_core)
    task = str(metadata["p11_task"])
    expected = metadata.get("p11_edit_spec") or {}
    owner = str(expected.get("source_id") or "")
    baseline_patch = baseline.get("patch")
    changed_patch = changed.get("patch")
    baseline_direction = baseline.get("diagnostics", {}).get("control_direction")
    changed_direction = changed.get("diagnostics", {}).get("control_direction")
    semantic_unchanged = (
        _semantic_signature(baseline["sceneplan"])
        == _semantic_signature(changed["sceneplan"])
    )
    p10_semantic_unchanged = (
        baseline["p10_conditions"]["semantic_caption"]
        == changed["p10_conditions"]["semantic_caption"]
    )
    discrete_unchanged = torch.equal(
        _ids(baseline["discrete_tokens"]), _ids(changed["discrete_tokens"])
    )
    numeric_changed = bool(changed_slots)
    patch_changed = baseline_patch != changed_patch
    owner_local = (
        None
        if not owner
        else all(slot == owner for slot in changed_slots)
    )
    direction_flip_exact = (
        None
        if intervention != "flip_control_direction"
        else baseline_direction in {-1, 1}
        and changed_direction == -int(baseline_direction)
    )
    if task in {"generation", "understanding"}:
        required_numeric_response = intervention == "zero_thought"
    elif intervention in {"flip_control_direction", "flip_retime_delta"}:
        required_numeric_response = True
    else:
        required_numeric_response = False
    required_pass = (
        bool(numeric_changed)
        if required_numeric_response
        else True
    )
    if intervention in {"flip_control_direction", "flip_retime_delta"}:
        required_pass = bool(required_pass and patch_changed and owner_local)
    if intervention == "flip_control_direction":
        required_pass = bool(required_pass and direction_flip_exact)
    thought_delta = (
        torch.as_tensor(changed["thought_core"]).float()
        - torch.as_tensor(baseline["thought_core"]).float()
    ).abs()
    return {
        "intervention": intervention,
        "rng_isolation_pass": rng_isolation_pass,
        "discrete_tokens_unchanged": bool(discrete_unchanged),
        "semantic_plan_unchanged": bool(semantic_unchanged),
        "p10_semantic_caption_unchanged": bool(p10_semantic_unchanged),
        "numeric_p10_changed": numeric_changed,
        "changed_slots": changed_slots,
        "changed_fields": changed_fields,
        "owner_local_numeric_change": owner_local,
        "patch_changed": patch_changed,
        "control_direction_before": baseline_direction,
        "control_direction_after": changed_direction,
        "control_direction_flip_exact": direction_flip_exact,
        "thought_mean_abs_delta": float(thought_delta.mean()),
        "thought_max_abs_delta": float(thought_delta.max()),
        "task_score_before": float(baseline_score["task_score"]),
        "task_score_after": float(changed_score["task_score"]),
        "task_score_delta": float(
            changed_score["task_score"] - baseline_score["task_score"]
        ),
        "baseline_patch": baseline_patch,
        "intervened_patch": changed_patch,
        "required_numeric_response": required_numeric_response,
        "required_causal_response_pass": required_pass,
        "valid": bool(
            changed_score.get("valid")
            and changed_score.get("roundtrip")
            and changed_score.get("finite")
        ),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for value in row.get("interventions", []):
            grouped[(str(row["task"]), str(value["intervention"]))].append(value)
    output: dict[str, Any] = {}
    for (task, name), values in sorted(grouped.items()):
        output[f"{task}:{name}"] = {
            "rows": len(values),
            "valid_rate": _mean([float(bool(value["valid"])) for value in values]),
            "rng_isolation_rate": _mean(
                [float(bool(value["rng_isolation_pass"])) for value in values]
            ),
            "discrete_authority_preservation_rate": _mean(
                [float(bool(value["discrete_tokens_unchanged"])) for value in values]
            ),
            "semantic_plan_preservation_rate": _mean(
                [float(bool(value["semantic_plan_unchanged"])) for value in values]
            ),
            "p10_semantic_preservation_rate": _mean(
                [
                    float(bool(value["p10_semantic_caption_unchanged"]))
                    for value in values
                ]
            ),
            "numeric_response_rate": _mean(
                [float(bool(value["numeric_p10_changed"])) for value in values]
            ),
            "required_causal_response_rate": _mean(
                [
                    float(bool(value["required_causal_response_pass"]))
                    for value in values
                    if value["required_numeric_response"]
                ]
            ),
            "task_score_delta_mean": _mean(
                [float(value["task_score_delta"]) for value in values]
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("flow", "direct"), default="flow")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("online", "ema"), default="ema")
    parser.add_argument("--rows-per-view", type=int, default=1)
    parser.add_argument("--families", default="")
    parser.add_argument(
        "--include-retime",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--discrete-decode-mode",
        choices=("cached", "prefix_recompute"),
        default="prefix_recompute",
    )
    parser.add_argument(
        "--qwen-kernel-mode",
        choices=(
            "fast",
            "fast_pinned_warps2",
            "fast_fixed_bv32_w2_s2",
            "torch_reference",
        ),
        default=CANONICAL_QWEN_KERNEL_MODE,
        help=(
            "Scientific causal evaluation uses torch_reference; FLA modes are "
            "retained only for diagnostic throughput probes."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.rows_per_view <= 0:
        raise ValueError("--rows-per-view must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    model_config_path = (
        MODEL_CONFIGS[args.arm]
        if args.model_config is None
        else args.model_config
    ).expanduser().resolve(strict=True)
    dataset_config_path = args.dataset_config.expanduser().resolve(strict=True)
    challenge_path = args.challenge.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    families = {
        value.strip() for value in args.families.split(",") if value.strip()
    } or None
    selected = _select_rows(
        challenge_path,
        rows_per_view=args.rows_per_view,
        families=families,
    )
    if args.include_retime and (
        families is None or "exact_compatibility" in families
    ):
        selected = sorted(set(selected + [_retime_ordinal(challenge_path)]))

    wrapper, planner, checkpoint_provenance = _load_model(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        weights=args.weights,
    )
    checkpoint_provenance["qwen_runtime_kernels"] = (
        planner.configure_qwen_runtime_kernels(args.qwen_kernel_mode)
    )
    challenge = _dataset(
        planner=planner,
        dataset_config_path=dataset_config_path,
        challenge_path=challenge_path,
    )
    materialized = []
    for ordinal in selected:
        _, metadata = challenge[ordinal]
        materialized.append((ordinal, metadata))
    del challenge

    started = time.perf_counter()
    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for ordinal, metadata in materialized:
        task = str(metadata["p11_task"])
        row: dict[str, Any] = {
            "ordinal": ordinal,
            "challenge_id": metadata["p11_challenge_id"],
            "task": task,
            "family": metadata["p11_challenge_family"],
            "view_id": metadata["p11_prompt_view_id"],
            "pair_id": metadata.get("p11_challenge_pair_id"),
            "expected_operation": (
                None
                if task != "editing"
                else str(metadata["p11_edit_spec"]["operation"])
            ),
            "interventions": [],
        }
        try:
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with torch.inference_mode(), autocast:
                baseline, baseline_rng = _decode_once(
                    planner,
                    metadata,
                    arm=args.arm,
                    root_seed=args.seed,
                    discrete_decode_mode=args.discrete_decode_mode,
                    intervention=None,
                )
            baseline_score = _score_draw(planner, metadata, baseline)
            decoded_operation = (
                None
                if task != "editing"
                else str(baseline["discrete_object"]["operation"])
            )
            row.update(
                {
                    "decoded_operation": decoded_operation,
                    "operation_matches_expected": (
                        None
                        if task != "editing"
                        else decoded_operation == row["expected_operation"]
                    ),
                    "baseline_rng_isolation_pass": baseline_rng,
                    "baseline": _public_score(baseline_score),
                    "baseline_valid": bool(
                        baseline_score.get("valid")
                        and baseline_score.get("roundtrip")
                        and baseline_score.get("finite")
                    ),
                }
            )
            interventions = _interventions_for(
                task, row["expected_operation"], decoded_operation
            )
            for name in interventions:
                with torch.inference_mode(), autocast:
                    changed, changed_rng = _decode_once(
                        planner,
                        metadata,
                        arm=args.arm,
                        root_seed=args.seed,
                        discrete_decode_mode=args.discrete_decode_mode,
                        intervention=name,
                    )
                changed_score = _score_draw(planner, metadata, changed)
                row["interventions"].append(
                    _compare(
                        planner=planner,
                        metadata=metadata,
                        baseline=baseline,
                        changed=changed,
                        baseline_score=baseline_score,
                        changed_score=changed_score,
                        intervention=name,
                        rng_isolation_pass=changed_rng,
                    )
                )
        except Exception as error:  # noqa: BLE001 - fail-closed row record.
            row["error"] = f"{type(error).__name__}: {error}"
            row["baseline_valid"] = False
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

    elapsed = time.perf_counter() - started
    interventions = [
        value for row in rows for value in row.get("interventions", [])
    ]
    required = [
        value for value in interventions if value["required_numeric_response"]
    ]
    evaluator_integrity = bool(rows) and all(
        row.get("baseline_valid") and row.get("baseline_rng_isolation_pass")
        for row in rows
    ) and all(value["valid"] and value["rng_isolation_pass"] for value in interventions)
    authority_isolation = bool(interventions) and all(
        value["discrete_tokens_unchanged"]
        and value["semantic_plan_unchanged"]
        and value["p10_semantic_caption_unchanged"]
        for value in interventions
    )
    operation_integrity = all(
        row.get("operation_matches_expected") is not False for row in rows
    )
    causal_response = bool(required) and all(
        value["required_causal_response_pass"] for value in required
    )
    status = (
        "PASS"
        if evaluator_integrity
        and authority_isolation
        and operation_integrity
        and causal_response
        else "FAIL"
    )
    runtime_provenance = _runtime_source_provenance(device)
    runtime_provenance["source_files"]["causal_intervention_evaluator"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256_file(Path(__file__).resolve()),
    }
    report = {
        "schema": "stable_audio_tools.p11_v4_causal_intervention_eval",
        "schema_version": 3,
        "evaluator_contract": (
            "authority_aligned_gu_flow_e_hybrid_torch_reference_intervention_v3"
            if args.qwen_kernel_mode == CANONICAL_QWEN_KERNEL_MODE
            else "authority_aligned_gu_flow_e_fast_kernel_diagnostic_v3"
        ),
        "status": status,
        "scope": (
            "mechanistic held-out ScenePlan/P10-conditioning dependency gate; "
            "no rendered FOA and no held-out superiority claim"
        ),
        "arm": args.arm,
        "checkpoint": checkpoint_provenance,
        "config_provenance": {
            "model": _resolved_config_provenance(model_config_path),
            "dataset": _resolved_config_provenance(dataset_config_path),
        },
        "challenge": str(challenge_path),
        "challenge_sha256": _sha256_file(challenge_path),
        "runtime_source_provenance": runtime_provenance,
        "weights": args.weights,
        "root_seed": args.seed,
        "discrete_decode_mode": args.discrete_decode_mode,
        "qwen_kernel_mode": args.qwen_kernel_mode,
        "rows_per_view": args.rows_per_view,
        "include_retime": args.include_retime,
        "selected_ordinals": selected,
        "rows": len(rows),
        "intervention_decodes": len(interventions),
        "elapsed_seconds": elapsed,
        "decodes_per_second": (len(rows) + len(interventions)) / elapsed,
        "peak_memory_gib": (
            torch.cuda.max_memory_allocated(device) / (2**30)
            if device.type == "cuda"
            else None
        ),
        "gates": {
            "evaluator_integrity": evaluator_integrity,
            "discrete_and_semantic_authority_isolation": authority_isolation,
            "editing_operation_matches_expected": operation_integrity,
            "required_numeric_causal_response": causal_response,
            "required_interventions": len(required),
        },
        "aggregate": _aggregate(rows),
        "claim_boundary": {
            "mechanistic_dependency_established_on_panel": status == "PASS",
            "quality_superiority_established": False,
            "frozen_p10_audio_closure_established": False,
            "eight_gpu_or_full_corpus_authorized": False,
        },
        "rows_detail": rows,
    }
    report["report_sha256_without_self"] = _json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    del planner, wrapper
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
