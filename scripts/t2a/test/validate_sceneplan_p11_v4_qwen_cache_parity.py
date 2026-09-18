#!/usr/bin/env python3
"""Audit P11's Qwen3.5 training/full-sequence vs cached-decode parity.

P11 trains every discrete SceneSketch/DeltaSketch in one dense causal forward,
but deployment decodes it with a prefill cache followed by one-token recurrent
updates.  Qwen3.5 combines full attention with GatedDeltaNet, whose chunk and
recurrent implementations are distinct kernels.  This diagnostic forces the
same target prefix through both paths and compares only grammar-legal logits.
It does not score model quality or promote a checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.evaluate_sceneplan_p11_v4_challenge import (  # noqa: E402
    DEFAULT_CHALLENGE,
    DEFAULT_DATASET,
    MODEL_CONFIGS,
    _dataset,
    _json_sha256,
    _load_model,
    _prepare_inputs,
    _runtime_source_provenance,
    _select_rows,
    _sha256_file,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    P11Task,
    normalize_p11_task,
)


DEFAULT_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/sceneplan_p11/"
    "qwen_cache_run/"
    "checkpoints/screen300.ckpt"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/"
    "p11_v4_qwen35_full_vs_cache_parity_20260901.json"
)


def _ids(value: Any) -> torch.Tensor:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    return torch.as_tensor(value, dtype=torch.long).flatten()


def _context_and_target(
    planner: Any, metadata: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, P11Task, dict[str, Any]]:
    values = _prepare_inputs(metadata)
    task = normalize_p11_task(values["task"])
    input_plan = (
        None
        if values["input_sceneplan"] is None
        else planner.plan_codec.encode(values["input_sceneplan"])
    )
    device = planner.plan_embedding.weight.device
    qwen_dtype = planner._ensure_qwen_device(device)
    context, context_metrics = planner._v4_context_embeddings(
        task=task,
        prompt=values["prompt"],
        input_foa=values["input_foa"],
        input_valid_mask=values["input_valid_mask"],
        input_semantic=values["input_semantic"],
        input_plan=input_plan,
        input_lexical=values["input_lexical"],
        device=device,
        qwen_dtype=qwen_dtype,
    )
    target = _ids(metadata["p11_target_tokens"]).to(device)
    return context, target, task, context_metrics


def _cached_forced_logits(
    planner: Any,
    context: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    device = context.device
    qwen_dtype = planner.qwen_backbone.embed_tokens.weight.dtype
    prefix = torch.cat(
        [
            context,
            planner.discrete_boundary[0:1].to(qwen_dtype),
            planner.output_start.view(1, -1).to(qwen_dtype),
        ],
        dim=0,
    ).unsqueeze(0)
    attention = torch.ones((1, prefix.shape[1]), device=device, dtype=torch.bool)
    output = planner._run_backbone(prefix, attention, use_cache=True)
    cache = output.past_key_values
    hidden = [output.last_hidden_state[0, -1].float()]
    for token in target[:-1]:
        embedding = planner.plan_embedding(token.view(1, 1)).to(qwen_dtype)
        attention = torch.ones(
            (1, attention.shape[1] + 1), device=device, dtype=torch.bool
        )
        output = planner._run_backbone(
            embedding,
            attention,
            use_cache=True,
            past_key_values=cache,
        )
        cache = output.past_key_values
        hidden.append(output.last_hidden_state[0, -1].float())
    return F.linear(
        torch.stack(hidden),
        planner.plan_embedding.weight.float(),
        planner.output_bias,
    )


def _full_teacher_logits(
    planner: Any,
    context: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the dense teacher-forced hidden positions used in training."""

    qwen_dtype = planner.qwen_backbone.embed_tokens.weight.dtype
    prefix = torch.cat(
        [context, planner.discrete_boundary[0:1].to(qwen_dtype)], dim=0
    )
    row = torch.cat(
        [
            prefix,
            planner.output_start.view(1, -1).to(qwen_dtype),
            planner.plan_embedding(target[:-1]).to(qwen_dtype),
        ],
        dim=0,
    ).unsqueeze(0)
    attention = torch.ones(
        (1, row.shape[1]), device=row.device, dtype=torch.bool
    )
    output = planner._run_backbone(row, attention, use_cache=False)
    start = int(prefix.shape[0])
    hidden = output.last_hidden_state[
        0, start : start + target.numel()
    ].float()
    return F.linear(
        hidden,
        planner.plan_embedding.weight.float(),
        planner.output_bias,
    )


def _prefix_recompute_forced_logits(
    planner: Any,
    context: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Decode every forced position through the training-style no-cache graph."""

    qwen_dtype = planner.qwen_backbone.embed_tokens.weight.dtype
    base = torch.cat(
        [
            context,
            planner.discrete_boundary[0:1].to(qwen_dtype),
            planner.output_start.view(1, -1).to(qwen_dtype),
        ],
        dim=0,
    )
    hidden = []
    for position in range(target.numel()):
        row = base
        if position:
            row = torch.cat(
                [
                    base,
                    planner.plan_embedding(target[:position]).to(qwen_dtype),
                ],
                dim=0,
            )
        attention = torch.ones(
            (1, row.shape[0]), device=row.device, dtype=torch.bool
        )
        output = planner._run_backbone(
            row.unsqueeze(0), attention, use_cache=False
        )
        hidden.append(output.last_hidden_state[0, -1].float())
    return F.linear(
        torch.stack(hidden),
        planner.plan_embedding.weight.float(),
        planner.output_bias,
    )


def _legal_ids(
    planner: Any,
    task: P11Task,
    target: torch.Tensor,
    position: int,
    input_sceneplan: Mapping[str, Any] | None,
) -> list[int]:
    prefix = [int(value) for value in target[:position].tolist()]
    if task is P11Task.EDITING:
        if input_sceneplan is None:
            raise ValueError("editing parity row lacks input ScenePlan")
        return sorted(
            planner.delta_sketch_codec.allowed_next_ids(
                prefix, input_sceneplan=input_sceneplan
            )
        )
    return sorted(planner.scene_sketch_codec.allowed_next_ids(prefix))


def _compare_one(
    planner: Any,
    metadata: Mapping[str, Any],
    *,
    decode_path: str,
) -> dict[str, Any]:
    context, target, task, context_metrics = _context_and_target(planner, metadata)
    full = _full_teacher_logits(planner, context, target).float()
    if decode_path == "cached":
        decoded = _cached_forced_logits(planner, context, target).float()
    elif decode_path == "prefix_recompute":
        decoded = _prefix_recompute_forced_logits(
            planner, context, target
        ).float()
    else:
        raise ValueError(f"unsupported decode path: {decode_path}")
    if full.shape != decoded.shape or full.shape[0] != target.numel():
        raise RuntimeError(
            f"full/cache parity shape mismatch: {tuple(full.shape)} vs "
            f"{tuple(decoded.shape)} target={target.numel()}"
        )

    input_sceneplan = metadata.get("p11_input_sceneplan")
    max_abs = 0.0
    abs_sum = 0.0
    compared = 0
    top1_mismatches = 0
    target_nll_deltas: list[float] = []
    first_mismatch = None
    full_teacher_hits = 0
    decoded_teacher_hits = 0
    for position in range(target.numel()):
        allowed = _legal_ids(
            planner, task, target, position, input_sceneplan
        )
        if not allowed:
            raise RuntimeError(f"empty grammar set at position {position}")
        legal = torch.tensor(allowed, device=full.device, dtype=torch.long)
        full_one = full[position].index_select(0, legal)
        decoded_one = decoded[position].index_select(0, legal)
        difference = (full_one - decoded_one).abs()
        max_abs = max(max_abs, float(difference.max()))
        abs_sum += float(difference.sum())
        compared += int(difference.numel())
        full_index = int(full_one.argmax())
        cache_index = int(decoded_one.argmax())
        full_token = allowed[full_index]
        cache_token = allowed[cache_index]
        teacher_token = int(target[position])
        full_teacher_hits += int(full_token == teacher_token)
        decoded_teacher_hits += int(cache_token == teacher_token)
        if full_token != cache_token:
            top1_mismatches += 1
            if first_mismatch is None:
                first_mismatch = {
                    "position": position,
                    "teacher_token": teacher_token,
                    "full_top1": full_token,
                    "decoded_top1": cache_token,
                    "full_top1_logit": float(full_one[full_index]),
                    "decoded_top1_logit": float(decoded_one[cache_index]),
                }
        teacher_index = allowed.index(teacher_token)
        full_nll = -F.log_softmax(full_one, dim=0)[teacher_index]
        cache_nll = -F.log_softmax(decoded_one, dim=0)[teacher_index]
        target_nll_deltas.append(float((full_nll - cache_nll).abs()))

    finite = bool(torch.isfinite(full).all() and torch.isfinite(decoded).all())
    return {
        "challenge_id": metadata["p11_challenge_id"],
        "task": task.value,
        "view_id": metadata["p11_prompt_view_id"],
        "decode_path": decode_path,
        "context_tokens": int(context.shape[0]),
        "target_tokens": int(target.numel()),
        "context_metrics": context_metrics,
        "finite": finite,
        "grammar_legal_logits_compared": compared,
        "legal_logit_max_abs": max_abs,
        "legal_logit_mean_abs": abs_sum / max(compared, 1),
        "target_nll_abs_delta_max": max(target_nll_deltas),
        "target_nll_abs_delta_mean": sum(target_nll_deltas)
        / len(target_nll_deltas),
        "top1_mismatch_count": top1_mismatches,
        "top1_match_rate": 1.0 - top1_mismatches / target.numel(),
        "full_teacher_top1_accuracy": full_teacher_hits / target.numel(),
        "decoded_teacher_top1_accuracy": decoded_teacher_hits / target.numel(),
        "first_top1_mismatch": first_mismatch,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tokens = sum(int(row["target_tokens"]) for row in rows)
    mismatches = sum(int(row["top1_mismatch_count"]) for row in rows)
    maximum_nll = max(float(row["target_nll_abs_delta_max"]) for row in rows)
    decision_gate = all(bool(row["finite"]) for row in rows) and mismatches == 0
    numerical_gate = maximum_nll <= 0.05
    return {
        "rows": len(rows),
        "target_tokens": tokens,
        "top1_mismatch_count": mismatches,
        "top1_match_rate": 1.0 - mismatches / max(tokens, 1),
        "legal_logit_max_abs": max(
            float(row["legal_logit_max_abs"]) for row in rows
        ),
        "target_nll_abs_delta_max": maximum_nll,
        "decision_parity_gate": "PASS" if decision_gate else "FAIL",
        "decision_gate_contract": "all finite; zero grammar-legal top1 mismatches",
        "numerical_parity_gate": "PASS" if numerical_gate else "FAIL",
        "numerical_gate_contract": "max teacher-token NLL delta <= 0.05",
        "strict_parity_gate": (
            "PASS" if decision_gate and numerical_gate else "FAIL"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIGS["flow"])
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("online", "ema"), default="online")
    parser.add_argument("--modes", default="fast,reference")
    parser.add_argument(
        "--decode-path", choices=("cached", "prefix_recompute"), default="cached"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not modes or any(value not in {"fast", "reference"} for value in modes):
        raise ValueError("--modes must contain fast and/or reference")
    torch.manual_seed(20260901)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    model_config = args.model_config.expanduser().resolve(strict=True)
    dataset_config = args.dataset_config.expanduser().resolve(strict=True)
    challenge_path = args.challenge.expanduser().resolve(strict=True)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    selected_ordinals = _select_rows(
        challenge_path, rows_per_view=1, families=None
    )
    results: dict[str, Any] = {}
    provenance = None
    selected_ids = None
    for mode in modes:
        wrapper, planner, one_provenance = _load_model(
            model_config_path=model_config,
            checkpoint_path=checkpoint,
            device=device,
            weights=args.weights,
        )
        if provenance is None:
            provenance = one_provenance
        challenge = _dataset(
            planner=planner,
            dataset_config_path=dataset_config,
            challenge_path=challenge_path,
        )
        materialized = []
        seen_tasks: set[str] = set()
        for ordinal in selected_ordinals:
            _, metadata = challenge[ordinal]
            task = str(metadata["p11_task"])
            if (
                metadata["p11_challenge_family"] != "exact_compatibility"
                or task in seen_tasks
            ):
                continue
            seen_tasks.add(task)
            materialized.append((ordinal, metadata))
        del challenge
        if seen_tasks != {"generation", "understanding", "editing"}:
            raise RuntimeError(f"did not find exact G/U/E rows: {sorted(seen_tasks)}")
        if selected_ids is None:
            selected_ids = [metadata["p11_challenge_id"] for _, metadata in materialized]
        reference_configuration = None
        if mode == "reference":
            reference_configuration = planner.configure_qwen_runtime_kernels(
                "torch_reference"
            )
        started = time.perf_counter()
        rows = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _, metadata in materialized:
                rows.append(
                    _compare_one(
                        planner, metadata, decode_path=args.decode_path
                    )
                )
        results[mode] = {
            "reference_configuration": reference_configuration,
            "elapsed_seconds": time.perf_counter() - started,
            "aggregate": _aggregate(rows),
            "rows": rows,
        }
        del planner, wrapper
        gc.collect()
        torch.cuda.empty_cache()

    passing = [
        mode
        for mode, value in results.items()
        if value["aggregate"]["decision_parity_gate"] == "PASS"
    ]
    report = {
        "schema": "stable_audio_tools.p11_v4_qwen35_cache_parity",
        "schema_version": 1,
        "status": "PASS" if passing else "FAIL",
        "scope": (
            "teacher-forced discrete-path diagnostic only; no checkpoint quality "
            "or P11 idea-validation claim"
        ),
        "model_config": str(model_config),
        "model_config_sha256": _sha256_file(model_config),
        "dataset_config": str(dataset_config),
        "challenge": str(challenge_path),
        "challenge_sha256": _sha256_file(challenge_path),
        "checkpoint": provenance,
        "weights": args.weights,
        "decode_path": args.decode_path,
        "selected_challenge_ids": selected_ids,
        "runtime_source_provenance": _runtime_source_provenance(device),
        "modes": results,
        "passing_modes": passing,
        "canonical_evaluator_mode": (
            f"{args.decode_path}:fast"
            if "fast" in passing
            else f"{args.decode_path}:reference"
            if passing
            else None
        ),
        "external_issue_context": {
            "transformers_qwen35_cache_issue": (
                "https://github.com/huggingface/transformers/issues/46190"
            ),
            "claim": (
                "external issue motivated this local audit; it is not accepted "
                "as evidence about this checkpoint or RTX 4090 runtime"
            ),
        },
    }
    report["report_sha256_without_self"] = _json_sha256(report)
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
