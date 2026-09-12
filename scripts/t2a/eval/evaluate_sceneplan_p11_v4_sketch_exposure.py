#!/usr/bin/env python3
"""Measure the P11-v4 teacher-context versus deployment-context exposure gap.

The normal path decodes its own SceneSketch/DeltaSketch.  The paired diagnostic
path reuses the same checkpoint, evidence, discrete seed, and Flow noise, but
substitutes the exact training target discrete span before continuous thought.
The oracle path is target-access diagnostics only; it is never a deployable
quality metric or a promotion score.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
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
    _active_mask,
    _dataset,
    _ids,
    _integer_tensor_sha256,
    _json_sha256,
    _load_model,
    _mean,
    _prepare_inputs,
    _resolved_config_provenance,
    _rng_equal,
    _rng_state,
    _row_seed_key,
    _runtime_source_provenance,
    _score_draw,
    _seed,
    _select_rows,
    _sha256_file,
)


SCHEMA = "stable_audio_tools.p11_v4_sketch_exposure_eval"
SCHEMA_VERSION = 1


def _target_discrete_tokens(metadata: Mapping[str, Any]) -> torch.Tensor:
    value = metadata["p11_target_tokens"]
    return _ids(value)


def _target_continuous_core(metadata: Mapping[str, Any]) -> torch.Tensor:
    key = (
        "p11_v4_delta_execution_core"
        if str(metadata["p11_task"]) == "editing"
        else "p11_v4_target_execution_core"
    )
    return torch.as_tensor(metadata[key], dtype=torch.float32).cpu()


def _masked_rmse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    prediction = torch.as_tensor(prediction, dtype=torch.float32).detach().cpu()
    target = torch.as_tensor(target, dtype=torch.float32).detach().cpu()
    mask = torch.as_tensor(mask, dtype=torch.bool).cpu()
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("exposure diagnostic core/mask shapes do not align")
    if not bool(mask.any()):
        return 0.0
    return float((prediction[mask] - target[mask]).square().mean().sqrt())


def _public_score(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "core"}


def _summarize_exposure_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task"])].append(row)

    def task_metric_delta(
        values: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, float]]:
        metric_rows = [
            row
            for row in values
            if isinstance(row.get("deployment", {}).get("task_metrics"), Mapping)
            and isinstance(
                row.get("teacher_context", {}).get("task_metrics"), Mapping
            )
        ]
        if not metric_rows:
            return {}
        common: set[str] | None = None
        for row in metric_rows:
            deployment = row["deployment"]["task_metrics"]
            teacher = row["teacher_context"]["task_metrics"]
            numeric = {
                key
                for key in set(deployment) & set(teacher)
                if isinstance(deployment[key], (int, float))
                and not isinstance(deployment[key], bool)
                and isinstance(teacher[key], (int, float))
                and not isinstance(teacher[key], bool)
            }
            common = numeric if common is None else common & numeric
        output: dict[str, dict[str, float]] = {}
        for key in sorted(common or set()):
            deployment_mean = _mean(
                [
                    float(row["deployment"]["task_metrics"][key])
                    for row in metric_rows
                ]
            )
            teacher_mean = _mean(
                [
                    float(row["teacher_context"]["task_metrics"][key])
                    for row in metric_rows
                ]
            )
            output[key] = {
                "deployment": deployment_mean,
                "teacher_context": teacher_mean,
                "teacher_minus_deployment": teacher_mean - deployment_mean,
            }
        return output

    def summarize(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        valid = [row for row in values if row.get("status") == "PASS"]
        mismatched = [row for row in valid if not row["discrete_exact"]]
        matched = [row for row in valid if row["discrete_exact"]]

        def structure_exact(row: Mapping[str, Any]) -> bool:
            metrics = row.get("deployment", {}).get("task_metrics", {})
            return all(
                float(metrics.get(key, 0.0)) >= 1.0 - 1.0e-9
                for key in (
                    "room_accuracy",
                    "source_count_accuracy",
                    "kind_accuracy",
                )
            )

        def subgroup(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
            return {
                "rows": len(values),
                "deployment_task_score": _mean(
                    [float(row["deployment"]["task_score"]) for row in values]
                ),
                "teacher_context_task_score": _mean(
                    [float(row["teacher_context"]["task_score"]) for row in values]
                ),
                "teacher_context_task_score_lift": _mean(
                    [float(row["task_score_lift"]) for row in values]
                ),
                "continuous_rmse_reduction": _mean(
                    [float(row["continuous_rmse_reduction"]) for row in values]
                ),
                "task_metric_means": task_metric_delta(values),
            }

        structure_exact_surface_mismatch = [
            row for row in mismatched if structure_exact(row)
        ]
        structure_mismatch = [
            row for row in mismatched if not structure_exact(row)
        ]
        return {
            "rows": len(values),
            "valid_rows": len(valid),
            "discrete_exact_rate": _mean(
                [float(bool(row["discrete_exact"])) for row in valid]
            ),
            "deployment_task_score": _mean(
                [float(row["deployment"]["task_score"]) for row in valid]
            ),
            "teacher_context_task_score": _mean(
                [float(row["teacher_context"]["task_score"]) for row in valid]
            ),
            "teacher_context_task_score_lift": _mean(
                [float(row["task_score_lift"]) for row in valid]
            ),
            "deployment_continuous_rmse": _mean(
                [float(row["deployment_continuous_rmse"]) for row in valid]
            ),
            "teacher_context_continuous_rmse": _mean(
                [float(row["teacher_context_continuous_rmse"]) for row in valid]
            ),
            "continuous_rmse_reduction": _mean(
                [float(row["continuous_rmse_reduction"]) for row in valid]
            ),
            "mismatched_rows": len(mismatched),
            "mismatched_continuous_rmse_reduction": _mean(
                [float(row["continuous_rmse_reduction"]) for row in mismatched]
            ),
            "matched_rows": len(matched),
            "matched_context_exact_parity_rate": _mean(
                [float(bool(row["matched_context_exact_parity"])) for row in matched]
            ),
            "task_metric_means": task_metric_delta(valid),
            "mismatched_task_metric_means": task_metric_delta(mismatched),
            "mismatch_attribution": {
                "structure_exact_surface_or_lexical_mismatch": subgroup(
                    structure_exact_surface_mismatch
                ),
                "room_count_or_kind_mismatch": subgroup(structure_mismatch),
            },
        }

    return {
        "overall": summarize(rows),
        "by_task": {
            task: summarize(values) for task, values in sorted(grouped.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=MODEL_CONFIGS["flow"])
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("online", "ema"), default="ema")
    parser.add_argument("--rows-per-view", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--families", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--views", default="")
    parser.add_argument("--seed", type=int, default=42)
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
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_per_view < 0:
        raise ValueError("--rows-per-view must be non-negative")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shards must satisfy 0 <= shard-index < num-shards")

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
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    model_config = args.model_config.expanduser().resolve(strict=True)
    dataset_config = args.dataset_config.expanduser().resolve(strict=True)
    challenge_path = args.challenge.expanduser().resolve(strict=True)
    families = {
        value.strip() for value in args.families.split(",") if value.strip()
    } or None
    tasks = {value.strip() for value in args.tasks.split(",") if value.strip()} or None
    views = {value.strip() for value in args.views.split(",") if value.strip()} or None
    if tasks is not None and not tasks <= {"generation", "understanding", "editing"}:
        raise ValueError("--tasks contains an unsupported P11 task")
    selected = _select_rows(
        challenge_path,
        rows_per_view=args.rows_per_view,
        families=families,
    )

    wrapper, planner, checkpoint_provenance = _load_model(
        model_config_path=model_config,
        checkpoint_path=checkpoint,
        device=device,
        weights=args.weights,
    )
    checkpoint_provenance["qwen_runtime_kernels"] = (
        planner.configure_qwen_runtime_kernels(args.qwen_kernel_mode)
    )
    challenge = _dataset(
        planner=planner,
        dataset_config_path=dataset_config,
        challenge_path=challenge_path,
    )
    materialized = []
    for ordinal in selected:
        metadata = challenge[ordinal][1]
        if tasks is not None and str(metadata["p11_task"]) not in tasks:
            continue
        if views is not None and str(metadata["p11_prompt_view_id"]) not in views:
            continue
        materialized.append((ordinal, metadata))
    del challenge
    if not materialized:
        raise RuntimeError("task/view filters removed every selected challenge row")
    pre_shard_rows = len(materialized)
    materialized = [
        item
        for index, item in enumerate(materialized)
        if index % int(args.num_shards) == int(args.shard_index)
    ]
    if not materialized:
        raise RuntimeError("selection shard contains no challenge rows")
    selected = [ordinal for ordinal, _ in materialized]

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for ordinal, metadata in materialized:
        inputs = _prepare_inputs(metadata)
        task = str(metadata["p11_task"])
        row_key = _row_seed_key(metadata)
        discrete_seed = _seed(args.seed, row_key, 0, "discrete")
        noise_seed = (
            None
            if task == "editing"
            else _seed(args.seed, row_key, 0, "continuous")
        )
        target_tokens = _target_discrete_tokens(metadata)
        before = _rng_state(device)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        try:
            with torch.inference_mode(), autocast:
                common = {
                    "task": inputs["task"],
                    "input_foa": inputs["input_foa"],
                    "input_valid_mask": inputs["input_valid_mask"],
                    "input_semantic": inputs["input_semantic"],
                    "input_lexical": inputs["input_lexical"],
                    "input_sceneplan": inputs["input_sceneplan"],
                    "duration_sec": inputs["duration_sec"],
                    "sample_id": inputs["sample_id"],
                    "temperature": 0.0,
                    "discrete_seed": discrete_seed,
                    "discrete_decode_mode": args.discrete_decode_mode,
                }
                if noise_seed is not None:
                    common["noise_seed"] = noise_seed
                deployment = planner.decode_transfusion_cot(
                    inputs["prompt"], **common
                )
                teacher = planner.decode_transfusion_cot(
                    inputs["prompt"],
                    diagnostic_discrete_tokens=target_tokens,
                    **common,
                )
            after = _rng_state(device)
            deployment_score = _score_draw(planner, metadata, deployment)
            teacher_score = _score_draw(planner, metadata, teacher)
            target_core = _target_continuous_core(metadata)
            mask = _active_mask(metadata)
            deployment_rmse = _masked_rmse(
                deployment["thought_core"], target_core, mask
            )
            teacher_rmse = _masked_rmse(
                teacher["thought_core"], target_core, mask
            )
            deployment_tokens = _ids(deployment["discrete_tokens"])
            discrete_exact = bool(torch.equal(deployment_tokens, target_tokens))
            exact_parity = bool(
                torch.equal(
                    torch.as_tensor(deployment["thought_core"]).detach().cpu(),
                    torch.as_tensor(teacher["thought_core"]).detach().cpu(),
                )
                and deployment["sceneplan"] == teacher["sceneplan"]
            )
            oracle_diagnostics = teacher["diagnostics"]
            invariants = {
                "rng_isolated": _rng_equal(before, after),
                "teacher_mode_explicit": (
                    oracle_diagnostics.get("discrete_authority_mode")
                    == "diagnostic_teacher_context_override"
                ),
                "teacher_tokens_are_exact_target": bool(
                    torch.equal(_ids(teacher["discrete_tokens"]), target_tokens)
                ),
                "paired_model_decode_replayed": (
                    oracle_diagnostics.get("model_decoded_discrete_sha256")
                    == _integer_tensor_sha256(deployment_tokens)
                ),
                "paired_noise_identical": (
                    deployment["diagnostics"].get("thought_noise_sha256")
                    == teacher["diagnostics"].get("thought_noise_sha256")
                ),
                "matched_context_exact_parity": (
                    exact_parity if discrete_exact else True
                ),
            }
            row = {
                "ordinal": ordinal,
                "challenge_id": metadata["p11_challenge_id"],
                "task": task,
                "family": metadata["p11_challenge_family"],
                "view_id": metadata["p11_prompt_view_id"],
                "status": "PASS" if all(invariants.values()) else "FAIL",
                "discrete_exact": discrete_exact,
                "matched_context_exact_parity": exact_parity,
                "deployment_discrete_sha256": _integer_tensor_sha256(
                    deployment_tokens
                ),
                "target_discrete_sha256": _integer_tensor_sha256(target_tokens),
                "noise_seed": noise_seed,
                "deployment_continuous_rmse": deployment_rmse,
                "teacher_context_continuous_rmse": teacher_rmse,
                "continuous_rmse_reduction": deployment_rmse - teacher_rmse,
                "task_score_lift": (
                    float(teacher_score["task_score"])
                    - float(deployment_score["task_score"])
                ),
                "deployment": _public_score(deployment_score),
                "teacher_context": _public_score(teacher_score),
                "invariants": invariants,
            }
        except Exception as error:  # noqa: BLE001 - retain fail-closed row.
            row = {
                "ordinal": ordinal,
                "challenge_id": metadata["p11_challenge_id"],
                "task": task,
                "family": metadata["p11_challenge_family"],
                "view_id": metadata["p11_prompt_view_id"],
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
            }
        rows.append(row)

    elapsed = time.perf_counter() - started
    summary = _summarize_exposure_rows(rows)
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
        "scope": (
            "paired diagnostic of target-discrete teacher context versus normal "
            "deployment-decoded context; no deployable oracle quality claim"
        ),
        "interpretation_contract": {
            "teacher_context_uses_hidden_target": True,
            "teacher_context_is_deployable": False,
            "positive_rmse_reduction_means_discrete_exposure_gap": True,
            "same_checkpoint_evidence_discrete_seed_and_flow_noise": True,
            "normal_inference_api_unchanged": True,
        },
        "checkpoint": checkpoint_provenance,
        "model_config": _resolved_config_provenance(model_config),
        "dataset_config": _resolved_config_provenance(dataset_config),
        "challenge": str(challenge_path),
        "challenge_sha256": _sha256_file(challenge_path),
        "runtime_source_provenance": _runtime_source_provenance(device),
        "qwen_kernel_mode": args.qwen_kernel_mode,
        "weights": args.weights,
        "root_seed": args.seed,
        "rows_per_view": args.rows_per_view,
        "selection_shard": {
            "contract": "post_filter_round_robin_v1",
            "pre_shard_rows": pre_shard_rows,
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
        },
        "families_filter": None if families is None else sorted(families),
        "tasks_filter": None if tasks is None else sorted(tasks),
        "views_filter": None if views is None else sorted(views),
        "selected_ordinals": selected,
        "elapsed_seconds": elapsed,
        "rows_per_second": len(rows) / elapsed,
        "peak_memory_gib": (
            torch.cuda.max_memory_allocated(device) / (2**30)
            if device.type == "cuda"
            else None
        ),
        "summary": summary,
        "rows": rows,
    }
    report["report_sha256_without_self"] = _json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "summary": summary,
        "elapsed_seconds": elapsed,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    del planner, wrapper
    gc.collect()
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
