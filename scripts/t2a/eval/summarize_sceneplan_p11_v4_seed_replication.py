#!/usr/bin/env python3
"""Fail-closed summary for exactly two canonical P11-v4 screening seeds.

This script measures replication of one frozen architecture/evaluator contract.
It deliberately does not compare architectures and cannot authorize a full or
multi-GPU run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


TASKS = ("generation", "understanding", "editing")
EXPECTED_SCHEMA = "stable_audio_tools.p11_v4_unified_challenge_eval"
EXPECTED_SCHEMA_VERSION = 10
EXPECTED_EVALUATOR = (
    "hybrid_flow_gu_direct_e_torch_reference_fair_lexical_boundary_v10"
)
REQUIRED_DECLINING_METRICS = (
    "train/loss",
    "train/generation_discrete_ce",
    "train/understanding_discrete_ce",
    "train/editing_discrete_ce",
    "train/text_end_ce",
    "train/scene_eos_ce",
    "train/control_direction_ce",
    "train/flow",
    "train/solve",
    "train/locality",
    "train/owner",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _last_log_object(payload: str, prefix: str) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for line in payload.replace("\r", "\n").splitlines():
        offset = line.find(prefix)
        if offset < 0:
            continue
        try:
            value = json.loads(line[offset + len(prefix) :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    if not values:
        raise RuntimeError(f"training log contains no complete {prefix}")
    return values[-1]


def _training(
    log_path: Path, expected_seed: int, *, expected_step: int
) -> dict[str, Any]:
    path = log_path.resolve(strict=True)
    payload = path.read_text(encoding="utf-8", errors="replace")
    seed_matches = re.findall(
        r"\[rng\] global_rank=0 global_step=0 training_seed=(\d+)",
        payload.replace("\r", "\n"),
    )
    if not seed_matches:
        raise RuntimeError(f"{path} lacks the rank-0 training seed declaration")
    seed_text = seed_matches[-1]
    if int(seed_text) != expected_seed:
        raise RuntimeError(
            f"declared seed {seed_text} does not match requested seed {expected_seed}"
        )

    benchmark = _last_log_object(payload, "SAT_BENCHMARK_RESULT=")
    gate = _last_log_object(payload, "SAT_TRAINING_GATE_RESULT=")
    max_steps = int(benchmark["target_final_global_step"])
    if max_steps != int(expected_step):
        raise RuntimeError(
            f"{path} completed step {max_steps}, expected {int(expected_step)}"
        )
    if (
        gate.get("status") != "PASS"
        or int(gate.get("global_step", -1)) != max_steps
        or int(benchmark.get("target_final_global_step", -1)) != max_steps
    ):
        raise RuntimeError(f"{path} did not complete its final training gate")
    if (
        int(benchmark.get("batch_size_per_gpu", -1)) != 8
        or int(benchmark.get("observed_batch_size_per_gpu_min", -1)) != 8
        or int(benchmark.get("observed_batch_size_per_gpu_max", -1)) != 8
        or int(benchmark.get("world_size", -1)) != 1
    ):
        raise RuntimeError(f"{path} is not the canonical single-GPU batch-8 screen")
    if (
        int(gate.get("optimizer_events", -1)) != max_steps
        or int(gate.get("optimizer_state_step_max", -1)) != max_steps
        or int(gate.get("ema_advances", {}).get("p11_ema", -1)) != max_steps
    ):
        raise RuntimeError(f"{path} lacks exact optimizer/EMA advancement")

    gradient_min = float(gate["gradient_norm_min"])
    gradient_max = float(gate["gradient_norm_max"])
    if not (
        math.isfinite(gradient_min)
        and math.isfinite(gradient_max)
        and gradient_min > 0.0
        and gradient_max >= gradient_min
    ):
        raise RuntimeError(f"{path} has invalid gradient evidence")

    windows = gate.get("metric_windows", {})
    ratios: dict[str, float] = {}
    for metric in REQUIRED_DECLINING_METRICS:
        if metric not in windows:
            raise RuntimeError(f"{path} lacks required metric window {metric}")
        ratio = float(windows[metric]["last_over_first"])
        if not math.isfinite(ratio) or ratio >= 0.99:
            raise RuntimeError(f"{path} failed decline gate for {metric}: {ratio}")
        ratios[metric] = ratio

    return {
        "seed": expected_seed,
        "log": str(path),
        "log_sha256": _sha256_file(path),
        "run_root": str(path.parent),
        "global_step": max_steps,
        "batch_size_per_gpu": 8,
        "world_size": 1,
        "optimizer_events": int(gate["optimizer_events"]),
        "ema_advances": int(gate["ema_advances"]["p11_ema"]),
        "samples_per_second": float(benchmark["global_samples_per_second"]),
        "seconds_per_step": float(benchmark["seconds_per_step"]),
        "peak_allocated_gib": float(benchmark["peak_allocated_gib"]),
        "gradient_norm_min": gradient_min,
        "gradient_norm_max": gradient_max,
        "loss_window_ratios": ratios,
    }


def _group(report: Mapping[str, Any], group: str) -> Mapping[str, Any]:
    try:
        value = report["aggregate"][group]["1"]
    except KeyError as exc:
        raise RuntimeError(f"report lacks K=1 aggregate group {group!r}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"aggregate group {group!r} is not an object")
    return value


def _evaluation(
    report_path: Path, *, expected_step: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = report_path.resolve(strict=True)
    report = _load(path)
    if (
        report.get("schema") != EXPECTED_SCHEMA
        or int(report.get("schema_version", -1)) != EXPECTED_SCHEMA_VERSION
        or report.get("evaluator_contract") != EXPECTED_EVALUATOR
        or report.get("status") != "PASS"
        or report.get("arm") != "flow"
        or report.get("weights") != "ema"
    ):
        raise RuntimeError(f"{path} is not a passing canonical Flow eval-v10 report")
    if (
        report.get("draws") != 1
        or report.get("k_values") != [1]
        or report.get("qwen_kernel_mode") != "torch_reference"
        or report.get("discrete_decode_mode") != "prefix_recompute"
        or float(report.get("rng_isolation_rate", -1.0)) != 1.0
    ):
        raise RuntimeError(f"{path} does not use the deterministic K=1 protocol")
    lexical = report.get("lexical_authority_intervention", {})
    lexical_source = report.get("lexical_evidence_provenance", {})
    if (
        lexical.get("mode") != "normal"
        or lexical.get("input_lexical_removed_before_model") is not False
        or lexical.get("target_transcript_access") != "forbidden"
        or lexical_source.get("target_transcript_access") != "forbidden"
    ):
        raise RuntimeError(f"{path} is not the input-only reliable-ASR route")

    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1:
        raise RuntimeError(f"{path} must contain exactly one checkpoint")
    checkpoint = checkpoints[0]
    if (
        checkpoint.get("checkpoint_model_config_matches_runtime") is not True
        or int(checkpoint.get("global_step", -1)) != int(expected_step)
        or int(checkpoint.get("ema_step", -1)) != int(expected_step)
        or checkpoint.get("qwen_runtime_kernels", {}).get("mode")
        != "torch_reference"
    ):
        raise RuntimeError(f"{path} lacks checkpoint/config/kernel identity proof")
    checkpoint_reload_identity = {
        key: value
        for key, value in checkpoint.items()
        if key != "qwen_runtime_kernels"
    }
    if report.get("scoring_checkpoint_reload") != checkpoint_reload_identity:
        raise RuntimeError(f"{path} scoring reload differs from checkpoint provenance")

    all_metrics = _group(report, "all")
    if (
        int(all_metrics["rows"]) <= 0
        or float(all_metrics["valid_rate"]) != 1.0
        or float(all_metrics["semantic_immutability_rate"]) != 1.0
        or report.get("p10_closure", {}).get(
            "conditioning_compiled_for_every_valid_draw"
        )
        is not True
    ):
        raise RuntimeError(f"{path} failed validity/immutability/P10 compilation")
    counterfactual = report.get("editing_counterfactual", {}).get("1", {})
    if not counterfactual:
        raise RuntimeError(f"{path} lacks editing counterfactual evaluation")

    tasks: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        value = _group(report, f"task:{task}")
        if float(value["valid_rate"]) != 1.0:
            raise RuntimeError(f"{path} has invalid {task} rows")
        tasks[task] = {
            "rows": int(value["rows"]),
            "task_score": float(value["task_score_mean"]),
            "valid_rate": float(value["valid_rate"]),
        }
    macro = statistics.fmean(value["task_score"] for value in tasks.values())
    return report, {
        "report": str(path),
        "report_file_sha256": _sha256_file(path),
        "report_sha256_without_self": report["report_sha256_without_self"],
        "checkpoint": checkpoint,
        "row_weighted_task_score": float(all_metrics["task_score_mean"]),
        "gue_macro_score": macro,
        "valid_rate": float(all_metrics["valid_rate"]),
        "semantic_immutability_rate": float(
            all_metrics["semantic_immutability_rate"]
        ),
        "tasks": tasks,
        "editing_counterfactual": counterfactual,
        "performance": report["performance"],
    }


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("seed replication requires at least two values")
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("TRAINING_SEED", "REPORT", "TRAIN_LOG"),
        required=True,
    )
    parser.add_argument("--expected-step", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run) != 2:
        raise ValueError("provide exactly two independent training seeds")
    if args.expected_step <= 0:
        raise ValueError("--expected-step must be positive")

    runs: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for seed_text, report_text, log_text in args.run:
        seed = int(seed_text)
        if seed in seen_seeds:
            raise ValueError(f"duplicate training seed: {seed}")
        seen_seeds.add(seed)
        training = _training(
            Path(log_text), seed, expected_step=args.expected_step
        )
        raw_report, evaluation = _evaluation(
            Path(report_text), expected_step=args.expected_step
        )
        checkpoint_path = Path(evaluation["checkpoint"]["checkpoint"]).resolve(
            strict=True
        )
        if checkpoint_path.parent.parent != Path(training["run_root"]).resolve(
            strict=True
        ):
            raise RuntimeError(f"seed {seed} report and training run roots differ")
        if evaluation["checkpoint"]["global_step"] != training["global_step"]:
            raise RuntimeError(f"seed {seed} report and log steps differ")
        runs.append(
            {
                "training_seed": seed,
                "training": training,
                "evaluation": evaluation,
            }
        )
        reports.append(raw_report)

    reference = reports[0]
    matched_keys = (
        "architecture",
        "challenge_sha256",
        "selected_ordinals",
        "rows",
        "rows_per_view",
        "root_seed",
        "weights",
        "evaluator_contract",
        "discrete_decode_mode",
        "rng_contract",
        "qwen_kernel_mode",
        "config_provenance",
        "lexical_evidence_provenance",
        "lexical_authority_intervention",
        "runtime_source_provenance",
    )
    for index, report in enumerate(reports[1:], start=1):
        for key in matched_keys:
            if report.get(key) != reference.get(key):
                raise RuntimeError(
                    f"seed index {index} differs on frozen evaluator invariant {key}"
                )
    checkpoint_hashes = {
        run["evaluation"]["checkpoint"]["checkpoint_sha256"] for run in runs
    }
    if len(checkpoint_hashes) != len(runs):
        raise RuntimeError("independent seeds unexpectedly reused a checkpoint")

    aggregate = {
        "row_weighted_task_score": _summary(
            [run["evaluation"]["row_weighted_task_score"] for run in runs]
        ),
        "gue_macro_score": _summary(
            [run["evaluation"]["gue_macro_score"] for run in runs]
        ),
        "tasks": {
            task: _summary(
                [run["evaluation"]["tasks"][task]["task_score"] for run in runs]
            )
            for task in TASKS
        },
        "training": {
            "samples_per_second": _summary(
                [run["training"]["samples_per_second"] for run in runs]
            ),
            "peak_allocated_gib": _summary(
                [run["training"]["peak_allocated_gib"] for run in runs]
            ),
        },
    }
    output = {
        "schema": "stable_audio_tools.p11_v4_seed_replication",
        "schema_version": 1,
        "status": "PASS",
        "decision": "TWO_SEED_CANONICAL_SCREENING_REPLICATION_COMPLETE",
        "scope": (
            f"two independent {args.expected_step}-step screening seeds on one "
            "frozen held-out evaluator; descriptive replication only"
        ),
        "claim_boundary": {
            "architecture_graph_learnable_in_both_seeds": True,
            "heldout_valid_in_both_seeds": True,
            "editing_counterfactual_exact_in_both_seeds": all(
                run["evaluation"]["editing_counterfactual"].get(
                    "both_counterfactual_targets_correct_rate"
                ) == 1.0
                and run["evaluation"]["editing_counterfactual"].get(
                    "prompt_counterfactual_changes_patch_rate"
                ) == 1.0
                for run in runs
            ),
            "quality_superiority_over_d0_or_direct_established": False,
            "statistical_significance_established": False,
            "full_corpus_training_authorized": False,
            "eight_gpu_training_authorized": False,
            "frozen_p10_rendered_foa_closure_established": False,
        },
        "frozen_evaluation_contract": {
            key: reference.get(key)
            for key in (
                "architecture",
                "challenge_sha256",
                "selected_ordinals",
                "root_seed",
                "weights",
                "evaluator_contract",
                "discrete_decode_mode",
                "rng_contract",
                "qwen_kernel_mode",
                "config_provenance",
                "lexical_evidence_provenance",
                "lexical_authority_intervention",
                "runtime_source_provenance",
            )
        },
        "runs": sorted(runs, key=lambda value: value["training_seed"]),
        "aggregate": aggregate,
    }
    output["report_sha256_without_self"] = _json_sha256(output)
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
