#!/usr/bin/env python3
"""Validate the fail-closed optimizer/EMA/DDP/throughput P11 preflight report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


BENCHMARK_PREFIX = "SAT_BENCHMARK_RESULT="
HEALTH_PREFIX = "SAT_TRAINING_GATE_RESULT="


def _last_prefixed_json(log_path: Path, prefix: str) -> dict[str, Any]:
    latest = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = line.find(prefix)
        if marker >= 0:
            latest = json.loads(line[marker + len(prefix) :])
    if latest is None:
        raise RuntimeError(f"{log_path} contains no {prefix.rstrip('=')}")
    if not isinstance(latest, dict):
        raise RuntimeError(f"{prefix.rstrip('=')} is not a JSON object")
    return latest


def _require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise RuntimeError(f"{name}: expected {expected!r}, observed {observed!r}")


def _require_positive_finite(name: str, value: Any) -> None:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise RuntimeError(f"{name} must be positive and finite, got {value!r}")


def _uniform_sample_batches(total_batches: int, budget: int = 64) -> list[int]:
    if total_batches <= budget:
        return list(range(1, total_batches + 1))
    denominator = budget - 1
    values = sorted(
        {
            1
            + (index * (total_batches - 1) + denominator // 2)
            // denominator
            for index in range(budget)
        }
    )
    if len(values) != budget or values[0] != 1 or values[-1] != total_batches:
        raise RuntimeError("invalid expected full-window utilization sample")
    return values


def _validate_benchmark(
    result: dict[str, Any],
    *,
    world_size: int,
    batch_size: int,
    initial_global_step: int,
    measured_steps: int,
    min_global_loader_samples_per_second: float,
    min_rank_mean_gpu_utilization_percent: float,
    min_peak_allocated_gib: float,
    row_identity_field: str,
) -> None:
    expected_rank_sum = world_size * (world_size - 1) // 2
    for name, expected in (
        ("world_size", world_size),
        ("expected_world_size", world_size),
        ("distributed_rank_count", world_size),
        ("distributed_rank_index_sum", expected_rank_sum),
        ("distributed_rank_index_sum_expected", expected_rank_sum),
        ("batch_size_per_gpu", batch_size),
        ("initial_global_step", initial_global_step),
        ("rank_observed_batch_size_min", batch_size),
        ("rank_observed_batch_size_max", batch_size),
        ("measured_optimizer_steps", measured_steps),
        ("rank_measured_batches_min", measured_steps),
        ("rank_measured_batches_max", measured_steps),
        ("strategy", "DDPStrategy"),
        ("distributed_backend", "nccl"),
        ("distributed_initialized", True),
        ("exact_local_batch_size_required", True),
        ("curriculum_identity_field", row_identity_field),
        ("curriculum_rows_disjoint_across_ranks", True),
        ("disjoint_curriculum_rows_required", True),
        ("cross_rank_curriculum_identity_overlap_count", 0),
        ("gpu_utilization_sample_errors", 0),
    ):
        _require_equal(f"benchmark.{name}", result.get(name), expected)
    identity_audit_batches = min(measured_steps, 8)
    identity_rows_per_rank = identity_audit_batches * batch_size
    for name, expected in (
        ("curriculum_identity_audit_batches", identity_audit_batches),
        ("rank_curriculum_identity_count_min", identity_rows_per_rank),
        ("rank_curriculum_identity_count_max", identity_rows_per_rank),
        ("rank_curriculum_identity_unique_min", identity_rows_per_rank),
        ("rank_curriculum_identity_unique_max", identity_rows_per_rank),
    ):
        _require_equal(f"benchmark.{name}", result.get(name), expected)
    if world_size == 8:
        for name, expected in (
            ("gue_each_local_batch_required", True),
            ("sampled_curriculum_batches_all_gue", True),
            ("rank_curriculum_gue_batch_count_min", identity_audit_batches),
            ("rank_curriculum_gue_batch_count_max", identity_audit_batches),
        ):
            _require_equal(f"benchmark.{name}", result.get(name), expected)
        rank_task_counts = result.get("rank_curriculum_task_counts") or []
        if len(rank_task_counts) != world_size:
            raise RuntimeError(
                "benchmark lacks one sampled G/U/E task count per DDP rank"
            )
        for rank, counts in enumerate(rank_task_counts):
            if (
                set(counts) != {"generation", "understanding", "editing"}
                or sum(int(value) for value in counts.values())
                != identity_rows_per_rank
                or any(int(value) <= 0 for value in counts.values())
            ):
                raise RuntimeError(
                    f"benchmark rank {rank} sampled invalid G/U/E counts: {counts}"
                )
    expected_utilization_batches = _uniform_sample_batches(measured_steps)
    _require_equal(
        "benchmark.gpu_utilization_sample_count",
        result.get("gpu_utilization_sample_count"),
        world_size * len(expected_utilization_batches),
    )
    _require_equal(
        "benchmark.gpu_utilization_sampling_contract",
        result.get("gpu_utilization_sampling_contract"),
        "uniform_over_measured_window_v1",
    )
    _require_equal(
        "benchmark.gpu_utilization_sample_batches",
        result.get("gpu_utilization_sample_batches"),
        expected_utilization_batches,
    )
    for name in (
        "elapsed_seconds",
        "global_loader_samples_per_second",
        "global_training_examples_per_second",
        "global_sequence_positions_per_second",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "rank_peak_allocated_gib_min",
        "rank_peak_allocated_gib_max",
        "rank_peak_reserved_gib_min",
        "rank_peak_reserved_gib_max",
    ):
        _require_positive_finite(f"benchmark.{name}", result.get(name))
    rank_peak_allocated = result.get("rank_peak_allocated_gib") or []
    rank_peak_reserved = result.get("rank_peak_reserved_gib") or []
    if len(rank_peak_allocated) != world_size or len(rank_peak_reserved) != world_size:
        raise RuntimeError(
            "benchmark lacks one CUDA memory peak per DDP rank: "
            f"allocated={len(rank_peak_allocated)}, "
            f"reserved={len(rank_peak_reserved)}, world_size={world_size}"
        )
    for rank, (allocated, reserved) in enumerate(
        zip(rank_peak_allocated, rank_peak_reserved)
    ):
        _require_positive_finite(
            f"benchmark.rank_peak_allocated_gib[{rank}]", allocated
        )
        _require_positive_finite(
            f"benchmark.rank_peak_reserved_gib[{rank}]", reserved
        )
        if float(reserved) < float(allocated):
            raise RuntimeError(
                f"benchmark rank {rank} reserved less CUDA memory than allocated"
            )
    final_metrics = result.get("final_scalar_metrics") or {}
    _require_positive_finite(
        "benchmark.final_scalar_metrics.train/owner_rows",
        final_metrics.get("train/owner_rows"),
    )
    owner_accuracy = float(final_metrics.get("train/owner_accuracy", -1.0))
    if not math.isfinite(owner_accuracy) or not 0.0 <= owner_accuracy <= 1.0:
        raise RuntimeError(
            "benchmark final owner accuracy must be finite and within [0,1], "
            f"got {owner_accuracy}"
        )
    if (
        float(result["global_loader_samples_per_second"])
        < min_global_loader_samples_per_second
    ):
        raise RuntimeError(
            "global loader throughput is below the canonical scale floor: "
            f"{result['global_loader_samples_per_second']} < "
            f"{min_global_loader_samples_per_second} samples/s"
        )
    if (
        float(result["rank_gpu_utilization_percent_mean_min"])
        < min_rank_mean_gpu_utilization_percent
    ):
        raise RuntimeError(
            "at least one rank has low mean GPU utilization: "
            f"{result['rank_gpu_utilization_percent_mean_min']} < "
            f"{min_rank_mean_gpu_utilization_percent}%"
        )
    if float(result["rank_peak_allocated_gib_min"]) < min_peak_allocated_gib:
        raise RuntimeError(
            "at least one rank's peak allocated memory is below the canonical "
            "scale floor: "
            f"{result['rank_peak_allocated_gib_min']} < "
            f"{min_peak_allocated_gib} GiB"
        )


def _validate_health(
    result: dict[str, Any],
    *,
    world_size: int,
    initial_global_step: int,
    optimizer_steps: int,
    arm: str,
) -> None:
    executed_steps = optimizer_steps - initial_global_step
    if executed_steps <= 0:
        raise RuntimeError(
            "health target optimizer step must exceed the restored global step"
        )
    _require_equal("health.status", result.get("status"), "PASS")
    _require_equal(
        "health.initial_global_step",
        result.get("initial_global_step"),
        initial_global_step,
    )
    _require_equal("health.global_step", result.get("global_step"), optimizer_steps)
    _require_equal(
        "health.optimizer_events", result.get("optimizer_events"), executed_steps
    )
    if int(result.get("optimizer_state_step_max", 0)) < optimizer_steps:
        raise RuntimeError("health optimizer state did not reach the requested step")
    _require_positive_finite("health.gradient_norm_min", result.get("gradient_norm_min"))
    _require_positive_finite("health.gradient_norm_max", result.get("gradient_norm_max"))

    ema_initial = result.get("ema_initial") or {}
    ema_final = result.get("ema_final") or {}
    ema_advances = result.get("ema_advances") or {}
    if set(ema_initial) != {"p11_ema"} or set(ema_final) != {"p11_ema"}:
        raise RuntimeError(
            "health report must contain exactly the canonical P11 EMA state"
        )
    if int(ema_final["p11_ema"]) <= int(ema_initial["p11_ema"]):
        raise RuntimeError("P11 EMA did not advance")
    _require_positive_finite("health.ema_advances.p11_ema", ema_advances.get("p11_ema"))

    distributed = result.get("distributed_health") or {}
    expected_rank_sum = world_size * (world_size - 1) // 2
    for name, expected in (
        ("world_size", world_size),
        ("expected_world_size", world_size),
        ("rank_count", world_size),
        ("rank_index_sum", expected_rank_sum),
        ("rank_index_sum_expected", expected_rank_sum),
        ("strategy", "DDPStrategy"),
        ("backend", "nccl"),
        ("initialized", True),
        ("optimizer_events_min", executed_steps),
        ("optimizer_events_max", executed_steps),
    ):
        _require_equal(f"health.distributed.{name}", distributed.get(name), expected)
    if int(distributed.get("optimizer_state_step_min", 0)) < optimizer_steps:
        raise RuntimeError("at least one DDP rank has stale optimizer state")
    for name in (
        "gradient_checks_min",
        "gradient_checks_max",
        "ema_advance_min",
        "ema_advance_max",
    ):
        _require_positive_finite(f"health.distributed.{name}", distributed.get(name))

    objectives = result.get("objectives") or {}
    if arm == "audio_aware_flow_r1_v1":
        required_objectives = {
            "train/loss",
            "train/generation_discrete_ce",
            "train/understanding_discrete_ce",
            "train/editing_discrete_ce",
            "train/observation_ce",
            "train/delta_ce",
            "train/text_end_ce",
            "train/scene_eos_ce",
            "train/u_inventory_ce",
            "train/u_source_count_ce",
            "train/u_room_ce",
            "train/u_kind_ce",
            "train/flow",
            "train/observation_solve",
            "train/delta_solve",
            "train/locality",
            "train/owner",
        }
    elif arm in {
        "sketch_first_transfusion_cot_v4",
        "sketch_first_direct_mse_v4",
    }:
        required_objectives = {
            "train/loss",
            "train/generation_discrete_ce",
            "train/understanding_discrete_ce",
            "train/editing_discrete_ce",
            "train/solve",
            "train/locality",
            "train/owner",
        }
        if arm == "sketch_first_transfusion_cot_v4":
            required_objectives.add("train/flow")
    else:
        required_objectives = {
            "train/loss",
            "train/generation_ce",
            "train/understanding_ce",
            "train/editing_ce",
        }
    missing = sorted(required_objectives - set(objectives))
    if missing:
        raise RuntimeError(f"health report lacks EST objectives: {missing}")
    for name in required_objectives:
        value = float(objectives[name])
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"health objective {name} is invalid: {value}")
    owner_window = (result.get("metric_windows") or {}).get("train/owner") or {}
    if (
        owner_window.get("activity_metric") != "train/owner_rows"
        or int(owner_window.get("active_observations", 0)) <= 0
        or float(owner_window.get("first_active_weight", 0.0)) <= 0.0
        or float(owner_window.get("last_active_weight", 0.0)) <= 0.0
    ):
        raise RuntimeError(
            "health report does not prove active DeltaSketch owner supervision"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument("--expected-batch-size", type=int, default=8)
    parser.add_argument(
        "--expected-arm",
        choices=(
            "discrete_v0",
            "sketch_first_direct_mse_v4",
            "sketch_first_transfusion_cot_v4",
            "audio_aware_flow_r1_v1",
        ),
        default="audio_aware_flow_r1_v1",
    )
    parser.add_argument("--expected-optimizer-steps", type=int, default=8)
    parser.add_argument("--expected-initial-global-step", type=int, default=0)
    parser.add_argument("--expected-measured-steps", type=int, default=5)
    parser.add_argument(
        "--min-global-loader-samples-per-second", type=float, default=0.0
    )
    parser.add_argument(
        "--min-rank-mean-gpu-utilization-percent", type=float, default=0.0
    )
    parser.add_argument("--min-peak-allocated-gib", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    benchmark = _last_prefixed_json(args.log, BENCHMARK_PREFIX)
    health = _last_prefixed_json(args.log, HEALTH_PREFIX)
    row_identity_field = (
        "p11_row_identity"
        if args.expected_arm == "audio_aware_flow_r1_v1"
        else "p11_curriculum_id"
    )
    try:
        _validate_benchmark(
            benchmark,
            world_size=args.expected_world_size,
            batch_size=args.expected_batch_size,
            initial_global_step=args.expected_initial_global_step,
            measured_steps=args.expected_measured_steps,
            min_global_loader_samples_per_second=(
                args.min_global_loader_samples_per_second
            ),
            min_rank_mean_gpu_utilization_percent=(
                args.min_rank_mean_gpu_utilization_percent
            ),
            min_peak_allocated_gib=args.min_peak_allocated_gib,
            row_identity_field=row_identity_field,
        )
        _validate_health(
            health,
            world_size=args.expected_world_size,
            initial_global_step=args.expected_initial_global_step,
            optimizer_steps=args.expected_optimizer_steps,
            arm=args.expected_arm,
        )
    except Exception as error:
        failure_report = {
            "status": "FAIL",
            "audit": "sceneplan_p11_gpu_preflight_v5",
            "log": str(args.log.resolve()),
            "error_type": type(error).__name__,
            "error": str(error),
            "expected": {
                "world_size": args.expected_world_size,
                "batch_size_per_gpu": args.expected_batch_size,
                "arm": args.expected_arm,
                "row_identity_field": row_identity_field,
                "optimizer_steps": args.expected_optimizer_steps,
                "initial_global_step": args.expected_initial_global_step,
                "measured_throughput_steps": args.expected_measured_steps,
                "min_global_loader_samples_per_second": (
                    args.min_global_loader_samples_per_second
                ),
                "min_rank_mean_gpu_utilization_percent": (
                    args.min_rank_mean_gpu_utilization_percent
                ),
                "min_rank_peak_allocated_gib": args.min_peak_allocated_gib,
            },
            "observed": {
                "global_loader_samples_per_second": benchmark.get(
                    "global_loader_samples_per_second"
                ),
                "rank_gpu_utilization_percent_mean_min": benchmark.get(
                    "rank_gpu_utilization_percent_mean_min"
                ),
                "peak_allocated_gib": benchmark.get("peak_allocated_gib"),
                "rank_peak_allocated_gib_min": benchmark.get(
                    "rank_peak_allocated_gib_min"
                ),
                "health_status": health.get("status"),
                "global_step": health.get("global_step"),
            },
        }
        rendered = json.dumps(failure_report, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        print(
            f"P11_GPU_PREFLIGHT_REPORT={json.dumps(failure_report, sort_keys=True)}"
        )
        raise
    report = {
        "status": "PASS",
        "audit": "sceneplan_p11_gpu_preflight_v5",
        "log": str(args.log.resolve()),
        "world_size": args.expected_world_size,
        "batch_size_per_gpu": args.expected_batch_size,
        "arm": args.expected_arm,
        "row_identity_field": row_identity_field,
        "optimizer_steps": args.expected_optimizer_steps,
        "initial_global_step": args.expected_initial_global_step,
        "measured_throughput_steps": args.expected_measured_steps,
        "global_loader_samples_per_second": benchmark[
            "global_loader_samples_per_second"
        ],
        "global_training_examples_per_second": benchmark[
            "global_training_examples_per_second"
        ],
        "global_sequence_positions_per_second": benchmark[
            "global_sequence_positions_per_second"
        ],
        "gpu_utilization_percent_mean": benchmark[
            "gpu_utilization_percent_mean"
        ],
        "gpu_utilization_sampling_contract": benchmark[
            "gpu_utilization_sampling_contract"
        ],
        "gpu_utilization_sample_batches": benchmark[
            "gpu_utilization_sample_batches"
        ],
        "rank_gpu_utilization_percent_mean_min": benchmark[
            "rank_gpu_utilization_percent_mean_min"
        ],
        "rank_gpu_utilization_percent_mean_max": benchmark[
            "rank_gpu_utilization_percent_mean_max"
        ],
        "curriculum_identity_audit_batches": benchmark[
            "curriculum_identity_audit_batches"
        ],
        "rank_curriculum_identity_count": benchmark[
            "rank_curriculum_identity_count_min"
        ],
        "cross_rank_curriculum_identity_overlap_count": benchmark[
            "cross_rank_curriculum_identity_overlap_count"
        ],
        "curriculum_rows_disjoint_across_ranks": benchmark[
            "curriculum_rows_disjoint_across_ranks"
        ],
        "gue_each_local_batch_required": benchmark.get(
            "gue_each_local_batch_required", False
        ),
        "sampled_curriculum_batches_all_gue": benchmark.get(
            "sampled_curriculum_batches_all_gue", False
        ),
        "rank_curriculum_gue_batch_count_min": benchmark.get(
            "rank_curriculum_gue_batch_count_min", 0
        ),
        "rank_curriculum_gue_batch_count_max": benchmark.get(
            "rank_curriculum_gue_batch_count_max", 0
        ),
        "rank_curriculum_task_counts": benchmark.get(
            "rank_curriculum_task_counts", []
        ),
        "peak_allocated_gib": benchmark["peak_allocated_gib"],
        "peak_reserved_gib": benchmark["peak_reserved_gib"],
        "rank_peak_allocated_gib": benchmark["rank_peak_allocated_gib"],
        "rank_peak_allocated_gib_min": benchmark[
            "rank_peak_allocated_gib_min"
        ],
        "rank_peak_allocated_gib_max": benchmark[
            "rank_peak_allocated_gib_max"
        ],
        "rank_peak_reserved_gib": benchmark["rank_peak_reserved_gib"],
        "rank_peak_reserved_gib_min": benchmark[
            "rank_peak_reserved_gib_min"
        ],
        "rank_peak_reserved_gib_max": benchmark[
            "rank_peak_reserved_gib_max"
        ],
        "optimizer_state_step_min": health["distributed_health"][
            "optimizer_state_step_min"
        ],
        "ema_advance_min": health["distributed_health"]["ema_advance_min"],
        "objectives": health["objectives"],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(f"P11_GPU_PREFLIGHT_REPORT={json.dumps(report, sort_keys=True)}")


if __name__ == "__main__":
    main()
