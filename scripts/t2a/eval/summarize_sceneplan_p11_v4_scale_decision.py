#!/usr/bin/env python3
"""Apply the frozen P11-v4 decision thresholds to evaluator-v10 evidence.

The quality thresholds and decision branches are unchanged from the v1 source
snapshot that predates the medium checkpoint.  This revision repairs three
evidence-protocol defects: canonical comparisons use ``torch_reference``;
P10 closure consumes frozen scored ScenePlans and checks each render against
its own legal duration; and reliable-ASR on/drop runs sequentially on one
physical GPU before applying predeclared causal thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402


QUALITY_SCHEMA = "stable_audio_tools.p11_v4_unified_challenge_eval"
QUALITY_SCHEMA_VERSION = 10
EXPECTED_EVALUATOR_CONTRACT = (
    "hybrid_flow_gu_direct_e_torch_reference_fair_lexical_boundary_v10"
)
CANONICAL_QWEN_KERNEL_MODE = "torch_reference"
FROZEN_THRESHOLD_RULE_SOURCE = REPO_ROOT / (
    "artifacts/sceneplan_p11/protocol_snapshots/"
    "summarize_sceneplan_p11_v4_scale_decision_v1_sha5b8f3dfb.py"
)
FROZEN_THRESHOLD_RULE_SHA256 = (
    "5b8f3dfbcc1bd240f803a58ed9148cd76fb87f4f224f07ecc698dfa4339de724"
)
CANONICAL_P10_SHA256 = (
    "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
)
P11_ATTRIBUTION_CONTRACT = "p11_planning_only_with_frozen_p10_oracle_ceiling_v1"
TASK_GROUPS = {
    "generation": "task:generation",
    "understanding": "task:understanding",
    "editing": "task:editing",
}
FROZEN_THRESHOLDS = {
    "overall_tie_absolute": 0.02,
    "material_scale_gain": 0.01,
    "editing_regression_tolerance": 0.02,
    "minimum_usable_row_weighted_k1": 0.75,
    "minimum_g_u_k8_valid_rate": 1.0,
    "minimum_g_u_k8_semantic_immutability_rate": 1.0,
    "minimum_g_u_k8_nondegenerate_numeric_rate": 0.95,
    "minimum_g_u_k8_oracle_lift": 0.01,
    "maximum_g_u_k8_absolute_calibration_error_95": 0.15,
    "minimum_global_loader_samples_per_second": 40.0,
    "minimum_rank_mean_gpu_utilization_percent": 45.0,
    "minimum_peak_allocated_gib": 30.0,
}


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


def _load(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"report is not a JSON object: {resolved}")
    claimed = value.get("report_sha256_without_self")
    if claimed is not None:
        unhashed = dict(value)
        unhashed.pop("report_sha256_without_self", None)
        actual = _json_sha256(unhashed)
        if claimed != actual:
            raise RuntimeError(
                f"report self-hash mismatch for {resolved}: {claimed} != {actual}"
            )
    return resolved, value


def _record(path: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "schema": report.get("schema"),
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "report_sha256_without_self": report.get("report_sha256_without_self"),
    }


def _quality_shape(
    name: str,
    report: Mapping[str, Any],
    *,
    arm: str,
    required_k: tuple[int, ...],
) -> None:
    if (
        report.get("schema") != QUALITY_SCHEMA
        or int(report.get("schema_version", -1)) != QUALITY_SCHEMA_VERSION
    ):
        raise RuntimeError(f"{name} is not evaluator-v10 output")
    if report.get("arm") != arm:
        raise RuntimeError(f"{name} arm mismatch: {report.get('arm')} != {arm}")
    if not set(required_k).issubset({int(value) for value in report.get("k_values", [])}):
        raise RuntimeError(f"{name} lacks required K values {required_k}")
    if int(report.get("rows", -1)) != 300:
        raise RuntimeError(f"{name} must contain the frozen 300-row panel")


def _point(report: Mapping[str, Any], group: str, k: int) -> Mapping[str, Any]:
    try:
        value = report["aggregate"][group][str(k)]
    except KeyError as error:
        raise RuntimeError(f"quality report lacks aggregate {group}/K={k}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"quality aggregate {group}/K={k} is malformed")
    return value


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


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


def _quality_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    tasks = {
        task: float(_point(report, group, 1)["task_score_mean"])
        for task, group in TASK_GROUPS.items()
    }
    return {
        "row_weighted_k1": float(_point(report, "all", 1)["task_score_mean"]),
        "gue_macro_k1": sum(tasks.values()) / len(tasks),
        "tasks_k1": tasks,
    }


def _posterior_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for task in ("generation", "understanding"):
        group = TASK_GROUPS[task]
        k1 = _point(report, group, 1)
        k8 = _point(report, group, 8)
        calibration = k8.get("calibration") or {}
        output[task] = {
            "valid_rate": float(k8["valid_rate"]),
            "semantic_immutability_rate": float(k8["semantic_immutability_rate"]),
            "nondegenerate_numeric_rate": float(k8["nondegenerate_numeric_rate"]),
            "oracle_lift_k8_minus_k1": float(
                k8["oracle_best_of_k_task_score"] - k1["task_score_mean"]
            ),
            "numeric_pairwise_rmse": float(k8["numeric_pairwise_rmse"]),
            "calibration": dict(calibration),
        }
    return output


def _posterior_gates(summary: Mapping[str, Any]) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    for task, values in summary.items():
        calibration = values["calibration"]
        prefix = f"{task}_k8"
        gates[f"{prefix}_valid"] = (
            values["valid_rate"] >= FROZEN_THRESHOLDS["minimum_g_u_k8_valid_rate"]
        )
        gates[f"{prefix}_semantic_immutable"] = (
            values["semantic_immutability_rate"]
            >= FROZEN_THRESHOLDS[
                "minimum_g_u_k8_semantic_immutability_rate"
            ]
        )
        gates[f"{prefix}_nondegenerate"] = (
            values["nondegenerate_numeric_rate"]
            >= FROZEN_THRESHOLDS["minimum_g_u_k8_nondegenerate_numeric_rate"]
        )
        gates[f"{prefix}_oracle_lift"] = (
            values["oracle_lift_k8_minus_k1"]
            >= FROZEN_THRESHOLDS["minimum_g_u_k8_oracle_lift"]
        )
        calibration_error = calibration.get("absolute_calibration_error_95")
        spread = calibration.get("spread_mean")
        gates[f"{prefix}_calibration"] = (
            _finite(calibration_error)
            and float(calibration_error)
            <= FROZEN_THRESHOLDS[
                "maximum_g_u_k8_absolute_calibration_error_95"
            ]
            and _finite(spread)
            and float(spread) > 0.0
        )
    return gates


def _causal_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    aggregates = report.get("aggregate") or {}
    fields = (
        "discrete_authority_preservation_rate",
        "numeric_response_rate",
        "p10_semantic_preservation_rate",
        "rng_isolation_rate",
        "semantic_plan_preservation_rate",
        "valid_rate",
    )
    return {
        "schema_v3": (
            report.get("schema")
            == "stable_audio_tools.p11_v4_causal_intervention_eval"
            and int(report.get("schema_version", -1)) == 3
        ),
        "status_pass": report.get("status") == "PASS",
        "nonempty": bool(aggregates),
        **{
            f"all_interventions_{field}": bool(aggregates)
            and all(float(value.get(field, -1.0)) == 1.0 for value in aggregates.values())
            for field in fields
        },
    }


def _asr_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    aggregate = report.get("aggregate") or {}
    reliable = aggregate.get("reliable_asr_understanding") or {}
    no_reliable = aggregate.get("no_reliable_asr_understanding") or {}
    causal = report.get("causal_gates") or {}
    comparability = report.get("comparability_gates") or {}
    return {
        "schema_v2": (
            report.get("schema")
            == "stable_audio_tools.p11_v4_reliable_asr_causal_ab"
            and int(report.get("schema_version", -1)) == 2
        ),
        "status_pass": report.get("status") == "PASS",
        "same_physical_gpu_sequential": comparability.get(
            "same_physical_gpu_sequential"
        )
        is True,
        "reliable_rows_positive": int(reliable.get("rows", 0)) > 0
        and float(reliable.get("positive_rate", 0.0)) >= 0.90
        and float(reliable.get("mean_delta", 0.0)) > 0.0,
        "non_reliable_rows_unchanged": int(no_reliable.get("rows", 0)) > 0
        and int(no_reliable.get("exactly_unchanged_rows", -1))
        == int(no_reliable.get("rows", 0)),
        "all_causal_gates_pass": bool(causal)
        and all(bool(value) for value in causal.values()),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-quality", type=Path, required=True)
    parser.add_argument("--candidate-causal", type=Path, required=True)
    parser.add_argument("--candidate-asr", type=Path, required=True)
    parser.add_argument("--candidate-repro", type=Path, required=True)
    parser.add_argument("--candidate-p10-closure", type=Path, required=True)
    parser.add_argument("--d0-quality", type=Path, required=True)
    parser.add_argument("--direct-quality", type=Path, required=True)
    parser.add_argument("--screen-flow-quality", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--launch-contract", type=Path, required=True)
    parser.add_argument(
        "--pretrain-report", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs: dict[str, tuple[Path, dict[str, Any]]] = {
        name: _load(getattr(args, name.replace("-", "_")))
        for name in (
            "candidate-quality",
            "candidate-causal",
            "candidate-asr",
            "candidate-repro",
            "candidate-p10-closure",
            "d0-quality",
            "direct-quality",
            "screen-flow-quality",
            "training-report",
            "launch-contract",
        )
    }
    for index, path in enumerate(args.pretrain_report):
        inputs[f"pretrain-report-{index}"] = _load(path)
    candidate = inputs["candidate-quality"][1]
    d0 = inputs["d0-quality"][1]
    direct = inputs["direct-quality"][1]
    screen = inputs["screen-flow-quality"][1]
    causal = inputs["candidate-causal"][1]
    asr = inputs["candidate-asr"][1]
    repro = inputs["candidate-repro"][1]
    closure = inputs["candidate-p10-closure"][1]
    training = inputs["training-report"][1]
    launch = inputs["launch-contract"][1]
    pretrain_reports = [
        value
        for name, (_, value) in inputs.items()
        if name.startswith("pretrain-report-")
    ]

    _quality_shape("candidate", candidate, arm="flow", required_k=(1, 4, 8))
    _quality_shape("d0", d0, arm="d0", required_k=(1,))
    _quality_shape("direct", direct, arm="direct", required_k=(1,))
    _quality_shape("screen-flow", screen, arm="flow", required_k=(1, 4, 8))
    quality_reports = (candidate, d0, direct, screen)
    comparability_gates = {
        "same_challenge_sha256": len(
            {str(value.get("challenge_sha256")) for value in quality_reports}
        )
        == 1,
        "same_selected_ordinals": all(
            value.get("selected_ordinals") == candidate.get("selected_ordinals")
            for value in quality_reports[1:]
        ),
        "same_dataset_resolved_sha256": len(
            {
                str(
                    (value.get("config_provenance") or {})
                    .get("dataset", {})
                    .get("resolved_sha256")
                )
                for value in quality_reports
            }
        )
        == 1,
        "seed_42": all(int(value.get("root_seed", -1)) == 42 for value in quality_reports),
        "ema_weights": all(value.get("weights") == "ema" for value in quality_reports),
        "prefix_recompute": all(
            value.get("discrete_decode_mode") == "prefix_recompute"
            for value in quality_reports
        ),
        "same_evaluator_contract": all(
            value.get("evaluator_contract") == EXPECTED_EVALUATOR_CONTRACT
            for value in quality_reports
        ),
        "deterministic_qwen_reference_kernel": all(
            value.get("qwen_kernel_mode") == CANONICAL_QWEN_KERNEL_MODE
            and (value.get("scientific_kernel_contract") or {}).get(
                "scientific_report"
            )
            is True
            for value in quality_reports
        ),
        "causal_uses_deterministic_qwen_reference_kernel": (
            causal.get("qwen_kernel_mode") == CANONICAL_QWEN_KERNEL_MODE
        ),
        "quality_reports_pass": all(
            value.get("status") == "PASS" for value in quality_reports
        ),
    }

    candidate_quality = _quality_summary(candidate)
    d0_quality = _quality_summary(d0)
    direct_quality = _quality_summary(direct)
    screen_quality = _quality_summary(screen)
    posterior = _posterior_summary(candidate)
    d0_posterior = {
        "status": "NOT_REQUIRED_FOR_DECISION",
        "reason": (
            "D0 K=8 autoregressive sampling is not a continuous posterior and "
            "does not enter any frozen decision gate; deterministic K=1 is used "
            "for the matched baseline comparison"
        ),
    }
    posterior_gates = _posterior_gates(posterior)
    causal_gates = _causal_gates(causal)
    asr_gates = _asr_gates(asr)

    candidate_checkpoint = candidate["checkpoints"][0]
    closure_p11 = closure.get("p11") or {}
    closure_attribution = closure.get("attribution_contract") or {}
    closure_gates = {
        "schema_v3": (
            closure.get("schema")
            == "stable_audio_tools.p11_v4_frozen_p10_foa_closure"
            and int(closure.get("schema_version", -1)) == 3
        ),
        "status_pass": closure.get("status") == "PASS",
        "all_integrity_gates_pass": bool(closure.get("integrity_gates"))
        and all(bool(value) for value in closure["integrity_gates"].values()),
        "candidate_checkpoint_matches": closure_p11.get("checkpoint_sha256")
        == candidate_checkpoint.get("checkpoint_sha256"),
        "canonical_p10_v11_150k": (closure.get("p10") or {}).get(
            "checkpoint_sha256"
        )
        == CANONICAL_P10_SHA256,
        "p11_only_error_attribution": (
            closure_attribution.get("contract")
            == "p11_plan_error_with_frozen_p10_oracle_ceiling_v1"
            and closure_attribution.get("p10_checkpoint_and_sampler_fixed") is True
            and closure_attribution.get("same_p10_noise_seed_per_paired_plan")
            is True
            and closure_attribution.get("p10_oracle_plan_residual_is_p11_error")
            is False
            and closure_attribution.get("p10_improvement_is_out_of_scope") is True
        ),
    }
    reproducibility_gates = {
        "schema_v2": (
            repro.get("schema")
            == "stable_audio_tools.p11_v4_cross_process_repro"
            and int(repro.get("schema_version", -1)) == 2
        ),
        "status_pass": repro.get("status") == "PASS",
        "deterministic_reference_kernel": (
            (repro.get("reproducibility_gates") or {}).get(
                "deterministic_torch_reference_kernel"
            )
            is True
        ),
    }

    expected_pretrain_schemas = {
        "stable_audio_tools.p11_lexical_cache_audit",
        "stable_audio_tools.p11_v4_ddp_sampler_audit",
        "stable_audio_tools.p11_v4_contract_validation",
        "stable_audio_tools.p11_v4_sequence_budget",
        "stable_audio_tools.p11_v4_curriculum_validation",
        "stable_audio_tools.p11_v4_control_direction_gate",
        "stable_audio_tools.p11_v4_delta_owner_gate",
        "stable_audio_tools.p11_v4_graph_smoke",
    }
    pretrain_by_schema = {
        str(report.get("schema")): report for report in pretrain_reports
    }
    curriculum_gate = pretrain_by_schema.get(
        "stable_audio_tools.p11_v4_curriculum_validation", {}
    )
    contract_gate = pretrain_by_schema.get(
        "stable_audio_tools.p11_v4_contract_validation", {}
    )
    sequence_gate = pretrain_by_schema.get(
        "stable_audio_tools.p11_v4_sequence_budget", {}
    )
    lexical_gate = pretrain_by_schema.get(
        "stable_audio_tools.p11_lexical_cache_audit", {}
    )
    ddp_sampler_gate = pretrain_by_schema.get(
        "stable_audio_tools.p11_v4_ddp_sampler_audit", {}
    )
    graph_gate = pretrain_by_schema.get(
        "stable_audio_tools.p11_v4_graph_smoke", {}
    )
    control_gate = pretrain_by_schema.get(
        "stable_audio_tools.p11_v4_control_direction_gate", {}
    )
    delta_owner_gate = pretrain_by_schema.get(
        "stable_audio_tools.p11_v4_delta_owner_gate", {}
    )
    pretrain_gates = {
        "exact_report_inventory": len(pretrain_reports)
        == len(pretrain_by_schema)
        == len(expected_pretrain_schemas)
        and set(pretrain_by_schema) == expected_pretrain_schemas,
        "all_reports_pass": bool(pretrain_reports)
        and all(report.get("status") == "PASS" for report in pretrain_reports),
        "curriculum_rows_269568": int(curriculum_gate.get("rows", -1))
        == 269568,
        "curriculum_no_leakage": bool(curriculum_gate.get("leakage"))
        and all(int(value) == 0 for value in curriculum_gate["leakage"].values()),
        "curriculum_p10_v11_150k": (
            curriculum_gate.get("p10_alignment") or {}
        ).get("checkpoint_sha256")
        == CANONICAL_P10_SHA256,
        "contract_p10_v11_150k": (contract_gate.get("p10") or {}).get(
            "checkpoint_sha256"
        )
        == CANONICAL_P10_SHA256,
        "sequence_exact_full_scan": int(sequence_gate.get("rows", -1))
        == 269568
        and int(sequence_gate.get("violations", -1)) == 0
        and int(sequence_gate.get("truncation", -1)) == 0
        and sequence_gate.get("full_row_scan") is True
        and int(sequence_gate.get("validator_workers", -1)) == 8,
        "lexical_input_only_wer_bound": int(lexical_gate.get("rows", -1))
        == 10000
        and float(lexical_gate.get("reliable_true_speech_word_error_rate", 1.0))
        <= 0.15
        and (lexical_gate.get("input_boundary") or {}).get(
            "target_transcript_access"
        )
        == "forbidden",
        "ddp8_rank_balanced_pair_layout": (
            ddp_sampler_gate.get("status") == "PASS"
            and ddp_sampler_gate.get("ordering_contract")
            == "p11_v4_ddp8_rank_balanced_pair_aware_batch8_v7"
            and int(ddp_sampler_gate.get("rows", -1)) == 269568
            and int(ddp_sampler_gate.get("world_size", -1)) == 8
            and int(ddp_sampler_gate.get("local_batch_size", -1)) == 8
            and int(ddp_sampler_gate.get("global_batch_size", -1)) == 64
            and int(ddp_sampler_gate.get("local_batches", -1))
            == int(ddp_sampler_gate.get("local_batches_with_all_tasks", -2))
            == int(
                ddp_sampler_gate.get(
                    "local_batches_with_complete_pair", -3
                )
            )
            and int(ddp_sampler_gate.get("local_batches_with_split_pair", -1))
            == 0
            and all(
                counts
                == {
                    "generation": 11232,
                    "understanding": 11232,
                    "editing": 11232,
                }
                for counts in ddp_sampler_gate.get("rank_task_counts", [])
            )
            and len(ddp_sampler_gate.get("rank_task_counts", [])) == 8
        ),
        "control_direction_complete": int(
            (control_gate.get("coverage") or {}).get("rows", -1)
        )
        == 269568
        and (control_gate.get("coverage") or {}).get(
            "population_has_both_directions_per_operation"
        )
        is True,
        "delta_owner_active": int(delta_owner_gate.get("rows", -1))
        == 269568
        and int(delta_owner_gate.get("owner_rows", 0)) > 0
        and int(delta_owner_gate.get("train_inference_legal_set_mismatches", -1))
        == 0,
        "real_graph_batch8_finite_optimizer_ema": int(
            graph_gate.get("batch_size", -1)
        )
        == 8
        and graph_gate.get("all_gradients_finite") is True
        and graph_gate.get("optimizer_step") == "PASS"
        and int((graph_gate.get("ema_step") or {}).get("after", 0))
        > int((graph_gate.get("ema_step") or {}).get("before", 0)),
    }

    utilization_batches = training.get("gpu_utilization_sample_batches") or []
    measured_steps = int(training.get("measured_throughput_steps", -1))
    expected_utilization_batches = (
        _uniform_sample_batches(measured_steps) if measured_steps > 0 else []
    )
    launch_training = launch.get("training") or {}
    launch_model = launch.get("model_config") or {}
    launch_curriculum = launch.get("curriculum") or {}
    checkpoint_path = Path(
        str(candidate_checkpoint.get("checkpoint", ""))
    ).expanduser().resolve(strict=True)
    threshold_rule_source = FROZEN_THRESHOLD_RULE_SOURCE.resolve(strict=True)
    if _sha256_file(threshold_rule_source) != FROZEN_THRESHOLD_RULE_SHA256:
        raise RuntimeError("frozen pre-checkpoint threshold rule snapshot changed")
    protocol_correction_source = Path(__file__).resolve(strict=True)
    threshold_rule_predates_checkpoint = (
        threshold_rule_source.stat().st_mtime_ns
        < checkpoint_path.stat().st_mtime_ns
    )
    operational_gates = {
        "training_report_pass": training.get("status") == "PASS",
        "training_report_v4": training.get("audit")
        == "sceneplan_p11_gpu_preflight_v5",
        "world_size_8": int(training.get("world_size", -1)) == 8,
        "batch_size_per_gpu_8": int(training.get("batch_size_per_gpu", -1)) == 8,
        "runtime_rank_local_gue_balance": (
            training.get("gue_each_local_batch_required") is True
            and training.get("sampled_curriculum_batches_all_gue") is True
            and int(training.get("rank_curriculum_gue_batch_count_min", -1))
            == 8
            and int(training.get("rank_curriculum_gue_batch_count_max", -1))
            == 8
            and len(training.get("rank_curriculum_task_counts", [])) == 8
            and all(
                set(counts) == {"generation", "understanding", "editing"}
                and all(int(value) > 0 for value in counts.values())
                for counts in training.get("rank_curriculum_task_counts", [])
            )
        ),
        "measured_steps_9992": measured_steps == 9992,
        "full_window_utilization_sampling": training.get(
            "gpu_utilization_sampling_contract"
        )
        == "uniform_over_measured_window_v1"
        and utilization_batches == expected_utilization_batches,
        "throughput_floor": float(
            training.get("global_loader_samples_per_second", 0.0)
        )
        >= FROZEN_THRESHOLDS["minimum_global_loader_samples_per_second"],
        "rank_utilization_floor": float(
            training.get("rank_gpu_utilization_percent_mean_min", 0.0)
        )
        >= FROZEN_THRESHOLDS[
            "minimum_rank_mean_gpu_utilization_percent"
        ],
        "memory_graph_floor": float(training.get("peak_allocated_gib", 0.0))
        >= FROZEN_THRESHOLDS["minimum_peak_allocated_gib"],
        "launch_contract_v2": int(launch.get("schema_version", -1)) == 2,
        "launch_profile_scale_trial": launch.get("profile") == "scale_trial",
        "launch_seed_42": int(launch_training.get("seed", -1)) == 42,
        "launch_global_batch_64": int(
            launch_training.get("global_batch_size", -1)
        )
        == 64,
        "launch_curriculum_rows_269568": int(
            launch_training.get("effective_curriculum_rows", -1)
        )
        == 269568,
        "launch_ddp8_ordering_v7": launch_curriculum.get(
            "ordering_contract"
        )
        == "p11_v4_ddp8_rank_balanced_pair_aware_batch8_v7"
        and int(launch_curriculum.get("ordering_world_size", -1)) == 8
        and int(launch_curriculum.get("ordering_global_batch_size", -1)) == 64
        and (launch_curriculum.get("ddp_layout_audit") or {}).get(
            "rank_task_counts_identical"
        )
        is True,
        "launch_sample_exposures_640000": int(
            launch_training.get("sample_exposures", -1)
        )
        == 640000,
        "candidate_step_10000": int(candidate_checkpoint.get("global_step", -1))
        == 10000,
        "candidate_belongs_to_launch_run": len(checkpoint_path.parents) >= 2
        and checkpoint_path.parents[1].name == launch.get("run_name"),
        "launch_model_matches_candidate": launch_model.get("resolved_sha256")
        == candidate_checkpoint.get("model_config_resolved_sha256"),
        "decision_threshold_rule_predates_candidate_checkpoint": (
            threshold_rule_predates_checkpoint
        ),
    }

    runtime_model = load_config(Path(str(candidate["model_config"])))
    transfusion = runtime_model["model"]["transfusion_cot"]
    boundary_gates = {
        "flow_g_u": transfusion["thought"]["continuous_objective"]
        == "rectified_flow",
        "direct_delta_e": transfusion["thought"]["editing_continuous_objective"]
        == "direct_mse",
        "semantic_not_owned_by_execution": transfusion.get(
            "semantic_from_execution_state"
        )
        is False,
        "numeric_not_owned_by_sketch": transfusion.get("numeric_from_scene_sketch")
        is False,
        "deterministic_assembler": transfusion.get("assembler")
        == "deterministic_sketch_execution_assembler_v1",
    }

    integrity_gate_groups = (
        comparability_gates,
        pretrain_gates,
        closure_gates,
        reproducibility_gates,
        operational_gates,
        boundary_gates,
    )
    evidence_integrity = all(
        all(bool(value) for value in group.values())
        for group in integrity_gate_groups
    )
    unique_capability = all(posterior_gates.values()) and all(
        causal_gates.values()
    )
    lexical_authority = all(asr_gates.values())

    baselines = {"d0": d0_quality, "direct": direct_quality}
    best_baseline_name = max(
        baselines, key=lambda name: baselines[name]["gue_macro_k1"]
    )
    best_baseline_macro = baselines[best_baseline_name]["gue_macro_k1"]
    macro_gap = candidate_quality["gue_macro_k1"] - best_baseline_macro
    macro_scale_gain = (
        candidate_quality["gue_macro_k1"] - screen_quality["gue_macro_k1"]
    )
    gu_scale_gain = sum(
        candidate_quality["tasks_k1"][task]
        - screen_quality["tasks_k1"][task]
        for task in ("generation", "understanding")
    ) / 2.0
    editing_scale_delta = (
        candidate_quality["tasks_k1"]["editing"]
        - screen_quality["tasks_k1"]["editing"]
    )
    near_best = macro_gap >= -FROZEN_THRESHOLDS["overall_tie_absolute"]
    scale_response = max(macro_scale_gain, gu_scale_gain) >= FROZEN_THRESHOLDS[
        "material_scale_gain"
    ]
    editing_stable = editing_scale_delta >= -FROZEN_THRESHOLDS[
        "editing_regression_tolerance"
    ]
    usable = candidate_quality["row_weighted_k1"] >= FROZEN_THRESHOLDS[
        "minimum_usable_row_weighted_k1"
    ]

    if not evidence_integrity:
        threshold_decision = "STOP_INVALID_OR_INCOMPLETE_EVIDENCE"
    elif not unique_capability:
        threshold_decision = "STOP_FLOW_UNIQUE_CAPABILITY_NOT_PRESERVED"
    elif not lexical_authority:
        threshold_decision = "REVISE_RELIABLE_ASR_BOUNDARY"
    elif near_best and scale_response and editing_stable:
        threshold_decision = "GO_HYBRID_FLOW_GU_DIRECT_E_CANONICAL"
    elif usable and scale_response and editing_stable:
        threshold_decision = "REVISE_FLOW_VALID_BUT_NOT_YET_BASELINE_COMPETITIVE"
    elif usable and gu_scale_gain >= FROZEN_THRESHOLDS[
        "material_scale_gain"
    ]:
        threshold_decision = "REVISE_DIRECT_DELTA_E_SUPERVISION"
    else:
        threshold_decision = "STOP_MEDIUM_SCALE_RESPONSE_NOT_ESTABLISHED"

    # The current candidate adds U finite-inventory supervision to the shared
    # SceneSketch decoder.  The historical Direct report predates that repair,
    # so it remains a useful reference but cannot by itself close the final
    # Flow-vs-Direct method comparison.  This provenance guard changes no
    # frozen quality threshold and was defined before the candidate checkpoint.
    candidate_inventory = (
        (transfusion.get("discrete_supervision") or {}).get(
            "understanding_inventory"
        )
        or {}
    )
    direct_runtime_path = Path(str(direct["model_config"])).resolve(strict=True)
    direct_runtime = load_config(direct_runtime_path)
    direct_transfusion = direct_runtime["model"]["transfusion_cot"]
    direct_inventory = (
        (direct_transfusion.get("discrete_supervision") or {}).get(
            "understanding_inventory"
        )
        or {}
    )
    candidate_runtime_resolved_sha256 = _json_sha256(runtime_model)
    candidate_report_uses_current_recipe = (
        candidate_checkpoint.get("model_config_resolved_sha256")
        == candidate_runtime_resolved_sha256
    )
    current_direct_resolved_sha256 = _json_sha256(direct_runtime)
    direct_checkpoint = direct["checkpoints"][0]
    screen_checkpoint = screen["checkpoints"][0]
    candidate_inventory_enabled = (
        candidate_report_uses_current_recipe
        and candidate_inventory.get("contract")
        == "u_same_decoder_finite_inventory_aux_v1"
        and candidate_inventory.get("objective") == "grammar_candidate_ce_v1"
        and candidate_inventory.get("authority")
        == "scene_sketch_autoregressive_logits_v1"
    )
    direct_current_recipe_inherits_same_inventory = (
        bool(candidate_inventory) and direct_inventory == candidate_inventory
    )
    direct_report_uses_current_recipe = (
        direct_checkpoint.get("model_config_resolved_sha256")
        == current_direct_resolved_sha256
    )
    matched_direct_confirmation_required = (
        candidate_inventory_enabled
        and direct_current_recipe_inherits_same_inventory
        and not direct_report_uses_current_recipe
    )
    if (
        threshold_decision == "GO_HYBRID_FLOW_GU_DIRECT_E_CANONICAL"
        and matched_direct_confirmation_required
    ):
        decision = "GO_MATCHED_DIRECT_INVENTORY_CONFIRMATION"
    else:
        decision = threshold_decision
    canonical_promotion_authorized = (
        decision == "GO_HYBRID_FLOW_GU_DIRECT_E_CANONICAL"
    )

    report: dict[str, Any] = {
        "schema": "stable_audio_tools.p11_v4_medium_scale_decision",
        "schema_version": 4,
        "status": "PASS",
        "decision": decision,
        "frozen_threshold_decision": threshold_decision,
        "canonical_promotion_authorized": canonical_promotion_authorized,
        "attribution_contract": {
            "contract": P11_ATTRIBUTION_CONTRACT,
            "selection_target": "P11_planner_only",
            "primary_selection_objects": [
                "SceneSketch_or_DeltaSketch",
                "ExecutionState_or_DeltaThought",
                "AtomicPatch_and_assembled_ScenePlan",
            ],
            "primary_selection_evidence": "pre_render_P11_quality_and_posterior_metrics",
            "p10_role": "fixed_oracle_executor_ceiling",
            "p10_checkpoint_sha256": CANONICAL_P10_SHA256,
            "p10_improvement_is_out_of_scope": True,
            "p10_oracle_plan_rendering_residual_is_p11_error": False,
            "p10_closure_role": (
                "secondary integration and causal-consequence evidence only"
            ),
            "audio_regret_interpretation": (
                "difference between target-plan and predicted-plan renders under "
                "the same frozen P10 checkpoint, sampler, and seed; never an "
                "independent P11 waveform-quality objective"
            ),
        },
        "decision_thresholds_frozen_before_candidate_results": (
            threshold_rule_predates_checkpoint
        ),
        "frozen_threshold_rule_source": {
            "path": str(threshold_rule_source),
            "bytes": threshold_rule_source.stat().st_size,
            "sha256": _sha256_file(threshold_rule_source),
            "mtime_ns": threshold_rule_source.stat().st_mtime_ns,
            "thresholds_sha256": _json_sha256(FROZEN_THRESHOLDS),
        },
        "evaluation_protocol_correction_source": {
            "path": str(protocol_correction_source),
            "bytes": protocol_correction_source.stat().st_size,
            "sha256": _sha256_file(protocol_correction_source),
            "mtime_ns": protocol_correction_source.stat().st_mtime_ns,
            "change_boundary": (
                "deterministic evaluator kernel, own-ScenePlan P10 render-length "
                "integrity, same-physical-GPU ASR pairing, and frozen prediction-"
                "artifact closure and matched-supervision promotion scope; "
                "core quality thresholds and threshold branches unchanged"
            ),
        },
        "thresholds": FROZEN_THRESHOLDS,
        "architecture_choice": "shared_qwen_plus_flow_g_u_plus_direct_delta_e",
        "quality": {
            "candidate_medium_flow": candidate_quality,
            "screen_flow": screen_quality,
            "d0": d0_quality,
            "direct": direct_quality,
            "direct_comparison_scope": "historical_pre_inventory_repair",
            "best_baseline": best_baseline_name,
            "candidate_macro_gap_vs_best_baseline": macro_gap,
            "candidate_macro_gain_vs_screen_flow": macro_scale_gain,
            "candidate_g_u_gain_vs_screen_flow": gu_scale_gain,
            "candidate_editing_delta_vs_screen_flow": editing_scale_delta,
        },
        "posterior": {
            "candidate": posterior,
            "d0": d0_posterior,
        },
        "decision_terms": {
            "evidence_integrity": evidence_integrity,
            "unique_capability": unique_capability,
            "reliable_asr_boundary": lexical_authority,
            "near_best_baseline": near_best,
            "material_scale_response": scale_response,
            "editing_stable": editing_stable,
            "minimum_usable_quality": usable,
            "matched_direct_confirmation_required": (
                matched_direct_confirmation_required
            ),
            "canonical_promotion_authorized": canonical_promotion_authorized,
        },
        "baseline_supervision_comparability": {
            "contract": "p11_u_inventory_repair_isolation_then_matched_method_ab_v1",
            "candidate_inventory": candidate_inventory,
            "candidate_checkpoint_model_config_resolved_sha256": (
                candidate_checkpoint.get("model_config_resolved_sha256")
            ),
            "candidate_runtime_model_config_resolved_sha256": (
                candidate_runtime_resolved_sha256
            ),
            "candidate_report_uses_current_recipe": (
                candidate_report_uses_current_recipe
            ),
            "screen_flow_checkpoint_model_config_resolved_sha256": (
                screen_checkpoint.get("model_config_resolved_sha256")
            ),
            "screen_flow_is_pre_inventory_repair": (
                screen_checkpoint.get("model_config_resolved_sha256")
                != candidate_checkpoint.get("model_config_resolved_sha256")
            ),
            "direct_current_recipe": {
                "path": str(direct_runtime_path),
                "resolved_sha256": current_direct_resolved_sha256,
                "inventory": direct_inventory,
            },
            "direct_historical_report": {
                "checkpoint": direct_checkpoint.get("checkpoint"),
                "checkpoint_sha256": direct_checkpoint.get("checkpoint_sha256"),
                "model_config_resolved_sha256": direct_checkpoint.get(
                    "model_config_resolved_sha256"
                ),
            },
            "direct_current_recipe_inherits_same_inventory": (
                direct_current_recipe_inherits_same_inventory
            ),
            "direct_report_uses_current_recipe": direct_report_uses_current_recipe,
            "matched_direct_confirmation_required": (
                matched_direct_confirmation_required
            ),
            "interpretation": (
                "the candidate report predates the current inventory recipe, "
                "so this repair-comparability classification does not apply"
                if not candidate_report_uses_current_recipe
                else "the historical Direct result remains a reference; a new "
                "seed-42 Direct 10k run with the identical U inventory objective "
                "is required before final Flow-vs-Direct canonical promotion"
                if matched_direct_confirmation_required
                else "the Direct comparison uses the current supervision recipe"
            ),
        },
        "gates": {
            "comparability": comparability_gates,
            "pretrain": pretrain_gates,
            "posterior": posterior_gates,
            "causal": causal_gates,
            "asr": asr_gates,
            "frozen_p10_closure": closure_gates,
            "cross_process_reproducibility": reproducibility_gates,
            "operational": operational_gates,
            "architecture_boundary": boundary_gates,
        },
        "remaining_risks": [
            "medium curriculum is synthetic representation stress, not real acoustic corruption",
            "the 269568-row trial is not evidence for a ten-million-row full-corpus optimum",
            "one fixed seed measures the canonical recipe but not seed variance",
            "the historical Direct checkpoint predates the U inventory repair and is not the final matched-supervision comparator",
            "P10 closure is only a fixed oracle-executor consequence of P11 plans; its residual quality is not attributed to P11",
            "P10 closure re-renders edits and does not claim waveform-local preservation",
        ],
        "next_action": {
            "GO_MATCHED_DIRECT_INVENTORY_CONFIRMATION": (
                "retain the positive Flow repair signal and run one seed-42, "
                "10k-step Direct-MSE comparison with the identical U inventory "
                "objective before any canonical/full-corpus promotion"
            ),
            "GO_HYBRID_FLOW_GU_DIRECT_E_CANONICAL": (
                "freeze this medium recipe and construct, but do not silently launch, "
                "the versioned approximately ten-million-row corpus"
            ),
            "REVISE_FLOW_VALID_BUT_NOT_YET_BASELINE_COMPETITIVE": (
                "retain Flow-R1 unique capability and revise G/U supervision before full scale"
            ),
            "REVISE_DIRECT_DELTA_E_SUPERVISION": (
                "keep Flow G/U and repair the already-direct DeltaThought E route"
            ),
            "REVISE_RELIABLE_ASR_BOUNDARY": (
                "retain Flow-R1 and repair reliable-speech lexical injection before full scale"
            ),
            "STOP_FLOW_UNIQUE_CAPABILITY_NOT_PRESERVED": (
                "retain D0 as accuracy baseline and archive Flow promotion"
            ),
            "STOP_MEDIUM_SCALE_RESPONSE_NOT_ESTABLISHED": (
                "stop scale-up and diagnose supervision/optimization response"
            ),
            "STOP_INVALID_OR_INCOMPLETE_EVIDENCE": (
                "repair the failed evidence artifact before any scientific conclusion"
            ),
        }[decision],
        "invocation": shlex.join([sys.executable, *sys.argv]),
        "inputs": {
            name: _record(path, value) for name, (path, value) in inputs.items()
        },
    }
    report["report_sha256_without_self"] = _json_sha256(report)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
