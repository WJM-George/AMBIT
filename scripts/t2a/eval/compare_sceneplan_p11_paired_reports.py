#!/usr/bin/env python3
"""Paired, contract-audited comparison of two ScenePlan P11 eval reports.

This script deliberately separates two questions:

1. Are the evaluation rows and contracts paired closely enough to compare the
   *existing checkpoints*?
2. Were the checkpoints trained under a matched recipe, so the comparison can
   be interpreted as an architecture A/B?

The second answer defaults to ``False`` and must never be inferred merely from
matching evaluation manifests.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


COMPARISON_CONTRACT_KEYS = (
    "evaluator",
    "field_metric_contract",
    "generation_constraint_contract",
    "manifest",
    "manifest_version",
    "codec_fingerprint",
    "decode_output_contract",
)

LOWER_IS_BETTER = {
    "azimuth_mae_deg",
    "distance_mae_m",
    "distance_relative_mae",
    "elevation_mae_deg",
    "offset_mae_frames",
    "onset_mae_frames",
}

METRICS_BY_TASK = {
    "generation": (
        "task_score",
        "constraint_duration_score",
        "constraint_room_score",
        "constraint_source_semantics_score",
        "constraint_temporal_score",
        "constraint_geometry_score",
        "constraint_gain_score",
        "source_count_accuracy",
        "semantic_token_f1",
        "activity_iou",
        "motion_type_accuracy",
        "azimuth_mae_deg",
        "elevation_mae_deg",
        "distance_mae_m",
        "onset_mae_frames",
        "offset_mae_frames",
    ),
    "understanding": (
        "task_score",
        "source_assignment_f1",
        "source_count_accuracy",
        "kind_accuracy",
        "semantic_token_f1",
        "activity_iou",
        "motion_type_accuracy",
        "room_accuracy",
        "azimuth_mae_deg",
        "elevation_mae_deg",
        "distance_mae_m",
        "onset_mae_frames",
        "offset_mae_frames",
    ),
    "editing": (
        "task_score",
        "edit_exact",
        "edit_success",
        "preservation_accuracy",
        "preservation_exact",
        "activity_iou",
        "onset_mae_frames",
        "offset_mae_frames",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-name", default="discrete_v0")
    parser.add_argument("--candidate-name", default="transfusion_v3")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_901)
    parser.add_argument(
        "--training-lineage-matched",
        action="store_true",
        help="Only set for a controlled A/B with the same train rows, optimizer budget, and recipe.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def percentile_interval(values: np.ndarray) -> list[float]:
    low, high = np.quantile(values, (0.025, 0.975))
    return [float(low), float(high)]


def harmonic(values: Iterable[float]) -> float:
    values = list(values)
    if any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def paired_summary(
    baseline: np.ndarray,
    candidate: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
    *,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired arrays must be one-dimensional and equally sized")
    if baseline.size == 0:
        raise ValueError("paired arrays must be non-empty")
    raw_delta = candidate - baseline
    favorable_delta = -raw_delta if lower_is_better else raw_delta
    sample_indices = rng.integers(0, baseline.size, size=(replicates, baseline.size))
    bootstrap = favorable_delta[sample_indices].mean(axis=1)
    tolerance = 1e-12
    return {
        "n": int(baseline.size),
        "baseline_mean": float(baseline.mean()),
        "candidate_mean": float(candidate.mean()),
        "raw_delta_candidate_minus_baseline": float(raw_delta.mean()),
        "favorable_delta_for_candidate": float(favorable_delta.mean()),
        "favorable_delta_ci95_paired_bootstrap": percentile_interval(bootstrap),
        "probability_candidate_favorable_bootstrap": float(np.mean(bootstrap > 0.0)),
        "candidate_win_rate": float(np.mean(favorable_delta > tolerance)),
        "tie_rate": float(np.mean(np.abs(favorable_delta) <= tolerance)),
        "candidate_loss_rate": float(np.mean(favorable_delta < -tolerance)),
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
    }


def load_manifest(path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, str]]:
    connection = sqlite3.connect(str(path))
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        rows = connection.execute(
            """
            SELECT ordinal, task, prompt, edit_kind, edit_spec_json, target_sceneplan_zlib
            FROM rows ORDER BY ordinal
            """
        ).fetchall()
    finally:
        connection.close()

    result: dict[int, dict[str, Any]] = {}
    for ordinal, task, prompt, edit_kind, edit_spec_json, compressed_plan in rows:
        plan = json.loads(zlib.decompress(compressed_plan))
        sources = plan.get("sources", [])
        linear_count = sum(
            source.get("trajectory", {}).get("type") == "linear" for source in sources
        )
        source_count = len(sources)
        if source_count == 1:
            source_complexity = "1_source"
        elif source_count == 2:
            source_complexity = "2_sources"
        else:
            source_complexity = "3_4_sources"
        duration = float(plan["duration_sec"])
        if duration <= 7.5:
            duration_bin = "about_5s"
        elif duration <= 12.5:
            duration_bin = "about_10s"
        else:
            duration_bin = "about_15s"
        result[int(ordinal)] = {
            "task": task,
            "prompt": prompt,
            "edit_kind": edit_kind,
            "edit_spec": json.loads(edit_spec_json) if edit_spec_json else None,
            "source_count": source_count,
            "source_complexity": source_complexity,
            "motion_profile": "has_linear" if linear_count else "static_only",
            "linear_source_count": linear_count,
            "speech_presence": (
                "has_speech" if any(source.get("kind") == "speech" for source in sources) else "no_speech"
            ),
            "duration_sec": duration,
            "duration_bin": duration_bin,
        }
    return result, metadata


def row_map(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = report.get("row_results")
    if not isinstance(rows, list):
        raise ValueError("report row_results must be a list")
    mapped = {int(row["ordinal"]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError("duplicate ordinals in row_results")
    return mapped


def eligible_for_promotion(row: dict[str, Any]) -> bool:
    return row["task"] != "editing" or row.get("edit_kind") != "no_op"


def metric_pairs(
    ordinals: Iterable[int],
    baseline_rows: dict[int, dict[str, Any]],
    candidate_rows: dict[int, dict[str, Any]],
    metric: str,
) -> tuple[np.ndarray, np.ndarray]:
    baseline_values: list[float] = []
    candidate_values: list[float] = []
    for ordinal in ordinals:
        baseline_metric = baseline_rows[ordinal].get("metrics", {}).get(metric)
        candidate_metric = candidate_rows[ordinal].get("metrics", {}).get(metric)
        if baseline_metric is None or candidate_metric is None:
            continue
        baseline_values.append(float(baseline_metric))
        candidate_values.append(float(candidate_metric))
    return np.asarray(baseline_values), np.asarray(candidate_values)


def audit_contracts(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_rows: dict[int, dict[str, Any]],
    candidate_rows: dict[int, dict[str, Any]],
    manifest_rows: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    contract_matches = {
        key: {
            "match": baseline.get(key) == candidate.get(key),
            "baseline": baseline.get(key),
            "candidate": candidate.get(key),
        }
        for key in COMPARISON_CONTRACT_KEYS
    }
    baseline_ordinals = sorted(baseline_rows)
    candidate_ordinals = sorted(candidate_rows)
    ordinal_match = baseline_ordinals == candidate_ordinals
    manifest_match = baseline_ordinals == sorted(manifest_rows)
    identity_mismatches: list[dict[str, Any]] = []
    if ordinal_match:
        for ordinal in baseline_ordinals:
            left = baseline_rows[ordinal]
            right = candidate_rows[ordinal]
            for key in ("task", "sample_id", "prompt_view_id", "edit_kind"):
                if left.get(key) != right.get(key):
                    identity_mismatches.append(
                        {
                            "ordinal": ordinal,
                            "field": key,
                            "baseline": left.get(key),
                            "candidate": right.get(key),
                        }
                    )
    pass_value = (
        all(item["match"] for item in contract_matches.values())
        and ordinal_match
        and manifest_match
        and not identity_mismatches
        and bool(baseline.get("strict_production_pass"))
        and bool(candidate.get("strict_production_pass"))
    )
    return {
        "pass": pass_value,
        "contract_matches": contract_matches,
        "ordinal_match": ordinal_match,
        "manifest_ordinal_match": manifest_match,
        "identity_mismatches": identity_mismatches,
        "baseline_strict_production_pass": bool(baseline.get("strict_production_pass")),
        "candidate_strict_production_pass": bool(candidate.get("strict_production_pass")),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_replicates < 1_000:
        raise ValueError("use at least 1,000 bootstrap replicates")

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    baseline_rows = row_map(baseline)
    candidate_rows = row_map(candidate)
    manifest_rows, manifest_metadata = load_manifest(args.manifest)
    audit = audit_contracts(
        baseline, candidate, baseline_rows, candidate_rows, manifest_rows
    )
    if not audit["pass"]:
        raise RuntimeError("evaluation reports are not paired under the required contracts")

    rng = np.random.default_rng(args.seed)
    task_ordinals: dict[str, list[int]] = {}
    task_summaries: dict[str, dict[str, Any]] = {}
    task_bootstrap_means: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for task in ("generation", "understanding", "editing"):
        ordinals = [
            ordinal
            for ordinal in sorted(baseline_rows)
            if baseline_rows[ordinal]["task"] == task
            and eligible_for_promotion(baseline_rows[ordinal])
        ]
        task_ordinals[task] = ordinals
        baseline_scores, candidate_scores = metric_pairs(
            ordinals, baseline_rows, candidate_rows, "task_score"
        )
        task_summaries[task] = paired_summary(
            baseline_scores,
            candidate_scores,
            rng,
            args.bootstrap_replicates,
        )
        sampled = rng.integers(
            0, baseline_scores.size, size=(args.bootstrap_replicates, baseline_scores.size)
        )
        task_bootstrap_means[task] = (
            baseline_scores[sampled].mean(axis=1),
            candidate_scores[sampled].mean(axis=1),
        )

    baseline_h = np.empty(args.bootstrap_replicates)
    candidate_h = np.empty(args.bootstrap_replicates)
    for index in range(args.bootstrap_replicates):
        baseline_h[index] = harmonic(
            task_bootstrap_means[task][0][index]
            for task in ("generation", "understanding", "editing")
        )
        candidate_h[index] = harmonic(
            task_bootstrap_means[task][1][index]
            for task in ("generation", "understanding", "editing")
        )
    harmonic_delta = candidate_h - baseline_h

    promotion_baseline = baseline["promotion"]
    promotion_candidate = candidate["promotion"]
    score_key = {
        "generation": "generation",
        "understanding": "understanding",
        "editing": "editing_non_noop",
    }
    for task, summary in task_summaries.items():
        expected_baseline = promotion_baseline["task_scores"][score_key[task]]
        expected_candidate = promotion_candidate["task_scores"][score_key[task]]
        if not math.isclose(summary["baseline_mean"], expected_baseline, abs_tol=1e-10):
            raise RuntimeError(f"baseline {task} mean does not reproduce promotion score")
        if not math.isclose(summary["candidate_mean"], expected_candidate, abs_tol=1e-10):
            raise RuntimeError(f"candidate {task} mean does not reproduce promotion score")

    metric_summaries: dict[str, dict[str, Any]] = {}
    for task, metrics in METRICS_BY_TASK.items():
        metric_summaries[task] = {}
        for metric in metrics:
            baseline_values, candidate_values = metric_pairs(
                task_ordinals[task], baseline_rows, candidate_rows, metric
            )
            if baseline_values.size == 0:
                continue
            metric_summaries[task][metric] = paired_summary(
                baseline_values,
                candidate_values,
                rng,
                args.bootstrap_replicates,
                lower_is_better=metric in LOWER_IS_BETTER,
            )

    dimensions = (
        "source_count",
        "source_complexity",
        "linear_source_count",
        "motion_profile",
        "speech_presence",
        "duration_bin",
        "prompt_view_id",
        "edit_kind",
    )
    stratified: dict[str, dict[str, dict[str, Any]]] = {}
    for task, ordinals in task_ordinals.items():
        stratified[task] = {}
        for dimension in dimensions:
            grouped: dict[str, list[int]] = defaultdict(list)
            for ordinal in ordinals:
                if dimension in manifest_rows[ordinal]:
                    value = manifest_rows[ordinal][dimension]
                else:
                    value = baseline_rows[ordinal].get(dimension)
                if value is not None:
                    grouped[str(value)].append(ordinal)
            if not grouped:
                continue
            stratified[task][dimension] = {}
            for value, group_ordinals in sorted(grouped.items()):
                baseline_values, candidate_values = metric_pairs(
                    group_ordinals, baseline_rows, candidate_rows, "task_score"
                )
                stratified[task][dimension][value] = paired_summary(
                    baseline_values,
                    candidate_values,
                    rng,
                    args.bootstrap_replicates,
                )

    report_task_scores = {
        "baseline": promotion_baseline["task_scores"],
        "candidate": promotion_candidate["task_scores"],
        "delta_candidate_minus_baseline": {
            key: float(
                promotion_candidate["task_scores"][key]
                - promotion_baseline["task_scores"][key]
            )
            for key in promotion_baseline["task_scores"]
        },
    }
    existing_checkpoint_comparison = {
        "task_scores": report_task_scores,
        "harmonic_gue": {
            "baseline": float(promotion_baseline["harmonic_gue_score"]),
            "candidate": float(promotion_candidate["harmonic_gue_score"]),
            "delta_candidate_minus_baseline": float(
                promotion_candidate["harmonic_gue_score"]
                - promotion_baseline["harmonic_gue_score"]
            ),
            "delta_ci95_paired_stratified_bootstrap": percentile_interval(harmonic_delta),
            "probability_candidate_superior_bootstrap": float(np.mean(harmonic_delta > 0.0)),
        },
        "paired_task_score": task_summaries,
        "paired_metrics": metric_summaries,
        "stratified_task_score": stratified,
    }

    candidate_mechanism = {
        "strict_production_pass": bool(candidate.get("strict_production_pass")),
        "thought_intervention_audit": candidate.get("thought_intervention_audit"),
        "learned_thought_intervention_gate": candidate.get(
            "learned_thought_intervention_gate"
        ),
        "scene_thought": candidate.get("scene_thought"),
    }

    conclusion = {
        "mechanism_status": "supported_by_existing_interventions",
        "existing_checkpoint_accuracy_status": (
            "candidate_below_baseline"
            if existing_checkpoint_comparison["harmonic_gue"][
                "delta_candidate_minus_baseline"
            ]
            < 0
            else "candidate_not_below_baseline"
        ),
        "architecture_superiority_status": (
            "eligible_for_causal_interpretation"
            if args.training_lineage_matched
            else "not_established_training_lineage_unmatched"
        ),
        "full_scale_training_recommendation": "do_not_start",
    }

    output = {
        "schema": "stable_audio_tools.sceneplan_p11_paired_checkpoint_comparison",
        "schema_version": 1,
        "baseline": {
            "name": args.baseline_name,
            "report": str(args.baseline),
            "checkpoint": baseline.get("checkpoint"),
            "checkpoint_step": baseline.get("checkpoint_step"),
        },
        "candidate": {
            "name": args.candidate_name,
            "report": str(args.candidate),
            "checkpoint": candidate.get("checkpoint"),
            "checkpoint_step": candidate.get("checkpoint_step"),
        },
        "manifest": {
            "path": str(args.manifest),
            "rows": len(manifest_rows),
            "metadata": manifest_metadata,
        },
        "comparability": {
            "evaluation_layer_matched": audit["pass"],
            "training_lineage_matched": args.training_lineage_matched,
            "interpretation": (
                "controlled_architecture_ab"
                if args.training_lineage_matched
                else "existing_checkpoint_diagnostic_only"
            ),
            "audit": audit,
        },
        "bootstrap": {
            "method": "paired percentile bootstrap; harmonic G/U/E resampled within task",
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "existing_checkpoint_comparison": existing_checkpoint_comparison,
        "candidate_mechanism_evidence": candidate_mechanism,
        "conclusion": conclusion,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(conclusion, ensure_ascii=False, sort_keys=True))
    print(json.dumps(existing_checkpoint_comparison["harmonic_gue"], sort_keys=True))
    print(args.output)


if __name__ == "__main__":
    main()
