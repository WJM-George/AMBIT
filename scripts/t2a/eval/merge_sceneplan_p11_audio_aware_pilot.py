#!/usr/bin/env python3
"""Merge disjoint learned P11 pilot-decode workers into one auditable gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean_metric(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [
        float(row["metrics"][name])
        for row in rows
        if name in (row.get("metrics") or {})
    ]
    return sum(values) / len(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=90)
    args = parser.parse_args()
    paths = [path.expanduser().resolve(strict=True) for path in args.inputs]
    if args.expected_rows <= 0:
        raise ValueError("expected rows must be positive")

    reports: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("schema") != "stable_audio_tools.p11_audio_aware_learned_pilot_gate":
            raise ValueError(f"{path} has the wrong learned-gate schema")
        if report.get("status") != "PASS" or not report.get("explicit_ordinal_worker"):
            raise ValueError(f"{path} is not a passing explicit-ordinal worker")
        reports.append(report)
        rows.extend(report["rows"])
    checkpoints = {report["checkpoint_sha256"] for report in reports}
    weights = {report["weights"] for report in reports}
    decode_modes = {report["decode_mode"] for report in reports}
    seeds = {int(report["seed"]) for report in reports}
    evaluator_versions = {int(report.get("version", -1)) for report in reports}
    metric_contracts = {report.get("metric_contract") for report in reports}
    ordinals = [int(row["ordinal"]) for row in rows]
    expected_ordinals = list(range(args.expected_rows))
    if (
        len(checkpoints) != 1
        or len(weights) != 1
        or decode_modes != {"prefix_recompute"}
        or seeds != {42}
        or evaluator_versions != {3}
        or metric_contracts != {"observation_conditional_edit_end_to_end_v2"}
    ):
        raise RuntimeError(
            "pilot workers used inconsistent checkpoint/weights/seed/evaluator "
            "or a non-canonical decode mode"
        )
    if sorted(ordinals) != expected_ordinals or len(set(ordinals)) != len(ordinals):
        raise RuntimeError(
            "pilot workers do not exactly cover disjoint manifest ordinals "
            f"0..{args.expected_rows - 1}"
        )

    task_summary: dict[str, dict[str, Any]] = {}
    for task in ("generation", "understanding", "editing"):
        values = [row for row in rows if row["task"] == task]
        expected = args.expected_rows // 3
        metric_names = sorted(
            {
                name
                for row in values
                for name in (row.get("metrics") or {})
            }
        )
        task_summary[task] = {
            "rows": len(values),
            "expected_rows": expected,
            "parse_rate": sum(bool(row["parsed"]) for row in values) / expected,
            "roundtrip_rate": sum(bool(row["roundtrip_exact"]) for row in values) / expected,
            "finite_rate": sum(bool(row["finite"]) for row in values) / expected,
            "plan_exact_rate": sum(bool(row["plan_exact"]) for row in values) / expected,
            "discrete_exact_rate": (
                sum(bool(row["discrete_exact"]) for row in values) / expected
            ),
            "mean_task_score": sum(float(row["task_score"]) for row in values) / expected,
            "mean_scene_score": sum(float(row["scene_score"]) for row in values) / expected,
            "budget_forced_row_rate": (
                sum(int(row.get("budget_forced_tokens", 0)) > 0 for row in values)
                / expected
            ),
            "budget_forced_tokens": sum(
                int(row.get("budget_forced_tokens", 0)) for row in values
            ),
            "mean_grammar_interventions": (
                sum(int(row.get("grammar_interventions", 0)) for row in values)
                / expected
            ),
            "mean_metrics": {
                name: sum(
                    float(row["metrics"][name])
                    for row in values
                    if name in row.get("metrics", {})
                )
                / sum(name in row.get("metrics", {}) for row in values)
                for name in metric_names
            },
            "patch_exact_rate": (
                sum(bool(row["patch_exact"]) for row in values) / expected
                if task == "editing"
                else None
            ),
            "patch_roundtrip_rate": (
                sum(bool(row["patch_roundtrip_exact"]) for row in values) / expected
                if task == "editing"
                else None
            ),
        }
    edit_kind_summary: dict[str, dict[str, Any]] = {}
    editing_rows = [row for row in rows if row["task"] == "editing"]
    for edit_kind in sorted({str(row["edit_kind"]) for row in editing_rows}):
        values = [row for row in editing_rows if str(row["edit_kind"]) == edit_kind]
        edit_kind_summary[edit_kind] = {
            "rows": len(values),
            "conditional_patch_exact_rate": (
                sum(bool(row["patch_exact"]) for row in values) / len(values)
            ),
            "conditional_applicable_rate": (
                sum(bool(row.get("conditional_edit_applicable")) for row in values)
                / len(values)
            ),
            "reference_patch_exact_rate": (
                sum(bool(row.get("reference_patch_exact")) for row in values)
                / len(values)
            ),
            "delta_sketch_exact_rate": sum(bool(row["discrete_exact"]) for row in values)
            / len(values),
            "mean_conditional_task_score": sum(float(row["task_score"]) for row in values)
            / len(values),
            "mean_conditional_scene_score": sum(float(row["scene_score"]) for row in values)
            / len(values),
            "mean_end_to_end_task_score": _mean_metric(
                values, "end_to_end_task_score"
            ),
            "mean_observation_task_score": _mean_metric(
                values, "observation_task_score"
            ),
            "mean_delta_control_rmse": _mean_metric(
                values, "delta_control_rmse"
            ),
            "mean_delta_zero_control_rmse": _mean_metric(
                values, "delta_zero_control_rmse"
            ),
            "mean_delta_control_skill_over_zero": _mean_metric(
                values, "delta_control_skill_over_zero"
            ),
        }
    hard_checks = {
        "exact_disjoint_coverage": sorted(ordinals) == expected_ordinals,
        "balanced_gue": all(
            task_summary[task]["rows"] == args.expected_rows // 3
            for task in task_summary
        ),
        "parse_100pct": all(
            task_summary[task]["parse_rate"] == 1.0 for task in task_summary
        ),
        "roundtrip_100pct": all(
            task_summary[task]["roundtrip_rate"] == 1.0 for task in task_summary
        ),
        "finite_100pct": all(
            task_summary[task]["finite_rate"] == 1.0 for task in task_summary
        ),
        "editing_patch_roundtrip_100pct": (
            task_summary["editing"]["patch_roundtrip_rate"] == 1.0
        ),
    }
    output = {
        "schema": "stable_audio_tools.p11_audio_aware_full_pilot_decode_gate",
        "version": 3,
        "metric_contract": "observation_conditional_edit_end_to_end_v2",
        "status": "PASS" if all(hard_checks.values()) else "FAIL",
        "scope": "P11_only_no_P10_render",
        "seed": 42,
        "checkpoint_sha256": next(iter(checkpoints)),
        "weights": next(iter(weights)),
        "decode_mode": next(iter(decode_modes)),
        "worker_evaluator_version": next(iter(evaluator_versions)),
        "expected_rows": args.expected_rows,
        "worker_count": len(paths),
        "elapsed_seconds_sum": sum(float(report["elapsed_seconds"]) for report in reports),
        "elapsed_seconds_max_worker": max(float(report["elapsed_seconds"]) for report in reports),
        "hard_checks": hard_checks,
        "summary_by_task": task_summary,
        "editing_by_kind": edit_kind_summary,
        "workers": [
            {
                "path": str(path),
                "sha256": _sha256(path),
                "ordinals": report["selected_ordinals"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
            for path, report in zip(paths, reports)
        ],
        "rows": sorted(rows, key=lambda row: int(row["ordinal"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "output": str(args.output.expanduser().resolve()),
                "summary_by_task": task_summary,
                "hard_checks": hard_checks,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if output["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
