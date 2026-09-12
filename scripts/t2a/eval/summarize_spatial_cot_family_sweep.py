#!/usr/bin/env python3
"""Summarize ordered Spatial-CoT checkpoint sweeps and gate promotion."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def _finite_mean(values) -> float | None:
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return mean(values) if values else None


def _load_family(
    directory: Path,
    *,
    max_target_error_deg: float,
    max_plan_excess_deg: float,
) -> dict[str, Any]:
    records = {}
    family_id = None
    family_rank = None
    for result_path in sorted(directory.glob("step*/family_*/RESULT.json")):
        step_text = result_path.parents[1].name
        if not step_text.startswith("step"):
            continue
        step = int(step_text[4:])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        turns = result.get("turn_results") or []
        if len(turns) < 4:
            raise RuntimeError(f"expected four turns in {result_path}")
        family_id = family_id or result.get("family_id")
        family_rank = family_rank if family_rank is not None else int(result["family_rank"])
        if result.get("family_id") != family_id or int(result["family_rank"]) != family_rank:
            raise RuntimeError(f"mixed families below {directory}")
        turn0 = turns[0]
        edits = turns[1:4]
        records[step] = {
            "result": str(result_path),
            "evaluator_status": result.get("status"),
            "creation_plan_error_deg": turn0["plan_spatial_alignment"][
                "angular_error_mean_deg"
            ],
            "creation_target_error_deg": turn0["spatial_alignment"][
                "angular_error_mean_deg"
            ],
            "creation_target_floor_deg": turn0["target_plan_spatial_alignment"][
                "angular_error_mean_deg"
            ],
            "edit_plan_error_mean_deg": _finite_mean(
                turn["plan_spatial_alignment"]["angular_error_mean_deg"]
                for turn in edits
            ),
            "edit_target_error_mean_deg": _finite_mean(
                turn["spatial_alignment"]["angular_error_mean_deg"]
                for turn in edits
            ),
            "edit_target_floor_mean_deg": _finite_mean(
                turn["target_plan_spatial_alignment"]["angular_error_mean_deg"]
                for turn in edits
            ),
            "edit_direction_cosine_mean": _finite_mean(
                turn["spatial_alignment"]["direction_cosine"] for turn in edits
            ),
        }
    required = {0, 100, 200, 300}
    if set(records) != required:
        raise RuntimeError(
            f"{directory} must contain steps {sorted(required)}, got {sorted(records)}"
        )
    baseline = records[0]
    candidate_steps = []
    for step in (100, 200, 300):
        record = records[step]
        record["creation_improved_vs_step0"] = (
            record["creation_plan_error_deg"] < baseline["creation_plan_error_deg"]
            and record["creation_target_error_deg"]
            < baseline["creation_target_error_deg"]
        )
        record["closed_loop_improved_vs_step0"] = (
            record["edit_target_error_mean_deg"]
            < baseline["edit_target_error_mean_deg"]
            and record["edit_plan_error_mean_deg"]
            < baseline["edit_plan_error_mean_deg"]
        )
        record["creation_quality_pass"] = (
            record["creation_target_error_deg"] <= max_target_error_deg
            and record["creation_plan_error_deg"]
            <= record["creation_target_floor_deg"] + max_plan_excess_deg
        )
        record["closed_loop_quality_pass"] = (
            record["edit_target_error_mean_deg"] <= max_target_error_deg
            and record["edit_plan_error_mean_deg"]
            <= record["edit_target_floor_mean_deg"] + max_plan_excess_deg
        )
        if (
            record["creation_improved_vs_step0"]
            and record["closed_loop_improved_vs_step0"]
            and record["creation_quality_pass"]
            and record["closed_loop_quality_pass"]
        ):
            candidate_steps.append(step)
    return {
        "family_id": family_id,
        "family_rank": family_rank,
        "directory": str(directory),
        "steps": {str(step): records[step] for step in sorted(records)},
        "candidate_steps": candidate_steps,
        "thresholds": {
            "max_target_error_deg": max_target_error_deg,
            "max_plan_excess_deg": max_plan_excess_deg,
        },
        "status": "PASS" if candidate_steps else "BLOCK",
    }


def _family_markdown(family: dict[str, Any]) -> str:
    lines = [
        f"# family{family['family_rank']} checkpoint sweep",
        "",
        "| step | create→Plan ° | create→target ° | edits→Plan mean ° | edits→target mean ° | eligible |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for step_text, record in family["steps"].items():
        step = int(step_text)
        lines.append(
            "| {step} | {cp:.2f} | {ct:.2f} | {ep:.2f} | {et:.2f} | {eligible} |".format(
                step=step,
                cp=record["creation_plan_error_deg"],
                ct=record["creation_target_error_deg"],
                ep=record["edit_plan_error_mean_deg"],
                et=record["edit_target_error_mean_deg"],
                eligible=(
                    "—"
                    if step == 0
                    else (
                        "yes"
                        if all(
                            record[name]
                            for name in (
                                "creation_improved_vs_step0",
                                "closed_loop_improved_vs_step0",
                                "creation_quality_pass",
                                "closed_loop_quality_pass",
                            )
                        )
                        else "no"
                    )
                ),
            )
        )
    candidates = family["candidate_steps"]
    lines.extend(
        [
            "",
            f"Status: **{family['status']}**.",
            "",
            (
                "Steps passing improvement and absolute-quality gates: "
                + ", ".join(map(str, candidates))
                if candidates
                else "No checkpoint passes both creation and closed-loop gates."
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _combined_step_score(families: list[dict[str, Any]], step: int) -> float:
    """Lower-is-better spatial diagnostic score in angular degrees.

    Generated-to-target errors measure absolute recovery. Positive excess over
    the retained target-to-Plan floor measures avoidable Plan disagreement
    without charging the model for compiler/room mismatch already present in
    the target. This score selects which already-evaluated checkpoint receives
    expensive semantic diagnostics; it never changes the promotion gates.
    """

    values = []
    for family in families:
        record = family["steps"][str(step)]
        creation_excess = max(
            0.0,
            record["creation_plan_error_deg"]
            - record["creation_target_floor_deg"],
        )
        edit_excess = max(
            0.0,
            record["edit_plan_error_mean_deg"]
            - record["edit_target_floor_mean_deg"],
        )
        values.extend(
            (
                record["creation_target_error_deg"],
                record["edit_target_error_mean_deg"],
                creation_excess,
                edit_excess,
            )
        )
    score = _finite_mean(values)
    if score is None:
        raise RuntimeError(f"step {step} has no finite spatial score")
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-target-error-deg", type=float, default=30.0)
    parser.add_argument("--max-plan-excess-deg", type=float, default=30.0)
    args = parser.parse_args()
    if args.max_target_error_deg < 0.0 or args.max_plan_excess_deg < 0.0:
        raise ValueError("angular thresholds must be non-negative")
    families = [
        _load_family(
            path.expanduser().resolve(),
            max_target_error_deg=args.max_target_error_deg,
            max_plan_excess_deg=args.max_plan_excess_deg,
        )
        for path in args.family_dir
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for family in families:
        prefix = args.output_dir / f"family_{family['family_rank']}"
        prefix.with_suffix(".json").write_text(
            json.dumps(family, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        prefix.with_suffix(".md").write_text(
            _family_markdown(family), encoding="utf-8"
        )

    common = set((100, 200, 300))
    for family in families:
        common &= set(family["candidate_steps"])
    common_steps = sorted(common)
    step_scores = {
        str(step): _combined_step_score(families, step)
        for step in (100, 200, 300)
    }
    recommended_step = (
        min(common_steps, key=lambda step: (step_scores[str(step)], step))
        if common_steps
        else None
    )
    # A blocked experiment still needs one representative checkpoint for
    # causal failure analysis. Selecting the least-bad observed spatial point
    # is more informative than silently assuming the final step is optimal.
    diagnostic_step = (
        recommended_step
        if recommended_step is not None
        else min((100, 200, 300), key=lambda step: (step_scores[str(step)], step))
    )
    combined = {
        "schema": "stable_audio_tools.spatial_cot_sequential_sweep",
        "schema_version": 1,
        "families": families,
        "common_candidate_steps": common_steps,
        "spatial_diagnostic_scores": step_scores,
        "recommended_step": recommended_step,
        "diagnostic_step": diagnostic_step,
        "status": "PASS" if common_steps else "BLOCK",
        "decision": (
            "eligible_for_next_stage"
            if common_steps
            else "do_not_unfreeze_or_extend"
        ),
    }
    (args.output_dir / "SUMMARY.json").write_text(
        json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Cross-family promotion gate",
        "",
        f"Status: **{combined['status']}**.",
        "",
        "A checkpoint is eligible only when the same step improves both turn0 creation and closed-loop editing over step0 for every family, keeps generated→target error at or below 30°, and stays within 30° of the target→Plan floor.",
        "",
        (
            "Common eligible steps: " + ", ".join(map(str, common_steps))
            if common_steps
            else "No common eligible checkpoint. Do not unfreeze the Transformer or start a longer run."
        ),
        "",
        (
            f"Recommended eligible step: {recommended_step}."
            if recommended_step is not None
            else f"Failure-analysis checkpoint: step {diagnostic_step} (promotion remains blocked)."
        ),
    ]
    (args.output_dir / "CONCLUSION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(combined, sort_keys=True))


if __name__ == "__main__":
    main()
