#!/usr/bin/env python3
"""Summarize fixed-family renderer probes without hiding missing sources.

The capacity panel deliberately supplies the target ScenePlan and no previous
FOA.  Its question is therefore narrow: can one checkpoint render every
target-discriminable source at the requested location?  Aggregate CLAP and a
mixture-level direction can improve after the model drops a weak source, so
neither is accepted as a source-binding gate on its own.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


def _finite(value: Any, *, name: str, path: Path) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{path}: {name} is not finite")
    return number


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required capacity diagnostic is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON diagnostic {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _load_case(
    result_path: Path,
    *,
    min_target_presence: float,
    min_source_retention: float,
    min_location_accuracy: float,
) -> dict[str, Any]:
    directory = result_path.parent
    result = _load_json(result_path)
    presence_path = directory / "SOURCE_PRESENCE_OPENFLAM.json"
    location_path = directory / "SOURCE_LOCATION_METRICS.json"
    presence = _load_json(presence_path)
    location = _load_json(location_path)

    if result.get("status") != "PASS":
        raise RuntimeError(f"checkpoint evaluator did not pass: {result_path}")
    turns = result.get("turn_results") or []
    if len(turns) != 1:
        raise RuntimeError(f"capacity case must contain exactly one turn: {result_path}")
    if (
        presence.get("family_id") != result.get("family_id")
        or location.get("family_id") != result.get("family_id")
    ):
        raise RuntimeError(f"mixed family diagnostics below {directory}")

    turn = turns[0]
    source_rows = []
    for source in presence.get("sources") or []:
        target_active = _finite(
            ((source.get("target") or {}).get("active") or {}).get("mean"),
            name="target active presence",
            path=presence_path,
        )
        generated_active = _finite(
            ((source.get("generated") or {}).get("active") or {}).get("mean"),
            name="generated active presence",
            path=presence_path,
        )
        eligible = target_active >= min_target_presence
        retention = generated_active / target_active if target_active > 0.0 else None
        source_rows.append(
            {
                "source_id": str(source.get("source_id")),
                "caption": str(source.get("caption")),
                "target_active_presence": target_active,
                "generated_active_presence": generated_active,
                "retention": retention,
                "target_discriminable": eligible,
                "presence_pass": bool(
                    eligible
                    and retention is not None
                    and retention >= min_source_retention
                ),
            }
        )
    eligible_rows = [row for row in source_rows if row["target_discriminable"]]
    if not eligible_rows:
        raise RuntimeError(
            f"no source reaches target-presence threshold in {presence_path}"
        )

    assignment = location.get("target_audio_assignment") or {}
    location_accuracy = _finite(
        assignment.get("accuracy_on_target_discriminable_sources"),
        name="target-audio assignment accuracy",
        path=location_path,
    )
    location_valid_sources = int(assignment.get("valid_source_count", 0))
    if location_valid_sources < 1:
        raise RuntimeError(f"no target-discriminable location source in {location_path}")

    content = turn.get("content_alignment") or {}
    spatial = turn.get("spatial_alignment") or {}
    plan_spatial = turn.get("plan_spatial_alignment") or {}
    return {
        "result": str(result_path),
        "family_id": str(result.get("family_id")),
        "family_rank": int(result["family_rank"]),
        "generated_target_clap": _finite(
            content.get("generated_target_audio_cosine"),
            name="generated-target CLAP",
            path=result_path,
        ),
        "target_error_deg": _finite(
            spatial.get("angular_error_mean_deg"),
            name="generated-target angular error",
            path=result_path,
        ),
        "plan_error_deg": _finite(
            plan_spatial.get("angular_error_mean_deg"),
            name="generated-Plan angular error",
            path=result_path,
        ),
        "source_rows": source_rows,
        "eligible_source_count": len(eligible_rows),
        "source_presence_pass_count": sum(
            int(row["presence_pass"]) for row in eligible_rows
        ),
        "min_source_retention": min(float(row["retention"]) for row in eligible_rows),
        "source_presence_pass": all(row["presence_pass"] for row in eligible_rows),
        "location_accuracy": location_accuracy,
        "location_valid_source_count": location_valid_sources,
        "location_pass": location_accuracy >= min_location_accuracy,
    }


def summarize(
    root: Path,
    *,
    min_target_presence: float = 1.0e-3,
    min_source_retention: float = 0.25,
    min_location_accuracy: float = 1.0,
    max_target_error_deg: float = 60.0,
    max_clap_regression: float = 0.02,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if min_target_presence <= 0.0 or min_source_retention < 0.0:
        raise ValueError("presence thresholds must be positive/non-negative")
    if not 0.0 <= min_location_accuracy <= 1.0:
        raise ValueError("min_location_accuracy must be in [0, 1]")
    if max_target_error_deg < 0.0 or max_clap_regression < 0.0:
        raise ValueError("error tolerances must be non-negative")

    families: dict[int, dict[int, dict[str, Any]]] = {}
    for result_path in sorted(root.glob("family*/step*/RESULT.json")):
        case = _load_case(
            result_path,
            min_target_presence=min_target_presence,
            min_source_retention=min_source_retention,
            min_location_accuracy=min_location_accuracy,
        )
        step_name = result_path.parent.name
        if not step_name.startswith("step"):
            raise RuntimeError(f"invalid capacity step directory: {result_path.parent}")
        step = int(step_name[4:])
        rank = int(case["family_rank"])
        if step in families.setdefault(rank, {}):
            raise RuntimeError(f"duplicate family{rank}/step{step} capacity result")
        families[rank][step] = case
    if not families:
        raise RuntimeError(f"no fixed-family RESULT.json files below {root}")

    step_sets = {tuple(sorted(records)) for records in families.values()}
    if len(step_sets) != 1:
        raise RuntimeError(f"families do not share one checkpoint set: {step_sets}")
    steps = list(next(iter(step_sets)))
    if 0 not in steps or len(steps) < 2:
        raise RuntimeError("capacity sweep requires step0 and at least one trained step")

    common_candidates = set(step for step in steps if step != 0)
    for rank, records in families.items():
        baseline = records[0]
        for step in steps:
            record = records[step]
            if step == 0:
                record["candidate_pass"] = False
                continue
            record["target_direction_improved"] = (
                record["target_error_deg"] < baseline["target_error_deg"]
            )
            record["plan_direction_improved"] = (
                record["plan_error_deg"] < baseline["plan_error_deg"]
            )
            record["clap_preserved"] = (
                record["generated_target_clap"]
                >= baseline["generated_target_clap"] - max_clap_regression
            )
            record["candidate_pass"] = all(
                (
                    record["source_presence_pass"],
                    record["location_pass"],
                    record["target_direction_improved"],
                    record["plan_direction_improved"],
                    record["clap_preserved"],
                    record["target_error_deg"] <= max_target_error_deg,
                )
            )
            if not record["candidate_pass"]:
                common_candidates.discard(step)

    common_candidates = sorted(common_candidates)
    if common_candidates:
        def score(step: int) -> tuple[float, float, int]:
            records = [families[rank][step] for rank in sorted(families)]
            return (
                -min(record["min_source_retention"] for record in records),
                mean(record["target_error_deg"] for record in records),
                step,
            )

        recommended_step = min(common_candidates, key=score)
    else:
        recommended_step = None

    return {
        "schema": "stable_audio_tools.spatial_cot_capacity_probe",
        "schema_version": 1,
        "status": "PASS" if common_candidates else "BLOCK",
        "decision": (
            "eligible_for_heldout_family274_then_family8179"
            if common_candidates
            else "do_not_extend_or_run_heldout"
        ),
        "thresholds": {
            "min_target_presence": min_target_presence,
            "min_source_retention": min_source_retention,
            "min_location_accuracy": min_location_accuracy,
            "max_target_error_deg": max_target_error_deg,
            "max_clap_regression": max_clap_regression,
        },
        "steps": steps,
        "families": {
            str(rank): {str(step): records[step] for step in steps}
            for rank, records in sorted(families.items())
        },
        "common_candidate_steps": common_candidates,
        "recommended_step": recommended_step,
    }


def _markdown(summary: dict[str, Any]) -> str:
    thresholds = summary["thresholds"]
    lines = [
        "# Fixed-family source-binding capacity gate",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "A checkpoint passes only when the same step recovers every "
        "target-discriminable source, assigns target-source audio to the "
        "correct spatial slot, improves both mixture→target and mixture→Plan "
        "direction over step0, and preserves aggregate content similarity.",
        "",
        "| family | step | weak-source min retention | presence | location | target ° | Plan ° | CLAP | eligible |",
        "|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|",
    ]
    for rank_text, records in summary["families"].items():
        for step_text, record in records.items():
            lines.append(
                "| {rank} | {step} | {ret:.4f} | {present}/{total} | "
                "{location:.3f} | {target:.2f} | {plan:.2f} | {clap:.4f} | {eligible} |".format(
                    rank=rank_text,
                    step=step_text,
                    ret=record["min_source_retention"],
                    present=record["source_presence_pass_count"],
                    total=record["eligible_source_count"],
                    location=record["location_accuracy"],
                    target=record["target_error_deg"],
                    plan=record["plan_error_deg"],
                    clap=record["generated_target_clap"],
                    eligible=("yes" if record["candidate_pass"] else "no"),
                )
            )
    lines.extend(
        [
            "",
            "Thresholds: target presence ≥ {target:g}; retention ≥ {ret:g}; "
            "location accuracy ≥ {location:g}; target angle ≤ {angle:g}°; "
            "CLAP regression ≤ {clap:g}.".format(
                target=thresholds["min_target_presence"],
                ret=thresholds["min_source_retention"],
                location=thresholds["min_location_accuracy"],
                angle=thresholds["max_target_error_deg"],
                clap=thresholds["max_clap_regression"],
            ),
            "",
            (
                "Common eligible steps: "
                + ", ".join(map(str, summary["common_candidate_steps"]))
                if summary["common_candidate_steps"]
                else "No common eligible checkpoint. Do not run held-out families or extend this arm."
            ),
        ]
    )
    if summary["recommended_step"] is not None:
        lines.extend(
            ["", f"Recommended checkpoint: step {summary['recommended_step']}."]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--min-target-presence", type=float, default=1.0e-3)
    parser.add_argument("--min-source-retention", type=float, default=0.25)
    parser.add_argument("--min-location-accuracy", type=float, default=1.0)
    parser.add_argument("--max-target-error-deg", type=float, default=60.0)
    parser.add_argument("--max-clap-regression", type=float, default=0.02)
    args = parser.parse_args()
    summary = summarize(
        args.root,
        min_target_presence=args.min_target_presence,
        min_source_retention=args.min_source_retention,
        min_location_accuracy=args.min_location_accuracy,
        max_target_error_deg=args.max_target_error_deg,
        max_clap_regression=args.max_clap_regression,
    )
    root = args.root.expanduser().resolve()
    (root / "CAPACITY_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "CAPACITY_CONCLUSION.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    for marker in ("CAPACITY_PASS", "CAPACITY_BLOCK"):
        (root / marker).unlink(missing_ok=True)
    (root / f"CAPACITY_{summary['status']}").touch()
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
