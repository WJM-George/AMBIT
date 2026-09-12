#!/usr/bin/env python3
"""Fail-closed summary for matched-seed fixed-family rollout probes.

Training loss is deliberately absent from this decision.  A seed passes only
when target-audio CLAP and independently calibrated AST both assign every
source correctly, every AST anchor retains a minimum fraction of its isolated
post-VAE reference evidence, and active signal energy stays within bounds.
Aggregate mixture CLAP and spatial angles are secondary non-regression gates.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required multiseed diagnostic is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid JSON diagnostic {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, *, name: str, path: Path) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{path}: {name} is not finite")
    return number


def _case(
    directory: Path,
    *,
    min_ast_reference_retention: float,
    min_active_rms_ratio: float,
    max_active_rms_ratio: float,
) -> dict[str, Any]:
    result_path = directory / "RESULT.json"
    location_path = directory / "SOURCE_LOCATION_METRICS.json"
    ast_path = directory / "SOURCE_SEMANTICS_AST.json"
    result = _load(result_path)
    location = _load(location_path)
    ast = _load(ast_path)
    if result.get("status") != "PASS":
        raise RuntimeError(f"checkpoint evaluator did not pass: {result_path}")
    turns = result.get("turn_results") or []
    if len(turns) != 1:
        raise RuntimeError(f"multiseed case must contain one turn: {result_path}")
    identities = {
        (str(value.get("family_id")), int(value.get("family_rank", -1)))
        for value in (result, location, ast)
    }
    if len(identities) != 1:
        raise RuntimeError(f"mixed family diagnostics below {directory}")
    if (
        location.get("schema")
        != "stable_audio_tools.source_location_semantic_scores"
        or ast.get("schema") != "stable_audio_tools.source_semantics_ast"
        or ast.get("anchor_source") != "independent_isolated_vae_target"
    ):
        raise RuntimeError(f"uncalibrated semantic diagnostics below {directory}")

    location_assignment = location.get("target_audio_assignment") or {}
    ast_assignment = ast.get("anchor_assignment") or {}
    source_ids = [str(value) for value in ast.get("source_ids") or []]
    if not source_ids or source_ids != [str(value) for value in location.get("source_ids") or []]:
        raise RuntimeError(f"source identities disagree below {directory}")
    source_count = len(source_ids)
    location_accuracy = _finite(
        location_assignment.get("accuracy_on_target_discriminable_sources"),
        name="target-audio location assignment",
        path=location_path,
    )
    ast_accuracy = _finite(
        ast_assignment.get("accuracy_on_target_discriminable_sources"),
        name="AST anchor assignment",
        path=ast_path,
    )
    location_valid = int(location_assignment.get("valid_source_count", 0))
    ast_valid = int(ast_assignment.get("valid_source_count", 0))
    location_rows = {
        str(row["source_id"]): row for row in location_assignment.get("rows") or []
    }
    ast_rows = {
        str(row["source_id"]): row for row in ast_assignment.get("rows") or []
    }
    signal_rows = {str(row["source_id"]): row for row in ast.get("sources") or []}
    if set(location_rows) != set(source_ids) or set(ast_rows) != set(source_ids) or set(signal_rows) != set(source_ids):
        raise RuntimeError(f"source rows are incomplete below {directory}")

    sources = []
    for source_id in source_ids:
        location_row = location_rows[source_id]
        ast_row = ast_rows[source_id]
        signal_row = signal_rows[source_id]
        retention = _finite(
            ast_row.get("generated_to_reference_anchor_ratio"),
            name=f"{source_id} AST reference retention",
            path=ast_path,
        )
        rms_ratio = _finite(
            signal_row.get("active_rms_ratio"),
            name=f"{source_id} active RMS ratio",
            path=ast_path,
        )
        source_pass = all(
            (
                bool(location_row.get("target_valid")),
                bool(location_row.get("generated_correct")),
                bool(ast_row.get("target_valid")),
                bool(ast_row.get("generated_correct")),
                retention >= min_ast_reference_retention,
                min_active_rms_ratio <= rms_ratio <= max_active_rms_ratio,
            )
        )
        sources.append(
            {
                "source_id": source_id,
                "anchor_label": str(ast_row.get("anchor_label")),
                "target_audio_correct": bool(location_row.get("generated_correct")),
                "target_audio_margin": _finite(
                    location_row.get("generated_diagonal_margin"),
                    name=f"{source_id} target-audio margin",
                    path=location_path,
                ),
                "ast_correct": bool(ast_row.get("generated_correct")),
                "ast_margin": _finite(
                    ast_row.get("generated_diagonal_margin"),
                    name=f"{source_id} AST margin",
                    path=ast_path,
                ),
                "ast_reference_retention": retention,
                "active_rms_ratio": rms_ratio,
                "source_pass": source_pass,
            }
        )

    turn = turns[0]
    content = turn.get("content_alignment") or {}
    spatial = turn.get("spatial_alignment") or {}
    plan_spatial = turn.get("plan_spatial_alignment") or {}
    semantic_pass = all(
        (
            location_valid == source_count,
            ast_valid == source_count,
            location_accuracy == 1.0,
            ast_accuracy == 1.0,
            all(row["source_pass"] for row in sources),
        )
    )
    return {
        "result": str(result_path),
        "family_id": str(result["family_id"]),
        "family_rank": int(result["family_rank"]),
        "generated_target_clap": _finite(
            content.get("generated_target_audio_cosine"),
            name="generated-target CLAP",
            path=result_path,
        ),
        "target_error_deg": _finite(
            spatial.get("angular_error_mean_deg"),
            name="target angular error",
            path=result_path,
        ),
        "plan_error_deg": _finite(
            plan_spatial.get("angular_error_mean_deg"),
            name="Plan angular error",
            path=result_path,
        ),
        "source_count": source_count,
        "location_valid_source_count": location_valid,
        "ast_valid_source_count": ast_valid,
        "location_accuracy": location_accuracy,
        "ast_accuracy": ast_accuracy,
        "sources": sources,
        "semantic_pass": semantic_pass,
    }


def summarize(
    root: Path,
    *,
    baseline_step: int = 0,
    candidate_step: int = 100,
    min_seed_pass_fraction: float = 0.75,
    min_ast_reference_retention: float = 0.25,
    min_active_rms_ratio: float = 0.25,
    max_active_rms_ratio: float = 4.0,
    max_clap_regression: float = 0.02,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if baseline_step < 0 or candidate_step <= baseline_step:
        raise ValueError("baseline/candidate steps are invalid")
    if not 0.0 < min_seed_pass_fraction <= 1.0:
        raise ValueError("min_seed_pass_fraction must lie in (0, 1]")
    if min_ast_reference_retention < 0.0 or min_active_rms_ratio < 0.0:
        raise ValueError("semantic retention thresholds must be non-negative")
    if max_active_rms_ratio < min_active_rms_ratio or max_clap_regression < 0.0:
        raise ValueError("upper/non-regression thresholds are invalid")

    seed_dirs = sorted(
        (path for path in root.glob("seed*") if path.is_dir()),
        key=lambda path: int(path.name[4:]),
    )
    if not seed_dirs:
        raise RuntimeError(f"no seed directories below {root}")
    cases: dict[int, dict[int, dict[str, Any]]] = {}
    for seed_dir in seed_dirs:
        seed = int(seed_dir.name[4:])
        cases[seed] = {}
        for step in (baseline_step, candidate_step):
            cases[seed][step] = _case(
                seed_dir / f"step{step}",
                min_ast_reference_retention=min_ast_reference_retention,
                min_active_rms_ratio=min_active_rms_ratio,
                max_active_rms_ratio=max_active_rms_ratio,
            )
    identities = {
        (case[step]["family_id"], case[step]["family_rank"])
        for case in cases.values()
        for step in (baseline_step, candidate_step)
    }
    if len(identities) != 1:
        raise RuntimeError("multiseed panel mixes families")
    family_id, family_rank = identities.pop()
    source_ids = [row["source_id"] for row in cases[next(iter(cases))][candidate_step]["sources"]]
    for seed_cases in cases.values():
        for step in (baseline_step, candidate_step):
            if [row["source_id"] for row in seed_cases[step]["sources"]] != source_ids:
                raise RuntimeError("multiseed source order changed")

    def aggregate(step: int) -> dict[str, Any]:
        rows = [cases[seed][step] for seed in sorted(cases)]
        source_summary = {}
        for source_index, source_id in enumerate(source_ids):
            sources = [row["sources"][source_index] for row in rows]
            source_summary[source_id] = {
                "anchor_label": sources[0]["anchor_label"],
                "target_audio_correct_seed_count": sum(
                    int(source["target_audio_correct"]) for source in sources
                ),
                "ast_correct_seed_count": sum(int(source["ast_correct"]) for source in sources),
                "joint_source_pass_seed_count": sum(int(source["source_pass"]) for source in sources),
                "median_ast_reference_retention": median(
                    source["ast_reference_retention"] for source in sources
                ),
                "median_active_rms_ratio": median(
                    source["active_rms_ratio"] for source in sources
                ),
            }
        return {
            "seed_count": len(rows),
            "semantic_pass_seed_count": sum(int(row["semantic_pass"]) for row in rows),
            "semantic_pass_seed_fraction": sum(int(row["semantic_pass"]) for row in rows) / len(rows),
            "median_generated_target_clap": median(row["generated_target_clap"] for row in rows),
            "median_target_error_deg": median(row["target_error_deg"] for row in rows),
            "median_plan_error_deg": median(row["plan_error_deg"] for row in rows),
            "sources": source_summary,
        }

    baseline = aggregate(baseline_step)
    candidate = aggregate(candidate_step)
    aggregate_gates = {
        "seed_semantics": candidate["semantic_pass_seed_fraction"] >= min_seed_pass_fraction,
        "clap_preserved": candidate["median_generated_target_clap"]
        >= baseline["median_generated_target_clap"] - max_clap_regression,
        "target_direction_improved": candidate["median_target_error_deg"]
        < baseline["median_target_error_deg"],
        "plan_direction_improved": candidate["median_plan_error_deg"]
        < baseline["median_plan_error_deg"],
    }
    status = "PASS" if all(aggregate_gates.values()) else "BLOCK"
    return {
        "schema": "stable_audio_tools.spatial_cot_multiseed_semantic_probe",
        "schema_version": 1,
        "status": status,
        "decision": (
            "eligible_for_independent_clean_family_replication"
            if status == "PASS"
            else "stop_this_arm_do_not_extend_or_add_auxiliary_tricks"
        ),
        "family_rank": family_rank,
        "family_id": family_id,
        "seeds": sorted(cases),
        "baseline_step": baseline_step,
        "candidate_step": candidate_step,
        "thresholds": {
            "min_seed_pass_fraction": min_seed_pass_fraction,
            "min_ast_reference_retention": min_ast_reference_retention,
            "min_active_rms_ratio": min_active_rms_ratio,
            "max_active_rms_ratio": max_active_rms_ratio,
            "max_clap_regression": max_clap_regression,
            "required_location_accuracy": 1.0,
            "required_ast_accuracy": 1.0,
        },
        "aggregate_gates": aggregate_gates,
        "baseline": baseline,
        "candidate": candidate,
        "cases": {
            str(seed): {
                str(step): cases[seed][step]
                for step in (baseline_step, candidate_step)
            }
            for seed in sorted(cases)
        },
    }


def _markdown(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    candidate = summary["candidate"]
    lines = [
        "# Matched-seed source-semantic rollout gate",
        "",
        f"Status: **{summary['status']}**.",
        "",
        "| step | semantic seeds | median CLAP | target ° | Plan ° |",
        "|---:|---:|---:|---:|---:|",
    ]
    for step, row in (
        (summary["baseline_step"], baseline),
        (summary["candidate_step"], candidate),
    ):
        lines.append(
            f"| {step} | {row['semantic_pass_seed_count']}/{row['seed_count']} | "
            f"{row['median_generated_target_clap']:.4f} | "
            f"{row['median_target_error_deg']:.2f} | {row['median_plan_error_deg']:.2f} |"
        )
    lines.extend(
        [
            "",
            "| source | AST anchor | CLAP correct seeds | AST correct seeds | joint passes | median AST retention | median RMS ratio |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for source_id, row in candidate["sources"].items():
        lines.append(
            f"| {source_id} | {row['anchor_label']} | "
            f"{row['target_audio_correct_seed_count']}/{candidate['seed_count']} | "
            f"{row['ast_correct_seed_count']}/{candidate['seed_count']} | "
            f"{row['joint_source_pass_seed_count']}/{candidate['seed_count']} | "
            f"{row['median_ast_reference_retention']:.4f} | "
            f"{row['median_active_rms_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Aggregate gates: "
            + ", ".join(
                f"{name}={'PASS' if passed else 'BLOCK'}"
                for name, passed in summary["aggregate_gates"].items()
            )
            + ".",
            "",
            "Decision: `" + summary["decision"] + "`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-step", type=int, default=0)
    parser.add_argument("--candidate-step", type=int, default=100)
    parser.add_argument("--min-seed-pass-fraction", type=float, default=0.75)
    parser.add_argument("--min-ast-reference-retention", type=float, default=0.25)
    parser.add_argument("--min-active-rms-ratio", type=float, default=0.25)
    parser.add_argument("--max-active-rms-ratio", type=float, default=4.0)
    parser.add_argument("--max-clap-regression", type=float, default=0.02)
    args = parser.parse_args()
    summary = summarize(
        args.root,
        baseline_step=args.baseline_step,
        candidate_step=args.candidate_step,
        min_seed_pass_fraction=args.min_seed_pass_fraction,
        min_ast_reference_retention=args.min_ast_reference_retention,
        min_active_rms_ratio=args.min_active_rms_ratio,
        max_active_rms_ratio=args.max_active_rms_ratio,
        max_clap_regression=args.max_clap_regression,
    )
    root = args.root.expanduser().resolve()
    (root / "MULTISEED_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (root / "MULTISEED_CONCLUSION.md").write_text(
        _markdown(summary), encoding="utf-8"
    )
    for marker in ("MULTISEED_PASS", "MULTISEED_BLOCK"):
        (root / marker).unlink(missing_ok=True)
    (root / f"MULTISEED_{summary['status']}").touch()
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
