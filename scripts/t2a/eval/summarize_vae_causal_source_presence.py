#!/usr/bin/env python3
"""Merge v2 direct-state VAE causal source-presence adjudications."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.summarize_vae_mixture_state_identifiability import PANEL_RANKS
from stable_audio_tools.data.t2a_artifacts import atomic_write_json


EXPECTED_SCHEMA = "stable_audio_tools.vae_causal_source_presence_adjudication"
MIN_CALIBRATED_FAMILY_FRACTION = 0.75
MAX_FAILURE_FRACTION = 0.25


def summarize(
    reports: Sequence[Mapping[str, Any]],
    *,
    expected_ranks: Sequence[int] = PANEL_RANKS,
    subtraction_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one causal-presence report is required")
    expected = sorted(set(int(rank) for rank in expected_ranks))
    reference = reports[0]
    contract_fields = (
        "schema_version",
        "adjudication",
        "draw_count",
        "shared_posterior_epsilon",
        "seed",
        "thresholds",
        "vae_checkpoint_sha256",
        "clap_model",
        "ast_model",
    )
    families_by_rank: dict[int, dict[str, Any]] = {}
    worker_rows = []
    for report in reports:
        if report.get("schema") != EXPECTED_SCHEMA:
            raise ValueError("an input is not a causal source-presence report")
        if int(report.get("schema_version", -1)) != 2:
            raise ValueError("only differential v2 adjudications may be merged")
        for field in contract_fields:
            if report.get(field) != reference.get(field):
                raise ValueError(f"worker adjudication contract differs at {field}")
        worker_ranks = []
        for family in report.get("families") or []:
            rank = int(family["family_rank"])
            if rank in families_by_rank:
                raise ValueError(f"duplicate family rank: {rank}")
            families_by_rank[rank] = dict(family)
            worker_ranks.append(rank)
        if sorted(worker_ranks) != sorted(int(rank) for rank in report["family_ranks"]):
            raise ValueError("worker family rows differ from declared ranks")
        worker_rows.append(worker_ranks)
    actual = sorted(families_by_rank)
    if actual != expected:
        raise ValueError(f"panel ranks changed: expected={expected}, actual={actual}")

    families = [families_by_rank[rank] for rank in expected]
    calibrated_families = [
        family for family in families if family["outcome"] != "ABSTAIN_CALIBRATION"
    ]
    calibrated_family_fraction = len(calibrated_families) / len(families)
    sources = [
        {"family_rank": int(family["family_rank"]), **dict(source)}
        for family in families
        for source in family["sources"]
    ]
    calibrated_sources = [
        source for source in sources if source["status"] != "ABSTAIN_CALIBRATION"
    ]
    failed_sources = [source for source in calibrated_sources if source["status"] != "PASS"]
    failure_fraction = len(failed_sources) / max(len(calibrated_sources), 1)
    family5 = families_by_rank.get(5)
    if family5 is None:
        raise ValueError("family 5 is mandatory")
    family5_abstains = any(
        source["status"] == "ABSTAIN_CALIBRATION" for source in family5["sources"]
    )
    family5_fails = any(source["status"] == "BLOCK" for source in family5["sources"])

    if calibrated_family_fraction < MIN_CALIBRATED_FAMILY_FRACTION or family5_abstains:
        direct_decision = "INCONCLUSIVE_CALIBRATION"
    elif family5_fails or failure_fraction > MAX_FAILURE_FRACTION:
        direct_decision = "CAUSAL_SOURCE_PRESENCE_LOSS"
    else:
        direct_decision = "CAUSAL_SOURCE_PRESENCE_PRESERVED"

    subtraction_decision = None
    if subtraction_summary is not None:
        if subtraction_summary.get("schema") != (
            "stable_audio_tools.vae_mixture_state_identifiability_summary"
        ):
            raise ValueError("subtraction summary schema changed")
        subtraction_decision = str(subtraction_summary["decision"])
    if direct_decision == "CAUSAL_SOURCE_PRESENCE_LOSS":
        combined_decision = "VAE_CAUSAL_SOURCE_PRESENCE_LOSS"
    elif direct_decision == "INCONCLUSIVE_CALIBRATION":
        combined_decision = direct_decision
    elif subtraction_decision == "VAE_SOURCE_IDENTITY_LOSS":
        combined_decision = "NONLINEAR_MIXTURE_STATE_ENTANGLEMENT"
    else:
        combined_decision = "STABLE_RECOVERABLE"

    failure_checks = Counter()
    failure_sources = Counter()
    for source in failed_sources:
        for check, count in source["failure_draw_counts"].items():
            failure_checks[check] += int(count)
            if int(count) > 1:
                failure_sources[check] += 1
    family_rows = []
    for family in families:
        family_sources = family["sources"]
        family_rows.append(
            {
                "family_rank": int(family["family_rank"]),
                "family_id": str(family["family_id"]),
                "source_count": int(family["source_count"]),
                "calibrated_source_count": sum(
                    source["status"] != "ABSTAIN_CALIBRATION"
                    for source in family_sources
                ),
                "pass_source_count": sum(
                    source["status"] == "PASS" for source in family_sources
                ),
                "outcome": str(family["outcome"]),
            }
        )
    return {
        "schema": "stable_audio_tools.vae_causal_source_presence_summary",
        "schema_version": 1,
        "direct_decision": direct_decision,
        "subtraction_audit_decision": subtraction_decision,
        "combined_decision": combined_decision,
        "family_count": len(families),
        "calibrated_family_count": len(calibrated_families),
        "calibrated_family_fraction": calibrated_family_fraction,
        "source_count": len(sources),
        "calibrated_source_count": len(calibrated_sources),
        "failed_source_count": len(failed_sources),
        "failed_source_fraction": failure_fraction,
        "family5": {
            "outcome": str(family5["outcome"]),
            "abstains": family5_abstains,
            "fails": family5_fails,
        },
        "failure_draw_counts": dict(sorted(failure_checks.items())),
        "failure_source_counts": dict(sorted(failure_sources.items())),
        "thresholds": {
            "min_calibrated_family_fraction": MIN_CALIBRATED_FAMILY_FRACTION,
            "max_failure_fraction": MAX_FAILURE_FRACTION,
            "family5_mandatory": True,
        },
        "audit_contract": {field: reference[field] for field in contract_fields},
        "worker_family_ranks": worker_rows,
        "families": family_rows,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen VAE causal source-presence conclusion",
        "",
        f"Combined decision: **{summary['combined_decision']}**",
        "",
        f"Direct-state decision: `{summary['direct_decision']}`; subtraction-audit decision: `{summary['subtraction_audit_decision']}`.",
        "",
        "| panel metric | value |",
        "|---|---:|",
        f"| calibrated families | {summary['calibrated_family_count']}/{summary['family_count']} ({summary['calibrated_family_fraction']:.3f}) |",
        f"| calibrated sources | {summary['calibrated_source_count']}/{summary['source_count']} |",
        f"| failed causal-presence sources | {summary['failed_source_count']} ({summary['failed_source_fraction']:.3f}) |",
        f"| family 5 | {summary['family5']['outcome']} |",
        "",
        "| family | sources | calibrated | pass | outcome |",
        "|---:|---:|---:|---:|---|",
    ]
    for family in summary["families"]:
        lines.append(
            f"| {family['family_rank']} | {family['source_count']} | "
            f"{family['calibrated_source_count']} | {family['pass_source_count']} | "
            f"{family['outcome']} |"
        )
    lines.extend(["", "Frozen 7/8 check failures by source (draw total):", ""])
    for name, count in summary["failure_source_counts"].items():
        lines.append(
            f"- `{name}`: {count} sources "
            f"({summary['failure_draw_counts'][name]} failed draws)"
        )
    lines.extend(
        [
            "",
            "This adjudication compares decoded full and leave-one-out states "
            "directly. It does not require decoder linearity and creates no "
            "training checkpoint or inference-time source route.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="+", type=Path)
    parser.add_argument("--subtraction-summary", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_markdown.exists():
        raise FileExistsError("refusing to overwrite an existing conclusion")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
    subtraction = (
        json.loads(args.subtraction_summary.read_text(encoding="utf-8"))
        if args.subtraction_summary is not None
        else None
    )
    summary = summarize(reports, subtraction_summary=subtraction)
    atomic_write_json(args.output_json, summary)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "combined_decision": summary["combined_decision"],
                "output": str(args.output_json.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
