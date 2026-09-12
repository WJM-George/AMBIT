#!/usr/bin/env python3
"""Merge GPU shards from the frozen VAE mixture-state identifiability audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.t2a_artifacts import atomic_write_json


EXPECTED_SCHEMA = "stable_audio_tools.vae_mixture_state_identifiability_audit"
PANEL_RANKS = (2, 5, 10, 13, 16, 17, 18, 19, 20, 24, 25, 30, 32, 38, 40, 42, 45, 46, 47)
MIN_CALIBRATED_FAMILY_FRACTION = 0.75
MAX_FAILURE_FRACTION = 0.25
MIN_STABLE_SOURCE_FRACTION = 0.75


def _family_source_rows(family: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = family.get("source_outcomes")
    if not isinstance(rows, list) or len(rows) != int(family.get("source_count", -1)):
        raise ValueError(f"family {family.get('family_rank')} has invalid source outcomes")
    return [dict(row) for row in rows]


def summarize(
    reports: Sequence[Mapping[str, Any]],
    *,
    expected_ranks: Sequence[int] = PANEL_RANKS,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one audit report is required")
    expected = sorted(set(int(rank) for rank in expected_ranks))
    if not expected or min(expected) < 0:
        raise ValueError("expected ranks must be non-negative")
    reference = reports[0]
    contract_fields = (
        "schema_version",
        "draw_count",
        "shared_posterior_epsilon",
        "seed",
        "thresholds",
        "rf_times",
        "vae_checkpoint_sha256",
        "clap_model",
        "ast_model",
    )
    families_by_rank: dict[int, dict[str, Any]] = {}
    worker_rows = []
    for report in reports:
        if report.get("schema") != EXPECTED_SCHEMA:
            raise ValueError("an input is not a VAE mixture-state audit")
        for field in contract_fields:
            if report.get(field) != reference.get(field):
                raise ValueError(f"worker audit contract differs at {field}")
        worker_ranks = []
        for family in report.get("families") or []:
            rank = int(family["family_rank"])
            if rank in families_by_rank:
                raise ValueError(f"duplicate family rank across workers: {rank}")
            families_by_rank[rank] = dict(family)
            worker_ranks.append(rank)
        if sorted(worker_ranks) != sorted(int(rank) for rank in report["family_ranks"]):
            raise ValueError("worker family list does not match its declared ranks")
        worker_rows.append(worker_ranks)
    actual = sorted(families_by_rank)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if missing or unexpected:
        raise ValueError(f"panel ranks changed: missing={missing}, unexpected={unexpected}")

    families = [families_by_rank[rank] for rank in expected]
    calibrated_families = [
        family for family in families if family["outcome"] != "ABSTAIN_CALIBRATION"
    ]
    calibrated_family_fraction = len(calibrated_families) / len(families)
    all_sources = [
        {"family_rank": int(family["family_rank"]), **source}
        for family in families
        for source in _family_source_rows(family)
    ]
    calibrated_sources = [
        source
        for source in all_sources
        if source["semantic_status"] != "ABSTAIN_CALIBRATION"
    ]
    semantic_failures = [
        source for source in calibrated_sources if source["semantic_status"] != "PASS"
    ]
    semantic_pass_sources = [
        source for source in calibrated_sources if source["semantic_status"] == "PASS"
    ]
    latent_failures = [
        source for source in semantic_pass_sources if source["latent_status"] != "PASS"
    ]
    stable_sources = [
        source
        for source in calibrated_sources
        if source["semantic_status"] == "PASS" and source["latent_status"] == "PASS"
    ]
    semantic_failure_fraction = len(semantic_failures) / max(len(calibrated_sources), 1)
    latent_failure_fraction = len(latent_failures) / max(len(semantic_pass_sources), 1)
    stable_source_fraction = len(stable_sources) / max(len(calibrated_sources), 1)

    branch_rows = []
    for family in families:
        semantic = family.get("semantic") or {}
        semantic_sources = semantic.get("sources") or []
        if len(semantic_sources) != int(family["source_count"]):
            raise ValueError(
                f"family {family['family_rank']} has invalid semantic branch rows"
            )
        for source in semantic_sources:
            branch_rows.append(
                {
                    "family_rank": int(family["family_rank"]),
                    "source_id": str(source["source_id"]),
                    "source_status": str(source["status"]),
                    "isolated": dict(source["isolated"]),
                    "intervention": dict(source["intervention"]),
                }
            )
    isolated_passes = [row for row in branch_rows if row["isolated"]["status"] == "PASS"]
    intervention_passes = [
        row for row in branch_rows if row["intervention"]["status"] == "PASS"
    ]
    intervention_clap_passes = [
        row
        for row in branch_rows
        if int(row["intervention"]["clap_correct_draw_count"]) >= 7
    ]
    intervention_ast_passes = [
        row
        for row in branch_rows
        if int(row["intervention"]["ast_correct_draw_count"]) >= 7
    ]
    intervention_ast_retention_passes = [
        row
        for row in branch_rows
        if float(row["intervention"]["ast_target_retention"]["median"]) >= 0.25
    ]
    intervention_rms_passes = [
        row
        for row in branch_rows
        if 0.25
        <= float(row["intervention"]["active_rms_retention"]["median"])
        <= 4.0
    ]
    clap_pass_ast_assignment_failures = [
        row
        for row in branch_rows
        if int(row["intervention"]["clap_correct_draw_count"]) >= 7
        and int(row["intervention"]["ast_correct_draw_count"]) < 7
    ]

    family5 = families_by_rank.get(5)
    if family5 is None:
        raise ValueError("family 5 is mandatory")
    family5_sources = _family_source_rows(family5)
    family5_abstains = any(
        source["semantic_status"] == "ABSTAIN_CALIBRATION"
        for source in family5_sources
    )
    family5_semantic_failure = any(
        source["semantic_status"] not in {"PASS", "ABSTAIN_CALIBRATION"}
        for source in family5_sources
    )
    family5_latent_failure = any(
        source["semantic_status"] == "PASS" and source["latent_status"] != "PASS"
        for source in family5_sources
    )

    if calibrated_family_fraction < MIN_CALIBRATED_FAMILY_FRACTION or family5_abstains:
        decision = "INCONCLUSIVE_CALIBRATION"
        reason = "target calibration coverage is below the frozen panel requirement"
    elif family5_semantic_failure or semantic_failure_fraction > MAX_FAILURE_FRACTION:
        decision = "VAE_SOURCE_IDENTITY_LOSS"
        reason = "decoded source intervention fails the frozen semantic identity gate"
    elif family5_latent_failure or latent_failure_fraction > MAX_FAILURE_FRACTION:
        decision = "RECOVERABLE_BUT_ENTANGLED"
        reason = "source identity survives decoding but latent directions are unstable or aliased"
    elif stable_source_fraction >= MIN_STABLE_SOURCE_FRACTION:
        decision = "STABLE_RECOVERABLE"
        reason = "semantic identity and latent intervention directions pass the frozen panel gate"
    else:
        decision = "INCONCLUSIVE_MIXED"
        reason = "failure fractions are mixed between the frozen categorical boundaries"

    family_rows = []
    for family in families:
        sources = _family_source_rows(family)
        family_rows.append(
            {
                "family_rank": int(family["family_rank"]),
                "family_id": str(family["family_id"]),
                "source_count": int(family["source_count"]),
                "semantic_pass_sources": sum(
                    source["semantic_status"] == "PASS" for source in sources
                ),
                "latent_pass_sources": sum(
                    source["latent_status"] == "PASS" for source in sources
                ),
                "outcome": str(family["outcome"]),
            }
        )
    return {
        "schema": "stable_audio_tools.vae_mixture_state_identifiability_summary",
        "schema_version": 1,
        "decision": decision,
        "reason": reason,
        "expected_family_ranks": expected,
        "family_count": len(families),
        "calibrated_family_count": len(calibrated_families),
        "calibrated_family_fraction": calibrated_family_fraction,
        "source_count": len(all_sources),
        "calibrated_source_count": len(calibrated_sources),
        "semantic_failure_source_count": len(semantic_failures),
        "semantic_failure_fraction": semantic_failure_fraction,
        "semantic_pass_source_count": len(semantic_pass_sources),
        "latent_failure_source_count": len(latent_failures),
        "latent_failure_fraction_on_semantic_pass": latent_failure_fraction,
        "stable_source_count": len(stable_sources),
        "stable_source_fraction": stable_source_fraction,
        "branch_diagnostics": {
            "isolated_pass_source_count": len(isolated_passes),
            "intervention_pass_source_count": len(intervention_passes),
            "intervention_clap_7of8_source_count": len(intervention_clap_passes),
            "intervention_ast_7of8_source_count": len(intervention_ast_passes),
            "intervention_ast_retention_pass_source_count": len(
                intervention_ast_retention_passes
            ),
            "intervention_rms_pass_source_count": len(intervention_rms_passes),
            "clap_pass_ast_assignment_fail_source_count": len(
                clap_pass_ast_assignment_failures
            ),
            "source_count": len(branch_rows),
        },
        "family5": {
            "outcome": family5["outcome"],
            "semantic_failure": family5_semantic_failure,
            "latent_failure": family5_latent_failure,
            "abstains": family5_abstains,
        },
        "thresholds": {
            "min_calibrated_family_fraction": MIN_CALIBRATED_FAMILY_FRACTION,
            "max_failure_fraction": MAX_FAILURE_FRACTION,
            "min_stable_source_fraction": MIN_STABLE_SOURCE_FRACTION,
            "family5_mandatory": True,
        },
        "audit_contract": {
            field: reference[field] for field in contract_fields
        },
        "worker_family_ranks": worker_rows,
        "families": family_rows,
    }


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Frozen VAE mixture-state identifiability conclusion",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        str(summary["reason"]),
        "",
        "| panel metric | value |",
        "|---|---:|",
        f"| calibrated families | {summary['calibrated_family_count']}/{summary['family_count']} ({summary['calibrated_family_fraction']:.3f}) |",
        f"| calibrated sources | {summary['calibrated_source_count']}/{summary['source_count']} |",
        f"| semantic failures | {summary['semantic_failure_source_count']} ({summary['semantic_failure_fraction']:.3f}) |",
        f"| latent failures after semantic pass | {summary['latent_failure_source_count']} ({summary['latent_failure_fraction_on_semantic_pass']:.3f}) |",
        f"| stable sources | {summary['stable_source_count']} ({summary['stable_source_fraction']:.3f}) |",
        f"| family 5 | {summary['family5']['outcome']} |",
        f"| isolated semantic passes | {summary['branch_diagnostics']['isolated_pass_source_count']}/{summary['branch_diagnostics']['source_count']} |",
        f"| decoded intervention semantic passes | {summary['branch_diagnostics']['intervention_pass_source_count']}/{summary['branch_diagnostics']['source_count']} |",
        f"| intervention CLAP at least 7/8 | {summary['branch_diagnostics']['intervention_clap_7of8_source_count']}/{summary['branch_diagnostics']['source_count']} |",
        f"| intervention AST at least 7/8 | {summary['branch_diagnostics']['intervention_ast_7of8_source_count']}/{summary['branch_diagnostics']['source_count']} |",
        f"| CLAP pass but AST assignment fail | {summary['branch_diagnostics']['clap_pass_ast_assignment_fail_source_count']} |",
        "",
        "| family | sources | semantic pass | latent pass | outcome |",
        "|---:|---:|---:|---:|---|",
    ]
    for family in summary["families"]:
        lines.append(
            f"| {family['family_rank']} | {family['source_count']} | "
            f"{family['semantic_pass_sources']} | {family['latent_pass_sources']} | "
            f"{family['outcome']} |"
        )
    lines.extend(
        [
            "",
            "The audit is target-side only. It creates no training checkpoint, no "
            "per-source inference route, and no post-hoc waveform mixture.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    if args.output_json.exists() or args.output_markdown.exists():
        raise FileExistsError("refusing to overwrite an existing summary")
    reports = []
    for path in args.report:
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    summary = summarize(reports)
    atomic_write_json(args.output_json, summary)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({"decision": summary["decision"], "output": str(args.output_json.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
