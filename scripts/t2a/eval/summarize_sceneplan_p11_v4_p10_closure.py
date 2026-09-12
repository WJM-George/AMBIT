#!/usr/bin/env python3
"""Fail-closed cross-arm summary for the frozen-P10 P11-v4 closure panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_SCHEMA = "stable_audio_tools.p11_v4_frozen_p10_foa_closure"
EXPECTED_VERSION = 2
ARMS = ("d0", "direct", "flow")


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path, *, arm: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        report.get("schema") != EXPECTED_SCHEMA
        or int(report.get("schema_version", -1)) != EXPECTED_VERSION
        or report.get("arm") != arm
        or report.get("status") != "PASS"
    ):
        raise RuntimeError(f"{arm} is not a passing P10 closure-v2 report")
    claimed = report.get("report_sha256_without_self")
    unhashed = dict(report)
    unhashed.pop("report_sha256_without_self", None)
    if claimed != _json_sha256(unhashed):
        raise RuntimeError(f"{arm} closure self-hash mismatch")
    report["_path"] = str(resolved)
    report["_file_sha256"] = _sha256_file(resolved)
    return report


def _row_map(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    rows = {int(row["ordinal"]): row for row in report["rows"]}
    if len(rows) != len(report["rows"]):
        raise RuntimeError("closure report repeats an ordinal")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = {arm: _load(getattr(args, arm), arm=arm) for arm in ARMS}
    reference = reports["flow"]
    invariant_fields = (
        "root_seed",
        "rows_per_task",
        "selected_ordinals",
        "selection_contract",
        "sampling",
        "p10",
        "weights",
    )
    comparability: dict[str, bool] = {}
    for arm, report in reports.items():
        for field in invariant_fields:
            comparability[f"{arm}:{field}"] = report.get(field) == reference.get(field)
        comparability[f"{arm}:challenge"] = (
            report["quality_report"]["challenge_sha256"]
            == reference["quality_report"]["challenge_sha256"]
        )
        comparability[f"{arm}:evaluator"] = (
            report["quality_report"]["evaluator_contract"]
            == reference["quality_report"]["evaluator_contract"]
        )
        comparability[f"{arm}:integrity"] = all(
            value is True for value in report["integrity_gates"].values()
        )
    failed = [key for key, value in comparability.items() if not value]
    if failed:
        raise RuntimeError(f"closure reports are incomparable: {failed}")

    row_maps = {arm: _row_map(report) for arm, report in reports.items()}
    row_checks = []
    for ordinal in reference["selected_ordinals"]:
        ordinal = int(ordinal)
        flow_row = row_maps["flow"][ordinal]
        identity_fields = (
            "challenge_id",
            "task",
            "family",
            "view_id",
            "edit_operation",
            "render_seed",
            "target_plan_sha256",
            "target_shape",
        )
        identity_exact = all(
            row_maps[arm][ordinal].get(field) == flow_row.get(field)
            for arm in ARMS
            for field in identity_fields
        )
        target_foa_exact = len(
            {row_maps[arm][ordinal]["target_foa_sha256"] for arm in ARMS}
        ) == 1
        current_hashes = {
            row_maps[arm][ordinal].get("current_foa_sha256") for arm in ARMS
        }
        current_foa_exact = len(current_hashes) == 1
        row_checks.append(
            {
                "ordinal": ordinal,
                "challenge_id": flow_row["challenge_id"],
                "task": flow_row["task"],
                "identity_exact": identity_exact,
                "target_foa_exact": target_foa_exact,
                "current_foa_exact": current_foa_exact,
                "pass": identity_exact and target_foa_exact and current_foa_exact,
            }
        )

    gates = {
        "reports_comparable": all(comparability.values()),
        "all_integrity_gates_pass": all(
            all(value is True for value in report["integrity_gates"].values())
            for report in reports.values()
        ),
        "same_target_and_current_render_per_row": all(
            row["pass"] for row in row_checks
        ),
        "canonical_p10_v11_150k": (
            reference["p10"]["release_id"]
            == "p10-sceneplan-dit-v11-step150000"
            and int(reference["p10"]["checkpoint_step"]) == 150000
        ),
        "canonical_100_step_same_seed_sampler": (
            int(reference["sampling"]["steps"]) == 100
            and all(row["render_seed"] is not None for row in reference["rows"])
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    output = {
        "schema": "stable_audio_tools.p11_v4_cross_arm_p10_closure_summary",
        "schema_version": 1,
        "status": status,
        "decision": (
            "MATCHED_FROZEN_P10_CLOSURE_ESTABLISHED"
            if status == "PASS"
            else "MATCHED_FROZEN_P10_CLOSURE_NOT_ESTABLISHED"
        ),
        "scope": (
            "matched 12-row G/U/E panel through canonical P10-v11 150k; "
            "this proves executor closure, not Flow quality superiority"
        ),
        "inputs": {
            arm: {
                "path": report["_path"],
                "file_sha256": report["_file_sha256"],
                "report_sha256_without_self": report["report_sha256_without_self"],
            }
            for arm, report in reports.items()
        },
        "comparability_gates": comparability,
        "gates": gates,
        "p10": reference["p10"],
        "sampling": reference["sampling"],
        "selected_ordinals": reference["selected_ordinals"],
        "row_checks": row_checks,
        "arms": {
            arm: {
                "aggregate": report["aggregate"],
                "performance": report["performance"],
                "p11_checkpoint_sha256": report["p11"]["checkpoint_sha256"],
            }
            for arm, report in reports.items()
        },
        "claim_boundary": {
            "frozen_p10_audio_closure_established": status == "PASS",
            "flow_quality_superiority_established": False,
            "full_corpus_training_authorized": False,
        },
    }
    output["report_sha256_without_self"] = _json_sha256(output)
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
