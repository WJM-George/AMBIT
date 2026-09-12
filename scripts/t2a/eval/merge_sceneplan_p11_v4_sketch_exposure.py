#!/usr/bin/env python3
"""Merge disjoint P11-v4 sketch-exposure evaluator shards fail-closed."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.evaluate_sceneplan_p11_v4_challenge import (  # noqa: E402
    _json_sha256,
    _sha256_file,
)
from scripts.t2a.eval.evaluate_sceneplan_p11_v4_sketch_exposure import (  # noqa: E402
    SCHEMA,
    SCHEMA_VERSION,
    _summarize_exposure_rows,
)


def _load(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        report.get("schema") != SCHEMA
        or int(report.get("schema_version", -1)) != SCHEMA_VERSION
        or report.get("status") != "PASS"
    ):
        raise RuntimeError(f"not a passing sketch-exposure report: {resolved}")
    claimed = report.get("report_sha256_without_self")
    unhashed = dict(report)
    unhashed.pop("report_sha256_without_self", None)
    if claimed != _json_sha256(unhashed):
        raise RuntimeError(f"sketch-exposure self-hash mismatch: {resolved}")
    report["_path"] = str(resolved)
    report["_file_sha256"] = _sha256_file(resolved)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [_load(path) for path in args.input]
    if len(reports) < 2:
        raise ValueError("exposure merge requires at least two shards")

    reference = reports[0]
    invariants = (
        "scope",
        "interpretation_contract",
        "checkpoint",
        "model_config",
        "dataset_config",
        "challenge",
        "challenge_sha256",
        "qwen_kernel_mode",
        "weights",
        "root_seed",
    )
    checks = {
        f"shard_{index}:{key}": report.get(key) == reference.get(key)
        for index, report in enumerate(reports)
        for key in invariants
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"incomparable exposure shards: {failed}")

    shard_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        selection_shard = report.get("selection_shard") or {}
        if selection_shard.get("contract") != "post_filter_round_robin_v1":
            raise RuntimeError("exposure report lacks canonical selection shard")
        key = (
            tuple(report.get("families_filter") or ()),
            tuple(report.get("tasks_filter") or ()),
            tuple(report.get("views_filter") or ()),
        )
        shard_groups[key].append(report)
    shard_coverage: dict[str, Any] = {}
    for key, group in sorted(shard_groups.items(), key=lambda item: str(item[0])):
        contracts = [report["selection_shard"] for report in group]
        num_shards = {int(value["num_shards"]) for value in contracts}
        pre_shard_rows = {int(value["pre_shard_rows"]) for value in contracts}
        indexes = {int(value["shard_index"]) for value in contracts}
        if len(num_shards) != 1 or len(pre_shard_rows) != 1:
            raise RuntimeError(f"inconsistent selection shards for {key}")
        expected_shards = next(iter(num_shards))
        expected_rows = next(iter(pre_shard_rows))
        observed_rows = sum(len(report["rows"]) for report in group)
        if indexes != set(range(expected_shards)) or observed_rows != expected_rows:
            raise RuntimeError(f"incomplete selection shard coverage for {key}")
        shard_coverage[str(key)] = {
            "num_shards": expected_shards,
            "shard_indexes": sorted(indexes),
            "pre_shard_rows": expected_rows,
            "observed_rows": observed_rows,
            "complete": True,
        }

    rows = sorted(
        [row for report in reports for row in report["rows"]],
        key=lambda row: int(row["ordinal"]),
    )
    ordinals = [int(row["ordinal"]) for row in rows]
    if len(ordinals) != len(set(ordinals)):
        raise RuntimeError("exposure shards repeat challenge ordinals")
    output = {
        "schema": "stable_audio_tools.p11_v4_sketch_exposure_merge",
        "schema_version": 1,
        "status": "PASS",
        "scope": reference["scope"],
        "interpretation_contract": reference["interpretation_contract"],
        "checkpoint": reference["checkpoint"],
        "model_config": reference["model_config"],
        "dataset_config": reference["dataset_config"],
        "challenge": reference["challenge"],
        "challenge_sha256": reference["challenge_sha256"],
        "qwen_kernel_mode": reference["qwen_kernel_mode"],
        "weights": reference["weights"],
        "root_seed": reference["root_seed"],
        "selected_ordinals": ordinals,
        "summary": _summarize_exposure_rows(rows),
        "rows": rows,
        "shards": [
            {
                "path": report["_path"],
                "file_sha256": report["_file_sha256"],
                "report_sha256_without_self": report[
                    "report_sha256_without_self"
                ],
                "selected_ordinals": report["selected_ordinals"],
                "tasks_filter": report["tasks_filter"],
                "views_filter": report["views_filter"],
                "selection_shard": report["selection_shard"],
            }
            for report in reports
        ],
        "selection_shard_coverage": shard_coverage,
        "comparability_gates": checks,
    }
    output["report_sha256_without_self"] = _json_sha256(output)
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "output": str(destination),
                "rows": len(rows),
                "summary": output["summary"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
