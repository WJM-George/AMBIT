#!/usr/bin/env python3
"""Audit P11-U source inventory before any frozen-P10 rendering.

This is deliberately a planner-only diagnostic.  It compares the first
deployable draw from matched evaluator-v10 reports, stratifies source-count
accuracy by the hidden held-out count, and separates rows whose decoded
room/count/kind structure is exact from rows with a structural error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHALLENGE = REPO_ROOT / (
    "artifacts/sceneplan_p11/challenges/"
    "p11_v4_heldout_challenge_v1_20260901.sqlite"
)
QUALITY_SCHEMA = "stable_audio_tools.p11_v4_unified_challenge_eval"
QUALITY_SCHEMA_VERSION = 10
ATTRIBUTION_CONTRACT = "p11_pre_render_understanding_inventory_audit_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _load_report(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        report.get("schema") != QUALITY_SCHEMA
        or int(report.get("schema_version", -1)) != QUALITY_SCHEMA_VERSION
        or report.get("status") != "PASS"
    ):
        raise RuntimeError(f"not a passing evaluator-v10 report: {resolved}")
    claimed = report.get("report_sha256_without_self")
    unhashed = dict(report)
    unhashed.pop("report_sha256_without_self", None)
    if claimed != _json_sha256(unhashed):
        raise RuntimeError(f"quality report self-hash mismatch: {resolved}")
    return resolved, report


def _parse_named_report(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--report must be NAME=PATH")
    return name.strip(), Path(path.strip())


def _target_counts(
    challenge: Path, ordinals: Sequence[int]
) -> dict[int, int]:
    connection = sqlite3.connect(str(challenge))
    try:
        output: dict[int, int] = {}
        for ordinal in sorted(set(int(value) for value in ordinals)):
            row = connection.execute(
                "SELECT target_sceneplan_zlib FROM rows WHERE ordinal = ?",
                (ordinal,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"challenge lacks ordinal {ordinal}")
            plan = json.loads(zlib.decompress(row[0]))
            count = len(plan.get("sources") or [])
            if count not in {1, 2, 3, 4}:
                raise RuntimeError(f"challenge ordinal {ordinal} has count {count}")
            output[ordinal] = count
        return output
    finally:
        connection.close()


def _structure_exact(metrics: Mapping[str, Any]) -> bool:
    return all(
        float(metrics.get(key, 0.0)) >= 1.0 - 1.0e-9
        for key in ("room_accuracy", "source_count_accuracy", "kind_accuracy")
    )


def _summarize(
    rows: Sequence[Mapping[str, Any]], target_counts: Mapping[int, int]
) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    for row in rows:
        scored = row.get("scored") or []
        if not scored:
            raise RuntimeError("quality row has no deployable draw")
        draw = scored[0]
        if draw.get("valid") is not True:
            raise RuntimeError("inventory audit requires valid first draws")
        metrics = draw["task_metrics"]
        predicted_count = len(draw["sceneplan"]["sources"])
        target_count = int(target_counts[int(row["ordinal"])])
        values.append(
            {
                "ordinal": int(row["ordinal"]),
                "target_count": target_count,
                "predicted_count": predicted_count,
                "count_exact": predicted_count == target_count,
                "structure_exact": _structure_exact(metrics),
                "task_score": float(draw["task_score"]),
                "continuous_reference_rmse": float(
                    draw["reference_anchor_nearest_core_rmse"]
                ),
            }
        )

    exact = [row for row in values if row["structure_exact"]]
    wrong = [row for row in values if not row["structure_exact"]]
    by_count: dict[str, Any] = {}
    for count in range(1, 5):
        subset = [row for row in values if row["target_count"] == count]
        predicted = Counter(int(row["predicted_count"]) for row in subset)
        by_count[str(count)] = {
            "rows": len(subset),
            "source_count_accuracy": _mean(
                [float(row["count_exact"]) for row in subset]
            ),
            "predicted_count_distribution": {
                str(key): int(value) for key, value in sorted(predicted.items())
            },
            "task_score": _mean([float(row["task_score"]) for row in subset]),
        }

    def subset_summary(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(subset),
            "task_score": _mean([float(row["task_score"]) for row in subset]),
            "continuous_reference_rmse": _mean(
                [float(row["continuous_reference_rmse"]) for row in subset]
            ),
        }

    return {
        "rows": len(values),
        "source_count_accuracy": _mean(
            [float(row["count_exact"]) for row in values]
        ),
        "structure_exact_rate": _mean(
            [float(row["structure_exact"]) for row in values]
        ),
        "task_score": _mean([float(row["task_score"]) for row in values]),
        "structure_exact": subset_summary(exact),
        "structure_mismatch": subset_summary(wrong),
        "by_target_source_count": by_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", action="append", type=_parse_named_report, required=True
    )
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--view", default="audio_evidence_exact_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.report) < 2:
        raise ValueError("inventory audit requires at least two matched arms")
    names = [name for name, _ in args.report]
    if len(names) != len(set(names)):
        raise ValueError("--report arm names must be unique")

    challenge = args.challenge.expanduser().resolve(strict=True)
    challenge_sha256 = _sha256_file(challenge)
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {
        name: _load_report(path) for name, path in args.report
    }
    for name, (_, report) in loaded.items():
        if report.get("challenge_sha256") != challenge_sha256:
            raise RuntimeError(f"{name} uses a different held-out challenge")
        if int(report.get("root_seed", -1)) != int(args.seed):
            raise RuntimeError(f"{name} does not use seed {args.seed}")

    selected: dict[str, list[Mapping[str, Any]]] = {}
    ordinal_sets: dict[str, set[int]] = {}
    for name, (_, report) in loaded.items():
        rows = [
            row
            for row in report["aggregate_rows"]
            if row["task"] == "understanding" and row["view_id"] == args.view
        ]
        if not rows:
            raise RuntimeError(f"{name} has no U rows for view {args.view}")
        selected[name] = rows
        ordinal_sets[name] = {int(row["ordinal"]) for row in rows}
    reference_ordinals = next(iter(ordinal_sets.values()))
    if any(values != reference_ordinals for values in ordinal_sets.values()):
        raise RuntimeError("inventory audit arms do not contain identical U rows")
    counts = _target_counts(challenge, sorted(reference_ordinals))

    output: dict[str, Any] = {
        "schema": "stable_audio_tools.p11_v4_u_inventory_audit",
        "schema_version": 1,
        "status": "PASS",
        "attribution_contract": {
            "contract": ATTRIBUTION_CONTRACT,
            "scope": "P11 planner outputs before rendering",
            "p10_loaded": False,
            "p10_metric_used": False,
            "p10_residual_attributed_to_p11": False,
        },
        "challenge": str(challenge),
        "challenge_sha256": challenge_sha256,
        "view": args.view,
        "root_seed": int(args.seed),
        "matched_ordinals": sorted(reference_ordinals),
        "arms": {
            name: {
                "source_report": str(path),
                "source_report_sha256": _sha256_file(path),
                "architecture": report.get("architecture"),
                "summary": _summarize(selected[name], counts),
            }
            for name, (path, report) in loaded.items()
        },
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
                "arms": {
                    name: value["summary"] for name, value in output["arms"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
