#!/usr/bin/env python3
"""Project evaluator-v10 K=1/4/8 output to its deterministic first draw.

The unified challenge evaluator already computes every K=1 metric from draw
zero before extending the same row to K=4 and K=8.  Frozen-P10 closure needs a
strict one-draw report so that it can re-decode exactly one prediction.  This
utility performs that lossless projection, recomputes all public aggregates,
and fails unless they are identical to the source report's K=1 prefixes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from merge_sceneplan_p11_v4_challenge_shards import (
    EXPECTED_SCHEMA,
    EXPECTED_SCHEMA_VERSION,
    _aggregate,
    _counterfactual_summary,
    _json_sha256,
    _sha256_file,
)


PROJECTION_CONTRACT = "p11_v4_evaluator_v10_first_draw_k1_projection_v1"


def _load_source(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise TypeError("source report must be one JSON object")
    if (
        report.get("schema") != EXPECTED_SCHEMA
        or int(report.get("schema_version", -1)) != EXPECTED_SCHEMA_VERSION
        or report.get("status") != "PASS"
    ):
        raise RuntimeError("source is not a passing evaluator-v10 report")
    claimed = report.get("report_sha256_without_self")
    unhashed = dict(report)
    unhashed.pop("report_sha256_without_self", None)
    if claimed != _json_sha256(unhashed):
        raise RuntimeError("source report self-hash mismatch")
    draws = int(report.get("draws", -1))
    k_values = [int(value) for value in report.get("k_values", [])]
    if draws < 1 or 1 not in k_values:
        raise RuntimeError("source report lacks a K=1 first draw")
    if int(report.get("rows", -1)) != len(report.get("selected_ordinals", [])):
        raise RuntimeError("source row count differs from selected ordinals")
    rows = report.get("aggregate_rows")
    if not isinstance(rows, list) or len(rows) != int(report["rows"]):
        raise RuntimeError("source public row payload is incomplete")
    return resolved, report


def _source_k1_aggregate(report: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise RuntimeError("source aggregate is missing")
    result: dict[str, Any] = {}
    for group, values in aggregate.items():
        if not isinstance(values, Mapping) or "1" not in values:
            raise RuntimeError(f"source aggregate group lacks K=1: {group}")
        result[str(group)] = {"1": values["1"]}
    return result


def _source_k1_counterfactual(report: Mapping[str, Any]) -> dict[str, Any]:
    counterfactual = report.get("editing_counterfactual")
    if not isinstance(counterfactual, Mapping) or "1" not in counterfactual:
        raise RuntimeError("source editing counterfactual lacks K=1")
    return {"1": counterfactual["1"]}


def _project_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    draws = int(report["draws"])
    projected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source_row in report["aggregate_rows"]:
        row = dict(source_row)
        ordinal = int(row["ordinal"])
        if ordinal in seen:
            raise RuntimeError(f"duplicate source ordinal: {ordinal}")
        seen.add(ordinal)
        scored = row.get("scored")
        prefixes = row.get("prefixes")
        if not isinstance(scored, list) or len(scored) != draws:
            raise RuntimeError(f"source draw count mismatch at ordinal {ordinal}")
        if not isinstance(prefixes, Mapping) or "1" not in prefixes:
            raise RuntimeError(f"source K=1 prefix missing at ordinal {ordinal}")
        row["scored"] = [scored[0]]
        row["prefixes"] = {"1": prefixes["1"]}
        projected.append(row)
    expected = sorted(int(value) for value in report["selected_ordinals"])
    if sorted(seen) != expected:
        raise RuntimeError("projected rows differ from selected ordinals")
    return projected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_path, source = _load_source(args.input)
    rows = _project_rows(source)
    aggregate = _aggregate(rows, [1])
    counterfactual = _counterfactual_summary(rows, [1])
    aggregate_exact = aggregate == _source_k1_aggregate(source)
    counterfactual_exact = (
        counterfactual == _source_k1_counterfactual(source)
    )
    if not aggregate_exact or not counterfactual_exact:
        raise RuntimeError(
            "first-draw projection does not reproduce source K=1 summaries"
        )

    projection_source = Path(__file__).resolve(strict=True)
    aggregate_source = Path(
        _aggregate.__code__.co_filename
    ).resolve(strict=True)
    report = dict(source)
    report.update(
        {
            "scope": (
                "strict first-draw K=1 projection of a passing evaluator-v10 "
                "report for frozen-P10 closure"
            ),
            "draws": 1,
            "k_values": [1],
            "performance": [],
            "aggregate": aggregate,
            "editing_counterfactual": counterfactual,
            "aggregate_rows": rows,
            "quality_decision": "NOT_ESTABLISHED_BY_K1_PROJECTION",
            "k1_projection": {
                "contract": PROJECTION_CONTRACT,
                "source_report": str(source_path),
                "source_file_sha256": _sha256_file(source_path),
                "source_report_sha256_without_self": source[
                    "report_sha256_without_self"
                ],
                "source_draws": int(source["draws"]),
                "source_k_values": [int(value) for value in source["k_values"]],
                "selected_draw_index": 0,
                "aggregate_k1_exact": aggregate_exact,
                "editing_counterfactual_k1_exact": counterfactual_exact,
                "inference_rerun": False,
                "projection_source": str(projection_source),
                "projection_source_sha256": _sha256_file(projection_source),
                "aggregate_source": str(aggregate_source),
                "aggregate_source_sha256": _sha256_file(aggregate_source),
            },
        }
    )
    posterior_fairness = dict(report.get("posterior_fairness") or {})
    posterior_fairness["k1_is_source_draw_zero_projection"] = True
    report["posterior_fairness"] = posterior_fairness
    report.pop("report_sha256_without_self", None)
    report["report_sha256_without_self"] = _json_sha256(report)

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing projection output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "rows": len(rows),
                "source_draws": int(source["draws"]),
                "aggregate_k1_exact": aggregate_exact,
                "editing_counterfactual_k1_exact": counterfactual_exact,
                "report_sha256_without_self": report[
                    "report_sha256_without_self"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
