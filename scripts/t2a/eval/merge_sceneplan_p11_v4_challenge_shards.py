#!/usr/bin/env python3
"""Fail-closed merge for disjoint P11-v4 challenge evaluator shards.

The evaluator intentionally emits a complete, independently auditable report
for every family shard.  This utility accepts only reports that are identical
on every scientific/runtime invariant, proves that their selected ordinals are
disjoint and exactly cover the requested challenge slice, and recomputes the
aggregate metrics from public per-row prefixes.  It never averages already
aggregated shard summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stable_audio_tools.data.p11_challenge_metrics import _mean, _aggregate
from functools import partial
from stable_audio_tools.data.artifact_io import digest, sha as _sha256_file

# Existing report hashes serialize nonfinite diagnostics using JSON's legacy policy.
_json_sha256 = partial(digest, allow_nan=True)


EXPECTED_SCHEMA = "stable_audio_tools.p11_v4_unified_challenge_eval"
EXPECTED_SCHEMA_VERSION = 10
DEFAULT_FAMILIES = (
    "editing_counterfactual_causality",
    "exact_compatibility",
    "generation_numeric_posterior",
    "understanding_degraded_evidence",
)






def _load_report(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"report must contain one JSON object: {resolved}")
    claimed = value.get("report_sha256_without_self")
    unhashed = dict(value)
    unhashed.pop("report_sha256_without_self", None)
    actual = _json_sha256(unhashed)
    if claimed != actual:
        raise RuntimeError(
            f"report self-hash mismatch for {resolved}: {claimed} != {actual}"
        )
    value["_source_path"] = str(resolved)
    value["_source_file_sha256"] = _sha256_file(resolved)
    return value






def _counterfactual_summary(
    rows: Sequence[Mapping[str, Any]], k_values: Sequence[int]
) -> dict[str, Any]:
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("pair_id"):
            pairs[str(row["pair_id"])].append(row)
    output: dict[str, Any] = {}
    for k in k_values:
        successes: list[bool] = []
        contrasts: list[bool] = []
        for values in pairs.values():
            if len(values) != 2 or any(
                len(value.get("scored", [])) < k for value in values
            ):
                continue
            for draw in range(k):
                left = values[0]["scored"][draw]
                right = values[1]["scored"][draw]
                if not left.get("valid") or not right.get("valid"):
                    successes.append(False)
                    contrasts.append(False)
                    continue
                successes.append(
                    left.get("patch_exact") is True
                    and right.get("patch_exact") is True
                )
                contrasts.append(left.get("patch") != right.get("patch"))
        output[str(k)] = {
            "pairs": len(pairs),
            "matched_draws": len(successes),
            "both_counterfactual_targets_correct_rate": _mean(
                [float(value) for value in successes]
            ),
            "prompt_counterfactual_changes_patch_rate": _mean(
                [float(value) for value in contrasts]
            ),
        }
    return output


def _expected_ordinals(
    challenge: Path,
    *,
    rows_per_view: int,
    families: set[str],
) -> list[int]:
    connection = sqlite3.connect(
        f"file:{challenge}?mode=ro&immutable=1", uri=True
    )
    try:
        placeholders = ",".join("?" for _ in families)
        records = connection.execute(
            "SELECT ordinal,family,view_id,pair_id FROM rows "
            f"WHERE family IN ({placeholders}) ORDER BY ordinal",
            sorted(families),
        ).fetchall()
    finally:
        connection.close()

    by_view: dict[tuple[str, str], list[int]] = defaultdict(list)
    edit_pairs: dict[str, list[int]] = defaultdict(list)
    for ordinal, family, view_id, pair_id in records:
        if str(family) == "editing_counterfactual_causality":
            edit_pairs[str(pair_id)].append(int(ordinal))
        else:
            by_view[(str(family), str(view_id))].append(int(ordinal))
    selected: list[int] = []
    if rows_per_view == 0:
        selected.extend(int(record[0]) for record in records)
    else:
        for values in by_view.values():
            selected.extend(values[:rows_per_view])
        for pair_id in sorted(edit_pairs)[:rows_per_view]:
            values = edit_pairs[pair_id]
            if len(values) != 2:
                raise RuntimeError(f"challenge edit pair is incomplete: {pair_id}")
            selected.extend(values)
    return sorted(set(selected))


def _strip_private(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument(
        "--required-families", default=",".join(DEFAULT_FAMILIES)
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = [_load_report(path) for path in args.shard]
    required_families = {
        item.strip() for item in args.required_families.split(",") if item.strip()
    }
    if not required_families:
        raise ValueError("--required-families must not be empty")
    reference = reports[0]
    if (
        reference.get("schema") != EXPECTED_SCHEMA
        or int(reference.get("schema_version", -1)) != EXPECTED_SCHEMA_VERSION
    ):
        raise RuntimeError("first shard is not a P11-v4 evaluator-v10 report")

    invariants = (
        "schema",
        "schema_version",
        "evaluator_contract",
        "status",
        "arm",
        "architecture",
        "model_config",
        "dataset_config",
        "config_provenance",
        "lexical_evidence_provenance",
        "lexical_authority_intervention",
        "qwen_kernel_mode",
        "scientific_kernel_contract",
        "challenge",
        "challenge_sha256",
        "runtime_source_provenance",
        "checkpoints",
        "scoring_checkpoint_reload",
        "weights",
        "root_seed",
        "draws",
        "k_values",
        "d0_temperature",
        "discrete_decode_mode",
        "rows_per_view",
        "rng_contract",
        "posterior_fairness",
        "calibration_contract",
        "p10_closure",
    )
    seen_families: set[str] = set()
    rows_by_ordinal: dict[int, Mapping[str, Any]] = {}
    source_shards: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    for report in reports:
        if report.get("status") != "PASS":
            raise RuntimeError(f"non-passing shard: {report['_source_path']}")
        for key in invariants:
            if report.get(key) != reference.get(key):
                raise RuntimeError(
                    f"shard invariant {key!r} differs: {report['_source_path']}"
                )
        declared = report.get("families_filter")
        if not isinstance(declared, list) or len(declared) != 1:
            raise RuntimeError("every merge input must select exactly one family")
        family = str(declared[0])
        if family in seen_families:
            raise RuntimeError(f"duplicate family shard: {family}")
        seen_families.add(family)
        selected = [int(value) for value in report.get("selected_ordinals", [])]
        public_rows = report.get("aggregate_rows")
        if not isinstance(public_rows, list) or len(public_rows) != len(selected):
            raise RuntimeError(f"row payload/count mismatch in family {family}")
        if int(report.get("rows", -1)) != len(selected):
            raise RuntimeError(f"selected row/count mismatch in family {family}")
        if float(report.get("rng_isolation_rate", -1.0)) != 1.0:
            raise RuntimeError(f"RNG isolation failed in family {family}")
        row_ordinals = [int(row["ordinal"]) for row in public_rows]
        if sorted(row_ordinals) != sorted(selected):
            raise RuntimeError(f"public rows disagree with selector in family {family}")
        if any(str(row["family"]) != family for row in public_rows):
            raise RuntimeError(f"foreign family row found in shard {family}")
        for row in public_rows:
            ordinal = int(row["ordinal"])
            if ordinal in rows_by_ordinal:
                raise RuntimeError(f"ordinal occurs in multiple shards: {ordinal}")
            rows_by_ordinal[ordinal] = row
            if not bool(row.get("rng_isolation_pass")):
                raise RuntimeError(f"row RNG isolation failed: {ordinal}")
            if len(row.get("scored", [])) != int(reference["draws"]):
                raise RuntimeError(f"row draw count mismatch: {ordinal}")
        source_shards.append(
            {
                "path": report["_source_path"],
                "file_sha256": report["_source_file_sha256"],
                "report_sha256_without_self": report[
                    "report_sha256_without_self"
                ],
                "family": family,
                "rows": len(public_rows),
            }
        )
        for item in report.get("performance", []):
            performance.append(
                {
                    **dict(item),
                    "source_family": family,
                    "source_report": report["_source_path"],
                }
            )

    if seen_families != required_families:
        raise RuntimeError(
            "family coverage mismatch: "
            f"seen={sorted(seen_families)}, required={sorted(required_families)}"
        )
    challenge = Path(str(reference["challenge"])).resolve(strict=True)
    if _sha256_file(challenge) != reference["challenge_sha256"]:
        raise RuntimeError("immutable challenge SHA256 changed during merge")
    expected = _expected_ordinals(
        challenge,
        rows_per_view=int(reference["rows_per_view"]),
        families=required_families,
    )
    actual = sorted(rows_by_ordinal)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise RuntimeError(
            f"merged challenge coverage mismatch: missing={missing[:20]}, "
            f"extra={extra[:20]}"
        )

    rows = [rows_by_ordinal[ordinal] for ordinal in actual]
    k_values = [int(value) for value in reference["k_values"]]
    merged = _strip_private(reference)
    merged.update(
        {
            "scope": (
                "strict merge of disjoint evaluator-v10 family shards; "
                "matched P10-conditioning challenge evaluation; no rendered FOA"
            ),
            "selected_ordinals": actual,
            "rows": len(rows),
            "families_filter": None,
            "rng_isolation_rate": _mean(
                [float(bool(row["rng_isolation_pass"])) for row in rows]
            ),
            "performance": performance,
            "aggregate": _aggregate(rows, k_values),
            "editing_counterfactual": _counterfactual_summary(rows, k_values),
            "aggregate_rows": rows,
            "quality_decision": "NOT_ESTABLISHED_BY_SHARD_MERGE",
            "shard_merge": {
                "contract": "p11_v4_fail_closed_disjoint_family_merge_v2",
                "required_families": sorted(required_families),
                "exact_challenge_coverage": True,
                "disjoint_ordinals": True,
                "aggregates_recomputed_from_public_rows": True,
                "source_shards": sorted(
                    source_shards, key=lambda item: item["family"]
                ),
            },
        }
    )
    merged.pop("report_sha256_without_self", None)
    merged["report_sha256_without_self"] = _json_sha256(merged)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(output),
                "rows": len(rows),
                "families": sorted(required_families),
                "report_sha256_without_self": merged[
                    "report_sha256_without_self"
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
