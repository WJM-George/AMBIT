#!/usr/bin/env python3
"""Freeze a deterministic 3,000-row slice of the complete P10 test split.

The outer quota preserves the complete test split's 1/2/3/4-source ratios
exactly up to largest-remainder rounding.  Selection within each source-count
quota is additionally stratified by scene composition, 432/648 length bucket,
source-kind multiset, and room type.  A namespace-keyed SHA256 ordering makes
the slice deterministic and independent of input row order.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Hashable, Iterable


DEFAULT_SOURCE = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_full_test_8000_semantic_v2"
)
DEFAULT_OUTPUT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_stratified_test_3000_semantic_v2"
)
NAMESPACE = "sceneplan-p10-v11-stratified-test-3000-v1-20260901"
EXPECTED_ROWS = 3_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def largest_remainder(
    counts: dict[Hashable, int], target: int
) -> dict[Hashable, int]:
    total = sum(counts.values())
    if target < 0 or target > total:
        raise ValueError(f"invalid quota target {target} for population {total}")
    exact = {key: target * value / total for key, value in counts.items()}
    quota = {key: int(value) for key, value in exact.items()}
    remaining = target - sum(quota.values())
    order = sorted(
        counts,
        key=lambda key: (
            -(exact[key] - quota[key]),
            -counts[key],
            repr(key),
        ),
    )
    for key in order[:remaining]:
        quota[key] += 1
    if sum(quota.values()) != target:
        raise RuntimeError("largest-remainder allocation failed")
    return quota


def fine_stratum(row: dict[str, Any]) -> tuple[Any, ...]:
    kind_signature = tuple(sorted(row["source_kind_counts"].items()))
    return (
        str(row["scene_composition"]),
        int(row["length_bucket"]),
        kind_signature,
        str(row["room_type"]),
    )


def selection_key(row: dict[str, Any]) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{NAMESPACE}\0{row['sample_id']}".encode("utf-8")
    ).hexdigest()
    return digest, str(row["sample_id"])


def distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_count = Counter(str(row["source_count"]) for row in rows)
    composition = Counter(str(row["scene_composition"]) for row in rows)
    length_bucket = Counter(str(row["length_bucket"]) for row in rows)
    room = Counter(str(row["room_type"]) for row in rows)
    source_kind_scene_coverage = {
        kind: sum(kind in row["source_kinds"] for row in rows)
        for kind in ("music", "sound", "speech")
    }
    source_kind_appearances = {
        kind: sum(int(row["source_kind_counts"].get(kind, 0)) for row in rows)
        for kind in ("music", "sound", "speech")
    }
    single_source = {
        kind: sum(
            int(row["source_count"]) == 1 and kind in row["source_kinds"]
            for row in rows
        )
        for kind in ("music", "sound", "speech")
    }
    return {
        "source_count": dict(sorted(source_count.items())),
        "scene_composition": dict(sorted(composition.items())),
        "length_bucket": dict(sorted(length_bucket.items())),
        "room_type": dict(sorted(room.items())),
        "source_kind_scene_coverage": source_kind_scene_coverage,
        "source_kind_appearances": source_kind_appearances,
        "single_source": single_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args()

    source = args.source_root.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve()
    target_rows = int(args.rows)
    if target_rows != EXPECTED_ROWS:
        raise ValueError(f"this frozen contract requires exactly {EXPECTED_ROWS} rows")

    source_contract_path = source / "EVAL_CONTRACT.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    if source_contract.get("status") != "FROZEN_FULL_TEST_MULTILABEL_SEMANTIC_V2":
        raise RuntimeError("source is not the canonical full semantic-v2 test contract")
    source_panel = source / source_contract["test_set"]["panel_filename"]
    if sha256_file(source_panel) != source_contract["test_set"]["panel_sha256"]:
        raise RuntimeError("source full-test panel digest changed")
    rows = read_jsonl(source_panel)
    if len(rows) != 8_000:
        raise RuntimeError(f"expected complete 8,000-row source panel, found {len(rows)}")

    by_source_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source_count[int(row["source_count"])].append(row)
    outer_counts = {key: len(value) for key, value in by_source_count.items()}
    outer_quota = largest_remainder(outer_counts, target_rows)

    selected: list[dict[str, Any]] = []
    fine_quota_audit: dict[str, dict[str, int]] = {}
    for source_count in sorted(by_source_count):
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in by_source_count[source_count]:
            grouped[fine_stratum(row)].append(row)
        counts = {key: len(value) for key, value in grouped.items()}
        quota = largest_remainder(counts, outer_quota[source_count])
        fine_quota_audit[str(source_count)] = {
            repr(key): int(quota[key]) for key in sorted(quota, key=repr)
        }
        for key, candidates in grouped.items():
            selected.extend(sorted(candidates, key=selection_key)[: quota[key]])

    selected.sort(key=lambda row: int(row["ordinal"]))
    if len(selected) != target_rows:
        raise RuntimeError(f"selection count changed: {len(selected)}")
    if len({row["sample_id"] for row in selected}) != target_rows:
        raise RuntimeError("stratified selection contains duplicate sample IDs")

    selected_distribution = distribution(selected)
    expected_outer = {str(key): value for key, value in sorted(outer_quota.items())}
    if selected_distribution["source_count"] != expected_outer:
        raise RuntimeError(
            "source-count quota mismatch: "
            f"{selected_distribution['source_count']} != {expected_outer}"
        )

    output.mkdir(parents=True, exist_ok=True)
    panel_path = output / "stratified_test_3000.jsonl"
    atomic_jsonl(panel_path, selected)

    contract = copy.deepcopy(source_contract)
    contract.update(
        {
            "schema": "stable_audio_tools.p10_v11_stratified_test_contract",
            "schema_version": 1,
            "status": "FROZEN_STRATIFIED_TEST_3000_SEMANTIC_V2",
            "purpose": (
                "publication-scale matched P10 and public-baseline evaluation "
                "with proportional 1/2/3/4-source coverage"
            ),
            "source_full_test_contract": str(source_contract_path),
            "source_full_test_contract_sha256": sha256_file(source_contract_path),
            "selection": {
                "namespace": NAMESPACE,
                "method": (
                    "largest-remainder source-count quotas, then largest-remainder "
                    "scene-composition/length/source-kind-multiset/room strata, "
                    "then SHA256 order"
                ),
                "source_rows": len(rows),
                "selected_rows": len(selected),
                "source_count_population": {
                    str(key): value for key, value in sorted(outer_counts.items())
                },
                "source_count_quota": expected_outer,
                "fine_quota": fine_quota_audit,
            },
        }
    )
    contract["test_set"].update(
        {
            "all_rows": 8_000,
            "evaluation_rows": target_rows,
            "evaluation_subset": (
                "deterministic proportional 3,000-row slice of the complete "
                "frozen 8,000-row test split"
            ),
            "panel_filename": panel_path.name,
            "panel_sha256": sha256_file(panel_path),
            "domain_counts": selected_distribution["scene_composition"],
            "source_kind_appearances": selected_distribution[
                "source_kind_appearances"
            ],
            "source_kind_scene_coverage": selected_distribution[
                "source_kind_scene_coverage"
            ],
            "source_count_counts": selected_distribution["source_count"],
            "single_source_counts": selected_distribution["single_source"],
            "length_bucket_counts": selected_distribution["length_bucket"],
            "stratified_distribution": selected_distribution,
        }
    )
    contract["metric_slices"] = {
        "complete_selected": target_rows,
        **selected_distribution,
        "source_kind_multi_label_note": (
            "Music/Sound/Speech presence slices overlap in multi-source scenes"
        ),
        "spatial": source_contract["metric_slices"]["spatial"],
    }
    contract["sampling"]["inference_batch_size"] = 1
    contract_path = output / "EVAL_CONTRACT.json"
    atomic_json(contract_path, contract)
    summary = {
        "status": "PASS",
        "output_root": str(output),
        "rows": len(selected),
        "panel_sha256": sha256_file(panel_path),
        "source_count_quota": expected_outer,
        "distribution": selected_distribution,
        "semantic_caption_compiler_version": contract["sampling"][
            "semantic_caption_compiler_version"
        ],
        "content_disjoint_from_train": contract["test_set"][
            "content_disjoint_from_train"
        ],
    }
    atomic_json(output / "BUILD_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
