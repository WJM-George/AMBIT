#!/usr/bin/env python3
"""Freeze the largest leak-safe Sound source universe for ScenePlan revision v1.

Inputs are (1) newly extracted VGGSound rows, (2) FSDKaggle2019 noisy-train
rows, and (3) eligible but previously unreferenced rows from the base signal
catalog.  The merger is fail-closed on exact hashes and recording lineage, caps
VGGSound at one segment per parent video, and assigns candidates only to train
and validation.  The already-complete base test split remains byte-for-byte
unchanged by this revision.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.build_p10_public_core_candidates import (
    Interval,
    add_interval,
    load_audiocaps_metadata,
    overlapping_assets,
)
from dataset.captioning.sceneplan_a2t_v2.build_source_universe import (
    INPUT_SCHEMA,
    INPUT_SCHEMA_VERSION,
    UNIVERSE_SCHEMA,
)


SPLIT_CAPACITY = {"train": 105_000, "validation": 1_750, "test": 0}
BASE_SPLIT_ROWS = {"train": 1_100_000, "validation": 20_000, "test": 4_000}


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def all_base_assets(index_path: Path) -> tuple[set[str], dict[str, set[str]]]:
    assets: set[str] = set()
    by_split: dict[str, set[str]] = defaultdict(set)
    parquet = pq.ParquetFile(index_path)
    for batch in parquet.iter_batches(columns=["split", "source_asset_ids"], batch_size=32_768):
        for split, values in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            for asset in values or ():
                assets.add(str(asset))
                by_split[str(split)].add(str(asset))
    return assets, by_split


def parse_any_youtube_asset_id(asset_id: str) -> str | None:
    match = re.fullmatch(
        r"(?:sound|music):audioset:audioset_(.{11})|"
        r"music:musiccaps:musiccaps_(.{11})",
        asset_id,
    )
    if not match:
        return None
    return match.group(1) or match.group(2)


def parse_any_vgg_asset_id(asset_id: str) -> tuple[str, int] | None:
    match = re.fullmatch(
        r"(?:sound|music):vggsound:vggsound_(.{11})_(\d{6})", asset_id
    )
    return (match.group(1), int(match.group(2))) if match else None


def parse_any_audiocaps_asset_id(asset_id: str) -> int | None:
    match = re.fullmatch(
        r"(?:sound|music):audiocaps:audiocaps_(\d+)", asset_id
    )
    return int(match.group(1)) if match else None


def audiocaps_lookup(root: Path) -> dict[int, tuple[str, float]]:
    files = sorted(root.glob("train-*.parquet"))
    result: dict[int, tuple[str, float]] = {}
    for row in load_audiocaps_metadata(files, with_locator=False):
        result[int(row["audiocap_id"])] = (str(row["youtube_id"]), float(row["start_time"]))
    return result


def build_base_lineage(
    assets: set[str], ac_lookup: dict[int, tuple[str, float]]
) -> tuple[set[str], dict[str, list[Interval]], dict[str, list[Interval]]]:
    whole: set[str] = set()
    vgg: dict[str, list[Interval]] = defaultdict(list)
    audiocaps: dict[str, list[Interval]] = defaultdict(list)
    for asset in assets:
        if youtube := parse_any_youtube_asset_id(asset):
            whole.add(youtube)
        elif parsed := parse_any_vgg_asset_id(asset):
            youtube, start = parsed
            add_interval(vgg, youtube, start, asset)
        elif (audiocap_id := parse_any_audiocaps_asset_id(asset)) is not None:
            locator = ac_lookup.get(audiocap_id)
            if locator is None:
                raise RuntimeError(f"missing AudioCaps lineage for base asset {asset}")
            youtube, start = locator
            add_interval(audiocaps, youtube, start, asset)
    return whole, vgg, audiocaps


def external_youtube_ids(root: Path) -> set[str]:
    values: set[str] = set()
    for name in ("candidate_pool.jsonl", "candidate_exclusions.jsonl"):
        path = root / name
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            if row.get("youtube_id"):
                values.add(str(row["youtube_id"]))
    return values


def external_hashes(root: Path) -> set[str]:
    values: set[str] = set()
    for name in ("candidate_pool.jsonl", "candidate_exclusions.jsonl"):
        path = root / name
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            if row.get("audio_sha256"):
                values.add(str(row["audio_sha256"]))
    return values


def candidate_parent(
    row: dict[str, Any], ac_lookup: dict[int, tuple[str, float]]
) -> tuple[str, str | None, float | None, float | None]:
    dataset = str(row["source_dataset"])
    if dataset == "vggsound":
        parsed = parse_any_vgg_asset_id(str(row["asset_id"]))
        youtube = str(row.get("youtube_id") or (parsed[0] if parsed else ""))
        start = float(
            row.get("start_time_sec")
            if row.get("start_time_sec") is not None
            else (parsed[1] if parsed else 0.0)
        )
        if not youtube:
            raise RuntimeError(f"could not parse VGGSound parent: {row['asset_id']}")
        return f"youtube:{youtube}", youtube, start, start + float(row.get("duration_sec") or 10.0)
    if dataset in {"audioset", "musiccaps"}:
        match = re.search(r"(?:audioset_|musiccaps_)(.{11})$", str(row["asset_id"]))
        if not match:
            raise RuntimeError(f"could not parse YouTube parent: {row['asset_id']}")
        youtube = match.group(1)
        return f"youtube:{youtube}", youtube, None, None
    if dataset == "audiocaps":
        parsed = parse_any_audiocaps_asset_id(str(row["asset_id"]))
        if parsed is None or parsed not in ac_lookup:
            raise RuntimeError(f"could not parse AudioCaps parent: {row['asset_id']}")
        youtube, start = ac_lookup[parsed]
        return f"youtube:{youtube}", youtube, start, start + 10.0
    if row.get("parent_asset_id"):
        return f"{row.get('parent_dataset') or dataset}:{row['parent_asset_id']}", None, None, None
    return f"{dataset}:{row['asset_id']}", None, None, None


def lineage_conflicts(
    row: dict[str, Any],
    ac_lookup: dict[int, tuple[str, float]],
    base_whole: set[str],
    base_vgg: dict[str, list[Interval]],
    base_ac: dict[str, list[Interval]],
    external_youtube: set[str],
) -> list[str]:
    _, youtube, start, end = candidate_parent(row, ac_lookup)
    if youtube is None:
        return []
    reasons: list[str] = []
    if youtube in external_youtube:
        reasons.append("youtube_parent_reserved_by_external_benchmark")
    if youtube in base_whole:
        reasons.append("youtube_parent_used_by_base_whole_clip")
    if start is None:
        if youtube in base_vgg or youtube in base_ac:
            reasons.append("youtube_parent_has_base_interval")
    else:
        if overlapping_assets(base_vgg, youtube, start, float(end)):
            reasons.append("overlaps_base_vggsound_interval")
        if overlapping_assets(base_ac, youtube, start, float(end)):
            reasons.append("overlaps_base_audiocaps_interval")
    return sorted(set(reasons))


def normalized_external(row: dict[str, Any]) -> dict[str, Any]:
    required = ("asset_id", "source_dataset", "source_audio_sha256", "audio_path", "model_num_samples")
    if any(not row.get(key) for key in required):
        raise RuntimeError(f"new Sound candidate lacks required data: {row.get('candidate_id')}")
    return {
        **row,
        "dry_audio_path": str(Path(str(row["audio_path"])).resolve(strict=True)),
        "raw_label": str(row.get("label") or "sound event"),
        "identity_hash": str(row["source_audio_sha256"]),
        "lineage_policy": "official_train_parent_and_exact_hash_disjoint_v1",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vgg-jsonl", type=Path, required=True)
    parser.add_argument("--fsdkaggle-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--base-index", type=Path,
        default=Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/sceneplans_model_v1/index.parquet"),
    )
    parser.add_argument(
        "--base-catalog", type=Path,
        default=Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/source_catalog/nonspeech/nonspeech_signal_catalog.parquet"),
    )
    parser.add_argument(
        "--audiocaps-root", type=Path,
        default=Path("/mnt/sdd/audio_dataset/datasets/audiocaps/snapshot/data"),
    )
    parser.add_argument(
        "--external-manifest-root", type=Path,
        default=Path("/mnt/sdb/audio_dataset/evaluation_benchmark/p10_evaluation_benchmark_v1/manifests"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.relative_to(Path("/mnt/sdb"))
    except ValueError as error:
        raise ValueError("Sound expansion universe must live on /mnt/sdb") from error
    output_root.mkdir(parents=True, exist_ok=True)
    base_index = args.base_index.expanduser().resolve(strict=True)
    base_catalog = args.base_catalog.expanduser().resolve(strict=True)
    ext_root = args.external_manifest_root.expanduser().resolve(strict=True)
    base_assets, _ = all_base_assets(base_index)
    ac_lookup = audiocaps_lookup(args.audiocaps_root.expanduser().resolve(strict=True))
    base_whole, base_vgg, base_ac = build_base_lineage(base_assets, ac_lookup)
    reserved_youtube = external_youtube_ids(ext_root)
    reserved_hashes = external_hashes(ext_root)

    all_catalog_rows = pq.read_table(
        base_catalog, filters=[("eligible", "=", True)]
    ).to_pylist()
    catalog_rows = [row for row in all_catalog_rows if str(row["kind"]) == "sound"]
    asset_to_hash = {
        str(row["asset_id"]): str(row["source_audio_sha256"])
        for row in all_catalog_rows
    }
    used_hashes = {asset_to_hash[asset] for asset in base_assets if asset in asset_to_hash}
    raw: list[dict[str, Any]] = []
    seen_catalog_hashes: set[str] = set()
    for row in sorted(catalog_rows, key=lambda item: (str(item["selection_rank"]), str(item["asset_id"]))):
        digest = str(row["source_audio_sha256"])
        if digest in used_hashes or digest in seen_catalog_hashes:
            continue
        seen_catalog_hashes.add(digest)
        raw.append(
            normalized_external(
                {
                    "schema": "stable_audio_tools.sound_expansion_candidate",
                    "schema_version": 1,
                    "candidate_id": f"base_catalog_unused:{row['asset_id']}",
                    "asset_id": str(row["asset_id"]),
                    "source_dataset": str(row["source_dataset"]),
                    "official_split": "train",
                    "label": str(row["description"]),
                    "modality": "sound",
                    "selection_rank": hashlib.sha256(
                        f"sceneplan-sound-expansion-v1\0catalog\0{digest}".encode()
                    ).hexdigest(),
                    "lineage_gate_status": "pending",
                    "exact_hash_gate_status": "pass",
                    "qc_status": "PASS",
                    "audio_path": str(row["dry_audio_path"]),
                    "source_audio_sha256": digest,
                    "native_sample_rate_hz": int(row["native_sample_rate_hz"]),
                    "native_num_samples": int(row["native_num_samples"]),
                    "native_channels": int(row["native_channels"]),
                    "model_num_samples": int(row["model_num_samples"]),
                    "duration_sec": float(row["duration_sec"]),
                }
            )
        )
    raw.extend(normalized_external(row) for row in read_jsonl(args.vgg_jsonl.expanduser().resolve(strict=True)))
    raw.extend(normalized_external(row) for row in read_jsonl(args.fsdkaggle_jsonl.expanduser().resolve(strict=True)))

    exclusions: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for row in raw:
        reasons = []
        if row.get("qc_status") != "PASS" or row.get("exact_hash_gate_status") != "pass":
            reasons.append("upstream_qc_or_hash_gate_failed")
        if str(row.get("modality")) != "sound" or str(row.get("official_split")) != "train":
            reasons.append("not_official_train_sound")
        if str(row["source_audio_sha256"]) in used_hashes:
            reasons.append("exact_hash_already_used_by_base")
        if str(row["source_audio_sha256"]) in reserved_hashes:
            reasons.append("exact_hash_reserved_by_external_benchmark")
        reasons.extend(
            lineage_conflicts(
                row, ac_lookup, base_whole, base_vgg, base_ac, reserved_youtube
            )
        )
        row["lineage_exclusion_reasons"] = sorted(set(reasons))
        row["lineage_gate_status"] = "fail" if reasons else "pass"
        (exclusions if reasons else eligible).append(row)

    # Prefer truly independent recording parents over multiple ten-second
    # excerpts from the same video, then remove exact waveform duplicates.
    eligible.sort(key=lambda row: (str(row["selection_rank"]), str(row["candidate_id"])))
    frozen: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_parents: set[str] = set()
    for row in eligible:
        digest = str(row["source_audio_sha256"])
        parent_key, _, _, _ = candidate_parent(row, ac_lookup)
        reasons = []
        if digest in seen_hashes:
            reasons.append("duplicate_exact_hash_in_expansion")
        if parent_key in seen_parents:
            reasons.append("duplicate_recording_parent_in_expansion")
        if reasons:
            row["lineage_gate_status"] = "fail"
            row["lineage_exclusion_reasons"] = reasons
            exclusions.append(row)
            continue
        seen_hashes.add(digest)
        seen_parents.add(parent_key)
        row["parent_lineage_key"] = parent_key
        frozen.append(row)

    total_capacity = sum(SPLIT_CAPACITY.values())
    if len(frozen) > total_capacity:
        raise RuntimeError(
            f"{len(frozen)} candidates exceed one-source replacement capacity {total_capacity}; "
            "extend the replacement planner before discarding valid sources"
        )
    split_ranked = sorted(
        frozen,
        key=lambda row: hashlib.sha256(
            f"sceneplan-sound-expansion-v1\0split\0{row['source_audio_sha256']}".encode()
        ).hexdigest(),
    )
    # Preserve the original annotation ordinal/shard contract while moving the
    # revision's formerly internal-test candidates into train/validation.  The
    # legacy labels are used only as a stable annotation ordering key; they are
    # never emitted as the final split.
    legacy_test_rows = round(len(frozen) * BASE_SPLIT_ROWS["test"] / sum(BASE_SPLIT_ROWS.values()))
    legacy_validation_rows = round(
        len(frozen) * BASE_SPLIT_ROWS["validation"] / sum(BASE_SPLIT_ROWS.values())
    )
    legacy_counts = {
        "test": legacy_test_rows,
        "validation": legacy_validation_rows,
        "train": len(frozen) - legacy_test_rows - legacy_validation_rows,
    }
    cursor = 0
    for split in ("test", "validation", "train"):
        for row in split_ranked[cursor : cursor + legacy_counts[split]]:
            row["_annotation_order_split"] = split
        cursor += legacy_counts[split]

    train_validation_total = BASE_SPLIT_ROWS["train"] + BASE_SPLIT_ROWS["validation"]
    validation_rows = round(
        len(frozen) * BASE_SPLIT_ROWS["validation"] / train_validation_total
    )
    extra_validation_rows = validation_rows - legacy_validation_rows
    if not 0 <= extra_validation_rows <= legacy_test_rows:
        raise RuntimeError(
            "train/validation repartition cannot be satisfied by the legacy test rows"
        )
    legacy_test = [
        row for row in frozen if row["_annotation_order_split"] == "test"
    ]
    legacy_test.sort(
        key=lambda row: hashlib.sha256(
            (
                "sceneplan-sound-expansion-v1\0test-to-validation\0"
                f"{row['source_audio_sha256']}"
            ).encode()
        ).hexdigest()
    )
    promoted_to_validation = {
        str(row["source_audio_sha256"])
        for row in legacy_test[:extra_validation_rows]
    }
    for row in frozen:
        legacy_split = str(row["_annotation_order_split"])
        if legacy_split == "test":
            row["split"] = (
                "validation"
                if str(row["source_audio_sha256"]) in promoted_to_validation
                else "train"
            )
        else:
            row["split"] = legacy_split
    split_counts = {
        split: sum(str(row["split"]) == split for row in frozen)
        for split in ("train", "validation", "test")
    }
    if any(split_counts[key] > SPLIT_CAPACITY[key] for key in split_counts):
        raise RuntimeError(f"split assignment exceeds one-source capacity: {split_counts}")
    if split_counts != {
        "train": len(frozen) - validation_rows,
        "validation": validation_rows,
        "test": 0,
    }:
        raise RuntimeError(f"unexpected train/validation split counts: {split_counts}")
    frozen.sort(
        key=lambda row: (
            str(row["_annotation_order_split"]),
            str(row["selection_rank"]),
            str(row["candidate_id"]),
        )
    )
    for row in frozen:
        del row["_annotation_order_split"]

    universe_rows = []
    for row in frozen:
        digest = str(row["source_audio_sha256"])
        universe_rows.append(
            {
                "annotation_id": f"sha256:{digest}",
                "source_audio_sha256": digest,
                "primary_asset_id": str(row["asset_id"]),
                "alias_asset_ids": [str(row["asset_id"])],
                "split": str(row["split"]),
                "kind": "sound",
                "source_dataset": str(row["source_dataset"]),
                "source_id": str(row["candidate_id"]),
                "raw_label": str(row["raw_label"]),
                "dry_audio_path": str(row["dry_audio_path"]),
                "identity_hash": digest,
                "native_sample_rate_hz": int(row["native_sample_rate_hz"]),
                "native_num_samples": int(row["native_num_samples"]),
                "native_channels": int(row["native_channels"]),
                "model_sample_rate_hz": 44_100,
                "model_num_samples": int(row["model_num_samples"]),
                "duration_sec": float(row["duration_sec"]),
                "selection_rank": str(row["selection_rank"]),
                "lineage_policy": str(row["lineage_policy"]),
                "sceneplan_reference_count": 1,
            }
        )
    universe_path = output_root / "source_universe.parquet"
    temporary = universe_path.with_name(f".{universe_path.name}.{os.getpid()}.tmp")
    pq.write_table(pa.Table.from_pylist(universe_rows, schema=UNIVERSE_SCHEMA), temporary, compression="zstd")
    os.replace(temporary, universe_path)
    input_path = output_root / "instruct_input.jsonl"
    atomic_jsonl(
        input_path,
        [
            {
                "schema": INPUT_SCHEMA,
                "schema_version": INPUT_SCHEMA_VERSION,
                "id": row["annotation_id"],
                "audio_path": row["dry_audio_path"],
                "kind": "sound",
                "split": row["split"],
                "source_audio_sha256": row["source_audio_sha256"],
                "asset_id": row["primary_asset_id"],
                "source_dataset": row["source_dataset"],
                "raw_label": row["raw_label"],
            }
            for row in universe_rows
        ],
    )
    atomic_jsonl(output_root / "candidates_frozen.jsonl", frozen)
    atomic_jsonl(output_root / "candidate_exclusions.jsonl", exclusions)
    summary = {
        "schema": "stable_audio_tools.sound_expansion_source_universe",
        "schema_version": 1,
        "status": "PASS" if frozen else "FAIL",
        "counts": {
            "raw_candidates": len(raw),
            "frozen_unique_waveforms_and_parents": len(frozen),
            "excluded": len(exclusions),
            "split": dict(sorted(split_counts.items())),
            "source_dataset": dict(sorted(Counter(row["source_dataset"] for row in frozen).items())),
        },
        "replacement_capacity": SPLIT_CAPACITY,
        "split_policy": {
            "policy": "base_test_unchanged_train_validation_only_v1",
            "base_train_validation_ratio": [
                BASE_SPLIT_ROWS["train"],
                BASE_SPLIT_ROWS["validation"],
            ],
            "legacy_internal_test_rows_reassigned": legacy_test_rows,
            "legacy_internal_test_to_train": legacy_test_rows - extra_validation_rows,
            "legacy_internal_test_to_validation": extra_validation_rows,
            "annotation_ordinal_order_preserved": True,
        },
        "contracts": {
            "base_exact_hash_disjoint": True,
            "external_exact_hash_disjoint": True,
            "base_parent_lineage_disjoint": True,
            "external_parent_lineage_disjoint": True,
            "one_recording_parent_per_new_source": True,
            "cross_split_hash_and_parent_disjoint": True,
            "one_source_replacement_capacity_sufficient": True,
            "new_sources_limited_to_train_validation": True,
            "base_test_unchanged_by_revision": True,
        },
        "artifacts": {
            "source_universe": str(universe_path),
            "source_universe_sha256": sha256_file(universe_path),
            "instruct_input": str(input_path),
            "instruct_input_sha256": sha256_file(input_path),
            "candidates_frozen": str(output_root / "candidates_frozen.jsonl"),
            "candidate_exclusions": str(output_root / "candidate_exclusions.jsonl"),
        },
    }
    atomic_json(output_root / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
