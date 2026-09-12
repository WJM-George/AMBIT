#!/usr/bin/env python3
"""Build the unique non-speech annotation universe referenced by P7.

One source hash becomes one annotation task even when the same mono donor is
used by multiple ScenePlans.  Split ownership is inherited from the P7 rows
and cross-split hash reuse is a hard error.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
DEFAULT_SCENEPLAN_ROOT = DATASET_ROOT / "sceneplans"
DEFAULT_CATALOG = DATASET_ROOT / "source_catalog/nonspeech/nonspeech_signal_catalog.parquet"
DEFAULT_OUTPUT = DATASET_ROOT / "source_annotations/nonspeech_instruct_v2"
SCHEMA_ROOT = Path(__file__).with_name("schemas")
INPUT_SCHEMA_PATH = SCHEMA_ROOT / "source_description_instruct_input_v1.schema.json"
REGISTRY_SCHEMA_PATH = SCHEMA_ROOT / "source_description_registry_v1.schema.json"
INPUT_SCHEMA = "stable_audio_tools.sceneplan_source_description_instruct_input"
INPUT_SCHEMA_VERSION = 1
KINDS = ("music", "sound")
SPLITS = ("train", "validation", "test")
SPLIT_ORDER = {split: ordinal for ordinal, split in enumerate(SPLITS)}
UNIVERSE_SCHEMA = pa.schema(
    [
        ("annotation_id", pa.string()),
        ("source_audio_sha256", pa.string()),
        ("primary_asset_id", pa.string()),
        ("alias_asset_ids", pa.list_(pa.string())),
        ("split", pa.string()),
        ("kind", pa.string()),
        ("source_dataset", pa.string()),
        ("source_id", pa.string()),
        ("raw_label", pa.string()),
        ("dry_audio_path", pa.string()),
        ("identity_hash", pa.string()),
        ("native_sample_rate_hz", pa.int32()),
        ("native_num_samples", pa.int64()),
        ("native_channels", pa.int16()),
        ("model_sample_rate_hz", pa.int32()),
        ("model_num_samples", pa.int64()),
        ("duration_sec", pa.float64()),
        ("selection_rank", pa.string()),
        ("lineage_policy", pa.string()),
        ("sceneplan_reference_count", pa.int32()),
    ]
)


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        raise ValueError("cannot compute percentile of an empty list")
    index = min(len(values) - 1, max(0, math.ceil(fraction * len(values)) - 1))
    return values[index]


def scan_references(sceneplan_root: Path) -> tuple[Counter[str], dict[str, tuple[str, str]], int, int]:
    files = sorted(sceneplan_root.glob("*/sceneplans-*.parquet"))
    if not files:
        raise ValueError(f"no ScenePlan shards under {sceneplan_root}")
    reference_counts: Counter[str] = Counter()
    ownership: dict[str, tuple[str, str]] = {}
    scene_rows = 0
    for batch in ds.dataset([str(path) for path in files], format="parquet").to_batches(
        columns=["split", "source_asset_ids", "source_kinds"],
        batch_size=65536,
    ):
        scene_rows += batch.num_rows
        splits = batch.column(0).to_pylist()
        asset_lists = batch.column(1).to_pylist()
        kind_lists = batch.column(2).to_pylist()
        for split, assets, kinds in zip(splits, asset_lists, kind_lists, strict=True):
            if split not in SPLIT_ORDER:
                raise ValueError(f"unexpected split: {split}")
            if len(assets) != len(kinds):
                raise ValueError("source_asset_ids/source_kinds length mismatch")
            for asset_id, kind in zip(assets, kinds, strict=True):
                if kind not in KINDS:
                    continue
                previous = ownership.setdefault(asset_id, (kind, split))
                if previous != (kind, split):
                    raise ValueError(
                        f"asset crosses kind or split: {asset_id}: {previous} != {(kind, split)}"
                    )
                reference_counts[asset_id] += 1
    return reference_counts, ownership, scene_rows, len(files)


def load_catalog(catalog_path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    columns = [field.name for field in UNIVERSE_SCHEMA if field.name not in {
        "annotation_id",
        "primary_asset_id",
        "alias_asset_ids",
        "split",
        "raw_label",
        "sceneplan_reference_count",
    }]
    columns.extend(["asset_id", "description", "eligible", "decoded_finite"])
    rows = pq.read_table(catalog_path, columns=columns).to_pylist()
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        asset_id = str(row["asset_id"])
        if asset_id not in wanted:
            continue
        if not row["eligible"] or not row["decoded_finite"]:
            raise ValueError(f"P7 references ineligible/non-finite source: {asset_id}")
        if int(row["native_channels"]) != 1:
            raise ValueError(f"P7 non-speech source is not mono: {asset_id}")
        if asset_id in selected:
            raise ValueError(f"duplicate catalog asset_id: {asset_id}")
        selected[asset_id] = row
    missing = sorted(wanted - selected.keys())
    if missing:
        raise ValueError(f"{len(missing)} referenced assets missing from eligible catalog; first={missing[0]}")
    return selected


def build_groups(
    catalog: dict[str, dict[str, Any]],
    references: Counter[str],
    ownership: dict[str, tuple[str, str]],
) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for asset_id, reference_count in references.items():
        row = catalog[asset_id]
        kind, split = ownership[asset_id]
        if row["kind"] != kind:
            raise ValueError(f"catalog kind mismatch for {asset_id}")
        source_hash = str(row["source_audio_sha256"])
        current = groups.get(source_hash)
        if current is None:
            current = {
                "annotation_id": f"sha256:{source_hash}",
                "source_audio_sha256": source_hash,
                "primary_asset_id": asset_id,
                "alias_asset_ids": [asset_id],
                "split": split,
                "kind": kind,
                "source_dataset": str(row["source_dataset"]),
                "source_id": str(row["source_id"]),
                "raw_label": str(row["description"]),
                "dry_audio_path": str(Path(row["dry_audio_path"]).resolve(strict=False)),
                "identity_hash": str(row["identity_hash"]),
                "native_sample_rate_hz": int(row["native_sample_rate_hz"]),
                "native_num_samples": int(row["native_num_samples"]),
                "native_channels": int(row["native_channels"]),
                "model_sample_rate_hz": int(row["model_sample_rate_hz"]),
                "model_num_samples": int(row["model_num_samples"]),
                "duration_sec": float(row["duration_sec"]),
                "selection_rank": str(row["selection_rank"]),
                "lineage_policy": str(row["lineage_policy"]),
                "sceneplan_reference_count": int(reference_count),
            }
            groups[source_hash] = current
            continue
        if current["kind"] != kind or current["split"] != split:
            raise ValueError(
                f"audio hash crosses kind/split: {source_hash}: "
                f"{(current['kind'], current['split'])} != {(kind, split)}"
            )
        current["alias_asset_ids"].append(asset_id)
        current["sceneplan_reference_count"] += int(reference_count)
        if str(row["selection_rank"]) < current["selection_rank"]:
            current.update(
                primary_asset_id=asset_id,
                source_dataset=str(row["source_dataset"]),
                source_id=str(row["source_id"]),
                raw_label=str(row["description"]),
                dry_audio_path=str(Path(row["dry_audio_path"]).resolve(strict=False)),
                identity_hash=str(row["identity_hash"]),
                native_sample_rate_hz=int(row["native_sample_rate_hz"]),
                native_num_samples=int(row["native_num_samples"]),
                native_channels=int(row["native_channels"]),
                model_sample_rate_hz=int(row["model_sample_rate_hz"]),
                model_num_samples=int(row["model_num_samples"]),
                duration_sec=float(row["duration_sec"]),
                selection_rank=str(row["selection_rank"]),
                lineage_policy=str(row["lineage_policy"]),
            )
    result = list(groups.values())
    for row in result:
        row["alias_asset_ids"].sort()
    result.sort(
        key=lambda row: (
            SPLIT_ORDER[row["split"]],
            row["kind"],
            row["selection_rank"],
            row["source_audio_sha256"],
        )
    )
    return result


def write_universe(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(temporary, UNIVERSE_SCHEMA, compression="zstd")
        for start in range(0, len(rows), 50000):
            writer.write_table(pa.Table.from_pylist(rows[start : start + 50000], schema=UNIVERSE_SCHEMA))
        writer.close()
        writer = None
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sceneplan-root", type=Path, default=DEFAULT_SCENEPLAN_ROOT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sceneplan_root = args.sceneplan_root.expanduser().resolve(strict=True)
    catalog_path = args.catalog.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    try:
        output_root.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"source annotation universe must persist on SDB: {output_root}") from error
    output_root.mkdir(parents=True, exist_ok=True)
    input_schema_path = INPUT_SCHEMA_PATH.resolve(strict=True)
    registry_schema_path = REGISTRY_SCHEMA_PATH.resolve(strict=True)
    universe_path = output_root / "source_universe.parquet"
    input_path = output_root / "instruct_input.jsonl"
    summary_path = output_root / "summary.json"
    ready_path = output_root / "READY"
    existing = [path for path in (universe_path, input_path, summary_path, ready_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"outputs already exist; pass --overwrite: {existing[0]}")

    references, ownership, scene_rows, sceneplan_files = scan_references(sceneplan_root)
    catalog = load_catalog(catalog_path, set(references))
    rows = build_groups(catalog, references, ownership)
    if len(rows) != len({row["source_audio_sha256"] for row in rows}):
        raise RuntimeError("source universe hash uniqueness changed")

    write_universe(universe_path, rows)
    input_lines = "".join(
        json.dumps(
            {
                "schema": INPUT_SCHEMA,
                "schema_version": INPUT_SCHEMA_VERSION,
                "id": row["annotation_id"],
                "audio_path": row["dry_audio_path"],
                "kind": row["kind"],
                "split": row["split"],
                "source_audio_sha256": row["source_audio_sha256"],
                "asset_id": row["primary_asset_id"],
                "source_dataset": row["source_dataset"],
                # Kept for lineage/audit only. The annotator never includes
                # this legacy label in the Instruct request.
                "raw_label": row["raw_label"],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    )
    atomic_text(input_path, input_lines)

    counts_by_kind = Counter(row["kind"] for row in rows)
    counts_by_split_kind = Counter((row["split"], row["kind"]) for row in rows)
    reuse = {}
    for kind in KINDS:
        values = sorted(row["sceneplan_reference_count"] for row in rows if row["kind"] == kind)
        reuse[kind] = {
            "mean": sum(values) / len(values),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p99": percentile(values, 0.99),
            "max": values[-1],
        }
    summary = {
        "schema": "stable_audio_tools.sceneplan_nonspeech_source_universe",
        "schema_version": 2,
        "sceneplan_root": str(sceneplan_root),
        "sceneplan_files": sceneplan_files,
        "scene_rows": scene_rows,
        "speech_a2t": False,
        "source_hash_rows": len(rows),
        "asset_ids": len(references),
        "sceneplan_nonspeech_references": sum(references.values()),
        "by_kind": dict(sorted(counts_by_kind.items())),
        "by_split_kind": {
            f"{split}|{kind}": counts_by_split_kind[(split, kind)]
            for split in SPLITS
            for kind in KINDS
        },
        "reuse": reuse,
        "universe_parquet": str(universe_path),
        "universe_parquet_sha256": file_sha256(universe_path),
        "instruct_input_jsonl": str(input_path),
        "instruct_input_sha256": file_sha256(input_path),
        "instruct_input_schema": str(input_schema_path),
        "instruct_input_schema_sha256": file_sha256(input_schema_path),
        "registry_schema": str(registry_schema_path),
        "registry_schema_sha256": file_sha256(registry_schema_path),
        "full_annotation_started": False,
    }
    atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    atomic_text(ready_path, file_sha256(summary_path) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
