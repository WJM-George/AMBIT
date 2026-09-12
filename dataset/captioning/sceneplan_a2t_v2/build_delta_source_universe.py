#!/usr/bin/env python3
"""Freeze a QC-passed non-speech delta into the production A2T contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from build_source_universe import (
    INPUT_SCHEMA,
    INPUT_SCHEMA_PATH,
    INPUT_SCHEMA_VERSION,
    REGISTRY_SCHEMA_PATH,
    UNIVERSE_SCHEMA,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    if args.expected_rows <= 0:
        raise ValueError("--expected-rows must be positive")
    try:
        output_root.relative_to(Path("/mnt/sdb"))
    except ValueError as error:
        raise ValueError("delta A2T artifacts must persist on /mnt/sdb") from error
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(source_path)
    if len(source_rows) != args.expected_rows:
        raise RuntimeError(
            f"expected {args.expected_rows} frozen rows, got {len(source_rows)}"
        )
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    seen_assets: set[str] = set()
    for source in source_rows:
        digest = str(source.get("source_audio_sha256") or "")
        asset_id = str(source.get("asset_id") or "")
        audio_path = Path(str(source.get("audio_path") or "")).resolve(strict=True)
        if len(digest) != 64 or not asset_id:
            raise RuntimeError("delta source lacks a frozen hash or asset id")
        if digest in seen_hashes or asset_id in seen_assets:
            raise RuntimeError("delta source hash/asset uniqueness changed")
        if source.get("qc_status") != "PASS" or source.get("modality") != "sound":
            raise RuntimeError(f"non-passing Sound entered delta universe: {asset_id}")
        if source.get("official_split") != "train":
            raise RuntimeError(f"non-train VGGSound source entered delta: {asset_id}")
        if source.get("lineage_gate_status") != "pass":
            raise RuntimeError(f"lineage-failing source entered delta: {asset_id}")
        if source.get("exact_hash_gate_status") != "pass":
            raise RuntimeError(f"hash-failing source entered delta: {asset_id}")
        seen_hashes.add(digest)
        seen_assets.add(asset_id)
        rows.append(
            {
                "annotation_id": f"sha256:{digest}",
                "source_audio_sha256": digest,
                "primary_asset_id": asset_id,
                "alias_asset_ids": [asset_id],
                "split": args.split,
                "kind": "sound",
                "source_dataset": str(source["source_dataset"]),
                "source_id": str(source["candidate_id"]),
                "raw_label": str(source["label"]),
                "dry_audio_path": str(audio_path),
                "identity_hash": digest,
                "native_sample_rate_hz": int(source["native_sample_rate_hz"]),
                "native_num_samples": int(source["native_num_samples"]),
                "native_channels": int(source["native_channels"]),
                "model_sample_rate_hz": 44_100,
                "model_num_samples": int(source["model_num_samples"]),
                "duration_sec": float(source["duration_sec"]),
                "selection_rank": str(source["selection_rank"]),
                "lineage_policy": (
                    "vggsound_official_train_interval_and_hash_disjoint_v1"
                ),
                "sceneplan_reference_count": 1,
            }
        )
    rows.sort(key=lambda row: (row["selection_rank"], row["source_audio_sha256"]))

    universe_path = output_root / "source_universe.parquet"
    temporary = universe_path.with_name(f".{universe_path.name}.{os.getpid()}.tmp")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=UNIVERSE_SCHEMA),
        temporary,
        compression="zstd",
    )
    os.replace(temporary, universe_path)
    input_path = output_root / "instruct_input.jsonl"
    atomic_text(
        input_path,
        "".join(
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
                    "raw_label": row["raw_label"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for row in rows
        ),
    )
    label_counts = Counter(row["raw_label"] for row in rows)
    summary = {
        "schema": "stable_audio_tools.sceneplan_delta_source_universe",
        "schema_version": 1,
        "status": "PASS",
        "source_jsonl": str(source_path),
        "source_jsonl_sha256": file_sha256(source_path),
        "rows": len(rows),
        "unique_source_audio_sha256": len(seen_hashes),
        "unique_asset_ids": len(seen_assets),
        "kind_counts": {"sound": len(rows)},
        "source_dataset_counts": dict(
            sorted(Counter(row["source_dataset"] for row in rows).items())
        ),
        "unique_raw_labels": len(label_counts),
        "most_common_raw_labels": label_counts.most_common(20),
        "universe_parquet": str(universe_path),
        "universe_parquet_sha256": file_sha256(universe_path),
        "instruct_input_jsonl": str(input_path),
        "instruct_input_sha256": file_sha256(input_path),
        "input_schema_sha256": file_sha256(INPUT_SCHEMA_PATH.resolve(strict=True)),
        "registry_schema_sha256": file_sha256(
            REGISTRY_SCHEMA_PATH.resolve(strict=True)
        ),
        "contracts": {
            "official_train_only": True,
            "sound_only": True,
            "mono": all(row["native_channels"] == 1 for row in rows),
            "base_and_eval_lineage_disjoint": True,
            "exact_hash_unique": True,
            "annotations_started": False,
        },
    }
    summary_path = output_root / "SUMMARY.json"
    atomic_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    atomic_text(output_root / "READY", file_sha256(summary_path) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
