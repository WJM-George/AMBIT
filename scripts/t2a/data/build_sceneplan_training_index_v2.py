#!/usr/bin/env python3
"""Build compact frozen SQLite split indexes for ScenePlan-v2 DiT loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
import zlib
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import (  # noqa: E402
    DATASET_ROOT,
    atomic_write_json,
    require_dataset_not_frozen,
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA page_size=65536;
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        PRAGMA locking_mode=EXCLUSIVE;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE latent_shards (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL
        );
        CREATE TABLE samples (
            ordinal INTEGER PRIMARY KEY,
            sample_id TEXT NOT NULL UNIQUE,
            model_num_samples INTEGER NOT NULL,
            latent_frames_valid INTEGER NOT NULL,
            renderer_caption_zlib BLOB NOT NULL,
            scene_plan_zlib BLOB NOT NULL,
            latent_shard_id INTEGER NOT NULL,
            latent_key TEXT NOT NULL,
            latent_tensor_sha256 TEXT NOT NULL,
            planned_record_sha256 TEXT NOT NULL,
            materialized_record_sha256 TEXT NOT NULL,
            FOREIGN KEY(latent_shard_id) REFERENCES latent_shards(id)
        );
        """
    )


def build_split(
    materialized_root: Path,
    output_root: Path,
    split: str,
    expected_rows: int,
    mode: str,
) -> dict[str, Any]:
    manifests = sorted(
        (materialized_root / "manifests" / split).glob(
            f"materialized-{split}-*.parquet"
        )
    )
    if not manifests:
        raise RuntimeError(f"no materialized manifests for split {split}")
    output = output_root / f"{split}.sqlite"
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(temporary)
    create_schema(connection)
    latent_ids: dict[str, tuple[int, str]] = {}
    ordinal = 0
    started = time.time()
    try:
        for manifest_index, manifest in enumerate(manifests, start=1):
            rows = pq.read_table(manifest).to_pylist()
            for row in rows:
                materialized_text = str(row["materialized_record_json"])
                record = json.loads(materialized_text)
                if canonical_json(record) != materialized_text:
                    raise RuntimeError(f"noncanonical materialized record: {row['sample_id']}")
                caption = record["renderer_caption"]
                scene = record["scene_plan"]
                latent_path, separator, latent_key = str(row["latent_ref"]).partition("#")
                if not separator or latent_key != row["sample_id"]:
                    raise RuntimeError(f"invalid latent reference: {row['sample_id']}")
                shard_sha = str(row["latent_shard_sha256"])
                existing = latent_ids.get(latent_path)
                if existing is None:
                    shard_id = len(latent_ids) + 1
                    latent_ids[latent_path] = (shard_id, shard_sha)
                    connection.execute(
                        "INSERT INTO latent_shards(id, path, sha256) VALUES (?, ?, ?)",
                        (shard_id, latent_path, shard_sha),
                    )
                else:
                    shard_id, previous_sha = existing
                    if previous_sha != shard_sha:
                        raise RuntimeError(f"latent shard checksum conflict: {latent_path}")
                connection.execute(
                    """
                    INSERT INTO samples(
                        ordinal, sample_id, model_num_samples,
                        latent_frames_valid, renderer_caption_zlib,
                        scene_plan_zlib, latent_shard_id, latent_key,
                        latent_tensor_sha256, planned_record_sha256,
                        materialized_record_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ordinal,
                        str(row["sample_id"]),
                        int(row["model_num_samples"]),
                        int(row["latent_frames_valid"]),
                        sqlite3.Binary(zlib.compress(canonical_json(caption).encode("utf-8"), 3)),
                        sqlite3.Binary(zlib.compress(canonical_json(scene).encode("utf-8"), 3)),
                        shard_id,
                        latent_key,
                        str(row["latent_tensor_sha256"]),
                        str(row["planned_record_sha256"]),
                        str(row["materialized_record_sha256"]),
                    ),
                )
                ordinal += 1
            if manifest_index % 16 == 0:
                connection.commit()
            if manifest_index % 50 == 0 or manifest_index == len(manifests):
                print(
                    json.dumps(
                        {
                            "split": split,
                            "indexed_manifests": manifest_index,
                            "total_manifests": len(manifests),
                            "rows": ordinal,
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
        if ordinal != int(expected_rows):
            raise RuntimeError(f"{split} training-index rows {ordinal} != {expected_rows}")
        metadata = {
            "schema": "stable_audio_tools.sceneplan_v2_training_index",
            "schema_version": "2",
            "contract_revision": "4",
            "conditioning_contract_revision": "2",
            "mode": mode,
            "split": split,
            "rows": str(ordinal),
            "latent_shards": str(len(latent_ids)),
            "latent_channels": "64",
            "max_latent_frames": "432",
            "caption_max_tokens": "512",
            "random_crop": "false",
            "frozen": "true",
            "materialized_root": str(materialized_root),
        }
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)", metadata.items()
        )
        connection.commit()
        connection.execute("ANALYZE")
        connection.commit()
        check = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise RuntimeError(f"SQLite integrity check failed for {split}: {check}")
        count = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
        unique = int(
            connection.execute("SELECT COUNT(DISTINCT sample_id) FROM samples").fetchone()[0]
        )
        if count != ordinal or unique != ordinal:
            raise RuntimeError(f"SQLite reopen counts failed before close for {split}")
    finally:
        connection.close()
    os.replace(temporary, output)
    reopened = sqlite3.connect(f"file:{output}?mode=ro&immutable=1", uri=True)
    try:
        if reopened.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"atomic SQLite reopen failed for {split}")
        if int(reopened.execute("SELECT COUNT(*) FROM samples").fetchone()[0]) != ordinal:
            raise RuntimeError(f"atomic SQLite row count changed for {split}")
    finally:
        reopened.close()
    return {
        "split": split,
        "rows": ordinal,
        "manifests": len(manifests),
        "latent_shards": len(latent_ids),
        "path": str(output),
        "num_bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "elapsed_sec": round(time.time() - started, 3),
    }


def main() -> int:
    require_dataset_not_frozen()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--materialized-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    materialized_root = (
        args.materialized_root
        or (
            DATASET_ROOT / "pilots/joint_4k/materialized"
            if args.mode == "pilot"
            else DATASET_ROOT / "materialized"
        )
    ).expanduser().resolve(strict=True)
    output_root = (
        args.output_root
        or (
            DATASET_ROOT / "pilots/joint_4k/training_index"
            if args.mode == "pilot"
            else DATASET_ROOT / "training_index"
        )
    ).expanduser().resolve(strict=False)
    try:
        output_root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
    except ValueError as error:
        raise ValueError(f"training indexes must be on SDB: {output_root}") from error
    expected = (
        {"train": 4_000}
        if args.mode == "pilot"
        else {"train": 1_100_000, "validation": 20_000, "test": 4_000}
    )
    summaries = [
        build_split(materialized_root, output_root, split, rows, args.mode)
        for split, rows in expected.items()
    ]
    summary = {
        "schema": "stable_audio_tools.sceneplan_v2_training_index_build",
        "schema_version": 2,
        "mode": args.mode,
        "rows": sum(item["rows"] for item in summaries),
        "splits": summaries,
        "random_crop": False,
        "latent_batch_padding_frames": 432,
        "conditioning_contract_revision": 2,
        "caption_max_tokens": 512,
    }
    atomic_write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
