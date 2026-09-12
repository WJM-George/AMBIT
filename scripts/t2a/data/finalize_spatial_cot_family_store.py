#!/usr/bin/env python3
"""Validate all Spatial-CoT work shards and publish one immutable family store."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.t2a_artifacts import atomic_write_json, atomic_write_jsonl  # noqa: E402


SCHEMA = "stable_audio_tools.spatial_family_latent_store"
VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--expected-families", type=int, default=None)
    parser.add_argument("--verify-shard-hashes", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.build_spec.resolve().read_text(encoding="utf-8"))
    expected = int(
        args.expected_families
        if args.expected_families is not None
        else spec["splits"][args.split]["families"]
    )
    if expected <= 0:
        raise SystemExit("expected family count must be positive")
    per_shard = int(spec["sharding"]["families_per_latent_shard"])
    expected_work_shards = (expected + per_shard - 1) // per_shard
    root = args.output_root.expanduser().resolve()
    if (root / "READY").is_file():
        raise SystemExit(f"store is already READY: {root}")

    all_rows: list[dict[str, Any]] = []
    done_rows = []
    for work_shard in range(expected_work_shards):
        done_path = root / "work_done" / f"work-{work_shard:05d}.json"
        index_path = root / "shard_indexes" / f"families-{work_shard:05d}.jsonl"
        if not done_path.is_file() or not index_path.is_file():
            raise SystemExit(f"work shard {work_shard} is incomplete")
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("split") != args.split or int(done["work_shard"]) != work_shard:
            raise RuntimeError(f"invalid DONE marker: {done_path}")
        rows = list(_rows(index_path))
        if len(rows) != int(done["families"]):
            raise RuntimeError(f"DONE/index count mismatch: {done_path}")
        if args.verify_shard_hashes:
            tensor_path = Path(done["tensor_shard"])
            if _sha256(tensor_path) != done["tensor_sha256"]:
                raise RuntimeError(f"tensor shard hash mismatch: {tensor_path}")
            if _sha256(index_path) != done["index_sha256"]:
                raise RuntimeError(f"index shard hash mismatch: {index_path}")
        all_rows.extend(rows)
        done_rows.append(done)
    if len(all_rows) != expected:
        raise RuntimeError(f"found {len(all_rows)} families, expected {expected}")
    all_rows.sort(key=lambda row: int(row["family_rank"]))
    ranks = [int(row["family_rank"]) for row in all_rows]
    if ranks != list(range(expected)):
        raise RuntimeError("family ranks are not exactly contiguous [0, expected)")
    family_ids = [str(row["family_id"]) for row in all_rows]
    if len(family_ids) != len(set(family_ids)):
        raise RuntimeError("duplicate family_id across work shards")
    for row in all_rows:
        if (
            row.get("split") != args.split
            or int(row["num_turns"]) != int(spec["splits"][args.split]["states_per_family"])
            or int(row["channels"]) != int(spec["audio"]["latent_channels"])
            or int(row["frames"]) != int(spec["audio"]["latent_frames"])
            or row["dtype"] != spec["audio"]["latent_dtype"]
        ):
            raise RuntimeError(f"family schema mismatch: {row['family_id']}")
        for key in ("tensor_shard", "metadata_shard"):
            if not (root / row[key]).is_file():
                raise FileNotFoundError(root / row[key])

    index_count, index_hash = atomic_write_jsonl(root / "index.jsonl", all_rows)
    if index_count != expected:
        raise RuntimeError("portable index count mismatch")
    output = root / "index.sqlite"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE families("
            "family_rank INTEGER PRIMARY KEY, family_id TEXT NOT NULL UNIQUE, "
            "tensor_shard TEXT NOT NULL, tensor_key TEXT NOT NULL, "
            "metadata_shard TEXT NOT NULL, metadata_offset INTEGER NOT NULL, "
            "metadata_length INTEGER NOT NULL, num_turns INTEGER NOT NULL, "
            "channels INTEGER NOT NULL, frames INTEGER NOT NULL, dtype TEXT NOT NULL"
            ") WITHOUT ROWID"
        )
        connection.executemany(
            "INSERT INTO families VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    row["family_rank"], row["family_id"], row["tensor_shard"],
                    row["tensor_key"], row["metadata_shard"], row["metadata_offset"],
                    row["metadata_length"], row["num_turns"], row["channels"],
                    row["frames"], row["dtype"],
                )
                for row in all_rows
            ],
        )
        connection.commit()
        if connection.execute("SELECT COUNT(*) FROM families").fetchone()[0] != expected:
            raise RuntimeError("SQLite family count mismatch")
    finally:
        connection.close()
    os.replace(temporary, output)

    vae_paths = {str(row["vae_checkpoint"]) for row in done_rows}
    vae_configs = {str(row["vae_config"]) for row in done_rows}
    if len(vae_paths) != 1 or len(vae_configs) != 1:
        raise RuntimeError("work shards used different VAE artifacts")
    vae_checkpoint = Path(next(iter(vae_paths)))
    atomic_write_json(
        root / "details.json",
        {
            "schema": SCHEMA,
            "schema_version": VERSION,
            "split": args.split,
            "families": expected,
            "states": expected * int(spec["splits"][args.split]["states_per_family"]),
            "work_shards": expected_work_shards,
            "build_spec": str(args.build_spec.resolve()),
            "vae_config": next(iter(vae_configs)),
            "vae_checkpoint": str(vae_checkpoint),
            "vae_checkpoint_sha256": _sha256(vae_checkpoint),
            "source_recoverability": (
                "catalog source lineage + persistent recipe shards + renderer version "
                "are sufficient to regenerate FOA before re-encoding"
            ),
        },
    )
    atomic_write_json(
        root / "READY",
        {
            "schema": SCHEMA,
            "schema_version": VERSION,
            "split": args.split,
            "families": expected,
            "states": expected * int(spec["splits"][args.split]["states_per_family"]),
            "work_shards": expected_work_shards,
            "index_sha256": index_hash,
            "sqlite_sha256": _sha256(output),
        },
    )
    print(
        json.dumps(
            {"status": "READY", "root": str(root), "families": expected},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
