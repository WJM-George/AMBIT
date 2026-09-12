#!/usr/bin/env python3
"""Strictly merge canonical P11 semantic-cache shards into one atomic DB."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from pathlib import Path


IDENTITY_KEYS = (
    "schema",
    "schema_version",
    "builder",
    "builder_sha256",
    "source_index",
    "source_manifest",
    "source_manifest_sha256",
    "ordinal_inventory",
    "ordinal_inventory_sha256",
    "ordinal_selection_contract",
    "dimension",
    "dtype",
    "encoder_repo",
    "encoder_revision",
    "encoder_model_sha256",
    "vae_config",
    "vae_checkpoint",
    "vae_checkpoint_sha256",
    "foa_channel",
    "normalization",
    "sample_rate",
    "representation",
    "window_sec",
    "hop_sec",
    "num_shards",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = [path.expanduser().resolve(strict=True) for path in args.inputs]
    if len(set(inputs)) != len(inputs):
        raise ValueError("semantic cache input shards must be unique")
    metadata_rows = []
    for path in inputs:
        connection = _readonly(path)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            rows = int(connection.execute("SELECT COUNT(*) FROM features").fetchone()[0])
        finally:
            connection.close()
        if metadata.get("schema") != "stable_audio_tools.p11_semantic_cache":
            raise RuntimeError(f"{path} is not a P11 semantic cache")
        if int(metadata.get("schema_version", -1)) != 2:
            raise RuntimeError(f"{path} is not semantic cache v2")
        if rows != int(metadata.get("rows", -1)):
            raise RuntimeError(f"{path} row metadata is stale")
        metadata_rows.append((path, metadata, rows))

    reference = metadata_rows[0][1]
    mismatches = {
        str(path): {
            key: (reference.get(key), metadata.get(key))
            for key in IDENTITY_KEYS
            if metadata.get(key) != reference.get(key)
        }
        for path, metadata, _ in metadata_rows
    }
    mismatches = {path: value for path, value in mismatches.items() if value}
    if mismatches:
        raise RuntimeError(f"semantic cache shard contracts differ: {mismatches}")
    num_shards = int(reference["num_shards"])
    shard_indices = sorted(int(metadata["shard_index"]) for _, metadata, _ in metadata_rows)
    if num_shards != len(inputs) or shard_indices != list(range(num_shards)):
        raise RuntimeError(
            f"semantic cache shard coverage is incomplete: {shard_indices}/{num_shards}"
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".sqlite", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    destination = sqlite3.connect(temporary)
    completed = False
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE features (
                ordinal INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                windows INTEGER NOT NULL,
                mean_l2_norm REAL NOT NULL
            );
            """
        )
        inserted = 0
        elapsed = 0.0
        for path, metadata, rows in sorted(
            metadata_rows, key=lambda value: int(value[1]["shard_index"])
        ):
            source = _readonly(path)
            try:
                cursor = source.execute(
                    "SELECT ordinal,embedding,windows,mean_l2_norm "
                    "FROM features ORDER BY ordinal"
                )
                while True:
                    chunk = cursor.fetchmany(4096)
                    if not chunk:
                        break
                    destination.executemany(
                        "INSERT INTO features VALUES (?,?,?,?)", chunk
                    )
            finally:
                source.close()
            inserted += rows
            elapsed += float(metadata.get("elapsed_sec", 0.0))
            destination.commit()

        manifest = Path(reference["source_manifest"]).resolve(strict=True)
        inventory_value = reference.get("ordinal_inventory")
        if inventory_value is not None:
            inventory = Path(inventory_value).resolve(strict=True)
            if _sha256(inventory) != reference.get("ordinal_inventory_sha256"):
                raise RuntimeError("semantic ordinal inventory SHA-256 changed")
            if _sha256(manifest) != reference.get("source_manifest_sha256"):
                raise RuntimeError("semantic source manifest SHA-256 changed")
            inventory_db = _readonly(inventory)
            try:
                inventory_metadata = dict(
                    inventory_db.execute("SELECT key,value FROM metadata")
                )
            finally:
                inventory_db.close()
            required_inventory = {
                "schema": "stable_audio_tools.p11_cache_ordinal_inventory",
                "schema_version": "2",
                "source_manifest": str(manifest),
                "source_manifest_sha256": reference["source_manifest_sha256"],
                "selection_contract": reference["ordinal_selection_contract"],
            }
            for key, expected_value in required_inventory.items():
                if inventory_metadata.get(key) != expected_value:
                    raise RuntimeError(
                        f"semantic ordinal inventory {key} changed"
                    )
            destination.execute(
                "ATTACH DATABASE ? AS inventory", (str(inventory),)
            )
            expected = int(
                destination.execute(
                    "SELECT COUNT(*) FROM inventory.ordinals"
                ).fetchone()[0]
            )
            missing = int(
                destination.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT source_ordinal AS ordinal FROM inventory.ordinals "
                    "EXCEPT SELECT ordinal FROM features)"
                ).fetchone()[0]
            )
            extra = int(
                destination.execute(
                    "SELECT COUNT(*) FROM (SELECT ordinal FROM features EXCEPT "
                    "SELECT source_ordinal AS ordinal FROM inventory.ordinals)"
                ).fetchone()[0]
            )
            destination.execute("DETACH DATABASE inventory")
        else:
            destination.execute("ATTACH DATABASE ? AS manifest", (str(manifest),))
            expected = int(
                destination.execute(
                    "SELECT COUNT(DISTINCT source_ordinal) FROM manifest.rows"
                ).fetchone()[0]
            )
            missing = int(
                destination.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT DISTINCT source_ordinal AS ordinal FROM manifest.rows "
                    "EXCEPT SELECT ordinal FROM features)"
                ).fetchone()[0]
            )
            extra = int(
                destination.execute(
                    "SELECT COUNT(*) FROM (SELECT ordinal FROM features EXCEPT "
                    "SELECT DISTINCT source_ordinal AS ordinal FROM manifest.rows)"
                ).fetchone()[0]
            )
            destination.execute("DETACH DATABASE manifest")
        if inserted != expected or missing or extra:
            raise RuntimeError(
                "merged semantic cache does not exactly cover its manifest: "
                f"inserted={inserted} expected={expected} missing={missing} extra={extra}"
            )
        merged_metadata = {
            key: value
            for key, value in reference.items()
            if key not in {"rows", "shard_index", "num_shards", "elapsed_sec"}
        }
        merged_metadata.update(
            {
                "rows": str(inserted),
                "shards_merged": str(num_shards),
            }
        )
        destination.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", merged_metadata.items()
        )
        destination.commit()
        destination.execute("VACUUM")
        destination.commit()
        completed = True
    finally:
        destination.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    if output.exists():
        output.unlink()
    os.replace(temporary, output)
    report = {
        "schema": "stable_audio_tools.p11_semantic_cache_merge",
        "schema_version": 1,
        "status": "PASS",
        "output": str(output),
        "output_sha256": _sha256(output),
        "rows": inserted,
        "shards": num_shards,
        "representation": reference["representation"],
        "source_elapsed_sec_sum": elapsed,
    }
    report_path = output.with_suffix(".build.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
