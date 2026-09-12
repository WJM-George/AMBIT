#!/usr/bin/env python3
"""Build one immutable ordinal inventory shared by P11 evidence caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path


SCHEMA = "stable_audio_tools.p11_cache_ordinal_inventory"
VERSION = 2
SELECTION_CONTRACT = "sorted_distinct_source_ordinal_then_strided_shard_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro&immutable=1", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def _validate_inventory(manifest_path: Path, output: Path) -> dict[str, object]:
    inventory_path = output.resolve(strict=True)
    connection = _readonly(inventory_path)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        rows = int(connection.execute("SELECT COUNT(*) FROM ordinals").fetchone()[0])
        bounds = connection.execute(
            "SELECT MIN(position),MAX(position) FROM ordinals"
        ).fetchone()
    finally:
        connection.close()
    required = {
        "schema": SCHEMA,
        "schema_version": str(VERSION),
        "selection_contract": SELECTION_CONTRACT,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"ordinal inventory {key}={metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    if rows != int(metadata.get("rows", -1)) or bounds != (0, rows - 1):
        raise RuntimeError("ordinal inventory rows are incomplete")
    return {
        "schema": "stable_audio_tools.p11_cache_ordinal_inventory_validation",
        "schema_version": 1,
        "status": "PASS",
        "inventory": str(inventory_path),
        "inventory_sha256": _sha256(inventory_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": required["source_manifest_sha256"],
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if args.validate_only:
        print(json.dumps(_validate_inventory(manifest_path, output), indent=2, sort_keys=True))
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    manifest_sha256 = _sha256(manifest_path)
    source = _readonly(manifest_path)
    source_metadata = dict(source.execute("SELECT key,value FROM metadata"))
    expected = int(source_metadata.get("base_samples", -1))
    if expected <= 0:
        raise RuntimeError("P11 manifest lacks a positive base_samples count")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".sqlite", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    destination = sqlite3.connect(temporary)
    completed = False
    started = time.perf_counter()
    written = 0
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE ordinals (
                position INTEGER PRIMARY KEY,
                source_ordinal INTEGER NOT NULL UNIQUE
            );
            """
        )
        cursor = source.execute(
            "SELECT DISTINCT source_ordinal FROM rows ORDER BY source_ordinal"
        )
        while True:
            values = cursor.fetchmany(8192)
            if not values:
                break
            destination.executemany(
                "INSERT INTO ordinals(position,source_ordinal) VALUES (?,?)",
                (
                    (written + offset, int(value[0]))
                    for offset, value in enumerate(values)
                ),
            )
            written += len(values)
            if written % 131072 == 0:
                destination.commit()
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "rows": written,
                            "expected": expected,
                            "elapsed_sec": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        if written != expected:
            raise RuntimeError(
                f"ordinal inventory selected {written} rows, expected {expected}"
            )
        bounds = destination.execute(
            "SELECT MIN(position),MAX(position),MIN(source_ordinal),"
            "MAX(source_ordinal) FROM ordinals"
        ).fetchone()
        if bounds[0:2] != (0, written - 1):
            raise RuntimeError("ordinal inventory positions are not contiguous")
        metadata = {
            "schema": SCHEMA,
            "schema_version": str(VERSION),
            "selection_contract": SELECTION_CONTRACT,
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": manifest_sha256,
            "source_manifest_schema": str(source_metadata.get("schema")),
            "source_manifest_schema_version": str(
                source_metadata.get("schema_version")
            ),
            "rows": str(written),
            "source_ordinal_min": str(bounds[2]),
            "source_ordinal_max": str(bounds[3]),
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": _sha256(Path(__file__).resolve()),
        }
        destination.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            sorted(metadata.items()),
        )
        destination.commit()
        completed = True
    finally:
        destination.close()
        source.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    os.replace(temporary, output)
    report = {
        "schema": "stable_audio_tools.p11_cache_ordinal_inventory_build",
        "schema_version": 1,
        "status": "PASS",
        "output": str(output),
        "output_sha256": _sha256(output),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "rows": written,
        "selection_contract": SELECTION_CONTRACT,
        "elapsed_sec": time.perf_counter() - started,
    }
    report_path = output.with_suffix(".build.json")
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
