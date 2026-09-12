#!/usr/bin/env python3
"""Compile the TTS Parquet locator JSONL into a read-only SQLite index."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.t2a_artifacts import atomic_write_json  # noqa: E402


SCHEMA = "stable_audio_tools.tts_source_index"
VERSION = 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    args = parser.parse_args()

    source = args.input_jsonl.expanduser().resolve()
    output = args.output_root.expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"TTS source locator JSONL is unavailable: {source}")
    if args.expected_count is not None and args.expected_count <= 0:
        raise SystemExit("--expected-count must be positive")

    ready_path = output / "READY"
    if ready_path.is_file():
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        count = int(ready.get("sources", -1))
        if args.expected_count is not None and count != args.expected_count:
            raise SystemExit(
                f"READY source index has {count} rows, expected {args.expected_count}"
            )
        print(
            json.dumps(
                {"status": "ALREADY_READY", "root": str(output), "sources": count},
                indent=2,
            )
        )
        return 0
    if output.exists():
        if any(output.iterdir()):
            raise SystemExit(f"output root must be new/empty unless READY: {output}")
        output.rmdir()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    temporary.mkdir()
    database_path = temporary / "index.sqlite"
    digest = hashlib.sha256()
    count = 0
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute(
            "CREATE TABLE sources("
            "source_dataset TEXT NOT NULL, source_id TEXT NOT NULL, "
            "locator_json TEXT NOT NULL, "
            "PRIMARY KEY(source_dataset, source_id)"
            ") WITHOUT ROWID"
        )
        batch: list[tuple[str, str, str]] = []
        with source.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, 1):
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line)
                try:
                    dataset = str(row["source_dataset"])
                    source_id = str(row["source_id"])
                    str(row["parquet_path"])
                    int(row["row_group"])
                    int(row["row_in_group"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid TTS locator at {source}:{line_number}"
                    ) from error
                batch.append(
                    (
                        dataset,
                        source_id,
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                )
                if len(batch) >= 10_000:
                    connection.executemany("INSERT INTO sources VALUES(?,?,?)", batch)
                    count += len(batch)
                    batch.clear()
            if batch:
                connection.executemany("INSERT INTO sources VALUES(?,?,?)", batch)
                count += len(batch)
        connection.commit()
        stored = int(connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0])
        if stored != count:
            raise RuntimeError(f"SQLite row count mismatch: {stored} != {count}")
    except BaseException:
        connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    else:
        connection.close()

    try:
        if args.expected_count is not None and count != args.expected_count:
            raise RuntimeError(
                f"source index contains {count} rows, expected {args.expected_count}"
            )
        details = {
            "schema": SCHEMA,
            "schema_version": VERSION,
            "sources": count,
            "input_jsonl": str(source),
            "input_sha256": digest.hexdigest(),
            "database": "index.sqlite",
            "lookup_key": ["source_dataset", "source_id"],
        }
        atomic_write_json(temporary / "details.json", details)
        atomic_write_json(
            temporary / "READY",
            {
                "schema": SCHEMA,
                "schema_version": VERSION,
                "sources": count,
                "input_sha256": digest.hexdigest(),
            },
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {"status": "READY", "root": str(output), "sources": count},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
