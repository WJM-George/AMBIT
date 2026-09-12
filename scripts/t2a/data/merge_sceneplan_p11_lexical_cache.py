#!/usr/bin/env python3
"""Strictly merge frozen-ASR P11 lexical-cache shards into one atomic DB."""

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
    "contract",
    "confidence_contract",
    "source_manifest",
    "source_manifest_sha256",
    "source_index",
    "ordinal_inventory",
    "ordinal_inventory_sha256",
    "ordinal_selection_contract",
    "source",
    "target_transcript_access",
    "encoder_revision",
    "encoder_model_sha256",
    "language",
    "beam_size",
    "vad_filter",
    "vae_config",
    "vae_checkpoint",
    "vae_checkpoint_sha256",
    "foa_channel",
    "normalization",
    "sample_rate",
    "num_shards",
)


HYPOTHESIS_COLUMNS = (
    "ordinal",
    "text",
    "has_speech",
    "confidence",
    "language",
    "language_probability",
    "mean_average_log_probability",
    "mean_no_speech_probability",
    "speech_seconds",
    "segment_count",
)


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


def _invalid_hypotheses(connection: sqlite3.Connection) -> int:
    return int(
        connection.execute(
            """
            SELECT COUNT(*) FROM hypotheses
            WHERE has_speech NOT IN (0,1)
               OR confidence IS NULL OR confidence < 0.0 OR confidence > 1.0
               OR language IS NULL OR trim(language) = ''
               OR language_probability IS NULL
               OR language_probability < 0.0 OR language_probability > 1.0
               OR speech_seconds IS NULL OR speech_seconds < 0.0
               OR segment_count IS NULL OR segment_count < 0
               OR (segment_count = 0 AND (
                      mean_average_log_probability IS NOT NULL
                   OR mean_no_speech_probability IS NOT NULL
               ))
               OR (segment_count > 0 AND (
                      mean_average_log_probability IS NULL
                   OR mean_no_speech_probability IS NULL
                   OR mean_no_speech_probability < 0.0
                   OR mean_no_speech_probability > 1.0
               ))
               OR (has_speech = 1 AND (text IS NULL OR trim(text) = ''))
               OR (has_speech = 1 AND segment_count = 0)
               OR (has_speech = 0 AND (text IS NULL OR trim(text) != ''))
            """
        ).fetchone()[0]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = [path.expanduser().resolve(strict=True) for path in args.inputs]
    if not inputs:
        raise ValueError("at least one lexical-cache shard is required")
    if len(set(inputs)) != len(inputs):
        raise ValueError("lexical cache input shards must be unique")

    shard_records: list[tuple[Path, dict[str, str], int]] = []
    for path in inputs:
        connection = _readonly(path)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            columns = tuple(
                str(row[1])
                for row in connection.execute("PRAGMA table_info(hypotheses)")
            )
            rows = int(
                connection.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
            )
            invalid = _invalid_hypotheses(connection)
        finally:
            connection.close()
        if metadata.get("schema") != "stable_audio_tools.p11_lexical_cache":
            raise RuntimeError(f"{path} is not a P11 lexical cache")
        if metadata.get("schema_version") != "2":
            raise RuntimeError(f"{path} is not lexical cache v2")
        if columns != HYPOTHESIS_COLUMNS:
            raise RuntimeError(f"{path} lexical-cache columns changed: {columns}")
        if rows != int(metadata.get("rows", -1)):
            raise RuntimeError(f"{path} row metadata is stale")
        if invalid:
            raise RuntimeError(f"{path} contains {invalid} invalid hypotheses")
        shard_records.append((path, metadata, rows))

    reference = shard_records[0][1]
    mismatches = {
        str(path): {
            key: (reference.get(key), metadata.get(key))
            for key in IDENTITY_KEYS
            if metadata.get(key) != reference.get(key)
        }
        for path, metadata, _ in shard_records
    }
    mismatches = {path: values for path, values in mismatches.items() if values}
    if mismatches:
        raise RuntimeError(f"lexical cache shard contracts differ: {mismatches}")

    num_shards = int(reference.get("num_shards", -1))
    shard_indices = sorted(
        int(metadata.get("shard_index", -1))
        for _, metadata, _ in shard_records
    )
    if num_shards != len(inputs) or shard_indices != list(range(num_shards)):
        raise RuntimeError(
            f"lexical cache shard coverage is incomplete: "
            f"{shard_indices}/{num_shards}"
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
    inserted = 0
    elapsed = 0.0
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE hypotheses (
                ordinal INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                has_speech INTEGER NOT NULL,
                confidence REAL NOT NULL,
                language TEXT NOT NULL,
                language_probability REAL NOT NULL,
                mean_average_log_probability REAL,
                mean_no_speech_probability REAL,
                speech_seconds REAL NOT NULL,
                segment_count INTEGER NOT NULL
            );
            """
        )
        select_columns = ",".join(HYPOTHESIS_COLUMNS)
        for path, metadata, rows in sorted(
            shard_records, key=lambda value: int(value[1]["shard_index"])
        ):
            source = _readonly(path)
            try:
                cursor = source.execute(
                    f"SELECT {select_columns} FROM hypotheses ORDER BY ordinal"
                )
                while True:
                    chunk = cursor.fetchmany(4096)
                    if not chunk:
                        break
                    destination.executemany(
                        "INSERT INTO hypotheses VALUES (?,?,?,?,?,?,?,?,?,?)",
                        chunk,
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
                raise RuntimeError("lexical ordinal inventory SHA-256 changed")
            if _sha256(manifest) != reference.get("source_manifest_sha256"):
                raise RuntimeError("lexical source manifest SHA-256 changed")
            inventory_db = _readonly(inventory)
            try:
                inventory_metadata = dict(
                    inventory_db.execute("SELECT key,value FROM metadata")
                )
            finally:
                inventory_db.close()
            required_inventory = {
                "schema": "stable_audio_tools.p11_cache_ordinal_inventory",
                "schema_version": "1",
                "source_manifest": str(manifest),
                "source_manifest_sha256": reference["source_manifest_sha256"],
                "selection_contract": reference["ordinal_selection_contract"],
            }
            for key, expected_value in required_inventory.items():
                if inventory_metadata.get(key) != expected_value:
                    raise RuntimeError(f"lexical ordinal inventory {key} changed")
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
                    "SELECT target_ordinal AS ordinal FROM inventory.ordinals "
                    "EXCEPT SELECT ordinal FROM hypotheses)"
                ).fetchone()[0]
            )
            extra = int(
                destination.execute(
                    "SELECT COUNT(*) FROM (SELECT ordinal FROM hypotheses EXCEPT "
                    "SELECT target_ordinal AS ordinal FROM inventory.ordinals)"
                ).fetchone()[0]
            )
            destination.execute("DETACH DATABASE inventory")
        else:
            destination.execute("ATTACH DATABASE ? AS manifest", (str(manifest),))
            expected = int(
                destination.execute(
                    "SELECT COUNT(DISTINCT target_ordinal) FROM manifest.rows"
                ).fetchone()[0]
            )
            missing = int(
                destination.execute(
                    "SELECT COUNT(*) FROM ("
                    "SELECT DISTINCT target_ordinal AS ordinal FROM manifest.rows "
                    "EXCEPT SELECT ordinal FROM hypotheses)"
                ).fetchone()[0]
            )
            extra = int(
                destination.execute(
                    "SELECT COUNT(*) FROM (SELECT ordinal FROM hypotheses EXCEPT "
                    "SELECT DISTINCT target_ordinal AS ordinal FROM manifest.rows)"
                ).fetchone()[0]
            )
            destination.execute("DETACH DATABASE manifest")
        if inserted != expected or missing or extra:
            raise RuntimeError(
                "merged lexical cache does not exactly cover its manifest: "
                f"inserted={inserted} expected={expected} "
                f"missing={missing} extra={extra}"
            )
        if _invalid_hypotheses(destination):
            raise RuntimeError("merged lexical cache contains invalid hypotheses")

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
            "INSERT INTO metadata(key,value) VALUES (?,?)",
            sorted(merged_metadata.items()),
        )
        destination.commit()
        destination.execute("VACUUM")
        destination.commit()
        completed = True
    finally:
        destination.close()
        if not completed:
            temporary.unlink(missing_ok=True)

    os.replace(temporary, output)
    report = {
        "schema": "stable_audio_tools.p11_lexical_cache_merge",
        "schema_version": 1,
        "status": "PASS",
        "output": str(output),
        "output_sha256": _sha256(output),
        "rows": inserted,
        "shards": num_shards,
        "source": reference["source"],
        "target_transcript_access": reference["target_transcript_access"],
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
