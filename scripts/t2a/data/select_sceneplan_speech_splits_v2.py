#!/usr/bin/env python3
"""Select the revision-4 speech splits and globally unique reserve ledger.

The selector never materializes or modifies source audio.  It assigns every
eligible speaker to exactly one replacement split, fills the exact per-corpus
quotas, and then records every remaining globally unique candidate as reserve.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import (  # noqa: E402
    CONTRACT_REVISION,
    DATASET_ROOT,
    atomic_write_json,
    deterministic_digest,
)


CATALOG = DATASET_ROOT / "source_catalog/speech/catalog.sqlite"
DEFAULT_OUTPUT = DATASET_ROOT / "split_ledgers/speech_v2"
SEED = 20260814

QUOTAS: dict[str, dict[str, int]] = {
    "libritts": {"train": 250_000, "validation": 5_000, "test": 1_000},
    "hifi_tts": {"train": 250_000, "validation": 5_000, "test": 1_000},
}

# HiFiTTS has ten speakers and no source-level speaker-disjoint official split.
# The two smallest audited speakers are held out entirely; all other speakers
# are train-only.  Surplus rows inherit the same replacement split in reserve.
HIFI_HELD_OUT = {
    "6670": "validation",
    "11697": "test",
}

LEDGER_SCHEMA = pa.schema(
    [
        ("asset_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_id", pa.string()),
        ("pool", pa.string()),
        ("replacement_split", pa.string()),
        ("speaker_id", pa.string()),
        ("speaker_key", pa.string()),
        ("chapter_id", pa.string()),
        ("source_split", pa.string()),
        ("source_text", pa.string()),
        ("renderer_text", pa.string()),
        ("normalized_transcript", pa.string()),
        ("normalized_transcript_sha256", pa.string()),
        ("parquet_path", pa.string()),
        ("row_group", pa.int32()),
        ("row_in_group", pa.int32()),
        ("source_audio_sha256", pa.string()),
        ("encoded_num_bytes", pa.int64()),
        ("native_sample_rate_hz", pa.int32()),
        ("native_num_samples", pa.int64()),
        ("native_channels", pa.int8()),
        ("model_sample_rate_hz", pa.int32()),
        ("model_num_samples", pa.int64()),
        ("duration_sec", pa.float64()),
        ("selection_rank", pa.string()),
        ("lineage_qc", pa.string()),
        ("signal_qc", pa.string()),
        ("endpoint_qc", pa.string()),
        ("asr_qc", pa.string()),
    ]
)


def ensure_sdb(path: Path) -> None:
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
    except ValueError as error:
        raise ValueError(f"revision-4 output must be on SDB: {resolved}") from error


def assigned_split(dataset: str, speaker_id: str, source_split: str) -> str:
    if dataset == "libritts":
        if source_split.startswith("train."):
            return "train"
        if source_split.startswith("dev."):
            return "validation"
        if source_split.startswith("test."):
            return "test"
        raise ValueError(f"unknown LibriTTS source split: {source_split}")
    if dataset == "hifi_tts":
        return HIFI_HELD_OUT.get(speaker_id, "train")
    raise ValueError(f"unapproved dataset: {dataset}")


def create_ledger_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE candidates(
          asset_id TEXT PRIMARY KEY,
          source_dataset TEXT NOT NULL,
          replacement_split TEXT NOT NULL,
          selection_rank TEXT NOT NULL
        );
        CREATE INDEX candidates_cell_rank
          ON candidates(source_dataset,replacement_split,selection_rank);
        CREATE TABLE allocations(
          asset_id TEXT PRIMARY KEY,
          pool TEXT NOT NULL,
          replacement_split TEXT NOT NULL,
          selection_rank TEXT NOT NULL
        );
        CREATE INDEX allocations_pool ON allocations(pool,replacement_split);
        """
    )


def populate_candidates(connection: sqlite3.Connection, catalog: sqlite3.Connection) -> int:
    cursor = catalog.execute(
        """
        SELECT asset_id,source_dataset,speaker_id,source_split
        FROM assets
        WHERE structurally_eligible=1
        ORDER BY asset_id
        """
    )
    inserted = 0
    batch: list[tuple[str, str, str, str]] = []
    for asset_id, dataset, speaker_id, source_split in cursor:
        split = assigned_split(str(dataset), str(speaker_id), str(source_split))
        rank = deterministic_digest(SEED, dataset, split, asset_id)
        batch.append((str(asset_id), str(dataset), split, rank))
        if len(batch) >= 10_000:
            connection.executemany("INSERT INTO candidates VALUES(?,?,?,?)", batch)
            connection.commit()
            inserted += len(batch)
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO candidates VALUES(?,?,?,?)", batch)
        connection.commit()
        inserted += len(batch)
    return inserted


def category_order(connection: sqlite3.Connection) -> list[tuple[str, str, int, int]]:
    cells: list[tuple[str, str, int, int]] = []
    for dataset, split_quotas in QUOTAS.items():
        for split, quota in split_quotas.items():
            available = int(
                connection.execute(
                    "SELECT COUNT(*) FROM candidates WHERE source_dataset=? AND replacement_split=?",
                    (dataset, split),
                ).fetchone()[0]
            )
            if available < quota:
                raise RuntimeError(f"{dataset}/{split}: {available} candidates < quota {quota}")
            cells.append((dataset, split, quota, available))
    # Fill the tightest cells first, protecting the partitions with least slack
    # from cross-corpus duplicate transcripts.
    return sorted(cells, key=lambda item: ((item[3] - item[2]) / item[2], item[0], item[1]))


def select_used(
    connection: sqlite3.Connection,
    catalog: sqlite3.Connection,
) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    used_audio: set[str] = set()
    used_text: set[str] = set()
    audit: list[dict[str, Any]] = []
    for dataset, split, quota, available in category_order(connection):
        selected = 0
        scanned = 0
        cursor = catalog.execute(
            """
            SELECT a.asset_id,a.source_audio_sha256,a.normalized_transcript_sha256,c.selection_rank
            FROM assets AS a
            JOIN ledger.candidates AS c USING(asset_id)
            WHERE c.source_dataset=? AND c.replacement_split=?
            ORDER BY c.selection_rank,a.asset_id
            """,
            (dataset, split),
        )
        batch: list[tuple[str, str, str, str]] = []
        for asset_id, audio_hash, text_hash, rank in cursor:
            scanned += 1
            if audio_hash in used_audio or text_hash in used_text:
                continue
            used_audio.add(str(audio_hash))
            used_text.add(str(text_hash))
            batch.append((str(asset_id), split, split, str(rank)))
            selected += 1
            if len(batch) >= 10_000:
                connection.executemany("INSERT INTO allocations VALUES(?,?,?,?)", batch)
                connection.commit()
                batch.clear()
            if selected == quota:
                break
        if batch:
            connection.executemany("INSERT INTO allocations VALUES(?,?,?,?)", batch)
            connection.commit()
        if selected != quota:
            raise RuntimeError(
                f"{dataset}/{split}: only {selected} globally unique rows for quota {quota}"
            )
        audit.append(
            {
                "source_dataset": dataset,
                "split": split,
                "quota": quota,
                "available_before_global_dedupe": available,
                "rows_scanned": scanned,
                "duplicate_rows_skipped": scanned - selected,
            }
        )
    return used_audio, used_text, audit


def select_reserve(
    connection: sqlite3.Connection,
    catalog: sqlite3.Connection,
    used_audio: set[str],
    used_text: set[str],
) -> tuple[int, dict[str, int]]:
    cursor = catalog.execute(
        """
        SELECT a.asset_id,a.source_audio_sha256,a.normalized_transcript_sha256,
               c.replacement_split,c.selection_rank,c.source_dataset
        FROM assets AS a
        JOIN ledger.candidates AS c USING(asset_id)
        LEFT JOIN ledger.allocations AS x USING(asset_id)
        WHERE x.asset_id IS NULL
        ORDER BY c.selection_rank,a.asset_id
        """
    )
    batch: list[tuple[str, str, str, str]] = []
    counts: defaultdict[str, int] = defaultdict(int)
    total = 0
    for asset_id, audio_hash, text_hash, split, rank, dataset in cursor:
        if audio_hash in used_audio or text_hash in used_text:
            continue
        used_audio.add(str(audio_hash))
        used_text.add(str(text_hash))
        batch.append((str(asset_id), "reserve", str(split), str(rank)))
        counts[f"{dataset}/{split}"] += 1
        total += 1
        if len(batch) >= 10_000:
            connection.executemany("INSERT INTO allocations VALUES(?,?,?,?)", batch)
            connection.commit()
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO allocations VALUES(?,?,?,?)", batch)
        connection.commit()
    return total, dict(sorted(counts.items()))


def export_parquet(
    ledger: sqlite3.Connection,
    catalog: sqlite3.Connection,
    destination: Path,
) -> int:
    query = catalog.execute(
        """
        SELECT
          a.asset_id,a.source_dataset,a.source_id,x.pool,x.replacement_split,
          a.speaker_id,a.speaker_key,a.chapter_id,a.source_split,a.source_text,
          a.renderer_text,a.normalized_transcript,a.normalized_transcript_sha256,
          a.parquet_path,a.row_group,a.row_in_group,a.source_audio_sha256,
          a.encoded_num_bytes,a.native_sample_rate_hz,a.native_num_samples,
          a.native_channels,a.model_sample_rate_hz,a.model_num_samples,a.duration_sec,
          x.selection_rank
        FROM assets AS a
        JOIN ledger.allocations AS x USING(asset_id)
        ORDER BY CASE x.pool WHEN 'train' THEN 0 WHEN 'validation' THEN 1
                            WHEN 'test' THEN 2 ELSE 3 END,
                 a.source_dataset,x.selection_rank,a.asset_id
        """
    )
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    writer = pq.ParquetWriter(temporary, LEDGER_SCHEMA, compression="zstd")
    written = 0
    try:
        while True:
            rows = query.fetchmany(20_000)
            if not rows:
                break
            records = [
                {
                    **dict(zip([field.name for field in LEDGER_SCHEMA][:-4], row)),
                    "lineage_qc": "structural_pass",
                    "signal_qc": "pending_P4_or_P9",
                    "endpoint_qc": "pending_P4_or_P9",
                    "asr_qc": "pending_P4_or_P9",
                }
                for row in rows
            ]
            writer.write_table(pa.Table.from_pylist(records, schema=LEDGER_SCHEMA))
            written += len(records)
    finally:
        writer.close()
    os.replace(temporary, destination)
    return written


def scalar(connection: sqlite3.Connection, sql: str, parameters: tuple[Any, ...] = ()) -> int:
    return int(connection.execute(sql, parameters).fetchone()[0])


def validate_and_summarize(
    ledger: sqlite3.Connection,
    catalog: sqlite3.Connection,
    selection_audit: list[dict[str, Any]],
    reserve_count: int,
    reserve_cells: dict[str, int],
    exported_rows: int,
) -> dict[str, Any]:
    used_counts: dict[str, dict[str, int]] = {}
    for dataset, split_quotas in QUOTAS.items():
        used_counts[dataset] = {}
        for split, quota in split_quotas.items():
            actual = scalar(
                ledger,
                """
                SELECT COUNT(*) FROM allocations AS x
                JOIN candidates AS c USING(asset_id)
                WHERE c.source_dataset=? AND x.pool=?
                """,
                (dataset, split),
            )
            if actual != quota:
                raise RuntimeError(f"{dataset}/{split}: selected {actual} != {quota}")
            used_counts[dataset][split] = actual
    used_total = scalar(ledger, "SELECT COUNT(*) FROM allocations WHERE pool!='reserve'")
    if used_total != 512_000:
        raise RuntimeError(f"used total {used_total} != 512000")
    if reserve_count < 50_000:
        raise RuntimeError(f"reserve {reserve_count} < required 50000")
    allocation_total = scalar(ledger, "SELECT COUNT(*) FROM allocations")
    if allocation_total != exported_rows:
        raise RuntimeError(f"Parquet rows {exported_rows} != allocations {allocation_total}")

    duplicate_audio = scalar(
        catalog,
        """
        SELECT COUNT(*) FROM (
          SELECT source_audio_sha256 FROM assets
          JOIN ledger.allocations USING(asset_id)
          GROUP BY source_audio_sha256 HAVING COUNT(*)>1
        )
        """,
    )
    duplicate_text = scalar(
        catalog,
        """
        SELECT COUNT(*) FROM (
          SELECT normalized_transcript_sha256 FROM assets
          JOIN ledger.allocations USING(asset_id)
          GROUP BY normalized_transcript_sha256 HAVING COUNT(*)>1
        )
        """,
    )
    speaker_leakage = scalar(
        catalog,
        """
        SELECT COUNT(*) FROM (
          SELECT speaker_key FROM assets
          JOIN ledger.allocations AS x USING(asset_id)
          GROUP BY speaker_key HAVING COUNT(DISTINCT x.replacement_split)>1
        )
        """,
    )
    if duplicate_audio or duplicate_text or speaker_leakage:
        raise RuntimeError(
            f"invariant failure audio={duplicate_audio} text={duplicate_text} "
            f"speaker_leakage={speaker_leakage}"
        )

    return {
        "schema": "stable_audio_tools.sceneplan_speech_split_summary",
        "schema_version": 1,
        "contract_revision": CONTRACT_REVISION,
        "seed": SEED,
        "used_counts": used_counts,
        "used_total": used_total,
        "reserve_total": reserve_count,
        "reserve_counts_by_dataset_and_replacement_split": reserve_cells,
        "allocation_total": allocation_total,
        "global_duplicate_audio_hashes": duplicate_audio,
        "global_duplicate_normalized_transcripts": duplicate_text,
        "speaker_split_leakage": speaker_leakage,
        "selection_cells": selection_audit,
        "hifi_tts_speaker_assignment": {
            "validation": ["6670"],
            "test": ["11697"],
            "train": "all_other_hifi_tts_speakers",
        },
        "libritts_speaker_assignment": {
            "train": "official train.* speakers",
            "validation": "official dev.* speakers",
            "test": "official test.* speakers",
        },
        "quality_control_state": {
            "lineage": "structural_pass",
            "signal_endpoint_asr": "pending_P4_or_P9",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    catalog_path = args.catalog.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    ensure_sdb(output)
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "READY"
    if ready.exists() and not args.force_rebuild:
        print(ready.read_text(encoding="utf-8"), end="")
        return 0
    paths = [
        output / "ledger.sqlite",
        output / "ledger.sqlite-wal",
        output / "ledger.sqlite-shm",
        output / "speech_split_ledger.parquet",
        output / "summary.json",
        ready,
    ]
    if args.force_rebuild:
        for path in paths:
            if path.exists():
                path.unlink()
    elif any(path.exists() for path in paths):
        raise RuntimeError(f"partial P3 output exists below {output}; rerun with --force-rebuild")

    ledger = sqlite3.connect(output / "ledger.sqlite", timeout=120)
    catalog = sqlite3.connect(catalog_path, timeout=120)
    catalog.execute("ATTACH DATABASE ? AS ledger", (str(output / "ledger.sqlite"),))
    try:
        create_ledger_database(ledger)
        candidates = populate_candidates(ledger, catalog)
        print(json.dumps({"eligible_candidates": candidates}), flush=True)
        used_audio, used_text, audit = select_used(ledger, catalog)
        print(json.dumps({"used": len(used_audio), "selection_cells": audit}), flush=True)
        reserve_count, reserve_cells = select_reserve(ledger, catalog, used_audio, used_text)
        print(json.dumps({"reserve": reserve_count, "reserve_cells": reserve_cells}), flush=True)
        exported = export_parquet(
            ledger,
            catalog,
            output / "speech_split_ledger.parquet",
        )
        summary = validate_and_summarize(
            ledger,
            catalog,
            audit,
            reserve_count,
            reserve_cells,
            exported,
        )
    finally:
        catalog.close()
        ledger.close()

    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(
        ready,
        {
            "schema": "stable_audio_tools.sceneplan_speech_split_ready",
            "schema_version": 1,
            "contract_revision": CONTRACT_REVISION,
            "used": 512_000,
            "reserve": reserve_count,
            "ledger": str(output / "speech_split_ledger.parquet"),
            "summary": str(output / "summary.json"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
