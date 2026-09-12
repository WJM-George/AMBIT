#!/usr/bin/env python3
"""Build the resumable revision-4 dry-speech catalog from local Parquets."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import (  # noqa: E402
    CONTRACT_REVISION,
    DATASET_ROOT,
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    atomic_write_json,
    clean_text,
    model_num_samples,
    normalized_transcript,
    source_split_from_parquet,
)


SOURCES = {
    "libritts": Path("/mnt/sdc/speech_dataset/mythicinfinity__libritts"),
    "hifi_tts": Path("/mnt/sdc/speech_dataset/MikhailT__hifi-tts"),
}
DEFAULT_OUTPUT = DATASET_ROOT / "source_catalog/speech"


def ensure_sdb(path: Path) -> None:
    resolved = path.expanduser().resolve(strict=False)
    try:
        resolved.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"revision-4 output must be on SDB: {resolved}") from error


def columns_for(dataset: str) -> list[str]:
    if dataset == "libritts":
        return [
            "audio",
            "id",
            "speaker_id",
            "chapter_id",
            "text_normalized",
            "text_original",
        ]
    if dataset == "hifi_tts":
        return [
            "audio",
            "speaker",
            "file",
            "duration",
            "text",
            "text_no_preprocessing",
            "text_normalized",
        ]
    raise ValueError(dataset)


def row_identity(dataset: str, row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    if dataset == "libritts":
        source_id = clean_text(row.get("id"))
        speaker = clean_text(row.get("speaker_id"))
        chapter = clean_text(row.get("chapter_id"))
        source_text = clean_text(row.get("text_original") or row.get("text_normalized"))
        renderer_text = clean_text(row.get("text_normalized") or row.get("text_original"))
    elif dataset == "hifi_tts":
        file_value = clean_text(row.get("file"))
        source_id = "hifi_tts_" + Path(file_value).stem if file_value else ""
        speaker = clean_text(row.get("speaker"))
        chapter = str(Path(file_value).parent) if file_value else ""
        source_text = clean_text(
            row.get("text_no_preprocessing") or row.get("text_normalized") or row.get("text")
        )
        renderer_text = clean_text(
            row.get("text_normalized") or row.get("text_no_preprocessing") or row.get("text")
        )
    else:
        raise ValueError(dataset)
    return source_id, speaker, chapter, source_text, renderer_text


def create_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE IF NOT EXISTS assets(
          asset_id TEXT PRIMARY KEY,
          source_dataset TEXT NOT NULL,
          source_id TEXT NOT NULL,
          speaker_id TEXT NOT NULL,
          speaker_key TEXT NOT NULL,
          chapter_id TEXT NOT NULL,
          source_split TEXT NOT NULL,
          source_text TEXT NOT NULL,
          renderer_text TEXT NOT NULL,
          normalized_transcript TEXT NOT NULL,
          normalized_transcript_sha256 TEXT NOT NULL,
          parquet_path TEXT NOT NULL,
          row_group INTEGER NOT NULL,
          row_in_group INTEGER NOT NULL,
          source_audio_sha256 TEXT NOT NULL,
          encoded_num_bytes INTEGER NOT NULL,
          native_sample_rate_hz INTEGER NOT NULL,
          native_num_samples INTEGER NOT NULL,
          native_channels INTEGER NOT NULL,
          model_sample_rate_hz INTEGER NOT NULL,
          model_num_samples INTEGER NOT NULL,
          duration_sec REAL NOT NULL,
          structurally_eligible INTEGER NOT NULL,
          rejection_reason TEXT,
          UNIQUE(source_dataset, source_id),
          UNIQUE(parquet_path, row_group, row_in_group)
        );
        CREATE TABLE IF NOT EXISTS source_files(
          parquet_path TEXT PRIMARY KEY,
          source_dataset TEXT NOT NULL,
          rows INTEGER NOT NULL,
          eligible_rows INTEGER NOT NULL,
          processed_at_unix REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS assets_dataset_eligible
          ON assets(source_dataset, structurally_eligible);
        CREATE INDEX IF NOT EXISTS assets_speaker
          ON assets(source_dataset, speaker_key, structurally_eligible);
        CREATE INDEX IF NOT EXISTS assets_text
          ON assets(normalized_transcript_sha256, structurally_eligible);
        CREATE INDEX IF NOT EXISTS assets_audio_hash
          ON assets(source_audio_sha256, structurally_eligible);
        """
    )


def existing_files(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT parquet_path FROM source_files")}


def rejection_reason(
    source_id: str,
    speaker: str,
    transcript: str,
    channels: int,
    native_frames: int,
    model_frames: int,
) -> str | None:
    if not source_id:
        return "missing_source_id"
    if not speaker:
        return "missing_speaker"
    if not transcript:
        return "empty_normalized_transcript"
    if channels != 1:
        return "not_mono"
    if native_frames <= 0 or model_frames <= 0:
        return "empty_audio"
    if model_frames > MAX_MODEL_SAMPLES:
        return "complete_utterance_exceeds_442368"
    return None


def process_file(
    connection: sqlite3.Connection,
    dataset: str,
    dataset_root: Path,
    parquet_path: Path,
) -> tuple[int, int, Counter[str]]:
    parquet_file = pq.ParquetFile(parquet_path)
    available = set(parquet_file.schema_arrow.names)
    required = columns_for(dataset)
    missing = [name for name in required if name not in available]
    if missing:
        raise RuntimeError(f"{parquet_path} missing columns: {missing}")
    source_split = source_split_from_parquet(parquet_path, dataset_root)
    inserted = 0
    eligible = 0
    reasons: Counter[str] = Counter()
    batch_rows: list[tuple[Any, ...]] = []
    connection.execute("BEGIN")
    try:
        for row_group in range(parquet_file.num_row_groups):
            rows = parquet_file.read_row_group(row_group, columns=required).to_pylist()
            for row_in_group, row in enumerate(rows):
                audio = row.pop("audio", None) or {}
                blob = audio.get("bytes")
                source_id, speaker, chapter, source_text, renderer_text = row_identity(dataset, row)
                transcript = normalized_transcript(renderer_text or source_text)
                audio_hash = hashlib.sha256(bytes(blob or b"")).hexdigest()
                text_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
                encoded_bytes = len(blob or b"")
                native_rate = native_frames = channels = 0
                header_error: str | None = None
                if blob:
                    try:
                        info = sf.info(io.BytesIO(blob))
                        native_rate = int(info.samplerate)
                        native_frames = int(info.frames)
                        channels = int(info.channels)
                    except Exception:
                        header_error = "invalid_audio_header"
                else:
                    header_error = "missing_audio_bytes"
                model_frames = model_num_samples(native_frames, native_rate)
                reason = header_error or rejection_reason(
                    source_id,
                    speaker,
                    transcript,
                    channels,
                    native_frames,
                    model_frames,
                )
                is_eligible = int(reason is None)
                eligible += is_eligible
                if reason:
                    reasons[reason] += 1
                asset_id = f"{dataset}:{source_id or parquet_path.name + ':' + str(row_group) + ':' + str(row_in_group)}"
                batch_rows.append(
                    (
                        asset_id,
                        dataset,
                        source_id,
                        speaker,
                        f"{dataset}:{speaker}",
                        chapter,
                        source_split,
                        source_text,
                        renderer_text,
                        transcript,
                        text_hash,
                        str(parquet_path),
                        row_group,
                        row_in_group,
                        audio_hash,
                        encoded_bytes,
                        native_rate,
                        native_frames,
                        channels,
                        MODEL_SAMPLE_RATE,
                        model_frames,
                        model_frames / MODEL_SAMPLE_RATE if model_frames else 0.0,
                        is_eligible,
                        reason,
                    )
                )
                if len(batch_rows) >= 2_000:
                    connection.executemany(
                        "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        batch_rows,
                    )
                    inserted += len(batch_rows)
                    batch_rows.clear()
        if batch_rows:
            connection.executemany(
                "INSERT INTO assets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                batch_rows,
            )
            inserted += len(batch_rows)
        connection.execute(
            "INSERT INTO source_files VALUES(?,?,?,?,?)",
            (str(parquet_path), dataset, inserted, eligible, time.time()),
        )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    return inserted, eligible, reasons


def query_scalar(connection: sqlite3.Connection, sql: str, parameters: Iterable[Any] = ()) -> int:
    return int(connection.execute(sql, tuple(parameters)).fetchone()[0])


def summarize(connection: sqlite3.Connection) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in SOURCES:
        datasets[dataset] = {
            "rows": query_scalar(connection, "SELECT COUNT(*) FROM assets WHERE source_dataset=?", [dataset]),
            "eligible": query_scalar(
                connection,
                "SELECT COUNT(*) FROM assets WHERE source_dataset=? AND structurally_eligible=1",
                [dataset],
            ),
            "eligible_unique_audio_hashes": query_scalar(
                connection,
                "SELECT COUNT(DISTINCT source_audio_sha256) FROM assets WHERE source_dataset=? AND structurally_eligible=1",
                [dataset],
            ),
            "eligible_unique_normalized_transcripts": query_scalar(
                connection,
                "SELECT COUNT(DISTINCT normalized_transcript_sha256) FROM assets WHERE source_dataset=? AND structurally_eligible=1",
                [dataset],
            ),
            "eligible_speakers": query_scalar(
                connection,
                "SELECT COUNT(DISTINCT speaker_key) FROM assets WHERE source_dataset=? AND structurally_eligible=1",
                [dataset],
            ),
        }
    rejection_rows = connection.execute(
        "SELECT COALESCE(rejection_reason,'eligible'),COUNT(*) FROM assets GROUP BY rejection_reason ORDER BY COUNT(*) DESC"
    ).fetchall()
    summary = {
        "schema": "stable_audio_tools.sceneplan_speech_catalog_summary",
        "schema_version": 1,
        "contract_revision": CONTRACT_REVISION,
        "database": "catalog.sqlite",
        "datasets": datasets,
        "totals": {
            "rows": query_scalar(connection, "SELECT COUNT(*) FROM assets"),
            "eligible": query_scalar(connection, "SELECT COUNT(*) FROM assets WHERE structurally_eligible=1"),
            "eligible_unique_audio_hashes": query_scalar(
                connection,
                "SELECT COUNT(DISTINCT source_audio_sha256) FROM assets WHERE structurally_eligible=1",
            ),
            "eligible_unique_normalized_transcripts": query_scalar(
                connection,
                "SELECT COUNT(DISTINCT normalized_transcript_sha256) FROM assets WHERE structurally_eligible=1",
            ),
            "source_files": query_scalar(connection, "SELECT COUNT(*) FROM source_files"),
        },
        "rejections": {str(reason): int(count) for reason, count in rejection_rows},
        "source_roots": {key: str(value) for key, value in SOURCES.items()},
        "output_mount": "/mnt/sdb",
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve(strict=False)
    ensure_sdb(output)
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "READY"
    if ready.exists() and not args.force_rebuild:
        print(ready.read_text(encoding="utf-8"), end="")
        return 0
    if args.force_rebuild:
        for name in ("catalog.sqlite", "catalog.sqlite-wal", "catalog.sqlite-shm", "READY", "summary.json"):
            path = output / name
            if path.exists():
                path.unlink()

    database = output / "catalog.sqlite"
    connection = sqlite3.connect(database, timeout=120)
    create_database(connection)
    complete = existing_files(connection)
    started = time.time()
    processed_now = 0
    total_files = sum(len(list((root / "data").rglob("*.parquet"))) for root in SOURCES.values())
    try:
        for dataset, dataset_root in SOURCES.items():
            for parquet_path in sorted((dataset_root / "data").rglob("*.parquet")):
                if str(parquet_path) in complete:
                    continue
                rows, eligible, reasons = process_file(
                    connection,
                    dataset,
                    dataset_root,
                    parquet_path,
                )
                processed_now += 1
                if processed_now % 10 == 0 or processed_now == 1:
                    print(
                        json.dumps(
                            {
                                "processed_now": processed_now,
                                "completed_files": len(complete) + processed_now,
                                "total_files": total_files,
                                "last": str(parquet_path),
                                "rows": rows,
                                "eligible": eligible,
                                "rejections": dict(reasons),
                                "elapsed_sec": round(time.time() - started, 1),
                            }
                        ),
                        flush=True,
                    )
        summary = summarize(connection)
    finally:
        connection.close()

    expected_rows = 375_086 + 323_978
    if summary["totals"]["rows"] != expected_rows:
        raise RuntimeError(f"catalog rows {summary['totals']['rows']} != expected {expected_rows}")
    if summary["totals"]["eligible"] < 512_000 + 50_000:
        raise RuntimeError("eligible speech inventory does not leave the required 50k reserve")
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(
        ready,
        {
            "schema": "stable_audio_tools.sceneplan_speech_catalog_ready",
            "schema_version": 1,
            "contract_revision": CONTRACT_REVISION,
            "rows": summary["totals"]["rows"],
            "eligible": summary["totals"]["eligible"],
            "database": str(database),
            "summary": str(output / "summary.json"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
