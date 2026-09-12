#!/usr/bin/env python3
"""Strong-ASR, signal, and endpoint audit of every speech-ledger candidate."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_tts_v2_pilot import score_channel, words  # noqa: E402
from sceneplan_v2_common import (  # noqa: E402
    DATASET_ROOT,
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    atomic_write_json,
    model_num_samples,
    normalized_transcript,
)


LEDGER = DATASET_ROOT / "split_ledgers/speech_v2/speech_split_ledger.parquet"
MODEL = DATASET_ROOT / "models/faster-distil-whisper-large-v3"
OUTPUT = DATASET_ROOT / "source_catalog/speech/strong_qc"
WORK_SHARDS = 128


SCHEMA = pa.schema(
    [
        ("asset_id", pa.string()),
        ("source_dataset", pa.string()),
        ("source_id", pa.string()),
        ("pool", pa.string()),
        ("replacement_split", pa.string()),
        ("speaker_key", pa.string()),
        ("selection_rank", pa.string()),
        ("status", pa.string()),
        ("failure_reasons", pa.list_(pa.string())),
        ("source_audio_sha256", pa.string()),
        ("native_sample_rate_hz", pa.int32()),
        ("native_num_samples", pa.int64()),
        ("model_num_samples", pa.int64()),
        ("signal_rms", pa.float64()),
        ("signal_peak", pa.float64()),
        ("tail_10ms_relative_db", pa.float64()),
        ("last_sample_abs_over_rms", pa.float64()),
        ("recognized_text", pa.string()),
        ("reference_word_count", pa.int32()),
        ("asr_word_count", pa.int32()),
        ("wer", pa.float64()),
        ("ordered_word_coverage", pa.float64()),
        ("suffix_word_coverage", pa.float64()),
        ("final_reference_word_matched", pa.bool_()),
        ("character_error_rate", pa.float64()),
        ("hypothesis_to_reference_char_ratio", pa.float64()),
        ("suffix_char_similarity", pa.float64()),
        ("semantic_complete", pa.bool_()),
        ("endpoint_natural", pa.bool_()),
        ("last_asr_word_end_sec", pa.float64()),
        ("tail_margin_after_last_asr_word_sec", pa.float64()),
        ("elapsed_sec", pa.float64()),
        ("work_shard", pa.int16()),
    ]
)


def source_record(
    row: dict[str, Any],
    cached_key: tuple[str, int] | None,
    cached_table: pa.Table | None,
) -> tuple[bytes, dict[str, Any], tuple[str, int], pa.Table]:
    key = (str(row["parquet_path"]), int(row["row_group"]))
    if key != cached_key or cached_table is None:
        parquet_file = pq.ParquetFile(key[0])
        columns = [
            name
            for name in parquet_file.schema_arrow.names
            if name
            in {
                "audio",
                "text",
                "text_original",
                "text_normalized",
                "text_no_preprocessing",
            }
        ]
        cached_table = parquet_file.read_row_group(key[1], columns=columns)
        cached_key = key
    index = int(row["row_in_group"])
    if not 0 <= index < cached_table.num_rows:
        raise IndexError(f"speech row outside Parquet row group: {row['asset_id']}")
    record = cached_table.slice(index, 1).to_pylist()[0]
    audio = record.pop("audio", None) or {}
    blob = audio.get("bytes")
    if not blob:
        raise ValueError("speech Parquet record has no embedded audio")
    return bytes(blob), record, cached_key, cached_table


def compact_score(score: dict[str, Any]) -> dict[str, Any]:
    alignment = score["alignment"]
    completion = score["completion_text_evidence"]
    return {
        "recognized_text": str(score["recognized_text"]),
        "asr_word_count": len(score["tokens"]),
        "wer": float(alignment["wer"]),
        "ordered_word_coverage": float(alignment["ordered_word_coverage"]),
        "suffix_word_coverage": float(alignment["suffix_word_coverage"]),
        "final_reference_word_matched": bool(alignment["final_reference_word_matched"]),
        "character_error_rate": float(completion["character_error_rate"]),
        "hypothesis_to_reference_char_ratio": float(
            completion["hypothesis_to_reference_char_ratio"]
        ),
        "suffix_char_similarity": float(completion["suffix_char_similarity"]),
        "semantic_complete": bool(score["semantic_complete"]),
        "endpoint_natural": bool(score["endpoint_natural"]),
        "last_asr_word_end_sec": score["last_asr_word_end_sec"],
        "tail_margin_after_last_asr_word_sec": score[
            "tail_margin_after_last_asr_word_sec"
        ],
        "tail_10ms_relative_db": float(score["acoustic_endpoint"]["tail_10ms_relative_db"]),
        "last_sample_abs_over_rms": float(
            score["acoustic_endpoint"]["last_sample_abs_over_rms"]
        ),
    }


def audit_row(
    model: Any,
    row: dict[str, Any],
    work_shard: int,
    cached_key: tuple[str, int] | None,
    cached_table: pa.Table | None,
) -> tuple[dict[str, Any], tuple[str, int], pa.Table]:
    started = time.time()
    failures: list[str] = []
    blob, metadata, cached_key, cached_table = source_record(
        row, cached_key, cached_table
    )
    digest = hashlib.sha256(blob).hexdigest()
    if digest != row["source_audio_sha256"]:
        failures.append("source_audio_sha256_changed")
    audio, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        failures.append("source_not_native_mono")
    mono = audio[:, 0]
    if int(rate) != int(row["native_sample_rate_hz"]):
        failures.append("native_sample_rate_changed")
    if len(mono) != int(row["native_num_samples"]):
        failures.append("native_num_samples_changed")
    derived_model_samples = model_num_samples(len(mono), int(rate))
    if derived_model_samples != int(row["model_num_samples"]):
        failures.append("model_num_samples_changed")
    if not 0 < derived_model_samples <= MAX_MODEL_SAMPLES - 40:
        failures.append("complete_utterance_outside_model_limit")
    if not np.isfinite(mono).all():
        failures.append("non_finite_signal")
    centered = mono.astype(np.float64) - float(np.mean(mono, dtype=np.float64))
    signal_rms = float(np.sqrt(np.mean(np.square(centered))))
    signal_peak = float(np.max(np.abs(centered)))
    if signal_rms < 1e-5 or signal_peak < 1e-4:
        failures.append("silent_or_low_signal")
    parquet_text = str(
        metadata.get("text_normalized")
        or metadata.get("text_no_preprocessing")
        or metadata.get("text_original")
        or metadata.get("text")
        or ""
    )
    if normalized_transcript(parquet_text) != normalized_transcript(
        row["renderer_text"]
    ):
        failures.append("parquet_transcript_lineage_mismatch")
    reference_tokens = words(row["renderer_text"])
    if not reference_tokens:
        failures.append("empty_reference_transcript")
    score = score_channel(model, mono, int(rate), reference_tokens)
    if not score["semantic_complete"]:
        failures.append("dry_reference_not_complete")
    if not score["endpoint_natural"]:
        failures.append("dry_endpoint_not_natural")
    compact = compact_score(score)
    result = {
        "asset_id": str(row["asset_id"]),
        "source_dataset": str(row["source_dataset"]),
        "source_id": str(row["source_id"]),
        "pool": str(row["pool"]),
        "replacement_split": str(row["replacement_split"]),
        "speaker_key": str(row["speaker_key"]),
        "selection_rank": str(row["selection_rank"]),
        "status": "pass" if not failures else "quarantine",
        "failure_reasons": sorted(set(failures)),
        "source_audio_sha256": digest,
        "native_sample_rate_hz": int(rate),
        "native_num_samples": len(mono),
        "model_num_samples": derived_model_samples,
        "signal_rms": signal_rms,
        "signal_peak": signal_peak,
        **compact,
        "reference_word_count": len(reference_tokens),
        "elapsed_sec": round(time.time() - started, 4),
        "work_shard": work_shard,
    }
    return result, cached_key, cached_table


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temporary, compression="zstd")
    if pq.read_metadata(temporary).num_rows != len(rows):
        raise RuntimeError(f"strong-QC shard reopen row count mismatch: {path}")
    os.replace(temporary, path)


def worker(
    worker_index: int,
    gpu_index: int,
    assignments: list[tuple[int, list[dict[str, Any]]]],
    model_path: str,
    output_root: str,
) -> None:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_path,
        device="cuda",
        device_index=gpu_index,
        compute_type="float16",
    )
    root = Path(output_root)
    for assignment_index, (work_shard, rows) in enumerate(assignments, start=1):
        output = root / f"part-{work_shard:05d}.parquet"
        expected_assets = [str(row["asset_id"]) for row in rows]
        if (
            output.is_file()
            and pq.read_metadata(output).num_rows == len(rows)
            and pq.read_table(output, columns=["asset_id"])["asset_id"].to_pylist()
            == expected_assets
        ):
            print(
                json.dumps(
                    {
                        "worker": worker_index,
                        "gpu": gpu_index,
                        "work_shard": work_shard,
                        "state": "skip_verified",
                        "rows": len(rows),
                    }
                ),
                flush=True,
            )
            continue
        cached_key = None
        cached_table = None
        results = []
        started = time.time()
        for position, row in enumerate(rows, start=1):
            try:
                result, cached_key, cached_table = audit_row(
                    model, row, work_shard, cached_key, cached_table
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "asset_id": str(row["asset_id"]),
                    "source_dataset": str(row["source_dataset"]),
                    "source_id": str(row["source_id"]),
                    "pool": str(row["pool"]),
                    "replacement_split": str(row["replacement_split"]),
                    "speaker_key": str(row["speaker_key"]),
                    "selection_rank": str(row["selection_rank"]),
                    "status": "error",
                    "failure_reasons": [f"audit_exception:{type(exc).__name__}"],
                    "source_audio_sha256": str(row["source_audio_sha256"]),
                    "native_sample_rate_hz": int(row["native_sample_rate_hz"]),
                    "native_num_samples": int(row["native_num_samples"]),
                    "model_num_samples": int(row["model_num_samples"]),
                    "signal_rms": None,
                    "signal_peak": None,
                    "tail_10ms_relative_db": None,
                    "last_sample_abs_over_rms": None,
                    "recognized_text": repr(exc),
                    "reference_word_count": len(words(row["renderer_text"])),
                    "asr_word_count": 0,
                    "wer": None,
                    "ordered_word_coverage": None,
                    "suffix_word_coverage": None,
                    "final_reference_word_matched": False,
                    "character_error_rate": None,
                    "hypothesis_to_reference_char_ratio": None,
                    "suffix_char_similarity": None,
                    "semantic_complete": False,
                    "endpoint_natural": False,
                    "last_asr_word_end_sec": None,
                    "tail_margin_after_last_asr_word_sec": None,
                    "elapsed_sec": 0.0,
                    "work_shard": work_shard,
                }
            results.append(result)
            if position % 500 == 0 or position == len(rows):
                print(
                    json.dumps(
                        {
                            "worker": worker_index,
                            "gpu": gpu_index,
                            "work_shard": work_shard,
                            "shard_progress": f"{position}/{len(rows)}",
                            "assignment_progress": f"{assignment_index}/{len(assignments)}",
                            "pass": sum(item["status"] == "pass" for item in results),
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
        atomic_parquet(output, results)


def balanced_work_shards(rows: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    groups: defaultdict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["parquet_path"]), int(row["row_group"]))].append(row)
    for values in groups.values():
        values.sort(key=lambda row: int(row["row_in_group"]))
    heap = [(0, index) for index in range(count)]
    heapq.heapify(heap)
    output: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    for _, values in sorted(
        groups.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        load, index = heapq.heappop(heap)
        output[index].extend(values)
        heapq.heappush(heap, (load + len(values), index))
    for values in output:
        values.sort(
            key=lambda row: (
                str(row["parquet_path"]),
                int(row["row_group"]),
                int(row["row_in_group"]),
            )
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--work-shards", type=int, default=WORK_SHARDS)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    ledger = args.ledger.expanduser().resolve(strict=True)
    model = args.model.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    try:
        output.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
    except ValueError as error:
        raise ValueError(f"speech strong-QC output must be on SDB: {output}") from error
    columns = [
        "asset_id",
        "source_dataset",
        "source_id",
        "pool",
        "replacement_split",
        "speaker_key",
        "renderer_text",
        "parquet_path",
        "row_group",
        "row_in_group",
        "source_audio_sha256",
        "native_sample_rate_hz",
        "native_num_samples",
        "model_num_samples",
        "selection_rank",
    ]
    rows = pq.read_table(ledger, columns=columns).to_pylist()
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if len({str(row["asset_id"]) for row in rows}) != len(rows):
        raise RuntimeError("speech ledger asset ids are not unique")
    work = balanced_work_shards(rows, int(args.work_shards))
    gpu_indices = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpu_indices:
        raise ValueError("--gpus must name at least one GPU")
    output.mkdir(parents=True, exist_ok=True)
    assignments = [
        [(index, work[index]) for index in range(worker, len(work), len(gpu_indices))]
        for worker in range(len(gpu_indices))
    ]
    context = mp.get_context("spawn")
    processes = []
    started = time.time()
    for worker_index, (gpu_index, worker_assignments) in enumerate(
        zip(gpu_indices, assignments)
    ):
        process = context.Process(
            target=worker,
            args=(
                worker_index,
                gpu_index,
                worker_assignments,
                str(model),
                str(output),
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"speech strong-QC worker exited {process.exitcode}")
    parts = sorted(output.glob("part-*.parquet"))
    if len(parts) != len(work):
        raise RuntimeError(f"speech strong-QC shard count mismatch: {len(parts)} != {len(work)}")
    table = pq.read_table(parts)
    if table.num_rows != len(rows):
        raise RuntimeError(f"speech strong-QC row count mismatch: {table.num_rows} != {len(rows)}")
    if pc.count_distinct(table["asset_id"]).as_py() != len(rows):
        raise RuntimeError("speech strong-QC asset ids are not unique")
    if set(map(str, table["asset_id"].to_pylist())) != {
        str(row["asset_id"]) for row in rows
    }:
        raise RuntimeError("speech strong-QC asset-id set does not match the ledger")
    status_counts = Counter(map(str, table["status"].to_pylist()))
    by_dataset_pool_status = Counter(
        zip(
            map(str, table["source_dataset"].to_pylist()),
            map(str, table["pool"].to_pylist()),
            map(str, table["status"].to_pylist()),
        )
    )
    reasons = Counter(
        reason
        for values in table["failure_reasons"].to_pylist()
        for reason in (values or [])
    )
    summary = {
        "schema": "stable_audio_tools.sceneplan_speech_catalog_strong_qc",
        "schema_version": 2,
        "audit_complete": True,
        "rows": len(rows),
        "work_shards": len(parts),
        "status_counts": dict(status_counts),
        "failure_reason_counts": dict(reasons),
        "dataset_pool_status_counts": {
            "|".join(key): value for key, value in sorted(by_dataset_pool_status.items())
        },
        "model": str(model),
        "source_ledger": str(ledger),
        "output_root": str(output),
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
