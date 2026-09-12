#!/usr/bin/env python3
"""Eight-GPU strong QC for one frozen speech-expansion candidate table."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from audit_sceneplan_speech_catalog_v2 import compact_score  # noqa: E402
from audit_tts_v2_pilot import score_channel, words  # noqa: E402
from sceneplan_v2_common import model_num_samples, normalized_transcript  # noqa: E402


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
MODEL = DATASET_ROOT / "models/faster-distil-whisper-large-v3"
MAX_MODEL_SAMPLES = 648 * 1024
MODEL_SAMPLE_RATE = 44_100
WORK_SHARDS = 128


SCHEMA = pa.schema(
    [
        ("candidate_id", pa.string()),
        ("source_family", pa.string()),
        ("source_dataset", pa.string()),
        ("source_id", pa.string()),
        ("speaker_key", pa.string()),
        ("duration_sec", pa.float64()),
        ("length_bucket_frames", pa.int16()),
        ("source_text", pa.string()),
        ("renderer_text", pa.string()),
        ("normalized_transcript", pa.string()),
        ("normalized_transcript_sha256", pa.string()),
        ("selection_rank", pa.string()),
        ("locator_json", pa.string()),
        ("source_audio_path", pa.string()),
        ("status", pa.string()),
        ("failure_reasons", pa.list_(pa.string())),
        ("source_audio_sha256", pa.string()),
        ("native_sample_rate_hz", pa.int32()),
        ("native_num_samples", pa.int64()),
        ("model_num_samples", pa.int64()),
        ("latent_frames_valid", pa.int16()),
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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _model_samples(native_frames: int, native_rate: int) -> int:
    # scipy.signal.resample_poly, used by the production renderer, emits the
    # ceiling of the rational output length.  Strong-QC geometry must use the
    # identical convention or some 24-kHz sources are understated by one
    # 44.1-kHz sample.
    return model_num_samples(int(native_frames), int(native_rate))


def _read_parquet_audio(
    locator: dict[str, Any],
    cached_key: tuple[str, int] | None,
    cached_table: pa.Table | None,
) -> tuple[bytes, dict[str, Any], tuple[str, int], pa.Table]:
    key = (str(locator["parquet_path"]), int(locator["row_group"]))
    if key != cached_key or cached_table is None:
        parquet = pq.ParquetFile(key[0])
        columns = [
            name
            for name in parquet.schema_arrow.names
            if name
            in {
                "audio",
                "text",
                "text_original",
                "text_normalized",
                "text_no_preprocessing",
            }
        ]
        cached_table = parquet.read_row_group(key[1], columns=columns)
        cached_key = key
    row_index = int(locator["row_in_group"])
    record = cached_table.slice(row_index, 1).to_pylist()[0]
    audio = record.pop("audio", None) or {}
    blob = audio.get("bytes")
    if not blob:
        raise RuntimeError("speech Parquet record has no embedded audio")
    return bytes(blob), record, cached_key, cached_table


def _read_candidate_audio(
    row: dict[str, Any],
    cached_key: tuple[str, int] | None,
    cached_table: pa.Table | None,
) -> tuple[np.ndarray, int, str, dict[str, Any], tuple[str, int] | None, pa.Table | None]:
    locator = json.loads(str(row["locator_json"]))
    if locator.get("type") == "parquet_row":
        blob, metadata, cached_key, cached_table = _read_parquet_audio(
            locator, cached_key, cached_table
        )
    elif locator.get("type") == "hifitts2_chapter_segment":
        path = Path(str(row.get("source_audio_path") or "")).resolve(strict=True)
        blob = path.read_bytes()
        metadata = {}
    else:
        raise RuntimeError(f"unsupported speech locator: {locator.get('type')!r}")
    decoded, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    if decoded.shape[1] != 1:
        raise RuntimeError("speech candidate is not native mono")
    return (
        decoded[:, 0],
        int(rate),
        _sha256_bytes(blob),
        metadata,
        cached_key,
        cached_table,
    )


def _audit_row(
    model: Any,
    row: dict[str, Any],
    work_shard: int,
    cached_key: tuple[str, int] | None,
    cached_table: pa.Table | None,
) -> tuple[dict[str, Any], tuple[str, int] | None, pa.Table | None]:
    started = time.time()
    failures: list[str] = []
    mono, rate, digest, metadata, cached_key, cached_table = _read_candidate_audio(
        row, cached_key, cached_table
    )
    expected_hash = str(row.get("source_audio_sha256") or "")
    if expected_hash and digest != expected_hash:
        failures.append("source_audio_sha256_changed")
    native_frames = len(mono)
    derived_samples = _model_samples(native_frames, rate)
    expected_duration = float(row["duration_sec"])
    if abs(native_frames / rate - expected_duration) > 0.012:
        failures.append("manifest_duration_changed")
    if not 0 < derived_samples <= MAX_MODEL_SAMPLES - 40:
        failures.append("complete_utterance_outside_648_frame_limit")
    expected_bucket = 432 if derived_samples <= 432 * 1024 else 648
    if expected_bucket != int(row["length_bucket_frames"]):
        failures.append("length_bucket_changed")
    if not np.isfinite(mono).all():
        failures.append("non_finite_signal")
    centered = mono.astype(np.float64) - float(np.mean(mono, dtype=np.float64))
    signal_rms = float(np.sqrt(np.mean(np.square(centered))))
    signal_peak = float(np.max(np.abs(centered)))
    if signal_rms < 1.0e-5 or signal_peak < 1.0e-4:
        failures.append("silent_or_low_signal")
    if metadata:
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
    reference_tokens = words(str(row["renderer_text"]))
    if not reference_tokens:
        failures.append("empty_reference_transcript")
    score = score_channel(model, mono, rate, reference_tokens)
    if not score["semantic_complete"]:
        failures.append("dry_reference_not_complete")
    if not score["endpoint_natural"]:
        failures.append("dry_endpoint_not_natural")
    compact = compact_score(score)
    result = {
        "candidate_id": str(row["candidate_id"]),
        "source_family": str(row["source_family"]),
        "source_dataset": str(row["source_dataset"]),
        "source_id": str(row["source_id"]),
        "speaker_key": str(row["speaker_key"]),
        "duration_sec": expected_duration,
        "length_bucket_frames": int(row["length_bucket_frames"]),
        "source_text": str(row["source_text"]),
        "renderer_text": str(row["renderer_text"]),
        "normalized_transcript": str(row["normalized_transcript"]),
        "normalized_transcript_sha256": str(
            row["normalized_transcript_sha256"]
        ),
        "selection_rank": str(row["selection_rank"]),
        "locator_json": str(row["locator_json"]),
        "source_audio_path": (
            str(row["source_audio_path"]) if row.get("source_audio_path") else None
        ),
        "status": "pass" if not failures else "quarantine",
        "failure_reasons": sorted(set(failures)),
        "source_audio_sha256": digest,
        "native_sample_rate_hz": rate,
        "native_num_samples": native_frames,
        "model_num_samples": derived_samples,
        "latent_frames_valid": math.ceil(derived_samples / 1024),
        "signal_rms": signal_rms,
        "signal_peak": signal_peak,
        **compact,
        "reference_word_count": len(reference_tokens),
        "elapsed_sec": round(time.time() - started, 4),
        "work_shard": work_shard,
    }
    return result, cached_key, cached_table


def _atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMA),
        temporary,
        compression="zstd",
    )
    if pq.read_metadata(temporary).num_rows != len(rows):
        raise RuntimeError("speech-expansion QC shard reopen mismatch")
    os.replace(temporary, path)


def _worker(
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
        expected = [str(row["candidate_id"]) for row in rows]
        if (
            output.is_file()
            and pq.read_metadata(output).num_rows == len(rows)
            and pq.read_table(output, columns=["candidate_id"])[
                "candidate_id"
            ].to_pylist()
            == expected
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
                result, cached_key, cached_table = _audit_row(
                    model, row, work_shard, cached_key, cached_table
                )
            except Exception as error:  # noqa: BLE001
                result = {
                    "candidate_id": str(row["candidate_id"]),
                    "source_family": str(row["source_family"]),
                    "source_dataset": str(row["source_dataset"]),
                    "source_id": str(row["source_id"]),
                    "speaker_key": str(row["speaker_key"]),
                    "duration_sec": float(row["duration_sec"]),
                    "length_bucket_frames": int(row["length_bucket_frames"]),
                    "source_text": str(row["source_text"]),
                    "renderer_text": str(row["renderer_text"]),
                    "normalized_transcript": str(row["normalized_transcript"]),
                    "normalized_transcript_sha256": str(
                        row["normalized_transcript_sha256"]
                    ),
                    "selection_rank": str(row["selection_rank"]),
                    "locator_json": str(row["locator_json"]),
                    "source_audio_path": (
                        str(row["source_audio_path"])
                        if row.get("source_audio_path")
                        else None
                    ),
                    "status": "error",
                    "failure_reasons": [
                        f"audit_exception:{type(error).__name__}"
                    ],
                    "source_audio_sha256": str(
                        row.get("source_audio_sha256") or ""
                    ),
                    "native_sample_rate_hz": None,
                    "native_num_samples": None,
                    "model_num_samples": None,
                    "latent_frames_valid": None,
                    "signal_rms": None,
                    "signal_peak": None,
                    "tail_10ms_relative_db": None,
                    "last_sample_abs_over_rms": None,
                    "recognized_text": repr(error),
                    "reference_word_count": len(words(str(row["renderer_text"]))),
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
            if position % 250 == 0 or position == len(rows):
                print(
                    json.dumps(
                        {
                            "worker": worker_index,
                            "gpu": gpu_index,
                            "work_shard": work_shard,
                            "shard_progress": f"{position}/{len(rows)}",
                            "assignment_progress": (
                                f"{assignment_index}/{len(assignments)}"
                            ),
                            "pass": sum(
                                item["status"] == "pass" for item in results
                            ),
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
        _atomic_parquet(output, results)


def _work_shards(rows: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    output = [[] for _ in range(count)]
    for row in rows:
        digest = hashlib.sha256(str(row["candidate_id"]).encode()).digest()
        output[int.from_bytes(digest[:4], "big") % count].append(row)
    for values in output:
        values.sort(
            key=lambda row: (
                json.loads(str(row["locator_json"])).get("parquet_path", ""),
                int(json.loads(str(row["locator_json"])).get("row_group", -1)),
                int(json.loads(str(row["locator_json"])).get("row_in_group", -1)),
                str(row["candidate_id"]),
            )
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--work-shards", type=int, default=WORK_SHARDS)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    candidates = args.candidates.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    model = args.model.expanduser().resolve(strict=True)
    if not str(output).startswith(os.environ.get("AMBIT_DATA_ROOT", "data")):
        raise ValueError("speech-expansion QC output must remain on SDB")
    rows = pq.read_table(candidates).to_pylist()
    if args.limit is not None:
        rows = rows[: int(args.limit)]
    if len({str(row["candidate_id"]) for row in rows}) != len(rows):
        raise RuntimeError("speech-expansion candidate IDs are not unique")
    work = _work_shards(rows, int(args.work_shards))
    if any(not values for values in work):
        raise RuntimeError("speech-expansion QC has an empty work shard")
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must name at least one GPU")
    output.mkdir(parents=True, exist_ok=True)
    assignments = [
        [(index, work[index]) for index in range(worker, len(work), len(gpus))]
        for worker in range(len(gpus))
    ]
    context = mp.get_context("spawn")
    processes = []
    started = time.time()
    for worker_index, (gpu, worker_assignments) in enumerate(
        zip(gpus, assignments)
    ):
        process = context.Process(
            target=_worker,
            args=(
                worker_index,
                gpu,
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
            raise RuntimeError(
                f"speech-expansion QC worker exited {process.exitcode}"
            )
    parts = sorted(output.glob("part-*.parquet"))
    table = pq.read_table(parts)
    if len(parts) != len(work) or table.num_rows != len(rows):
        raise RuntimeError("speech-expansion QC coverage changed")
    if pc.count_distinct(table["candidate_id"]).as_py() != len(rows):
        raise RuntimeError("speech-expansion QC candidate IDs are not unique")
    status_counts = Counter(map(str, table["status"].to_pylist()))
    reason_counts = Counter(
        reason
        for values in table["failure_reasons"].to_pylist()
        for reason in (values or ())
    )
    passed_by_bucket = Counter(
        str(bucket)
        for bucket, status in zip(
            table["length_bucket_frames"].to_pylist(),
            table["status"].to_pylist(),
        )
        if status == "pass"
    )
    summary = {
        "schema": "stable_audio_tools.speech_expansion_strong_qc",
        "schema_version": 1,
        "state": "complete",
        "rows": len(rows),
        "status_counts": dict(status_counts),
        "passed_by_bucket": dict(passed_by_bucket),
        "failure_reason_counts": dict(reason_counts),
        "candidate_manifest": str(candidates),
        "model": str(model),
        "work_shards": len(parts),
        "elapsed_sec": round(time.time() - started, 3),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
