#!/usr/bin/env python3
"""Build external exact-transcript speech candidates from FLEURS en_us.

The external speech benchmark is deliberately separate from both the native
ScenePlan-FOA test split and the public mono music/sound benchmark.  Candidate
clips must fit the P10 duration contract, have a non-empty exact transcript,
and not duplicate a normalized transcript or exact audio file used by P10
train.  Acoustic fingerprint comparison is a later mandatory gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

# Direct script execution puts scripts/evaluation, rather than the repository
# root, on sys.path.  Import the one canonical transcript normalizer instead of
# duplicating its contract here.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.sceneplan_v2_common import normalized_transcript


SCHEMA = "sceneplan_foa.p10_external_speech_candidate_pool"
SCHEMA_VERSION = 1
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
MAX_DURATION_SEC = 10.0
MIN_DURATION_SEC = 3.0
SAMPLE_RATE_HZ = 16_000
GENDER_NAMES = {0: "male", 1: "female", 2: "speaker"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fleurs-root",
        type=Path,
        default=Path(
            "/mnt/sdb/audio_dataset/evaluation_benchmark/"
            "fleurs_en_us_test_70bb2e84"
        ),
    )
    parser.add_argument(
        "--sceneplan-root",
        type=Path,
        default=Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/sdb/audio_dataset/evaluation_benchmark/"
            "p10_evaluation_benchmark_v1"
        ),
    )
    return parser.parse_args()


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record) + "\n")
    os.replace(temporary, path)


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    keys = sorted({key for record in records for key in record})
    normalized = [{key: record.get(key) for key in keys} for record in records]
    pq.write_table(
        pa.Table.from_pylist(normalized),
        temporary,
        compression="zstd",
        compression_level=9,
    )
    os.replace(temporary, path)


def load_p10_train_identity(sceneplan_root: Path) -> tuple[set[str], set[str]]:
    ledger_path = (
        sceneplan_root / "split_ledgers/speech_v2/speech_split_ledger.parquet"
    )
    transcript_hashes: set[str] = set()
    audio_hashes: set[str] = set()
    parquet = pq.ParquetFile(ledger_path)
    for batch in parquet.iter_batches(
        columns=[
            "pool",
            "normalized_transcript_sha256",
            "source_audio_sha256",
        ],
        batch_size=32_768,
    ):
        pools = batch.column("pool").to_pylist()
        texts = batch.column("normalized_transcript_sha256").to_pylist()
        audios = batch.column("source_audio_sha256").to_pylist()
        for pool, text_hash, audio_hash in zip(pools, texts, audios):
            if pool != "train":
                continue
            if text_hash:
                transcript_hashes.add(str(text_hash))
            if audio_hash:
                audio_hashes.add(str(audio_hash))
    return transcript_hashes, audio_hashes


def metadata_rows(parquet_path: Path, split: str) -> list[dict[str, Any]]:
    columns = [
        "id",
        "num_samples",
        "path",
        "transcription",
        "raw_transcription",
        "gender",
        "language",
    ]
    table = pq.read_table(parquet_path, columns=columns)
    rows: list[dict[str, Any]] = []
    for row_index, row in enumerate(table.to_pylist()):
        row["upstream_split"] = split
        row["source_parquet_path"] = str(parquet_path)
        row["source_parquet_row"] = row_index
        rows.append(row)
    return rows


def make_prompt(gender: str, transcript: str) -> str:
    return f'An adult {gender} speaker clearly says "{transcript}".'


def extract_audio(
    records: list[dict[str, Any]], audio_root: Path, train_audio_hashes: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_parquet: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_parquet.setdefault(record["source_parquet_path"], []).append(record)
    audio_root.mkdir(parents=True, exist_ok=True)
    passed: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for parquet_path, selected in sorted(by_parquet.items()):
        audio_rows = pq.read_table(parquet_path, columns=["audio"]).column("audio").to_pylist()
        for record in selected:
            payload = audio_rows[int(record["source_parquet_row"])]["bytes"]
            digest = sha256_bytes(payload)
            target = audio_root / f"{record['external_clip_id']}.wav"
            if target.exists() and sha256_file(target) != digest:
                raise RuntimeError(f"existing FLEURS extraction has wrong hash: {target}")
            if not target.exists():
                temporary = target.with_suffix(".wav.tmp")
                with temporary.open("wb") as handle:
                    handle.write(payload)
                os.replace(temporary, target)
            record["audio_path"] = str(target)
            record["audio_sha256"] = digest
            if digest in train_audio_hashes:
                record["exact_hash_gate_status"] = "fail"
                record["exclusion_reasons"] = ["exact_file_sha256_used_by_P10_train"]
                excluded.append(record)
            else:
                record["exact_hash_gate_status"] = "pass"
                passed.append(record)
    return passed, excluded


def main() -> None:
    args = parse_args()
    train_transcript_hashes, train_audio_hashes = load_p10_train_identity(
        args.sceneplan_root
    )
    raw_root = args.fleurs_root / "raw_parquet"
    source_files = {
        "validation": raw_root / "validation-0000.parquet",
        "test": raw_root / "test-0000.parquet",
    }
    for path in source_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    upstream_counts: Counter[str] = Counter()
    for split, path in source_files.items():
        for row in metadata_rows(path, split):
            transcript = str(row["raw_transcription"] or "").strip()
            normalized = normalized_transcript(transcript)
            normalized_hash = sha256_text(normalized)
            duration = float(row["num_samples"]) / SAMPLE_RATE_HZ
            gender_id = int(row["gender"])
            gender = GENDER_NAMES.get(gender_id, "speaker")
            # FLEURS `id` is not row-unique (hundreds of ids repeat within a
            # split), so the immutable parquet row locator is part of the key.
            external_clip_id = (
                f"{split}_row_{int(row['source_parquet_row']):05d}_"
                f"id_{int(row['id']):05d}"
            )
            reasons: list[str] = []
            if not normalized:
                reasons.append("empty_normalized_transcript")
            if normalized_hash in train_transcript_hashes:
                reasons.append("normalized_transcript_used_by_P10_train")
            if not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
                reasons.append("duration_outside_P10_3_to_10_sec")
            record = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "candidate_id": f"fleurs:en_us:{external_clip_id}",
                "source_dataset": "google_fleurs",
                "dataset_revision": FLEURS_REVISION,
                "dataset_license": "cc-by-4.0",
                "locale": "en_us",
                "upstream_split": split,
                "external_clip_id": external_clip_id,
                "fleurs_id": int(row["id"]),
                "duration_sec": duration,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "num_channels": 1,
                "gender_id": gender_id,
                "gender": gender,
                "speaker_description": f"an adult {gender} speaker with clear delivery",
                "speaker_description_provenance": "FLEURS_gender_metadata_template",
                "exact_transcript": transcript,
                "normalized_transcript": normalized,
                "normalized_transcript_sha256": normalized_hash,
                "prompt": make_prompt(gender, transcript),
                "source_parquet_path": row["source_parquet_path"],
                "source_parquet_row": int(row["source_parquet_row"]),
                "audio_path": None,
                "audio_sha256": None,
                "lineage_gate_status": "pass" if not reasons else "fail",
                "exact_hash_gate_status": "pending" if not reasons else "not_run",
                "acoustic_fingerprint_gate_status": "pending" if not reasons else "not_run",
                "exclusion_reasons": reasons,
                "selection_rank": stable_rank(f"fleurs:en_us:{external_clip_id}"),
            }
            upstream_counts[split] += 1
            if reasons:
                exclusions.append(record)
            else:
                candidates.append(record)

    candidates, exact_hash_exclusions = extract_audio(
        candidates,
        args.output_root / "staging/fleurs_external_speech_audio",
        train_audio_hashes,
    )
    exclusions.extend(exact_hash_exclusions)
    candidates.sort(key=lambda row: row["selection_rank"])
    exclusions.sort(key=lambda row: row["selection_rank"])

    manifest_root = args.output_root / "manifests/external_speech"
    write_jsonl(manifest_root / "candidate_pool.jsonl", candidates)
    write_parquet(manifest_root / "candidate_pool.parquet", candidates)
    write_jsonl(manifest_root / "candidate_exclusions.jsonl", exclusions)
    write_parquet(manifest_root / "candidate_exclusions.parquet", exclusions)

    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_pool_only_acoustic_fingerprint_gate_pending",
        "source_revision": FLEURS_REVISION,
        "source_license": "cc-by-4.0",
        "source_upstream_rows": dict(sorted(upstream_counts.items())),
        "P10_train_normalized_transcript_hashes": len(train_transcript_hashes),
        "P10_train_speech_audio_hashes": len(train_audio_hashes),
        "candidate_rows": len(candidates),
        "candidate_gender_counts": dict(
            sorted(Counter(row["gender"] for row in candidates).items())
        ),
        "candidate_split_counts": dict(
            sorted(Counter(row["upstream_split"] for row in candidates).items())
        ),
        "excluded_rows": len(exclusions),
        "exclusion_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in exclusions
                    for reason in row["exclusion_reasons"]
                ).items()
            )
        ),
        "target_component_rows": min(500, len(candidates)),
        "overall_speech_core_target_rows": 1000,
        "gate_contract": {
            "duration_and_exact_transcript_gate": "complete",
            "normalized_transcript_vs_P10_train_gate": "complete",
            "exact_file_sha256_vs_P10_train_gate": "complete",
            "acoustic_fingerprint_vs_P10_train_gate": "pending",
        },
    }
    summary_path = args.output_root / "EXTERNAL_SPEECH_CANDIDATE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
