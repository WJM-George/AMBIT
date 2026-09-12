#!/usr/bin/env python3
"""Freeze the 200k source-disjoint speech donors for revision 6.

The candidate planner deliberately over-selects HiFiTTS-2 before ASR QC.  This
stage is the only place that turns those candidates into the exact 100k short
and 100k long inventory used by the 500k ScenePlan delta.  It fails closed on
audio/text collisions with the formal base splits and on missing file-backed
audio lineage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
DEFAULT_RESERVE = (
    REVISION_ROOT / "sources/candidates/existing_train_reserve.parquet"
)
DEFAULT_LONG_QC = REVISION_ROOT / "sources/qc/existing_train_long"
DEFAULT_EXTERNAL_QC = REVISION_ROOT / "sources/qc/nvidia_hifitts2_44khz"
DEFAULT_LEDGER = (
    DATASET_ROOT / "split_ledgers/speech_v2/speech_split_ledger.parquet"
)
DEFAULT_OUTPUT = REVISION_ROOT / "sources/registry/final_speech_donors.parquet"
FORMAL_POOLS = {"train", "validation", "test"}
SHORT_TARGET = 100_000
LONG_TARGET = 100_000
MAX_MODEL_SAMPLES = 648 * 1024


SCHEMA = pa.schema(
    [
        ("schema", pa.string()),
        ("schema_version", pa.int16()),
        ("donor_ordinal", pa.int32()),
        ("length_bucket_frames", pa.int16()),
        ("asset_id", pa.string()),
        ("source_family", pa.string()),
        ("source_dataset", pa.string()),
        ("source_id", pa.string()),
        ("speaker_id", pa.string()),
        ("speaker_key", pa.string()),
        ("source_text", pa.string()),
        ("renderer_text", pa.string()),
        ("normalized_transcript", pa.string()),
        ("normalized_transcript_sha256", pa.string()),
        ("source_audio_sha256", pa.string()),
        ("source_audio_path", pa.string()),
        ("native_sample_rate_hz", pa.int32()),
        ("native_num_samples", pa.int64()),
        ("model_num_samples", pa.int64()),
        ("latent_frames_valid", pa.int16()),
        ("duration_sec", pa.float64()),
        ("locator_json", pa.string()),
        ("selection_rank", pa.string()),
        ("strong_qc_status", pa.string()),
    ]
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMA),
        temporary,
        compression="zstd",
        row_group_size=8192,
    )
    if pq.read_metadata(temporary).num_rows != len(rows):
        raise RuntimeError("speech donor registry reopen count mismatch")
    os.replace(temporary, path)


def qc_rows(root: Path) -> list[dict[str, Any]]:
    summary = root / "summary.json"
    if not summary.is_file():
        raise RuntimeError(f"strong QC is not complete: {root}")
    value = json.loads(summary.read_text(encoding="utf-8"))
    if value.get("state") != "complete":
        raise RuntimeError(f"strong QC completion marker is invalid: {root}")
    parts = sorted(root.glob("part-*.parquet"))
    if len(parts) != int(value["work_shards"]):
        raise RuntimeError(f"strong QC shard coverage changed: {root}")
    table = pq.read_table(parts)
    if table.num_rows != int(value["rows"]):
        raise RuntimeError(f"strong QC row coverage changed: {root}")
    return [row for row in table.to_pylist() if row["status"] == "pass"]


def formal_lineage(ledger: Path) -> tuple[set[str], set[str], dict[str, dict[str, Any]]]:
    rows = pq.read_table(ledger).to_pylist()
    formal = [row for row in rows if str(row["pool"]) in FORMAL_POOLS]
    audio = {str(row["source_audio_sha256"]) for row in formal}
    text = {str(row["normalized_transcript_sha256"]) for row in formal}
    by_hash = {str(row["source_audio_sha256"]): row for row in rows}
    if len(audio) != len(formal) or len(text) != len(formal):
        raise RuntimeError("formal speech base is no longer audio/text unique")
    return audio, text, by_hash


def select_unique(
    candidates: Iterable[dict[str, Any]],
    *,
    count: int,
    occupied_audio: set[str],
    occupied_text: set[str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: str(item["selection_rank"])):
        audio_hash = str(row.get("source_audio_sha256") or "")
        text_hash = str(row.get("normalized_transcript_sha256") or "")
        if not audio_hash or not text_hash:
            raise RuntimeError("candidate has incomplete audio/text lineage")
        if audio_hash in occupied_audio or text_hash in occupied_text:
            continue
        occupied_audio.add(audio_hash)
        occupied_text.add(text_hash)
        selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} unique passed donors, need {count}")
    return selected


def canonical_row(
    row: dict[str, Any],
    *,
    ordinal: int,
    ledger_by_hash: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    audio_hash = str(row["source_audio_sha256"])
    existing = ledger_by_hash.get(audio_hash)
    locator = json.loads(str(row["locator_json"]))
    source_dataset = str(row["source_dataset"])
    source_id = str(row["source_id"])
    speaker_key = str(row["speaker_key"])
    speaker_id = speaker_key.split(":", 1)[-1]
    lineage_geometry = existing if existing is not None else row
    if "model_num_samples" not in lineage_geometry:
        raise RuntimeError(f"{source_dataset}:{source_id}: missing recorded model geometry")
    recorded_model_samples = int(lineage_geometry["model_num_samples"])
    if existing is not None:
        asset_id = str(existing["asset_id"])
        native_rate = int(existing["native_sample_rate_hz"])
        native_samples = int(existing["native_num_samples"])
        source_path = None
    elif locator.get("type") == "parquet_row":
        # Long LibriTTS/HiFiTTS candidates can be source-disjoint from the
        # frozen formal ledger while still living in immutable Parquet rows.
        # Their strong-QC record owns the decoded geometry; absence from the
        # ledger does not turn them into file-backed HiFiTTS-2 material.
        asset_id = f"{source_dataset}:{source_id}"
        native_rate = int(row["native_sample_rate_hz"])
        native_samples = int(row["native_num_samples"])
        source_path = None
    else:
        asset_id = f"{source_dataset}:{source_id}"
        native_rate = int(row["native_sample_rate_hz"])
        native_samples = int(row["native_num_samples"])
        source_path = str(row.get("source_audio_path") or "")
        if not source_path:
            raise RuntimeError(f"{asset_id}: file-backed donor has no audio path")
        path = Path(source_path).resolve(strict=True)
        if sha256_file(path) != audio_hash:
            raise RuntimeError(f"{asset_id}: file-backed donor hash changed")
        source_path = str(path)
    # Canonical model geometry follows the production polyphase resampler:
    # ceil(native_frames * 44_100 / native_rate).  Legacy revision-6 strong-QC
    # receipts used round(), which can be exactly one sample shorter.  The QC
    # semantic/endpoint decision remains valid; only this derived geometry is
    # corrected here.  Larger disagreement is a real lineage failure.
    model_samples = math.ceil(native_samples * 44_100 / native_rate)
    if abs(model_samples - recorded_model_samples) > 1:
        raise RuntimeError(
            f"{asset_id}: recorded/canonical resample geometry disagrees by more than one sample"
        )
    latent_frames = math.ceil(model_samples / 1024)
    expected_frames = latent_frames
    expected_bucket = 432 if expected_frames <= 432 else 648
    if (
        latent_frames != expected_frames
        or expected_bucket != int(row["length_bucket_frames"])
        or not 0 < model_samples <= MAX_MODEL_SAMPLES - 40
    ):
        raise RuntimeError(f"{asset_id}: donor geometry is outside revision 6")
    if source_path:
        locator = {
            "type": "file",
            "dry_audio_path": source_path,
        }
    elif locator.get("type") == "parquet_row":
        parquet_path = Path(str(locator.get("parquet_path") or "")).resolve(
            strict=True
        )
        if not parquet_path.is_file():
            raise RuntimeError(f"{asset_id}: Parquet locator is not a file")
        row_group = int(locator.get("row_group", -1))
        row_in_group = int(locator.get("row_in_group", -1))
        if row_group < 0 or row_in_group < 0:
            raise RuntimeError(f"{asset_id}: invalid Parquet row coordinates")
        locator = {
            "type": "parquet_row",
            "parquet_path": str(parquet_path),
            "row_group": row_group,
            "row_in_group": row_in_group,
        }
    else:
        raise RuntimeError(f"{asset_id}: unsupported donor locator")
    return {
        "schema": "stable_audio_tools.sceneplan_speech_donor",
        "schema_version": 1,
        "donor_ordinal": ordinal,
        "length_bucket_frames": expected_bucket,
        "asset_id": asset_id,
        "source_family": str(row["source_family"]),
        "source_dataset": source_dataset,
        "source_id": source_id,
        "speaker_id": speaker_id,
        "speaker_key": speaker_key,
        "source_text": str(row["source_text"]),
        "renderer_text": str(row["renderer_text"]),
        "normalized_transcript": str(row["normalized_transcript"]),
        "normalized_transcript_sha256": str(
            row["normalized_transcript_sha256"]
        ),
        "source_audio_sha256": audio_hash,
        "source_audio_path": source_path,
        "native_sample_rate_hz": native_rate,
        "native_num_samples": native_samples,
        "model_num_samples": model_samples,
        "latent_frames_valid": latent_frames,
        "duration_sec": model_samples / 44_100.0,
        "locator_json": json.dumps(locator, sort_keys=True),
        "selection_rank": str(row["selection_rank"]),
        "strong_qc_status": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve", type=Path, default=DEFAULT_RESERVE)
    parser.add_argument("--long-qc", type=Path, default=DEFAULT_LONG_QC)
    parser.add_argument("--external-qc", type=Path, default=DEFAULT_EXTERNAL_QC)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reserve_path = args.reserve.expanduser().resolve(strict=True)
    long_root = args.long_qc.expanduser().resolve(strict=True)
    external_root = args.external_qc.expanduser().resolve(strict=True)
    ledger = args.ledger.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    if not str(output).startswith(os.environ.get("AMBIT_DATA_ROOT", "data")):
        raise ValueError("speech donor registry must remain on SDB")

    formal_audio, formal_text, ledger_by_hash = formal_lineage(ledger)
    occupied_audio = set(formal_audio)
    occupied_text = set(formal_text)
    reserve = pq.read_table(reserve_path).to_pylist()
    long_pass = qc_rows(long_root)
    external_pass = qc_rows(external_root)

    reserve_selected = select_unique(
        reserve,
        count=len(reserve),
        occupied_audio=occupied_audio,
        occupied_text=occupied_text,
    )
    external_short = [
        row for row in external_pass if int(row["length_bucket_frames"]) == 432
    ]
    short_external_selected = select_unique(
        external_short,
        count=SHORT_TARGET - len(reserve_selected),
        occupied_audio=occupied_audio,
        occupied_text=occupied_text,
    )
    existing_long_selected = select_unique(
        long_pass,
        count=len(long_pass),
        occupied_audio=occupied_audio,
        occupied_text=occupied_text,
    )
    if len(existing_long_selected) > LONG_TARGET:
        existing_long_selected = existing_long_selected[:LONG_TARGET]
    external_long = [
        row for row in external_pass if int(row["length_bucket_frames"]) == 648
    ]
    long_external_selected = select_unique(
        external_long,
        count=LONG_TARGET - len(existing_long_selected),
        occupied_audio=occupied_audio,
        occupied_text=occupied_text,
    )
    selected = (
        reserve_selected
        + short_external_selected
        + existing_long_selected
        + long_external_selected
    )
    rows = [
        canonical_row(row, ordinal=index, ledger_by_hash=ledger_by_hash)
        for index, row in enumerate(selected)
    ]
    if len(rows) != 200_000:
        raise RuntimeError("final donor registry is not exactly 200k")
    counts = Counter(row["length_bucket_frames"] for row in rows)
    if counts != Counter({432: SHORT_TARGET, 648: LONG_TARGET}):
        raise RuntimeError(f"final donor bucket counts changed: {counts}")
    if len({row["asset_id"] for row in rows}) != len(rows):
        raise RuntimeError("final donor asset IDs are not unique")
    if len({row["source_audio_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("final donor audio hashes are not unique")
    if len({row["normalized_transcript_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("final donor transcripts are not unique")
    atomic_parquet(output, rows)
    summary = {
        "schema": "stable_audio_tools.sceneplan_speech_donor_registry_summary",
        "schema_version": 1,
        "state": "complete",
        "rows": len(rows),
        "length_bucket_counts": {str(key): value for key, value in counts.items()},
        "source_family_counts": dict(Counter(row["source_family"] for row in rows)),
        "source_dataset_counts": dict(Counter(row["source_dataset"] for row in rows)),
        "unique_speakers": len({row["speaker_key"] for row in rows}),
        "formal_base_audio_collisions": 0,
        "formal_base_transcript_collisions": 0,
        "canonical_resample_geometry_corrections": sum(
            int(
                (ledger_by_hash.get(str(source["source_audio_sha256"])) or source)[
                    "model_num_samples"
                ]
            )
            != int(canonical["model_num_samples"])
            for source, canonical in zip(selected, rows)
        ),
        "registry": str(output),
        "registry_sha256": sha256_file(output),
    }
    atomic_json(output.with_name("summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
