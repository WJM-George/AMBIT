#!/usr/bin/env python3
"""Extract leak-safe FSDKaggle2019 noisy-train Sound sources for ScenePlan.

The curated split is deliberately excluded: every curated Freesound parent is
already present in FSD50K, so treating those files as new would inflate source
uniqueness without adding independent recordings.  The noisy split uses unique
Flickr parent videos.  Music, singing, and linguistic-speech labels are removed;
the remaining clips are normalized to mono PCM16/48 kHz and, when necessary,
deterministically cropped to at most ten seconds.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf


MODEL_SAMPLE_RATE = 44_100
MAX_MODEL_SAMPLES = 442_368
REQUIRED_TAIL_SAMPLES = 40
OUTPUT_SAMPLE_RATE = 48_000
MAX_SOURCE_SECONDS = 10.0
MIN_RMS = 1.0e-5
MIN_PEAK = 1.0e-4

MUSIC_OR_SINGING = {
    "Accordion", "Acoustic_guitar", "Bass_drum", "Bass_guitar",
    "Electric_guitar", "Female_singing", "Glockenspiel", "Gong",
    "Harmonica", "Hi-hat", "Male_singing", "Marimba_and_xylophone",
    "Strum",
}
LINGUISTIC_SPEECH = {
    "Child_speech_and_kid_speaking",
    "Female_speech_and_woman_speaking",
    "Male_speech_and_man_speaking",
    "Whispering",
}


def canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(canonical_json(row) + "\n" for row in rows))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_audio(path: Path) -> bool:
    try:
        info = sf.info(str(path))
        return (
            int(info.channels) == 1
            and int(info.samplerate) == OUTPUT_SAMPLE_RATE
            and int(info.frames) > 0
        )
    except Exception:
        return False


def deterministic_crop_start(parent_id: str, available: int) -> int:
    if available <= 0:
        return 0
    value = int(hashlib.sha256(f"sceneplan-sound-expansion-v1\0{parent_id}".encode()).hexdigest()[:16], 16)
    return value % (available + 1)


def convert_one(task: tuple[dict[str, Any], str, str]) -> dict[str, Any]:
    row, source_text, target_text = task
    source, target = Path(source_text), Path(target_text)
    try:
        info = sf.info(str(source))
        native_rate = int(info.samplerate)
        native_frames = int(info.frames)
        wanted_native = min(native_frames, round(MAX_SOURCE_SECONDS * native_rate))
        crop_start = deterministic_crop_start(
            str(row["parent_asset_id"]), max(0, native_frames - wanted_native)
        )
        crop_duration = wanted_native / native_rate
        temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.wav")
        temporary.unlink(missing_ok=True)
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-ss", f"{crop_start / native_rate:.9f}", "-i", str(source),
            "-t", f"{crop_duration:.9f}", "-map", "0:a:0",
            "-ac", "1", "-ar", str(OUTPUT_SAMPLE_RATE),
            "-c:a", "pcm_s16le", str(temporary),
        ]
        result = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        if result.returncode or not valid_audio(temporary):
            temporary.unlink(missing_ok=True)
            return {**row, "conversion_status": "FAIL", "conversion_error": result.stderr[-1000:]}
        os.replace(temporary, target)
        return {
            **row,
            "conversion_status": "PASS",
            "audio_path": str(target),
            "parent_native_sample_rate_hz": native_rate,
            "parent_native_num_samples": native_frames,
            "parent_start_sample": crop_start,
            "parent_end_sample": crop_start + wanted_native,
        }
    except Exception as error:  # noqa: BLE001
        return {**row, "conversion_status": "FAIL", "conversion_error": f"{type(error).__name__}: {error}"}


def qc_one(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    reasons: list[str] = []
    try:
        path = Path(str(row["audio_path"]))
        blob = path.read_bytes()
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        frames, channels = audio.shape
        mono = audio[:, 0] if channels == 1 else np.zeros(1, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
        peak = float(np.max(np.abs(mono)))
        model_frames = math.ceil(frames * MODEL_SAMPLE_RATE / int(rate))
        if channels != 1:
            reasons.append("not_mono")
        if int(rate) != OUTPUT_SAMPLE_RATE:
            reasons.append("wrong_sample_rate")
        if not bool(np.isfinite(audio).all()):
            reasons.append("non_finite")
        if rms < MIN_RMS:
            reasons.append("signal_rms_too_low")
        if peak < MIN_PEAK:
            reasons.append("signal_peak_too_low")
        if model_frames + REQUIRED_TAIL_SAMPLES > MAX_MODEL_SAMPLES:
            reasons.append("complete_source_plus_pyroom_delay_over_limit")
        output.update(
            {
                "source_audio_sha256": hashlib.sha256(blob).hexdigest(),
                "native_sample_rate_hz": int(rate),
                "native_num_samples": int(frames),
                "native_channels": int(channels),
                "model_num_samples": int(model_frames),
                "duration_sec": float(frames / rate),
                "file_num_bytes": len(blob),
                "signal_rms": rms,
                "signal_peak": peak,
            }
        )
    except Exception as error:  # noqa: BLE001
        reasons.append(f"decode_or_checksum_error:{type(error).__name__}")
    output["qc_status"] = "PASS" if not reasons else "FAIL"
    output["qc_exclusion_reasons"] = sorted(set(reasons))
    return output


def load_hashes(path: Path) -> set[str]:
    result: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["source_audio_sha256", "eligible"], batch_size=65_536):
        for digest, eligible in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            if eligible and digest:
                result.add(str(digest))
    return result


def load_eval_hashes(root: Path) -> set[str]:
    result: set[str] = set()
    for name in ("candidate_pool.jsonl", "candidate_exclusions.jsonl"):
        path = root / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip() and (digest := json.loads(line).get("audio_sha256")):
                    result.add(str(digest))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-root", type=Path,
        default=Path("/mnt/sdd/audio_dataset/datasets/fsdkaggle2019/archives"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--base-signal-catalog", type=Path,
        default=Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/source_catalog/nonspeech/nonspeech_signal_catalog.parquet"),
    )
    parser.add_argument(
        "--external-manifest-root", type=Path,
        default=Path("/mnt/sdb/audio_dataset/evaluation_benchmark/p10_evaluation_benchmark_v1/manifests"),
    )
    parser.add_argument("--jobs", type=int, default=min(48, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    archive_root = args.archive_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    try:
        output_root.relative_to(Path("/mnt/sdb"))
    except ValueError as error:
        raise ValueError("Sound expansion outputs must live on /mnt/sdb") from error
    output_root.mkdir(parents=True, exist_ok=True)
    completed_summary = output_root / "SUMMARY.json"
    if completed_summary.is_file() and (output_root / "qc_passed.jsonl").is_file():
        prior = json.loads(completed_summary.read_text(encoding="utf-8"))
        if prior.get("status") == "PASS":
            print(json.dumps(prior, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
    meta_zip = archive_root / "FSDKaggle2019.meta.zip"
    noisy_zip = archive_root / "FSDKaggle2019.audio_train_noisy.zip"
    with zipfile.ZipFile(meta_zip) as archive:
        rows = list(
            csv.DictReader(
                io.TextIOWrapper(
                    archive.open("FSDKaggle2019.meta/train_noisy_post_competition.csv"),
                    encoding="utf-8",
                )
            )
        )
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for raw in rows:
        labels = {value.strip() for value in str(raw["labels"]).split(",") if value.strip()}
        reasons = []
        if labels & MUSIC_OR_SINGING:
            reasons.append("music_or_singing_label")
        if labels & LINGUISTIC_SPEECH:
            reasons.append("linguistic_speech_label")
        fname = str(raw["fname"])
        parent = str(raw["flickr_video_URL"])
        record = {
            "schema": "stable_audio_tools.sound_expansion_candidate",
            "schema_version": 1,
            "candidate_id": f"fsdkaggle2019:noisy:{Path(fname).stem}",
            "asset_id": f"sound:fsdkaggle2019:noisy_{Path(fname).stem}",
            "source_dataset": "fsdkaggle2019",
            "official_split": "train",
            "parent_dataset": "flickr",
            "parent_asset_id": parent,
            "member_path": f"FSDKaggle2019.audio_train_noisy/{fname}",
            "label": ", ".join(sorted(labels)),
            "labels": sorted(labels),
            "license": str(raw.get("license") or ""),
            "modality": "sound",
            "selection_rank": hashlib.sha256(
                f"sceneplan-sound-expansion-v1\0fsdkaggle2019\0{parent}".encode()
            ).hexdigest(),
            "lineage_gate_status": "pass" if not reasons else "fail",
            "lineage_exclusion_reasons": reasons,
            "exact_hash_gate_status": "pending",
        }
        (exclusions if reasons else selected).append(record)
    if len({row["parent_asset_id"] for row in selected}) != len(selected):
        raise RuntimeError("FSDKaggle noisy parent IDs are not unique")
    selected.sort(key=lambda row: (row["selection_rank"], row["candidate_id"]))
    extraction_list = output_root / "selected_archive_members.txt"
    atomic_text(extraction_list, "".join(row["member_path"] + "\n" for row in selected))
    temporary_root = output_root / "temporary_extracted"
    extracted_root = temporary_root / "FSDKaggle2019.audio_train_noisy"
    audio_root = output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    conversion_receipt = output_root / "conversion_receipt.jsonl"
    prior_conversions: dict[str, dict[str, Any]] = {}
    if conversion_receipt.is_file():
        with conversion_receipt.open(encoding="utf-8") as handle:
            prior_conversions = {
                row["candidate_id"]: row
                for line in handle
                if line.strip() and (row := json.loads(line))
            }
    missing = [
        row for row in selected
        if (
            not valid_audio(audio_root / f"{Path(row['member_path']).stem}.wav")
            or row["candidate_id"] not in prior_conversions
        )
    ]
    if missing:
        temporary_root.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["7z", "x", "-y", f"-o{temporary_root}", str(noisy_zip), f"@{extraction_list}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        atomic_text(output_root / "7z_extract.log", result.stdout)
        if result.returncode:
            raise RuntimeError(f"7z extraction failed with code {result.returncode}")
    tasks = []
    ready_rows = []
    for row in selected:
        target = audio_root / f"{Path(row['member_path']).stem}.wav"
        prior = prior_conversions.get(row["candidate_id"])
        if valid_audio(target) and prior is not None:
            ready_rows.append({**prior, **row, "conversion_status": "PASS", "audio_path": str(target)})
        else:
            tasks.append((row, str(temporary_root / row["member_path"]), str(target)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        converted = list(pool.map(convert_one, tasks, chunksize=8))
    converted_by_id = {row["candidate_id"]: row for row in ready_rows + converted}
    atomic_jsonl(
        conversion_receipt,
        sorted(converted_by_id.values(), key=lambda row: (row["selection_rank"], row["candidate_id"])),
    )
    audited_input = [converted_by_id[row["candidate_id"]] for row in selected if converted_by_id[row["candidate_id"]].get("conversion_status") == "PASS"]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        audited = list(pool.map(qc_one, audited_input, chunksize=8))
    base_hashes = load_hashes(args.base_signal_catalog.expanduser().resolve(strict=True))
    eval_hashes = load_eval_hashes(args.external_manifest_root.expanduser().resolve(strict=True))
    seen_hashes: set[str] = set()
    passed: list[dict[str, Any]] = []
    qc_exclusions: list[dict[str, Any]] = []
    for row in audited:
        reasons = list(row.get("qc_exclusion_reasons") or ())
        digest = str(row.get("source_audio_sha256") or "")
        if digest in base_hashes:
            reasons.append("exact_file_sha256_in_base_catalog")
        if digest in eval_hashes:
            reasons.append("exact_file_sha256_in_external_benchmark")
        if digest in seen_hashes:
            reasons.append("duplicate_exact_file_sha256_in_expansion")
        if not reasons and digest:
            seen_hashes.add(digest)
            row["qc_status"] = "PASS"
            row["exact_hash_gate_status"] = "pass"
            passed.append(row)
        else:
            row["qc_status"] = "FAIL"
            row["exact_hash_gate_status"] = "fail"
            row["qc_exclusion_reasons"] = sorted(set(reasons))
            qc_exclusions.append(row)
    passed.sort(key=lambda row: (row["selection_rank"], row["candidate_id"]))
    atomic_jsonl(output_root / "candidate_metadata_exclusions.jsonl", exclusions)
    atomic_jsonl(output_root / "qc_passed.jsonl", passed)
    atomic_jsonl(output_root / "qc_exclusions.jsonl", qc_exclusions)
    shutil.rmtree(temporary_root, ignore_errors=True)
    summary = {
        "schema": "stable_audio_tools.fsdkaggle2019_sound_expansion",
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "counts": {
            "metadata_rows": len(rows),
            "metadata_sound_candidates": len(selected),
            "metadata_excluded": len(exclusions),
            "converted_or_resumed": len(audited_input),
            "qc_passed_unique": len(passed),
            "qc_excluded": len(qc_exclusions),
        },
        "contracts": {
            "official_noisy_train_only": True,
            "curated_freesound_overlap_excluded": True,
            "music_singing_and_linguistic_speech_excluded": True,
            "unique_flickr_parent_per_candidate": True,
            "mono_pcm16_48khz": True,
            "deterministic_max_10s_parent_crop": True,
            "base_and_external_exact_hash_disjoint": True,
        },
        "artifacts": {
            "audio_root": str(audio_root),
            "qc_passed": str(output_root / "qc_passed.jsonl"),
            "qc_exclusions": str(output_root / "qc_exclusions.jsonl"),
        },
    }
    atomic_json(output_root / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
