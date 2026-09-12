#!/usr/bin/env python3
"""Build leak-auditable candidates for the P10 evaluation benchmarks.

This script intentionally separates three states:

1. lineage/exact-hash-clean candidates;
2. candidates awaiting the acoustic fingerprint gate;
3. the final frozen core set (created by a separate finalizer).

It never treats an official split name by itself as proof of train/test
separation.  AudioCaps and VGGSound are also cross-checked against the YouTube
lineage used through AudioSet, MusicCaps, and each other.  FSD50K eval is
checked against the exact set of assets referenced by P10 train.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf


SCHEMA = "sceneplan_foa.p10_cross_system_candidate_pool"
SCHEMA_VERSION = 1
CLIP_DURATION_SEC = 10.0

SPEECH_TERMS = {
    "babbling",
    "chatter",
    "child speech",
    "conversation",
    "female speech",
    "male speech",
    "narration",
    "parrot talking",
    "people babbling",
    "people whispering",
    "police radio chatter",
    "rapping",
    "speech",
    "talking",
    "whispering",
}

VOCAL_TERMS = {
    "beat boxing",
    "child singing",
    "choir",
    "female singing",
    "humming",
    "male singing",
    "rapping",
    "singing",
    "vocal",
    "yodelling",
}

MUSIC_INSTRUMENT_TERMS = {
    "accordion",
    "acoustic guitar",
    "bagpipes",
    "banjo",
    "bass drum",
    "bass guitar",
    "bassoon",
    "bongo",
    "bowed string",
    "brass instrument",
    "bugle",
    "castanets",
    "cello",
    "clarinet",
    "congas",
    "cornet",
    "cymbal",
    "didgeridoo",
    "djembe",
    "double bass",
    "drum",
    "electric guitar",
    "electronic organ",
    "erhu",
    "flute",
    "french horn",
    "glockenspiel",
    "gong",
    "guitar",
    "guiro",
    "harmonica",
    "hammond organ",
    "harp",
    "harpsichord",
    "hi-hat",
    "keyboard (musical)",
    "mallet percussion",
    "mandolin",
    "marimba",
    "musical instrument",
    "oboe",
    "orchestra",
    "organ",
    "percussion",
    "piano",
    "saxophone",
    "shofar",
    "singing bowl",
    "sitar",
    "snare drum",
    "steel guitar",
    "steelpan",
    "synthesizer",
    "tabla",
    "tambourine",
    "tapping guitar",
    "theremin",
    "timbales",
    "timpani",
    "tympani",
    "trombone",
    "trumpet",
    "tuning fork",
    "ukulele",
    "vibraphone",
    "violin",
    "washboard",
    "wind instrument",
    "xylophone",
    "zither",
}


@dataclass(frozen=True)
class Interval:
    start: float
    end: float
    asset_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/sdb/audio_dataset/evaluation_benchmark/"
            "p10_evaluation_benchmark_v1"
        ),
    )
    parser.add_argument(
        "--sceneplan-root",
        type=Path,
        default=Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m"),
    )
    parser.add_argument(
        "--audiocaps-root",
        type=Path,
        default=Path(
            "/mnt/sdb/audio_dataset/evaluation_benchmark/"
            "audiocaps_test_b29b3243/raw_parquet/test"
        ),
    )
    parser.add_argument(
        "--audiocaps-full-root",
        type=Path,
        default=Path("/mnt/sdd/audio_dataset/datasets/audiocaps/snapshot/data"),
    )
    parser.add_argument(
        "--vggsound-csv",
        type=Path,
        default=Path(
            "/mnt/sdc/audio_dataset/datasets/vggsound/snapshot/vggsound.csv"
        ),
    )
    parser.add_argument(
        "--vggsound-audio-root",
        type=Path,
        default=Path("/mnt/sdc/audio_dataset/datasets/vggsound/extracted/audio"),
    )
    parser.add_argument(
        "--fsd50k-root",
        type=Path,
        default=Path("/mnt/sdd/audio_dataset/datasets/FSD50k"),
    )
    parser.add_argument(
        "--musiccaps-csv",
        type=Path,
        default=Path(
            "/mnt/sdd/audio_dataset/datasets/musiccaps/snapshot/"
            "musiccaps-public.csv"
        ),
    )
    parser.add_argument(
        "--musiccaps-audio-root",
        type=Path,
        default=Path("/mnt/sdd/audio_dataset/datasets/musiccaps/audio"),
    )
    parser.add_argument("--hash-workers", type=int, default=32)
    return parser.parse_args()


def canonical_json(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_rank(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(canonical_json(record))
            handle.write("\n")
    os.replace(tmp, path)


def write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    keys = sorted({key for record in records for key in record})
    normalized = [{key: record.get(key) for key in keys} for record in records]
    table = pa.Table.from_pylist(normalized)
    pq.write_table(table, tmp, compression="zstd", compression_level=9)
    os.replace(tmp, path)


def iter_index_train_assets(index_path: Path) -> set[str]:
    result: set[str] = set()
    parquet = pq.ParquetFile(index_path)
    for batch in parquet.iter_batches(
        columns=["split", "source_asset_ids"], batch_size=16_384
    ):
        for split, assets in zip(
            batch.column("split").to_pylist(),
            batch.column("source_asset_ids").to_pylist(),
        ):
            if split == "train":
                result.update(assets or [])
    return result


def parse_vgg_asset_id(asset_id: str) -> tuple[str, int] | None:
    match = re.fullmatch(
        r"sound:vggsound:vggsound_(.{11})_(\d{6})", asset_id
    )
    if not match:
        return None
    return match.group(1), int(match.group(2))


def parse_audiocaps_asset_id(asset_id: str) -> int | None:
    match = re.fullmatch(r"sound:audiocaps:audiocaps_(\d+)", asset_id)
    return int(match.group(1)) if match else None


def parse_youtube_asset_id(asset_id: str) -> str | None:
    prefixes = (
        "sound:audioset:audioset_",
        "music:musiccaps:musiccaps_",
    )
    for prefix in prefixes:
        if asset_id.startswith(prefix):
            return asset_id[len(prefix) :]
    return None


def load_audiocaps_metadata(
    files: Iterable[Path], *, with_locator: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    columns = ["audiocap_id", "youtube_id", "start_time", "caption", "audio_length"]
    for path in sorted(files):
        table = pq.read_table(path, columns=columns)
        for row_index, row in enumerate(table.to_pylist()):
            if with_locator:
                row["parquet_path"] = str(path)
                row["row_index"] = row_index
            rows.append(row)
    return rows


def add_interval(
    index: dict[str, list[Interval]],
    youtube_id: str,
    start: float,
    asset_id: str,
) -> None:
    index[youtube_id].append(
        Interval(start=float(start), end=float(start) + CLIP_DURATION_SEC, asset_id=asset_id)
    )


def overlapping_assets(
    index: dict[str, list[Interval]], youtube_id: str, start: float, end: float
) -> list[str]:
    return [
        item.asset_id
        for item in index.get(youtube_id, ())
        if max(float(start), item.start) < min(float(end), item.end)
    ]


def load_train_lineage(
    sceneplan_root: Path, audiocaps_full_root: Path
) -> tuple[
    set[str],
    set[str],
    dict[str, list[Interval]],
    dict[str, list[Interval]],
    set[str],
]:
    train_assets = iter_index_train_assets(sceneplan_root / "sceneplans_model_v1/index.parquet")
    youtube_whole_clip: set[str] = set()
    vgg_intervals: dict[str, list[Interval]] = defaultdict(list)
    audiocaps_intervals: dict[str, list[Interval]] = defaultdict(list)

    used_audiocap_ids: dict[int, str] = {}
    for asset_id in train_assets:
        if youtube_id := parse_youtube_asset_id(asset_id):
            youtube_whole_clip.add(youtube_id)
            continue
        if parsed := parse_vgg_asset_id(asset_id):
            youtube_id, start = parsed
            add_interval(vgg_intervals, youtube_id, start, asset_id)
            continue
        if (audiocap_id := parse_audiocaps_asset_id(asset_id)) is not None:
            used_audiocap_ids[audiocap_id] = asset_id

    train_parquets = sorted(audiocaps_full_root.glob("train-*.parquet"))
    for row in load_audiocaps_metadata(train_parquets, with_locator=False):
        audiocap_id = int(row["audiocap_id"])
        if audiocap_id not in used_audiocap_ids:
            continue
        add_interval(
            audiocaps_intervals,
            str(row["youtube_id"]),
            float(row["start_time"]),
            used_audiocap_ids[audiocap_id],
        )

    catalog_path = (
        sceneplan_root / "source_catalog/nonspeech/nonspeech_signal_catalog.parquet"
    )
    train_hashes: set[str] = set()
    catalog = pq.ParquetFile(catalog_path)
    for batch in catalog.iter_batches(
        columns=["asset_id", "source_audio_sha256"], batch_size=32_768
    ):
        for asset_id, audio_hash in zip(
            batch.column("asset_id").to_pylist(),
            batch.column("source_audio_sha256").to_pylist(),
        ):
            if asset_id in train_assets and audio_hash:
                train_hashes.add(str(audio_hash))

    for values in vgg_intervals.values():
        values.sort(key=lambda item: item.start)
    for values in audiocaps_intervals.values():
        values.sort(key=lambda item: item.start)
    return (
        train_assets,
        youtube_whole_clip,
        vgg_intervals,
        audiocaps_intervals,
        train_hashes,
    )


def contains_term(text: str, terms: set[str]) -> bool:
    normalized = text.lower().replace("_", " ")
    return any(term in normalized for term in terms)


def classify_audiocaps(captions: list[str]) -> str | None:
    text = " ".join(captions).lower()
    if contains_term(text, SPEECH_TERMS | VOCAL_TERMS):
        return None
    if contains_term(text, MUSIC_INSTRUMENT_TERMS | {"melody", "music", "song", "tune"}):
        return "music"
    return "sound"


def classify_vggsound(label: str) -> str | None:
    normalized = label.lower().strip()
    if contains_term(normalized, SPEECH_TERMS | VOCAL_TERMS):
        return None
    if normalized == "orchestra" or contains_term(normalized, MUSIC_INSTRUMENT_TERMS):
        return "music"
    return "sound"


def classify_fsd50k(labels: list[str]) -> str | None:
    normalized = {label.replace("_", " ").lower() for label in labels}
    if any(contains_term(label, SPEECH_TERMS | VOCAL_TERMS) for label in normalized):
        return None
    if "music" in normalized or "musical instrument" in normalized:
        return "music"
    return "sound"


def extract_audiocaps_audio(
    candidates: list[dict[str, Any]], output_root: Path
) -> None:
    wanted: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        wanted[row["source_parquet_path"]].append(row)
    audio_root = output_root / "staging/audiocaps_lineage_clean_audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    for parquet_path, selected in sorted(wanted.items()):
        audio_rows = pq.read_table(parquet_path, columns=["audio"]).column("audio").to_pylist()
        for row in selected:
            payload = audio_rows[int(row["source_parquet_row"])]["bytes"]
            target = audio_root / f"{row['external_clip_id']}.wav"
            digest = hashlib.sha256(payload).hexdigest()
            if target.exists() and sha256_file(target) != digest:
                raise RuntimeError(f"existing AudioCaps extraction has wrong hash: {target}")
            if not target.exists():
                tmp = target.with_suffix(".wav.tmp")
                with tmp.open("wb") as handle:
                    handle.write(payload)
                os.replace(tmp, target)
            row["audio_path"] = str(target)
            row["audio_sha256"] = digest


def build_audiocaps_candidates(
    args: argparse.Namespace,
    youtube_whole_clip: set[str],
    vgg_intervals: dict[str, list[Interval]],
    audiocaps_intervals: dict[str, list[Interval]],
    train_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_files = sorted(args.audiocaps_root.glob("*.parquet"))
    raw_rows = load_audiocaps_metadata(test_files, with_locator=True)
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in raw_rows:
        key = (str(row["youtube_id"]), int(row["start_time"]))
        clip = grouped.setdefault(
            key,
            {
                "youtube_id": key[0],
                "start_time": key[1],
                "captions": [],
                "audiocap_ids": [],
                "source_parquet_path": row["parquet_path"],
                "source_parquet_row": int(row["row_index"]),
                "audio_length": int(row["audio_length"]),
            },
        )
        clip["captions"].append(str(row["caption"]))
        clip["audiocap_ids"].append(int(row["audiocap_id"]))

    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for (youtube_id, start_time), clip in sorted(grouped.items()):
        reasons: list[str] = []
        matched_train_assets: set[str] = set()
        if youtube_id in youtube_whole_clip:
            reasons.append("youtube_id_used_by_audioset_or_musiccaps_train")
        for label, intervals in (
            ("overlapping_vggsound_train_interval", vgg_intervals),
            ("overlapping_audiocaps_train_interval", audiocaps_intervals),
        ):
            matched = overlapping_assets(
                intervals, youtube_id, start_time, start_time + CLIP_DURATION_SEC
            )
            if matched:
                reasons.append(label)
                matched_train_assets.update(matched)
        captions = sorted(set(clip["captions"]))
        modality = classify_audiocaps(captions)
        if modality is None:
            reasons.append("unsupported_or_spoken_language_without_exact_transcript")
        record = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "candidate_id": f"audiocaps:{youtube_id}:{start_time}",
            "source_dataset": "audiocaps",
            "official_split": "test",
            "external_clip_id": f"{youtube_id}_{start_time:06d}",
            "youtube_id": youtube_id,
            "start_time_sec": float(start_time),
            "duration_sec": CLIP_DURATION_SEC,
            "modality": modality,
            "prompt": captions[0],
            "reference_captions": captions,
            "labels": [],
            "audiocap_ids": sorted(set(clip["audiocap_ids"])),
            "source_parquet_path": clip["source_parquet_path"],
            "source_parquet_row": clip["source_parquet_row"],
            "matched_train_asset_ids": sorted(matched_train_assets),
            "lineage_gate_status": "pass" if not reasons else "fail",
            "lineage_exclusion_reasons": sorted(set(reasons)),
            "exact_hash_gate_status": "pending",
            "acoustic_fingerprint_gate_status": "pending",
            "selection_rank": stable_rank(f"audiocaps:{youtube_id}:{start_time}"),
        }
        if reasons:
            exclusions.append(record)
        else:
            candidates.append(record)

    extract_audiocaps_audio(candidates, args.output_root)
    passed: list[dict[str, Any]] = []
    for record in candidates:
        if record["audio_sha256"] in train_hashes:
            record["exact_hash_gate_status"] = "fail"
            record["lineage_exclusion_reasons"] = ["exact_file_sha256_used_by_train"]
            exclusions.append(record)
        else:
            record["exact_hash_gate_status"] = "pass"
            passed.append(record)
    return passed, exclusions


def load_vggsound_rows(csv_path: Path) -> Iterator[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) != 4:
                raise ValueError(f"unexpected VGGSound row: {row!r}")
            youtube_id, start_time, label, split = row
            yield {
                "youtube_id": youtube_id,
                "start_time": int(start_time),
                "label": label,
                "split": split,
            }


def probe_and_hash(path: Path) -> tuple[str, float, int, int]:
    info = sf.info(str(path))
    return sha256_file(path), float(info.duration), int(info.samplerate), int(info.channels)


def build_vggsound_candidates(
    args: argparse.Namespace,
    youtube_whole_clip: set[str],
    vgg_intervals: dict[str, list[Interval]],
    audiocaps_intervals: dict[str, list[Interval]],
    train_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in load_vggsound_rows(args.vggsound_csv):
        if row["split"] != "test":
            continue
        youtube_id = row["youtube_id"]
        start_time = row["start_time"]
        reasons: list[str] = []
        matched_train_assets: set[str] = set()
        if youtube_id in youtube_whole_clip:
            reasons.append("youtube_id_used_by_audioset_or_musiccaps_train")
        for label, intervals in (
            ("overlapping_vggsound_train_interval", vgg_intervals),
            ("overlapping_audiocaps_train_interval", audiocaps_intervals),
        ):
            matched = overlapping_assets(
                intervals, youtube_id, start_time, start_time + CLIP_DURATION_SEC
            )
            if matched:
                reasons.append(label)
                matched_train_assets.update(matched)
        modality = classify_vggsound(row["label"])
        if modality is None:
            reasons.append("unsupported_or_spoken_language_without_exact_transcript")
        audio_path = args.vggsound_audio_root / f"{youtube_id}_{start_time:06d}.wav"
        if not audio_path.is_file():
            reasons.append("audio_unavailable_in_local_snapshot")
        record = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "candidate_id": f"vggsound:{youtube_id}:{start_time}",
            "source_dataset": "vggsound",
            "official_split": "test",
            "external_clip_id": f"{youtube_id}_{start_time:06d}",
            "youtube_id": youtube_id,
            "start_time_sec": float(start_time),
            "duration_sec": CLIP_DURATION_SEC,
            "modality": modality,
            "prompt": f"The sound of {row['label']}.",
            "reference_captions": [],
            "labels": [row["label"]],
            "audiocap_ids": [],
            "audio_path": str(audio_path),
            "audio_sha256": None,
            "sample_rate_hz": None,
            "num_channels": None,
            "matched_train_asset_ids": sorted(matched_train_assets),
            "lineage_gate_status": "pass" if not reasons else "fail",
            "lineage_exclusion_reasons": sorted(set(reasons)),
            "exact_hash_gate_status": "pending",
            "acoustic_fingerprint_gate_status": "pending",
            "selection_rank": stable_rank(f"vggsound:{youtube_id}:{start_time}"),
        }
        if reasons:
            exclusions.append(record)
        else:
            candidates.append(record)

    with ThreadPoolExecutor(max_workers=args.hash_workers) as pool:
        results = pool.map(
            probe_and_hash, (Path(record["audio_path"]) for record in candidates)
        )
        for record, (digest, duration, sample_rate, channels) in zip(candidates, results):
            record["audio_sha256"] = digest
            record["duration_sec"] = duration
            record["sample_rate_hz"] = sample_rate
            record["num_channels"] = channels

    passed: list[dict[str, Any]] = []
    for record in candidates:
        if record["audio_sha256"] in train_hashes:
            record["exact_hash_gate_status"] = "fail"
            record["lineage_exclusion_reasons"] = ["exact_file_sha256_used_by_train"]
            exclusions.append(record)
        else:
            record["exact_hash_gate_status"] = "pass"
            passed.append(record)
    return passed, exclusions


def load_fsd_labels(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "fname": str(row["fname"]),
                    "labels": str(row["labels"]).split(","),
                    "mids": str(row["mids"]).split(","),
                }
            )
    return rows


def clean_fsd_prompt(labels: list[str]) -> str:
    specific = [
        label.replace("_", " ")
        for label in labels
        if label not in {"Music", "Musical_instrument"}
    ]
    chosen = specific[:3] or [label.replace("_", " ") for label in labels[:3]]
    return "An audio recording of " + ", ".join(chosen).lower() + "."


def build_fsd50k_candidates(
    args: argparse.Namespace,
    train_assets: set[str],
    train_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = load_fsd_labels(args.fsd50k_root / "labels/eval.csv")
    metadata = json.loads(
        (args.fsd50k_root / "metadata/eval_clips_info_FSD50K.json").read_text(
            encoding="utf-8"
        )
    )
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for row in labels:
        clip_id = row["fname"]
        reasons: list[str] = []
        modality = classify_fsd50k(row["labels"])
        if modality is None:
            reasons.append("unsupported_or_spoken_language_without_exact_transcript")
        direct_asset_id = f"sound:fsd50k:fsd50k_eval_{clip_id}"
        if direct_asset_id in train_assets:
            reasons.append("fsd50k_eval_asset_directly_used_by_train")
        audio_path = args.fsd50k_root / "clips/eval" / f"{clip_id}.wav"
        if not audio_path.is_file():
            reasons.append("audio_unavailable_in_local_snapshot")
        info = metadata.get(clip_id, metadata.get(str(int(clip_id)), {}))
        record = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "candidate_id": f"fsd50k:{clip_id}",
            "source_dataset": "fsd50k",
            "official_split": "eval",
            "external_clip_id": clip_id,
            "youtube_id": None,
            "start_time_sec": None,
            "duration_sec": None,
            "modality": modality,
            "prompt": clean_fsd_prompt(row["labels"]),
            "reference_captions": [],
            "labels": row["labels"],
            "mids": row["mids"],
            "raw_title": str(info.get("title", "")),
            "raw_description": str(info.get("description", "")),
            "license": str(info.get("license", "")),
            "uploader": str(info.get("uploader", "")),
            "audiocap_ids": [],
            "audio_path": str(audio_path),
            "audio_sha256": None,
            "sample_rate_hz": None,
            "num_channels": None,
            "matched_train_asset_ids": [],
            "lineage_gate_status": "pass" if not reasons else "fail",
            "lineage_exclusion_reasons": sorted(set(reasons)),
            "exact_hash_gate_status": "pending",
            "acoustic_fingerprint_gate_status": "pending",
            "selection_rank": stable_rank(f"fsd50k:{clip_id}"),
        }
        if reasons:
            exclusions.append(record)
        else:
            candidates.append(record)

    with ThreadPoolExecutor(max_workers=args.hash_workers) as pool:
        results = pool.map(
            probe_and_hash, (Path(record["audio_path"]) for record in candidates)
        )
        for record, (digest, duration, sample_rate, channels) in zip(candidates, results):
            record["audio_sha256"] = digest
            record["duration_sec"] = duration
            record["sample_rate_hz"] = sample_rate
            record["num_channels"] = channels

    passed: list[dict[str, Any]] = []
    for record in candidates:
        reasons: list[str] = []
        if record["audio_sha256"] in train_hashes:
            reasons.append("exact_file_sha256_used_by_train")
        if not (3.0 <= float(record["duration_sec"]) <= 10.0):
            reasons.append("duration_outside_core_3_to_10_sec")
        if reasons:
            record["exact_hash_gate_status"] = (
                "fail" if "exact_file_sha256_used_by_train" in reasons else "pass"
            )
            record["lineage_exclusion_reasons"] = reasons
            exclusions.append(record)
        else:
            record["exact_hash_gate_status"] = "pass"
            passed.append(record)
    return passed, exclusions


MUSICCAPS_INSTRUMENTAL_TERMS = re.compile(
    r"\b(instrumental|no voice|no vocals|without vocals|without a vocal|no singing)\b",
    re.IGNORECASE,
)
MUSICCAPS_VOCAL_TERMS = re.compile(
    r"\b(vocal|vocals|voice|voices|sing|sings|singer|singers|singing|rap|rapper|"
    r"rapping|speech|spoken|chant|chanting|choir|lyrics?|a cappella|acapella)\b",
    re.IGNORECASE,
)
MUSICCAPS_NEGATED_VOCAL_TERMS = re.compile(
    r"\b(no|without)(?:\s+\w+){0,2}\s+(voice|voices|vocal|vocals|singing)\b",
    re.IGNORECASE,
)


def is_strict_instrumental_musiccaps(row: dict[str, str]) -> bool:
    text = f"{row['caption']} {row['aspect_list']}"
    without_negated_vocals = MUSICCAPS_NEGATED_VOCAL_TERMS.sub("", text)
    return bool(MUSICCAPS_INSTRUMENTAL_TERMS.search(text)) and not bool(
        MUSICCAPS_VOCAL_TERMS.search(without_negated_vocals)
    )


def parse_musiccaps_aspects(raw: str) -> list[str]:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def build_musiccaps_candidates(
    args: argparse.Namespace,
    youtube_whole_clip: set[str],
    vgg_intervals: dict[str, list[Interval]],
    audiocaps_intervals: dict[str, list[Interval]],
    train_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    with args.musiccaps_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        youtube_id = str(row["ytid"])
        start_time = int(row["start_s"])
        end_time = int(row["end_s"])
        reasons: list[str] = []
        matched_train_assets: set[str] = set()
        if not is_strict_instrumental_musiccaps(row):
            reasons.append("not_strictly_instrumental_or_contains_vocal_language")
        if youtube_id in youtube_whole_clip:
            reasons.append("youtube_id_used_by_audioset_or_musiccaps_train")
        for label, intervals in (
            ("overlapping_vggsound_train_interval", vgg_intervals),
            ("overlapping_audiocaps_train_interval", audiocaps_intervals),
        ):
            matched = overlapping_assets(intervals, youtube_id, start_time, end_time)
            if matched:
                reasons.append(label)
                matched_train_assets.update(matched)
        audio_path = args.musiccaps_audio_root / f"{youtube_id}.wav"
        if not audio_path.is_file():
            reasons.append("audio_unavailable_in_local_snapshot")
        caption = str(row["caption"]).strip()
        aspects = parse_musiccaps_aspects(str(row["aspect_list"]))
        record = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "candidate_id": f"musiccaps:{youtube_id}:{start_time}",
            "source_dataset": "musiccaps",
            "official_split": "public_instrumental_subset",
            "external_clip_id": f"{youtube_id}_{start_time:06d}",
            "youtube_id": youtube_id,
            "start_time_sec": float(start_time),
            "duration_sec": float(end_time - start_time),
            "modality": "music",
            "prompt": caption,
            "reference_captions": [caption],
            "labels": aspects,
            "audiocap_ids": [],
            "is_audioset_eval": str(row["is_audioset_eval"]).lower() == "true",
            "audio_path": str(audio_path),
            "audio_sha256": None,
            "sample_rate_hz": None,
            "num_channels": None,
            "matched_train_asset_ids": sorted(matched_train_assets),
            "lineage_gate_status": "pass" if not reasons else "fail",
            "lineage_exclusion_reasons": sorted(set(reasons)),
            "exact_hash_gate_status": "pending",
            "acoustic_fingerprint_gate_status": "pending",
            "selection_rank": stable_rank(f"musiccaps:{youtube_id}:{start_time}"),
        }
        if reasons:
            exclusions.append(record)
        else:
            candidates.append(record)

    with ThreadPoolExecutor(max_workers=args.hash_workers) as pool:
        results = pool.map(
            probe_and_hash, (Path(record["audio_path"]) for record in candidates)
        )
        for record, (digest, duration, sample_rate, channels) in zip(candidates, results):
            record["audio_sha256"] = digest
            record["duration_sec"] = duration
            record["sample_rate_hz"] = sample_rate
            record["num_channels"] = channels

    passed: list[dict[str, Any]] = []
    for record in candidates:
        reasons: list[str] = []
        if record["audio_sha256"] in train_hashes:
            reasons.append("exact_file_sha256_used_by_train")
        if not (8.0 <= float(record["duration_sec"]) <= 10.1):
            reasons.append("duration_outside_nominal_musiccaps_8_to_10p1_sec")
        if reasons:
            record["exact_hash_gate_status"] = (
                "fail" if "exact_file_sha256_used_by_train" in reasons else "pass"
            )
            record["lineage_exclusion_reasons"] = reasons
            exclusions.append(record)
        else:
            record["exact_hash_gate_status"] = "pass"
            passed.append(record)
    return passed, exclusions


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(
        f"{record.get('source_dataset')}:{record.get('modality')}" for record in records
    )
    return dict(sorted(counter.items()))


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (
        train_assets,
        youtube_whole_clip,
        vgg_intervals,
        audiocaps_intervals,
        train_hashes,
    ) = load_train_lineage(args.sceneplan_root, args.audiocaps_full_root)

    audiocaps, audiocaps_excluded = build_audiocaps_candidates(
        args,
        youtube_whole_clip,
        vgg_intervals,
        audiocaps_intervals,
        train_hashes,
    )
    vggsound, vggsound_excluded = build_vggsound_candidates(
        args,
        youtube_whole_clip,
        vgg_intervals,
        audiocaps_intervals,
        train_hashes,
    )
    fsd50k, fsd50k_excluded = build_fsd50k_candidates(
        args, train_assets, train_hashes
    )
    musiccaps, musiccaps_excluded = build_musiccaps_candidates(
        args,
        youtube_whole_clip,
        vgg_intervals,
        audiocaps_intervals,
        train_hashes,
    )
    candidates = sorted(
        audiocaps + vggsound + fsd50k + musiccaps,
        key=lambda item: (item["source_dataset"], item["selection_rank"]),
    )
    exclusions = sorted(
        audiocaps_excluded
        + vggsound_excluded
        + fsd50k_excluded
        + musiccaps_excluded,
        key=lambda item: (item["source_dataset"], item["selection_rank"]),
    )
    manifest_root = args.output_root / "manifests"
    write_jsonl(manifest_root / "candidate_pool.jsonl", candidates)
    write_parquet(manifest_root / "candidate_pool.parquet", candidates)
    write_jsonl(manifest_root / "candidate_exclusions.jsonl", exclusions)
    write_parquet(manifest_root / "candidate_exclusions.parquet", exclusions)

    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_pool_only_acoustic_fingerprint_gate_pending",
        "final_sets": {
            "music_sound_public_core2000": {
                "target_rows": 2000,
                "target_by_modality": {"music": 1000, "sound": 1000},
            },
        },
        "train_sceneplan_asset_count": len(train_assets),
        "train_audio_sha256_count": len(train_hashes),
        "youtube_whole_clip_train_lineage_count": len(youtube_whole_clip),
        "vggsound_train_lineage_video_count": len(vgg_intervals),
        "audiocaps_train_lineage_video_count": len(audiocaps_intervals),
        "candidate_rows": len(candidates),
        "candidate_counts": summarize(candidates),
        "excluded_rows": len(exclusions),
        "excluded_counts": summarize(exclusions),
        "gate_contract": {
            "official_split_only_is_sufficient": False,
            "lineage_gate": "complete",
            "exact_file_sha256_gate": "complete",
            "acoustic_fingerprint_gate": "pending_for_external_audio",
        },
    }
    summary_path = args.output_root / "CANDIDATE_POOL_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
