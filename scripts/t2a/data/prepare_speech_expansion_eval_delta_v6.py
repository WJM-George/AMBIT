#!/usr/bin/env python3
"""Build and audit the revision-6 validation/test coverage extension.

The frozen revision-5 validation/test rows remain an immutable legacy subset.
This module appends source-disjoint scenes so the final evaluation views cover
the same semantic families and 432/648-frame envelope as the 1.6M train view.
It deliberately does not use a word-level aligner.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time
from typing import Any
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.t2a.data.build_model_sceneplan_manifests_v1 import (  # noqa: E402
    INDEX_SCHEMA,
    atomic_jsonl,
    canonical_json,
    rounded,
    sha256_file,
    sha256_text,
)
from scripts.t2a.data.build_sceneplan_manifests_v2 import (  # noqa: E402
    ROOM_CLASSES,
    TAIL_TARGET,
    motion_plan,
)
from scripts.t2a.data.build_speech_expansion_sceneplans_v6 import (  # noqa: E402
    DurationSelector,
    LONG_CEILING,
    MODEL_SAMPLE_RATE,
    SHORT_CEILING,
    SHARD_ROWS,
    VAE_HOP_SAMPLES,
    build_scene,
    load_nonspeech,
    nonspeech_asset_ref,
    source_model_value,
)
from scripts.t2a.data.materialize_model_sceneplan_v1_shard import (  # noqa: E402
    load_shard_rows,
)
from scripts.t2a.data.plan_sceneplan_speech_expansion_noalign_15s import (  # noqa: E402
    canonical_candidate,
)
from scripts.t2a.data.render_tts_v2_pilot import room_recipe  # noqa: E402
from scripts.t2a.data.sceneplan_v2_common import (  # noqa: E402
    deterministic_digest,
    model_num_samples,
)
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_44_controls,
    compile_model_renderer_caption,
    validate_model_sceneplan,
)


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
LEDGER = DATASET_ROOT / "split_ledgers/speech_v2/speech_split_ledger.parquet"
CATALOG = DATASET_ROOT / "source_catalog/speech/catalog.sqlite"
BASE_SPEAKERS = DATASET_ROOT / (
    "source_annotations/speech_speaker_instruct_v1/registry/"
    "speech_speaker_description_registry.parquet"
)
TRAIN_DONORS = REVISION_ROOT / (
    "source_annotations/speech_speaker_instruct_v1/registry/"
    "final_speech_donors_with_speakers.parquet"
)
EVAL_SOURCES = REVISION_ROOT / "eval_sources"
CANDIDATES = EVAL_SOURCES / "candidates/eval_long_pending_qc.parquet"
QC_ROOT = EVAL_SOURCES / "qc/eval_long"
SCENEPLAN_ROOT = REVISION_ROOT / "sceneplans_model_v2_eval_delta"
MATERIALIZED_ROOT = REVISION_ROOT / "materialized_eval_delta"

SHORT_FRAMES = 432
LONG_FRAMES = 648
LONG_DRY_MAX_SEC = 14.5

# These additions turn the immutable 20k/4k legacy views into exact 32k/8k
# representatives of the final train marginals.
NO_SPEECH_SOURCE_COUNTS = {
    "validation": {1: 700, 2: 700, 3: 400, 4: 200},
    "test": {1: 350, 2: 200, 3: 300, 4: 150},
}
DIRECT_COUNTS = {
    "validation": {
        432: {"speech_only": 750, "speech_plus_music": 625, "speech_plus_sound": 625},
        648: {"speech_only": 750, "speech_plus_music": 625, "speech_plus_sound": 625},
    },
    "test": {
        432: {"speech_only": 363, "speech_plus_music": 318, "speech_plus_sound": 319},
        648: {"speech_only": 187, "speech_plus_music": 157, "speech_plus_sound": 156},
    },
}
SEQUENTIAL_COUNTS = {
    "validation": {
        "speech_then_sound": 1_500,
        "sound_then_speech": 1_500,
        "speech_then_music": 1_500,
        "music_then_speech": 1_500,
    },
    "test": {
        "speech_then_sound": 375,
        "sound_then_speech": 375,
        "speech_then_music": 375,
        "music_then_speech": 375,
    },
}
DELTA_ROWS = {"validation": 12_000, "test": 4_000}
DELTA_BUCKETS = {
    "validation": {432: 4_000, 648: 8_000},
    "test": {432: 2_000, 648: 2_000},
}
DELTA_PATTERNS = {
    "validation": {
        "no_speech_1": 700,
        "no_speech_2": 700,
        "no_speech_3": 400,
        "no_speech_4": 200,
        "direct_speech_only": 1_500,
        "direct_speech_plus_music": 1_250,
        "direct_speech_plus_sound": 1_250,
        "sequential_speech_then_sound": 1_500,
        "sequential_sound_then_speech": 1_500,
        "sequential_speech_then_music": 1_500,
        "sequential_music_then_speech": 1_500,
    },
    "test": {
        "no_speech_1": 350,
        "no_speech_2": 200,
        "no_speech_3": 300,
        "no_speech_4": 150,
        "direct_speech_only": 550,
        "direct_speech_plus_music": 475,
        "direct_speech_plus_sound": 475,
        "sequential_speech_then_sound": 375,
        "sequential_sound_then_speech": 375,
        "sequential_speech_then_music": 375,
        "sequential_music_then_speech": 375,
    },
}
DELTA_SOURCE_COUNTS = {
    "validation": {1: 2_200, 2: 9_200, 3: 400, 4: 200},
    "test": {1: 900, 2: 2_650, 3: 300, 4: 150},
}
DELTA_KIND_APPEARANCES = {
    "validation": {"music": 6_300, "sound": 6_300, "speech": 10_000},
    "test": {"music": 2_350, "sound": 2_350, "speech": 3_000},
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    if pq.read_metadata(temporary).num_rows != len(rows):
        raise RuntimeError(f"atomic parquet reopen changed row count: {path}")
    os.replace(temporary, path)


def _formal_split_speakers(ledger_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    return {
        split: {
            str(row["speaker_key"])
            for row in ledger_rows
            if str(row["pool"]) == split
        }
        for split in ("validation", "test")
    }


def plan_candidates() -> dict[str, Any]:
    ledger_rows = pq.read_table(LEDGER).to_pylist()
    speakers = _formal_split_speakers(ledger_rows)
    if speakers["validation"] & speakers["test"]:
        raise RuntimeError("legacy validation/test speakers overlap")
    occupied_audio = {
        str(row["source_audio_sha256"])
        for row in ledger_rows
        if row.get("source_audio_sha256")
    }
    occupied_text = {
        str(row["normalized_transcript_sha256"])
        for row in ledger_rows
        if row.get("normalized_transcript_sha256")
    }
    for row in pq.read_table(TRAIN_DONORS).to_pylist():
        occupied_audio.add(str(row["source_audio_sha256"]))
        occupied_text.add(str(row["normalized_transcript_sha256"]))

    connection = sqlite3.connect(f"file:{CATALOG}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    output: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    seen_audio = set(occupied_audio)
    seen_text = set(occupied_text)
    try:
        for split in ("validation", "test"):
            keys = sorted(speakers[split])
            placeholders = ",".join("?" for _ in keys)
            query = f"""
                SELECT * FROM assets
                WHERE speaker_key IN ({placeholders})
                  AND model_num_samples > ?
                  AND duration_sec <= ?
                  AND rejection_reason = 'complete_utterance_exceeds_442368'
                ORDER BY source_dataset,source_id
            """
            candidates = []
            for sqlite_row in connection.execute(
                query, (*keys, SHORT_CEILING, LONG_DRY_MAX_SEC)
            ):
                row = dict(sqlite_row)
                audio_hash = str(row["source_audio_sha256"])
                text_hash = str(row["normalized_transcript_sha256"])
                if audio_hash in seen_audio or text_hash in seen_text:
                    continue
                candidate = canonical_candidate(
                    row,
                    source_family=f"existing_{split}_long",
                    length_bucket=648,
                    qc_state="pending_strong_qc_eval_15s",
                    locator={
                        "type": "parquet_row",
                        "parquet_path": row["parquet_path"],
                        "row_group": int(row["row_group"]),
                        "row_in_group": int(row["row_in_group"]),
                    },
                )
                candidate.update(
                    {
                        "split": split,
                        "asset_id": str(row["asset_id"]),
                        "speaker_id": str(row["speaker_id"]),
                        "source_split": str(row["source_split"]),
                        "source_audio_path": None,
                    }
                )
                candidates.append(candidate)
            candidates.sort(key=lambda row: str(row["selection_rank"]))
            for row in candidates:
                audio_hash = str(row["source_audio_sha256"])
                text_hash = str(row["normalized_transcript_sha256"])
                if audio_hash in seen_audio or text_hash in seen_text:
                    continue
                seen_audio.add(audio_hash)
                seen_text.add(text_hash)
                output.append(row)
                counts[split] += 1
    finally:
        connection.close()

    if counts["validation"] < 500 or counts["test"] < 500:
        raise RuntimeError(f"insufficient split-disjoint long eval candidates: {counts}")
    if len({row["candidate_id"] for row in output}) != len(output):
        raise RuntimeError("eval long candidate IDs are not unique")
    atomic_parquet(CANDIDATES, output)
    summary = {
        "schema": "stable_audio_tools.sceneplan_eval_long_candidate_summary",
        "schema_version": 1,
        "state": "ready_for_strong_qc",
        "rows": len(output),
        "split_counts": dict(counts),
        "speaker_counts": {key: len(value) for key, value in speakers.items()},
        "audio_hash_collisions_with_train_or_legacy": 0,
        "transcript_hash_collisions_with_train_or_legacy": 0,
        "word_level_timestamp_teacher": False,
        "path": str(CANDIDATES),
        "sha256": sha256_file(CANDIDATES),
    }
    atomic_json(CANDIDATES.parent / "summary.json", summary)
    return summary


@lru_cache(maxsize=1)
def _eval_ledger_rows() -> tuple[dict[str, Any], ...]:
    table = pq.read_table(
        LEDGER,
        filters=[("pool", "in", ["validation", "test"])],
    )
    return tuple(table.to_pylist())


@lru_cache(maxsize=1)
def _speaker_profiles() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    ledger_rows = _eval_ledger_rows()
    ledger_by_asset = {str(row["asset_id"]): row for row in ledger_rows}
    profile_by_asset = {
        str(row["asset_id"]): row
        for row in pq.read_table(
            BASE_SPEAKERS,
            filters=[("split", "in", ["validation", "test"])],
        ).to_pylist()
    }
    by_speaker: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for asset_id, profile in profile_by_asset.items():
        ledger = ledger_by_asset.get(asset_id)
        if ledger is None:
            continue
        by_speaker[str(ledger["speaker_key"])].append((asset_id, profile))
    stable_by_speaker: dict[str, dict[str, Any]] = {}
    for speaker_key, values in by_speaker.items():
        values.sort(key=lambda value: value[0])
        stable_by_speaker[speaker_key] = values[0][1]
    return profile_by_asset, stable_by_speaker


def _base_speech_rows(split: str) -> list[dict[str, Any]]:
    ledger_rows = _eval_ledger_rows()
    profiles, _ = _speaker_profiles()
    rows = []
    for row in ledger_rows:
        if str(row["pool"]) != split:
            continue
        profile = profiles.get(str(row["asset_id"]))
        if profile is None:
            raise RuntimeError(f"missing legacy {split} speaker profile: {row['asset_id']}")
        value = dict(row)
        value.update(
            {
                "source_audio_path": None,
                "locator_json": json.dumps(
                    {
                        "type": "parquet_row",
                        "parquet_path": row["parquet_path"],
                        "row_group": int(row["row_group"]),
                        "row_in_group": int(row["row_in_group"]),
                    },
                    sort_keys=True,
                ),
                "model_num_samples": model_num_samples(
                    int(row["native_num_samples"]), int(row["native_sample_rate_hz"])
                ),
                "speaker_description": str(profile["speaker_description"]),
                "speaker_description_provenance": "frozen_legacy_split_profile",
                "length_bucket_frames": 432,
            }
        )
        if int(value["model_num_samples"]) > SHORT_CEILING:
            raise RuntimeError(f"legacy {split} speech escaped the short bucket")
        rows.append(value)
    rows.sort(
        key=lambda row: (
            deterministic_digest("eval-short", split, row["source_dataset"]),
            str(row["selection_rank"]),
        )
    )
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["source_dataset"])].append(row)
    interleaved = []
    for index in range(max(map(len, by_dataset.values()))):
        for dataset in sorted(by_dataset):
            if index < len(by_dataset[dataset]):
                interleaved.append(by_dataset[dataset][index])
    return interleaved


def _long_speech_rows(split: str) -> list[dict[str, Any]]:
    parts = sorted(QC_ROOT.glob("part-*.parquet"))
    if not parts:
        raise RuntimeError("eval long-source strong QC has not run")
    qc_rows = {
        str(row["candidate_id"]): row
        for row in pq.read_table(parts).to_pylist()
        if str(row["status"]) == "pass"
    }
    candidate_rows = {
        str(row["candidate_id"]): row
        for row in pq.read_table(CANDIDATES).to_pylist()
        if str(row["split"]) == split
    }
    _, profiles = _speaker_profiles()
    output = []
    for candidate_id, candidate in candidate_rows.items():
        qc = qc_rows.get(candidate_id)
        if qc is None:
            continue
        profile = profiles.get(str(candidate["speaker_key"]))
        if profile is None:
            raise RuntimeError(f"missing same-speaker profile for {candidate_id}")
        output.append(
            {
                "asset_id": str(candidate["asset_id"]),
                "source_dataset": str(candidate["source_dataset"]),
                "source_id": str(candidate["source_id"]),
                "speaker_id": str(candidate["speaker_id"]),
                "speaker_key": str(candidate["speaker_key"]),
                "renderer_text": str(candidate["renderer_text"]),
                "normalized_transcript_sha256": str(
                    candidate["normalized_transcript_sha256"]
                ),
                "source_audio_sha256": str(qc["source_audio_sha256"]),
                "native_sample_rate_hz": int(qc["native_sample_rate_hz"]),
                "native_num_samples": int(qc["native_num_samples"]),
                "model_num_samples": int(qc["model_num_samples"]),
                "locator_json": str(candidate["locator_json"]),
                "source_audio_path": None,
                "selection_rank": str(candidate["selection_rank"]),
                "speaker_description": str(profile["speaker_description"]),
                "speaker_description_provenance": "same_split_same_speaker_profile_reuse",
                "length_bucket_frames": 648,
            }
        )
    output.sort(key=lambda row: str(row["selection_rank"]))
    if len(output) < 500:
        raise RuntimeError(f"{split} has only {len(output)} passing long eval donors")
    return output


class Cycle:
    def __init__(self, rows: list[dict[str, Any]], label: str) -> None:
        if not rows:
            raise RuntimeError(f"empty deterministic cycle: {label}")
        self.rows = rows
        self.label = label
        self.cursor = 0

    def take(self) -> dict[str, Any]:
        row = self.rows[self.cursor % len(self.rows)]
        self.cursor += 1
        return row


def _nonspeech_kinds(source_count: int, ordinal: int) -> list[str]:
    if source_count == 1:
        return ["music" if ordinal % 2 else "sound"]
    mode = ordinal % 4
    if mode == 0:
        return ["sound"] * source_count
    if mode == 1:
        return ["music"] * source_count
    first = "sound" if mode == 2 else "music"
    second = "music" if first == "sound" else "sound"
    return [first if index % 2 == 0 else second for index in range(source_count)]


def build_no_speech_scene(
    *,
    sample_id: str,
    split: str,
    source_count: int,
    selectors: dict[str, DurationSelector],
    ordinal: int,
) -> dict[str, Any]:
    seed = int(deterministic_digest(20260828, "eval-no-speech-v6", sample_id)[:16], 16)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    used: set[str] = set()
    for source_index, kind in enumerate(_nonspeech_kinds(source_count, ordinal)):
        for attempt in range(32):
            row = selectors[kind].take(
                max_samples=SHORT_CEILING - 40,
                key=f"{sample_id}:{source_index}:{attempt}",
            )
            identity = str(row["source_audio_sha256"])
            if identity not in used:
                used.add(identity)
                rows.append(row)
                break
        else:
            raise RuntimeError(f"{sample_id}: could not select distinct sources")

    room_class = ROOM_CLASSES[ordinal % len(ROOM_CLASSES)]
    room = room_recipe(room_class, seed ^ 0x9E3779B97F4A7C15)
    longest = max(int(row["model_num_samples"]) for row in rows)
    tail = min(int(TAIL_TARGET[room_class]), SHORT_CEILING - longest)
    if tail < 40:
        raise RuntimeError(f"{sample_id}: no room for renderer tail")
    slack = min(2 * MODEL_SAMPLE_RATE, SHORT_CEILING - longest - tail)
    scene_samples = longest + tail + rng.randint(0, max(0, slack))
    active_capacity = scene_samples - tail
    slots = list(range(4))
    rng.shuffle(slots)
    model_sources = []
    recipe_sources = []
    windows = []
    for source_index, (row, slot) in enumerate(zip(rows, slots)):
        length = int(row["model_num_samples"])
        onset = rng.randint(0, max(0, active_capacity - length))
        offset = onset + length
        windows.append((onset, offset))
        dynamic = (ordinal + source_index) % 5 < 3
        trajectory = motion_plan(dynamic, onset, offset, rng, room)
        source_id = f"source_{slot}"
        model_sources.append(
            source_model_value(
                source_id=source_id,
                kind=str(row["kind"]),
                description=str(row["description"]),
                transcript=None,
                onset=onset,
                offset=offset,
                trajectory=trajectory,
                gain_db=0.0,
            )
        )
        recipe_sources.append(
            {
                "source_id": source_id,
                "kind": str(row["kind"]),
                "asset_ref": nonspeech_asset_ref(row),
                "exact_source_sample_window": {
                    "model_onset_sample": onset,
                    "model_offset_sample": offset,
                    "dry_start_sample": 0,
                    "dry_end_sample": length,
                },
            }
        )
    model_sources.sort(key=lambda row: int(str(row["source_id"])[7:]))
    recipe_sources.sort(key=lambda row: int(str(row["source_id"])[7:]))
    frames = math.ceil(scene_samples / VAE_HOP_SAMPLES)
    model_sceneplan = {
        "sample_id": sample_id,
        "duration_sec": rounded(scene_samples / MODEL_SAMPLE_RATE, 6),
        "room": {"type": room_class},
        "sources": model_sources,
    }
    validate_model_sceneplan(model_sceneplan)
    model_text = canonical_json(model_sceneplan)
    model_hash = sha256_text(model_text)
    render_recipe = {
        "schema": "stable_audio_tools.sceneplan_render_recipe",
        "schema_version": 2,
        "dataset_contract_revision": 6,
        "sample_id": sample_id,
        "model_sceneplan_sha256": model_hash,
        "recipe_seed": seed,
        "temporal_pattern": f"no_speech_{source_count}",
        "mixing": {"speech_background_mode": "not_applicable"},
        "audio_execution": {
            "model_num_samples": scene_samples,
            "latent_frames_valid": frames,
            "vae_padded_num_samples": frames * VAE_HOP_SAMPLES,
            "render_tail_samples": scene_samples - max(offset for _, offset in windows),
        },
        "resolved_room": {
            "room_id": str(room["room_id"]),
            "dimensions_m": list(map(float, room["dimensions_m"])),
            "rt60_sec": float(room["rt60_sec"]),
            "max_order": int(room["max_order"]),
            "microphone_xyz_m": list(map(float, room["microphone_xyz_m"])),
        },
        "sources": recipe_sources,
    }
    caption = compile_model_renderer_caption(model_sceneplan)
    controls = compile_model_44_controls(
        model_sceneplan,
        model_num_samples=scene_samples,
        latent_frames_valid=frames,
    )
    if controls["source_event_frame_ids"].shape != (4, frames):
        raise RuntimeError(f"{sample_id}: 4+4 event geometry changed")
    return {
        "sample_id": sample_id,
        "split": split,
        "family": "no_speech",
        "source_count": source_count,
        "temporal_pattern": f"no_speech_{source_count}",
        "room_type": room_class,
        "model_num_samples": scene_samples,
        "latent_frames_valid": frames,
        "model_sceneplan": model_sceneplan,
        "model_sceneplan_sha256": model_hash,
        "render_recipe": render_recipe,
        "render_recipe_sha256": sha256_text(canonical_json(render_recipe)),
        "caption": caption,
        "renderer_caption_sha256": sha256_text(canonical_json(caption)),
        "speech_asset_id": None,
        "source_asset_ids": [str(row["primary_asset_id"]) for row in rows],
        "source_kinds": [str(row["kind"]) for row in rows],
    }


class ManifestWriter:
    def __init__(self, root: Path, tokenizer: Any) -> None:
        self.root = root
        self.tokenizer = tokenizer
        self.index_tmp = root / "index.parquet.tmp"
        self.index_writer = pq.ParquetWriter(self.index_tmp, INDEX_SCHEMA, compression="zstd")
        self.rows: dict[str, list[dict[str, Any]]] = {"validation": [], "test": []}
        self.shards = {"validation": 0, "test": 0}
        self.total = Counter()
        self.patterns: dict[str, Counter[str]] = defaultdict(Counter)
        self.buckets: dict[str, Counter[int]] = defaultdict(Counter)
        self.source_counts: dict[str, Counter[int]] = defaultdict(Counter)
        self.caption_tokens: list[int] = []

    def append(self, row: dict[str, Any]) -> None:
        split = str(row["split"])
        self.rows[split].append(row)
        self.total[split] += 1
        self.patterns[split][str(row["temporal_pattern"])] += 1
        self.buckets[split][432 if int(row["latent_frames_valid"]) <= 432 else 648] += 1
        self.source_counts[split][int(row["source_count"])] += 1
        if len(self.rows[split]) >= SHARD_ROWS:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.rows[split]
        if not rows:
            return
        tokenized = self.tokenizer(
            [row["caption"]["text"] for row in rows],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]
        token_counts = [len(value) for value in tokenized]
        if max(token_counts) > 512:
            raise RuntimeError(f"{split}: eval caption exceeded 512 tokens")
        self.caption_tokens.extend(token_counts)
        shard = self.shards[split]
        stem = f"{split}-{shard:05d}"
        model_path = self.root / split / f"model-sceneplans-{stem}.jsonl"
        recipe_path = self.root / split / f"render-recipes-{stem}.jsonl"
        conditioning_path = self.root / split / f"conditioning-{stem}.jsonl"
        model_offsets = atomic_jsonl(
            model_path, [canonical_json(row["model_sceneplan"]) for row in rows]
        )
        recipe_offsets = atomic_jsonl(
            recipe_path, [canonical_json(row["render_recipe"]) for row in rows]
        )
        condition_offsets = atomic_jsonl(
            conditioning_path,
            [
                canonical_json(
                    {"sample_id": row["sample_id"], "renderer_caption": row["caption"]}
                )
                for row in rows
            ],
        )
        index_rows = []
        for row_index, row in enumerate(rows):
            index_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "family": row["family"],
                    "source_count": row["source_count"],
                    "room_type": row["room_type"],
                    "model_num_samples": row["model_num_samples"],
                    "latent_frames_valid": row["latent_frames_valid"],
                    "work_shard": shard,
                    "row_in_shard": row_index,
                    "sceneplan_path": str(model_path),
                    "sceneplan_byte_offset": model_offsets[row_index][0],
                    "sceneplan_byte_length": model_offsets[row_index][1],
                    "model_sceneplan_sha256": row["model_sceneplan_sha256"],
                    "render_recipe_path": str(recipe_path),
                    "render_recipe_byte_offset": recipe_offsets[row_index][0],
                    "render_recipe_byte_length": recipe_offsets[row_index][1],
                    "render_recipe_sha256": row["render_recipe_sha256"],
                    "conditioning_path": str(conditioning_path),
                    "conditioning_byte_offset": condition_offsets[row_index][0],
                    "conditioning_byte_length": condition_offsets[row_index][1],
                    "renderer_caption_sha256": row["renderer_caption_sha256"],
                    "caption_qwen_tokens": token_counts[row_index],
                    "speech_asset_id": row["speech_asset_id"],
                    "source_asset_ids": row["source_asset_ids"],
                    "source_kinds": row["source_kinds"],
                }
            )
        self.index_writer.write_table(pa.Table.from_pylist(index_rows, schema=INDEX_SCHEMA))
        self.shards[split] += 1
        self.rows[split] = []

    def close(self) -> dict[str, Any]:
        for split in self.rows:
            self.flush(split)
        self.index_writer.close()
        os.replace(self.index_tmp, self.root / "index.parquet")
        return {
            "split_counts": dict(self.total),
            "shard_counts": self.shards,
            "temporal_pattern_counts": {
                split: dict(values) for split, values in self.patterns.items()
            },
            "length_bucket_counts": {
                split: {str(key): value for key, value in values.items()}
                for split, values in self.buckets.items()
            },
            "source_count_counts": {
                split: {str(key): value for key, value in values.items()}
                for split, values in self.source_counts.items()
            },
            "caption_qwen_tokens": {
                "p99": float(np.percentile(self.caption_tokens, 99)),
                "max": max(self.caption_tokens),
                "hard_max": 512,
                "truncated": 0,
            },
        }


def build_scenes() -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B",
        local_files_only=True,
    )
    SCENEPLAN_ROOT.mkdir(parents=True, exist_ok=True)
    writer = ManifestWriter(SCENEPLAN_ROOT, tokenizer)
    started = time.monotonic()
    for split in ("validation", "test"):
        pools = load_nonspeech(split)
        selectors = {kind: DurationSelector(rows, kind) for kind, rows in pools.items()}
        short_cycle = Cycle(_base_speech_rows(split), f"{split}-short-speech")
        long_cycle = Cycle(_long_speech_rows(split), f"{split}-long-speech")
        ordinal = 0
        for source_count, count in NO_SPEECH_SOURCE_COUNTS[split].items():
            for _ in range(count):
                sample_id = f"spv2e_{split}_{ordinal:07d}"
                writer.append(
                    build_no_speech_scene(
                        sample_id=sample_id,
                        split=split,
                        source_count=source_count,
                        selectors=selectors,
                        ordinal=ordinal,
                    )
                )
                ordinal += 1
        for bucket in (432, 648):
            speech_cycle = short_cycle if bucket == 432 else long_cycle
            for pattern, count in DIRECT_COUNTS[split][bucket].items():
                for _ in range(count):
                    speech = speech_cycle.take()
                    kind = (
                        "music" if pattern.endswith("music") else
                        "sound" if pattern.endswith("sound") else None
                    )
                    background = None
                    if kind is not None:
                        background = selectors[kind].take(
                            max_samples=bucket * VAE_HOP_SAMPLES - 40,
                            key=f"direct:{split}:{bucket}:{pattern}:{speech['asset_id']}",
                        )
                    sample_id = f"spv2e_{split}_{ordinal:07d}"
                    writer.append(
                        build_scene(
                            sample_id=sample_id,
                            speech=speech,
                            background=background,
                            bucket_frames=bucket,
                            temporal_pattern=f"direct_{pattern}",
                            ordinal=ordinal,
                            split=split,
                        )
                    )
                    ordinal += 1
        for pattern, count in SEQUENTIAL_COUNTS[split].items():
            kind = "sound" if "sound" in pattern else "music"
            for _ in range(count):
                speech = short_cycle.take()
                room_class = ROOM_CLASSES[ordinal % len(ROOM_CLASSES)]
                max_background = (
                    LONG_CEILING
                    - int(speech["model_num_samples"])
                    - int(TAIL_TARGET[room_class])
                    - round(0.40 * MODEL_SAMPLE_RATE)
                )
                background = selectors[kind].take(
                    max_samples=max_background,
                    key=f"sequential:{split}:{pattern}:{speech['asset_id']}",
                )
                sample_id = f"spv2e_{split}_{ordinal:07d}"
                writer.append(
                    build_scene(
                        sample_id=sample_id,
                        speech=speech,
                        background=background,
                        bucket_frames=648,
                        temporal_pattern=f"sequential_{pattern}",
                        ordinal=ordinal,
                        split=split,
                    )
                )
                ordinal += 1
        if ordinal != DELTA_ROWS[split]:
            raise RuntimeError(f"{split}: built {ordinal}, expected {DELTA_ROWS[split]}")
    summary = {
        "schema": "stable_audio_tools.sceneplan_eval_expansion_manifest_summary",
        "schema_version": 1,
        "dataset_contract_revision": 6,
        "model_sceneplan_schema_version": 2,
        "state": "ready_for_audit",
        "rows": sum(DELTA_ROWS.values()),
        **writer.close(),
        "formal_speech_sources_per_scene_max": 1,
        "spoken_language_background_rows": 0,
        "speech_timing_sidecar": None,
        "word_level_timestamp_teacher": False,
        "index": str(SCENEPLAN_ROOT / "index.parquet"),
        "index_sha256": sha256_file(SCENEPLAN_ROOT / "index.parquet"),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    for split in DELTA_ROWS:
        actual = {int(key): value for key, value in summary["length_bucket_counts"][split].items()}
        if actual != DELTA_BUCKETS[split]:
            raise RuntimeError(f"{split}: delta length distribution changed: {actual}")
    atomic_json(SCENEPLAN_ROOT / "summary.json", summary)
    atomic_json(
        SCENEPLAN_ROOT / "READY",
        {
            "schema": "stable_audio_tools.sceneplan_eval_expansion_ready",
            "schema_version": 1,
            "rows": summary["rows"],
            "split_counts": summary["split_counts"],
            "p8_started": False,
        },
    )
    return summary


def audit_scenes() -> dict[str, Any]:
    summary = json.loads((SCENEPLAN_ROOT / "summary.json").read_text(encoding="utf-8"))
    if summary.get("split_counts") != DELTA_ROWS:
        raise RuntimeError("eval ScenePlan summary split counts changed")
    index = pq.read_table(SCENEPLAN_ROOT / "index.parquet").to_pylist()
    if len(index) != sum(DELTA_ROWS.values()):
        raise RuntimeError("eval ScenePlan index coverage changed")
    seen: set[str] = set()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    buckets: dict[str, Counter[int]] = defaultdict(Counter)
    source_counts: dict[str, Counter[int]] = defaultdict(Counter)
    kinds: dict[str, Counter[str]] = defaultdict(Counter)
    for split in ("validation", "test"):
        for path in sorted((SCENEPLAN_ROOT / split).glob(f"model-sceneplans-{split}-*.jsonl")):
            for row in load_shard_rows(path):
                sample_id = str(row["sample_id"])
                if sample_id in seen:
                    raise RuntimeError(f"duplicate eval ScenePlan ID: {sample_id}")
                seen.add(sample_id)
                scene = json.loads(str(row["model_sceneplan_json"]))
                recipe = json.loads(str(row["render_recipe_json"]))
                if str(row["split"]) != split or not sample_id.startswith(f"spv2e_{split}_"):
                    raise RuntimeError(f"{sample_id}: split identity changed")
                validate_model_sceneplan(scene)
                speech = sum(source["kind"] == "speech" for source in scene["sources"])
                if speech > 1:
                    raise RuntimeError(f"{sample_id}: multiple formal speech sources")
                pattern = str(recipe["temporal_pattern"])
                counts[split][pattern] += 1
                buckets[split][432 if int(row["latent_frames_valid"]) <= 432 else 648] += 1
                source_counts[split][len(scene["sources"])] += 1
                kinds[split].update(str(source["kind"]) for source in scene["sources"])
                controls = compile_model_44_controls(
                    scene,
                    model_num_samples=int(row["model_num_samples"]),
                    latent_frames_valid=int(row["latent_frames_valid"]),
                )
                if controls["source_event_frame_ids"].shape != (
                    4, int(row["latent_frames_valid"])
                ):
                    raise RuntimeError(f"{sample_id}: 4+4 alignment changed")
                if pattern.startswith("sequential_"):
                    windows = [
                        value["exact_source_sample_window"]
                        for value in recipe["sources"]
                    ]
                    if min(windows[0]["model_offset_sample"], windows[1]["model_offset_sample"]) > max(
                        windows[0]["model_onset_sample"], windows[1]["model_onset_sample"]
                    ):
                        raise RuntimeError(f"{sample_id}: sequential sources overlap")
    if len(seen) != sum(DELTA_ROWS.values()):
        raise RuntimeError("eval ScenePlan shard coverage is incomplete")
    for split, expected in DELTA_BUCKETS.items():
        if buckets[split] != Counter(expected):
            raise RuntimeError(f"{split}: audited buckets changed: {buckets[split]}")
        if counts[split] != Counter(DELTA_PATTERNS[split]):
            raise RuntimeError(f"{split}: audited temporal patterns changed")
        if source_counts[split] != Counter(DELTA_SOURCE_COUNTS[split]):
            raise RuntimeError(f"{split}: audited source-count distribution changed")
        if kinds[split] != Counter(DELTA_KIND_APPEARANCES[split]):
            raise RuntimeError(f"{split}: audited source-kind appearances changed")
    audit = {
        "schema": "stable_audio_tools.sceneplan_eval_expansion_manifest_audit",
        "schema_version": 1,
        "dataset_contract_revision": 6,
        "ok": True,
        "rows": len(seen),
        "split_counts": DELTA_ROWS,
        "length_bucket_counts": {
            split: {str(key): value for key, value in counter.items()}
            for split, counter in buckets.items()
        },
        "temporal_pattern_counts": {
            split: dict(counter) for split, counter in counts.items()
        },
        "source_count_counts": {
            split: {str(key): value for key, value in counter.items()}
            for split, counter in source_counts.items()
        },
        "source_kind_counts": {
            split: dict(counter) for split, counter in kinds.items()
        },
        "formal_speech_sources_per_scene_max": 1,
        "all_sequential_windows_nonoverlapping": True,
        "caption_and_4plus4_recompiled": True,
        "speech_timing_sidecar": None,
        "word_level_timestamp_teacher": False,
        "p8_started": False,
    }
    atomic_json(SCENEPLAN_ROOT / "audit.json", audit)
    return audit


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def audit_materialized() -> dict[str, Any]:
    p8 = json.loads((MATERIALIZED_ROOT / "P8_SUMMARY.json").read_text(encoding="utf-8"))
    if p8.get("status") != "complete" or int(p8.get("rows", -1)) != sum(DELTA_ROWS.values()):
        raise RuntimeError("eval P8 is incomplete")
    seen: set[str] = set()
    split_counts: Counter[str] = Counter()
    buckets: dict[str, Counter[int]] = defaultdict(Counter)
    for split in ("validation", "test"):
        manifests = sorted(
            (MATERIALIZED_ROOT / "manifests" / split).glob(f"materialized-{split}-*.parquet")
        )
        for manifest in manifests:
            rows = pq.read_table(manifest).to_pylist()
            refs = {str(row["latent_ref"]).split("#", 1)[0] for row in rows}
            hashes = {str(row["latent_shard_sha256"]) for row in rows}
            if len(refs) != 1 or len(hashes) != 1:
                raise RuntimeError(f"{manifest}: latent shard lineage changed")
            latent_path = Path(refs.pop()).resolve(strict=True)
            if sha256_file(latent_path) != hashes.pop():
                raise RuntimeError(f"{latent_path}: latent shard checksum changed")
            with safe_open(str(latent_path), framework="pt", device="cpu") as handle:
                for row in rows:
                    sample_id = str(row["sample_id"])
                    if sample_id in seen:
                        raise RuntimeError(f"duplicate materialized eval ID: {sample_id}")
                    seen.add(sample_id)
                    split_counts[split] += 1
                    frames = int(row["latent_frames_valid"])
                    buckets[split][432 if frames <= 432 else 648] += 1
                    tensor = handle.get_tensor(sample_id)
                    if tensor.dtype != torch.float16 or tuple(tensor.shape) != (64, frames):
                        raise RuntimeError(f"{sample_id}: eval latent shape/dtype changed")
                    if not torch.isfinite(tensor).all() or _tensor_sha256(tensor) != str(
                        row["latent_tensor_sha256"]
                    ):
                        raise RuntimeError(f"{sample_id}: eval latent checksum/nonfinite")
                    foa = Path(str(row["foa_path"])).resolve(strict=True)
                    if sha256_file(foa) != str(row["foa_sha256"]):
                        raise RuntimeError(f"{sample_id}: retained eval FOA checksum changed")
                    result = json.loads(str(row["render_result_json"]))
                    if result.get("status") != "ok" or int(
                        result.get("dataset_contract_revision", -1)
                    ) != 6:
                        raise RuntimeError(f"{sample_id}: eval render did not pass")
                    for source in result.get("source_qc") or ():
                        lineage = source["content_lineage"]
                        if float(lineage["coverage_fraction"]) != 1.0 or lineage["random_crop"] is not False:
                            raise RuntimeError(f"{sample_id}: eval source was cropped")
    if dict(split_counts) != DELTA_ROWS:
        raise RuntimeError(f"materialized eval split counts changed: {split_counts}")
    for split, expected in DELTA_BUCKETS.items():
        if buckets[split] != Counter(expected):
            raise RuntimeError(f"{split}: materialized eval buckets changed")
    audit = {
        "schema": "stable_audio_tools.sceneplan_eval_expansion_materialized_audit",
        "schema_version": 1,
        "dataset_contract_revision": 6,
        "ok": True,
        "status": "complete",
        "rows": len(seen),
        "split_counts": dict(split_counts),
        "length_bucket_counts": {
            split: {str(key): value for key, value in counter.items()}
            for split, counter in buckets.items()
        },
        "all_sources_complete_and_uncropped": True,
        "all_latents_finite_float16_with_checksums": True,
        "all_eval_foa_retained_with_checksums": True,
        "speech_timing_sidecar": None,
        "word_level_timestamp_teacher": False,
        "ready_for_training_index_merge": True,
    }
    atomic_json(MATERIALIZED_ROOT / "audit.json", audit)
    atomic_json(REVISION_ROOT / "EVAL_READY_FOR_MERGE.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan-candidates", "build-scenes", "audit-scenes", "audit-materialized"),
    )
    args = parser.parse_args()
    functions = {
        "plan-candidates": plan_candidates,
        "build-scenes": build_scenes,
        "audit-scenes": audit_scenes,
        "audit-materialized": audit_materialized,
    }
    result = functions[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
