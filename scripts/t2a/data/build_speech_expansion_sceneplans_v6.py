#!/usr/bin/env python3
"""Build the 500k revision-6 ScenePlan delta for 0--15 second P10.

The compact model ScenePlan remains the sole semantic/time/space state.  Asset
locators, exact sample windows, room execution values, and overlap calibration
mode live only in the renderer recipe.  The resulting three-view JSONL is
directly consumable by the existing P8 materializer after its revision-6
compatibility checks.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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
from scripts.t2a.data.render_tts_v2_pilot import room_recipe  # noqa: E402
from scripts.t2a.data.sceneplan_v2_common import deterministic_digest  # noqa: E402
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_44_controls,
    compile_model_renderer_caption,
    validate_model_sceneplan,
)


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
DEFAULT_DONORS = REVISION_ROOT / (
    "source_annotations/speech_speaker_instruct_v1/registry/"
    "final_speech_donors_with_speakers.parquet"
)
DEFAULT_LEDGER = DATASET_ROOT / "split_ledgers/speech_v2/speech_split_ledger.parquet"
DEFAULT_SPEAKERS = DATASET_ROOT / (
    "source_annotations/speech_speaker_instruct_v1/registry/"
    "speech_speaker_description_registry.parquet"
)
DEFAULT_OUTPUT = REVISION_ROOT / "sceneplans_model_v2_delta"
NONSPEECH_INPUTS = (
    (
        DATASET_ROOT / "source_annotations/nonspeech_instruct_v2/source_universe.parquet",
        DATASET_ROOT / (
            "source_annotations/nonspeech_instruct_v2/registry/"
            "source_description_registry.parquet"
        ),
    ),
    (
        DATASET_ROOT / "revisions/sound_expansion_v1/sources/universe/source_universe.parquet",
        DATASET_ROOT / (
            "revisions/sound_expansion_v1/source_annotations/registry/"
            "source_description_registry.parquet"
        ),
    ),
)
MODEL_SAMPLE_RATE = 44_100
VAE_HOP_SAMPLES = 1024
SHORT_FRAMES = 432
LONG_FRAMES = 648
SHORT_CEILING = SHORT_FRAMES * VAE_HOP_SAMPLES
LONG_CEILING = LONG_FRAMES * VAE_HOP_SAMPLES
SHARD_ROWS = 1024
DIRECT_COUNTS = {
    432: {"speech_only": 37_500, "speech_plus_music": 31_250, "speech_plus_sound": 31_250},
    648: {"speech_only": 37_500, "speech_plus_music": 31_250, "speech_plus_sound": 31_250},
}
SEQUENTIAL_COUNTS = {
    "speech_then_sound": 75_000,
    "sound_then_speech": 75_000,
    "speech_then_music": 75_000,
    "music_then_speech": 75_000,
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def speech_asset_ref(row: dict[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "asset_id": str(row["asset_id"]),
        "dataset_id": str(row["source_dataset"]),
        "identity_hash": str(row["source_audio_sha256"]),
        "native_sample_rate_hz": int(row["native_sample_rate_hz"]),
        "native_num_samples": int(row["native_num_samples"]),
        "input_audio_domain": "dry_mono",
        "canonical_channels": 1,
        "spatialization_passes_before_scene": 0,
        "eligible_as_scene_source": True,
        "parent_asset_id": None,
        "parent_start_sample": None,
        "parent_end_sample": None,
        "segment_method": "full_native_utterance",
        "normalized_transcript_sha256": str(
            row["normalized_transcript_sha256"]
        ),
    }
    source_path = str(row.get("source_audio_path") or "")
    if source_path:
        value.update(
            {
                "dry_audio_path": source_path,
                "parquet_path": None,
                "row_group": None,
                "row_in_group": None,
            }
        )
    else:
        locator = json.loads(str(row["locator_json"]))
        if locator.get("type") != "parquet_row":
            raise RuntimeError(f"{row['asset_id']}: invalid speech locator")
        value.update(
            {
                "dry_audio_path": None,
                "parquet_path": str(locator["parquet_path"]),
                "row_group": int(locator["row_group"]),
                "row_in_group": int(locator["row_in_group"]),
            }
        )
    return value


def nonspeech_asset_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_id": str(row["primary_asset_id"]),
        "dataset_id": str(row["source_dataset"]),
        "identity_hash": str(row["source_audio_sha256"]),
        "native_sample_rate_hz": int(row["native_sample_rate_hz"]),
        "native_num_samples": int(row["native_num_samples"]),
        "input_audio_domain": "dry_mono",
        "canonical_channels": 1,
        "spatialization_passes_before_scene": 0,
        "eligible_as_scene_source": True,
        "dry_audio_path": str(row["dry_audio_path"]),
        "parent_asset_id": None,
        "parent_start_sample": None,
        "parent_end_sample": None,
        "segment_method": "complete_registry_asset",
        "parquet_path": None,
        "row_group": None,
        "row_in_group": None,
    }


def load_nonspeech(split: str = "train") -> dict[str, list[dict[str, Any]]]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported non-speech split: {split}")
    by_hash: dict[str, dict[str, Any]] = {}
    for universe_path, registry_path in NONSPEECH_INPUTS:
        universe = {
            str(row["source_audio_sha256"]): row
            for row in pq.read_table(
                universe_path, filters=[("split", "=", split)]
            ).to_pylist()
        }
        for annotation in pq.read_table(
            registry_path, filters=[("split", "=", split)]
        ).to_pylist():
            source_hash = str(annotation["source_audio_sha256"])
            source = universe.get(source_hash)
            if source is None:
                raise RuntimeError(f"nonspeech registry has no universe row: {source_hash}")
            if (
                str(annotation["split"]) != split
                or bool(annotation["spoken_language_background"])
            ):
                continue
            row = dict(source)
            row["description"] = str(annotation["source_description"])
            row["spoken_language_background"] = False
            existing = by_hash.get(source_hash)
            if existing is None or str(row["selection_rank"]) < str(
                existing["selection_rank"]
            ):
                by_hash[source_hash] = row
    output = {"music": [], "sound": []}
    for row in by_hash.values():
        kind = str(row["kind"])
        if kind in output:
            output[kind].append(row)
    for kind in output:
        output[kind].sort(
            key=lambda row: (int(row["model_num_samples"]), str(row["selection_rank"]))
        )
        if not output[kind]:
            raise RuntimeError(f"empty clean {split} nonspeech pool: {kind}")
    return output


class DurationSelector:
    def __init__(self, rows: list[dict[str, Any]], kind: str) -> None:
        self.rows = rows
        self.kind = kind
        self.lengths = [int(row["model_num_samples"]) for row in rows]
        self.counter = 0

    def take(self, *, max_samples: int, key: str) -> dict[str, Any]:
        stop = bisect_right(self.lengths, int(max_samples))
        if stop <= 0:
            raise RuntimeError(
                f"no {self.kind} source fits {max_samples} model samples"
            )
        index = int(
            deterministic_digest("speech-expansion-background", self.kind, key)[:16],
            16,
        ) % stop
        # Mix a hash-derived starting point with a monotonically advancing
        # cursor so repeated assets are spread over the whole eligible prefix.
        index = (index + self.counter) % stop
        self.counter += 1
        return self.rows[index]


def formal_sequential_speech(ledger: Path, speakers: Path) -> list[dict[str, Any]]:
    speaker_by_asset = {
        str(row["asset_id"]): row
        for row in pq.read_table(speakers).to_pylist()
        if str(row["split"]) == "train"
    }
    rows = pq.read_table(ledger, filters=[("pool", "=", "train")]).to_pylist()
    rows.sort(
        key=lambda row: deterministic_digest(
            "speech-expansion-sequential", row["selection_rank"], row["asset_id"]
        )
    )
    selected = rows[:300_000]
    if len(selected) != 300_000 or len({row["asset_id"] for row in selected}) != 300_000:
        raise RuntimeError("formal sequential speech selection is not 300k unique")
    output = []
    for row in selected:
        profile = speaker_by_asset.get(str(row["asset_id"]))
        if profile is None:
            raise RuntimeError(f"missing frozen speaker description: {row['asset_id']}")
        value = dict(row)
        value["source_audio_path"] = None
        value["locator_json"] = json.dumps(
            {
                "type": "parquet_row",
                "parquet_path": row["parquet_path"],
                "row_group": int(row["row_group"]),
                "row_in_group": int(row["row_in_group"]),
            },
            sort_keys=True,
        )
        value["speaker_description"] = str(profile["speaker_description"])
        value["speaker_description_provenance"] = str(
            profile["identity_provenance"]
        )
        output.append(value)
    return output


def source_model_value(
    *,
    source_id: str,
    kind: str,
    description: str,
    transcript: str | None,
    onset: int,
    offset: int,
    trajectory: dict[str, Any],
    gain_db: float,
) -> dict[str, Any]:
    if trajectory["type"] == "static":
        compact_trajectory = {
            "type": "static",
            "position": {
                key: rounded(value, 4 if key == "distance_m" else 3)
                for key, value in trajectory["keyframes"][0]["position"].items()
            },
        }
    else:
        compact_trajectory = {
            "type": "linear",
            "start": {
                key: rounded(value, 4 if key == "distance_m" else 3)
                for key, value in trajectory["keyframes"][0]["position"].items()
            },
            "end": {
                key: rounded(value, 4 if key == "distance_m" else 3)
                for key, value in trajectory["keyframes"][-1]["position"].items()
            },
        }
    value: dict[str, Any] = {
        "source_id": source_id,
        "kind": kind,
        "activity": {
            "onset_sec": rounded(onset / MODEL_SAMPLE_RATE, 6),
            "offset_sec": rounded(offset / MODEL_SAMPLE_RATE, 6),
        },
        "trajectory": compact_trajectory,
        "gain_db": rounded(gain_db, 4),
    }
    if kind == "speech":
        value["speaker_description"] = description
        value["transcript"] = str(transcript)
    else:
        value["description"] = description
    return value


def planned_background_gain(seed: int) -> float:
    quantile = int(deterministic_digest(seed, "mix")[:8], 16) / 0xFFFFFFFF
    median = 20.0 * math.log10(0.6 / 0.4)
    if quantile <= 0.5:
        target = 2.0 + (median - 2.0) * (quantile / 0.5)
    else:
        target = median + (6.0 - median) * ((quantile - 0.5) / 0.5)
    return -target


def choose_direct_activities(
    *,
    lengths: list[int],
    active_capacity: int,
    mixed: bool,
    rng: random.Random,
) -> list[tuple[int, int]]:
    values = []
    for length in lengths:
        onset = rng.randint(0, max(0, active_capacity - length))
        values.append((onset, onset + length))
    if mixed:
        speech_onset, speech_offset = values[0]
        bg_onset, bg_offset = values[1]
        minimum = round(0.10 * MODEL_SAMPLE_RATE)
        overlap = min(speech_offset, bg_offset) - max(speech_onset, bg_onset)
        if overlap < minimum:
            length = lengths[1]
            low = max(0, speech_onset - length + minimum)
            high = min(active_capacity - length, speech_offset - minimum)
            if low > high:
                raise RuntimeError("cannot place required direct-scene overlap")
            bg_onset = rng.randint(low, high)
            values[1] = (bg_onset, bg_onset + length)
    return values


def build_scene(
    *,
    sample_id: str,
    speech: dict[str, Any],
    background: dict[str, Any] | None,
    bucket_frames: int,
    temporal_pattern: str,
    ordinal: int,
    split: str = "train",
) -> dict[str, Any]:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unsupported ScenePlan split: {split}")
    seed = int(deterministic_digest(20260828, "scene-v6", sample_id)[:16], 16)
    rng = random.Random(seed)
    ceiling = int(bucket_frames) * VAE_HOP_SAMPLES
    room_class = ROOM_CLASSES[ordinal % len(ROOM_CLASSES)]
    room = room_recipe(room_class, seed ^ 0x9E3779B97F4A7C15)
    speech_samples = int(speech["model_num_samples"])
    background_samples = int(background["model_num_samples"]) if background else 0
    tail_target = int(TAIL_TARGET[room_class])
    if temporal_pattern.startswith("sequential_"):
        gap = rng.randint(round(0.05 * MODEL_SAMPLE_RATE), round(0.40 * MODEL_SAMPLE_RATE))
        event_span = speech_samples + background_samples + gap
        minimum_scene = max(SHORT_CEILING + 1, event_span + tail_target)
        if minimum_scene > ceiling:
            raise RuntimeError(f"{sample_id}: sequential sources do not fit long envelope")
        scene_samples = rng.randint(minimum_scene, ceiling)
        free = scene_samples - event_span - tail_target
        front = rng.randint(0, max(0, free))
        if temporal_pattern in {"sequential_speech_then_sound", "sequential_speech_then_music"}:
            speech_window = (front, front + speech_samples)
            background_window = (
                speech_window[1] + gap,
                speech_window[1] + gap + background_samples,
            )
        else:
            background_window = (front, front + background_samples)
            speech_window = (
                background_window[1] + gap,
                background_window[1] + gap + speech_samples,
            )
        mixing_mode = "sequential_nonoverlap"
    else:
        longest = max(speech_samples, background_samples)
        tail_target = min(tail_target, ceiling - longest)
        if tail_target < 40:
            raise RuntimeError(f"{sample_id}: no room for residual RIR tail")
        minimum_scene = longest + tail_target
        if minimum_scene > ceiling:
            raise RuntimeError(f"{sample_id}: direct sources exceed bucket envelope")
        slack_limit = min(2 * MODEL_SAMPLE_RATE, ceiling - minimum_scene)
        scene_samples = minimum_scene + rng.randint(0, max(0, slack_limit))
        # A long donor must remain in the long bucket even when it is only one
        # sample beyond the 432-frame boundary.
        if bucket_frames == LONG_FRAMES:
            scene_samples = max(scene_samples, SHORT_CEILING + 1)
        active_capacity = scene_samples - tail_target
        activities = choose_direct_activities(
            lengths=[speech_samples] + ([background_samples] if background else []),
            active_capacity=active_capacity,
            mixed=background is not None,
            rng=rng,
        )
        speech_window = activities[0]
        background_window = activities[1] if background else None
        mixing_mode = "overlap_calibrated" if background else "not_applicable"

    frames = math.ceil(scene_samples / VAE_HOP_SAMPLES)
    if frames > bucket_frames or (bucket_frames == LONG_FRAMES and frames <= SHORT_FRAMES):
        raise RuntimeError(f"{sample_id}: scene landed in the wrong length bucket")
    sources = [("speech", speech, speech_window)]
    if background is not None and background_window is not None:
        sources.append((str(background["kind"]), background, background_window))
    slots = list(range(4))
    rng.shuffle(slots)
    model_sources = []
    recipe_sources = []
    background_gain = planned_background_gain(seed)
    for source_index, ((kind, row, (onset, offset)), slot) in enumerate(
        zip(sources, slots)
    ):
        dynamic = (ordinal + source_index) % 5 < (1 if kind == "speech" else 3)
        trajectory = motion_plan(dynamic, onset, offset, rng, room)
        description = str(
            row["speaker_description"] if kind == "speech" else row["description"]
        )
        transcript = str(row["renderer_text"]).strip() if kind == "speech" else None
        source_id = f"source_{slot}"
        gain = 0.0 if kind == "speech" else background_gain
        model_sources.append(
            source_model_value(
                source_id=source_id,
                kind=kind,
                description=description,
                transcript=transcript,
                onset=onset,
                offset=offset,
                trajectory=trajectory,
                gain_db=gain,
            )
        )
        recipe_value = {
            "source_id": source_id,
            "kind": kind,
            "asset_ref": (
                speech_asset_ref(row) if kind == "speech" else nonspeech_asset_ref(row)
            ),
            "exact_source_sample_window": {
                "model_onset_sample": onset,
                "model_offset_sample": offset,
                "dry_start_sample": 0,
                "dry_end_sample": int(row["model_num_samples"]),
            },
        }
        if kind == "speech":
            recipe_value["speaker_id"] = str(row["speaker_id"])
        recipe_sources.append(recipe_value)
    model_sources.sort(key=lambda source: int(source["source_id"][7:]))
    recipe_sources.sort(key=lambda source: int(source["source_id"][7:]))
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
        "temporal_pattern": temporal_pattern,
        "mixing": {"speech_background_mode": mixing_mode},
        "audio_execution": {
            "model_num_samples": scene_samples,
            "latent_frames_valid": frames,
            "vae_padded_num_samples": frames * VAE_HOP_SAMPLES,
            "render_tail_samples": scene_samples
            - max(window[1] for _, _, window in sources),
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
        raise RuntimeError(f"{sample_id}: event track geometry changed")
    return {
        "sample_id": sample_id,
        "split": split,
        "family": "speech",
        "source_count": len(sources),
        "temporal_pattern": temporal_pattern,
        "room_type": room_class,
        "model_num_samples": scene_samples,
        "latent_frames_valid": frames,
        "model_sceneplan": model_sceneplan,
        "model_sceneplan_sha256": model_hash,
        "render_recipe": render_recipe,
        "render_recipe_sha256": sha256_text(canonical_json(render_recipe)),
        "caption": caption,
        "renderer_caption_sha256": sha256_text(canonical_json(caption)),
        "speech_asset_id": str(speech["asset_id"]),
        "source_asset_ids": [
            str(speech["asset_id"]),
            *([str(background["primary_asset_id"])] if background else []),
        ],
        "source_kinds": ["speech", *([str(background["kind"])] if background else [])],
    }


def direct_specs(donors: list[dict[str, Any]]) -> Iterable[tuple[int, str, dict[str, Any]]]:
    by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in donors:
        by_bucket[int(row["length_bucket_frames"])].append(row)
    for bucket in (432, 648):
        rows = sorted(by_bucket[bucket], key=lambda row: int(row["donor_ordinal"]))
        cursor = 0
        for pattern, count in DIRECT_COUNTS[bucket].items():
            for row in rows[cursor : cursor + count]:
                yield bucket, pattern, row
            cursor += count
        if cursor != len(rows) or cursor != 100_000:
            raise RuntimeError(f"direct donor allocation changed for bucket {bucket}")


def pilot_direct_specs(
    donors: list[dict[str, Any]],
) -> Iterable[tuple[int, str, dict[str, Any]]]:
    """Cover both buckets and every direct family in a 60-scene pilot."""

    by_bucket: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in donors:
        by_bucket[int(row["length_bucket_frames"])].append(row)
    for bucket in (432, 648):
        rows = sorted(by_bucket[bucket], key=lambda row: int(row["donor_ordinal"]))
        cursor = 0
        for pattern, count in DIRECT_COUNTS[bucket].items():
            cell = rows[cursor : cursor + count]
            indices = np.linspace(0, len(cell) - 1, 10, dtype=np.int64)
            for index in indices:
                yield bucket, pattern, cell[int(index)]
            cursor += count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donors", type=Path, default=DEFAULT_DONORS)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--speaker-registry", type=Path, default=DEFAULT_SPEAKERS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--pilot-balanced",
        action="store_true",
        help="build 100 rows covering both buckets and all seven temporal patterns",
    )
    args = parser.parse_args()
    donors_path = args.donors.expanduser().resolve(strict=True)
    ledger = args.ledger.expanduser().resolve(strict=True)
    speakers = args.speaker_registry.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    if not str(output).startswith("/mnt/sdb/audio_dataset/"):
        raise ValueError("revision-6 ScenePlans must remain on SDB")
    output.mkdir(parents=True, exist_ok=True)
    donors = pq.read_table(donors_path).to_pylist()
    if len(donors) != 200_000:
        raise RuntimeError("revision-6 direct donor registry is not 200k")
    sequential_speech = formal_sequential_speech(ledger, speakers)
    pools = load_nonspeech()
    selectors = {
        kind: DurationSelector(rows, kind) for kind, rows in pools.items()
    }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B",
        local_files_only=True,
    )
    index_tmp = output / "index.parquet.tmp"
    index_writer = pq.ParquetWriter(index_tmp, INDEX_SCHEMA, compression="zstd")
    shard_rows: list[dict[str, Any]] = []
    shard_index = 0
    global_rows = 0
    counts: Counter[str] = Counter()
    bucket_counts: Counter[int] = Counter()
    caption_tokens: list[int] = []
    started = time.monotonic()

    def flush() -> None:
        nonlocal shard_rows, shard_index
        if not shard_rows:
            return
        tokenized = tokenizer(
            [row["caption"]["text"] for row in shard_rows],
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]
        token_counts = [len(ids) for ids in tokenized]
        if max(token_counts) > 512:
            offender = shard_rows[token_counts.index(max(token_counts))]
            raise RuntimeError(
                f"{offender['sample_id']}: caption exceeds 512 Qwen tokens"
            )
        caption_tokens.extend(token_counts)
        stem = f"train-{shard_index:05d}"
        model_path = output / "train" / f"model-sceneplans-{stem}.jsonl"
        recipe_path = output / "train" / f"render-recipes-{stem}.jsonl"
        conditioning_path = output / "train" / f"conditioning-{stem}.jsonl"
        model_texts = [canonical_json(row["model_sceneplan"]) for row in shard_rows]
        recipe_texts = [canonical_json(row["render_recipe"]) for row in shard_rows]
        condition_texts = [
            canonical_json(
                {"sample_id": row["sample_id"], "renderer_caption": row["caption"]}
            )
            for row in shard_rows
        ]
        model_offsets = atomic_jsonl(model_path, model_texts)
        recipe_offsets = atomic_jsonl(recipe_path, recipe_texts)
        condition_offsets = atomic_jsonl(conditioning_path, condition_texts)
        index_rows = []
        for row_index, row in enumerate(shard_rows):
            index_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": "train",
                    "family": "speech",
                    "source_count": row["source_count"],
                    "room_type": row["room_type"],
                    "model_num_samples": row["model_num_samples"],
                    "latent_frames_valid": row["latent_frames_valid"],
                    "work_shard": shard_index,
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
        index_writer.write_table(pa.Table.from_pylist(index_rows, schema=INDEX_SCHEMA))
        shard_index += 1
        shard_rows = []

    def append_scene(scene: dict[str, Any]) -> bool:
        nonlocal global_rows
        if args.limit is not None and global_rows >= int(args.limit):
            return False
        shard_rows.append(scene)
        global_rows += 1
        counts[scene["temporal_pattern"]] += 1
        bucket_counts[432 if scene["latent_frames_valid"] <= 432 else 648] += 1
        if len(shard_rows) >= SHARD_ROWS:
            flush()
        if global_rows % 10_000 == 0:
            print(
                json.dumps(
                    {
                        "planned": global_rows,
                        "elapsed_sec": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
        return True

    try:
        direct_iterator = (
            pilot_direct_specs(donors) if args.pilot_balanced else direct_specs(donors)
        )
        for bucket, pattern, speech in direct_iterator:
            background = None
            kind = None
            if pattern.endswith("music"):
                kind = "music"
            elif pattern.endswith("sound"):
                kind = "sound"
            if kind:
                ceiling = bucket * VAE_HOP_SAMPLES
                background = selectors[kind].take(
                    max_samples=ceiling - 40,
                    key=f"direct:{bucket}:{pattern}:{speech['asset_id']}",
                )
            sample_id = f"spv2e_train_{global_rows:07d}"
            scene = build_scene(
                sample_id=sample_id,
                speech=speech,
                background=background,
                bucket_frames=bucket,
                temporal_pattern=f"direct_{pattern}",
                ordinal=global_rows,
            )
            if not append_scene(scene):
                break
        if args.limit is None or global_rows < int(args.limit):
            cursor = 0
            sequential_counts = (
                {key: 10 for key in SEQUENTIAL_COUNTS}
                if args.pilot_balanced
                else SEQUENTIAL_COUNTS
            )
            for pattern, count in sequential_counts.items():
                kind = "sound" if "sound" in pattern else "music"
                for _ in range(count):
                    speech = sequential_speech[cursor]
                    cursor += 1
                    room_class = ROOM_CLASSES[global_rows % len(ROOM_CLASSES)]
                    max_background = (
                        LONG_CEILING
                        - int(speech["model_num_samples"])
                        - int(TAIL_TARGET[room_class])
                        - round(0.40 * MODEL_SAMPLE_RATE)
                    )
                    background = selectors[kind].take(
                        max_samples=max_background,
                        key=f"sequential:{pattern}:{speech['asset_id']}",
                    )
                    sample_id = f"spv2e_train_{global_rows:07d}"
                    scene = build_scene(
                        sample_id=sample_id,
                        speech=speech,
                        background=background,
                        bucket_frames=648,
                        temporal_pattern=f"sequential_{pattern}",
                        ordinal=global_rows,
                    )
                    if not append_scene(scene):
                        break
                if args.limit is not None and global_rows >= int(args.limit):
                    break
            expected_sequential = 40 if args.pilot_balanced else 300_000
            if args.limit is None and cursor != expected_sequential:
                raise RuntimeError("sequential speech allocation changed")
        flush()
    finally:
        index_writer.close()
    os.replace(index_tmp, output / "index.parquet")
    expected_rows = (
        int(args.limit)
        if args.limit is not None
        else (100 if args.pilot_balanced else 500_000)
    )
    if global_rows != expected_rows:
        raise RuntimeError(f"planned {global_rows}, expected {expected_rows}")
    summary = {
        "schema": "stable_audio_tools.sceneplan_speech_expansion_manifest_summary",
        "schema_version": 1,
        "dataset_contract_revision": 6,
        "model_sceneplan_schema_version": 2,
        "state": (
            "pilot_ready_for_p8"
            if args.pilot_balanced or args.limit is not None
            else "ready_for_p8"
        ),
        "rows": global_rows,
        "shards": shard_index,
        "temporal_pattern_counts": dict(counts),
        "length_bucket_counts": {str(key): value for key, value in bucket_counts.items()},
        "caption_qwen_tokens": {
            "p99": float(np.percentile(caption_tokens, 99)),
            "max": max(caption_tokens),
            "hard_max": 512,
            "truncated": 0,
        },
        "formal_speech_sources_per_scene_max": 1,
        "spoken_language_background_rows": 0,
        "speech_timing_sidecar": None,
        "word_level_timestamp_teacher": False,
        "index": str(output / "index.parquet"),
        "index_sha256": sha256_file(output / "index.parquet"),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    if args.limit is None and not args.pilot_balanced:
        if bucket_counts != Counter({432: 100_000, 648: 400_000}):
            raise RuntimeError(f"final delta length buckets changed: {bucket_counts}")
        expected_patterns = {
            "direct_speech_only": 75_000,
            "direct_speech_plus_music": 62_500,
            "direct_speech_plus_sound": 62_500,
            **{f"sequential_{key}": value for key, value in SEQUENTIAL_COUNTS.items()},
        }
        if dict(counts) != expected_patterns:
            raise RuntimeError(f"final temporal quotas changed: {counts}")
    if args.pilot_balanced:
        expected_pilot_patterns = {
            "direct_speech_only": 20,
            "direct_speech_plus_music": 20,
            "direct_speech_plus_sound": 20,
            **{f"sequential_{key}": 10 for key in SEQUENTIAL_COUNTS},
        }
        if bucket_counts != Counter({432: 30, 648: 70}):
            raise RuntimeError(f"pilot length buckets changed: {bucket_counts}")
        if dict(counts) != expected_pilot_patterns:
            raise RuntimeError(f"pilot temporal quotas changed: {counts}")
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "READY",
        {
            "schema": "stable_audio_tools.sceneplan_speech_expansion_manifest_ready",
            "schema_version": 1,
            "rows": global_rows,
            "index": str(output / "index.parquet"),
            "p8_started": False,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
