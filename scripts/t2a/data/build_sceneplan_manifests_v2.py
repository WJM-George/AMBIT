#!/usr/bin/env python3
"""Plan the P6 pilot or all P7 one-shot ScenePlan-v2 records on SDB."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_tts_v2_pilot import room_recipe  # noqa: E402
from sceneplan_v2_common import (  # noqa: E402
    CONTRACT_REVISION,
    DATASET_ROOT,
    MAX_LATENT_FRAMES,
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    VAE_CHECKPOINT_SHA256,
    VAE_HOP_SAMPLES,
    atomic_write_json,
    clean_text,
    deterministic_digest,
    require_dataset_not_frozen,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_v2 import (  # noqa: E402
    compile_renderer_caption,
    compile_structured_source_controls,
)


CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/construct_dataset/"
    "sceneplan_renderer_v2_1p124m.json"
)
CONDITIONING_AMENDMENT = (
    REPO_ROOT
    / "docs/sceneplan_v2/sceneplan_conditioning_amendment_v2_512.json"
)
SPEECH_LEDGER = DATASET_ROOT / "split_ledgers/speech_v2/speech_split_ledger.parquet"
NONSPEECH_CATALOG = (
    DATASET_ROOT / "source_catalog/nonspeech/nonspeech_signal_catalog.parquet"
)
SOURCE_DESCRIPTION_REGISTRY = (
    DATASET_ROOT
    / "source_annotations/nonspeech_instruct_v2/registry/"
    "source_description_registry.parquet"
)
SPEAKER_DESCRIPTION_REGISTRY = (
    DATASET_ROOT
    / "source_annotations/speech_speaker_instruct_v1/registry/"
    "speech_speaker_description_registry.parquet"
)
DEFAULT_FULL_OUTPUT = DATASET_ROOT / "sceneplans"
DEFAULT_PILOT_OUTPUT = DATASET_ROOT / "pilots/joint_4k/sceneplans"
ROOM_CLASSES = ("dry", "moderate", "reverberant", "outdoor")
TAIL_TARGET = {
    "dry": 40 + round(0.05 * MODEL_SAMPLE_RATE),
    "moderate": 40 + round(0.12 * MODEL_SAMPLE_RATE),
    "reverberant": 40 + round(0.25 * MODEL_SAMPLE_RATE),
    "outdoor": 40,
}
SHARD_ROWS = 1024


PARQUET_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("split", pa.string()),
        ("family", pa.string()),
        ("source_count", pa.int8()),
        ("room_class", pa.string()),
        ("model_num_samples", pa.int32()),
        ("latent_frames_valid", pa.int16()),
        ("caption", pa.string()),
        ("qwen_token_count", pa.int16()),
        ("record_json", pa.string()),
        ("record_sha256", pa.string()),
        ("sceneplan_sha256", pa.string()),
        ("recipe_seed", pa.uint64()),
        ("speech_asset_id", pa.string()),
        ("source_asset_ids", pa.list_(pa.string())),
        ("source_kinds", pa.list_(pa.string())),
        ("work_shard", pa.int32()),
        ("row_in_shard", pa.int16()),
    ]
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_description(value: Any, kind: str) -> str:
    """Preserve the registry sentence; only compact whitespace."""
    text = re.sub(r"\s+", " ", str(value or kind)).strip()
    return text or ("instrumental music" if kind == "music" else "a sound event")


def split_bucket(selection_rank: str) -> str:
    value = int(selection_rank[:8], 16) % 100
    if value < 90:
        return "train"
    if value < 98:
        return "validation"
    return "test"


class AssetCycler:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.cursor = 0
        # A candidate rejected only because another source in the *current*
        # scene has the same asset/description must not lose its one-time
        # coverage opportunity.  Keep never-selected rows in a queue until a
        # later compatible scene consumes them; only then enter ordinary
        # cyclic reuse.
        self.unseen = deque(rows)
        if not rows:
            raise RuntimeError("empty source asset cycle")

    def take(
        self,
        excluded_assets: set[str],
        excluded_descriptions: set[str],
        *,
        forbid_spoken_language_background: bool = False,
    ) -> dict[str, Any]:
        def eligible(row: dict[str, Any]) -> bool:
            description_key = clean_description(
                row["description"], row["kind"]
            ).casefold()
            return not (
                row["asset_id"] in excluded_assets
                or description_key in excluded_descriptions
                or (
                    forbid_spoken_language_background
                    and bool(row.get("spoken_language_background", False))
                )
            )

        # Rotate incompatible unseen rows instead of consuming them.  Scene
        # exclusions are ephemeral, so a row skipped here remains eligible
        # for first-use coverage in a later scene.
        for _ in range(len(self.unseen)):
            row = self.unseen.popleft()
            if eligible(row):
                return row
            self.unseen.append(row)

        # Once every row has been selected at least once, reuse deterministically.
        for _ in range(len(self.rows)):
            row = self.rows[self.cursor % len(self.rows)]
            self.cursor += 1
            if eligible(row):
                return row
        raise RuntimeError("could not select a unique nonspeech source within scene")


def load_source_registry(path: Path) -> dict[str, dict[str, Any]]:
    columns = [
        "schema",
        "schema_version",
        "annotation_id",
        "source_audio_sha256",
        "primary_asset_id",
        "alias_asset_ids",
        "split",
        "kind",
        "source_description",
        "spoken_language_background",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_hash = str(row["source_audio_sha256"])
        if (
            row["schema"]
            != "stable_audio_tools.sceneplan_source_description_registry_entry"
            or int(row["schema_version"]) != 1
            or str(row["annotation_id"]) != f"sha256:{source_hash}"
        ):
            raise RuntimeError(f"invalid source registry entry: {source_hash}")
        if source_hash in registry:
            raise RuntimeError(f"duplicate source registry hash: {source_hash}")
        if not str(row["source_description"]).strip():
            raise RuntimeError(f"empty source registry description: {source_hash}")
        registry[source_hash] = row
    if not registry:
        raise RuntimeError(f"empty source description registry: {path}")
    return registry


def load_nonspeech(
    catalog: Path,
    split: str,
    source_registry: Path | None = None,
    family_quotas: dict[str, dict[int, int]] | None = None,
) -> dict[str, AssetCycler] | dict[str, dict[str, AssetCycler]]:
    table = pq.read_table(catalog, filters=[("eligible", "=", True)])
    rows = sorted(table.to_pylist(), key=lambda row: str(row["selection_rank"]))
    registry = load_source_registry(source_registry) if source_registry else None
    by_kind = {"sound": [], "music": []}
    seen_content_hashes: set[str] = set()
    for row in rows:
        content_hash = str(row["source_audio_sha256"] or "")
        if len(content_hash) != 64:
            raise RuntimeError(f"missing nonspeech content SHA256: {row['asset_id']}")
        if registry is not None:
            # The catalog is a superset of the sources referenced by the
            # frozen 1.124M plan universe.  Registry-driven rebuilding must
            # ignore eligible-but-unreferenced catalog candidates, and for a
            # shared audio hash it must select only an alias recorded in the
            # registry lineage.
            entry = registry.get(content_hash)
            if entry is None:
                continue
            if str(row["asset_id"]) not in {
                str(value) for value in entry["alias_asset_ids"]
            }:
                continue
            if content_hash in seen_content_hashes:
                continue
            seen_content_hashes.add(content_hash)
            if str(entry["split"]) != split:
                continue
            if (
                split_bucket(str(row["selection_rank"])) != split
                or str(entry["kind"]) != str(row["kind"])
            ):
                raise RuntimeError(
                    f"source registry lineage mismatch: {content_hash}"
                )
            row = dict(row)
            row["description"] = str(entry["source_description"])
            row["spoken_language_background"] = bool(
                entry["spoken_language_background"]
            )
            row["source_description_registry_id"] = str(entry["annotation_id"])
            by_kind[str(row["kind"])].append(row)
            continue
        if content_hash in seen_content_hashes:
            continue
        seen_content_hashes.add(content_hash)
        if split_bucket(str(row["selection_rank"])) == split:
            row = dict(row)
            row["spoken_language_background"] = False
            by_kind[str(row["kind"])].append(row)
    if registry is not None:
        expected_hashes = {
            source_hash
            for source_hash, entry in registry.items()
            if str(entry["split"]) == split
        }
        actual_hashes = {
            str(row["source_audio_sha256"])
            for values in by_kind.values()
            for row in values
        }
        if actual_hashes != expected_hashes:
            missing = sorted(expected_hashes - actual_hashes)
            extra = sorted(actual_hashes - expected_hashes)
            raise RuntimeError(
                f"registry/catalog split coverage mismatch for {split}: "
                f"missing={missing[:1]} extra={extra[:1]}"
            )
    for kind in by_kind:
        by_kind[kind].sort(key=lambda row: str(row["selection_rank"]))
    if registry is None:
        return {kind: AssetCycler(values) for kind, values in by_kind.items()}
    if family_quotas is None:
        raise RuntimeError("registry-driven source allocation requires family quotas")

    def kind_capacity(family: str, kind: str) -> int:
        capacity = 0
        for source_count, scene_count in family_quotas[family].items():
            background_count = int(source_count) - int(family == "speech")
            cycles, remainder = divmod(int(scene_count), 4)
            for pattern_index in range(4):
                occurrences = cycles + int(pattern_index < remainder)
                capacity += occurrences * nonspeech_pattern(
                    background_count, pattern_index
                ).count(kind)
        return capacity

    allocated: dict[str, dict[str, AssetCycler]] = {
        "speech": {},
        "no_speech": {},
    }
    for kind, values in by_kind.items():
        speech_capacity = kind_capacity("speech", kind)
        no_speech_capacity = kind_capacity("no_speech", kind)
        flagged = [
            row for row in values if bool(row["spoken_language_background"])
        ]
        clean = [
            row for row in values if not bool(row["spoken_language_background"])
        ]
        if len(flagged) > no_speech_capacity:
            raise RuntimeError(
                f"{split}/{kind}: {len(flagged)} spoken-language sources exceed "
                f"{no_speech_capacity} no-speech slots"
            )
        if len(values) > speech_capacity + no_speech_capacity:
            raise RuntimeError(
                f"{split}/{kind}: source registry exceeds total planned slots"
            )
        speech_exclusive_count = max(0, len(values) - no_speech_capacity)
        if speech_exclusive_count > len(clean):
            raise RuntimeError(
                f"{split}/{kind}: not enough clean sources for formal-TTS slots"
            )
        speech_exclusive_ids = {
            str(row["source_audio_sha256"])
            for row in clean[:speech_exclusive_count]
        }
        no_speech_rows = [
            row
            for row in values
            if str(row["source_audio_sha256"]) not in speech_exclusive_ids
        ]
        # Use as many distinct clean rows as the formal-speech capacity permits.
        # Rows after the exclusive prefix may also appear in no-speech scenes;
        # that overlap is exactly the deterministic reuse budget.
        speech_rows = clean[: min(len(clean), speech_capacity)]
        if speech_capacity and not speech_rows:
            raise RuntimeError(f"{split}/{kind}: no clean formal-TTS backgrounds")
        if len(speech_rows) > speech_capacity or len(no_speech_rows) > no_speech_capacity:
            raise RuntimeError(f"{split}/{kind}: family source allocation exceeds capacity")
        allocated["speech"][kind] = AssetCycler(speech_rows)
        allocated["no_speech"][kind] = AssetCycler(no_speech_rows)
    return allocated


@lru_cache(maxsize=2)
def load_speaker_registry(path: Path) -> dict[str, dict[str, Any]]:
    columns = [
        "schema", "schema_version", "asset_id", "source_audio_sha256",
        "source_dataset", "speaker_id", "split", "speaker_profile_key",
        "speaker_identity_description", "delivery_description",
        "speaker_description", "identity_provenance", "delivery_provenance",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        asset_id = str(row["asset_id"])
        if (
            row["schema"]
            != "stable_audio_tools.sceneplan_speech_speaker_registry_entry"
            or int(row["schema_version"]) != 1
            or not str(row["speaker_description"]).strip()
        ):
            raise RuntimeError(f"invalid speech speaker registry entry: {asset_id}")
        if asset_id in registry:
            raise RuntimeError(f"duplicate speech speaker registry asset: {asset_id}")
        registry[asset_id] = row
    if len(registry) != 512_000:
        raise RuntimeError(f"speech speaker registry must have 512000 rows, got {len(registry)}")
    return registry


def load_speech(
    ledger: Path,
    split: str,
    pilot: bool,
    speaker_registry: Path | None = None,
) -> list[dict[str, Any]]:
    if pilot:
        table = pq.read_table(
            ledger,
            filters=[("pool", "=", "reserve"), ("replacement_split", "=", "train")],
        ).sort_by([("selection_rank", "ascending")])
        # Rank-prefix of the mixed reserve is LibriTTS-heavy.  Take 1000 passed
        # rows from each corpus so the 2k speech half of the joint pilot covers
        # both frozen donors without touching the formal 512k pool.
        selected_by_dataset: dict[str, list[dict[str, Any]]] = {
            "libritts": [],
            "hifi_tts": [],
        }
        for row in table.to_pylist():
            dataset = str(row["source_dataset"])
            if dataset not in selected_by_dataset:
                continue
            if len(selected_by_dataset[dataset]) >= 1_000:
                continue
            selected_by_dataset[dataset].append(row)
            if all(len(values) == 1_000 for values in selected_by_dataset.values()):
                break
        if any(len(values) != 1_000 for values in selected_by_dataset.values()):
            raise RuntimeError(
                "pilot reserve-train cannot supply 1000 strong-QC rows per speech corpus: "
                + ", ".join(
                    f"{dataset}={len(values)}"
                    for dataset, values in selected_by_dataset.items()
                )
            )
        selected = selected_by_dataset["libritts"] + selected_by_dataset["hifi_tts"]
    else:
        selected = pq.read_table(ledger, filters=[("pool", "=", split)]).to_pylist()
    # Pool membership remains ledger-authoritative.  Within that frozen pool,
    # interleave the two corpora while keeping each corpus row-group-local.
    # Every source-count cell therefore stays corpus-balanced, and the render
    # workers can reuse a decoded Parquet row group instead of reopening its
    # 10--23 MB payload for every utterance.
    by_dataset: dict[str, list[dict[str, Any]]] = {"libritts": [], "hifi_tts": []}
    for row in selected:
        dataset = str(row["source_dataset"])
        if dataset not in by_dataset:
            raise RuntimeError(f"unapproved speech dataset in ledger: {dataset}")
        by_dataset[dataset].append(row)
    for dataset, values in by_dataset.items():
        groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for row in values:
            key = (str(row["parquet_path"]), int(row["row_group"]))
            groups.setdefault(key, []).append(row)
        for group_rows in groups.values():
            group_rows.sort(
                key=lambda row: (int(row["row_in_group"]), str(row["selection_rank"]))
            )
        ordered_group_keys = sorted(
            groups,
            key=lambda key: deterministic_digest(
                20260814, "speech-render-row-group", split, dataset, key[0], key[1]
            ),
        )
        by_dataset[dataset] = [
            row for key in ordered_group_keys for row in groups[key]
        ]
    rows = []
    for index in range(max(map(len, by_dataset.values()))):
        for dataset in ("libritts", "hifi_tts"):
            values = by_dataset[dataset]
            if index < len(values):
                rows.append(values[index])
    if any(
        row["signal_qc"] != "pass"
        or row["endpoint_qc"] != "pass"
        or row["asr_qc"] != "pass_distil_large_v3"
        for row in rows
    ):
        raise RuntimeError(f"{split} speech pool contains a non-passing strong-QC donor")
    if speaker_registry is not None:
        registry = load_speaker_registry(speaker_registry)
        expected = {str(row["asset_id"]) for row in rows}
        available = {
            asset_id for asset_id, entry in registry.items()
            if str(entry["split"]) == split
        }
        if expected != available:
            raise RuntimeError(
                f"{split} speech speaker registry coverage mismatch: "
                f"expected={len(expected)} available={len(available)}"
            )
        for row in rows:
            entry = registry[str(row["asset_id"])]
            if (
                str(entry["source_audio_sha256"]) != str(row["source_audio_sha256"])
                or str(entry["source_dataset"]) != str(row["source_dataset"])
                or str(entry["speaker_id"]) != str(row["speaker_id"])
            ):
                raise RuntimeError(f"speaker registry lineage mismatch: {row['asset_id']}")
            row.update(
                speaker_description=str(entry["speaker_description"]),
                speaker_identity_description=str(entry["speaker_identity_description"]),
                delivery_description=str(entry["delivery_description"]),
                speaker_profile_key=str(entry["speaker_profile_key"]),
                speaker_identity_provenance=str(entry["identity_provenance"]),
                speaker_delivery_provenance=str(entry["delivery_provenance"]),
            )
    return rows


def nonspeech_pattern(count: int, index: int) -> list[str]:
    if count <= 0:
        return []
    if count == 1:
        return ["sound" if index % 2 == 0 else "music"]
    mode = index % 4
    if mode == 0:
        return ["sound"] * count
    if mode == 1:
        return ["music"] * count
    first = "sound" if mode == 2 else "music"
    second = "music" if first == "sound" else "sound"
    return [first if position % 2 == 0 else second for position in range(count)]


def position(rng: random.Random, room: dict[str, Any]) -> dict[str, float]:
    microphone = [float(value) for value in room["microphone_xyz_m"]]
    dimensions = [float(value) for value in room["dimensions_m"]]
    margin = 0.32
    for _ in range(100):
        candidate = {
            "azimuth_deg": rng.uniform(-180.0, 180.0),
            "elevation_deg": rng.uniform(-22.0, 22.0),
            "distance_m": rng.uniform(0.9, 2.2),
        }
        azimuth = math.radians(candidate["azimuth_deg"])
        elevation = math.radians(candidate["elevation_deg"])
        distance = candidate["distance_m"]
        cosine = math.cos(elevation)
        xyz = [
            microphone[0] + distance * cosine * math.cos(azimuth),
            microphone[1] + distance * cosine * math.sin(azimuth),
            microphone[2] + distance * math.sin(elevation),
        ]
        if all(margin <= value <= bound - margin for value, bound in zip(xyz, dimensions)):
            return candidate
    raise RuntimeError("could not sample a source position inside the Pyroom margin")


def cartesian_midpoint_position(
    start: dict[str, float], stop: dict[str, float], room: dict[str, Any]
) -> dict[str, float]:
    microphone = np.asarray(room["microphone_xyz_m"], dtype=np.float64)

    def xyz(value: dict[str, float]) -> np.ndarray:
        azimuth = math.radians(value["azimuth_deg"])
        elevation = math.radians(value["elevation_deg"])
        distance = value["distance_m"]
        cosine = math.cos(elevation)
        return microphone + distance * np.asarray(
            [cosine * math.cos(azimuth), cosine * math.sin(azimuth), math.sin(elevation)],
            dtype=np.float64,
        )

    delta = (xyz(start) + xyz(stop)) * 0.5 - microphone
    distance = float(np.linalg.norm(delta))
    if distance <= 0:
        raise RuntimeError("dynamic trajectory midpoint collapsed onto the microphone")
    return {
        "azimuth_deg": math.degrees(math.atan2(float(delta[1]), float(delta[0]))),
        "elevation_deg": math.degrees(math.asin(float(delta[2]) / distance)),
        "distance_m": distance,
    }


def motion_plan(
    dynamic: bool,
    onset_sample: int,
    offset_sample: int,
    rng: random.Random,
    room: dict[str, Any],
) -> dict[str, Any]:
    onset = onset_sample / MODEL_SAMPLE_RATE
    offset = offset_sample / MODEL_SAMPLE_RATE
    start = position(rng, room)
    if not dynamic or offset_sample - onset_sample < round(0.40 * MODEL_SAMPLE_RATE):
        return {"type": "static", "keyframes": [{"time_sec": onset, "position": start}]}
    stop = position(rng, room)
    for _ in range(100):
        delta = abs(((stop["azimuth_deg"] - start["azimuth_deg"] + 180.0) % 360.0) - 180.0)
        if delta >= 60.0:
            break
        stop = position(rng, room)
    # Interpolate in room Cartesian coordinates.  The midpoint of two points
    # inside the convex Pyroom margin remains inside it, and this also avoids
    # the +180/-180 degree wrap discontinuity of a raw azimuth average.
    middle = cartesian_midpoint_position(start, stop, room)
    return {
        "type": "linear",
        "keyframes": [
            {"time_sec": onset, "position": start},
            {"time_sec": (onset + offset) * 0.5, "position": middle},
            {"time_sec": offset, "position": stop},
        ],
    }


def speech_source(row: dict[str, Any]) -> dict[str, Any]:
    speaker_description = str(
        row.get("speaker_description") or "an English audiobook narrator"
    ).strip()
    return {
        "kind": "speech",
        "asset_id": str(row["asset_id"]),
        "source_dataset": str(row["source_dataset"]),
        "description": speaker_description,
        "model_num_samples": int(row["model_num_samples"]),
        "asset_ref": {
            "asset_id": str(row["asset_id"]),
            "dataset_id": str(row["source_dataset"]),
            "identity_hash": str(row["source_audio_sha256"]),
            "native_sample_rate_hz": int(row["native_sample_rate_hz"]),
            "native_num_samples": int(row["native_num_samples"]),
            "input_audio_domain": "dry_mono",
            "canonical_channels": 1,
            "spatialization_passes_before_scene": 0,
            "eligible_as_scene_source": True,
            "dry_audio_path": None,
            "parent_asset_id": None,
            "parent_start_sample": None,
            "parent_end_sample": None,
            "segment_method": "full_native_utterance",
            "parquet_path": str(row["parquet_path"]),
            "row_group": int(row["row_group"]),
            "row_in_group": int(row["row_in_group"]),
        },
        "speech": {
            "speaker_id": str(row["speaker_id"]),
            "speaker_description": speaker_description,
            "speaker_identity_description": str(
                row.get("speaker_identity_description") or speaker_description
            ),
            "delivery_description": str(row.get("delivery_description") or ""),
            "speaker_profile_key": row.get("speaker_profile_key"),
            "description_provenance": {
                "identity": row.get("speaker_identity_provenance"),
                "delivery": row.get("speaker_delivery_provenance"),
            },
            # Normalize only punctuation/case-preserving whitespace artifacts
            # before both ScenePlan storage and caption compilation. A small
            # set of HiFiTTS ledger rows retain boundary spaces exposed when
            # their source wrapper quotes were removed.
            "transcript": clean_text(row["renderer_text"]),
            "transcript_normalization": "punctuation_case_whitespace_only_v1",
        },
    }


def nonspeech_source(row: dict[str, Any]) -> dict[str, Any]:
    kind = str(row["kind"])
    return {
        "kind": kind,
        "asset_id": str(row["asset_id"]),
        "source_dataset": str(row["source_dataset"]),
        "description": clean_description(row["description"], kind),
        "model_num_samples": int(row["model_num_samples"]),
        "asset_ref": {
            "asset_id": str(row["asset_id"]),
            "dataset_id": str(row["source_dataset"]),
            "identity_hash": str(row["source_audio_sha256"]),
            "native_sample_rate_hz": int(row["native_sample_rate_hz"]),
            "native_num_samples": int(row["native_num_samples"]),
            "input_audio_domain": "dry_mono",
            "canonical_channels": 1,
            "spatialization_passes_before_scene": 0,
            "eligible_as_scene_source": True,
            "dry_audio_path": str(row["dry_audio_path"]),
            "parent_asset_id": row.get("parent_asset_id"),
            "parent_start_sample": row.get("parent_start_sample"),
            "parent_end_sample": row.get("parent_end_sample"),
            "segment_method": str(
                row.get("segment_method") or "full_native_utterance"
            ),
            "parquet_path": None,
            "row_group": None,
            "row_in_group": None,
        },
        "speech": None,
    }


def plan_scene(
    *,
    sample_id: str,
    split: str,
    family: str,
    source_count: int,
    cell_index: int,
    speech_row: dict[str, Any] | None,
    cyclers: dict[str, AssetCycler],
    nonspeech_kinds_override: list[str] | None = None,
) -> dict[str, Any]:
    recipe_seed = int(deterministic_digest(20260814, "scene", sample_id)[:16], 16)
    rng = random.Random(recipe_seed)
    sources: list[dict[str, Any]] = []
    if family == "speech":
        if speech_row is None:
            raise RuntimeError("speech scene has no speech donor")
        sources.append(speech_source(speech_row))
    elif speech_row is not None:
        raise RuntimeError("no-speech scene received a speech donor")
    background_count = source_count - len(sources)
    kinds = (
        list(nonspeech_kinds_override)
        if nonspeech_kinds_override is not None
        else nonspeech_pattern(background_count, cell_index)
    )
    if len(kinds) != background_count or any(
        kind not in {"sound", "music"} for kind in kinds
    ):
        raise RuntimeError("explicit non-speech kind plan does not match source count")
    excluded_assets = {source["asset_id"] for source in sources}
    excluded_descriptions = {source["description"].casefold() for source in sources}
    for kind in kinds:
        row = cyclers[kind].take(
            excluded_assets,
            excluded_descriptions,
            forbid_spoken_language_background=family == "speech",
        )
        source = nonspeech_source(row)
        sources.append(source)
        excluded_assets.add(source["asset_id"])
        excluded_descriptions.add(source["description"].casefold())
    if len(sources) != source_count:
        raise RuntimeError("planned source count changed")

    room_class = ROOM_CLASSES[cell_index % len(ROOM_CLASSES)]
    room = room_recipe(room_class, recipe_seed ^ 0x9E3779B97F4A7C15)
    longest = max(int(source["model_num_samples"]) for source in sources)
    tail_target = min(TAIL_TARGET[room_class], MAX_MODEL_SAMPLES - longest)
    if tail_target < 40:
        raise RuntimeError("complete source leaves no fixed Pyroom delay tail")
    slack_limit = min(MAX_MODEL_SAMPLES - longest - tail_target, 2 * MODEL_SAMPLE_RATE)
    extra_slack = rng.randint(0, max(0, slack_limit))
    scene_samples = longest + tail_target + extra_slack
    active_capacity = scene_samples - tail_target

    # Persistent slots are not tied to source count or source kind.
    slots = list(range(4))
    rng.shuffle(slots)
    assigned_slots = slots[:source_count]
    scene_dynamic = cell_index % 5 < 3
    speech_dynamic = family == "speech" and cell_index % 5 == 0
    activities = []
    for source_index, source in enumerate(sources):
        length = int(source["model_num_samples"])
        onset = rng.randint(0, max(0, active_capacity - length))
        offset = onset + length
        dynamic = speech_dynamic if source["kind"] == "speech" else (
            scene_dynamic and (source_index == (1 if family == "speech" else 0))
        ) or (rng.random() < 0.20)
        activities.append([onset, offset, dynamic])

    if family == "speech" and background_count:
        speech_index = next(index for index, source in enumerate(sources) if source["kind"] == "speech")
        speech_onset, speech_offset, _ = activities[speech_index]
        minimum_overlap = round(0.10 * MODEL_SAMPLE_RATE)
        overlap = any(
            min(speech_offset, offset) - max(speech_onset, onset) >= minimum_overlap
            for index, (onset, offset, _) in enumerate(activities)
            if index != speech_index
        )
        if not overlap:
            bg_index = next(index for index, source in enumerate(sources) if source["kind"] != "speech")
            length = activities[bg_index][1] - activities[bg_index][0]
            if min(length, speech_offset - speech_onset) < minimum_overlap:
                raise RuntimeError("source is too short for the 100 ms speech/background overlap")
            feasible_low = max(0, speech_onset - length + minimum_overlap)
            feasible_high = min(active_capacity - length, speech_offset - minimum_overlap)
            if feasible_low > feasible_high:
                raise RuntimeError("could not place background with the required speech overlap")
            onset = rng.randint(feasible_low, feasible_high)
            activities[bg_index][0] = onset
            activities[bg_index][1] = onset + length

    source_slots: list[dict[str, Any]] = [
        {
            "source_id": f"source_{slot}",
            "slot": slot,
            "present": False,
            "kind": "empty",
            "description": None,
            "asset_ref": None,
            "gain_db": None,
            "activity": [],
            "motion": None,
            "speech": None,
        }
        for slot in range(4)
    ]
    mix_quantile = int(deterministic_digest(recipe_seed, "mix")[:8], 16) / 0xFFFFFFFF
    median_target_db = 20.0 * math.log10(0.6 / 0.4)
    if mix_quantile <= 0.5:
        target_db = 2.0 + (median_target_db - 2.0) * (mix_quantile / 0.5)
    else:
        target_db = median_target_db + (6.0 - median_target_db) * (
            (mix_quantile - 0.5) / 0.5
        )
    background_gain_db = -target_db - 5.0 * math.log10(max(1, background_count))
    for source, slot, (onset, offset, dynamic) in zip(sources, assigned_slots, activities):
        source_slots[slot] = {
            "source_id": f"source_{slot}",
            "slot": slot,
            "present": True,
            "kind": source["kind"],
            "description": source["description"],
            "asset_ref": source["asset_ref"],
            "gain_db": 0.0 if source["kind"] == "speech" else (
                background_gain_db if family == "speech" else 0.0
            ),
            "activity": [
                {
                    "onset_sec": onset / MODEL_SAMPLE_RATE,
                    "offset_sec": offset / MODEL_SAMPLE_RATE,
                    "model_onset_sample": onset,
                    "model_offset_sample": offset,
                    "dry_start_sample": 0,
                    "dry_end_sample": int(source["model_num_samples"]),
                }
            ],
            "motion": motion_plan(dynamic, onset, offset, rng, room),
            "speech": source["speech"],
        }
    max_offset = max(
        interval["model_offset_sample"]
        for source in source_slots
        for interval in source["activity"]
    )
    render_tail = scene_samples - max_offset
    frames = math.ceil(scene_samples / VAE_HOP_SAMPLES)
    padded = frames * VAE_HOP_SAMPLES
    if not (0 < scene_samples <= MAX_MODEL_SAMPLES and frames <= MAX_LATENT_FRAMES):
        raise RuntimeError("planned variable length violates model ceiling")
    scene_plan = {
        "schema": "stable_audio_tools.spatial_scene_plan",
        "schema_version": 2,
        "audio": {
            "duration_sec": scene_samples / MODEL_SAMPLE_RATE,
            "render_sample_rate_hz": MODEL_SAMPLE_RATE,
            "render_num_samples": scene_samples,
            "model_sample_rate_hz": MODEL_SAMPLE_RATE,
            "model_num_samples": scene_samples,
            "vae_hop_samples": VAE_HOP_SAMPLES,
            "vae_padded_num_samples": padded,
            "latent_frames_valid": frames,
            "render_tail_samples": render_tail,
            "channels": 4,
            "channel_layout": "WYZX_ACN_SN3D",
        },
        "room": room,
        "sources": source_slots,
    }
    caption = compile_renderer_caption(scene_plan)
    controls = compile_structured_source_controls(scene_plan)
    if controls["source_present_mask"].sum() != source_count:
        raise RuntimeError("compiled source-present mask changed source count")
    record = {
        "schema": "stable_audio_tools.sceneplan_renderer_sample",
        "schema_version": 2,
        "sample_id": sample_id,
        "split": split,
        "renderer_caption": caption,
        "scene_plan": scene_plan,
        "target": {
            "materialization_state": "planned",
            "retention": "transient_train_foa" if split == "train" else "retained_eval_foa",
            "foa_path": None,
            "foa_sha256": None,
            "latent_ref": None,
            "latent_sha256": None,
            "vae_encode_seed": None,
            "latent_dtype": "float16",
            "latent_channels": 64,
        },
        "lineage": {
            "dataset_contract_version": CONTRACT_REVISION,
            "recipe_seed": recipe_seed,
            "renderer_version": "sceneplan_v2_renderer.py@revision4",
            "renderer_backend": "pyroomacoustics_single_pass_v2",
            "source_input_domain": "canonical_dry_mono_only",
            "spatialization_passes": 1,
            "foa_normalization": "WYZX_ACN_SN3D",
            "direct_arrival_alignment": "remove_geometric_delay_record_fixed_fractional_filter_delay",
            "residual_algorithmic_delay_samples": 40,
            "tail_policy": "complete_direct_speech_before_declared_room_tail",
            "vae_checkpoint_sha256": VAE_CHECKPOINT_SHA256,
        },
    }
    present_sources = [source for source in source_slots if source["present"]]
    return {
        "sample_id": sample_id,
        "split": split,
        "family": family,
        "source_count": source_count,
        "room_class": room_class,
        "model_num_samples": scene_samples,
        "latent_frames_valid": frames,
        "caption": caption["text"],
        "qwen_token_count": 0,
        "record_json": canonical_json(record),
        "record_sha256": sha256_json(record),
        "sceneplan_sha256": sha256_json(scene_plan),
        "recipe_seed": recipe_seed,
        "speech_asset_id": next(
            (source["asset_ref"]["asset_id"] for source in present_sources if source["kind"] == "speech"),
            None,
        ),
        "source_asset_ids": [source["asset_ref"]["asset_id"] for source in present_sources],
        "source_kinds": [source["kind"] for source in present_sources],
        "work_shard": 0,
        "row_in_shard": 0,
    }


def quotas(config: dict[str, Any], mode: str) -> dict[str, dict[str, dict[int, int]]]:
    if mode == "pilot":
        return {
            "train": {
                "speech": {1: 700, 2: 700, 3: 400, 4: 200},
                "no_speech": {1: 700, 2: 700, 3: 400, 4: 200},
            }
        }
    output = {}
    for split, spec in config["splits"].items():
        output[split] = {
            family: {int(count): int(value) for count, value in cells.items()}
            for family, cells in spec["joint_quotas"].items()
        }
    return output


def atomic_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows, schema=PARQUET_SCHEMA), temporary, compression="zstd")
    reopened = pq.read_table(temporary, columns=["sample_id", "record_sha256"])
    if reopened.num_rows != len(rows):
        raise RuntimeError("atomic ScenePlan shard reopen row count mismatch")
    os.replace(temporary, path)


def atomic_record_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one complete renderer-sample/ScenePlan record per JSONL line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(str(row["record_json"]) + "\n")
        sink.flush()
        os.fsync(sink.fileno())
    with temporary.open("r", encoding="utf-8") as source:
        reopened_rows = sum(1 for line in source if line.strip())
    if reopened_rows != len(rows):
        raise RuntimeError("atomic ScenePlan JSONL reopen row count mismatch")
    os.replace(temporary, path)


def main() -> int:
    require_dataset_not_frozen()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--conditioning-amendment",
        type=Path,
        default=CONDITIONING_AMENDMENT,
    )
    parser.add_argument("--speech-ledger", type=Path, default=SPEECH_LEDGER)
    parser.add_argument("--nonspeech-catalog", type=Path, default=NONSPEECH_CATALOG)
    parser.add_argument(
        "--source-registry",
        type=Path,
        default=SOURCE_DESCRIPTION_REGISTRY,
    )
    parser.add_argument(
        "--speaker-registry",
        type=Path,
        default=SPEAKER_DESCRIPTION_REGISTRY,
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-tokenizer", action="store_true")
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    amendment_path = args.conditioning_amendment.expanduser().resolve(strict=True)
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    if (
        amendment.get("schema")
        != "stable_audio_tools.sceneplan_conditioning_amendment"
        or int(amendment.get("conditioning_contract_revision", -1)) != 2
        or int(amendment.get("base_dataset_contract_revision", -1))
        != CONTRACT_REVISION
    ):
        raise RuntimeError(f"invalid conditioning amendment: {amendment_path}")
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if str(amendment.get("base_build_spec_sha256")) != config_sha256:
        raise RuntimeError("conditioning amendment/base build-spec SHA256 mismatch")
    caption_max_tokens = int(amendment["caption"]["max_tokens"])
    caption_p99_target = int(amendment["caption"]["p99_target_tokens"])
    if caption_max_tokens != 512 or caption_p99_target != 384:
        raise RuntimeError("revised ScenePlan conditioning envelope must be p99<=384/max<=512")
    source_registry = args.source_registry.expanduser().resolve(strict=False)
    if args.mode == "full" and not source_registry.is_file():
        raise FileNotFoundError(
            "full revised ScenePlans require the finalized source registry: "
            f"{source_registry}"
        )
    if source_registry.is_file():
        source_registry = source_registry.resolve(strict=True)
    else:
        source_registry = None
    speaker_registry = args.speaker_registry.expanduser().resolve(strict=False)
    if args.mode == "full" and not speaker_registry.is_file():
        raise FileNotFoundError(
            "full revised ScenePlans require the finalized speech speaker registry: "
            f"{speaker_registry}"
        )
    speaker_registry = (
        speaker_registry.resolve(strict=True) if speaker_registry.is_file() else None
    )
    output = (args.output_root or (DEFAULT_PILOT_OUTPUT if args.mode == "pilot" else DEFAULT_FULL_OUTPUT))
    output = output.expanduser().resolve(strict=False)
    try:
        output.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"ScenePlans must be persisted on SDB: {output}") from error
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "READY"
    if ready.is_file():
        print(ready.read_text(encoding="utf-8"), end="")
        return 0

    tokenizer = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B", local_files_only=True
        )
    target_quotas = quotas(config, args.mode)
    counts: Counter[tuple[Any, ...]] = Counter()
    token_max = 0
    token_counts: list[int] = []
    global_rows = 0
    started = time.time()
    index_writer = pq.ParquetWriter(
        output / "index.parquet.tmp",
        pa.schema(
            [
                ("sample_id", pa.string()),
                ("split", pa.string()),
                ("family", pa.string()),
                ("source_count", pa.int8()),
                ("work_shard", pa.int32()),
                ("row_in_shard", pa.int16()),
                ("shard_path", pa.string()),
                ("jsonl_path", pa.string()),
                ("model_num_samples", pa.int32()),
                ("latent_frames_valid", pa.int16()),
                ("record_sha256", pa.string()),
                ("speech_asset_id", pa.string()),
            ]
        ),
        compression="zstd",
    )
    summaries = {}
    try:
        for split, families in target_quotas.items():
            speech_rows = load_speech(
                args.speech_ledger,
                split,
                args.mode == "pilot",
                speaker_registry if args.mode == "full" else None,
            )
            speech_cursor = 0
            loaded_cyclers = load_nonspeech(
                args.nonspeech_catalog,
                "train" if args.mode == "pilot" else split,
                source_registry,
                families if source_registry is not None else None,
            )
            shard_rows: list[dict[str, Any]] = []
            shard_index = 0
            split_rows = 0

            def flush() -> None:
                nonlocal shard_rows, shard_index, split_rows, token_max
                if not shard_rows:
                    return
                if tokenizer is not None:
                    encoded = tokenizer(
                        [row["caption"] for row in shard_rows],
                        add_special_tokens=True,
                        truncation=False,
                        padding=False,
                    )
                    for row, ids in zip(shard_rows, encoded["input_ids"]):
                        row["qwen_token_count"] = len(ids)
                        token_counts.append(len(ids))
                        token_max = max(token_max, len(ids))
                        if len(ids) > caption_max_tokens:
                            raise RuntimeError(
                                f"caption exceeds Qwen token contract: {row['sample_id']} {len(ids)}"
                            )
                shard_path = output / split / f"sceneplans-{split}-{shard_index:05d}.parquet"
                jsonl_path = output / split / f"sceneplans-{split}-{shard_index:05d}.jsonl"
                for row_index, row in enumerate(shard_rows):
                    row["work_shard"] = shard_index
                    row["row_in_shard"] = row_index
                atomic_parquet(shard_path, shard_rows)
                atomic_record_jsonl(jsonl_path, shard_rows)
                index_rows = [
                    {
                        "sample_id": row["sample_id"],
                        "split": row["split"],
                        "family": row["family"],
                        "source_count": row["source_count"],
                        "work_shard": shard_index,
                        "row_in_shard": row["row_in_shard"],
                        "shard_path": str(shard_path),
                        "jsonl_path": str(jsonl_path),
                        "model_num_samples": row["model_num_samples"],
                        "latent_frames_valid": row["latent_frames_valid"],
                        "record_sha256": row["record_sha256"],
                        "speech_asset_id": row["speech_asset_id"],
                    }
                    for row in shard_rows
                ]
                index_writer.write_table(pa.Table.from_pylist(index_rows, schema=index_writer.schema))
                split_rows += len(shard_rows)
                shard_index += 1
                shard_rows = []

            for family in ("speech", "no_speech"):
                for source_count in (1, 2, 3, 4):
                    count = int(families[family][source_count])
                    for cell_index in range(count):
                        speech_row = None
                        if family == "speech":
                            if speech_cursor >= len(speech_rows):
                                raise RuntimeError(f"{split} speech ledger exhausted")
                            speech_row = speech_rows[speech_cursor]
                            speech_cursor += 1
                        sample_id = (
                            f"jointpilot_{family}_{source_count}_{cell_index:07d}"
                            if args.mode == "pilot"
                            else f"spv2_{split}_{family}_{source_count}_{cell_index:07d}"
                        )
                        row = plan_scene(
                            sample_id=sample_id,
                            split=split,
                            family=family,
                            source_count=source_count,
                            cell_index=cell_index,
                            speech_row=speech_row,
                            cyclers=(
                                loaded_cyclers[family]
                                if source_registry is not None
                                else loaded_cyclers
                            ),
                        )
                        shard_rows.append(row)
                        global_rows += 1
                        counts[(split, family, source_count)] += 1
                        if len(shard_rows) >= SHARD_ROWS:
                            flush()
                        if global_rows % 10_000 == 0:
                            print(
                                json.dumps(
                                    {
                                        "planned": global_rows,
                                        "split": split,
                                        "family": family,
                                        "source_count": source_count,
                                        "token_max": token_max,
                                        "elapsed_sec": round(time.time() - started, 1),
                                    }
                                ),
                                flush=True,
                            )
            flush()
            if speech_cursor != sum(families["speech"].values()):
                raise RuntimeError(f"{split} speech consumption mismatch")
            if args.mode == "full" and speech_cursor != len(speech_rows):
                raise RuntimeError(
                    f"{split} did not consume every formal speech donor: {speech_cursor}/{len(speech_rows)}"
                )
            summaries[split] = {
                "rows": split_rows,
                "shards": shard_index,
                "speech_donors": speech_cursor,
            }
    finally:
        index_writer.close()
    os.replace(output / "index.parquet.tmp", output / "index.parquet")
    expected = sum(
        value
        for families in target_quotas.values()
        for cells in families.values()
        for value in cells.values()
    )
    if global_rows != expected:
        raise RuntimeError(f"planned rows {global_rows} != expected {expected}")
    token_p99 = (
        float(np.percentile(np.asarray(token_counts, dtype=np.float64), 99))
        if token_counts
        else None
    )
    if tokenizer is not None and (
        token_max > caption_max_tokens
        or token_p99 is None
        or token_p99 > caption_p99_target
    ):
        raise RuntimeError(
            f"caption envelope failed: p99={token_p99}/{caption_p99_target} "
            f"max={token_max}/{caption_max_tokens}"
        )
    summary = {
        "schema": "stable_audio_tools.sceneplan_manifest_build_summary",
        "schema_version": 2,
        "contract_revision": CONTRACT_REVISION,
        "mode": args.mode,
        "rows": global_rows,
        "expected_rows": expected,
        "joint_counts": {"|".join(map(str, key)): value for key, value in sorted(counts.items())},
        "splits": summaries,
        "max_qwen_tokens": token_max if tokenizer is not None else None,
        "p99_qwen_tokens": token_p99,
        "p99_qwen_tokens_target": caption_p99_target,
        "caption_max_tokens": caption_max_tokens,
        "caption_truncation": False,
        "speech_reuse": False,
        "speech_render_order": "corpus_interleaved_deterministically_shuffled_row_groups",
        "nonspeech_source_policy": "complete_original_dry_mono_no_crop_split_disjoint",
        "source_description_registry": (
            str(source_registry) if source_registry is not None else None
        ),
        "source_description_registry_sha256": (
            sha256_file(source_registry)
            if source_registry is not None
            else None
        ),
        "speech_speaker_registry": (
            str(speaker_registry) if speaker_registry is not None else None
        ),
        "speech_speaker_registry_sha256": (
            sha256_file(speaker_registry) if speaker_registry is not None else None
        ),
        "speech_speaker_description_is_registry_driven": (
            speaker_registry is not None
        ),
        "formal_tts_spoken_language_background_forbidden": True,
        "index": str(output / "index.parquet"),
        "config_sha256": config_sha256,
        "conditioning_contract_revision": int(
            amendment["conditioning_contract_revision"]
        ),
        "conditioning_amendment": str(amendment_path),
        "conditioning_amendment_sha256": hashlib.sha256(
            amendment_path.read_bytes()
        ).hexdigest(),
        "logical_sceneplan_jsonl": True,
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(
        ready,
        {
            "schema": "stable_audio_tools.sceneplan_manifest_ready",
            "schema_version": 2,
            "mode": args.mode,
            "rows": global_rows,
            "summary": str(output / "summary.json"),
            "index": str(output / "index.parquet"),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
