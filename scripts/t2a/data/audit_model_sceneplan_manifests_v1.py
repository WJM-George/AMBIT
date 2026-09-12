#!/usr/bin/env python3
"""Fail-closed audit for compact P7.5 model ScenePlan manifests."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, BinaryIO

import numpy as np
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from audit_sceneplan_manifests_v2 import (  # noqa: E402
    expected_background_patterns,
    expected_quotas,
    load_nonspeech_assets,
    load_source_registry,
    load_speech_assets,
)
from build_model_sceneplan_manifests_v1 import (  # noqa: E402
    CONFIG,
    DATASET_CONTRACT,
    DEFAULT_FULL_OUTPUT,
    DEFAULT_PILOT_OUTPUT_V1,
    MODEL_SCHEMA,
    NONSPEECH_CATALOG,
    SOURCE_DESCRIPTION_REGISTRY,
    SPEAKER_DESCRIPTION_REGISTRY,
    SPEECH_LEDGER,
    canonical_json,
)
from build_sceneplan_manifests_v2 import load_speaker_registry  # noqa: E402
from sceneplan_v2_common import (  # noqa: E402
    MAX_LATENT_FRAMES,
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
    atomic_write_json,
    clean_text,
)
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_renderer_caption,
    compile_model_structured_controls,
    validate_model_sceneplan,
)
from stable_audio_tools.data.sceneplan_v2 import compile_442_token_masks  # noqa: E402


TOKENIZER = Path("/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class IndexedJsonlReaders:
    """Keep one seekable handle per artifact kind while shards advance."""

    def __init__(self) -> None:
        self.current: dict[str, tuple[Path, BinaryIO]] = {}
        self.last_end: dict[tuple[str, str], int] = {}
        self.last_row: dict[tuple[str, str], int] = {}

    def read(
        self,
        kind: str,
        path_value: str,
        offset: int,
        length: int,
        *,
        row_in_shard: int,
    ) -> str:
        path = Path(path_value).resolve(strict=True)
        state = self.current.get(kind)
        if state is None or state[0] != path:
            if state is not None:
                state[1].close()
            handle = path.open("rb")
            self.current[kind] = (path, handle)
        else:
            handle = state[1]
        key = (kind, str(path))
        expected_offset = self.last_end.get(key, 0)
        require(
            int(offset) == expected_offset,
            f"{path}: non-contiguous JSONL offset {offset} != {expected_offset}",
        )
        expected_row = self.last_row.get(key, -1) + 1
        require(
            int(row_in_shard) == expected_row,
            f"{path}: non-contiguous row index {row_in_shard} != {expected_row}",
        )
        handle.seek(int(offset))
        payload = handle.read(int(length))
        newline = handle.read(1)
        require(len(payload) == int(length) and newline == b"\n", f"{path}: invalid indexed line")
        self.last_end[key] = int(offset) + int(length) + 1
        self.last_row[key] = int(row_in_shard)
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise AssertionError(f"{path}: indexed line is not UTF-8") from error

    def assert_fully_consumed(self) -> None:
        for (_, path_value), indexed_end in self.last_end.items():
            path = Path(path_value)
            require(
                indexed_end == path.stat().st_size,
                f"{path}: {path.stat().st_size - indexed_end} unindexed trailing bytes",
            )

    def close(self) -> None:
        for _, handle in self.current.values():
            handle.close()
        self.current.clear()


def position_values(source: dict[str, Any]) -> list[dict[str, float]]:
    trajectory = source["trajectory"]
    if trajectory["type"] == "static":
        return [trajectory["position"]]
    if trajectory["type"] == "linear":
        return [trajectory["start"], trajectory["end"]]
    return [item["position"] for item in trajectory["keyframes"]]


def validate_position_in_room(
    position: dict[str, Any], room: dict[str, Any], sample_id: str
) -> None:
    azimuth = math.radians(float(position["azimuth_deg"]))
    elevation = math.radians(float(position["elevation_deg"]))
    distance = float(position["distance_m"])
    microphone = [float(value) for value in room["microphone_xyz_m"]]
    dimensions = [float(value) for value in room["dimensions_m"]]
    cosine = math.cos(elevation)
    xyz = [
        microphone[0] + distance * cosine * math.cos(azimuth),
        microphone[1] + distance * cosine * math.sin(azimuth),
        microphone[2] + distance * math.sin(elevation),
    ]
    require(
        all(0.3 <= value <= bound - 0.3 for value, bound in zip(xyz, dimensions)),
        f"{sample_id}: model trajectory is outside resolved room: {xyz}",
    )


def validate_token_batch(
    tokenizer: Any,
    pending: list[dict[str, Any]],
    *,
    hard_max: int,
) -> int:
    if not pending:
        return 0
    encoded = tokenizer(
        [item["caption"]["text"] for item in pending],
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_offsets_mapping=True,
    )
    maximum = 0
    for item, ids, attention, offsets in zip(
        pending,
        encoded["input_ids"],
        encoded["attention_mask"],
        encoded["offset_mapping"],
    ):
        length = len(ids)
        maximum = max(maximum, length)
        require(length == item["stored_tokens"], f"{item['sample_id']}: stored token count drift")
        require(length <= hard_max, f"{item['sample_id']}: caption exceeds {hard_max}")
        masks = compile_442_token_masks(item["caption"], offsets, attention)
        present_slots = item["present_slots"]
        semantic = masks["source_semantic_token_masks"]
        motion = masks["source_motion_activity_token_masks"]
        for slot in range(4):
            require(
                bool(semantic[slot].any()) == (slot in present_slots)
                and bool(motion[slot].any()) == (slot in present_slots),
                f"{item['sample_id']}: 4+4 source mask/slot mismatch",
            )
        has_speech = item["family"] == "speech"
        require(
            bool(masks["speaker_info_token_mask"].any()) == has_speech
            and bool(masks["quoted_transcript_token_mask"].any()) == has_speech,
            f"{item['sample_id']}: 2 speech-mask mismatch",
        )
    pending.clear()
    return maximum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--speech-ledger", type=Path, default=SPEECH_LEDGER)
    parser.add_argument("--nonspeech-catalog", type=Path, default=NONSPEECH_CATALOG)
    parser.add_argument(
        "--source-registry", type=Path, default=SOURCE_DESCRIPTION_REGISTRY
    )
    parser.add_argument(
        "--speaker-registry", type=Path, default=SPEAKER_DESCRIPTION_REGISTRY
    )
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--token-batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.token_batch_size <= 0:
        raise ValueError("--token-batch-size must be positive")
    root = (
        args.root
        or (DEFAULT_PILOT_OUTPUT_V1 if args.mode == "pilot" else DEFAULT_FULL_OUTPUT)
    ).expanduser().resolve(strict=True)
    index_path = (root / "index.parquet").resolve(strict=True)
    config = json.loads(args.config.expanduser().resolve(strict=True).read_text())
    contract_path = DATASET_CONTRACT.resolve(strict=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    schema_path = MODEL_SCHEMA.resolve(strict=True)
    require(
        contract["model_sceneplan"]["schema_sha256"] == sha256_file(schema_path),
        "model schema/dataset contract checksum mismatch",
    )
    expected = expected_quotas(config, args.mode)
    expected_rows = sum(expected.values())
    source_registry_path = args.source_registry.expanduser().resolve(strict=False)
    if args.mode == "full":
        source_registry_path = source_registry_path.resolve(strict=True)
        source_registry = load_source_registry(source_registry_path)
        speaker_registry_path = args.speaker_registry.expanduser().resolve(strict=True)
        speaker_registry = load_speaker_registry(speaker_registry_path)
    else:
        source_registry = None
        speaker_registry = None
    speech_assets = load_speech_assets(args.speech_ledger.expanduser().resolve(strict=True))
    nonspeech_assets = load_nonspeech_assets(
        args.nonspeech_catalog.expanduser().resolve(strict=True)
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer.expanduser().resolve(strict=True), local_files_only=True
    )
    hard_max = int(contract["renderer_caption_contract"]["hard_max_qwen_tokens"])
    p99_target = int(contract["renderer_caption_contract"]["p99_target_qwen_tokens"])
    readers = IndexedJsonlReaders()
    pending_tokens: list[dict[str, Any]] = []
    token_counts: list[int] = []
    token_max = 0
    counts: Counter[tuple[str, str, int]] = Counter()
    room_cell_counts: Counter[tuple[str, str, int, str]] = Counter()
    background_patterns: Counter[tuple[str, str, int, int, int]] = Counter()
    source_kind_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()
    registry_references: Counter[str] = Counter()
    speech_ids: set[str] = set()
    speech_hashes: set[str] = set()
    speech_transcripts: set[str] = set()
    sample_ids: set[str] = set()
    speech_dataset_counts: Counter[str] = Counter()
    speaker_descriptions: Counter[str] = Counter()
    speaker_profile_keys: set[str] = set()
    formal_tts_background_violations = 0
    structured_recomputed = 0
    rows_seen = 0
    started = time.time()
    parquet = pq.ParquetFile(index_path)
    try:
        for batch in parquet.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                sample_id = str(row["sample_id"])
                require(sample_id not in sample_ids, f"{sample_id}: duplicate sample id")
                sample_ids.add(sample_id)
                row_in_shard = int(row["row_in_shard"])
                scene_text = readers.read(
                    "sceneplan",
                    row["sceneplan_path"],
                    int(row["sceneplan_byte_offset"]),
                    int(row["sceneplan_byte_length"]),
                    row_in_shard=row_in_shard,
                )
                recipe_text = readers.read(
                    "recipe",
                    row["render_recipe_path"],
                    int(row["render_recipe_byte_offset"]),
                    int(row["render_recipe_byte_length"]),
                    row_in_shard=row_in_shard,
                )
                conditioning_text = readers.read(
                    "conditioning",
                    row["conditioning_path"],
                    int(row["conditioning_byte_offset"]),
                    int(row["conditioning_byte_length"]),
                    row_in_shard=row_in_shard,
                )
                scene = json.loads(scene_text)
                recipe = json.loads(recipe_text)
                conditioning = json.loads(conditioning_text)
                require(scene_text == canonical_json(scene), f"{sample_id}: non-canonical ScenePlan JSON")
                require(recipe_text == canonical_json(recipe), f"{sample_id}: non-canonical recipe JSON")
                require(
                    conditioning_text == canonical_json(conditioning),
                    f"{sample_id}: non-canonical conditioning JSON",
                )
                validate_model_sceneplan(scene)
                require(scene["sample_id"] == sample_id, f"{sample_id}: ScenePlan id mismatch")
                scene_sha = sha256_text(scene_text)
                require(
                    scene_sha == row["model_sceneplan_sha256"]
                    and recipe["model_sceneplan_sha256"] == scene_sha,
                    f"{sample_id}: ScenePlan hash binding mismatch",
                )
                require(
                    sha256_text(recipe_text) == row["render_recipe_sha256"],
                    f"{sample_id}: render recipe hash mismatch",
                )
                require(
                    set(recipe)
                    == {
                        "schema",
                        "schema_version",
                        "sample_id",
                        "model_sceneplan_sha256",
                        "recipe_seed",
                        "audio_execution",
                        "resolved_room",
                        "sources",
                    }
                    and "lineage" not in recipe,
                    f"{sample_id}: render recipe separation changed",
                )
                require(
                    recipe["schema"] == "stable_audio_tools.sceneplan_render_recipe"
                    and int(recipe["schema_version"]) == 1
                    and recipe["sample_id"] == sample_id,
                    f"{sample_id}: invalid render recipe header",
                )
                audio = recipe["audio_execution"]
                model_num_samples = int(audio["model_num_samples"])
                frames = int(audio["latent_frames_valid"])
                require(
                    model_num_samples == int(row["model_num_samples"])
                    and 0 < model_num_samples <= MAX_MODEL_SAMPLES
                    and frames == int(row["latent_frames_valid"])
                    == math.ceil(model_num_samples / VAE_HOP_SAMPLES)
                    <= MAX_LATENT_FRAMES,
                    f"{sample_id}: model/latent length mismatch",
                )
                require(
                    math.isclose(
                        float(scene["duration_sec"]),
                        model_num_samples / MODEL_SAMPLE_RATE,
                        abs_tol=1.1e-6,
                    ),
                    f"{sample_id}: compact duration/sample mismatch",
                )
                require(
                    conditioning["sample_id"] == sample_id,
                    f"{sample_id}: conditioning id mismatch",
                )
                caption = conditioning["renderer_caption"]
                require(
                    caption == compile_model_renderer_caption(scene),
                    f"{sample_id}: renderer caption compiler mismatch",
                )
                require(
                    sha256_text(canonical_json(caption))
                    == row["renderer_caption_sha256"],
                    f"{sample_id}: caption hash mismatch",
                )
                family = str(row["family"])
                sources = scene["sources"]
                recipe_sources = recipe["sources"]
                require(
                    len(sources) == int(row["source_count"]) == len(recipe_sources),
                    f"{sample_id}: source count mismatch",
                )
                recipe_by_id = {
                    str(source["source_id"]): source for source in recipe_sources
                }
                require(
                    list(recipe_by_id) == [source["source_id"] for source in sources],
                    f"{sample_id}: recipe/model persistent-source mismatch",
                )
                speech_sources = [source for source in sources if source["kind"] == "speech"]
                require(
                    len(speech_sources) == (1 if family == "speech" else 0),
                    f"{sample_id}: family/formal-speech mismatch",
                )
                present_slots = {
                    int(str(source["source_id"])[7:]) for source in sources
                }
                pending_tokens.append(
                    {
                        "sample_id": sample_id,
                        "caption": caption,
                        "stored_tokens": int(row["caption_qwen_tokens"]),
                        "present_slots": present_slots,
                        "family": family,
                    }
                )
                token_counts.append(int(row["caption_qwen_tokens"]))
                if len(pending_tokens) >= args.token_batch_size:
                    token_max = max(
                        token_max,
                        validate_token_batch(
                            tokenizer, pending_tokens, hard_max=hard_max
                        ),
                    )
                resolved_room = recipe["resolved_room"]
                for source in sources:
                    source_id = str(source["source_id"])
                    execution_source = recipe_by_id[source_id]
                    require(
                        execution_source["kind"] == source["kind"],
                        f"{sample_id}: recipe/model kind mismatch",
                    )
                    source_kind_counts[str(source["kind"])] += 1
                    motion_counts[
                        "static"
                        if source["trajectory"]["type"] == "static"
                        else "dynamic"
                    ] += 1
                    for position in position_values(source):
                        validate_position_in_room(position, resolved_room, sample_id)
                    asset = execution_source["asset_ref"]
                    asset_id = str(asset["asset_id"])
                    window = execution_source["exact_source_sample_window"]
                    onset_sample = int(window["model_onset_sample"])
                    offset_sample = int(window["model_offset_sample"])
                    require(
                        int(window["dry_start_sample"]) == 0
                        and int(window["dry_end_sample"]) == offset_sample - onset_sample
                        and 0 <= onset_sample < offset_sample <= model_num_samples,
                        f"{sample_id}: exact execution window mismatch",
                    )
                    require(
                        math.isclose(
                            float(source["activity"]["onset_sec"]),
                            onset_sample / MODEL_SAMPLE_RATE,
                            abs_tol=1.1e-6,
                        )
                        and math.isclose(
                            float(source["activity"]["offset_sec"]),
                            offset_sample / MODEL_SAMPLE_RATE,
                            abs_tol=1.1e-6,
                        ),
                        f"{sample_id}: model/execution activity mismatch",
                    )
                    if source["kind"] == "speech":
                        require(asset_id in speech_assets, f"{sample_id}: unknown speech asset")
                        ledger = speech_assets[asset_id]
                        expected_pool = "reserve" if args.mode == "pilot" else str(row["split"])
                        require(ledger["pool"] == expected_pool, f"{sample_id}: speech pool mismatch")
                        require(
                            source["transcript"] == clean_text(ledger["renderer_text"])
                            and str(execution_source["speaker_id"])
                            == str(ledger["speaker_id"]),
                            f"{sample_id}: authoritative speech metadata mismatch",
                        )
                        if speaker_registry is not None:
                            speaker_entry = speaker_registry.get(asset_id)
                            require(
                                speaker_entry is not None
                                and source["speaker_description"]
                                == speaker_entry["speaker_description"]
                                and str(speaker_entry["source_audio_sha256"])
                                == str(ledger["source_audio_sha256"]),
                                f"{sample_id}: speaker-description registry mismatch",
                            )
                            speaker_descriptions[str(source["speaker_description"])] += 1
                            speaker_profile_keys.add(
                                str(speaker_entry["speaker_profile_key"])
                            )
                        require(
                            int(window["dry_end_sample"])
                            == int(ledger["model_num_samples"]),
                            f"{sample_id}: incomplete speech donor",
                        )
                        require(asset_id not in speech_ids, f"{sample_id}: speech donor reused")
                        require(
                            str(ledger["source_audio_sha256"]) not in speech_hashes
                            and str(ledger["normalized_transcript"])
                            not in speech_transcripts,
                            f"{sample_id}: speech hash/transcript reused",
                        )
                        speech_ids.add(asset_id)
                        speech_hashes.add(str(ledger["source_audio_sha256"]))
                        speech_transcripts.add(str(ledger["normalized_transcript"]))
                        speech_dataset_counts[str(ledger["source_dataset"])] += 1
                    else:
                        require(asset_id in nonspeech_assets, f"{sample_id}: unknown nonspeech asset")
                        catalog = nonspeech_assets[asset_id]
                        source_hash = str(asset["identity_hash"])
                        require(
                            source["kind"] == catalog["kind"]
                            and source_hash == str(catalog["source_audio_sha256"])
                            and int(window["dry_end_sample"])
                            == int(catalog["model_num_samples"]),
                            f"{sample_id}: nonspeech lineage mismatch",
                        )
                        if source_registry is not None:
                            registry = source_registry.get(source_hash)
                            require(registry is not None, f"{sample_id}: source absent from registry")
                            require(
                                source["description"] == registry["source_description"]
                                and source["kind"] == registry["kind"]
                                and str(row["split"]) == registry["split"]
                                and asset_id in {str(value) for value in registry["alias_asset_ids"]},
                                f"{sample_id}: registry description/lineage mismatch",
                            )
                            violation = family == "speech" and bool(
                                registry["spoken_language_background"]
                            )
                            formal_tts_background_violations += int(violation)
                            require(not violation, f"{sample_id}: TTS/spoken-background violation")
                            registry_references[source_hash] += 1
                if rows_seen < 16 or rows_seen % 10_000 == 0:
                    structured = compile_model_structured_controls(
                        scene,
                        model_num_samples=model_num_samples,
                        latent_frames_valid=frames,
                    )
                    features = structured[
                        "source_position_activity_gain_features"
                    ]
                    activity = structured["source_activity_frame_masks"]
                    require(
                        features.shape == (4, frames, 9)
                        and activity.shape == (4, frames)
                        and np.isfinite(features).all()
                        and np.array_equal(
                            features[..., 0].astype(np.uint8), activity
                        )
                        and np.all(features[activity == 0] == 0),
                        f"{sample_id}: structured control audit failed",
                    )
                    structured_recomputed += 1
                split = str(row["split"])
                source_count = int(row["source_count"])
                counts[(split, family, source_count)] += 1
                room_type = str(scene["room"]["type"])
                require(room_type == row["room_type"], f"{sample_id}: room index mismatch")
                room_cell_counts[(split, family, source_count, room_type)] += 1
                backgrounds = [source for source in sources if source["kind"] != "speech"]
                sound_count = sum(source["kind"] == "sound" for source in backgrounds)
                music_count = len(backgrounds) - sound_count
                background_patterns[(split, family, source_count, sound_count, music_count)] += 1
                require(
                    [source["asset_ref"]["asset_id"] for source in recipe_sources]
                    == row["source_asset_ids"]
                    and [source["kind"] for source in sources] == row["source_kinds"],
                    f"{sample_id}: source index columns mismatch",
                )
                rows_seen += 1
            if rows_seen and rows_seen % 50_000 < len(batch):
                print(
                    json.dumps(
                        {
                            "audited_rows": rows_seen,
                            "expected_rows": expected_rows,
                            "max_caption_tokens": token_max,
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
        readers.assert_fully_consumed()
    finally:
        readers.close()
    token_max = max(
        token_max,
        validate_token_batch(tokenizer, pending_tokens, hard_max=hard_max),
    )
    require(rows_seen == expected_rows, f"row count {rows_seen} != {expected_rows}")
    require(counts == expected, f"joint quota mismatch: {counts - expected}/{expected - counts}")
    require(
        background_patterns == expected_background_patterns(expected),
        "music/sound pattern quota mismatch",
    )
    expected_rooms = Counter(
        {
            (split, family, source_count, room_type): rows // 4
            for (split, family, source_count), rows in expected.items()
            for room_type in ("dry", "moderate", "reverberant", "outdoor")
        }
    )
    require(room_cell_counts == expected_rooms, "room quota mismatch")
    require(motion_counts["static"] > 0 and motion_counts["dynamic"] > 0, "motion coverage incomplete")
    expected_speech = sum(
        value for (split, family, count), value in expected.items() if family == "speech"
    )
    require(len(speech_ids) == expected_speech, "formal speech uniqueness mismatch")
    if args.mode == "full":
        require(
            speech_dataset_counts == {"libritts": 256_000, "hifi_tts": 256_000},
            f"speech corpus quota mismatch: {speech_dataset_counts}",
        )
        require(
            len(speaker_descriptions) > 1_000
            and speaker_descriptions["an English audiobook narrator"] == 0
            and len(speaker_profile_keys) >= 2_452,
            "speech speaker descriptions are not sufficiently diverse/complete",
        )
        require(
            set(registry_references) == set(source_registry),
            "not every finalized source registry row is referenced",
        )
    p99 = float(np.percentile(np.asarray(token_counts, dtype=np.float64), 99))
    require(token_max <= hard_max and p99 <= p99_target, "renderer caption token envelope failed")
    require(formal_tts_background_violations == 0, "formal TTS background gate failed")
    output = (
        args.output.expanduser().resolve(strict=False)
        if args.output
        else root / "audit.json"
    )
    summary = {
        "schema": "stable_audio_tools.model_sceneplan_manifest_audit",
        "schema_version": 1,
        "ok": True,
        "dataset_contract_revision": 5,
        "mode": args.mode,
        "root": str(root),
        "rows": rows_seen,
        "joint_counts": {"|".join(map(str, key)): value for key, value in sorted(counts.items())},
        "room_cell_counts": {"|".join(map(str, key)): value for key, value in sorted(room_cell_counts.items())},
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "motion_counts": dict(sorted(motion_counts.items())),
        "globally_unique_speech_assets": len(speech_ids),
        "globally_unique_speech_audio_hashes": len(speech_hashes),
        "globally_unique_normalized_speech_transcripts": len(speech_transcripts),
        "source_description_registry_rows": len(source_registry) if source_registry is not None else None,
        "source_description_registry_rows_referenced": len(registry_references) if source_registry is not None else None,
        "speech_speaker_registry_rows": len(speaker_registry) if speaker_registry is not None else None,
        "speech_speaker_registry_rows_referenced": len(speech_ids) if speaker_registry is not None else None,
        "unique_speaker_descriptions": len(speaker_descriptions) if speaker_registry is not None else None,
        "unique_speaker_profile_keys": len(speaker_profile_keys) if speaker_registry is not None else None,
        "constant_generic_speaker_description_rows": (
            speaker_descriptions["an English audiobook narrator"]
            if speaker_registry is not None else None
        ),
        "formal_tts_spoken_language_background_violations": formal_tts_background_violations,
        "caption_tokens": {
            "p99": p99,
            "p99_target": p99_target,
            "max": token_max,
            "hard_max": hard_max,
            "truncated": 0,
        },
        "caption_character_spans_and_4_4_2_all_rows": True,
        "structured_feature_dim": 9,
        "structured_controls_recomputed": structured_recomputed,
        "model_sceneplan_contains_renderer_lineage": False,
        "model_sceneplan_contains_asset_refs": False,
        "render_recipe_separate": True,
        "foa_materialization_started": False,
        "latent_materialization_started": False,
        "p8_started": False,
        "p9_started": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
