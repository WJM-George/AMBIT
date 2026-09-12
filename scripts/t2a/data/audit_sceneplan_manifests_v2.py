#!/usr/bin/env python3
"""Fail-closed P6/P7 audit of frozen ScenePlan-v2 manifest shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sceneplan_v2_common import (  # noqa: E402
    expected_quotas,
    CONTRACT_REVISION,
    DATASET_ROOT,
    MAX_LATENT_FRAMES,
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
    atomic_write_json,
    clean_text,
)
from stable_audio_tools.data.sceneplan_v2 import (  # noqa: E402
    compile_442_token_masks,
    compile_renderer_caption,
    compile_structured_source_controls,
)


CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/construct_dataset/"
    "sceneplan_renderer_v2_1p124m.json"
)
SPEECH_LEDGER = DATASET_ROOT / "split_ledgers/speech_v2/speech_split_ledger.parquet"
SIGNAL_CATALOG = (
    DATASET_ROOT / "source_catalog/nonspeech/nonspeech_signal_catalog.parquet"
)
TOKENIZER = Path("/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B")
CONDITIONING_AMENDMENT = (
    REPO_ROOT
    / "docs/sceneplan_v2/sceneplan_conditioning_amendment_v2_512.json"
)
SOURCE_REGISTRY = (
    DATASET_ROOT
    / "source_annotations/nonspeech_instruct_v2/registry/"
    "source_description_registry.parquet"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)




def expected_background_patterns(
    cells: Counter[tuple[str, str, int]],
) -> Counter[tuple[str, str, int, int, int]]:
    output: Counter[tuple[str, str, int, int, int]] = Counter()
    for (split, family, source_count), rows in cells.items():
        background_count = source_count - int(family == "speech")
        for cell_index in range(rows):
            if background_count == 0:
                sound_count, music_count = 0, 0
            elif background_count == 1:
                sound_count = int(cell_index % 2 == 0)
                music_count = 1 - sound_count
            elif cell_index % 4 == 0:
                sound_count, music_count = background_count, 0
            elif cell_index % 4 == 1:
                sound_count, music_count = 0, background_count
            else:
                sound_first = cell_index % 4 == 2
                sound_count = sum(
                    (position % 2 == 0) == sound_first
                    for position in range(background_count)
                )
                music_count = background_count - sound_count
            output[(split, family, source_count, sound_count, music_count)] += 1
    return output


def load_speech_assets(path: Path) -> dict[str, dict[str, Any]]:
    columns = [
        "asset_id",
        "source_dataset",
        "pool",
        "replacement_split",
        "speaker_id",
        "renderer_text",
        "normalized_transcript",
        "parquet_path",
        "row_group",
        "row_in_group",
        "source_audio_sha256",
        "native_sample_rate_hz",
        "native_num_samples",
        "model_num_samples",
        "signal_qc",
        "endpoint_qc",
        "asr_qc",
    ]
    rows = pq.read_table(path, columns=columns).to_pylist()
    assets = {str(row["asset_id"]): row for row in rows}
    require(len(assets) == len(rows), "speech ledger contains duplicate asset ids")
    return assets


def load_nonspeech_assets(path: Path) -> dict[str, dict[str, Any]]:
    columns = [
        "asset_id",
        "kind",
        "dry_audio_path",
        "source_audio_sha256",
        "native_sample_rate_hz",
        "native_num_samples",
        "model_num_samples",
        "eligible",
    ]
    rows = pq.read_table(path, columns=columns, filters=[("eligible", "=", True)]).to_pylist()
    assets = {str(row["asset_id"]): row for row in rows}
    require(len(assets) == len(rows), "nonspeech signal catalog contains duplicate asset ids")
    return assets


def load_source_registry(path: Path) -> dict[str, dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    registry: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_hash = str(row["source_audio_sha256"])
        require(
            row["schema"]
            == "stable_audio_tools.sceneplan_source_description_registry_entry"
            and int(row["schema_version"]) == 1
            and str(row["annotation_id"]) == f"sha256:{source_hash}",
            f"invalid source registry row: {source_hash}",
        )
        require(source_hash not in registry, f"duplicate source registry hash: {source_hash}")
        registry[source_hash] = row
    require(bool(registry), "source description registry is empty")
    return registry


def validate_position(position: dict[str, Any], scene: dict[str, Any], sample_id: str) -> None:
    azimuth = math.radians(float(position["azimuth_deg"]))
    elevation = math.radians(float(position["elevation_deg"]))
    distance = float(position["distance_m"])
    require(distance > 0, f"{sample_id}: non-positive source distance")
    microphone = [float(value) for value in scene["room"]["microphone_xyz_m"]]
    dimensions = [float(value) for value in scene["room"]["dimensions_m"]]
    cosine = math.cos(elevation)
    xyz = [
        microphone[0] + distance * cosine * math.cos(azimuth),
        microphone[1] + distance * cosine * math.sin(azimuth),
        microphone[2] + distance * math.sin(elevation),
    ]
    require(
        all(0.3 <= value <= bound - 0.3 for value, bound in zip(xyz, dimensions)),
        f"{sample_id}: trajectory keyframe outside Pyroom margin: {xyz}",
    )


def validate_token_batch(tokenizer: Any, pending: list[dict[str, Any]], max_tokens: int) -> int:
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
        sample_id = item["sample_id"]
        require(len(ids) <= max_tokens, f"{sample_id}: caption requires {len(ids)} tokens")
        require(
            len(ids) == item["stored_token_count"],
            f"{sample_id}: stored/recomputed Qwen token count mismatch",
        )
        masks = compile_442_token_masks(item["caption"], offsets, attention)
        present = item["present_slots"]
        semantic = masks["source_semantic_token_masks"]
        motion = masks["source_motion_activity_token_masks"]
        for slot in range(4):
            require(
                bool(semantic[slot].sum()) == (slot in present),
                f"{sample_id}: semantic token mask/presence mismatch for slot {slot}",
            )
            require(
                bool(motion[slot].sum()) == (slot in present),
                f"{sample_id}: motion token mask/presence mismatch for slot {slot}",
            )
        has_speech = item["family"] == "speech"
        require(
            bool(masks["speaker_info_token_mask"].sum()) == has_speech,
            f"{sample_id}: speaker token mask/family mismatch",
        )
        require(
            bool(masks["quoted_transcript_token_mask"].sum()) == has_speech,
            f"{sample_id}: transcript token mask/family mismatch",
        )
        require(
            not np.any(
                masks["speaker_info_token_mask"]
                & masks["quoted_transcript_token_mask"]
            ),
            f"{sample_id}: speaker/transcript masks overlap",
        )
        maximum = max(maximum, len(ids))
    pending.clear()
    return maximum


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--sceneplan-root", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--speech-ledger", type=Path, default=SPEECH_LEDGER)
    parser.add_argument("--nonspeech-catalog", type=Path, default=SIGNAL_CATALOG)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument(
        "--conditioning-amendment",
        type=Path,
        default=CONDITIONING_AMENDMENT,
    )
    parser.add_argument("--source-registry", type=Path, default=SOURCE_REGISTRY)
    parser.add_argument("--token-batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = (
        args.sceneplan_root
        or (
            DATASET_ROOT / "pilots/joint_4k/sceneplans"
            if args.mode == "pilot"
            else DATASET_ROOT / "sceneplans"
        )
    ).expanduser().resolve(strict=True)
    output = (
        args.output
        or (
            DATASET_ROOT / "pilots/joint_4k/qc/planned_manifest_audit.json"
            if args.mode == "pilot"
            else DATASET_ROOT / "qc/p7_sceneplan_manifest_audit.json"
        )
    ).expanduser().resolve(strict=False)
    try:
        output.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"audit output must be on SDB: {output}") from error
    config = json.loads(args.config.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    registry_path = args.source_registry.expanduser().resolve(strict=False)
    source_registry = (
        load_source_registry(registry_path.resolve(strict=True))
        if registry_path.is_file()
        else None
    )
    if args.mode == "full":
        require(
            source_registry is not None,
            "revised full audit requires the finalized source registry",
        )
    if source_registry is not None:
        amendment_path = args.conditioning_amendment.expanduser().resolve(strict=True)
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        require(
            int(amendment["conditioning_contract_revision"]) == 2,
            "conditioning amendment revision drift",
        )
        caption_max_tokens = int(amendment["caption"]["max_tokens"])
        caption_p99_target = int(amendment["caption"]["p99_target_tokens"])
    else:
        caption_max_tokens = int(config["conditioning"]["caption_max_tokens"])
        caption_p99_target = caption_max_tokens
    expected = expected_quotas(config, args.mode)
    expected_rows = sum(expected.values())
    speech_assets = load_speech_assets(args.speech_ledger.expanduser().resolve(strict=True))
    nonspeech_assets = load_nonspeech_assets(
        args.nonspeech_catalog.expanduser().resolve(strict=True)
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer.expanduser().resolve(strict=True), local_files_only=True
    )
    index_path = root / "index.parquet"
    index = pq.read_table(index_path)
    require(index.num_rows == expected_rows, "ScenePlan index row count mismatch")
    require(
        pc.count_distinct(index["sample_id"]).as_py() == expected_rows,
        "ScenePlan index sample ids are not unique",
    )
    index_by_id = {str(row["sample_id"]): row for row in index.to_pylist()}
    shards = sorted(root.glob("*/sceneplans-*.parquet"))
    require(bool(shards), "no ScenePlan shards found")
    counts: Counter[tuple[str, str, int]] = Counter()
    room_counts: Counter[str] = Counter()
    room_cell_counts: Counter[tuple[str, str, int, str]] = Counter()
    source_kind_counts: Counter[str] = Counter()
    source_motion_counts: Counter[str] = Counter()
    scene_motion_counts: Counter[str] = Counter()
    background_pattern_counts: Counter[tuple[str, str, int, int, int]] = Counter()
    speech_dataset_counts: Counter[str] = Counter()
    speech_dataset_cell_counts: Counter[tuple[str, int, str]] = Counter()
    seen_sample_ids: set[str] = set()
    speech_ids: set[str] = set()
    speech_hashes: set[str] = set()
    speech_transcripts: set[str] = set()
    speaker_split: dict[str, str] = {}
    nonspeech_hash_split: dict[str, str] = {}
    pending_tokens: list[dict[str, Any]] = []
    token_max = 0
    token_counts: list[int] = []
    registry_hash_reference_counts: Counter[str] = Counter()
    rows_seen = 0
    min_samples = MAX_MODEL_SAMPLES
    max_samples = 0
    min_frames = MAX_LATENT_FRAMES
    max_frames = 0
    structured_samples = 0
    speech_background_targets_db: list[float] = []
    started = time.time()
    for shard_index, path in enumerate(shards, start=1):
        rows = pq.read_table(path).to_pylist()
        require(bool(rows), f"empty ScenePlan shard: {path}")
        jsonl_path = path.with_suffix(".jsonl")
        if source_registry is not None:
            require(jsonl_path.is_file(), f"missing canonical ScenePlan JSONL: {jsonl_path}")
            jsonl_records = [
                line.rstrip("\n")
                for line in jsonl_path.open("r", encoding="utf-8")
                if line.strip()
            ]
            require(
                len(jsonl_records) == len(rows),
                f"ScenePlan JSONL/Parquet row mismatch: {jsonl_path}",
            )
        else:
            jsonl_records = []
        for row_index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            require(sample_id not in seen_sample_ids, f"{sample_id}: duplicate ScenePlan sample id")
            seen_sample_ids.add(sample_id)
            require(sample_id in index_by_id, f"{sample_id}: absent from ScenePlan index")
            index_row = index_by_id[sample_id]
            require(int(row["row_in_shard"]) == row_index, f"{sample_id}: row position drift")
            require(
                int(row["work_shard"]) == int(path.stem.rsplit("-", 1)[-1]),
                f"{sample_id}: work shard/path mismatch",
            )
            require(
                index_row["split"] == row["split"]
                and index_row["family"] == row["family"]
                and int(index_row["source_count"]) == int(row["source_count"])
                and int(index_row["work_shard"]) == int(row["work_shard"])
                and int(index_row["row_in_shard"]) == int(row["row_in_shard"])
                and index_row["shard_path"] == str(path)
                and (
                    source_registry is None
                    or index_row["jsonl_path"] == str(jsonl_path)
                )
                and int(index_row["model_num_samples"]) == int(row["model_num_samples"])
                and int(index_row["latent_frames_valid"])
                == int(row["latent_frames_valid"])
                and index_row["record_sha256"] == row["record_sha256"]
                and index_row["speech_asset_id"] == row["speech_asset_id"],
                f"{sample_id}: ScenePlan index/shard mapping mismatch",
            )
            record_text = str(row["record_json"])
            if source_registry is not None:
                require(
                    jsonl_records[row_index] == record_text,
                    f"{sample_id}: canonical JSONL/Parquet record mismatch",
                )
            record = json.loads(record_text)
            require(record_text == canonical_json(record), f"{sample_id}: record is not canonical JSON")
            require(sha256_json(record) == row["record_sha256"], f"{sample_id}: record SHA mismatch")
            require(record["sample_id"] == sample_id, f"{sample_id}: record sample id mismatch")
            require(record["split"] == row["split"], f"{sample_id}: record split mismatch")
            require(
                record["lineage"]["dataset_contract_version"] == CONTRACT_REVISION,
                f"{sample_id}: contract revision drift",
            )
            require(
                record["lineage"]["spatialization_passes"] == 1,
                f"{sample_id}: spatialization pass count drift",
            )
            target = record["target"]
            require(
                target["materialization_state"] == "planned"
                and target["retention"]
                == ("transient_train_foa" if row["split"] == "train" else "retained_eval_foa")
                and target["foa_path"] is None
                and target["foa_sha256"] is None
                and target["latent_ref"] is None
                and target["latent_sha256"] is None
                and target["vae_encode_seed"] is None
                and target["latent_dtype"] == "float16"
                and int(target["latent_channels"]) == 64,
                f"{sample_id}: planned target state/retention lineage mismatch",
            )
            scene = record["scene_plan"]
            require(sha256_json(scene) == row["sceneplan_sha256"], f"{sample_id}: ScenePlan SHA mismatch")
            audio = scene["audio"]
            num_samples = int(audio["model_num_samples"])
            frames = int(audio["latent_frames_valid"])
            require(
                0 < num_samples <= MAX_MODEL_SAMPLES,
                f"{sample_id}: model length outside contract",
            )
            require(
                frames == math.ceil(num_samples / VAE_HOP_SAMPLES) <= MAX_LATENT_FRAMES,
                f"{sample_id}: valid latent frame count mismatch",
            )
            require(
                int(audio["vae_padded_num_samples"]) == frames * VAE_HOP_SAMPLES,
                f"{sample_id}: VAE padded length mismatch",
            )
            require(
                math.isclose(float(audio["duration_sec"]), num_samples / MODEL_SAMPLE_RATE, abs_tol=1e-12),
                f"{sample_id}: duration/sample count mismatch",
            )
            require(
                int(audio["render_num_samples"]) == num_samples
                and int(audio["render_sample_rate_hz"]) == MODEL_SAMPLE_RATE
                and int(audio["channels"]) == 4,
                f"{sample_id}: render geometry mismatch",
            )
            sources = scene["sources"]
            require(len(sources) == 4, f"{sample_id}: source slot count is not four")
            present = [source for source in sources if source["present"]]
            require(len(present) == int(row["source_count"]), f"{sample_id}: source count mismatch")
            require(
                [source["asset_ref"]["asset_id"] for source in present]
                == row["source_asset_ids"],
                f"{sample_id}: source asset list mismatch",
            )
            require(
                [source["kind"] for source in present] == row["source_kinds"],
                f"{sample_id}: source kind list mismatch",
            )
            require(
                len({source["asset_ref"]["identity_hash"] for source in present}) == len(present),
                f"{sample_id}: duplicate source content within scene",
            )
            speech_sources = [source for source in present if source["kind"] == "speech"]
            family = str(row["family"])
            require(
                len(speech_sources) == (1 if family == "speech" else 0),
                f"{sample_id}: speech family/source invariant failed",
            )
            max_offset = 0
            present_slots: set[int] = set()
            scene_has_dynamic_source = False
            for slot, source in enumerate(sources):
                require(
                    source["slot"] == slot and source["source_id"] == f"source_{slot}",
                    f"{sample_id}: persistent source slot mismatch",
                )
                if not source["present"]:
                    require(
                        source["kind"] == "empty"
                        and source["asset_ref"] is None
                        and not source["activity"],
                        f"{sample_id}: empty slot carries source data",
                    )
                    continue
                present_slots.add(slot)
                source_kind_counts[str(source["kind"])] += 1
                asset = source["asset_ref"]
                require(
                    asset["input_audio_domain"] == "dry_mono"
                    and int(asset["canonical_channels"]) == 1
                    and int(asset["spatialization_passes_before_scene"]) == 0
                    and asset["eligible_as_scene_source"] is True,
                    f"{sample_id}: source is not canonical dry mono",
                )
                require(len(str(asset["identity_hash"])) == 64, f"{sample_id}: invalid source SHA256")
                require(len(source["activity"]) == 1, f"{sample_id}: source activity count drift")
                activity = source["activity"][0]
                onset = int(activity["model_onset_sample"])
                offset = int(activity["model_offset_sample"])
                require(
                    int(activity["dry_start_sample"]) == 0
                    and int(activity["dry_end_sample"]) == offset - onset,
                    f"{sample_id}: source is cropped or activity length changed",
                )
                require(0 <= onset < offset <= num_samples, f"{sample_id}: activity outside scene")
                require(
                    math.isclose(float(activity["onset_sec"]), onset / MODEL_SAMPLE_RATE, abs_tol=1e-12)
                    and math.isclose(float(activity["offset_sec"]), offset / MODEL_SAMPLE_RATE, abs_tol=1e-12),
                    f"{sample_id}: activity seconds/sample mismatch",
                )
                max_offset = max(max_offset, offset)
                motion = source["motion"]
                motion_type = str(motion["type"])
                require(
                    motion_type in {"static", "linear", "keyframed"},
                    f"{sample_id}: unsupported source motion type {motion_type!r}",
                )
                keyframes = motion["keyframes"]
                require(bool(keyframes), f"{sample_id}: missing trajectory")
                coverage = (
                    "static"
                    if motion_type == "static" or len(keyframes) == 1
                    else "dynamic"
                )
                require(
                    (coverage == "static" and motion_type == "static" and len(keyframes) == 1)
                    or (
                        coverage == "dynamic"
                        and motion_type in {"linear", "keyframed"}
                        and len(keyframes) >= 2
                    ),
                    f"{sample_id}: motion type/keyframe count mismatch",
                )
                source_motion_counts[coverage] += 1
                scene_has_dynamic_source = scene_has_dynamic_source or coverage == "dynamic"
                times = [float(item["time_sec"]) for item in keyframes]
                require(
                    all(left < right for left, right in zip(times, times[1:])),
                    f"{sample_id}: trajectory times are not strictly increasing",
                )
                require(
                    times[0] >= onset / MODEL_SAMPLE_RATE - 1e-12
                    and times[-1] <= offset / MODEL_SAMPLE_RATE + 1e-12,
                    f"{sample_id}: trajectory lies outside source activity",
                )
                for keyframe in keyframes:
                    validate_position(keyframe["position"], scene, sample_id)
                asset_id = str(asset["asset_id"])
                if source["kind"] == "speech":
                    require(asset_id in speech_assets, f"{sample_id}: speech asset absent from ledger")
                    ledger = speech_assets[asset_id]
                    expected_pool = "reserve" if args.mode == "pilot" else str(row["split"])
                    require(ledger["pool"] == expected_pool, f"{sample_id}: speech split/pool mismatch")
                    if args.mode == "pilot":
                        require(
                            ledger["replacement_split"] == "train",
                            f"{sample_id}: pilot speech reserve split mismatch",
                        )
                    speech = source["speech"]
                    require(
                        speech["transcript"] == clean_text(ledger["renderer_text"]),
                        f"{sample_id}: transcript is not the canonical complete ledger renderer text",
                    )
                    require(
                        ledger["signal_qc"] == "pass"
                        and ledger["endpoint_qc"] == "pass"
                        and ledger["asr_qc"] == "pass_distil_large_v3",
                        f"{sample_id}: speech donor did not pass full signal/endpoint/ASR QC",
                    )
                    require(
                        asset["identity_hash"] == ledger["source_audio_sha256"]
                        and asset["parquet_path"] == ledger["parquet_path"]
                        and int(asset["row_group"]) == int(ledger["row_group"])
                        and int(asset["row_in_group"]) == int(ledger["row_in_group"]),
                        f"{sample_id}: speech audio lineage mismatch",
                    )
                    require(
                        int(activity["dry_end_sample"]) == int(ledger["model_num_samples"]),
                        f"{sample_id}: complete speech sample count/ledger mismatch",
                    )
                    require(asset_id not in speech_ids, f"{sample_id}: speech donor reused")
                    require(
                        ledger["source_audio_sha256"] not in speech_hashes,
                        f"{sample_id}: speech audio content reused",
                    )
                    require(
                        ledger["normalized_transcript"] not in speech_transcripts,
                        f"{sample_id}: normalized speech transcript reused",
                    )
                    speech_ids.add(asset_id)
                    speech_hashes.add(str(ledger["source_audio_sha256"]))
                    speech_transcripts.add(str(ledger["normalized_transcript"]))
                    speech_dataset_counts[str(ledger["source_dataset"])] += 1
                    speech_dataset_cell_counts[
                        (
                            str(row["split"]),
                            int(row["source_count"]),
                            str(ledger["source_dataset"]),
                        )
                    ] += 1
                    speaker = f"{ledger['source_dataset']}:{ledger['speaker_id']}"
                    previous_split = speaker_split.setdefault(speaker, str(row["split"]))
                    require(previous_split == row["split"], f"{sample_id}: speaker leaks across splits")
                else:
                    require(asset_id in nonspeech_assets, f"{sample_id}: nonspeech asset absent from signal catalog")
                    catalog = nonspeech_assets[asset_id]
                    require(
                        source["kind"] == catalog["kind"]
                        and asset["identity_hash"] == catalog["source_audio_sha256"]
                        and asset["dry_audio_path"] == catalog["dry_audio_path"],
                        f"{sample_id}: nonspeech content lineage mismatch",
                    )
                    require(
                        int(activity["dry_end_sample"]) == int(catalog["model_num_samples"]),
                        f"{sample_id}: complete nonspeech sample count/catalog mismatch",
                    )
                    content_hash = str(asset["identity_hash"])
                    if source_registry is not None:
                        registry = source_registry.get(content_hash)
                        require(
                            registry is not None,
                            f"{sample_id}: nonspeech hash absent from source registry",
                        )
                        require(
                            source["description"] == registry["source_description"]
                            and source["kind"] == registry["kind"]
                            and str(row["split"]) == registry["split"]
                            and asset_id in {
                                str(value) for value in registry["alias_asset_ids"]
                            },
                            f"{sample_id}: source registry semantic/lineage mismatch",
                        )
                        require(
                            family != "speech"
                            or not bool(registry["spoken_language_background"]),
                            f"{sample_id}: formal TTS scene contains spoken-language background",
                        )
                        registry_hash_reference_counts[content_hash] += 1
                    previous_split = nonspeech_hash_split.setdefault(content_hash, str(row["split"]))
                    require(
                        previous_split == row["split"],
                        f"{sample_id}: nonspeech content leaks across splits",
                    )
            scene_motion_counts["dynamic" if scene_has_dynamic_source else "static"] += 1
            require(
                int(audio["render_tail_samples"]) == num_samples - max_offset >= 40,
                f"{sample_id}: declared render tail mismatch/too short",
            )
            backgrounds = [source for source in present if source["kind"] != "speech"]
            sound_count = sum(source["kind"] == "sound" for source in backgrounds)
            music_count = sum(source["kind"] == "music" for source in backgrounds)
            background_pattern_counts[
                (
                    str(row["split"]),
                    family,
                    int(row["source_count"]),
                    sound_count,
                    music_count,
                )
            ] += 1
            if family == "speech" and backgrounds:
                gains = [float(source["gain_db"]) for source in backgrounds]
                require(
                    max(gains) - min(gains) <= 1e-12,
                    f"{sample_id}: planned aggregate background gains are inconsistent",
                )
                target_db = -gains[0] - 5.0 * math.log10(len(backgrounds))
                require(2.0 <= target_db <= 6.0, f"{sample_id}: speech/background target outside 2-6 dB")
                speech_background_targets_db.append(target_db)
            caption = record["renderer_caption"]
            require(
                caption == compile_renderer_caption(scene),
                f"{sample_id}: renderer caption is not deterministic compiler output",
            )
            require(row["caption"] == caption["text"], f"{sample_id}: caption column mismatch")
            if family == "speech":
                speech = speech_sources[0]["speech"]["transcript"]
                region = caption["transcript_regions"][0]
                require(
                    caption["text"][int(region["start"]): int(region["end"])] == speech,
                    f"{sample_id}: caption does not contain the complete exact transcript span",
                )
            pending_tokens.append(
                {
                    "sample_id": sample_id,
                    "caption": caption,
                    "stored_token_count": int(row["qwen_token_count"]),
                    "present_slots": present_slots,
                    "family": family,
                }
            )
            token_counts.append(int(row["qwen_token_count"]))
            if len(pending_tokens) >= args.token_batch_size:
                token_max = max(
                    token_max,
                    validate_token_batch(
                        tokenizer,
                        pending_tokens,
                        caption_max_tokens,
                    ),
                )
            if rows_seen % 10_000 == 0 or rows_seen < 16:
                controls = compile_structured_source_controls(scene)
                activity_masks = controls["source_activity_frame_masks"]
                position_features = controls["source_position_activity_features"]
                expected_present_mask = np.asarray(
                    [slot in present_slots for slot in range(4)], dtype=np.uint8
                )
                require(
                    controls["source_present_mask"].shape == (4,)
                    and activity_masks.shape == (4, frames)
                    and position_features.shape == (4, frames, 8),
                    f"{sample_id}: structured 4-source control shape mismatch",
                )
                require(
                    np.array_equal(controls["source_present_mask"], expected_present_mask),
                    f"{sample_id}: structured source-present binding mismatch",
                )
                require(
                    np.isfinite(position_features).all()
                    and np.array_equal(
                        position_features[..., 0].astype(np.uint8), activity_masks
                    ),
                    f"{sample_id}: position/activity feature binding mismatch",
                )
                require(
                    np.all(position_features[activity_masks == 0] == 0),
                    f"{sample_id}: structured position leaks outside activity",
                )
                structured_samples += 1
            counts[(str(row["split"]), family, int(row["source_count"]))] += 1
            room_counts[str(row["room_class"])] += 1
            room_cell_counts[
                (
                    str(row["split"]),
                    family,
                    int(row["source_count"]),
                    str(row["room_class"]),
                )
            ] += 1
            rows_seen += 1
            min_samples = min(min_samples, num_samples)
            max_samples = max(max_samples, num_samples)
            min_frames = min(min_frames, frames)
            max_frames = max(max_frames, frames)
        if shard_index % 50 == 0 or shard_index == len(shards):
            print(
                json.dumps(
                    {
                        "audited_shards": shard_index,
                        "total_shards": len(shards),
                        "rows": rows_seen,
                        "max_qwen_tokens": token_max,
                        "elapsed_sec": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    token_max = max(
        token_max,
        validate_token_batch(
            tokenizer,
            pending_tokens,
            caption_max_tokens,
        ),
    )
    require(rows_seen == expected_rows, "audited ScenePlan row total mismatch")
    token_p99 = float(
        np.percentile(np.asarray(token_counts, dtype=np.float64), 99)
    )
    require(
        token_max <= caption_max_tokens,
        f"caption hard ceiling exceeded: {token_max} > {caption_max_tokens}",
    )
    if source_registry is not None:
        require(
            token_p99 <= caption_p99_target,
            f"caption p99 target exceeded: {token_p99} > {caption_p99_target}",
        )
    if source_registry is not None:
        require(
            set(registry_hash_reference_counts) == set(source_registry),
            "not every frozen source-registry hash is referenced by revised ScenePlans",
        )
    require(len(seen_sample_ids) == expected_rows, "ScenePlan shard sample-id set mismatch")
    require(counts == expected, f"joint quota mismatch: {counts - expected} / {expected - counts}")
    expected_patterns = expected_background_patterns(expected)
    require(
        background_pattern_counts == expected_patterns,
        "music/sound pattern quotas drift: "
        f"{background_pattern_counts - expected_patterns} / "
        f"{expected_patterns - background_pattern_counts}",
    )
    expected_rooms = Counter(
        {
            (split, family, source_count, room_class): rows // 4
            for (split, family, source_count), rows in expected.items()
            for room_class in ("dry", "moderate", "reverberant", "outdoor")
        }
    )
    require(
        room_cell_counts == expected_rooms,
        f"room-class cell quotas drift: {room_cell_counts - expected_rooms} / "
        f"{expected_rooms - room_cell_counts}",
    )
    require(
        source_motion_counts["static"] > 0
        and source_motion_counts["dynamic"] > 0
        and scene_motion_counts["static"] > 0
        and scene_motion_counts["dynamic"] > 0,
        "static/dynamic source and scene coverage is incomplete",
    )
    expected_speech = sum(value for (split, family, count), value in expected.items() if family == "speech")
    require(len(speech_ids) == expected_speech, "global unique speech count mismatch")
    if args.mode == "full":
        require(
            speech_dataset_counts == {"libritts": 256_000, "hifi_tts": 256_000},
            f"formal speech dataset quotas drift: {speech_dataset_counts}",
        )
        expected_dataset_cells = Counter(
            {
                (split, source_count, dataset): cell_rows // 2
                for (split, family, source_count), cell_rows in expected.items()
                if family == "speech"
                for dataset in ("libritts", "hifi_tts")
            }
        )
        require(
            speech_dataset_cell_counts == expected_dataset_cells,
            "speech corpus/source-count cell balance drift: "
            f"{speech_dataset_cell_counts - expected_dataset_cells} / "
            f"{expected_dataset_cells - speech_dataset_cell_counts}",
        )
    target_median = float(np.median(speech_background_targets_db))
    required_median = 20.0 * math.log10(0.6 / 0.4)
    median_tolerance = 0.1 if args.mode == "pilot" else 0.02
    require(
        abs(target_median - required_median) <= median_tolerance,
        f"speech/background median {target_median} does not match 0.6:0.4 ({required_median})",
    )
    summary = {
        "schema": "stable_audio_tools.sceneplan_manifest_audit",
        "schema_version": 2,
        "ok": True,
        "mode": args.mode,
        "root": str(root),
        "rows": rows_seen,
        "shards": len(shards),
        "joint_counts": {"|".join(map(str, key)): value for key, value in sorted(counts.items())},
        "room_counts": dict(sorted(room_counts.items())),
        "room_cell_counts": {
            "|".join(map(str, key)): value for key, value in sorted(room_cell_counts.items())
        },
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "source_motion_counts": dict(sorted(source_motion_counts.items())),
        "scene_motion_counts": dict(sorted(scene_motion_counts.items())),
        "static_dynamic_coverage": True,
        "background_pattern_counts": {
            "|".join(map(str, key)): value
            for key, value in sorted(background_pattern_counts.items())
        },
        "speech_dataset_counts": dict(sorted(speech_dataset_counts.items())),
        "speech_dataset_cell_counts": {
            "|".join(map(str, key)): value
            for key, value in sorted(speech_dataset_cell_counts.items())
        },
        "globally_unique_speech_assets": len(speech_ids),
        "globally_unique_speech_audio_hashes": len(speech_hashes),
        "globally_unique_normalized_speech_transcripts": len(speech_transcripts),
        "speaker_split_leakage": 0,
        "nonspeech_content_split_leakage": 0,
        "complete_source_no_crop": True,
        "speech_to_aggregate_background_db": {
            "minimum": min(speech_background_targets_db),
            "median": target_median,
            "maximum": max(speech_background_targets_db),
            "required_median": required_median,
        },
        "exact_caption_transcript_spans": True,
        "caption_truncation": False,
        "max_qwen_tokens": token_max,
        "p99_qwen_tokens": token_p99,
        "p99_qwen_tokens_target": caption_p99_target,
        "p99_qwen_tokens_target_ok": token_p99 <= caption_p99_target,
        "caption_max_tokens": caption_max_tokens,
        "source_description_registry_rows": (
            len(source_registry) if source_registry is not None else None
        ),
        "source_description_registry_rows_referenced": (
            len(registry_hash_reference_counts)
            if source_registry is not None
            else None
        ),
        "formal_tts_spoken_language_background_violations": 0,
        "canonical_sceneplan_jsonl_mirrors": source_registry is not None,
        "structured_control_samples_recomputed": structured_samples,
        "model_num_samples_range": [min_samples, max_samples],
        "latent_frames_valid_range": [min_frames, max_frames],
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
