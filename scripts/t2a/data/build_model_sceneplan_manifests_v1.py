#!/usr/bin/env python3
"""Build compact P7.5 ScenePlans, render recipes, and conditioning views.

The revision-4 planner remains the deterministic source allocator and execution
recipe generator.  This script normalizes its output into three deliberately
separate artifacts:

* model ScenePlan JSONL: only state that P10/P11 may consume;
* render-recipe JSONL: asset locators and exact renderer execution values;
* conditioning JSONL: deterministic renderer caption and character regions.

No FOA or latent materialization is performed here, so P8/P9 remain closed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_sceneplan_manifests_v2 import (  # noqa: E402
    CONFIG,
    DEFAULT_PILOT_OUTPUT,
    NONSPEECH_CATALOG,
    SOURCE_DESCRIPTION_REGISTRY,
    SPEAKER_DESCRIPTION_REGISTRY,
    SPEECH_LEDGER,
    SHARD_ROWS,
    load_nonspeech,
    load_speech,
    plan_scene,
    quotas,
)
from sceneplan_v2_common import (  # noqa: E402
    DATASET_ROOT,
    atomic_write_json,
    require_dataset_not_frozen,
)
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_44_controls,
    compile_model_renderer_caption,
    validate_model_sceneplan,
)


MODEL_SCHEMA = REPO_ROOT / "docs/sceneplan_v2/model_sceneplan_v1.schema.json"
DATASET_CONTRACT = (
    REPO_ROOT / "docs/sceneplan_v2/sceneplan_dataset_contract_v5.json"
)
DEFAULT_FULL_OUTPUT = DATASET_ROOT / "sceneplans_model_v1"
DEFAULT_PILOT_OUTPUT_V1 = DEFAULT_PILOT_OUTPUT.parent / "sceneplans_model_v1"


INDEX_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("split", pa.string()),
        ("family", pa.string()),
        ("source_count", pa.int8()),
        ("room_type", pa.string()),
        ("model_num_samples", pa.int32()),
        ("latent_frames_valid", pa.int16()),
        ("work_shard", pa.int32()),
        ("row_in_shard", pa.int16()),
        ("sceneplan_path", pa.string()),
        ("sceneplan_byte_offset", pa.int64()),
        ("sceneplan_byte_length", pa.int32()),
        ("model_sceneplan_sha256", pa.string()),
        ("render_recipe_path", pa.string()),
        ("render_recipe_byte_offset", pa.int64()),
        ("render_recipe_byte_length", pa.int32()),
        ("render_recipe_sha256", pa.string()),
        ("conditioning_path", pa.string()),
        ("conditioning_byte_offset", pa.int64()),
        ("conditioning_byte_length", pa.int32()),
        ("renderer_caption_sha256", pa.string()),
        ("caption_qwen_tokens", pa.int16()),
        ("speech_asset_id", pa.string()),
        ("source_asset_ids", pa.list_(pa.string())),
        ("source_kinds", pa.list_(pa.string())),
    ]
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded(value: Any, digits: int) -> float:
    number = round(float(value), digits)
    return 0.0 if number == 0.0 else number


def compact_position(value: dict[str, Any]) -> dict[str, float]:
    return {
        "azimuth_deg": rounded(value["azimuth_deg"], 3),
        "elevation_deg": rounded(value["elevation_deg"], 3),
        "distance_m": rounded(value["distance_m"], 4),
    }


def model_sceneplan_from_renderer_record(record: dict[str, Any]) -> dict[str, Any]:
    """Project a revision-4 execution record onto the frozen model schema."""

    execution = record["scene_plan"]
    sources: list[dict[str, Any]] = []
    for source in execution["sources"]:
        if not bool(source["present"]):
            continue
        activity = source["activity"]
        if len(activity) != 1:
            raise RuntimeError(f"{record['sample_id']}: expected one activity interval")
        interval = activity[0]
        motion = source["motion"]
        keyframes = motion["keyframes"]
        motion_type = str(motion["type"])
        if motion_type == "static":
            trajectory = {
                "type": "static",
                "position": compact_position(keyframes[0]["position"]),
            }
        elif motion_type == "linear":
            trajectory = {
                "type": "linear",
                "start": compact_position(keyframes[0]["position"]),
                "end": compact_position(keyframes[-1]["position"]),
            }
        elif motion_type == "keyframed":
            trajectory = {
                "type": "keyframed",
                "keyframes": [
                    {
                        "time_sec": rounded(item["time_sec"], 6),
                        "position": compact_position(item["position"]),
                    }
                    for item in keyframes
                ],
            }
        else:
            raise RuntimeError(
                f"{record['sample_id']}: unsupported motion type {motion_type!r}"
            )
        value: dict[str, Any] = {
            "source_id": str(source["source_id"]),
            "kind": str(source["kind"]),
            "activity": {
                "onset_sec": rounded(interval["onset_sec"], 6),
                "offset_sec": rounded(interval["offset_sec"], 6),
            },
            "trajectory": trajectory,
            "gain_db": rounded(source["gain_db"], 4),
        }
        if source["kind"] == "speech":
            value["speaker_description"] = str(
                source["speech"]["speaker_description"]
            )
            value["transcript"] = str(source["speech"]["transcript"])
        else:
            # This is the direct registry response after whitespace compaction.
            value["description"] = str(source["description"])
        sources.append(value)
    sources.sort(key=lambda source: int(str(source["source_id"])[7:]))
    model_sceneplan = {
        "sample_id": str(record["sample_id"]),
        "duration_sec": rounded(execution["audio"]["duration_sec"], 6),
        "room": {"type": str(execution["room"]["class"])},
        "sources": sources,
    }
    validate_model_sceneplan(model_sceneplan)
    return model_sceneplan


def render_recipe_from_renderer_record(
    record: dict[str, Any], model_sceneplan_sha256: str
) -> dict[str, Any]:
    """Keep only per-sample execution state that cannot live in model JSON."""

    execution = record["scene_plan"]
    audio = execution["audio"]
    room = execution["room"]
    sources = []
    for source in execution["sources"]:
        if not bool(source["present"]):
            continue
        interval = source["activity"][0]
        value: dict[str, Any] = {
            "source_id": str(source["source_id"]),
            "kind": str(source["kind"]),
            "asset_ref": source["asset_ref"],
            "exact_source_sample_window": {
                "model_onset_sample": int(interval["model_onset_sample"]),
                "model_offset_sample": int(interval["model_offset_sample"]),
                "dry_start_sample": int(interval["dry_start_sample"]),
                "dry_end_sample": int(interval["dry_end_sample"]),
            },
        }
        if source["kind"] == "speech":
            value["speaker_id"] = str(source["speech"]["speaker_id"])
        sources.append(value)
    sources.sort(key=lambda source: int(str(source["source_id"])[7:]))
    return {
        "schema": "stable_audio_tools.sceneplan_render_recipe",
        "schema_version": 1,
        "sample_id": str(record["sample_id"]),
        "model_sceneplan_sha256": model_sceneplan_sha256,
        "recipe_seed": int(record["lineage"]["recipe_seed"]),
        "audio_execution": {
            "model_num_samples": int(audio["model_num_samples"]),
            "latent_frames_valid": int(audio["latent_frames_valid"]),
            "vae_padded_num_samples": int(audio["vae_padded_num_samples"]),
            "render_tail_samples": int(audio["render_tail_samples"]),
        },
        "resolved_room": {
            "room_id": str(room["room_id"]),
            "dimensions_m": [float(value) for value in room["dimensions_m"]],
            "rt60_sec": float(room["rt60_sec"]),
            "max_order": int(room["max_order"]),
            "microphone_xyz_m": [
                float(value) for value in room["microphone_xyz_m"]
            ],
        },
        "sources": sources,
    }


def atomic_jsonl(path: Path, texts: list[str]) -> list[tuple[int, int]]:
    """Write canonical lines and return byte offset/length pairs."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    offsets: list[tuple[int, int]] = []
    cursor = 0
    with temporary.open("wb") as sink:
        for text in texts:
            encoded = text.encode("utf-8")
            offsets.append((cursor, len(encoded)))
            sink.write(encoded)
            sink.write(b"\n")
            cursor += len(encoded) + 1
        sink.flush()
        os.fsync(sink.fileno())
    if sum(1 for line in temporary.open("rb") if line.strip()) != len(texts):
        raise RuntimeError(f"JSONL reopen count mismatch: {temporary}")
    os.replace(temporary, path)
    return offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--speech-ledger", type=Path, default=SPEECH_LEDGER)
    parser.add_argument("--nonspeech-catalog", type=Path, default=NONSPEECH_CATALOG)
    parser.add_argument(
        "--source-registry", type=Path, default=SOURCE_DESCRIPTION_REGISTRY
    )
    parser.add_argument(
        "--speaker-registry", type=Path, default=SPEAKER_DESCRIPTION_REGISTRY
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--no-tokenizer", action="store_true")
    return parser.parse_args()


def main() -> int:
    require_dataset_not_frozen()
    args = parse_args()
    config_path = args.config.expanduser().resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_schema = MODEL_SCHEMA.resolve(strict=True)
    dataset_contract = DATASET_CONTRACT.resolve(strict=True)
    contract = json.loads(dataset_contract.read_text(encoding="utf-8"))
    if int(contract.get("dataset_contract_revision", -1)) != 5:
        raise RuntimeError("P7.5 requires dataset contract revision 5")
    if str(contract["model_sceneplan"]["schema_sha256"]) != sha256_file(
        model_schema
    ):
        raise RuntimeError("model ScenePlan schema/contract checksum mismatch")
    source_registry_path = args.source_registry.expanduser().resolve(strict=False)
    if args.mode == "full" and not source_registry_path.is_file():
        raise FileNotFoundError(
            f"P7.5 full build requires finalized P7 registry: {source_registry_path}"
        )
    source_registry = (
        source_registry_path.resolve(strict=True)
        if source_registry_path.is_file()
        else None
    )
    speaker_registry_path = args.speaker_registry.expanduser().resolve(strict=False)
    if args.mode == "full" and not speaker_registry_path.is_file():
        raise FileNotFoundError(
            f"P7.5 full build requires finalized speech registry: {speaker_registry_path}"
        )
    speaker_registry = (
        speaker_registry_path.resolve(strict=True)
        if speaker_registry_path.is_file()
        else None
    )
    expected_registry_hashes = (
        {
            str(value)
            for value in pq.read_table(
                source_registry, columns=["source_audio_sha256"]
            ).column("source_audio_sha256").to_pylist()
        }
        if args.mode == "full" and source_registry is not None
        else set()
    )
    referenced_registry_hashes: set[str] = set()
    output = (
        args.output_root
        or (DEFAULT_PILOT_OUTPUT_V1 if args.mode == "pilot" else DEFAULT_FULL_OUTPUT)
    ).expanduser().resolve(strict=False)
    try:
        output.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"P7.5 artifacts must persist on SDB: {output}") from error
    output.mkdir(parents=True, exist_ok=True)
    ready = output / "READY"
    if ready.is_file():
        print(ready.read_text(encoding="utf-8"), end="")
        return 0

    tokenizer = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B",
            local_files_only=True,
        )
    caption_max = int(contract["renderer_caption_contract"]["hard_max_qwen_tokens"])
    caption_p99_target = int(
        contract["renderer_caption_contract"]["p99_target_qwen_tokens"]
    )
    target_quotas = quotas(config, args.mode)
    index_tmp = output / "index.parquet.tmp"
    index_writer = pq.ParquetWriter(index_tmp, INDEX_SCHEMA, compression="zstd")
    counts: Counter[tuple[str, str, int]] = Counter()
    caption_token_counts: list[int] = []
    summaries: dict[str, Any] = {}
    global_rows = 0
    started = time.time()
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
                nonlocal shard_rows, shard_index, split_rows
                if not shard_rows:
                    return
                captions = [row["caption"]["text"] for row in shard_rows]
                if tokenizer is None:
                    token_counts = [0] * len(shard_rows)
                else:
                    encoded = tokenizer(
                        captions,
                        add_special_tokens=True,
                        truncation=False,
                        padding=False,
                    )["input_ids"]
                    token_counts = [len(ids) for ids in encoded]
                    if max(token_counts) > caption_max:
                        offender = shard_rows[token_counts.index(max(token_counts))]
                        raise RuntimeError(
                            f"caption exceeds {caption_max}: "
                            f"{offender['sample_id']}={max(token_counts)}"
                        )
                    caption_token_counts.extend(token_counts)
                stem = f"{split}-{shard_index:05d}"
                sceneplan_path = output / split / f"model-sceneplans-{stem}.jsonl"
                recipe_path = output / split / f"render-recipes-{stem}.jsonl"
                conditioning_path = output / split / f"conditioning-{stem}.jsonl"
                sceneplan_texts = [
                    canonical_json(row["model_sceneplan"]) for row in shard_rows
                ]
                recipe_texts = [
                    canonical_json(row["render_recipe"]) for row in shard_rows
                ]
                conditioning_texts = [
                    canonical_json(
                        {
                            "sample_id": row["sample_id"],
                            "renderer_caption": row["caption"],
                        }
                    )
                    for row in shard_rows
                ]
                sceneplan_offsets = atomic_jsonl(sceneplan_path, sceneplan_texts)
                recipe_offsets = atomic_jsonl(recipe_path, recipe_texts)
                conditioning_offsets = atomic_jsonl(
                    conditioning_path, conditioning_texts
                )
                index_rows = []
                for row_index, row in enumerate(shard_rows):
                    scene_offset, scene_length = sceneplan_offsets[row_index]
                    recipe_offset, recipe_length = recipe_offsets[row_index]
                    condition_offset, condition_length = conditioning_offsets[row_index]
                    index_rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "split": row["split"],
                            "family": row["family"],
                            "source_count": row["source_count"],
                            "room_type": row["model_sceneplan"]["room"]["type"],
                            "model_num_samples": row["model_num_samples"],
                            "latent_frames_valid": row["latent_frames_valid"],
                            "work_shard": shard_index,
                            "row_in_shard": row_index,
                            "sceneplan_path": str(sceneplan_path),
                            "sceneplan_byte_offset": scene_offset,
                            "sceneplan_byte_length": scene_length,
                            "model_sceneplan_sha256": row[
                                "model_sceneplan_sha256"
                            ],
                            "render_recipe_path": str(recipe_path),
                            "render_recipe_byte_offset": recipe_offset,
                            "render_recipe_byte_length": recipe_length,
                            "render_recipe_sha256": row["render_recipe_sha256"],
                            "conditioning_path": str(conditioning_path),
                            "conditioning_byte_offset": condition_offset,
                            "conditioning_byte_length": condition_length,
                            "renderer_caption_sha256": row[
                                "renderer_caption_sha256"
                            ],
                            "caption_qwen_tokens": token_counts[row_index],
                            "speech_asset_id": row["speech_asset_id"],
                            "source_asset_ids": row["source_asset_ids"],
                            "source_kinds": row["source_kinds"],
                        }
                    )
                index_writer.write_table(
                    pa.Table.from_pylist(index_rows, schema=INDEX_SCHEMA)
                )
                split_rows += len(shard_rows)
                shard_index += 1
                shard_rows = []

            for family in ("speech", "no_speech"):
                for source_count in (1, 2, 3, 4):
                    for cell_index in range(int(families[family][source_count])):
                        speech_row = None
                        if family == "speech":
                            if speech_cursor >= len(speech_rows):
                                raise RuntimeError(f"{split} speech ledger exhausted")
                            speech_row = speech_rows[speech_cursor]
                            speech_cursor += 1
                        sample_id = (
                            f"modelpilot_{family}_{source_count}_{cell_index:07d}"
                            if args.mode == "pilot"
                            else f"spv2_{split}_{family}_{source_count}_{cell_index:07d}"
                        )
                        legacy_row = plan_scene(
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
                        legacy_record = json.loads(legacy_row["record_json"])
                        for source in legacy_record["scene_plan"]["sources"]:
                            if bool(source["present"]) and source["kind"] != "speech":
                                referenced_registry_hashes.add(
                                    str(source["asset_ref"]["identity_hash"])
                                )
                        model_sceneplan = model_sceneplan_from_renderer_record(
                            legacy_record
                        )
                        model_text = canonical_json(model_sceneplan)
                        model_sha = sha256_text(model_text)
                        caption = compile_model_renderer_caption(model_sceneplan)
                        controls = compile_model_44_controls(
                            model_sceneplan,
                            model_num_samples=int(legacy_row["model_num_samples"]),
                            latent_frames_valid=int(legacy_row["latent_frames_valid"]),
                        )
                        frames = int(legacy_row["latent_frames_valid"])
                        if (
                            controls["source_event_frame_ids"].shape != (4, frames)
                            or controls["source_trajectory_features"].shape
                            != (4, frames, 5)
                        ):
                            raise RuntimeError(f"{sample_id}: 4+4 control shape drift")
                        render_recipe = render_recipe_from_renderer_record(
                            legacy_record, model_sha
                        )
                        recipe_text = canonical_json(render_recipe)
                        caption_text = canonical_json(caption)
                        shard_rows.append(
                            {
                                "sample_id": sample_id,
                                "split": split,
                                "family": family,
                                "source_count": source_count,
                                "model_num_samples": int(
                                    legacy_row["model_num_samples"]
                                ),
                                "latent_frames_valid": int(
                                    legacy_row["latent_frames_valid"]
                                ),
                                "model_sceneplan": model_sceneplan,
                                "model_sceneplan_sha256": model_sha,
                                "render_recipe": render_recipe,
                                "render_recipe_sha256": sha256_text(recipe_text),
                                "caption": caption,
                                "renderer_caption_sha256": sha256_text(caption_text),
                                "speech_asset_id": legacy_row["speech_asset_id"],
                                "source_asset_ids": legacy_row["source_asset_ids"],
                                "source_kinds": legacy_row["source_kinds"],
                            }
                        )
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
                                        "elapsed_sec": round(time.time() - started, 1),
                                    }
                                ),
                                flush=True,
                            )
            flush()
            expected_speech = sum(families["speech"].values())
            if speech_cursor != expected_speech:
                raise RuntimeError(f"{split} speech consumption mismatch")
            if args.mode == "full" and speech_cursor != len(speech_rows):
                raise RuntimeError(f"{split} did not consume every speech donor")
            summaries[split] = {
                "rows": split_rows,
                "shards": shard_index,
                "speech_donors": speech_cursor,
            }
    finally:
        index_writer.close()
    os.replace(index_tmp, output / "index.parquet")
    expected_rows = sum(
        value
        for families in target_quotas.values()
        for cells in families.values()
        for value in cells.values()
    )
    if global_rows != expected_rows:
        raise RuntimeError(f"planned {global_rows} != expected {expected_rows}")
    if args.mode == "full" and referenced_registry_hashes != expected_registry_hashes:
        missing = expected_registry_hashes - referenced_registry_hashes
        extra = referenced_registry_hashes - expected_registry_hashes
        raise RuntimeError(
            "source registry coverage failed before READY: "
            f"referenced={len(referenced_registry_hashes)} "
            f"expected={len(expected_registry_hashes)} "
            f"missing={len(missing)} extra={len(extra)}"
        )
    token_p99 = (
        float(np.percentile(caption_token_counts, 99))
        if caption_token_counts
        else None
    )
    token_max = max(caption_token_counts) if caption_token_counts else None
    if tokenizer is not None and (
        token_max is None
        or token_max > caption_max
        or token_p99 is None
        or token_p99 > caption_p99_target
    ):
        raise RuntimeError(
            f"renderer caption envelope failed: p99={token_p99}, max={token_max}"
        )
    summary = {
        "schema": "stable_audio_tools.model_sceneplan_manifest_build_summary",
        "schema_version": 1,
        "dataset_contract_revision": 5,
        "mode": args.mode,
        "rows": global_rows,
        "expected_rows": expected_rows,
        "joint_counts": {
            "|".join(map(str, key)): value for key, value in sorted(counts.items())
        },
        "splits": summaries,
        "model_sceneplan_schema": str(model_schema),
        "model_sceneplan_schema_sha256": sha256_file(model_schema),
        "dataset_contract": str(dataset_contract),
        "dataset_contract_sha256": sha256_file(dataset_contract),
        "source_description_registry": (
            str(source_registry) if source_registry is not None else None
        ),
        "source_description_registry_sha256": (
            sha256_file(source_registry) if source_registry is not None else None
        ),
        "source_description_registry_rows_referenced": (
            len(referenced_registry_hashes) if source_registry is not None else None
        ),
        "source_description_registry_exact_coverage": (
            referenced_registry_hashes == expected_registry_hashes
            if args.mode == "full" and source_registry is not None
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
        "caption_qwen_tokens": {
            "p99": token_p99,
            "p99_target": caption_p99_target,
            "max": token_max,
            "hard_max": caption_max,
            "truncated": 0,
        },
        "structured_feature_dim": 9,
        "model_sceneplan_contains_renderer_lineage": False,
        "model_sceneplan_contains_asset_refs": False,
        "render_recipe_is_separate": True,
        "conditioning_is_separate": True,
        "foa_materialization_started": False,
        "latent_materialization_started": False,
        "p8_started": False,
        "p9_started": False,
        "index": str(output / "index.parquet"),
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(
        ready,
        {
            "schema": "stable_audio_tools.model_sceneplan_manifest_ready",
            "schema_version": 1,
            "rows": global_rows,
            "summary": str(output / "summary.json"),
            "index": str(output / "index.parquet"),
            "p8_started": False,
            "p9_started": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
