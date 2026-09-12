#!/usr/bin/env python3
"""Build a train-only, one-source Sound delta in the frozen P7.5 contract."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_model_sceneplan_manifests_v1 import (  # noqa: E402
    DATASET_CONTRACT,
    INDEX_SCHEMA,
    MODEL_SCHEMA,
    atomic_jsonl,
    canonical_json,
    model_sceneplan_from_renderer_record,
    render_recipe_from_renderer_record,
    sha256_file,
    sha256_text,
)
from build_sceneplan_manifests_v2 import AssetCycler, plan_scene  # noqa: E402
from sceneplan_v2_common import atomic_write_json  # noqa: E402
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_44_controls,
    compile_model_renderer_caption,
)


SHARD_ROWS = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-parquet", type=Path, required=True)
    parser.add_argument("--registry-parquet", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B"),
    )
    return parser.parse_args()


def load_sources(
    universe_path: Path, registry_path: Path, expected_rows: int
) -> list[dict[str, Any]]:
    universe_rows = pq.read_table(universe_path).to_pylist()
    registry_rows = pq.read_table(registry_path).to_pylist()
    if len(universe_rows) != expected_rows or len(registry_rows) != expected_rows:
        raise RuntimeError("delta universe/registry row count mismatch")
    registry = {
        str(row["source_audio_sha256"]): row for row in registry_rows
    }
    if len(registry) != expected_rows:
        raise RuntimeError("delta registry source hashes are not unique")
    sources: list[dict[str, Any]] = []
    for universe in universe_rows:
        digest = str(universe["source_audio_sha256"])
        annotation = registry.get(digest)
        if annotation is None:
            raise RuntimeError(f"delta registry misses source hash {digest}")
        if (
            universe["kind"] != "sound"
            or universe["split"] != "train"
            or annotation["kind"] != "sound"
            or annotation["split"] != "train"
            or annotation["annotation_id"] != f"sha256:{digest}"
            or annotation["primary_asset_id"] != universe["primary_asset_id"]
            or annotation["audio_path"] != universe["dry_audio_path"]
        ):
            raise RuntimeError(f"delta registry lineage mismatch: {digest}")
        sources.append(
            {
                "asset_id": str(universe["primary_asset_id"]),
                "source_dataset": str(universe["source_dataset"]),
                "kind": "sound",
                "description": str(annotation["source_description"]),
                "model_num_samples": int(universe["model_num_samples"]),
                "source_audio_sha256": digest,
                "native_sample_rate_hz": int(universe["native_sample_rate_hz"]),
                "native_num_samples": int(universe["native_num_samples"]),
                "dry_audio_path": str(
                    Path(universe["dry_audio_path"]).resolve(strict=True)
                ),
                "selection_rank": str(universe["selection_rank"]),
                "spoken_language_background": bool(
                    annotation["spoken_language_background"]
                ),
                "source_description_registry_id": str(
                    annotation["annotation_id"]
                ),
            }
        )
    sources.sort(key=lambda row: (row["selection_rank"], row["source_audio_sha256"]))
    if len({row["source_audio_sha256"] for row in sources}) != expected_rows:
        raise RuntimeError("delta source hashes are not unique after join")
    return sources


def main() -> int:
    args = parse_args()
    if args.expected_rows <= 0:
        raise ValueError("--expected-rows must be positive")
    universe_path = args.universe_parquet.expanduser().resolve(strict=True)
    registry_path = args.registry_parquet.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    tokenizer_root = args.tokenizer_root.expanduser().resolve(strict=True)
    try:
        output_root.relative_to(Path(os.environ.get("AMBIT_DATA_ROOT", "data")))
    except ValueError as error:
        raise ValueError("Sound delta ScenePlans must persist on ${AMBIT_DATA_ROOT}") from error
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    model_schema = MODEL_SCHEMA.resolve(strict=True)
    dataset_contract = DATASET_CONTRACT.resolve(strict=True)
    contract = json.loads(dataset_contract.read_text(encoding="utf-8"))
    if int(contract.get("dataset_contract_revision", -1)) != 5:
        raise RuntimeError("Sound delta requires the revision-5 P7.5 contract")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    caption_max = int(contract["renderer_caption_contract"]["hard_max_qwen_tokens"])
    caption_p99_target = int(
        contract["renderer_caption_contract"]["p99_target_qwen_tokens"]
    )
    sources = load_sources(universe_path, registry_path, args.expected_rows)
    cycler = AssetCycler(sources)
    referenced_hashes: set[str] = set()
    index_path = output_root / "index.parquet"
    index_temporary = index_path.with_name(f".{index_path.name}.{os.getpid()}.tmp")
    index_writer = pq.ParquetWriter(index_temporary, INDEX_SCHEMA, compression="zstd")
    shard_rows: list[dict[str, Any]] = []
    shard_index = 0
    global_rows = 0
    caption_token_counts: list[int] = []
    room_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()
    slot_counts: Counter[int] = Counter()
    started = time.time()

    def flush() -> None:
        nonlocal shard_rows, shard_index
        if not shard_rows:
            return
        captions = [row["caption"]["text"] for row in shard_rows]
        encoded = tokenizer(
            captions,
            add_special_tokens=True,
            truncation=False,
            padding=False,
        )["input_ids"]
        token_counts = [len(ids) for ids in encoded]
        if max(token_counts) > caption_max:
            raise RuntimeError(
                f"delta renderer caption exceeds {caption_max} Qwen tokens"
            )
        caption_token_counts.extend(token_counts)
        stem = f"train-{shard_index:05d}"
        model_path = output_root / "train" / f"model-sceneplans-{stem}.jsonl"
        recipe_path = output_root / "train" / f"render-recipes-{stem}.jsonl"
        conditioning_path = output_root / "train" / f"conditioning-{stem}.jsonl"
        model_offsets = atomic_jsonl(
            model_path,
            [canonical_json(row["model_sceneplan"]) for row in shard_rows],
        )
        recipe_offsets = atomic_jsonl(
            recipe_path,
            [canonical_json(row["render_recipe"]) for row in shard_rows],
        )
        conditioning_offsets = atomic_jsonl(
            conditioning_path,
            [
                canonical_json(
                    {
                        "sample_id": row["sample_id"],
                        "renderer_caption": row["caption"],
                    }
                )
                for row in shard_rows
            ],
        )
        index_rows = []
        for row_index, row in enumerate(shard_rows):
            model_offset, model_length = model_offsets[row_index]
            recipe_offset, recipe_length = recipe_offsets[row_index]
            conditioning_offset, conditioning_length = conditioning_offsets[row_index]
            index_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": "train",
                    "family": "no_speech",
                    "source_count": 1,
                    "room_type": row["model_sceneplan"]["room"]["type"],
                    "model_num_samples": row["model_num_samples"],
                    "latent_frames_valid": row["latent_frames_valid"],
                    "work_shard": shard_index,
                    "row_in_shard": row_index,
                    "sceneplan_path": str(model_path),
                    "sceneplan_byte_offset": model_offset,
                    "sceneplan_byte_length": model_length,
                    "model_sceneplan_sha256": row["model_sceneplan_sha256"],
                    "render_recipe_path": str(recipe_path),
                    "render_recipe_byte_offset": recipe_offset,
                    "render_recipe_byte_length": recipe_length,
                    "render_recipe_sha256": row["render_recipe_sha256"],
                    "conditioning_path": str(conditioning_path),
                    "conditioning_byte_offset": conditioning_offset,
                    "conditioning_byte_length": conditioning_length,
                    "renderer_caption_sha256": row["renderer_caption_sha256"],
                    "caption_qwen_tokens": token_counts[row_index],
                    "speech_asset_id": None,
                    "source_asset_ids": row["source_asset_ids"],
                    "source_kinds": ["sound"],
                }
            )
        index_writer.write_table(pa.Table.from_pylist(index_rows, schema=INDEX_SCHEMA))
        shard_rows = []
        shard_index += 1

    try:
        for source_index in range(args.expected_rows):
            sample_id = f"spv2_delta_vggsound_sound_{source_index:07d}"
            legacy_row = plan_scene(
                sample_id=sample_id,
                split="train",
                family="no_speech",
                source_count=1,
                cell_index=source_index,
                speech_row=None,
                cyclers={"sound": cycler},
                nonspeech_kinds_override=["sound"],
            )
            legacy_record = json.loads(legacy_row["record_json"])
            present = [
                source
                for source in legacy_record["scene_plan"]["sources"]
                if source["present"]
            ]
            if len(present) != 1 or present[0]["kind"] != "sound":
                raise RuntimeError(f"{sample_id}: Sound-only contract changed")
            referenced_hashes.add(str(present[0]["asset_ref"]["identity_hash"]))
            model_sceneplan = model_sceneplan_from_renderer_record(legacy_record)
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
                or controls["source_trajectory_features"].shape != (4, frames, 5)
            ):
                raise RuntimeError(f"{sample_id}: 4+4 frame-control shape drift")
            recipe = render_recipe_from_renderer_record(legacy_record, model_sha)
            source = model_sceneplan["sources"][0]
            room_counts[model_sceneplan["room"]["type"]] += 1
            motion_counts[source["trajectory"]["type"]] += 1
            slot_counts[int(source["source_id"].split("_")[-1])] += 1
            recipe_text = canonical_json(recipe)
            caption_text = canonical_json(caption)
            shard_rows.append(
                {
                    "sample_id": sample_id,
                    "model_num_samples": int(legacy_row["model_num_samples"]),
                    "latent_frames_valid": int(legacy_row["latent_frames_valid"]),
                    "model_sceneplan": model_sceneplan,
                    "model_sceneplan_sha256": model_sha,
                    "render_recipe": recipe,
                    "render_recipe_sha256": sha256_text(recipe_text),
                    "caption": caption,
                    "renderer_caption_sha256": sha256_text(caption_text),
                    "source_asset_ids": legacy_row["source_asset_ids"],
                }
            )
            global_rows += 1
            if len(shard_rows) >= SHARD_ROWS:
                flush()
            if global_rows % 10_000 == 0:
                print(
                    json.dumps(
                        {
                            "planned": global_rows,
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
        flush()
    finally:
        index_writer.close()
    os.replace(index_temporary, index_path)

    expected_hashes = {row["source_audio_sha256"] for row in sources}
    if referenced_hashes != expected_hashes or global_rows != args.expected_rows:
        raise RuntimeError("Sound delta failed exact one-source coverage")
    token_p99 = float(np.percentile(caption_token_counts, 99))
    token_max = max(caption_token_counts)
    if token_p99 > caption_p99_target or token_max > caption_max:
        raise RuntimeError("Sound delta caption envelope failed")
    summary = {
        "schema": "stable_audio_tools.model_sceneplan_sound_delta_build",
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_revision": 5,
        "rows": global_rows,
        "split": "train",
        "family": "no_speech",
        "source_count": 1,
        "unique_source_audio_sha256": len(referenced_hashes),
        "exact_one_scene_per_delta_source": True,
        "shards": shard_index,
        "room_counts": dict(sorted(room_counts.items())),
        "motion_counts": dict(sorted(motion_counts.items())),
        "source_slot_counts": {str(key): value for key, value in sorted(slot_counts.items())},
        "caption_qwen_tokens": {
            "p99": token_p99,
            "p99_target": caption_p99_target,
            "max": token_max,
            "hard_max": caption_max,
            "truncated": 0,
        },
        "universe_parquet": str(universe_path),
        "universe_parquet_sha256": sha256_file(universe_path),
        "source_registry": str(registry_path),
        "source_registry_sha256": sha256_file(registry_path),
        "model_sceneplan_schema": str(model_schema),
        "model_sceneplan_schema_sha256": sha256_file(model_schema),
        "dataset_contract": str(dataset_contract),
        "dataset_contract_sha256": sha256_file(dataset_contract),
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "base_p9_mutated": False,
        "foa_materialization_started": False,
        "latent_materialization_started": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    summary_path = output_root / "summary.json"
    atomic_write_json(summary_path, summary)
    atomic_write_json(
        output_root / "READY",
        {
            "schema": "stable_audio_tools.model_sceneplan_sound_delta_ready",
            "schema_version": 1,
            "rows": global_rows,
            "summary": str(summary_path),
            "index": str(index_path),
            "base_p9_mutated": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
