#!/usr/bin/env python3
"""Migrate frozen P7.5 speaker text while preserving every non-text plan field."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_model_sceneplan_manifests_v1 import (
    DATASET_CONTRACT, INDEX_SCHEMA, canonical_json, atomic_jsonl,
)
from build_sceneplan_manifests_v2 import (
    SPEAKER_DESCRIPTION_REGISTRY, load_speaker_registry,
)
from sceneplan_v2_common import DATASET_ROOT, atomic_write_json, require_dataset_not_frozen
from stable_audio_tools.data.model_sceneplan import (
    compile_model_renderer_caption, validate_model_sceneplan,
)


DEFAULT_OLD_ROOT = (
    DATASET_ROOT / "audit/superseded_speaker_constant_20260819_0924/sceneplans_model_v1"
)
DEFAULT_NEW_ROOT = DATASET_ROOT / "sceneplans_model_v1"


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(raw)
        if canonical_json(value) != raw:
            raise RuntimeError(f"noncanonical JSONL: {path}:{number}")
        result.append(value)
    return result


def neutral_scene(value: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(canonical_json(value))
    for source in copied["sources"]:
        if source["kind"] == "speech":
            source["speaker_description"] = "<speaker-description>"
    return copied


def main() -> int:
    require_dataset_not_frozen()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--speaker-registry", type=Path, default=SPEAKER_DESCRIPTION_REGISTRY)
    args = parser.parse_args()
    old_root = args.old_root.expanduser().resolve(strict=True)
    new_root = args.new_root.expanduser().resolve(strict=False)
    registry_path = args.speaker_registry.expanduser().resolve(strict=True)
    if new_root.exists() and any(new_root.iterdir()):
        raise RuntimeError(f"new P7.5 root is not empty: {new_root}")
    new_root.mkdir(parents=True, exist_ok=True)
    registry = load_speaker_registry(registry_path)
    contract = json.loads(DATASET_CONTRACT.read_text(encoding="utf-8"))
    hard_max = int(contract["renderer_caption_contract"]["hard_max_qwen_tokens"])
    p99_target = int(contract["renderer_caption_contract"]["p99_target_qwen_tokens"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B", local_files_only=True
    )
    old_index = pq.ParquetFile(old_root / "index.parquet")
    shard_files = [
        path
        for split in ("train", "validation", "test")
        for path in sorted((old_root / split).glob("model-sceneplans-*.jsonl"))
    ]
    if len(shard_files) != 1_099 or old_index.num_row_groups != len(shard_files):
        raise RuntimeError("old P7.5 shard/index layout is not the frozen 1099-shard layout")
    writer = pq.ParquetWriter(new_root / "index.parquet.tmp", INDEX_SCHEMA, compression="zstd")
    token_counts: list[int] = []
    counts: Counter[tuple[str, str, int]] = Counter()
    changed_speech = 0
    unchanged_nonspeech = 0
    rows_seen = 0
    referenced_assets: set[str] = set()
    started = time.time()
    try:
        for shard_ordinal, model_path in enumerate(shard_files):
            name = model_path.name
            suffix = name[len("model-sceneplans-"):]
            split = model_path.parent.name
            old_recipe_path = model_path.with_name("render-recipes-" + suffix)
            old_condition_path = model_path.with_name("conditioning-" + suffix)
            models = read_canonical_jsonl(model_path)
            recipes = read_canonical_jsonl(old_recipe_path)
            conditions = read_canonical_jsonl(old_condition_path)
            index_rows = old_index.read_row_group(shard_ordinal).to_pylist()
            if not (len(models) == len(recipes) == len(conditions) == len(index_rows)):
                raise RuntimeError(f"old P7.5 row mismatch: {model_path}")
            new_model_texts: list[str] = []
            new_recipe_texts: list[str] = []
            new_condition_texts: list[str] = []
            pending: list[dict[str, Any]] = []
            captions: list[str] = []
            for scene, recipe, condition, old_row in zip(models, recipes, conditions, index_rows):
                sample_id = str(scene["sample_id"])
                if not (
                    recipe["sample_id"] == condition["sample_id"] == old_row["sample_id"] == sample_id
                ):
                    raise RuntimeError(f"{sample_id}: old three-view order mismatch")
                before = json.loads(canonical_json(scene))
                recipe_by_id = {str(source["source_id"]): source for source in recipe["sources"]}
                speech_sources = [source for source in scene["sources"] if source["kind"] == "speech"]
                for source in speech_sources:
                    asset_id = str(recipe_by_id[str(source["source_id"])]["asset_ref"]["asset_id"])
                    entry = registry.get(asset_id)
                    if entry is None or str(entry["split"]) != split:
                        raise RuntimeError(f"{sample_id}: missing/wrong-split speaker registry asset {asset_id}")
                    source["speaker_description"] = str(entry["speaker_description"])
                    referenced_assets.add(asset_id)
                validate_model_sceneplan(scene)
                if neutral_scene(before) != neutral_scene(scene):
                    raise RuntimeError(f"{sample_id}: non-speaker ScenePlan state changed")
                model_text = canonical_json(scene)
                model_sha = sha256_text(model_text)
                recipe["model_sceneplan_sha256"] = model_sha
                recipe_text = canonical_json(recipe)
                caption = compile_model_renderer_caption(scene)
                condition = {"sample_id": sample_id, "renderer_caption": caption}
                condition_text = canonical_json(condition)
                new_model_texts.append(model_text)
                new_recipe_texts.append(recipe_text)
                new_condition_texts.append(condition_text)
                captions.append(caption["text"])
                pending.append({
                    "old": old_row,
                    "model_sha": model_sha,
                    "recipe_sha": sha256_text(recipe_text),
                    "caption_sha": sha256_text(canonical_json(caption)),
                })
                if speech_sources:
                    if model_sha == str(old_row["model_sceneplan_sha256"]):
                        raise RuntimeError(f"{sample_id}: speech ScenePlan hash did not change")
                    changed_speech += 1
                else:
                    if (
                        model_sha != str(old_row["model_sceneplan_sha256"])
                        or pending[-1]["caption_sha"] != str(old_row["renderer_caption_sha256"])
                    ):
                        raise RuntimeError(f"{sample_id}: no-speech text changed")
                    unchanged_nonspeech += 1
            encoded = tokenizer(captions, add_special_tokens=True, truncation=False, padding=False)["input_ids"]
            shard_tokens = [len(value) for value in encoded]
            if max(shard_tokens) > hard_max:
                raise RuntimeError(f"caption exceeds {hard_max} in {model_path}")
            token_counts.extend(shard_tokens)
            destination_model = new_root / split / name
            destination_recipe = new_root / split / old_recipe_path.name
            destination_condition = new_root / split / old_condition_path.name
            model_offsets = atomic_jsonl(destination_model, new_model_texts)
            recipe_offsets = atomic_jsonl(destination_recipe, new_recipe_texts)
            condition_offsets = atomic_jsonl(destination_condition, new_condition_texts)
            output_rows: list[dict[str, Any]] = []
            for index, item in enumerate(pending):
                row = dict(item["old"])
                row.update({
                    "sceneplan_path": str(destination_model),
                    "sceneplan_byte_offset": model_offsets[index][0],
                    "sceneplan_byte_length": model_offsets[index][1],
                    "model_sceneplan_sha256": item["model_sha"],
                    "render_recipe_path": str(destination_recipe),
                    "render_recipe_byte_offset": recipe_offsets[index][0],
                    "render_recipe_byte_length": recipe_offsets[index][1],
                    "render_recipe_sha256": item["recipe_sha"],
                    "conditioning_path": str(destination_condition),
                    "conditioning_byte_offset": condition_offsets[index][0],
                    "conditioning_byte_length": condition_offsets[index][1],
                    "renderer_caption_sha256": item["caption_sha"],
                    "caption_qwen_tokens": shard_tokens[index],
                })
                output_rows.append(row)
                counts[(str(row["split"]), str(row["family"]), int(row["source_count"]))] += 1
            writer.write_table(pa.Table.from_pylist(output_rows, schema=INDEX_SCHEMA))
            rows_seen += len(output_rows)
            if (shard_ordinal + 1) % 50 == 0:
                print(json.dumps({
                    "migrated_shards": shard_ordinal + 1,
                    "rows": rows_seen,
                    "elapsed_sec": round(time.time() - started, 1),
                }), flush=True)
    finally:
        writer.close()
    os.replace(new_root / "index.parquet.tmp", new_root / "index.parquet")
    if (
        rows_seen != 1_124_000 or changed_speech != 512_000
        or unchanged_nonspeech != 612_000 or len(referenced_assets) != 512_000
    ):
        raise RuntimeError(
            f"migration totals failed rows={rows_seen} speech={changed_speech} "
            f"nonspeech={unchanged_nonspeech} registry={len(referenced_assets)}"
        )
    p99 = float(np.percentile(np.asarray(token_counts, dtype=np.float64), 99))
    maximum = max(token_counts)
    if p99 > p99_target or maximum > hard_max:
        raise RuntimeError(f"caption envelope failed p99={p99} max={maximum}")
    old_summary = json.loads((old_root / "summary.json").read_text(encoding="utf-8"))
    summary = dict(old_summary)
    summary.update({
        "rows": rows_seen,
        "index": str(new_root / "index.parquet"),
        "speech_speaker_registry": str(registry_path),
        "speech_speaker_registry_sha256": sha256_file(registry_path),
        "speech_speaker_description_is_registry_driven": True,
        "speaker_description_migration": {
            "source": str(old_root),
            "method": "exact_asset_join_text_only_v1",
            "changed_speech_rows": changed_speech,
            "unchanged_nonspeech_rows": unchanged_nonspeech,
            "non_text_sceneplan_state_preserved": True,
            "exact_transcript_preserved": True,
        },
        "caption_qwen_tokens": {
            "p99": p99, "p99_target": p99_target, "max": maximum,
            "hard_max": hard_max, "truncated": 0,
        },
        "foa_materialization_started": False,
        "latent_materialization_started": False,
        "p8_started": False,
        "p9_started": False,
        "elapsed_sec": round(time.time() - started, 3),
    })
    atomic_write_json(new_root / "summary.json", summary)
    atomic_write_json(new_root / "READY", {
        "schema": "stable_audio_tools.model_sceneplan_manifest_ready",
        "schema_version": 1,
        "rows": rows_seen,
        "summary": str(new_root / "summary.json"),
        "index": str(new_root / "index.parquet"),
        "speaker_description_migrated": True,
        "p8_started": False,
        "p9_started": False,
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
