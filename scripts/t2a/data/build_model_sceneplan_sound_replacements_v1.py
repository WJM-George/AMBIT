#!/usr/bin/env python3
"""Replace one-source Sound scenes with new unique sources in all P10 splits.

The base 1.124M dataset is immutable.  This builder emits only changed rows plus
an exact replacement map.  Train targets are occurrences of assets that remain
present elsewhere; validation/test targets are exchanged one-for-one so their
row and modality quotas stay unchanged.  Every changed scene is replanned from
its original sample id and cell index, preserving deterministic room/motion
distributions while adapting duration to the complete new dry asset.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
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
SAMPLE_RE = re.compile(
    r"spv2_(train|validation|test)_no_speech_1_(\d{7})"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-parquet", type=Path, required=True)
    parser.add_argument("--registry-parquet", type=Path, required=True)
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument(
        "--base-index", type=Path,
        default=Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/sceneplans_model_v1/index.parquet"),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--tokenizer-root", type=Path,
        default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_sources(
    universe_path: Path, registry_path: Path, candidate_path: Path
) -> dict[str, list[dict[str, Any]]]:
    universe = pq.read_table(universe_path).to_pylist()
    registry_rows = pq.read_table(registry_path).to_pylist()
    candidates = {
        str(row["source_audio_sha256"]): row for row in read_jsonl(candidate_path)
    }
    registry = {
        str(row["source_audio_sha256"]): row for row in registry_rows
    }
    if not (len(universe) == len(registry) == len(candidates)):
        raise RuntimeError("Sound expansion universe/registry/candidate counts differ")
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in universe:
        digest = str(row["source_audio_sha256"])
        annotation = registry.get(digest)
        candidate = candidates.get(digest)
        if annotation is None or candidate is None:
            raise RuntimeError(f"Sound expansion join misses {digest}")
        if not (
            row["kind"] == annotation["kind"] == "sound"
            and row["split"] == annotation["split"]
            and annotation["annotation_id"] == f"sha256:{digest}"
            and annotation["primary_asset_id"] == row["primary_asset_id"]
            and annotation["audio_path"] == row["dry_audio_path"]
        ):
            raise RuntimeError(f"Sound expansion registry lineage mismatch: {digest}")
        dataset = str(row["source_dataset"])
        parent = candidate.get("parent_lineage_key")
        if dataset == "fsdkaggle2019":
            segment_method = "deterministic_parent_crop_max10s_v1"
        elif dataset == "vggsound":
            segment_method = "official_fixed_10s_segment_v1"
        else:
            segment_method = "full_native_utterance"
        by_split[str(row["split"])].append(
            {
                "asset_id": str(row["primary_asset_id"]),
                "source_dataset": dataset,
                "kind": "sound",
                "description": str(annotation["source_description"]),
                "model_num_samples": int(row["model_num_samples"]),
                "source_audio_sha256": digest,
                "native_sample_rate_hz": int(row["native_sample_rate_hz"]),
                "native_num_samples": int(row["native_num_samples"]),
                "dry_audio_path": str(Path(row["dry_audio_path"]).resolve(strict=True)),
                "selection_rank": str(row["selection_rank"]),
                "spoken_language_background": bool(
                    annotation["spoken_language_background"]
                ),
                "source_description_registry_id": str(annotation["annotation_id"]),
                "parent_asset_id": str(parent) if parent else None,
                "parent_start_sample": None,
                "parent_end_sample": None,
                "segment_method": segment_method,
            }
        )
    for rows in by_split.values():
        rows.sort(key=lambda row: (row["selection_rank"], row["source_audio_sha256"]))
    return dict(by_split)


def load_targets(base_index: Path, needed: dict[str, int]) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    asset_counts: Counter[str] = Counter()
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parquet = pq.ParquetFile(base_index)
    columns = ["sample_id", "split", "family", "source_count", "source_asset_ids", "source_kinds"]
    for batch in parquet.iter_batches(columns=columns, batch_size=32_768):
        values = [batch.column(name).to_pylist() for name in columns]
        for sample_id, split, family, count, assets, kinds in zip(*values):
            for asset, kind in zip(assets or (), kinds or ()):
                if kind == "sound":
                    asset_counts[str(asset)] += 1
            if (
                str(family) == "no_speech"
                and int(count) == 1
                and list(kinds or ()) == ["sound"]
            ):
                candidates[str(split)].append(
                    {"sample_id": str(sample_id), "old_asset_id": str(assets[0])}
                )
    selected: dict[str, list[dict[str, Any]]] = {}
    for split, count in needed.items():
        rows = candidates[split]
        if split == "train":
            rows = [row for row in rows if asset_counts[row["old_asset_id"]] > 1]
        rows.sort(
            key=lambda row: (
                -asset_counts[row["old_asset_id"]] if split == "train" else 0,
                hashlib.sha256(
                    f"sceneplan-sound-replacement-v1\0{row['sample_id']}".encode()
                ).hexdigest(),
            )
        )
        if split == "train":
            # Prefer the most reused donors, but never select every occurrence
            # of an old asset.  The expansion adds unique sources while
            # retaining all pre-existing train-source diversity.
            chosen_per_asset: Counter[str] = Counter()
            picked = []
            for row in rows:
                asset = row["old_asset_id"]
                if chosen_per_asset[asset] >= asset_counts[asset] - 1:
                    continue
                picked.append(row)
                chosen_per_asset[asset] += 1
                if len(picked) == count:
                    break
        else:
            picked = rows[:count]
        if len(picked) < count:
            raise RuntimeError(
                f"{split}: only {len(picked)} retention-safe replacement "
                f"targets for {count} sources"
            )
        selected[split] = picked
    return selected, asset_counts


def main() -> int:
    args = parse_args()
    universe_path = args.universe_parquet.expanduser().resolve(strict=True)
    registry_path = args.registry_parquet.expanduser().resolve(strict=True)
    candidate_path = args.candidate_jsonl.expanduser().resolve(strict=True)
    base_index = args.base_index.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    tokenizer_root = args.tokenizer_root.expanduser().resolve(strict=True)
    try:
        output_root.relative_to(Path(os.environ.get("AMBIT_DATA_ROOT", "data")))
    except ValueError as error:
        raise ValueError("replacement ScenePlans must live on ${AMBIT_DATA_ROOT}") from error
    if output_root.exists() and any(output_root.iterdir()):
        ready = output_root / "READY"
        if ready.is_file():
            print(ready.read_text(encoding="utf-8"), end="")
            return 0
        raise FileExistsError(f"refusing non-empty incomplete output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    contract = json.loads(DATASET_CONTRACT.read_text(encoding="utf-8"))
    caption_max = int(contract["renderer_caption_contract"]["hard_max_qwen_tokens"])
    caption_p99_target = int(contract["renderer_caption_contract"]["p99_target_qwen_tokens"])
    sources = load_sources(universe_path, registry_path, candidate_path)
    needed = {split: len(rows) for split, rows in sources.items()}
    targets, asset_counts = load_targets(base_index, needed)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    replacement_rows: list[dict[str, Any]] = []
    index_writer = pq.ParquetWriter(
        output_root / "index.parquet.tmp", INDEX_SCHEMA, compression="zstd"
    )
    caption_tokens: list[int] = []
    room_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter(
        {"train": 0, "validation": 0, "test": 0}
    )
    started = time.time()
    try:
        for split in ("train", "validation", "test"):
            split_sources = sources.get(split, [])
            split_targets = targets.get(split, [])
            shard_rows: list[dict[str, Any]] = []
            shard_index = 0

            def flush() -> None:
                nonlocal shard_rows, shard_index
                if not shard_rows:
                    return
                encoded = tokenizer(
                    [row["caption"]["text"] for row in shard_rows],
                    add_special_tokens=True, truncation=False, padding=False,
                )["input_ids"]
                lengths = [len(ids) for ids in encoded]
                if max(lengths) > caption_max:
                    raise RuntimeError(f"{split}: replacement caption exceeds {caption_max}")
                caption_tokens.extend(lengths)
                stem = f"{split}-{shard_index:05d}"
                model_path = output_root / split / f"model-sceneplans-{stem}.jsonl"
                recipe_path = output_root / split / f"render-recipes-{stem}.jsonl"
                conditioning_path = output_root / split / f"conditioning-{stem}.jsonl"
                model_offsets = atomic_jsonl(
                    model_path, [canonical_json(row["model_sceneplan"]) for row in shard_rows]
                )
                recipe_offsets = atomic_jsonl(
                    recipe_path, [canonical_json(row["render_recipe"]) for row in shard_rows]
                )
                conditioning_offsets = atomic_jsonl(
                    conditioning_path,
                    [canonical_json({"sample_id": row["sample_id"], "renderer_caption": row["caption"]}) for row in shard_rows],
                )
                index_rows = []
                for row_index, row in enumerate(shard_rows):
                    mo, ml = model_offsets[row_index]
                    ro, rl = recipe_offsets[row_index]
                    co, cl = conditioning_offsets[row_index]
                    index_rows.append(
                        {
                            "sample_id": row["sample_id"],
                            "split": split,
                            "family": "no_speech",
                            "source_count": 1,
                            "room_type": row["model_sceneplan"]["room"]["type"],
                            "model_num_samples": row["model_num_samples"],
                            "latent_frames_valid": row["latent_frames_valid"],
                            "work_shard": shard_index,
                            "row_in_shard": row_index,
                            "sceneplan_path": str(model_path),
                            "sceneplan_byte_offset": mo,
                            "sceneplan_byte_length": ml,
                            "model_sceneplan_sha256": row["model_sceneplan_sha256"],
                            "render_recipe_path": str(recipe_path),
                            "render_recipe_byte_offset": ro,
                            "render_recipe_byte_length": rl,
                            "render_recipe_sha256": row["render_recipe_sha256"],
                            "conditioning_path": str(conditioning_path),
                            "conditioning_byte_offset": co,
                            "conditioning_byte_length": cl,
                            "renderer_caption_sha256": row["renderer_caption_sha256"],
                            "caption_qwen_tokens": lengths[row_index],
                            "speech_asset_id": None,
                            "source_asset_ids": row["source_asset_ids"],
                            "source_kinds": ["sound"],
                        }
                    )
                index_writer.write_table(pa.Table.from_pylist(index_rows, schema=INDEX_SCHEMA))
                shard_rows = []
                shard_index += 1

            for source, target in zip(split_sources, split_targets):
                match = SAMPLE_RE.fullmatch(target["sample_id"])
                if match is None or match.group(1) != split:
                    raise RuntimeError(f"invalid one-source target id: {target['sample_id']}")
                cell_index = int(match.group(2))
                legacy = plan_scene(
                    sample_id=target["sample_id"], split=split,
                    family="no_speech", source_count=1, cell_index=cell_index,
                    speech_row=None, cyclers={"sound": AssetCycler([source])},
                    nonspeech_kinds_override=["sound"],
                )
                record = json.loads(legacy["record_json"])
                model = model_sceneplan_from_renderer_record(record)
                model_text = canonical_json(model)
                model_sha = sha256_text(model_text)
                recipe = render_recipe_from_renderer_record(record, model_sha)
                caption = compile_model_renderer_caption(model)
                controls = compile_model_44_controls(
                    model,
                    model_num_samples=int(legacy["model_num_samples"]),
                    latent_frames_valid=int(legacy["latent_frames_valid"]),
                )
                frames = int(legacy["latent_frames_valid"])
                if (
                    controls["source_event_frame_ids"].shape != (4, frames)
                    or controls["source_trajectory_features"].shape != (4, frames, 5)
                    or int((controls["source_event_frame_ids"] > 0).any(axis=1).sum()) != 1
                ):
                    raise RuntimeError(f"{target['sample_id']}: replacement 4+4 contract failed")
                recipe_text = canonical_json(recipe)
                caption_text = canonical_json(caption)
                shard_rows.append(
                    {
                        "sample_id": target["sample_id"],
                        "model_num_samples": int(legacy["model_num_samples"]),
                        "latent_frames_valid": frames,
                        "model_sceneplan": model,
                        "model_sceneplan_sha256": model_sha,
                        "render_recipe": recipe,
                        "render_recipe_sha256": sha256_text(recipe_text),
                        "caption": caption,
                        "renderer_caption_sha256": sha256_text(caption_text),
                        "source_asset_ids": legacy["source_asset_ids"],
                    }
                )
                replacement_rows.append(
                    {
                        "sample_id": target["sample_id"],
                        "split": split,
                        "old_asset_id": target["old_asset_id"],
                        "old_asset_occurrences_before": int(asset_counts[target["old_asset_id"]]),
                        "new_asset_id": source["asset_id"],
                        "new_source_audio_sha256": source["source_audio_sha256"],
                        "new_source_dataset": source["source_dataset"],
                    }
                )
                room_counts[model["room"]["type"]] += 1
                motion_counts[model["sources"][0]["trajectory"]["type"]] += 1
                split_counts[split] += 1
                if len(shard_rows) >= SHARD_ROWS:
                    flush()
            flush()
    finally:
        index_writer.close()
    os.replace(output_root / "index.parquet.tmp", output_root / "index.parquet")
    if len({row["sample_id"] for row in replacement_rows}) != len(replacement_rows):
        raise RuntimeError("replacement sample IDs are not unique")
    if len({row["new_source_audio_sha256"] for row in replacement_rows}) != len(replacement_rows):
        raise RuntimeError("replacement sources are not unique")
    if any(
        row["split"] == "train" and row["old_asset_occurrences_before"] < 2
        for row in replacement_rows
    ):
        raise RuntimeError("train replacement would remove an old asset entirely")
    train_replaced = Counter(
        row["old_asset_id"]
        for row in replacement_rows
        if row["split"] == "train"
    )
    if any(
        count >= asset_counts[asset]
        for asset, count in train_replaced.items()
    ):
        raise RuntimeError("train replacement selected every occurrence of an old asset")
    token_p99 = float(np.percentile(caption_tokens, 99))
    if token_p99 > caption_p99_target:
        raise RuntimeError("replacement caption p99 target failed")
    replacement_path = output_root / "replacement_map.jsonl"
    replacement_rows.sort(key=lambda row: (row["split"], row["sample_id"]))
    with replacement_path.open("w", encoding="utf-8") as handle:
        for row in replacement_rows:
            handle.write(canonical_json(row) + "\n")
    summary = {
        "schema": "stable_audio_tools.model_sceneplan_sound_replacements",
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_revision": 5,
        "rows": len(replacement_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "unique_new_source_audio_sha256": len(replacement_rows),
        "train_old_assets_retained_elsewhere": True,
        "train_target_policy": "highest_reuse_first_with_count_minus_one_cap",
        "base_p9_mutated": False,
        "room_counts": dict(sorted(room_counts.items())),
        "motion_counts": dict(sorted(motion_counts.items())),
        "caption_qwen_tokens": {
            "p99": token_p99, "p99_target": caption_p99_target,
            "max": max(caption_tokens), "hard_max": caption_max, "truncated": 0,
        },
        "replacement_map": str(replacement_path),
        "replacement_map_sha256": sha256_file(replacement_path),
        "index": str(output_root / "index.parquet"),
        "index_sha256": sha256_file(output_root / "index.parquet"),
        "elapsed_sec": round(time.time() - started, 3),
    }
    summary_path = output_root / "summary.json"
    atomic_write_json(summary_path, summary)
    atomic_write_json(
        output_root / "READY",
        {
            "schema": "stable_audio_tools.sound_replacement_sceneplans_ready",
            "schema_version": 1,
            "status": "PASS",
            "rows": len(replacement_rows),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "base_p9_mutated": False,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
