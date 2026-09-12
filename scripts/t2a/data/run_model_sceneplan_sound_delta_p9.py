#!/usr/bin/env python3
"""Audit, index, and freeze an isolated train-only Sound supplement."""

from __future__ import annotations
import os

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq
from safetensors import safe_open
import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from build_model_sceneplan_training_index_v1 import (  # noqa: E402
    build_split,
    read_jsonl_with_offsets,
)
from materialize_model_sceneplan_v1_shard import (  # noqa: E402
    canonical_json,
    sha256_text,
)
from sceneplan_v2_common import DATASET_ROOT, atomic_write_json  # noqa: E402
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_44_controls,
    compile_model_renderer_caption,
    validate_model_sceneplan,
)
from stable_audio_tools.data.sceneplan_v2_dataset import ScenePlanV2Dataset  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supplement-root", type=Path, required=True)
    parser.add_argument("--sceneplan-root", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.expected_rows <= 0:
        raise ValueError("--expected-rows must be positive")
    supplement_root = args.supplement_root.expanduser().resolve(strict=True)
    sceneplan_root = args.sceneplan_root.expanduser().resolve(strict=True)
    materialized_root = args.materialized_root.expanduser().resolve(strict=True)
    tokenizer_root = args.tokenizer_root.expanduser().resolve(strict=True)
    allowed_root = (DATASET_ROOT / "supplements").resolve(strict=True)
    for path in (supplement_root, sceneplan_root, materialized_root):
        try:
            path.relative_to(allowed_root)
        except ValueError as error:
            raise ValueError(f"Sound delta P9 path escapes {allowed_root}: {path}") from error
    if not (DATASET_ROOT / "FROZEN_P9.json").is_file():
        raise RuntimeError("Sound delta P9 requires the frozen base P9")
    p8 = json.loads(
        (materialized_root / "P8_SUMMARY.json").read_text(encoding="utf-8")
    )
    if (
        p8.get("schema") != "stable_audio_tools.sound_delta_p8_summary"
        or p8.get("status") != "PASS"
        or int(p8.get("rows", -1)) != args.expected_rows
        or p8.get("base_p9_mutated") is not False
    ):
        raise RuntimeError("Sound delta P8 is not an all-pass P9 gate")
    started = time.time()
    plan_rows = pq.read_table(sceneplan_root / "index.parquet").to_pylist()
    if len(plan_rows) != args.expected_rows:
        raise RuntimeError("Sound delta P7.5 index row count changed")
    planned = {str(row["sample_id"]): row for row in plan_rows}
    if len(planned) != args.expected_rows:
        raise RuntimeError("Sound delta sample IDs are not unique")
    manifests = sorted(
        (materialized_root / "manifests" / "train").glob(
            "materialized-train-*.parquet"
        )
    )
    if len(manifests) != int(p8["materialized_shards"]):
        raise RuntimeError("Sound delta materialized shard count changed")

    seen_ids: set[str] = set()
    source_hashes: set[str] = set()
    source_assets: set[str] = set()
    latent_shards: set[str] = set()
    min_frames = 432
    max_frames = 0
    static = 0
    linear = 0
    for manifest in manifests:
        rows = sorted(
            pq.read_table(manifest).to_pylist(),
            key=lambda row: int(row["row_in_shard"]),
        )
        if not rows:
            raise RuntimeError(f"empty Sound delta manifest: {manifest}")
        shard = int(rows[0]["work_shard"])
        model_path = sceneplan_root / "train" / f"model-sceneplans-train-{shard:05d}.jsonl"
        recipe_path = sceneplan_root / "train" / f"render-recipes-train-{shard:05d}.jsonl"
        conditioning_path = sceneplan_root / "train" / f"conditioning-train-{shard:05d}.jsonl"
        models = read_jsonl_with_offsets(model_path)
        recipes = read_jsonl_with_offsets(recipe_path)
        conditionings = read_jsonl_with_offsets(conditioning_path)
        if not (len(rows) == len(models) == len(recipes) == len(conditionings)):
            raise RuntimeError(f"Sound delta three-view row mismatch: {manifest}")
        latent_refs = {str(row["latent_ref"]).split("#", 1)[0] for row in rows}
        latent_hashes = {str(row["latent_shard_sha256"]) for row in rows}
        if len(latent_refs) != 1 or len(latent_hashes) != 1:
            raise RuntimeError(f"Sound delta latent shard lineage differs: {manifest}")
        latent_path = Path(latent_refs.pop()).resolve(strict=True)
        expected_latent_sha = latent_hashes.pop()
        if file_sha256(latent_path) != expected_latent_sha:
            raise RuntimeError(f"Sound delta latent shard checksum changed: {latent_path}")
        latent_shards.add(str(latent_path))
        row_by_id = {str(row["sample_id"]): row for row in rows}
        with safe_open(str(latent_path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != set(row_by_id):
                raise RuntimeError(f"Sound delta latent keys changed: {latent_path}")
            for sample_id, row in row_by_id.items():
                tensor = handle.get_tensor(sample_id)
                frames = int(row["latent_frames_valid"])
                if (
                    tensor.dtype != torch.float16
                    or tuple(tensor.shape) != (64, frames)
                    or not torch.isfinite(tensor).all().item()
                    or tensor_sha256(tensor) != row["latent_tensor_sha256"]
                ):
                    raise RuntimeError(f"{sample_id}: Sound delta latent QC failed")
                min_frames = min(min_frames, frames)
                max_frames = max(max_frames, frames)

        for row_index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            if sample_id in seen_ids or sample_id not in planned:
                raise RuntimeError(f"{sample_id}: duplicate or unplanned Sound delta row")
            seen_ids.add(sample_id)
            model, model_text, _, _ = models[row_index]
            recipe, recipe_text, _, _ = recipes[row_index]
            conditioning, _, _, _ = conditionings[row_index]
            if not (
                model["sample_id"]
                == recipe["sample_id"]
                == conditioning["sample_id"]
                == sample_id
            ):
                raise RuntimeError(f"{sample_id}: Sound delta three-view ID mismatch")
            validate_model_sceneplan(model)
            if len(model["sources"]) != 1 or model["sources"][0]["kind"] != "sound":
                raise RuntimeError(f"{sample_id}: Sound-only contract changed")
            source = model["sources"][0]
            recipe_source = recipe["sources"][0]
            if source["source_id"] != recipe_source["source_id"]:
                raise RuntimeError(f"{sample_id}: source slot lineage changed")
            digest = str(recipe_source["asset_ref"]["identity_hash"])
            asset = str(recipe_source["asset_ref"]["asset_id"])
            if digest in source_hashes or asset in source_assets:
                raise RuntimeError(f"{sample_id}: delta source reused unexpectedly")
            source_hashes.add(digest)
            source_assets.add(asset)
            caption = compile_model_renderer_caption(model)
            if caption != conditioning["renderer_caption"]:
                raise RuntimeError(f"{sample_id}: Sound delta caption compiler drift")
            model_sha = sha256_text(model_text)
            recipe_sha = sha256_text(recipe_text)
            caption_sha = sha256_text(canonical_json(caption))
            plan = planned[sample_id]
            if not (
                row["model_sceneplan_sha256"] == plan["model_sceneplan_sha256"] == model_sha
                and row["render_recipe_sha256"] == plan["render_recipe_sha256"] == recipe_sha
                and row["renderer_caption_sha256"] == plan["renderer_caption_sha256"] == caption_sha
                and recipe["model_sceneplan_sha256"] == model_sha
            ):
                raise RuntimeError(f"{sample_id}: Sound delta hash lineage changed")
            controls = compile_model_44_controls(
                model,
                model_num_samples=int(row["model_num_samples"]),
                latent_frames_valid=int(row["latent_frames_valid"]),
            )
            frames = int(row["latent_frames_valid"])
            if (
                controls["source_event_frame_ids"].shape != (4, frames)
                or controls["source_trajectory_features"].shape != (4, frames, 5)
                or int((controls["source_event_frame_ids"] > 0).any(axis=1).sum()) != 1
            ):
                raise RuntimeError(f"{sample_id}: Sound delta 4+4 contract failed")
            motion = str(source["trajectory"]["type"])
            static += int(motion == "static")
            linear += int(motion == "linear")
            result = json.loads(str(row["render_result_json"]))
            if (
                result.get("status") != "ok"
                or result.get("sample_id") != sample_id
                or int(result.get("num_samples", -1)) != int(row["model_num_samples"])
                or len(result.get("source_qc") or ()) != 1
                or result["source_qc"][0].get("kind") != "sound"
                or result["source_qc"][0].get("asset_id") != asset
            ):
                raise RuntimeError(f"{sample_id}: Sound delta renderer QC failed")
    if not (
        len(seen_ids)
        == len(source_hashes)
        == len(source_assets)
        == args.expected_rows
    ):
        raise RuntimeError("Sound delta P9 exact coverage failed")

    audit_root = supplement_root / "qc"
    audit_root.mkdir(parents=True, exist_ok=True)
    training_index_root = supplement_root / "training_index"
    split_summary = build_split(
        materialized_root,
        sceneplan_root,
        training_index_root,
        "train",
        args.expected_rows,
    )
    training_summary = {
        "schema": "stable_audio_tools.model_sceneplan_training_index_build",
        "schema_version": 1,
        "dataset_contract_revision": 5,
        "rows": args.expected_rows,
        "splits": [split_summary],
        "caption_max_tokens": 512,
        "structured_feature_dim": 9,
        "latent_batch_padding_frames": 432,
        "random_crop": False,
        "p10_training_started": False,
        "p11_training_started": False,
        "base_p9_mutated": False,
    }
    training_summary_path = training_index_root / "summary.json"
    atomic_write_json(training_summary_path, training_summary)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)
    dataset = ScenePlanV2Dataset(
        split_summary["path"],
        tokenizer_spec=(tokenizer, 512),
        expected_num_samples=args.expected_rows,
        latent_crop_length=432,
        caption_max_tokens=512,
        random_crop=False,
        require_frozen=False,
    )
    sampled = []
    for ordinal in (0, args.expected_rows // 2, args.expected_rows - 1):
        latent, metadata = dataset[ordinal]
        if tuple(latent.shape) != (64, 432):
            raise RuntimeError("Sound delta loader padded shape changed")
        sampled.append(
            {
                "ordinal": ordinal,
                "sample_id": metadata["sample_id"],
                "caption_tokens": int(metadata["prompt"]["attention_mask"].sum()),
                "latent_frames_valid": int(metadata["latent_stored_length"]),
                "present_source_tracks": int(
                    (metadata["sceneplan_44"]["source_event_frame_ids"] > 0)
                    .any(dim=1)
                    .sum()
                ),
            }
        )
    loader_report = {
        "schema": "stable_audio_tools.sound_delta_loader_smoke",
        "schema_version": 1,
        "status": "PASS",
        "rows": args.expected_rows,
        "sampled": sampled,
        "base_p9_mutated": False,
    }
    loader_path = audit_root / "loader_smoke.json"
    atomic_write_json(loader_path, loader_report)
    audit = {
        "schema": "stable_audio_tools.sound_delta_p9_audit",
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_revision": 5,
        "rows": len(seen_ids),
        "unique_source_audio_sha256": len(source_hashes),
        "unique_source_asset_ids": len(source_assets),
        "latent_shards": len(latent_shards),
        "latent_frames_min": min_frames,
        "latent_frames_max": max_frames,
        "motion_counts": {"static": static, "linear": linear},
        "quarantine_rows": 0,
        "training_index": split_summary,
        "loader_smoke": str(loader_path),
        "loader_smoke_sha256": file_sha256(loader_path),
        "base_p9_mutated": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    audit_path = audit_root / "P9_AUDIT.json"
    atomic_write_json(audit_path, audit)
    contracts_root = supplement_root / "contracts"
    contracts_root.mkdir(parents=True, exist_ok=True)
    freeze = {
        "schema": "stable_audio_tools.sceneplan_v2_p9_freeze_manifest",
        "schema_version": 3,
        "state": "P9_complete_frozen_waiting_for_user_acceptance",
        "dataset_contract_revision": 5,
        "dataset_id": "sceneplan_vggsound_sound_delta_v1",
        "rows": args.expected_rows,
        "splits": {"train": args.expected_rows},
        "p8_summary": str(materialized_root / "P8_SUMMARY.json"),
        "p8_summary_sha256": file_sha256(materialized_root / "P8_SUMMARY.json"),
        "p9_report": str(audit_path),
        "p9_report_sha256": file_sha256(audit_path),
        "loader_smoke_report": str(loader_path),
        "loader_smoke_report_sha256": file_sha256(loader_path),
        "training_index_summary": str(training_summary_path),
        "training_index_summary_sha256": file_sha256(training_summary_path),
        "renderer_caption_max_tokens": 512,
        "max_latent_frames": 432,
        "random_crop": False,
        "structured_feature_dim": 9,
        "base_p9_mutated": False,
        "p10_training_started": False,
        "p11_training_started": False,
    }
    freeze_path = contracts_root / "freeze_manifest.json"
    atomic_write_json(freeze_path, freeze)
    marker = {
        "schema": "stable_audio_tools.sceneplan_v2_p9_marker",
        "schema_version": 3,
        "dataset_contract_revision": 5,
        "state": "P9_complete_frozen_waiting_for_user_acceptance",
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": file_sha256(freeze_path),
        "p10_training_started": False,
        "p11_training_started": False,
        "base_p9_mutated": False,
    }
    atomic_write_json(supplement_root / "FROZEN_P9.json", marker)
    # Reopen through the formal frozen path after the marker is durable.
    frozen_dataset = ScenePlanV2Dataset(
        split_summary["path"],
        tokenizer_spec=(tokenizer, 512),
        expected_num_samples=args.expected_rows,
        latent_crop_length=432,
        caption_max_tokens=512,
        random_crop=False,
        require_frozen=True,
    )
    frozen_dataset[args.expected_rows - 1]
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
