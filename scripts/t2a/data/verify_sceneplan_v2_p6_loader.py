#!/usr/bin/env python3
"""Verify the real P6 SQLite/safetensors loader and 4+4+2 fusion path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sceneplan_v2_common import DATASET_ROOT, atomic_write_json  # noqa: E402
from stable_audio_tools.data.dataset import collation_fn  # noqa: E402
from stable_audio_tools.data.sceneplan_v2_dataset import ScenePlanV2Dataset  # noqa: E402
from stable_audio_tools.models.conditioners import (  # noqa: E402
    ScenePlan442FusionConditioner,
)


INDEX = DATASET_ROOT / "pilots/joint_4k/training_index/train.sqlite"
TOKENIZER = Path("/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B")
OUTPUT = DATASET_ROOT / "pilots/joint_4k/qc/loader_conditioner_smoke.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_item(latent: torch.Tensor, metadata: dict) -> dict:
    sample_id = str(metadata["sample_id"])
    valid = int(metadata["latent_stored_length"])
    require(latent.dtype == torch.float16, f"{sample_id}: latent dtype changed")
    require(tuple(latent.shape) == (64, 432), f"{sample_id}: latent shape changed")
    require(0 < valid <= 432, f"{sample_id}: invalid latent frame count")
    require(not latent[:, valid:].any().item(), f"{sample_id}: latent padding is nonzero")
    require(torch.equal(latent, metadata["audio"]), f"{sample_id}: metadata audio differs")

    padding = metadata["padding_mask"]
    require(
        isinstance(padding, list) and len(padding) == 1,
        f"{sample_id}: padding-mask wrapper changed",
    )
    padding = padding[0]
    require(tuple(padding.shape) == (432,), f"{sample_id}: padding-mask shape changed")
    require(
        bool(padding[:valid].all()) and not bool(padding[valid:].any()),
        f"{sample_id}: loss-valid padding mask drifted",
    )

    prompt = metadata["prompt"]
    require(tuple(prompt["input_ids"].shape) == (256,), f"{sample_id}: prompt shape changed")
    require(
        tuple(prompt["attention_mask"].shape) == (256,),
        f"{sample_id}: prompt attention shape changed",
    )
    controls = metadata["sceneplan_442"]
    required_shapes = {
        "source_semantic_token_masks": (4, 256),
        "source_motion_activity_token_masks": (4, 256),
        "speaker_info_token_mask": (256,),
        "quoted_transcript_token_mask": (256,),
        "source_present_mask": (4,),
        "source_kind_ids": (4,),
        "source_slot_ids": (4,),
        "source_activity_frame_masks": (4, 432),
        "source_position_activity_features": (4, 432, 8),
    }
    for key, shape in required_shapes.items():
        require(tuple(controls[key].shape) == shape, f"{sample_id}: {key} shape changed")

    present = controls["source_present_mask"].to(torch.bool)
    semantic = controls["source_semantic_token_masks"].to(torch.bool)
    motion = controls["source_motion_activity_token_masks"].to(torch.bool)
    activity = controls["source_activity_frame_masks"].to(torch.bool)
    position = controls["source_position_activity_features"]
    require(
        torch.equal(position[..., 0].to(torch.bool), activity),
        f"{sample_id}: structured feature-0/activity binding changed",
    )
    require(
        not activity[:, valid:].any().item() and not position[:, valid:].any().item(),
        f"{sample_id}: structured variable-length padding is nonzero",
    )
    for slot in range(4):
        require(
            bool(semantic[slot].any()) == bool(present[slot]),
            f"{sample_id}: semantic mask/present mismatch for slot {slot}",
        )
        require(
            bool(motion[slot].any()) == bool(present[slot]),
            f"{sample_id}: motion mask/present mismatch for slot {slot}",
        )
        if not bool(present[slot]):
            require(
                not activity[slot].any().item() and not position[slot].any().item(),
                f"{sample_id}: absent source has structured energy",
            )
    speech_count = int(controls["source_kind_ids"].eq(1).sum().item())
    require(speech_count in (0, 1), f"{sample_id}: more than one speech source")
    require(
        bool(controls["speaker_info_token_mask"].any()) == bool(speech_count),
        f"{sample_id}: speaker mask/speech mismatch",
    )
    require(
        bool(controls["quoted_transcript_token_mask"].any()) == bool(speech_count),
        f"{sample_id}: transcript mask/speech mismatch",
    )
    require(
        not (
            controls["speaker_info_token_mask"].to(torch.bool)
            & controls["quoted_transcript_token_mask"].to(torch.bool)
        ).any().item(),
        f"{sample_id}: speaker/transcript masks overlap",
    )
    return {
        "sample_id": sample_id,
        "valid_frames": valid,
        "present_sources": int(present.sum().item()),
        "speech_sources": speech_count,
        "qwen_tokens": int(prompt["attention_mask"].sum().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=INDEX)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    index = args.index.expanduser().resolve(strict=True)
    tokenizer_path = args.tokenizer.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    try:
        index.relative_to("/mnt/sdb")
        output.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError("P6 loader index/report must remain on SDB") from error

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    dataset = ScenePlanV2Dataset(
        index,
        tokenizer_spec=(tokenizer, 256),
        expected_num_samples=4_000,
        latent_crop_length=432,
        caption_max_tokens=256,
        random_crop=False,
        require_frozen=False,
    )
    indices = [0, 699, 700, 1_399, 1_400, 1_799, 1_800, 1_999, 2_000, 3_999]
    items = [dataset[index_value] for index_value in indices]
    checked = [validate_item(*item) for item in items]
    require(
        {row["present_sources"] for row in checked} == {1, 2, 3, 4},
        "P6 real-loader sample did not cover all source counts",
    )
    require(
        {row["speech_sources"] for row in checked} == {0, 1},
        "P6 real-loader sample did not cover both families",
    )

    batch = collation_fn([items[0], items[-1]])
    require(len(batch) == 2, "training collation tuple width changed")
    latent_batch, metadata_batch = batch
    require(tuple(latent_batch.shape) == (2, 64, 432), "collated latent shape changed")
    require(len(metadata_batch) == 2, "collated metadata batch size changed")

    torch.manual_seed(20260814)
    prompt_tokens = torch.randn(2, 256, 768)
    prompt_attention = torch.stack(
        [metadata["prompt"]["attention_mask"] for metadata in metadata_batch]
    )
    fusion = ScenePlan442FusionConditioner(
        output_dim=256,
        prompt_id="prompt",
        text_dim=768,
        event_dim=256,
        position_feature_dim=8,
        per_source_stream_dim=64,
        max_sources=4,
    ).eval()
    with torch.inference_mode():
        fused, fusion_mask = fusion.forward_with_context(
            [metadata["sceneplan_442"] for metadata in metadata_batch],
            prompt_tokens=prompt_tokens,
            prompt_attention_mask=prompt_attention,
            device=torch.device("cpu"),
        )
    require(tuple(fused.shape) == (2, 256, 432), "ScenePlan fusion output shape changed")
    require(tuple(fusion_mask.shape) == (2, 432), "ScenePlan fusion mask shape changed")
    require(torch.isfinite(fused).all().item(), "ScenePlan fusion produced non-finite values")
    for batch_index, metadata in enumerate(metadata_batch):
        activity = metadata["sceneplan_442"]["source_activity_frame_masks"]
        scene_active = activity.to(torch.bool).any(dim=0)
        require(
            not fused[batch_index, :, ~scene_active].any().item(),
            f"{metadata['sample_id']}: local conditioning leaks outside source activity",
        )

    report = {
        "schema": "stable_audio_tools.sceneplan_v2_p6_loader_conditioner_smoke",
        "schema_version": 1,
        "ok": True,
        "index": str(index),
        "index_sha256": sha256_file(index),
        "dataset_rows": len(dataset),
        "checked_items": checked,
        "checked_indices": indices,
        "collated_latent_shape": list(latent_batch.shape),
        "fusion_output_shape": list(fused.shape),
        "fusion_mask_shape": list(fusion_mask.shape),
        "random_crop": False,
        "caption_truncation": False,
        "conditioning": "caption_cross_attention_plus_4+4+2_to_four_local_streams",
        "p10_training_started": False,
    }
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
