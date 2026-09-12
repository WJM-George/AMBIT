#!/usr/bin/env python3
"""P9 smoke test for the frozen revision-5 loader and 4+4+2 conditioner."""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sceneplan_v2_common import DATASET_ROOT, atomic_write_json  # noqa: E402
from stable_audio_tools.data.sceneplan_v2_dataset import ScenePlanV2Dataset  # noqa: E402
from stable_audio_tools.models.sceneplan_conditioning import ScenePlan442Conditioner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument(
        "--tokenizer-root", type=Path, default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B")
    )
    parser.add_argument(
        "--output", type=Path, default=DATASET_ROOT / "qc/p9_model_sceneplan_loader_smoke.json"
    )
    args = parser.parse_args()
    root = args.dataset_root.expanduser().resolve(strict=True)
    tokenizer_root = args.tokenizer_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    try:
        root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
        output.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
    except ValueError as error:
        raise ValueError("P9 dataset and smoke report must be on SDB") from error
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_root, trust_remote_code=True, local_files_only=True
    )
    expected = {"train": 1_100_000, "validation": 20_000, "test": 4_000}
    datasets: dict[str, ScenePlanV2Dataset] = {}
    sampled: dict[str, list[dict]] = {}
    for split, rows in expected.items():
        dataset = ScenePlanV2Dataset(
            root / "training_index" / f"{split}.sqlite",
            tokenizer_spec=(tokenizer, 512),
            expected_num_samples=rows,
            latent_crop_length=432,
            caption_max_tokens=512,
            random_crop=False,
            require_frozen=False,
        )
        if dataset.contract_revision != 5 or dataset.structured_feature_dim != 9:
            raise RuntimeError(f"{split}: loader did not select revision-5 controls")
        datasets[split] = dataset
        sampled[split] = []
        for ordinal in (0, rows // 2, rows - 1):
            latent, metadata = dataset[ordinal]
            controls = metadata["sceneplan_442"]
            if tuple(latent.shape) != (64, 432) or latent.dtype != torch.float16:
                raise RuntimeError(f"{split}:{ordinal}: padded latent geometry changed")
            if tuple(controls["source_semantic_token_masks"].shape) != (4, 512):
                raise RuntimeError(f"{split}:{ordinal}: semantic token masks changed")
            if tuple(controls["source_motion_activity_token_masks"].shape) != (4, 512):
                raise RuntimeError(f"{split}:{ordinal}: motion token masks changed")
            if tuple(controls["speaker_info_token_mask"].shape) != (512,):
                raise RuntimeError(f"{split}:{ordinal}: speaker mask changed")
            if tuple(controls["quoted_transcript_token_mask"].shape) != (512,):
                raise RuntimeError(f"{split}:{ordinal}: transcript mask changed")
            if tuple(controls["source_position_activity_features"].shape) != (4, 432, 9):
                raise RuntimeError(f"{split}:{ordinal}: structured feature geometry changed")
            valid = int(metadata["latent_stored_length"])
            if not controls["source_position_activity_features"][:, valid:].eq(0).all():
                raise RuntimeError(f"{split}:{ordinal}: controls leak into latent padding")
            if not metadata["padding_mask"][0][:valid].all() or metadata["padding_mask"][0][valid:].any():
                raise RuntimeError(f"{split}:{ordinal}: loss padding mask changed")
            sampled[split].append(
                {
                    "ordinal": ordinal,
                    "sample_id": metadata["sample_id"],
                    "latent_frames_valid": valid,
                    "caption_tokens": int(metadata["prompt"]["attention_mask"].sum()),
                    "present_sources": int(controls["source_present_mask"].sum()),
                }
            )

    # Exercise the real fusion module on two formal training rows without
    # loading the frozen Qwen weights: its actual hidden width and masks are
    # used, and the P10 model config independently pins the Qwen checkpoint.
    batch_items = [datasets["train"][0][1], datasets["train"][expected["train"] - 1][1]]
    controls = [item["sceneplan_442"] for item in batch_items]
    prompt_attention = torch.stack([item["prompt"]["attention_mask"] for item in batch_items])
    conditioner = ScenePlan442Conditioner(
        text_dim=1024,
        event_dim=256,
        position_feature_dim=9,
        per_source_stream_dim=64,
        input_concat_dim=256,
        max_sources=4,
    ).eval()
    torch.manual_seed(20260818)
    with torch.inference_mode():
        fused = conditioner(
            caption_hidden=torch.randn(2, 512, 1024),
            caption_attention_mask=prompt_attention,
            source_semantic_token_masks=torch.stack(
                [value["source_semantic_token_masks"] for value in controls]
            ),
            source_motion_activity_token_masks=torch.stack(
                [value["source_motion_activity_token_masks"] for value in controls]
            ),
            speaker_info_token_mask=torch.stack(
                [value["speaker_info_token_mask"] for value in controls]
            ),
            quoted_transcript_token_mask=torch.stack(
                [value["quoted_transcript_token_mask"] for value in controls]
            ),
            source_present_mask=torch.stack(
                [value["source_present_mask"] for value in controls]
            ),
            source_kind_ids=torch.stack([value["source_kind_ids"] for value in controls]),
            source_slot_ids=torch.stack([value["source_slot_ids"] for value in controls]),
            source_position_activity_features=torch.stack(
                [value["source_position_activity_features"] for value in controls]
            ),
        )
    if tuple(fused["input_concat_cond"].shape) != (2, 256, 432):
        raise RuntimeError("real 4+4+2 fusion output geometry changed")
    if tuple(fused["event_embeddings"].shape) != (2, 4, 256):
        raise RuntimeError("real source event embedding geometry changed")
    summary = {
        "schema": "stable_audio_tools.model_sceneplan_v1_loader_smoke",
        "schema_version": 1,
        "dataset_contract_revision": 5,
        "ok": True,
        "dataset_rows": expected,
        "sampled": sampled,
        "caption_tokens": 512,
        "token_masks": "4+4+2",
        "structured_feature_dim": 9,
        "latent_padded_shape": [64, 432],
        "fusion_output_shape": [2, 256, 432],
        "event_embedding_shape": [2, 4, 256],
        "random_crop": False,
        "p10_training_started": False,
        "p11_training_started": False,
    }
    atomic_write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
