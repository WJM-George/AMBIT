#!/usr/bin/env python3
"""Executable P5 verification for ScenePlan-v2 4+4+2 conditioning."""

from __future__ import annotations
import os

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.sceneplan_v2 import (  # noqa: E402
    compile_442_token_masks,
    compile_renderer_caption,
    compile_structured_source_controls,
    connector_character_mask,
    tokenize_renderer_caption,
)
from stable_audio_tools.models.sceneplan_conditioning import (  # noqa: E402
    ScenePlan442Conditioner,
)


QWEN = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B")
MODEL_CONFIG = (
    REPO_ROOT
    / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_sceneplan_v2.json"
)


def interval(start: int, stop: int) -> dict:
    return {
        "onset_sec": start / 44_100,
        "offset_sec": stop / 44_100,
        "model_onset_sample": start,
        "model_offset_sample": stop,
        "dry_start_sample": 0,
        "dry_end_sample": stop - start,
    }


def keyframe(time_sec: float, azimuth: float, distance: float = 1.5) -> dict:
    return {
        "time_sec": time_sec,
        "position": {
            "azimuth_deg": azimuth,
            "elevation_deg": 0.0,
            "distance_m": distance,
        },
    }


def example() -> dict:
    samples = 176_400
    return {
        "audio": {
            "model_sample_rate_hz": 44_100,
            "model_num_samples": samples,
            "vae_hop_samples": 1024,
            "latent_frames_valid": math.ceil(samples / 1024),
        },
        "sources": [
            {
                "source_id": "source_0",
                "slot": 0,
                "present": True,
                "kind": "speech",
                "description": "an English audiobook narrator",
                "activity": [interval(0, 88_200)],
                "motion": {"type": "static", "keyframes": [keyframe(0.0, -30.0)]},
                "speech": {
                    "speaker_description": "an English audiobook narrator",
                    "transcript": "The complete final word is audible",
                },
            },
            {
                "source_id": "source_1",
                "slot": 1,
                "present": True,
                "kind": "sound",
                "description": "a dog barking",
                "activity": [interval(22_050, 132_300)],
                "motion": {
                    "type": "linear",
                    "keyframes": [keyframe(0.5, 45.0), keyframe(3.0, -90.0, 2.0)],
                },
                "speech": None,
            },
            {
                "source_id": "source_2",
                "slot": 2,
                "present": True,
                "kind": "music",
                "description": "soft piano music",
                "activity": [interval(0, samples)],
                "motion": {"type": "static", "keyframes": [keyframe(0.0, 150.0)]},
                "speech": None,
            },
            {
                "source_id": "source_3",
                "slot": 3,
                "present": False,
                "kind": "empty",
                "description": None,
                "activity": [],
                "motion": None,
                "speech": None,
            },
        ],
    }


def main() -> int:
    scene = example()
    caption = compile_renderer_caption(scene)
    text = caption["text"]
    transcript = caption["transcript_regions"][0]
    assert text[transcript["start"] - 1] == '"'
    assert text[transcript["end"]] == '"'
    assert text[transcript["start"] : transcript["end"]] == (
        "The complete final word is audible"
    )
    assert [
        len(caption["source_semantic_regions"]),
        len(caption["source_motion_activity_regions"]),
        len(caption["speaker_info_regions"]),
        len(caption["transcript_regions"]),
    ] == [3, 3, 1, 1]
    outside = connector_character_mask(caption)
    for index in range(len(text) - 1):
        if text[index : index + 2] == "; ":
            assert outside[index : index + 2].all()
    says = text.index(" says ")
    assert outside[says : says + len(" says ")].all()
    nested_scene = example()
    nested_scene["sources"][0]["speech"]["transcript"] = (
        'He said "go now," then stopped.'
    )
    nested_caption = compile_renderer_caption(nested_scene)
    nested_region = nested_caption["transcript_regions"][0]
    assert nested_caption["text"][nested_region["start"] : nested_region["end"]] == (
        'He said "go now," then stopped.'
    )

    offsets = np.asarray([(index, index + 1) for index in range(len(text))])
    masks = compile_442_token_masks(caption, offsets, np.ones(len(text), dtype=np.uint8))
    assert masks["source_semantic_token_masks"].shape == (4, len(text))
    assert masks["source_motion_activity_token_masks"].shape == (4, len(text))
    assert not masks["source_semantic_token_masks"][3].any()
    assert not masks["source_motion_activity_token_masks"][3].any()
    assert not np.any(
        masks["speaker_info_token_mask"] & masks["quoted_transcript_token_mask"]
    )

    controls = compile_structured_source_controls(scene)
    frames = math.ceil(176_400 / 1024)
    assert controls["source_present_mask"].tolist() == [1, 1, 1, 0]
    assert controls["source_kind_ids"].tolist() == [1, 3, 2, 0]
    assert controls["source_slot_ids"].tolist() == [1, 2, 3, 4]
    assert controls["source_activity_frame_masks"].shape == (4, frames)
    assert controls["source_position_activity_features"].shape == (4, frames, 8)

    token_count = None
    token_mask_sums = None
    if QWEN.is_dir():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
        tokenized = tokenize_renderer_caption(caption, tokenizer, max_length=512)
        token_count = len(tokenized["input_ids"])
        assert 0 < token_count <= 512
        token_mask_sums = {
            "semantic": tokenized["source_semantic_token_masks"].sum(axis=1).tolist(),
            "motion_activity": tokenized[
                "source_motion_activity_token_masks"
            ].sum(axis=1).tolist(),
            "speaker": int(tokenized["speaker_info_token_mask"].sum()),
            "transcript": int(tokenized["quoted_transcript_token_mask"].sum()),
        }
        assert token_mask_sums["transcript"] > 0

    module = ScenePlan442Conditioner(
        text_dim=16,
        event_dim=12,
        per_source_stream_dim=5,
        input_concat_dim=9,
    )
    hidden = torch.randn(2, 20, 16)
    attention = torch.ones(2, 20, dtype=torch.bool)
    semantic = torch.zeros(2, 4, 20, dtype=torch.bool)
    motion = torch.zeros_like(semantic)
    semantic[:, 0, 1:4] = True
    semantic[:, 1, 5:8] = True
    motion[:, 0, 10:12] = True
    motion[:, 1, 12:14] = True
    speaker = torch.zeros(2, 20, dtype=torch.bool)
    quoted = torch.zeros_like(speaker)
    speaker[:, 1:4] = True
    quoted[:, 15:19] = True
    present = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    kinds = torch.tensor([[1, 3, 0, 0], [1, 3, 0, 0]])
    slots = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    position = torch.randn(2, 4, 17, 8)
    position[..., 0] = 0
    position[:, 0, :8, 0] = 1
    position[:, 1, 4:13, 0] = 1
    output = module(
        caption_hidden=hidden,
        caption_attention_mask=attention,
        source_semantic_token_masks=semantic,
        source_motion_activity_token_masks=motion,
        speaker_info_token_mask=speaker,
        quoted_transcript_token_mask=quoted,
        source_present_mask=present,
        source_kind_ids=kinds,
        source_slot_ids=slots,
        source_position_activity_features=position,
    )
    assert output["cross_attention_tokens"] is hidden
    assert output["event_embeddings"].shape == (2, 4, 12)
    assert not output["event_embeddings"][:, 2:].any()
    assert output["position_activity_embeddings"].shape == (2, 4, 17, 5)
    assert not output["position_activity_embeddings"][:, 0, 8:].any()
    assert not output["position_activity_embeddings"][:, 1, :4].any()
    assert not output["position_activity_embeddings"][:, 1, 13:].any()
    assert output["input_concat_cond"].shape == (2, 9, 17)
    assert not output["input_concat_cond"][:, :, 13:].any()

    config = load_config(MODEL_CONFIG)
    assert config["model"]["diffusion"]["cross_attention_cond_ids"] == ["prompt"]
    assert config["model"]["diffusion"]["input_concat_ids"] == ["sceneplan_442"]
    assert config["model"]["diffusion"]["config"]["input_concat_dim"] == 256
    prompt_config = config["model"]["conditioning"]["configs"][0]["config"]
    assert prompt_config["max_length"] == 512
    assert prompt_config["fail_on_truncation"] is True
    assert prompt_config["project_out"] is True
    report = {
        "ok": True,
        "caption_characters": len(text),
        "qwen_tokens": token_count,
        "token_mask_sums": token_mask_sums,
        "structured_shapes": {
            "activity": list(controls["source_activity_frame_masks"].shape),
            "position_activity": list(
                controls["source_position_activity_features"].shape
            ),
        },
        "conditioner_contract": module.contract(),
        "model_routing": {
            "cross_attention": ["prompt"],
            "input_concat": ["sceneplan_442"],
            "sceneplan_token_concat": False,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
