from __future__ import annotations

import os
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from stable_audio_tools.data.sceneplan_v2 import (
    compile_442_token_masks,
    compile_renderer_caption,
    compile_structured_source_controls,
    connector_character_mask,
    tokenize_renderer_caption,
)
from stable_audio_tools.models.sceneplan_conditioning import ScenePlan442Conditioner
from scripts.t2a.data.build_sceneplan_manifests_v2 import speech_source


QWEN_TOKENIZER = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B")


def _interval(start: int, stop: int) -> dict:
    return {
        "onset_sec": start / 44_100,
        "offset_sec": stop / 44_100,
        "model_onset_sample": start,
        "model_offset_sample": stop,
        "dry_start_sample": 0,
        "dry_end_sample": stop - start,
    }


def _keyframe(time_sec: float, azimuth: float, distance: float = 1.5) -> dict:
    return {
        "time_sec": time_sec,
        "position": {
            "azimuth_deg": azimuth,
            "elevation_deg": 0.0,
            "distance_m": distance,
        },
    }


def scene_plan() -> dict:
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
                "activity": [_interval(0, 88_200)],
                "motion": {"type": "static", "keyframes": [_keyframe(0.0, -30.0)]},
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
                "activity": [_interval(22_050, 132_300)],
                "motion": {
                    "type": "linear",
                    "keyframes": [_keyframe(0.5, 45.0), _keyframe(3.0, -90.0, 2.0)],
                },
                "speech": None,
            },
            {
                "source_id": "source_2",
                "slot": 2,
                "present": True,
                "kind": "music",
                "description": "soft piano music",
                "activity": [_interval(0, samples)],
                "motion": {"type": "static", "keyframes": [_keyframe(0.0, 150.0)]},
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


def test_caption_has_strict_quote_and_connector_free_regions() -> None:
    caption = compile_renderer_caption(scene_plan())
    text = caption["text"]
    transcript = caption["transcript_regions"][0]
    assert text[transcript["start"] - 1] == '"'
    assert text[transcript["end"]] == '"'
    assert text[transcript["start"] : transcript["end"]] == (
        "The complete final word is audible"
    )
    assert len(caption["source_semantic_regions"]) == 3
    assert len(caption["source_motion_activity_regions"]) == 3
    assert len(caption["speaker_info_regions"]) == 1
    assert len(caption["transcript_regions"]) == 1
    outside = connector_character_mask(caption)
    for match in (index for index in range(len(text) - 1) if text[index : index + 2] == "; "):
        assert outside[match : match + 2].all()
    says = text.index(" says ")
    assert outside[says : says + len(" says ")].all()


def test_caption_preserves_transcript_internal_double_quotes() -> None:
    plan = scene_plan()
    plan["sources"][0]["speech"]["transcript"] = 'He said "go now," then stopped.'
    caption = compile_renderer_caption(plan)
    region = caption["transcript_regions"][0]
    assert caption["text"][region["start"] : region["end"]] == (
        'He said "go now," then stopped.'
    )
    assert caption["text"][region["start"] - 1] == '"'
    assert caption["text"][region["end"]] == '"'


def test_sceneplan_speech_source_canonicalizes_only_boundary_whitespace() -> None:
    row = {
        "asset_id": "hifi_tts:example",
        "source_dataset": "hifi_tts",
        "source_audio_sha256": "a" * 64,
        "native_sample_rate_hz": 44_100,
        "native_num_samples": 44_100,
        "model_num_samples": 44_100,
        "parquet_path": "/tmp/example.parquet",
        "row_group": 0,
        "row_in_group": 0,
        "speaker_id": "speaker",
        "renderer_text": '  He said "go now," then stopped.  ',
    }
    source = speech_source(row)
    assert source["speech"]["transcript"] == 'He said "go now," then stopped.'
    assert source["speech"]["transcript_normalization"] == (
        "punctuation_case_whitespace_only_v1"
    )


def test_character_aligned_442_masks_exclude_wrappers_and_empty_slot() -> None:
    caption = compile_renderer_caption(scene_plan())
    text = caption["text"]
    offsets = np.asarray([(index, index + 1) for index in range(len(text))])
    masks = compile_442_token_masks(caption, offsets, np.ones(len(text), dtype=np.uint8))
    assert masks["source_semantic_token_masks"].shape == (4, len(text))
    assert masks["source_motion_activity_token_masks"].shape == (4, len(text))
    assert not masks["source_semantic_token_masks"][3].any()
    assert not masks["source_motion_activity_token_masks"][3].any()
    says = text.index("says")
    combined = np.concatenate(
        [
            masks["source_semantic_token_masks"],
            masks["source_motion_activity_token_masks"],
            masks["speaker_info_token_mask"][None, :],
            masks["quoted_transcript_token_mask"][None, :],
        ]
    )
    assert not combined[:, says : says + 4].any()
    transcript = caption["transcript_regions"][0]
    assert masks["quoted_transcript_token_mask"][transcript["start"] : transcript["end"]].sum() > 0
    assert not np.any(
        masks["speaker_info_token_mask"] & masks["quoted_transcript_token_mask"]
    )


def test_structured_controls_have_four_bound_streams_and_variable_frames() -> None:
    controls = compile_structured_source_controls(scene_plan())
    frames = math.ceil(176_400 / 1024)
    assert controls["source_present_mask"].tolist() == [1, 1, 1, 0]
    assert controls["source_kind_ids"].tolist() == [1, 3, 2, 0]
    assert controls["source_slot_ids"].tolist() == [1, 2, 3, 4]
    assert controls["source_activity_frame_masks"].shape == (4, frames)
    assert controls["source_position_activity_features"].shape == (4, frames, 8)
    assert not controls["source_position_activity_features"][3].any()
    assert controls["source_position_activity_features"][1, :, 7].max() == 1.0


@pytest.mark.skipif(not QWEN_TOKENIZER.is_dir(), reason="local Qwen tokenizer unavailable")
def test_real_qwen_offsets_compile_without_truncation() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_TOKENIZER, local_files_only=True)
    compiled = tokenize_renderer_caption(
        compile_renderer_caption(scene_plan()), tokenizer, max_length=256
    )
    length = len(compiled["input_ids"])
    assert 0 < length <= 256
    assert compiled["source_semantic_token_masks"].shape == (4, length)
    assert compiled["source_motion_activity_token_masks"].shape == (4, length)
    assert compiled["quoted_transcript_token_mask"].sum() > 0


def test_sceneplan_conditioner_keeps_qwen_cross_attention_and_concats_sources() -> None:
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
    transcript = torch.zeros_like(speaker)
    speaker[:, 1:4] = True
    transcript[:, 15:19] = True
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
        quoted_transcript_token_mask=transcript,
        source_present_mask=present,
        source_kind_ids=kinds,
        source_slot_ids=slots,
        source_position_activity_features=position,
    )
    assert output["cross_attention_tokens"] is hidden
    assert output["event_embeddings"].shape == (2, 4, 12)
    assert not output["event_embeddings"][:, 2:].any()
    assert output["position_activity_embeddings"].shape == (2, 4, 17, 5)
    assert not output["position_activity_embeddings"][:, 2:].any()
    assert not output["position_activity_embeddings"][:, 0, 8:].any()
    assert not output["position_activity_embeddings"][:, 1, :4].any()
    assert not output["position_activity_embeddings"][:, 1, 13:].any()
    assert output["input_concat_cond"].shape == (2, 9, 17)
    assert not output["input_concat_cond"][:, :, 13:].any()
