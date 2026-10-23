from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import torch
from .schema import (
    compile_model_44_controls,
    compile_model_semantic_caption_v2,
    make_sceneplan_cfg_unknown_metadata,
    tokenize_model_semantic_caption,
    validate_model_sceneplan,
)


def compile_plan_condition(
    model_sceneplan: Mapping[str, Any],
    *,
    tokenizer: Any,
    model_num_samples: int,
    latent_frames_valid: int,
    latent_crop_length: int,
    caption_max_tokens: int = 512,
) -> dict[str, Any]:
    """Compile one complete new ScenePlan into the exact Editing-DiT condition."""
    validate_model_sceneplan(model_sceneplan)
    valid_frames = int(latent_frames_valid)
    crop_length = int(latent_crop_length)
    if not 1 <= valid_frames <= crop_length <= 648:
        raise ValueError("Editing-DiT plan condition has invalid latent geometry")
    semantic_caption = compile_model_semantic_caption_v2(model_sceneplan)
    tokenized = tokenize_model_semantic_caption(
        semantic_caption, tokenizer, max_length=int(caption_max_tokens)
    )
    prompt = {
        "input_ids": torch.as_tensor(tokenized["input_ids"], dtype=torch.long),
        "attention_mask": torch.as_tensor(tokenized["attention_mask"], dtype=torch.bool),
        "event_source_ids": torch.as_tensor(tokenized["event_source_ids"], dtype=torch.int8),
        "speech_source_ids": torch.as_tensor(tokenized["speech_source_ids"], dtype=torch.int8),
        "speech_lexical_mask": torch.as_tensor(tokenized["speech_lexical_mask"], dtype=torch.bool),
    }
    structured = compile_model_44_controls(
        model_sceneplan, model_num_samples=int(model_num_samples), latent_frames_valid=valid_frames
    )
    if structured["source_trajectory_features"].shape != (4, valid_frames, 5):
        raise RuntimeError("compiled Editing-DiT trajectory shape changed")
    trajectory = np.zeros((4, crop_length, 5), dtype=np.float32)
    trajectory[:, :valid_frames] = structured["source_trajectory_features"]
    event_ids = np.zeros((4, crop_length), dtype=np.int8)
    event_ids[:, :valid_frames] = structured["source_event_frame_ids"]
    frame_valid = torch.zeros(crop_length, dtype=torch.bool)
    frame_valid[:valid_frames] = True
    speech_active = np.zeros(crop_length, dtype=np.uint8)
    speech_active[:valid_frames] = structured["speech_active_frame_mask"]
    controls = {
        "source_event_frame_ids": torch.as_tensor(event_ids, dtype=torch.int8),
        "source_trajectory_features": torch.as_tensor(trajectory),
        "frame_valid_mask": frame_valid,
        "speech_active_frame_mask": torch.as_tensor(speech_active, dtype=torch.bool),
    }
    return {
        "model_sceneplan": model_sceneplan,
        "prompt": prompt,
        "prompt_text": semantic_caption["text"],
        "semantic_caption_compiler_version": int(semantic_caption["compiler_version"]),
        "sceneplan_44": controls,
    }


def make_editing_dit_cfg_unknown_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Unknown the new-plan branches while retaining the exact clean source.

    Editing guidance contrasts ``[new plan, source]`` with
    ``[unknown plan, same source]``.  Dropping the source in the negative
    branch would turn CFG into an edit-vs-generation contrast and undermine
    waveform preservation.
    """
    source = metadata.get("source_foa_latent")
    if not isinstance(source, torch.Tensor) or source.ndim != 2 or source.shape[0] != 64:
        raise ValueError("Editing CFG requires a clean aligned [64,T] source latent")
    negative = make_sceneplan_cfg_unknown_metadata(metadata)
    if negative.get("source_foa_latent") is not source:
        raise RuntimeError("Editing CFG changed or copied the source latent")
    return negative
