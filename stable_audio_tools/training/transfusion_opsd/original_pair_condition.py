"""Compile an original Generation pair without routing through AR quantization.

The caller verifies dataset/latent provenance. This function enforces geometry
and uses the existing P10 compilers and tokenizer. It supplies no new teacher
and does not change normal AR inference. Editing uses its native joint dataset.
"""
from __future__ import annotations

import math

import torch

from .adapters import RenderCondition


def original_generation_pair_condition(sceneplan, tokenizer, *, model_num_samples,
                                       latent_frames_valid, device):
    from ...data.model_sceneplan import (
        MODEL_SAMPLE_RATE, VAE_HOP_SAMPLES, validate_model_sceneplan,
        tokenize_model_semantic_caption, make_sceneplan_cfg_unknown_metadata,
    )
    from ...data.sceneplan_p11_single_turn import (
        compile_p10_aligned_target_conditions, SEMANTIC_CAPTION_MAX_TOKENS,
        P10_SEMANTIC_CAPTION_COMPILER_VERSION, P10_SEMANTIC_CAPTION_CONTRACT,
        P10_TRANSCRIPT_STATE_AUTHORITY, P10_MAX_LATENT_FRAMES,
    )

    plan=validate_model_sceneplan(sceneplan)
    if (type(model_num_samples) is not int or model_num_samples<=0
            or type(latent_frames_valid) is not int
            or not 1<=latent_frames_valid<=P10_MAX_LATENT_FRAMES
            or round(plan['duration_sec']*MODEL_SAMPLE_RATE)!=model_num_samples
            or math.ceil(model_num_samples/VAE_HOP_SAMPLES)!=latent_frames_valid):
        raise ValueError('Original paired duration, sample count and latent frame count must agree.')
    compiled=compile_p10_aligned_target_conditions(plan,model_num_samples=model_num_samples,
        latent_frames_valid=latent_frames_valid)
    tokenized=tokenize_model_semantic_caption(compiled['semantic_caption'],tokenizer,
        max_length=SEMANTIC_CAPTION_MAX_TOKENS)
    prompt={key:torch.as_tensor(tokenized[key],dtype=dtype) for key,dtype in [
        ('input_ids',torch.long),('attention_mask',torch.bool),('event_source_ids',torch.int8),
        ('speech_source_ids',torch.int8),('speech_lexical_mask',torch.bool)]}
    valid=torch.ones(latent_frames_valid,dtype=torch.bool)
    controls={key:torch.as_tensor(compiled['sceneplan_44'][key],dtype=dtype) for key,dtype in [
        ('source_event_frame_ids',torch.int8),('source_trajectory_features',torch.float32),
        ('speech_active_frame_mask',torch.bool)]}
    controls['frame_valid_mask']=valid.clone()
    metadata=dict(sample_id=plan['sample_id'],model_sceneplan=plan,model_num_samples=model_num_samples,
        prompt=prompt,prompt_text=compiled['semantic_caption']['text'],sceneplan_44=controls,
        semantic_caption_compiler_version=P10_SEMANTIC_CAPTION_COMPILER_VERSION,
        semantic_caption_contract=P10_SEMANTIC_CAPTION_CONTRACT,
        transcript_state_authority=P10_TRANSCRIPT_STATE_AUTHORITY,padding_mask=[valid],
        seconds_start=0.,seconds_total=model_num_samples/MODEL_SAMPLE_RATE,
        latent_stored_length=latent_frames_valid,latent_crop_length=latent_frames_valid,latent_crop_start=0,
        pair_condition_contract='original_verified_pair_without_AR_codec_projection_v1')
    return RenderCondition(plan,[metadata],[make_sceneplan_cfg_unknown_metadata(metadata)],
        valid[None].to(device),model_num_samples)
