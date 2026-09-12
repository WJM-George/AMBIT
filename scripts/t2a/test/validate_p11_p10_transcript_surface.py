#!/usr/bin/env python3
"""Fail-closed regression for P11's deterministic transcript handoff to P10."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config, validate_t2a_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
    compile_model_44_controls,
    compile_model_semantic_caption,
    compile_model_semantic_caption_v2,
    tokenize_model_semantic_caption,
)
from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    P10_SEMANTIC_CAPTION_COMPILER_VERSION,
    P10_SEMANTIC_CAPTION_CONTRACT,
    P10_SEMANTIC_CAPTION_SURFACE,
    P10_TRANSCRIPT_STATE_AUTHORITY,
    P11_MAX_LATENT_FRAMES,
    compile_p10_aligned_target_conditions,
    finalize_sceneplan_for_p10,
    p10_p11_alignment_contract,
)


DEFAULT_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
)
DEFAULT_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_pilot90_curriculum_pair_aware_transfusion_cot_v4_reliable_asr_v1.json"
)


def _assert_equal_array(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if left.dtype.kind == "f":
        equal = np.array_equal(left, right)
    else:
        equal = np.array_equal(left, right)
    if not equal:
        raise AssertionError(f"{label} changed across the transcript surface boundary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = load_config(args.dataset)
    validate_t2a_config(config, dataset)
    executor = config["model"]["executor"]
    expected_executor = {
        "semantic_caption_contract": P10_SEMANTIC_CAPTION_CONTRACT,
        "semantic_caption_compiler_version": P10_SEMANTIC_CAPTION_COMPILER_VERSION,
        "semantic_caption_surface": P10_SEMANTIC_CAPTION_SURFACE,
        "transcript_state_authority": P10_TRANSCRIPT_STATE_AUTHORITY,
    }
    for key, expected in expected_executor.items():
        if executor.get(key) != expected:
            raise AssertionError(f"executor.{key} is not frozen to {expected!r}")

    transcript = "'Tis John's \"blue\" hour: wait—now!"
    plan = {
        "sample_id": "p11_p10_transcript_surface_regression",
        "duration_sec": (
            P11_MAX_LATENT_FRAMES * VAE_HOP_SAMPLES / MODEL_SAMPLE_RATE
        ),
        "room": {"type": "dry"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "speech",
                "speaker_description": "a calm adult narrator",
                "transcript": transcript,
                "activity": {"onset_sec": 0.0, "offset_sec": 2.0},
                "trajectory": {
                    "type": "static",
                    "position": {
                        "azimuth_deg": -30.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.5,
                    },
                },
                "gain_db": 0.0,
            }
        ],
    }

    codec = load_model_sceneplan_codec(config["model"]["text"]["plan_codec_path"])
    projected = codec.project_plan(plan)
    encoded = codec.encode(projected, max_tokens=1024)
    decoded = codec.decode(encoded["input_ids"], sample_id=plan["sample_id"])
    if decoded["sources"][0]["transcript"] != transcript:
        raise AssertionError("codec-v4 did not preserve the exact transcript")

    legacy = compile_model_semantic_caption(decoded)
    canonical = compile_model_semantic_caption_v2(decoded)
    if int(canonical["compiler_version"]) != P10_SEMANTIC_CAPTION_COMPILER_VERSION:
        raise AssertionError("canonical handoff did not use semantic compiler v2")
    if f"who says: {transcript}" not in canonical["text"]:
        raise AssertionError("canonical P10 transcript surface is not colon-delimited")
    if f'who says "{transcript}"' not in legacy["text"]:
        raise AssertionError("v1 control surface did not retain protocol quotes")
    if legacy["text"] == canonical["text"]:
        raise AssertionError("v1/v2 transcript surface controls unexpectedly match")
    speech_region = canonical["speech_regions"][0]
    if canonical["text"][speech_region["start"] : speech_region["end"]] != transcript:
        raise AssertionError("v2 speech region is not the exact ScenePlan transcript")

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text"]["model_path"])
    tokenized = tokenize_model_semantic_caption(canonical, tokenizer, max_length=512)
    event_ids = tokenized["event_source_ids"]
    speech_ids = tokenized["speech_source_ids"]
    if np.any((event_ids > 0) & (speech_ids > 0)):
        raise AssertionError("event and speech token roles overlap")
    if set(int(value) for value in event_ids if value > 0) != {1}:
        raise AssertionError("speaker event role lost source_0 ownership")
    if set(int(value) for value in speech_ids if value > 0) != {1}:
        raise AssertionError("exact transcript role lost source_0 ownership")

    colon_offset = canonical["event_regions"][0]["end"]
    if canonical["text"][colon_offset] != ":":
        raise AssertionError("event region is not followed by the v2 colon")
    colon_tokens = [
        index
        for index, ((start, end), valid) in enumerate(
            zip(tokenized["offset_mapping"], tokenized["attention_mask"])
        )
        if valid and int(start) <= colon_offset < int(end)
    ]
    if not colon_tokens or any(
        int(event_ids[index]) != 0 or int(speech_ids[index]) != 0
        for index in colon_tokens
    ):
        raise AssertionError("v2 separator colon is not a neutral role-0 token")

    samples = P11_MAX_LATENT_FRAMES * VAE_HOP_SAMPLES
    direct_controls = compile_model_44_controls(
        decoded,
        model_num_samples=samples,
        latent_frames_valid=P11_MAX_LATENT_FRAMES,
    )
    handoff = compile_p10_aligned_target_conditions(
        decoded,
        model_num_samples=samples,
        latent_frames_valid=P11_MAX_LATENT_FRAMES,
    )
    for key in (
        "source_event_frame_ids",
        "source_trajectory_features",
        "speech_active_frame_mask",
    ):
        _assert_equal_array(direct_controls[key], handoff["sceneplan_44"][key], key)

    bundle = finalize_sceneplan_for_p10(
        codec,
        encoded,
        tokenizer=tokenizer,
        task="generation",
        sample_id=plan["sample_id"],
    )
    metadata = bundle.p10_metadata
    if metadata["prompt_text"] != canonical["text"]:
        raise AssertionError("runtime bundle did not carry the canonical v2 text")
    for key, expected in expected_executor.items():
        metadata_key = key
        if key == "semantic_caption_surface":
            continue
        if metadata.get(metadata_key) != expected:
            raise AssertionError(f"runtime metadata {metadata_key} changed")
    if bundle.latent_frames_valid != P11_MAX_LATENT_FRAMES:
        raise AssertionError("max-duration transcript handoff was truncated")

    alignment = p10_p11_alignment_contract()
    if alignment["p10_semantic_caption_contract"] != P10_SEMANTIC_CAPTION_CONTRACT:
        raise AssertionError("alignment contract does not name semantic-caption v2")
    report = {
        "schema": "stable_audio_tools.p11_p10_transcript_surface_report",
        "schema_version": 1,
        "status": "PASS",
        "semantic_caption_contract": P10_SEMANTIC_CAPTION_CONTRACT,
        "compiler_version": P10_SEMANTIC_CAPTION_COMPILER_VERSION,
        "surface": P10_SEMANTIC_CAPTION_SURFACE,
        "transcript_state_authority": P10_TRANSCRIPT_STATE_AUTHORITY,
        "codec_exact_transcript_roundtrip": True,
        "separator_colon_role": 0,
        "event_speech_positive_roles_disjoint": True,
        "v1_v2_structured_controls_identical": True,
        "latent_frames": bundle.latent_frames_valid,
        "no_manifest_rebuild_required": True,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
