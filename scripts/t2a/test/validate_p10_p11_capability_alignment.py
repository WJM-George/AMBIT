#!/usr/bin/env python3
"""Fail-closed audit of the P11 planner profile against the active P10 DiT."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sqlite3
import sys
import zlib
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config, validate_t2a_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    MAX_LATENT_FRAMES as COMPILER_MAX_FRAMES,
    MAX_MODEL_SAMPLES,
    MAX_SOURCES,
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
)
from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    MAX_FRAMES as CODEC_MAX_FRAMES,
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    P10_CANONICAL_CHECKPOINT,
    P10_CANONICAL_CHECKPOINT_SHA256,
    P10_CANONICAL_CHECKPOINT_STEP,
    P10_CANONICAL_EXECUTOR_FAMILY,
    P10_CANONICAL_MODEL_CONFIG,
    P10_CANONICAL_MODEL_CONFIG_SHA256,
    P10_MAX_LATENT_FRAMES,
    P10_SEMANTIC_CAPTION_COMPILER_VERSION,
    P10_SEMANTIC_CAPTION_CONTRACT,
    P10_SEMANTIC_CAPTION_SURFACE,
    P10_TRANSCRIPT_STATE_AUTHORITY,
    P11_GAIN_POLICY,
    P11_MAX_LATENT_FRAMES,
    P11_SUPPORTED_MOTION_TYPES,
    compile_p10_aligned_target_conditions,
    p10_p11_alignment_contract,
    validate_p11_executor_profile,
)


DEFAULT_CAPABILITY = (
    REPO_ROOT / "docs/sceneplan_v2/p10_sceneplan_44_capability_v1_20260830.json"
)
DEFAULT_P10_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s_resume_cosine_40k.json"
)
DEFAULT_P11_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
)
DEFAULT_P11_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_pilot90_curriculum_pair_aware_transfusion_cot_v4_reliable_asr_v1.json"
)
DEFAULT_P10_INDEX = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/training_index/train.sqlite"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _read_plan(index: Path, where: str) -> tuple[dict[str, Any], int, int]:
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    try:
        row = connection.execute(
            "SELECT scene_plan_zlib, model_num_samples, latent_frames_valid "
            f"FROM samples WHERE {where} LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise RuntimeError(f"no P10 index row satisfies {where!r}")
    return json.loads(zlib.decompress(row[0])), int(row[1]), int(row[2])


def _expect_rejected(plan: dict[str, Any], fragment: str) -> str:
    try:
        validate_p11_executor_profile(plan)
    except ValueError as error:
        message = str(error)
        if fragment not in message:
            raise AssertionError(f"unexpected rejection: {message}") from error
        return message
    raise AssertionError("out-of-profile ScenePlan was accepted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability", type=Path, default=DEFAULT_CAPABILITY)
    parser.add_argument("--p10-config", type=Path, default=DEFAULT_P10_CONFIG)
    parser.add_argument("--p11-config", type=Path, default=DEFAULT_P11_CONFIG)
    parser.add_argument("--p11-dataset", type=Path, default=DEFAULT_P11_DATASET)
    parser.add_argument("--p10-index", type=Path, default=DEFAULT_P10_INDEX)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    capability_path = args.capability.expanduser().resolve(strict=True)
    capability = json.loads(capability_path.read_text(encoding="utf-8"))
    if capability.get("schema") != "stable_audio_tools.p10_executor_capability":
        raise AssertionError("wrong P10 capability schema")
    if capability.get("contract") != "p10_sceneplan_44_capability_v1":
        raise AssertionError("wrong P10 capability contract")

    p10 = load_config(args.p10_config)
    p11 = load_config(args.p11_config)
    p11_dataset = load_config(args.p11_dataset)
    validate_t2a_config(p10, load_config(
        REPO_ROOT / "stable_audio_tools/configs/dataset_configs/"
        "sceneplan_v2_speech_expansion_noalign_15s_v1_train_semantic_v2.json"
    ))
    validate_t2a_config(p11, p11_dataset)

    executor = capability["executor"]
    assert int(p10["sample_size"]) == int(executor["max_model_num_samples"])
    assert int(p10["sample_rate"]) == MODEL_SAMPLE_RATE == int(executor["sample_rate_hz"])
    assert int(p10["audio_channels"]) == int(executor["foa_channels"]) == 4
    assert COMPILER_MAX_FRAMES == P10_MAX_LATENT_FRAMES == int(executor["max_latent_frames"])
    assert MAX_MODEL_SAMPLES == P10_MAX_LATENT_FRAMES * VAE_HOP_SAMPLES
    assert MAX_SOURCES == int(executor["max_sources"]) == 4
    assert executor["family"] == P10_CANONICAL_EXECUTOR_FAMILY
    assert int(executor["canonical_checkpoint_step"]) == P10_CANONICAL_CHECKPOINT_STEP
    assert executor["canonical_checkpoint"] == P10_CANONICAL_CHECKPOINT
    assert executor["canonical_checkpoint_sha256"] == P10_CANONICAL_CHECKPOINT_SHA256
    assert executor["canonical_model_config"] == P10_CANONICAL_MODEL_CONFIG
    assert (
        executor["canonical_model_config_sha256"]
        == P10_CANONICAL_MODEL_CONFIG_SHA256
    )

    conditioning = p10["model"]["conditioning"]["configs"]
    by_id = {item["id"]: item for item in conditioning}
    local = by_id["sceneplan_44"]["config"]
    assert int(local["max_sources"]) == 4
    assert int(local["trajectory_feature_dim"]) == 5
    assert int(local["output_dim"]) == 256
    diffusion = p10["model"]["diffusion"]
    assert diffusion["cross_attention_cond_ids"] == ["prompt"]
    assert diffusion["input_concat_ids"] == ["sceneplan_44"]
    semantic_route = capability["conditioning_routes"]["semantic_cross_attention"]
    assert (
        semantic_route["semantic_caption_contract"]
        == P10_SEMANTIC_CAPTION_CONTRACT
    )
    assert (
        int(semantic_route["semantic_caption_compiler_version"])
        == P10_SEMANTIC_CAPTION_COMPILER_VERSION
    )
    assert semantic_route["semantic_caption_surface"] == P10_SEMANTIC_CAPTION_SURFACE
    assert (
        semantic_route["transcript_state_authority"]
        == P10_TRANSCRIPT_STATE_AUTHORITY
    )

    profile = capability["p11_canonical_profile"]
    assert CODEC_MAX_FRAMES == P11_MAX_LATENT_FRAMES == int(profile["max_latent_frames"])
    assert P11_MAX_LATENT_FRAMES == P10_MAX_LATENT_FRAMES
    assert profile["temporal_envelope_matches_executor"] is True
    assert tuple(profile["motion_types"]) == P11_SUPPORTED_MOTION_TYPES
    assert profile["gain_policy"] == P11_GAIN_POLICY
    p11_executor = p11["model"]["executor"]
    assert p11_executor["capability_contract"] == capability["contract"]
    assert (
        p11_executor["semantic_caption_contract"]
        == P10_SEMANTIC_CAPTION_CONTRACT
    )
    assert (
        int(p11_executor["semantic_caption_compiler_version"])
        == P10_SEMANTIC_CAPTION_COMPILER_VERSION
    )
    assert p11_executor["semantic_caption_surface"] == P10_SEMANTIC_CAPTION_SURFACE
    assert (
        p11_executor["transcript_state_authority"]
        == P10_TRANSCRIPT_STATE_AUTHORITY
    )
    assert p11_executor["canonical_executor_family"] == P10_CANONICAL_EXECUTOR_FAMILY
    assert p11_executor["canonical_model_config"] == P10_CANONICAL_MODEL_CONFIG
    assert (
        p11_executor["canonical_model_config_sha256"]
        == P10_CANONICAL_MODEL_CONFIG_SHA256
    )
    assert (
        int(p11_executor["canonical_checkpoint_step"])
        == P10_CANONICAL_CHECKPOINT_STEP
    )
    assert p11_executor["canonical_checkpoint"] == P10_CANONICAL_CHECKPOINT
    assert (
        p11_executor["canonical_checkpoint_sha256"]
        == P10_CANONICAL_CHECKPOINT_SHA256
    )
    assert int(p11_executor["p10_max_latent_frames"]) == P10_MAX_LATENT_FRAMES
    assert int(p11_executor["planner_max_latent_frames"]) == P11_MAX_LATENT_FRAMES
    assert tuple(p11_executor["planner_motion_types"]) == P11_SUPPORTED_MOTION_TYPES
    assert p11_executor["gain_policy"] == P11_GAIN_POLICY
    assert p11_executor["word_level_timing_supported"] is False

    codec = load_model_sceneplan_codec(p11["model"]["text"]["plan_codec_path"])
    if not isinstance(codec, ModelScenePlanCodecV4):
        raise AssertionError("P11 does not use the canonical 648-frame codec-v4")
    short_plan, short_samples, short_frames = _read_plan(
        Path(p11_dataset["datasets"][0]["path"]), "latent_frames_valid <= 432"
    )
    short_projected = codec.project_plan(short_plan)
    accepted = validate_p11_executor_profile(short_projected)
    short_handoff = compile_p10_aligned_target_conditions(
        accepted,
        model_num_samples=short_frames * VAE_HOP_SAMPLES,
        latent_frames_valid=short_frames,
    )
    assert short_handoff["semantic_caption"]["compiler_version"] == 2
    controls = short_handoff["sceneplan_44"]
    assert controls["source_event_frame_ids"].shape == (4, short_frames)
    assert controls["source_trajectory_features"].shape == (4, short_frames, 5)

    long_plan, long_samples, long_frames = _read_plan(
        args.p10_index.expanduser().resolve(strict=True), "latent_frames_valid = 648"
    )
    long_projected = codec.project_plan(long_plan)
    long_accepted = validate_p11_executor_profile(long_projected)
    long_handoff = compile_p10_aligned_target_conditions(
        long_accepted,
        model_num_samples=long_frames * VAE_HOP_SAMPLES,
        latent_frames_valid=long_frames,
    )
    assert long_handoff["semantic_caption"]["compiler_version"] == 2
    long_controls = long_handoff["sceneplan_44"]
    assert long_controls["source_event_frame_ids"].shape == (4, long_frames)
    assert long_controls["source_trajectory_features"].shape == (4, long_frames, 5)

    keyframed = copy.deepcopy(short_projected)
    source = keyframed["sources"][0]
    trajectory = source["trajectory"]
    start = trajectory.get("start", trajectory.get("position"))
    end = trajectory.get("end", trajectory.get("position"))
    source["trajectory"] = {
        "type": "keyframed",
        "keyframes": [
            {"time_sec": source["activity"]["onset_sec"], "position": start},
            {"time_sec": source["activity"]["offset_sec"], "position": end},
        ],
    }
    keyframe_rejection = _expect_rejected(keyframed, "outside the trained P11-v2 profile")

    nonzero_gain = copy.deepcopy(short_projected)
    nonzero_gain["sources"][0]["gain_db"] = 1.0
    gain_rejection = _expect_rejected(nonzero_gain, "gain_db is not a P10 condition")

    alignment = p10_p11_alignment_contract()
    assert int(alignment["p10_max_latent_frames"]) == P10_MAX_LATENT_FRAMES
    assert int(alignment["p11_profile_max_latent_frames"]) == P11_MAX_LATENT_FRAMES
    assert alignment["p11_duration_matches_p10"] is True
    assert alignment["p11_is_strict_p10_subset"] is False
    assert alignment["word_level_timing_supported"] is False
    assert (
        alignment["p10_semantic_caption_contract"]
        == P10_SEMANTIC_CAPTION_CONTRACT
    )
    assert (
        int(alignment["p10_semantic_caption_compiler_version"])
        == P10_SEMANTIC_CAPTION_COMPILER_VERSION
    )
    assert alignment["p10_semantic_caption_surface"] == P10_SEMANTIC_CAPTION_SURFACE
    assert (
        alignment["transcript_state_authority"]
        == P10_TRANSCRIPT_STATE_AUTHORITY
    )

    resolved_hash = _canonical_sha256(p10)
    if resolved_hash != capability["evidence"]["resolved_active_model_config_sha256"]:
        raise AssertionError("active resolved P10 config hash changed")
    report = {
        "schema": "stable_audio_tools.p10_p11_capability_alignment_report",
        "schema_version": 2,
        "status": "PASS",
        "capability_contract": capability["contract"],
        "capability_sha256": _sha256(capability_path.read_bytes()),
        "p10": {
            "max_latent_frames": P10_MAX_LATENT_FRAMES,
            "max_duration_sec": P10_MAX_LATENT_FRAMES * VAE_HOP_SAMPLES / MODEL_SAMPLE_RATE,
            "max_sources": MAX_SOURCES,
            "hard_control_shape": [4, "T", 5],
            "semantic_route": "frozen_qwen_cross_attention",
            "semantic_caption_contract": P10_SEMANTIC_CAPTION_CONTRACT,
            "semantic_caption_compiler_version": (
                P10_SEMANTIC_CAPTION_COMPILER_VERSION
            ),
            "semantic_caption_surface": P10_SEMANTIC_CAPTION_SURFACE,
            "transcript_state_authority": P10_TRANSCRIPT_STATE_AUTHORITY,
        },
        "p11_canonical_profile": {
            "max_latent_frames": P11_MAX_LATENT_FRAMES,
            "max_duration_sec": P11_MAX_LATENT_FRAMES * VAE_HOP_SAMPLES / MODEL_SAMPLE_RATE,
            "motion_types": list(P11_SUPPORTED_MOTION_TYPES),
            "gain_policy": P11_GAIN_POLICY,
            "temporal_envelope_matches_executor": True,
        },
        "positive_control": {
            "short": {
                "frames": short_frames,
                "model_num_samples_in_index": short_samples,
                "compiled_event_shape": list(controls["source_event_frame_ids"].shape),
                "compiled_trajectory_shape": list(controls["source_trajectory_features"].shape),
            },
            "long": {
                "frames": long_frames,
                "model_num_samples_in_index": long_samples,
                "compiled_event_shape": list(long_controls["source_event_frame_ids"].shape),
                "compiled_trajectory_shape": list(long_controls["source_trajectory_features"].shape),
            },
        },
        "negative_controls": {
            "keyframed_rejection": keyframe_rejection,
            "nonzero_gain_rejection": gain_rejection,
        },
        "claim_gaps": capability["capability_tiers"]["empirically_verified_150k"][
            "not_verified_by_this_panel"
        ],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
