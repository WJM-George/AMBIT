#!/usr/bin/env python3
"""Read-only real-checkpoint smoke for the P11 -> external P10 handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        type=Path,
        default=REPO_ROOT
        / "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
            "qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s_resume_cosine_40k.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/mnt/sdc/ckpts/dit/"
            "sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
            "checkpoints/epoch=48-step=150000.ckpt"
        ),
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=Path(
            "/mnt/sdc/ckpts/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
        ),
    )
    parser.add_argument(
        "--codec",
        type=Path,
        default=Path(
            "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
            "model_sceneplan_codec_v4"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "artifacts/sceneplan_p11/p11_v4_p10_same_seed_handoff_20260901.json",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from stable_audio_tools.data.model_sceneplan_codec import load_model_sceneplan_codec
    from stable_audio_tools.data.sceneplan_edit_patch import ScenePlanEditPatchCodec
    from stable_audio_tools.data.sceneplan_p11_single_turn import (
        P10_CANONICAL_CHECKPOINT,
        P10_CANONICAL_CHECKPOINT_SHA256,
        P10_CANONICAL_CHECKPOINT_STEP,
        P10_CANONICAL_MODEL_CONFIG,
        P10_CANONICAL_MODEL_CONFIG_SHA256,
        P10_SEMANTIC_CAPTION_COMPILER_VERSION,
        P10_SEMANTIC_CAPTION_CONTRACT,
        P11Task,
        finalize_sceneplan_for_p10,
    )
    from stable_audio_tools.inference.sceneplan_cot import P10ScenePlanDiTExecutor

    plan = {
        "sample_id": "p11_external_p10_smoke",
        "duration_sec": 0.1,
        "room": {"type": "dry"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "sound",
                "description": (
                    "A small metal bell rings once with a clear bright tone."
                ),
                "activity": {"onset_sec": 0.0, "offset_sec": 0.1},
                "trajectory": {
                    "type": "linear",
                    "start": {
                        "azimuth_deg": -60.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.0,
                    },
                    "end": {
                        "azimuth_deg": 60.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.5,
                    },
                },
                "gain_db": 0.0,
            }
        ],
    }
    codec = load_model_sceneplan_codec(args.codec.resolve(strict=True))
    encoded = codec.encode(plan, max_tokens=1024)
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B",
        local_files_only=True,
        use_fast=True,
    )
    bundle = finalize_sceneplan_for_p10(
        codec,
        encoded,
        tokenizer=tokenizer,
        task=P11Task.GENERATION,
        sample_id=plan["sample_id"],
    )
    model_config_path = args.model_config.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    if str(model_config_path) != P10_CANONICAL_MODEL_CONFIG:
        raise RuntimeError(f"non-canonical P10 model config: {model_config_path}")
    if sha256(model_config_path) != P10_CANONICAL_MODEL_CONFIG_SHA256:
        raise RuntimeError("canonical P10 model-config SHA256 changed")
    if str(checkpoint_path) != P10_CANONICAL_CHECKPOINT:
        raise RuntimeError(f"non-canonical P10 checkpoint: {checkpoint_path}")
    checkpoint_sha256 = sha256(checkpoint_path)
    if checkpoint_sha256 != P10_CANONICAL_CHECKPOINT_SHA256:
        raise RuntimeError("canonical P10 checkpoint SHA256 changed")
    executor = P10ScenePlanDiTExecutor.from_checkpoints(
        model_config_path=model_config_path,
        checkpoint_path=checkpoint_path,
        vae_checkpoint_path=args.vae_checkpoint,
        device=args.device,
        steps=args.steps,
        cfg_scale=1.0,
        rescale_cfg=False,
        apg_scale=0.0,
    )
    render_seed = 20260822
    foa = executor.render(bundle, seed=render_seed)
    repeated = executor.render(bundle, seed=render_seed)
    if not torch.equal(foa, repeated):
        raise RuntimeError("P10 same-seed repeat is not exactly deterministic")
    patch_codec = ScenePlanEditPatchCodec(codec)
    patch = {
        "operation": "rotate_source",
        "source_id": "source_0",
        "delta_azimuth_deg": 45,
    }
    edited_plan = patch_codec.apply(plan, patch)
    edited_bundle = finalize_sceneplan_for_p10(
        codec,
        codec.encode(edited_plan, max_tokens=1024),
        tokenizer=tokenizer,
        task=P11Task.EDITING,
        sample_id=plan["sample_id"],
        edit_patch_token_ids=patch_codec.encode(patch)["input_ids"],
        edit_patch=patch,
    )
    edited_foa = executor.render(edited_bundle, seed=render_seed)
    edit_mean_absolute_delta = float((edited_foa - foa).abs().mean())
    if not edit_mean_absolute_delta > 0.0:
        raise RuntimeError("P10 same-seed control edit had no waveform effect")
    report = {
        "status": "PASS",
        "architecture": "p11_bundle_to_external_p10_sceneplan_dit",
        "checkpoint_step": P10_CANONICAL_CHECKPOINT_STEP,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "model_config": str(model_config_path),
        "model_config_sha256": P10_CANONICAL_MODEL_CONFIG_SHA256,
        "shape": list(foa.shape),
        "finite": bool(torch.isfinite(foa).all()),
        "mean": float(foa.mean()),
        "std": float(foa.std()),
        "same_seed": render_seed,
        "same_seed_repeat_exact": bool(torch.equal(foa, repeated)),
        "same_seed_edit_finite": bool(torch.isfinite(edited_foa).all()),
        "same_seed_edit_mean_absolute_delta": edit_mean_absolute_delta,
        "input_audio_forwarded": False,
        "p11_audio_flow_used": False,
        "semantic_caption_contract": P10_SEMANTIC_CAPTION_CONTRACT,
        "semantic_caption_compiler_version": (
            P10_SEMANTIC_CAPTION_COMPILER_VERSION
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
