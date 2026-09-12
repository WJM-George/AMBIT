#!/usr/bin/env python3
"""Generate a frozen text-only panel with released AudioX family checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audiox.inference.generation import generate_diffusion_cond
from audiox.models.factory import create_model_from_config
from audiox.models.utils import load_ckpt_state_dict

from baseline_common import (
    load_requests,
    pending_rows,
    save_result,
    seed_process,
)


ASSET_ROOT = Path("/mnt/sdc/ckpts/baselines/p10_60k_15row_v1")
SHARED_VAE = ASSET_ROOT / "models/AudioX-Turbo/pretransform/vae.ckpt"
VARIANTS = {
    "audiox_maf": {
        "display_name": "AudioX-MAF",
        "model_dir": ASSET_ROOT / "models/AudioX-MAF",
        "revision": "0a6575a6fd58039281584ad1c6f9e895233e8ca7",
    },
    "audiox_maf_mmdit": {
        "display_name": "AudioX-MAF-MMDiT",
        "model_dir": ASSET_ROOT / "models/AudioX-MAF-MMDiT",
        "revision": "2db085273efb141facde8d15a22a4f7cd9734df4",
    },
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-id", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--sigma-min", type=float, default=0.3)
    parser.add_argument("--sigma-max", type=float, default=500.0)
    parser.add_argument("--sampler", default="dpmpp-3m-sde")
    return parser.parse_args()


def _patch_local_assets(config: dict, baseline_id: str) -> None:
    """Resolve released working-directory-relative conditioner assets locally."""

    if baseline_id != "audiox_maf_mmdit":
        return
    conditioners = config["model"]["conditioning"]["configs"]
    audio_prompt = [item for item in conditioners if item.get("id") == "audio_prompt"]
    if len(audio_prompt) != 1:
        raise RuntimeError(
            "AudioX-MAF-MMDiT must contain exactly one audio_prompt conditioner"
        )
    SHARED_VAE.resolve(strict=True)
    audio_prompt[0]["config"]["pretransform_ckpt_path"] = str(SHARED_VAE)


def main() -> int:
    args = _arguments()
    variant = VARIANTS[args.baseline_id]
    rows = pending_rows(
        load_requests(
            args.manifest,
            args.baseline_id,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        ),
        force=args.force,
    )
    if not rows:
        print('{"status":"SKIP","reason":"all outputs valid"}')
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("AudioX family benchmark requires CUDA")
    model_dir = Path(variant["model_dir"]).resolve(strict=True)
    config_path = model_dir / "config.json"
    checkpoint_path = model_dir / "model.ckpt"
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    _patch_local_assets(model_config, args.baseline_id)

    model = create_model_from_config(model_config)
    state_dict = load_ckpt_state_dict(str(checkpoint_path.resolve(strict=True)))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"strict AudioX load failed: {incompatible}")
    del state_dict
    model.to(device).eval().requires_grad_(False)

    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])
    target_fps = int(model_config.get("video_fps", 5))
    conditioning_seconds = 10
    for row in rows:
        seed_process(int(row["seed"]))
        video_tensor = torch.zeros(
            conditioning_seconds * target_fps, 3, 224, 224
        )
        sync_features = torch.zeros(1, 240, 768, device=device)
        audio_tensor = torch.zeros(
            2, sample_rate * conditioning_seconds, device=device
        )
        conditioning = [
            {
                "video_prompt": {
                    "video_tensors": video_tensor.unsqueeze(0),
                    "video_sync_frames": sync_features,
                },
                "text_prompt": row["semantic_prompt"],
                "audio_prompt": audio_tensor.unsqueeze(0),
                "seconds_start": 0,
                "seconds_total": conditioning_seconds,
            }
        ]
        with torch.inference_mode():
            output = generate_diffusion_cond(
                model,
                conditioning=conditioning,
                steps=int(args.steps),
                cfg_scale=float(args.cfg_scale),
                batch_size=1,
                sample_size=sample_size,
                sample_rate=sample_rate,
                seed=int(row["seed"]),
                device=str(device),
                sampler_type=args.sampler,
                sigma_min=float(args.sigma_min),
                sigma_max=float(args.sigma_max),
                scale_phi=0.0,
            )[0].float().cpu()
        save_result(
            row,
            output,
            sample_rate,
            backend_metadata={
                "model": variant["display_name"],
                "mode": "text-only",
                "revision": variant["revision"],
                "steps": int(args.steps),
                "cfg_scale": float(args.cfg_scale),
                "sampler": args.sampler,
                "sigma_min": float(args.sigma_min),
                "sigma_max": float(args.sigma_max),
                "conditioning_seconds": conditioning_seconds,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
