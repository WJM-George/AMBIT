#!/usr/bin/env python3
"""Generate the frozen Music/Sound panel with Stable Audio Open 1.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

from baseline_common import (
    add_common_arguments,
    load_requests,
    pending_rows,
    save_result,
    seed_process,
)


DEFAULT_CONFIG = Path(
    "/mnt/sdc/ckpts/baselines/p10_60k_15row_v1/models/"
    "stable-audio-open-1.0-config/model_config.json"
)
DEFAULT_WEIGHTS = Path(
    "/mnt/sdc/ckpts/pretrained/stable-audio-open-1.0/model.safetensors"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "stable_audio_open_1_0")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    args = parser.parse_args()

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
    config_path = args.config.expanduser().resolve(strict=True)
    weights_path = args.weights.expanduser().resolve(strict=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stable Audio Open benchmark requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = create_model_from_config(config)
    model.load_state_dict(load_ckpt_state_dict(str(weights_path)), strict=True)
    model = model.to(device).eval().requires_grad_(False)
    sample_rate = int(config["sample_rate"])
    sample_size = int(config["sample_size"])

    for row in rows:
        seed_process(int(row["seed"]))
        conditioning = [
            {
                "prompt": row["semantic_prompt"],
                "seconds_start": 0,
                "seconds_total": float(row["duration_sec"]),
            }
        ]
        with torch.inference_mode():
            output = generate_diffusion_cond(
                model,
                steps=int(args.steps),
                cfg_scale=float(args.cfg_scale),
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=0.3,
                sigma_max=500.0,
                sampler_type="dpmpp-3m-sde",
                device=str(device),
                seed=int(row["seed"]),
            )[0].float().cpu()
        save_result(
            row,
            output,
            sample_rate,
            backend_metadata={
                "model": "Stable Audio Open 1.0",
                "steps": int(args.steps),
                "cfg_scale": float(args.cfg_scale),
                "sampler": "dpmpp-3m-sde",
                "sigma_min": 0.3,
                "sigma_max": 500.0,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
