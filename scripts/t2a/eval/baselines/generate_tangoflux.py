#!/usr/bin/env python3
"""Generate the frozen Music/Sound panel with local TangoFlux weights."""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderOobleck
from safetensors.torch import load_file
from tangoflux.model import TangoFlux

from baseline_common import (
    add_common_arguments,
    load_requests,
    pending_rows,
    save_result,
    seed_process,
)


DEFAULT_MODEL = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/baselines/p10_60k_15row_v1/models/TangoFlux")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "tangoflux")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
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
    model_dir = args.model_dir.expanduser().resolve(strict=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("TangoFlux benchmark requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    vae = AutoencoderOobleck()
    vae.load_state_dict(load_file(str(model_dir / "vae.safetensors")))
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    model = TangoFlux(config)
    model.load_state_dict(load_file(str(model_dir / "tangoflux.safetensors")), strict=False)
    vae = vae.to(device).eval().requires_grad_(False)
    model = model.to(device).eval().requires_grad_(False)

    for row in rows:
        seed_process(int(row["seed"]))
        with torch.inference_mode():
            latents = model.inference_flow(
                row["semantic_prompt"],
                duration=float(row["duration_sec"]),
                num_inference_steps=int(args.steps),
                guidance_scale=float(args.guidance_scale),
            )
            wave = vae.decode(latents.transpose(2, 1)).sample[0].float().cpu()
        save_result(
            row,
            wave,
            int(vae.config.sampling_rate),
            backend_metadata={
                "model": "declare-lab/TangoFlux",
                "steps": int(args.steps),
                "guidance_scale": float(args.guidance_scale),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
