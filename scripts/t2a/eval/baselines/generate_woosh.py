#!/usr/bin/env python3
"""Generate the frozen Sound panel with Woosh-Flow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from woosh.components.base import LoadConfig
from woosh.inference.flowmatching_sampler import flowmatching_integrate
from woosh.model.ldm import LatentDiffusionModel

from baseline_common import (
    add_common_arguments,
    load_requests,
    pending_rows,
    save_result,
    seed_process,
)


DEFAULT_MODEL = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/baselines/p10_60k_15row_v1/models/checkpoints/Woosh-Flow"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "woosh_flow")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.001)
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
        raise RuntimeError("Woosh benchmark requires CUDA")
    # The released config intentionally links its shared components through
    # paths such as ``checkpoints/Woosh-AE``.  Resolve those against the release
    # extraction root, independent of the caller's working directory.
    os.chdir(model_dir.parents[1])
    ldm = LatentDiffusionModel(LoadConfig(path=str(model_dir))).eval().to(device)

    for row in rows:
        seed_process(int(row["seed"]))
        noise = torch.randn(1, 128, 501, device=device)
        cond = ldm.get_cond(
            {"audio": None, "description": [row["semantic_prompt"]]},
            no_dropout=True,
            device=device,
        )
        with torch.inference_mode():
            latent, used_steps = flowmatching_integrate(
                ldm,
                noise=noise,
                cond=cond,
                cfg=float(args.cfg_scale),
                atol=float(args.atol),
                rtol=float(args.rtol),
                return_steps=True,
                device=str(device),
                dtype=torch.float64,
            )
            audio = ldm.autoencoder.inverse(latent)[0].float().cpu()
        save_result(
            row,
            audio,
            48_000,
            backend_metadata={
                "model": "Woosh-Flow",
                "cfg_scale": float(args.cfg_scale),
                "atol": float(args.atol),
                "rtol": float(args.rtol),
                "adaptive_steps": int(used_steps) + 1,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
