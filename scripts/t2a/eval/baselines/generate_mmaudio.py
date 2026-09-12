#!/usr/bin/env python3
"""Generate the frozen Sound panel with MMAudio-L v2 in text-only mode."""

from __future__ import annotations

import argparse
from pathlib import Path

import open_clip
import torch
from torchvision.transforms import Normalize

from mmaudio.eval_utils import generate
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils, patch_clip

from baseline_common import (
    add_common_arguments,
    load_requests,
    pending_rows,
    save_result,
    seed_process,
)


DEFAULT_MODEL = Path("/mnt/sdc/ckpts/baselines/p10_60k_15row_v1/models/MMAudio")


def _text_only_features(vae_path: Path, device: torch.device) -> FeaturesUtils:
    """Load the text encoder without the unused 907 MB Synchformer."""

    features = FeaturesUtils(
        tod_vae_ckpt=str(vae_path),
        synchformer_ckpt=None,
        enable_conditions=False,
        mode="44k",
        bigvgan_vocoder_ckpt=None,
        need_vae_encoder=False,
    )
    clip_model = open_clip.create_model_from_pretrained(
        "hf-hub:apple/DFN5B-CLIP-ViT-H-14-384", return_transform=False
    )
    features.clip_model = patch_clip(clip_model)
    features.clip_preprocess = Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    features.tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
    return features.to(device, torch.bfloat16).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "mmaudio_large_44k_v2_text_only")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg-scale", type=float, default=4.5)
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
        raise RuntimeError("MMAudio benchmark requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = torch.bfloat16

    net = get_my_mmaudio("large_44k_v2").to(device, dtype).eval()
    net.load_weights(
        torch.load(
            model_dir / "weights/mmaudio_large_44k_v2.pth",
            map_location=device,
            weights_only=True,
        )
    )
    features = _text_only_features(model_dir / "ext_weights/v1-44.pth", device)

    for row in rows:
        duration = float(row["duration_sec"])
        seq_cfg = net.seq_cfg if hasattr(net, "seq_cfg") else None
        if seq_cfg is None:
            from mmaudio.model.sequence_config import CONFIG_44K

            seq_cfg = CONFIG_44K
        seq_cfg.duration = duration
        net.update_seq_lengths(
            seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len
        )
        seed_process(int(row["seed"]))
        rng = torch.Generator(device=device).manual_seed(int(row["seed"]))
        fm = FlowMatching(
            min_sigma=0, inference_mode="euler", num_steps=int(args.steps)
        )
        with torch.inference_mode():
            audio = generate(
                None,
                None,
                [row["semantic_prompt"]],
                negative_text=[""],
                feature_utils=features,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=float(args.cfg_scale),
            ).float().cpu()[0]
        save_result(
            row,
            audio,
            int(seq_cfg.sampling_rate),
            backend_metadata={
                "model": "MMAudio-L 44.1kHz v2",
                "mode": "text-only",
                "steps": int(args.steps),
                "cfg_scale": float(args.cfg_scale),
                "synchformer_loaded": False,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
