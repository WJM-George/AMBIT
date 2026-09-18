#!/usr/bin/env python3
"""Generate a deterministic sound/music/speech listening panel with dense DiT."""
from __future__ import annotations
import os

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchaudio

from scripts.t2a.train.gpu_preflight import assert_gpu_driver_healthy
from stable_audio_tools.configuration import load_config
from stable_audio_tools.inference.sampling import sample_diffusion
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.factory import create_training_wrapper_from_config


DEFAULT_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m.json"
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/"
    "t2a_dit_qwen35_0p8b_300m_wdmix_v2_300k/"
    "checkpoints/epoch=75-step=300000.ckpt"
)
DEFAULT_VAE = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)
PROMPTS = (
    ("sound", "01_campfire", "From 0.0 to 7.0 seconds, a close dry campfire crackles with distinct wooden pops. From 7.0 seconds to the end, complete silence. No speech and no music."),
    ("sound", "02_rain_thunder", "From 0.0 to 7.0 seconds, heavy rain strikes a metal roof with distant thunder. From 7.0 seconds to the end, complete silence. No speech and no music."),
    ("sound", "03_dog_barks", "A quiet yard with one clear dog bark near 2 seconds, one near 4 seconds, and one near 6 seconds. After 7.0 seconds, complete silence. No speech and no music."),
    ("music", "01_piano", "From 0.0 to 7.0 seconds, a clean solo grand piano plays a gentle arpeggiated melody, then stops naturally. From 7.0 seconds to the end, complete silence. No vocals."),
    ("music", "02_funk", "From 0.0 to 7.0 seconds, an energetic instrumental funk groove with drums, bass, and clean guitar, then a clean stop. From 7.0 seconds to the end, complete silence. No vocals."),
    ("music", "03_orchestra", "From 0.0 to 7.0 seconds, a cinematic string orchestra builds to a short cadence and stops. From 7.0 seconds to the end, complete silence. No vocals."),
    ("speech", "01_english_female", "At a normal speaking speed, one adult female speaker clearly says once: 'The morning train arrives at seven, so please do not be late.' She finishes before 7.0 seconds. The remainder is complete silence. No music or background sounds."),
    ("speech", "02_mandarin_male", "At a normal speaking speed, one adult male speaker clearly says once in Mandarin: '今天的天气很好，我们下午一起去公园散步。' He finishes before 7.0 seconds. The remainder is complete silence. No music or background sounds."),
    ("speech", "03_dialogue", "At normal speaking speed, two adult English speakers have one short exchange. Speaker one asks, 'Did you lock the front door?' Speaker two replies, 'Yes, I checked it twice.' They finish before 7.0 seconds. The remainder is complete silence. No music or background sounds."),
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=4200)
    args = parser.parse_args()
    assert_gpu_driver_healthy()
    for path in (args.model_config, args.checkpoint, args.vae_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_config(args.model_config)
    model = create_model_from_config(config)
    wrapper = create_training_wrapper_from_config(config, model)
    state = load_ckpt_state_dict(str(args.checkpoint))
    incompatible = wrapper.load_state_dict(state, strict=False)
    del state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys[:8]} "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    vae_state = load_ckpt_state_dict(str(args.vae_checkpoint))
    if hasattr(model, "load_pretransform_state_dict"):
        model.load_pretransform_state_dict(vae_state, strict=False)
    else:
        model.pretransform.load_state_dict(vae_state, strict=False)
    del vae_state

    device = torch.device(args.device)
    diffusion = wrapper.diffusion
    sample_model = wrapper.diffusion_ema.ema_model
    sample_model.eval().requires_grad_(False).to(device)
    diffusion.conditioner.eval().requires_grad_(False).to(device)
    diffusion.pretransform.eval().requires_grad_(False).to(device)
    dtype = next(sample_model.parameters()).dtype
    frames = int(config["sample_size"]) // int(diffusion.pretransform.downsampling_ratio)
    seconds = float(config["sample_size"]) / float(config["sample_rate"])
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for index, (domain, name, prompt) in enumerate(PROMPTS):
        seed = args.seed + index
        seed_all(seed)
        conditioning = [{
            "prompt": prompt,
            "spatial_format": "foa",
            "seconds_start": 0.0,
            "seconds_total": seconds,
        }]
        condition_tensors = diffusion.conditioner(conditioning, device)
        condition_inputs = diffusion.get_conditioning_inputs(condition_tensors)
        condition_inputs = {
            key: value.to(dtype) if isinstance(value, torch.Tensor) else value
            for key, value in condition_inputs.items()
        }
        noise = torch.randn(1, diffusion.io_channels, frames, device=device, dtype=dtype)
        with torch.inference_mode(), autocast:
            audio = sample_diffusion(
                model=sample_model,
                noise=noise,
                cond_inputs=condition_inputs,
                diffusion_objective=diffusion.diffusion_objective,
                steps=args.steps,
                cfg_scale=args.cfg_scale,
                conditioning=conditioning,
                sample_rate=int(config["sample_rate"]),
                pretransform=diffusion.pretransform,
                mask_padding_attention=diffusion.mask_padding_attention,
                use_effective_length_for_schedule=diffusion.use_effective_length_for_schedule,
                dist_shift=diffusion.sampling_dist_shift,
                sampler_type="euler",
                batch_cfg=True,
                rescale_cfg=True,
                apg_scale=1.0,
                decode=True,
                disable_tqdm=False,
            )[0].float().cpu()
        if not bool(torch.isfinite(audio).all()):
            raise RuntimeError(f"non-finite audio for {domain}/{name}")
        output = args.output_dir / domain / f"{name}.wav"
        output.parent.mkdir(parents=True, exist_ok=True)
        clipped_fraction = float(audio.abs().gt(1.0).float().mean())
        boundary = min(audio.shape[-1], int(7.0 * int(config["sample_rate"])))
        active_rms = audio[..., :boundary].square().mean().sqrt()
        tail_rms = audio[..., boundary:].square().mean().sqrt()
        tail_dbfs = 20.0 * torch.log10(tail_rms.clamp_min(1e-12))
        tail_to_active = tail_rms / active_rms.clamp_min(1e-12)
        torchaudio.save(str(output), audio.clamp(-1.0, 1.0), int(config["sample_rate"]))
        results.append({
            "domain": domain,
            "name": name,
            "prompt": prompt,
            "seed": seed,
            "path": str(output.resolve()),
            "peak_before_clamp": float(audio.abs().max()),
            "rms": float(audio.square().mean().sqrt()),
            "clipped_fraction_before_clamp": clipped_fraction,
            "active_0_7s_rms": float(active_rms),
            "tail_7s_end_rms": float(tail_rms),
            "tail_7s_end_dbfs": float(tail_dbfs),
            "tail_to_active_rms_ratio": float(tail_to_active),
        })
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    report = {
        "schema": "stable_audio_tools.dit_domain_listening_panel",
        "status": "PASS",
        "checkpoint": str(args.checkpoint.resolve()),
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "settings": {"steps": args.steps, "cfg_scale": args.cfg_scale, "seconds": seconds},
        "samples": results,
    }
    (args.output_dir / "MANIFEST.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
