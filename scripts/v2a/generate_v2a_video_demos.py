#!/usr/bin/env python
"""Generate V2A video demos from Sphere360 dynamic10 test latents.

Input: pre-encoded test metadata with VideoMAE feature paths.
Output: generated WAV plus MP4 muxed with the corresponding video segment.
"""

import argparse
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import torch
import torchaudio

from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.training.factory import create_training_wrapper_from_config
from stable_audio_tools.inference.sampling import sample_diffusion


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return text.strip("_")[:96] or "sample"


def load_custom_metadata_fn(path: str):
    spec = importlib.util.spec_from_file_location("sphere360_video_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_custom_metadata


def iter_test_items(test_latents_dir: Path, limit: int):
    json_files = sorted(test_latents_dir.glob("*/*.json"))
    for md_path in json_files[:limit]:
        npy_path = md_path.with_suffix(".npy")
        if npy_path.exists():
            yield npy_path, md_path


def audio_to_wav_tensor(audio: torch.Tensor) -> torch.Tensor:
    audio = audio.detach().float().cpu()
    if audio.ndim == 3:
        audio = audio[0]
    peak = audio.abs().max().item()
    if peak > 0:
        audio = audio / peak * 0.95
    return audio.clamp(-1, 1)


def mux_video_audio(video_path: str, video_start: float, duration: float, wav_path: Path, mp4_path: Path):
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{video_start:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        video_path,
        "-i",
        str(wav_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "512k",
        "-shortest",
        str(mp4_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test-latents-dir", default="/mnt/sdd/audio_dataset/Sphere360_processed/latents_dynamic10/test")
    p.add_argument("--custom-metadata-module", default="/mnt/sdd/audio_dataset/Sphere360_processed/metadata/sphere360_video_metadata.py")
    p.add_argument("--out-dir", default="/mnt/sdc/video_demos")
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=20260703)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sampler-type", default="euler")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "wav"
    mp4_dir = out_dir / "mp4"
    wav_dir.mkdir(parents=True, exist_ok=True)
    mp4_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] checkpoint: {args.ckpt}", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=False)
    model_config = ckpt["model_config"]
    state_dict = ckpt["state_dict"]

    print("[build] model + training wrapper", flush=True)
    model = create_model_from_config(model_config)
    wrapper = create_training_wrapper_from_config(model_config, model)
    incompatible = wrapper.load_state_dict(state_dict, strict=False)
    print(f"[load] missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}", flush=True)
    if incompatible.missing_keys:
        print(f"[load] missing sample: {incompatible.missing_keys[:8]}", flush=True)
    if incompatible.unexpected_keys:
        print(f"[load] unexpected sample: {incompatible.unexpected_keys[:8]}", flush=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    diffusion = wrapper.diffusion
    diffusion.conditioner.eval().requires_grad_(False).to(device)
    diffusion.pretransform.eval().requires_grad_(False).to(device)

    if getattr(wrapper, "diffusion_ema", None) is not None and wrapper.diffusion_ema is not None:
        sample_model = wrapper.diffusion_ema.ema_model
        print("[sample] using EMA diffusion model", flush=True)
    else:
        sample_model = diffusion.model
        print("[sample] using online diffusion model", flush=True)

    sample_model.eval().requires_grad_(False).to(device)
    model_dtype = next(sample_model.parameters()).dtype

    default_latent_len = model_config["sample_size"] // diffusion.pretransform.downsampling_ratio
    sample_rate = int(model_config["sample_rate"])
    custom_metadata_fn = load_custom_metadata_fn(args.custom_metadata_module)
    results_path = out_dir / "generation_results.jsonl"

    items = list(iter_test_items(Path(args.test_latents_dir), args.num_samples))
    print(
        f"[sample] items={len(items)} default_latent_len={default_latent_len} steps={args.steps} cfg={args.cfg_scale} out={out_dir}",
        flush=True,
    )

    with results_path.open("w", encoding="utf-8") as results_f:
        for i, (npy_path, md_path) in enumerate(items):
            with md_path.open("r", encoding="utf-8") as f:
                info = json.load(f)

            latents = torch.from_numpy(np.load(npy_path))
            latent_len = int(latents.shape[-1])
            info["latent_filename"] = str(npy_path)
            info.update(custom_metadata_fn(info, latents))

            cond = {
                "video_motion_aligned": info["video_motion_aligned"],
                "spatial_format": info.get("spatial_format", "foa"),
                "seconds_start": info.get("seconds_start", 0),
                "seconds_total": info.get("seconds_total", info.get("duration", 10.0)),
            }
            seconds = float(cond["seconds_total"])
            seed = args.seed + i
            torch.manual_seed(seed)
            noise = torch.randn(
                [1, diffusion.io_channels, latent_len],
                device=device,
                dtype=model_dtype,
            )

            sample_id = info.get("id", md_path.stem)
            clip_id = info.get("clip_id", sample_id)
            print(f"[{i + 1:02d}/{len(items)}] {sample_id} latent_len={latent_len} seed={seed}", flush=True)

            conditioning = diffusion.conditioner([cond], device)
            cond_inputs = diffusion.get_conditioning_inputs(conditioning)
            cond_inputs = {
                k: v.to(model_dtype) if isinstance(v, torch.Tensor) else v
                for k, v in cond_inputs.items()
            }

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                audio = sample_diffusion(
                    model=sample_model,
                    noise=noise,
                    cond_inputs=cond_inputs,
                    diffusion_objective=diffusion.diffusion_objective,
                    steps=args.steps,
                    cfg_scale=args.cfg_scale,
                    conditioning=[cond],
                    sample_rate=sample_rate,
                    pretransform=diffusion.pretransform,
                    mask_padding_attention=diffusion.mask_padding_attention,
                    use_effective_length_for_schedule=diffusion.use_effective_length_for_schedule,
                    headroom_seconds=5.0,
                    dist_shift=diffusion.sampling_dist_shift,
                    sampler_type=args.sampler_type,
                    batch_cfg=True,
                    decode=True,
                    disable_tqdm=False,
                )

            trim = min(audio.shape[-1], int(seconds * sample_rate))
            wav = audio_to_wav_tensor(audio[..., :trim])
            base_name = f"{i + 1:02d}_{slugify(clip_id)}_{slugify(sample_id)}"
            wav_path = wav_dir / f"{base_name}.wav"
            torchaudio.save(str(wav_path), wav, sample_rate)

            video_path = info.get("source_video_path") or info.get("video_path")
            video_start = float(info.get("video_start", 0.0))
            mp4_path = mp4_dir / f"{base_name}.mp4"
            mux_video_audio(video_path, video_start, seconds, wav_path, mp4_path)

            result = {
                "id": sample_id,
                "clip_id": clip_id,
                "seed": seed,
                "steps": args.steps,
                "cfg_scale": args.cfg_scale,
                "checkpoint": args.ckpt,
                "latent_json": str(md_path),
                "video_path": video_path,
                "video_start": video_start,
                "duration": seconds,
                "generated_wav": str(wav_path),
                "muxed_mp4": str(mp4_path),
                "sample_rate": sample_rate,
                "channels": int(wav.shape[0]),
                "num_samples": int(wav.shape[-1]),
            }
            results_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            results_f.flush()

    print(f"[done] wrote {results_path}", flush=True)


if __name__ == "__main__":
    main()
