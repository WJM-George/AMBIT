#!/usr/bin/env python3
"""Generate the frozen Music/Sound panel with AudioX-Turbo text-only."""

from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import torch

from audiox_turbo.data.utils import load_and_process_audio
from audiox_turbo.inference import load_audiox_turbo_model
from audiox_turbo.inference.generation import generate_diffusion_cond_dmd

from baseline_common import (
    add_common_arguments,
    load_requests,
    pending_rows,
    save_result,
    seed_process,
)


REPO = Path("third_party" + "/AudioX-Turbo")
DEFAULT_MODEL = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/baselines/p10_60k_15row_v1/models/AudioX-Turbo")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, "audiox_turbo")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=4)
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
        raise RuntimeError("AudioX-Turbo benchmark requires CUDA")

    # AudioX-Turbo's released config contains a working-directory-relative
    # checkpoint path for the audio-prompt conditioner.  Loading the config as
    # a dictionary lets the benchmark resolve that released asset explicitly,
    # without editing the frozen upstream checkout or depending on caller CWD.
    config_path = REPO / "configs/audiox_turbo_infer_4step.json"
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)
    audio_prompt_configs = [
        item
        for item in model_config["model"]["conditioning"]["configs"]
        if item.get("id") == "audio_prompt"
    ]
    if len(audio_prompt_configs) != 1:
        raise RuntimeError(
            "Expected exactly one AudioX-Turbo audio_prompt conditioner; "
            f"found {len(audio_prompt_configs)}"
        )
    audio_prompt_configs[0]["config"]["pretransform_ckpt_path"] = str(
        model_dir / "pretransform/vae.ckpt"
    )
    model, config = load_audiox_turbo_model(
        model_config,
        str(model_dir / "audiox_turbo/audiox_turbo.ckpt"),
        pretransform_ckpt_path=str(model_dir / "pretransform/vae.ckpt"),
        device=str(device),
    )
    sample_rate = int(config["sample_rate"])
    sample_size = int(config["sample_size"])
    target_fps = int(config.get("video_fps", 5))
    conditioning_seconds = int(config.get("conditioning_seconds", 10))

    for row in rows:
        duration = float(row["duration_sec"])
        seed_process(int(row["seed"]))
        # The released text-only path is trained with a fixed 10-second empty
        # video/audio conditioning window (50 frames at 5 fps).  Panel clips
        # differ by a few milliseconds, so generate the native fixed window and
        # let ``save_result`` crop/pad it deterministically to the reference.
        video_tensor = torch.zeros(
            conditioning_seconds * target_fps, 3, 224, 224
        )
        sync_features = torch.zeros(1, 240, 768, device=device)
        audio_tensor = load_and_process_audio(
            None, sample_rate, 0, conditioning_seconds
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
            output = generate_diffusion_cond_dmd(
                model,
                steps=int(args.steps),
                conditioning=conditioning,
                sample_size=sample_size,
                seed=int(row["seed"]),
                device=str(device),
            )[0].float().cpu()
        save_result(
            row,
            output,
            sample_rate,
            backend_metadata={
                "model": "AudioX-Turbo",
                "mode": "text-only",
                "steps": int(args.steps),
                "cfg": "distilled-no-cfg",
                "conditioning_seconds": conditioning_seconds,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
