#!/usr/bin/env python3
"""Generate one retained Spatial-CoT family with the dense FOA DiT.

This is a matched acoustic-prior control, not a Spatial-CoT evaluator.  It
uses the same persisted semantic caption, target FOA latent, frozen VAE,
duration, and seed as the Transfusion renderer panels.  The emitted report is
compatible with the source-location and OpenFLAM diagnostics so a dense DiT
checkpoint must demonstrate useful source-composition evidence before it is
accepted as a Transfusion teacher.
"""
from __future__ import annotations
import os

import argparse
import json
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.train.gpu_preflight import assert_gpu_driver_healthy  # noqa: E402

assert_gpu_driver_healthy()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.spatial_caption_templates import (  # noqa: E402
    SEMANTIC_CAPTION_TEMPLATE_VERSION,
)
from stable_audio_tools.data.spatial_conversation_metadata import (  # noqa: E402
    SpatialFamilyMetadata,
    evaluation_metadata_provider as _provider,
)
from stable_audio_tools.data.spatial_family_dataset import (  # noqa: E402
    SpatialFamilyDataset,
)
from stable_audio_tools.inference.sampling import sample_diffusion  # noqa: E402
from stable_audio_tools.models import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import load_ckpt_state_dict  # noqa: E402
from stable_audio_tools.training.factory import (  # noqa: E402
    create_training_wrapper_from_config,
)


DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m.json"
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/"
    "t2a_dit_qwen35_0p8b_300m_wdmix_v2_300k/"
    "checkpoints/epoch=75-step=300000.ckpt"
)
DEFAULT_LATENT_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/latents/train")
DEFAULT_CODEC_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/codec")
DEFAULT_PRETRANSFORM = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)




def _load_family(
    latent_root: Path,
    codec_root: Path,
    caption_overlay_root: Path | None,
    family_rank: int,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
    dataset_entry: dict[str, Any] = {
        "path": str(latent_root),
        "custom_metadata_fn": _provider(codec_root),
    }
    if caption_overlay_root is not None:
        dataset_entry["caption_overlay_path"] = str(caption_overlay_root)
    dataset = SpatialFamilyDataset(
        [dataset_entry],
        require_ready=True,
        max_open_shards=1,
    )
    family_latents, family_info = dataset[family_rank]
    turns = list(family_info.get("family_turn_metadata") or [])
    if not turns or int(family_latents.shape[0]) < 1:
        raise RuntimeError(f"family rank {family_rank} has no retained turn")
    return family_latents[0], family_info, turns[0]


def _save_foa(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    audio = torch.as_tensor(audio, dtype=torch.float32).detach().cpu()
    if audio.ndim == 3 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim != 2 or audio.shape[0] != 4:
        raise ValueError(f"FOA audio must be [4,N], got {tuple(audio.shape)}")
    if not bool(torch.isfinite(audio).all()):
        raise ValueError("refusing to save non-finite FOA audio")
    torchaudio.save(str(path), audio.clamp(-1.0, 1.0), sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument(
        "--pretransform-checkpoint", type=Path, default=DEFAULT_PRETRANSFORM
    )
    parser.add_argument("--latent-root", type=Path, default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--codec-root", type=Path, default=DEFAULT_CODEC_ROOT)
    parser.add_argument("--caption-overlay-root", type=Path)
    parser.add_argument(
        "--no-caption-overlay",
        action="store_true",
        help="Use semantic captions embedded in the family store.",
    )
    parser.add_argument("--family-rank", type=int, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.checkpoint,
        args.model_config,
        args.pretransform_checkpoint,
        args.latent_root / "READY",
        args.codec_root / "READY",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.family_rank < 0 or args.steps < 2 or args.cfg_scale <= 0.0:
        raise ValueError("family-rank>=0, steps>=2, and cfg-scale>0 are required")

    if args.no_caption_overlay and args.caption_overlay_root is not None:
        raise ValueError(
            "--no-caption-overlay and --caption-overlay-root are mutually exclusive"
        )
    caption_overlay_root = None if args.no_caption_overlay else args.caption_overlay_root
    if caption_overlay_root is None and not args.no_caption_overlay:
        caption_overlay_root = (
            args.latent_root.resolve().parents[1]
            / "captions"
            / SEMANTIC_CAPTION_TEMPLATE_VERSION
            / args.latent_root.resolve().name
        )
    target_latent, family_info, turn = _load_family(
        args.latent_root,
        args.codec_root,
        caption_overlay_root,
        args.family_rank,
    )
    semantic_caption = str(turn.get("semantic_caption") or "").strip()
    if not semantic_caption:
        raise RuntimeError("retained turn has no semantic caption")
    scene_plan = turn.get("scene_plan")
    if not isinstance(scene_plan, dict):
        raise RuntimeError("retained turn has no authoritative ScenePlan")

    model_config = load_config(args.model_config)
    if model_config.get("model_type") != "diffusion_cond":
        raise ValueError("matched control requires a conditional diffusion DiT config")
    model = create_model_from_config(model_config)
    wrapper = create_training_wrapper_from_config(model_config, model)
    state = load_ckpt_state_dict(str(args.checkpoint))
    incompatible = wrapper.load_state_dict(state, strict=False)
    del state
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "DiT checkpoint/config mismatch: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )

    pretransform_state = load_ckpt_state_dict(str(args.pretransform_checkpoint))
    if hasattr(model, "load_pretransform_state_dict"):
        model.load_pretransform_state_dict(pretransform_state, strict=False)
    else:
        model.pretransform.load_state_dict(pretransform_state, strict=False)
    del pretransform_state

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
    diffusion = wrapper.diffusion
    if wrapper.diffusion_ema is None:
        raise RuntimeError("the retained dense DiT checkpoint has no EMA model")
    sample_model = wrapper.diffusion_ema.ema_model
    sample_model.eval().requires_grad_(False).to(device)
    diffusion.conditioner.eval().requires_grad_(False).to(device)
    diffusion.pretransform.eval().requires_grad_(False).to(device)
    model_dtype = next(sample_model.parameters()).dtype

    seconds_total = float(turn.get("seconds_total") or scene_plan["audio"]["duration_sec"])
    conditioning = [
        {
            "prompt": semantic_caption,
            "spatial_format": "foa",
            "seconds_start": float(turn.get("seconds_start") or 0.0),
            "seconds_total": seconds_total,
        }
    ]
    _seed_everything(args.seed)
    noise = torch.randn(
        1,
        diffusion.io_channels,
        int(target_latent.shape[-1]),
        device=device,
        dtype=model_dtype,
    )
    condition_tensors = diffusion.conditioner(conditioning, device)
    condition_inputs = diffusion.get_conditioning_inputs(condition_tensors)
    condition_inputs = {
        key: value.to(model_dtype) if isinstance(value, torch.Tensor) else value
        for key, value in condition_inputs.items()
    }
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        generated_audio = sample_diffusion(
            model=sample_model,
            noise=noise,
            cond_inputs=condition_inputs,
            diffusion_objective=diffusion.diffusion_objective,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            conditioning=conditioning,
            sample_rate=int(model_config["sample_rate"]),
            pretransform=diffusion.pretransform,
            mask_padding_attention=diffusion.mask_padding_attention,
            use_effective_length_for_schedule=(
                diffusion.use_effective_length_for_schedule
            ),
            dist_shift=diffusion.sampling_dist_shift,
            sampler_type="euler",
            batch_cfg=True,
            rescale_cfg=True,
            apg_scale=1.0,
            decode=True,
            disable_tqdm=False,
        )
        target_audio = diffusion.pretransform.decode(
            torch.as_tensor(target_latent, device=device, dtype=model_dtype).unsqueeze(0)
        )

    sample_rate = int(model_config["sample_rate"])
    sample_count = min(
        int(generated_audio.shape[-1]),
        int(target_audio.shape[-1]),
        int(model_config["sample_size"]),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    generated_path = (args.output_dir / "turn_00.wav").resolve()
    target_path = (args.output_dir / "turn_00.target.wav").resolve()
    plan_path = (args.output_dir / "turn_00.target_scene_plan.json").resolve()
    _save_foa(generated_path, generated_audio[..., :sample_count], sample_rate)
    _save_foa(target_path, target_audio[..., :sample_count], sample_rate)
    _atomic_json(plan_path, scene_plan)

    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluator_version": 3,
        "family_id": str(family_info.get("family_id") or ""),
        "family_rank": int(args.family_rank),
        "status": "PASS",
        "settings": {
            "control": "dense_dit_acoustic_prior",
            "cfg_scale": float(args.cfg_scale),
            "modality_steps": int(args.steps),
            "seed": int(args.seed),
            "sample_rate": sample_rate,
            "vae_checkpoint": str(args.pretransform_checkpoint.resolve()),
        },
        "turn_results": [
            {
                "audio_path": str(generated_path),
                "target_audio_path": str(target_path),
                "target_plan_path": str(plan_path),
                "semantic_caption": semantic_caption,
                "spatial_alignment": {
                    "hop": int(diffusion.pretransform.downsampling_ratio),
                    "frame_count": int(target_latent.shape[-1]),
                },
                "turn": 0,
            }
        ],
    }
    _atomic_json(args.output_dir / "RESULT.json", report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
