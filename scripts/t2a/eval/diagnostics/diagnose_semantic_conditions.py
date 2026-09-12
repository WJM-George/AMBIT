#!/usr/bin/env python3
"""Same-noise causal diagnostic for Spatial-CoT semantic conditioning.

The creation-turn renderer receives semantic information twice: as a Qwen
caption prefix and as event/content fields inside ScenePlan.  This diagnostic
crosses correct versus donor values for those two paths while holding sampler
noise, source count, source identity, activity, gain, motion, room, and the
compiled 4-D spatial anchor fixed.  It therefore measures whether each
semantic path can causally change the rendered FOA without conflating that
effect with geometry steering.
"""
from __future__ import annotations
import os

import argparse
import copy
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from stable_audio_tools.configuration import load_config
from stable_audio_tools.data.spatial_caption_templates import (
    SEMANTIC_CAPTION_TEMPLATE_VERSION,
)
from stable_audio_tools.data.spatial_family_dataset import SpatialFamilyDataset
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

from scripts.t2a.eval.diagnostics.diagnose_spatial_conditions import (
    _atomic_json,
    _decode_latent,
    _render,
    _reset_seed,
    _save_audio,
    _structured_semantic_caption,
)
from scripts.t2a.eval.evaluate_spatial_cot_checkpoint import (
    _provider,
    _spatial_alignment_metrics,
)


def _sources(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = ((plan.get("scene") or {}).get("sources") or [])
    if not isinstance(values, list) or not all(isinstance(x, dict) for x in values):
        raise ValueError("ScenePlan sources must be a list of dictionaries")
    return values


def _swap_source_semantics(
    base_plan: Mapping[str, Any], donor_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Copy only codec-visible event/content fields from a donor ScenePlan."""

    swapped = copy.deepcopy(dict(base_plan))
    base_sources = _sources(swapped)
    donor_sources = _sources(donor_plan)
    if len(base_sources) != len(donor_sources):
        raise ValueError(
            "semantic donor must have the same source count: "
            f"base={len(base_sources)}, donor={len(donor_sources)}"
        )
    if not base_sources:
        raise ValueError("semantic diagnostic requires at least one source")
    for base_source, donor_source in zip(base_sources, donor_sources):
        base_source["event"] = copy.deepcopy(donor_source.get("event") or {})
        base_source["content"] = copy.deepcopy(donor_source.get("content") or {})
    return swapped


def _source_semantics(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for source in _sources(plan):
        event = source.get("event") or {}
        content = source.get("content") or {}
        result.append(
            {
                "source_id": source.get("source_id"),
                "label": event.get("label"),
                "category": event.get("category"),
                "transcript": content.get("transcript"),
                "speaker_id": content.get("speaker_id"),
            }
        )
    return result


def _tensor_pair_metrics(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().cpu()
    reference = reference.detach().float().cpu()
    if value.shape != reference.shape:
        raise ValueError(
            f"pair metric shapes differ: {tuple(value.shape)} != {tuple(reference.shape)}"
        )
    delta = value - reference
    reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-12)
    value_flat = value.flatten()
    reference_flat = reference.flatten()
    cosine = torch.nn.functional.cosine_similarity(
        value_flat.unsqueeze(0), reference_flat.unsqueeze(0)
    )[0]
    return {
        "mae": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_rmse": float(delta.square().mean().sqrt() / reference_rms),
        "cosine": float(cosine),
    }


def _channel_correlations(value: torch.Tensor, reference: torch.Tensor) -> list[float]:
    value = value.detach().float().cpu()
    reference = reference.detach().float().cpu()
    if value.shape != reference.shape or value.ndim != 2:
        raise ValueError("channel correlations require matching [C,T] tensors")
    correlations = []
    for channel in range(value.shape[0]):
        x = value[channel] - value[channel].mean()
        y = reference[channel] - reference[channel].mean()
        denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
        correlations.append(
            0.0 if float(denominator) <= 1.0e-12 else float((x * y).sum() / denominator)
        )
    return correlations


def _log_spectral_mae(value: torch.Tensor, reference: torch.Tensor) -> float:
    value = value.detach().float().cpu()
    reference = reference.detach().float().cpu()
    if value.shape != reference.shape or value.ndim != 2:
        raise ValueError("spectral metric requires matching [C,T] tensors")
    losses = []
    for n_fft in (512, 1024, 2048):
        window = torch.hann_window(n_fft)
        value_stft = torch.stft(
            value,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
        )
        reference_stft = torch.stft(
            reference,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
        )
        losses.append(
            (torch.log1p(value_stft.abs()) - torch.log1p(reference_stft.abs()))
            .abs()
            .mean()
        )
    return float(torch.stack(losses).mean())


def _output_pair_metrics(
    latent: torch.Tensor,
    audio: torch.Tensor,
    *,
    baseline_latent: torch.Tensor,
    baseline_audio: torch.Tensor,
    hop: int,
) -> dict[str, Any]:
    audio_metrics = _tensor_pair_metrics(audio, baseline_audio)
    audio_metrics["channel_correlations_wyzx"] = _channel_correlations(
        audio, baseline_audio
    )
    audio_metrics["multiresolution_log_spectral_mae"] = _log_spectral_mae(
        audio, baseline_audio
    )
    return {
        "latent": _tensor_pair_metrics(latent, baseline_latent),
        "audio": audio_metrics,
        "field": _spatial_alignment_metrics(audio, baseline_audio, hop=hop),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("online", "ema"), default="online")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--attention-backend", choices=("configured", "dense"), default="configured"
    )
    parser.add_argument("--pretransform-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/latents/validation"),
    )
    parser.add_argument("--caption-overlay-root", type=Path)
    parser.add_argument(
        "--codec-root",
        type=Path,
        default=Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/codec"),
    )
    parser.add_argument("--family-rank", type=int, required=True)
    parser.add_argument("--donor-family-rank", type=int, required=True)
    parser.add_argument("--modality-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    caption_overlay_root = args.caption_overlay_root
    if caption_overlay_root is None:
        caption_overlay_root = (
            args.latent_root.resolve().parents[1]
            / "captions"
            / SEMANTIC_CAPTION_TEMPLATE_VERSION
            / args.latent_root.resolve().name
        )
    for path in (
        args.checkpoint,
        args.model_config,
        args.pretransform_checkpoint,
        args.latent_root / "READY",
        args.codec_root / "READY",
        caption_overlay_root / "READY",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.family_rank < 0
        or args.donor_family_rank < 0
        or args.family_rank == args.donor_family_rank
        or args.modality_steps < 2
        or not math.isfinite(args.cfg_scale)
        or args.cfg_scale <= 0.0
    ):
        raise ValueError("invalid family/donor/modality-step/CFG arguments")

    _reset_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    dataset = SpatialFamilyDataset(
        [
            {
                "path": str(args.latent_root),
                "caption_overlay_path": str(caption_overlay_root),
                "custom_metadata_fn": _provider(args.codec_root),
            }
        ],
        require_ready=True,
        max_open_shards=2,
    )
    family_latents, family = dataset[args.family_rank]
    _, donor = dataset[args.donor_family_rank]
    creation = (family.get("family_turn_metadata") or [])[0]
    donor_creation = (donor.get("family_turn_metadata") or [])[0]

    model_config = load_config(args.model_config)
    if args.attention_backend == "dense":
        model_config["model"]["transfusion"]["transformer"]["use_flex_attn"] = False
    model = create_model_from_config(model_config)
    checkpoint_state, checkpoint_metadata = load_ckpt_state_dict(
        str(args.checkpoint), return_metadata=True
    )
    warmstart = model.load_pretrained_route_state_dict(
        checkpoint_state,
        prefer_ema=args.weights == "ema",
        source_model_config=checkpoint_metadata.get("model_config"),
        source_text_conditioner_ema_names=checkpoint_metadata.get(
            "text_conditioner_ema_parameter_names"
        ),
    )
    del checkpoint_state, checkpoint_metadata
    pretransform_state = load_ckpt_state_dict(str(args.pretransform_checkpoint))
    model.load_pretransform_state_dict(pretransform_state)
    del pretransform_state

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    model = model.eval().requires_grad_(False).to(device)
    codec = model.get_plan_codec()
    hop = int(model.downsampling_ratio)
    sample_rate = int(model_config["sample_rate"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode(), autocast:
        correct_tokens = torch.as_tensor(
            creation["spatial_plan_tokens"]["input_ids"],
            device=device,
            dtype=torch.long,
        )
        donor_tokens = torch.as_tensor(
            donor_creation["spatial_plan_tokens"]["input_ids"],
            device=device,
            dtype=torch.long,
        )
        correct_plan = codec.decode(correct_tokens)
        donor_plan = codec.decode(donor_tokens)
        swapped_plan_source = _swap_source_semantics(correct_plan, donor_plan)
        swapped_tokens = codec.encode(swapped_plan_source, max_tokens=1024)[
            "input_ids"
        ].to(device)
        swapped_plan = codec.decode(swapped_tokens)
        correct_anchor = model.compile_plan_modalities(correct_tokens, device=device)[
            "spatial_traj"
        ]
        swapped_anchor = model.compile_plan_modalities(swapped_tokens, device=device)[
            "spatial_traj"
        ]
        anchor_delta = (swapped_anchor.float() - correct_anchor.float()).abs()
        anchor_invariance = {
            "mae": float(anchor_delta.mean().cpu()),
            "max_abs": float(anchor_delta.max().cpu()),
        }
        if anchor_invariance["max_abs"] > 1.0e-6:
            raise RuntimeError(
                "semantic swap changed compiled spatial anchor: "
                f"max_abs={anchor_invariance['max_abs']}"
            )

        correct_caption = _structured_semantic_caption(creation)
        donor_caption = _structured_semantic_caption(donor_creation)
        if correct_caption["text"] == donor_caption["text"]:
            raise RuntimeError("correct and donor semantic captions are identical")
        conditions = {
            "plan_semantics_correct__caption_correct": (
                correct_tokens,
                correct_caption,
            ),
            "plan_semantics_correct__caption_donor": (
                correct_tokens,
                donor_caption,
            ),
            "plan_semantics_donor__caption_correct": (
                swapped_tokens,
                correct_caption,
            ),
            "plan_semantics_donor__caption_donor": (
                swapped_tokens,
                donor_caption,
            ),
        }
        render_seed = args.seed + 30_000
        outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        results: dict[str, Any] = {}
        for name, (plan_tokens, caption) in conditions.items():
            latent, audio = _render(
                model,
                plan_tokens,
                anchor=correct_anchor,
                previous_foa=None,
                semantic_caption=caption,
                modality_steps=args.modality_steps,
                cfg_scale=args.cfg_scale,
                seed=render_seed,
            )
            outputs[name] = (latent, audio)
            audio_path = args.output_dir / f"turn00.{name}.wav"
            _save_audio(audio_path, audio, sample_rate)
            results[name] = {"audio_path": str(audio_path)}

        baseline_name = "plan_semantics_correct__caption_correct"
        baseline_latent, baseline_audio = outputs[baseline_name]
        for name, (latent, audio) in outputs.items():
            results[name]["vs_baseline"] = _output_pair_metrics(
                latent,
                audio,
                baseline_latent=baseline_latent,
                baseline_audio=baseline_audio,
                hop=hop,
            )

        target_latent = family_latents[0].to(device=device, dtype=torch.float32)
        target_audio = _decode_latent(model, target_latent)
        target_path = args.output_dir / "turn00.target.wav"
        _save_audio(target_path, target_audio, sample_rate)
        for name, (latent, audio) in outputs.items():
            target_metrics = _output_pair_metrics(
                latent,
                audio,
                baseline_latent=target_latent,
                baseline_audio=target_audio,
                hop=hop,
            )
            results[name]["vs_target"] = target_metrics
            # Retain the historical key for downstream reports while making
            # the full target-relative acoustic diagnostic available.
            results[name]["vs_target_field"] = target_metrics["field"]

    report = {
        "schema": "stable_audio_tools.semantic_condition_diagnostic",
        "schema_version": 2,
        "checkpoint": str(args.checkpoint.resolve()),
        "weights": args.weights,
        "family_rank": args.family_rank,
        "family_id": family["family_id"],
        "donor_family_rank": args.donor_family_rank,
        "donor_family_id": donor["family_id"],
        "settings": {
            "modality_steps": args.modality_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "shared_render_seed": render_seed,
            "attention_backend": args.attention_backend,
            "latent_root": str(args.latent_root.resolve()),
            "sample_rate": sample_rate,
            "downsampling_ratio": hop,
        },
        "intervention_contract": {
            "fixed": [
                "sampler_noise",
                "source_count",
                "source_id",
                "activity",
                "gain",
                "motion",
                "room",
                "spatial_anchor",
            ],
            "varied": ["ScenePlan.event/content", "semantic_caption"],
            "anchor_invariance": anchor_invariance,
        },
        "correct_caption": correct_caption["text"],
        "donor_caption": donor_caption["text"],
        # Keep the exact codec-decoded control state beside the audio.  This
        # makes source-location scoring self-contained and prevents a later
        # dataset/template revision from silently changing the evaluator's
        # geometry or activity windows.
        "correct_scene_plan": correct_plan,
        "correct_plan_semantics": _source_semantics(correct_plan),
        "donor_plan_semantics": _source_semantics(donor_plan),
        "swapped_plan_semantics": _source_semantics(swapped_plan),
        "plan_token_counts": {
            "correct": int(correct_tokens.numel()),
            "semantic_swap": int(swapped_tokens.numel()),
        },
        "warmstart": warmstart,
        "target_audio_path": str(target_path),
        "conditions": results,
    }
    _atomic_json(args.output_dir / "RESULT.json", report)
    print(
        json.dumps(
            {
                "family_id": report["family_id"],
                "donor_family_id": report["donor_family_id"],
                "anchor_invariance": anchor_invariance,
                "condition_relative_latent_rmse": {
                    name: value["vs_baseline"]["latent"]["relative_rmse"]
                    for name, value in results.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
