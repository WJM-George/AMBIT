#!/usr/bin/env python3
"""Causal renderer diagnostics for Plan, spatial anchor, and previous FOA.

The creation diagnostic is a 2x2 intervention with identical sampler noise:
correct/azimuth-rotated ScenePlan x correct/azimuth-rotated 4-D anchor.  Edit
turns compare no context, the correct target previous FOA, and a foreign
previous FOA while holding the current Plan and anchor fixed.
"""
from __future__ import annotations
import os

import argparse
import copy
import json
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torchaudio

from stable_audio_tools.configuration import load_config
from stable_audio_tools.data.spatial_caption_templates import (
    SEMANTIC_CAPTION_TEMPLATE_VERSION,
)
from stable_audio_tools.data.spatial_family_dataset import SpatialFamilyDataset
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

from scripts.t2a.eval.evaluate_spatial_cot_checkpoint import (
    _extract_target_latent,
    _plan_spatial_alignment_metrics,
    _provider,
    _spatial_alignment_metrics,
)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _reset_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _wrapped_azimuth(value: float) -> float:
    return ((float(value) + 180.0) % 360.0) - 180.0


_DIRECTION_CENTERS = (
    (0.0, "front"),
    (45.0, "front-left"),
    (90.0, "left"),
    (135.0, "rear-left"),
    (-180.0, "behind"),
    (-135.0, "rear-right"),
    (-90.0, "right"),
    (-45.0, "front-right"),
)


def _direction_label(azimuth: float) -> str:
    return min(
        _DIRECTION_CENTERS,
        key=lambda item: abs(_wrapped_azimuth(azimuth - item[0])),
    )[1]


def _rotate_scene_plan(plan: Mapping[str, Any], degrees: float) -> dict[str, Any]:
    rotated = copy.deepcopy(dict(plan))
    sources = ((rotated.get("scene") or {}).get("sources") or [])
    for source in sources:
        keyframes = ((source.get("motion") or {}).get("keyframes") or [])
        for keyframe in keyframes:
            position = keyframe.get("position") or {}
            azimuth = position.get("azimuth_deg")
            if azimuth is None:
                continue
            new_azimuth = _wrapped_azimuth(float(azimuth) + float(degrees))
            position["azimuth_deg"] = new_azimuth
            position["direction"] = _direction_label(new_azimuth)
    return rotated


def _rotate_anchor(anchor: torch.Tensor, degrees: float) -> torch.Tensor:
    value = anchor.clone()
    radians = math.radians(float(degrees))
    cosine = math.cos(radians)
    sine = math.sin(radians)
    x = value[:, 0].clone()
    y = value[:, 1].clone()
    value[:, 0] = cosine * x - sine * y
    value[:, 1] = sine * x + cosine * y
    return value


def _scale_anchor_direction(anchor: torch.Tensor, scale: float) -> torch.Tensor:
    value = anchor.clone()
    value[:, :3] *= float(scale)
    return value


def _decode_latent(model, latent: torch.Tensor) -> torch.Tensor:
    audio = model.decode_latent(latent.unsqueeze(0))[0].detach().float().cpu()
    if audio.ndim != 2 or audio.shape[0] != 4 or not bool(torch.isfinite(audio).all()):
        raise RuntimeError(f"invalid decoded FOA shape/value: {tuple(audio.shape)}")
    return audio


def _structured_semantic_caption(turn: Mapping[str, Any]) -> dict[str, Any]:
    text = turn.get("semantic_caption")
    metadata = turn.get("semantic_caption_metadata")
    if not isinstance(text, str) or not text or not isinstance(metadata, Mapping):
        raise ValueError(
            "diagnostic turn requires caption text and source-region metadata"
        )
    source_regions = metadata.get("source_regions")
    if not isinstance(source_regions, list) or not source_regions:
        raise ValueError("diagnostic caption has no source regions")
    return {
        "text": text,
        "source_regions": source_regions,
        "transcript_regions": metadata.get("transcript_regions") or [],
        "caption_template_version": metadata.get("version"),
        "caption_template_id": metadata.get("template_id"),
    }


def _render(
    model,
    plan_tokens: torch.Tensor,
    *,
    anchor: torch.Tensor,
    previous_foa: Optional[torch.Tensor],
    semantic_caption: Any,
    modality_steps: int,
    cfg_scale: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    provided = {"spatial_traj": anchor}
    if previous_foa is not None:
        provided["previous_foa"] = previous_foa
    _reset_seed(seed)
    sample = model.render_plan(
        plan_tokens,
        provided_modalities=provided,
        semantic_caption=semantic_caption,
        modality_steps=modality_steps,
        cfg_scale=cfg_scale,
    )
    latent = _extract_target_latent(model, sample)
    return latent.detach(), _decode_latent(model, latent)


def _alignment_summary(
    generated: torch.Tensor,
    *,
    current_plan: Mapping[str, Any],
    rotated_plan: Optional[Mapping[str, Any]],
    current_target: torch.Tensor,
    hop: int,
) -> dict[str, Any]:
    result = {
        "current_plan": _plan_spatial_alignment_metrics(
            generated, current_plan, hop=hop
        ),
        "current_target": _spatial_alignment_metrics(
            generated, current_target, hop=hop
        ),
    }
    if rotated_plan is not None:
        result["rotated_plan"] = _plan_spatial_alignment_metrics(
            generated, rotated_plan, hop=hop
        )
    return result


def _save_audio(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    torchaudio.save(str(path), audio.clamp(-1.0, 1.0), sample_rate)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("online", "ema"), default="online")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument(
        "--attention-backend",
        choices=("configured", "dense"),
        default="configured",
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
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--creation-only", action="store_true")
    parser.add_argument("--rotation-deg", type=float, default=180.0)
    parser.add_argument("--anchor-direction-scale", type=float, default=4.0)
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
        or args.turns < (1 if args.creation_only else 2)
        or args.modality_steps < 2
        or args.anchor_direction_scale <= 1.0
    ):
        raise ValueError("invalid family/donor/turn/modality-step arguments")

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
    donor_latents, donor = dataset[args.donor_family_rank]
    turns = list(family.get("family_turn_metadata") or [])[: args.turns]
    if len(turns) != args.turns or int(donor_latents.shape[0]) < args.turns:
        raise RuntimeError("family or donor does not contain the requested turns")

    model_config = load_config(args.model_config)
    if args.attention_backend == "dense":
        model_config["model"]["transfusion"]["transformer"][
            "use_flex_attn"
        ] = False
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

    target_latents = family_latents.to(device=device, dtype=torch.float32)
    donor_latents = donor_latents.to(device=device, dtype=torch.float32)
    plan_factorial: dict[str, Any] = {}
    context_interventions: list[dict[str, Any]] = []
    with torch.inference_mode(), autocast:
        creation = turns[0]
        correct_tokens = torch.as_tensor(
            creation["spatial_plan_tokens"]["input_ids"],
            device=device,
            dtype=torch.long,
        )
        correct_plan = codec.decode(correct_tokens)
        rotated_source_plan = _rotate_scene_plan(correct_plan, args.rotation_deg)
        rotated_tokens = codec.encode(rotated_source_plan, max_tokens=1024)[
            "input_ids"
        ].to(device)
        rotated_plan = codec.decode(rotated_tokens)
        correct_anchor = model.compile_plan_modalities(
            correct_tokens, device=device
        )["spatial_traj"]
        manual_rotated_anchor = _rotate_anchor(correct_anchor, args.rotation_deg)
        compiled_rotated_anchor = model.compile_plan_modalities(
            rotated_tokens, device=device
        )["spatial_traj"]
        scaled_anchor = _scale_anchor_direction(
            correct_anchor, args.anchor_direction_scale
        )
        scaled_rotated_anchor = _scale_anchor_direction(
            manual_rotated_anchor, args.anchor_direction_scale
        )
        q_rotation_consistency_mae = float(
            (manual_rotated_anchor - compiled_rotated_anchor).abs().mean().cpu()
        )
        current_target_audio = _decode_latent(model, target_latents[0])
        _save_audio(
            args.output_dir / "turn00.current_target.wav",
            current_target_audio,
            sample_rate,
        )
        factorial_conditions = {
            "plan_correct__q_correct": (correct_tokens, correct_anchor),
            "plan_correct__q_rotated": (correct_tokens, manual_rotated_anchor),
            "plan_rotated__q_correct": (rotated_tokens, correct_anchor),
            "plan_rotated__q_rotated": (rotated_tokens, compiled_rotated_anchor),
            "plan_correct__q_scaled": (correct_tokens, scaled_anchor),
            "plan_correct__q_rotated_scaled": (
                correct_tokens,
                scaled_rotated_anchor,
            ),
        }
        factorial_outputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        factorial_seed = args.seed + 10_000
        for name, (tokens, anchor) in factorial_conditions.items():
            latent, audio = _render(
                model,
                tokens,
                anchor=anchor,
                previous_foa=None,
                semantic_caption=_structured_semantic_caption(creation),
                modality_steps=args.modality_steps,
                cfg_scale=args.cfg_scale,
                seed=factorial_seed,
            )
            factorial_outputs[name] = (latent, audio)
            audio_path = args.output_dir / f"turn00.{name}.wav"
            _save_audio(audio_path, audio, sample_rate)
            plan_factorial[name] = {
                "audio_path": str(audio_path),
                "alignment": _alignment_summary(
                    audio,
                    current_plan=correct_plan,
                    rotated_plan=rotated_plan,
                    current_target=current_target_audio,
                    hop=hop,
                ),
            }
        baseline_latent, baseline_audio = factorial_outputs[
            "plan_correct__q_correct"
        ]
        for name, (latent, audio) in factorial_outputs.items():
            plan_factorial[name]["latent_mae_vs_baseline"] = float(
                (latent.float() - baseline_latent.float()).abs().mean().cpu()
            )
            plan_factorial[name]["audio_field_vs_baseline"] = (
                _spatial_alignment_metrics(audio, baseline_audio, hop=hop)
            )

        for turn_index in range(1, 1 if args.creation_only else args.turns):
            turn = turns[turn_index]
            tokens = torch.as_tensor(
                turn["spatial_plan_tokens"]["input_ids"],
                device=device,
                dtype=torch.long,
            )
            plan = codec.decode(tokens)
            anchor = model.compile_plan_modalities(tokens, device=device)[
                "spatial_traj"
            ]
            current_target = target_latents[turn_index]
            previous_target = target_latents[turn_index - 1]
            foreign_previous = donor_latents[turn_index - 1]
            current_target_audio = _decode_latent(model, current_target)
            previous_target_audio = _decode_latent(model, previous_target)
            foreign_previous_audio = _decode_latent(model, foreign_previous)
            context_values = {
                "none": None,
                "target_previous": previous_target,
                "foreign_previous": foreign_previous,
            }
            turn_seed = args.seed + 20_000 + turn_index
            for context_name, context_latent in context_values.items():
                latent, audio = _render(
                    model,
                    tokens,
                    anchor=anchor,
                    previous_foa=context_latent,
                    semantic_caption=_structured_semantic_caption(turn),
                    modality_steps=args.modality_steps,
                    cfg_scale=args.cfg_scale,
                    seed=turn_seed,
                )
                audio_path = (
                    args.output_dir
                    / f"turn{turn_index:02d}.context_{context_name}.wav"
                )
                _save_audio(audio_path, audio, sample_rate)
                context_interventions.append(
                    {
                        "turn": turn_index,
                        "context": context_name,
                        "audio_path": str(audio_path),
                        "current_plan": _plan_spatial_alignment_metrics(
                            audio, plan, hop=hop
                        ),
                        "current_target": _spatial_alignment_metrics(
                            audio, current_target_audio, hop=hop
                        ),
                        "previous_target": _spatial_alignment_metrics(
                            audio, previous_target_audio, hop=hop
                        ),
                        "foreign_previous": _spatial_alignment_metrics(
                            audio, foreign_previous_audio, hop=hop
                        ),
                        "latent_mae_to_current_target": float(
                            (latent.float() - current_target.float()).abs().mean().cpu()
                        ),
                        "latent_mae_to_context": (
                            None
                            if context_latent is None
                            else float(
                                (latent.float() - context_latent.float())
                                .abs()
                                .mean()
                                .cpu()
                            )
                        ),
                    }
                )

    report = {
        "schema": "stable_audio_tools.spatial_condition_diagnostic",
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "weights": args.weights,
        "family_rank": args.family_rank,
        "family_id": family["family_id"],
        "donor_family_rank": args.donor_family_rank,
        "donor_family_id": donor["family_id"],
        "settings": {
            "rotation_deg": args.rotation_deg,
            "anchor_direction_scale": args.anchor_direction_scale,
            "creation_only": args.creation_only,
            "modality_steps": args.modality_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "latent_root": str(args.latent_root.resolve()),
            "attention_backend": args.attention_backend,
        },
        "warmstart": warmstart,
        "q_rotation_consistency_mae": q_rotation_consistency_mae,
        "plan_q_factorial": plan_factorial,
        "context_interventions": context_interventions,
    }
    _atomic_json(args.output_dir / "RESULT.json", report)
    print(
        json.dumps(
            {
                "family_id": report["family_id"],
                "q_rotation_consistency_mae": q_rotation_consistency_mae,
                "conditions": list(plan_factorial),
                "context_cases": len(context_interventions),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
