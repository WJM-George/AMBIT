#!/usr/bin/env python3
"""Same-q source-slot permutation diagnostic for Spatial-CoT renderers.

The intervention swaps activity, motion, and gain between two persistent
source slots while keeping source identity, event/content semantics, caption,
room, sampler noise, and the compiled mixture anchor fixed.  The multiset of
source tracks is therefore unchanged and the global 4-D anchor must remain
bitwise-equivalent.  Any causal response beyond the discrete ScenePlan tokens
can be attributed to the source-slot-resolved frame adapter, including its
explicit source-semantic x geometry branch when configured.
"""
from __future__ import annotations

import argparse
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
from stable_audio_tools.data.spatial_counterfactual import (
    swap_scene_plan_source_controls,
)
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

from scripts.t2a.eval.diagnostics.diagnose_semantic_conditions import (
    _output_pair_metrics,
)
from scripts.t2a.eval.diagnostics.diagnose_spatial_conditions import (
    _atomic_json,
    _render,
    _reset_seed,
    _save_audio,
    _structured_semantic_caption,
)
from scripts.t2a.eval.evaluate_spatial_cot_checkpoint import _provider


def _sources_by_id(plan: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    sources = ((plan.get("scene") or {}).get("sources") or [])
    if not isinstance(sources, list) or not all(isinstance(x, dict) for x in sources):
        raise ValueError("ScenePlan sources must be a list of dictionaries")
    result = {str(source.get("source_id") or ""): source for source in sources}
    if len(result) != len(sources) or "" in result:
        raise ValueError("ScenePlan source IDs must be present and unique")
    return result


def _tensor_delta(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().cpu()
    reference = reference.detach().float().cpu()
    if value.shape != reference.shape:
        raise ValueError(f"tensor shapes differ: {value.shape} != {reference.shape}")
    delta = value - reference
    reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-12)
    return {
        "mae": float(delta.abs().mean()),
        "rmse": float(delta.square().mean().sqrt()),
        "relative_rmse": float(delta.square().mean().sqrt() / reference_rms),
        "max_abs": float(delta.abs().max()),
    }


def _tensor_stats(value: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().cpu()
    return {
        "mean_abs": float(value.abs().mean()),
        "rms": float(value.square().mean().sqrt()),
        "max_abs": float(value.abs().max()),
    }


def _source_semantic_metrics(
    summaries: torch.Tensor, mask: torch.Tensor
) -> dict[str, Any]:
    """Report whether exact source-span summaries are distinct and non-zero."""

    summaries = summaries.detach().float().cpu()
    mask = mask.detach().bool().cpu().flatten()
    if summaries.ndim != 2 or mask.shape != summaries.shape[:1]:
        raise ValueError(
            "source summaries/mask must be [S,D]/[S], got "
            f"{tuple(summaries.shape)}/{tuple(mask.shape)}"
        )
    active_indices = mask.nonzero(as_tuple=False).flatten()
    active = summaries[active_indices]
    if active.numel() == 0:
        return {
            "active_slots": [],
            "norms": [],
            "cosine_similarity": [],
            "maximum_off_diagonal_cosine": None,
        }
    norms = active.norm(dim=-1)
    normalized = active / norms[:, None].clamp_min(1.0e-12)
    cosine = normalized @ normalized.transpose(0, 1)
    if cosine.shape[0] > 1:
        off_diagonal = cosine[~torch.eye(cosine.shape[0], dtype=torch.bool)]
        maximum_off_diagonal = float(off_diagonal.max())
    else:
        maximum_off_diagonal = None
    return {
        "active_slots": [int(index) for index in active_indices.tolist()],
        "norms": [float(value) for value in norms.tolist()],
        "cosine_similarity": [
            [float(value) for value in row] for row in cosine.tolist()
        ],
        "maximum_off_diagonal_cosine": maximum_off_diagonal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("online", "ema"), default="online")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--pretransform-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--latent-root",
        type=Path,
        default=Path("/mnt/sdb/audio_dataset/spatial_cot_v1/latents/validation"),
    )
    parser.add_argument("--caption-overlay-root", type=Path)
    parser.add_argument(
        "--codec-root",
        type=Path,
        default=Path("/mnt/sdb/audio_dataset/spatial_cot_v1/codec"),
    )
    parser.add_argument("--family-rank", type=int, required=True)
    parser.add_argument("--source-a", default="source_0")
    parser.add_argument("--source-b", default="source_1")
    parser.add_argument("--modality-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--interface-only",
        action="store_true",
        help=(
            "Measure the compiled anchor, source summaries, and frame-aligned "
            "source residual without running the expensive sampler/decoder."
        ),
    )
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
        or args.source_a == args.source_b
        or args.modality_steps < 2
        or not math.isfinite(args.cfg_scale)
        or args.cfg_scale <= 0.0
    ):
        raise ValueError("invalid family/source/modality-step/CFG arguments")

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
        max_open_shards=1,
    )
    _, family = dataset[args.family_rank]
    creation = (family.get("family_turn_metadata") or [])[0]

    model_config = load_config(args.model_config)
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
    if not args.interface_only:
        pretransform_state = load_ckpt_state_dict(str(args.pretransform_checkpoint))
        model.load_pretransform_state_dict(pretransform_state)
        del pretransform_state

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    model = model.eval().requires_grad_(False).to(device)
    codec = model.get_plan_codec()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    with torch.inference_mode(), autocast:
        base_tokens = torch.as_tensor(
            creation["spatial_plan_tokens"]["input_ids"],
            device=device,
            dtype=torch.long,
        )
        base_plan = codec.decode(base_tokens)
        swapped_source = swap_scene_plan_source_controls(
            base_plan, args.source_a, args.source_b
        )
        swapped_tokens = codec.encode(swapped_source, max_tokens=1024)[
            "input_ids"
        ].to(device)
        swapped_plan = codec.decode(swapped_tokens)

        base_sources = _sources_by_id(base_plan)
        swapped_sources = _sources_by_id(swapped_plan)
        for source_id in base_sources:
            for semantic_key in ("event", "content"):
                if base_sources[source_id].get(semantic_key) != swapped_sources[
                    source_id
                ].get(semantic_key):
                    raise RuntimeError(
                        f"slot permutation changed {source_id}.{semantic_key}"
                    )

        base_anchor = model.compile_plan_modalities(base_tokens, device=device)[
            "spatial_traj"
        ]
        swapped_anchor = model.compile_plan_modalities(
            swapped_tokens, device=device
        )["spatial_traj"]
        anchor_delta = _tensor_delta(swapped_anchor, base_anchor)
        if anchor_delta["max_abs"] > 1.0e-6:
            raise RuntimeError(
                "source-slot permutation changed global anchor: "
                f"max_abs={anchor_delta['max_abs']}"
            )

        base_tracks = model.compile_plan_source_region_tracks(
            base_tokens, device=device
        )
        swapped_tracks = model.compile_plan_source_region_tracks(
            swapped_tokens, device=device
        )
        if base_tracks is None or swapped_tracks is None:
            raise RuntimeError("source-regional tracks are not enabled")
        caption = _structured_semantic_caption(creation)
        _, _, source_summaries, source_summary_masks = (
            model._encode_qwen_prefix([caption], device)
        )
        source_semantics = (
            source_summaries[0] if source_summaries is not None else None
        )
        source_semantic_mask = (
            source_summary_masks[0]
            if source_summary_masks is not None
            else None
        )
        if source_semantics is None or source_semantic_mask is None:
            raise RuntimeError(
                "source-binding diagnostic requires source semantic summaries"
            )
        source_semantic_metrics = _source_semantic_metrics(
            source_semantics, source_semantic_mask
        )
        base_residual = model.project_frame_aligned_source_regions(
            base_tracks,
            source_semantics=source_semantics,
            source_semantic_mask=source_semantic_mask,
        )
        swapped_residual = model.project_frame_aligned_source_regions(
            swapped_tracks,
            source_semantics=source_semantics,
            source_semantic_mask=source_semantic_mask,
        )
        if base_residual is None or swapped_residual is None:
            raise RuntimeError("source-regional residual is not enabled")

        core = model.model
        adapter = getattr(core, "frame_aligned_source_region_adapter", None)
        if adapter is None:
            raise RuntimeError("source-regional adapter is missing from the core")
        base_geometry_residual = model.project_frame_aligned_source_regions(
            base_tracks
        )
        swapped_geometry_residual = model.project_frame_aligned_source_regions(
            swapped_tracks
        )
        if base_geometry_residual is None or swapped_geometry_residual is None:
            raise RuntimeError("source geometry residual is not enabled")
        base_semantic_residual = base_residual - base_geometry_residual
        swapped_semantic_residual = swapped_residual - swapped_geometry_residual

        target_index = model.modality_ids.index("foa_latent")
        target_config = model.modalities[target_index]
        target_frames = int(target_config["default_shape"][0])
        target_channels = int(target_config["dim_latent"])
        noise_generator = torch.Generator(device=device).manual_seed(
            args.seed + 90_000
        )
        target_noise = torch.randn(
            (1, target_channels, target_frames),
            device=device,
            generator=noise_generator,
        )
        target_tokens = core.get_modality_info(target_index).latent_to_model(
            target_noise
        )[0]
        base_slot_residual = adapter.forward_slot_attention(
            target_tokens,
            base_tracks,
            source_semantics=source_semantics,
            source_semantic_mask=source_semantic_mask,
        )
        swapped_slot_residual = adapter.forward_slot_attention(
            target_tokens,
            swapped_tracks,
            source_semantics=source_semantics,
            source_semantic_mask=source_semantic_mask,
        )
        anchor_residual = model.project_frame_aligned_anchor(base_anchor)
        if anchor_residual is None:
            raise RuntimeError("frame-aligned global anchor residual is not enabled")

        interface_scales = {
            "target_noise_projection": _tensor_stats(target_tokens),
            "global_anchor_residual": _tensor_stats(anchor_residual),
            "source_early_full_base": _tensor_stats(base_residual),
            "source_early_full_swap_delta": _tensor_delta(
                swapped_residual, base_residual
            ),
            "source_early_geometry_base": _tensor_stats(
                base_geometry_residual
            ),
            "source_early_geometry_swap_delta": _tensor_delta(
                swapped_geometry_residual, base_geometry_residual
            ),
            "source_early_semantic_base": _tensor_stats(
                base_semantic_residual
            ),
            "source_early_semantic_swap_delta": _tensor_delta(
                swapped_semantic_residual, base_semantic_residual
            ),
            "source_late_slot_base": _tensor_stats(base_slot_residual),
            "source_late_slot_swap_delta": _tensor_delta(
                swapped_slot_residual, base_slot_residual
            ),
        }

        render_seed = None
        output_delta = None
        audio_paths: dict[str, str] = {}
        if not args.interface_only:
            render_seed = args.seed + 40_000
            base_latent, base_audio = _render(
                model,
                base_tokens,
                anchor=base_anchor,
                previous_foa=None,
                semantic_caption=caption,
                modality_steps=args.modality_steps,
                cfg_scale=args.cfg_scale,
                seed=render_seed,
            )
            swapped_latent, swapped_audio = _render(
                model,
                swapped_tokens,
                anchor=base_anchor,
                previous_foa=None,
                semantic_caption=caption,
                modality_steps=args.modality_steps,
                cfg_scale=args.cfg_scale,
                seed=render_seed,
            )
            base_path = args.output_dir / "turn00.binding_base.wav"
            swapped_path = args.output_dir / "turn00.binding_swapped.wav"
            _save_audio(base_path, base_audio, int(model_config["sample_rate"]))
            _save_audio(swapped_path, swapped_audio, int(model_config["sample_rate"]))
            output_delta = _output_pair_metrics(
                swapped_latent,
                swapped_audio,
                baseline_latent=base_latent,
                baseline_audio=base_audio,
                hop=int(model.downsampling_ratio),
            )
            audio_paths = {
                "base": str(base_path),
                "swapped": str(swapped_path),
            }

    report = {
        "schema": "stable_audio_tools.source_binding_diagnostic",
        "schema_version": 2,
        "checkpoint": str(args.checkpoint.resolve()),
        "weights": args.weights,
        "family_rank": args.family_rank,
        "family_id": family["family_id"],
        "source_pair": [args.source_a, args.source_b],
        "settings": {
            "modality_steps": args.modality_steps,
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "shared_render_seed": render_seed,
            "interface_only": args.interface_only,
            "latent_root": str(args.latent_root.resolve()),
        },
        "intervention_contract": {
            "fixed": [
                "sampler_noise",
                "source_id",
                "event/content semantics",
                "semantic_caption and exact source regions",
                "room",
                "global 4-D anchor",
                "multiset of source control tracks",
            ],
            "swapped": ["activity", "motion", "acoustics/gain"],
            "anchor_delta": anchor_delta,
        },
        "source_residual_delta": _tensor_delta(swapped_residual, base_residual),
        "interface_scales": interface_scales,
        "source_semantics": source_semantic_metrics,
        "output_delta": output_delta,
        "audio_paths": audio_paths,
        "plan_token_counts": {
            "base": int(base_tokens.numel()),
            "swapped": int(swapped_tokens.numel()),
        },
        "warmstart": warmstart,
    }
    _atomic_json(args.output_dir / "RESULT.json", report)
    print(
        json.dumps(
            {
                "family_id": report["family_id"],
                "anchor_max_abs": anchor_delta["max_abs"],
                "source_residual_relative_rmse": report[
                    "source_residual_delta"
                ]["relative_rmse"],
                "output_latent_relative_rmse": (
                    output_delta["latent"]["relative_rmse"]
                    if output_delta is not None
                    else None
                ),
                "output_audio_relative_rmse": (
                    output_delta["audio"]["relative_rmse"]
                    if output_delta is not None
                    else None
                ),
                "source_semantic_maximum_off_diagonal_cosine": (
                    source_semantic_metrics[
                        "maximum_off_diagonal_cosine"
                    ]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
