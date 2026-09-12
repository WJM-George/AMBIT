#!/usr/bin/env python3
"""Measure source-binding reliance across the Rectified-Flow time axis.

The probe keeps the complete target mixture, semantic caption, global mixture
trajectory, room, and RF noise fixed.  It swaps only the activity, geometry,
and gain bundles owned by two persistent source identities.  For each fixed
student time it compares the correct and control-swapped Dense fields.

Student time follows this repository's Transfusion convention: ``t=0`` is
pure Gaussian noise and ``t=1`` is clean target data.  No audio is decoded and
no source waveform is generated or mixed.
"""
from __future__ import annotations
import os

import argparse
import json
import math
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from stable_audio_tools.configuration import load_config
from stable_audio_tools.data.spatial_caption_templates import (
    SEMANTIC_CAPTION_TEMPLATE_VERSION,
)
from stable_audio_tools.data.spatial_counterfactual import (
    swap_scene_plan_source_controls,
)
from stable_audio_tools.data.spatial_family_dataset import SpatialFamilyDataset
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from scripts.t2a.eval.diagnostics.diagnose_spatial_conditions import (
    _structured_semantic_caption,
)
from scripts.t2a.eval.evaluate_spatial_cot_checkpoint import _provider


DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/spatial_cot/"
    "bootstrap/qwen35_0p8b_spatial_chat_dense_joint_source_binding_base.json"
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/spatial_cot/probes/"
    "spcot_dense_source_kv_v15_identity100_20260813_073308/"
    "checkpoints/epoch=99-step=100.ckpt"
)
DEFAULT_LATENT_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/latents/train")
DEFAULT_CODEC_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/spatial_cot_v1/codec")
DEFAULT_TIMES = (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def _reset_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_times(value: str) -> tuple[float, ...]:
    try:
        times = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("times must be comma-separated numbers") from exc
    if not times or any(not math.isfinite(item) or not 0.0 <= item <= 1.0 for item in times):
        raise argparse.ArgumentTypeError("times must be finite values inside [0, 1]")
    if len(set(times)) != len(times):
        raise argparse.ArgumentTypeError("times must be unique")
    return times


def _tensor_delta(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    value = value.detach().float().cpu()
    reference = reference.detach().float().cpu()
    if value.shape != reference.shape:
        raise ValueError(f"tensor shapes differ: {value.shape} != {reference.shape}")
    delta = value - reference
    reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-12)
    return {
        "rms": float(delta.square().mean().sqrt()),
        "relative_rms": float(delta.square().mean().sqrt() / reference_rms),
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
    }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(statistics.fmean(values))


def _population_std(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _run_field(
    model,
    sample,
    *,
    times: torch.Tensor,
    target_index: int,
    dense_conditioning: Mapping[str, Any],
    flow_masks,
    flow_element_weights,
    seed: int,
) -> tuple[float, torch.Tensor]:
    captured: list[torch.Tensor] = []

    def capture_flow(
        modality_index,
        predicted_flows,
        _noised_modalities,
        _modality_times,
    ):
        if modality_index == target_index:
            if len(predicted_flows) != 1:
                raise RuntimeError(
                    "time-curve probe expects one target occurrence per forward"
                )
            captured.append(predicted_flows[0].detach().float().cpu())
        return sum(value.sum() * 0.0 for value in predicted_flows)

    _reset_seed(seed)
    _, breakdown = model.forward_renderer(
        [sample],
        times=times,
        return_breakdown=True,
        qwen_dropout_prob=0.0,
        modality_flow_masks=flow_masks,
        modality_flow_element_weights=flow_element_weights,
        dense_dit_conditioning=[dense_conditioning],
        dense_dit_active_masks=[
            None if flow_masks is None else flow_masks[0][target_index]
        ],
        modality_flow_auxiliary_loss_fn=capture_flow,
    )
    if len(captured) != 1:
        raise RuntimeError(f"captured {len(captured)} target fields, expected one")
    return float(breakdown.flow[target_index].detach()), captured[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--weights", choices=("online", "ema"), default="online")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--latent-root", type=Path, default=DEFAULT_LATENT_ROOT)
    parser.add_argument("--caption-overlay-root", type=Path)
    parser.add_argument("--codec-root", type=Path, default=DEFAULT_CODEC_ROOT)
    parser.add_argument("--family-rank", type=int, default=5)
    parser.add_argument("--source-a", default="source_0")
    parser.add_argument("--source-b", default="source_1")
    parser.add_argument(
        "--times",
        type=_parse_times,
        default=DEFAULT_TIMES,
        help="Comma-separated Transfusion student times; t=0 is pure noise.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
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
        args.latent_root / "READY",
        caption_overlay_root / "READY",
        args.codec_root / "READY",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (
        args.family_rank < 0
        or args.repeats < 1
        or args.source_a == args.source_b
    ):
        raise ValueError("family-rank/repeats/source arguments are invalid")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe requested but unavailable")
    torch.set_float32_matmul_precision("high")
    _reset_seed(args.seed)

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
    family_latents, family = dataset[args.family_rank]
    rows = list(family.get("family_turn_metadata") or [])
    if not rows or int(family_latents.shape[0]) != len(rows):
        raise RuntimeError("family latent/metadata rows are not aligned")
    creation = rows[0]

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
    model = model.eval().requires_grad_(False).to(device)
    _reset_seed(args.seed)

    codec = model.get_plan_codec()
    base_tokens = torch.as_tensor(
        creation["spatial_plan_tokens"]["input_ids"],
        device=device,
        dtype=torch.long,
    )
    base_plan = codec.decode(base_tokens)
    swapped_plan = swap_scene_plan_source_controls(
        base_plan, args.source_a, args.source_b
    )
    swapped_tokens = codec.encode(swapped_plan, max_tokens=1024)["input_ids"].to(
        device
    )
    base_anchor = model.compile_plan_modalities(base_tokens, device=device)[
        "spatial_traj"
    ]
    swapped_anchor = model.compile_plan_modalities(swapped_tokens, device=device)[
        "spatial_traj"
    ]
    anchor_delta = _tensor_delta(swapped_anchor, base_anchor)
    if anchor_delta["max_abs"] > 1.0e-6:
        raise RuntimeError(
            "source-control permutation changed the global mixture anchor: "
            f"max_abs={anchor_delta['max_abs']}"
        )

    base_tracks = model.compile_plan_source_region_tracks(base_tokens, device=device)
    swapped_tracks = model.compile_plan_source_region_tracks(
        swapped_tokens, device=device
    )
    if base_tracks is None or swapped_tracks is None:
        raise RuntimeError("source-resolved tracks are not enabled")
    caption = _structured_semantic_caption(creation)
    target = family_latents[0].to(device=device, dtype=torch.float32)
    modalities = {
        "previous_foa": torch.zeros_like(target),
        "spatial_traj": base_anchor,
        "foa_latent": target,
    }
    base_sample = model.build_renderer_sample(
        base_tokens,
        modalities,
        semantic_caption=caption,
        source_region_tracks=base_tracks,
        plan_after_modality="previous_foa",
    )
    swapped_sample = model.build_renderer_sample(
        swapped_tokens,
        modalities,
        semantic_caption=caption,
        source_region_tracks=swapped_tracks,
        plan_after_modality="previous_foa",
    )
    flow_masks = model.build_renderer_flow_masks([modalities])
    flow_element_weights = model.build_renderer_flow_element_weights([modalities])
    target_index = model.modality_ids.index("foa_latent")
    dense_conditioning = model.dense_dit_conditioning(caption)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    per_time: list[dict[str, Any]] = []
    with torch.inference_mode(), autocast:
        for student_time in args.times:
            repetitions = []
            for repeat in range(args.repeats):
                noise_seed = int(args.seed) + repeat * 1009
                times = torch.ones(
                    (1, len(model.modality_ids)),
                    device=device,
                    dtype=torch.float32,
                )
                times[0, target_index] = float(student_time)
                correct_loss, correct_field = _run_field(
                    model,
                    base_sample,
                    times=times,
                    target_index=target_index,
                    dense_conditioning=dense_conditioning,
                    flow_masks=flow_masks,
                    flow_element_weights=flow_element_weights,
                    seed=noise_seed,
                )
                swapped_loss, swapped_field = _run_field(
                    model,
                    swapped_sample,
                    times=times,
                    target_index=target_index,
                    dense_conditioning=dense_conditioning,
                    flow_masks=flow_masks,
                    flow_element_weights=flow_element_weights,
                    seed=noise_seed,
                )
                repetitions.append(
                    {
                        "seed": noise_seed,
                        "correct_loss": correct_loss,
                        "swapped_loss": swapped_loss,
                        "swapped_minus_correct_loss": swapped_loss - correct_loss,
                        "swapped_over_correct_loss": swapped_loss
                        / max(correct_loss, 1.0e-12),
                        "field_delta": _tensor_delta(
                            swapped_field, correct_field
                        ),
                    }
                )
            correct_losses = [row["correct_loss"] for row in repetitions]
            swapped_losses = [row["swapped_loss"] for row in repetitions]
            advantages = [
                row["swapped_minus_correct_loss"] for row in repetitions
            ]
            relative_field_deltas = [
                row["field_delta"]["relative_rms"] for row in repetitions
            ]
            per_time.append(
                {
                    "student_time": float(student_time),
                    "noise_fraction": float(1.0 - student_time),
                    "correct_loss_mean": _mean(correct_losses),
                    "correct_loss_std": _population_std(correct_losses),
                    "swapped_loss_mean": _mean(swapped_losses),
                    "swapped_loss_std": _population_std(swapped_losses),
                    "condition_advantage_mean": _mean(advantages),
                    "condition_advantage_std": _population_std(advantages),
                    "field_delta_relative_rms_mean": _mean(
                        relative_field_deltas
                    ),
                    "field_delta_relative_rms_std": _population_std(
                        relative_field_deltas
                    ),
                    "repetitions": repetitions,
                }
            )
            print(
                "[binding-time-curve] "
                f"t={student_time:.3f} "
                f"correct={per_time[-1]['correct_loss_mean']:.6f} "
                f"swapped={per_time[-1]['swapped_loss_mean']:.6f} "
                f"field_delta={per_time[-1]['field_delta_relative_rms_mean']:.6f}",
                flush=True,
            )

    report = {
        "schema": "stable_audio_tools.dense_source_binding_time_curve",
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "weights": args.weights,
        "model_config": str(args.model_config.resolve()),
        "family_rank": args.family_rank,
        "family_id": family["family_id"],
        "source_pair": [args.source_a, args.source_b],
        "rf_convention": {
            "student_time_0": "pure_gaussian_noise",
            "student_time_1": "clean_target_data",
            "noised_state": "t * data + (1 - t) * noise",
        },
        "intervention_contract": {
            "fixed": [
                "complete target FOA latent",
                "RF noise seed",
                "semantic caption and exact source spans",
                "source identities and event semantics",
                "room",
                "global mixture trajectory",
                "multiset of source control tracks",
            ],
            "swapped": ["activity", "motion/geometry", "gain/acoustics"],
            "decoded_audio": False,
            "per_source_waveforms": False,
            "global_anchor_delta": anchor_delta,
        },
        "settings": {
            "times": list(args.times),
            "repeats": args.repeats,
            "base_seed": args.seed,
            "device": str(device),
            "activity_flow_mask": flow_masks is not None,
            "latent_channel_weights": flow_element_weights is not None,
        },
        "per_time": per_time,
        "warmstart": warmstart,
    }
    _atomic_json(args.output, report)
    print(json.dumps({"output": str(args.output), "times": len(per_time)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
