#!/usr/bin/env python3
"""Fail-closed generation/editing check for a Spatial-CoT checkpoint.

The selected persisted family supplies real planner prompts, semantic captions,
and ground-truth ScenePlan tokens.  Sampling is closed loop: every edit receives
the previous *generated* FOA latent and generated ScenePlan, then the frozen FOA
VAE decodes each target latent to a four-channel waveform for inspection.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.train.gpu_preflight import assert_gpu_driver_healthy  # noqa: E402

assert_gpu_driver_healthy()

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.foa_intensity import (  # noqa: E402
    foa_to_intensity_trajectory,
)
from stable_audio_tools.data.spatial_conversation_metadata import (  # noqa: E402
    SpatialFamilyMetadata,
    evaluation_metadata_provider as _provider,
)
from stable_audio_tools.data.spatial_family_dataset import (  # noqa: E402
    SpatialFamilyDataset,
)
from stable_audio_tools.data.spatial_caption_templates import (  # noqa: E402
    SEMANTIC_CAPTION_TEMPLATE_VERSION,
)
from stable_audio_tools.data.spatial_story import compile_source_tracks  # noqa: E402
from stable_audio_tools.models import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import load_ckpt_state_dict  # noqa: E402


DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/spatial_cot/"
    "qwen35_0p8b_spatial_chat_500m.json"
)
DEFAULT_LATENT_ROOT = Path(
    "/mnt/sdb/audio_dataset/spatial_cot_v1/latents/train"
)
DEFAULT_CODEC_ROOT = Path("/mnt/sdb/audio_dataset/spatial_cot_v1/codec")
DEFAULT_PRETRANSFORM = Path(
    "/mnt/sdc/ckpts/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)
EVALUATOR_VERSION = 4


def _seed_everything(seed: int) -> None:
    """Seed data selection or sampling independently of model construction."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def _token_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    predicted = predicted.detach().long().cpu().flatten()
    target = target.detach().long().cpu().flatten()
    denominator = max(int(predicted.numel()), int(target.numel()))
    if denominator == 0:
        return 1.0
    overlap = min(int(predicted.numel()), int(target.numel()))
    matches = int((predicted[:overlap] == target[:overlap]).sum())
    return matches / denominator


def _select_renderer_previous_latent(
    mode: str,
    *,
    turn_index: int,
    closed_loop_previous: Optional[torch.Tensor],
    family_latents: torch.Tensor,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Select the FOA context for a controlled renderer intervention.

    ``closed_loop`` preserves the production path, ``target_previous`` removes
    accumulated renderer error while retaining the edit transport, and
    ``none`` isolates current-Plan rendering by starting every turn from fresh
    Gaussian state.
    """

    if mode == "closed_loop":
        return closed_loop_previous
    if mode == "none" or turn_index == 0:
        return None
    if mode != "target_previous":
        raise ValueError(f"unknown renderer context mode {mode!r}")
    if int(family_latents.shape[0]) < turn_index:
        raise RuntimeError(
            f"family latent tensor has no previous state for turn {turn_index}"
        )
    return torch.as_tensor(
        family_latents[turn_index - 1], device=device, dtype=torch.float32
    )


def _extract_target_latent(model, sample: list) -> torch.Tensor:
    if model._latent_id is None:
        raise RuntimeError("Spatial-CoT model has no target FOA latent modality")
    latent_index = model.modality_ids.index(model._latent_id)
    matches = [
        item[1]
        for item in sample
        if isinstance(item, tuple) and len(item) >= 2 and item[0] == latent_index
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"sampling returned {len(matches)} target FOA latent blocks"
        )
    latent = matches[0]
    if latent.ndim != 2:
        raise RuntimeError(f"target latent must be [C,T], got {tuple(latent.shape)}")
    if not bool(torch.isfinite(latent).all()):
        raise RuntimeError("generated target latent is non-finite")
    return latent


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _spatial_alignment_metrics(
    generated_foa: torch.Tensor,
    target_foa: torch.Tensor,
    *,
    hop: int,
    min_coherence: float = 0.1,
) -> dict[str, Any]:
    """Compare generated and target FOA active-intensity trajectories."""

    generated = generated_foa.detach().float().cpu()
    target = target_foa.detach().float().cpu()
    if (
        generated.ndim != 2
        or target.ndim != 2
        or generated.shape[0] != 4
        or target.shape[0] != 4
    ):
        raise ValueError("spatial alignment expects generated/target FOA [4,N]")
    if hop < 1:
        raise ValueError("spatial alignment hop must be positive")

    frame_count = min(generated.shape[-1], target.shape[-1]) // hop
    if frame_count < 1:
        raise ValueError("decoded FOA is shorter than one spatial frame")
    sample_count = frame_count * hop
    generated = generated[:, :sample_count]
    target = target[:, :sample_count]
    generated_traj = foa_to_intensity_trajectory(generated, hop=hop)
    target_traj = foa_to_intensity_trajectory(target, hop=hop)

    target_energy = (
        target.reshape(4, frame_count, hop).square().mean(dim=(0, 2))
    )
    energy_floor = max(1.0e-12, float(target_energy.max()) * 1.0e-4)
    active = target_energy >= energy_floor
    if not bool(active.any()):
        raise ValueError("target FOA has no active spatial frames")
    generated_coherence = (1.0 - generated_traj[:, 3]).clamp(0.0, 1.0)
    target_coherence = (1.0 - target_traj[:, 3]).clamp(0.0, 1.0)
    generated_direction = generated_traj[:, :3]
    target_direction = target_traj[:, :3]
    generated_norm = generated_direction.norm(dim=-1)
    target_norm = target_direction.norm(dim=-1)
    valid_direction = (
        active
        & (generated_coherence >= min_coherence)
        & (target_coherence >= min_coherence)
        & (generated_norm > 1.0e-6)
        & (target_norm > 1.0e-6)
    )

    direction_cosine = None
    angular_error_mean_deg = None
    angular_error_median_deg = None
    angular_error_p90_deg = None
    if bool(valid_direction.any()):
        generated_unit = (
            generated_direction[valid_direction]
            / generated_norm[valid_direction, None]
        )
        target_unit = (
            target_direction[valid_direction]
            / target_norm[valid_direction, None]
        )
        cosine = (generated_unit * target_unit).sum(dim=-1).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosine))
        direction_weights = (
            target_energy[valid_direction] * target_coherence[valid_direction]
        ).clamp_min(1.0e-12)
        direction_cosine = float(
            (cosine * direction_weights).sum() / direction_weights.sum()
        )
        angular_error_mean_deg = float(
            (angles * direction_weights).sum() / direction_weights.sum()
        )
        angular_error_median_deg = float(angles.median())
        angular_error_p90_deg = float(torch.quantile(angles, 0.9))

    diffuse_weights = target_energy[active].clamp_min(1.0e-12)
    diffuseness_error = (
        generated_traj[active, 3] - target_traj[active, 3]
    ).abs()
    diffuseness_mae = float(
        (diffuseness_error * diffuse_weights).sum() / diffuse_weights.sum()
    )
    return {
        "frame_count": int(frame_count),
        "active_frame_count": int(active.sum()),
        "valid_direction_frame_count": int(valid_direction.sum()),
        "valid_direction_fraction": float(
            valid_direction.sum().float() / active.sum().float()
        ),
        "direction_cosine": direction_cosine,
        "angular_error_mean_deg": angular_error_mean_deg,
        "angular_error_median_deg": angular_error_median_deg,
        "angular_error_p90_deg": angular_error_p90_deg,
        "diffuseness_mae": diffuseness_mae,
        "generated_mean_diffuseness": float(generated_traj[active, 3].mean()),
        "target_mean_diffuseness": float(target_traj[active, 3].mean()),
        "min_coherence": float(min_coherence),
        "hop": int(hop),
    }


@torch.inference_mode()
def _clap_content_metrics(
    clap_model,
    generated_foa: torch.Tensor,
    target_foa: torch.Tensor,
    caption: str,
    *,
    sample_rate: int,
) -> dict[str, Any]:
    """Measure semantic content independently of waveform phase and FOA DoA.

    CLAP consumes the W/omnidirectional channel. Each item is peak-normalized
    independently, so the score catches wrong events/timbre rather than merely
    repeating the loudness and silence checks enforced elsewhere.
    """

    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("CLAP content evaluation requires a non-empty caption")
    device = next(clap_model.model.parameters()).device

    def prepare(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.shape[0] != 4:
            raise ValueError("CLAP content evaluation expects FOA [4,N]")
        # ACN/SN3D channel zero is W and preserves content without a
        # direction-dependent cancellation caused by averaging W/Y/Z/X.
        mono = value[:1].float().to(device)
        peak = mono.abs().amax().clamp_min(1.0e-8)
        mono = mono / peak * (10.0 ** (-1.0 / 20.0))
        if sample_rate != 48_000:
            mono = torchaudio.functional.resample(mono, sample_rate, 48_000)
        return mono.clamp(-1.0, 1.0)

    # LAION-CLAP's fusion path assigns a float32 local branch into a global
    # tensor. Running it under the renderer's outer BF16 autocast makes that
    # destination BF16 and fails at the indexed assignment. CLAP weights and
    # its published evaluation path are float32, so isolate the metric from the
    # renderer autocast rather than mutating third-party code.
    autocast_off = (
        torch.autocast(device_type=device.type, enabled=False)
        if device.type in {"cpu", "cuda"}
        else nullcontext()
    )
    with autocast_off:
        generated = prepare(generated_foa)
        target = prepare(target_foa)
        audio_embeddings = clap_model.get_audio_embedding_from_data(
            x=torch.cat((generated, target), dim=0), use_tensor=True
        ).float()
        text_kwargs: dict[str, Any] = {"use_tensor": True}
        tokenizer_backend = getattr(clap_model, "tokenize", None)
        if callable(tokenizer_backend):
            # laion_clap 1.1.7's default wrapper unconditionally squeezes
            # dimension zero. A one-caption batch then becomes [T] instead of
            # [1,T] and recent Transformers' RoBERTa rejects it. Supplying the
            # package's own tokenizer without that squeeze preserves the batch.
            def tokenize_batch(texts):
                return tokenizer_backend(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )

            text_kwargs["tokenizer"] = tokenize_batch
        text_embedding = clap_model.get_text_embedding(
            [caption.strip()], **text_kwargs
        ).float()
    generated_embedding, target_embedding = audio_embeddings.unbind(0)
    generated_text = torch.nn.functional.cosine_similarity(
        generated_embedding[None], text_embedding, dim=-1
    )[0]
    target_text = torch.nn.functional.cosine_similarity(
        target_embedding[None], text_embedding, dim=-1
    )[0]
    generated_target = torch.nn.functional.cosine_similarity(
        generated_embedding[None], target_embedding[None], dim=-1
    )[0]
    return {
        "generated_target_audio_cosine": float(generated_target.cpu()),
        "generated_text_cosine": float(generated_text.cpu()),
        "target_text_cosine": float(target_text.cpu()),
        "semantic_deficit_vs_target": float((target_text - generated_text).cpu()),
        "channel": "W",
        "sample_rate": 48_000,
    }


def _latent_alignment_metrics(
    generated: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Report target-latent fit without treating it as a generation gate."""

    generated = torch.as_tensor(generated).detach().float()
    target = torch.as_tensor(target).detach().float()
    if generated.shape != target.shape:
        raise ValueError(
            "generated and target latents must have equal shape, got "
            f"{tuple(generated.shape)} and {tuple(target.shape)}"
        )
    if not bool(torch.isfinite(generated).all() and torch.isfinite(target).all()):
        raise ValueError("latent alignment requires finite tensors")
    difference = generated - target
    generated_rms = generated.square().mean().sqrt()
    target_rms = target.square().mean().sqrt()
    rmse = difference.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        generated.flatten().unsqueeze(0),
        target.flatten().unsqueeze(0),
        dim=-1,
        eps=1.0e-8,
    )[0]
    return {
        "mae": float(difference.abs().mean().cpu()),
        "rmse": float(rmse.cpu()),
        "relative_rmse": float((rmse / target_rms.clamp_min(1.0e-8)).cpu()),
        "cosine": float(cosine.cpu()),
        "generated_rms": float(generated_rms.cpu()),
        "target_rms": float(target_rms.cpu()),
    }


def _plan_spatial_alignment_metrics(
    generated_foa: torch.Tensor,
    plan: Mapping[str, Any],
    *,
    hop: int,
    min_coherence: float = 0.1,
) -> dict[str, Any]:
    """Measure DoA only where a ScenePlan has one active source.

    Target-waveform intensity is not a stable directional reference while
    several independently generated sources overlap. Single-source plan frames
    are unambiguous and directly test whether source tracks control FOA.
    """

    generated = generated_foa.detach().float().cpu()
    if generated.ndim != 2 or generated.shape[0] != 4 or hop < 1:
        raise ValueError("plan spatial alignment expects generated FOA [4,N]")
    frame_count = generated.shape[-1] // hop
    if frame_count < 1:
        raise ValueError("decoded FOA is shorter than one spatial frame")
    generated = generated[:, : frame_count * hop]
    trajectory = foa_to_intensity_trajectory(generated, hop=hop)
    compiled = compile_source_tracks(
        plan,
        num_frames=frame_count,
        max_sources=4,
    )["tracks"]
    active = compiled[:, 0].gt(0.5)
    single_source = active.sum(dim=0).eq(1)
    expected_direction = (
        compiled[:, 1:4] * active[:, None].to(compiled)
    ).sum(dim=0).transpose(0, 1)
    generated_direction = trajectory[:, :3]
    generated_norm = generated_direction.norm(dim=-1)
    generated_coherence = (1.0 - trajectory[:, 3]).clamp(0.0, 1.0)
    valid = (
        single_source
        & generated_coherence.ge(float(min_coherence))
        & generated_norm.gt(1.0e-6)
    )
    direction_cosine = None
    angular_error_mean_deg = None
    angular_error_median_deg = None
    angular_error_p90_deg = None
    if bool(valid.any()):
        generated_unit = generated_direction[valid] / generated_norm[valid, None]
        expected_unit = torch.nn.functional.normalize(
            expected_direction[valid], dim=-1
        )
        cosine = (generated_unit * expected_unit).sum(dim=-1).clamp(-1.0, 1.0)
        angles = torch.rad2deg(torch.acos(cosine))
        direction_cosine = float(cosine.mean())
        angular_error_mean_deg = float(angles.mean())
        angular_error_median_deg = float(angles.median())
        angular_error_p90_deg = float(torch.quantile(angles, 0.9))
    single_count = int(single_source.sum())
    valid_count = int(valid.sum())
    return {
        "frame_count": int(frame_count),
        "single_source_frame_count": single_count,
        "valid_direction_frame_count": valid_count,
        "valid_direction_fraction": (
            valid_count / single_count if single_count else None
        ),
        "direction_cosine": direction_cosine,
        "angular_error_mean_deg": angular_error_mean_deg,
        "angular_error_median_deg": angular_error_median_deg,
        "angular_error_p90_deg": angular_error_p90_deg,
        "min_coherence": float(min_coherence),
        "hop": int(hop),
    }


def _silence_alignment_metrics(
    generated_foa: torch.Tensor,
    target_foa: torch.Tensor,
    *,
    frame_samples: int,
    absolute_silence_dbfs: float = -60.0,
    relative_silence_db: float = -40.0,
    settling_frames: int = 8,
) -> dict[str, Any]:
    """Measure generated energy specifically where the target is silent."""

    generated = generated_foa.detach().float().cpu()
    target = target_foa.detach().float().cpu()
    if generated.ndim != 2 or target.ndim != 2 or generated.shape[0] != 4:
        raise ValueError("silence alignment expects generated/target FOA [4,N]")
    if frame_samples < 1:
        raise ValueError("frame_samples must be positive")
    frame_count = min(generated.shape[-1], target.shape[-1]) // frame_samples
    if frame_count < 1:
        raise ValueError("audio is shorter than one silence-analysis frame")
    sample_count = frame_count * frame_samples
    generated_frames = generated[:, :sample_count].reshape(
        4, frame_count, frame_samples
    )
    target_frames = target[:, :sample_count].reshape(
        4, frame_count, frame_samples
    )
    generated_rms = generated_frames.square().mean(dim=(0, 2)).sqrt()
    target_rms = target_frames.square().mean(dim=(0, 2)).sqrt()
    absolute = 10.0 ** (float(absolute_silence_dbfs) / 20.0)
    relative = float(target_rms.max()) * (
        10.0 ** (float(relative_silence_db) / 20.0)
    )
    threshold = max(absolute, relative)
    silent = target_rms.le(threshold)
    active = ~silent
    generated_silent_rms = None
    target_silent_rms = None
    max_generated_silent_frame_rms = None
    if bool(silent.any()):
        generated_silent_rms = float(
            generated_frames[:, silent].square().mean().sqrt()
        )
        target_silent_rms = float(target_frames[:, silent].square().mean().sqrt())
        max_generated_silent_frame_rms = float(generated_rms[silent].max())
    trailing_silent_frames = 0
    for value in reversed(silent.tolist()):
        if not value:
            break
        trailing_silent_frames += 1
    settled_trailing_rms = None
    settled_start_frame = None
    if trailing_silent_frames > settling_frames:
        settled_start_frame = (
            frame_count - trailing_silent_frames + int(settling_frames)
        )
        settled_trailing_rms = float(
            generated_frames[:, settled_start_frame:].square().mean().sqrt()
        )
    return {
        "frame_count": int(frame_count),
        "silent_frame_count": int(silent.sum()),
        "silent_frame_fraction": float(silent.float().mean()),
        "threshold_dbfs": 20.0 * math.log10(max(threshold, 1.0e-12)),
        "generated_silent_rms": generated_silent_rms,
        "generated_silent_rms_dbfs": (
            None
            if generated_silent_rms is None
            else 20.0 * math.log10(max(generated_silent_rms, 1.0e-12))
        ),
        "target_silent_rms": target_silent_rms,
        "max_generated_silent_frame_rms": max_generated_silent_frame_rms,
        "target_active_rms": (
            float(target_frames[:, active].square().mean().sqrt())
            if bool(active.any())
            else None
        ),
        "trailing_silent_frames": trailing_silent_frames,
        "settling_frames": int(settling_frames),
        "settled_trailing_start_frame": settled_start_frame,
        "generated_settled_trailing_rms": settled_trailing_rms,
        "generated_settled_trailing_rms_dbfs": (
            None
            if settled_trailing_rms is None
            else 20.0 * math.log10(max(settled_trailing_rms, 1.0e-12))
        ),
        "frame_samples": int(frame_samples),
    }


def _renderer_audio_gate_failures(
    *,
    plan_spatial: Mapping[str, Any],
    target_plan_spatial: Mapping[str, Any],
    silence: Mapping[str, Any],
    max_plan_spatial_excess_deg: float,
    min_plan_spatial_valid_fraction: float,
    max_settled_trailing_rms: float,
) -> list[str]:
    failures: list[str] = []
    single_frames = int(plan_spatial.get("single_source_frame_count") or 0)
    target_angle = target_plan_spatial.get("angular_error_mean_deg")
    generated_angle = plan_spatial.get("angular_error_mean_deg")
    if single_frames and target_angle is not None:
        if generated_angle is None:
            failures.append("plan spatial direction has no valid generated frames")
        elif float(generated_angle) - float(target_angle) > float(
            max_plan_spatial_excess_deg
        ):
            failures.append(
                "plan spatial angular excess "
                f"{float(generated_angle) - float(target_angle):.3f}deg > "
                f"{float(max_plan_spatial_excess_deg):.3f}deg "
                f"(generated={float(generated_angle):.3f}, "
                f"target_codec_floor={float(target_angle):.3f})"
            )
        valid_fraction = plan_spatial.get("valid_direction_fraction")
        if valid_fraction is None or float(valid_fraction) < float(
            min_plan_spatial_valid_fraction
        ):
            failures.append(
                "plan spatial valid fraction "
                f"{valid_fraction} < {float(min_plan_spatial_valid_fraction):.3f}"
            )
    settled_rms = silence.get("generated_settled_trailing_rms")
    if settled_rms is not None and float(settled_rms) > float(
        max_settled_trailing_rms
    ):
        failures.append(
            "settled trailing-silence RMS "
            f"{float(settled_rms):.8g} > "
            f"{float(max_settled_trailing_rms):.8g}"
        )
    return failures


def _describe_token(codec, token_id: int) -> str:
    token_id = int(token_id)
    if 0 <= token_id < len(codec.id_to_token):
        return str(codec.id_to_token[token_id])
    for name, spec in codec.numeric.items():
        offset = int(spec["offset"])
        if offset <= token_id < offset + int(spec["count"]):
            return f"<{name}:{codec._dequantize(name, token_id)}>"
    piece = token_id - int(codec.text_offset)
    if 0 <= piece < int(codec.text_vocab_size):
        return f"<text_piece:{codec.text_processor.id_to_piece(piece)!r}>"
    return f"<unused:{token_id}>"


def _roundtrip_diagnostic(codec, predicted: torch.Tensor, roundtrip: torch.Tensor) -> dict:
    predicted_ids = predicted.detach().long().cpu().flatten().tolist()
    roundtrip_ids = roundtrip.detach().long().cpu().flatten().tolist()
    overlap = min(len(predicted_ids), len(roundtrip_ids))
    mismatch = next(
        (
            index
            for index in range(overlap)
            if predicted_ids[index] != roundtrip_ids[index]
        ),
        overlap if len(predicted_ids) != len(roundtrip_ids) else None,
    )
    start = max(0, (mismatch or 0) - 8)
    stop = min(max(len(predicted_ids), len(roundtrip_ids)), (mismatch or 0) + 9)

    def window(values: list[int]) -> list[dict]:
        return [
            {
                "index": index,
                "id": int(values[index]),
                "token": _describe_token(codec, int(values[index])),
            }
            for index in range(start, min(stop, len(values)))
        ]

    return {
        "predicted_length": len(predicted_ids),
        "roundtrip_length": len(roundtrip_ids),
        "first_mismatch": mismatch,
        "predicted_ids": predicted_ids,
        "roundtrip_ids": roundtrip_ids,
        "predicted_window": window(predicted_ids),
        "roundtrip_window": window(roundtrip_ids),
    }


def _plan_field_metrics(predicted: dict, target: dict) -> dict:
    """Compare decoded plans by stable state fields, not BPE segmentation."""

    predicted_sources = {
        str(source.get("source_id")): source
        for source in (predicted.get("scene") or {}).get("sources") or []
    }
    target_sources = {
        str(source.get("source_id")): source
        for source in (target.get("scene") or {}).get("sources") or []
    }
    predicted_ids = set(predicted_sources)
    target_ids = set(target_sources)
    component_checks = []
    component_names = ("event", "content", "activity", "acoustics", "motion")
    for source_id in sorted(predicted_ids | target_ids):
        predicted_source = predicted_sources.get(source_id)
        target_source = target_sources.get(source_id)
        for component in component_names:
            component_checks.append(
                bool(
                    predicted_source is not None
                    and target_source is not None
                    and predicted_source.get(component) == target_source.get(component)
                )
            )
    component_count = len(component_checks)
    component_matches = sum(component_checks)
    top_level = {
        "audio_exact": predicted.get("audio") == target.get("audio"),
        "mix_exact": predicted.get("mix") == target.get("mix"),
        "room_exact": (predicted.get("scene") or {}).get("room")
        == (target.get("scene") or {}).get("room"),
        "source_count_exact": len(predicted_sources) == len(target_sources),
        "source_ids_exact": predicted_ids == target_ids,
    }
    return {
        "decoded_plan_exact": predicted == target,
        **top_level,
        "predicted_source_ids": sorted(predicted_ids),
        "target_source_ids": sorted(target_ids),
        "source_component_matches": component_matches,
        "source_component_count": component_count,
        "source_component_accuracy": (
            component_matches / component_count if component_count else 1.0
        ),
    }


_EDIT_PATH_COMPONENT = re.compile(r"([^.\[\]]+)(?:\[(\d+)\])?")
_MISSING_EDIT_VALUE = object()


def _edit_field_metrics(
    predicted: Mapping[str, Any],
    target: Mapping[str, Any],
    previous: Optional[Mapping[str, Any]],
    diff: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure whether the requested state delta was applied exactly.

    Whole-plan equality and positional token accuracy obscure small edits.  A
    gain operation, for example, changes one quantized token in a roughly
    500-token plan.  The persisted recipe diff names the authoritative source
    and field paths, so this metric evaluates those fields directly after codec
    canonicalization and separately reports an incorrect-but-nonzero change.
    """

    if previous is None or not isinstance(diff, Mapping):
        return {
            "applicable": False,
            "changed_field_count": 0,
            "changed_field_matches": 0,
            "changed_field_accuracy": None,
            "changed_fields_exact": None,
            "applied_field_count": 0,
            "applied_field_fraction": None,
            "fields": [],
        }

    def sources(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        return {
            str(source.get("source_id")): source
            for source in (plan.get("scene") or {}).get("sources") or []
            if isinstance(source, Mapping)
        }

    def path_value(value: Any, path: str):
        current = value
        for name, raw_index in _EDIT_PATH_COMPONENT.findall(path):
            if not isinstance(current, Mapping) or name not in current:
                return _MISSING_EDIT_VALUE
            current = current[name]
            if raw_index:
                index = int(raw_index)
                if not isinstance(current, list) or index >= len(current):
                    return _MISSING_EDIT_VALUE
                current = current[index]
        return current

    predicted_sources = sources(predicted)
    target_sources = sources(target)
    previous_sources = sources(previous)
    fields: list[dict[str, Any]] = []

    for source_id in diff.get("added") or []:
        source_id = str(source_id)
        if source_id not in target_sources or source_id in previous_sources:
            continue
        matched = source_id in predicted_sources
        fields.append(
            {
                "source_id": source_id,
                "path": "<source_added>",
                "target_match": matched,
                "changed_from_previous": matched,
            }
        )
    for source_id in diff.get("removed") or []:
        source_id = str(source_id)
        if source_id in target_sources or source_id not in previous_sources:
            continue
        matched = source_id not in predicted_sources
        fields.append(
            {
                "source_id": source_id,
                "path": "<source_removed>",
                "target_match": matched,
                "changed_from_previous": matched,
            }
        )
    for change in diff.get("changed") or []:
        if not isinstance(change, Mapping):
            continue
        source_id = str(change.get("source_id") or "")
        predicted_source = predicted_sources.get(source_id)
        target_source = target_sources.get(source_id)
        previous_source = previous_sources.get(source_id)
        for path in change.get("fields") or []:
            path = str(path)
            target_value = path_value(target_source, path)
            previous_value = path_value(previous_source, path)
            # Recipe provenance contains dry-audio/source-dataset fields which
            # are deliberately outside the executable ScenePlan codec.  It can
            # also contain sub-bin changes that canonicalize to the same token.
            if (
                target_value is _MISSING_EDIT_VALUE
                and previous_value is _MISSING_EDIT_VALUE
            ) or target_value == previous_value:
                continue
            predicted_value = path_value(predicted_source, path)
            fields.append(
                {
                    "source_id": source_id,
                    "path": path,
                    "target_match": predicted_value == target_value,
                    "changed_from_previous": predicted_value != previous_value,
                }
            )

    count = len(fields)
    matches = sum(bool(item["target_match"]) for item in fields)
    applied = sum(bool(item["changed_from_previous"]) for item in fields)
    return {
        "applicable": count > 0,
        "changed_field_count": count,
        "changed_field_matches": matches,
        "changed_field_accuracy": matches / count if count else None,
        "changed_fields_exact": matches == count if count else None,
        "applied_field_count": applied,
        "applied_field_fraction": applied / count if count else None,
        "fields": fields,
    }


def _plan_gate_failures(
    *,
    plan_accuracy: float,
    field_metrics: dict,
    min_plan_token_accuracy: float,
    min_source_component_accuracy: float,
    require_exact_plan: bool,
) -> list[str]:
    failures = []
    if plan_accuracy < min_plan_token_accuracy:
        failures.append(
            "plan_token_accuracy "
            f"{plan_accuracy:.6f} < {min_plan_token_accuracy:.6f}"
        )
    if field_metrics["source_component_accuracy"] < min_source_component_accuracy:
        failures.append(
            "source_component_accuracy "
            f"{field_metrics['source_component_accuracy']:.6f} < "
            f"{min_source_component_accuracy:.6f}"
        )
    if require_exact_plan and not field_metrics["decoded_plan_exact"]:
        failures.append("decoded ScenePlan is not exact")
    return failures


PLAN_GROUP_NAMES = {
    0: "ignore",
    1: "grammar",
    2: "semantic",
    3: "room",
    4: "spatial_categorical",
    5: "spatial_metric",
    6: "motion",
    7: "speech_content",
}


def _teacher_forced_token_metrics(
    *,
    codec,
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    loss_group_ids: Optional[torch.Tensor],
    min_sources: int,
    max_sources: int,
    fixed_duration_sec: float,
) -> dict:
    """Score teacher-prefix choices under the same FSM mask as free decoding."""

    logits = logits.detach().float().cpu()
    targets = target_tokens.detach().long().cpu().flatten()
    groups = (
        None
        if loss_group_ids is None
        else loss_group_ids.detach().long().cpu().flatten()
    )
    if logits.ndim != 2 or logits.shape[0] != targets.numel():
        raise ValueError(
            "teacher-forced logits/target shape mismatch: "
            f"{tuple(logits.shape)} vs {tuple(targets.shape)}"
        )
    if groups is not None and groups.numel() != targets.numel():
        raise ValueError("teacher-forced loss groups do not match target length")

    errors = []
    ambiguous_tokens = 0
    for index, target_tensor in enumerate(targets):
        prefix = targets[:index].tolist()
        allowed = sorted(
            codec.allowed_next_ids(
                prefix,
                min_sources=min_sources,
                max_sources=max_sources,
                fixed_duration_sec=fixed_duration_sec,
            )
        )
        target = int(target_tensor)
        if target not in allowed:
            raise RuntimeError(
                f"canonical target token {target} is not FSM-allowed at index {index}"
            )
        ambiguous_tokens += int(len(allowed) > 1)
        allowed_tensor = torch.tensor(allowed, dtype=torch.long)
        candidate_logits = logits[index].index_select(0, allowed_tensor)
        best_position = int(candidate_logits.argmax())
        predicted = int(allowed[best_position])
        if predicted == target:
            continue
        target_position = allowed.index(target)
        target_logit = float(candidate_logits[target_position])
        margin = target_logit - float(candidate_logits[best_position])
        rank = 1 + int((candidate_logits > target_logit).sum())
        top_count = min(5, len(allowed))
        top_positions = torch.topk(candidate_logits, k=top_count).indices.tolist()
        group_id = int(groups[index]) if groups is not None else None
        errors.append(
            {
                "index": index,
                "group_id": group_id,
                "group": PLAN_GROUP_NAMES.get(group_id, str(group_id)),
                "target_id": target,
                "target": _describe_token(codec, target),
                "predicted_id": predicted,
                "predicted": _describe_token(codec, predicted),
                "target_rank": rank,
                "target_minus_best_logit": margin,
                "allowed_candidate_count": len(allowed),
                "top_candidates": [
                    {
                        "id": int(allowed[position]),
                        "token": _describe_token(codec, int(allowed[position])),
                        "logit": float(candidate_logits[position]),
                    }
                    for position in top_positions
                ],
            }
        )
    token_count = int(targets.numel())
    error_count = len(errors)
    return {
        "token_count": token_count,
        "ambiguous_token_count": ambiguous_tokens,
        "error_count": error_count,
        "token_accuracy": (token_count - error_count) / token_count,
        "exact": error_count == 0,
        "errors": errors,
    }


def _teacher_forced_plan_diagnostic(
    model,
    *,
    caption: Any,
    plan_tokens: Any,
    context_modalities: Optional[Mapping[str, torch.Tensor]] = None,
    prefix_plan_tokens: Any = None,
) -> dict:
    """Run one aligned forward pass and inspect every FSM-constrained choice."""

    core = model.model
    device = next(core.parameters()).device
    ids_list, soft_list, _, _ = model._encode_qwen_prefix([caption], device)
    caption_ids, soft = ids_list[0], soft_list[0]
    target = model._plan_input_ids(plan_tokens).to(device)
    groups = model._plan_group_ids(plan_tokens, expected_length=int(target.numel()))
    context = dict(context_modalities or {})
    use_mixed_context = bool(context) or prefix_plan_tokens is not None

    if use_mixed_context:
        items: list = [
            torch.tensor([core.sos_id], device=device, dtype=torch.long),
            caption_ids,
        ]
        expanded_offset = 1 + int(caption_ids.numel())
        for modality_id, raw_value in context.items():
            if modality_id not in model.modality_ids:
                raise KeyError(f"unknown teacher-forced context {modality_id!r}")
            modality_index = model.modality_ids.index(modality_id)
            value = torch.as_tensor(raw_value, device=device, dtype=torch.float32)
            items.append((modality_index, value))
            expanded_offset += model._expanded_modality_length(
                core, modality_index, value
            )
        if prefix_plan_tokens is not None:
            prefix = model._plan_input_ids(prefix_plan_tokens).to(device)
            items.append(prefix)
            expanded_offset += int(prefix.numel())
        plan_start = expanded_offset
        items.append(target)
        times = (
            torch.ones(
                (1, int(core.num_modalities)),
                device=device,
                dtype=torch.float32,
            )
            if context
            else None
        )
        soft_embed = model._set_soft_list_on_core(core, [soft])
        try:
            all_logits = core(
                [items], times=times, return_loss=False, prob_uncond=0.0
            )[0]
        finally:
            soft_embed.set_soft_list(None)
        stop = plan_start - 1 + int(target.numel())
        selected = all_logits[plan_start - 1 : stop]
    else:
        ignored_prefix = torch.full(
            (soft.shape[0],), -1, dtype=torch.long, device=device
        )
        sequence = torch.cat((ignored_prefix, target))
        soft_embed = model._set_soft_list_on_core(core, [soft])
        try:
            all_logits = core.forward_text(
                sequence[:-1].unsqueeze(0), return_loss=False
            )[0]
        finally:
            soft_embed.set_soft_list(None)
        start = int(soft.shape[0]) - 1
        selected = all_logits[start : start + int(target.numel())]

    if selected.shape[0] != target.numel():
        raise RuntimeError("teacher-forced planner logit alignment failed")
    selected = selected.masked_fill(
        ~core.text_only_logits_mask,
        torch.finfo(selected.dtype).min,
    )
    min_sources, max_sources = model._plan_source_limits()
    result = _teacher_forced_token_metrics(
        codec=model.get_plan_codec(),
        logits=selected,
        target_tokens=target,
        loss_group_ids=groups,
        min_sources=min_sources,
        max_sources=max_sources,
        fixed_duration_sec=model._default_plan_duration_sec(),
    )
    result["qwen_prefix_tokens"] = int(soft.shape[0])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "online"), default="ema")
    parser.add_argument(
        "--modality-warmstart-checkpoint",
        type=Path,
        help=(
            "optionally transplant only named modality interfaces after loading "
            "the primary checkpoint"
        ),
    )
    parser.add_argument(
        "--modality-warmstart-ids", default="spatial_traj"
    )
    parser.add_argument(
        "--modality-warmstart-weights",
        choices=("ema", "online"),
        default="ema",
    )
    parser.add_argument("--teacher-forced-only", action="store_true")
    parser.add_argument(
        "--require-teacher-forced-exact",
        action="store_true",
        help="make teacher-forced-only fail unless planner and understanding are exact",
    )
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--pretransform-checkpoint", type=Path, default=DEFAULT_PRETRANSFORM)
    parser.add_argument("--latent-root", type=Path, default=DEFAULT_LATENT_ROOT)
    parser.add_argument(
        "--caption-overlay-root",
        type=Path,
        help=(
            "versioned semantic-caption overlay; defaults to the canonical "
            "overlay matching the latent split"
        ),
    )
    parser.add_argument("--codec-root", type=Path, default=DEFAULT_CODEC_ROOT)
    parser.add_argument("--family-rank", type=int, default=0)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--modality-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument(
        "--renderer-plan",
        choices=("generated", "target"),
        default="generated",
        help=(
            "render the free planner output or the authoritative target plan; "
            "target isolates renderer quality from planner errors"
        ),
    )
    parser.add_argument(
        "--renderer-context",
        choices=("closed_loop", "target_previous", "none"),
        default="closed_loop",
        help=(
            "select generated previous FOA, ground-truth previous FOA, or no "
            "previous FOA; the latter two are causal renderer interventions"
        ),
    )
    parser.add_argument(
        "--render-seed-policy",
        choices=("stream", "same_each_turn"),
        default="stream",
        help=(
            "consume one RNG stream across turns, or reset to --seed before "
            "every renderer call for matched-condition causal comparisons"
        ),
    )
    parser.add_argument("--plan-temperature", type=float, default=0.0)
    parser.add_argument("--max-plan-tokens", type=int, default=1024)
    parser.add_argument("--min-plan-token-accuracy", type=float, default=1.0)
    parser.add_argument(
        "--min-source-component-accuracy", type=float, default=1.0
    )
    parser.add_argument("--min-edit-field-accuracy", type=float, default=0.0)
    parser.add_argument(
        "--require-exact-plan",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check-understanding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also require free FOA-to-ScenePlan understanding decode to pass",
    )
    parser.add_argument("--min-audio-rms", type=float, default=1.0e-6)
    parser.add_argument(
        "--audio-quality-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail on plan-relative FOA direction or settled tail-silence leakage",
    )
    parser.add_argument(
        "--max-plan-spatial-excess-deg", type=float, default=30.0
    )
    parser.add_argument(
        "--min-plan-spatial-valid-fraction", type=float, default=0.5
    )
    parser.add_argument(
        "--max-settled-trailing-rms", type=float, default=1.0e-3
    )
    parser.add_argument(
        "--clap-content-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="measure generated-target and text-audio semantic similarity with CLAP",
    )
    parser.add_argument(
        "--clap-model",
        default="music_speech_audioset_epoch_15_esc_89.98.pt",
    )
    parser.add_argument("--min-target-audio-clap-cosine", type=float)
    parser.add_argument("--max-clap-semantic-deficit", type=float)
    parser.add_argument("--min-edit-latent-mae", type=float, default=1.0e-7)
    parser.add_argument("--require-edit-plan-change", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260809)
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
    if (
        args.modality_warmstart_checkpoint is not None
        and not args.modality_warmstart_checkpoint.is_file()
    ):
        raise FileNotFoundError(args.modality_warmstart_checkpoint)
    if args.family_rank < 0 or args.turns < 1 or args.modality_steps < 2:
        raise ValueError("family-rank must be non-negative; turns>=1 and modality-steps>=2")
    if not 0.0 <= args.min_plan_token_accuracy <= 1.0:
        raise ValueError("min-plan-token-accuracy must be in [0,1]")
    if not 0.0 <= args.min_source_component_accuracy <= 1.0:
        raise ValueError("min-source-component-accuracy must be in [0,1]")
    if not 0.0 <= args.min_edit_field_accuracy <= 1.0:
        raise ValueError("min-edit-field-accuracy must be in [0,1]")
    if not 0.0 <= args.min_plan_spatial_valid_fraction <= 1.0:
        raise ValueError("min-plan-spatial-valid-fraction must be in [0,1]")
    if (
        args.max_plan_spatial_excess_deg < 0.0
        or args.max_settled_trailing_rms < 0.0
    ):
        raise ValueError("renderer audio-quality thresholds must be non-negative")
    _seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    caption_overlay_root = args.caption_overlay_root
    if caption_overlay_root is None:
        default_caption_overlay = (
            args.latent_root.resolve().parents[1]
            / "captions"
            / SEMANTIC_CAPTION_TEMPLATE_VERSION
            / args.latent_root.resolve().name
        )
        # Derived curriculum stores already carry immutable, versioned caption
        # metadata in each turn.  Prefer the canonical external overlay when it
        # exists, but do not manufacture a missing path and thereby make those
        # self-contained stores impossible to evaluate.  Explicitly supplied
        # overlays remain fail-closed in ``SpatialFamilyDataset``.
        caption_overlay_root = (
            default_caption_overlay
            if (default_caption_overlay / "READY").is_file()
            else None
        )
    store = {
        "path": str(args.latent_root),
        "custom_metadata_fn": _provider(args.codec_root),
    }
    if caption_overlay_root is not None:
        store["caption_overlay_path"] = str(caption_overlay_root)
    dataset = SpatialFamilyDataset(
        [store],
        require_ready=True,
        max_open_shards=1,
    )
    family_latents, family_info = dataset[args.family_rank]
    turns = list(family_info.get("family_turn_metadata") or [])[: args.turns]
    if len(turns) != args.turns:
        raise RuntimeError(
            f"requested {args.turns} turns but family has {len(turns)}"
        )

    model_config = load_config(args.model_config)
    model = create_model_from_config(model_config)
    checkpoint_state, checkpoint_metadata = load_ckpt_state_dict(
        str(args.checkpoint), return_metadata=True
    )
    warmstart_report = model.load_pretrained_route_state_dict(
        checkpoint_state,
        prefer_ema=args.weights == "ema",
        source_model_config=checkpoint_metadata.get("model_config"),
        source_text_conditioner_ema_names=checkpoint_metadata.get(
            "text_conditioner_ema_parameter_names"
        ),
    )
    del checkpoint_state, checkpoint_metadata
    modality_warmstart_report = None
    if args.modality_warmstart_checkpoint is not None:
        modality_ids = [
            value.strip()
            for value in args.modality_warmstart_ids.split(",")
            if value.strip()
        ]
        if not modality_ids:
            raise ValueError("modality-warmstart-ids must not be empty")
        modality_state, modality_metadata = load_ckpt_state_dict(
            str(args.modality_warmstart_checkpoint), return_metadata=True
        )
        modality_warmstart_report = model.load_pretrained_modality_state_dict(
            modality_state,
            modality_ids=modality_ids,
            prefer_ema=args.modality_warmstart_weights == "ema",
            source_model_config=modality_metadata.get("model_config"),
        )
        del modality_state, modality_metadata

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    model = model.eval().requires_grad_(False).to(device)
    # Architecture probes can add zero-initialized modules whose random input
    # projections consume a different number of RNG draws during construction.
    # Reset here so --seed controls ODE noise and decoding rather than the
    # architecture-dependent initialization cursor.
    _seed_everything(args.seed)
    codec = model.get_plan_codec()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    previous_latent = None
    previous_plan = None
    records = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    if args.teacher_forced_only:
        objectives_by_id = {
            str(objective.get("id")): objective
            for objective in model_config.get("training", {}).get("objectives", [])
            if isinstance(objective, dict)
        }
        planner_objective = objectives_by_id.get("state_planner")
        understanding_objective = objectives_by_id.get("understanding")
        if not isinstance(planner_objective, dict) or not isinstance(
            understanding_objective, dict
        ):
            raise RuntimeError(
                "model config must contain state_planner and understanding objectives"
            )
        teacher_forced = []
        with torch.inference_mode(), autocast:
            for turn_index, turn in enumerate(turns):
                previous_foa = torch.as_tensor(
                    turn["previous_foa"], device=device, dtype=torch.float32
                )
                loss = model.forward_planner(
                    [turn["planner_prompt"]],
                    [turn["spatial_plan_tokens"]],
                    loss_group_weights=planner_objective.get("loss_group_weights"),
                    context_modalities=[{"previous_foa": previous_foa}],
                    prefix_plan_tokens=[turn.get("previous_plan_tokens")],
                )
                understanding_loss = model.forward_planner(
                    [turn["understanding_prompt"]],
                    [turn["spatial_plan_tokens"]],
                    loss_group_weights=understanding_objective.get(
                        "loss_group_weights"
                    ),
                    context_modalities=[
                        {
                            "foa_latent": torch.as_tensor(
                                family_latents[turn_index],
                                device=device,
                                dtype=torch.float32,
                            )
                        }
                    ],
                )
                planner_diagnostic = _teacher_forced_plan_diagnostic(
                    model,
                    caption=turn["planner_prompt"],
                    plan_tokens=turn["spatial_plan_tokens"],
                    context_modalities={"previous_foa": previous_foa},
                    prefix_plan_tokens=turn.get("previous_plan_tokens"),
                )
                understanding_diagnostic = _teacher_forced_plan_diagnostic(
                    model,
                    caption=turn["understanding_prompt"],
                    plan_tokens=turn["spatial_plan_tokens"],
                    context_modalities={
                        "foa_latent": torch.as_tensor(
                            family_latents[turn_index],
                            device=device,
                            dtype=torch.float32,
                        )
                    },
                )
                teacher_forced.append(
                    {
                        "turn": turn_index,
                        "state_planner_ce": float(loss.float().cpu()),
                        "understanding_ce": float(
                            understanding_loss.float().cpu()
                        ),
                        "state_planner": planner_diagnostic,
                        "understanding": understanding_diagnostic,
                    }
                )
        exact = all(
            item["state_planner"]["exact"] and item["understanding"]["exact"]
            for item in teacher_forced
        )
        report = {
            "status": (
                "PASS"
                if exact
                else ("FAIL" if args.require_teacher_forced_exact else "DIAGNOSTIC")
            ),
            "mode": "teacher_forced_only",
            "checkpoint": str(args.checkpoint.resolve()),
            "weights": args.weights,
            "family_id": family_info["family_id"],
            "family_rank": args.family_rank,
            "exact": exact,
            "require_exact": args.require_teacher_forced_exact,
            "turn_results": teacher_forced,
        }
        _atomic_json(args.output_dir / "TEACHER_FORCED_RESULT.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if exact or not args.require_teacher_forced_exact else 1

    pretransform_state = load_ckpt_state_dict(str(args.pretransform_checkpoint))
    model.load_pretransform_state_dict(pretransform_state)
    del pretransform_state
    clap_model = None
    if args.clap_content_metrics:
        from stable_audio_tools.training.metrics.fad_metrics import load_clap_model

        clap_model = load_clap_model(args.clap_model, device=str(device))

    # Auxiliary VAE/CLAP construction may also draw from PyTorch RNG. Sampling
    # starts from the same seed regardless of which diagnostics are enabled.
    _seed_everything(args.seed)
    with torch.inference_mode(), autocast:
        for turn_index, turn in enumerate(turns):
            stem = f"turn_{turn_index:02d}"
            planner_prompt = str(turn["planner_prompt"])
            semantic_caption_text = str(turn["semantic_caption"])
            semantic_caption_metadata = turn.get("semantic_caption_metadata")
            if not isinstance(semantic_caption_metadata, dict):
                raise RuntimeError(
                    "evaluation turn lacks semantic_caption_metadata; "
                    "enable the versioned Spatial-CoT caption overlay"
                )
            semantic_caption = {
                "text": semantic_caption_text,
                "source_regions": semantic_caption_metadata.get("source_regions") or [],
                "transcript_regions": semantic_caption_metadata.get("transcript_regions") or [],
                "caption_template_version": semantic_caption_metadata.get("version"),
                "caption_template_id": semantic_caption_metadata.get("template_id"),
            }
            target_tokens = turn["spatial_plan_tokens"]["input_ids"]
            target_plan = codec.decode(target_tokens)
            target_plan_path = args.output_dir / f"{stem}.target_scene_plan.json"
            _atomic_json(target_plan_path, target_plan)
            understanding_result = None
            if args.check_understanding:
                if int(family_latents.shape[0]) <= turn_index:
                    raise RuntimeError(
                        f"family latent tensor has no turn {turn_index}"
                    )
                understanding_prompt = str(turn["understanding_prompt"])
                target_latent = torch.as_tensor(
                    family_latents[turn_index], device=device, dtype=torch.float32
                )
                understanding_plan = model.generate_plan(
                    understanding_prompt,
                    max_plan_tokens=args.max_plan_tokens,
                    temperature=args.plan_temperature,
                    context_modalities={"foa_latent": target_latent},
                )
                decoded_understanding = codec.decode(understanding_plan)
                understanding_roundtrip = codec.encode(
                    decoded_understanding, max_tokens=args.max_plan_tokens
                )["input_ids"]
                if not torch.equal(
                    understanding_plan.cpu(), understanding_roundtrip.cpu()
                ):
                    diagnostic = {
                        "status": "FAIL",
                        "reason": "understanding_plan_roundtrip",
                        "checkpoint": str(args.checkpoint.resolve()),
                        "weights": args.weights,
                        "family_id": family_info["family_id"],
                        "family_rank": args.family_rank,
                        "turn": turn_index,
                        **_roundtrip_diagnostic(
                            codec, understanding_plan, understanding_roundtrip
                        ),
                    }
                    _atomic_json(args.output_dir / "FAILURE.json", diagnostic)
                    raise RuntimeError(
                        "understanding plan round-trip failed at turn "
                        f"{turn_index}; first_mismatch={diagnostic['first_mismatch']}"
                    )
                understanding_accuracy = _token_accuracy(
                    understanding_plan, target_tokens
                )
                understanding_fields = _plan_field_metrics(
                    decoded_understanding, target_plan
                )
                understanding_path = (
                    args.output_dir / f"{stem}.understanding.scene_plan.json"
                )
                _atomic_json(understanding_path, decoded_understanding)
                understanding_failures = _plan_gate_failures(
                    plan_accuracy=understanding_accuracy,
                    field_metrics=understanding_fields,
                    min_plan_token_accuracy=args.min_plan_token_accuracy,
                    min_source_component_accuracy=args.min_source_component_accuracy,
                    require_exact_plan=args.require_exact_plan,
                )
                understanding_result = {
                    "prompt": understanding_prompt,
                    "predicted_plan_tokens": int(understanding_plan.numel()),
                    "plan_token_accuracy": understanding_accuracy,
                    "field_metrics": understanding_fields,
                    "plan_path": str(understanding_path),
                }
                if understanding_failures:
                    diagnostic = {
                        "status": "FAIL",
                        "reason": "generated_understanding_quality",
                        "checkpoint": str(args.checkpoint.resolve()),
                        "weights": args.weights,
                        "family_id": family_info["family_id"],
                        "family_rank": args.family_rank,
                        "turn": turn_index,
                        "failures": understanding_failures,
                        **understanding_result,
                        "target_plan_path": str(target_plan_path),
                    }
                    _atomic_json(args.output_dir / "FAILURE.json", diagnostic)
                    raise RuntimeError(
                        f"turn {turn_index} understanding ScenePlan gate failed: "
                        + "; ".join(understanding_failures)
                    )
            renderer_previous_latent = _select_renderer_previous_latent(
                args.renderer_context,
                turn_index=turn_index,
                closed_loop_previous=previous_latent,
                family_latents=family_latents,
                device=device,
            )
            provided_modalities = (
                None
                if renderer_previous_latent is None
                else {"previous_foa": renderer_previous_latent}
            )
            if args.render_seed_policy == "same_each_turn":
                _seed_everything(args.seed)
            if args.renderer_plan == "target":
                sample = model.render_plan(
                    target_tokens,
                    provided_modalities=provided_modalities,
                    semantic_caption=semantic_caption,
                    modality_steps=args.modality_steps,
                    cfg_scale=args.cfg_scale,
                )
            else:
                sample = model.generate(
                    planner_prompt,
                    provided_modalities=provided_modalities,
                    previous_plan_tokens=previous_plan,
                    semantic_caption=semantic_caption,
                    modality_steps=args.modality_steps,
                    cfg_scale=args.cfg_scale,
                    plan_temperature=args.plan_temperature,
                    max_plan_tokens=args.max_plan_tokens,
                )
            if len(sample) < 2 or not torch.is_tensor(sample[1]):
                raise RuntimeError("Spatial-CoT sample does not contain generated plan tokens")
            plan = sample[1].detach()
            decoded_plan = codec.decode(plan)
            roundtrip = codec.encode(decoded_plan, max_tokens=args.max_plan_tokens)[
                "input_ids"
            ]
            if not torch.equal(plan.cpu(), roundtrip.cpu()):
                diagnostic = {
                    "status": "FAIL",
                    "reason": "generated_plan_roundtrip",
                    "checkpoint": str(args.checkpoint.resolve()),
                    "weights": args.weights,
                    "family_id": family_info["family_id"],
                    "family_rank": args.family_rank,
                    "turn": turn_index,
                    **_roundtrip_diagnostic(codec, plan, roundtrip),
                }
                _atomic_json(args.output_dir / "FAILURE.json", diagnostic)
                raise RuntimeError(
                    "generated plan round-trip failed at turn "
                    f"{turn_index}; first_mismatch={diagnostic['first_mismatch']} "
                    f"predicted_length={diagnostic['predicted_length']} "
                    f"roundtrip_length={diagnostic['roundtrip_length']}"
                )

            plan_accuracy = _token_accuracy(plan, target_tokens)
            field_metrics = _plan_field_metrics(decoded_plan, target_plan)
            edit_metrics = _edit_field_metrics(
                decoded_plan,
                target_plan,
                turn.get("previous_scene_plan"),
                turn.get("diff"),
            )
            plan_path = args.output_dir / f"{stem}.scene_plan.json"
            _atomic_json(plan_path, decoded_plan)
            gate_failures = _plan_gate_failures(
                plan_accuracy=plan_accuracy,
                field_metrics=field_metrics,
                min_plan_token_accuracy=args.min_plan_token_accuracy,
                min_source_component_accuracy=args.min_source_component_accuracy,
                require_exact_plan=args.require_exact_plan,
            )
            if (
                edit_metrics["applicable"]
                and float(edit_metrics["changed_field_accuracy"])
                < args.min_edit_field_accuracy
            ):
                gate_failures.append(
                    "edit_field_accuracy "
                    f"{float(edit_metrics['changed_field_accuracy']):.6f} < "
                    f"{args.min_edit_field_accuracy:.6f}"
                )
            if gate_failures:
                diagnostic = {
                    "status": "FAIL",
                    "reason": "generated_plan_quality",
                    "checkpoint": str(args.checkpoint.resolve()),
                    "weights": args.weights,
                    "family_id": family_info["family_id"],
                    "family_rank": args.family_rank,
                    "turn": turn_index,
                    "failures": gate_failures,
                    "plan_token_accuracy": plan_accuracy,
                    "field_metrics": field_metrics,
                    "edit_metrics": edit_metrics,
                    "predicted_plan_path": str(plan_path),
                    "target_plan_path": str(target_plan_path),
                }
                _atomic_json(args.output_dir / "FAILURE.json", diagnostic)
                raise RuntimeError(
                    f"turn {turn_index} ScenePlan gate failed: "
                    + "; ".join(gate_failures)
                )
            plan_changed = (
                None
                if previous_plan is None
                else not torch.equal(plan.cpu(), previous_plan.cpu())
            )
            if (
                turn_index > 0
                and args.require_edit_plan_change
                and plan_changed is not True
            ):
                _atomic_json(
                    args.output_dir / "FAILURE.json",
                    {
                        "status": "FAIL",
                        "reason": "edit_plan_unchanged",
                        "checkpoint": str(args.checkpoint.resolve()),
                        "weights": args.weights,
                        "family_id": family_info["family_id"],
                        "family_rank": args.family_rank,
                        "turn": turn_index,
                        "edit_metrics": edit_metrics,
                        "predicted_plan_path": str(plan_path),
                        "target_plan_path": str(target_plan_path),
                    },
                )
                raise RuntimeError(f"edit turn {turn_index} did not change ScenePlan")

            latent = _extract_target_latent(model, sample)
            latent_mae = (
                None
                if renderer_previous_latent is None
                else float(
                    (latent.float() - renderer_previous_latent.float())
                    .abs()
                    .mean()
                    .cpu()
                )
            )
            if (
                latent_mae is not None
                and latent_mae < args.min_edit_latent_mae
            ):
                raise RuntimeError(
                    f"edit turn {turn_index} latent MAE {latent_mae:.8g} "
                    f"is below {args.min_edit_latent_mae:.8g}"
                )

            audio = model.decode_latent(latent.unsqueeze(0))[0].detach().float().cpu()
            if audio.ndim != 2 or audio.shape[0] != 4:
                raise RuntimeError(
                    f"decoded FOA must be [4,N], got {tuple(audio.shape)}"
                )
            if not bool(torch.isfinite(audio).all()):
                raise RuntimeError(f"decoded audio is non-finite at turn {turn_index}")
            audio_rms = float(audio.square().mean().sqrt())
            audio_peak = float(audio.abs().max())
            if not math.isfinite(audio_rms) or audio_rms < args.min_audio_rms:
                raise RuntimeError(
                    f"decoded audio RMS {audio_rms:.8g} failed at turn {turn_index}"
                )

            audio_path = args.output_dir / f"{stem}.wav"
            torchaudio.save(
                str(audio_path), audio.clamp(-1.0, 1.0), model_config["sample_rate"]
            )
            target_latent = torch.as_tensor(
                family_latents[turn_index], device=device, dtype=torch.float32
            )
            latent_alignment = _latent_alignment_metrics(latent, target_latent)
            target_audio = (
                model.decode_latent(target_latent.unsqueeze(0))[0]
                .detach()
                .float()
                .cpu()
            )
            if target_audio.ndim != 2 or target_audio.shape[0] != 4:
                raise RuntimeError(
                    f"decoded target FOA must be [4,N], got {tuple(target_audio.shape)}"
                )
            if not bool(torch.isfinite(target_audio).all()):
                raise RuntimeError(
                    f"decoded target audio is non-finite at turn {turn_index}"
                )
            target_audio_path = args.output_dir / f"{stem}.target.wav"
            torchaudio.save(
                str(target_audio_path),
                target_audio.clamp(-1.0, 1.0),
                model_config["sample_rate"],
            )
            spatial_alignment = _spatial_alignment_metrics(
                audio,
                target_audio,
                hop=int(model.downsampling_ratio),
            )
            plan_spatial_alignment = _plan_spatial_alignment_metrics(
                audio,
                decoded_plan,
                hop=int(model.downsampling_ratio),
            )
            target_plan_spatial_alignment = _plan_spatial_alignment_metrics(
                target_audio,
                target_plan,
                hop=int(model.downsampling_ratio),
            )
            silence_alignment = _silence_alignment_metrics(
                audio,
                target_audio,
                frame_samples=int(model.downsampling_ratio),
            )
            content_alignment = (
                _clap_content_metrics(
                    clap_model,
                    audio,
                    target_audio,
                    semantic_caption_text,
                    sample_rate=int(model_config["sample_rate"]),
                )
                if clap_model is not None
                else None
            )
            record = {
                "turn": turn_index,
                "planner_prompt": planner_prompt,
                "semantic_caption": semantic_caption_text,
                "semantic_caption_template_id": semantic_caption_metadata.get(
                    "template_id"
                ),
                "predicted_plan_tokens": int(plan.numel()),
                "target_plan_tokens": int(torch.as_tensor(target_tokens).numel()),
                "plan_token_accuracy": plan_accuracy,
                "field_metrics": field_metrics,
                "edit_metrics": edit_metrics,
                "renderer_context": args.renderer_context,
                "renderer_previous_foa_present": (
                    renderer_previous_latent is not None
                ),
                "plan_changed_from_previous": plan_changed,
                "edit_latent_mae": latent_mae,
                "latent_alignment": latent_alignment,
                "audio_rms": audio_rms,
                "audio_peak": audio_peak,
                "audio_path": str(audio_path),
                "target_audio_path": str(target_audio_path),
                "spatial_alignment": spatial_alignment,
                "plan_spatial_alignment": plan_spatial_alignment,
                "target_plan_spatial_alignment": target_plan_spatial_alignment,
                "silence_alignment": silence_alignment,
                "content_alignment": content_alignment,
                "plan_path": str(plan_path),
                "target_plan_path": str(target_plan_path),
                "understanding": understanding_result,
            }
            records.append(record)
            audio_gate_failures = _renderer_audio_gate_failures(
                plan_spatial=plan_spatial_alignment,
                target_plan_spatial=target_plan_spatial_alignment,
                silence=silence_alignment,
                max_plan_spatial_excess_deg=args.max_plan_spatial_excess_deg,
                min_plan_spatial_valid_fraction=(
                    args.min_plan_spatial_valid_fraction
                ),
                max_settled_trailing_rms=args.max_settled_trailing_rms,
            )
            if content_alignment is not None:
                if (
                    args.min_target_audio_clap_cosine is not None
                    and content_alignment["generated_target_audio_cosine"]
                    < args.min_target_audio_clap_cosine
                ):
                    audio_gate_failures.append(
                        "generated-target CLAP cosine "
                        f"{content_alignment['generated_target_audio_cosine']:.6f} < "
                        f"{args.min_target_audio_clap_cosine:.6f}"
                    )
                if (
                    args.max_clap_semantic_deficit is not None
                    and content_alignment["semantic_deficit_vs_target"]
                    > args.max_clap_semantic_deficit
                ):
                    audio_gate_failures.append(
                        "CLAP semantic deficit "
                        f"{content_alignment['semantic_deficit_vs_target']:.6f} > "
                        f"{args.max_clap_semantic_deficit:.6f}"
                    )
            if args.audio_quality_gate and audio_gate_failures:
                diagnostic = {
                    "status": "FAIL",
                    "reason": "renderer_audio_quality",
                    "checkpoint": str(args.checkpoint.resolve()),
                    "weights": args.weights,
                    "family_id": family_info["family_id"],
                    "family_rank": args.family_rank,
                    "turn": turn_index,
                    "failures": audio_gate_failures,
                    "turn_result": record,
                }
                _atomic_json(args.output_dir / "FAILURE.json", diagnostic)
                raise RuntimeError(
                    f"turn {turn_index} Renderer audio gate failed: "
                    + "; ".join(audio_gate_failures)
                )
            previous_latent = latent.detach()
            previous_plan = plan.detach()

    report = {
        "evaluator_version": EVALUATOR_VERSION,
        "status": "PASS",
        "checkpoint": str(args.checkpoint.resolve()),
        "weights": args.weights,
        "family_id": family_info["family_id"],
        "family_rank": args.family_rank,
        "closed_loop": args.renderer_context == "closed_loop",
        "warmstart": warmstart_report,
        "modality_warmstart": modality_warmstart_report,
        "settings": {
            "turns": args.turns,
            "modality_steps": args.modality_steps,
            "cfg_scale": args.cfg_scale,
            "renderer_plan": args.renderer_plan,
            "renderer_context": args.renderer_context,
            "render_seed_policy": args.render_seed_policy,
            "plan_temperature": args.plan_temperature,
            "min_plan_token_accuracy": args.min_plan_token_accuracy,
            "min_source_component_accuracy": args.min_source_component_accuracy,
            "min_edit_field_accuracy": args.min_edit_field_accuracy,
            "require_exact_plan": args.require_exact_plan,
            "check_understanding": args.check_understanding,
            "require_edit_plan_change": args.require_edit_plan_change,
            "audio_quality_gate": args.audio_quality_gate,
            "max_plan_spatial_excess_deg": args.max_plan_spatial_excess_deg,
            "min_plan_spatial_valid_fraction": (
                args.min_plan_spatial_valid_fraction
            ),
            "max_settled_trailing_rms": args.max_settled_trailing_rms,
            "clap_content_metrics": args.clap_content_metrics,
            "clap_model": args.clap_model if args.clap_content_metrics else None,
            "min_target_audio_clap_cosine": args.min_target_audio_clap_cosine,
            "max_clap_semantic_deficit": args.max_clap_semantic_deficit,
        },
        "turn_results": records,
    }
    _atomic_json(args.output_dir / "RESULT.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
