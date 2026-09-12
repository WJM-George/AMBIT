"""Experimental training-only decoded-audio supervision for Editing DiT.

This module is deliberately not registered in the production training factory.
It accepts audio supervision separately from model conditioning and owns no VAE
parameters. A successful backward pass is not evidence of edit quality.
"""

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


SPATIAL_OPERATIONS = frozenset(
    {"linear_to_static", "static_to_linear", "stationary_spatial_relocation"}
)


@dataclass(frozen=True)
class EditingAudioAuxiliaryConfig:
    max_timestep: float = 0.3
    max_rows: int = 1
    raw_foa_weight: float = 1.0
    w_spectral_weight: float = 1.0
    spatial_weight: float = 0.25
    spatial_edit_weight: float = 0.25

    def __post_init__(self):
        if not 0 < self.max_timestep <= 1 or self.max_rows < 1:
            raise ValueError("invalid auxiliary timestep or row budget")
        weights = (
            self.raw_foa_weight, self.w_spectral_weight,
            self.spatial_weight, self.spatial_edit_weight,
        )
        if not all(math.isfinite(x) and x >= 0 for x in weights) or not any(weights):
            raise ValueError("auxiliary weights must be finite, nonnegative and nonzero")


@dataclass
class EditingAudioAuxiliaryResult:
    loss: Tensor
    selected_rows: tuple[int, ...]
    metrics: dict[str, float]


def select_audio_auxiliary_rows(
    timesteps: Tensor,
    operations: Sequence[str],
    config: EditingAudioAuxiliaryConfig,
    *,
    selection_offset: int = 0,
) -> tuple[int, ...]:
    """Deterministically rotate through eligible rows without drawing RNG."""
    if timesteps.ndim != 1 or len(operations) != timesteps.numel():
        raise ValueError("one timestep and operation required per row")
    cpu_times = timesteps.detach().cpu()
    times = cpu_times.tolist()
    if not all(math.isfinite(t) and 0 <= t <= 1 for t in times):
        raise ValueError("timesteps must be finite and in [0,1]")
    # Compare in the tensor's dtype so float32(0.3) is included at a 0.3
    # boundary instead of being excluded after conversion to Python float.
    in_window = ((cpu_times > 0) & (cpu_times <= config.max_timestep)).tolist()
    eligible = [i for i, op in enumerate(operations) if op in SPATIAL_OPERATIONS and in_window[i]]
    if not eligible:
        return ()
    start = int(selection_offset) % len(eligible)
    rotated = eligible[start:] + eligible[:start]
    return tuple(rotated[:config.max_rows])


def _smooth_time(value: Tensor) -> Tensor:
    if value.is_complex():
        parts = torch.view_as_real(value).movedim(-1, -2)
        smoothed = _smooth_time(parts).movedim(-2, -1).contiguous()
        return torch.view_as_complex(smoothed)
    shape = value.shape
    return F.avg_pool1d(
        value.reshape(-1, 1, shape[-1]), 5, stride=1, padding=2,
        count_include_pad=False,
    ).reshape(shape)


def _audio_features(audio: Tensor):
    i, j = torch.triu_indices(4, 4, device=audio.device)
    multiplicity = torch.where(i == j, 1.0, 2.0)
    features = []
    for fft in (2048, 512):
        spectrum = torch.stft(
            audio, n_fft=fft, hop_length=fft // 4,
            window=torch.hann_window(fft, device=audio.device),
            center=False, return_complex=True,
        )
        covariance = _smooth_time(spectrum[i] * spectrum[j].conj())
        trace = covariance[i == j].real.sum(dim=0).clamp_min(0)
        features.append((covariance, trace, multiplicity, spectrum[0].abs()))
    return features


def _audio_distances(predicted, target, source):
    spectral, spatial, spatial_edit = [], [], []
    for p, r, s in zip(predicted, target, source):
        pc, pt, mult, pm = p
        rc, rt, _, rm = r
        sc, st, _, _ = s
        spectral.append(
            (pm - rm).norm() / rm.norm().clamp_min(1e-8)
            + (pm.clamp_min(1e-7).log() - rm.clamp_min(1e-7).log()).abs().mean()
        )
        if float(rt.sum()) <= 1e-12:
            raise ValueError("silent target cannot supervise spatial structure")
        floor = (rt.mean() * 1e-6).clamp_min(1e-12)
        pn, rn, sn = pc / pt.clamp_min(floor), rc / rt.clamp_min(floor), sc / st.clamp_min(floor)
        error = ((pn - rn).abs().square() * mult[:, None, None]).sum(dim=0)
        norm = (rn.abs().square() * mult[:, None, None]).sum(dim=0)
        weight = rt / rt.sum()
        spatial.append((weight * error).sum() / (weight * norm).sum().clamp_min(1e-12))
        change = ((sn - rn).abs().square() * mult[:, None, None]).sum(dim=0).sqrt()
        edit_weight = (weight * change).detach()
        if float(edit_weight.sum()) > 1e-12:
            spatial_edit.append(
                (edit_weight * error).sum()
                / (edit_weight * norm).sum().clamp_min(1e-12)
            )
    return (
        torch.stack(spectral).mean(), torch.stack(spatial).mean(),
        torch.stack(spatial_edit).mean() if spatial_edit else None,
        len(spatial_edit),
    )


class EditingDecodedAudioAuxiliary(nn.Module):
    """Low-noise RF clean-estimate supervision through an external frozen VAE.

    ``source_audio`` and ``target_audio`` contain only the selected rows. They
    are offline supervision, detached here, and never enter DiT conditioning.
    Audio is 44.1 kHz WYZX FOA; latents have 64 channels and a 1024-sample hop.
    ``loss`` is averaged over selected rows. The outer RF trainer must choose
    its auxiliary coefficient explicitly after gradient/memory calibration.
    """

    def __init__(self, config: EditingAudioAuxiliaryConfig | None = None):
        super().__init__()
        self.config = config or EditingAudioAuxiliaryConfig()

    def forward(
        self,
        noised_latents: Tensor,
        velocity: Tensor,
        timesteps: Tensor,
        *,
        decoder: nn.Module,
        operations: Sequence[str],
        model_num_samples: Sequence[int],
        valid_latent_frames: Sequence[int],
        source_audio: Mapping[int, Tensor],
        target_audio: Mapping[int, Tensor],
        selection_offset: int = 0,
    ) -> EditingAudioAuxiliaryResult:
        if noised_latents.ndim != 3 or noised_latents.shape[1] != 64:
            raise ValueError("latents must have shape [B,64,T]")
        if velocity.shape != noised_latents.shape or velocity.device != noised_latents.device:
            raise ValueError("RF velocity and noised latents must be aligned")
        batch = velocity.shape[0]
        if any(len(x) != batch for x in (operations, model_num_samples, valid_latent_frames)):
            raise ValueError("audio geometry and operations must cover the batch")
        selected = select_audio_auxiliary_rows(
            timesteps, operations, self.config, selection_offset=selection_offset,
        )
        if not selected:
            return EditingAudioAuxiliaryResult(velocity.sum() * 0, (), {"selected_rows": 0.0})
        if any(m.training for m in decoder.modules()) or any(p.requires_grad for p in decoder.parameters()):
            raise ValueError("auxiliary requires an external frozen eval-mode VAE")
        losses, per_row = [], []
        with torch.autocast(device_type=velocity.device.type, enabled=False):
            for index in selected:
                n, frames = int(model_num_samples[index]), int(valid_latent_frames[index])
                if n < 2048 or frames != math.ceil(n / 1024) or frames > velocity.shape[-1]:
                    raise ValueError("audio/latent geometry is not valid 44.1kHz FOA")
                references = []
                for collection in (source_audio, target_audio):
                    value = collection[index]
                    if value.ndim != 2 or value.shape[0] != 4 or value.shape[1] < n:
                        raise ValueError("selected supervision must contain [4,valid_samples] FOA")
                    value = value[:, :n].detach().to(device=velocity.device, dtype=torch.float32)
                    if not bool(torch.isfinite(value).all()):
                        raise ValueError("non-finite audio supervision")
                    references.append(value)
                source, target = references
                with torch.no_grad():
                    source_features, target_features = _audio_features(source), _audio_features(target)
                mask = torch.arange(velocity.shape[-1], device=velocity.device) < frames
                estimate = (
                    noised_latents[index:index + 1].detach().float()
                    - timesteps[index].detach().to(velocity.device).float()
                    * velocity[index:index + 1].float()
                ) * mask[None, None]
                waveform = decoder.decode(estimate).float()
                if waveform.ndim != 3 or waveform.shape[:2] != (1, 4) or waveform.shape[-1] < n:
                    raise ValueError("VAE decoder returned incompatible FOA audio")
                waveform = waveform[0, :, :n]
                if not bool(torch.isfinite(waveform).all()):
                    raise ValueError("non-finite decoded clean estimate")
                raw = (waveform - target).square().sum() / target.square().sum().clamp_min(1e-12)
                spec, spatial, edit, edit_resolutions = _audio_distances(
                    _audio_features(waveform), target_features, source_features,
                )
                edit_term = edit if edit is not None else spatial * 0
                loss = (
                    self.config.raw_foa_weight * raw + self.config.w_spectral_weight * spec
                    + self.config.spatial_weight * spatial + self.config.spatial_edit_weight * edit_term
                )
                losses.append(loss)
                per_row.append({
                    "raw_foa_nmse": float(raw.detach()), "w_mrstft": float(spec.detach()),
                    "scm_full": float(spatial.detach()), "scm_edit": float(edit_term.detach()),
                    "edit_scm_available_resolutions": float(edit_resolutions),
                    "timestep": float(timesteps[index].detach()),
                })
        metrics = {key: sum(row[key] for row in per_row) / len(per_row) for key in per_row[0]}
        metrics["selected_rows"] = float(len(selected))
        return EditingAudioAuxiliaryResult(torch.stack(losses).mean(), selected, metrics)
