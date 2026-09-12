"""Fixed audio-space distillation objectives; these are not quality certificates.

The caller decodes a student's visited clean prediction through the frozen
native codec with autograd enabled. Teacher and reference audio come from the
same query, plan, decoder and sample geometry, and stay fixed for the fit.
"""
from __future__ import annotations

import math
from numbers import Real

import torch


def relative_decoded_fit_loss(audio, teacher, reference, mask, *,
                              fixed_error_scale, reference_power_floor):
    """Mean per-query error over all channels and valid waveform samples.

    Normalize by detached reference energy, never by the student's energy.
    Both scalar constants must be declared before fitting. The mask describes
    actual valid samples, not model-dependent activity or a selected crop.
    Variable-length plans get separate scalar losses before plan weighting.

    All channels receive equal weight. This preserves a fixed target's channel
    relationships but does not separately certify transcript, semantics or
    directional observability. No inference-time waveform edit is implied.
    """
    for value in (fixed_error_scale, reference_power_floor):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError('Audio fitting requires positive, finite, fixed scalar normalizers.')
    if audio.ndim != 3 or any(size <= 0 for size in audio.shape):
        raise ValueError('Audio fitting requires nonempty [batch, channels, samples] tensors.')
    if teacher.shape != audio.shape or reference.shape != audio.shape:
        raise ValueError('Student, fixed teacher and reference must share the actual audio geometry.')
    if any(not value.is_floating_point() or value.device != audio.device
           or not torch.isfinite(value).all() for value in (audio, teacher, reference)):
        raise ValueError('Audio fitting requires finite floating tensors on the same device.')
    if mask.shape != (audio.shape[0], audio.shape[-1]) or mask.dtype != torch.bool or mask.device != audio.device:
        raise ValueError('Valid waveform samples must be a matching boolean mask.')
    if not mask.any(-1).all():
        raise ValueError('Every query needs nonempty valid waveform support.')

    dtype = torch.float64 if any(value.dtype == torch.float64 for value in (audio, teacher, reference)) else torch.float32
    prediction = audio.to(dtype)
    target = teacher.detach().to(dtype)
    anchor = reference.detach().to(dtype)
    weights = mask[:, None].to(dtype)
    reference_energy = (anchor.square() * weights).sum((1, 2))
    valid_elements = weights.sum((1, 2)) * audio.shape[1]
    denominator = reference_energy.clamp_min(valid_elements * reference_power_floor)
    error = ((prediction - target).square() * weights).sum((1, 2)) / denominator
    return error.mean() / fixed_error_scale
