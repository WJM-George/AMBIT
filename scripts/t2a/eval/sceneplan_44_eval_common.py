"""Small I/O and listening helpers for ScenePlan 4+4 evaluation."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import soundfile as sf
import torch


def atomic_wav(
    path: Path,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    subtype: str,
) -> None:
    """Write a complete WAV atomically after validating channel geometry."""

    if audio.ndim != 2:
        raise ValueError(f"audio must be [channels,samples], got {tuple(audio.shape)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    sf.write(
        str(temporary),
        audio.detach().to(torch.float32).cpu().transpose(0, 1).numpy(),
        int(sample_rate),
        format="WAV",
        subtype=subtype,
    )
    temporary.replace(path)


def virtual_stereo(foa: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Decode WYZX/ACN/SN3D to a stable +/-30 degree stereo preview."""

    if tuple(foa.shape[:1]) != (4,):
        raise ValueError(f"FOA must be [4,N], got {tuple(foa.shape)}")
    w, y, _z, x = foa.to(torch.float32)
    cos30 = math.cos(math.radians(30.0))
    sin30 = math.sin(math.radians(30.0))
    left = math.sqrt(0.5) * w + 0.5 * cos30 * x + 0.5 * sin30 * y
    right = math.sqrt(0.5) * w + 0.5 * cos30 * x - 0.5 * sin30 * y
    stereo = torch.stack([left, right])
    rms_before = float(stereo.square().mean().sqrt())
    peak_before = float(stereo.abs().max())
    target_rms = 10.0 ** (-18.0 / 20.0)
    gain = (
        1.0
        if rms_before <= 1.0e-12 or peak_before <= 1.0e-12
        else min(target_rms / rms_before, 0.98 / peak_before)
    )
    stereo = stereo * float(gain)
    return stereo, {
        "preview_gain": float(gain),
        "preview_rms_before": rms_before,
        "preview_peak_before": peak_before,
        "preview_peak_after": float(stereo.abs().max()),
    }


def audio_qc(audio: torch.Tensor) -> dict[str, Any]:
    """Return fail-closed finite, geometry, level, and clipping diagnostics."""

    finite = bool(torch.isfinite(audio).all())
    if not finite:
        return {"finite": False}
    channel_rms = audio.to(torch.float32).square().mean(dim=-1).sqrt()
    return {
        "finite": True,
        "channels": int(audio.shape[0]),
        "samples": int(audio.shape[1]),
        "peak": float(audio.abs().max()),
        "rms": float(audio.to(torch.float32).square().mean().sqrt()),
        "channel_rms": [float(value) for value in channel_rms],
        "fraction_abs_ge_1": float(audio.abs().ge(1.0).to(torch.float32).mean()),
    }


__all__ = ["atomic_wav", "audio_qc", "virtual_stereo"]
