"""Reusable frame-energy activity measurements for rendered audio."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


def frame_rms(audio: np.ndarray, frame_samples: int) -> np.ndarray:
    """Channel-joint RMS for non-overlapping frames."""

    if frame_samples <= 0:
        raise ValueError("frame_samples must be positive")
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.ndim != 2 or audio.shape[0] == 0:
        raise ValueError(f"audio must be non-empty [samples, channels], got {audio.shape}")
    count = math.ceil(audio.shape[0] / frame_samples)
    padded = np.pad(audio, ((0, count * frame_samples - audio.shape[0]), (0, 0)))
    frames = padded.reshape(count, frame_samples, audio.shape[1])
    return np.sqrt(
        np.mean(np.square(frames, dtype=np.float64), axis=(1, 2)) + 1e-12
    )


def measure_audio_activity(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 50.0,
    absolute_dbfs: float = -55.0,
    relative_db: float = -40.0,
) -> dict[str, Any]:
    """Measure audible extent using a conservative absolute+relative threshold."""

    if sample_rate <= 0 or frame_ms <= 0:
        raise ValueError("sample_rate and frame_ms must be positive")
    if audio.ndim == 1:
        audio = audio[:, None]
    frame_samples = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    rms = frame_rms(audio, frame_samples)
    peak_rms = float(rms.max())
    absolute = 10.0 ** (absolute_dbfs / 20.0)
    relative = peak_rms * (10.0 ** (relative_db / 20.0))
    threshold = max(absolute, relative)
    active = rms > threshold
    if active.any():
        first = int(np.flatnonzero(active)[0])
        last = int(np.flatnonzero(active)[-1]) + 1
    else:
        first = len(active)
        last = 0
    frame_sec = frame_samples / float(sample_rate)
    duration = audio.shape[0] / float(sample_rate)
    onset = min(duration, first * frame_sec) if active.any() else None
    offset = min(duration, last * frame_sec) if active.any() else None
    return {
        "audio_duration_sec": duration,
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "peak": float(np.max(np.abs(audio))),
        "peak_frame_rms_dbfs": 20.0 * math.log10(max(peak_rms, 1e-12)),
        "activity_threshold_dbfs": 20.0 * math.log10(max(threshold, 1e-12)),
        "activity_onset_sec": onset,
        "activity_offset_sec": offset,
        "active_ratio": float(active.mean()),
        "leading_silence_sec": min(duration, first * frame_sec),
        "trailing_silence_sec": max(0.0, duration - last * frame_sec),
        "all_silent": not bool(active.any()),
        "frame_ms": float(frame_ms),
        "absolute_dbfs": float(absolute_dbfs),
        "relative_db": float(relative_db),
    }


def measure_audio_file(
    path: str | Path,
    *,
    frame_ms: float = 50.0,
    absolute_dbfs: float = -55.0,
    relative_db: float = -40.0,
) -> dict[str, Any]:
    audio, sample_rate = sf.read(
        str(Path(path).expanduser()), always_2d=True, dtype="float32"
    )
    return measure_audio_activity(
        audio,
        int(sample_rate),
        frame_ms=frame_ms,
        absolute_dbfs=absolute_dbfs,
        relative_db=relative_db,
    )


__all__ = ["frame_rms", "measure_audio_activity", "measure_audio_file"]
