#!/usr/bin/env python3
"""Single-pass Pyroomacoustics renderer primitives for ScenePlan v2."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTHESIS_ROOT = REPO_ROOT / "dataset/synthesis"
for value in (REPO_ROOT, SYNTHESIS_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from render_spatial_edit_families import _align_rirs_to_direct_arrival  # noqa: E402
from synthesize_foa_pyroom import az_el_dist_to_xyz, room_rirs  # noqa: E402


FOA_LAYOUT = "WYZX_ACN_SN3D"
RESIDUAL_ALGORITHMIC_DELAY_SAMPLES = 40
DIRECT_ARRIVAL_PEAK_MIN = RESIDUAL_ALGORITHMIC_DELAY_SAMPLES - 1
DIRECT_ARRIVAL_PEAK_MAX = RESIDUAL_ALGORITHMIC_DELAY_SAMPLES + 1


def position_xyz(position: dict[str, Any], room: dict[str, Any]) -> tuple[float, float, float]:
    azimuth = math.radians(float(position["azimuth_deg"]))
    elevation = math.radians(float(position["elevation_deg"]))
    distance = float(position["distance_m"])
    if distance <= 0:
        raise ValueError("source distance must be positive")
    xyz = az_el_dist_to_xyz(
        tuple(map(float, room["microphone_xyz_m"])),
        azimuth,
        elevation,
        distance,
    )
    dimensions = tuple(map(float, room["dimensions_m"]))
    # room_rirs historically clamps invalid positions.  Silent clamping would
    # make the ScenePlan trajectory differ from the rendered FOA, so v2 rejects
    # such a plan instead and requires the planner to sample a valid interior
    # point explicitly.
    margin = 0.3
    if any(value < margin or value > bound - margin for value, bound in zip(xyz, dimensions)):
        raise ValueError(f"planned source position lies outside Pyroom margin: {xyz}")
    return xyz


def _rir_direction_diagnostic(
    rirs: list[np.ndarray],
    source_xyz: tuple[float, float, float],
    microphone_xyz: tuple[float, float, float],
) -> dict[str, Any]:
    # `_align_rirs_to_direct_arrival` deliberately places the fractional-delay
    # direct response at sample 40 (with its peak allowed at 39--41).  Search
    # that contract window, not the first 256 samples: in a reflective room a
    # later reflection can legitimately be slightly stronger than the direct
    # response and must not be mistaken for the direct-arrival direction.
    if len(rirs[0]) <= DIRECT_ARRIVAL_PEAK_MAX:
        raise RuntimeError("FOA RIR is shorter than the direct-arrival window")
    direct_window = np.abs(
        rirs[0][DIRECT_ARRIVAL_PEAK_MIN : DIRECT_ARRIVAL_PEAK_MAX + 1]
    )
    peak_index = DIRECT_ARRIVAL_PEAK_MIN + int(np.argmax(direct_window))
    w = float(rirs[0][peak_index])
    if abs(w) < 1e-12:
        raise RuntimeError("FOA W direct response is zero")
    measured = np.asarray(
        [rirs[1][peak_index] / w, rirs[2][peak_index] / w, rirs[3][peak_index] / w],
        dtype=np.float64,
    )
    delta = np.asarray(source_xyz, dtype=np.float64) - np.asarray(
        microphone_xyz, dtype=np.float64
    )
    target = delta / np.linalg.norm(delta)
    # Cartesian target is X/Y/Z; ACN channel ratios are Y/Z/X.
    target_yzx = target[[1, 2, 0]]
    return {
        "residual_direct_peak_sample": peak_index,
        "measured_yzx_over_w": measured.tolist(),
        "target_yzx_over_w": target_yzx.tolist(),
        "max_abs_direction_ratio_error": float(np.max(np.abs(measured - target_yzx))),
    }


def _render_rir_banks(
    signal: np.ndarray,
    banks: list[list[np.ndarray]],
    num_samples: int,
) -> list[np.ndarray]:
    max_rir = max(len(impulse) for bank in banks for impulse in bank)
    fft_size = next_fast_len(len(signal) + max_rir - 1)
    signal_fft = rfft(signal, n=fft_size, workers=1)
    rendered: list[np.ndarray] = []
    for rirs in banks:
        padded = np.zeros((4, max_rir), dtype=np.float32)
        for channel, impulse in enumerate(rirs):
            padded[channel, : len(impulse)] = impulse
        response = irfft(
            rfft(padded, n=fft_size, axis=-1, workers=1) * signal_fft[None, :],
            n=fft_size,
            axis=-1,
            workers=1,
        )
        rendered.append(response[:, :num_samples].astype(np.float32, copy=False))
    return rendered


def _trajectory_weights(
    keyframes: list[dict[str, Any]],
    num_samples: int,
    sample_rate: int,
) -> np.ndarray:
    if len(keyframes) == 1:
        return np.ones((1, num_samples), dtype=np.float32)
    times = np.asarray([float(item["time_sec"]) for item in keyframes], dtype=np.float64)
    if np.any(np.diff(times) <= 0):
        raise ValueError("trajectory keyframe times must be strictly increasing")
    seconds = np.arange(num_samples, dtype=np.float64) / sample_rate
    weights = np.zeros((len(keyframes), num_samples), dtype=np.float32)
    weights[0, seconds <= times[0]] = 1.0
    weights[-1, seconds >= times[-1]] = 1.0
    for index in range(len(keyframes) - 1):
        left = times[index]
        right = times[index + 1]
        mask = (seconds >= left) & (seconds <= right)
        fraction = ((seconds[mask] - left) / (right - left)).astype(np.float32)
        weights[index, mask] = 1.0 - fraction
        weights[index + 1, mask] = fraction
    error = float(np.max(np.abs(np.sum(weights, axis=0) - 1.0)))
    if error > 1e-5:
        raise RuntimeError(f"trajectory weights do not sum to one: {error}")
    return weights


def render_complete_mono_source(
    mono: np.ndarray,
    *,
    sample_rate: int,
    scene_num_samples: int,
    onset_sample: int,
    room: dict[str, Any],
    keyframes: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render one complete mono source exactly once into a shared FOA scene."""

    value = np.asarray(mono, dtype=np.float32)
    if value.ndim != 1 or not len(value):
        raise ValueError(f"renderer accepts non-empty mono only, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("source contains non-finite samples")
    onset = int(onset_sample)
    if onset < 0 or onset + len(value) > int(scene_num_samples):
        raise ValueError(
            "complete source does not fit scene activity window: "
            f"{onset}+{len(value)}>{scene_num_samples}"
        )
    signal = np.zeros(int(scene_num_samples), dtype=np.float32)
    signal[onset : onset + len(value)] = value
    microphone = tuple(map(float, room["microphone_xyz_m"]))
    room_dim = tuple(map(float, room["dimensions_m"]))
    banks: list[list[np.ndarray]] = []
    diagnostics: list[dict[str, Any]] = []
    for keyframe in keyframes:
        xyz = position_xyz(keyframe["position"], room)
        raw = room_rirs(
            room_dim,
            float(room["rt60_sec"]),
            int(room["max_order"]),
            microphone,
            xyz,
            int(sample_rate),
        )
        aligned = _align_rirs_to_direct_arrival(
            raw,
            source_xyz=xyz,
            microphone_xyz=microphone,
            sample_rate=int(sample_rate),
        )
        diagnostic = _rir_direction_diagnostic(aligned, xyz, microphone)
        if not 39 <= int(diagnostic["residual_direct_peak_sample"]) <= 41:
            raise RuntimeError(f"unexpected residual Pyroom delay: {diagnostic}")
        # Reflections can already contribute a very small amount at the
        # fractional-delay peak in non-anechoic rooms.  A 0.5% ratio bound is
        # still far below an audible directional error while avoiding a false
        # failure caused by that physically valid overlap.
        if float(diagnostic["max_abs_direction_ratio_error"]) > 5e-3:
            raise RuntimeError(f"FOA normalization/direction mismatch: {diagnostic}")
        diagnostics.append(diagnostic)
        banks.append(aligned)
    rendered = _render_rir_banks(signal, banks, int(scene_num_samples))
    weights = _trajectory_weights(keyframes, int(scene_num_samples), int(sample_rate))
    output = np.zeros((4, int(scene_num_samples)), dtype=np.float32)
    for index, response in enumerate(rendered):
        output += response * weights[index][None, :]
    if not np.isfinite(output).all():
        raise RuntimeError("renderer produced non-finite FOA")
    return output, {
        "spatialization_passes": 1,
        "foa_layout": FOA_LAYOUT,
        "residual_algorithmic_delay_samples": RESIDUAL_ALGORITHMIC_DELAY_SAMPLES,
        "keyframe_rir_diagnostics": diagnostics,
        "trajectory_weight_sum_max_abs_error": float(
            np.max(np.abs(np.sum(weights, axis=0) - 1.0))
        ),
    }


def active_rms(audio: np.ndarray) -> float:
    value = np.asarray(audio, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(value)))) if value.size else 0.0


def true_peak(audio: np.ndarray, oversample: int = 4) -> float:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim == 1:
        value = value[None, :]
    oversampled = resample_poly(value, oversample, 1, axis=-1)
    return float(np.max(np.abs(oversampled))) if oversampled.size else 0.0


def peak_safe_master_gain(
    audio: np.ndarray,
    *,
    target_w_rms_dbfs: float = -23.0,
    true_peak_ceiling_dbfs: float = -1.0,
) -> tuple[float, dict[str, float]]:
    raw_w_rms = active_rms(np.asarray(audio)[0])
    if raw_w_rms < 1e-10:
        raise ValueError("rendered W channel is silent")
    rms_gain = 10.0 ** (target_w_rms_dbfs / 20.0) / raw_w_rms
    raw_true_peak = true_peak(audio)
    ceiling = 10.0 ** (true_peak_ceiling_dbfs / 20.0)
    peak_gain = ceiling / raw_true_peak if raw_true_peak > 0 else 1.0
    gain = min(rms_gain, peak_gain)
    return float(gain), {
        "raw_w_rms": raw_w_rms,
        "raw_true_peak": raw_true_peak,
        "rms_target_gain": float(rms_gain),
        "true_peak_ceiling_gain": float(peak_gain),
    }
