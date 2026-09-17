"""Request-window FOA geometry credit for native Transfusion decisions.

This is a differentiable training proxy, not a replacement for normal-output
evaluation. Windows and targets must be fixed from a request or a qualified
teacher before inspecting the student prediction. Use one isolated source per
window; the intensity of an overlapping mixture is not a source direction.
The native_prediction_credit operator supplies the finite-candidate surrogate;
execution_decision_derivative can route its gradient only to AR logits. DiT
receives a separate, plan-matched velocity objective.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

import torch


@dataclass(frozen=True)
class SpatialCreditWindow:
    start_sample: int
    end_sample: int
    azimuth_deg: float
    tolerance_deg: float
    source_id: str
    elevation_deg: float | None = None

    def __post_init__(self):
        if (type(self.start_sample) is not int or type(self.end_sample) is not int
                or not 0 <= self.start_sample < self.end_sample or not self.source_id
                or not all(math.isfinite(v) for v in (self.azimuth_deg, self.tolerance_deg))
                or not 0 < self.tolerance_deg < 90
                or (self.elevation_deg is not None and
                    (not math.isfinite(self.elevation_deg) or not -90 <= self.elevation_deg <= 90))):
            raise ValueError('Require a fixed nonempty window and a valid request cone.')


def native_spatial_credit(waveform: torch.Tensor, windows: Sequence[SpatialCreditWindow],
                          *, minimum_energy: float = 1e-7,
                          minimum_coherence: float = .1) -> dict:
    """WYZX input; horizontal-only unless elevation was explicitly requested.

    All fixed windows contribute equally, including silent/incoherent ones.
    No output-dependent window selection or silent-window dropping is allowed.
    Energy and coherence shortfalls are diagnostics/penalties, not a proof of
    content or identity. This proxy does not estimate distance or exact timing.
    """
    if (waveform.ndim != 3 or waveform.shape[:2] != (1, 4)
            or waveform.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(waveform).all() or not windows
            or not math.isfinite(minimum_energy) or minimum_energy <= 0
            or not math.isfinite(minimum_coherence) or not 0 < minimum_coherence <= 1):
        raise ValueError('Require finite FP32/64 FOA, fixed windows and positive observation floors.')
    rows = []
    for window in windows:
        if window.end_sample > waveform.shape[-1]:
            raise ValueError('Fixed window exceeds the actual audio; do not silently truncate it.')
        wave = waveform[0, :, window.start_sample:window.end_sample]
        axes = wave[[3, 1]] if window.elevation_deg is None else wave[[3, 1, 2]]
        energy = wave[0].square().mean()
        intensity = (wave[0, None] * axes).mean(-1)
        # Floors are fixed absolute values, never inferred from a candidate.
        norm = intensity.norm().clamp_min(1e-12)
        unit = intensity / norm
        azimuth = math.radians(window.azimuth_deg)
        if window.elevation_deg is None:
            target = wave.new_tensor([math.cos(azimuth), math.sin(azimuth)])
        else:
            elevation = math.radians(window.elevation_deg)
            target = wave.new_tensor([math.cos(elevation)*math.cos(azimuth),
                                      math.cos(elevation)*math.sin(azimuth), math.sin(elevation)])
        cosine = (unit * target).sum().clamp(-1, 1)
        coherence = intensity.norm() / (energy * axes.square().sum(0).mean()).clamp_min(1e-24).sqrt()
        # Normalize by the largest possible cone violation. In-cone directions
        # have zero angular pressure; no hidden precise coordinate is fitted.
        boundary = math.cos(math.radians(window.tolerance_deg))
        direction = torch.relu(boundary - cosine) / (1 + boundary)
        energy_loss = torch.relu(1 - energy / minimum_energy).square()
        coherence_loss = torch.relu(1 - coherence / minimum_coherence).square()
        loss = direction + energy_loss + coherence_loss
        observable = (energy >= minimum_energy) & (coherence >= minimum_coherence)
        rows.append(dict(loss=loss, direction=direction, energy_loss=energy_loss,
                         coherence_loss=coherence_loss, cosine=cosine, unit=unit,
                         energy=energy, coherence=coherence, observable=observable,
                         source_id=window.source_id))
    return dict(loss=torch.stack([r['loss'] for r in rows]).mean(), windows=rows,
                scope='fixed isolated-source request geometry; no semantics, distance or complete path certification')
