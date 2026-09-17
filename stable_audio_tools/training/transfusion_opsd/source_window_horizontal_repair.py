"""Adapt bounded physical FOA repair to one source's fixed isolated windows.

The desired directions come from the active native plan. The caller supplies
independently fixed, source-isolated windows and must validate calibration,
VAE transport, and actual continuation. This is a teacher proposal, not an
audio separator or a guarantee that a mixture contains only the named source.
"""
from __future__ import annotations

import math

import torch

from .foa_horizontal_repair import HorizontalRepairConfig, bounded_horizontal_repair


def _position_vector(position):
    azimuth = math.radians(position['azimuth_deg'])
    elevation = math.radians(position['elevation_deg'])
    distance = position['distance_m']
    return [distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation)]


def native_azimuth_at(source, seconds):
    trajectory = source['trajectory']
    if trajectory['type'] == 'static':
        return float(trajectory['position']['azimuth_deg'])
    if trajectory['type'] != 'linear':
        raise ValueError('Only native static and linear source trajectories are supported.')
    activity = source['activity']
    u = (seconds - activity['onset_sec']) / (activity['offset_sec'] - activity['onset_sec'])
    u = max(0., min(1., u))
    start, end = _position_vector(trajectory['start']), _position_vector(trajectory['end'])
    x, y, _ = [(1-u)*a+u*b for a, b in zip(start, end)]
    if math.hypot(x, y) < 1e-8:
        raise ValueError('Horizontal direction is undefined at a trajectory crossing.')
    return math.degrees(math.atan2(y, x))


class FixedSourceRepairView:
    @torch.no_grad()
    def __init__(self, waveform, source, windows, *, tolerance_deg=30.,
                 minimum_energy=1e-7, minimum_coherence=.1):
        if waveform.shape[:2] != (1, 4) or not torch.isfinite(waveform).all():
            raise ValueError('Expected one finite WYZX waveform.')
        if not windows or any(w.source_id != source['source_id'] for w in windows):
            raise ValueError('All fixed windows must belong to the same identified source.')
        self.num_samples = waveform.shape[-1]
        self.hop = 1024
        self.minimum_energy = minimum_energy
        self.min_coherence = minimum_coherence
        self.cos_tolerance = math.cos(math.radians(tolerance_deg))
        frames = math.ceil(self.num_samples / self.hop)
        covered = torch.zeros(frames, dtype=torch.bool)
        self.sample_support = torch.zeros(self.num_samples, dtype=torch.bool, device=waveform.device)
        for w in windows:
            if not 0 <= w.start_sample < w.end_sample <= self.num_samples:
                raise ValueError('A repair window lies outside the waveform.')
            # A complete analysis frame and its interpolation support must lie
            # within the declared interval; no rotation of an overlapping source.
            first = math.ceil(w.start_sample / self.hop)
            stop = math.floor(w.end_sample / self.hop)
            # One fixed zero-angle frame at either edge keeps interpolation
            # inside the source window before the explicit sample protection.
            covered[first+1:max(first+1, stop-1)] = True
            self.sample_support[w.start_sample:w.end_sample] = True
        indices = covered.nonzero().flatten()
        if not len(indices):
            raise ValueError('No complete FOA analysis frame inside the fixed windows.')
        self.windows = torch.zeros((len(indices), frames), dtype=torch.bool)
        self.windows[torch.arange(len(indices)), indices] = True
        angles = [native_azimuth_at(source, (int(i)*self.hop+511.5)/44100) for i in indices]
        self.directions = torch.tensor([[math.cos(math.radians(a)), math.sin(math.radians(a)), 0.] for a in angles])
        self.evidence = dict(requested_windows=[dict(source_id=source['source_id'], frame=int(i)) for i in indices],
            direction_authority='Active native plan trajectory at each analysis frame, not hidden target coordinates.')

    def _directions(self, waveform):
        from ...data.foa_intensity import foa_to_intensity_trajectory
        padded = torch.nn.functional.pad(waveform[0].float(), (0, (-self.num_samples) % self.hop))
        trajectory = foa_to_intensity_trajectory(padded, hop=self.hop)
        unit = trajectory[:, :3]
        unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        energy = padded[0].reshape(-1, self.hop).square().mean(-1)
        return unit, 1-trajectory[:, 3], energy/(energy+self.minimum_energy)


@torch.no_grad()
def repair_source_windows(waveform, source, windows, *, config=HorizontalRepairConfig(), tolerance_deg=30.):
    reference = FixedSourceRepairView(waveform, source, windows, tolerance_deg=tolerance_deg)
    repaired, evidence = bounded_horizontal_repair(reference, waveform, config=config)
    # The bounded solver interpolates frame angles. Explicitly keep all samples
    # outside the independently fixed source support bitwise unchanged.
    repaired[..., ~reference.sample_support] = waveform[..., ~reference.sample_support]
    outside_exact = torch.equal(repaired[..., ~reference.sample_support], waveform[..., ~reference.sample_support])
    assert outside_exact and torch.equal(repaired[:, 0], waveform[:, 0]) and torch.equal(repaired[:, 2], waveform[:, 2])
    evidence.update(outside_fixed_windows_exact=outside_exact, fixed_support_samples=int(reference.sample_support.sum()),
        native_source_id=source['source_id'], native_trajectory=source['trajectory'],
        direction_authority=reference.evidence['direction_authority'],
        no_op_exact=torch.equal(repaired, waveform),
        scope='Bounded horizontal rotation inside one source support. W and Z are unchanged before VAE; other physical, content, source identity, continuation and student properties require actual validation.')
    return repaired, evidence
