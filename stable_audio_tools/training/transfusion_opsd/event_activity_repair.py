"""Freeze observed query-audio windows for a bounded physical teacher proposal."""
from __future__ import annotations

import math

import torch

from ...data.foa_intensity import foa_to_intensity_trajectory
from ...data.sceneplan_generation_ar_natural_constraints import COMPASS_CENTERS, COMPASS_HALF_WIDTH


class FrozenActivityRepairView:
    """Minimal geometry adapter for bounded_horizontal_repair, not a scorer.

    Windows come from the current query's W-energy activity and are held fixed
    while constructing the teacher. Final acceptance always re-measures actual
    audio with the independent request observer and content/ASR protections.
    """

    @torch.no_grad()
    def __init__(self, observer, waveform):
        observed = observer.measure(waveform)
        if observed['costs']['source_presence_failure']:
            raise ValueError('Cannot rotate an absent source into a valid physical teacher.')
        self.num_samples = waveform.shape[-1]
        self.hop, self.sample_rate = observer.hop, observer.sample_rate
        self.energy_floor = observer.config.energy_floor
        self.min_coherence = observer.config.min_coherence
        self.cos_tolerance = math.cos(math.radians(COMPASS_HALF_WIDTH))
        energy = self._energy(waveform)
        active = energy >= self.energy_floor
        indices = active.nonzero().flatten()
        count = observer.config.endpoint_frames
        choices = {'both': active.clone(), 'start': torch.zeros_like(active), 'end': torch.zeros_like(active)}
        choices['start'][indices[:count]] = True
        choices['end'][indices[-count:]] = True
        windows, directions, evidence = [], [], []
        for constraint in observer.compass:
            mask = choices[constraint['point']]
            if int(mask.sum()) < observer.config.minimum_active_frames:
                raise ValueError('Query lacks the declared minimum endpoint activity evidence.')
            angle = math.radians(COMPASS_CENTERS[constraint['value']])
            windows.append(mask.cpu())
            directions.append([math.cos(angle), math.sin(angle), 0.])
            evidence.append(dict(source_id=observer.source['key'], point=constraint['point'], frames=int(mask.sum())))
        self.windows = torch.stack(windows)
        self.directions = torch.tensor(directions, dtype=torch.float32)
        self.evidence = dict(requested_windows=evidence,
            source='Frozen activity in the actual current clean-query audio; no plan/GT activity window.',
            is_acceptance_scorer=False)

    def _audio(self, waveform):
        if waveform.shape != (1, 4, self.num_samples) or not torch.isfinite(waveform).all():
            raise ValueError('Physical repair changed the actual query audio geometry.')
        frames = math.ceil(self.num_samples/self.hop)
        return torch.nn.functional.pad(waveform[0].float(), (0, frames*self.hop-self.num_samples))

    def _energy(self, waveform):
        audio = self._audio(waveform)
        energy = audio[0].reshape(-1, self.hop).square().mean(-1)
        tail = self.num_samples % self.hop
        if tail:
            energy[-1] *= self.hop/tail
        return energy

    def _directions(self, waveform):
        trajectory = foa_to_intensity_trajectory(self._audio(waveform), hop=self.hop)
        unit = trajectory[:, :3]
        unit = unit/unit.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        energy = self._energy(waveform)
        return unit, 1-trajectory[:, 3], energy/(energy+self.energy_floor)
