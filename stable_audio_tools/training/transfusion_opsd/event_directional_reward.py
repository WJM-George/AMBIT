"""Request-angle reward with a fixed reference for observation reliability.

The shared FOA feature is a coherence-damped direction, not a unit vector.
Using it directly as a cosine rewards increased coherence as well as angle.
This explicit experimental profile separates the two. Existing reward versions
remain unchanged; old results must not be relabeled with the new metric.
"""
from __future__ import annotations

import math

import torch

from .event_rewards import EventRequestFoaRewardV2
from .rewards import UnobservableAudio


class FixedReferenceDirectionalReward(EventRequestFoaRewardV2):
    """Fixed baseline confidence weights; live audibility and direction checks.

    Compare every candidate for a request/noise against the same raw baseline
    reference. Do not re-estimate favorable frame weights per candidate. A
    reliable direction is required, but extra coherence earns no angle bonus.
    This is not a calibrated room-acoustics or perceptual-quality verifier.
    """
    profile = 'event_fixed_reference_unit_direction_v3_experimental'

    def __init__(self, request_reward, plan, *, reference_audio, min_coherence=.1, **kwargs):
        if not math.isfinite(min_coherence) or not 0 < min_coherence < 1:
            raise ValueError('declare a finite observation reliability floor')
        super().__init__(request_reward, plan, **kwargs)
        self.min_coherence = min_coherence
        with torch.no_grad():
            _, coherence, level = self._directions(reference_audio)
            observable = (coherence >= min_coherence) & (level >= .5)
            windows = self.windows.to(reference_audio.device)
            if ((windows & observable).sum(-1) < 3).any():
                raise UnobservableAudio('reference lacks three reliable direction frames in a requested window')
            weights = windows * observable * coherence * level
            self.reference_weights = (weights / weights.sum(-1, keepdim=True)).detach().cpu()
        self.evidence = {'requested_windows': self.evidence,
            'profile': self.profile, 'min_coherence': min_coherence,
            'weights': 'normalized baseline coherence*audibility on observable requested frames; fixed across candidates',
            'coherence_is_not_an_angle_bonus': True,
            'reference_uses_model_output_not_hidden_gt': True}

    def _directions(self, waveform):
        from ...data.foa_intensity import foa_to_intensity_trajectory
        if waveform.shape != (1, 4, self.num_samples) or not torch.isfinite(waveform).all():
            raise ValueError('direction reward needs finite exact-length WYZX FOA')
        frames = len(self.active)
        audio = torch.nn.functional.pad(waveform[0].float(), (0, frames * 1024 - self.num_samples))
        trajectory = foa_to_intensity_trajectory(audio)
        damped, coherence = trajectory[:, :3], 1. - trajectory[:, 3]
        unit = damped / damped.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        energy = audio[0].view(frames, 1024).square().mean(-1)
        return unit, coherence, energy / (energy + self.energy_floor)

    def _measure(self, waveform):
        _, costs = super()._measure(waveform)
        unit, coherence, level = self._directions(waveform)
        observable = (coherence >= self.min_coherence) & (level >= .5)
        cosine = (self.directions.to(waveform.device) @ unit.T).clamp(-1, 1)
        weights = self.reference_weights.to(waveform.device)
        utility = (weights * ((cosine + 1.) / 2) * observable).sum(-1).mean()
        # Keep a fixed denominator: losing difficult directions cannot improve
        # the measured success rate by removing those frames from evaluation.
        windows = self.windows.to(waveform.device)
        failures = (cosine < self.cos_tolerance) | ~observable
        costs['requested_sector_failure'] = ((windows & failures).sum(-1).float() / windows.sum(-1)).mean()
        costs['direction_unobservable_fraction'] = ((windows & ~observable).sum(-1).float() / windows.sum(-1)).mean()
        return utility, costs

    def soft_sector_cost(self, waveform):
        """No angle penalty inside the requested sector; differentiable outside.

        Hard audibility, transcript and coverage checks remain separate. This
        guides a local semantic repair and is not a replacement for them.
        """
        unit, _, _ = self._directions(waveform)
        cosine = (self.directions.to(waveform.device) @ unit.T).clamp(-1, 1)
        deficit = (self.cos_tolerance - cosine).clamp_min(0.)
        return (self.reference_weights.to(waveform.device) * deficit).sum(-1).mean()

    @torch.no_grad()
    def diagnostics(self, waveform):
        legacy_utility, legacy_costs = super()._measure(waveform)
        unit, coherence, _ = self._directions(waveform)
        cosine = (self.directions.to(waveform.device) @ unit.T).clamp(-1, 1)
        windows = self.windows.to(waveform.device)
        return {'legacy_coherence_weighted_utility': float(legacy_utility),
            'legacy_sector_failure': float(legacy_costs['requested_sector_failure']),
            'mean_coherence': float(coherence[self.active.to(waveform.device)].mean()),
            'mean_angle_error_deg': float(((torch.rad2deg(torch.acos(cosine)) * windows).sum(-1) / windows.sum(-1)).mean()),
            'room_acoustics_preservation_verified': False}
