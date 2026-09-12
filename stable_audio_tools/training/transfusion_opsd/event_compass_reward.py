"""Experimental horizontal request supervision; historical 3D rewards stay intact."""
from __future__ import annotations

import torch

from .event_directional_reward import FixedReferenceDirectionalReward


class CompassOnlyDirectionalReward(FixedReferenceDirectionalReward):
    """Use the azimuth predicate of a compass request, without inventing elevation.

    This standalone profile currently covers one source with compass, motion,
    time and content constraints. Additional spatial requirements need their
    own audio verifier; reject them instead of silently omitting them. Native
    plan admissibility and external content/ASR costs remain mandatory in the
    caller. This class is not a full semantic or motion-path certificate.
    """

    profile = 'event_single_source_horizontal_compass_v1_experimental'
    horizontal_projection_epsilon = 1e-6

    def __init__(self, request_reward, plan, **kwargs):
        requirements = request_reward.requirements
        if len(requirements['sources']) != 1 or requirements.get('relations'):
            raise ValueError('compass-only profile requires one source and no relational audio requirements')
        allowed = {'compass', 'motion', 'time_phase', 'full_scene', 'ends_scene', 'starts_scene', 'transcript'}
        time_fields = {'onset_sec', 'offset_sec', 'event_duration_sec'}
        for constraint in requirements['sources'][0]['constraints']:
            if constraint['op'] == 'numeric' and constraint.get('field') in time_fields:
                continue
            if constraint['op'] not in allowed:
                raise ValueError('compass-only profile cannot silently drop additional spatial requirements')
        super().__init__(request_reward, plan, **kwargs)
        self.evidence.update(
            direction_scope='Horizontal azimuth, matching the natural-request compass predicate. Unspecified elevation is free.',
            horizontal_projection_epsilon=self.horizontal_projection_epsilon,
            vertical_or_undefined_azimuth='Unobservable, never a successful compass direction.',
            full_content_and_motion_verification=False,
        )

    def _directions(self, waveform):
        unit, coherence, level = super()._directions(waveform)
        horizontal = torch.cat((unit[:, :2], torch.zeros_like(unit[:, :1])), dim=-1)
        norm = horizontal.norm(dim=-1, keepdim=True)
        direction = horizontal / norm.clamp_min(self.horizontal_projection_epsilon)
        defined = norm[:, 0] > self.horizontal_projection_epsilon
        # Keep the existing coherence/energy checks. A purely vertical field
        # has no defined azimuth, even when its 3D direction is very coherent.
        coherence = torch.where(defined, coherence, torch.zeros_like(coherence))
        return direction, coherence, level
