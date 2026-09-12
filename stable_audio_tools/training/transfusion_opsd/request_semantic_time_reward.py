"""Versioned audio timing semantics for unqualified early/late requests.

An early event may start at the beginning; a late event may stop at the end.
Those words alone do not request silent gaps at either edge. Explicit numeric
or scene-boundary constraints remain independent checks. Native AR label
ranges and historical audio profiles are deliberately left unchanged.
"""
from __future__ import annotations

import torch

from ...data.sceneplan_generation_ar_natural_constraints import TIME_PHASE_RANGES
from .event_activity_compass_reward import RequestActivityCompassReward
from .request_coarse_spatial_reward import RequestCoarseSpatialReward, CoarseSpatialConfig


class RequestSemanticTimeActivityReward(RequestActivityCompassReward):
    profile = 'event_request_audio_edge_inclusive_time_v1'

    def _time_checks(self, onset, offset, duration, present):
        rows = super()._time_checks(onset, offset, duration, present)
        tolerance = self.config.time_resolution_frames*self.hop/self.sample_rate
        for row in rows:
            constraint = row['constraint']
            if constraint['op'] != 'time_phase' or constraint['value'] not in ('early', 'late'):
                continue
            lower, upper = TIME_PHASE_RANGES[constraint['value']]
            if constraint['value'] == 'early':
                lower = 0.
            else:
                upper = 1.
            value = onset if constraint['field'] == 'onset_sec' else offset
            row['pass'] = bool(present and value is not None and lower*duration-tolerance <= value <= upper*duration+tolerance)
            row['audio_phase_range'] = [lower, upper]
            row['time_semantics'] = 'early_includes_beginning_late_includes_ending_v1'
        return rows


class RequestSemanticTimeCoarseReward(RequestCoarseSpatialReward):
    profile = 'request_coarse_horizontal_semantic_time_v2'

    def __init__(self, request_reward, *, config=CoarseSpatialConfig()):
        super().__init__(request_reward, config=config)
        self.local = RequestSemanticTimeActivityReward(request_reward)

    @torch.no_grad()
    def measure(self, waveform):
        result = super().measure(waveform)
        result['time_semantics'] = dict(
            rule='early_includes_beginning_late_includes_ending_v1',
            early=[0., TIME_PHASE_RANGES['early'][1]],
            late=[TIME_PHASE_RANGES['late'][0], 1.],
            other_phases='unchanged', numeric_and_boundary_constraints='unchanged',
            resolution_tolerance_sec=self.local.config.time_resolution_frames*self.local.hop/self.local.sample_rate,
            native_AR_constraints='unchanged',
            scope='Unqualified early/late constraints in the current request schema; explicit additional limits remain mandatory.')
        return result
