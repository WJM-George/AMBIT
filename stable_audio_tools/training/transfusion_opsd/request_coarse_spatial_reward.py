"""Coarse requested horizontal direction from integrated FOA evidence.

This deliberately versioned development profile complements the historical
20ms frame profile. It never changes old reports or certifies speech/semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from ...data.sceneplan_generation_ar_natural_constraints import COMPASS_CENTERS, COMPASS_HALF_WIDTH
from .event_activity_compass_reward import RequestActivityCompassReward
from .objectives import RewardScore


@dataclass(frozen=True)
class CoarseSpatialConfig:
    energy_intervals: int = 5
    extra_angle_tolerance_deg: float = 5.
    minimum_horizontal_coherence: float = .1
    minimum_motion_progress: float = .1

    def __post_init__(self):
        if type(self.energy_intervals) is not int or self.energy_intervals < 3:
            raise ValueError('Use at least three declared energy intervals.')
        if not (math.isfinite(self.extra_angle_tolerance_deg) and 0 <= self.extra_angle_tolerance_deg <= 10):
            raise ValueError('Declare a bounded coarse-angle tolerance.')
        if not all(math.isfinite(x) and 0 < x < 1 for x in (
                self.minimum_horizontal_coherence, self.minimum_motion_progress)):
            raise ValueError('Invalid observation or movement threshold.')


class RequestCoarseSpatialReward:
    """A bounded-scope observer for approximate single-source directions.

    Windows contain equal W energy, selected from the output without a plan or
    reference. Integrated W*X/W*Y gives horizontal evidence; whole-clip averaging
    alone cannot hide left/right changes in a static request. Movement uses the
    first/last windows, so this is coarse endpoint evidence, not exact endpoints
    or a guarantee about the full trajectory or uniform speed.
    """
    profile = 'request_coarse_horizontal_equal_W_energy_v1'

    def __init__(self, request_reward, *, config=CoarseSpatialConfig()):
        self.local = RequestActivityCompassReward(request_reward)
        self.config = config

    @torch.no_grad()
    def measure(self, waveform):
        previous = self.local.measure(waveform)
        wave = waveform[0].double()
        energy = wave[0].square()
        total = energy.sum()
        count = self.config.energy_intervals
        boundaries = torch.searchsorted(energy.cumsum(0), torch.linspace(0, 1, count+1, device=wave.device)*total)
        boundaries[0], boundaries[-1] = 0, wave.shape[-1]
        intervals = []
        present = previous['costs']['source_presence_failure'] == 0
        for index in range(count):
            start, end = int(boundaries[index]), int(boundaries[index+1])
            window = wave[:, start:end]
            intensity = (window[0, None]*window[[3, 1, 2]]).sum(-1)
            denominator = (window[0].square().sum()*window[[3, 1, 2]].square().sum()).sqrt()
            horizontal = intensity[:2].norm()
            coherence = float(horizontal/denominator.clamp_min(1e-20))
            observable = bool(present and end > start and coherence >= self.config.minimum_horizontal_coherence and horizontal > 1e-12)
            angle = float(torch.rad2deg(torch.atan2(intensity[1], intensity[0]))) if observable else None
            intervals.append(dict(index=index, start_sample=start, end_sample=end,
                start_sec=start/44100, end_sec=end/44100, horizontal_coherence=coherence,
                observable=observable, azimuth_deg=angle))
        half_width = COMPASS_HALF_WIDTH+self.config.extra_angle_tolerance_deg
        direction_rows, failures, unobservables, excesses = [], [], [], []
        for constraint in self.local.compass:
            selected = intervals if constraint['point'] == 'both' else [intervals[0 if constraint['point'] == 'start' else -1]]
            center = COMPASS_CENTERS[constraint['value']]
            local_excess = [max(0., abs((item['azimuth_deg']-center+180) % 360-180)-half_width)
                            if item['observable'] else 180. for item in selected]
            failure = sum(value > 1e-9 for value in local_excess)/len(selected)
            unobservable = sum(not item['observable'] for item in selected)/len(selected)
            mean_excess = sum(local_excess)/len(selected)
            failures.append(failure)
            unobservables.append(unobservable)
            excesses.append(mean_excess)
            direction_rows.append(dict(constraint=constraint, interval_indices=[x['index'] for x in selected],
                failure=failure, unobservable=unobservable, mean_excess_angle_deg=mean_excess))
        progress = None
        motion_failure = float(not present)
        if present and self.local.motion == 'linear':
            if not (intervals[0]['observable'] and intervals[-1]['observable']):
                motion_failure = 1.
            else:
                unit = lambda angle: (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
                a, b = unit(self.local.start_angle), unit(self.local.end_angle)
                axis = (b[0]-a[0], b[1]-a[1])
                norm = math.hypot(*axis)
                first, last = unit(intervals[0]['azimuth_deg']), unit(intervals[-1]['azimuth_deg'])
                progress = sum((last[i]-first[i])*axis[i]/norm for i in (0, 1))
                motion_failure = float(progress < self.config.minimum_motion_progress)
        costs = dict(previous['costs'], requested_sector_failure=sum(failures)/len(failures),
            direction_unobservable_fraction=sum(unobservables)/len(unobservables),
            motion_trend_failure=motion_failure)
        mean_excess = sum(excesses)/len(excesses)
        return dict(profile=self.profile, config=asdict(self.config), nominal_half_width_deg=COMPASS_HALF_WIDTH,
            effective_half_width_deg=half_width, intervals=intervals, direction_checks=direction_rows,
            mean_excess_angle_deg=mean_excess, motion_progress=progress, costs=costs,
            utility=1-mean_excess/180., original_frame_costs=previous['costs'],
            time_checks=previous['time_checks'], active_frames=previous['active_frames'],
            observed_onset_sec=previous['observed_onset_sec'], observed_offset_sec=previous['observed_offset_sec'],
            limitations=['Single-source and horizontal coarse observations only.',
                'The first/last fifths of W energy do not establish exact instantaneous endpoint positions.',
                'Speech, sound semantics, room and source binding need independent checks.',
                'Five intervals can miss shorter changes; complete motion-path correctness is not certified.',
                'Historical per-frame experiments are not reclassified by this profile.'])

    def __call__(self, waveform):
        result = self.measure(waveform)
        return RewardScore(result['utility'], result['costs'])
