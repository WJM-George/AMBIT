"""Request-relative horizontal/time measurements from actual single-source audio.

No candidate plan or reference waveform supplies activity windows. This is an
explicitly scoped measurement component, not a full semantic/room verifier.
Historical fixed-window rewards remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from ...data.sceneplan_generation_ar_natural_constraints import (
    COMPASS_CENTERS, COMPASS_HALF_WIDTH, NUMERIC_TOLERANCES, TIME_PHASE_RANGES,
)
from .objectives import RewardScore


@dataclass(frozen=True)
class ActivityCompassConfig:
    energy_floor: float = 1e-6
    min_coherence: float = .1
    endpoint_frames: int = 3
    minimum_active_frames: int = 3
    time_resolution_frames: int = 1
    minimum_motion_progress: float = .1

    def __post_init__(self):
        values = (self.energy_floor, self.min_coherence, self.minimum_motion_progress)
        if not all(math.isfinite(x) for x in values):
            raise ValueError('Activity measurement parameters must be finite.')
        if not (self.energy_floor > 0 and 0 < self.min_coherence < 1 and 0 < self.minimum_motion_progress < 1):
            raise ValueError('Invalid activity/direction calibration.')
        if (type(self.minimum_active_frames) is not int or self.minimum_active_frames < 3
                or type(self.endpoint_frames) is not int or self.endpoint_frames < self.minimum_active_frames
                or type(self.time_resolution_frames) is not int or not 0 <= self.time_resolution_frames <= 2):
            raise ValueError('Declare positive evidence support and a bounded frame-resolution allowance.')


class RequestActivityCompassReward:
    """Separate observable direction, requested time, presence and motion trend.

    The activity detector uses W energy only, so unreliable spatial frames are
    still in the direction denominator. Audio silence cannot count as success.
    Removing content is not fully observable here: external ASR/content checks
    remain necessary, and direction never certifies semantics or source count.
    """
    profile = 'event_request_audio_activity_horizontal_v1_experimental'
    sample_rate = 44100
    hop = 1024

    def __init__(self, request_reward, *, config=ActivityCompassConfig()):
        self.requirements = request_reward.requirements
        self.config = config
        if len(self.requirements['sources']) != 1 or self.requirements.get('relations'):
            raise ValueError('Observed single-source activity cannot certify overlapping sources or relations.')
        self.source = self.requirements['sources'][0]
        allowed = {'compass', 'motion', 'time_phase', 'full_scene', 'starts_scene', 'ends_scene', 'transcript'}
        self.time_constraints, self.compass, self.motion = [], [], None
        for constraint in self.source['constraints']:
            op = constraint['op']
            if op == 'numeric':
                if constraint['field'] not in {'onset_sec', 'offset_sec', 'event_duration_sec'}:
                    raise ValueError('Additional spatial requirements need a separate audio verifier.')
            elif op not in allowed:
                raise ValueError('Additional spatial requirements need a separate audio verifier.')
            if op == 'compass':
                self.compass.append(constraint)
            elif op == 'motion':
                self.motion = constraint['value']
            elif op != 'transcript':
                self.time_constraints.append(constraint)
        if not self.compass:
            raise ValueError('This profile needs an explicit requested compass direction.')
        if self.motion not in {'static', 'linear'}:
            raise ValueError('Declare a supported static or linear source request.')
        self.start_angle = next((COMPASS_CENTERS[c['value']] for c in self.compass if c['point'] in ('start', 'both')), None)
        self.end_angle = next((COMPASS_CENTERS[c['value']] for c in self.compass if c['point'] in ('end', 'both')), None)
        if self.motion == 'linear':
            if (self.start_angle is None or self.end_angle is None
                    or abs((self.end_angle-self.start_angle+180) % 360-180) < 1e-6):
                raise ValueError('Same-sector or unspecified linear motion needs an additional motion/range observer.')

    def _time_checks(self, onset, offset, duration, present):
        tolerance = self.config.time_resolution_frames * self.hop / self.sample_rate
        rows = []
        for c in self.time_constraints:
            op = c['op']
            if not present:
                passed = False
            elif op == 'time_phase':
                value = onset if c['field'] == 'onset_sec' else offset
                lower, upper = TIME_PHASE_RANGES[c['value']]
                passed = lower*duration-tolerance <= value <= upper*duration+tolerance
            elif op == 'numeric':
                value = {'onset_sec': onset, 'offset_sec': offset, 'event_duration_sec': offset-onset}[c['field']]
                passed = abs(value-c['value']) <= NUMERIC_TOLERANCES[c['field']]+tolerance
            elif op == 'starts_scene':
                passed = onset <= .25+tolerance
            elif op == 'ends_scene':
                passed = duration-offset <= .25+tolerance
            elif op == 'full_scene':
                passed = onset <= .25+tolerance and duration-offset <= .25+tolerance
            else:
                raise ValueError('Unhandled request time constraint.')
            rows.append({'constraint': c, 'pass': bool(passed)})
        for c in self.requirements.get('scene', []):
            if c['op'] == 'numeric':
                passed = abs(duration-c['value']) <= NUMERIC_TOLERANCES['duration_sec']
            elif c['op'] == 'duration_range':
                passed = c['min'] <= duration and (duration <= c['max'] if c.get('max_inclusive', True) else duration < c['max'])
            else:
                continue  # Room is explicitly outside this observer's scope.
            rows.append({'constraint': c, 'pass': bool(passed)})
        return rows

    @torch.no_grad()
    def measure(self, waveform):
        from ...data.foa_intensity import foa_to_intensity_trajectory
        if (waveform.ndim != 3 or waveform.shape[:2] != (1, 4) or waveform.shape[-1] < self.hop
                or not torch.isfinite(waveform).all()):
            raise ValueError('Expected finite single-scene WYZX audio at 44100 Hz.')
        samples = waveform.shape[-1]
        frames = math.ceil(samples/self.hop)
        audio = torch.nn.functional.pad(waveform[0].float(), (0, frames*self.hop-samples))
        valid_samples = torch.full((frames,), self.hop, device=audio.device, dtype=torch.float32)
        valid_samples[-1] = samples-(frames-1)*self.hop
        energy = audio[0].reshape(frames, self.hop).square().sum(-1)/valid_samples
        active = energy >= self.config.energy_floor
        indices = active.nonzero().flatten()
        present = len(indices) >= self.config.minimum_active_frames
        duration = samples/self.sample_rate
        onset = float(indices[0])*self.hop/self.sample_rate if len(indices) else None
        offset = min(samples, (int(indices[-1])+1)*self.hop)/self.sample_rate if len(indices) else None
        trajectory = foa_to_intensity_trajectory(audio, hop=self.hop)
        horizontal = trajectory[:, :2]
        norms = horizontal.norm(dim=-1)
        unit = horizontal/norms[:, None].clamp_min(1e-8)
        coherence = 1-trajectory[:, 3]
        reliable = active & (coherence >= self.config.min_coherence) & (norms > 1e-6)
        # A start/end requirement does not ask for endpoint dwell throughout
        # 20% of the event. Use the declared minimum local evidence instead.
        count = self.config.endpoint_frames
        windows = {'both': active.clone(), 'start': torch.zeros_like(active), 'end': torch.zeros_like(active)}
        windows['start'][indices[:count]] = True
        windows['end'][indices[-count:]] = True
        direction_rows = []
        failures, unobservables = [], []
        for c in self.compass:
            mask = windows[c['point']]
            angle = math.radians(COMPASS_CENTERS[c['value']])
            target = unit.new_tensor([math.cos(angle), math.sin(angle)])
            cosine = unit @ target
            if int(mask.sum()) < self.config.minimum_active_frames:
                failure, unobservable = 1., 1.
            else:
                failure = float(((cosine < math.cos(math.radians(COMPASS_HALF_WIDTH))) | ~reliable)[mask].float().mean())
                unobservable = float((~reliable)[mask].float().mean())
            failures.append(failure)
            unobservables.append(unobservable)
            direction_rows.append(dict(constraint=c, frames=int(mask.sum()), failure=failure, unobservable=unobservable))
        time_rows = self._time_checks(onset, offset, duration, present)
        progress, motion_failure = None, float(not present)
        if present and self.motion == 'linear':
            means = []
            for point in ('start', 'end'):
                mask = windows[point] & reliable
                means.append(unit[mask].mean(0) if int(mask.sum()) >= self.config.minimum_active_frames else None)
            if any(value is None for value in means):
                motion_failure = 1.
            else:
                start, end = (math.radians(value) for value in (self.start_angle, self.end_angle))
                axis = unit.new_tensor([math.cos(end)-math.cos(start), math.sin(end)-math.sin(start)])
                axis = axis/axis.norm().clamp_min(1e-8)
                progress = float((means[1]-means[0]) @ axis)
                motion_failure = float(progress < self.config.minimum_motion_progress)
        costs = dict(requested_sector_failure=sum(failures)/len(failures),
            direction_unobservable_fraction=sum(unobservables)/len(unobservables),
            source_presence_failure=float(not present), requested_time_failure=float(any(not r['pass'] for r in time_rows)),
            motion_trend_failure=motion_failure, clipping=float((waveform.abs() > 1.).float().mean()))
        return dict(profile=self.profile, config=asdict(self.config), duration_sec=duration,
            observed_onset_sec=onset, observed_offset_sec=offset, active_frames=len(indices), total_frames=frames,
            active_frame_fraction=float(active.float().mean()), direction_checks=direction_rows, time_checks=time_rows,
            motion_progress=progress, costs=costs, utility=1-costs['requested_sector_failure'],
            window_source='Actual W-energy activity with fixed thresholds; no reference or candidate-plan windows.',
            limitations=['Single-source hypothesis requires independent content/source verification.',
                'ASR, semantics, room acoustics, source binding and perceptual quality are not certified.',
                'Linear-motion check measures endpoint progress, not a complete path or constant velocity.',
                'Static direction is checked across active frames; within-sector jitter is not separately penalized.',
                'Activity deletion can remove content; direction must never be accepted without content/presence/time protection.'])

    def __call__(self, waveform):
        result = self.measure(waveform)
        return RewardScore(result['utility'], result['costs'])
