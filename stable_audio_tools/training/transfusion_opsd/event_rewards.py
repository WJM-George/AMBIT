"""Request-grounded EVENT feedback. Hidden numerical ScenePlan targets are absent."""
from __future__ import annotations

import copy
import math

import torch

from ...data.sceneplan_generation_ar_exact_proof import exact_core_labels
from ...data.sceneplan_generation_ar_natural_constraints import evaluate_natural_request, validate_requirements
from ...models.sceneplan_generation_ar_qualitative_head import ATTRIBUTES, CENTERS
from .objectives import RewardScore
from .rewards import UnobservableAudio


class EventRequestReward:
    """Training-side verifier with one consistent source assignment for all fields.

    Exact literal agreement is sufficient evidence. Unresolved paraphrases stay
    unknown, never fabricated semantic failures. There is no GT-plan argument.
    """

    def __init__(self, request, requirements, *, semantic_labels=None):
        validate_requirements(request, requirements)
        self.request = request
        self.requirements = copy.deepcopy(requirements)
        self.semantic_labels = dict(semantic_labels or {})

    def evaluate(self, plan):
        labels = dict(self.semantic_labels)
        labels.update(exact_core_labels(self.requirements, plan))
        return evaluate_natural_request(self.request, self.requirements, plan, labels)

    def admissible(self, plan):
        return self.evaluate(plan)['request_constraints_joint']

    def __call__(self, plan):
        score = self.evaluate(plan)
        constraints = [c for row in score['sources'] for c in row['constraints']]
        constraints += score['scene_constraints'] + score['relations']
        utility = sum(c['pass'] for c in constraints) / max(1, len(constraints))
        return RewardScore(utility, {'invalid': float(not score['valid']),
            'source_count': float(not score['count_correct']),
            'content_or_binding_unverified': float(any(r['core_semantics'] is not True for r in score['sources'])),
            'transcript': float(any(not c['pass'] for c in constraints if c['constraint']['op'] == 'transcript'))})


class EventRequestFoaReward:
    """Observable requested compass directions, with fixed evidence windows.

    The dense preference is angular margin toward the *requested sector center*,
    not a hidden witness or the candidate's exact angle. Request-sector failure
    is reported separately. This verifier cannot judge audio content or range.
    """

    def __init__(self, request_reward, plan, *, model_num_samples, energy_floor=1e-6,
                 endpoint_fraction=.2, sector_tolerance_deg=27.5, extra_costs=None):
        from ...data.model_sceneplan import compile_model_44_controls
        if not 0 < endpoint_fraction <= .5 or energy_floor <= 0 or not 0 < sector_tolerance_deg < 90:
            raise ValueError('invalid fixed spatial feedback calibration')
        evaluated = request_reward.evaluate(plan)
        if not evaluated['request_constraints_joint']:
            raise ValueError('spatial feedback requires verified request-admissible plan semantics and binding')
        self.num_samples, self.energy_floor = model_num_samples, energy_floor
        self.extra_costs = extra_costs
        controls = compile_model_44_controls(plan, model_num_samples=model_num_samples)
        active = torch.as_tensor(controls['source_event_frame_ids']) != 0
        self.active, self.solo = active.any(0), active.sum(0) == 1
        self.cos_tolerance = math.cos(math.radians(sector_tolerance_deg))
        source_ids = [s['source_id'] for s in plan['sources']]
        windows, directions, identities = [], [], []
        for source in request_reward.requirements['sources']:
            slot = source_ids.index(evaluated['assignment'][source['key']])
            active_indices = active[slot].nonzero().flatten()
            if not len(active_indices):
                raise UnobservableAudio('requested source has no fixed active frames')
            start, end = int(active_indices[0]), int(active_indices[-1]) + 1
            count = max(1, math.ceil((end - start) * endpoint_fraction))
            for constraint in source['constraints']:
                if constraint['op'] != 'compass':
                    continue
                point = constraint['point']
                region = torch.zeros_like(self.solo)
                if point == 'both':
                    region[start:end] = True
                elif point == 'start':
                    region[start:start + count] = True
                elif point == 'end':
                    region[end - count:end] = True
                else:
                    raise ValueError('unsupported compass evidence scope')
                observed = region & self.solo & active[slot]
                if int(observed.sum()) < 3:
                    raise UnobservableAudio('a requested source/endpoint lacks at least three solo frames')
                angle = math.radians(CENTERS[ATTRIBUTES['start'].index(constraint['value'])])
                windows.append(observed)
                directions.append([math.cos(angle), math.sin(angle), 0.])
                identities.append({'request_source': source['key'], 'source_id': source_ids[slot],
                    'point': point, 'frames': int(observed.sum()), 'possible_frames': int(region.sum())})
        if not windows:
            raise UnobservableAudio('request has no supported observable compass constraints')
        self.windows = torch.stack(windows)
        self.directions = torch.tensor(directions, dtype=torch.float32)
        self.evidence = identities

    def _measure(self, waveform):
        from ...data.foa_intensity import foa_to_intensity_trajectory
        if waveform.shape != (1, 4, self.num_samples) or not torch.isfinite(waveform).all():
            raise ValueError('request spatial feedback requires finite exact-length WYZX FOA')
        frames = len(self.active)
        audio = torch.nn.functional.pad(waveform[0].float(), (0, frames * 1024 - self.num_samples))
        direction = foa_to_intensity_trajectory(audio)[:, :3]
        energy = audio[0].view(frames, 1024).square().mean(-1)
        level = energy / (energy + self.energy_floor)
        windows = self.windows.to(waveform.device)
        dot = (self.directions.to(waveform.device) @ direction.T).clamp(-1, 1)
        weights = windows / windows.sum(-1, keepdim=True)
        utility = (weights * ((dot + 1.) / 2) * level[None]).sum(-1).mean()
        failures = (dot < self.cos_tolerance) | (level[None] < .5)
        # Count failures before division. Summing floating uniform weights can
        # give different rates for identical counts at different frame indices.
        angular_violation = ((windows & failures).sum(-1).float() / windows.sum(-1)).mean()
        active = self.active.to(waveform.device)
        silence = (1 - level)[active].mean()
        leakage = energy[~active].mean() / energy[active].mean().clamp_min(self.energy_floor) if (~active).any() else energy.sum() * 0
        clipping = (waveform.abs() > 1.).float().mean()
        return utility, {'silence': silence, 'activity_leakage': leakage, 'clipping': clipping,
            'requested_sector_failure': angular_violation}

    def differentiable(self, waveform):
        return self._measure(waveform)[0]

    @torch.no_grad()
    def __call__(self, waveform):
        utility, costs = self._measure(waveform)
        result = {key: float(value) for key, value in costs.items()}
        if self.extra_costs is not None:
            additional = self.extra_costs(waveform)
            if set(additional) & result.keys():
                raise ValueError('external protection cannot replace built-in measurements')
            result.update(additional)
        return RewardScore(float(utility), result)


class EventRequestFoaRewardV2(EventRequestFoaReward):
    """Request-level presence/time protection; retain raw diagnostics separately.

    A freely chosen exact onset is not a request boundary. Likewise, a change
    in already audible energy is not by itself disappearance of a source. The
    energy floor is unchanged from v1; it is not fitted to candidate outcomes.
    This remains a spatial signal profile, requiring separate content/ASR checks.
    """
    profile = 'event_request_foa_presence_and_allowed_time_v2'

    def __init__(self, request_reward, plan, **kwargs):
        from ...data.sceneplan_generation_ar_natural_constraints import TIME_PHASE_RANGES, NUMERIC_TOLERANCES
        super().__init__(request_reward, plan, **kwargs)
        times = (torch.arange(len(self.active), dtype=torch.float32) + .5) * 1024 / 44100
        allowed = torch.zeros_like(self.active)
        for source in request_reward.requirements['sources']:
            earliest, latest = 0., plan['duration_sec']
            for constraint in source['constraints']:
                if constraint['op'] == 'time_phase':
                    low, high = TIME_PHASE_RANGES[constraint['value']]
                    if constraint['field'] == 'onset_sec':
                        earliest = max(earliest, low * plan['duration_sec'])
                    elif constraint['field'] == 'offset_sec':
                        latest = min(latest, high * plan['duration_sec'])
                elif constraint['op'] == 'numeric':
                    if constraint['field'] == 'onset_sec':
                        earliest = max(earliest, constraint['value'] - NUMERIC_TOLERANCES['onset_sec'])
                    elif constraint['field'] == 'offset_sec':
                        latest = min(latest, constraint['value'] + NUMERIC_TOLERANCES['offset_sec'])
            allowed |= (times >= earliest) & (times <= latest)
        self.request_allowed_activity = allowed

    def _measure(self, waveform):
        utility, raw = super()._measure(waveform)
        frames = len(self.active)
        omni = torch.nn.functional.pad(waveform[0, 0].float(), (0, frames * 1024 - self.num_samples))
        energy = omni.view(frames, 1024).square().mean(-1)
        windows = self.windows.to(waveform.device)
        average_energy = (windows * energy[None]).sum(-1) / windows.sum(-1)
        presence = (average_energy < self.energy_floor).float().mean()
        forbidden = ~self.request_allowed_activity.to(waveform.device)
        forbidden_activity = (energy[forbidden] >= self.energy_floor).float().mean() if forbidden.any() else energy.sum() * 0
        return utility, {'source_presence_failure': presence, 'forbidden_activity_fraction': forbidden_activity,
            'clipping': raw['clipping'], 'requested_sector_failure': raw['requested_sector_failure']}

    @torch.no_grad()
    def diagnostics(self, waveform):
        utility, raw = super()._measure(waveform)
        return {'utility': float(utility), **{key: float(value) for key, value in raw.items()},
            'exact_proposal_activity_is_a_diagnostic_not_a_requested_number': True}
