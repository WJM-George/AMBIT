"""Paired execution when an admissible EVENT update changes free duration.

At the reference length, noise and Euler math exactly match native generation.
At a different length, common latent time positions share the reference noise;
new positions use additional standard-normal draws. This is an explicit noise
coupling protocol, not native same-integer-seed equality at a different shape.
"""
from __future__ import annotations

import copy

import torch


def paired_gaussian_noise(seed, shape, *, reference_frames):
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or len(shape) < 2 or any(not isinstance(value, int) or value <= 0 for value in shape)
            or not isinstance(reference_frames, int) or reference_frames <= 0):
        raise ValueError('paired noise requires a nonnegative seed and positive fixed channel/time geometry')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    reference = torch.randn((*shape[:-1], reference_frames), generator=generator, dtype=torch.float32)
    if shape[-1] <= reference_frames:
        return reference[..., :shape[-1]].contiguous()
    tail = torch.randn((*shape[:-1], shape[-1] - reference_frames), generator=generator, dtype=torch.float32)
    return torch.cat((reference, tail), dim=-1)


@torch.no_grad()
def paired_native_rollout(bundle, condition, *, seed, reference_frames, steps=100):
    from ...inference.sampling import sample_discrete_euler
    from .objectives import EulerTrace
    if steps != 100:
        raise ValueError('the current EVENT adapter retains its native 100-step schedule')
    noise = paired_gaussian_noise(seed, (1, 64, condition.mask.shape[-1]),
        reference_frames=reference_frames).to(bundle.device)
    times = bundle.schedule(steps, noise.shape[-1])
    states = []
    final = sample_discrete_euler(bundle.velocity_function(condition, differentiable=False), noise,
        torch.tensor(times, dtype=torch.float32, device=bundle.device), disable_tqdm=True,
        callback=lambda values: states.append(values['x'].detach().clone()))
    states.append(final.detach().clone())
    return EulerTrace(tuple(states), times, condition.mask.detach().clone())


class FixedRequestAudioProtocol:
    """Fixed reference observation windows, consistently mapped across duration.

The candidate supplies only its output length; it cannot move the observation
windows to its own favorable regions. Relative/free reference activity bounds
scale with total duration. Explicit requested seconds stay in seconds. Request
admissibility of the actual candidate remains a separate mandatory check.
    """
    def __init__(self, request_reward, reference_plan, *, model_num_samples, extra_costs=None):
        from .event_rewards import EventRequestFoaRewardV2
        self.request = request_reward
        self.plan = copy.deepcopy(reference_plan)
        self.num_samples, self.extra_costs = model_num_samples, extra_costs
        self.initial = EventRequestFoaRewardV2(request_reward, reference_plan,
            model_num_samples=model_num_samples, extra_costs=extra_costs)
        verification = request_reward.evaluate(reference_plan)
        if not verification['request_constraints_joint']:
            raise ValueError('fixed observation reference must obey the request')
        self.bindings = {value: key for key, value in verification['assignment'].items()}

    def reward_for_samples(self, num_samples):
        from .event_rewards import EventRequestFoaRewardV2
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError('counterfactual waveform needs a positive actual length')
        if num_samples == self.num_samples:
            return self.initial
        plan = copy.deepcopy(self.plan)
        duration = num_samples / 44100.
        ratio = duration / plan['duration_sec']
        plan['duration_sec'] = duration
        requested = {source['key']: source for source in self.request.requirements['sources']}
        for source in plan['sources']:
            requirement = requested[self.bindings[source['source_id']]]
            absolute = {constraint['field'] for constraint in requirement['constraints']
                if constraint['op'] == 'numeric' and constraint['field'] in ('onset_sec', 'offset_sec')}
            for key in ('onset_sec', 'offset_sec'):
                value = source['activity'][key]
                source['activity'][key] = min(duration, max(0., value if key in absolute else value * ratio))
        return EventRequestFoaRewardV2(self.request, plan, model_num_samples=num_samples, extra_costs=self.extra_costs)


class FixedDirectionalAudioProtocol(FixedRequestAudioProtocol):
    """Map baseline direction weights when a permitted free duration changes.

    Observation windows come from the fixed reference protocol, including its
    treatment of explicit requested seconds. Candidates supply only length.
    """
    def __init__(self, request_reward, reference_plan, *, reference_audio, model_num_samples,
            min_coherence=.1, extra_costs=None):
        from .event_directional_reward import FixedReferenceDirectionalReward
        super().__init__(request_reward, reference_plan, model_num_samples=model_num_samples, extra_costs=extra_costs)
        self.initial = FixedReferenceDirectionalReward(request_reward, reference_plan, reference_audio=reference_audio,
            model_num_samples=model_num_samples, min_coherence=min_coherence, extra_costs=extra_costs)

    def reward_for_samples(self, num_samples):
        if num_samples == self.num_samples:
            return self.initial
        from .rewards import UnobservableAudio
        mapped = super().reward_for_samples(num_samples)
        result = copy.copy(self.initial)
        for key in ('num_samples', 'active', 'solo', 'windows', 'directions', 'request_allowed_activity'):
            setattr(result, key, getattr(mapped, key))
        weights = torch.nn.functional.interpolate(self.initial.reference_weights[None],
            size=len(mapped.active), mode='linear', align_corners=False)[0] * mapped.windows
        if (weights.sum(-1) <= 0).any():
            raise UnobservableAudio('fixed direction weights have no support in the mapped requested windows')
        result.reference_weights = weights / weights.sum(-1, keepdim=True)
        result.evidence = {**self.initial.evidence, 'actual_samples': num_samples,
            'duration_mapping': 'linear reference confidence remap, intersect fixed request windows; explicit seconds retained'}
        return result
