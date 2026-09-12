"""Execution supervision on the native feasible onset/offset decision.

Collection must bind rewards to actual paired executions and qualify them.
This module constructs a stopped finite teacher, not an execution certificate.
"""
from __future__ import annotations

import math

import torch

from ...models.sceneplan_generation_ar_qualitative_head import PHASES
from .identity_preserving_kl import identity_preserving_kl


def feasible_timing_pairs(frames):
    if not isinstance(frames, int) or isinstance(frames, bool) or frames < 1:
        raise ValueError('A positive integer native duration is required.')
    grid = [(max(0, math.ceil(low * frames - 1e-8)),
             min(frames, math.floor(high * frames + 1e-8))) for low, high in PHASES]
    return tuple((i, j) for i, (low_on, high_on) in enumerate(grid)
                 for j, (low_off, high_off) in enumerate(grid)
                 if low_on <= min(high_on, frames - 1)
                 and max(1, low_off) <= high_off and low_on < high_off)


def timing_pair_logits(onset, offset, frames):
    if (onset.shape != offset.shape or onset.ndim != 1 or onset.numel() != len(PHASES)
            or onset.device != offset.device or not onset.is_floating_point()
            or not offset.is_floating_point()
            or not torch.isfinite(onset).all() or not torch.isfinite(offset).all()):
        raise ValueError('Require aligned finite native onset and offset vectors.')
    pairs = feasible_timing_pairs(frames)
    if not pairs:
        raise ValueError('No feasible native timing pair exists at this duration.')
    return torch.stack([onset[i] + offset[j] for i, j in pairs])


def timing_execution_teacher(onset, offset, frames, rewards, *, temperature):
    """Tilt qualified measured pairs, keeping all unmeasured mass unchanged.

    rewards maps (onset_index, offset_index) to mean actual execution reward.
    The caller qualifies every complete paired-noise panel before supplying it.
    """
    if not math.isfinite(temperature) or temperature <= 0 or len(rewards) < 2:
        raise ValueError('Require a finite positive temperature and multiple measured pairs.')
    pairs = feasible_timing_pairs(frames)
    if any(key not in pairs or not math.isfinite(value) for key, value in rewards.items()):
        raise ValueError('Measured rewards must be finite on feasible native pairs.')
    reference = timing_pair_logits(onset, offset, frames).detach().double()
    indices = [i for i, pair in enumerate(pairs) if pair in rewards]
    values = reference.new_tensor([rewards[pairs[i]] for i in indices])
    target = reference.clone()
    if not bool((values == values[0]).all()):
        shifted = reference[indices] + (values - values.max()) / temperature
        if not torch.isfinite(shifted).all():
            raise ValueError('Reward scale is outside finite numerical range.')
        target[indices] = shifted + reference[indices].logsumexp(0) - shifted.logsumexp(0)
    before, after = reference.softmax(0), target.softmax(0)
    return dict(pairs=pairs, reference_logits=reference, target_logits=target,
        probabilities=after, measured_indices=indices,
        measured_probability_mass=float(before[indices].sum()),
        expected_measured_reward_before=float((before[indices] * values).sum()),
        expected_measured_reward_target=float((after[indices] * values).sum()),
        reference_choice=pairs[int(reference.argmax())], target_choice=pairs[int(target.argmax())],
        temperature=float(temperature))


def native_timing_loss(onset, offset, frames, target_logits):
    """Differentiate actual additive native head scores, not detached labels."""
    return identity_preserving_kl(timing_pair_logits(onset, offset, frames), target_logits)


def factorized_timing_target(onset, offset, frames, target_logits, *, max_iter=64):
    """Finite CPU KL projection onto the existing additive head family.

    This is only a stopped logit target for native readout fitting. It does not
    certify that parameters can fit it or that the resulting audio improves.
    """
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError('Declare a finite positive projection budget.')
    initial = torch.stack((onset, offset)).detach().double().cpu()
    target = target_logits.detach().double().cpu()
    if target.shape != timing_pair_logits(*initial, frames).shape or not torch.isfinite(target).all():
        raise ValueError('Target must cover every feasible native timing pair.')
    parameters = initial.clone().requires_grad_(True)
    q = target.softmax(0)
    optimizer = torch.optim.LBFGS([parameters], lr=1., max_iter=max_iter,
        tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    calls = 0

    def closure():
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=True)
        logits = timing_pair_logits(*parameters, frames)
        loss = (q * (target.log_softmax(0) - logits.log_softmax(0))).sum()
        loss.backward()
        return loss

    before = float(closure().detach())
    optimizer.step(closure)
    with torch.no_grad():
        # Additive offsets do not identify probabilities; keep reference means.
        parameters.add_(initial.mean(-1, keepdim=True) - parameters.mean(-1, keepdim=True))
        logits = timing_pair_logits(*parameters, frames)
        after = float((q * (target.log_softmax(0) - logits.log_softmax(0))).sum())
    if not torch.isfinite(parameters).all() or after > before + 1e-9:
        raise ValueError('Finite native logit projection did not improve its own objective.')
    return parameters.detach(), dict(kl_before=before, kl_after=after,
        closure_calls=calls, maximum_iterations=max_iter,
        projected_choice=feasible_timing_pairs(frames)[int(logits.argmax())],
        target_choice=feasible_timing_pairs(frames)[int(target.argmax())])
