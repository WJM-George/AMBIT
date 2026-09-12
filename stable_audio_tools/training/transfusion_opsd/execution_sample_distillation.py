"""A finite execution-distribution teacher, distinct from trajectory repair.

The stopped weights tilt a measured empirical sample distribution. They do
not establish an exact flow likelihood, a policy gradient, or student gain.
"""
from __future__ import annotations

import math

import torch


def execution_sample_teacher(rewards, *, temperature, prior=None):
    """Maximize mean reward minus temperature*KL on a fixed sample panel.

    Every panel member remains represented. Callers bind each reward to its
    actual plan, executor and output; eligibility is not inferred here. The
    empirical target may include imperfect samples, so its average advantage
    is not a statement that every sample is correct or improves each noise.
    """
    if (rewards.ndim != 1 or rewards.numel() < 2 or not rewards.is_floating_point()
            or not torch.isfinite(rewards).all() or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Require a finite multi-sample reward panel and positive temperature.')
    values = rewards.detach().double()
    if prior is None:
        fixed = torch.full_like(values, 1 / values.numel())
    else:
        if (prior.shape != values.shape or prior.device != values.device or not prior.is_floating_point()
                or not torch.isfinite(prior).all() or not (prior > 0).all()
                or not torch.isclose(prior.double().sum(), values.new_tensor(1.), atol=1e-8, rtol=0)):
            raise ValueError('Use a positive normalized prior on the complete measured panel.')
        fixed = prior.detach().double()
    logits = fixed.log() + (values - values.max()) / temperature
    if not torch.isfinite(logits).all():
        raise ValueError('Reward temperature exceeds finite numerical range.')
    log_target = logits.log_softmax(-1)
    target = log_target.exp()
    return dict(probabilities=target, prior=fixed,
        mean_reward_before=(fixed * values).sum(), mean_reward_target=(target * values).sum(),
        target_kl=(target * (log_target - fixed.log())).sum())
