"""Stopped execution credit relative to the actual native default plan.

The default is a comparator, not an on-policy sampled action. Protection gates
positive credit; negative mean execution differences are retained. This changes
the planner surrogate, including the usual group-centered GRPO advantage.
It does not guarantee a better greedy planner after a shared-parameter step.
"""
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DefaultRelativeCredit:
    paired_differences: torch.Tensor
    mean_differences: torch.Tensor
    positive_eligible: torch.Tensor
    safe_credit: torch.Tensor
    scale: torch.Tensor
    normalized_advantage: torch.Tensor


def default_relative_credit(rewards, default_rewards, protected, same_plan, *,
                            legal=None, reward_epsilon=1e-6, scale_epsilon=1e-4):
    """Evaluate P plans on the same K>=2 execution seeds as the default.

    Failed outputs remain finite negative rewards supplied by the task scorer.
    Identical plans receive exactly zero credit, irrespective of score noise.
    Normalize by the population standard deviation of the safe plan means and
    the zero default anchor, without recentering them around sampled plans.
    """
    if (rewards.ndim != 2 or rewards.shape[1] < 2 or rewards.shape[0] < 1
            or default_rewards.shape != rewards.shape[1:]
            or protected.shape != rewards.shape or protected.dtype != torch.bool
            or same_plan.shape != rewards.shape[:1] or same_plan.dtype != torch.bool):
        raise ValueError('Expected aligned plan/noise rewards, protection and exact-plan identity.')
    if (not torch.isfinite(rewards).all() or not torch.isfinite(default_rewards).all()
            or not math.isfinite(reward_epsilon) or reward_epsilon < 0
            or not math.isfinite(scale_epsilon) or scale_epsilon <= 0):
        raise ValueError('Credit requires finite rewards and declared finite tolerances.')
    if legal is None:
        legal = torch.ones_like(same_plan)
    if legal.shape != same_plan.shape or legal.dtype != torch.bool:
        raise ValueError('Plan legality must match the candidate support.')
    with torch.no_grad():
        differences = rewards.detach().double() - default_rewards.detach().double()[None]
        means = differences.mean(-1)
        eligible = (~same_plan & legal & protected.all(-1)
                    & (differences > reward_epsilon).all(-1))
        safe = torch.where((means > 0) & ~eligible, torch.zeros_like(means), means)
        safe = torch.where(same_plan, torch.zeros_like(safe), safe)
        scale = torch.cat([safe.new_zeros(1), safe]).std(unbiased=False) + scale_epsilon
        return DefaultRelativeCredit(differences, means, eligible, safe, scale, safe / scale)


def default_relative_teacher(reference_log_probabilities, safe_credit, *, temperature):
    """Finite unique native-trace teacher; first support entry is the default.

    Caller deduplicates identical decision traces. These are nominal categorical
    probabilities, not probabilities under deterministic greedy selection and
    not a sum over every latent trace that may decode to the same ScenePlan.
    """
    if (reference_log_probabilities.ndim != 1
            or reference_log_probabilities.shape != safe_credit.shape
            or reference_log_probabilities.numel() < 1
            or not torch.isfinite(reference_log_probabilities).all()
            or not torch.isfinite(safe_credit).all()
            or float(safe_credit[0]) != 0
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Provide finite unique trace logits with zero credit for the default.')
    logits = reference_log_probabilities.detach().double() + safe_credit.detach().double() / temperature
    return logits.detach(), logits.softmax(0).detach()
