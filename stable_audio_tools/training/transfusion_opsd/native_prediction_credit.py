"""Hard native predictions with finite candidate prediction credit to AR.

This optional surrogate uses actually evaluated executor predictions rather
than the local Jacobian of an interpolated condition. It preserves the hard
forward and its ordinary gradient; alternative predictions stop gradients.
It is a biased finite-support surrogate, not an argmax derivative or a
guarantee about the final task reward. Candidate eligibility is the caller's
responsibility, including an explicit distinction between field and full-plan
teachers. Generation and Editing use the same operator.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def hard_native_prediction(
    hard: torch.Tensor,
    alternatives: Sequence[torch.Tensor],
    logits: torch.Tensor,
    *,
    hard_index: int,
    temperature: float = 1.,
    connect_decisions: bool = True,
) -> torch.Tensor:
    """Add a zero-valued prediction secant; never sample a soft waveform.

    For two choices, hard v0 and target v1, mean-squared-error credit to logit
    1 is -2*p0*p1*mean((v1-v0)^2)/temperature. Thus it cannot point away from
    that exact prediction target, unlike a nonlinear condition Jacobian.
    This narrow identity does not imply improved normal audio or retention.
    """
    n = len(alternatives)
    if (not isinstance(hard, torch.Tensor) or not hard.is_floating_point()
            or not torch.isfinite(hard).all() or n < 2
            or logits.ndim != 1 or logits.numel() != n
            or not logits.is_floating_point() or not torch.isfinite(logits).all()
            or logits.device != hard.device
            or not math.isfinite(temperature) or temperature <= 0
            or isinstance(hard_index, bool) or not 0 <= hard_index < n):
        raise ValueError('Require a finite hard prediction and aligned candidate logits.')
    for value in alternatives:
        if (not isinstance(value, torch.Tensor) or value.shape != hard.shape
                or value.dtype != hard.dtype or value.device != hard.device
                or not torch.isfinite(value).all()):
            raise ValueError('Candidate predictions must have identical geometry and finite values.')
    if not torch.equal(hard.detach(), alternatives[hard_index].detach()):
        raise ValueError('Hard prediction must be the actual selected candidate prediction.')
    if not connect_decisions:
        return hard
    weights = (logits / temperature).softmax(0).to(hard.dtype)
    soft = sum(weight * value.detach() for weight, value in zip(weights, alternatives))
    return hard + (soft - soft.detach())
