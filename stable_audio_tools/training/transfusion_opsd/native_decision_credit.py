"""Route execution credit to native decision logits, separating repair duties.

The caller evaluates an actual hard native DiT forward with an ST condition.
This extracts its logit derivative without accumulating executor gradients.
Separate plan-matched residual losses supply the ordinary DiT/shared update.
It is a declared gradient routing rule, not the gradient of the sum of all
displayed diagnostic scalars with respect to every parameter.
"""
from __future__ import annotations

import torch


def execution_decision_derivative(loss, logits, condition_probe):
    """Keep the AR graph upstream of logits for a subsequent weighted backward.

    condition_probe is a leaf with identical hard-condition values in both
    arms. Its derivative keeps the detached control's DiT backward comparable.
    None for logits means that the declared decision connection was detached.
    """
    if (loss.numel()!=1 or not loss.requires_grad or not torch.isfinite(loss)
            or logits.ndim!=1 or not logits.requires_grad
            or not condition_probe.is_leaf or not condition_probe.requires_grad):
        raise ValueError('Require a finite differentiable scalar, native logits and a condition leaf.')
    condition_grad, decision_grad = torch.autograd.grad(
        loss, (condition_probe, logits), allow_unused=True)
    if condition_grad is None or not torch.isfinite(condition_grad).all():
        raise ValueError('Execution credit lost the declared condition derivative.')
    if decision_grad is not None and not torch.isfinite(decision_grad).all():
        raise ValueError('Execution credit has a nonfinite native decision derivative.')
    return (None if decision_grad is None else decision_grad.detach()), condition_grad.detach()
