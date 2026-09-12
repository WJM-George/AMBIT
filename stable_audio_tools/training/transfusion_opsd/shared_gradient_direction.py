"""A bounded shared-parameter direction control, not an efficacy guarantee.

Normalize the two task-gradient directions, then restore the norm of their
ordinary sum. Private gradients, global clipping and optimizer rules are not
changed. This is a gradient-composition experiment, not exact GradNorm, not the
gradient of the original fixed scalar objective, and not gradient-through-plan.
"""
from __future__ import annotations

import math
import torch


def shared_direction_coefficients(planning, execution, *, epsilon=1e-12):
    if not planning or len(planning) != len(execution) or epsilon <= 0:
        raise ValueError('Two aligned nonempty shared-gradient lists are required.')
    aa = dd = ad = 0.
    for a, d in zip(planning, execution):
        if a.shape != d.shape or not torch.isfinite(a).all() or not torch.isfinite(d).all():
            raise ValueError('Shared gradients must have matching finite coordinates.')
        af, df = a.double(), d.double()
        aa += float(af.square().sum()); dd += float(df.square().sum()); ad += float((af * df).sum())
    na, nd = math.sqrt(aa), math.sqrt(dd)
    raw = math.sqrt(max(0., aa + dd + 2 * ad))
    cosine = max(-1., min(1., ad / (na * nd))) if na > epsilon and nd > epsilon else None
    reason = None
    if cosine is None:
        wa = wd = 1.; reason = 'zero_task_gradient'
    elif 2 + 2 * cosine <= epsilon:
        wa = wd = 1.; reason = 'opposite_task_directions'
    else:
        common = raw / math.sqrt(2 + 2 * cosine)
        wa, wd = common / na, common / nd
    composed = math.sqrt(max(0., wa * wa * aa + wd * wd * dd + 2 * wa * wd * ad))
    return dict(planning_coefficient=wa, execution_coefficient=wd,
        planning_norm=na, execution_norm=nd, cosine=cosine, ordinary_sum_norm=raw,
        composed_norm=composed, fallback=reason,
        planning_dot_composed=wa * aa + wd * ad, execution_dot_composed=wa * ad + wd * dd)


@torch.no_grad()
def first_adam_delta(parameter, gradient, *, learning_rate, clip_coefficient, epsilon=1e-8):
    """FP32 first AdamW step with zero moments/weight decay, as a diagnostic.

    Return actual representable parameter difference. A directional derivative
    against this vector is a local proxy, not a measured finite-loss decrease.
    """
    if parameter.dtype != torch.float32 or learning_rate <= 0 or not 0 < clip_coefficient <= 1:
        raise ValueError('Use FP32 parameters and a positive bounded first-step scale.')
    g = gradient.float() * clip_coefficient
    next_parameter = parameter - learning_rate * g / (g.abs() + epsilon)
    return next_parameter - parameter
