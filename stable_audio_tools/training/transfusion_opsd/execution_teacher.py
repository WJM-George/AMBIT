"""Finite execution tilt and independent paired qualification.

Reusable controls for the EVENT teacher studies. A score tilt by itself does
not protect content; callers must qualify it before using it as supervision.
These functions do not assume that a decision is an ordinary AR token.
"""
import math

import torch

from .objectives import legal_log_probs


def execution_teacher(logits, legal, action_ids, values, *, ar_temperature,
                      temperature, strength):
    """Return finite logits for q=(1-lambda)q_task+lambda*q_execution.

    All unqueried actions keep exactly their original probability analytically.
    The returned logits use the existing AR temperature, not the score temperature.
    """
    if not 0 <= strength <= 1 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid execution teacher temperature/strength")
    ids = torch.as_tensor(action_ids, dtype=torch.long, device=logits.device)
    scores = torch.as_tensor(values, dtype=torch.float64, device=logits.device)
    if (logits.ndim != 1 or len(ids) < 2 or len(ids.unique()) != len(ids)
            or scores.shape != ids.shape or not torch.isfinite(scores).all()
            or (ids < 0).any() or (ids >= len(logits)).any() or not legal[ids].all()):
        raise ValueError("teacher needs distinct, legal, evaluated actions")
    base_log = legal_log_probs(logits.detach(), legal, ar_temperature).double()
    log_mass = torch.logsumexp(base_log[ids], 0)
    conditional_log = base_log[ids] - log_mass
    adjusted = torch.log_softmax(conditional_log + scores / temperature, 0)
    if strength == 0:
        mixed = conditional_log
    elif strength == 1:
        mixed = adjusted
    else:
        mixed = torch.logaddexp(conditional_log + math.log1p(-strength), adjusted + math.log(strength))
    # A common shift is unnecessary; modify only the queried finite logits.
    output = logits.detach().clone()
    output[ids] += ((mixed - conditional_log) * ar_temperature).to(output.dtype)
    return output, conditional_log.exp(), mixed.exp(), float(log_mass.exp())

def qualify_execution_teacher(base, target, construction, validation, *, min_gain,
                              min_utility, cost_limits):
    """Paired empirical qualification, not a population confidence theorem."""
    values = [*construction, *validation]
    if not construction or not validation or any(len(row) != len(base) for row in values):
        raise ValueError("independent construction and validation matrices required")
    cost_keys = values[0][0].costs.keys()
    if any(score.costs.keys() != cost_keys for row in values for score in row):
        raise ValueError("counterfactual protection coverage changed")
    if not set(cost_limits).issubset(cost_keys):
        raise ValueError("declared execution protection is missing")
    if any(s.utility < min_utility or any(s.costs[k] > v for k, v in cost_limits.items())
           for row in values for s in row):
        return False, {"reason": "execution_floor_or_protection_failed"}
    delta = target.double().cpu() - base.double().cpu()
    gains = []
    for row in values:
        gain = float(delta @ torch.tensor([s.utility for s in row], dtype=torch.float64))
        gains.append(gain)
        if gain <= min_gain:
            return False, {"reason": "paired_gain_not_reproduced", "paired_gains": gains}
        for key in cost_keys:
            change = float(delta @ torch.tensor([s.costs[key] for s in row], dtype=torch.float64))
            if change > 1e-12:
                return False, {"reason": "execution_cost_regression", "cost": key}
    return True, {"reason": "paired_execution_qualified", "paired_gains": gains}
