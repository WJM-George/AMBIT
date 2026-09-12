"""Hard native conditions with a declared surrogate gradient to AR decisions.

This is a straight-through estimator, not differentiation of a discrete
decoder. Only aligned, request-admissible alternatives may be supplied by the
caller. Discrete masks, sequence lengths and negative CFG inputs never mix.
The finite support is conditional; it is not the full native policy objective.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


def hard_native_condition(
    hard: Mapping[str, torch.Tensor | None],
    alternatives: Sequence[Mapping[str, torch.Tensor | None]],
    logits: torch.Tensor,
    *,
    hard_index: int,
    differentiable_keys: Sequence[str],
    temperature: float = 1.,
    connect_decisions: bool = True,
) -> dict:
    """Keep hard forward and encoder gradients; add only the logit derivative.

    Alternative encodings are stopped. Thus the usual hard-condition encoder
    path occurs exactly once, while the extra path reaches the native logits.
    A detached control has exactly the same values and ordinary gradients.
    """
    n = len(alternatives)
    if (n < 2 or logits.ndim != 1 or logits.numel() != n
            or not logits.is_floating_point() or not torch.isfinite(logits).all()
            or not math.isfinite(temperature) or temperature <= 0
            or isinstance(hard_index, bool) or not 0 <= hard_index < n):
        raise ValueError('Require finite logits for at least two aligned alternatives.')
    keys = set(differentiable_keys)
    if not keys or not keys <= hard.keys() or len(keys) != len(differentiable_keys):
        raise ValueError('Explicitly declare unique continuous conditioning keys.')
    if any('mask' in key or 'negative' in key for key in keys):
        raise ValueError('Masks and negative CFG inputs cannot carry this surrogate.')
    for alt in alternatives:
        if set(alt) != set(hard):
            raise ValueError('Native conditioning keys must match exactly.')
        for key, value in hard.items():
            other = alt[key]
            if value is None:
                if other is not None or key in keys:
                    raise ValueError('Optional native conditioning must remain identical.')
                continue
            if (not isinstance(value, torch.Tensor) or not isinstance(other, torch.Tensor)
                    or value.shape != other.shape or value.dtype != other.dtype
                    or value.device != other.device):
                raise ValueError('Reject changes to native condition geometry or dtype.')
            if key in keys:
                if (not value.is_floating_point() or value.device != logits.device
                        or not torch.isfinite(value).all() or not torch.isfinite(other).all()):
                    raise ValueError('The continuous path requires finite aligned floating tensors.')
            elif not torch.equal(value, other):
                raise ValueError('Undeclared inputs, masks and negative CFG must stay identical.')
    for key, value in hard.items():
        if value is not None and not torch.equal(value, alternatives[hard_index][key]):
            raise ValueError('Hard forward must be an actual member of the native alternatives.')
    result = dict(hard)
    if connect_decisions:
        probabilities = (logits / temperature).softmax(0)
        for key in keys:
            value = hard[key]
            weights = probabilities.to(value.dtype)
            soft = sum(weights[i] * alt[key].detach() for i, alt in enumerate(alternatives))
            # Parentheses are intentional: subtract identical values before
            # adding to the native condition, preserving its forward exactly.
            result[key] = value + (soft - soft.detach())
    return result
