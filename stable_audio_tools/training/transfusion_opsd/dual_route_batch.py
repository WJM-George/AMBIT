"""A fresh, independently populated pair of objectives for a shared model.

An AR teacher and a diffusion teacher need not exist on the same example.
Every transaction still requires both objectives, and the existing actual
Adam-displacement projection and execution validation remain mandatory.
No audio, EVENT, or task-specific reward is assumed by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class RouteLossTerm:
    sample_id: str
    model_version: int
    loss: Callable[[], torch.Tensor]
    kind: str
    weight: float = 1.

    def __post_init__(self):
        if (not self.sample_id or self.model_version < 0 or not callable(self.loss)
                or not self.kind or not math.isfinite(self.weight) or self.weight <= 0):
            raise ValueError('route terms require fresh identified evidence and a positive finite weight')


def dual_route_loss_closure(ar_terms, dit_terms, *, model_version, checkpoint_terms=True,
        ar_denominator=None, dit_denominator=None):
    """Mean of weighted losses, with independent per-route eligibility.

The denominator is the number of eligible terms, not sum(weights). This is
deliberate for importance weights q(action)/p(action): normalizing by their
realized sum would change that estimator. An explicit denominator counts all
sampled examples when unqualified targets contribute zero. Terms and detached targets are fixed
for the entire transaction. Non-reentrant checkpointing recomputes each term
for autograd.grad without retaining all examples' transformer activations.
    """
    routes = tuple(tuple(values) for values in (ar_terms, dit_terms))
    if any(not values for values in routes):
        raise ValueError('a shared transaction requires nonempty AR and DiT supervision')
    if any(term.model_version != model_version for values in routes for term in values):
        raise ValueError('stale teacher: recollect both route pools at the current model version')
    if any(len({term.sample_id for term in values}) != len(values) for values in routes):
        raise ValueError('duplicate sample in a route pool would silently change its weight')
    denominators = tuple(len(values) if denominator is None else denominator
        for values, denominator in zip(routes, (ar_denominator, dit_denominator)))
    if any(not isinstance(n, int) or isinstance(n, bool) or n < len(values)
            for n, values in zip(denominators, routes)):
        raise ValueError('route denominators must count at least every eligible term')

    def closure():
        need_graph = torch.is_grad_enabled()
        losses = []
        with torch.enable_grad():
            for terms, denominator in zip(routes, denominators):
                values = []
                for term in terms:
                    value = (checkpoint(term.loss, use_reentrant=False, preserve_rng_state=True)
                        if checkpoint_terms and need_graph else term.loss())
                    if value.ndim != 0:
                        raise ValueError('each route term must reduce to a scalar')
                    values.append(value * term.weight)
                losses.append(torch.stack(values).sum() / denominator)
        return (*losses, None) if need_graph else (*(value.detach() for value in losses), None)

    closure.evidence = {'model_version': model_version, 'checkpoint_terms': checkpoint_terms,
        'reduction': 'weighted sum divided by declared sample count per route',
        'denominators': list(denominators),
        'ar': [dict(sample_id=t.sample_id, kind=t.kind, weight=t.weight) for t in routes[0]],
        'dit': [dict(sample_id=t.sample_id, kind=t.kind, weight=t.weight) for t in routes[1]],
        'execution_coupled': any(t.kind == 'execution_teacher' for t in routes[0])}
    return closure
