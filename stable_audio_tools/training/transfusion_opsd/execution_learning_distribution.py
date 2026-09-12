"""Explicit stopped plan coverage and attainable direction-gain preconditions.

These utilities define a training distribution and a protocol check. Neither
certifies teacher quality, student improvement, or unseen-request protection.
"""
from __future__ import annotations

import math

import torch


def covered_execution_weights(planning, coverage, supervised, *, coverage_fraction):
    """Mix full legal-support probabilities, retaining all unsupported mass.

    ``supervised`` identifies plans with verified repair or retention targets.
    The caller still checks each plan/noise target's qualification. Missing
    targets receive zero loss weight; remaining weights are not renormalized.
    Both distributions stop gradients for a fixed teacher/update block.
    """
    if (planning.ndim != 1 or planning.numel() == 0
            or coverage.shape != planning.shape or supervised.shape != planning.shape
            or supervised.dtype != torch.bool
            or coverage.device != planning.device or supervised.device != planning.device):
        raise ValueError('Aligned full-support distributions and Boolean availability are required.')
    if not math.isfinite(coverage_fraction) or not 0 <= coverage_fraction <= 1:
        raise ValueError('Declare a coverage fraction in [0, 1].')
    fixed = []
    for distribution in (planning, coverage):
        if (not distribution.is_floating_point() or not torch.isfinite(distribution).all()
                or (distribution < 0).any()):
            raise ValueError('Use finite, nonnegative probabilities, not logits.')
        value = distribution.detach().double()
        if not torch.isclose(value.sum(), value.new_tensor(1.), rtol=0, atol=1e-6):
            raise ValueError('Supply normalized probabilities over the complete declared support.')
        fixed.append(value)
    mixed = (1 - coverage_fraction) * fixed[0] + coverage_fraction * fixed[1]
    applied = torch.where(supervised, mixed, torch.zeros_like(mixed))
    return dict(distribution=mixed, supervised_weights=applied,
                unsupported_mass=mixed[~supervised].sum())


def require_direction_gain_headroom(baseline_failure_rates, *, minimum_mean_gain):
    """Reject a mean direction-gain gate above its mathematical upper bound.

    Direction failure is in [0, 1] and cannot be lower than zero. A reachable
    gate still need not be attainable by any model; this is not a power test.
    No threshold or result is changed automatically.
    """
    values = tuple(float(x) for x in baseline_failure_rates)
    if not values or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError('Use a nonempty declared panel of failure rates in [0, 1].')
    if not math.isfinite(minimum_mean_gain) or minimum_mean_gain <= 0:
        raise ValueError('Declare a finite positive mean improvement requirement.')
    maximum = math.fsum(values) / len(values)
    if minimum_mean_gain > maximum:
        raise ValueError(f'Unreachable direction-gain gate: {minimum_mean_gain:g} exceeds {maximum:g}.')
    return dict(baseline_failure_rates=values, maximum_possible_mean_gain=maximum,
                minimum_mean_gain=minimum_mean_gain, mathematical_headroom_available=True)
