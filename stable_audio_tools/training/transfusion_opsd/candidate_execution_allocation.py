"""Allocate an explicit execution-learning budget inside verified plan support.

This changes the conditional training objective. It does not silently preserve
the full-support policy objective or imply efficient unbiased resampling.
"""
from __future__ import annotations

import math

import torch

from .execution_learning_distribution import covered_execution_weights


def candidate_execution_allocation(planning, coverage, available, *, coverage_fraction,
                                   supervised_budget):
    if (planning.ndim != 1 or coverage.shape != planning.shape or available.shape != planning.shape
            or available.dtype != torch.bool or available.device != planning.device
            or coverage.device != planning.device or not available.any()
            or not math.isfinite(supervised_budget) or not 0 < supervised_budget <= 1):
        raise ValueError('Declare aligned candidate support and a positive budget no greater than one.')
    # Reuse the probability-contract checks without altering the original inputs.
    covered_execution_weights(planning, coverage, available, coverage_fraction=coverage_fraction)
    prior = planning.detach().double()
    cover = coverage.detach().double()
    measured = prior.masked_fill(~available, 0)
    mass = measured.sum()
    if mass <= 0 or cover[~available].sum() > 1e-12:
        raise ValueError('The teacher needs measured mass and coverage must stay on available targets.')
    conditional = measured / mass
    mixed = covered_execution_weights(conditional, cover, available, coverage_fraction=coverage_fraction)
    return dict(conditional_teacher=conditional, conditional_distribution=mixed['distribution'],
        supervised_weights=mixed['supervised_weights'] * supervised_budget,
        measured_teacher_mass=mass, unassigned_budget=1. - supervised_budget,
        supervised_budget=float(supervised_budget))
