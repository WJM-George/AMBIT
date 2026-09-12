"""Native planning retention and fixed clean-target fitting, without a refiner.

Retention may start at zero. It is a constraint against forgetting, not a
requirement to manufacture a separate AR improvement for every DiT update.
"""
from __future__ import annotations

import math

import torch

from .objectives import clean_prediction, forward_kl


def native_head_teacher(output):
    return {family: {name: value.detach().clone() for name, value in output[family].items()}
        for family in ('inventory', 'qualitative')}


def native_head_retention_loss(output, teacher):
    """Keep complete native inventory/span and qualitative distributions."""
    families = []
    for family in ('inventory', 'qualitative'):
        if set(output[family]) != set(teacher[family]):
            raise ValueError('Native retention head support changed.')
        terms = [forward_kl(output[family][name], target, torch.isfinite(target), temperature=1.)
            for name, target in teacher[family].items()]
        families.append(torch.stack(terms).mean())
    return torch.stack(families).mean()


def native_clean_fit_loss(z, time, velocity, target, mask, *, fixed_mse_scale):
    """Separate per-plan scalar objectives; retain the declared normalizer."""
    if not math.isfinite(fixed_mse_scale) or fixed_mse_scale <= 0:
        raise ValueError('A positive fixed fitting scale is required.')
    if target.shape != z.shape or mask.shape != (z.shape[0], z.shape[-1]) or mask.dtype != torch.bool:
        raise ValueError('Clean fitting must match the actual query geometry.')
    if not mask.any(-1).all() or not torch.isfinite(target).all():
        raise ValueError('Every query requires finite targets and nonempty valid latent support.')
    error = (clean_prediction(z, time, velocity)-target.detach()).square()
    weights = mask[:, None].to(error.dtype)
    return (error*weights).sum()/(weights.sum()*z.shape[1]*fixed_mse_scale)
