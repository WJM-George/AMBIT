"""Original paired-data RF retention, distinct from on-policy repair targets.

The caller must verify the audio/latent and ScenePlan pair. This module checks
the tensor contract; it does not certify semantics, transcripts or geometry.
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch


def paired_rf_example(clean, noise, time, mask):
    """Fresh straight-path state and velocity with no teacher/data gradients.

    t=0 is valid here: unlike repair reconstruction, no division by t occurs.
    Padded coordinates are zero and never acquire supervised velocity weight.
    """
    if (clean.ndim != 3 or noise.shape != clean.shape or time.shape != (clean.shape[0],)
            or mask.shape != (clean.shape[0], clean.shape[-1]) or mask.dtype != torch.bool
            or not mask.any(-1).all() or not all(x.device == clean.device for x in (noise, time, mask))
            or not all(x.is_floating_point() and torch.isfinite(x).all() for x in (clean, noise, time))
            or not ((time >= 0) & (time <= 1)).all()):
        raise ValueError('Require a finite aligned RF pair, t in [0,1], and nonempty boolean masks.')
    z0, eps, t = clean.detach().float(), noise.detach().float(), time.detach().float()[:, None, None]
    return ((1-t)*z0+t*eps).masked_fill(~mask[:, None], 0), (eps-z0).masked_fill(~mask[:, None], 0)


def paired_rf_loss(velocity, target, mask, *, fixed_mse_scale=1.):
    """Mean per-example supervised velocity MSE, with stopped data targets."""
    if (velocity.ndim != 3 or target.shape != velocity.shape or mask.shape != (velocity.shape[0], velocity.shape[-1])
            or mask.dtype != torch.bool or not mask.any(-1).all()
            or target.device != velocity.device or mask.device != velocity.device
            or not all(x.is_floating_point() and torch.isfinite(x).all() for x in (velocity, target))
            or not math.isfinite(fixed_mse_scale) or fixed_mse_scale <= 0):
        raise ValueError('Require finite aligned velocities, masks and positive fixed scale.')
    squared = (velocity.float()-target.detach().float()).square().masked_fill(~mask[:, None], 0)
    return (squared.sum((1, 2))/(mask.sum(-1)*velocity.shape[1]*fixed_mse_scale)).mean()


def paired_rf_velocity_function(bundle, condition):
    """Original positive conditional RF model; no inference CFG or rescaling.

    Preserve native conditioning/DiT autocast and gradients to the existing
    conditioner/shared parameters. This retention path has no auxiliary bridge;
    the separate on-policy repair objective trains that experimental interface.
    No mutable changes are made to the normal sampling settings on the bundle.
    """
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if bundle.device.type == 'cuda' else nullcontext()
    with amp():
        positive = bundle.diffusion.conditioner(condition.positive, bundle.device)
        inputs = bundle.diffusion.get_conditioning_inputs(positive)
        dtype = next(bundle.diffusion.model.parameters()).dtype
        inputs = {key: value.to(dtype) if torch.is_tensor(value) else value for key, value in inputs.items()}

    def velocity(z, t):
        if z.shape[-1] != condition.mask.shape[-1]:
            raise ValueError('Paired latent and compiled original-plan lengths must match.')
        with amp():
            # DiTWrapper requires this dispatch flag even at CFG1. Guidance
            # remains disabled by cfg_scale=1, with no negative inputs.
            return bundle.diffusion.model(z, t, **inputs, cfg_scale=1., batch_cfg=True,
                rescale_cfg=False, scale_phi=0., apg_scale=0., cfg_dropout_prob=0., padding_mask=condition.mask)

    return velocity
