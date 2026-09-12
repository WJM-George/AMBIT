"""Fit stopped flow-velocity targets stored as native clean predictions."""
from __future__ import annotations

import math

import torch


def native_velocity_distillation_loss(state, time, velocity, target_clean, mask, *, fixed_mse_scale):
    """Mean per-example velocity error; output length cannot change its weight.

    Direct velocity fitting avoids an implicit t-squared weight from clean
    regression. Collection-time state/time and teacher predictions stop
    gradients. The student velocity keeps its entire computation graph.
    """
    if (state.ndim != 3 or velocity.shape != state.shape or target_clean.shape != state.shape
            or time.shape != (state.shape[0],) or mask.shape != (state.shape[0],state.shape[-1])
            or mask.dtype != torch.bool or not mask.any(-1).all()
            or not all(x.device == velocity.device for x in (state,time,target_clean,mask))
            or not all(torch.isfinite(x).all() for x in (state,time,velocity,target_clean))
            or not ((time>0)&(time<=1)).all()
            or not math.isfinite(fixed_mse_scale) or fixed_mse_scale<=0):
        raise ValueError('Require aligned finite student queries, native masks and a positive fixed scale.')
    target=(state.detach().float()-target_clean.detach().float())/time.detach()[:,None,None]
    error=(velocity.float()-target).square().masked_fill(~mask[:,None],0)
    per_example=error.sum((1,2))/(mask.sum(-1)*state.shape[1]*fixed_mse_scale)
    return per_example.mean()


def stratified_native_queries(update_index):
    """A fixed four-stratum rotation; 25 updates cover all100 native indices."""
    if type(update_index) is not int or update_index<0:
        raise ValueError('The update index must be a nonnegative integer.')
    offset=update_index%25
    return tuple(offset+25*i for i in range(4))
