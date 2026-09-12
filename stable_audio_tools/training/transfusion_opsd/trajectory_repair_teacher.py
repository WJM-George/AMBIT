"""Stopped velocity targets for transporting an existing endpoint correction.

The correction is a verified VAE-latent difference, never physical FOA axes.
For noise-at-one flow, subtracting delta / start_time from velocity distributes
the nominal endpoint correction across the remaining Euler time interval.
The executor's response can amplify, erase or reverse it: actual continuation
and content validation are mandatory before treating this as a useful teacher.
"""
from __future__ import annotations

import math

import torch


def transport_velocity_target(velocity, endpoint_delta, mask, *, remaining_time,
                              direction=1):
    """Return a detached target in the executor's native output dtype.

    The positive direction denotes the desired endpoint displacement; Euler
    time decreases, hence the minus sign in velocity. No current model state,
    parameter or inference waveform is edited by this function.
    """
    if (velocity.ndim != 3 or endpoint_delta.shape != velocity.shape
            or mask.shape != (velocity.shape[0], velocity.shape[-1])
            or mask.dtype != torch.bool or not mask.any(-1).all()
            or velocity.device != endpoint_delta.device or mask.device != velocity.device
            or not velocity.is_floating_point() or not endpoint_delta.is_floating_point()
            or not torch.isfinite(velocity).all() or not torch.isfinite(endpoint_delta).all()):
        raise ValueError('Require aligned finite latent velocities, corrections and valid masks')
    if (isinstance(remaining_time, bool) or not isinstance(remaining_time, (float, int))
            or not math.isfinite(remaining_time) or not 0 < remaining_time <= 1
            or isinstance(direction, bool) or direction not in (-1, 1)):
        raise ValueError('Declare positive remaining time and a signed correction direction')
    delta = endpoint_delta.detach().float().masked_fill(~mask[:, None], 0)
    target = velocity.detach().float() - direction * delta / remaining_time
    return target.to(velocity.dtype)
