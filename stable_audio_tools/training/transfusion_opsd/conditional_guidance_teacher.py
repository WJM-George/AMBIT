"""Stopped, bounded clean targets from same-state conditioning contrasts.

The caller supplies predictions from one frozen executor at the same state.
This algebra does not qualify a teacher: decoded continuation, requested-word
correctness, and the other capability checks remain the caller's responsibility.
"""
from __future__ import annotations

import math

import torch

from .objectives import clean_prediction, valid_mask


@torch.no_grad()
def conditional_clean_target(state, time, native_velocity, positive_velocity,
        negative_velocity, mask, *, weight=1., relative_radius=.02):
    """Add a conditional velocity contrast, bounded per example in clean space.

    For the current flow convention clean = state - time * velocity, so the
    proposed clean displacement is -time * weight * (positive - negative).
    Padding receives no displacement. Every returned tensor is detached.
    """
    if (not math.isfinite(weight) or weight < 0 or
            not math.isfinite(relative_radius) or not 0 < relative_radius <= 1):
        raise ValueError('Declare finite nonnegative guidance and a bounded positive radius.')
    if state.ndim != 3 or any(v.shape != state.shape for v in
            (native_velocity, positive_velocity, negative_velocity)):
        raise ValueError('All same-state predictions must have matching [batch, channels, time] shapes.')
    if time.ndim != 1 or time.shape[0] != state.shape[0] or not bool(((time > 0) & (time <= 1)).all()):
        raise ValueError('A positive flow time is required for every example.')
    if any(v.device != state.device for v in (time, native_velocity, positive_velocity, negative_velocity, mask)):
        raise ValueError('State, predictions, time and valid mask must share one device.')
    if not all(bool(torch.isfinite(v).all()) for v in
            (state, time, native_velocity, positive_velocity, negative_velocity)):
        raise ValueError('A teacher cannot contain nonfinite values.')
    full = valid_mask(state, mask)
    if not bool(full.flatten(1).any(-1).all()):
        raise ValueError('Every example needs at least one valid frame.')
    anchor = clean_prediction(state.float(), time.float(), native_velocity.float())
    delta = -time.float()[:, None, None] * weight * (positive_velocity.float() - negative_velocity.float())
    delta = delta.masked_fill(~full, 0.)
    anchor_norm = anchor.masked_fill(~full, 0.).flatten(1).norm(dim=1)
    delta_norm = delta.flatten(1).norm(dim=1)
    radius = relative_radius * anchor_norm
    fraction = (radius / delta_norm.clamp_min(1e-12)).clamp(max=1.)
    target = anchor + delta * fraction[:, None, None]
    actual_norm = (target - anchor).masked_fill(~full, 0.).flatten(1).norm(dim=1)
    return dict(anchor=anchor.detach(), positive=target.detach(),
        proposed_norm=delta_norm.detach(), actual_norm=actual_norm.detach(),
        radius=radius.detach(), fraction=fraction.detach())
