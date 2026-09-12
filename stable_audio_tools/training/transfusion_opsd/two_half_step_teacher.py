"""Bounded same-plan RF targets from two frozen half steps at a visited state.

This is a numerical distillation proposal, not a reward or teacher-eligibility
test. Callers must validate the resulting continuation before training on it.
No source, plan or inference-time conditioning is replaced by this operator.
"""
from __future__ import annotations

import math
import torch


@torch.no_grad()
def bounded_two_half_step_target(state, time, step, velocity, mask, *,
                                 maximum_relative_velocity_change=0.1,
                                 anchor_velocity=None):
    """Return a detached velocity whose Euler step approximates two half steps.

    For descending RF time, z_half=z-step*v0/2 and v_target=(v0+v_half)/2.
    Bound its change from v0 independently per example over the valid region.
    Teacher construction may use extra compute; normal inference stays fixed.
    """
    if (state.ndim != 3 or not state.is_floating_point() or not torch.isfinite(state).all()
            or time.shape != state.shape[:1] or time.device != state.device
            or not time.is_floating_point() or not torch.isfinite(time).all()
            or mask.shape != (state.shape[0], state.shape[2]) or mask.dtype != torch.bool
            or mask.device != state.device or not mask.any(-1).all()
            or not math.isfinite(step) or step <= 0 or bool((time < step).any())
            or not math.isfinite(maximum_relative_velocity_change)
            or maximum_relative_velocity_change <= 0):
        raise ValueError('Require finite RF states, positive valid time step and aligned nonempty masks.')
    keep = mask[:, None]
    z = state.detach() * keep

    def checked(value):
        if (not isinstance(value, torch.Tensor) or value.shape != z.shape
                or value.device != z.device or not value.is_floating_point()
                or not torch.isfinite(value).all()):
            raise ValueError('Teacher velocity must be a finite aligned prediction.')
        return value.detach().to(z.dtype) * keep

    v0 = checked(velocity(z, time) if anchor_velocity is None else anchor_velocity)
    midpoint = (z - (step / 2) * v0) * keep
    vmid = checked(velocity(midpoint, time - step / 2))
    delta = (vmid - v0) / 2
    norm0 = v0.double().flatten(1).norm(dim=1)
    norm_delta = delta.double().flatten(1).norm(dim=1)
    limit = maximum_relative_velocity_change * norm0
    scale = (limit / norm_delta.clamp_min(1e-30)).clamp(max=1.).to(z.dtype)
    scale = torch.where(norm_delta == 0, torch.ones_like(scale), scale)
    target = v0 + scale[:, None, None] * delta
    return dict(anchor_velocity=v0, target_velocity=target,
                next_state=(z - step * target) * keep, midpoint=midpoint,
                repair_scale=scale, unbounded_repair_norm=norm_delta,
                anchor_norm=norm0, velocity_radius=limit)
