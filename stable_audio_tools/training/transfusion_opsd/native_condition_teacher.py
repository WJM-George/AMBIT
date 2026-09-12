"""Bounded native predictions from an execution-validated alternate plan.

Both predictions must be evaluated at the same noisy student state and time.
An alternate condition's final reward is not enough: the bounded continuation
itself still needs actual output validation before supplying repair targets.
"""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def bounded_condition_velocity_target(reference, alternate, mask, *, maximum_relative_rms=.05):
    """Stop gradients and enforce the bound after native output quantization.

    This modifies a teacher prediction only. It does not edit a student model,
    sampling state, physical FOA axes, or inference waveform. Identical native
    predictions return an exactly identical target. Padding retains reference.
    """
    if (reference.ndim != 3 or alternate.shape != reference.shape
            or reference.dtype != alternate.dtype or reference.device != alternate.device
            or not reference.is_floating_point() or not torch.isfinite(reference).all()
            or not torch.isfinite(alternate).all() or mask.dtype != torch.bool
            or mask.shape != (reference.shape[0], reference.shape[-1])
            or mask.device != reference.device or not mask.any(-1).all()):
        raise ValueError('Require finite native predictions at the same aligned state and valid mask.')
    if (not math.isfinite(maximum_relative_rms) or not 0 < maximum_relative_rms <= .1):
        raise ValueError('Declare a positive relative RMS correction bound no greater than .1.')
    base = reference.detach().float()
    delta = (alternate.detach().float() - base).masked_fill(~mask[:, None], 0)
    count = mask.sum(-1).float() * reference.shape[1]
    rms = lambda x: (x.masked_fill(~mask[:, None], 0).square().sum((1,2))/count).sqrt()
    base_rms, delta_rms = rms(base), rms(delta)
    limit = maximum_relative_rms * base_rms
    scale = (limit/delta_rms.clamp_min(1e-30)).clamp(max=1)
    # Bisection is a numerical feasibility operation, not a reward search.
    # Float/BF16 rounding can move an otherwise clipped target outside its cap.
    target = (base + scale[:,None,None]*delta).to(reference.dtype)
    overshoot = rms(target.float()-base) > limit
    if overshoot.any():
        lo, hi = torch.zeros_like(scale), scale.clone()
        for _ in range(16):
            mid = (lo+hi)/2
            candidate = (base+mid[:,None,None]*delta).to(reference.dtype)
            feasible = rms(candidate.float()-base) <= limit
            lo, hi = torch.where(feasible,mid,lo), torch.where(feasible,hi,mid)
        scale = torch.where(overshoot,lo,scale)
        target = (base+scale[:,None,None]*delta).to(reference.dtype)
    target = torch.where(mask[:,None],target,reference).detach()
    achieved = rms(target.float()-base)
    if not (achieved <= limit).all():
        raise RuntimeError('Native-rounded correction exceeded its declared RMS bound.')
    return target,dict(reference_rms=base_rms.cpu().tolist(),raw_difference_rms=delta_rms.cpu().tolist(),
        accepted_scale=scale.cpu().tolist(),relative_correction=(achieved/base_rms.clamp_min(1e-30)).cpu().tolist(),
        maximum_relative_rms=maximum_relative_rms,native_dtype=str(reference.dtype),
        identity_exact=torch.equal(target,reference),post_rounding_bound_verified=True)
