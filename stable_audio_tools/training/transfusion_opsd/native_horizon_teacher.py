"""Same-coordinate teacher queries across a verified shorter native horizon."""
from __future__ import annotations

import torch

from .native_condition_teacher import bounded_condition_velocity_target


@torch.no_grad()
def shorter_horizon_velocity_target(student_state, time, reference_velocity,
        shorter_velocity, *, student_mask, shorter_mask, maximum_relative_rms=.05):
    """Query the actual student's prefix at the same flow time, without warping.

    The caller binds the shorter model condition and verifies its legal plan.
    No future teacher trajectory or separately generated latent enters this
    query. The final unmatched frames keep the full-horizon model prediction.
    The returned target is training-only and requires output qualification.
    """
    if (student_state.ndim != 3 or reference_velocity.shape != student_state.shape
            or student_mask.shape != (student_state.shape[0],student_state.shape[-1])
            or shorter_mask.ndim != 2 or shorter_mask.shape[0] != student_state.shape[0]
            or student_mask.dtype != torch.bool or shorter_mask.dtype != torch.bool
            or not student_mask.all() or not shorter_mask.all()
            or not 0 < shorter_mask.shape[-1] < student_state.shape[-1]
            or time.shape != (student_state.shape[0],) or not torch.isfinite(time).all()
            or not ((time>0)&(time<=1)).all()
            or any(x.device != student_state.device for x in (time,reference_velocity,student_mask,shorter_mask))
            or not torch.isfinite(student_state).all()):
        raise ValueError('Require native unpadded horizons with a strictly shorter aligned prefix and valid current time.')
    frames=shorter_mask.shape[-1]
    alternate=shorter_velocity(student_state[...,:frames].contiguous(),time)
    if (alternate.shape != (*student_state.shape[:-1],frames)
            or alternate.dtype != reference_velocity.dtype or alternate.device != student_state.device
            or not torch.isfinite(alternate).all()):
        raise ValueError('Shorter executor must return finite native velocities on the exact queried prefix.')
    aligned=reference_velocity.detach().clone()
    aligned[...,:frames]=alternate
    target,receipt=bounded_condition_velocity_target(reference_velocity,aligned,student_mask,
        maximum_relative_rms=maximum_relative_rms)
    if not torch.equal(target[...,frames:],reference_velocity[...,frames:]):
        raise RuntimeError('The unmatched suffix must retain the current full-horizon prediction.')
    return target,dict(receipt,student_frames=student_state.shape[-1],teacher_frames=frames,
        correspondence='same latent time coordinates; prefix slice only, no interpolation or channel transform',
        unmatched_suffix_exact=True,teacher_queries_current_student_prefix=True)
