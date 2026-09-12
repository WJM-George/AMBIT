"""Stopped signed velocity teachers, with explicit reference anchoring.

For a fixed anchor b, endpoint velocity u and coefficient a in [-1, 1],
the target b + a * (u - b) has the same student gradient as
    a * MSE(v, u) + (1 - a) * MSE(v, b).
This is an algebraic instance of RVM / DiffusionNFT, not a new efficacy
theorem or a likelihood estimator. Sampling, reward eligibility, normal
generation validation and cross-branch credit remain the caller's duties.

References: https://arxiv.org/html/2608.23664
            https://arxiv.org/html/2509.16117v2
"""
from __future__ import annotations

import math
from numbers import Real

import torch


def execution_teacher_coefficients(probabilities, *, has_qualified_positive: bool):
    """A matched positive/centered comparison on one empirical plan group.

    The caller must establish positive eligibility from actual request,
    content and execution evidence. Agreement or high relative reward alone
    does not establish a qualified positive. With no such positive, both
    teachers reduce to reference retention. Uncertain sample evidence must
    be handled by the caller before forming the empirical group.
    """
    if (not torch.is_tensor(probabilities) or probabilities.ndim != 1 or probabilities.numel() < 2
            or not probabilities.is_floating_point() or not torch.isfinite(probabilities).all()
            or (probabilities < 0).any() or not isinstance(has_qualified_positive, bool)
            or not torch.isclose(probabilities.detach().double().sum(),
                torch.tensor(1., device=probabilities.device, dtype=torch.float64), atol=1e-6, rtol=0)):
        raise ValueError('Require one finite normalized empirical group and explicit positive eligibility.')
    weights=probabilities.detach().double()
    weights=weights/weights.sum()
    if not has_qualified_positive:
        return dict(positive=torch.zeros_like(weights),signed=torch.zeros_like(weights),active=False)
    scale=weights.max()
    signed=torch.zeros_like(weights) if torch.equal(weights,weights[0].expand_as(weights)) else (weights-weights.mean())/scale
    return dict(positive=weights/scale,signed=signed,active=True)


def anchored_velocity_teacher(reference, endpoint, coefficients, mask, *, max_rms_delta=None):
    """Build one stopped B,C,T teacher at the same declared model state.

    Positive coefficients move toward an endpoint; negative coefficients
    reflect away; zero preserves the reference prediction. Optional RMS
    clipping changes the coefficient and therefore the training objective;
    the returned effective coefficient records that change. It guarantees
    neither a bounded parameter update nor improved final audio.

    The caller must bind reference to a frozen round executor. Passing the
    changing student's prediction at every minibatch silently removes the
    intended fixed reference, despite this function stopping its gradients.
    """
    if (not all(torch.is_tensor(x) for x in (reference,endpoint,coefficients,mask))
            or reference.ndim != 3 or min(reference.shape) <= 0 or endpoint.shape != reference.shape
            or coefficients.shape != (reference.shape[0],) or mask.shape != (reference.shape[0],reference.shape[-1])
            or mask.dtype != torch.bool or not mask.any(-1).all()
            or any(x.device != reference.device for x in (endpoint,coefficients,mask))
            or not all(x.is_floating_point() and torch.isfinite(x).all() for x in (reference,endpoint,coefficients))
            or (coefficients.abs() > 1+1e-7).any()
            or (max_rms_delta is not None and (not isinstance(max_rms_delta, Real)
                or isinstance(max_rms_delta, bool) or not math.isfinite(max_rms_delta) or max_rms_delta <= 0))):
        raise ValueError('Require aligned finite velocities, bounded per-example coefficients and nonempty masks.')
    anchor=reference.detach().float().masked_fill(~mask[:,None],0)
    clean=endpoint.detach().float().masked_fill(~mask[:,None],0)
    a=coefficients.detach().float().clamp(-1,1)
    delta=(clean-anchor)*a[:,None,None]
    denominator=mask.sum(-1)*reference.shape[1]
    rms=(delta.double().square().sum((1,2))/denominator).sqrt()
    scale=torch.ones_like(rms)
    if max_rms_delta is not None:
        scale=(max_rms_delta/rms.clamp_min(torch.finfo(rms.dtype).tiny)).clamp(max=1)
    effective=a*scale.to(a.dtype)
    target=(anchor+(clean-anchor)*effective[:,None,None]).masked_fill(~mask[:,None],0)
    actual_rms=((target-anchor).double().square().sum((1,2))/denominator).sqrt()
    if not torch.isfinite(target).all():
        raise ValueError('Velocity teacher overflowed after interpolation.')
    return dict(target=target,coefficients=a,effective_coefficients=effective,
        requested_rms_delta=rms,actual_rms_delta=actual_rms,scale=scale)
