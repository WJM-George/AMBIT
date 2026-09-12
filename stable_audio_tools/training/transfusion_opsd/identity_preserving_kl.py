"""Stopped-teacher KL with its analytic probability-difference gradient.

At identical student/teacher logits both value and gradient are exactly zero.
This avoids cancellation in a log-softmax backward at a preservation fixed
point; it does not limit an optimizer step or certify preserved behavior.
"""
from __future__ import annotations

import math

import torch


class _StoppedTeacherKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, allowed, temperature):
        ctx.save_for_backward(student, teacher.detach(), allowed)
        ctx.temperature = temperature
        lp = (student.double()/temperature).masked_fill(~allowed, -torch.inf).log_softmax(-1)
        lq = (teacher.detach().double()/temperature).masked_fill(~allowed, -torch.inf).log_softmax(-1)
        delta = lq.masked_fill(~allowed, 0)-lp.masked_fill(~allowed, 0)
        return (lq.exp()*delta).sum(-1)

    @staticmethod
    def backward(ctx, upstream):
        student, teacher, allowed = ctx.saved_tensors
        temperature = ctx.temperature
        # Identical inputs take the same softmax path; subtraction then has an
        # exact zero fixed point. Recompute from student to support gradgrad.
        p = (student.double()/temperature).masked_fill(~allowed, -torch.inf).softmax(-1)
        q = (teacher.double()/temperature).masked_fill(~allowed, -torch.inf).softmax(-1)
        gradient = upstream[..., None]*(p-q)/temperature
        return gradient.to(student.dtype), None, None, None


def identity_preserving_kl(student, teacher, allowed=None, *, temperature=1.):
    """Mean KL(q_teacher || p_student), with a stopped teacher and legal mask."""
    if student.shape != teacher.shape or student.ndim < 1 or not student.is_floating_point() or not teacher.is_floating_point():
        raise ValueError('Require aligned floating-point logits.')
    if student.device != teacher.device or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('Require a common device and positive temperature.')
    if allowed is None:
        allowed = torch.isfinite(teacher)
    if (allowed.shape != student.shape or allowed.dtype != torch.bool or allowed.device != student.device
            or not allowed.any(-1).all() or not torch.isfinite(student[allowed]).all()
            or not torch.isfinite(teacher[allowed]).all()):
        raise ValueError('Require a nonempty common finite legal support per row.')
    return _StoppedTeacherKL.apply(student, teacher.detach(), allowed, float(temperature)).mean().float()


def native_identity_head_retention(output, teacher):
    """Same complete-head averaging as native_head_retention_loss."""
    families = []
    for family in ('inventory', 'qualitative'):
        if set(output[family]) != set(teacher[family]):
            raise ValueError('The native retention head support changed.')
        terms = [identity_preserving_kl(output[family][name], target, torch.isfinite(target))
                 for name, target in teacher[family].items()]
        families.append(torch.stack(terms).mean())
    return torch.stack(families).mean()
