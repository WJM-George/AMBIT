"""Frozen native semantic and acoustic retention objectives.

Legacy LAION continuous-fusion code was archived in V67; native FOA-latent
CLAP supplies the semantic score. Penalties alone do not certify output.
"""
from __future__ import annotations
import math
import torch
from .objectives import forward_kl


def semantic_retention_penalty(score, reference_score, *, free_drop=.0025, guard_drop=.005):
    """Allow small score changes and penalize losses before the output guard.

    Zero inside the free interval, one at guard_drop. The frozen reference
    cannot improve jointly with the student to erase a real regression.
    """
    if (not all(math.isfinite(x) for x in (free_drop, guard_drop))
            or not 0 <= free_drop < guard_drop):
        raise ValueError('Require finite 0 <= free_drop < guard_drop.')
    if (score.numel() != 1 or reference_score.numel() != 1 or score.device != reference_score.device
            or not score.is_floating_point() or not reference_score.is_floating_point()
            or not torch.isfinite(score).all() or not torch.isfinite(reference_score).all()):
        raise ValueError('Retention scores must be matching finite floating scalars.')
    drop = reference_score.detach() - score
    return ((drop - free_drop).clamp_min(0) / (guard_drop - free_drop)).square().mean()


def acoustic_distribution_retention(student_logits, reference_logits):
    """Mean frame KL for a separately qualified, frozen same-geometry reference.

    This function cannot decide whether a reference contains correct words.
    The caller must gate references using the pinned transcription evidence.
    No time interpolation or transcript relabeling is performed here.
    """
    if (student_logits.ndim != 3 or student_logits.shape[0] != 1
            or student_logits.shape != reference_logits.shape or min(student_logits.shape) == 0
            or student_logits.device != reference_logits.device
            or not torch.isfinite(student_logits).all() or not torch.isfinite(reference_logits).all()):
        raise ValueError('Acoustic distributions require finite, aligned [1,frames,vocab] logits.')
    return forward_kl(student_logits, reference_logits,
        torch.ones_like(student_logits, dtype=torch.bool), temperature=1.)
