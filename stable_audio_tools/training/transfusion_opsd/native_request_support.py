"""Training-side correction of explicit request constraints at native AR sites.

Grammar support and request compatibility are different. The student softmax
still spans every grammatical action, so it receives gradients that reduce
known-incompatible probability. No annotation or mask is used at inference.
This constraint teacher does not assert unmeasured execution quality.
"""
from __future__ import annotations

import torch

from ...data.model_sceneplan import MODEL_SAMPLE_RATE, VAE_HOP_SAMPLES
from ...data.sceneplan_generation_ar_natural_constraints import validate_requirements, _numeric


def requested_duration_support(codec, request, requirements, prefix, legal_ids):
    """Return compatibility at the original scene-duration site, or None.

    Only explicit scene duration/range annotations are consumed. Evidence must
    be a literal request span. This checks traceability, not annotator truth;
    callers remain responsible for faithful natural-language annotations.
    """
    validate_requirements(request, requirements)
    constraints = [c for c in requirements.get('scene', ())
                   if c['op'] == 'duration_range'
                   or (c['op'] == 'numeric' and c['field'] == 'duration_sec')]
    if not constraints:
        return None
    expected = (codec.bos_id, codec.token_to_id['<duration_frames>'])
    if tuple(prefix) != expected or tuple(legal_ids) != tuple(sorted(codec.allowed_next_ids(expected))):
        raise ValueError('Explicit duration correction requires the full original native duration site.')
    compatible = []
    for token in legal_ids:
        seconds = codec.frame_ids.index(token) * VAE_HOP_SAMPLES / MODEL_SAMPLE_RATE
        compatible.append(all(_numeric(seconds, c['value'], 'duration_sec') if c['op'] == 'numeric'
                              else c['min'] <= seconds <= c['max'] for c in constraints))
    if not any(compatible):
        raise ValueError('Requested duration constraints have no executable codec support.')
    return tuple(compatible)


class _RequestConditionedKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, reference, compatible):
        ctx.save_for_backward(student, reference.detach(), compatible)
        lp = student.double().log_softmax(-1)
        lq = reference.detach().double().masked_fill(~compatible, -torch.inf).log_softmax(-1)
        return (lq.exp() * (lq.masked_fill(~compatible, 0) - lp)).sum(-1)

    @staticmethod
    def backward(ctx, upstream):
        student, reference, compatible = ctx.saved_tensors
        p = student.double().softmax(-1)
        q = reference.double().masked_fill(~compatible, -torch.inf).softmax(-1)
        return (upstream[..., None] * (p - q)).to(student.dtype), None, None


def request_conditioned_kl(student, reference, compatible):
    """KL(pi_reference(. | compatible) || pi_student) on full grammar support.

    The stopped teacher keeps relative prior probabilities inside the allowed
    set; it does not select one hidden numerical ground truth. Unlike a KL that
    masks both distributions, this penalizes student mass outside that set.
    It is auxiliary request correction, not measured execution preference.
    """
    if (student.shape != reference.shape or student.shape != compatible.shape or student.ndim < 1
            or not student.is_floating_point() or not reference.is_floating_point()
            or compatible.dtype != torch.bool or student.device != reference.device
            or student.device != compatible.device or not compatible.any(-1).all()
            or not torch.isfinite(student).all() or not torch.isfinite(reference).all()):
        raise ValueError('Require aligned finite full-support logits and nonempty boolean compatible sets.')
    return _RequestConditionedKL.apply(student, reference.detach(), compatible).mean().float()
