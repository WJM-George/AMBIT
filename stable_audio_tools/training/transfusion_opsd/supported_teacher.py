"""A local execution comparison must not label unexecuted choices as failures.

For one native categorical decision, redistribute only the reference mass of
the evaluated support. The remaining choices keep their reference targets.
This defines a conservative distillation target, not a policy improvement
guarantee or a differentiable discrete sampler.
"""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SupportedTeacher:
    probabilities: Tensor
    reference_probabilities: Tensor
    support_mass: Tensor
    support_indices: Tensor


def preserve_reference_support_mass(reference_logits: Tensor, support_indices,
                                    conditional_teacher: Tensor) -> SupportedTeacher:
    """Lift a stopped teacher on measured choices to the full native simplex.

    A caller applying a conditional decision-credit objective may weight that
    objective by the returned stopped support mass. That weighting is an
    explicit objective choice, not an unbiased full-policy gradient claim.
    Teachers and support must have been fixed before student optimization.
    """
    if (reference_logits.ndim != 1 or reference_logits.numel() < 2
            or reference_logits.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(reference_logits).all())):
        raise ValueError('Use one finite FP32/FP64 native logit vector.')
    indices = torch.as_tensor(support_indices, device=reference_logits.device)
    if indices.dtype not in (torch.int32, torch.int64):
        raise ValueError('Measured support indices must be integers.')
    indices = indices.long().detach()
    if (indices.ndim != 1 or indices.numel() < 2
            or indices.unique().numel() != indices.numel()
            or int(indices.min()) < 0 or int(indices.max()) >= reference_logits.numel()):
        raise ValueError('Use at least two distinct in-range measured choices.')
    conditional = conditional_teacher.detach().to(reference_logits)
    if (conditional.shape != indices.shape or not bool(torch.isfinite(conditional).all())
            or bool((conditional < 0).any())
            or not torch.isclose(conditional.sum(), conditional.new_tensor(1.), rtol=1e-6, atol=1e-7)):
        raise ValueError('The conditional teacher must be a normalized probability vector.')
    reference = reference_logits.detach().softmax(-1)
    mass = reference[indices].sum()
    if not bool(torch.isfinite(mass)) or float(mass) <= 0:
        raise ValueError('The evaluated support needs positive reference mass.')
    target = reference.clone()
    target[indices] = mass * conditional
    if not torch.isclose(target.sum(), target.new_tensor(1.), rtol=1e-6, atol=1e-7):
        raise FloatingPointError('The lifted native teacher is not normalized.')
    return SupportedTeacher(target, reference, mass, indices)
