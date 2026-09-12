import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.auditory_retention import (
    acoustic_distribution_retention, semantic_retention_penalty,
)


def test_semantic_margin_allows_small_change_and_is_one_at_guard():
    reference = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    assert semantic_retention_penalty(torch.tensor(.399, dtype=torch.float64), reference) == 0
    score = torch.tensor(.395, dtype=torch.float64, requires_grad=True)
    loss = semantic_retention_penalty(score, reference)
    torch.testing.assert_close(loss, torch.tensor(1., dtype=torch.float64))
    loss.backward()
    assert score.grad < 0 and reference.grad is None


def test_better_score_is_not_penalized():
    score = torch.tensor(.5, requires_grad=True)
    loss = semantic_retention_penalty(score, torch.tensor(.4))
    loss.backward()
    assert loss == 0 and score.grad == 0


def test_acoustic_reference_is_fixed_and_actual_student_gradient_reduces_kl():
    teacher = torch.tensor([[[2., -1., 0.], [-1., 3., 1.]]], requires_grad=True)
    student = torch.zeros_like(teacher, requires_grad=True)
    loss = acoustic_distribution_retention(student, teacher)
    loss.backward()
    assert teacher.grad is None and student.grad.norm() > 0
    assert acoustic_distribution_retention(student.detach() - .2 * student.grad, teacher) < loss
    assert acoustic_distribution_retention(teacher.detach(), teacher) == 0


@pytest.mark.parametrize('free,guard', [(0.005,0.005),(-1.,.005),(0.,float('nan'))])
def test_invalid_semantic_budget_is_rejected(free,guard):
    with pytest.raises(ValueError):
        semantic_retention_penalty(torch.tensor(.4), torch.tensor(.4), free_drop=free, guard_drop=guard)


def test_acoustic_alignment_is_not_silently_warped():
    with pytest.raises(ValueError):
        acoustic_distribution_retention(torch.zeros(1,4,3), torch.zeros(1,5,3))
    with pytest.raises(ValueError):
        acoustic_distribution_retention(torch.full((1,4,3), float('nan')), torch.zeros(1,4,3))
