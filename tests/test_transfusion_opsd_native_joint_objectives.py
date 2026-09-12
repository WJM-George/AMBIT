import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_joint_objectives import (
    native_head_teacher, native_head_retention_loss, native_clean_fit_loss)


def test_correct_native_planning_has_zero_retention_loss_without_an_extra_action_head():
    output = {family: {'choice': torch.tensor([[1., 2., -float('inf')]], requires_grad=True)}
        for family in ('inventory', 'qualitative')}
    teacher = native_head_teacher(output)
    loss = native_head_retention_loss(output, teacher)
    assert float(loss) == pytest.approx(0., abs=1e-8)
    loss.backward()
    assert all(not value.requires_grad for heads in teacher.values() for value in heads.values())
    assert all(torch.isfinite(heads['choice'].grad).all() for heads in output.values())


def test_native_decision_drift_is_penalized_and_teacher_remains_detached():
    old = {family: {'choice': torch.tensor([[3., 0.]])} for family in ('inventory', 'qualitative')}
    teacher = native_head_teacher(old)
    changed = {family: {'choice': torch.tensor([[0., 3.]], requires_grad=True)} for family in old}
    loss = native_head_retention_loss(changed, teacher)
    loss.backward()
    assert float(loss)>0
    assert all(heads['choice'].grad[0, 0]<0 for heads in changed.values())


def test_fixed_clean_target_gradient_respects_padding_and_does_not_train_teacher():
    z = torch.zeros(1, 2, 3)
    velocity = torch.zeros_like(z, requires_grad=True)
    target = torch.ones_like(z, requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    loss = native_clean_fit_loss(z, torch.tensor([.2]), velocity, target, mask, fixed_mse_scale=.25)
    loss.backward()
    assert float(loss) == pytest.approx(4.)
    assert target.grad is None
    assert torch.equal(velocity.grad[..., -1], torch.zeros(1, 2))
    assert (velocity.grad[..., :2]>0).all()
