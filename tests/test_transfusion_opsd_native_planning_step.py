import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_planning_step import native_planning_constrained_step


def setup():
    ar=torch.nn.Parameter(torch.tensor([0.]))
    shared=torch.nn.Parameter(torch.tensor([0.]))
    dit=torch.nn.Parameter(torch.tensor([0.]))
    optimizer=torch.optim.AdamW([dict(name='ar_body',params=[ar]),dict(name='shared',params=[shared]),
                                dict(name='dit_private',params=[dit])],lr=1.,weight_decay=0.,foreach=False)
    for p in (ar,shared,dit):p.grad=torch.tensor([-1.])
    return ar,shared,dit,optimizer


def test_backtracking_protects_planner_without_shrinking_dit_private_step():
    ar,shared,dit,optimizer=setup()
    result=native_planning_constrained_step(optimizer,lambda:dict(passed=float(shared)<=.3 and float(ar)<=.3))
    assert result['accepted'] and result['planner_shared_factor']==.25
    torch.testing.assert_close(ar,torch.tensor([.25]))
    torch.testing.assert_close(shared,torch.tensor([.25]))
    torch.testing.assert_close(dit,torch.tensor([1.]))
    assert optimizer.state[dit]['step']==1


def test_full_step_is_unchanged_when_the_constraint_passes():
    ar,shared,dit,optimizer=setup()
    result=native_planning_constrained_step(optimizer,lambda:dict(passed=True))
    assert result['planner_shared_factor']==1 and len(result['attempts'])==1
    for p in (ar,shared,dit):torch.testing.assert_close(p,torch.tensor([1.]))


def test_zero_shared_step_is_possible_with_full_dit_step():
    ar,shared,dit,optimizer=setup()
    result=native_planning_constrained_step(optimizer,lambda:dict(passed=float(shared)==0 and float(ar)==0))
    assert result['accepted'] and result['planner_shared_factor']==0
    assert torch.equal(ar,torch.zeros(1)) and torch.equal(shared,torch.zeros(1))
    torch.testing.assert_close(dit,torch.tensor([1.]))


def test_failed_private_constraint_rolls_back_parameters_and_optimizer():
    ar,shared,dit,optimizer=setup()
    result=native_planning_constrained_step(optimizer,lambda:dict(passed=float(dit)==0))
    assert not result['accepted'] and not optimizer.state
    for p in (ar,shared,dit):assert torch.equal(p,torch.zeros(1))


def test_preexisting_failure_cannot_advance_optimizer():
    ar,shared,dit,optimizer=setup()
    with pytest.raises(ValueError):native_planning_constrained_step(optimizer,lambda:dict(passed=False))
    assert not optimizer.state and torch.equal(dit,torch.zeros(1))
