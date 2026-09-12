import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.shared_gradient_direction import (
    first_adam_delta, shared_direction_coefficients)


def test_unequal_tasks_preserve_sum_norm_and_give_both_descent():
    a = [torch.tensor([100., 0., 0.]), torch.tensor([0., 4.])]
    d = [torch.tensor([-0.1, 1., 0.]), torch.tensor([0.5, 0.])]
    r = shared_direction_coefficients(a, d)
    mixed = torch.cat([r['planning_coefficient'] * x + r['execution_coefficient'] * y for x,y in zip(a,d)])
    raw = torch.cat([x+y for x,y in zip(a,d)])
    assert mixed.norm().item() == pytest.approx(raw.norm().item(), rel=2e-6)
    assert torch.cat(a).dot(-mixed) < 0
    assert torch.cat(d).dot(-mixed) < 0


@pytest.mark.parametrize('a,d,reason', [([0., 0.], [1., 2.], 'zero_task_gradient'),
    ([1., 2.], [-2., -4.], 'opposite_task_directions'), ([0., 0.], [0., 0.], 'zero_task_gradient')])
def test_degenerate_directions_keep_the_ordinary_update(a, d, reason):
    r = shared_direction_coefficients([torch.tensor(a)], [torch.tensor(d)])
    assert r['planning_coefficient'] == r['execution_coefficient'] == 1
    assert r['fallback'] == reason


def test_invalid_nonfinite_gradient_rejected():
    with pytest.raises(ValueError):
        shared_direction_coefficients([torch.tensor([float('nan')])], [torch.ones(1)])


def test_balanced_direction_does_not_depend_on_relative_task_units():
    a, d = torch.tensor([4., 1.]), torch.tensor([-0.1, 2.])
    directions = []
    for scale in (1., 100.):
        r = shared_direction_coefficients([a*scale], [d])
        x = r['planning_coefficient']*a*scale + r['execution_coefficient']*d
        directions.append(x/x.norm())
    torch.testing.assert_close(*directions)


def test_first_adam_proposal_matches_real_update():
    p = torch.nn.Parameter(torch.tensor([0.01, -0.3, 1., 0.]))
    grad = torch.tensor([1., -3., 0., 1e-9]); before = p.detach().clone()
    expected = first_adam_delta(before, grad, learning_rate=1e-6, clip_coefficient=0.125)
    optimizer = torch.optim.AdamW([p], lr=1e-6, weight_decay=0., foreach=False)
    p.grad = grad*0.125; optimizer.step()
    torch.testing.assert_close(p-before, expected, atol=2e-9, rtol=0.)
