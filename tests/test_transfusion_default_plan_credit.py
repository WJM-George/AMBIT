import torch

from stable_audio_tools.training.transfusion_opsd.default_plan_credit import (
    default_relative_credit, default_relative_teacher)
from stable_audio_tools.training.transfusion_opsd.identity_preserving_kl import identity_preserving_kl


def test_both_sampled_plans_worse_than_default_never_get_positive_advantage():
    result = default_relative_credit(torch.tensor([[.4, .5], [.1, .2]]),
        torch.tensor([.8, .9]), torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    assert (result.normalized_advantage < 0).all()


def test_positive_guard_failure_and_identical_plan_cannot_create_credit():
    result = default_relative_credit(torch.tensor([[.8, .9], [.7, .7], [.1, .1]]),
        torch.tensor([.5, .5]), torch.tensor([[True, False], [True, True], [True, True]]),
        torch.tensor([False, True, False]))
    torch.testing.assert_close(result.safe_credit, torch.tensor([0., 0., -.4], dtype=torch.float64))
    assert not result.positive_eligible.any()


def test_zero_credit_has_zero_preservation_gradient():
    prior = torch.tensor([-.2, -1.2, -2.7], dtype=torch.float64, requires_grad=True)
    credit = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    teacher, q = default_relative_teacher(prior, credit, temperature=.1)
    assert not teacher.requires_grad and not q.requires_grad
    loss = identity_preserving_kl(prior, teacher)
    loss.backward()
    assert float(loss) == 0 and torch.equal(prior.grad, torch.zeros_like(prior))
    assert credit.grad is None


def test_negative_credit_reduces_relative_odds_and_duplicate_noise_is_not_a_gain():
    prior = torch.tensor([-.3, -.8, -1.7], dtype=torch.float64)
    _, q = default_relative_teacher(prior, torch.tensor([0., -.1, 0.]), temperature=.1)
    old = prior.softmax(0)
    assert q[1] / q[0] < old[1] / old[0]
    torch.testing.assert_close(q[2] / q[0], old[2] / old[0])
    result = default_relative_credit(torch.ones(2, 2), torch.ones(2),
        torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    assert torch.equal(result.normalized_advantage, torch.zeros(2, dtype=torch.float64))
