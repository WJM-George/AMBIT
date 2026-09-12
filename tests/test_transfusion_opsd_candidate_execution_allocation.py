import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.candidate_execution_allocation import candidate_execution_allocation


def example():
    return (torch.tensor([.0013, .0288, .9699], dtype=torch.float64, requires_grad=True),
            torch.tensor([.5, .5, 0.], dtype=torch.float64, requires_grad=True),
            torch.tensor([True, True, False]))


def test_conditioning_changes_priority_without_increasing_execution_budget():
    q, mu, support = example()
    result = candidate_execution_allocation(q, mu, support, coverage_fraction=.5, supervised_budget=.515)
    torch.testing.assert_close(result['supervised_weights'].sum(), torch.tensor(.515, dtype=torch.float64))
    assert result['supervised_weights'][1] / .515 > .72
    assert result['supervised_weights'][2] == 0 and result['unassigned_budget'] == .485
    assert not result['supervised_weights'].requires_grad and not result['conditional_teacher'].requires_grad


def test_changing_unmeasured_mass_does_not_change_the_conditional_curriculum():
    q, mu, support = example()
    alt = q.detach().clone(); alt[:2] *= 2; alt[2] = 1 - alt[:2].sum()
    a = candidate_execution_allocation(q, mu, support, coverage_fraction=.5, supervised_budget=.515)
    b = candidate_execution_allocation(alt, mu, support, coverage_fraction=.5, supervised_budget=.515)
    torch.testing.assert_close(a['supervised_weights'], b['supervised_weights'])
    assert a['measured_teacher_mass'] != b['measured_teacher_mass']


def test_full_coverage_recovers_coverage_allocation():
    q, mu, support = example()
    result = candidate_execution_allocation(q, mu, support, coverage_fraction=1., supervised_budget=.4)
    torch.testing.assert_close(result['supervised_weights'], mu.detach() * .4)


@pytest.mark.parametrize('bad_budget', [0., -1., 1.01, float('nan')])
def test_invalid_budget_is_rejected(bad_budget):
    with pytest.raises(ValueError):
        candidate_execution_allocation(*example(), coverage_fraction=.5, supervised_budget=bad_budget)


def test_coverage_without_a_target_is_not_silently_renormalized():
    q, mu, support = example(); mu = torch.tensor([.4, .4, .2], dtype=torch.float64)
    with pytest.raises(ValueError):
        candidate_execution_allocation(q, mu, support, coverage_fraction=.5, supervised_budget=.5)
