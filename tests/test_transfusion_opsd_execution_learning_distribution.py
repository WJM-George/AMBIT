import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.execution_learning_distribution import (
    covered_execution_weights, require_direction_gain_headroom,
)


def test_unsupported_teacher_mass_is_not_redistributed_or_differentiated():
    q = torch.tensor([.2, .7, .1], requires_grad=True)
    mu = torch.tensor([.5, .5, 0.], requires_grad=True)
    value = covered_execution_weights(q, mu, torch.tensor([True, True, False]), coverage_fraction=.5)
    torch.testing.assert_close(value['distribution'], torch.tensor([.35, .6, .05], dtype=torch.float64))
    torch.testing.assert_close(value['supervised_weights'], torch.tensor([.35, .6, 0.], dtype=torch.float64))
    assert value['unsupported_mass'].item() == pytest.approx(.05)
    assert not any(x.requires_grad for x in value.values())


def test_coverage_is_explicit_objective_change_and_not_teacher_qualification():
    q = torch.tensor([.2, .7, .1])
    mu = torch.tensor([0., 0., 1.])
    supported = torch.tensor([True, True, False])
    value = covered_execution_weights(q, mu, supported, coverage_fraction=1.)
    assert value['supervised_weights'].sum().item() == 0.
    assert value['unsupported_mass'].item() == 1.


def test_truncated_plan_bank_must_not_silently_become_a_distribution():
    with pytest.raises(ValueError, match='complete declared support'):
        covered_execution_weights(torch.tensor([.2, .7]), torch.tensor([.5, .5]),
                                  torch.tensor([True, True]), coverage_fraction=.5)


def test_v20_comparison_gate_is_rejected_before_training():
    with pytest.raises(ValueError, match='Unreachable'):
        require_direction_gain_headroom([0., .0031645570416003466], minimum_mean_gain=.0025)
    with pytest.raises(ValueError, match='Unreachable'):
        require_direction_gain_headroom([0., 0.], minimum_mean_gain=.00001)


def test_headroom_is_a_bound_and_does_not_relabel_baselines():
    baseline = [.02, .04]
    result = require_direction_gain_headroom(baseline, minimum_mean_gain=.01)
    assert result['maximum_possible_mean_gain'] == pytest.approx(.03)
    assert baseline == [.02, .04]


@pytest.mark.parametrize('rates', [[], [-.01], [1.01], [float('nan')], [float('inf')]])
def test_invalid_failure_panel_cannot_define_acceptance(rates):
    with pytest.raises(ValueError):
        require_direction_gain_headroom(rates, minimum_mean_gain=.01)
