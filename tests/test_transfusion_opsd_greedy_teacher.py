import pytest

from stable_audio_tools.training.transfusion_opsd.greedy_execution_teacher import (
    paired_execution_change, select_greedy_execution_teacher, qualify_fixed_execution_teacher)
from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore


def score(utility, content=0., speech=0.):
    return RewardScore(utility, {'content': content, 'speech': speech})


def test_validation_cannot_switch_to_another_action():
    construction = [[score(0), score(2), score(1)]] * 2
    validation = [[score(0), score(-1), score(1)]] * 2
    result = select_greedy_execution_teacher(construction, validation,
        legal=[True] * 3, baseline_action=0)
    assert result['chosen_on_construction'] == 1
    assert not result['qualified']


def test_each_cost_is_protected_and_noise_regression_is_disclosed():
    before = [score(0), score(0)]
    result = paired_execution_change(before, [score(1, .2), score(1, -.4)])
    assert result['empirical_protected_gain']
    assert result['cost_changes']['content']['per_noise'][0] > 0
    assert not paired_execution_change(before, [score(1, -1, .01)] * 2)['empirical_protected_gain']
    with pytest.raises(ValueError, match='same protection metrics'):
        paired_execution_change(before, [RewardScore(1, {'content': 0})] * 2)


def test_actual_baseline_and_legal_actions_determine_comparison():
    values = [[score(2), score(0), score(1)]] * 2
    result = select_greedy_execution_teacher(values, values,
        legal=[False, True, True], baseline_action=1)
    assert result['qualified'] and result['chosen_on_construction'] == 2
    assert paired_execution_change([score(0)] * 2, [score(0)] * 2,
        require_gain=False)['empirical_protected_gain']


def test_fixed_teacher_cannot_hide_a_failed_half_or_missing_pair():
    before = [score(0)] * 4
    groups = dict(first_indices=[0, 1], second_indices=[2, 3])
    assert qualify_fixed_execution_teacher(before, [score(1)] * 4, **groups)['qualified']
    result = qualify_fixed_execution_teacher(before, [score(3), score(3), score(-1), score(-1)], **groups)
    assert result['checks']['pooled']['empirical_protected_gain'] and not result['qualified']
    assert not qualify_fixed_execution_teacher(before, [score(1), score(1), score(1), None], **groups)['qualified']
    with pytest.raises(ValueError, match='exactly once'):
        qualify_fixed_execution_teacher(before, [score(1)] * 4, first_indices=[0, 1], second_indices=[1, 2])
