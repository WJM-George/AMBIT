import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.joint_repair import propose_bounded_repair, select_joint_repair_teacher
from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore


def variant(utility, fraction=0., cost=0., certified=True):
    return dict(score=RewardScore(utility, {'protected': cost}), repair_fraction=fraction, certified=certified)


def select(rows, penalty=.01):
    return select_joint_repair_teacher(rows, legal=[True] * len(rows), tolerances={'protected': .01},
        minimum_gain=.01, semantic_tolerance=.005, repair_penalty=penalty)


def test_ar_is_compared_to_equally_repaired_original():
    result = select([[variant(.1), variant(.15, .5)], [variant(.12), variant(.14, .5)]])
    assert result['action'] == 0
    assert result['trial'] == result['baseline_repair_trial'] == 1
    assert not result['qualified']


def test_combination_can_supply_genuine_additional_plan_credit():
    result = select([[variant(.1), variant(.12, .5)], [variant(.09), variant(.15, .5)]])
    assert result['qualified'] and result['action'] == 1 and result['trial'] == 1
    assert result['ar_credit'] == pytest.approx(.03)


def test_repair_cost_can_change_plan_choice():
    result = select([[variant(.1)], [variant(.1), variant(.15, 1.)], [variant(.1), variant(.148, .1)]])
    assert result['action'] == 2


def test_uncertified_or_unprotected_benefits_are_not_teachers():
    result = select([[variant(.1)], [variant(.1), variant(.2, .5, certified=False)], [variant(.1), variant(.3, .5, cost=.02)]])
    assert result['action'] == 0 and not result['qualified']


def test_a_repaired_candidate_cannot_hide_its_unmodified_outcome():
    with pytest.raises(ValueError, match='every legal plan'):
        select([[variant(.1)], [variant(.2, .5)]])


def test_common_absolute_budget_and_padding_survive_different_anchors():
    mask = torch.tensor([[True, False]])
    for value in [1., 100.]:
        anchor = torch.tensor([[[value, 3.]]], requires_grad=True)
        repaired = propose_bounded_repair(anchor, mask, objective=lambda x: x.sum(), absolute_budget=.2, steps=2)
        assert float((repaired - anchor).norm()) == pytest.approx(.2, abs=5e-6)
        assert repaired[0, 0, 1] == anchor[0, 0, 1]
        assert anchor.grad is None and not repaired.requires_grad


def test_zero_gradient_preserves_anchor_exactly():
    anchor = torch.tensor([[[1., 3.]]])
    repaired = propose_bounded_repair(anchor, torch.tensor([[True, False]]),
        objective=lambda x: x.sum() * 0, absolute_budget=.2, steps=2)
    assert torch.equal(anchor, repaired)
