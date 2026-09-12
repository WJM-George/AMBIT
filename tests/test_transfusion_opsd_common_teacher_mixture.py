import copy

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.common_teacher_mixture import (
    best_positive_contexts, positive_context_mixture,
)


def selection():
    return dict(action=0, best_per_plan=[
        dict(trial=0, eligible=True), dict(trial=1, eligible=True), dict(trial=0, eligible=True)],
        trials=[[dict(trial=0, eligible=True)],
                [dict(trial=0, eligible=True), dict(trial=1, eligible=True)],
                [dict(trial=0, eligible=True)]])


def test_a_valid_nonwinning_repair_is_not_discarded():
    terms, evidence = positive_context_mixture(selection(), [.6, .25, .15], legal=[True] * 3)
    assert selection()['action'] == 0
    assert best_positive_contexts(selection()) == ((1, 1),)
    assert terms == [dict(action=1, trial=1, weight=.25)]
    assert evidence['positive_probability_mass'] == .25
    assert evidence['unrepaired_probability_mass'] == .75
    assert not evidence['renormalized_positive_targets']


def test_teacher_marginal_is_detached_from_student_gradient():
    q = torch.tensor([.5, .25, .25], requires_grad=True)
    terms, _ = positive_context_mixture(selection(), q, legal=[True] * 3)
    student = torch.tensor(2., requires_grad=True)
    sum(term['weight'] * student.square() for term in terms).backward()
    assert q.grad is None
    assert student.grad == 1.


def test_no_positive_target_does_not_create_a_fake_dit_update():
    empty = selection()
    empty['best_per_plan'][1]['trial'] = 0
    terms, evidence = positive_context_mixture(empty, [.5, .25, .25], legal=[True] * 3)
    assert not terms
    assert evidence['positive_probability_mass'] == 0.
    assert evidence['unrepaired_probability_mass'] == 1.


@pytest.mark.parametrize('q,legal', [([.5, .2, .1], [True] * 3),
    ([.5, .25, .25], [True, False, True]), ([.5, float('nan'), .5], [True] * 3)])
def test_invalid_probability_mass_cannot_hide_coverage(q, legal):
    with pytest.raises(ValueError, match='marginal'):
        positive_context_mixture(selection(), q, legal=legal)


def test_a_target_losing_protection_evidence_is_rejected():
    invalid = copy.deepcopy(selection())
    invalid['trials'][1][1]['eligible'] = False
    with pytest.raises(ValueError, match='verified'):
        positive_context_mixture(invalid, [.5, .25, .25], legal=[True] * 3)
