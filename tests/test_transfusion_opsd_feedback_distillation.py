import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.event_feedback_distillation import query_teacher_distribution
from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore


def score(similarity, wer=0.):
    return RewardScore(0., {'clap_content_cost/source_0': 1 - similarity, 'asr_wer': wer})


def distribution(scores, **kwargs):
    return query_teacher_distribution(scores, legal=[True] * len(scores),
        initial_logits=torch.zeros(len(scores)), tolerances={'asr_wer': 0.},
        objective='protected_soft_teacher', **kwargs)


def test_semantic_gain_cannot_buy_transcript_regression():
    value = distribution([score(.4), score(.42), score(.8, .1)], smoothing=0.)
    assert value[2] == 0
    assert value[1] > value[0] > 0
    assert value.sum() == pytest.approx(1.)


def test_rejected_or_tiny_improvements_keep_actual_baseline():
    value = distribution([score(.4), score(.409), score(.5, .1)], smoothing=0.)
    assert torch.equal(value, torch.tensor([1., 0., 0.], dtype=torch.double))


def test_unobservable_query_produces_no_fabricated_supervision():
    assert distribution([score(.4), None]) is None


def test_soft_teacher_retains_initial_probability_and_detaches_it():
    logits = torch.tensor([0., -3., 0.], requires_grad=True)
    value = query_teacher_distribution([score(.4), score(.42), score(.42)],
        legal=[True, True, True], initial_logits=logits, tolerances={'asr_wer': 0.},
        objective='protected_soft_teacher', smoothing=0.)
    assert value[2] / value[1] == pytest.approx(float(torch.exp(torch.tensor(3.))), rel=1e-6)
    assert not value.requires_grad


@pytest.mark.parametrize('objective', ['winner_cross_entropy', 'protected_soft_teacher'])
def test_illegal_action_never_receives_target_mass(objective):
    value = query_teacher_distribution([score(.4), score(.42), score(.9)],
        legal=[True, True, False], initial_logits=torch.tensor([0., 0., -torch.inf]),
        tolerances={'asr_wer': 0.}, objective=objective)
    assert value[2] == 0
    assert value.sum() == pytest.approx(1.)


def test_teacher_can_return_to_zero_from_a_changed_actor():
    value = query_teacher_distribution([score(.5), score(.4)],
        legal=[True, True], initial_logits=torch.zeros(2), tolerances={'asr_wer': 0.},
        objective='winner_cross_entropy', baseline_action=1, smoothing=0.)
    assert torch.equal(value, torch.tensor([1., 0.], dtype=torch.double))


def test_neutral_query_retains_probability_with_zero_initial_gradient_but_penalizes_drift():
    logits = torch.tensor([.1, -.2, .0, -torch.inf], dtype=torch.double, requires_grad=True)
    target = query_teacher_distribution([score(.4), score(.409), score(.6, .1), score(.9)],
        legal=[True, True, True, False], initial_logits=logits, tolerances={'asr_wer': 0.},
        objective='protected_soft_teacher', neutral_query_target='initial_policy')
    assert not target.requires_grad and target[3] == 0
    torch.testing.assert_close(target, logits.detach().softmax(-1))
    loss = -(target[:3] * logits.log_softmax(-1)[:3]).sum()
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient, torch.zeros_like(logits), atol=1e-14, rtol=0.)
    moved = (logits.detach() + torch.tensor([.3, 0., 0., 0.])).requires_grad_()
    gradient, = torch.autograd.grad(-(target[:3] * moved.log_softmax(-1)[:3]).sum(), moved)
    assert gradient[0] > 0  # Descent restores the initial distribution.


@pytest.mark.parametrize('objective', ['winner_cross_entropy', 'protected_soft_teacher'])
def test_neutral_option_changes_neither_qualified_teachers_nor_missing_queries(objective):
    common = dict(legal=[True, True], initial_logits=torch.tensor([.1, 0.]),
        tolerances={'asr_wer': 0.}, objective=objective)
    actual = query_teacher_distribution([score(.4), score(.45)],
        neutral_query_target='initial_policy', **common)
    legacy = query_teacher_distribution([score(.4), score(.45)], **common)
    assert torch.equal(actual, legacy)
    assert query_teacher_distribution([score(.4), None], neutral_query_target='initial_policy', **common) is None


def test_neutral_projection_retains_safe_ratios_and_excludes_observed_harm():
    logits = torch.tensor([.1, -.2, .0, -.3], dtype=torch.double, requires_grad=True)
    target = query_teacher_distribution([score(.4), score(.398), score(.5, .1), score(.3)],
        legal=[True]*4, initial_logits=logits, tolerances={'asr_wer': 0.},
        objective='protected_soft_teacher', neutral_query_target='protected_policy', neutral_semantic_tolerance=.005)
    torch.testing.assert_close(target[:2], logits[:2].detach().softmax(-1))
    assert torch.equal(target[2:], torch.zeros(2, dtype=target.dtype))
    assert not target.requires_grad
    gradient, = torch.autograd.grad(-(target * logits.log_softmax(-1)).sum(), logits)
    assert bool((gradient[2:] > 0).all())  # Harmful alternatives receive a suppressing gradient.


def test_neutral_projection_has_no_target_shift_when_every_candidate_is_safe():
    logits = torch.tensor([.1, 0.], dtype=torch.double)
    target = query_teacher_distribution([score(.4), score(.398)], legal=[True, True],
        initial_logits=logits, tolerances={'asr_wer': 0.}, objective='protected_soft_teacher',
        neutral_query_target='protected_policy')
    torch.testing.assert_close(target, logits.softmax(-1))
