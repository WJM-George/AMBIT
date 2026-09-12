import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.event_feedback_distillation import query_teacher_distribution
from stable_audio_tools.training.transfusion_opsd.joint_repair_distillation import repair_teacher_distribution
from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore


def trial(utility, fraction=0., cost=0., certified=True):
    return dict(score=RewardScore(utility, {'space': cost}), repair_fraction=fraction, certified=certified)


def target(variants, mode, logits=None):
    return repair_teacher_distribution(variants, legal=[True] * len(variants),
        initial_logits=torch.tensor([.1] + [0.] * (len(variants)-1)) if logits is None else logits,
        mode=mode, tolerances={'space': .01})


@pytest.mark.parametrize('utilities,costs', [([.1, .14, .12], [0., 0., .02]), ([.1, .102, .09], [0., 0., 0.])])
def test_raw_control_matches_existing_protected_soft_teacher(utilities, costs):
    variants = [[trial(u, cost=c)] for u, c in zip(utilities, costs)]
    logits = torch.tensor([.1, .03, -.04], requires_grad=True)
    actual, _ = target(variants, 'raw_plan', logits)
    scores = [RewardScore(0., {'clap_content_cost/source': 1-u, 'space': c}) for u, c in zip(utilities, costs)]
    expected = query_teacher_distribution(scores, legal=[True] * 3, initial_logits=logits,
        tolerances={'space': .01}, objective='protected_soft_teacher', neutral_query_target='protected_policy')
    assert torch.allclose(actual, expected, atol=1e-14, rtol=1e-14)
    assert not actual.requires_grad


def test_repaired_original_reduces_plan_credit_without_needing_action_change():
    variants = [[trial(.1), trial(.125, .5)], [trial(.15)]]
    raw, raw_info = target(variants, 'raw_plan')
    joint, joint_info = target(variants, 'joint_repair')
    assert raw_info['qualified'] and joint_info['qualified']
    assert joint.argmax() == raw.argmax() == 1
    assert joint[1] < raw[1]
    assert joint_info['candidates'][1]['value_gain'] == pytest.approx(.0275)


def test_repair_can_remove_unsupported_plan_improvement_label():
    variants = [[trial(.1), trial(.13, .5)], [trial(.135)]]
    _, raw_info = target(variants, 'raw_plan')
    joint, joint_info = target(variants, 'joint_repair')
    assert raw_info['qualified'] and not joint_info['qualified']
    assert torch.allclose(joint, torch.tensor([.1, 0.], dtype=torch.float64).softmax(-1))


def test_uncertified_high_reward_cannot_change_teacher():
    variants = [[trial(.1), trial(.3, .5, certified=False)], [trial(.15)]]
    raw, _ = target(variants, 'raw_plan')
    joint, _ = target(variants, 'joint_repair')
    assert torch.equal(raw, joint)
