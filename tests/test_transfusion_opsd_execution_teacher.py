import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.execution_teacher import execution_teacher, qualify_execution_teacher
from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore, forward_kl


def test_dit_feedback_changes_private_ar_gradient_and_preserves_unmeasured_mass():
    temperature = .8
    original = temperature * torch.tensor([.4, .4, .2]).log()
    legal = torch.ones(3, dtype=torch.bool)
    gradients = []
    for scores in ([.4, .6], [.9, .65]):
        teacher, _, _, mass = execution_teacher(original, legal, [0, 1], scores,
            ar_temperature=temperature, temperature=.2, strength=1.)
        assert mass == pytest.approx(.8)
        distribution = (teacher / temperature).softmax(-1)
        assert distribution[2].item() == pytest.approx(.2)
        student = original.clone().requires_grad_()
        forward_kl(student[None, None], teacher[None, None], legal[None, None], temperature=temperature).backward()
        gradients.append(student.grad.clone())
    assert gradients[0][0] > 0 > gradients[1][0]
    for strength, scores in [(0., [.9, .1]), (1., [.5, .5])]:
        teacher, _, _, _ = execution_teacher(original, legal, [0, 1], scores,
            ar_temperature=temperature, temperature=.2, strength=strength)
        torch.testing.assert_close(teacher, original)
    with pytest.raises(ValueError, match="legal"):
        execution_teacher(original, torch.tensor([True, False, True]), [0, 1], [.1, .2],
            ar_temperature=.8, temperature=.2, strength=.5)

def test_teacher_requires_independent_paired_gain_and_no_cost_regression():
    p, q = torch.tensor([.5, .5]), torch.tensor([.2, .8])
    make = lambda a, b: [RewardScore(a, {"speech": 0.}), RewardScore(b, {"speech": 0.})]
    options = dict(min_gain=.001, min_utility=0., cost_limits={"speech": 0.})
    train = [make(.4, .8), make(.45, .85)]
    assert qualify_execution_teacher(p, q, train, [make(.5, .9)], **options)[0]
    assert not qualify_execution_teacher(p, q, train, [make(.9, .5)], **options)[0]
    damaged = [[RewardScore(.4, {"speech": 0.}), RewardScore(.9, {"speech": .2})]]
    assert not qualify_execution_teacher(p, q, train, damaged, **options)[0]
    with pytest.raises(ValueError, match="coverage"):
        qualify_execution_teacher(p, q, train, [[RewardScore(.4, {}), RewardScore(.9, {})]], **options)
