import pytest
import torch

from stable_audio_tools.models.sceneplan_generation_ar_qualitative_head import complete_from_logits, ATTRIBUTES
from stable_audio_tools.training.transfusion_opsd.native_timing_execution import (
    feasible_timing_pairs, timing_pair_logits, timing_execution_teacher,
    native_timing_loss, factorized_timing_target)


@pytest.mark.parametrize('frames', [10, 123, 432])
def test_joint_choice_matches_native_completion(frames):
    generator = torch.Generator().manual_seed(frames)
    for _ in range(4):
        values = {key: torch.randn(len(labels), generator=generator).tolist()
                  for key, labels in ATTRIBUTES.items()}
        result = complete_from_logits(values, frames, seed_key='timing-test', speech=True)
        choice = feasible_timing_pairs(frames)[int(timing_pair_logits(
            torch.tensor(values['onset']), torch.tensor(values['offset']), frames).argmax())]
        assert tuple(result['labels'][key] for key in ('onset', 'offset')) == (
            ATTRIBUTES['onset'][choice[0]], ATTRIBUTES['offset'][choice[1]])


def test_teacher_uses_gain_without_changing_unmeasured_mass():
    onset = torch.tensor([0., 1., 2., 3., 5.], requires_grad=True)
    offset = torch.tensor([0., 0., 0., 0., 9.], requires_grad=True)
    teacher = timing_execution_teacher(onset, offset, 123, {(0, 4): 1., (4, 4): 0.}, temperature=.1)
    reference = teacher['reference_logits'].softmax(0)
    outside = [i for i, pair in enumerate(teacher['pairs']) if pair not in [(0, 4), (4, 4)]]
    assert torch.allclose(reference[outside], teacher['probabilities'][outside], atol=1e-14, rtol=0)
    assert teacher['expected_measured_reward_target'] > teacher['expected_measured_reward_before']
    assert teacher['target_choice'] == (0, 4)
    assert not teacher['target_logits'].requires_grad
    native_timing_loss(onset, offset, 123, teacher['target_logits']).backward()
    assert onset.grad[0] < 0 and onset.grad[4] > 0
    assert offset.grad is not None and torch.isfinite(offset.grad).all()


def test_tied_execution_retains_exact_zero_gradient():
    a = torch.arange(5, dtype=torch.float32, requires_grad=True)
    b = torch.arange(5, dtype=torch.float32, requires_grad=True)
    teacher = timing_execution_teacher(a, b, 123, {(0, 4): .5, (4, 4): .5}, temperature=.1)
    loss = native_timing_loss(a, b, 123, teacher['target_logits'])
    loss.backward()
    assert float(loss) == 0 and torch.count_nonzero(a.grad) == torch.count_nonzero(b.grad) == 0


def test_nonfactorized_teacher_is_projected_not_assumed_representable():
    a = torch.zeros(5, dtype=torch.float64)
    b = torch.zeros(5, dtype=torch.float64)
    teacher = timing_execution_teacher(a, b, 123, {(0, 4): 1., (4, 4): 0.}, temperature=.25)
    fitted, report = factorized_timing_target(a, b, 123, teacher['target_logits'])
    assert report['kl_after'] < report['kl_before']
    assert report['kl_after'] > 1e-6  # Finite-support tilt has an interaction the two heads cannot represent.
    assert not fitted.requires_grad


def test_invalid_observations_are_rejected():
    with pytest.raises(ValueError):
        timing_execution_teacher(torch.zeros(5), torch.zeros(5), 123,
            {(4, 0): 1., (4, 4): 0.}, temperature=.1)
