import pytest
import torch
import torch.nn.functional as F

from stable_audio_tools.training.transfusion_opsd.supported_teacher import preserve_reference_support_mass


def test_no_initial_logit_credit_assigned_to_unexecuted_alternatives():
    probabilities = torch.tensor([.6, .1, .2, .1], dtype=torch.float64)
    logits = probabilities.log().requires_grad_(True)
    teacher = preserve_reference_support_mass(logits, [0, 2], torch.tensor([.2, .8], dtype=torch.float64))
    torch.testing.assert_close(teacher.probabilities, torch.tensor([.16, .1, .64, .1], dtype=torch.float64))
    loss = F.kl_div(logits.log_softmax(-1), teacher.probabilities, reduction='sum')
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient, torch.tensor([.44, 0., -.44, 0.], dtype=torch.float64), atol=1e-15, rtol=0)
    assert torch.equal(teacher.probabilities[[1, 3]], teacher.reference_probabilities[[1, 3]])


def test_small_measured_mass_can_still_change_native_argmax():
    probabilities = torch.full((361,), (1-.0235393327-.00360987335)/359, dtype=torch.float64)
    probabilities[0], probabilities[1] = .0235393327, .00360987335
    teacher = preserve_reference_support_mass(probabilities.log(), [0, 1], torch.tensor([0., 1.], dtype=torch.float64))
    assert int(probabilities.argmax()) == 0 and int(teacher.probabilities.argmax()) == 1
    assert float(teacher.support_mass) < .028
    assert torch.equal(teacher.probabilities[2:], teacher.reference_probabilities[2:])


def test_teacher_reference_and_conditional_distribution_are_stopped():
    reference = torch.tensor([0., 1., 2.], requires_grad=True)
    conditional = torch.tensor([.3, .7], requires_grad=True)
    teacher = preserve_reference_support_mass(reference, [0, 2], conditional)
    assert not teacher.probabilities.requires_grad
    assert not teacher.reference_probabilities.requires_grad
    assert not teacher.support_mass.requires_grad


def test_full_evaluated_support_reduces_to_the_full_teacher():
    q = torch.tensor([.1, .2, .7], dtype=torch.float64)
    teacher = preserve_reference_support_mass(torch.tensor([1., 3., -2.], dtype=torch.float64), [0, 1, 2], q)
    torch.testing.assert_close(teacher.probabilities, q)


@pytest.mark.parametrize('indices,conditional', [([0, 0], [.5, .5]), ([0, 3], [.5, .5]),
    ([0., 1.], [.5, .5]), ([0, 1], [.8, .8]), ([0, 1], [-.1, 1.1])])
def test_invalid_support_or_teacher_is_rejected(indices, conditional):
    with pytest.raises(ValueError):
        preserve_reference_support_mass(torch.tensor([0., 1., 2.]), indices, torch.tensor(conditional))
