import math

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.decoded_audio_objectives import relative_decoded_fit_loss
from stable_audio_tools.training.transfusion_opsd.objectives import clean_prediction


def loss(audio, teacher, reference, mask=None, **kwargs):
    if mask is None:
        mask = torch.ones(audio.shape[0], audio.shape[-1], dtype=torch.bool, device=audio.device)
    return relative_decoded_fit_loss(audio, teacher, reference, mask,
        fixed_error_scale=kwargs.get('fixed_error_scale', 1.), reference_power_floor=kwargs.get('reference_power_floor', 1e-12))


def test_gradient_crosses_frozen_decoder_but_not_teacher_or_reference():
    decoder = torch.nn.Conv1d(2, 4, 1, bias=False).double().requires_grad_(False)
    with torch.no_grad():
        decoder.weight.copy_(torch.tensor([[[1.], [0.]], [[0.], [1.]], [[1.], [1.]], [[1.], [-1.]]]))
    z = torch.ones(1, 2, 5, dtype=torch.float64)
    velocity = torch.nn.Parameter(torch.zeros_like(z))
    waveform = decoder(clean_prediction(z, torch.tensor([.2], dtype=torch.float64), velocity))
    teacher = torch.zeros_like(waveform, requires_grad=True)
    reference = torch.ones_like(waveform, requires_grad=True)
    objective = loss(waveform, teacher, reference)
    objective.backward()
    assert torch.isfinite(velocity.grad).all() and velocity.grad.abs().sum() > 0
    assert decoder.weight.grad is None and teacher.grad is None and reference.grad is None
    with torch.no_grad():
        velocity -= .1 * velocity.grad
        updated = loss(decoder(clean_prediction(z, torch.tensor([.2], dtype=torch.float64), velocity)), teacher, reference)
    assert updated < objective


def test_fixed_target_has_zero_gradient_and_muting_is_not_a_solution():
    teacher = torch.arange(1., 25.).reshape(1, 4, 6)
    matched = teacher.clone().requires_grad_()
    objective = loss(matched, teacher, teacher)
    objective.backward()
    assert objective.item() == 0 and torch.count_nonzero(matched.grad) == 0
    assert loss(torch.zeros_like(teacher), teacher, teacher).item() == pytest.approx(1.)


def test_equal_channel_weight_is_equivariant_to_horizontal_foa_rotation():
    generator = torch.Generator().manual_seed(32)
    reference, teacher, audio = [torch.randn(2, 4, 7, generator=generator, dtype=torch.float64) for _ in range(3)]
    c, s = math.cos(.47), math.sin(.47)
    # WYZX: rotate X and Y together while W and Z stay fixed.
    rotation = torch.tensor([[1, 0, 0, 0], [0, c, 0, s], [0, 0, 1, 0], [0, -s, 0, c]], dtype=torch.float64)
    rotate = lambda value: torch.einsum('ij,bjt->bit', rotation, value)
    assert torch.allclose(loss(audio, teacher, reference), loss(rotate(audio), rotate(teacher), rotate(reference)), rtol=1e-12, atol=1e-12)


def test_queries_have_equal_weight_despite_valid_length_and_padding():
    reference = torch.ones(2, 4, 6)
    teacher = reference.clone()
    audio = torch.stack((torch.full((4, 6), 2.), torch.full((4, 6), 4.))).requires_grad_()
    mask = torch.tensor([[True, True, False, False, False, False], [True] * 6])
    objective = loss(audio, teacher, reference, mask)
    assert objective.item() == pytest.approx(5.)  # Query errors 1 and 9 have equal mass.
    objective.backward()
    assert torch.count_nonzero(audio.grad[0, :, 2:]) == 0
    longer = audio.detach().repeat_interleave(3, -1)
    assert loss(longer, teacher.repeat_interleave(3, -1), reference.repeat_interleave(3, -1), mask.repeat_interleave(3, -1)) == objective.detach()


def test_low_amplitude_error_accumulates_without_half_precision_underflow():
    audio = torch.full((1, 4, 8), 1e-4, dtype=torch.float16, requires_grad=True)
    reference = torch.ones_like(audio)
    value = loss(audio, torch.zeros_like(audio), reference)
    assert value.dtype == torch.float32 and 5e-9 < value.item() < 2e-8
    value.backward()
    assert torch.isfinite(audio.grad).all() and torch.count_nonzero(audio.grad) == audio.numel()


def test_fixed_floor_handles_silent_reference_without_dividing_by_student_energy():
    reference = torch.zeros(1, 4, 3)
    teacher = torch.full_like(reference, .1)
    error = loss(reference, teacher, reference, reference_power_floor=.01)
    assert error.item() == pytest.approx(1.)
    assert loss(teacher, teacher, reference, reference_power_floor=.01).item() == 0


@pytest.mark.parametrize('bad_scale', [0., -1., float('inf'), float('nan'), True, torch.tensor(1., requires_grad=True)])
def test_normalizer_cannot_be_invalid_or_trainable(bad_scale):
    audio = torch.ones(1, 4, 3)
    with pytest.raises(ValueError, match='fixed scalar'):
        loss(audio, audio, audio, fixed_error_scale=bad_scale)


def test_empty_support_and_nonfinite_target_fail_instead_of_disappearing():
    audio = torch.ones(1, 4, 3)
    with pytest.raises(ValueError, match='nonempty valid'):
        loss(audio, audio, audio, torch.zeros(1, 3, dtype=torch.bool))
    target = audio.clone()
    target[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite floating'):
        loss(audio, target, audio)
