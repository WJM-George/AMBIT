"""Analytic ODE checks for the sign, integrated budget and teacher boundary."""
import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.trajectory_repair_teacher import transport_velocity_target


@pytest.mark.parametrize('direction', [1, -1])
def test_constant_executor_has_declared_endpoint_displacement(direction):
    # A constant executor has an analytic endpoint. A nonuniform schedule must
    # not change the total correction when the suffix's time interval is fixed.
    state = torch.tensor([[[2., -1., 3.]]], dtype=torch.float64)
    velocity = torch.tensor([[[.5, 2., -1.]]], dtype=torch.float64)
    delta = torch.tensor([[[.1, -.2, 8.]]], dtype=torch.float64)
    mask = torch.tensor([[True, True, False]])
    times = [.5, .49, .3, .12, 0.]
    expected = state - times[0] * velocity + direction * delta.masked_fill(~mask[:, None], 0)
    for current, following in zip(times, times[1:]):
        target = transport_velocity_target(velocity, delta, mask, remaining_time=times[0], direction=direction)
        state = state + (following-current) * target
    torch.testing.assert_close(state, expected, rtol=0, atol=1e-7)


def test_frozen_teacher_preserves_output_boundary_and_zero_identity():
    velocity = torch.randn(1, 4, 7, dtype=torch.bfloat16, requires_grad=True)
    delta = torch.zeros_like(velocity, requires_grad=True)
    mask = torch.ones(1, 7, dtype=torch.bool)
    target = transport_velocity_target(velocity, delta, mask, remaining_time=.5)
    assert target.dtype == velocity.dtype and not target.requires_grad
    assert torch.equal(target, velocity)
    with pytest.raises(ValueError):
        transport_velocity_target(velocity, delta, mask, remaining_time=0.)
