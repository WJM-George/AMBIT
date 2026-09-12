import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.two_half_step_teacher import bounded_two_half_step_target


def test_two_half_steps_reduce_linear_flow_discretization_error():
    z = torch.ones(1, 2, 3, dtype=torch.float64, requires_grad=True)
    r = bounded_two_half_step_target(z, torch.ones(1, dtype=z.dtype), .05,
                                    lambda x, t: 2*x, torch.ones(1, 3, dtype=torch.bool))
    expected = z.detach() * .95**2
    assert torch.allclose(r['next_state'], expected)
    exact = z.detach() * torch.exp(torch.tensor(-.1, dtype=z.dtype))
    assert (r['next_state']-exact).abs().max() < (z.detach()*.9-exact).abs().max()
    assert not r['target_velocity'].requires_grad and not r['next_state'].requires_grad


def test_radius_is_per_example_and_padding_is_not_a_teacher_signal():
    z = torch.tensor([[[1., 1., 900.]], [[100., 100., -900.]]], dtype=torch.float64)
    keep = torch.tensor([[True, True, False], [True, True, False]])
    r = bounded_two_half_step_target(z, torch.ones(2, dtype=z.dtype), .05,
        lambda x,t: 2*x, keep, maximum_relative_velocity_change=.01)
    delta = (r['target_velocity']-r['anchor_velocity']).flatten(1).norm(dim=1)
    assert torch.allclose(delta, r['velocity_radius'])
    assert torch.all(r['next_state'][..., -1] == 0)
    assert torch.all(r['target_velocity'][..., -1] == 0)


def test_constant_velocity_has_no_artificial_repair_and_cached_anchor_is_equivalent():
    z = torch.randn(1, 2, 3); mask = torch.ones(1, 3, dtype=torch.bool); t = torch.ones(1)
    v = lambda x,t: torch.ones_like(x)
    a = bounded_two_half_step_target(z,t,.05,v,mask)
    b = bounded_two_half_step_target(z,t,.05,v,mask,anchor_velocity=v(z,t))
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert torch.equal(a['target_velocity'], torch.ones_like(z))
    assert torch.all(a['unbounded_repair_norm'] == 0)


def test_invalid_time_or_nonfinite_teacher_is_rejected():
    z = torch.ones(1,2,3);mask = torch.ones(1,3,dtype=torch.bool)
    with pytest.raises(ValueError):
        bounded_two_half_step_target(z,torch.tensor([.01]),.05,lambda x,t:x,mask)
    with pytest.raises(ValueError):
        bounded_two_half_step_target(z,torch.ones(1),.05,lambda x,t:x*float('nan'),mask)
