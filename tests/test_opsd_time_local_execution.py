import pytest
import torch
from torch import nn

from stable_audio_tools.training.transfusion_opsd.time_local_execution import TimeLocalExecution


class ToyBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False)
        self.shared_ar = self.projection
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(2))

    def velocity_function(self, condition, *, differentiable):
        def forward(state, time):
            assert self.shared_ar.weight is self.projection.weight
            if condition == 'fail':
                raise RuntimeError('injected forward failure')
            return self.projection(state.transpose(1, 2)).transpose(1, 2) * condition
        return forward


def make():
    b = ToyBundle()
    gate = TimeLocalExecution(b, maximum_trainable_time=.24,
        conditional_factory=lambda module, c: module.velocity_function(c, differentiable=True))
    return b, gate


def test_anchor_identity_and_optimizer_leaves():
    b, g = make()
    ids = [id(p) for p in b.parameters()]
    z = torch.randn(1, 2, 3)
    for t in [.1, .8]:
        assert torch.equal(g.velocity_function(2., differentiable=False)(z, torch.tensor([t])), 2*z)
    assert [id(p) for p in b.parameters()] == ids
    assert b.shared_ar is b.projection


def test_gate_changes_only_low_noise_and_keeps_ar_current():
    b, g = make()
    with torch.no_grad():
        b.projection.weight.mul_(3)
    z = torch.ones(1, 2, 3)
    assert torch.equal(g.velocity_function(2., differentiable=False)(z, torch.tensor([.8])), 2*z)
    assert torch.equal(g.velocity_function(2., differentiable=False)(z, torch.tensor([.1])), 6*z)
    assert torch.equal(b.shared_ar(torch.ones(1, 2)), torch.full((1, 2), 3.))


def test_gradient_routes_and_frozen_conditional_backward():
    b, g = make()
    z = torch.ones(1, 2, 3, requires_grad=True)
    g.conditional_velocity_function(2.)(z, torch.tensor([.8])).sum().backward()
    assert torch.equal(b.projection.weight.grad, torch.zeros_like(b.projection.weight))
    assert torch.equal(z.grad, torch.full_like(z, 2.))
    b.zero_grad(set_to_none=True)
    g.velocity_function(2., differentiable=True)(z.detach(), torch.tensor([.1])).sum().backward()
    assert torch.count_nonzero(b.projection.weight.grad) == 4


def test_tied_parameter_restoration_on_failure():
    b, g = make()
    parameter = b.projection.weight
    with torch.no_grad():
        parameter.mul_(3)
    with pytest.raises(RuntimeError, match='injected'):
        g.velocity_function('fail', differentiable=False)(torch.ones(1, 2, 3), torch.tensor([.8]))
    assert b.shared_ar.weight is b.projection.weight is parameter
    assert torch.equal(parameter, 3*torch.eye(2))


def test_invalid_and_mixed_gate_microbatches_rejected():
    _, g = make()
    for times in [torch.tensor([-.1]), torch.tensor([float('nan')]), torch.tensor([1.1])]:
        with pytest.raises(ValueError):
            g.velocity_function(1., differentiable=False)(torch.ones(1, 2, 3), times)
    with pytest.raises(ValueError, match='homogeneous'):
        g.velocity_function(1., differentiable=False)(torch.ones(2, 2, 3), torch.tensor([.1, .8]))
    with pytest.raises(ValueError):
        TimeLocalExecution(ToyBundle(), maximum_trainable_time=1.)
