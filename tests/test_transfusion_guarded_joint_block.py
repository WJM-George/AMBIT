import copy

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.guarded_joint_block import guarded_joint_adam_block


def make_model():
    m = torch.nn.Linear(2, 1, bias=False).double()
    m.weight.data.copy_(torch.tensor([[.2, -.4]], dtype=torch.float64))
    m.register_buffer('sentinel', torch.tensor(3.))
    o = torch.optim.AdamW([dict(name='shared', params=list(m.parameters()))], lr=.03)
    return m, o


def backward(m, i):
    prediction = m(torch.tensor([[1., 2.]], dtype=torch.float64))
    loss = (prediction - 1).square().sum()
    loss.backward()
    return dict(step=i, loss=float(loss.detach()))


def test_full_block_equals_manual_joint_steps_and_counts_all_steps():
    m, o = make_model()
    reference, ro = make_model()
    for i in range(3):
        ro.zero_grad(set_to_none=True)
        backward(reference, i)
        ro.step()
    result = guarded_joint_adam_block(m, o, backward_step=lambda i: backward(m, i), inner_steps=3,
        trial_scales=[{'shared': 1.}], validate_native=lambda: {'passed': True})
    assert result.accepted and result.accepted_optimizer_steps == result.attempted_optimizer_steps == 3
    torch.testing.assert_close(m.weight, reference.weight, rtol=0, atol=0)
    for k, v in o.state[m.weight].items():
        torch.testing.assert_close(v, ro.state[reference.weight][k], rtol=0, atol=0)


def test_scaled_acceptance_is_relative_to_start_of_whole_block():
    m, o = make_model()
    start = m.weight.detach().clone()
    end, eo = make_model()
    for i in range(2):
        eo.zero_grad(set_to_none=True)
        backward(end, i)
        eo.step()
    calls = []
    def validate():
        calls.append(m.weight.detach().clone())
        return {'passed': len(calls) == 2}
    result = guarded_joint_adam_block(m, o, backward_step=lambda i: backward(m, i), inner_steps=2,
        trial_scales=[{'shared': 1.}, {'shared': .25}], validate_native=validate)
    torch.testing.assert_close(calls[0], end.weight, rtol=0, atol=0)
    torch.testing.assert_close(m.weight, start + .25 * (end.weight - start), rtol=0, atol=0)
    assert result.accepted and result.accepted_optimizer_steps == 2


def test_rejected_block_restores_parameters_moments_buffers_and_rng():
    m, o = make_model()
    backward(m, 0); o.step(); o.zero_grad(set_to_none=True)
    state = copy.deepcopy(m.state_dict()); opt = copy.deepcopy(o.state_dict())
    rng = torch.get_rng_state().clone()
    def step(i):
        torch.rand(3)
        m.sentinel.add_(2)
        return backward(m, i)
    def reject():
        torch.rand(7)
        return {'passed': False}
    result = guarded_joint_adam_block(m, o, backward_step=step, inner_steps=2,
        trial_scales=[{'shared': 1.}, {'shared': .1}], validate_native=reject)
    assert not result.accepted and result.attempted_optimizer_steps == 2 and result.accepted_optimizer_steps == 0
    assert torch.equal(torch.get_rng_state(), rng)
    for k, v in m.state_dict().items():
        assert torch.equal(v, state[k])
    for k, v in o.state_dict()['state'][0].items():
        assert torch.equal(v, opt['state'][0][k])
    assert m.weight.grad is None


def test_callback_failure_rolls_back_earlier_internal_step():
    m, o = make_model(); start = m.weight.detach().clone()
    def step(i):
        if i == 1:
            raise RuntimeError('deliberate second backward failure')
        return backward(m, i)
    with pytest.raises(RuntimeError, match='second backward'):
        guarded_joint_adam_block(m, o, backward_step=step, inner_steps=2,
            trial_scales=[{'shared': 1.}], validate_native=lambda: {'passed': True})
    assert torch.equal(m.weight, start) and not o.state and m.weight.grad is None
