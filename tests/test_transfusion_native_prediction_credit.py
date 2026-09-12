import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_decision_condition import hard_native_condition
from stable_audio_tools.training.transfusion_opsd.native_prediction_credit import hard_native_prediction


def test_nonlinear_condition_can_reverse_credit_but_prediction_secant_does_not():
    # A real finite endpoint can be better even when the local condition
    # derivative initially leads away from it: f(c)=c^3-c, c0=0, c1=2.
    logits = torch.zeros(2, requires_grad=True)
    c0, c1 = torch.tensor(0.), torch.tensor(2.)
    mixed = hard_native_condition({'c': c0}, [{'c': c0}, {'c': c1}], logits,
                                  hard_index=0, differentiable_keys=['c'])['c']
    target = c1**3-c1
    condition_loss = ((mixed**3-mixed)-target).square()
    condition_credit = torch.autograd.grad(condition_loss, logits)[0]
    prediction = hard_native_prediction(c0, [c0, target], logits, hard_index=0)
    prediction_loss = (prediction-target).square()
    prediction_credit = torch.autograd.grad(prediction_loss, logits)[0]
    assert prediction.item() == mixed.item() == 0.
    assert condition_loss.item() == prediction_loss.item() == 36.
    assert condition_credit[1] > 0  # gradient descent discourages the better endpoint
    assert prediction_credit[1] == -18.


def test_binary_prediction_target_has_analytic_credit_and_stopped_alternatives():
    logits = torch.tensor([.7, -.4], dtype=torch.float64, requires_grad=True)
    hard = torch.tensor([1., -2., 4.], dtype=torch.float64, requires_grad=True)
    other = torch.tensor([-3., 1., 2.], dtype=torch.float64, requires_grad=True)
    t = 2.
    output = hard_native_prediction(hard, [hard, other], logits, hard_index=0, temperature=t)
    loss = (output-other.detach()).square().mean()
    decision, direct, alternative = torch.autograd.grad(loss, (logits, hard, other), allow_unused=True)
    p = (logits.detach()/t).softmax(0)
    expected = -2*p.prod()*(other.detach()-hard.detach()).square().mean()/t
    torch.testing.assert_close(decision[1], expected)
    torch.testing.assert_close(direct, 2*(hard.detach()-other.detach())/hard.numel())
    assert alternative is None
    assert torch.equal(output, hard)


def test_detached_control_keeps_native_gradient_and_has_no_decision_gradient():
    hard = torch.tensor([3., -1.], requires_grad=True)
    logits = torch.tensor([.1, .2], requires_grad=True)
    other = torch.tensor([1., 2.])
    output = hard_native_prediction(hard, [other, hard], logits, hard_index=1, connect_decisions=False)
    d, h = torch.autograd.grad(output.square().sum(), (logits, hard), allow_unused=True)
    assert d is None
    torch.testing.assert_close(h, 2*hard.detach())
    assert torch.equal(output, hard)


def test_correct_hard_target_has_zero_decision_credit():
    hard = torch.tensor([3., -1.])
    logits = torch.tensor([1., -1.], requires_grad=True)
    output = hard_native_prediction(hard, [torch.zeros_like(hard), hard], logits, hard_index=1)
    (output-hard).square().mean().backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize('alternatives,index,temperature', [
    ([torch.zeros(2), torch.zeros(3)], 0, 1.),
    ([torch.ones(2), torch.zeros(2)], 0, 1.),
    ([torch.zeros(2), torch.full((2,), float('nan'))], 0, 1.),
    ([torch.zeros(2), torch.zeros(2)], 0, 0.),
])
def test_rejects_invalid_counterfactual_support(alternatives, index, temperature):
    with pytest.raises(ValueError):
        hard_native_prediction(torch.zeros(2), alternatives, torch.zeros(2),
                               hard_index=index, temperature=temperature)
