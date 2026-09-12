import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_decision_condition import hard_native_condition
from stable_audio_tools.training.transfusion_opsd.native_decision_credit import execution_decision_derivative


@pytest.mark.parametrize('connected',[False,True])
def test_execution_credit_routes_to_actual_decision_graph_without_private_dit_update(connected):
    shared=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    ar=torch.tensor([.2,-.4],dtype=torch.float64,requires_grad=True)
    dit=torch.tensor(1.2,dtype=torch.float64,requires_grad=True)
    logits=ar+shared*torch.tensor([1.,-1.],dtype=torch.float64)
    c=torch.tensor([[[.7,1.1]]],dtype=torch.float64,requires_grad=True)
    alternative=torch.tensor([[[1.2,.4]]],dtype=torch.float64)
    inputs=hard_native_condition({'input_concat_cond':c},
        [{'input_concat_cond':c.detach()},{'input_concat_cond':alternative}],logits,
        hard_index=0,differentiable_keys=['input_concat_cond'],temperature=4.,
        connect_decisions=connected)
    prediction=(dit+shared)*inputs['input_concat_cond']
    target=torch.zeros_like(prediction,requires_grad=True)
    loss=(prediction-target.detach()).square().mean()
    decision,condition=execution_decision_derivative(loss,logits,c)
    assert all(p.grad is None for p in (shared,ar,dit,c,target))
    assert torch.equal(inputs['input_concat_cond'],c)
    torch.testing.assert_close(condition,(dit.detach()+shared.detach())**2*c.detach())
    if connected:
        assert decision is not None and decision.abs().sum()>0
        logits.backward(decision)
        torch.testing.assert_close(ar.grad,decision)
        # The shared leaf gets only its AR-path derivative in this credit term.
        torch.testing.assert_close(shared.grad,decision[0]-decision[1])
    else:
        assert decision is None and ar.grad is None and shared.grad is None
    assert dit.grad is None and target.grad is None


def test_plan_matched_residual_still_updates_executor_after_credit_extraction():
    ar=torch.nn.Linear(2,2,bias=False,dtype=torch.float64)
    dit=torch.nn.Linear(2,1,bias=False,dtype=torch.float64)
    x=torch.tensor([[.2,.7]],dtype=torch.float64)
    logits=ar(x)[0]
    c=x.detach().clone().requires_grad_(True);alt=torch.tensor([[.5,1.]],dtype=torch.float64)
    inp=hard_native_condition({'c':c},[{'c':c.detach()},{'c':alt}],logits,
        hard_index=0,differentiable_keys=['c'])['c']
    loss=(dit(inp)-2.).square().mean()
    g,_=execution_decision_derivative(loss,logits,c);logits.backward(g)
    assert ar.weight.grad is not None and dit.weight.grad is None
    residual=(dit(alt)-1.).square().mean();residual.backward()
    assert dit.weight.grad is not None and dit.weight.grad.abs().sum()>0
