import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.anchored_velocity_teacher import (
    anchored_velocity_teacher, execution_teacher_coefficients)


def test_positive_negative_and_neutral_teacher_directions():
    anchor=torch.tensor([[[1.,2.]],[[1.,2.]],[[1.,2.]]],requires_grad=True)
    endpoint=torch.tensor([[[3.,6.]],[[3.,6.]],[[3.,6.]]],requires_grad=True)
    coefficients=torch.tensor([1.,-1.,0.],requires_grad=True)
    result=anchored_velocity_teacher(anchor,endpoint,coefficients,torch.ones(3,2,dtype=torch.bool))
    assert torch.equal(result['target'],torch.tensor([[[3.,6.]],[[-1.,-2.]],[[1.,2.]]]))
    assert all(not x.requires_grad for x in result.values())
    student=anchor.detach().clone().requires_grad_()
    ((student-result['target'])**2).sum().backward()
    assert (student.grad[0]<0).all() and (student.grad[1]>0).all()
    assert (student.grad[2]==0).all()
    assert anchor.grad is endpoint.grad is coefficients.grad is None


def test_teacher_gradient_matches_signed_rvm_at_arbitrary_student():
    anchor=torch.tensor([[[1.,-2.,9.]],[[2.,3.,0.]]])
    endpoint=torch.tensor([[[4.,1.,-9.]],[[1.,-2.,5.]]])
    mask=torch.tensor([[True,True,False],[True,True,True]])
    coefficients=torch.tensor([-.75,.25])
    student=torch.tensor([[[3.,2.,8.]],[[4.,-1.,2.]]],requires_grad=True)
    target=anchored_velocity_teacher(anchor,endpoint,coefficients,mask)['target']
    def mse(value):
        return ((student-value).square()*mask[:,None]).sum((1,2))/(mask.sum(-1)*student.shape[1])
    teacher_loss=mse(target).mean()
    rvm_loss=(coefficients*mse(endpoint)+(1-coefficients)*mse(anchor)).mean()
    g1=torch.autograd.grad(teacher_loss,student,retain_graph=True)[0]
    g2=torch.autograd.grad(rvm_loss,student)[0]
    assert torch.allclose(g1,g2,atol=1e-6,rtol=0)
    assert g1[0,0,2]==0


def test_teacher_matches_nft_implicit_positive_negative_gradients():
    anchor=torch.tensor([[[1.,2.]],[[2.,-1.]]])
    endpoint=torch.tensor([[[3.,4.]],[[0.,3.]]])
    coefficient=torch.tensor([-.5,.75])
    mask=torch.ones(2,2,dtype=torch.bool)
    student=torch.tensor([[[4.,0.]],[[3.,2.]]],requires_grad=True)
    target=anchored_velocity_teacher(anchor,endpoint,coefficient,mask)['target']
    reward=(coefficient+1)/2
    nft=(reward*(student-endpoint).square().mean((1,2))
        +(1-reward)*(2*anchor-student-endpoint).square().mean((1,2))).mean()
    distill=(student-target).square().mean()
    a=torch.autograd.grad(nft,student,retain_graph=True)[0]
    b=torch.autograd.grad(distill,student)[0]
    assert torch.allclose(a,b,atol=1e-6,rtol=0)


def test_rms_cap_ignores_padding_and_reports_changed_coefficients():
    mask=torch.tensor([[True,True,False],[True,True,True]])
    anchor=torch.zeros(2,1,3)
    endpoint=torch.tensor([[[4.,4.,10000.]],[[.1,.1,.1]]])
    result=anchored_velocity_teacher(anchor,endpoint,torch.tensor([-1.,1.]),mask,max_rms_delta=.5)
    assert torch.all(result['actual_rms_delta']<=.5+1e-6)
    assert torch.allclose(result['effective_coefficients'],torch.tensor([-.125,1.]))
    assert result['target'][0,0,2]==0


def test_relative_teacher_weights_are_centered_stopped_and_bounded():
    probabilities=torch.tensor([.01,.33,.33,.33],requires_grad=True)
    weights=execution_teacher_coefficients(probabilities,has_qualified_positive=True)
    assert abs(float(weights['signed'].sum()))<1e-6
    assert weights['signed'][0]<0 and (weights['signed'][1:]>0).all()
    assert (weights['signed'].abs()<=1).all() and (weights['positive']<=1).all()
    assert not weights['signed'].requires_grad
    neutral=execution_teacher_coefficients(probabilities,has_qualified_positive=False)
    assert not neutral['active'] and torch.count_nonzero(neutral['signed'])==0 and torch.count_nonzero(neutral['positive'])==0


@pytest.mark.parametrize('group_size',[3,4,7])
def test_equal_rewards_do_not_create_signed_learning_signal(group_size):
    weights=execution_teacher_coefficients(torch.full((group_size,),1/group_size),has_qualified_positive=True)
    assert torch.count_nonzero(weights['signed'])==0


def test_teacher_cannot_receive_gradients_through_trust_radius():
    with pytest.raises(ValueError):
        anchored_velocity_teacher(torch.zeros(1,1,2),torch.ones(1,1,2),torch.ones(1),
            torch.ones(1,2,dtype=torch.bool),max_rms_delta=torch.tensor(.5,requires_grad=True))


@pytest.mark.parametrize('value',[float('nan'),1.1,-1.1])
def test_invalid_coefficient_is_rejected(value):
    with pytest.raises(ValueError):
        anchored_velocity_teacher(torch.zeros(1,1,2),torch.ones(1,1,2),torch.tensor([value]),torch.ones(1,2,dtype=torch.bool))


@pytest.mark.parametrize('radius',[0.,-1.,float('inf')])
def test_invalid_radius_is_rejected(radius):
    with pytest.raises(ValueError):
        anchored_velocity_teacher(torch.zeros(1,1,2),torch.ones(1,1,2),torch.zeros(1),torch.ones(1,2,dtype=torch.bool),max_rms_delta=radius)
