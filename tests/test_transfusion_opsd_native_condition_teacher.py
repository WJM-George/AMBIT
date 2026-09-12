import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_condition_teacher import bounded_condition_velocity_target


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_identity_and_padding_are_exact_and_teacher_stops(dtype):
    x=torch.randn(2,3,11,dtype=dtype,generator=torch.Generator().manual_seed(42),requires_grad=True)
    mask=torch.tensor([[True]*8+[False]*3,[True]*11])
    y,r=bounded_condition_velocity_target(x,x,mask)
    assert torch.equal(x,y) and r['identity_exact']
    assert y.grad_fn is None and not y.requires_grad
    other=x.detach().clone();other[:,:,8:]+=100
    z,r=bounded_condition_velocity_target(x,other,mask)
    assert torch.equal(z[0,:,8:],x[0,:,8:]) and max(r['relative_correction'])<=.05+1e-7


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_bound_is_applied_after_output_rounding_per_example(dtype):
    g=torch.Generator().manual_seed(14)
    x=torch.randn(4,64,37,generator=g).to(dtype)
    other=x+torch.randn(x.shape,generator=g).to(dtype)*.4
    mask=torch.ones(4,37,dtype=torch.bool);mask[0,12:]=False
    y,r=bounded_condition_velocity_target(x,other,mask,maximum_relative_rms=.013)
    count=mask.sum(-1)*64
    rms=lambda q:(q.float().masked_fill(~mask[:,None],0).square().sum((1,2))/count).sqrt()
    assert (rms(y.float()-x.float())<=.013*rms(x)).all()
    assert not torch.equal(y,x) and y.dtype==dtype


def test_zero_reference_cannot_acquire_unbounded_velocity():
    x=torch.zeros(1,2,3)
    y,r=bounded_condition_velocity_target(x,torch.ones_like(x),torch.ones(1,3,dtype=torch.bool))
    assert torch.equal(x,y) and r['identity_exact']


def test_foreign_geometry_or_invalid_budget_rejected():
    x=torch.ones(1,2,3);mask=torch.ones(1,3,dtype=torch.bool)
    with pytest.raises(ValueError):bounded_condition_velocity_target(x,x[:,:,:2],mask)
    with pytest.raises(ValueError):bounded_condition_velocity_target(x,x,mask,maximum_relative_rms=.2)
