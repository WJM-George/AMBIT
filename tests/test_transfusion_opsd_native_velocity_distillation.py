import torch
import pytest

from stable_audio_tools.training.transfusion_opsd.native_velocity_distillation import (
    native_velocity_distillation_loss, stratified_native_queries)


def test_equivalent_velocity_errors_at_different_flow_times_get_equal_weight():
    z=torch.ones(2,3,11);time=torch.tensor([.1,.9]);v=torch.ones_like(z)
    target=z-time[:,None,None]*2
    mask=torch.ones(2,11,dtype=torch.bool)
    losses=[native_velocity_distillation_loss(z[i:i+1],time[i:i+1],v[i:i+1],target[i:i+1],mask[i:i+1],fixed_mse_scale=1.) for i in range(2)]
    torch.testing.assert_close(losses[0],losses[1])
    torch.testing.assert_close(losses[0],torch.tensor(1.))


def test_horizon_length_and_padding_do_not_change_plan_weight():
    t=torch.tensor([.5]);losses=[]
    for frames in [11,17]:
        z=torch.ones(1,3,frames);v=torch.zeros_like(z);target=z-t[:,None,None]
        mask=torch.ones(1,frames,dtype=torch.bool);mask[:,-3:]=False;v[:,:,-3:]=1000
        losses.append(native_velocity_distillation_loss(z,t,v,target,mask,fixed_mse_scale=1.))
    torch.testing.assert_close(losses[0],losses[1])


def test_only_student_velocity_receives_gradient():
    z=torch.ones(1,2,4,requires_grad=True);t=torch.tensor([.5],requires_grad=True)
    target=torch.zeros_like(z,requires_grad=True);v=torch.zeros_like(z,requires_grad=True)
    mask=torch.tensor([[True,True,True,False]])
    loss=native_velocity_distillation_loss(z,t,v,target,mask,fixed_mse_scale=1.)
    loss.backward()
    assert z.grad is target.grad is t.grad is None
    assert v.grad[:,:,:3].abs().sum()>0 and v.grad[:,:,3].abs().sum()==0


def test_fixed_time_rotation_covers_the_whole_native_trajectory():
    indices=[q for step in range(25) for q in stratified_native_queries(step)]
    assert sorted(indices)==list(range(100))
    assert stratified_native_queries(25)==stratified_native_queries(0)
    with pytest.raises(ValueError):stratified_native_queries(-1)
