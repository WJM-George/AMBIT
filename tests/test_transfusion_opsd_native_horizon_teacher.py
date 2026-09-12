import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_horizon_teacher import shorter_horizon_velocity_target


def test_teacher_reads_same_prefix_and_time_and_preserves_unmatched_suffix():
    z=torch.randn(1,4,12,requires_grad=True);t=torch.tensor([.6])
    base=torch.ones_like(z,dtype=torch.bfloat16)
    seen=[]
    def shorter(x,time):
        seen.append((x.clone(),time.clone(),torch.is_grad_enabled()))
        return torch.full_like(x,1.03,dtype=torch.bfloat16)
    target,receipt=shorter_horizon_velocity_target(z,t,base,shorter,
        student_mask=torch.ones(1,12,dtype=torch.bool),shorter_mask=torch.ones(1,9,dtype=torch.bool))
    assert torch.equal(seen[0][0],z[...,:9]) and torch.equal(seen[0][1],t) and not seen[0][2]
    assert torch.equal(target[...,9:],base[...,9:]) and not torch.equal(target[...,:9],base[...,:9])
    assert target.grad_fn is None and not target.requires_grad and receipt['unmatched_suffix_exact']


@pytest.mark.parametrize('frames',[12,13,0])
def test_nonshorter_or_empty_horizon_rejected(frames):
    z=torch.ones(1,2,12)
    with pytest.raises(ValueError):shorter_horizon_velocity_target(z,torch.tensor([.5]),z,lambda x,t:x,
        student_mask=torch.ones(1,12,dtype=torch.bool),shorter_mask=torch.ones(1,frames,dtype=torch.bool))


def test_wrong_teacher_layout_rejected():
    z=torch.ones(1,2,12)
    with pytest.raises(ValueError):shorter_horizon_velocity_target(z,torch.tensor([.5]),z,lambda x,t:x[:,:,:-1],
        student_mask=torch.ones(1,12,dtype=torch.bool),shorter_mask=torch.ones(1,9,dtype=torch.bool))
