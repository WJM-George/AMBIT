import pytest
import torch

from stable_audio_tools.inference.sampling import sample_discrete_euler
from stable_audio_tools.training.transfusion_opsd.native_continuation_diagnostic import continue_native_query


def test_same_state_continuation_matches_full_native_trajectory_exactly():
    times=torch.linspace(1.,0.,101);initial=torch.randn(1,2,8,generator=torch.Generator().manual_seed(42))
    velocity=lambda z,t: z*.2+t[:,None,None]*.1
    states=[]
    expected=sample_discrete_euler(velocity,initial,times,disable_tqdm=True,
        callback=lambda v:states.append(v['x'].clone()))
    query=dict(z=states[50],time=times[50].reshape(1),query_index=50)
    actual=continue_native_query(velocity,query,times)
    injected=continue_native_query(velocity,query,times,first_velocity=velocity(query['z'],query['time']))
    assert torch.equal(actual,expected) and torch.equal(injected,expected)


def test_only_first_velocity_is_injected_and_time_does_not_restart():
    times=torch.linspace(1.,0.,101);seen=[]
    def zero(z,t):seen.append(float(t));return torch.zeros_like(z)
    query=dict(z=torch.zeros(1,2,8),time=times[75].reshape(1),query_index=75)
    actual=continue_native_query(zero,query,times,first_velocity=torch.ones_like(query['z']))
    assert len(seen)==24 and seen[0]==float(times[76])
    assert torch.equal(actual,torch.full_like(actual,times[76]-times[75]))


def test_unaligned_query_time_is_rejected():
    times=torch.linspace(1.,0.,101)
    query=dict(z=torch.zeros(1,2,8),time=torch.tensor([1.]),query_index=50)
    with pytest.raises(ValueError,match='State and time'):
        continue_native_query(lambda z,t:z,query,times)
