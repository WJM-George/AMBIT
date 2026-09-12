import copy
import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.guarded_joint_update import guarded_joint_adam_update


class TwoRoutes(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.planner=torch.nn.Parameter(torch.tensor(0.))
        self.shared=torch.nn.Parameter(torch.tensor(0.))
        self.executor=torch.nn.Parameter(torch.tensor(0.))
        self.register_buffer('seen',torch.tensor(0.))


def setup():
    m=TwoRoutes()
    o=torch.optim.AdamW([dict(params=[getattr(m,n)],name=n,lr=.001,weight_decay=0.) for n in ['planner','shared','executor']])
    loss=(m.planner+m.shared-1).square()+(m.executor+m.shared-1).square()
    loss.backward()
    return m,o


FULL=dict(planner=1.,shared=1.,executor=1.)
LIMITED=dict(planner=.25,shared=.25,executor=1.)


def equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def test_actual_native_boundary_limits_planning_but_keeps_executor_update():
    m,o=setup();control,co=setup();co.step()
    # This models a finite greedy boundary that a zero-at-initialization KL
    # cannot prevent crossing on the first unconstrained update.
    result=guarded_joint_adam_update(m,o,trial_scales=[FULL,LIMITED],
        validate_native=lambda:dict(passed=bool(m.planner+m.shared<.0008)))
    assert result.accepted and len(result.trials)==2 and result.scales==LIMITED
    assert result.trials[0]['validation']['passed'] is False
    assert m.planner>0 and m.shared>0 and torch.equal(m.executor,control.executor)
    assert all(float(s['step'])==1 for s in o.state.values())


def test_unscaled_pass_matches_standard_adam_bitwise():
    m,o=setup();reference,ro=setup();ro.step()
    result=guarded_joint_adam_update(m,o,trial_scales=[FULL],validate_native=lambda:dict(passed=True))
    assert result.accepted and equal(m.state_dict(),reference.state_dict())
    assert equal(o.state_dict(),ro.state_dict())


@pytest.mark.parametrize('raises',[False,True])
def test_rejection_restores_existing_moments_buffers_and_random_state(raises):
    m,o=setup();o.step();o.zero_grad();(m.planner+m.executor+m.shared).backward()
    weights=copy.deepcopy(m.state_dict());state=copy.deepcopy(o.state_dict());rng=torch.get_rng_state().clone()
    def bad():
        m.seen.add_(1);torch.rand(5)
        if raises:raise RuntimeError('invalid teacher evidence')
        return dict(passed=False)
    if raises:
        with pytest.raises(RuntimeError,match='invalid teacher'):
            guarded_joint_adam_update(m,o,trial_scales=[FULL,LIMITED],validate_native=bad)
    else:
        result=guarded_joint_adam_update(m,o,trial_scales=[FULL,LIMITED],validate_native=bad)
        assert not result.accepted
    assert equal(m.state_dict(),weights) and equal(o.state_dict(),state)
    assert torch.equal(torch.get_rng_state(),rng)


def test_validation_randomness_does_not_change_accepted_training_rng():
    m,o=setup();rng=torch.get_rng_state().clone()
    def accept():
        torch.rand(3)
        return dict(passed=True)
    guarded_joint_adam_update(m,o,trial_scales=[FULL],validate_native=accept)
    assert torch.equal(torch.get_rng_state(),rng)


@pytest.mark.parametrize('scale',[dict(planner=1.,shared=1.),dict(FULL,executor=0.),dict(FULL,planner=float('nan'))])
def test_missing_or_disabled_branch_scales_are_rejected_before_update(scale):
    m,o=setup();weights=copy.deepcopy(m.state_dict())
    with pytest.raises(ValueError):
        guarded_joint_adam_update(m,o,trial_scales=[scale],validate_native=lambda:dict(passed=True))
    assert equal(m.state_dict(),weights) and not o.state
