import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_execution_teacher import NativeDecisionSite
from stable_audio_tools.training.transfusion_opsd.native_reward_execution_teacher import (
    RewardExecutionObservation, build_native_reward_execution_teacher,
)


def teacher(*, gain=.3, offset=0., temperature=.2, observations=None):
    logits=torch.tensor([2.,0.,-1.,-2.],dtype=torch.float64,requires_grad=True)
    site=NativeDecisionSite.from_request('example','spoken content at the front',(1,2),(10,11,12,13))
    rows=observations if observations is not None else [
        RewardExecutionObservation(token,seed,offset+(-.5 if token==10 else -.5+gain),'passed')
        for token in (10,11) for seed in (31,32)]
    result=build_native_reward_execution_teacher(logits,site=site,executor_fingerprint='1'*64,
        observer_fingerprint='2'*64,evidence_sha256='3'*64,reference_token=10,
        paired_seeds=(31,32),observations=rows,temperature=temperature)
    return logits,result


def test_reward_tilt_preserves_unmeasured_mass_and_stops_teacher_gradient():
    logits,result=teacher()
    p,q=logits.detach().softmax(-1),result.target_logits.softmax(-1)
    torch.testing.assert_close(q[2:],p[2:],rtol=1e-14,atol=1e-14)
    torch.testing.assert_close(q[:2].sum(),p[:2].sum(),rtol=1e-14,atol=1e-14)
    torch.testing.assert_close((q[1]/q[0]).log(),(p[1]/p[0]).log()+torch.tensor(1.5,dtype=torch.float64))
    assert not result.target_logits.requires_grad
    assert result.measured_probability_mass+result.unsupported_probability_mass==1


def test_conditional_variational_identity_matches_arbitrary_distribution():
    logits,result=teacher()
    prior=logits.detach()[:2].softmax(-1)
    optimum=result.target_logits[:2].softmax(-1)
    alternative=torch.tensor([.35,.65],dtype=torch.float64)
    rewards=torch.tensor([-.5,-.2],dtype=torch.float64)
    objective=lambda q: (q*rewards).sum()-.2*(q*(q.log()-prior.log())).sum()
    lhs=objective(optimum)-objective(alternative)
    rhs=.2*(alternative*(alternative.log()-optimum.log())).sum()
    torch.testing.assert_close(lhs,rhs,rtol=1e-12,atol=1e-12)
    assert objective(optimum)>=objective(prior)


def test_reward_strength_matters_and_common_offset_does_not():
    _,a=teacher(gain=.1)
    _,b=teacher(gain=.3)
    _,c=teacher(gain=.3,offset=7.)
    assert b.target_logits.softmax(-1)[1]>a.target_logits.softmax(-1)[1]
    torch.testing.assert_close(b.target_logits,c.target_logits,rtol=1e-14,atol=1e-14)


@pytest.mark.parametrize('problem',['missing_noise','duplicate_noise','uncertain_alternative'])
def test_incomplete_or_unprotected_execution_cannot_supply_teacher(problem):
    rows=[RewardExecutionObservation(token,seed,-.5 if token==10 else -.2,'passed') for token in (10,11) for seed in (31,32)]
    if problem=='missing_noise':rows.pop()
    elif problem=='duplicate_noise':rows.append(rows[-1])
    else:rows[-1]=RewardExecutionObservation(11,32,-.2,'uncertain')
    with pytest.raises(ValueError):teacher(observations=rows)
