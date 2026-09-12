import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_joint_step import native_private_first_joint_step


def optimizer():
    a, s, d = [torch.nn.Parameter(torch.zeros(())) for _ in range(3)]
    opt = torch.optim.AdamW([
        dict(params=[a],name='ar_body'),dict(params=[s],name='shared'),dict(params=[d],name='dit_private')
    ],lr=1.,weight_decay=0.,eps=0.)
    for p in (a,s,d):
        p.grad = torch.tensor(-1.)
    return opt,a,s,d


def test_reduces_private_AR_without_throttling_shared_execution():
    opt,a,s,d = optimizer()
    result = native_private_first_joint_step(opt,lambda:dict(passed=bool(a <= .25)),factors=(1.,.5,.25,0.))
    assert result['accepted'] and result['ar_private_factor'] == .25
    assert result['shared_factor'] == result['dit_private_factor'] == 1.
    assert float(a) == pytest.approx(.25)
    assert float(s) == pytest.approx(1.) and float(d) == pytest.approx(1.)
    assert len(result['passes']) == 1


def test_fallback_restores_Adam_before_reproposing_and_keeps_full_DiT():
    opt,a,s,d = optimizer()
    result = native_private_first_joint_step(opt,lambda:dict(passed=bool(a <= .5 and s <= .5)),factors=(1.,.5,0.))
    assert result['accepted'] and len(result['passes']) == 2
    assert result['ar_private_factor'] == result['shared_factor'] == .5
    assert result['dit_private_factor'] == 1.
    assert (float(a),float(s),float(d)) == pytest.approx((.5,.5,1.))
    for p in (a,s,d):
        assert opt.state[p]['step'].item() == 1
        assert opt.state[p]['exp_avg'].item() == pytest.approx(-.1)


def test_infeasible_unconstrained_DiT_restores_everything():
    opt,a,s,d = optimizer()
    result = native_private_first_joint_step(opt,lambda:dict(passed=bool(d == 0)),factors=(1.,.5,0.))
    assert not result['accepted'] and not result['optimizer_moments_advanced']
    assert all(p.item() == 0 for p in (a,s,d))
    assert not opt.state
