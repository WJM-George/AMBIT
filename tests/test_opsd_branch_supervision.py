import copy
import math

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd import branch_request_supervision as branch
from stable_audio_tools.training.transfusion_opsd.request_paired_supervision import covered_request, feedback_coverage
from test_opsd_request_paired_fallback import OPERATIONS, learner_for, stats


def receipt(ar, rf):
    return dict(enabled=ar or rf, recipe=branch.RECIPE['version'], AR=ar, RF=rf,
        execution_teacher=False, native_joint_loss=2., AR_CE=1. if ar else None,
        RF_MSE=1. if rf else None, structured_loss=0.)


@pytest.mark.parametrize('ar,rf', [(False,False),(True,False),(False,True),(True,True)])
def test_exactly_one_source_per_branch(ar, rf):
    route=dict(execution_AR=ar,execution_RF=rf,execution_joint=ar and rf,execution_RF_mass=float(rf))
    result=covered_request(route,receipt(not ar,not rf))
    assert result['AR_budget']==result['RF_budget']==1.
    assert result['covered_AR'] and result['covered_RF']
    for key in ('AR','RF'):
        broken=receipt(not ar,not rf);broken[key]=not broken[key]
        broken['enabled']=broken['AR'] or broken['RF']
        broken['AR_CE']=1. if broken['AR'] else None
        broken['RF_MSE']=1. if broken['RF'] else None
        with pytest.raises(RuntimeError):covered_request(route,broken)


@pytest.mark.parametrize('coefficients', [[0.,.03,0.,.02],[.5,.5],[0.,0.],[]])
def test_fixed_terminal_budget_keeps_selection_and_relative_preferences(coefficients):
    result,mass=branch.normalize_selected_coefficients(coefficients)
    assert math.fsum(result)==pytest.approx(float(mass>0))
    assert [i for i,c in enumerate(result) if c>0]==[i for i,c in enumerate(coefficients) if c>0]
    for i,c in enumerate(coefficients):assert result[i]*mass==pytest.approx(c)


@pytest.mark.parametrize('bad', [[-1.],[float('nan')],[float('inf')]])
def test_nonfinite_or_negative_selected_mass_is_rejected(bad):
    with pytest.raises(ValueError):branch.normalize_selected_coefficients(bad)


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('qualified,selected', [(False,0),(True,0),(True,1)])
def test_actual_backward_skips_gt_ar_when_only_dit_needs_correction(monkeypatch,operation,qualified,selected):
    initial=stats(qualified,selected)
    initial['terminal_RF_coefficients']=[1.]*selected
    learner,item,seen,accesses=learner_for(monkeypatch,operation,initial)
    learner.q['request_paired_correction']=copy.deepcopy(branch.RECIPE)
    calls=[]
    def objective(adapter,ar,target,metadata,mask,*,use_AR,use_RF,seed):
        calls.append((use_AR,use_RF,metadata))
        a=(adapter.ar_weight.square()+adapter.shared_weight.square()) if use_AR else None
        r=(adapter.dit_weight.square()+adapter.shared_weight.square()) if use_RF else None
        return sum(v for v in (a,r) if v is not None),a,r
    monkeypatch.setattr(branch,'branch_objective',objective)
    result=learner.backward_self(item,scale=.5)
    correction=result['paired_request_correction']
    assert result['request_supervision']['covered']
    assert not learner.adapter.training
    if qualified and selected:
        assert not accesses and not calls
    else:
        assert accesses==[3]
        assert calls[0][:2]==(not qualified,True)
        assert correction['structured_loss']==0
        assert all(v>0 and math.isfinite(v) for v in correction['gradient_probe'].values())
        assert set(correction['gradient_probe'])==set(branch.gradient_parameters(correction))
        if qualified:assert learner.adapter.ar_weight.grad is None or learner.adapter.ar_weight.grad==0
        assert float(learner.adapter.dit_weight.grad)/8==pytest.approx(4./16)
    result.update(requested_operation=operation,terminal_selection=None)
    row=feedback_coverage([result],require_complete=True)[operation]
    assert row['covered_AR_requests']==row['covered_RF_requests']==1
    assert row['execution_AR_requests']+row['GT_AR_requests']==1
    assert row['execution_RF_requests']+row['GT_RF_requests']==1


def test_zero_selected_budget_cannot_be_reported_as_full_rf_coverage():
    route=dict(execution_AR=True,execution_RF=True,execution_joint=True,execution_RF_mass=.2)
    with pytest.raises(ValueError,match='fixed branch budget'):covered_request(route,receipt(False,False))


def test_branch_configuration_rejects_budget_drift():
    import json
    from scripts.t2a.rl.train_editing_opsd_repaired_fresh import validate_configuration
    from stable_audio_tools.paths import opsd_config_path
    path=opsd_config_path()
    if not path.exists():
        pytest.skip('Pinned exclusive-branch config is not installed.')
    with open(path) as f:q=json.load(f)
    q['request_paired_correction']=copy.deepcopy(branch.RECIPE)
    validate_configuration(q)
    q['spatial_recipe']['terminal_RF_weight']=.5
    with pytest.raises(ValueError):validate_configuration(q)
