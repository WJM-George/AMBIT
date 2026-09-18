import copy
import json

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.top_checkpoints import (
    METRIC_DIRECTIONS, OPERATIONS, SCALAR_METRICS, guarded_selection,
    relative_metric_score, retain_top_checkpoints, _sha,
)
from scripts.t2a.rl.train_editing_opsd_repaired_fresh import POLICY, validate_configuration


def diagnostics(gain=1.):
    return dict(groups={op:dict(requests=10,scalar_metrics={
        m:dict(relative_gain_percent=gain) for m in SCALAR_METRICS}) for op in OPERATIONS})


def test_large_spatial_gain_cannot_qualify_a_regressing_model():
    base={m:1. for m in METRIC_DIRECTIONS}
    metrics={m:1.+METRIC_DIRECTIONS[m]*.01 for m in base}
    metrics.update(CRW=.5,LSD=1.02)
    score=relative_metric_score(metrics,base)
    assert score['score']>0
    result=guarded_selection(score,diagnostics(),POLICY)
    assert not result['eligible']
    assert any(x['metric']=='LSD' and x['scope']=='overall' for x in result['guardrail_failures'])


def test_operation_regression_blocks_good_aggregate_and_missing_data_fails_closed():
    score=relative_metric_score({m:1.+METRIC_DIRECTIONS[m]*.01 for m in METRIC_DIRECTIONS},
                                {m:1. for m in METRIC_DIRECTIONS})
    d=diagnostics();d['groups']['event_removal']['scalar_metrics']['Paired CLAP']['relative_gain_percent']=-4.
    assert not guarded_selection(score,d,POLICY)['eligible']
    d['groups']['event_removal']['scalar_metrics']['Paired CLAP']['relative_gain_percent']=None
    with pytest.raises(ValueError):guarded_selection(score,d,POLICY)
    with pytest.raises(ValueError):guarded_selection(score,None,POLICY)


def test_real_checkpoint_manager_filters_before_ranking_and_uses_median(tmp_path):
    config=tmp_path/'config.json';config.write_text(json.dumps(dict(validation_ordinals=[1],evaluation_seeds=[2])))
    baseline={m:1. for m in METRIC_DIRECTIONS}
    def evaluation(step,metrics):
        p=tmp_path/f'EVALUATION_step{step:06d}.json'
        p.write_text(json.dumps(dict(step=step,metrics=metrics,requests=1,outputs=1,target_audio_in_inference=False)))
        return p
    evaluation(0,baseline)
    for step,gains in [(1,dict.fromkeys(baseline,2.)),
                       (2,dict(dict.fromkeys(baseline,1.),CRW=50.)),
                       (3,dict(dict.fromkeys(baseline,10.),LSD=-2.))]:
        metrics={m:1.+METRIC_DIRECTIONS[m]*gains[m]/100 for m in baseline}
        temp=tmp_path/'new.pt'
        torch.save(dict(step=step,model_sha256=str(step),config_sha256=_sha(config)),temp)
        temp.replace(tmp_path/'resume_latest.pt')
        (tmp_path/'RESUME.json').write_text(json.dumps(dict(step=step,model_sha256=str(step))))
        ledger=retain_top_checkpoints(tmp_path,evaluation(step,metrics),config,POLICY,diagnostics=diagnostics())
    assert ledger['ranked_steps']==[1,2]
    assert not ledger['history']['3']['eligible']
    assert not (tmp_path/'top_checkpoints/step-00000003.pt').exists()
    assert (tmp_path/'top_checkpoints/step-00000001.pt').exists()


def test_fresh_repair_contract_requires_new_optimizer_scope_and_fixed_guardrails():
    from stable_audio_tools.paths import opsd_config_path
    parent=opsd_config_path()
    if not parent.exists():
        pytest.skip('Pinned eight-GPU fresh config is not installed.')
    q=json.load(open(parent));q.pop('resize_parent_config')
    q.update(removal_repair=dict(version='removal_paired_v1',paired_native_weight=1.),top_checkpoint_policy=copy.deepcopy(POLICY))
    q['paired_catalog_scope']='full_train_1m'
    validate_configuration(q)
    q['top_checkpoint_policy']['maximum_operation_regression_percent']=10.
    with pytest.raises(ValueError):validate_configuration(q)
