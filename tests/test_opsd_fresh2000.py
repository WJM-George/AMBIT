import copy
import json
from pathlib import Path

import pytest
import torch

from scripts.t2a.rl.train_editing_opsd_fresh2000 import validate_configuration, validate_state, freeze_candidate, base
from scripts.t2a.rl.launch_editing_opsd_fresh2000 import training_command, evaluation_specs
from scripts.t2a.rl.report_editing_opsd_fresh2000 import matched_diagnostics


def configuration():
    return dict(initial_overlay=None,base_checkpoint=dict(step=40000,path='original.pt',sha256='original'),
        maximum_updates=2000,physical_gpus=[4,5,6,7],save_every=500,global_request_batch=16,
        global_paired_batch=512,request_rows_per_rank=4,paired_rows_per_rank=128,paired_microbatch=48,
        connected_credit=False,defer_audio_evaluation=True,evaluation_save_audio=False,
        validation_ordinals=list(range(500)),evaluation_seeds=[10,11],candidate_every=250,
        top_checkpoint_policy=dict(keep=5,ranking='mean_signed_relative_improvement_percent'),
        selective_recipe=dict(reference_native_prefix=True,select_same_plan_improvements=True),
        native_plan_audit_every=25,learning_rates={'a':1e-6},seed=7)


@pytest.mark.parametrize('change', [dict(initial_overlay={'path':'old'}),
    dict(initialization_checkpoint={'step':500}),dict(resize_parent_config='old.json'),
    dict(top_checkpoint_policy=dict(keep=3,ranking='mean_signed_relative_improvement_percent')),
    dict(validation_ordinals=[1]*500)])
def test_fresh_contract_rejects_old_initialization_or_wrong_retention(change):
    q=configuration();validate_configuration(q);q.update(change)
    with pytest.raises(ValueError):validate_configuration(q)


def test_recovery_rejects_old_config_and_moments_at_step_zero(tmp_path):
    q=configuration();p=tmp_path/'config.json';p.write_text(json.dumps(q));q['config_path']=str(p)
    ranks=[dict(random=(),numpy=(),cpu_rng=[],cuda_rng=[],
        request=dict(rank=r,world=4,seed=8),paired=dict(rank=r,world=4,seed=9)) for r in range(4)]
    state=dict(step=0,world_size=4,config_sha256=base.sha(p),original_checkpoint=q['base_checkpoint'],
        initial_overlay=None,execution={k:q[k] for k in ('request_rows_per_rank','global_request_batch',
        'paired_rows_per_rank','paired_microbatch','global_paired_batch','native_plan_audit_every')},
        optimizer=dict(param_groups=[dict(group_name='a',lr=1e-6,betas=(.9,.95),weight_decay=.001)],state={}),
        model={'x':torch.zeros(1)},rank_states=ranks)
    validate_state(q,state)
    wrong=copy.deepcopy(state);wrong['config_sha256']='old500'
    with pytest.raises(ValueError):validate_state(q,wrong)
    wrong=copy.deepcopy(state);wrong['optimizer']['state']={0:dict(step=500)}
    with pytest.raises(ValueError):validate_state(q,wrong)
    wrong=copy.deepcopy(state);wrong['step']=2
    with pytest.raises(ValueError):validate_state(q,wrong)


def test_async_candidate_survives_atomic_recovery_rotation(tmp_path):
    torch.save(dict(step=250,value=torch.tensor([1.])),tmp_path/'resume_latest.pt')
    candidate=freeze_candidate(tmp_path,250)
    torch.save(dict(step=251,value=torch.tensor([2.])),tmp_path/'next.pt')
    (tmp_path/'next.pt').replace(tmp_path/'resume_latest.pt')
    assert freeze_candidate(tmp_path,250)==candidate
    assert torch.load(candidate,weights_only=False)['step']==250
    assert torch.load(candidate,weights_only=False)['value'].item()==1.


def test_fresh_start_and_all_eight_candidates_are_scheduled(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps(dict(output=str(tmp_path/'training'))))
    (tmp_path/'PROTOCOL.json').write_text('{}')
    command=training_command(tmp_path,None,2)
    assert '--resume' not in command
    assert command[command.index('--limit-updates')+1]=='2'
    jobs=evaluation_specs(tmp_path)
    assert [j['step'] for j in jobs]==list(range(0,2001,250))
    assert jobs[0]['old'] and not any(j['old'] for j in jobs[1:])


def test_zero_baseline_scalar_ratios_are_null_not_nonfinite():
    rows=[dict(pair_ordinal=i,operation='event_removal',target_domain='speech_only') for i in range(5)]
    b={(i,s):dict(scalar={'GCC':0.}) for i in range(5) for s in [10,11]}
    c={(i,s):dict(scalar={'GCC':1.}) for i in range(5) for s in [10,11]}
    report=matched_diagnostics(b,c,dict(rows=rows),[10,11])
    assert report['paired_scalar_metrics']['GCC']['relative_gain_percent'] is None
    assert report['groups']['speech']['scalar_metrics']['GCC']['relative_gain_percent'] is None
    json.dumps(report,allow_nan=False)
