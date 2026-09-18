from types import SimpleNamespace

import pytest
import torch

from stable_audio_tools.paths import data_path

from stable_audio_tools.training.transfusion_opsd.editing_request_constraints import (
    parse_edit_request, bind_edit_target, native_field_sites, frozen_text_targets,
)
from stable_audio_tools.training.transfusion_opsd.editing_spatial_retention import reference_field_targets
from stable_audio_tools.training.transfusion_opsd.native_prefix_supervision import _sites
from stable_audio_tools.training.transfusion_opsd.reference_prefix_retention import reference_prefix_targets
from stable_audio_tools.training.transfusion_opsd.removal_retention import apply_removal_retention
from stable_audio_tools.training.transfusion_opsd.removal_paired_supervision import (
    validate_removal_pair, removal_loss,
)


@pytest.fixture(scope='module')
def codec():
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    path = data_path('sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    if not path.exists():
        pytest.skip('Native codec artifact is not installed.')
    return ModelScenePlanCodecV4(path)


def scene():
    def source(sid, text):
        return dict(source_id=sid, kind='sound', description=text, gain_db=0.,
            activity=dict(onset_sec=.5, offset_sec=3.5), trajectory=dict(type='static',
            position=dict(azimuth_deg=10., elevation_deg=0., distance_m=1.)))
    return dict(sample_id='test', duration_sec=4., room=dict(type='dry'), sources=[
        source('source_0', 'A sharp electronic alarm rings continuously.'),
        source('source_1', 'A deep resonant didgeridoo plays a low drone.')])


def current_targets(codec, request, operation='event_removal'):
    tokens = codec.encode(scene())['input_ids']
    plan = codec.decode(tokens.tolist())
    logits = torch.zeros(len(tokens)-1, codec.vocab_size)
    holds, coarse = reference_field_targets(codec, tokens.tolist(), plan, codec.allowed_next_ids, logits)
    item = dict(tokens=tokens, plan=plan, row=dict(operation=operation, request=request),
                holds=holds, coarse=coarse, text_reference=frozen_text_targets(
                    codec, tokens.tolist(), logits, codec.allowed_next_ids))
    facts = parse_edit_request(request, operation)
    return item, facts, bind_edit_target(plan, facts), logits


def test_current_and_reference_prefixes_release_removal_identity_count_and_geometry(codec):
    request = 'Remove the sound described as "A deep resonant didgeridoo plays a low drone." entirely.'
    item, facts, binding, logits = current_targets(codec, request)
    ids = item['tokens'].tolist()
    numeric, _ = native_field_sites(codec, ids)
    atoms, _ = _sites(codec, ids)
    original = {h['position'] for h in item['holds']}
    assert numeric[('scene', 'num_sources')] in original  # Reproduces the old conflict.
    assert atoms[('source_1', 'identity')] in original
    policy = apply_removal_retention(codec, item, facts, binding)
    retained = {h['position'] for h in item['holds']}
    assert numeric[('scene', 'num_sources')] not in retained
    assert atoms[('source_1', 'identity')] not in retained
    assert numeric[('source_0', 'kind')] in retained
    assert all(pos not in retained for (sid, _), pos in numeric.items() if sid == 'source_1')
    assert all(h['source_id'] != 'source_1' for h in item['coarse'])
    assert {h['field'] for h in item['text_reference']} == {'source_0/<description>'}
    ref = reference_prefix_targets(codec, ids, item['plan'], facts, binding, logits, codec.allowed_next_ids)
    assert ref['excluded_removed_sources'] == policy['excluded_removed_sources'] == ['source_1']
    assert 'scene/num_sources' not in {h['field'] for h in ref['structure']}
    assert all(not h['field'].startswith('source_1/') for h in ref['structure'] + ref['text'])
    assert not policy['reference_velocity_allowed']


@pytest.mark.parametrize('instruction,excluded', [
    ('Remove the sound described as "Unidentifiable resonances." entirely.', ['source_0','source_1']),
    ('Remove the music described as "A jazz guitar solo." entirely.', []),
    ('Delete the specified event.', ['source_0','source_1']),
])
def test_ambiguous_absent_and_unparsed_removal_never_anchor_count(codec, instruction, excluded):
    item, facts, binding, logits = current_targets(codec, instruction)
    assert not binding['available']
    policy = apply_removal_retention(codec, item, facts, binding)
    assert policy['excluded_removed_sources'] == excluded
    count = native_field_sites(codec, item['tokens'].tolist())[0][('scene','num_sources')]
    assert count not in {h['position'] for h in item['holds']}
    ref = reference_prefix_targets(codec, item['tokens'].tolist(), item['plan'], facts, binding,
        logits, codec.allowed_next_ids, operation='event_removal')
    assert ref['excluded_removed_sources'] == excluded
    assert 'scene/num_sources' not in {h['field'] for h in ref['structure']}


@pytest.mark.parametrize('operation', ['event_addition','stationary_spatial_relocation',
                                     'static_to_linear','linear_to_static'])
def test_removal_mask_does_not_change_other_operations(codec, operation):
    item, facts, binding, _ = current_targets(codec,
        'Move the sound described as "A deep resonant didgeridoo plays a low drone." to azimuth -90 degrees.', operation)
    before = {k: [h['position'] for h in item[k]] for k in ('holds','coarse','text_reference')}
    policy = apply_removal_retention(codec, item, facts, binding)
    assert before == {k: [h['position'] for h in item[k]] for k in before}
    assert policy['reference_velocity_allowed']


def test_complete_collect_uses_removal_mask_and_skips_conflicting_velocity(codec, monkeypatch):
    from scripts.t2a.rl.train_editing_opsd_complete import CompleteLearner, spatial
    request = 'Remove the sound described as "A deep resonant didgeridoo plays a low drone." entirely.'
    item, _, _, logits = current_targets(codec, request)
    item.update(obs=object(), terminals=[], enabled=False)
    monkeypatch.setattr(spatial.SpatialLearner, 'collect', lambda self, ordinal: item)
    learner = CompleteLearner.__new__(CompleteLearner)
    learner.q = dict(complete_recipe=dict(request_tolerances={}))
    learner.costs = dict(execution_teachers=0)
    learner.reference = SimpleNamespace(student_logits=lambda *args: logits[None])
    learner.adapter = SimpleNamespace(codec=codec, allowed_next_ids=lambda obs, prefix: codec.allowed_next_ids(prefix))
    result = learner.collect(1)
    assert result['velocity_reference'] is None
    assert result['removal_retention']['released_positions']['holds'] > 0
    assert result['request_constraints']['targets'] == []  # No invented count label.
    assert learner.costs['execution_teachers'] == 0


def paired_fixture():
    row = dict(operation='event_removal', pair_ordinal=3, pair_id='pair3', request='Remove the bell.',
               model_num_samples=1024, source_latent_tensor_sha256='a'*64)
    metadata = dict(operation=row['operation'],pair_ordinal=3,pair_id='pair3',raw_edit_request=row['request'],
        model_num_samples=1024,source_foa_latent_tensor_sha256='a'*64,
        target_foa_latent_tensor_sha256='b'*64, editing_split='train',
        model_sceneplan=dict(sources=[dict(source_id='source_0'),dict(source_id='source_2')]))
    return row, dict(metadata=[metadata])


@pytest.mark.parametrize('key,value', [('pair_id','wrong'), ('raw_edit_request','Add the bell.'),
    ('source_foa_latent_tensor_sha256','c'*64), ('editing_split','validation'),
    ('operation','event_addition'), ('target_foa_latent_tensor_sha256',None)])
def test_paired_removal_rejects_wrong_provenance(key, value):
    row,batch = paired_fixture();batch['metadata'][0][key] = value
    with pytest.raises(ValueError):validate_removal_pair(row,batch)


def test_paired_removal_uses_true_count_and_normalizes_over_all_requests():
    row,batch = paired_fixture()
    params = [torch.tensor(2.,requires_grad=True) for _ in range(3)]
    seen = []
    def move(batch, device):
        return dict(plan_labels=torch.tensor([[1,2,-100]])), torch.zeros(1,64,3), batch['metadata'], torch.tensor([[True,True,False]])
    def native_loss(adapter,teacher,batch,cfg,device,step,rank,world,den):
        seen.append((cfg['seed'],step,rank,world,den.tolist()))
        sums = torch.stack([p.square()*n for p,n in zip(params, [2,128,1])])
        return (sums/den).sum(), sums.detach(), (), None
    native = SimpleNamespace(_move_joint_batch=move, batch_loss=native_loss)
    loss, report = removal_loss(native,None,None,batch,dict(seed=42),'cpu',row=row,step=540,scale=.5,weight=1.)
    loss.backward()
    # Two requests/rank and eight-rank averaging give dL/16 per request,
    # regardless of how many other requests have qualified teachers.
    assert all(float(p.grad)/8 == pytest.approx(4./16) for p in params)
    assert seen[0][1:] == (0,0,1,[2.,128.,1.])
    assert report['target_source_ids'] == ['source_0','source_2']
    assert report['execution_teacher'] is False and report['enabled']
    again,_ = removal_loss(native,None,None,batch,dict(seed=42),'cpu',row=row,step=540,scale=.5,weight=1.)
    assert seen[-1][0] == seen[0][0]
    removal_loss(native,None,None,batch,dict(seed=42),'cpu',row=row,step=541,scale=.5,weight=1.)
    assert seen[-1][0] != seen[0][0]


def test_repair_transition_preserves_full_state_and_rejects_unrelated_changes(tmp_path):
    from scripts.t2a.rl.train_editing_opsd_removal_repair import transition_state, validate_repair, base
    from scripts.t2a.rl.train_editing_opsd_eight_gpu import EXECUTION
    parent = dict(output=str(tmp_path/'parent'),physical_gpus=list(range(8)),
        learning_rates={'a':1e-6},seed=10,request_rows_per_rank=2,global_request_batch=16,
        paired_rows_per_rank=64,paired_microbatch=48,global_paired_batch=512,native_plan_audit_every=25,
        base_checkpoint=dict(step=40000,path='base',sha256='base'))
    parent_path = tmp_path/'parent.json';parent_path.write_text(json.dumps(parent))
    checkpoint = tmp_path/'protected.pt';checkpoint.write_bytes(b'identified full state')
    q = copy.deepcopy(parent);q['output'] = str(tmp_path/'repaired')
    q['removal_repair'] = dict(version='removal_paired_v1',paired_native_weight=1.,
        parent_config=dict(path=str(parent_path),sha256=base.sha(parent_path)),
        parent_checkpoint=dict(path=str(checkpoint),sha256=base.sha(checkpoint),step=540,model_sha256='model'))
    qpath = tmp_path/'repaired.json';qpath.write_text(json.dumps(q));q['config_path'] = str(qpath)
    state = dict(step=540,world_size=8,model={'p':torch.ones(1)},model_sha256='model',
        config_sha256=base.sha(parent_path),original_checkpoint=parent['base_checkpoint'],initial_overlay=None,
        execution={k:q[k] for k in EXECUTION},optimizer=dict(
            param_groups=[dict(group_name='a',lr=1e-6,betas=(.9,.95),weight_decay=.001)],
            state={0:dict(step=540,exp_avg=torch.ones(1),exp_avg_sq=torch.ones(1))}),
        rank_states=[dict(random=(),numpy=(),cpu_rng=[],cuda_rng=[],
            request=dict(seed=11,world=8,rank=r,cursor=4),paired=dict(seed=12,world=8,rank=r,cursor=128)) for r in range(8)])
    new, changed = transition_state(q,parent,state,checkpoint)
    assert changed and new['config_sha256'] == base.sha(qpath)
    for key in ('optimizer','model','rank_states'):
        assert new[key] is state[key]
    wrong = copy.deepcopy(q);wrong['learning_rates']['a'] *= 2
    with pytest.raises(ValueError):validate_repair(wrong,parent)
    wrong = copy.deepcopy(q);wrong['output'] = parent['output']
    with pytest.raises(ValueError):validate_repair(wrong,parent)
    state['optimizer']['state'][0]['step'] = 539
    with pytest.raises(ValueError):transition_state(q,parent,state,checkpoint)
import copy
import json
