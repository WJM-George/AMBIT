import copy
import math
from types import SimpleNamespace

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.request_paired_supervision import (
    OPERATIONS, RECIPE, validate_request_pair, request_loss, execution_supervision,
    covered_request, feedback_coverage,
)


def pair(operation, kind='sound'):
    row = dict(operation=operation, pair_ordinal=3, pair_id='pair3', request='The actual instruction',
               model_num_samples=1024, source_latent_tensor_sha256='a'*64)
    meta = dict(operation=operation, pair_ordinal=3, pair_id='pair3', raw_edit_request=row['request'],
        model_num_samples=1024, source_foa_latent_tensor_sha256='a'*64,
        target_foa_latent_tensor_sha256='b'*64, editing_split='train',
        model_sceneplan=dict(sources=[dict(source_id='source_2',kind=kind)]))
    return row, dict(metadata=[meta])


def stats(enabled=False, selected=0):
    return dict(enabled=enabled, terminal_RF=[.5]*selected,
                terminal_RF_coefficients=[.25]*selected)


WEIGHTS = dict(ar_teacher_weight=1., terminal_RF_weight=1.)


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('kind', ['sound','music','speech'])
def test_supported_pairs_use_actual_identity_and_target(operation, kind):
    row,batch = pair(operation,kind)
    result = validate_request_pair(row,batch)
    assert result['operation'] == operation
    assert result['target_source_ids'] == ['source_2']
    assert result['target_kinds'] == [kind]


@pytest.mark.parametrize('key,value', [('pair_ordinal',4),('pair_id','wrong'),
    ('operation','event_removal'),('raw_edit_request','Different request'),('model_num_samples',2048),
    ('source_foa_latent_tensor_sha256','c'*64),('editing_split','test'),
    ('target_foa_latent_tensor_sha256','g'*64)])
def test_mismatched_or_nontraining_pair_is_rejected(key,value):
    row,batch = pair('event_addition');batch['metadata'][0][key]=value
    with pytest.raises(ValueError):validate_request_pair(row,batch)


@pytest.mark.parametrize('enabled,selected,ar_weight,rf_weight,joint', [
    (False,0,1.,1.,False), (True,0,1.,1.,False), (True,1,1.,1.,True),
    (True,2,1.,0.,False), (True,2,0.,1.,False),
])
def test_qualified_teacher_alone_does_not_hide_missing_dit_supervision(enabled,selected,ar_weight,rf_weight,joint):
    result=execution_supervision(stats(enabled,selected),dict(ar_teacher_weight=ar_weight,terminal_RF_weight=rf_weight))
    assert result['execution_joint'] is joint
    if not joint:
        with pytest.raises(RuntimeError):covered_request(result,dict(enabled=False))


class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ar_weight=torch.nn.Parameter(torch.tensor(2.))
        self.dit_weight=torch.nn.Parameter(torch.tensor(2.))
        self.shared_weight=torch.nn.Parameter(torch.tensor(2.))
        self.codec=SimpleNamespace(pad_id=0)

    def student_logits(self,*args):
        return self.ar_weight.expand(1,3,4)


def native(adapter, seen, *, omit_shared=False):
    def move(batch,device):
        return (dict(plan_labels=torch.tensor([[1,2,-100]])),torch.zeros(1,64,3),
                batch['metadata'],torch.tensor([[True,True,False]]))
    def loss(adapter,teacher,batch,cfg,device,step,rank,world,den):
        seen.append(dict(seed=cfg['seed'],den=den.tolist(),step=step,rank=rank,world=world))
        shared=adapter.shared_weight.detach() if omit_shared else adapter.shared_weight
        sums=torch.stack([adapter.ar_weight.square()*2,adapter.dit_weight.square()*128,shared.square()])
        return (sums/den).sum(),sums.detach(),(),None
    return SimpleNamespace(_move_joint_batch=move,batch_loss=loss)


def learner_for(monkeypatch,operation,initial_stats, *, omit_shared=False):
    from scripts.t2a.rl.train_editing_opsd_complete import CompleteLearner,spatial
    from scripts.t2a.experiments.ar_structured_v1 import data
    row,batch=pair(operation)
    monkeypatch.setattr(spatial.SpatialLearner,'backward_self',lambda *args,**kwargs:copy.deepcopy(initial_stats))
    monkeypatch.setattr(data,'collate',lambda *args,**kwargs:batch)
    learner=CompleteLearner.__new__(CompleteLearner)
    learner.q=dict(request_paired_correction=copy.deepcopy(RECIPE),spatial_recipe=WEIGHTS,
        complete_recipe=dict(request_constraint_weight=1.,retained_text_KL_weight=.5,reference_velocity_weight=.1))
    learner.adapter=Adapter().eval();learner.teacher=None;learner.cfg=dict(seed=42)
    learner.device='cpu';learner.step=0;learner.costs={};learner.progress=lambda *args,**kwargs:None
    seen=[];learner.native=native(learner.adapter,seen,omit_shared=omit_shared)
    accesses=[]
    class Pairs:
        def __getitem__(self,ordinal):
            accesses.append(ordinal);return batch
    learner.paired=Pairs()
    learner.trainable={'ar.plan_adapter.plan_head.weight':learner.adapter.ar_weight,
        'ar.editing_dit.postprocess_conv.weight':learner.adapter.dit_weight,
        'ar.editing_dit.transformer.layers.0.pre_norm.gamma':learner.adapter.shared_weight}
    item=dict(row=row,obs=None,tokens=torch.tensor([1,2,3]),request_constraints=dict(targets=[],unavailable=['binding_failed']),
        text_reference=[],velocity_reference=None,removal_retention={},binding=dict(available=False))
    return learner,item,seen,accesses


@pytest.mark.parametrize('operation', OPERATIONS)
@pytest.mark.parametrize('qualified,selected', [(False,0),(True,0),(True,1)])
def test_actual_complete_backward_routes_all_operations_and_reaches_ar_dit_shared(monkeypatch,operation,qualified,selected):
    learner,item,seen,accesses=learner_for(monkeypatch,operation,stats(qualified,selected))
    result=learner.backward_self(item,scale=.5)
    correction=result['paired_request_correction']
    needed=not (qualified and selected)
    assert correction['enabled'] is needed
    assert result['request_supervision']['covered']
    assert accesses==([3] if needed else [])  # Never borrow another paired sample.
    assert not learner.adapter.training
    if needed:
        assert correction['execution_teacher'] is False
        assert all(math.isfinite(v) and v>0 for v in correction['gradient_probe'].values())
        assert all(float(p.grad)/8==pytest.approx(4./16) for p in learner.adapter.parameters())
        assert seen[0]['den']==[2.,128.,1.]
        assert seen[0]['world']==1
        assert learner.costs['paired_request_corrections']==1
        assert result['enabled'] is qualified  # No manufactured execution feedback.
    assert result['paired_removal_correction']['enabled']==(needed and operation=='event_removal')


def test_missing_actual_gradient_stops_and_restores_training_mode(monkeypatch):
    learner,item,_,_=learner_for(monkeypatch,'static_to_linear',stats(),omit_shared=True)
    with pytest.raises(RuntimeError,match='AR/DiT/shared gradients'):
        learner.backward_self(item,scale=.5)
    assert not learner.adapter.training
    assert not learner.costs.get('paired_request_corrections')


def test_request_noise_is_deterministic_separate_and_does_not_consume_rng():
    row,batch=pair('event_addition');adapter=Adapter();seen=[];api=native(adapter,seen)
    rng=torch.get_rng_state().clone()
    for step in [7,7,8]:
        request_loss(api,adapter,None,batch,dict(seed=42),'cpu',row=row,step=step,scale=.5,weight=1.)
    assert seen[0]['seed']==seen[1]['seed']!=seen[2]['seed']
    assert torch.equal(rng,torch.get_rng_state())


def test_coverage_counts_fallback_separately_and_rejects_a_missing_receipt(monkeypatch):
    learner,item,_,_=learner_for(monkeypatch,'event_removal',stats())
    row=learner.backward_self(item,scale=.5)
    row.update(requested_operation='event_removal',terminal_selection=None)
    coverage=feedback_coverage([row],require_complete=True)['event_removal']
    assert coverage['requests']==coverage['covered_requests']==coverage['paired_request_corrections']==1
    assert coverage['qualified_teachers']==coverage['joint_execution_supervision']==coverage['uncovered_requests']==0
    row['request_supervision']=None
    with pytest.raises(RuntimeError):feedback_coverage([row],require_complete=True)


def test_fallback_never_accepts_nonfinite_or_relabelled_execution_feedback():
    route=execution_supervision(stats(),WEIGHTS)
    correction=dict(enabled=True,execution_teacher=False,native_joint_loss=1.,AR_CE=1.,RF_MSE=1.,structured_loss=0.)
    assert covered_request(route,correction)['covered']
    for key,value in [('execution_teacher',True),('RF_MSE',float('nan'))]:
        invalid=dict(correction);invalid[key]=value
        with pytest.raises(ValueError):covered_request(route,invalid)


def test_configuration_prevents_double_removal_correction_and_unrecorded_recipe_changes():
    import json
    from scripts.t2a.rl.train_editing_opsd_repaired_fresh import validate_configuration
    from stable_audio_tools.paths import opsd_config_path
    path=opsd_config_path()
    if not path.exists():
        pytest.skip('Pinned repaired-fresh config is not installed.')
    with open(path) as f:q=json.load(f)
    q.pop('removal_repair');q['request_paired_correction']=copy.deepcopy(RECIPE)
    validate_configuration(q)
    q['removal_repair']=dict(version='removal_paired_v1',paired_native_weight=1.)
    with pytest.raises(ValueError):validate_configuration(q)
    q.pop('removal_repair');q['request_paired_correction']['require_full_coverage']=False
    with pytest.raises(ValueError):validate_configuration(q)
