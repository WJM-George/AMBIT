from types import SimpleNamespace

import pytest
import torch

from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import TIME_PHASE_RANGES
from stable_audio_tools.training.transfusion_opsd.request_coarse_spatial_reward import RequestCoarseSpatialReward
from stable_audio_tools.training.transfusion_opsd.request_semantic_time_reward import RequestSemanticTimeCoarseReward
from test_transfusion_opsd_activity_compass_reward import tone


def request(field, phase, *, extra=(), scene=()):
    constraints=[dict(op='motion',value='static'),dict(op='compass',point='both',value='front'),
        dict(op='time_phase',field=field,value=phase)]+list(extra)
    return SimpleNamespace(requirements=dict(scene=list(scene),relations=[],sources=[dict(constraints=constraints)]))


@pytest.mark.parametrize('duration',[2.,4.,10.031])
@pytest.mark.parametrize('field,phase,bounds',[
    ('onset_sec','early',(0.,.5)),
    ('onset_sec','early',(.03,.5)),
    ('onset_sec','early',(.2,.5)),
    ('offset_sec','late',(.1,.7)),
    ('offset_sec','late',(.1,.94)),
    ('offset_sec','late',(.1,1.)),
])
def test_unqualified_early_and_late_do_not_require_edge_silence(duration,field,phase,bounds):
    result=RequestSemanticTimeCoarseReward(request(field,phase)).measure(tone(duration,*bounds))
    assert result['costs']['requested_time_failure']==0
    assert result['costs']['source_presence_failure']==0


@pytest.mark.parametrize('field,phase,bounds',[
    ('onset_sec','early',(.7,.95)),
    ('offset_sec','late',(.02,.3)),
    ('onset_sec','beginning',(.5,.8)),
    ('offset_sec','ending',(.02,.4)),
    ('onset_sec','middle',(.02,.8)),
    ('offset_sec','middle',(.1,.98)),
])
def test_clearly_wrong_time_still_fails(field,phase,bounds):
    assert RequestSemanticTimeCoarseReward(request(field,phase)).measure(tone(4.,*bounds))['costs']['requested_time_failure']==1


def test_numeric_limit_still_rejects_an_otherwise_late_event():
    req=request('offset_sec','late',extra=[dict(op='numeric',field='offset_sec',value=2.8)])
    observer=RequestSemanticTimeCoarseReward(req)
    assert observer.measure(tone(4.,.1,.7))['costs']['requested_time_failure']==0
    rows=observer.measure(tone(4.,.1,1.))['time_checks']
    assert rows[0]['pass'] and not rows[1]['pass']


def test_presence_and_exclusive_scene_duration_are_not_relaxed():
    req=request('offset_sec','late',scene=[dict(op='duration_range',min=2.,max=4.,max_inclusive=False)])
    observer=RequestSemanticTimeCoarseReward(req)
    assert observer.measure(tone(4.,.1,.94))['costs']['requested_time_failure']==1
    costs=observer.measure(torch.zeros(1,4,44100*3))['costs']
    assert costs['requested_time_failure']==costs['source_presence_failure']==1


def test_only_time_changes_and_historical_AR_ranges_remain_intact():
    req=request('offset_sec','late');wave=tone(4.,.1,.97,azimuth=12.)
    before=RequestCoarseSpatialReward(req).measure(wave)
    after=RequestSemanticTimeCoarseReward(req).measure(wave)
    assert before['profile']!=after['profile']
    assert before['costs']['requested_time_failure']==1 and after['costs']['requested_time_failure']==0
    for key,value in before['costs'].items():
        if key!='requested_time_failure':assert value==after['costs'][key]
    for key in ('intervals','direction_checks','mean_excess_angle_deg','motion_progress','active_frames',
        'observed_onset_sec','observed_offset_sec'):
        assert before[key]==after[key]
    assert TIME_PHASE_RANGES['early']==(.1,.4) and TIME_PHASE_RANGES['late']==(.6,.9)


def test_uncovered_radial_motion_is_still_rejected():
    req=request('offset_sec','late',extra=[dict(op='distance_change',value='approaching')])
    with pytest.raises(ValueError,match='separate audio verifier'):
        RequestSemanticTimeCoarseReward(req)
