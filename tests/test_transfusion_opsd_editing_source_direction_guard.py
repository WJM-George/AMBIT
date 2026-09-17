import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.coarse_native_plan_retention import NativePlanTolerance
from stable_audio_tools.training.transfusion_opsd.editing_source_direction_guard import (
    source_direction_evidence,source_supported_plan_retention,guarded_source_supported_editing_update)


TOLERANCE=NativePlanTolerance(seconds=.5,azimuth_deg=10.,gain_db=1.)


def foa(angle):
    angle=torch.as_tensor(angle,dtype=torch.float32)*torch.pi/180
    w=torch.sin(torch.arange(2000)*.17)*.2
    return torch.stack((w,w*angle.sin(),torch.zeros_like(w),w*angle.cos()))[None]


def evidence(angle=145., count=1):
    return source_direction_evidence(foa(angle),sample_rate=2000,source_kind='music',source_count=count,
        window_seconds=.25,quantiles=(.2,.5,.8),minimum_energy=1e-7,minimum_coherence=.1,maximum_span_deg=10.)


def plan(angle,room='moderate'):
    return dict(duration_sec=1.,room=dict(type=room),sources=[dict(source_id='s0',kind='music',
        description='Guitar',activity=dict(onset_sec=0.,offset_sec=1.),gain_db=0.,
        trajectory=dict(type='static',position=dict(azimuth_deg=angle,elevation_deg=0.,distance_m=1.)))])


def check(before,after,e):
    return source_supported_plan_retention(before,after,tolerance=TOLERANCE,preserve_room=True,
                                           source_evidence=e,source_tolerance_deg=32.5)


def test_source_evidence_allows_correction_without_old_plan_agreement():
    e=evidence();r=check(plan(-163.),plan(165.),e)
    assert e['available'] and r['passed'] and not r['old_plan_change_budget_passed']
    assert len(r['source_supported_exceptions'])==1
    assert len(r['source_supported_exceptions'][0]['comparisons'])==3


def test_closer_but_still_outside_source_cone_is_not_enough():
    assert not check(plan(-163.),plan(-175.),evidence())['passed']


def test_input_exception_cannot_waive_room_or_added_source_changes():
    a,b=plan(-163.),plan(165.,'dry')
    r=check(a,b,evidence());assert not r['passed']
    assert [x['field'] for x in r['failures']]==['room.type']
    a,b=plan(-163.),plan(165.)
    speech=copy.deepcopy(plan(0.)['sources'][0]);speech.update(kind='speech',source_id='s1')
    a['sources'].append(speech);b['sources'].append(copy.deepcopy(speech))
    b['sources'][1]['trajectory']['position']['azimuth_deg']=50.
    r=check(a,b,evidence());assert not r['passed']
    assert [x['field'] for x in r['failures']]==['sources.speech.trajectory.position.azimuth_deg']


def test_already_accurate_input_direction_cannot_be_degraded_using_exception():
    assert not check(plan(145.),plan(165.),evidence())['passed']


def test_silence_multiple_sources_and_changing_direction_authorize_nothing():
    kwargs=dict(sample_rate=2000,source_kind='music',source_count=1,window_seconds=.25,
                quantiles=(.2,.5,.8),minimum_energy=1e-7,minimum_coherence=.1,maximum_span_deg=10.)
    silent=source_direction_evidence(torch.zeros(1,4,2000),**kwargs)
    moving=source_direction_evidence(foa(torch.linspace(0,150,2000)),**kwargs)
    for e in [silent,moving,evidence(count=2)]:
        assert not e['available'] and not check(plan(-163.),plan(165.),e)['passed']


def test_wraparound_uses_small_circular_error():
    assert check(plan(140.),plan(179.),evidence(-175.))['passed']


@pytest.mark.parametrize('early',[False,True])
def test_candidate_acceptance_requires_every_anchor_even_with_early_rejection(early):
    class Adapter:
        training=False
        def native_plan(self,obs):return obs,None
    names=['AR_adapters','shared_Transformer','Editing_DiT_adapters_and_conditioning','structured_heads']
    optimizer=SimpleNamespace(param_groups=[dict(group_name=n,params=[object()]) for n in names])
    anchors=[dict(identifier=i,plan=plan(0.),observation=plan(0.)) for i in range(2)]
    kwargs=dict(anchors=anchors,tolerance=TOLERANCE,trial_scales=[],preserve_room=True,
                source_tolerance_deg=32.5,stop_on_first_failure=early)
    with patch('stable_audio_tools.training.transfusion_opsd.editing_source_direction_guard.guarded_joint_adam_update',
               side_effect=lambda *a,**k:k['validate_native']()):
        result=guarded_source_supported_editing_update(Adapter(),optimizer,**kwargs)
        assert result['passed'] and result['checked_anchor_count']==2
        anchors[0]['observation']=plan(80.)
        result=guarded_source_supported_editing_update(Adapter(),optimizer,**kwargs)
        assert not result['passed'] and result['checked_anchor_count']==(1 if early else 2)
