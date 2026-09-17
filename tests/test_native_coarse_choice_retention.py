import copy
import pytest
import torch

from stable_audio_tools.paths import data_path
from stable_audio_tools.training.transfusion_opsd.native_coarse_choice_retention import (
    native_azimuth_cone_targets, field_balanced_native_set_ce,
)


@pytest.fixture
def codec():
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    p = data_path('sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    if not p.exists():
        pytest.skip('Native codec artifact is not installed.')
    return ModelScenePlanCodecV4(p)


def plan(azimuth=-179.):
    return dict(sample_id='coarse', duration_sec=3., room=dict(type='dry'), sources=[
        dict(source_id='source_0', kind='music', description='a quiet accordion',
             activity=dict(onset_sec=0., offset_sec=3.), gain_db=0.,
             trajectory=dict(type='static', position=dict(azimuth_deg=azimuth, elevation_deg=0., distance_m=1.)))])


def build(codec, p, **kwargs):
    ids = codec.encode(p)['input_ids'].tolist()
    return native_azimuth_cone_targets(codec, ids, p, codec.allowed_next_ids, radius_deg=10., **kwargs)


def evidence():
    return dict(available=True, source_kind='music', source_count=1,
                windows=[dict(observable=True, azimuth_deg=a) for a in [144., 147., 145.]])


def test_cone_wraps_and_keeps_multiple_native_choices(codec):
    t = build(codec, plan())[0]
    a = lambda degree: codec.azimuth_ids[degree + 180]
    assert len(t['acceptable_ids']) == 21
    assert a(179) in t['acceptable_ids'] and a(-170) in t['acceptable_ids']
    assert a(160) not in t['acceptable_ids']


def test_source_improvement_allowed_without_requiring_wrong_old_direction(codec):
    t = build(codec, plan(165.), source_evidence=evidence())[0]
    assert codec.azimuth_ids[145 + 180] in t['source_supported_extra_ids']
    assert codec.azimuth_ids[-163 + 180] not in t['acceptable_ids']
    assert t['observed_token_id'] in t['acceptable_ids']


def test_ambiguous_kind_does_not_authorize_source_correction(codec):
    p = plan(165.)
    q = copy.deepcopy(p['sources'][0]); q['source_id'] = 'source_1'; p['sources'].append(q)
    targets = build(codec, p, source_evidence=evidence())
    assert len(targets) == 2
    assert all(not t['source_evidence_applied'] and len(t['acceptable_ids']) == 21 for t in targets)


def test_linear_waypoints_are_separate_and_no_static_source_exception(codec):
    p = plan()
    pos = p['sources'][0]['trajectory']['position']
    end = dict(pos, azimuth_deg=20.)
    p['sources'][0]['trajectory'] = dict(type='linear', start=pos, end=end)
    ts = build(codec, p, source_evidence=evidence())
    assert [t['waypoint_index'] for t in ts] == [0, 1]
    assert not any(t['source_evidence_applied'] for t in ts)
    assert ts[0]['acceptable_ids'] != ts[1]['acceptable_ids']


def test_loss_moves_only_legal_probability_into_compatible_set():
    logits = torch.zeros(2, 5, requires_grad=True)
    target = dict(position=1, allowed_ids=[0, 1, 2, 3], acceptable_ids=[0, 1], field='source_0/azimuth')
    loss = field_balanced_native_set_ce(logits, [target])
    assert loss.item() == pytest.approx(0.69314718)
    loss.backward()
    assert torch.all(logits.grad[0, :2] < 0) and torch.all(logits.grad[0, 2:4] > 0)
    assert logits.grad[0, 4] == 0 and torch.all(logits.grad[1] == 0)
    assert logits.grad[0, 0] == logits.grad[0, 1]


def test_invalid_compatible_set_cannot_train_forbidden_choice():
    logits = torch.zeros(1, 4, requires_grad=True)
    target = dict(position=1, allowed_ids=[0, 1], acceptable_ids=[2], field='azimuth')
    with pytest.raises(ValueError, match='Invalid legal support'):
        field_balanced_native_set_ce(logits, [target])
