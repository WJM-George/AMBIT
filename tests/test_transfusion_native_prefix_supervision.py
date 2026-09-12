from pathlib import Path

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_prefix_supervision import (
    build_native_prefix_targets, field_balanced_native_ce,
)


@pytest.fixture
def codec():
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    path = Path('/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    if not path.exists():
        pytest.skip('Native Editing codec artifact is not installed.')
    return ModelScenePlanCodecV4(path)


def plan(words, source='source_0'):
    return dict(sample_id='prefix_test', duration_sec=3., room=dict(type='dry'), sources=[
        dict(source_id=source, kind='speech', speaker_description='a calm voice',
             transcript=words, activity=dict(onset_sec=0., offset_sec=3.), gain_db=0.,
             trajectory=dict(type='static', position=dict(azimuth_deg=0., elevation_deg=0., distance_m=1.)))])


def test_first_error_uses_visited_prefix_and_never_labels_after_divergence(codec):
    old=codec.encode(plan('and the biggest,'))['input_ids'].tolist()
    expected=plan('and the mossy tree-trunk,')
    result=build_native_prefix_targets(codec, old, expected, codec.allowed_next_ids)
    assert len(result['frontiers'])==1
    f=result['frontiers'][0]
    assert f['token_id'] != f['observed_token_id'] and f['role']=='paired_transcript_first_error'
    transcript_anchors=[a for a in result['anchors'] if a['field'].endswith('<transcript>')]
    assert transcript_anchors and max(a['position'] for a in transcript_anchors)<f['position']
    desired=codec._text(expected['sources'][0]['transcript'])
    start=old.index(codec._tid('<transcript>'))+1
    assert old[start:f['position']]==desired[:f['position']-start]
    assert desired[f['position']-start]==f['token_id']


def test_mismatched_source_is_not_assigned_other_source_transcript(codec):
    old=codec.encode(plan('wrong words', source='source_0'))['input_ids'].tolist()
    result=build_native_prefix_targets(codec, old, plan('requested words', source='source_1'), codec.allowed_next_ids)
    assert not result['frontiers']
    assert all(a['field'].startswith('scene/') for a in result['anchors'])


def test_matching_transcript_is_retained_without_correction(codec):
    expected=plan('and love,')
    old=codec.encode(expected)['input_ids'].tolist()
    result=build_native_prefix_targets(codec, old, expected, codec.allowed_next_ids)
    assert not result['frontiers']
    assert any(a['field'].endswith('<transcript>') for a in result['anchors'])
    assert all(a['token_id']==a['observed_token_id'] for a in result['anchors'])


def test_correct_choice_has_initial_gradient_only_on_legal_support():
    logits=torch.tensor([[0.,2.,1.,8.]],requires_grad=True)
    target=dict(position=1, token_id=1, allowed_ids=[0,1,2], field='room')
    field_balanced_native_ce(logits,[target]).backward()
    assert logits.grad[0,1]<0 and logits.grad[0,3]==0


def test_repeating_text_tokens_does_not_dilute_a_separate_room_field():
    logits=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
    room=dict(position=1,token_id=0,allowed_ids=[0,1],field='room')
    text=dict(position=2,token_id=1,allowed_ids=[0,1],field='transcript')
    one=field_balanced_native_ce(logits,[room,text])
    many=field_balanced_native_ce(logits,[room]+[text]*20)
    torch.testing.assert_close(one,many)
