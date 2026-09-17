import copy
import pytest

from stable_audio_tools.paths import data_path
from stable_audio_tools.training.transfusion_opsd.request_grounded_text_retention import (
    request_quoted_text_targets, quoted_field_behavior,
)


@pytest.fixture
def codec():
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    path = data_path('sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    if not path.exists():
        pytest.skip('Native Editing codec artifact is not installed.')
    return ModelScenePlanCodecV4(path)


def plan():
    return dict(sample_id='quoted_test', duration_sec=3., room=dict(type='dry'), sources=[
        dict(source_id='source_0', kind='speech', speaker_description='a calm male voice',
             transcript='incorrect original words', activity=dict(onset_sec=0., offset_sec=3.), gain_db=0.,
             trajectory=dict(type='static', position=dict(azimuth_deg=0., elevation_deg=0., distance_m=1.)))])


def test_wrong_transcript_is_not_reinforced_by_matching_speaker(codec):
    p = plan(); ids = codec.encode(p)['input_ids'].tolist()
    request = 'Add the speech saying "please come here" in the voice described as "a calm male voice".'
    result = request_quoted_text_targets(codec, ids, p, request, codec.allowed_next_ids)
    assert len(result['fields']) == 1 and result['fields'][0]['field'] == 'speaker_description'
    assert result['targets'] and all(x['field'].endswith('<speaker_description>') for x in result['targets'])
    assert all(x['token_id'] == ids[x['position']] for x in result['targets'])
    a, b = result['fields'][0]['request_span']
    assert request[a:b] == p['sources'][0]['speaker_description']


def test_ambiguous_source_binding_is_skipped(codec):
    p = plan(); source = p['sources'][0]
    source.pop('speaker_description'); source.pop('transcript')
    source.update(kind='sound', description='a rhythmic hand drum')
    other = copy.deepcopy(source); other['source_id'] = 'source_1';p['sources'].append(other)
    ids = codec.encode(p)['input_ids'].tolist()
    r = request_quoted_text_targets(codec, ids, p, 'Add the sound described as "a rhythmic hand drum".', codec.allowed_next_ids)
    assert not r['targets'] and not r['fields']


def test_text_change_is_not_automatically_semantic_failure():
    f = [dict(source_id='source_0', field='speaker_description', text='a calm male voice')]
    p = plan(); p['sources'][0]['speaker_description'] = 'a relaxed male voice'
    r = quoted_field_behavior(f, p)[0]
    assert not r['text_unchanged'] and not r['explicit_voice_label_conflict']
    p['sources'][0]['speaker_description'] = 'a calm female voice'
    assert quoted_field_behavior(f, p)[0]['explicit_voice_label_conflict']
