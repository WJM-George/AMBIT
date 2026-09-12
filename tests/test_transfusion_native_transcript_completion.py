import copy
from pathlib import Path

import pytest

from stable_audio_tools.training.transfusion_opsd.native_transcript_completion import build_native_transcript_completions


@pytest.fixture(scope='module')
def codec():
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    p=Path('/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    if not p.exists():pytest.skip('Native codec is not installed.')
    return ModelScenePlanCodecV4(p)


def plan(words,source='source_0'):
    return dict(sample_id='completion_test',duration_sec=3.,room=dict(type='dry'),sources=[dict(
        source_id=source,kind='speech',speaker_description='a calm voice',transcript=words,
        activity=dict(onset_sec=.2,offset_sec=2.8),gain_db=0.,
        trajectory=dict(type='static',position=dict(azimuth_deg=30.,elevation_deg=0.,distance_m=1.)))])


@pytest.mark.parametrize('old_words,new_words',[
    ('and the biggest,','and the mossy tree-trunk,'),
    ('and love,','and love, and love again.'),
    ('and love, and love again.','and love,'),
])
def test_complete_suffix_and_native_structure_are_separated(codec,old_words,new_words):
    old_plan=plan(old_words);truth=plan(new_words)
    truth['sources'][0]['trajectory']['position']['azimuth_deg']=-80.
    old=codec.encode(old_plan)['input_ids'].tolist();before=old.copy()
    items=build_native_transcript_completions(codec,old,truth,codec.allowed_next_ids)
    assert old==before and len(items)==1
    item=items[0];new=item['tokens'];first=item['first_divergence']
    assert old[:first]==new[:first] and old[first]!=new[first]
    assert old[item['original_suffix_start']:]==new[item['corrected_suffix_start']:]
    decoded=codec.decode(new);baseline=codec.decode(old)
    baseline['sources'][0]['transcript']=new_words
    assert decoded==baseline
    assert item['targets'][0]['prefix_origin']=='student_visited'
    assert all(t['prefix_origin']=='paired_teacher_forced' for t in item['targets'][1:])
    assert item['targets'][-1]['token_id']==codec._tid('<text_end>')
    assert all(t['field']=='source_0/<transcript>' for t in item['targets'])


def test_correct_transcript_needs_no_completion(codec):
    p=plan('Keep these words.')
    assert not build_native_transcript_completions(codec,codec.encode(p)['input_ids'].tolist(),p,codec.allowed_next_ids)


def test_other_source_transcript_is_never_attached(codec):
    old=codec.encode(plan('wrong words'))['input_ids'].tolist()
    assert not build_native_transcript_completions(codec,old,plan('right words','source_1'),codec.allowed_next_ids)


def test_malformed_completion_support_is_rejected(codec):
    old=codec.encode(plan('and the biggest,'))['input_ids'].tolist()
    desired=plan('and the mossy tree-trunk,')
    def allowed(prefix):
        return [x for x in codec.allowed_next_ids(prefix) if x!=codec._tid('<text_end>')]
    with pytest.raises(ValueError,match='incompatible'):
        build_native_transcript_completions(codec,old,desired,allowed)
