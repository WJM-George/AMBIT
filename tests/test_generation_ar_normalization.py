"""Text cleanup cannot silently change identities, numeric fields or content."""
import copy
from pathlib import Path
import pytest
from stable_audio_tools.inference.sceneplan_generation_ar_normalization import normalize_generated_sceneplan


class CodecFixture:
    def __init__(self,raw,canonical):self.raw=raw;self.canonical=canonical
    def decode(self,ids,*,sample_id):
        result=copy.deepcopy(self.raw if ids==[1] else self.canonical);result['sample_id']=sample_id;return result
    def canonicalize(self,ids,*,max_tokens=None):return {'input_ids':[2]}


def fixture():
    return {'sources':[{'source_id':'source_0','transcript':'  Hello,\n world. ','activity':{'onset_sec':1.,'offset_sec':2.}}]}


def test_retains_raw_output_and_whitespace_proof():
    raw=fixture();canonical=copy.deepcopy(raw);canonical['sources'][0]['transcript']='Hello, world.'
    result=normalize_generated_sceneplan(CodecFixture(raw,canonical),[1],sample_id='fixture')
    assert result['raw_plan']['sources'][0]['transcript']=='  Hello,\n world. '
    assert result['p10_plan']['sources'][0]['transcript']=='Hello, world.'
    assert result['raw_plan_sha256']!=result['p10_plan_sha256']
    assert result['whitespace_changes'][0]['path']==['sources',0,'transcript']


@pytest.mark.parametrize('change',['time','meaning','identity','count'])
def test_rejects_changes_beyond_text_whitespace(change):
    raw=fixture();canonical=copy.deepcopy(raw)
    if change=='time':canonical['sources'][0]['activity']['offset_sec']=2.1
    elif change=='meaning':canonical['sources'][0]['transcript']='Goodbye, world.'
    elif change=='identity':canonical['sources'][0]['source_id']=' source_0 '
    else:canonical['sources']=[]
    with pytest.raises(ValueError):normalize_generated_sceneplan(CodecFixture(raw,canonical),[1],sample_id='fixture')


def test_actual_codec_removes_only_a_generated_leading_space():
    artifact=Path('/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4')
    if not artifact.is_dir():pytest.skip('requires existing project codec artifact')
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    codec=ModelScenePlanCodecV4(artifact)
    plan={'sample_id':'whitespace-regression','duration_sec':2.,'room':{'type':'dry'},
          'sources':[{'source_id':'source_0','kind':'sound','description':'A bell rings.',
                      'activity':{'onset_sec':0.,'offset_sec':2.},'gain_db':0.,
                      'trajectory':{'type':'static','position':{'azimuth_deg':0.,'elevation_deg':0.,'distance_m':1.}}}]}
    canonical=codec.encode(plan)['input_ids'].tolist()
    raw=canonical.copy()
    raw.insert(raw.index(codec._tid('<text_begin>'))+1,codec.text_offset+1467)
    result=normalize_generated_sceneplan(codec,raw,sample_id=plan['sample_id'])
    assert result['raw_plan']['sources'][0]['description']==' A bell rings.'
    assert result['p10_plan']['sources'][0]['description']=='A bell rings.'
    assert result['p10_token_ids']==canonical
    assert result['whitespace_changes']==[{'path':['sources',0,'description'],'raw':' A bell rings.','canonical':'A bell rings.'}]
