import copy
from pathlib import Path

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.original_pair_condition import original_generation_pair_condition


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    p=Path('/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B')
    if not p.exists():pytest.skip('Native tokenizer is not installed.')
    return AutoTokenizer.from_pretrained(str(p),local_files_only=True)


def plan():
    return dict(sample_id='original_pair_test',duration_sec=88744/44100,room=dict(type='dry'),sources=[
        dict(source_id='source_0',kind='speech',speaker_description='a calm voice',transcript='Keep these exact words.',
            activity=dict(onset_sec=.12345,offset_sec=1.91234),gain_db=0.,
            trajectory=dict(type='static',position=dict(azimuth_deg=12.345,elevation_deg=-3.456,distance_m=1.23456)))])


def make(p,tokenizer,**kwargs):
    import math
    samples=round(p['duration_sec']*44100)
    return original_generation_pair_condition(p,tokenizer,model_num_samples=kwargs.get('samples',samples),
        latent_frames_valid=kwargs.get('frames',math.ceil(samples/1024)),device='cpu')


def test_raw_values_and_compiler_controls_preserved(tokenizer):
    from stable_audio_tools.data.sceneplan_p11_single_turn import compile_p10_aligned_target_conditions
    p=plan();before=copy.deepcopy(p);c=make(p,tokenizer)
    expected=compile_p10_aligned_target_conditions(p,model_num_samples=c.model_num_samples,latent_frames_valid=c.mask.shape[-1])
    assert p==before==c.plan==c.positive[0]['model_sceneplan']
    assert torch.equal(c.positive[0]['sceneplan_44']['source_trajectory_features'],
        torch.as_tensor(expected['sceneplan_44']['source_trajectory_features'],dtype=torch.float32))
    assert c.mask.dtype==torch.bool and c.mask.all() and c.mask.shape[0]==1
    assert c.positive[0]['prompt_text']==expected['semantic_caption']['text']


@pytest.mark.parametrize('bad', [{'samples':44100},{'frames':1},{'samples':True}])
def test_mismatched_pair_geometry_rejected(tokenizer,bad):
    with pytest.raises(ValueError,match='must agree'):make(plan(),tokenizer,**bad)


def test_roles_and_CFG_unknown_do_not_corrupt_positive(tokenizer):
    c=make(plan(),tokenizer);p,n=c.positive[0],c.negative[0]
    roles=p['prompt'];speech=roles['speech_source_ids']>0
    assert speech.any() and roles['speech_lexical_mask'].any()
    assert not torch.any((roles['event_source_ids']>0)&speech)
    assert not torch.any(roles['speech_lexical_mask']&~speech)
    assert 'cfg_unknown' not in p['prompt'] and 'cfg_unknown' not in p['sceneplan_44']
    assert n['prompt']['cfg_unknown'] and n['sceneplan_44']['cfg_unknown']
