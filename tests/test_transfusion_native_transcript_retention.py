import os
import copy
from pathlib import Path

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_prefix_supervision import build_native_prefix_targets, _sites
from stable_audio_tools.training.transfusion_opsd.native_transcript_retention import unverified_transcript_logit_positions


@pytest.fixture(scope='module')
def codec():
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    path=Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    if not path.exists():pytest.skip('Native codec is not installed.')
    return ModelScenePlanCodecV4(path)


def plan(words):
    return dict(sample_id='retention_scope_test',duration_sec=3.,room=dict(type='dry'),sources=[
        dict(source_id='source_0',kind='speech',speaker_description='a calm voice',transcript=words,
             activity=dict(onset_sec=0.,offset_sec=3.),gain_db=0.,
             trajectory=dict(type='static',position=dict(azimuth_deg=0.,elevation_deg=0.,distance_m=1.)))])


def targets(codec, old, truth):
    tokens=codec.encode(old)['input_ids'].tolist()
    paired=build_native_prefix_targets(codec,tokens,truth,codec.allowed_next_ids)
    return tokens,paired['frontiers']


def test_suffix_exemption_stops_at_transcript_boundary(codec):
    tokens,frontiers=targets(codec,plan('and the biggest,'),plan('and the mossy tree-trunk,'))
    removed=unverified_transcript_logit_positions(codec,tokens,frontiers)
    first=frontiers[0]['position'];end=tokens.index(codec._tid('<text_end>'),first)
    assert removed==list(range(first-1,end))
    assert first-2 not in removed and end not in removed
    assert tokens[removed[-1]+1]==codec._tid('<text_end>')


def test_matching_transcript_keeps_all_reference_positions(codec):
    truth=plan('and love,')
    tokens,frontiers=targets(codec,truth,truth)
    assert unverified_transcript_logit_positions(codec,tokens,frontiers)==[]


@pytest.mark.parametrize('old_words,new_words',[
    ('and the','and the mossy tree'),('and the mossy tree','and the')])
def test_early_or_late_end_is_scoped_to_its_text_field(codec,old_words,new_words):
    tokens,frontiers=targets(codec,plan(old_words),plan(new_words))
    assert len(frontiers)==1
    removed=unverified_transcript_logit_positions(codec,tokens,frontiers)
    _,fields=_sites(codec,tokens)
    span=fields[('source_0','<transcript>')]
    assert removed and set(removed)<={p-1 for p in span}
    assert span[-1]-1 in removed


def test_only_the_matching_source_is_exempted(codec):
    old=plan('keep this speech')
    second=copy.deepcopy(old['sources'][0]);second.update(source_id='source_1',transcript='wrong second words')
    first=old['sources'][0]
    first.pop('transcript');first.pop('speaker_description')
    first.update(kind='music',description='gentle piano music')
    old['sources'].append(second)
    truth=copy.deepcopy(old);truth['sources'][1]['transcript']='requested second words'
    tokens,frontiers=targets(codec,old,truth)
    assert len(frontiers)==1 and frontiers[0]['field']=='source_1/<transcript>'
    removed=unverified_transcript_logit_positions(codec,tokens,frontiers)
    _,fields=_sites(codec,tokens)
    assert not set(removed)&{p-1 for p in fields[('source_0','<description>')]}


def test_stale_or_wrongly_bound_frontier_is_rejected(codec):
    tokens,frontiers=targets(codec,plan('wrong words'),plan('requested words'))
    invalid=copy.deepcopy(frontiers);invalid[0]['observed_token_id']=-1
    with pytest.raises(ValueError,match='native sequence'):
        unverified_transcript_logit_positions(codec,tokens,invalid)
    invalid=copy.deepcopy(frontiers);invalid[0]['field']='source_1/<transcript>'
    with pytest.raises(ValueError,match='absent'):
        unverified_transcript_logit_positions(codec,tokens,invalid)


def test_exemption_removes_only_those_gradients_without_rescaling_others(codec):
    tokens,frontiers=targets(codec,plan('and the biggest,'),plan('and the mossy tree-trunk,'))
    logits=torch.linspace(-1,1,(len(tokens)-1)*3).reshape(-1,3).requires_grad_()
    reference=torch.tensor([.7,.2,.1]).expand_as(logits)
    valid=torch.ones(len(tokens)-1,dtype=torch.bool)
    for f in frontiers:valid[f['position']-1]=False
    denominator=valid.sum()
    base=torch.nn.functional.kl_div(logits[valid].log_softmax(-1),reference[valid],reduction='sum')/denominator
    before,=torch.autograd.grad(base,logits,retain_graph=True)
    new_valid=valid.clone();removed=unverified_transcript_logit_positions(codec,tokens,frontiers)
    new_valid[removed]=False
    revised=torch.nn.functional.kl_div(logits[new_valid].log_softmax(-1),reference[new_valid],reduction='sum')/denominator
    after,=torch.autograd.grad(revised,logits)
    assert before[removed].abs().sum()>0 and after[removed].abs().sum()==0
    torch.testing.assert_close(before[new_valid],after[new_valid],atol=0,rtol=0)
