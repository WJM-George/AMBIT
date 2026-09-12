import pytest
import torch

from stable_audio_tools.data.model_sceneplan_codec_v3 import STRUCTURAL_TOKENS
from stable_audio_tools.training.transfusion_opsd.native_activity_choice_retention import activity_choice_references


class Codec:
    names = {s: i for i, s in enumerate(STRUCTURAL_TOKENS)}
    frame_ids = tuple(range(len(STRUCTURAL_TOKENS), len(STRUCTURAL_TOKENS)+649))

    def _tid(self, name):
        return self.names[name]

def fixture():
    codec = Codec()
    values = list(codec.frame_ids)
    ids = [codec._tid('<plan_bos>')]
    for slot in (0, 2):
        ids += [codec._tid('<source_begin>'), codec._tid(f'<source_slot_{slot}>'),
                codec._tid('<activity_begin>'), codec._tid('<onset_frame>'), values[1],
                codec._tid('<offset_frame>'), values[4], codec._tid('<activity_end>'),
                codec._tid('<source_end>')]
    ids += [codec._tid('<plan_eos>')]
    logits = torch.zeros(len(ids)-1, max(codec.frame_ids)+1)
    for pos in range(1, len(ids)):
        logits[pos-1, ids[pos]] = 1
    return codec, ids, logits, values


def test_source_binding_atomic_frames_and_explicit_exemption():
    codec, ids, logits, values = fixture()
    refs = activity_choice_references(codec, ids, logits, lambda prefix: values,
                                      exempt_fields=['source_2/<onset_frame>'])
    assert len(refs) == 3
    assert {r['field'] for r in refs} == {'source_0/<onset_frame>', 'source_0/<offset_frame>', 'source_2/<offset_frame>'}
    assert all(r['reference_gap'] == 1.0 and r['token_id'] == ids[r['position']] for r in refs)


def test_forced_frames_have_no_surrogate_decision_and_non_greedy_is_rejected():
    codec, ids, logits, values = fixture()
    refs = activity_choice_references(codec, ids, logits, lambda prefix: [ids[len(prefix)]])
    assert refs == []
    pos = ids.index(codec._tid('<onset_frame>'))+1
    logits[pos-1, values[10]] = 5
    with pytest.raises(ValueError, match='not greedy'):
        activity_choice_references(codec, ids, logits, lambda prefix: values)


def test_unbound_source_and_non_frame_payload_are_rejected():
    codec, ids, logits, values = fixture()
    ids[1] = codec._tid('<plan_bos>')
    with pytest.raises(ValueError, match='bound native source'):
        activity_choice_references(codec, ids, logits, lambda prefix: values)
    codec, ids, logits, values = fixture()
    ids[ids.index(codec._tid('<onset_frame>'))+1] = codec._tid('<text_begin>')
    with pytest.raises(ValueError, match='native atomic frame vocabulary'):
        activity_choice_references(codec, ids, logits, lambda prefix: values)
