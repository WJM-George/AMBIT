from types import SimpleNamespace

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_request_room import (
    native_requested_room_loss, requested_room,
)


def policy_and_proposal():
    names = {'<room>':6, '<room_dry>':7, '<room_moderate>':8,
             '<room_reverberant>':9, '<room_outdoor>':10}
    logits = torch.nn.Parameter(torch.tensor([2., 0., 0., 0.]))
    seen = []
    def ar(tokens, mask, context, context_mask):
        seen.append(tokens[0].tolist())
        output = logits.new_zeros(1, tokens.shape[-1], 11)
        output[0,-1,7:11] = logits
        return output
    codec = SimpleNamespace(_tid=names.__getitem__, allowed_next_ids=lambda prefix: {7,8,9,10})
    bundle = SimpleNamespace(codec=codec, ar=ar,
        encode_event_requests=lambda requests,device: (None,None))
    policy = SimpleNamespace(bundle=bundle, device=torch.device('cpu'))
    prefix=(1,5,474,6)
    proposal=SimpleNamespace(tokens=prefix+(7,), observation=SimpleNamespace(request='Moderate reverberation.'),
        free_decisions=[SimpleNamespace(prefix=prefix,legal_ids=(7,8,9,10))], plan={'room':{'type':'dry'}})
    return policy,proposal,logits,seen


def test_wrong_current_room_learns_request_at_changed_duration_prefix():
    policy,proposal,logits,seen=policy_and_proposal()
    loss,details=native_requested_room_loss(policy,proposal,{'scene':[{'op':'room','value':'moderate'}]})
    loss.backward()
    assert seen==[[1,5,474,6]]
    assert not details['correct'] and details['target_token']==8
    assert logits.grad[1] < 0 < logits.grad[0]
    assert proposal.plan['room']['type']=='dry'


def test_unspecified_room_is_not_copied_from_completed_plan():
    policy,proposal,logits,seen=policy_and_proposal()
    assert native_requested_room_loss(policy,proposal,{'scene':[]}) is None
    assert not seen and logits.grad is None


def test_old_prefix_cannot_be_used_as_current_room_supervision():
    policy,proposal,_,_=policy_and_proposal()
    proposal.tokens=(1,5,479,6,7)
    with pytest.raises(ValueError,match='current sampled prefix'):
        native_requested_room_loss(policy,proposal,{'scene':[{'op':'room','value':'moderate'}]})


def test_conflicting_request_constraints_are_not_silently_chosen():
    with pytest.raises(ValueError,match='consistent'):
        requested_room({'scene':[{'op':'room','value':'dry'},{'op':'room','value':'outdoor'}]})
