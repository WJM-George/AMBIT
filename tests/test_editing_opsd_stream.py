import numpy as np
import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.editing_stream import (
    OrdinalStream, request_facts, request_spatial_measure, training_partitions, homogeneous_microbatches, resize_stream_position)
from stable_audio_tools.training.transfusion_opsd.native_prediction_credit import hard_native_prediction


def test_sampler_resume_and_disjoint_rank_epoch():
    streams = [OrdinalStream(range(103), seed=42, rank=r, world=4) for r in range(4)]
    first = [s.take(25) for s in streams]
    assert len(set(sum(first, []))) == 100
    saved = streams[0].state_dict()
    continued = OrdinalStream(range(103), seed=42, rank=0, world=4, state=saved)
    assert continued.take(51) == streams[0].take(51)


def test_request_rows_never_enter_paired_partition():
    request, paired = training_partitions(1000, 4, request_only_extra=[9, 17])
    assert {9, 17} <= set(request)
    assert not set(request) & set(paired)
    assert sorted(np.concatenate((request, paired))) == list(range(1000))


def test_affirmative_request_and_removal_boundary():
    text = 'Between 1.5 and 4.7 seconds, include the music described as "a sustained tone", stationary at azimuth -34.8 degrees.'
    f = request_facts(text, 'event_addition')
    assert f['activity'] == [1.5, 4.7] and f['azimuths'] == [-34.8]
    assert request_facts(text, 'event_removal') is None
    text = 'Add the speech saying "come here" in the voice described as "a calm voice". Set its active interval to 0.2 through 2.1 seconds.'
    assert request_facts(text, 'event_addition')['activity'] == [.2, 2.1]


def test_unobservable_spatial_windows_are_failures_not_dropped():
    p = dict(sources=[dict(source_id='source_0', kind='music', activity=dict(onset_sec=0., offset_sec=1.), trajectory=dict(type='static'))])
    facts = dict(kind='music', azimuths=[-30.], activity=[0., 1.])
    result = request_spatial_measure(torch.zeros(1, 4, 44100), p, facts)
    assert result['available'] and result['unobservable'] == result['failures'] == 3
    assert result['mean_capped_angle_deg'] == 90.


def test_overlapping_sources_do_not_produce_direction_reward():
    p = dict(sources=[dict(source_id=f'source_{i}', kind=k, activity=dict(onset_sec=0., offset_sec=1.), trajectory=dict(type='static')) for i,k in enumerate(['music','speech'])])
    facts = dict(kind='music', azimuths=[-30.], activity=[0., 1.])
    assert not request_spatial_measure(torch.ones(1, 4, 44100), p, facts)['available']


def test_straight_through_credit_keeps_forward_and_exact_surrogate_direction():
    hard = torch.tensor([1., -2., .5], requires_grad=True)
    alternative = torch.tensor([-1., 3., 2.], requires_grad=True)
    logits = torch.tensor([.3, -.4], requires_grad=True)
    output = hard_native_prediction(hard, [hard, alternative], logits, hard_index=0)
    assert torch.equal(output.detach(), hard.detach())
    loss = (output - alternative.detach()).square().mean()
    gh, ga, gl = torch.autograd.grad(loss, (hard, alternative, logits), allow_unused=True)
    p = logits.softmax(0)
    expected = -2*p[0]*p[1]*(hard.detach()-alternative.detach()).square().mean()
    assert torch.allclose(gl[1], expected) and gl[1] < 0
    assert ga is None
    assert torch.allclose(gh, 2*(hard.detach()-alternative.detach())/hard.numel())


def test_detach_removes_only_decision_gradient():
    hard = torch.tensor([1., 2.], requires_grad=True)
    other = torch.tensor([2., 4.], requires_grad=True)
    logits = torch.tensor([.1, .4], requires_grad=True)
    output = hard_native_prediction(hard, [hard, other], logits, hard_index=0, connect_decisions=False)
    grads = torch.autograd.grad((output-other.detach()).square().mean(), (hard,other,logits), allow_unused=True)
    assert torch.equal(output, hard) and grads[0] is not None
    assert grads[1] is None and grads[2] is None


def test_decision_proxy_reaches_shared_and_ar_head_but_not_alternative_executor():
    torch.manual_seed(17)
    shared = torch.nn.Linear(3, 4)
    ar_head = torch.nn.Linear(4, 2)
    logits = ar_head(shared(torch.ones(3)))
    hard = torch.tensor([0., 1.], requires_grad=True)
    other = torch.tensor([1., -1.], requires_grad=True)
    output = hard_native_prediction(hard, [hard, other], logits, hard_index=0)
    gs, ga, gd = torch.autograd.grad((output-other.detach()).square().mean(), (shared.weight,ar_head.weight,other), allow_unused=True)
    assert gs.abs().sum() > 0 and ga.abs().sum() > 0 and gd is None


def test_checkpoint_digest_handles_scalar_native_parameters():
    from scripts.t2a.rl.train_editing_opsd_stream import tensor_digest
    state = {'scalar':torch.tensor(1.), 'vector':torch.arange(3.).to(torch.bfloat16)}
    digest = tensor_digest(state)
    assert digest == tensor_digest({k:v.clone() for k,v in state.items()})
    state['scalar'].add_(1)
    assert digest != tensor_digest(state)


def test_native_buckets_keep_every_drawn_paired_row():
    rows = [5, 1, 6, 3, 9, 4, 2]
    bucket = lambda i:432 if i % 2 else 648
    batches = homogeneous_microbatches(rows, bucket, 2)
    assert sorted(sum(batches, [])) == sorted(rows)
    assert all(len({bucket(i) for i in b}) == 1 and 1 <= len(b) <= 2 for b in batches)


def test_two_to_four_ranks_preserve_next_global_sample_block():
    old = [OrdinalStream(range(100), seed=79, rank=r, world=2) for r in range(2)]
    for s in old:s.take(8)
    new = [OrdinalStream(range(100), seed=79, rank=r, world=4,
        state=resize_stream_position([s.state_dict() for s in old], new_world=4, rank=r)) for r in range(4)]
    assert sorted(sum([s.take(4) for s in old], [])) == sorted(sum([s.take(2) for s in new], []))
def test_native_microbatch_caps_preserve_all_sampled_rows():
    from stable_audio_tools.training.transfusion_opsd.editing_stream import homogeneous_microbatches
    rows = list(range(151))
    batches = homogeneous_microbatches(rows, lambda i:432 if i < 66 else 648, 64,
                                      bucket_limits={432:64,648:40})
    assert [len(batch) for batch in batches] == [64,2,40,40,5]
    assert [i for batch in batches for i in batch] == rows


def test_native_observation_accepts_real_aligned_duration_above_15_seconds():
    from stable_audio_tools.training.transfusion_opsd.adapters import EditingObservation
    z = torch.zeros(1,64,648)
    mask = torch.ones(1,648,dtype=torch.bool)
    # Real request991445 caused both distributed arms to stop at update53.
    observation = EditingObservation('991445', 'move the speech', z, mask, 663159)
    assert observation.model_num_samples == 663159
    with pytest.raises(ValueError, match='duration'):
        EditingObservation('too_long', 'move it', z, mask, 663553)
    wrong = mask.clone(); wrong[0,-1] = False
    with pytest.raises(ValueError, match='duration'):
        EditingObservation('bad_mask', 'move it', z, wrong, 663159)
    with pytest.raises(ValueError, match='duration'):
        EditingObservation('short_bucket', 'move it', z[...,:432], mask[:,:432], 663159)
