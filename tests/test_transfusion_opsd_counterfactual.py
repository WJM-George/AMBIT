import torch

from stable_audio_tools.training.transfusion_opsd.event_counterfactual import paired_gaussian_noise, FixedRequestAudioProtocol, FixedDirectionalAudioProtocol
from test_transfusion_opsd_event import codec, example


def test_noise_matches_native_at_reference_length_and_keeps_shared_time_positions():
    state = torch.get_rng_state().clone()
    native = torch.randn((1, 64, 131), generator=torch.Generator(device='cpu').manual_seed(42))
    same = paired_gaussian_noise(42, (1, 64, 131), reference_frames=131)
    shorter = paired_gaussian_noise(42, (1, 64, 129), reference_frames=131)
    longer = paired_gaussian_noise(42, (1, 64, 134), reference_frames=131)
    assert torch.equal(native, same)
    assert torch.equal(shorter, native[..., :129])
    assert torch.equal(longer[..., :131], native)
    assert torch.equal(longer, paired_gaussian_noise(42, (1, 64, 134), reference_frames=131))
    assert torch.equal(state, torch.get_rng_state())
    assert torch.isfinite(longer).all() and longer.shape == (1, 64, 134)
    assert not torch.equal(longer[:, 0, 131:], longer[:, 1, 131:])


def test_duration_protocol_reuses_reference_scoring_and_cannot_follow_candidate_activity(codec):
    plan, _, request = example(codec)
    samples = round(plan['duration_sec'] * 44100)
    protocol = FixedRequestAudioProtocol(request, plan, model_num_samples=samples)
    assert protocol.reward_for_samples(samples) is protocol.initial
    other = protocol.reward_for_samples(samples + 1024)
    assert other.num_samples == samples + 1024
    assert torch.equal(other.directions, protocol.initial.directions)
    # Even on a changed shape, silence is not an improved spatial completion.
    score = other(torch.zeros(1, 4, samples + 1024))
    assert score.utility == 0. and score.costs['source_presence_failure'] == 1.


def test_fixed_direction_observations_allow_free_duration_changes(codec):
    from test_transfusion_opsd_directional_reward import field
    plan, _, request = example(codec)
    samples = round(plan['duration_sec'] * 44100)
    protocol = FixedDirectionalAudioProtocol(request, plan, reference_audio=field(samples), model_num_samples=samples)
    assert protocol.reward_for_samples(samples) is protocol.initial
    mapped = protocol.reward_for_samples(samples + 2048)
    torch.testing.assert_close(mapped.reference_weights.sum(-1), torch.ones(len(mapped.windows)))
    assert not mapped.reference_weights[~mapped.windows].count_nonzero()
    assert mapped(field(samples + 2048)).utility > .99
    assert mapped(torch.zeros(1, 4, samples + 2048)).costs['direction_unobservable_fraction'] == 1.
