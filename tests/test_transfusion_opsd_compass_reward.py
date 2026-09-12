import copy
import math

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.event_compass_reward import CompassOnlyDirectionalReward
from stable_audio_tools.training.transfusion_opsd.event_directional_reward import FixedReferenceDirectionalReward
from stable_audio_tools.training.transfusion_opsd.event_rewards import EventRequestReward
from stable_audio_tools.training.transfusion_opsd.rewards import UnobservableAudio
from test_transfusion_opsd_event import codec, example


def plane_wave(samples, *, azimuth=45., elevation=0.):
    time = torch.arange(samples, dtype=torch.float32) / 44100
    wave = .03 * torch.sin(2 * math.pi * 440 * time)
    azimuth, elevation = math.radians(azimuth), math.radians(elevation)
    direction = torch.tensor([math.cos(azimuth)*math.cos(elevation),
        math.sin(azimuth)*math.cos(elevation), math.sin(elevation)])
    xyz = direction[:, None] * wave * math.sqrt(2.)
    return torch.stack([wave, xyz[1], xyz[2], xyz[0]])[None]


def setup(codec):
    plan, _, request = example(codec)
    samples = round(plan['duration_sec']*44100)
    baseline = plane_wave(samples)
    return plan, request, samples, baseline


@pytest.mark.parametrize('elevation', [-70., 0., 70.])
def test_unspecified_elevation_does_not_change_valid_compass(codec, elevation):
    plan, request, samples, baseline = setup(codec)
    changed = copy.deepcopy(plan)
    changed['sources'][0]['trajectory']['position']['elevation_deg'] = elevation
    assert request.admissible(changed)
    reward = CompassOnlyDirectionalReward(request, plan, reference_audio=baseline, model_num_samples=samples)
    audio = plane_wave(samples, elevation=elevation)
    assert torch.equal(audio[:, 0], baseline[:, 0])
    assert reward(audio).costs['requested_sector_failure'] == 0
    assert reward.soft_sector_cost(audio).item() == pytest.approx(0., abs=1e-6)


def test_historical_cone_and_new_compass_profiles_are_distinct(codec):
    plan, request, samples, baseline = setup(codec)
    tilted = plane_wave(samples, elevation=70.)
    old = FixedReferenceDirectionalReward(request, plan, reference_audio=baseline, model_num_samples=samples)
    new = CompassOnlyDirectionalReward(request, plan, reference_audio=baseline, model_num_samples=samples)
    assert old(tilted).costs['requested_sector_failure'] == 1
    assert new(tilted).costs['requested_sector_failure'] == 0
    assert new.profile != old.profile


def test_wrong_horizontal_direction_is_penalized_at_every_free_elevation(codec):
    plan, request, samples, baseline = setup(codec)
    reward = CompassOnlyDirectionalReward(request, plan, reference_audio=baseline, model_num_samples=samples)
    losses = []
    for elevation in (-70., 0., 70.):
        audio = plane_wave(samples, azimuth=-45., elevation=elevation).requires_grad_(True)
        assert reward(audio).costs['requested_sector_failure'] == 1
        loss = reward.soft_sector_cost(audio)
        loss.backward()
        assert torch.isfinite(audio.grad).all() and audio.grad.norm() > 0
        losses.append(float(loss))
    assert losses == pytest.approx([losses[1]]*3, abs=1e-6)


def test_silence_and_undefined_vertical_azimuth_cannot_hide_failure(codec):
    plan, request, samples, baseline = setup(codec)
    reward = CompassOnlyDirectionalReward(request, plan, reference_audio=baseline, model_num_samples=samples)
    for audio in (torch.zeros_like(baseline), plane_wave(samples, elevation=90.)):
        result = reward(audio)
        assert result.costs['requested_sector_failure'] == 1
        assert result.costs['direction_unobservable_fraction'] == 1
        with pytest.raises(UnobservableAudio, match='reliable'):
            CompassOnlyDirectionalReward(request, plan, reference_audio=audio, model_num_samples=samples)


@pytest.mark.parametrize('constraint,text', [
    ({'op':'numeric','field':'start.elevation_deg','value':20.,'evidence':'elevation 20 degrees'}, 'elevation 20 degrees'),
    ({'op':'direct','point':'both','value':'left','evidence':'directly left'}, 'directly left'),
    ({'op':'distance_range','point':'both','min':2.,'max':4.,'evidence':'between 2 and 4 meters'}, 'between 2 and 4 meters'),
])
def test_additional_spatial_requests_require_explicit_audio_verifiers(codec, constraint, text):
    plan, request, samples, baseline = setup(codec)
    requirements = copy.deepcopy(request.requirements)
    requirements['sources'][0]['constraints'].append(constraint)
    other = EventRequestReward(request.request+' '+text, requirements)
    with pytest.raises(ValueError, match='additional spatial'):
        CompassOnlyDirectionalReward(other, plan, reference_audio=baseline, model_num_samples=samples)


def test_external_content_costs_and_time_presence_protection_remain(codec):
    plan, request, samples, baseline = setup(codec)
    extra = lambda audio: {'clap_content_cost/source_0': .2, 'asr_wer': .1}
    old = FixedReferenceDirectionalReward(request, plan, reference_audio=baseline, model_num_samples=samples, extra_costs=extra)
    new = CompassOnlyDirectionalReward(request, plan, reference_audio=baseline, model_num_samples=samples, extra_costs=extra)
    assert old(baseline).costs.keys() == new(baseline).costs.keys()
    for key in ['source_presence_failure','forbidden_activity_fraction','clipping','clap_content_cost/source_0','asr_wer']:
        assert old(baseline).costs[key] == new(baseline).costs[key]
