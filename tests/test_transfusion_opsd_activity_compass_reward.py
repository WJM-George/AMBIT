import copy
import math

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.event_activity_compass_reward import RequestActivityCompassReward
from stable_audio_tools.training.transfusion_opsd.event_rewards import EventRequestReward


def request(moving=False):
    text = 'A tone begins near the beginning and ends near the ending, staying static in front.'
    constraints = [dict(op='time_phase', field='onset_sec', value='beginning', evidence='beginning'),
        dict(op='time_phase', field='offset_sec', value='ending', evidence='ending')]
    if moving:
        text = 'A tone begins near the beginning and ends near the ending, moving from left to right.'
        constraints += [dict(op='motion', value='linear', evidence='moving'),
            dict(op='compass', point='start', value='left', evidence='left'),
            dict(op='compass', point='end', value='right', evidence='right')]
    else:
        constraints += [dict(op='motion', value='static', evidence='static'),
            dict(op='compass', point='both', value='front', evidence='front')]
    requirements = dict(schema='generation_ar_natural_requirements_v2', scene=[], relations=[],
        sources=[dict(key='tone', kind='sound', core='A tone', evidence='A tone', constraints=constraints)])
    return EventRequestReward(text, requirements)


def tone(duration=4., onset=.1, offset=.9, *, azimuth=0., elevation=0., moving=False, reverse=False, silence=False, endpoint_hold=True):
    samples = round(duration*44100)
    time = torch.arange(samples, dtype=torch.float32)/44100
    active = (time >= onset*duration) & (time < offset*duration)
    wave = .03 * torch.sin(2*math.pi*440*time) * active
    if silence:
        wave.zero_()
    if moving:
        # Hold endpoints for 25% of activity, then move continuously between them.
        progress = (time/duration-onset)/(offset-onset)
        progress = ((progress-.25)*2).clamp(0, 1) if endpoint_hold else progress.clamp(0, 1)
        angle = 90-180*progress
        if reverse:
            angle = -angle
    else:
        angle = torch.full_like(time, azimuth)
    angle = torch.deg2rad(angle)
    elevation = math.radians(elevation)
    return torch.stack([wave, wave*2**.5*angle.sin()*math.cos(elevation),
        wave*2**.5*math.sin(elevation), wave*2**.5*angle.cos()*math.cos(elevation)])[None]


@pytest.mark.parametrize('duration', [2., 8., 10.031])
@pytest.mark.parametrize('bounds', [(0., 1.), (.15, .85)])
def test_request_legal_time_and_duration_do_not_inherit_a_reference_window(duration, bounds):
    result = RequestActivityCompassReward(request()).measure(tone(duration, *bounds))
    assert all(result['costs'][key] == 0 for key in
        ['requested_sector_failure', 'source_presence_failure', 'requested_time_failure', 'direction_unobservable_fraction'])
    assert result['duration_sec'] == pytest.approx(duration, abs=1/44100)


@pytest.mark.parametrize('elevation', [-70., 70.])
def test_unspecified_elevation_is_free(elevation):
    assert RequestActivityCompassReward(request())(tone(elevation=elevation)).costs['requested_sector_failure'] == 0.


def test_wrong_time_is_independent_of_correct_direction():
    costs = RequestActivityCompassReward(request())(tone(onset=.5, offset=.6)).costs
    assert costs['requested_sector_failure'] == 0.
    assert costs['requested_time_failure'] == 1.


def test_silence_never_passes_direction_presence_or_requested_time():
    costs = RequestActivityCompassReward(request())(tone(silence=True)).costs
    assert all(costs[k] == 1. for k in ['requested_sector_failure', 'direction_unobservable_fraction',
        'source_presence_failure', 'requested_time_failure'])


def test_spatially_unobservable_active_frames_stay_in_denominator():
    observer = RequestActivityCompassReward(request())
    original = tone()
    changed = original.clone()
    changed[:, 1:] = 0
    a, b = observer.measure(original), observer.measure(changed)
    assert a['active_frames'] == b['active_frames']
    assert b['costs']['source_presence_failure'] == 0
    assert b['costs']['requested_sector_failure'] == b['costs']['direction_unobservable_fraction'] == 1


def test_real_motion_reverse_and_stationary_are_distinguished():
    observer = RequestActivityCompassReward(request(moving=True))
    forward, backward, stationary = [observer.measure(w) for w in
        [tone(moving=True), tone(moving=True, reverse=True), tone(azimuth=90.)]]
    assert forward['costs']['requested_sector_failure'] == forward['costs']['motion_trend_failure'] == 0
    assert backward['costs']['requested_sector_failure'] == backward['costs']['motion_trend_failure'] == 1
    assert stationary['costs']['motion_trend_failure'] == 1
    assert stationary['costs']['requested_sector_failure'] > 0


@pytest.mark.parametrize('duration', [2., 8.])
def test_valid_continuous_movement_does_not_require_endpoint_dwell(duration):
    costs = RequestActivityCompassReward(request(moving=True))(
        tone(duration=duration, moving=True, endpoint_hold=False)).costs
    assert costs['requested_sector_failure'] == costs['motion_trend_failure'] == 0


def test_missing_motion_endpoint_is_not_relabelled_as_success():
    audio = tone(moving=True)
    audio[..., round(4.*.35*44100):] = 0
    costs = RequestActivityCompassReward(request(moving=True))(audio).costs
    assert costs['requested_time_failure'] == 1
    assert costs['requested_sector_failure'] > 0
    assert costs['motion_trend_failure'] == 1


def test_pause_does_not_become_a_direction_failure():
    audio = tone()
    audio[..., 44100:3*44100] = 0
    result = RequestActivityCompassReward(request()).measure(audio)
    assert result['active_frame_fraction'] < .5
    assert result['costs']['requested_sector_failure'] == result['costs']['requested_time_failure'] == 0


@pytest.mark.parametrize('extra', [dict(op='distance_change', value='approaching', evidence='approaching'),
    dict(op='numeric', field='start.elevation_deg', value=20., evidence='20 degrees')])
def test_uncovered_spatial_requirements_are_rejected(extra):
    original = request()
    requirements = copy.deepcopy(original.requirements)
    requirements['sources'][0]['constraints'].append(extra)
    other = EventRequestReward(original.request+' '+extra['evidence'], requirements)
    with pytest.raises(ValueError, match='separate audio verifier'):
        RequestActivityCompassReward(other)


def test_scene_duration_range_keeps_its_exclusive_endpoint():
    original = request()
    requirements = copy.deepcopy(original.requirements)
    requirements['scene'] = [dict(op='duration_range', min=2., max=4., max_inclusive=False, evidence='under four seconds')]
    observer = RequestActivityCompassReward(EventRequestReward(original.request+' under four seconds', requirements))
    assert observer(tone(duration=3.5)).costs['requested_time_failure'] == 0
    assert observer(tone(duration=4.)).costs['requested_time_failure'] == 1
