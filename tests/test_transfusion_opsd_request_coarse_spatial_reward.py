import math
from types import SimpleNamespace

import torch

from stable_audio_tools.training.transfusion_opsd.request_coarse_spatial_reward import RequestCoarseSpatialReward


def evaluator(start='front', end=None):
    constraints = [dict(op='motion', value='static' if end is None else 'linear')]
    if end is None:
        constraints.append(dict(op='compass', point='both', value=start))
    else:
        constraints += [dict(op='compass', point='start', value=start), dict(op='compass', point='end', value=end)]
    return RequestCoarseSpatialReward(SimpleNamespace(requirements=dict(sources=[dict(constraints=constraints)], scene=[], relations=[])))


def plane(angles, gain=.1):
    angles = torch.as_tensor(angles, dtype=torch.float64)
    if angles.ndim == 0:
        angles = angles.expand(44100)
    w = gain*torch.sin(torch.arange(len(angles), dtype=torch.float64)*2*math.pi*440/44100)
    radians = torch.deg2rad(angles)
    return torch.stack([w, w*radians.sin(), torch.zeros_like(w), w*radians.cos()])[None].float()


def test_coarse_angle_accepts_small_boundary_difference_rejects_wrong_sector():
    reward = evaluator()
    assert reward.measure(plane(25.))['costs']['requested_sector_failure'] == 0
    assert reward.measure(plane(90.))['costs']['requested_sector_failure'] == 1


def test_global_front_average_cannot_hide_left_then_right():
    angles = torch.cat([torch.full((22050,), -60.), torch.full((22050,), 60.)])
    result = evaluator().measure(plane(angles))
    assert result['costs']['requested_sector_failure'] >= .8
    assert result['intervals'][0]['azimuth_deg'] < -59
    assert result['intervals'][-1]['azimuth_deg'] > 59


def test_movement_correct_reverse_and_stationary_are_distinguished():
    reward = evaluator('left', 'front')
    good = reward.measure(plane(torch.linspace(90, 0, 44100)))['costs']
    reverse = reward.measure(plane(torch.linspace(0, 90, 44100)))['costs']
    still = reward.measure(plane(45.))['costs']
    assert good['requested_sector_failure'] == good['motion_trend_failure'] == 0
    assert reverse['requested_sector_failure'] == reverse['motion_trend_failure'] == 1
    assert still['motion_trend_failure'] == 1


def test_level_and_yaw_covariance_within_declared_presence_range():
    front, left = evaluator(), evaluator('left')
    a, b, c = front.measure(plane(12.)), front.measure(plane(12., gain=.05)), left.measure(plane(102.))
    assert a['costs'] == b['costs'] == c['costs']
    assert abs(a['mean_excess_angle_deg']-c['mean_excess_angle_deg']) < 1e-6


def test_silence_and_vertical_only_cannot_satisfy_horizontal_request():
    reward = evaluator()
    silent = reward.measure(torch.zeros(1, 4, 44100))['costs']
    vertical = plane(0.)
    vertical[:, 2] = vertical[:, 3].clone()
    vertical[:, 3] = 0
    result = reward.measure(vertical)['costs']
    assert silent['source_presence_failure'] == silent['direction_unobservable_fraction'] == 1
    assert result['source_presence_failure'] == 0 and result['direction_unobservable_fraction'] == 1
