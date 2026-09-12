import copy

import numpy as np

from stable_audio_tools.training.transfusion_opsd.coarse_audio_retention import (
    coarse_audio_retention, foa_activity,
)


def observation(onset=.3, offset=1.6, angle=8.):
    return dict(angle_deg=angle,unobservable_fraction=0.,activity=dict(
        all_silent=False,activity_onset_sec=onset,activity_offset_sec=offset,audio_duration_sec=2.))


def test_identical_output_passes_and_coarse_slack_accepts_small_time_difference():
    a=observation(); b=observation(onset=.4,offset=1.5)
    assert coarse_audio_retention(a,a,a)['passed']
    assert coarse_audio_retention(a,b,a)['passed']


def test_direction_and_large_time_regression_both_fail():
    a=observation(); b=observation(onset=.9,angle=30.)
    r=coarse_audio_retention(a,b,a)
    assert not r['passed']
    assert set(r['failures'])=={'audio_direction','activity_onset_sec'}


def test_reference_calibration_floor_and_missing_generated_audio():
    a=observation(angle=8.); ref=observation(angle=38.); b=observation(angle=39.)
    assert coarse_audio_retention(a,b,ref)['passed']
    b=copy.deepcopy(a);b['activity'].update(all_silent=True,activity_onset_sec=None,activity_offset_sec=None)
    assert 'generated_audio_silent' in coarse_audio_retention(a,b,ref)['failures']


def test_w_activity_ignores_rotating_directional_channels():
    x=np.zeros((2000,4));x[400:1600,0]=.1
    a=foa_activity(x,1000);x[:,1:]=10
    b=foa_activity(x,1000)
    assert a==b and a['activity_onset_sec']==.4 and a['activity_offset_sec']==1.6
