import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.event_activity_compass_reward import RequestActivityCompassReward
from stable_audio_tools.training.transfusion_opsd.event_activity_repair import FrozenActivityRepairView
from stable_audio_tools.training.transfusion_opsd.foa_horizontal_repair import bounded_horizontal_repair
from test_transfusion_opsd_activity_compass_reward import request, tone


def test_physical_target_uses_actual_query_activity_and_independent_acceptance():
    observer = RequestActivityCompassReward(request())
    audio = tone(duration=2.1, onset=.15, offset=.85, azimuth=40., elevation=60.)
    view = FrozenActivityRepairView(observer, audio)
    before = observer.measure(audio)
    assert int(view.windows.sum()) == before['active_frames']
    windows = view.windows.clone()
    repaired, evidence = bounded_horizontal_repair(view, audio)
    after = observer.measure(repaired)
    assert after['costs']['requested_sector_failure'] < before['costs']['requested_sector_failure']
    assert after['costs']['requested_time_failure'] == 0
    assert torch.equal(audio[:, 0], repaired[:, 0]) and torch.equal(audio[:, 2], repaired[:, 2])
    assert evidence['max_applied_rotation_deg'] <= 30
    assert evidence['xy_instantaneous_energy_relative_error'] < 1e-6
    assert torch.equal(view.windows, windows)
    assert not view.evidence['is_acceptance_scorer']


def test_request_correct_query_is_exact_retention_target():
    observer = RequestActivityCompassReward(request())
    audio = tone(duration=2.01, elevation=70.)
    repaired, evidence = bounded_horizontal_repair(FrozenActivityRepairView(observer, audio), audio)
    assert torch.equal(repaired, audio) and evidence['no_op_exact']


def test_motion_windows_are_first_and_last_observed_active_frames():
    observer = RequestActivityCompassReward(request(moving=True))
    audio = tone(moving=True, endpoint_hold=False)
    view = FrozenActivityRepairView(observer, audio)
    assert view.windows.sum(-1).tolist() == [3, 3]
    assert not (view.windows[0] & view.windows[1]).any()
    repaired, evidence = bounded_horizontal_repair(view, audio)
    assert evidence['no_op_exact'] and torch.equal(repaired, audio)


def test_absent_source_has_no_physical_repair_teacher():
    observer = RequestActivityCompassReward(request())
    with pytest.raises(ValueError, match='absent source'):
        FrozenActivityRepairView(observer, tone(silence=True))


def test_frozen_view_rejects_different_audio_geometry():
    observer = RequestActivityCompassReward(request())
    audio = tone()
    view = FrozenActivityRepairView(observer, audio)
    with pytest.raises(ValueError, match='geometry'):
        view._directions(audio[..., :-1])
