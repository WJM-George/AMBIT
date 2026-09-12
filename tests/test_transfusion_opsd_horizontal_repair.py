from dataclasses import replace

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.event_directional_reward import FixedReferenceDirectionalReward
from stable_audio_tools.training.transfusion_opsd.foa_horizontal_repair import (
    HorizontalRepairConfig,bounded_horizontal_repair,horizontal_components)
from test_transfusion_opsd_compass_reward import plane_wave
from test_transfusion_opsd_event import codec,example


def fixture(codec,*,azimuth,elevation=0.):
    plan,_,request=example(codec)
    samples=round(plan['duration_sec']*44100)
    audio=plane_wave(samples,azimuth=azimuth,elevation=elevation)
    reference=FixedReferenceDirectionalReward(request,plan,reference_audio=audio,model_num_samples=samples)
    return reference,audio


def test_physical_repair_changes_wrong_direction_and_preserves_content_field(codec):
    reference,audio=fixture(codec,azimuth=90.,elevation=30.)
    repaired,evidence=bounded_horizontal_repair(reference,audio)
    assert horizontal_components(reference,audio)['requested_horizontal_failure']==1
    assert horizontal_components(reference,repaired)['requested_horizontal_failure']<.1
    assert torch.equal(audio[:,0],repaired[:,0]) and torch.equal(audio[:,2],repaired[:,2])
    assert evidence['xy_instantaneous_energy_relative_error']<1e-6
    assert evidence['objective_after']<evidence['objective_before']


@pytest.mark.parametrize('elevation',[-70.,0.,70.])
def test_already_valid_horizontal_audio_is_exact_identity(codec,elevation):
    reference,audio=fixture(codec,azimuth=60.,elevation=elevation)
    repaired,evidence=bounded_horizontal_repair(reference,audio)
    assert torch.equal(repaired,audio)
    assert evidence['reachable_wrong_frames']==0
    assert evidence['normalized_mean_squared_rotation']==0


def test_too_small_declared_budget_does_not_fabricate_a_repair(codec):
    reference,audio=fixture(codec,azimuth=100.)
    repaired,evidence=bounded_horizontal_repair(reference,audio,config=replace(HorizontalRepairConfig(),max_rotation_deg=5.))
    assert torch.equal(repaired,audio)
    assert horizontal_components(reference,repaired)['requested_horizontal_failure']==1
    assert evidence['reachable_wrong_frames']==0


def test_unobservable_frames_have_no_invented_direction_target(codec):
    reference,audio=fixture(codec,azimuth=90.)
    changed=audio.clone();changed[:,:,20*1024:40*1024]=0
    repaired,evidence=bounded_horizontal_repair(reference,changed)
    assert torch.equal(repaired[:,:,20*1024:40*1024],changed[:,:,20*1024:40*1024])
    assert horizontal_components(reference,repaired)['direction_unobservable_fraction'] == pytest.approx(
        horizontal_components(reference,changed)['direction_unobservable_fraction'],abs=1e-6)


def test_negative_smoothness_budget_is_rejected():
    with pytest.raises(ValueError):
        HorizontalRepairConfig(smoothness_penalty=-1.)
