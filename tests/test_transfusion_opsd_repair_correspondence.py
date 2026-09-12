import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.repair_correspondence import (
    RepairCorrespondenceError, certify_repair_correspondence, repair_fit_gain_decomposition,
)


def test_actual_anchor_and_small_arithmetic_drift_are_accepted():
    base = torch.ones(1, 2, 4)
    target = base + .01
    mask = torch.ones(1, 4, dtype=torch.bool)
    exact = certify_repair_correspondence(base, base, target, mask)
    jitter = certify_repair_correspondence(base + 1e-7, base, target, mask)
    assert exact['certified'] and exact['drift_to_repair_ratio'] == 0.
    assert jitter['certified'] and jitter['drift_to_repair_ratio'] > 0.


def test_large_anchor_shift_cannot_be_misreported_as_repair_learning():
    base = torch.zeros(1, 2, 4)
    target = base + .01
    with pytest.raises(RepairCorrespondenceError) as caught:
        certify_repair_correspondence(base - .03, base, target, torch.ones(1, 4, dtype=torch.bool))
    evidence = caught.value.diagnostics
    assert evidence['drift_to_repair_ratio'] > 2.9
    assert evidence['actual_squared_error'] > 15 * evidence['nominal_squared_error']


def test_padding_is_not_part_of_the_query_correspondence():
    base = torch.tensor([[[0., 0., float('nan')]]])
    target = torch.tensor([[[.1, .1, float('inf')]]])
    assert certify_repair_correspondence(base, base, target, torch.tensor([[True, True, False]]))['certified']


def test_zero_residual_is_not_a_certified_positive_training_signal():
    base = torch.zeros(1, 2, 4)
    with pytest.raises(ValueError, match='numerical resolution'):
        certify_repair_correspondence(base, base, base, torch.ones(1, 4, dtype=torch.bool))


def test_loss_can_improve_without_moving_in_the_certified_repair_direction():
    base = torch.zeros(1, 2, 1)
    positive = torch.tensor([[[.1], [0.]]])
    before = torch.tensor([[[0.], [.2]]])
    after = torch.tensor([[[0.], [.1]]])
    result = repair_fit_gain_decomposition(before, after, base, positive, torch.ones(1, 1, dtype=torch.bool))
    assert result['observed_loss_gain'] > 0
    assert result['repair_alignment_gain'] == 0
    assert result['anchor_drift_removal_gain'] > 0
    assert abs(result['identity_error']) < 1e-12
