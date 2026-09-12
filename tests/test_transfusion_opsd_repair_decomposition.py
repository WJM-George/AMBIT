import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.repair_decomposition import repair_response_geometry


def measure(responses, repair=(1., 0.), mask=(True, True), legal=None):
    predictions = torch.tensor(responses, dtype=torch.float64)[:, None, None, :]
    return repair_response_geometry(predictions, torch.tensor(repair)[None, None, :],
        torch.tensor(mask)[None, :], torch.tensor(legal or [True] * len(responses)))


def test_exact_repair_is_a_discrete_action():
    result = measure([[0., 0.], [1., 0.]])
    assert result['best_discrete_action'] == 1
    assert result['best_discrete_explained_fraction'] == pytest.approx(1.)
    assert result['span_explained_fraction'] == pytest.approx(1.)


def test_orthogonal_change_has_no_repair_control():
    result = measure([[0., 0.], [0., 1.]])
    assert result['best_discrete_action'] == 0
    assert result['span_explained_fraction'] == 0
    assert result['actions'][1]['discrete_explained_fraction'] == -1


def test_span_and_interpolation_do_not_certify_discrete_control():
    result = measure([[0., 0.], [2., 0.]])
    assert result['best_discrete_action'] == 0
    assert result['best_discrete_explained_fraction'] == 0
    assert result['span_explained_fraction'] == pytest.approx(1.)
    assert result['actions'][1]['optimistic_segment_alpha'] == .5
    assert result['actions'][1]['optimistic_segment_explained_fraction'] == 1
    assert not result['interpolation_is_available_action']


def test_collinear_actions_have_rank_one():
    result = measure([[0., 0.], [1., 0.], [2., 0.], [-1., 0.]])
    assert result['effective_response_rank'] == 1
    assert result['span_explained_fraction'] == pytest.approx(1.)


def test_padded_coordinates_cannot_create_control():
    result = measure([[0., 0.], [0., 10.]], repair=(1., 10.), mask=(True, False))
    assert result['span_explained_fraction'] == 0
    assert result['repair_norm'] == 1
    assert result['effective_response_rank'] == 0


def test_illegal_action_is_excluded_from_span_and_choice():
    result = measure([[0., 0.], [1., 0.], [0., 1.]], legal=[True, False, True])
    assert result['actions'][1] is None
    assert result['span_explained_fraction'] == 0
    assert result['best_discrete_action'] == 0


def test_zero_repair_is_not_a_failed_control_test():
    result = measure([[0., 0.], [1., 0.]], repair=(0., 0.))
    assert not result['nonzero_repair']
    assert result['span_explained_fraction'] is None


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_input_is_rejected(bad):
    with pytest.raises(ValueError, match='non-finite'):
        measure([[0., 0.], [bad, 1.]])


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError, match='expected predictions'):
        repair_response_geometry(torch.zeros(2, 1, 2), torch.zeros(1, 1, 2),
            torch.ones(1, 2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match='baseline'):
        measure([[0., 0.], [1., 0.]], legal=[False, True])
