import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.execution_query import ExecutionQuery, execution_query_features


def test_execution_feedback_preserves_temporal_information_and_ignores_padding():
    z = torch.arange(16).float().reshape(1, 2, 8)
    q = ExecutionQuery(z, z * 2, torch.tensor([.5]), torch.ones(1, 8, dtype=torch.bool), 0)
    features = execution_query_features(q)
    padded = ExecutionQuery(torch.nn.functional.pad(z, (0, 3), value=999.),
        torch.nn.functional.pad(z * 2, (0, 3), value=-999.), q.time,
        torch.nn.functional.pad(q.mask, (0, 3)), 0)
    torch.testing.assert_close(features, execution_query_features(padded))
    reversed_q = ExecutionQuery(z.flip(-1), (z * 2).flip(-1), q.time, q.mask, 0)
    assert not torch.equal(features, execution_query_features(reversed_q))
    assert features.shape == (1, 2 * 4 * 4 + 1) and features[0, -1] == .5


def test_execution_feedback_rejects_differentiable_or_finished_observations():
    z = torch.ones(1, 2, 8)
    mask = torch.ones(1, 8, dtype=torch.bool)
    with pytest.raises(ValueError, match='detached'):
        ExecutionQuery(z.requires_grad_(), z, torch.tensor([.5]), mask, 0)
    with pytest.raises(ValueError, match='endpoint'):
        ExecutionQuery(z.detach(), z.detach(), torch.tensor([0.]), mask, 0)
