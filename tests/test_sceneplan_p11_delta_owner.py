from __future__ import annotations

import pytest
import torch

from stable_audio_tools.models.sceneplan_p11_v4 import _delta_owner_objective


def _objective(
    logits: torch.Tensor,
    target_slot: int,
    legal: list[bool],
) -> tuple[torch.Tensor, torch.Tensor]:
    return _delta_owner_objective(
        logits,
        target_slot,
        torch.tensor(legal),
        margin=2.0,
        margin_weight=1.0,
    )


def test_delta_owner_uses_only_inference_legal_source_logits() -> None:
    logits = torch.tensor([3.0, 1.0, 100.0, -100.0], requires_grad=True)
    loss, correct = _objective(logits, 0, [True, True, False, False])
    loss.backward()

    assert float(correct) == 1.0
    assert logits.grad is not None
    assert float(logits.grad[0]) < 0.0
    assert float(logits.grad[1]) > 0.0
    assert torch.equal(logits.grad[2:], torch.zeros_like(logits.grad[2:]))


def test_delta_owner_target_changes_the_same_autoregressive_decision() -> None:
    first = torch.zeros(4, requires_grad=True)
    second = torch.zeros(4, requires_grad=True)
    first_loss, _ = _objective(first, 0, [True, True, False, False])
    second_loss, _ = _objective(second, 1, [True, True, False, False])
    first_loss.backward()
    second_loss.backward()

    assert first.grad is not None and second.grad is not None
    assert torch.allclose(first.grad[:2], second.grad[:2].flip(0))
    assert float(first.grad[0]) < 0.0 < float(first.grad[1])
    assert float(second.grad[1]) < 0.0 < float(second.grad[0])


def test_delta_owner_one_source_scene_has_no_artificial_loss() -> None:
    logits = torch.randn(4, requires_grad=True)
    loss, correct = _objective(logits, 2, [False, False, True, False])
    loss.backward()

    assert float(loss) == 0.0
    assert float(correct) == 1.0
    assert logits.grad is not None
    assert torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_delta_owner_fails_closed_when_teacher_owner_is_absent() -> None:
    with pytest.raises(ValueError, match="absent from input ScenePlan"):
        _objective(torch.zeros(4), 1, [True, False, False, False])
