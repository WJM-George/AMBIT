import math

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.native_choice_margin_retention import native_choice_margin_loss


def target(position=1, field='source_0/<transcript>', others=(1, 2)):
    return dict(position=position, field=field, token_id=0, other_ids=list(others), reference_gap=1.0)


def test_reference_has_reinforcement_gradient_only_on_legal_current_competitor():
    logits = torch.tensor([[2.0, 1.0, -2.0, 100.0]], requires_grad=True)
    loss = native_choice_margin_loss(logits, [target()])
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert torch.allclose(logits.grad, torch.tensor([[-0.5, 0.5, 0.0, 0.0]]))


def test_new_competitor_and_crossed_decision_are_penalized_more():
    a = torch.tensor([[2.0, 1.0, -2.0]], requires_grad=True)
    b = torch.tensor([[2.0, 1.0, 3.0]], requires_grad=True)
    assert native_choice_margin_loss(b, [target()]) > native_choice_margin_loss(a, [target()])
    native_choice_margin_loss(b, [target()]).backward()
    assert b.grad[0, 1] == 0 and b.grad[0, 2] > 0 and b.grad[0, 0] < 0


def test_equal_field_weight_and_no_other_positions():
    logits = torch.tensor([[2.0, 1.0, 0.0]] * 4, requires_grad=True)
    refs = [target(1), target(2), target(3, 'source_1/<transcript>')]
    native_choice_margin_loss(logits, refs).backward()
    assert logits.grad[2, 0] == 2 * logits.grad[0, 0]
    assert logits.grad[3].abs().sum() == 0


def test_empty_and_invalid_inputs():
    logits = torch.zeros((1, 3), requires_grad=True)
    native_choice_margin_loss(logits, []).backward()
    assert logits.grad.abs().sum() == 0
    with pytest.raises(ValueError):
        native_choice_margin_loss(logits, [target(others=(0, 1))])
    with pytest.raises(ValueError):
        native_choice_margin_loss(logits, [target()], temperature=0)
