import pytest
import torch
from stable_audio_tools.training.transfusion_opsd.native_stochastic_policy import ordered_span_log_prob, kind_assignments

def test_ordered_span_distribution_normalizes_and_has_head_gradients():
    a = torch.tensor([.4, -.2, .7, .1], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([-.1, .3, -.8, .6], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([True, False, True, True])
    support = [(i, j) for i in range(4) for j in range(i, 4) if mask[i] and mask[j]]
    logits = torch.stack([(a[i] + b[j]) / .7 for i, j in support])
    probabilities = torch.stack([ordered_span_log_prob(a, b, mask, s, temperature=.7) for s in support])
    torch.testing.assert_close(probabilities, logits.log_softmax(0))
    assert probabilities.exp().sum().item() == pytest.approx(1.)
    (-probabilities[-1]).backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert a.grad[1] == 0 and b.grad[1] == 0


def test_full_source_kind_support_preserves_codec_speech_limit():
    assert len(kind_assignments(4)) == 48
    assert all(row.count(2) <= 1 for row in kind_assignments(4))
