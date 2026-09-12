import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.objectives import RewardScore
from stable_audio_tools.training.transfusion_opsd.reference_repair import (
    protected_against_references, build_reference_repair, normalized_spectral_reference)


def test_candidate_improvement_does_not_hide_original_content_regression():
    original = RewardScore(1., {'content': .1, 'speech': .1})
    candidate = RewardScore(2., {'content': .4, 'speech': .2})
    improved = RewardScore(3., {'content': .3, 'speech': .1})
    assert improved.improves(candidate)
    assert not protected_against_references(improved, [original, candidate], min_gain=.001)
    assert protected_against_references(RewardScore(3., {'content': .1, 'speech': .1}),
        [original, candidate], min_gain=.001)
    with pytest.raises(ValueError, match='identical protection'):
        protected_against_references(RewardScore(3., {'content': 0}), [original], min_gain=.001)


def test_reference_target_is_detached_and_protects_both_contexts():
    anchor = torch.ones(1, 1, 4)
    mask = torch.ones(1, 4, dtype=torch.bool)
    def score(value):
        amount = float(value.mean())
        return RewardScore(amount, {'content': 2. - amount})
    ref = RewardScore(.5, {'content': .9})
    result, evidence = build_reference_repair(anchor, mask, objective=lambda value: value.sum(),
        score_clean=score, score_suffix=score, reference_clean=ref, reference_suffix=ref,
        radius=.2, target_steps=2)
    assert evidence['qualified'] and result is not None
    assert not result.positive.requires_grad and score(result.positive).costs['content'] <= .9
    assert torch.equal(anchor, torch.ones_like(anchor))


def test_spectral_reference_keeps_gradient_and_ignores_directional_channels():
    torch.manual_seed(17)
    reference = torch.randn(1, 4, 8192)
    equal = reference.clone().requires_grad_(True)
    assert normalized_spectral_reference(equal, reference).item() == 0.
    altered = reference.clone()
    altered[:, 0, 2000:3000] *= .2
    altered.requires_grad_(True)
    loss = normalized_spectral_reference(altered, reference)
    gradient, = torch.autograd.grad(loss, altered)
    assert loss > 0 and gradient[:, 0].abs().sum() > 0
    assert gradient[:, 1:].count_nonzero() == 0
    assert torch.allclose(normalized_spectral_reference(reference * 2., reference), torch.tensor(0.), atol=1e-10)
