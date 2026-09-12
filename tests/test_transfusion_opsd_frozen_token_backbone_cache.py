from types import SimpleNamespace

import pytest
import torch
from torch import nn

from stable_audio_tools.training.transfusion_opsd.frozen_token_backbone_cache import FrozenTokenBackboneCache


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 3)
        self.calls = 0

    def forward(self, *, input_ids, attention_mask, use_cache, return_dict):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids) * attention_mask[..., None])


def fixture():
    model = Backbone().eval().requires_grad_(False)
    cache = FrozenTokenBackboneCache(model, maximum_bytes=48)
    kw = dict(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.ones(1, 2, dtype=torch.long),
              use_cache=False, return_dict=True)
    return model, cache, kw


def test_hit_is_exact_and_downstream_projection_keeps_its_gradient():
    model, cache, kw = fixture()
    with torch.no_grad():
        reference = cache(**kw).last_hidden_state
        actual = cache(**kw).last_hidden_state
    assert torch.equal(reference, actual) and model.calls == 1
    projection = nn.Linear(3, 1)
    projection(actual).square().sum().backward()
    assert projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_mask_and_ids_are_distinct_and_cache_is_bounded():
    model, cache, kw = fixture()
    with torch.no_grad():
        cache(**kw)
        cache(**dict(kw, attention_mask=torch.tensor([[1, 0]])))
        cache(**dict(kw, input_ids=torch.tensor([[3, 4]])))
    assert model.calls == 3 and cache.bytes <= 48


def test_mutating_returned_hidden_cannot_poison_a_later_hit():
    _, cache, kw = fixture()
    with torch.no_grad():
        original = cache(**kw).last_hidden_state.clone()
        cache(**kw).last_hidden_state.zero_()
        assert torch.equal(original, cache(**kw).last_hidden_state)


def test_frozen_weight_change_is_rejected():
    model, cache, kw = fixture()
    with torch.no_grad():
        cache(**kw)
        model.embedding.weight.add_(1)
        with pytest.raises(ValueError, match='changed'):
            cache(**kw)
        assert not cache.receipt()['identity_unchanged']
