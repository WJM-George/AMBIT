import pytest
import torch
from types import SimpleNamespace
from stable_audio_tools.inference.sceneplan_generation_ar_count_expert import CountExpertDecoder


class FakeAR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ar_adapter = torch.nn.Linear(1, 1, bias=False)
        self.ar_lora = torch.nn.Linear(1, 1, bias=False)
        self.ar_adapter.weight.data.fill_(1.)
        self.ar_lora.weight.data.fill_(2.)
        self.fail = False
        self.eval()

    def prepare_decode_cache(self, *args, **kwargs):
        return {'position': 0, 'seen_weights': []}

    def decode_step(self, tokens, cache):
        cache['position'] += 1
        cache['seen_weights'].append(float(self.ar_adapter.weight.item()))
        return tokens[:, None].float() + self.ar_adapter.weight + self.ar_lora.weight

    def forward(self, prefix, attention_mask, context, mask):
        if self.fail:
            raise RuntimeError('expert failure')
        return prefix[:, :, None].float() + self.ar_adapter.weight + self.ar_lora.weight


def setup():
    base = FakeAR()
    wrapper = CountExpertDecoder(base, SimpleNamespace(token_to_id={'<num_sources>': 9}),
                                 {'weight': torch.tensor([[10.]])}, {'weight': torch.tensor([[20.]])})
    cache = wrapper.prepare_decode_cache(torch.zeros(2, 1, 1), torch.ones(2, 1), max_plan_tokens=20)
    return base, wrapper, cache


def test_only_count_logits_change_and_base_cache_uses_base_weights():
    base, wrapper, cache = setup()
    for step, token in enumerate([1, 2, 3, 4, 5, 9, 7, 8]):
        out = wrapper.decode_step(torch.full((2,), token), cache)
        assert torch.equal(out, torch.full((2, 1), float(token + (30 if step == 5 else 3))))
    assert cache.base_cache['seen_weights'] == [1.] * 8
    assert wrapper.count_calls == 1
    assert base.ar_adapter.weight.item() == 1. and base.ar_lora.weight.item() == 2.


def test_exception_restores_both_weight_sets():
    base, wrapper, cache = setup()
    for token in [1, 2, 3, 4, 5]:
        wrapper.decode_step(torch.full((2,), token), cache)
    base.fail = True
    with pytest.raises(RuntimeError, match='expert failure'):
        wrapper.decode_step(torch.full((2,), 9), cache)
    assert base.ar_adapter.weight.item() == 1. and base.ar_lora.weight.item() == 2.


def test_unexpected_header_fails_closed():
    base, wrapper, cache = setup()
    for token in [1, 2, 3, 4, 5]:
        wrapper.decode_step(torch.full((2,), token), cache)
    with pytest.raises(ValueError, match='six-token'):
        wrapper.decode_step(torch.full((2,), 8), cache)
    assert base.ar_adapter.weight.item() == 1.
