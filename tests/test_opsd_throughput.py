from types import SimpleNamespace

import pytest
import torch

from stable_audio_tools.training.transfusion_opsd.frozen_forward_cache import FrozenForwardCache
from stable_audio_tools.training.transfusion_opsd.editing_stream import OrdinalStream
from scripts.t2a.rl.train_editing_opsd_throughput import choose_microbatch, peek, native_plans


class FrozenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.eval()

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=(input_ids.float() * attention_mask).unsqueeze(-1))


def inputs(value=1):
    return dict(input_ids=torch.tensor([[value, 2]]), attention_mask=torch.ones(1, 2, dtype=torch.bool),
                use_cache=False, return_dict=True)


def test_cache_preserves_values_without_reusing_trainable_projection_graph():
    model = FrozenModel()
    cache = FrozenForwardCache(model)
    projection = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        a = model(**inputs()).last_hidden_state
    projection(a).sum().backward()
    first_gradient = projection.weight.grad.clone()
    projection.zero_grad()
    with torch.no_grad():
        b = model(**inputs()).last_hidden_state
    projection(b).sum().backward()
    assert torch.equal(a, b)
    assert torch.equal(first_gradient, projection.weight.grad)
    assert model.calls == 1 and cache.hits == 1


def test_downstream_mutation_cannot_corrupt_cached_backbone_output():
    model = FrozenModel()
    cache = FrozenForwardCache(model)
    with torch.no_grad():
        first = model(**inputs()).last_hidden_state
        first.zero_()
        second = model(**inputs()).last_hidden_state
        second.fill_(99)
        third = model(**inputs()).last_hidden_state
    assert torch.equal(third, torch.tensor([[[1.], [2.]]]))
    assert model.calls == 1 and cache.hits == 2


def test_frozen_cache_is_bounded_and_keys_the_mask_and_tokens():
    model = FrozenModel()
    cache = FrozenForwardCache(model, maximum_bytes=8)
    with torch.no_grad():
        model(**inputs(1)); model(**inputs(3)); model(**inputs(1))
        altered = inputs(1); altered['attention_mask'][0, 1] = False
        output = model(**altered)
    assert model.calls == 4
    assert output.last_hidden_state[0, 1] == 0
    assert cache.bytes <= 8 and len(cache.entries) == 1


def test_trainable_or_training_backbone_cannot_be_cached():
    with pytest.raises(ValueError):
        FrozenForwardCache(torch.nn.Linear(1, 1))


def test_sampler_peek_does_not_advance_checkpoint_cursor():
    stream = OrdinalStream(range(64), seed=12, rank=2, world=4)
    stream.take(3)
    state = stream.state_dict()
    preview = peek(stream, 4)
    assert stream.state_dict() == state
    assert preview == stream.take(4)


def test_microbatch_selection_excludes_oom_and_memory_pressure():
    records = [dict(microbatch=48, ok=True, seconds=10., peak_allocated_MiB=35000),
               dict(microbatch=64, ok=True, seconds=7., peak_allocated_MiB=47500),
               dict(microbatch=56, ok=True, seconds=8., peak_allocated_MiB=41500)]
    assert choose_microbatch(records, memory_limit_MiB=44000) == 56
    records[2]['ok'] = False
    assert choose_microbatch(records, memory_limit_MiB=44000) == 48


def test_larger_microbatch_without_measured_speed_gain_is_not_selected():
    records = [dict(microbatch=48, ok=True, seconds=10., peak_allocated_MiB=35000),
               dict(microbatch=64, ok=True, seconds=9.8, peak_allocated_MiB=41000)]
    assert choose_microbatch(records, memory_limit_MiB=44000) == 48


def test_native_planning_keeps_buckets_duration_and_scalar_sample_ids(monkeypatch):
    import stable_audio_tools.models.sceneplan_transfusion_editing_clap44_pipeline as pipeline
    calls = []
    class View:
        def __init__(self, **unused):
            pass
        def generate_new_sceneplans(self, source, mask, requests, *, duration_sec):
            calls.append((source.shape, requests, duration_sec))
            return ([dict(sample_id=f'edited_{i:06d}', text=text) for i, text in enumerate(requests)],
                    [torch.tensor([int(text)]) for text in requests])
    monkeypatch.setattr(pipeline, 'ScenePlanTransfusionEditingCLAP44Pipeline', View)
    adapter = SimpleNamespace(diffusion=None, ar=None, codec=None,
        native_plan=lambda obs: (dict(sample_id='edited_000000', text=obs.request), torch.tensor([int(obs.request)])))
    observations = {i: SimpleNamespace(source_foa_latent=torch.zeros(1, 64, frames),
        source_attention_mask=torch.ones(1, frames, dtype=torch.bool), model_num_samples=44100 * (i + 1),
        request=str(i)) for i, frames in enumerate([432, 648, 432, 432])}
    results = native_plans(adapter, observations, 2)
    assert all(plan['sample_id'] == 'edited_000000' for plan, _ in results.values())
    assert len(calls) == 1 and calls[0][0] == (2, 64, 432)
    assert calls[0][1] == ['0', '2'] and calls[0][2] == [1., 3.]
