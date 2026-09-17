from types import SimpleNamespace

import pytest
import torch

import stable_audio_tools.models.sceneplan_transfusion_editing_ar as native
import stable_audio_tools.training.transfusion_opsd.native_greedy_fastpath as optimized


class Model(torch.nn.Module):
    generate_batch = native.ScenePlanTransfusionEditingAR.generate_batch

    def __init__(self):
        super().__init__()
        self.pad_id = 0
        self.forward_prefixes = []
        self.context_calls = 0
        self.source_clap_model = SimpleNamespace(source_features=lambda source, mask: source.sum(-1))
        self.eval()

    def encode_edit_instructions(self, instructions, *, device):
        self.context_calls += 1
        return torch.zeros(len(instructions), 1, 2), torch.ones(len(instructions), 1, dtype=torch.bool)

    def forward(self, source, mask, ids, plan_mask, context, context_mask, **semantic):
        assert torch.equal(semantic['source_clap_features'], source.sum(-1))
        self.forward_prefixes.append(ids.tolist())
        return torch.arange(10.).expand(ids.shape[0], ids.shape[1], 10)


@pytest.fixture
def grammar(monkeypatch):
    def allowed(codec, prefix, *, fixed_duration_sec):
        assert fixed_duration_sec == 10.
        return [{2}, {3, 4}, {5}, {6, 7}, {8}][len(prefix) - 1]
    monkeypatch.setattr(native, 'editing_ar_allowed_next_ids', allowed)
    monkeypatch.setattr(optimized, 'editing_ar_allowed_next_ids', allowed)
    return dict(codec=SimpleNamespace(bos_id=1, eos_id=8), max_plan_tokens=10, fixed_duration_sec=[10.])


def test_forced_tokens_keep_native_output_and_choice_prefixes_exact(grammar):
    model = Model()
    source, mask = torch.zeros(1, 64, 432), torch.ones(1, 432, dtype=torch.bool)
    expected = model.generate_batch(source, mask, ['edit'], **grammar)
    original_prefixes = list(model.forward_prefixes)
    fastpath = optimized.NativeGreedyFastpath(model)
    fastpath.enabled = True
    model.forward_prefixes.clear()
    state = torch.get_rng_state().clone()
    actual = model.generate_batch(source, mask, ['edit'], **grammar)
    assert torch.equal(actual[0], expected[0])
    assert actual[0].tolist() == [1, 2, 4, 5, 7, 8]
    assert model.forward_prefixes == [original_prefixes[1], original_prefixes[3]]
    assert fastpath.statistics() == dict(skipped_forwards=3, choice_forwards=2)
    assert torch.equal(state, torch.get_rng_state())
    assert model.context_calls == 2


def test_disabled_and_training_paths_use_original_generation(grammar):
    model = Model()
    fastpath = optimized.NativeGreedyFastpath(model)
    source, mask = torch.zeros(1, 64, 432), torch.ones(1, 432, dtype=torch.bool)
    model.generate_batch(source, mask, ['edit'], **grammar)
    assert len(model.forward_prefixes) == 5
    model.train(); fastpath.enabled = True
    model.generate_batch(source, mask, ['edit'], **grammar)
    assert len(model.forward_prefixes) == 10
    assert fastpath.skipped_forwards == fastpath.choice_forwards == 0


def test_batched_generation_remains_native(grammar):
    model = Model()
    fastpath = optimized.NativeGreedyFastpath(model); fastpath.enabled = True
    grammar['fixed_duration_sec'] = [10., 10.]
    actual = model.generate_batch(torch.zeros(2, 64, 432), torch.ones(2, 432, dtype=torch.bool),
                                  ['first', 'second'], **grammar)
    assert len(actual) == 2 and torch.equal(actual[0], actual[1])
    assert len(model.forward_prefixes) == 5 and fastpath.skipped_forwards == 0


def test_native_token_limit_failure_is_retained(grammar):
    model = Model()
    fastpath = optimized.NativeGreedyFastpath(model); fastpath.enabled = True
    grammar['max_plan_tokens'] = 4
    with pytest.raises(RuntimeError, match='did not emit EOS'):
        model.generate_batch(torch.zeros(1, 64, 432), torch.ones(1, 432, dtype=torch.bool), ['edit'], **grammar)
