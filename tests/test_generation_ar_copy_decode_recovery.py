from types import SimpleNamespace

import pytest
import torch

from stable_audio_tools.inference.sceneplan_generation_ar_learned_copy import (
    CopyDecodeLimitError,
    generate_with_learned_copy,
)


def test_one_unfinished_row_preserves_completed_rows_and_removes_the_hook():
    # The small streaming grammar makes one request terminate immediately and
    # the other exhaust the envelope. Neither row enters a literal field.
    # This exercises the actual batched loop, rather than a mocked exception.
    class Codec:
        bos_id, eos_id = 1, 2
        ids = {'<text_begin>': 4, **{f'<source_slot_{i}>': i + 5 for i in range(4)}}
        _tid = ids.__getitem__
        def allowed_next_ids(self, prefix): return {2, 3}
    class Model:
        def __init__(self):
            self.ar_adapter = SimpleNamespace(output_norm=torch.nn.Identity())
            self.prompt_conditioner = SimpleNamespace(tokenizer=None)
        def encode_requests(self, requests, device):
            return torch.zeros(len(requests), 1, 8), torch.ones(len(requests), 1).bool()
        def prepare_decode_cache(self, *args, **kw): return {}
        def decode_step(self, current, cache):
            self.ar_adapter.output_norm(torch.zeros(2, 1, 8))
            logits = torch.zeros(2, 9); logits[0, 2] = 10; logits[1, 3] = 10
            return logits
    model = Model()
    module = SimpleNamespace(FIELD_TYPES={}, encode_character_alignment=lambda *a, **kw: None)
    pointer = SimpleNamespace(prepare_keys=lambda *a, **kw: None)
    with pytest.raises(CopyDecodeLimitError) as failure:
        generate_with_learned_copy(model, pointer, module, ['short', 'nonterminating'], Codec(),
            device=torch.device('cpu'), max_plan_tokens=5)
    assert failure.value.finished == [True, False]
    assert failure.value.prefixes == [[1, 2], [1, 3, 3, 3, 3]]
    assert failure.value.traces == [[], []]
    assert not model.ar_adapter.output_norm._forward_hooks
