import importlib.util
from pathlib import Path
import sys

import pytest
import torch

path = Path(__file__).resolve().parents[1] / 'stable_audio_tools/models/sceneplan_generation_ar_copy_pointer.py'
spec = importlib.util.spec_from_file_location('copy_pointer_test_module', path)
copy = importlib.util.module_from_spec(spec); sys.modules[spec.name] = copy; spec.loader.exec_module(copy)


def test_unicode_byte_offsets_and_padding_do_not_change_raw_features():
    offsets = [[(0, 0), (0, 1), (2, 5), (5, 6), (5, 6)]]
    a = copy.character_alignment(['a café'], offsets, [[1] * 5])
    b = copy.character_alignment(['a café'], [[(0, 0)] + offsets[0] + [(0, 0)]], [[0] + [1] * 5 + [0]])
    assert a.token_indices[0, 5] == 4  # Last byte token covering é.
    assert a.endpoint_mask.tolist() == [[True, False, True, True, True, True]]
    for field in ('characters', 'token_char_offsets', 'relative_positions', 'endpoint_mask'):
        assert torch.equal(getattr(a, field), getattr(b, field))
    assert torch.equal(a.token_indices[a.endpoint_mask] + 1, b.token_indices[b.endpoint_mask])


def test_best_span_matches_exhaustive_ordered_search_and_excludes_padding():
    torch.manual_seed(42)
    start, end = torch.randn(3, 2, 9), torch.randn(3, 2, 9)
    mask = torch.tensor([[1, 0, 1, 1, 0, 1, 1, 0, 0]] * 3).bool()
    s, e = copy.best_ordered_span(start, end, mask)
    for b in range(3):
        for q in range(2):
            allowed = [(float(start[b, q, i] + end[b, q, j]), i, j) for i in range(9) for j in range(i, 9) if mask[b, i] and mask[b, j]]
            best = max(allowed)
            assert (int(s[b, q]), int(e[b, q])) == best[1:]
    with pytest.raises(ValueError, match='No valid endpoint'):
        copy.best_ordered_span(start, end, torch.zeros_like(mask))


class ByteCodec:
    text_offset = 100
    ids = {'<description>': 1, '<speaker_description>': 2, '<transcript>': 3, '<text_begin>': 4, '<text_end>': 5}
    _tid = ids.__getitem__
    class Processor:
        def get_piece_size(self): return 256
        def decode(self, ids): return bytes(ids).decode()
    text_processor = Processor()


def test_training_spans_preserve_nested_quotes_and_distinguish_repeated_values():
    codec = ByteCodec(); words = 'No. “Yes”.'
    request = f'The first voice says “{words}”; the other voice also says “{words}”.'
    field = [3, 4, *[v + 100 for v in words.encode()], 5]
    targets = copy.literal_field_targets(codec, field + field, request)
    assert len(targets) == 2 and targets[0]['start'] < targets[1]['start']
    for t in targets:
        assert request[t['start']:t['end'] + 1] == words
        assert t['field'] == 'transcript'
    with pytest.raises(ValueError, match='not a literal span'):
        copy.literal_field_targets(codec, field, 'A different English request.')


def test_learned_head_ignores_encoder_padding_in_span_features():
    torch.manual_seed(7)
    head = copy.LiteralCopyPointer(hidden_dim=8, width=8).eval()
    text = ['abc']; offsets = [[(0, 1), (1, 2), (2, 3)]]
    a = copy.character_alignment(text, offsets, [[1, 1, 1]])
    b = copy.character_alignment(text, [[(0, 0)] + offsets[0] + [(0, 0)]], [[0, 1, 1, 1, 0]])
    context = torch.randn(1, 3, 8); padded = torch.cat([torch.randn(1, 1, 8), context, torch.randn(1, 1, 8)], dim=1)
    query = torch.randn(1, 1, 8); kind = torch.tensor([[2]])
    first = head(query, kind, context, torch.ones(1, 3).bool(), a)
    second = head(query, kind, padded, torch.tensor([[0, 1, 1, 1, 0]]).bool(), b)
    for x, y in zip(first, second): torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-6)
