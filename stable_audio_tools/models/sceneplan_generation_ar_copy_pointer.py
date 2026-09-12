"""Experimental learned character-span copying for Generation AR text fields.

Inference features come only from the raw request, its frozen token context,
and the generated AR prefix. Source counts, source spans, GT plans and template
parsers are not inference inputs. The existing codec/P10 schema is unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FIELD_TYPES = {'description': 0, 'speaker_description': 1, 'transcript': 2}
MAX_TOKEN_CHAR_OFFSET = 63


@dataclass
class CharRequestBatch:
    token_indices: torch.Tensor
    characters: torch.Tensor
    token_char_offsets: torch.Tensor
    relative_positions: torch.Tensor
    endpoint_mask: torch.Tensor

    def to(self, device):
        return CharRequestBatch(**{k: v.to(device) for k, v in vars(self).items()})


def _character_id(value):
    if value is None: return 0
    code = ord(value)
    return code + 1 if code < 128 else 129 + code % 383


def character_alignment(requests: Sequence[str], offsets, attention_mask) -> CharRequestBatch:
    """Map raw Unicode character positions to the actual encoder token grid.

    Byte-token splits may cover the same Unicode character. Use the final
    covering token consistently, whose causal context contains preceding bytes.
    Only non-whitespace, encoder-covered endpoints can be selected. Spaces
    inside a selected span are copied directly from the unchanged raw string.
    """
    assert len(requests) == len(offsets) == len(attention_mask) and requests
    if any(not isinstance(r, str) or not r.strip() for r in requests): raise ValueError('Empty raw request')
    batch = len(requests); width = max(map(len, requests))
    if width > 8192: raise ValueError('Copy pilot raw request exceeds its explicit 8192-character capability limit')
    indices = np.zeros((batch, width), dtype=np.int64)
    chars = np.zeros((batch, width, 3), dtype=np.int64)
    within = np.zeros((batch, width), dtype=np.int64)
    positions = np.zeros((batch, width, 2), dtype=np.float32)
    valid = np.zeros((batch, width), dtype=np.bool_)
    for row, text in enumerate(requests):
        token_for_char = np.full(len(text), -1, dtype=np.int64); token_start = np.zeros(len(text), dtype=np.int64)
        active_tokens = np.asarray(attention_mask[row], dtype=np.bool_)
        real_tokens = int(active_tokens.sum()); token_ordinals = active_tokens.cumsum() - 1
        for token, ((start, end), active) in enumerate(zip(offsets[row], attention_mask[row])):
            start, end = int(start), int(end)
            if not active or start == end: continue
            if not 0 <= start < end <= len(text): raise ValueError('Tokenizer offsets do not address the raw request')
            token_for_char[start:end] = token; token_start[start:end] = start
        size = len(text); char_positions = np.arange(size); safe_tokens = np.maximum(token_for_char, 0)
        values = np.fromiter((_character_id(c) for c in text), dtype=np.int64, count=size)
        indices[row, :size] = safe_tokens; chars[row, :size, 1] = values
        chars[row, 1:size, 0] = values[:-1]; chars[row, :size - 1, 2] = values[1:]
        within[row, :size] = np.minimum(MAX_TOKEN_CHAR_OFFSET, char_positions - token_start)
        positions[row, :size, 0] = char_positions / max(1, size - 1)
        # Nonpadding token ordinals make features independent of batch padding.
        positions[row, :size, 1] = np.maximum(token_ordinals[safe_tokens], 0) / max(1, real_tokens - 1)
        valid[row, :size] = (token_for_char >= 0) & np.fromiter((not c.isspace() for c in text), dtype=np.bool_, count=size)
        if not bool(valid[row].any()) or real_tokens == 0: raise ValueError('Raw request has no selectable encoder-covered character')
    return CharRequestBatch(*(torch.from_numpy(a) for a in (indices, chars, within, positions, valid)))


def encode_character_alignment(tokenizer, requests, context_mask, *, device):
    encoded = tokenizer(list(requests), add_special_tokens=True, padding=True, truncation=False,
        return_offsets_mapping=True, return_tensors='pt')
    attention = torch.as_tensor(encoded['attention_mask'], dtype=torch.bool)
    if not torch.equal(attention, context_mask.detach().cpu().bool()):
        raise ValueError('Pointer token offsets are not aligned with the actual AR request context')
    return character_alignment(requests, encoded['offset_mapping'].tolist(), attention.tolist()).to(device)


def literal_field_targets(codec, token_ids, request):
    """Training-only literal span labels at each GT <text_begin> query.

    Repeated literal values use successive occurrences, wrapping when needed;
    all alternatives are retained, and quality is scored on the copied string.
    These labels must never enter a raw-generation call.
    """
    ids = [int(x) for x in token_ids]
    tags = {codec._tid('<' + name + '>'): name for name in FIELD_TYPES}
    begin, end = codec._tid('<text_begin>'), codec._tid('<text_end>')
    targets = []; used = {}
    for position, token in enumerate(ids):
        if token not in tags: continue
        if ids[position + 1] != begin: raise ValueError('Codec text field is missing <text_begin>')
        stop = ids.index(end, position + 2)
        pieces = [v - codec.text_offset for v in ids[position + 2:stop]]
        if not pieces or any(v < 0 or v >= codec.text_processor.get_piece_size() for v in pieces):
            raise ValueError('Invalid codec text pieces in copy supervision')
        text = ' '.join(codec.text_processor.decode(pieces).split())
        alternatives = []; start = request.find(text)
        while start >= 0:
            alternatives.append((start, start + len(text) - 1)); start = request.find(text, start + 1)
        if not alternatives: raise ValueError('GT text field is not a literal span in the raw request: ' + repr(text))
        occurrence = used.get(text, 0); used[text] = occurrence + 1
        start, stop = alternatives[occurrence % len(alternatives)]
        targets.append({'query_position': position + 1, 'field': tags[token], 'field_type': FIELD_TYPES[tags[token]],
            'start': start, 'end': stop, 'text': text, 'alternative_spans': alternatives})
    if not targets: raise ValueError('No text fields in GT codec plan')
    return targets


def best_ordered_span(start_logits, end_logits, endpoint_mask):
    """Exact O(C) maximizer of start[s] + end[e] with s <= e, no GT length."""
    if start_logits.shape != end_logits.shape: raise ValueError('Start/end logits must have the same shape')
    mask = endpoint_mask
    while mask.ndim < start_logits.ndim: mask = mask.unsqueeze(-2)
    if not bool(mask.any(-1).all()): raise ValueError('No valid endpoint in a raw request')
    start = start_logits.masked_fill(~mask, -torch.inf)
    end = end_logits.masked_fill(~mask, -torch.inf)
    prefix_max, prefix_index = start.cummax(-1)
    end_index = (prefix_max + end).argmax(-1)
    start_index = prefix_index.gather(-1, end_index.unsqueeze(-1)).squeeze(-1)
    return start_index, end_index


class LiteralCopyPointer(nn.Module):
    """Small learned span head over raw characters and frozen AR features."""
    def __init__(self, hidden_dim=1024, width=128):
        super().__init__(); self.hidden_dim = hidden_dim; self.width = width
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.context_projection = nn.Linear(hidden_dim, width)
        # A short bidirectional context window exposes adjacent word boundaries
        # while retaining the frozen encoder's contextual features.
        self.token_mix = nn.Conv1d(width, width, 5, padding=2)
        self.characters = nn.Embedding(512, 16, padding_idx=0)
        self.within_token = nn.Embedding(MAX_TOKEN_CHAR_OFFSET + 1, 8)
        self.boundaries = nn.Linear(3 * 16 + 8 + 2, width)
        self.key_norm = nn.LayerNorm(width)
        self.start_keys = nn.Linear(width, width)
        self.end_keys = nn.Linear(width, width)
        self.start_query = nn.Linear(hidden_dim, width)
        self.end_query = nn.Linear(hidden_dim, width)
        self.field_embedding = nn.Embedding(len(FIELD_TYPES), width)
        self.start_bias = nn.Linear(width, 1)
        self.end_bias = nn.Linear(width, 1)

    def prepare_keys(self, request_context, context_mask, alignment: CharRequestBatch):
        if request_context.shape[:2] != context_mask.shape: raise ValueError('Context/mask mismatch')
        context = self.context_projection(self.context_norm(request_context.float()))
        context = context * context_mask[..., None]
        context = context + F.gelu(self.token_mix(context.transpose(1, 2)).transpose(1, 2))
        gathered = context.gather(1, alignment.token_indices[..., None].expand(-1, -1, self.width))
        local = torch.cat([self.characters(alignment.characters).flatten(-2), self.within_token(alignment.token_char_offsets),
            alignment.relative_positions], dim=-1)
        keys = self.key_norm(gathered + self.boundaries(local))
        return self.start_keys(keys), self.end_keys(keys), self.start_bias(keys).squeeze(-1), self.end_bias(keys).squeeze(-1), alignment.endpoint_mask

    def score_queries(self, queries, field_types, prepared):
        a, b, a_bias, b_bias, valid = prepared
        kind = self.field_embedding(field_types)
        start = torch.einsum('bqd,bcd->bqc', self.start_query(queries.float()) + kind, a) / math.sqrt(self.width) + a_bias[:, None]
        end = torch.einsum('bqd,bcd->bqc', self.end_query(queries.float()) + kind, b) / math.sqrt(self.width) + b_bias[:, None]
        return start.masked_fill(~valid[:, None], -torch.inf), end.masked_fill(~valid[:, None], -torch.inf)

    def forward(self, queries, field_types, request_context, context_mask, alignment):
        return self.score_queries(queries, field_types, self.prepare_keys(request_context, context_mask, alignment))
