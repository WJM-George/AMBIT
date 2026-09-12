"""Learn source inventory from raw context; no teacher prefix or parser input.

Learned queries predict count, source kinds and Unicode character endpoints.
The raw text is copied only after the endpoint model has selected a span.
Targets and source-owned evidence are used by the training loss, never forward.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

KINDS = ('sound', 'music', 'speech', 'absent')
TEXT_FIELDS = ('identity', 'transcript')


class InventoryBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width = width
        self.heads = heads
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width),
                                 nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, query, context, mask):
        normalized = self.self_norm(query)
        query = query + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        batch, slots, width = query.shape
        q = self.query(self.cross_norm(query)).reshape(batch, slots, self.heads, -1).transpose(1, 2)
        k = self.key(context).reshape(batch, -1, self.heads, width // self.heads).transpose(1, 2)
        v = self.value(context).reshape(batch, -1, self.heads, width // self.heads).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(width // self.heads)
        attention = scores.masked_fill(~mask[:, None, None], -torch.inf).softmax(-1)
        query = query + self.output((attention @ v).transpose(1, 2).reshape(batch, slots, width))
        query = query + self.ffn(query)
        return query, attention.mean(1)


class SourceInventoryHead(nn.Module):
    def __init__(self, hidden_dim=1024, width=128, heads=4, layers=2, max_sources=4, include_event_span=False):
        super().__init__()
        if max_sources != 4 or width % heads:
            raise ValueError('This experiment supports one to four sources and divisible attention width')
        self.width = width
        self.max_sources = max_sources
        self.text_fields = TEXT_FIELDS + ('event',) if include_event_span else TEXT_FIELDS
        self.context_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width))
        self.context_local = nn.Conv1d(width, width, 5, padding=2)
        self.position_projection = nn.Linear(6, width)
        self.slot_queries = nn.Embedding(max_sources + 1, width)
        nn.init.normal_(self.slot_queries.weight, std=.02)
        self.blocks = nn.ModuleList(InventoryBlock(width, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.count = nn.Linear(width, max_sources)
        self.kind = nn.Linear(width, len(KINDS))
        self.characters = nn.Embedding(512, 16, padding_idx=0)
        self.within_token = nn.Embedding(64, 8)
        self.boundaries = nn.Linear(3 * 16 + 8 + 2, width)
        self.key_norm = nn.LayerNorm(width)
        self.start_keys = nn.Linear(width, width)
        self.end_keys = nn.Linear(width, width)
        self.start_queries = nn.Linear(width, len(self.text_fields) * width)
        self.end_queries = nn.Linear(width, len(self.text_fields) * width)
        self.start_bias = nn.Linear(width, len(self.text_fields))
        self.end_bias = nn.Linear(width, len(self.text_fields))

    def forward(self, context, mask, alignment, *, return_attention=False):
        if context.shape[:2] != mask.shape or not bool(mask.any(-1).all()):
            raise ValueError('Invalid raw context or padding mask')
        values = self.context_projection(context.float()).masked_fill(~mask[..., None], 0.)
        # Explicit remasking prevents padding-dependent leakage through the convolution.
        values = values + F.gelu(self.context_local(values.transpose(1, 2)).transpose(1, 2))
        relative = (mask.long().cumsum(-1) - 1).clamp_min(0) / (mask.sum(-1, keepdim=True) - 1).clamp_min(1)
        positions = torch.stack((relative, relative.square(),
            (math.pi * relative).sin(), (math.pi * relative).cos(),
            (2 * math.pi * relative).sin(), (2 * math.pi * relative).cos()), -1)
        values = (values + self.position_projection(positions)).masked_fill(~mask[..., None], 0.)
        query = self.slot_queries.weight[None].expand(len(context), -1, -1)
        for block in self.blocks:
            query, attention = block(query, values, mask)
        query = self.norm(query)
        source_query = query[:, 1:]
        gathered = values.gather(1, alignment.token_indices[..., None].expand(-1, -1, self.width))
        local = torch.cat((self.characters(alignment.characters).flatten(-2),
            self.within_token(alignment.token_char_offsets), alignment.relative_positions), -1)
        keys = self.key_norm(gathered + self.boundaries(local))
        start_q = self.start_queries(source_query).reshape(len(context), self.max_sources, len(self.text_fields), self.width)
        end_q = self.end_queries(source_query).reshape_as(start_q)
        start = torch.einsum('bsfd,bcd->bsfc', start_q, self.start_keys(keys)) / math.sqrt(self.width)
        end = torch.einsum('bsfd,bcd->bsfc', end_q, self.end_keys(keys)) / math.sqrt(self.width)
        start = start + self.start_bias(keys).transpose(1, 2)[:, None]
        end = end + self.end_bias(keys).transpose(1, 2)[:, None]
        valid = alignment.endpoint_mask[:, None, None]
        result = {'count': self.count(query[:, 0]), 'kind': self.kind(source_query),
            'start': start.masked_fill(~valid, -torch.inf), 'end': end.masked_fill(~valid, -torch.inf)}
        return (result, attention[:, 1:]) if return_attention else result


def decode_inventory(logits, requests, endpoint_mask, best_ordered_span):
    """Use learned count and endpoints; no raw-template parsing or reference input."""
    starts, ends = best_ordered_span(logits['start'], logits['end'], endpoint_mask)
    counts = (logits['count'].argmax(-1) + 1).tolist()
    # Absent is a training target on unused slots. On slots admitted by the learned
    # count, choose the highest scoring executable source kind.
    kinds = logits['kind'][..., :3].argmax(-1).tolist()
    starts, ends = starts.tolist(), ends.tolist()
    fields_to_decode = TEXT_FIELDS + ('event',) if logits['start'].shape[2] == 3 else TEXT_FIELDS
    result = []
    for i, request in enumerate(requests):
        sources = []
        for slot in range(counts[i]):
            fields = {}
            for j, field in enumerate(fields_to_decode):
                low, high = starts[i][slot][j], ends[i][slot][j]
                if not 0 <= low <= high < len(request):
                    raise ValueError('Predicted character span outside the raw request')
                fields[field] = {'start': low, 'end': high, 'text': request[low:high + 1]}
            sources.append({'kind': KINDS[kinds[i][slot]], **fields})
        result.append({'count': counts[i], 'sources': sources})
    return result
