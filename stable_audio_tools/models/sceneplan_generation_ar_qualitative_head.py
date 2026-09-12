"""Learn qualitative source controls from raw context and a learned content span.

The inference path receives no request annotations or numerical witness. The
small head predicts categories; deterministic seeded completions map those
predictions to the existing numerical ScenePlan/codec representation.
"""
import hashlib
import math

import torch
from torch import nn

ATTRIBUTES = {
    'motion': ('static', 'linear'),
    'start': ('front', 'front_left', 'left', 'rear_left', 'rear', 'rear_right', 'right', 'front_right'),
    'end': ('front', 'front_left', 'left', 'rear_left', 'rear', 'rear_right', 'right', 'front_right'),
    'onset': ('beginning', 'early', 'middle', 'late', 'ending'),
    'offset': ('beginning', 'early', 'middle', 'late', 'ending'),
    'radial': ('free', 'approaching', 'receding'),
}
CENTERS = (0., 45., 90., 135., 180., -135., -90., -45.)
PHASES = ((0., .2), (.1, .4), (.3, .7), (.6, .9), (.8, 1.))


def annotation_targets(requirements):
    """Training-only requested-category labels, without consulting a GT plan."""
    result = []
    for source in requirements['sources']:
        value = {'radial': 'free'}
        for constraint in source['constraints']:
            op = constraint['op']
            if op == 'motion': value['motion'] = constraint['value']
            elif op == 'compass':
                for point in ('start', 'end') if constraint['point'] == 'both' else (constraint['point'],):
                    value[point] = constraint['value']
            elif op == 'time_phase':
                value[{'onset_sec': 'onset', 'offset_sec': 'offset'}[constraint['field']]] = constraint['value']
            elif op == 'distance_change': value['radial'] = constraint['value']
        if set(value) != set(ATTRIBUTES): raise ValueError('This pilot requires all qualitative source-control fields')
        if value['motion'] == 'static' and (value['start'] != value['end'] or value['radial'] != 'free'):
            raise ValueError('Inconsistent requested static-source controls')
        result.append({name: ATTRIBUTES[name].index(value[name]) for name in ATTRIBUTES})
    return result


class SourceCrossAttention(nn.Module):
    def __init__(self, width, heads, relative_clip):
        super().__init__()
        assert width % heads == 0
        self.heads = heads; self.width = width; self.relative_clip = relative_clip
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.relative_bias = nn.Embedding(2 * relative_clip + 1, heads)
        self.output = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def prepare(self, context):
        batch, tokens, _ = context.shape
        return tuple(layer(context).reshape(batch, tokens, self.heads, -1).transpose(1, 2)
            for layer in (self.key, self.value))

    def forward(self, query, prepared, mask, relative, *, return_attention=False):
        batch, queries, _ = query.shape
        q = self.query(self.norm(query)).reshape(batch, queries, self.heads, -1).transpose(1, 2)
        key, value = prepared
        scores = (q @ key.transpose(-1, -2)) / math.sqrt(self.width // self.heads)
        scores = scores + self.relative_bias(relative).permute(0, 3, 1, 2)
        attention_mask = mask[:, None, None, :] if mask.ndim == 2 else mask[:, None, :, :]
        scores = scores.masked_fill(~attention_mask, -torch.inf)
        attention = scores.softmax(-1)
        attended = (attention @ value).transpose(1, 2).reshape(batch, queries, self.width)
        query = query + self.output(attended)
        query = query + self.ffn(query)
        return (query, attention.mean(1)) if return_attention else query


class QualitativeExecutionHead(nn.Module):
    def __init__(self, hidden_dim=1024, width=128, heads=4, layers=2, relative_clip=96, use_ar_query=True, attribute_queries=False, source_local_attention=False):
        super().__init__()
        self.width = width; self.relative_clip = relative_clip
        self.source_local_attention = bool(source_local_attention)
        self.context_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width))
        self.context_local = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.query_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width))
        self.use_ar_query = bool(use_ar_query)
        if not self.use_ar_query: self.query_projection.requires_grad_(False)
        self.anchor_projection = nn.Linear(3 * width, width)
        self.field_embedding = nn.Embedding(3, width)
        self.blocks = nn.ModuleList(SourceCrossAttention(width, heads, relative_clip) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.outputs = nn.ModuleDict({name: nn.Linear(width, len(values)) for name, values in ATTRIBUTES.items()})
        self.attribute_queries = bool(attribute_queries)
        if self.attribute_queries:
            self.attribute_embedding = nn.Embedding(len(ATTRIBUTES), width)
            nn.init.normal_(self.attribute_embedding.weight, std=.02)

    def prepare_context(self, context, mask):
        values = self.context_projection(context).masked_fill(~mask[..., None], 0.)
        values = values + self.context_local(values.transpose(1, 2)).transpose(1, 2)
        values = values.masked_fill(~mask[..., None], 0.)
        return {'values': values, 'mask': mask, 'blocks': [block.prepare(values) for block in self.blocks]}

    def score_sources(self, hidden, field_types, start_tokens, end_tokens, prepared, *, return_attention=False, source_spans=None):
        values, mask = prepared['values'], prepared['mask']
        if not bool(((start_tokens >= 0) & (start_tokens <= end_tokens) & (end_tokens < values.shape[1])).all()):
            raise ValueError('Invalid learned source-span token endpoints')
        if not bool(mask.gather(1, start_tokens).all() & mask.gather(1, end_tokens).all()):
            raise ValueError('Learned source endpoints include encoder padding')
        positions = torch.arange(values.shape[1], device=values.device)[None, None, :]
        inside = (positions >= start_tokens[..., None]) & (positions <= end_tokens[..., None]) & mask[:, None, :]
        mean = inside.to(values.dtype) @ values / inside.sum(-1, keepdim=True).clamp_min(1)
        first = values.gather(1, start_tokens[..., None].expand(-1, -1, self.width))
        last = values.gather(1, end_tokens[..., None].expand(-1, -1, self.width))
        anchor = self.anchor_projection(torch.cat([first, mean, last], -1))
        query = (self.query_projection(hidden) + anchor + self.field_embedding(field_types)
            if self.use_ar_query else anchor + self.field_embedding(field_types))
        distance = torch.where(positions < start_tokens[..., None], positions - start_tokens[..., None],
            torch.where(positions > end_tokens[..., None], positions - end_tokens[..., None], 0))
        relative = distance.clamp(-self.relative_clip, self.relative_clip) + self.relative_clip
        batch, sources, _ = query.shape; attributes = len(ATTRIBUTES)
        if self.source_local_attention:
            if source_spans is None or source_spans.shape != (*start_tokens.shape, 2):
                raise ValueError('Source-local attention requires predicted event-span token endpoints')
            low, high = source_spans.unbind(-1)
            if not bool(((low >= 0) & (low <= start_tokens) & (end_tokens <= high) & (high < values.shape[1])).all()):
                raise ValueError('Event span must enclose the selected literal anchor')
            mask = mask[:, None, :] & (positions >= low[..., None]) & (positions <= high[..., None])
        elif source_spans is not None:
            raise ValueError('Event spans were provided to a head that does not use source-local attention')
        if self.attribute_queries:
            query = (query[:, :, None, :] + self.attribute_embedding.weight[None, None, :, :]).reshape(batch, sources * attributes, self.width)
            relative = relative[:, :, None, :].expand(-1, -1, attributes, -1).reshape(batch, sources * attributes, -1)
            if self.source_local_attention:
                mask = mask[:, :, None, :].expand(-1, -1, attributes, -1).reshape(batch, sources * attributes, -1)
        elif return_attention:
            raise ValueError('Field attention supervision requires separate attribute queries')
        for index, (block, cache) in enumerate(zip(self.blocks, prepared['blocks'])):
            if return_attention and index == len(self.blocks) - 1:
                query, attention = block(query, cache, mask, relative, return_attention=True)
            else: query = block(query, cache, mask, relative)
        query = self.norm(query)
        if self.attribute_queries:
            query = query.reshape(batch, sources, attributes, self.width)
            result = {name: head(query[:, :, index, :]) for index, (name, head) in enumerate(self.outputs.items())}
        else: result = {name: head(query) for name, head in self.outputs.items()}
        return (result, attention.reshape(batch, sources, attributes, -1)) if return_attention else result

    def forward(self, hidden, field_types, start_tokens, end_tokens, context, mask, *, return_attention=False, source_spans=None):
        return self.score_sources(hidden, field_types, start_tokens, end_tokens, self.prepare_context(context, mask),
            return_attention=return_attention, source_spans=source_spans)


def complete_from_logits(logits, duration_frames, *, seed_key, speech=False):
    """Construct one valid completion using only learned categories and a seed.

    Any other valid completion satisfying the same request categories remains
    acceptable. The seed does not encode or recover a training witness.
    """
    if not isinstance(duration_frames, int) or not 1 <= duration_frames <= 648:
        raise ValueError('Duration must be a valid generated codec frame count')
    if set(logits) != set(ATTRIBUTES): raise ValueError('Missing learned attribute logits')
    for name, values in logits.items():
        if len(values) != len(ATTRIBUTES[name]) or any(not math.isfinite(float(v)) for v in values):
            raise ValueError('Invalid qualitative logits')
    def maximum(values): return max(range(len(values)), key=lambda index: (values[index], -index))
    def uniform(name):
        data = hashlib.sha256((str(seed_key) + '/' + name).encode()).digest()[:8]
        return int.from_bytes(data, 'big') / 2**64
    labels = {name: maximum(values) for name, values in logits.items()}
    static = labels['motion'] == 0
    if static:
        labels['start'] = labels['end'] = maximum([a + b for a, b in zip(logits['start'], logits['end'])])
        labels['radial'] = 0
    grid = []
    for low, high in PHASES:
        grid.append((max(0, math.ceil(low * duration_frames - 1e-8)),
            min(duration_frames, math.floor(high * duration_frames + 1e-8))))
    feasible = []
    for onset, (low_on, high_on) in enumerate(grid):
        high_on = min(high_on, duration_frames - 1)
        for offset, (low_off, high_off) in enumerate(grid):
            low_off = max(1, low_off)
            if low_on <= high_on and low_off <= high_off and low_on < high_off:
                feasible.append((float(logits['onset'][onset]) + float(logits['offset'][offset]), -onset, -offset))
    if not feasible: raise ValueError('No executable learned time-phase pair')
    _, negative_onset, negative_offset = max(feasible)
    labels['onset'], labels['offset'] = -negative_onset, -negative_offset
    low_on, high_on = grid[labels['onset']]; low_off, high_off = grid[labels['offset']]
    gap = min(10, high_off - low_on)  # Prefer at least ten frames where feasible.
    high_on = min(high_on, high_off - gap, duration_frames - 1)
    if speech:
        onset = low_on + int(uniform('onset') * (min(2, high_on - low_on) + 1))
        offset = high_off - int(uniform('offset') * (min(2, high_off - max(low_off, onset + gap)) + 1))
    else:
        onset = low_on + int(uniform('onset') * (high_on - low_on + 1))
        lower = max(1, low_off, onset + gap)
        offset = lower + int(uniform('offset') * (high_off - lower + 1))
    assert 0 <= onset < offset <= duration_frames
    def azimuth(point):
        angle = CENTERS[labels[point]] + 16 * (uniform(point + '_azimuth') - .5)
        return float(round((angle + 180.) % 360. - 180.))
    start_azimuth = azimuth('start'); end_azimuth = start_azimuth if static else azimuth('end')
    distance = 2.25 + uniform('distance'); end_distance = distance
    if not static and labels['radial']:
        delta = .75 + .25 * uniform('radial_delta')
        end_distance = distance + (-delta if labels['radial'] == 1 else delta)
    if not static and end_distance == distance and start_azimuth == end_azimuth:
        # Keep same-sector movement nonzero without crossing the compass cell.
        end_azimuth = float((start_azimuth + 6. + 180.) % 360. - 180.)
    start = {'azimuth_deg': start_azimuth, 'elevation_deg': 0., 'distance_m': distance}
    end = {'azimuth_deg': end_azimuth, 'elevation_deg': 0., 'distance_m': end_distance}
    return {'labels': {name: ATTRIBUTES[name][index] for name, index in labels.items()},
        'onset_frame': onset, 'offset_frame': offset,
        'trajectory': {'type': 'static', 'position': start} if static else {'type': 'linear', 'start': start, 'end': end},
        'completion_policy': 'Seeded valid values inside learned qualitative categories; no witness values.'}
