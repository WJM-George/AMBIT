"""Prediction heads used by the ScenePlan planner."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from ambit._runtime.sceneplan.schema import MAX_DURATION_SEC, MAX_SOURCES

import hashlib

ATTRIBUTES = {
    "motion": ("static", "linear"),
    "start": (
        "front",
        "front_left",
        "left",
        "rear_left",
        "rear",
        "rear_right",
        "right",
        "front_right",
    ),
    "end": (
        "front",
        "front_left",
        "left",
        "rear_left",
        "rear",
        "rear_right",
        "right",
        "front_right",
    ),
    "onset": ("beginning", "early", "middle", "late", "ending"),
    "offset": ("beginning", "early", "middle", "late", "ending"),
    "radial": ("free", "approaching", "receding"),
}
CENTERS = (0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0)
PHASES = ((0.0, 0.2), (0.1, 0.4), (0.3, 0.7), (0.6, 0.9), (0.8, 1.0))


class SourceCrossAttention(nn.Module):
    def __init__(self, width, heads, relative_clip):
        super().__init__()
        assert width % heads == 0
        self.heads = heads
        self.width = width
        self.relative_clip = relative_clip
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.relative_bias = nn.Embedding(2 * relative_clip + 1, heads)
        self.output = nn.Linear(width, width)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def prepare(self, context):
        (batch, tokens, _) = context.shape
        return tuple(
            (
                layer(context).reshape(batch, tokens, self.heads, -1).transpose(1, 2)
                for layer in (self.key, self.value)
            )
        )

    def forward(self, query, prepared, mask, relative, *, return_attention=False):
        (batch, queries, _) = query.shape
        q = self.query(self.norm(query)).reshape(batch, queries, self.heads, -1).transpose(1, 2)
        (key, value) = prepared
        scores = q @ key.transpose(-1, -2) / math.sqrt(self.width // self.heads)
        scores = scores + self.relative_bias(relative).permute(0, 3, 1, 2)
        attention_mask = mask[:, None, None, :] if mask.ndim == 2 else mask[:, None, :, :]
        scores = scores.masked_fill(~attention_mask, -torch.inf)
        attention = scores.softmax(-1)
        attended = (attention @ value).transpose(1, 2).reshape(batch, queries, self.width)
        query = query + self.output(attended)
        query = query + self.ffn(query)
        return (query, attention.mean(1)) if return_attention else query


class QualitativeExecutionHead(nn.Module):
    def __init__(
        self,
        hidden_dim=1024,
        width=128,
        heads=4,
        layers=2,
        relative_clip=96,
        use_ar_query=True,
        attribute_queries=False,
        source_local_attention=False,
    ):
        super().__init__()
        self.width = width
        self.relative_clip = relative_clip
        self.source_local_attention = bool(source_local_attention)
        self.context_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width)
        )
        self.context_local = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.query_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width)
        )
        self.use_ar_query = bool(use_ar_query)
        if not self.use_ar_query:
            self.query_projection.requires_grad_(False)
        self.anchor_projection = nn.Linear(3 * width, width)
        self.field_embedding = nn.Embedding(3, width)
        self.blocks = nn.ModuleList(
            (SourceCrossAttention(width, heads, relative_clip) for _ in range(layers))
        )
        self.norm = nn.LayerNorm(width)
        self.outputs = nn.ModuleDict(
            {name: nn.Linear(width, len(values)) for (name, values) in ATTRIBUTES.items()}
        )
        self.attribute_queries = bool(attribute_queries)
        if self.attribute_queries:
            self.attribute_embedding = nn.Embedding(len(ATTRIBUTES), width)
            nn.init.normal_(self.attribute_embedding.weight, std=0.02)

    def prepare_context(self, context, mask):
        values = self.context_projection(context).masked_fill(~mask[..., None], 0.0)
        values = values + self.context_local(values.transpose(1, 2)).transpose(1, 2)
        values = values.masked_fill(~mask[..., None], 0.0)
        return {
            "values": values,
            "mask": mask,
            "blocks": [block.prepare(values) for block in self.blocks],
        }

    def score_sources(
        self,
        hidden,
        field_types,
        start_tokens,
        end_tokens,
        prepared,
        *,
        return_attention=False,
        source_spans=None,
    ):
        (values, mask) = (prepared["values"], prepared["mask"])
        if not bool(
            (
                (start_tokens >= 0) & (start_tokens <= end_tokens) & (end_tokens < values.shape[1])
            ).all()
        ):
            raise ValueError("Invalid learned source-span token endpoints")
        if not bool(mask.gather(1, start_tokens).all() & mask.gather(1, end_tokens).all()):
            raise ValueError("Learned source endpoints include encoder padding")
        positions = torch.arange(values.shape[1], device=values.device)[None, None, :]
        inside = (
            (positions >= start_tokens[..., None])
            & (positions <= end_tokens[..., None])
            & mask[:, None, :]
        )
        mean = inside.to(values.dtype) @ values / inside.sum(-1, keepdim=True).clamp_min(1)
        first = values.gather(1, start_tokens[..., None].expand(-1, -1, self.width))
        last = values.gather(1, end_tokens[..., None].expand(-1, -1, self.width))
        anchor = self.anchor_projection(torch.cat([first, mean, last], -1))
        query = (
            self.query_projection(hidden) + anchor + self.field_embedding(field_types)
            if self.use_ar_query
            else anchor + self.field_embedding(field_types)
        )
        distance = torch.where(
            positions < start_tokens[..., None],
            positions - start_tokens[..., None],
            torch.where(positions > end_tokens[..., None], positions - end_tokens[..., None], 0),
        )
        relative = distance.clamp(-self.relative_clip, self.relative_clip) + self.relative_clip
        (batch, sources, _) = query.shape
        attributes = len(ATTRIBUTES)
        if self.source_local_attention:
            if source_spans is None or source_spans.shape != (*start_tokens.shape, 2):
                raise ValueError(
                    "Source-local attention requires predicted event-span token endpoints"
                )
            (low, high) = source_spans.unbind(-1)
            if not bool(
                (
                    (low >= 0)
                    & (low <= start_tokens)
                    & (end_tokens <= high)
                    & (high < values.shape[1])
                ).all()
            ):
                raise ValueError("Event span must enclose the selected literal anchor")
            mask = mask[:, None, :] & (positions >= low[..., None]) & (positions <= high[..., None])
        elif source_spans is not None:
            raise ValueError(
                "Event spans were provided to a head that does not use source-local attention"
            )
        if self.attribute_queries:
            query = (
                query[:, :, None, :] + self.attribute_embedding.weight[None, None, :, :]
            ).reshape(batch, sources * attributes, self.width)
            relative = (
                relative[:, :, None, :]
                .expand(-1, -1, attributes, -1)
                .reshape(batch, sources * attributes, -1)
            )
            if self.source_local_attention:
                mask = (
                    mask[:, :, None, :]
                    .expand(-1, -1, attributes, -1)
                    .reshape(batch, sources * attributes, -1)
                )
        elif return_attention:
            raise ValueError("Field attention supervision requires separate attribute queries")
        for index, (block, cache) in enumerate(zip(self.blocks, prepared["blocks"])):
            if return_attention and index == len(self.blocks) - 1:
                (query, attention) = block(query, cache, mask, relative, return_attention=True)
            else:
                query = block(query, cache, mask, relative)
        query = self.norm(query)
        if self.attribute_queries:
            query = query.reshape(batch, sources, attributes, self.width)
            result = {
                name: head(query[:, :, index, :])
                for (index, (name, head)) in enumerate(self.outputs.items())
            }
        else:
            result = {name: head(query) for (name, head) in self.outputs.items()}
        return (
            (result, attention.reshape(batch, sources, attributes, -1))
            if return_attention
            else result
        )

    def forward(
        self,
        hidden,
        field_types,
        start_tokens,
        end_tokens,
        context,
        mask,
        *,
        return_attention=False,
        source_spans=None,
    ):
        return self.score_sources(
            hidden,
            field_types,
            start_tokens,
            end_tokens,
            self.prepare_context(context, mask),
            return_attention=return_attention,
            source_spans=source_spans,
        )


def complete_from_logits(logits, duration_frames, *, seed_key, speech=False):
    """Construct one valid completion using only learned categories and a seed.

    Any other valid completion satisfying the same request categories remains
    acceptable. The seed does not encode or recover a training witness.
    """
    if not isinstance(duration_frames, int) or not 1 <= duration_frames <= 648:
        raise ValueError("Duration must be a valid generated codec frame count")
    if set(logits) != set(ATTRIBUTES):
        raise ValueError("Missing learned attribute logits")
    for name, values in logits.items():
        if len(values) != len(ATTRIBUTES[name]) or any(
            (not math.isfinite(float(v)) for v in values)
        ):
            raise ValueError("Invalid qualitative logits")

    def maximum(values):
        return max(range(len(values)), key=lambda index: (values[index], -index))

    def uniform(name):
        data = hashlib.sha256((str(seed_key) + "/" + name).encode()).digest()[:8]
        return int.from_bytes(data, "big") / 2**64

    labels = {name: maximum(values) for (name, values) in logits.items()}
    static = labels["motion"] == 0
    if static:
        labels["start"] = labels["end"] = maximum(
            [a + b for (a, b) in zip(logits["start"], logits["end"])]
        )
        labels["radial"] = 0
    grid = []
    for low, high in PHASES:
        grid.append(
            (
                max(0, math.ceil(low * duration_frames - 1e-08)),
                min(duration_frames, math.floor(high * duration_frames + 1e-08)),
            )
        )
    feasible = []
    for onset, (low_on, high_on) in enumerate(grid):
        high_on = min(high_on, duration_frames - 1)
        for offset, (low_off, high_off) in enumerate(grid):
            low_off = max(1, low_off)
            if low_on <= high_on and low_off <= high_off and (low_on < high_off):
                feasible.append(
                    (
                        float(logits["onset"][onset]) + float(logits["offset"][offset]),
                        -onset,
                        -offset,
                    )
                )
    if not feasible:
        raise ValueError("No executable learned time-phase pair")
    (_, negative_onset, negative_offset) = max(feasible)
    (labels["onset"], labels["offset"]) = (-negative_onset, -negative_offset)
    (low_on, high_on) = grid[labels["onset"]]
    (low_off, high_off) = grid[labels["offset"]]
    gap = min(10, high_off - low_on)
    high_on = min(high_on, high_off - gap, duration_frames - 1)
    if speech:
        onset = low_on + int(uniform("onset") * (min(2, high_on - low_on) + 1))
        offset = high_off - int(
            uniform("offset") * (min(2, high_off - max(low_off, onset + gap)) + 1)
        )
    else:
        onset = low_on + int(uniform("onset") * (high_on - low_on + 1))
        lower = max(1, low_off, onset + gap)
        offset = lower + int(uniform("offset") * (high_off - lower + 1))
    assert 0 <= onset < offset <= duration_frames

    def azimuth(point):
        angle = CENTERS[labels[point]] + 16 * (uniform(point + "_azimuth") - 0.5)
        return float(round((angle + 180.0) % 360.0 - 180.0))

    start_azimuth = azimuth("start")
    end_azimuth = start_azimuth if static else azimuth("end")
    distance = 2.25 + uniform("distance")
    end_distance = distance
    if not static and labels["radial"]:
        delta = 0.75 + 0.25 * uniform("radial_delta")
        end_distance = distance + (-delta if labels["radial"] == 1 else delta)
    if not static and end_distance == distance and (start_azimuth == end_azimuth):
        end_azimuth = float((start_azimuth + 6.0 + 180.0) % 360.0 - 180.0)
    start = {"azimuth_deg": start_azimuth, "elevation_deg": 0.0, "distance_m": distance}
    end = {"azimuth_deg": end_azimuth, "elevation_deg": 0.0, "distance_m": end_distance}
    return {
        "labels": {name: ATTRIBUTES[name][index] for (name, index) in labels.items()},
        "onset_frame": onset,
        "offset_frame": offset,
        "trajectory": {"type": "static", "position": start}
        if static
        else {"type": "linear", "start": start, "end": end},
        "completion_policy": "Seeded valid values inside learned qualitative categories; no witness values.",
    }


KINDS = ("sound", "music", "speech", "absent")
TEXT_FIELDS = ("identity", "transcript")


class InventoryBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.width = width
        self.heads = heads
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.cross_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, query, context, mask):
        normalized = self.self_norm(query)
        query = (
            query + self.self_attention(normalized, normalized, normalized, need_weights=False)[0]
        )
        (batch, slots, width) = query.shape
        q = self.query(self.cross_norm(query)).reshape(batch, slots, self.heads, -1).transpose(1, 2)
        k = self.key(context).reshape(batch, -1, self.heads, width // self.heads).transpose(1, 2)
        v = self.value(context).reshape(batch, -1, self.heads, width // self.heads).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)
        attention = scores.masked_fill(~mask[:, None, None], -torch.inf).softmax(-1)
        query = query + self.output((attention @ v).transpose(1, 2).reshape(batch, slots, width))
        query = query + self.ffn(query)
        return (query, attention.mean(1))


class SourceInventoryHead(nn.Module):
    def __init__(
        self, hidden_dim=1024, width=128, heads=4, layers=2, max_sources=4, include_event_span=False
    ):
        super().__init__()
        if max_sources != 4 or width % heads:
            raise ValueError(
                "This experiment supports one to four sources and divisible attention width"
            )
        self.width = width
        self.max_sources = max_sources
        self.text_fields = TEXT_FIELDS + ("event",) if include_event_span else TEXT_FIELDS
        self.context_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, width)
        )
        self.context_local = nn.Conv1d(width, width, 5, padding=2)
        self.position_projection = nn.Linear(6, width)
        self.slot_queries = nn.Embedding(max_sources + 1, width)
        nn.init.normal_(self.slot_queries.weight, std=0.02)
        self.blocks = nn.ModuleList((InventoryBlock(width, heads) for _ in range(layers)))
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
            raise ValueError("Invalid raw context or padding mask")
        values = self.context_projection(context.float()).masked_fill(~mask[..., None], 0.0)
        values = values + F.gelu(self.context_local(values.transpose(1, 2)).transpose(1, 2))
        relative = (mask.long().cumsum(-1) - 1).clamp_min(0) / (
            mask.sum(-1, keepdim=True) - 1
        ).clamp_min(1)
        positions = torch.stack(
            (
                relative,
                relative.square(),
                (math.pi * relative).sin(),
                (math.pi * relative).cos(),
                (2 * math.pi * relative).sin(),
                (2 * math.pi * relative).cos(),
            ),
            -1,
        )
        values = (values + self.position_projection(positions)).masked_fill(~mask[..., None], 0.0)
        query = self.slot_queries.weight[None].expand(len(context), -1, -1)
        for block in self.blocks:
            (query, attention) = block(query, values, mask)
        query = self.norm(query)
        source_query = query[:, 1:]
        gathered = values.gather(1, alignment.token_indices[..., None].expand(-1, -1, self.width))
        local = torch.cat(
            (
                self.characters(alignment.characters).flatten(-2),
                self.within_token(alignment.token_char_offsets),
                alignment.relative_positions,
            ),
            -1,
        )
        keys = self.key_norm(gathered + self.boundaries(local))
        start_q = self.start_queries(source_query).reshape(
            len(context), self.max_sources, len(self.text_fields), self.width
        )
        end_q = self.end_queries(source_query).reshape_as(start_q)
        start = torch.einsum("bsfd,bcd->bsfc", start_q, self.start_keys(keys)) / math.sqrt(
            self.width
        )
        end = torch.einsum("bsfd,bcd->bsfc", end_q, self.end_keys(keys)) / math.sqrt(self.width)
        start = start + self.start_bias(keys).transpose(1, 2)[:, None]
        end = end + self.end_bias(keys).transpose(1, 2)[:, None]
        valid = alignment.endpoint_mask[:, None, None]
        result = {
            "count": self.count(query[:, 0]),
            "kind": self.kind(source_query),
            "start": start.masked_fill(~valid, -torch.inf),
            "end": end.masked_fill(~valid, -torch.inf),
        }
        return (result, attention[:, 1:]) if return_attention else result


def decode_inventory(logits, requests, endpoint_mask, best_ordered_span):
    """Use learned count and endpoints; no raw-template parsing or reference input."""
    (starts, ends) = best_ordered_span(logits["start"], logits["end"], endpoint_mask)
    counts = (logits["count"].argmax(-1) + 1).tolist()
    kinds = logits["kind"][..., :3].argmax(-1).tolist()
    (starts, ends) = (starts.tolist(), ends.tolist())
    fields_to_decode = TEXT_FIELDS + ("event",) if logits["start"].shape[2] == 3 else TEXT_FIELDS
    result = []
    for i, request in enumerate(requests):
        sources = []
        for slot in range(counts[i]):
            fields = {}
            for j, field in enumerate(fields_to_decode):
                (low, high) = (starts[i][slot][j], ends[i][slot][j])
                if not 0 <= low <= high < len(request):
                    raise ValueError("Predicted character span outside the raw request")
                fields[field] = {"start": low, "end": high, "text": request[low : high + 1]}
            sources.append({"kind": KINDS[kinds[i][slot]], **fields})
        result.append({"count": counts[i], "sources": sources})
    return result


MAX_KEYFRAMES = 8


class AudioEventReadout(nn.Module):
    """Learned source queries see only the unpooled CLAP audio sequence."""

    def __init__(self, *, audio_width=384, width=192, heads=6, layers=2, text_dim=1024):
        super().__init__()
        self.input_projection = nn.Linear(audio_width, width)
        self.queries = nn.Parameter(torch.randn(MAX_SOURCES, width) * 0.02)
        layer = nn.TransformerDecoderLayer(
            width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(width))
        sizes = {
            "kind_logits": 4,
            "motion_logits": 3,
            "frame_count_logits": MAX_KEYFRAMES,
            "content": text_dim,
            "activity_sec": 2,
            "gain_db": 1,
            "keyframe_time_sec": MAX_KEYFRAMES,
            "direction_xyz": 3 * MAX_KEYFRAMES,
            "log_distance": MAX_KEYFRAMES,
        }
        self.output_heads = nn.ModuleDict(
            {key: nn.Linear(width, size) for (key, size) in sizes.items()}
        )

    def forward(self, sequence, sequence_mask):
        if (
            sequence.ndim != 3
            or sequence_mask.shape != sequence.shape[:2]
            or sequence_mask.dtype != torch.bool
        ):
            raise ValueError("Event readout requires masked audio sequence features")
        if not bool(sequence_mask.any(-1).all()) or not bool(torch.isfinite(sequence).all()):
            raise ValueError("Missing or invalid audio sequence")
        memory = self.input_projection(sequence)
        hidden = self.decoder(
            self.queries[None].expand(sequence.shape[0], -1, -1),
            memory,
            memory_key_padding_mask=~sequence_mask,
        )
        out = {key: head(hidden).float() for (key, head) in self.output_heads.items()}
        out["activity_sec"] = out["activity_sec"].sigmoid() * MAX_DURATION_SEC
        out["keyframe_time_sec"] = out["keyframe_time_sec"].sigmoid() * MAX_DURATION_SEC
        out["gain_db"] = out["gain_db"].squeeze(-1).sigmoid() * 36 - 24
        out["direction_xyz"] = out["direction_xyz"].reshape(
            sequence.shape[0], MAX_SOURCES, MAX_KEYFRAMES, 3
        )
        return out
