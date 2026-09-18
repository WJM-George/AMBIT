"""Editing Transfusion event-description head (route C).

This is part of the editing AR+DiT Transfusion, not OPSD. The discrete
ScenePlan and ``sceneplan_44`` path stay native. Only event-description
spans inside the existing Qwen caption are replaced. Scene and speech
tokens stay in their original positions. There is no B pooled-and-rebuild
route here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class EventHeadConfig:
    hidden_dim: int = 1024
    condition_dim: int = 768
    head_dim: int = 1024
    adapter_dim: int = 256
    max_events: int = 4
    speech_tokens: bool = True


def plan_readout_positions(codec, ids: Tensor, mask: Tensor, *, max_events=4, allow_speech=False):
    """Return [EOS, source_0_end, ..., source_3_end]; absent slots are -1."""
    if ids.ndim != 2 or ids.shape != mask.shape or ids.shape[0] == 0:
        raise ValueError("plan ids/mask must be nonempty aligned [B,L] tensors")
    mask = mask.bool()
    lengths = mask.sum(-1)
    prefix = torch.arange(ids.shape[1], device=mask.device)[None] < lengths[:, None]
    if not torch.equal(prefix, mask) or bool((lengths < 2).any()):
        raise ValueError("complete plans require contiguous right padding")
    tokens = codec.token_to_id
    begin, end = tokens["<source_begin>"], tokens["<source_end>"]
    slots = {tokens[f"<source_slot_{i}>"]: i for i in range(max_events)}
    result = torch.full((ids.shape[0], max_events + 1), -1, dtype=torch.long)
    for row, (values, length) in enumerate(zip(ids.detach().cpu().tolist(), lengths.tolist())):
        values = values[:length]
        if values[0] != codec.bos_id or values[-1] != codec.eos_id:
            raise ValueError("readout requires the complete plan including BOS/EOS")
        if values.count(codec.bos_id) != 1 or values.count(codec.eos_id) != 1:
            raise ValueError("multiple plans in one row")
        if tokens["<kind_speech>"] in values and not allow_speech:
            raise ValueError("speech plans require the speech-capable event head")
        active = None
        seen = set()
        for position, token in enumerate(values):
            if token == begin:
                if active is not None or position + 1 >= length or values[position + 1] not in slots:
                    raise ValueError("malformed source boundary or missing source slot")
                active = slots[values[position + 1]]
                if active in seen:
                    raise ValueError("duplicate source slot")
                seen.add(active)
            elif token in slots:
                if position == 0 or values[position - 1] != begin:
                    raise ValueError("source slot outside source header")
            elif token == end:
                if active is None:
                    raise ValueError("source_end without source_begin")
                result[row, active + 1] = position
                active = None
        if active is not None or not seen:
            raise ValueError("unfinished or empty source list")
        result[row, 0] = length - 1
    return result.to(ids.device)


def gather_plan_states(hidden: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
    if hidden.ndim != 3 or positions.ndim != 2 or hidden.shape[0] != positions.shape[0]:
        raise ValueError("hidden states and readout positions do not align")
    valid = positions >= 0
    if bool((positions >= hidden.shape[1]).any()):
        raise ValueError("readout position exceeds consumed plan length")
    index = positions.clamp_min(0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    states = hidden.gather(1, index)
    return states.masked_fill(~valid[..., None], 0), valid


def pool_qwen_event_targets(
    embeddings: Tensor,
    attention_mask: Tensor,
    event_source_ids: Tensor,
    speech_source_ids: Tensor,
    *,
    max_events=4,
    allow_speech=False,
) -> tuple[Tensor, Tensor]:
    """Pool the existing 768-d caption for distillation only. Not a generation route."""
    shape = embeddings.shape[:2]
    if embeddings.ndim != 3 or any(
        value.shape != shape
        for value in (attention_mask, event_source_ids, speech_source_ids)
    ):
        raise ValueError("teacher embeddings, token roles and masks must align")
    valid = attention_mask.bool()
    if bool((speech_source_ids[valid] != 0).any()) and not allow_speech:
        raise ValueError("speech/unknown teacher roles are unsupported")
    if bool(((speech_source_ids[valid] < 0) | (speech_source_ids[valid] > max_events)).any()):
        raise ValueError("speech source role outside source slots")
    if bool(((event_source_ids > 0) & (speech_source_ids > 0) & valid).any()):
        raise ValueError("a teacher token cannot belong to both event and speech roles")
    roles = event_source_ids[valid]
    if bool(((roles < 0) | (roles > max_events)).any()):
        raise ValueError("teacher event role outside scene + source slots")
    values = embeddings.detach().float()
    if not bool(torch.isfinite(values[valid]).all()):
        raise ValueError("non-finite teacher features")
    pools, masks = [], []
    for slot in range(max_events + 1):
        selected = valid & (event_source_ids == slot) & (speech_source_ids == 0)
        count = selected.sum(-1)
        pool = values.masked_fill(~selected[..., None], 0).sum(1) / count.clamp_min(1)[:, None]
        pools.append(pool)
        masks.append(count > 0)
    output_mask = torch.stack(masks, dim=1)
    if not bool(output_mask[:, 0].all()) or not bool(output_mask[:, 1:].any(-1).all()):
        raise ValueError("teacher needs scene context and at least one complete event")
    return torch.stack(pools, dim=1), output_mask


def inplace_caption_condition(
    caption,
    caption_mask,
    event_source_ids,
    speech_source_ids,
    event_vectors,
    event_mask,
    *,
    max_events=4,
):
    """Keep A's caption order and length. Only event-description spans change."""
    if caption.ndim != 3 or caption.shape[:2] != caption_mask.shape:
        raise ValueError("caption embeddings and mask must align")
    if any(value.shape != caption_mask.shape for value in (event_source_ids, speech_source_ids)):
        raise ValueError("caption role ids must align with the Qwen sequence")
    if event_vectors.ndim != 3 or event_vectors.shape[:2] != event_mask.shape:
        raise ValueError("event vectors and event mask must align")
    if event_vectors.shape[0] != caption.shape[0] or event_vectors.shape[1] != max_events + 1:
        raise ValueError("event vectors must be [B, scene+sources, D]")
    if event_vectors.shape[-1] != caption.shape[-1]:
        raise ValueError("event vectors must match the caption conditioner width")
    valid = caption_mask.bool()
    event_ids = event_source_ids.to(device=caption.device, dtype=torch.long)
    speech_ids = speech_source_ids.to(device=caption.device, dtype=torch.long)
    if bool(((speech_ids[valid] < 0) | (speech_ids[valid] > max_events)).any()):
        raise ValueError("speech source role outside source slots")
    if bool(((event_ids[valid] < 0) | (event_ids[valid] > max_events)).any()):
        raise ValueError("event source role outside scene + source slots")
    if bool(((event_ids > 0) & (speech_ids > 0) & valid).any()):
        raise ValueError("a caption token cannot be both event description and speech")
    output = caption.to(device=event_vectors.device, dtype=event_vectors.dtype)
    description = valid & (speech_ids == 0) & (event_ids > 0)
    for slot in range(1, max_events + 1):
        selected = description & (event_ids == slot)
        present = selected.any(-1)
        available = event_mask[:, slot].bool()
        if bool((present & ~available).any()) or bool((available & ~present).any()):
            raise ValueError("inplace event spans and event vectors disagree")
        output = torch.where(selected.unsqueeze(-1), event_vectors[:, slot].unsqueeze(1), output)
    return output.masked_fill(~valid.unsqueeze(-1), 0), valid


class EditingEventHead(nn.Module):
    """New Editing parameters: source-end readout and a private prompt adapter."""

    def __init__(self, config: EventHeadConfig = EventHeadConfig()):
        super().__init__()
        if any(value <= 0 for key, value in asdict(config).items() if key != "speech_tokens"):
            raise ValueError("event-head dimensions must be positive")
        self.config = config
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.head_dim),
            nn.SiLU(),
            nn.Linear(config.head_dim, config.condition_dim),
        )
        self.condition_adapter = nn.Sequential(
            nn.LayerNorm(config.condition_dim),
            nn.Linear(config.condition_dim, config.adapter_dim),
            nn.SiLU(),
            nn.Linear(config.adapter_dim, config.condition_dim),
        )
        nn.init.zeros_(self.condition_adapter[-1].weight)
        nn.init.zeros_(self.condition_adapter[-1].bias)

    def reset_condition_adapter(self, seed: int) -> None:
        generator = torch.Generator(device=self.condition_adapter[1].weight.device).manual_seed(seed)
        for module in self.condition_adapter:
            if isinstance(module, nn.LayerNorm):
                module.reset_parameters()
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
                bound = 1 / math.sqrt(module.in_features)
                nn.init.uniform_(module.bias, -bound, bound, generator=generator)
        nn.init.zeros_(self.condition_adapter[-1].weight)
        nn.init.zeros_(self.condition_adapter[-1].bias)

    def forward(self, states: Tensor, mask: Tensor) -> Tensor:
        if states.shape[:2] != mask.shape or states.shape[-1] != self.config.hidden_dim:
            raise ValueError("event state geometry does not match the editing event head")
        states = states.to(self.head[1].weight.dtype)
        return self.head(states).masked_fill(~mask.bool()[..., None], 0)

    def adapt(self, embeddings: Tensor, mask: Tensor) -> Tensor:
        if embeddings.shape[:2] != mask.shape or embeddings.shape[-1] != self.config.condition_dim:
            raise ValueError("condition geometry does not match the editing event head")
        values = embeddings.to(self.condition_adapter[1].weight.dtype)
        return (values + self.condition_adapter(values)).masked_fill(~mask.bool()[..., None], 0)


def distillation_loss(
    prediction: Tensor, target: Tensor, mask: Tensor, *, cosine_weight: float = 1.0
) -> dict[str, Tensor]:
    if prediction.shape != target.shape or prediction.shape[:2] != mask.shape:
        raise ValueError("distillation target geometry mismatch")
    if cosine_weight < 0 or not bool(mask.bool().any()):
        raise ValueError("invalid distillation weight or empty supervision")
    predicted = prediction.float()[mask.bool()]
    teacher = target.detach().float()[mask.bool()]
    mse = F.mse_loss(predicted, teacher)
    cosine = (1 - F.cosine_similarity(predicted, teacher, dim=-1)).mean()
    return {"loss": mse + cosine_weight * cosine, "mse": mse, "cosine": cosine}
