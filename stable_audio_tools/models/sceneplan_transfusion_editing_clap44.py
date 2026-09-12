"""Native 44.1 kHz FOA latent/audio-text contrastive model for Editing AR.

No M2D or pretrained CLAP weights are used. The existing frozen 44.1 kHz
FOA VAE is the audio frontend; this module learns the audio tower and both
language projections. Captions/scene descriptions are supervision only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, distributed as dist, nn
from torch.nn import functional as F

CLAP44_CONTRACT = "editing_clap44_foa_latent_dual_head_v1"
CLAP44_SAMPLE_RATE = 44100
CLAP44_VAE_HOP = 1024
CLAP44_SIDE_INPUT = "own_clap44_768_from_clean_source_foa_latent"


@dataclass(frozen=True)
class CLAP44Config:
    sample_rate: int = CLAP44_SAMPLE_RATE
    vae_hop: int = CLAP44_VAE_HOP
    latent_channels: int = 64
    max_frames: int = 648
    stride: int = 4
    width: int = 384
    layers: int = 6
    heads: int = 6
    semantic_dim: int = 512
    scene_dim: int = 256
    text_dim: int = 1024
    dropout: float = 0.1

    def __post_init__(self):
        if (self.sample_rate, self.vae_hop, self.latent_channels) != (44100, 1024, 64):
            raise ValueError("CLAP44 requires the 44.1 kHz / hop-1024 / 64-channel FOA VAE")
        if self.max_frames != 648 or self.stride < 1:
            raise ValueError("CLAP44 requires complete <=648-frame references")
        if min(self.width, self.layers, self.heads, self.semantic_dim, self.scene_dim, self.text_dim) < 1:
            raise ValueError("CLAP44 dimensions must be positive")
        if self.width % self.heads or self.width % 2 or not 0 <= self.dropout < 1:
            raise ValueError("invalid CLAP44 attention/dropout configuration")


def canonical_source_latent(latent: Tensor, mask: Tensor) -> Tensor:
    """Use the same FP16 source boundary offline and after online VAE encoding."""
    if latent.ndim != 3 or latent.shape[1] != 64 or not 1 <= latent.shape[2] <= 648:
        raise ValueError("CLAP44 source must be [B,64,1<=T<=648]")
    if mask.shape != (latent.shape[0], latent.shape[2]) or mask.dtype != torch.bool:
        raise ValueError("CLAP44 mask must be boolean [B,T]")
    mask = mask.to(latent.device)
    if not bool(mask.any(1).all()) or bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError("CLAP44 needs nonempty right-padded source references")
    value = torch.where(mask[:, None], latent, 0).to(torch.float16).float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("CLAP44 source contains invalid or FP16-overflowing values")
    return value


def canonical_clap44_embedding(semantic: Tensor, scene: Tensor) -> Tensor:
    """Stable fused inference boundary; independent heads retain separate losses."""
    if semantic.ndim != 2 or scene.ndim != 2 or semantic.shape[0] != scene.shape[0]:
        raise ValueError("CLAP44 heads must be two aligned embedding matrices")
    with torch.autocast(device_type="cpu", enabled=False):
        a, b = semantic.detach().float().cpu(), scene.detach().float().cpu()
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("non-finite CLAP44 embedding")
        if bool((a.norm(dim=-1) < 1e-8).any() or (b.norm(dim=-1) < 1e-8).any()):
            raise ValueError("zero CLAP44 embedding")
        return torch.cat((F.normalize(a, dim=-1), F.normalize(b, dim=-1)), -1).div(math.sqrt(2)).half()


class EditingCLAP44(nn.Module):
    """Trainable audio tower with content and time/space contrastive heads."""

    def __init__(self, config: CLAP44Config = CLAP44Config()):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.latent_channels)
        self.input_projection = nn.Linear(config.latent_channels, config.width)
        self.level_projection = nn.Linear(2, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width, config.heads, 4 * config.width, config.dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.audio_transformer = nn.TransformerEncoder(
            layer, config.layers, norm=nn.LayerNorm(config.width), enable_nested_tensor=False,
        )
        # TransformerEncoder clones an initial layer; independently initialize
        # the clones so different layers do not start with identical weights.
        for block in self.audio_transformer.layers:
            for name, parameter in block.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        self.audio_semantic = nn.Linear(config.width, config.semantic_dim, bias=False)
        self.audio_scene = nn.Linear(config.width, config.scene_dim, bias=False)
        self.text_semantic = nn.Sequential(nn.LayerNorm(config.text_dim), nn.Linear(config.text_dim, config.semantic_dim, bias=False))
        self.text_scene = nn.Sequential(nn.LayerNorm(config.text_dim), nn.Linear(config.text_dim, config.scene_dim, bias=False))
        self.logit_scale_semantic = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.logit_scale_scene = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_audio(self, latent: Tensor, mask: Tensor, *, sample_rate: int = 44100, return_sequence: bool = False) -> dict[str, Tensor]:
        if sample_rate != CLAP44_SAMPLE_RATE:
            raise ValueError("CLAP44 does not resample: input provenance must be 44.1 kHz")
        values = canonical_source_latent(latent, mask).transpose(1, 2)
        mask = mask.to(values.device)
        stride = self.config.stride
        pad = (-values.shape[1]) % stride
        values = F.pad(values, (0, 0, 0, pad))
        weights = F.pad(mask, (0, pad)).float().reshape(values.shape[0], -1, stride)
        values = values.reshape(values.shape[0], -1, stride, 64)
        values = (values * weights[..., None]).sum(2) / weights.sum(2).clamp_min(1)[..., None]
        reduced_mask = weights.any(2)
        levels = torch.stack((values.mean(-1), values.square().mean(-1).add(1e-8).sqrt()), -1)
        values = values.to(self.input_projection.weight.dtype)
        levels = levels.to(self.level_projection.weight.dtype)
        hidden = self.input_projection(self.input_norm(values)) + self.level_projection(levels)
        seconds = torch.arange(hidden.shape[1], device=hidden.device).float() * stride * self.config.vae_hop / self.config.sample_rate
        frequency = torch.exp(torch.arange(0, self.config.width, 2, device=hidden.device).float() * (-math.log(10000) / self.config.width))
        phase = seconds[:, None] * frequency[None]
        position = torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)
        hidden = hidden + position.to(hidden.dtype)[None]
        hidden = self.audio_transformer(hidden, src_key_padding_mask=~reduced_mask)
        pooled = (hidden * reduced_mask[..., None]).sum(1) / reduced_mask.sum(1, keepdim=True)
        result = {
            "semantic": F.normalize(self.audio_semantic(pooled).float(), dim=-1),
            "scene": F.normalize(self.audio_scene(pooled).float(), dim=-1),
        }
        if return_sequence:
            result.update(sequence=hidden * reduced_mask[..., None], sequence_mask=reduced_mask)
        return result

    def encode_text_features(self, semantic_features: Tensor, scene_features: Tensor) -> dict[str, Tensor]:
        if semantic_features.ndim != 2 or semantic_features.shape != scene_features.shape or semantic_features.shape[-1] != self.config.text_dim:
            raise ValueError("CLAP44 requires aligned pooled text features")
        if not bool(torch.isfinite(semantic_features).all() and torch.isfinite(scene_features).all()):
            raise ValueError("non-finite CLAP44 text features")
        return {
            "semantic": F.normalize(self.text_semantic(semantic_features.to(self.text_semantic[-1].weight.dtype)).float(), dim=-1),
            "scene": F.normalize(self.text_scene(scene_features.to(self.text_scene[-1].weight.dtype)).float(), dim=-1),
        }

    def forward(self, latent: Tensor, mask: Tensor, semantic_text: Tensor, scene_text: Tensor):
        return self.encode_audio(latent, mask), self.encode_text_features(semantic_text, scene_text)

    @torch.no_grad()
    def source_embedding(self, latent: Tensor, mask: Tensor) -> Tensor:
        if self.training:
            raise RuntimeError("freeze/eval CLAP44 before deriving AR source features")
        output = self.encode_audio(latent, mask)
        return canonical_clap44_embedding(output["semantic"], output["scene"])

    @torch.no_grad()
    def source_features(self, latent: Tensor, mask: Tensor) -> dict[str, Any]:
        if self.training:
            raise RuntimeError("freeze/eval CLAP44 before deriving AR source features")
        output = self.encode_audio(latent, mask, return_sequence=True)
        return {"global": canonical_clap44_embedding(output["semantic"], output["scene"]).to(latent.device), "sequence": output["sequence"].detach().half(), "sequence_mask": output["sequence_mask"], "stride": self.config.stride}

    def configuration(self) -> dict[str, Any]:
        return {"contract": CLAP44_CONTRACT, **asdict(self.config)}


def contrastive_relations(labels: Sequence[Mapping[str, Any]], head: str, device=None, *, include_edit_negatives: bool = True) -> tuple[Tensor, Tensor]:
    """Positive equivalence, audited edit negatives, otherwise exclude shared assets.

    Spatial-only source/target pairs are semantic positives. They are scene
    negatives. Addition/removal pairs with different content are negatives in
    both heads. Arbitrary shared-asset or partial-content overlaps are ignored.
    """
    if head not in {"semantic", "scene"} or not labels:
        raise ValueError("invalid CLAP44 relation request")
    keys = [str(x[f"{head}_key"]) for x in labels]
    if any(not k for k in keys):
        raise ValueError("empty CLAP44 equivalence key")
    assets = [set(x["asset_ids"]) for x in labels]
    contents = [set(x["content_ids"]) for x in labels]
    positives, allowed = [], []
    for i, a in enumerate(labels):
        pos_row, allowed_row = [], []
        for j, b in enumerate(labels):
            same = keys[i] == keys[j]
            paired_edit = include_edit_negatives and (a["pair_id"] == b["pair_id"] and a["role"] != b["role"])
            shared = bool(assets[i] & assets[j]) or bool(contents[i] & contents[j])
            pos_row.append(same); allowed_row.append(same or paired_edit or not shared)
        positives.append(pos_row); allowed.append(allowed_row)
    return torch.tensor(positives, dtype=torch.bool, device=device), torch.tensor(allowed, dtype=torch.bool, device=device)


def symmetric_multi_positive_loss(audio: Tensor, text: Tensor, positives: Tensor, allowed: Tensor, logit_scale: Tensor) -> Tensor:
    if audio.ndim != 2 or text.shape != audio.shape or positives.shape != (audio.shape[0], text.shape[0]) or allowed.shape != positives.shape:
        raise ValueError("invalid contrastive shapes")
    if positives.dtype != torch.bool or allowed.dtype != torch.bool or bool((positives & ~allowed).any()):
        raise ValueError("contrastive masks must include every positive")
    if not bool(positives.any(0).all() and positives.any(1).all()):
        raise ValueError("every audio and text requires at least one positive")
    logits = F.normalize(audio.float(), dim=-1) @ F.normalize(text.float(), dim=-1).T
    logits = logits * logit_scale.float().clamp(math.log(1), math.log(100)).exp()

    def direction(scores, pos, valid):
        log_z = scores.masked_fill(~valid, -torch.inf).logsumexp(-1)
        # Average positive log probabilities, rather than only rewarding the
        # easiest caption of a multi-positive group.
        positive_score = scores.masked_fill(~pos, 0).sum(-1) / pos.sum(-1)
        return (log_z - positive_score).mean()
    return (direction(logits, positives, allowed) + direction(logits.T, positives.T, allowed.T)) / 2


def clap44_objective(model: EditingCLAP44, audio: Mapping[str, Tensor], text: Mapping[str, Tensor], labels: Sequence[Mapping[str, Any]], *, scene_weight: float = 0.25, include_edit_negatives: bool = True) -> dict[str, Tensor]:
    """Global differentiable negatives on every rank; equal rank batch sizes required."""
    if scene_weight <= 0:
        raise ValueError("scene loss weight must be positive")
    labels = list(labels)
    if dist.is_available() and dist.is_initialized():
        from torch.distributed.nn.functional import all_gather
        gathered: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, labels)
        if len({len(x) for x in gathered}) != 1:
            raise ValueError("use drop_last=True for equal CLAP44 DDP batches")
        labels = [x for group in gathered for x in group]
        audio = {k: torch.cat(tuple(all_gather(audio[k])), 0) for k in ("semantic", "scene")}
        text = {k: torch.cat(tuple(all_gather(text[k])), 0) for k in ("semantic", "scene")}
    losses = {}
    for head in ("semantic", "scene"):
        positive, allowed = contrastive_relations(labels, head, audio[head].device, include_edit_negatives=include_edit_negatives)
        losses[head] = symmetric_multi_positive_loss(audio[head], text[head], positive, allowed, getattr(model, f"logit_scale_{head}"))
    losses["loss"] = losses["semantic"] + scene_weight * losses["scene"]
    return losses


def binding_negative_loss(audio_scene: Tensor, positive_scene: Tensor, negative_scene: Tensor, owners: Tensor, *, margin: float = 0.1) -> Tensor:
    """Only the originating audio contrasts with a synthetic binding caption."""
    if owners.numel() == 0:
        return audio_scene.sum() * 0
    owners = owners.to(device=audio_scene.device, dtype=torch.long)
    if negative_scene.shape != (owners.numel(), audio_scene.shape[-1]) or positive_scene.shape != audio_scene.shape or bool((owners < 0).any() or (owners >= audio_scene.shape[0]).any()):
        raise ValueError("invalid anchor-local CLAP44 counterfactuals")
    positive = (audio_scene * positive_scene).sum(-1)[owners]
    negative = (audio_scene[owners] * negative_scene).sum(-1)
    per_negative = F.softplus((negative - positive + margin) * 10) / 10
    sums = audio_scene.new_zeros(audio_scene.shape[0]).index_add(0, owners, per_negative)
    counts = audio_scene.new_zeros(audio_scene.shape[0]).index_add(0, owners, torch.ones_like(per_negative))
    # Include zero-negative anchors with zero loss. This denominator gives
    # equal example weighting across DDP ranks with different source counts.
    return (sums / counts.clamp_min(1)).mean()


class EditingCLAP44SourceBridge(nn.Module):
    """Global and temporal residuals without adding or shifting AR plan tokens."""
    mode = "clap44_audio_caption_aux"
    inject_audio = True
    align_caption = True

    def __init__(self, hidden_dim: int, config: CLAP44Config, *, use_global=True, use_sequence=True, audio_feature_dropout: float = 0.0):
        super().__init__()
        self.use_global, self.use_sequence = bool(use_global), bool(use_sequence)
        if not 0 <= audio_feature_dropout <= 1:
            raise ValueError("CLAP44 feature dropout must be in [0,1]")
        self.audio_feature_dropout = float(audio_feature_dropout)
        self.global_projection = nn.Linear(config.semantic_dim + config.scene_dim, hidden_dim)
        self.sequence_projection = nn.Linear(config.width, hidden_dim)
        self.source_to_caption = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, config.semantic_dim + config.scene_dim, bias=False))
        for module in (self.global_projection, self.sequence_projection):
            nn.init.zeros_(module.weight); nn.init.zeros_(module.bias)
        if not self.use_global:
            self.global_projection.requires_grad_(False)
        if not self.use_sequence:
            self.sequence_projection.requires_grad_(False)

    def inject(self, hidden: Tensor, features: Mapping[str, Any], keep_mask: Tensor | None = None) -> Tensor:
        if not isinstance(features, Mapping):
            raise ValueError("CLAP44 bridge requires source-derived global and sequence features")
        global_value = features["global"].to(hidden)
        sequence = features["sequence"].to(hidden)
        sequence_mask = features["sequence_mask"].to(device=hidden.device, dtype=torch.bool)
        if global_value.shape != (hidden.shape[0], self.global_projection.in_features) or sequence.ndim != 3 or sequence.shape[0] != hidden.shape[0] or sequence.shape[-1] != self.sequence_projection.in_features or sequence_mask.shape != sequence.shape[:2]:
            raise ValueError("CLAP44 feature shapes do not match AR source prefix")
        if not bool(torch.isfinite(global_value).all() and torch.isfinite(sequence).all()):
            raise ValueError("non-finite CLAP44 source features")
        residual = torch.zeros_like(hidden)
        if self.use_global:
            residual = residual + self.global_projection(global_value)[:, None]
        if self.use_sequence:
            stride = int(features["stride"])
            if stride < 1 or sequence.shape[1] * stride < hidden.shape[1]:
                raise ValueError("CLAP44 temporal features cannot cover source prefix")
            local = self.sequence_projection(sequence) * sequence_mask[..., None]
            residual = residual + local.repeat_interleave(stride, dim=1)[:, :hidden.shape[1]]
        if keep_mask is None and self.training and self.audio_feature_dropout:
            keep_mask = torch.rand(hidden.shape[0], device=hidden.device) >= self.audio_feature_dropout
        if keep_mask is not None:
            if keep_mask.shape != hidden.shape[:1]:
                raise ValueError("CLAP44 ablation keep mask must be [B]")
            if not bool(((keep_mask == 0) | (keep_mask == 1)).all()):
                raise ValueError("CLAP44 keep mask must be binary")
            residual = residual * keep_mask.to(hidden)[:, None, None]
        return hidden + residual

    def contrastive_query(self, hidden: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(hidden)[..., None]
        return F.normalize(self.source_to_caption((hidden * weights).sum(1) / weights.sum(1).clamp_min(1)).float(), dim=-1)
