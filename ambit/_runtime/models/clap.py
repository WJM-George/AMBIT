from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping
import torch
from torch import Tensor, nn
from torch.nn import functional as F

CLAP44_CONTRACT = "editing_clap44_foa_latent_dual_head_v1"
CLAP44_SAMPLE_RATE = 44100
CLAP44_VAE_HOP = 1024


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
        if (
            min(
                self.width,
                self.layers,
                self.heads,
                self.semantic_dim,
                self.scene_dim,
                self.text_dim,
            )
            < 1
        ):
            raise ValueError("CLAP44 dimensions must be positive")
        if self.width % self.heads or self.width % 2 or (not 0 <= self.dropout < 1):
            raise ValueError("invalid CLAP44 attention/dropout configuration")


def canonical_source_latent(latent: Tensor, mask: Tensor) -> Tensor:
    """Use the same FP16 source boundary offline and after online VAE encoding."""
    if latent.ndim != 3 or latent.shape[1] != 64 or (not 1 <= latent.shape[2] <= 648):
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
    """Fused content and spatial audio embedding."""
    if semantic.ndim != 2 or scene.ndim != 2 or semantic.shape[0] != scene.shape[0]:
        raise ValueError("CLAP44 heads must be two aligned embedding matrices")
    with torch.autocast(device_type="cpu", enabled=False):
        (a, b) = (semantic.detach().float().cpu(), scene.detach().float().cpu())
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("non-finite CLAP44 embedding")
        if bool((a.norm(dim=-1) < 1e-08).any() or (b.norm(dim=-1) < 1e-08).any()):
            raise ValueError("zero CLAP44 embedding")
        return (
            torch.cat((F.normalize(a, dim=-1), F.normalize(b, dim=-1)), -1).div(math.sqrt(2)).half()
        )


class EditingCLAP44(nn.Module):
    """Audio tower with content and temporal/spatial embedding heads."""

    def __init__(self, config: CLAP44Config = CLAP44Config()):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.latent_channels)
        self.input_projection = nn.Linear(config.latent_channels, config.width)
        self.level_projection = nn.Linear(2, config.width)
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            4 * config.width,
            config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_transformer = nn.TransformerEncoder(
            layer, config.layers, norm=nn.LayerNorm(config.width), enable_nested_tensor=False
        )
        for block in self.audio_transformer.layers:
            for name, parameter in block.named_parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
        self.audio_semantic = nn.Linear(config.width, config.semantic_dim, bias=False)
        self.audio_scene = nn.Linear(config.width, config.scene_dim, bias=False)

    def encode_audio(
        self,
        latent: Tensor,
        mask: Tensor,
        *,
        sample_rate: int = 44100,
        return_sequence: bool = False,
    ) -> dict[str, Tensor]:
        if sample_rate != CLAP44_SAMPLE_RATE:
            raise ValueError("CLAP44 does not resample: input provenance must be 44.1 kHz")
        values = canonical_source_latent(latent, mask).transpose(1, 2)
        mask = mask.to(values.device)
        stride = self.config.stride
        pad = -values.shape[1] % stride
        values = F.pad(values, (0, 0, 0, pad))
        weights = F.pad(mask, (0, pad)).float().reshape(values.shape[0], -1, stride)
        values = values.reshape(values.shape[0], -1, stride, 64)
        values = (values * weights[..., None]).sum(2) / weights.sum(2).clamp_min(1)[..., None]
        reduced_mask = weights.any(2)
        levels = torch.stack((values.mean(-1), values.square().mean(-1).add(1e-08).sqrt()), -1)
        values = values.to(self.input_projection.weight.dtype)
        levels = levels.to(self.level_projection.weight.dtype)
        hidden = self.input_projection(self.input_norm(values)) + self.level_projection(levels)
        seconds = (
            torch.arange(hidden.shape[1], device=hidden.device).float()
            * stride
            * self.config.vae_hop
            / self.config.sample_rate
        )
        frequency = torch.exp(
            torch.arange(0, self.config.width, 2, device=hidden.device).float()
            * (-math.log(10000) / self.config.width)
        )
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

    def encode_text_features(
        self, semantic_features: Tensor, scene_features: Tensor
    ) -> dict[str, Tensor]:
        if (
            semantic_features.ndim != 2
            or semantic_features.shape != scene_features.shape
            or semantic_features.shape[-1] != self.config.text_dim
        ):
            raise ValueError("CLAP44 requires aligned pooled text features")
        if not bool(
            torch.isfinite(semantic_features).all() and torch.isfinite(scene_features).all()
        ):
            raise ValueError("non-finite CLAP44 text features")
        return {
            "semantic": F.normalize(
                self.text_semantic(
                    semantic_features.to(self.text_semantic[-1].weight.dtype)
                ).float(),
                dim=-1,
            ),
            "scene": F.normalize(
                self.text_scene(scene_features.to(self.text_scene[-1].weight.dtype)).float(), dim=-1
            ),
        }

    @torch.no_grad()
    def source_embedding(self, latent: Tensor, mask: Tensor) -> Tensor:
        output = self.encode_audio(latent, mask)
        return canonical_clap44_embedding(output["semantic"], output["scene"])

    @torch.no_grad()
    def source_features(self, latent: Tensor, mask: Tensor) -> dict[str, Any]:
        output = self.encode_audio(latent, mask, return_sequence=True)
        return {
            "global": canonical_clap44_embedding(output["semantic"], output["scene"]).to(
                latent.device
            ),
            "sequence": output["sequence"].detach().half(),
            "sequence_mask": output["sequence_mask"],
            "stride": self.config.stride,
        }

    def configuration(self) -> dict[str, Any]:
        return {"contract": CLAP44_CONTRACT, **asdict(self.config)}


class EditingCLAP44SourceBridge(nn.Module):
    """Global and temporal residuals without adding or shifting AR plan tokens."""

    mode = "clap44_audio_caption_aux"
    inject_audio = True
    align_caption = True

    def __init__(
        self,
        hidden_dim: int,
        config: CLAP44Config,
        *,
        use_global=True,
        use_sequence=True,
        audio_feature_dropout: float = 0.0,
    ):
        super().__init__()
        (self.use_global, self.use_sequence) = (bool(use_global), bool(use_sequence))
        if not 0 <= audio_feature_dropout <= 1:
            raise ValueError("CLAP44 feature dropout must be in [0,1]")
        self.audio_feature_dropout = float(audio_feature_dropout)
        self.global_projection = nn.Linear(config.semantic_dim + config.scene_dim, hidden_dim)
        self.sequence_projection = nn.Linear(config.width, hidden_dim)
        for module in (self.global_projection, self.sequence_projection):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        if not self.use_global:
            self.global_projection.requires_grad_(False)
        if not self.use_sequence:
            self.sequence_projection.requires_grad_(False)

    def inject(
        self, hidden: Tensor, features: Mapping[str, Any], keep_mask: Tensor | None = None
    ) -> Tensor:
        if not isinstance(features, Mapping):
            raise ValueError("CLAP44 bridge requires source-derived global and sequence features")
        global_value = features["global"].to(hidden)
        sequence = features["sequence"].to(hidden)
        sequence_mask = features["sequence_mask"].to(device=hidden.device, dtype=torch.bool)
        if (
            global_value.shape != (hidden.shape[0], self.global_projection.in_features)
            or sequence.ndim != 3
            or sequence.shape[0] != hidden.shape[0]
            or (sequence.shape[-1] != self.sequence_projection.in_features)
            or (sequence_mask.shape != sequence.shape[:2])
        ):
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
            residual = residual + local.repeat_interleave(stride, dim=1)[:, : hidden.shape[1]]
        return hidden + residual
