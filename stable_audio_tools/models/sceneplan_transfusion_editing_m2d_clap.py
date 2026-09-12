"""Editing-only M2D-CLAP semantic bridge and contrastive objective.

The clean, frame-aligned FOA latent remains the primary Editing-AR audio
input.  This module adds only a source-derived semantic side channel:

* a frozen M2D-CLAP audio embedding can be broadcast into the existing source
  prefix through a zero-initialized residual projection; and
* a trainable projection of the pre-Transformer source prefix can be aligned
  with a frozen source-caption embedding by multi-positive InfoNCE.

Caption embeddings are training labels.  They are deliberately not accepted
by the AR forward path and are never required at inference.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, distributed as dist, nn
from torch.nn import functional as F


EDITING_M2D_CLAP_CONTRACT = (
    "source_latent_plus_vae_decode_W_frozen_m2d_cpu_fp16_zero_init_caption_aux_v4"
)
EDITING_M2D_CLAP_SIDE_INPUT = (
    "frozen_m2d_clap_audio_768_from_source_latent_via_frozen_vae_decode_W"
)
EDITING_M2D_CLAP_EMBED_DIM = 768
EDITING_M2D_CLAP_MODEL_BOUNDARY_DTYPE = torch.float16
EDITING_M2D_CLAP_MODES = frozenset(
    {
        "latent_only",
        "m2d_audio",
        "caption_aux",
        "m2d_audio_caption_aux",
    }
)


def editing_m2d_mode_flags(mode: str) -> tuple[bool, bool]:
    """Return ``(audio_injection, caption_auxiliary)`` for one A/B arm."""

    normalized = str(mode)
    if normalized not in EDITING_M2D_CLAP_MODES:
        raise ValueError(
            "Editing M2D mode must be one of "
            f"{sorted(EDITING_M2D_CLAP_MODES)}, got {normalized!r}"
        )
    return (
        normalized in {"m2d_audio", "m2d_audio_caption_aux"},
        normalized in {"caption_aux", "m2d_audio_caption_aux"},
    )


def canonicalize_editing_m2d_embedding(value: Tensor) -> Tensor:
    """Apply the one frozen M2D-to-model serialization boundary.

    Offline training caches store normalized float16 vectors. Real-audio
    inference must cross the identical boundary before AR sees the embedding;
    otherwise cache and online paths can disagree after rounding.
    """

    if value.ndim not in {1, 2} or int(value.shape[-1]) != EDITING_M2D_CLAP_EMBED_DIM:
        raise ValueError("Editing M2D embedding must end in 768 dimensions")
    # An outer CPU autocast context would otherwise remain active after the
    # device transfer. This serialization edge must use one exact FP32 CPU
    # reduction for both offline caches and online source-derived embeddings.
    with torch.autocast(device_type="cpu", enabled=False):
        canonical = value.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(canonical).all()):
            raise ValueError("Editing M2D embedding contains non-finite values")
        # This is a serialization boundary, not a differentiable layer. Force
        # its 768-value reduction onto CPU so cache construction and online CUDA
        # inference cannot choose different fp16 rounding at the final L2 norm.
        normalized = F.normalize(canonical, dim=-1)
        quantized = normalized.to(dtype=EDITING_M2D_CLAP_MODEL_BOUNDARY_DTYPE)
        norms = quantized.float().norm(dim=-1)
        if bool((norms < 0.99).any() or (norms > 1.01).any()):
            raise ValueError("Editing M2D embedding cannot cross the fp16 boundary")
        return quantized


class EditingARSourceSemanticBridge(nn.Module):
    """Small trainable bridge around frozen M2D-CLAP representations.

    The audio residual projection is initialized to exactly zero.  Therefore
    enabling the M2D branch starts from the latent-only AR function without
    adding a token or shifting any plan-token position.
    """

    def __init__(
        self,
        *,
        mode: str,
        hidden_dim: int,
        semantic_dim: int = EDITING_M2D_CLAP_EMBED_DIM,
        audio_feature_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mode = str(mode)
        self.inject_audio, self.align_caption = editing_m2d_mode_flags(self.mode)
        self.hidden_dim = int(hidden_dim)
        self.semantic_dim = int(semantic_dim)
        self.audio_feature_dropout = float(audio_feature_dropout)
        if self.hidden_dim <= 0 or self.semantic_dim <= 0:
            raise ValueError("Editing semantic bridge dimensions must be positive")
        if not 0.0 <= self.audio_feature_dropout < 1.0:
            raise ValueError("M2D audio feature dropout must be in [0,1)")

        if self.inject_audio:
            self.audio_norm: nn.Module | None = nn.LayerNorm(
                self.semantic_dim, elementwise_affine=False, eps=1.0e-6
            )
            self.audio_to_hidden: nn.Module | None = nn.Linear(
                self.semantic_dim, self.hidden_dim
            )
            nn.init.zeros_(self.audio_to_hidden.weight)
            nn.init.zeros_(self.audio_to_hidden.bias)
        else:
            self.audio_norm = None
            self.audio_to_hidden = None

        if self.align_caption:
            self.source_pool_norm: nn.Module | None = nn.LayerNorm(self.hidden_dim)
            self.source_to_caption: nn.Module | None = nn.Linear(
                self.hidden_dim, self.semantic_dim, bias=False
            )
            nn.init.normal_(self.source_to_caption.weight, mean=0.0, std=0.02)
        else:
            self.source_pool_norm = None
            self.source_to_caption = None

    def _validate_audio_embedding(
        self, value: Tensor | None, *, batch: int
    ) -> Tensor:
        if value is None:
            raise ValueError(
                "this Editing AR semantic mode requires a source M2D audio embedding"
            )
        if tuple(value.shape) != (int(batch), self.semantic_dim):
            raise ValueError(
                "source M2D audio embedding must be "
                f"[batch,{self.semantic_dim}]"
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError("source M2D audio embedding contains non-finite values")
        norms = value.float().norm(dim=-1)
        if bool((norms < 0.90).any() or (norms > 1.10).any()):
            raise ValueError("source M2D audio embeddings must be L2-normalized")
        return value

    def inject(
        self,
        source_hidden: Tensor,
        source_m2d_audio_embedding: Tensor | None,
        source_m2d_audio_keep_mask: Tensor | None = None,
    ) -> Tensor:
        """Broadcast one semantic residual over all source-prefix frames."""

        if source_hidden.ndim != 3 or int(source_hidden.shape[-1]) != self.hidden_dim:
            raise ValueError("source hidden states have invalid semantic-bridge shape")
        if not self.inject_audio:
            if (
                source_m2d_audio_embedding is not None
                or source_m2d_audio_keep_mask is not None
            ):
                raise ValueError(
                    "latent-only/caption-only ablations must not consume M2D audio"
                )
            return source_hidden
        embedding = self._validate_audio_embedding(
            source_m2d_audio_embedding, batch=int(source_hidden.shape[0])
        ).to(device=source_hidden.device, dtype=source_hidden.dtype)
        assert self.audio_norm is not None and self.audio_to_hidden is not None
        residual = self.audio_to_hidden(self.audio_norm(embedding))
        # Mask the projected residual, not the normalized input.  Masking
        # before LayerNorm/Linear would leave a learned Linear bias behind, so
        # ``m2d_zero`` would not actually recover the latent-only route.  It
        # would also make inverted-dropout scaling ineffective because
        # LayerNorm removes the scale before projection.
        if source_m2d_audio_keep_mask is not None:
            if tuple(source_m2d_audio_keep_mask.shape) != (
                int(embedding.shape[0]),
            ):
                raise ValueError("source M2D keep mask must be [batch]")
            residual = residual * source_m2d_audio_keep_mask.to(
                device=residual.device, dtype=residual.dtype
            )[:, None]
        if self.training and self.audio_feature_dropout > 0.0:
            # Drop complete projected examples, rather than coordinates, so
            # the missing-M2D route is exact and E[residual] matches inference.
            keep = torch.rand(
                (int(residual.shape[0]), 1), device=residual.device
            ).ge(self.audio_feature_dropout)
            residual = (
                residual
                * keep.to(residual.dtype)
                / (1.0 - self.audio_feature_dropout)
            )
        return source_hidden + residual[:, None, :]

    def contrastive_query(
        self, source_hidden: Tensor, source_attention_mask: Tensor
    ) -> Tensor:
        """Project the pre-block source prefix into frozen caption space."""

        if not self.align_caption:
            raise RuntimeError("this Editing semantic mode has no caption auxiliary")
        if (
            source_hidden.ndim != 3
            or int(source_hidden.shape[-1]) != self.hidden_dim
            or tuple(source_attention_mask.shape) != tuple(source_hidden.shape[:2])
        ):
            raise ValueError("source prefix/mask cannot form a contrastive query")
        mask = source_attention_mask.to(
            device=source_hidden.device, dtype=source_hidden.dtype
        )
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (source_hidden * mask[:, :, None]).sum(dim=1) / denominator
        assert self.source_pool_norm is not None
        assert self.source_to_caption is not None
        query = self.source_to_caption(self.source_pool_norm(pooled))
        if not bool(torch.isfinite(query).all()):
            raise RuntimeError("Editing source-caption query is non-finite")
        return F.normalize(query.float(), dim=-1)


def _distributed_all_gather_with_grad(value: Tensor) -> Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    from torch.distributed.nn.functional import all_gather

    return torch.cat(tuple(all_gather(value)), dim=0)


def _distributed_all_gather_no_grad(value: Tensor) -> Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return value
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value.detach())
    return torch.cat(gathered, dim=0)


def multi_positive_source_caption_infonce(
    source_queries: Tensor,
    caption_targets: Tensor,
    caption_group_ids: Tensor,
    source_group_ids: Tensor,
    *,
    temperature: float = 0.07,
    gather_distributed: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Symmetric multi-positive InfoNCE for source audio and source caption.

    Two rows are positives when either their complete source-caption SHA or
    their source-latent SHA is identical.  This prevents duplicate captions
    and duplicate source audio from becoming false negatives.
    """

    if (
        source_queries.ndim != 2
        or caption_targets.ndim != 2
        or tuple(source_queries.shape) != tuple(caption_targets.shape)
        or int(source_queries.shape[1]) != EDITING_M2D_CLAP_EMBED_DIM
    ):
        raise ValueError("source/caption contrastive embeddings must be [B,768]")
    batch = int(source_queries.shape[0])
    if (
        tuple(caption_group_ids.shape) != (batch, 2)
        or tuple(source_group_ids.shape) != (batch, 2)
        or caption_group_ids.dtype != torch.int64
        or source_group_ids.dtype != torch.int64
    ):
        raise ValueError("contrastive group ids must be int64 [B,2] SHA halves")
    tau = float(temperature)
    if not 0.0 < tau <= 1.0:
        raise ValueError("source-caption InfoNCE temperature must be in (0,1]")
    if not bool(torch.isfinite(source_queries).all()) or not bool(
        torch.isfinite(caption_targets).all()
    ):
        raise ValueError("source-caption contrastive inputs must be finite")

    queries = F.normalize(source_queries.float(), dim=-1)
    targets = F.normalize(caption_targets.detach().float(), dim=-1)
    if gather_distributed:
        queries = _distributed_all_gather_with_grad(queries)
        targets = _distributed_all_gather_no_grad(targets)
        caption_group_ids = _distributed_all_gather_no_grad(caption_group_ids)
        source_group_ids = _distributed_all_gather_no_grad(source_group_ids)
    if int(queries.shape[0]) < 2:
        raise ValueError("source-caption InfoNCE needs at least two global rows")

    caption_equal = (
        caption_group_ids[:, None, :] == caption_group_ids[None, :, :]
    ).all(dim=-1)
    source_equal = (
        source_group_ids[:, None, :] == source_group_ids[None, :, :]
    ).all(dim=-1)
    positives = caption_equal | source_equal
    if not bool(positives.any(dim=1).all() and positives.any(dim=0).all()):
        raise RuntimeError("every contrastive row must retain at least one positive")

    cosine = queries @ targets.transpose(0, 1)
    logits = cosine / tau
    negative_infinity = torch.finfo(logits.dtype).min
    positive_logits = logits.masked_fill(~positives, negative_infinity)
    audio_to_text = -(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(logits, dim=1)
    ).mean()
    text_to_audio = -(
        torch.logsumexp(positive_logits.transpose(0, 1), dim=1)
        - torch.logsumexp(logits.transpose(0, 1), dim=1)
    ).mean()
    loss = 0.5 * (audio_to_text + text_to_audio)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("source-caption InfoNCE became non-finite")
    positive_cosines = cosine.masked_select(positives)
    metrics = {
        "audio_to_text": audio_to_text.detach(),
        "text_to_audio": text_to_audio.detach(),
        "positive_cosine": positive_cosines.mean().detach(),
        "audio_top1_positive": positives.gather(
            1, cosine.argmax(dim=1, keepdim=True)
        ).float().mean().detach(),
        "text_top1_positive": positives.transpose(0, 1).gather(
            1, cosine.transpose(0, 1).argmax(dim=1, keepdim=True)
        ).float().mean().detach(),
        "global_rows": torch.tensor(
            float(queries.shape[0]), device=loss.device
        ),
        "mean_positives_per_row": positives.float().sum(dim=1).mean().detach(),
    }
    return loss, metrics


def sha256_group_id(value: str) -> tuple[int, int]:
    """Represent a full SHA256 as two stable, signed-int64-safe chunks."""

    text = str(value).lower()
    if len(text) != 64:
        raise ValueError("contrastive group identity must be one SHA256")
    try:
        first = int(text[:16], 16) & ((1 << 63) - 1)
        second = int(text[48:], 16) & ((1 << 63) - 1)
    except ValueError as error:
        raise ValueError("contrastive group identity is not hexadecimal") from error
    return first, second


__all__ = [
    "EDITING_M2D_CLAP_CONTRACT",
    "EDITING_M2D_CLAP_EMBED_DIM",
    "EDITING_M2D_CLAP_MODES",
    "EDITING_M2D_CLAP_SIDE_INPUT",
    "EditingARSourceSemanticBridge",
    "editing_m2d_mode_flags",
    "multi_positive_source_caption_infonce",
    "sha256_group_id",
]
