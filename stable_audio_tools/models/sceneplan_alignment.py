"""Frame-to-text alignment for the ScenePlan P10 semantic attention route.

The 4+4 local conditioner tells the DiT *which* persistent source is active at
each latent frame.  The Qwen caption tells it *what* each persistent source is
and, for formal speech, the exact transcript.  This module connects those two
views only through cross-attention scores.  It never copies semantic features
into the local 4+4 channel block.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _assert_tensor_true(condition: Tensor, message: str) -> None:
    condition = condition.reshape(())
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _inverse_softplus(value: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("a ScenePlan attention-bias scale must be positive")
    return math.log(math.expm1(value))


class ScenePlanFrameTextAlignment(nn.Module):
    """Create a source-local, monotonic additive cross-attention bias.

    Event-description tokens are boosted for every frame where their matching
    4+4 source is active.  Lexical transcript tokens receive a smooth monotonic
    allocation inside the formal speech source's known activity interval.  A
    small duration head predicts the allocation from contextual Qwen tokens;
    forced-alignment targets are used only for an auxiliary training loss and
    are never required at inference.
    """

    def __init__(
        self,
        *,
        token_dim: int,
        max_sources: int = 4,
        duration_hidden_dim: int = 256,
        duration_dropout: float = 0.1,
        event_bias_init: float = 1.0,
        speech_bias_init: float = 2.0,
        speech_sigma_scale: float = 0.5,
        speech_sigma_floor_frames: float = 1.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.max_sources = int(max_sources)
        self.duration_hidden_dim = int(duration_hidden_dim)
        self.duration_dropout = float(duration_dropout)
        self.speech_sigma_scale = float(speech_sigma_scale)
        self.speech_sigma_floor_frames = float(speech_sigma_floor_frames)
        if self.token_dim <= 0 or self.duration_hidden_dim <= 0:
            raise ValueError("ScenePlan alignment dimensions must be positive")
        if self.max_sources != 4:
            raise ValueError("ScenePlan frame/text alignment requires four slots")
        if not 0.0 <= self.duration_dropout < 1.0:
            raise ValueError("duration_dropout must be in [0,1)")
        if (
            not math.isfinite(self.speech_sigma_scale)
            or self.speech_sigma_scale <= 0.0
            or not math.isfinite(self.speech_sigma_floor_frames)
            or self.speech_sigma_floor_frames <= 0.0
        ):
            raise ValueError("speech-bias widths must be finite and positive")

        # Qwen's final token states are already contextual.  A compact
        # per-token head is enough to predict a positive relative allocation;
        # softmax below supplies positivity and a fixed activity budget.
        self.duration_predictor = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.duration_hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.duration_dropout),
            nn.Linear(self.duration_hidden_dim, 1),
        )
        nn.init.zeros_(self.duration_predictor[-1].weight)
        nn.init.zeros_(self.duration_predictor[-1].bias)

        self.event_bias_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(float(event_bias_init)))
        )
        self.speech_bias_raw = nn.Parameter(
            torch.tensor(_inverse_softplus(float(speech_bias_init)))
        )

    @staticmethod
    def _tensor(
        values: Mapping[str, Any],
        key: str,
        *,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if key not in values:
            raise ValueError(f"ScenePlan alignment auxiliary is missing {key!r}")
        value = values[key]
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        return tensor.to(device=device, dtype=dtype, non_blocking=True)

    @staticmethod
    def _masked_fractions(logits: Tensor, mask: Tensor) -> Tensor:
        """Stable masked softmax that returns exact zeros for empty rows."""

        logits32 = logits.to(torch.float32)
        mask = mask.to(torch.bool)
        floor = torch.finfo(logits32.dtype).min
        masked = logits32.masked_fill(~mask, floor)
        row_has_value = mask.any(dim=-1, keepdim=True)
        maximum = torch.where(
            row_has_value,
            masked.max(dim=-1, keepdim=True).values,
            torch.zeros_like(masked[:, :1]),
        )
        numerator = torch.exp(masked - maximum) * mask.to(logits32.dtype)
        denominator = numerator.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(logits32.dtype).tiny
        )
        return torch.where(
            row_has_value,
            numerator / denominator,
            torch.zeros_like(numerator),
        )

    @staticmethod
    def _resize_frame_aux(
        event_ids: Tensor,
        valid: Tensor,
        *,
        frames: int,
    ) -> tuple[Tensor, Tensor]:
        if int(event_ids.shape[-1]) == int(frames):
            return event_ids, valid
        resized_events = F.interpolate(
            event_ids.to(torch.float32), size=int(frames), mode="nearest"
        ).to(event_ids.dtype)
        resized_valid = F.interpolate(
            valid[:, None, :].to(torch.float32),
            size=int(frames),
            mode="nearest",
        )[:, 0, :].to(torch.bool)
        return resized_events, resized_valid

    def _duration_metrics(
        self,
        *,
        predicted_fractions: Tensor,
        lexical_mask: Tensor,
        token_aux: Mapping[str, Any],
    ) -> dict[str, Tensor]:
        zero = predicted_fractions.sum() * 0.0
        if (
            "speech_duration_target_fraction" not in token_aux
            or "speech_duration_target_mask" not in token_aux
        ):
            return {
                "duration_kl": zero,
                "duration_uniform_kl": zero.detach(),
                "duration_center_mae": zero.detach(),
                "duration_teacher_rows": torch.zeros(
                    (), device=predicted_fractions.device, dtype=torch.long
                ),
            }

        target = self._tensor(
            token_aux,
            "speech_duration_target_fraction",
            device=predicted_fractions.device,
            dtype=torch.float32,
        )
        target_mask = self._tensor(
            token_aux,
            "speech_duration_target_mask",
            device=predicted_fractions.device,
            dtype=torch.bool,
        )
        if target.shape != predicted_fractions.shape or target_mask.shape != lexical_mask.shape:
            raise ValueError("speech duration targets do not align with Qwen tokens")
        _assert_tensor_true(
            (target >= 0.0).all() & torch.isfinite(target).all(),
            "speech duration targets must be finite and non-negative",
        )
        _assert_tensor_true(
            ~(target_mask & ~lexical_mask).any(),
            "speech duration supervision may target only lexical transcript tokens",
        )
        teacher_rows = target_mask.any(dim=-1)
        target_sum = (target * target_mask.to(target.dtype)).sum(dim=-1)
        _assert_tensor_true(
            (~teacher_rows | ((target_sum - 1.0).abs() <= 1.0e-5)).all(),
            "speech duration target fractions must sum to one per teacher row",
        )
        # A teacher row must cover every lexical token.  Otherwise the
        # predicted softmax would be normalized over a different support and
        # the duration loss would silently train the wrong distribution.
        _assert_tensor_true(
            (~teacher_rows[:, None] | target_mask.eq(lexical_mask)).all(),
            "speech duration teacher support differs from lexical token support",
        )

        eps = torch.finfo(torch.float32).eps
        pred = predicted_fractions.to(torch.float32).clamp_min(eps)
        safe_target = target.clamp_min(eps)
        per_row_kl = (
            target
            * (safe_target.log() - pred.log())
            * target_mask.to(target.dtype)
        ).sum(dim=-1)

        token_count = target_mask.sum(dim=-1).clamp_min(1).to(torch.float32)
        uniform = target_mask.to(torch.float32) / token_count[:, None]
        per_row_uniform_kl = (
            target
            * (safe_target.log() - uniform.clamp_min(eps).log())
            * target_mask.to(target.dtype)
        ).sum(dim=-1)

        predicted_centers = predicted_fractions.cumsum(dim=-1) - (
            0.5 * predicted_fractions
        )
        target_centers = target.cumsum(dim=-1) - (0.5 * target)
        center_error = (
            (predicted_centers - target_centers).abs()
            * target_mask.to(torch.float32)
        ).sum(dim=-1) / token_count

        count = teacher_rows.sum().clamp_min(1).to(torch.float32)
        duration_kl = (per_row_kl * teacher_rows).sum() / count
        uniform_kl = (per_row_uniform_kl * teacher_rows).sum() / count
        center_mae = (center_error * teacher_rows).sum() / count
        return {
            "duration_kl": duration_kl,
            "duration_uniform_kl": uniform_kl.detach(),
            "duration_center_mae": center_mae.detach(),
            "duration_teacher_rows": teacher_rows.sum().detach(),
        }

    def forward(
        self,
        token_embeddings: Tensor,
        token_attention_mask: Tensor,
        token_aux: Mapping[str, Any],
        frame_aux: Mapping[str, Any],
        *,
        query_frames: int,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if token_embeddings.ndim != 3:
            raise ValueError("ScenePlan alignment expects token embeddings [B,K,D]")
        batch, tokens, dim = token_embeddings.shape
        if int(dim) != self.token_dim:
            raise ValueError(
                f"ScenePlan alignment token dim changed: {dim} != {self.token_dim}"
            )
        if token_attention_mask.shape != (batch, tokens):
            raise ValueError("ScenePlan alignment token mask shape changed")
        if int(query_frames) <= 0:
            raise ValueError("ScenePlan alignment requires positive query frames")
        device = token_embeddings.device
        token_attention_mask = token_attention_mask.to(
            device=device, dtype=torch.bool
        )

        event_source_ids = self._tensor(
            token_aux, "event_source_ids", device=device, dtype=torch.long
        )
        speech_source_ids = self._tensor(
            token_aux, "speech_source_ids", device=device, dtype=torch.long
        )
        lexical_mask = self._tensor(
            token_aux, "speech_lexical_mask", device=device, dtype=torch.bool
        )
        for name, value in (
            ("event_source_ids", event_source_ids),
            ("speech_source_ids", speech_source_ids),
            ("speech_lexical_mask", lexical_mask),
        ):
            if value.shape != (batch, tokens):
                raise ValueError(f"{name} does not align with Qwen tokens")
        _assert_tensor_true(
            (
                (event_source_ids >= -1)
                & (event_source_ids <= self.max_sources)
                & (speech_source_ids >= -1)
                & (speech_source_ids <= self.max_sources)
            ).all(),
            "ScenePlan alignment roles must lie in {-1,0,1,2,3,4}",
        )
        _assert_tensor_true(
            ~((event_source_ids > 0) & (speech_source_ids > 0)).any(),
            "event and speech roles must remain disjoint",
        )
        lexical_mask = lexical_mask & token_attention_mask
        _assert_tensor_true(
            ~(lexical_mask & (speech_source_ids <= 0)).any(),
            "lexical speech tokens require a positive speech source id",
        )

        frame_event_ids = self._tensor(
            frame_aux,
            "source_event_frame_ids",
            device=device,
            dtype=torch.long,
        )
        frame_valid = self._tensor(
            frame_aux, "frame_valid_mask", device=device, dtype=torch.bool
        )
        if frame_event_ids.ndim != 3 or frame_event_ids.shape[:2] != (
            batch,
            self.max_sources,
        ):
            raise ValueError("frame event auxiliary must have shape [B,4,T]")
        if frame_valid.shape != (batch, int(frame_event_ids.shape[-1])):
            raise ValueError("frame-valid auxiliary does not align with event tracks")
        frame_event_ids, frame_valid = self._resize_frame_aux(
            frame_event_ids, frame_valid, frames=int(query_frames)
        )
        _assert_tensor_true(
            (
                (frame_event_ids >= -1)
                & (frame_event_ids <= self.max_sources)
            ).all(),
            "frame event ids must lie in {-1,0,1,2,3,4}",
        )

        # [B,T,4,1] == [B,1,1,K] -> source-local event-token support.
        frame_slots = frame_event_ids.permute(0, 2, 1).unsqueeze(-1)
        event_roles = event_source_ids[:, None, None, :]
        event_match = (
            (frame_slots == event_roles)
            & (frame_slots > 0)
            & (event_roles > 0)
        ).any(dim=2)
        event_match &= frame_valid[:, :, None] & token_attention_mask[:, None, :]

        duration_logits = self.duration_predictor(token_embeddings).squeeze(-1)
        predicted_fractions = self._masked_fractions(duration_logits, lexical_mask)
        duration_metrics = self._duration_metrics(
            predicted_fractions=predicted_fractions,
            lexical_mask=lexical_mask,
            token_aux=token_aux,
        )

        # A ScenePlan allows at most one formal speech source.  Validate that
        # all positive transcript roles within a row agree on its slot.
        positive_speech = speech_source_ids > 0
        speech_min = torch.where(
            positive_speech,
            speech_source_ids,
            torch.full_like(speech_source_ids, self.max_sources + 1),
        ).min(dim=-1).values
        speech_max = torch.where(
            positive_speech, speech_source_ids, torch.zeros_like(speech_source_ids)
        ).max(dim=-1).values
        has_speech_tokens = positive_speech.any(dim=-1)
        _assert_tensor_true(
            (~has_speech_tokens | speech_min.eq(speech_max)).all(),
            "one caption row contains multiple formal speech source ids",
        )
        speech_label = speech_max
        speech_active = (
            frame_event_ids
            == speech_label[:, None, None]
        ).any(dim=1)
        speech_active &= frame_valid & has_speech_tokens[:, None]

        positions = torch.arange(
            int(query_frames), device=device, dtype=torch.long
        )[None, :]
        start = torch.where(
            speech_active,
            positions,
            torch.full_like(positions, int(query_frames)),
        ).min(dim=-1).values
        stop = torch.where(
            speech_active, positions + 1, torch.zeros_like(positions)
        ).max(dim=-1).values
        active_count = speech_active.sum(dim=-1)
        has_speech_interval = active_count > 0
        interval = (
            (positions >= start[:, None]) & (positions < stop[:, None])
        ) & has_speech_interval[:, None]
        _assert_tensor_true(
            (~has_speech_interval[:, None] | interval.eq(speech_active)).all(),
            "formal speech activity must be one contiguous interval",
        )

        fractions = predicted_fractions.to(torch.float32)
        centers_normalized = fractions.cumsum(dim=-1) - (0.5 * fractions)
        count32 = active_count.to(torch.float32).clamp_min(1.0)
        centers = start.to(torch.float32)[:, None] + (
            centers_normalized * count32[:, None]
        )
        widths = fractions * count32[:, None]
        sigma = torch.maximum(
            widths * self.speech_sigma_scale,
            torch.full_like(widths, self.speech_sigma_floor_frames),
        )
        frame_centers = (
            torch.arange(int(query_frames), device=device, dtype=torch.float32)
            + 0.5
        )[None, :, None]
        speech_score = torch.exp(
            -0.5
            * ((frame_centers - centers[:, None, :]) / sigma[:, None, :]).square()
        )
        # Do not mutate ExpBackward's output in place. The score also carries
        # gradients into predicted token durations, and an in-place mask bumps
        # its autograd version before cross-attention backward consumes it.
        speech_score = speech_score * (
            speech_active[:, :, None] & lexical_mask[:, None, :]
        ).to(speech_score.dtype)

        event_strength = F.softplus(self.event_bias_raw.float())
        speech_strength = F.softplus(self.speech_bias_raw.float())
        bias = event_strength * event_match.to(torch.float32)
        bias = bias + speech_strength * speech_score
        bias = bias * token_attention_mask[:, None, :].to(bias.dtype)

        metrics = {
            **duration_metrics,
            "predicted_fractions": predicted_fractions,
            "duration_logits": duration_logits,
            "event_bias_strength": event_strength.detach(),
            "speech_bias_strength": speech_strength.detach(),
            "bias_nonzero_fraction": (
                bias.ne(0).sum().to(torch.float32)
                / max(int(bias.numel()), 1)
            ).detach(),
            "speech_interval_rows": has_speech_interval.sum().detach(),
        }
        # One shared score map is broadcast over attention heads by SDPA.  This
        # keeps the bias memory O(B*T*K), not O(B*H*T*K).
        return bias.unsqueeze(1).to(token_embeddings.dtype), metrics


class ScenePlanSoftBlockAlignment(nn.Module):
    """Build source-aligned event and speech masks without forced alignment.

    The masks contain only zeros and ones.  Per-layer bounded gates live in the
    Transformer blocks, so a newly upcycled model starts with an exact zero
    additive score bias while every caption token remains globally visible.
    """

    def __init__(self, *, max_sources: int = 4) -> None:
        super().__init__()
        self.max_sources = int(max_sources)
        if self.max_sources != 4:
            raise ValueError("ScenePlan soft-block attention requires four slots")

    @staticmethod
    def _tensor(
        values: Mapping[str, Any],
        key: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if key not in values:
            raise ValueError(f"ScenePlan soft-block auxiliary is missing {key!r}")
        value = values[key]
        tensor = value if isinstance(value, Tensor) else torch.as_tensor(value)
        return tensor.to(device=device, dtype=dtype, non_blocking=True)

    def forward(
        self,
        token_embeddings: Tensor,
        token_attention_mask: Tensor,
        token_aux: Mapping[str, Any],
        frame_aux: Mapping[str, Any],
        *,
        query_frames: int,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if token_embeddings.ndim != 3:
            raise ValueError("ScenePlan soft-block expects token embeddings [B,K,D]")
        batch, tokens, _ = token_embeddings.shape
        if token_attention_mask.shape != (batch, tokens):
            raise ValueError("ScenePlan soft-block token mask shape changed")
        if int(query_frames) <= 0:
            raise ValueError("ScenePlan soft-block requires positive query frames")
        device = token_embeddings.device
        token_attention_mask = token_attention_mask.to(
            device=device, dtype=torch.bool
        )

        event_source_ids = self._tensor(
            token_aux,
            "event_source_ids",
            device=device,
            dtype=torch.long,
        )
        speech_source_ids = self._tensor(
            token_aux,
            "speech_source_ids",
            device=device,
            dtype=torch.long,
        )
        for name, value in (
            ("event_source_ids", event_source_ids),
            ("speech_source_ids", speech_source_ids),
        ):
            if value.shape != (batch, tokens):
                raise ValueError(f"{name} does not align with Qwen tokens")
        _assert_tensor_true(
            (
                (event_source_ids >= -1)
                & (event_source_ids <= self.max_sources)
                & (speech_source_ids >= -1)
                & (speech_source_ids <= self.max_sources)
            ).all(),
            "ScenePlan soft-block roles must lie in {-1,0,1,2,3,4}",
        )
        _assert_tensor_true(
            ~((event_source_ids > 0) & (speech_source_ids > 0)).any(),
            "event and speech roles must remain disjoint",
        )

        frame_event_ids = self._tensor(
            frame_aux,
            "source_event_frame_ids",
            device=device,
            dtype=torch.long,
        )
        frame_valid = self._tensor(
            frame_aux,
            "frame_valid_mask",
            device=device,
            dtype=torch.bool,
        )
        if frame_event_ids.ndim != 3 or frame_event_ids.shape[:2] != (
            batch,
            self.max_sources,
        ):
            raise ValueError("frame event auxiliary must have shape [B,4,T]")
        if frame_valid.shape != (batch, int(frame_event_ids.shape[-1])):
            raise ValueError("frame-valid auxiliary does not align with event tracks")
        frame_event_ids, frame_valid = ScenePlanFrameTextAlignment._resize_frame_aux(
            frame_event_ids,
            frame_valid,
            frames=int(query_frames),
        )
        _assert_tensor_true(
            (
                (frame_event_ids >= -1)
                & (frame_event_ids <= self.max_sources)
            ).all(),
            "frame event ids must lie in {-1,0,1,2,3,4}",
        )

        frame_slots = frame_event_ids.permute(0, 2, 1).unsqueeze(-1)
        valid = frame_valid[:, :, None] & token_attention_mask[:, None, :]

        event_roles = event_source_ids[:, None, None, :]
        event_match = (
            (frame_slots == event_roles)
            & (frame_slots > 0)
            & (event_roles > 0)
        ).any(dim=2)
        event_match &= valid

        speech_roles = speech_source_ids[:, None, None, :]
        speech_match = (
            (frame_slots == speech_roles)
            & (frame_slots > 0)
            & (speech_roles > 0)
        ).any(dim=2)
        speech_match &= valid

        metrics = {
            "event_match_fraction": event_match.float().mean().detach(),
            "speech_match_fraction": speech_match.float().mean().detach(),
            "active_frame_fraction": (
                frame_event_ids.gt(0).any(dim=1) & frame_valid
            ).float().mean().detach(),
        }
        # Broadcast one shared source map over all attention heads.
        return (
            event_match[:, None],
            speech_match[:, None],
            metrics,
        )


__all__ = ["ScenePlanFrameTextAlignment", "ScenePlanSoftBlockAlignment"]
