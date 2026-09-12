"""ScenePlan-aligned chunk routing for the P10 Dense-DiT upgrade.

The proven Dense FFN remains the shared path in ``TransformerBlock``.  This
module owns only zero-initialized delta experts and their router, so adding it
does not rename or perturb any P10-v11 parameter.  Routing is constant inside a
short audio chunk and a chunk is restarted whenever the active ScenePlan source
set changes.

Router prior:
    diffusion timestep + active-source caption summary + activity state

Router evidence:
    the current post-attention audio hidden state

Only the small fixed expert loop remains in Python.  Chunk construction,
pooling, token dispatch and scatter are tensorized; there is no per-chunk or
per-token Python loop and no ``Tensor.tolist()`` host synchronization.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _assert_tensor_true(condition: Tensor, message: str) -> None:
    condition = condition.reshape(())
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _linear_fp32(layer: nn.Linear, values: Tensor) -> Tensor:
    """Run a router projection in FP32 even when the acoustic model is BF16."""

    def project() -> Tensor:
        bias = None if layer.bias is None else layer.bias.float()
        return F.linear(values.float(), layer.weight.float(), bias)

    if values.device.type in {"cpu", "cuda"}:
        # Lightning mixed precision installs an ambient autocast context.  An
        # explicit local disable is required; merely calling ``.float()`` does
        # not stop autocast from selecting a BF16/FP16 GEMM kernel.
        with torch.autocast(device_type=values.device.type, enabled=False):
            return project()
    return project()


def _mlp_fp32(module: nn.Sequential, values: Tensor) -> Tensor:
    return _linear_fp32(module[2], F.silu(_linear_fp32(module[0], values)))


@dataclass
class CPEMoEOutput:
    hidden_states: Tensor
    auxiliary_loss: Tensor
    routing: Dict[str, Tensor]


class GatedFeedForward(nn.Module):
    """The exact Linear-SwiGLU-Linear layout used by the native Dense FFN."""

    def __init__(
        self,
        dim: int,
        *,
        inner_dim: Optional[int] = None,
        expansion_factor: float = 4.0,
        dropout: float = 0.0,
        zero_init_output: bool = False,
    ) -> None:
        super().__init__()
        resolved_inner = (
            int(inner_dim) if inner_dim is not None else int(dim * expansion_factor)
        )
        if dim <= 0 or resolved_inner <= 0:
            raise ValueError("feed-forward dimensions must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.dim = int(dim)
        self.inner_dim = resolved_inner
        self.in_proj = nn.Linear(self.dim, self.inner_dim * 2)
        self.dropout = nn.Dropout(float(dropout))
        self.out_proj = nn.Linear(self.inner_dim, self.dim)
        if zero_init_output:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden_states: Tensor) -> Tensor:
        values, gates = self.in_proj(hidden_states).chunk(2, dim=-1)
        return self.out_proj(self.dropout(values * F.silu(gates)))


class CPEMoEDeltaFeedForward(nn.Module):
    """Vectorized source-boundary-aware chunk router and delta experts."""

    def __init__(
        self,
        dim: int,
        *,
        expert_inner_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_experts: int = 4,
        top_k: int = 2,
        chunk_size: int = 4,
        router_hidden_dim: Optional[int] = None,
        router_temperature: float = 1.0,
        max_sources: int = 4,
        boundary_aware: bool = True,
        conflict_gate_min: float = 0.0,
        conflict_gate_max: float = 1.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_experts < 1 or not 1 <= top_k <= num_experts:
            raise ValueError("top_k must lie in [1, num_experts]")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if max_sources < 1:
            raise ValueError("max_sources must be positive")
        if not 0.0 < float(router_temperature):
            raise ValueError("router_temperature must be positive")
        if not (
            0.0 <= float(conflict_gate_min) < float(conflict_gate_max) <= 1.0
        ):
            raise ValueError(
                "conflict gate bounds must satisfy 0 <= min < max <= 1"
            )

        router_hidden_dim = int(router_hidden_dim or dim)
        expert_inner_dim = int(expert_inner_dim or dim)
        if router_hidden_dim <= 0 or expert_inner_dim <= 0:
            raise ValueError("router and expert dimensions must be positive")

        self.dim = int(dim)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.chunk_size = int(chunk_size)
        self.max_sources = int(max_sources)
        self.boundary_aware = bool(boundary_aware)
        self.router_temperature = float(router_temperature)
        self.conflict_gate_min = float(conflict_gate_min)
        self.conflict_gate_max = float(conflict_gate_max)

        self.experts = nn.ModuleList(
            GatedFeedForward(
                self.dim,
                inner_dim=expert_inner_dim,
                dropout=dropout,
                zero_init_output=True,
            )
            for _ in range(self.num_experts)
        )

        self.activity_embedding = nn.Embedding(self.max_sources + 2, self.dim)
        self.prior_router = nn.Sequential(
            nn.Linear(self.dim * 2, router_hidden_dim),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, self.num_experts),
        )
        self.evidence_router = nn.Sequential(
            nn.Linear(self.dim, router_hidden_dim),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, self.num_experts),
        )
        self.prior_alignment = nn.Linear(self.dim * 2, self.dim)
        self.conflict_norm = nn.LayerNorm(self.dim * 4)
        self.conflict_gate = nn.Sequential(
            nn.Linear(self.dim * 4, router_hidden_dim),
            nn.SiLU(),
            nn.Linear(router_hidden_dim, 1),
        )
        nn.init.zeros_(self.prior_router[-1].bias)
        nn.init.zeros_(self.evidence_router[-1].bias)
        nn.init.zeros_(self.conflict_gate[-1].bias)

    @staticmethod
    def _validate_mask(name: str, mask: Tensor, batch: int, length: int) -> Tensor:
        if tuple(mask.shape) != (batch, length):
            raise ValueError(
                f"{name} must have shape {(batch, length)}, got {tuple(mask.shape)}"
            )
        return mask.to(dtype=torch.bool)

    def _chunk_assignments(
        self,
        audio_mask: Tensor,
        frame_source_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return flattened audio indices and vectorized chunk membership."""

        batch, length = audio_mask.shape
        positions = torch.arange(length, device=audio_mask.device)[None, :]
        previous_audio = F.pad(audio_mask[:, :-1], (1, 0), value=False)
        run_start = audio_mask & ~previous_audio
        if self.boundary_aware and length > 1:
            source_by_frame = frame_source_ids.transpose(1, 2)
            activity_change = F.pad(
                source_by_frame[:, 1:].ne(source_by_frame[:, :-1]).any(dim=-1),
                (1, 0),
                value=False,
            )
            run_start = run_start | (audio_mask & activity_change)

        last_start = torch.where(run_start, positions, 0).cummax(dim=1).values
        offset = positions - last_start
        chunk_start = audio_mask & (
            run_start | offset.remainder(self.chunk_size).eq(0)
        )
        local_chunk = chunk_start.long().cumsum(dim=1) - 1

        flat_audio = audio_mask.flatten().nonzero(as_tuple=False).flatten()
        token_batch = torch.div(flat_audio, length, rounding_mode="floor")
        token_position = flat_audio.remainder(length)
        selected_local = local_chunk.flatten().index_select(0, flat_audio)
        encoded_chunk = token_batch * (length + 1) + selected_local
        unique_chunk, inverse, counts = torch.unique_consecutive(
            encoded_chunk,
            return_inverse=True,
            return_counts=True,
        )
        del unique_chunk
        first = counts.cumsum(dim=0) - counts
        chunk_batch = token_batch.index_select(0, first)
        chunk_start_position = token_position.index_select(0, first)
        return flat_audio, inverse, counts, chunk_batch, chunk_start_position

    def _context_summaries(
        self,
        context: Tensor,
        context_mask: Tensor,
        context_source_ids: Tensor,
        chunk_batch: Tensor,
        frame_state: Tensor,
    ) -> Tensor:
        mask = context_mask.to(dtype=torch.bool)
        weights = mask[..., None].to(context.dtype)
        global_summary = (context * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)

        positive_role = context_source_ids.gt(0) & mask
        source_one_hot = F.one_hot(
            context_source_ids.clamp(0, self.max_sources),
            num_classes=self.max_sources + 1,
        )[..., 1:].to(context.dtype)
        source_one_hot = source_one_hot * positive_role[..., None].to(context.dtype)
        source_sum = torch.einsum("bks,bkd->bsd", source_one_hot, context)
        source_count = source_one_hot.sum(dim=1)
        source_summary = source_sum / source_count[..., None].clamp_min(1.0)

        active_one_hot = F.one_hot(
            frame_state.clamp(0, self.max_sources),
            num_classes=self.max_sources + 1,
        )[..., 1:].to(torch.bool)
        active_sources = active_one_hot.any(dim=1)
        available_sources = source_count.index_select(0, chunk_batch).gt(0)
        usable = active_sources & available_sources
        selected = source_summary.index_select(0, chunk_batch)
        local_sum = (selected * usable[..., None].to(selected.dtype)).sum(dim=1)
        local_count = usable.sum(dim=1, keepdim=True)
        local_summary = local_sum / local_count.clamp_min(1).to(local_sum.dtype)
        fallback = global_summary.index_select(0, chunk_batch)
        summary = torch.where(local_count.gt(0), local_summary, fallback)

        activity_indices = (frame_state + 1).clamp(0, self.max_sources + 1)
        activity_state = F.embedding(
            activity_indices,
            self.activity_embedding.weight,
        ).mean(dim=1)
        return summary + activity_state.to(summary.dtype)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        audio_mask: Tensor,
        time_condition: Tensor,
        context: Tensor,
        context_mask: Tensor,
        context_source_ids: Tensor,
        frame_source_ids: Tensor,
        valid_mask: Optional[Tensor] = None,
    ) -> CPEMoEOutput:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.dim:
            raise ValueError(
                f"hidden_states must be [B,N,{self.dim}], got "
                f"{tuple(hidden_states.shape)}"
            )
        batch, length, _ = hidden_states.shape
        audio_mask = self._validate_mask("audio_mask", audio_mask, batch, length)
        valid_mask = (
            torch.ones_like(audio_mask)
            if valid_mask is None
            else self._validate_mask("valid_mask", valid_mask, batch, length)
        )
        _assert_tensor_true(
            ~(audio_mask & ~valid_mask).any(),
            "audio_mask cannot select padded tokens",
        )
        if tuple(time_condition.shape) != (batch, self.dim):
            raise ValueError(
                f"time_condition must be {(batch, self.dim)}, got "
                f"{tuple(time_condition.shape)}"
            )
        if context.ndim != 3 or context.shape[0] != batch or context.shape[-1] != self.dim:
            raise ValueError(
                f"context must be [B,K,{self.dim}], got {tuple(context.shape)}"
            )
        context_tokens = int(context.shape[1])
        context_mask = self._validate_mask(
            "context_mask", context_mask, batch, context_tokens
        )
        if tuple(context_source_ids.shape) != (batch, context_tokens):
            raise ValueError("context_source_ids must align with context tokens")
        context_source_ids = context_source_ids.to(
            device=hidden_states.device, dtype=torch.long
        )
        _assert_tensor_true(
            (
                (context_source_ids >= -1)
                & (context_source_ids <= self.max_sources)
            ).all(),
            "context source ids are outside {-1,0,1,...,max_sources}",
        )
        if tuple(frame_source_ids.shape) != (batch, self.max_sources, length):
            raise ValueError(
                "frame_source_ids must match [B,max_sources,query_length], got "
                f"{tuple(frame_source_ids.shape)}"
            )
        frame_source_ids = frame_source_ids.to(
            device=hidden_states.device, dtype=torch.long
        )
        _assert_tensor_true(
            (
                (frame_source_ids >= -1)
                & (frame_source_ids <= self.max_sources)
            ).all(),
            "frame source ids are outside {-1,0,1,...,max_sources}",
        )

        flat_audio = audio_mask.flatten().nonzero(as_tuple=False).flatten()
        if flat_audio.numel() == 0:
            zero = hidden_states.sum() * 0.0
            return CPEMoEOutput(
                hidden_states=torch.zeros_like(hidden_states),
                auxiliary_loss=zero,
                routing={
                    "topk_indices": torch.empty(
                        0, self.top_k, dtype=torch.long, device=hidden_states.device
                    ),
                    "topk_weights": hidden_states.new_empty(0, self.top_k),
                    "probabilities": hidden_states.new_empty(0, self.num_experts),
                    "conflict_gate": hidden_states.new_empty(0),
                    "chunk_batch": torch.empty(
                        0, dtype=torch.long, device=hidden_states.device
                    ),
                    "chunk_start": torch.empty(
                        0, dtype=torch.long, device=hidden_states.device
                    ),
                    "chunk_length": torch.empty(
                        0, dtype=torch.long, device=hidden_states.device
                    ),
                    "expert_dispatch": hidden_states.new_zeros(self.num_experts),
                    "raw_load_balance_loss": zero,
                    "router_entropy": zero.detach(),
                    "router_max_probability": zero.detach(),
                    "top1_weight_mean": zero.detach(),
                    "conflict_gate_saturation_fraction": zero.detach(),
                    "conflict_logit_l2": zero,
                    "prior_evidence_top1_agreement": zero.detach(),
                },
            )

        (
            flat_audio,
            inverse,
            chunk_lengths,
            chunk_batch,
            chunk_start,
        ) = self._chunk_assignments(audio_mask, frame_source_ids)
        flat_hidden = hidden_states.reshape(batch * length, self.dim)
        audio_hidden = flat_hidden.index_select(0, flat_audio)
        chunk_count = int(chunk_lengths.shape[0])
        evidence = hidden_states.new_zeros(chunk_count, self.dim).index_add(
            0, inverse, audio_hidden
        )
        evidence = evidence / chunk_lengths[:, None].to(evidence.dtype)

        frame_by_token = frame_source_ids.transpose(1, 2).reshape(
            batch * length, self.max_sources
        )
        audio_frame_state = frame_by_token.index_select(0, flat_audio)
        first = chunk_lengths.cumsum(dim=0) - chunk_lengths
        frame_state = audio_frame_state.index_select(0, first)
        context_summary = self._context_summaries(
            context,
            context_mask,
            context_source_ids,
            chunk_batch,
            frame_state,
        )
        prior_state = torch.cat(
            (time_condition.index_select(0, chunk_batch), context_summary), dim=-1
        )

        prior_logits = _mlp_fp32(self.prior_router, prior_state)
        evidence_logits = _mlp_fp32(self.evidence_router, evidence)
        aligned_prior = _linear_fp32(self.prior_alignment, prior_state)
        conflict_features = torch.cat(
            (
                aligned_prior,
                evidence.float(),
                aligned_prior * evidence.float(),
                (aligned_prior - evidence.float()).abs(),
            ),
            dim=-1,
        )
        conflict_features = F.layer_norm(
            conflict_features,
            self.conflict_norm.normalized_shape,
            self.conflict_norm.weight.float(),
            self.conflict_norm.bias.float(),
            self.conflict_norm.eps,
        )
        conflict_logits = _mlp_fp32(self.conflict_gate, conflict_features)
        raw_conflict_gate = conflict_logits.sigmoid()
        conflict_gate = self.conflict_gate_min + (
            self.conflict_gate_max - self.conflict_gate_min
        ) * raw_conflict_gate
        fused_logits = (
            (1.0 - conflict_gate) * prior_logits
            + conflict_gate * evidence_logits
        )
        probabilities = (fused_logits / self.router_temperature).softmax(dim=-1)
        topk_weights, topk_indices = probabilities.topk(self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-12)

        token_topk_indices = topk_indices.index_select(0, inverse)
        token_topk_weights = topk_weights.index_select(0, inverse)
        dispatch_token_indices = []
        dispatch_values = []
        for expert_index, expert in enumerate(self.experts):
            assignments = (token_topk_indices == expert_index).nonzero(
                as_tuple=False
            )
            token_indices = assignments[:, 0]
            slots = assignments[:, 1]
            # Running an empty tensor through an unselected expert keeps every
            # replicated expert in the autograd graph.  This yields explicit
            # zero gradients instead of DDP-unused parameters without adding
            # fake routing assignments or a capacity/drop policy.
            expert_output = expert(audio_hidden.index_select(0, token_indices))
            weighted = expert_output * token_topk_weights[
                token_indices, slots
            ][:, None].to(expert_output.dtype)
            dispatch_token_indices.append(token_indices)
            # The residual stream intentionally remains FP32 under Lightning's
            # BF16 mixed precision while autocast makes expert GEMMs return
            # BF16. ``index_add`` does not promote its source, so align the
            # sparse contribution with the residual dtype before the one
            # consolidated scatter. The cast is differentiable and also gives
            # the Top-2 accumulation FP32 precision in production.
            dispatch_values.append(weighted.to(audio_hidden.dtype))
        # One scatter avoids materializing a full [audio_tokens,dim] result for
        # every expert while preserving the fixed-expert autograd dependencies.
        sparse_audio = torch.zeros_like(audio_hidden).index_add(
            0,
            torch.cat(dispatch_token_indices, dim=0),
            torch.cat(dispatch_values, dim=0),
        )

        sparse_flat = torch.zeros_like(flat_hidden).index_copy(
            0, flat_audio, sparse_audio
        )
        sparse_output = sparse_flat.reshape_as(hidden_states)
        sparse_output = sparse_output * valid_mask[..., None].to(sparse_output.dtype)

        dispatch = F.one_hot(
            topk_indices, num_classes=self.num_experts
        ).float().sum(dim=1) / float(self.top_k)
        raw_balance = self.num_experts * (
            probabilities.mean(dim=0) * dispatch.mean(dim=0)
        ).sum()
        entropy = -(
            probabilities * probabilities.clamp_min(1.0e-12).log()
        ).sum(dim=-1).mean()
        conflict_logit_l2 = conflict_logits.square().mean()
        return CPEMoEOutput(
            hidden_states=sparse_output,
            auxiliary_loss=raw_balance,
            routing={
                "topk_indices": topk_indices,
                "topk_weights": topk_weights,
                "probabilities": probabilities,
                "conflict_gate": conflict_gate.squeeze(-1),
                "chunk_batch": chunk_batch,
                "chunk_start": chunk_start,
                "chunk_length": chunk_lengths,
                "chunk_source_ids": frame_state,
                "expert_dispatch": dispatch.mean(dim=0),
                "raw_load_balance_loss": raw_balance,
                "router_entropy": entropy,
                "router_max_probability": probabilities.max(dim=-1).values.mean(),
                "top1_weight_mean": topk_weights[:, 0].mean(),
                "conflict_gate_saturation_fraction": (
                    (raw_conflict_gate < 0.05) | (raw_conflict_gate > 0.95)
                ).float().mean(),
                "conflict_logit_l2": conflict_logit_l2,
                "prior_evidence_top1_agreement": (
                    prior_logits.argmax(dim=-1) == evidence_logits.argmax(dim=-1)
                ).float().mean(),
            },
        )


class CPEMoEFeedForward(nn.Module):
    """Standalone shared+deltas wrapper used by probes and upcycle tests.

    The production Transformer integration retains its native ``FeedForward``
    object as the shared path and instantiates only ``CPEMoEDeltaFeedForward``.
    """

    def __init__(
        self,
        dim: int,
        *,
        expansion_factor: float = 4.0,
        shared_inner_dim: Optional[int] = None,
        expert_inner_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_experts: int = 4,
        top_k: int = 2,
        chunk_size: int = 4,
        router_hidden_dim: Optional[int] = None,
        router_temperature: float = 1.0,
        max_sources: int = 4,
        boundary_aware: bool = True,
        load_balance_weight: float = 0.01,
        router_entropy_loss_weight: float = 0.0,
        router_entropy_target: Optional[float] = None,
        conflict_logit_l2_loss_weight: float = 0.0,
        conflict_gate_min: float = 0.0,
        conflict_gate_max: float = 1.0,
        shared_audio_scale: float = 1.0,
        sparse_audio_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if load_balance_weight < 0.0:
            raise ValueError("load_balance_weight cannot be negative")
        if router_entropy_loss_weight < 0.0:
            raise ValueError("router_entropy_loss_weight cannot be negative")
        if router_entropy_target is not None and not (
            0.0 < float(router_entropy_target) < math.log(float(num_experts))
        ):
            raise ValueError(
                "router_entropy_target must lie strictly between zero and "
                "log(num_experts)"
            )
        if conflict_logit_l2_loss_weight < 0.0:
            raise ValueError("conflict_logit_l2_loss_weight cannot be negative")
        if shared_audio_scale < 0.0 or sparse_audio_scale < 0.0:
            raise ValueError("composition scales cannot be negative")
        self.dim = int(dim)
        self.load_balance_weight = float(load_balance_weight)
        self.router_entropy_loss_weight = float(router_entropy_loss_weight)
        self.router_entropy_target = (
            None
            if router_entropy_target is None
            else float(router_entropy_target)
        )
        self.conflict_logit_l2_loss_weight = float(
            conflict_logit_l2_loss_weight
        )
        self.shared_audio_scale = float(shared_audio_scale)
        self.sparse_audio_scale = float(sparse_audio_scale)
        self.shared = GatedFeedForward(
            self.dim,
            inner_dim=shared_inner_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )
        self.delta = CPEMoEDeltaFeedForward(
            self.dim,
            expert_inner_dim=expert_inner_dim,
            dropout=dropout,
            num_experts=num_experts,
            top_k=top_k,
            chunk_size=chunk_size,
            router_hidden_dim=router_hidden_dim,
            router_temperature=router_temperature,
            max_sources=max_sources,
            boundary_aware=boundary_aware,
            conflict_gate_min=conflict_gate_min,
            conflict_gate_max=conflict_gate_max,
        )

    @property
    def experts(self) -> nn.ModuleList:
        return self.delta.experts

    @property
    def prior_router(self) -> nn.Sequential:
        return self.delta.prior_router

    @torch.no_grad()
    def upcycle_from_dense(
        self, input_projection: nn.Linear, output_projection: nn.Linear
    ) -> None:
        if input_projection.weight.shape != self.shared.in_proj.weight.shape:
            raise ValueError("dense input projection shape is incompatible")
        if output_projection.weight.shape != self.shared.out_proj.weight.shape:
            raise ValueError("dense output projection shape is incompatible")
        self.shared.in_proj.load_state_dict(input_projection.state_dict())
        self.shared.out_proj.load_state_dict(output_projection.state_dict())

    def forward(
        self,
        hidden_states: Tensor,
        *,
        audio_mask: Tensor,
        time_condition: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        context_source_ids: Optional[Tensor] = None,
        frame_source_ids: Optional[Tensor] = None,
        valid_mask: Optional[Tensor] = None,
    ) -> CPEMoEOutput:
        batch, length, _ = hidden_states.shape
        if context is None:
            context = hidden_states
            context_mask = (
                (~audio_mask)
                if context_mask is None
                else context_mask.to(dtype=torch.bool)
            )
        if context_mask is None:
            context_mask = torch.ones(
                context.shape[:2], device=hidden_states.device, dtype=torch.bool
            )
        if context_source_ids is None:
            context_source_ids = torch.zeros(
                context.shape[:2], device=hidden_states.device, dtype=torch.long
            )
        if frame_source_ids is None:
            frame_source_ids = torch.zeros(
                batch,
                self.delta.max_sources,
                length,
                device=hidden_states.device,
                dtype=torch.long,
            )
        routed = self.delta(
            hidden_states,
            audio_mask=audio_mask,
            time_condition=time_condition,
            context=context,
            context_mask=context_mask,
            context_source_ids=context_source_ids,
            frame_source_ids=frame_source_ids,
            valid_mask=valid_mask,
        )
        router_entropy_loss = routed.routing["router_entropy"]
        if self.router_entropy_target is not None:
            router_entropy_loss = (
                router_entropy_loss - self.router_entropy_target
            ).square()
        shared = self.shared(hidden_states)
        audio_output = (
            self.shared_audio_scale * shared
            + self.sparse_audio_scale * routed.hidden_states
        )
        output = torch.where(audio_mask[..., None], audio_output, shared)
        if valid_mask is not None:
            output = output * valid_mask[..., None].to(output.dtype)
        return CPEMoEOutput(
            hidden_states=output,
            auxiliary_loss=(
                routed.auxiliary_loss * self.load_balance_weight
                + router_entropy_loss * self.router_entropy_loss_weight
                + routed.routing["conflict_logit_l2"]
                * self.conflict_logit_l2_loss_weight
            ),
            routing=routed.routing,
        )


__all__ = [
    "CPEMoEDeltaFeedForward",
    "CPEMoEFeedForward",
    "CPEMoEOutput",
    "GatedFeedForward",
]
