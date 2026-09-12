"""Sketch-first continuous execution/delta thought for P11-v4.

The five slots are exactly ``global + source_0..source_3`` and the fifteen
features are exactly the numeric fields consumed by the deterministic P10
assembler.  No room, kind, text, transcript, gain, or source-count coordinate
exists in this state space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.scene_sketch_v1 import (
    EXECUTION_FEATURE_DIM,
    EXECUTION_SLOT_COUNT,
)
from ..data.sceneplan_p11_single_turn import P11Task


P11_V4_FLOW_ARM = "sketch_first_transfusion_cot_v4"
P11_V4_DIRECT_MSE_ARM = "sketch_first_direct_mse_v4"
# Backward-compatible name for the canonical (non-ablation) arm.
P11_V4_THOUGHT_ARM = P11_V4_FLOW_ARM
P11_V4_THOUGHT_OBJECTIVES = {
    P11_V4_FLOW_ARM: "rectified_flow",
    P11_V4_DIRECT_MSE_ARM: "direct_mse",
}
P11_V4_THOUGHT_CONTRACT = "p10_numeric_scene_delta_thought_v1"


def _masked_mse(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("P11-v4 thought prediction/target/mask must align")
    squared = (prediction.float() - target.float()).square() * mask.float()
    denominator = mask.float().sum(dim=(1, 2)).clamp_min(1.0)
    return squared.sum(dim=(1, 2)) / denominator


@dataclass
class P11V4ThoughtOutput:
    hidden: Tensor
    core: Tensor
    thought_tokens: Tensor
    flow_loss_per_row: Tensor
    solve_loss_per_row: Tensor
    locality_loss_per_row: Tensor
    owner_loss_per_row: Tensor
    inference_noise: Tensor | None = None
    inference_noise_seeds: tuple[int, ...] | None = None
    inference_noise_source: str | None = None


class SketchFirstExecutionReasoner(nn.Module):
    """Rectified-flow thought slots inside the shared causal Qwen backbone."""

    def __init__(self, config: Mapping[str, Any], *, output_dim: int) -> None:
        super().__init__()
        self.arm = str(config.get("arm") or "")
        if self.arm not in P11_V4_THOUGHT_OBJECTIVES:
            raise ValueError(
                "P11-v4 thought.arm must be one of "
                f"{sorted(P11_V4_THOUGHT_OBJECTIVES)}"
            )
        if config.get("contract") != P11_V4_THOUGHT_CONTRACT:
            raise ValueError(
                f"P11-v4 thought.contract must be {P11_V4_THOUGHT_CONTRACT!r}"
            )
        if config.get("backbone") != "shared_qwen_causal_v1":
            raise ValueError("P11-v4 thought requires shared_qwen_causal_v1")
        self.continuous_objective = str(config.get("continuous_objective") or "")
        expected_objective = P11_V4_THOUGHT_OBJECTIVES[self.arm]
        if self.continuous_objective != expected_objective:
            raise ValueError(
                f"P11-v4 arm {self.arm!r} requires {expected_objective!r}"
            )
        self.editing_continuous_objective = str(
            config.get("editing_continuous_objective", self.continuous_objective)
        )
        if self.arm == P11_V4_FLOW_ARM:
            if self.editing_continuous_objective not in {
                "rectified_flow",
                "direct_mse",
            }:
                raise ValueError(
                    "P11-v4 Flow G/U requires Editing to use rectified_flow or "
                    "direct_mse"
                )
        elif self.editing_continuous_objective != "direct_mse":
            raise ValueError("P11-v4 Direct-MSE must also use direct Editing")
        self.editing_uses_direct = (
            self.editing_continuous_objective == "direct_mse"
        )
        self.editing_output_head = str(
            config.get("editing_output_head", "shared_velocity_v1")
        )
        if self.editing_output_head not in {
            "shared_velocity_v1",
            "dedicated_delta_v1",
        }:
            raise ValueError("P11-v4 Editing output head is unsupported")
        if (
            self.editing_output_head == "dedicated_delta_v1"
            and not self.editing_uses_direct
        ):
            raise ValueError(
                "P11-v4 dedicated delta head requires direct-mse Editing"
            )
        if int(config.get("slot_count", -1)) != EXECUTION_SLOT_COUNT:
            raise ValueError(
                f"P11-v4 thought requires {EXECUTION_SLOT_COUNT} execution slots"
            )
        if int(config.get("core_dim", -1)) != EXECUTION_FEATURE_DIM:
            raise ValueError(
                f"P11-v4 thought requires core{EXECUTION_FEATURE_DIM}"
            )
        self.dim = int(config.get("dim", output_dim))
        if self.dim != int(output_dim):
            raise ValueError("P11-v4 slots must use the Qwen hidden width")
        self.inference_steps = int(config.get("inference_steps", 1))
        if not 1 <= self.inference_steps <= 8:
            raise ValueError("P11-v4 inference_steps must be within [1,8]")
        if int(config.get("training_inference_steps", self.inference_steps)) != self.inference_steps:
            raise ValueError("P11-v4 train/inference flow solvers must match")

        if "counterfactual_forcing" in config or "executable_axis" in config:
            raise ValueError(
                "retired endpoint-pair/executable-axis heads are forbidden; "
                "P10 binary direction is owned by DeltaSketch"
            )

        self.query = nn.Parameter(torch.empty(EXECUTION_SLOT_COUNT, self.dim))
        self.role_embedding = nn.Embedding(EXECUTION_SLOT_COUNT, self.dim)
        self.task_embedding = nn.Embedding(len(P11Task), self.dim)
        self.core_projection = nn.Linear(EXECUTION_FEATURE_DIM, self.dim)
        self.previous_projection = nn.Linear(EXECUTION_FEATURE_DIM, self.dim)
        self.time_projection = nn.Sequential(
            nn.Linear(16, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )
        self.token_norm = nn.LayerNorm(self.dim)
        self.velocity_head = nn.Linear(self.dim, EXECUTION_FEATURE_DIM)
        self.delta_head = (
            nn.Linear(self.dim, EXECUTION_FEATURE_DIM)
            if self.editing_output_head == "dedicated_delta_v1"
            else None
        )
        # Retained only so checkpoints written before the inference-aligned
        # DeltaSketch-owner correction remain loadable. It is retired from the
        # forward objective: the autoregressive owner token is the sole source
        # authority at both train and inference time.
        self.owner_head = nn.Linear(self.dim, 1)
        nn.init.normal_(self.query, std=0.02)

        self.inference_noise_seed = int(config.get("inference_noise_seed", 41))
        if not 0 <= self.inference_noise_seed < 2**63:
            raise ValueError("P11-v4 inference_noise_seed must be within [0,2**63)")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.inference_noise_seed)
        self.register_buffer(
            "inference_noise",
            torch.randn(
                EXECUTION_SLOT_COUNT,
                EXECUTION_FEATURE_DIM,
                generator=generator,
            ),
            persistent=False,
        )

    @staticmethod
    def _normalize_noise_seeds(
        noise_seed: int | Sequence[int] | Tensor | None,
        *,
        batch: int,
        default_seed: int,
    ) -> tuple[int, ...]:
        """Return one explicit local-RNG seed per sample.

        A scalar seed is deliberately broadcast instead of offset by row index.
        This makes a sample's draw invariant to batching and row order.  Callers
        requesting distinct posterior draws pass a distinct scalar per decode or
        an explicit sequence for a batched decode.
        """

        if noise_seed is None:
            values = [int(default_seed)] * batch
        elif isinstance(noise_seed, Tensor):
            if noise_seed.ndim == 0:
                values = [int(noise_seed.item())] * batch
            else:
                values = [int(value) for value in noise_seed.detach().cpu().flatten()]
        elif isinstance(noise_seed, Sequence) and not isinstance(
            noise_seed, (str, bytes)
        ):
            values = [int(value) for value in noise_seed]
        else:
            values = [int(noise_seed)] * batch
        if len(values) != batch:
            raise ValueError(
                f"P11-v4 requires one noise seed per sample: got {len(values)} for batch {batch}"
            )
        if any(value < 0 or value >= 2**63 for value in values):
            raise ValueError("P11-v4 noise seeds must be within [0,2**63)")
        return tuple(values)

    def _resolve_inference_noise(
        self,
        reference: Tensor,
        *,
        noise_seed: int | Sequence[int] | Tensor | None,
        noise: Tensor | None,
    ) -> tuple[Tensor, tuple[int, ...] | None, str]:
        """Resolve flow initial state without touching process-global RNG state."""

        batch = int(reference.shape[0])
        if noise is not None and noise_seed is not None:
            raise ValueError("P11-v4 accepts either noise_seed or noise, not both")
        if noise is not None:
            value = torch.as_tensor(noise, device=reference.device, dtype=reference.dtype)
            if value.ndim == 2 and batch == 1:
                value = value.unsqueeze(0)
            if value.shape != reference.shape:
                raise ValueError(
                    f"P11-v4 explicit noise must have shape {tuple(reference.shape)}"
                )
            if not bool(torch.isfinite(value).all()):
                raise ValueError("P11-v4 explicit noise must be finite")
            return value.clone(), None, "explicit_tensor"

        seeds = self._normalize_noise_seeds(
            noise_seed,
            batch=batch,
            default_seed=self.inference_noise_seed,
        )
        rows = []
        for seed in seeds:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            rows.append(
                torch.randn(
                    EXECUTION_SLOT_COUNT,
                    EXECUTION_FEATURE_DIM,
                    generator=generator,
                    dtype=torch.float32,
                    device="cpu",
                )
            )
        value = torch.stack(rows).to(device=reference.device, dtype=reference.dtype)
        source = "configured_seed" if noise_seed is None else "explicit_seed"
        return value, seeds, source

    @staticmethod
    def _time_features(time: Tensor) -> Tensor:
        frequencies = torch.arange(1, 9, device=time.device, dtype=torch.float32)
        phase = time.float().view(-1, 1) * frequencies.view(1, -1) * torch.pi
        return torch.cat([phase.sin(), phase.cos()], dim=-1)

    def _tokens(
        self,
        state: Tensor,
        *,
        time: Tensor,
        task_ids: Tensor,
        previous_core: Tensor,
        editing: Tensor,
    ) -> Tensor:
        expected = (EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM)
        if state.ndim != 3 or tuple(state.shape[1:]) != expected:
            raise ValueError(f"P11-v4 thought state must be [B,{expected[0]},{expected[1]}]")
        if previous_core.shape != state.shape:
            raise ValueError("P11-v4 previous execution core must align")
        batch = state.shape[0]
        roles = torch.arange(EXECUTION_SLOT_COUNT, device=state.device)
        value = self.query.unsqueeze(0).expand(batch, -1, -1)
        value = value + self.role_embedding(roles).unsqueeze(0)
        value = value + self.task_embedding(task_ids).unsqueeze(1)
        value = value + self.core_projection(state.float())
        value = value + self.time_projection(self._time_features(time)).unsqueeze(1)
        value = value + self.previous_projection(previous_core.float()) * editing.view(
            batch, 1, 1
        ).to(value)
        return self.token_norm(value)

    @staticmethod
    def _mask(
        *,
        editing: Tensor,
        target_source_mask: Tensor,
        input_source_mask: Tensor,
    ) -> Tensor:
        batch = editing.shape[0]
        if tuple(target_source_mask.shape) != (batch, 4):
            raise ValueError("P11-v4 target source mask must be [B,4]")
        if tuple(input_source_mask.shape) != (batch, 4):
            raise ValueError("P11-v4 input source mask must be [B,4]")
        source_mask = torch.where(
            editing[:, None],
            target_source_mask | input_source_mask,
            target_source_mask,
        )
        slot_mask = torch.cat(
            [torch.ones((batch, 1), device=editing.device, dtype=torch.bool), source_mask],
            dim=1,
        )
        return slot_mask[:, :, None].expand(
            -1, -1, EXECUTION_FEATURE_DIM
        )

    def _solve(
        self,
        *,
        run_slots: Callable[[Tensor], Tensor],
        task_ids: Tensor,
        previous_core: Tensor,
        editing: Tensor,
        reference: Tensor,
        initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch = reference.shape[0]
        if initial_state is None:
            state = self.inference_noise.to(reference).unsqueeze(0).expand_as(reference).clone()
        else:
            if initial_state.shape != reference.shape:
                raise ValueError("P11-v4 flow initial state must align with reference")
            state = initial_state.to(reference).clone()
        hidden = None
        for step in range(self.inference_steps):
            hidden = run_slots(
                self._tokens(
                    state,
                    time=reference.new_full(
                        (batch,), float(step) / float(self.inference_steps)
                    ),
                    task_ids=task_ids,
                    previous_core=previous_core,
                    editing=editing,
                )
            )
            state = state + self.velocity_head(hidden) / float(self.inference_steps)
        if hidden is None:
            raise RuntimeError("P11-v4 flow solver executed no steps")
        return hidden, state

    def _direct(
        self,
        *,
        run_slots: Callable[[Tensor], Tensor],
        task_ids: Tensor,
        previous_core: Tensor,
        editing: Tensor,
        reference: Tensor,
        output_head: nn.Linear | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Matched connector baseline: one shared-backbone direct regression."""

        batch = reference.shape[0]
        state = torch.zeros_like(reference)
        hidden = run_slots(
            self._tokens(
                state,
                time=reference.new_zeros((batch,)),
                task_ids=task_ids,
                previous_core=previous_core,
                editing=editing,
            )
        )
        head = self.velocity_head if output_head is None else output_head
        return hidden, head(hidden)

    def forward_train(
        self,
        *,
        run_slots: Callable[[Tensor], Tensor],
        task_ids: Tensor,
        target_core: Tensor,
        input_core: Tensor,
        delta_core: Tensor,
        target_source_mask: Tensor,
        input_source_mask: Tensor,
    ) -> P11V4ThoughtOutput:
        expected = (EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM)
        for label, value in (
            ("target", target_core),
            ("input", input_core),
            ("delta", delta_core),
        ):
            if value.ndim != 3 or tuple(value.shape[1:]) != expected:
                raise ValueError(
                    f"P11-v4 {label} core must be [B,{expected[0]},{expected[1]}]"
                )
        editing = task_ids.eq(list(P11Task).index(P11Task.EDITING))
        batch = target_core.shape[0]
        continuous_target = torch.where(
            editing[:, None, None], delta_core.float(), target_core.float()
        )
        mask = self._mask(
            editing=editing,
            target_source_mask=target_source_mask.bool(),
            input_source_mask=input_source_mask.bool(),
        )
        if self.continuous_objective == "rectified_flow":
            noise = torch.randn_like(continuous_target)
            time = torch.rand(batch, device=target_core.device).clamp_(
                1.0e-4, 1.0 - 1.0e-4
            )
            mix = time[:, None, None]
            bridge = (1.0 - mix) * noise + mix * continuous_target
            bridge_hidden = run_slots(
                self._tokens(
                    bridge,
                    time=time,
                    task_ids=task_ids,
                    previous_core=input_core.float(),
                    editing=editing,
                )
            )
            velocity = self.velocity_head(bridge_hidden)
            flow_loss = _masked_mse(velocity, continuous_target - noise, mask)
            hidden, solved = self._solve(
                run_slots=run_slots,
                task_ids=task_ids,
                previous_core=input_core.float(),
                editing=editing,
                reference=continuous_target,
                initial_state=None,
            )
            if self.editing_uses_direct:
                direct_hidden, direct_solved = self._direct(
                    run_slots=run_slots,
                    task_ids=task_ids,
                    previous_core=input_core.float(),
                    editing=editing,
                    reference=continuous_target,
                    output_head=self.delta_head,
                )
                selector = editing[:, None, None]
                hidden = torch.where(selector, direct_hidden, hidden)
                solved = torch.where(selector, direct_solved, solved)
                flow_loss = torch.where(
                    editing, torch.zeros_like(flow_loss), flow_loss
                )
        else:
            hidden, solved = self._direct(
                run_slots=run_slots,
                task_ids=task_ids,
                previous_core=input_core.float(),
                editing=editing,
                reference=continuous_target,
            )
            if self.delta_head is not None and bool(editing.any()):
                solved = torch.where(
                    editing[:, None, None], self.delta_head(hidden), solved
                )
            flow_loss = continuous_target.new_zeros((batch,))
        solve_loss = _masked_mse(solved, continuous_target, mask)
        changed = delta_core.abs().gt(1.0e-8)
        preserved = mask & ~changed
        locality = _masked_mse(
            solved,
            continuous_target,
            torch.where(editing[:, None, None], preserved, torch.zeros_like(preserved)),
        )

        # Compatibility-only field. Owner supervision now lives on the exact
        # DeltaSketch decoder logits consumed by grammar-constrained inference.
        owner_loss = hidden[:, 0, 0].float().mul(0.0)

        thought_tokens = self._tokens(
            solved,
            time=target_core.new_ones((batch,)),
            task_ids=task_ids,
            previous_core=input_core.float(),
            editing=editing,
        )
        return P11V4ThoughtOutput(
            hidden=hidden,
            core=solved,
            thought_tokens=thought_tokens,
            flow_loss_per_row=flow_loss,
            solve_loss_per_row=solve_loss,
            locality_loss_per_row=locality,
            owner_loss_per_row=owner_loss,
        )

    @torch.no_grad()
    def infer(
        self,
        *,
        run_slots: Callable[[Tensor], Tensor],
        task_ids: Tensor,
        previous_core: Tensor | None = None,
        noise_seed: int | Sequence[int] | Tensor | None = None,
        noise: Tensor | None = None,
    ) -> P11V4ThoughtOutput:
        batch = task_ids.shape[0]
        reference = self.query.new_zeros(
            (batch, EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM)
        )
        editing = task_ids.eq(list(P11Task).index(P11Task.EDITING))
        if previous_core is None:
            previous_core = torch.zeros_like(reference)
        if previous_core.shape != reference.shape:
            raise ValueError("P11-v4 inference previous core has the wrong shape")
        inference_noise = None
        inference_noise_seeds = None
        inference_noise_source = None
        if (
            self.continuous_objective == "rectified_flow"
            and self.editing_uses_direct
            and bool(editing.all())
        ):
            if noise_seed is not None or noise is not None:
                raise ValueError(
                    "P11-v4 direct DeltaThought Editing is deterministic and "
                    "does not accept Flow noise"
                )
            hidden, core = self._direct(
                run_slots=run_slots,
                task_ids=task_ids,
                previous_core=previous_core,
                editing=editing,
                reference=reference,
                output_head=self.delta_head,
            )
        elif self.continuous_objective == "rectified_flow":
            inference_noise, inference_noise_seeds, inference_noise_source = (
                self._resolve_inference_noise(
                    reference,
                    noise_seed=noise_seed,
                    noise=noise,
                )
            )
            hidden, core = self._solve(
                run_slots=run_slots,
                task_ids=task_ids,
                previous_core=previous_core,
                editing=editing,
                reference=reference,
                initial_state=inference_noise,
            )
            if self.editing_uses_direct and bool(editing.any()):
                direct_hidden, direct_core = self._direct(
                    run_slots=run_slots,
                    task_ids=task_ids,
                    previous_core=previous_core,
                    editing=editing,
                    reference=reference,
                    output_head=self.delta_head,
                )
                selector = editing[:, None, None]
                hidden = torch.where(selector, direct_hidden, hidden)
                core = torch.where(selector, direct_core, core)
        else:
            if noise_seed is not None or noise is not None:
                raise ValueError(
                    "P11-v4 Direct-MSE has no stochastic flow state; use an ensemble checkpoint"
                )
            hidden, core = self._direct(
                run_slots=run_slots,
                task_ids=task_ids,
                previous_core=previous_core,
                editing=editing,
                reference=reference,
            )
            if self.delta_head is not None and bool(editing.any()):
                core = torch.where(
                    editing[:, None, None], self.delta_head(hidden), core
                )
        thought_tokens = self._tokens(
            core,
            time=reference.new_ones((batch,)),
            task_ids=task_ids,
            previous_core=previous_core,
            editing=editing,
        )
        zero = reference.new_zeros((batch,))
        return P11V4ThoughtOutput(
            hidden=hidden,
            core=core,
            thought_tokens=thought_tokens,
            flow_loss_per_row=zero,
            solve_loss_per_row=zero,
            locality_loss_per_row=zero,
            owner_loss_per_row=zero,
            inference_noise=inference_noise,
            inference_noise_seeds=inference_noise_seeds,
            inference_noise_source=inference_noise_source,
        )


__all__ = [
    "P11_V4_DIRECT_MSE_ARM",
    "P11_V4_FLOW_ARM",
    "P11_V4_THOUGHT_ARM",
    "P11_V4_THOUGHT_CONTRACT",
    "P11_V4_THOUGHT_OBJECTIVES",
    "P11V4ThoughtOutput",
    "SketchFirstExecutionReasoner",
]
