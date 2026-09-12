"""Frozen dense-DiT acoustic teacher for staged Transfusion training.

The dense DiT and Transfusion use opposite rectified-flow conventions:

``DiT``: ``x_t = (1-t) * data + t * noise``, target ``noise - data``.
``Transfusion``: ``x_t = t * data + (1-t) * noise``, target ``data - noise``.

Consequently a Transfusion state at ``t`` is evaluated by the DiT at ``1-t``
and its predicted velocity is negated.  The teacher is deliberately kept out
of the Lightning module tree: it is frozen, lazily materialized per rank, and
never duplicated into student checkpoints or DDP buckets.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from ..models.dense_dit_renderer import FrozenDenseDiTRenderer


class DenseDiTRectifiedFlowTeacher(FrozenDenseDiTRenderer):
    """Lazily loaded EMA DiT used for content-channel velocity distillation."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.loss_weight = float(self.config.get("loss_weight", 0.1))
        student_time_range = self.config.get("student_time_range", (0.0, 1.0))
        if not isinstance(student_time_range, (list, tuple)) or len(student_time_range) != 2:
            raise ValueError("DiT teacher student_time_range must contain two values")
        self.student_time_min = float(student_time_range[0])
        self.student_time_max = float(student_time_range[1])
        self.channel_start = int(self.config.get("channel_start", 0))
        self.channel_end = int(self.config.get("channel_end", 40))
        self.creation_only = bool(self.config.get("creation_only", True))
        self.disable_student_qwen_dropout = bool(
            self.config.get("disable_student_qwen_dropout", True)
        )
        self.context_presence_key = str(
            self.config.get("context_presence_key", "previous_foa_present")
        )
        self.metadata_equals = dict(self.config.get("metadata_equals") or {})
        self.max_examples_per_group = int(
            self.config.get("max_examples_per_group", 0)
        )
        self.group_key = str(self.config.get("group_key", "family_id"))
        self.rotation_key = str(
            self.config.get("rotation_key", "family_turn_index")
        )
        self.every_n_steps = int(self.config.get("every_n_steps", 1))
        self.step_offset = int(self.config.get("step_offset", 0))
        if not self.enabled:
            raise ValueError(
                "DenseDiTRectifiedFlowTeacher should not be constructed when disabled"
            )
        if not math.isfinite(self.loss_weight) or self.loss_weight <= 0.0:
            raise ValueError("DiT teacher loss_weight must be finite and positive")
        if not (
            math.isfinite(self.student_time_min)
            and math.isfinite(self.student_time_max)
            and 0.0 <= self.student_time_min < self.student_time_max <= 1.0
        ):
            raise ValueError(
                "DiT teacher student_time_range must satisfy 0 <= min < max <= 1"
            )
        if self.channel_start < 0 or self.channel_end <= self.channel_start:
            raise ValueError("DiT teacher channel range must be non-empty")
        if self.max_examples_per_group < 0:
            raise ValueError("DiT teacher max_examples_per_group cannot be negative")
        if self.every_n_steps < 1:
            raise ValueError("DiT teacher every_n_steps must be positive")
        if not 0 <= self.step_offset < self.every_n_steps:
            raise ValueError(
                "DiT teacher step_offset must be in [0, every_n_steps)"
            )

        self.last_metrics: dict[str, Tensor] = {}
        self._selection_fraction: Tensor | None = None
        self._pulse_active: Tensor | None = None

    @staticmethod
    def _metadata_value_matches(value: Any, expected: Any) -> bool:
        allowed = expected if isinstance(expected, (list, tuple, set)) else (expected,)
        return value in allowed

    def select_enabled_samples(
        self,
        metadata: Sequence[Mapping[str, Any]],
        base_enabled: Sequence[bool],
        *,
        step: int,
        device: torch.device,
    ) -> list[bool]:
        """Apply fail-closed metadata filters and rotating per-family sampling."""

        if len(metadata) != len(base_enabled):
            raise ValueError("DiT teacher metadata and enable masks must align")
        teacher_step = int(step) - self.step_offset
        schedule_active = teacher_step >= 0 and not (
            teacher_step % self.every_n_steps
        )
        self._pulse_active = torch.tensor(
            float(schedule_active), device=device, dtype=torch.float32
        )
        if not schedule_active:
            self._selection_fraction = torch.tensor(
                0.0, device=device, dtype=torch.float32
            )
            return [False] * len(base_enabled)
        rotation_step = teacher_step // self.every_n_steps
        enabled = [bool(value) for value in base_enabled]
        for index, item in enumerate(metadata):
            if not enabled[index]:
                continue
            for key, expected in self.metadata_equals.items():
                if key not in item:
                    raise KeyError(
                        f"DiT teacher metadata filter key {key!r} is missing"
                    )
                if not self._metadata_value_matches(item[key], expected):
                    enabled[index] = False
                    break

        if self.max_examples_per_group > 0:
            groups: dict[str, list[int]] = {}
            for index, (item, is_enabled) in enumerate(zip(metadata, enabled)):
                if not is_enabled:
                    continue
                if self.group_key not in item or self.rotation_key not in item:
                    raise KeyError(
                        "DiT teacher grouped selection requires metadata keys "
                        f"{self.group_key!r} and {self.rotation_key!r}"
                    )
                groups.setdefault(str(item[self.group_key]), []).append(index)
            selected = [False] * len(enabled)
            for indices in groups.values():
                ordered = sorted(
                    indices,
                    key=lambda index: int(metadata[index][self.rotation_key]),
                )
                count = min(self.max_examples_per_group, len(ordered))
                start = rotation_step % len(ordered)
                for offset in range(count):
                    selected[ordered[(start + offset) % len(ordered)]] = True
            enabled = selected

        self._selection_fraction = torch.tensor(
            sum(enabled) / max(1, len(enabled)),
            device=device,
            dtype=torch.float32,
        )
        return enabled

    _pad_channel_first = staticmethod(FrozenDenseDiTRenderer.pad_channel_first)

    @staticmethod
    def _time_mask(mask: Tensor | None, length: int, device) -> Tensor:
        if mask is None:
            return torch.ones(length, device=device, dtype=torch.bool)
        value = torch.as_tensor(mask, device=device)
        if value.ndim == 0:
            return value.gt(0).expand(length)
        while value.ndim > 1:
            value = value.amax(dim=0)
        value = value.flatten()
        if value.numel() < length:
            value = torch.nn.functional.pad(
                value, (0, length - value.numel()), value=0.0
            )
        return value[:length].gt(0)

    @torch.no_grad()
    def _teacher_velocity(
        self,
        noised: Tensor,
        student_times: Tensor,
        conditioning: Sequence[Mapping[str, Any]],
        valid_mask: Tensor,
    ) -> Tensor:
        velocity = self.predict_velocity(
            noised,
            student_times,
            conditioning,
            valid_mask,
        )
        if self.channel_end > int(velocity.shape[1]):
            raise ValueError(
                f"DiT teacher channel_end={self.channel_end} exceeds "
                f"io_channels={velocity.shape[1]}"
            )
        return velocity

    def loss(
        self,
        predicted_flows: Sequence[Tensor],
        noised_modalities: Sequence[Tensor],
        student_times: Sequence[Tensor],
        *,
        conditioning: Sequence[Mapping[str, Any]],
        active_masks: Sequence[Tensor | None],
        enabled_samples: Sequence[bool],
    ) -> Tensor:
        batch_size = len(predicted_flows)
        if not (
            batch_size
            == len(noised_modalities)
            == len(student_times)
            == len(conditioning)
            == len(active_masks)
            == len(enabled_samples)
        ):
            raise ValueError("DiT teacher inputs must have equal batch lengths")
        if batch_size < 1:
            raise ValueError("DiT teacher received an empty batch")

        metadata_selected_indices = [
            index for index, enabled in enumerate(enabled_samples) if enabled
        ]
        if metadata_selected_indices:
            candidate_times = torch.stack(
                [
                    torch.as_tensor(
                        student_times[index], device=predicted_flows[0].device
                    ).reshape(())
                    for index in metadata_selected_indices
                ]
            ).detach()
            time_is_selected = (
                (candidate_times >= self.student_time_min)
                & (candidate_times <= self.student_time_max)
            ).tolist()
            selected_indices = [
                index
                for index, selected in zip(
                    metadata_selected_indices, time_is_selected
                )
                if selected
            ]
        else:
            selected_indices = []
        metric_device = predicted_flows[0].device
        time_selection_fraction = torch.tensor(
            len(selected_indices) / max(1, len(metadata_selected_indices)),
            device=metric_device,
            dtype=torch.float32,
        )
        effective_selection_fraction = torch.tensor(
            len(selected_indices) / batch_size,
            device=metric_device,
            dtype=torch.float32,
        )
        if not selected_indices:
            zero = sum(value.sum() * 0.0 for value in predicted_flows)
            self.last_metrics = {
                "loss_unweighted": zero.detach(),
                "active_fraction": zero.detach(),
                "time_selection_fraction": time_selection_fraction,
                "effective_selection_fraction": effective_selection_fraction,
                "selection_fraction": (
                    self._selection_fraction
                    if self._selection_fraction is not None
                    else zero.detach()
                ),
                "pulse_active": (
                    self._pulse_active
                    if self._pulse_active is not None
                    else zero.detach()
                ),
            }
            return zero

        selected_predicted = [predicted_flows[index] for index in selected_indices]
        selected_noised = [noised_modalities[index] for index in selected_indices]
        selected_times = [student_times[index] for index in selected_indices]
        selected_conditioning = [conditioning[index] for index in selected_indices]
        selected_masks = [active_masks[index] for index in selected_indices]

        predicted, predicted_valid = self._pad_channel_first(selected_predicted)
        noised, noised_valid = self._pad_channel_first(selected_noised)
        if tuple(predicted.shape) != tuple(noised.shape):
            raise ValueError("predicted and noised latent batches must align")
        valid = predicted_valid & noised_valid
        times = torch.stack(
            [torch.as_tensor(value, device=noised.device).reshape(()) for value in selected_times]
        )
        active = torch.zeros_like(valid)
        for index, (mask, value) in enumerate(zip(selected_masks, selected_noised)):
            active[index, : value.shape[-1]] = self._time_mask(
                mask, int(value.shape[-1]), noised.device
            )
        supervised = valid & active
        if not bool(supervised.any()):
            # Keep a zero connection to student predictions so static DDP sees
            # exactly the same parameter-use graph on edit-only batches.
            zero = predicted.sum() * 0.0
            self.last_metrics = {
                "loss_unweighted": zero.detach(),
                "active_fraction": supervised.float().mean(),
                "time_selection_fraction": time_selection_fraction,
                "effective_selection_fraction": effective_selection_fraction,
                "selection_fraction": (
                    self._selection_fraction
                    if self._selection_fraction is not None
                    else supervised.new_tensor(1.0, dtype=torch.float32)
                ),
                "pulse_active": (
                    self._pulse_active
                    if self._pulse_active is not None
                    else supervised.new_tensor(1.0, dtype=torch.float32)
                ),
            }
            return zero

        teacher_velocity = self._teacher_velocity(
            noised, times, selected_conditioning, valid
        )
        channel_slice = slice(self.channel_start, self.channel_end)
        difference = (
            predicted[:, channel_slice].float()
            - teacher_velocity[:, channel_slice].float()
        )
        weights = supervised[:, None, :].to(difference.dtype)
        unweighted = (difference.square() * weights).sum() / (
            weights.sum().clamp_min(1.0) * difference.shape[1]
        )
        teacher_rms = (
            teacher_velocity[:, channel_slice].float().square() * weights
        ).sum().div(
            weights.sum().clamp_min(1.0) * difference.shape[1]
        ).sqrt()
        self.last_metrics = {
            "loss_unweighted": unweighted.detach(),
            "active_fraction": supervised.float().mean().detach(),
            "teacher_velocity_rms": teacher_rms.detach(),
            "time_selection_fraction": time_selection_fraction,
            "effective_selection_fraction": effective_selection_fraction,
            "selection_fraction": (
                self._selection_fraction
                if self._selection_fraction is not None
                else supervised.new_tensor(1.0, dtype=torch.float32)
            ),
            "pulse_active": (
                self._pulse_active
                if self._pulse_active is not None
                else supervised.new_tensor(1.0, dtype=torch.float32)
            ),
        }
        return unweighted * self.loss_weight
