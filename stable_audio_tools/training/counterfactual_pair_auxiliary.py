"""Same-state flow supervision for source-binding counterfactual pairs."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
import math
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


class CounterfactualPairDeltaLoss:
    """Make source-control ownership identifiable in the RF vector field.

    Every matched base/swap pair receives the same target time and Gaussian
    base state. A rotating subset can be anchored at one or more configured
    times. At ``t=0`` the noised FOA inputs are naturally identical. For later
    times, ``same_noised_state_at_anchor`` symmetrically adjusts the two base
    states so the model sees the same active-region latent under both source
    conditions. The exact target-flow difference is then ``delta / (1 - t)``.
    This prevents the target-bearing noised input from becoming a shortcut for
    the source condition. The auxiliary adds no inference-time module and
    never shortens the fixed 432-frame primary target.
    """

    def __init__(self, config: Mapping[str, Any], *, target_modality: str):
        self.config = dict(config)
        self.target_modality = str(
            self.config.get("target_modality", target_modality)
        )
        self.weight = float(self.config.get("weight", 0.25))
        self.anchor_pairs_per_rank = int(
            self.config.get("anchor_pairs_per_rank", 1)
        )
        self.anchor_time = float(self.config.get("anchor_time", 0.0))
        raw_anchor_times = self.config.get("anchor_times")
        self.anchor_times = (
            (self.anchor_time,)
            if raw_anchor_times is None
            else tuple(float(value) for value in raw_anchor_times)
        )
        self.same_noised_state_at_anchor = bool(
            self.config.get("same_noised_state_at_anchor", False)
        )
        self.same_state_tolerance = float(
            self.config.get("same_state_tolerance", 1.0e-5)
        )
        raw_pair_keys = self.config.get(
            "pair_metadata_keys",
            ("curriculum_source_family_id", "curriculum_pair_index"),
        )
        if not isinstance(raw_pair_keys, (list, tuple)):
            raise ValueError("counterfactual pair_metadata_keys must be a sequence")
        self.pair_keys = tuple(str(key) for key in raw_pair_keys)
        self.role_key = str(
            self.config.get("role_metadata_key", "curriculum_role")
        )
        self.base_role = str(self.config.get("base_role", "base"))
        self.swapped_role = str(
            self.config.get("swapped_role", "control_swapped")
        )
        self.require_all_rows = bool(self.config.get("require_all_rows", True))
        self.epsilon = float(self.config.get("epsilon", 1.0e-6))
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("counterfactual pair loss weight must be positive")
        if self.anchor_pairs_per_rank <= 0:
            raise ValueError("anchor_pairs_per_rank must be positive")
        if not self.anchor_times or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.anchor_times
        ):
            raise ValueError(
                "counterfactual anchor_times must be a non-empty sequence in [0,1]"
            )
        if self.same_noised_state_at_anchor and any(
            value >= 1.0 for value in self.anchor_times
        ):
            raise ValueError(
                "same-state counterfactual anchors require every time < 1"
            )
        if (
            not math.isfinite(self.same_state_tolerance)
            or self.same_state_tolerance <= 0.0
        ):
            raise ValueError("counterfactual same_state_tolerance must be positive")
        if (
            not self.pair_keys
            or any(not key for key in self.pair_keys)
            or len(set(self.pair_keys)) != len(self.pair_keys)
            or not self.role_key
        ):
            raise ValueError("counterfactual metadata keys must be non-empty")
        if self.base_role == self.swapped_role:
            raise ValueError("counterfactual base/swapped roles must differ")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0.0:
            raise ValueError("counterfactual epsilon must be positive")
        self._pairs: list[tuple[int, int]] = []
        self._anchored_pairs: list[tuple[int, int]] = []
        self._target_masks: Optional[Sequence[Optional[Tensor]]] = None
        self._active_anchor_time = self.anchor_times[0]
        self.last_metrics: dict[str, Tensor] = {}

    def _resolve_pairs(
        self, examples: Sequence[Mapping[str, Any]]
    ) -> list[tuple[int, int]]:
        grouped: dict[tuple[Hashable, ...], dict[str, int]] = defaultdict(dict)
        eligible_rows = 0
        for index, example in enumerate(examples):
            metadata = example.get("metadata") or {}
            pair_values = tuple(metadata.get(key) for key in self.pair_keys)
            role = metadata.get(self.role_key)
            if all(value is None for value in pair_values) and role is None:
                if self.require_all_rows:
                    raise ValueError(
                        f"counterfactual row {index} lacks {self.pair_keys!r}/"
                        f"{self.role_key!r}"
                    )
                continue
            if any(value is None for value in pair_values) or role is None:
                raise ValueError(
                    f"counterfactual row {index} has incomplete pair metadata"
                )
            role = str(role)
            if role not in {self.base_role, self.swapped_role}:
                raise ValueError(
                    f"counterfactual row {index} has unknown role {role!r}"
                )
            if any(not isinstance(value, Hashable) for value in pair_values):
                raise TypeError(
                    f"counterfactual row {index} has an unhashable pair key"
                )
            if role in grouped[pair_values]:
                raise ValueError(
                    f"duplicate {role!r} row for pair {pair_values!r}"
                )
            grouped[pair_values][role] = index
            eligible_rows += 1
        pairs = []
        for pair_id in sorted(grouped, key=repr):
            roles = grouped[pair_id]
            if set(roles) != {self.base_role, self.swapped_role}:
                raise ValueError(
                    f"counterfactual pair {pair_id!r} is incomplete: {sorted(roles)}"
                )
            pairs.append((roles[self.base_role], roles[self.swapped_role]))
        if not pairs:
            raise ValueError("counterfactual batch contains no complete pairs")
        if eligible_rows != 2 * len(pairs):
            raise ValueError("counterfactual eligible rows are not disjoint pairs")
        return pairs

    @staticmethod
    def _target_occurrence_states(
        base_states: Optional[list[list[Optional[Tensor]]]],
        *,
        batch_size: int,
        modality_count: int,
    ) -> list[list[Optional[Tensor]]]:
        if base_states is None:
            return [[None] * modality_count for _ in range(batch_size)]
        if len(base_states) != batch_size or any(
            len(row) != modality_count for row in base_states
        ):
            raise ValueError("renderer base states do not align with the batch")
        return [list(row) for row in base_states]

    def prepare(
        self,
        *,
        examples: Sequence[Mapping[str, Any]],
        times: Tensor,
        target_index: int,
        modality_count: int,
        target_masks: Optional[Sequence[Optional[Tensor]]],
        base_states: Optional[list[list[Optional[Tensor]]]],
        step: int,
    ) -> tuple[Tensor, list[list[Optional[Tensor]]]]:
        if times.ndim != 2 or times.shape[0] != len(examples):
            raise ValueError("counterfactual times must be [batch, modality]")
        if not 0 <= target_index < times.shape[1] == modality_count:
            raise ValueError("counterfactual target modality index is invalid")
        pairs = self._resolve_pairs(examples)
        if len(pairs) < self.anchor_pairs_per_rank:
            raise ValueError(
                "counterfactual batch has fewer pairs than anchor_pairs_per_rank"
            )
        states = self._target_occurrence_states(
            base_states,
            batch_size=len(examples),
            modality_count=modality_count,
        )
        times = times.clone()
        for base_index, swapped_index in pairs:
            # Retain the base row's sampled marginal time and give its mate the
            # same value. Pair order is deterministic in the immutable store.
            times[swapped_index, target_index] = times[base_index, target_index]
            base_target = examples[base_index]["modalities"][self.target_modality]
            swapped_target = examples[swapped_index]["modalities"][
                self.target_modality
            ]
            if tuple(base_target.shape) != tuple(swapped_target.shape):
                raise ValueError("counterfactual targets have unequal shapes")
            if states[base_index][target_index] is not None or states[
                swapped_index
            ][target_index] is not None:
                raise ValueError(
                    "counterfactual same-noise forcing cannot overwrite an "
                    "existing target flow base state"
                )
            shared_noise = torch.randn_like(base_target)
            states[base_index][target_index] = shared_noise
            states[swapped_index][target_index] = shared_noise

        start = (int(step) * self.anchor_pairs_per_rank) % len(pairs)
        anchored = [
            pairs[(start + offset) % len(pairs)]
            for offset in range(self.anchor_pairs_per_rank)
        ]
        active_anchor_time = self.anchor_times[int(step) % len(self.anchor_times)]
        for base_index, swapped_index in anchored:
            times[base_index, target_index] = active_anchor_time
            times[swapped_index, target_index] = active_anchor_time
            if self.same_noised_state_at_anchor:
                base_target = examples[base_index]["modalities"][
                    self.target_modality
                ]
                swapped_target = examples[swapped_index]["modalities"][
                    self.target_modality
                ]
                shared_center = states[base_index][target_index]
                assert shared_center is not None
                target_delta = (swapped_target - base_target).detach()
                half_adjustment = (
                    0.5
                    * active_anchor_time
                    / (1.0 - active_anchor_time)
                    * target_delta
                )
                states[base_index][target_index] = (
                    shared_center + half_adjustment
                )
                states[swapped_index][target_index] = (
                    shared_center - half_adjustment
                )

        if target_masks is not None:
            if len(target_masks) != len(examples):
                raise ValueError("counterfactual target masks do not align")
            for base_index, swapped_index in pairs:
                left, right = target_masks[base_index], target_masks[swapped_index]
                if (left is None) != (right is None) or (
                    left is not None and not torch.equal(left, right)
                ):
                    raise ValueError(
                        "counterfactual pair activity masks must be identical"
                    )
        self._pairs = pairs
        self._anchored_pairs = anchored
        self._target_masks = target_masks
        self._active_anchor_time = active_anchor_time
        return times, states

    def __call__(
        self,
        predicted_flows: Sequence[Tensor],
        noised_modalities: Sequence[Tensor],
        modality_times: Sequence[Tensor],
        *,
        target_latents: Sequence[Tensor],
    ) -> Tensor:
        count = len(predicted_flows)
        if not (
            count
            == len(noised_modalities)
            == len(modality_times)
            == len(target_latents)
        ):
            raise ValueError("counterfactual flow occurrences do not align")
        if not self._pairs or not self._anchored_pairs:
            raise RuntimeError("counterfactual loss was called before prepare")

        losses = []
        pred_rms_values = []
        target_rms_values = []
        cosine_values = []
        same_state_error_values = []
        same_state_error_max_values = []
        for base_index, swapped_index in self._anchored_pairs:
            base_pred = predicted_flows[base_index].float()
            swapped_pred = predicted_flows[swapped_index].float()
            raw_target_delta = (
                target_latents[swapped_index].float()
                - target_latents[base_index].float()
            )
            mask = None
            if self._target_masks is not None:
                mask = self._target_masks[base_index]
            if mask is not None:
                mask = mask.to(
                    device=raw_target_delta.device,
                    dtype=raw_target_delta.dtype,
                )
                while mask.ndim < raw_target_delta.ndim:
                    mask = mask.unsqueeze(0)
            target_delta = raw_target_delta
            if mask is not None:
                target_delta = target_delta * mask
            if self.same_noised_state_at_anchor:
                target_delta = target_delta / (1.0 - self._active_anchor_time)
                observed_state_delta = (
                    noised_modalities[swapped_index].float()
                    - noised_modalities[base_index].float()
                )
                expected_state_delta = (
                    torch.zeros_like(raw_target_delta)
                    if mask is None
                    else raw_target_delta * (1.0 - mask)
                )
                state_error = observed_state_delta - expected_state_delta
                state_error_rms = state_error.square().mean().sqrt()
                state_error_max = state_error.abs().max()
                if float(state_error_max.detach().cpu()) > self.same_state_tolerance:
                    raise RuntimeError(
                        "same-state counterfactual anchor drifted: "
                        f"max_abs={float(state_error_max):.8g} exceeds "
                        f"{self.same_state_tolerance:.8g}"
                    )
                same_state_error_values.append(state_error_rms)
                same_state_error_max_values.append(state_error_max)
            predicted_delta = swapped_pred - base_pred
            target_rms = target_delta.square().mean().sqrt().clamp_min(self.epsilon)
            pred_rms = predicted_delta.square().mean().sqrt()
            normalized_error = (predicted_delta - target_delta) / target_rms.detach()
            losses.append(F.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error)))
            pred_rms_values.append(pred_rms)
            target_rms_values.append(target_rms)
            cosine_values.append(
                F.cosine_similarity(
                    predicted_delta.flatten().unsqueeze(0),
                    target_delta.flatten().unsqueeze(0),
                    dim=-1,
                    eps=self.epsilon,
                )[0]
            )

        raw_loss = torch.stack(losses).mean()
        weighted_loss = raw_loss * self.weight
        target_rms = torch.stack(target_rms_values).mean()
        pred_rms = torch.stack(pred_rms_values).mean()
        device = raw_loss.device
        zero = torch.zeros((), device=device)
        self.last_metrics = {
            "loss_raw": raw_loss.detach(),
            "loss_weighted": weighted_loss.detach(),
            "pair_count": torch.tensor(float(len(self._pairs)), device=device),
            "anchor_pair_count": torch.tensor(
                float(len(self._anchored_pairs)), device=device
            ),
            "anchor_example_fraction": torch.tensor(
                2.0 * len(self._anchored_pairs) / count, device=device
            ),
            "anchor_time": torch.tensor(
                self._active_anchor_time, device=device
            ),
            "anchor_time_count": torch.tensor(
                float(len(self.anchor_times)), device=device
            ),
            "same_noised_state": torch.tensor(
                float(self.same_noised_state_at_anchor), device=device
            ),
            "same_state_error_rms": (
                torch.stack(same_state_error_values).mean().detach()
                if same_state_error_values
                else zero
            ),
            "same_state_error_max_abs": (
                torch.stack(same_state_error_max_values).max().detach()
                if same_state_error_max_values
                else zero
            ),
            "predicted_delta_rms": pred_rms.detach(),
            "target_delta_rms": target_rms.detach(),
            "delta_rms_ratio": (pred_rms / target_rms).detach(),
            "delta_cosine": torch.stack(cosine_values).mean().detach(),
        }
        return weighted_loss


__all__ = ["CounterfactualPairDeltaLoss"]
