"""Lightning wrapper for sketch-first P11-v4 Transfusion-CoT."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch import Tensor

from ..data.scene_sketch_v1 import (
    CONTROL_DIRECTION_OPERATIONS,
)
from ..data.sceneplan_p11_single_turn import P11Task, normalize_p11_task
from ..data.sceneplan_p11_v4_dataset import (
    P11_AUDIO_AWARE_DATA_CONTRACT,
    P11_V4_DATA_CONTRACT,
    P11_V4_CONTROL_DIRECTION_CONTRACT,
)
from ..models.sceneplan_p11_v4 import (
    ScenePlanP11AudioAwarePlanner,
    ScenePlanP11V4Planner,
)
from .ema import TrainableParameterEMA
from .utils import create_optimizer_from_config, create_scheduler_from_config


class ScenePlanP11V4TrainingWrapper(pl.LightningModule):
    """Train sketch CE and executable flow losses on balanced G/U/E rows."""

    def __init__(
        self,
        model: ScenePlanP11V4Planner,
        *,
        optimizer_configs: Mapping[str, Any],
        planner_loss_group_weights: Optional[Mapping[int | str, float]] = None,
        transfusion_cot_loss_weights: Optional[Mapping[str, float]] = None,
        use_ema: bool = True,
        ema_beta: float = 0.999,
        ema_power: float = 0.75,
        ema_update_every: int = 1,
        ema_update_after_step: int = 1,
        log_every_n_steps: int = 10,
    ) -> None:
        super().__init__()
        if not isinstance(model, ScenePlanP11V4Planner):
            raise TypeError("P11-v4 wrapper requires ScenePlanP11V4Planner")
        self.p11 = model
        self.optimizer_configs = dict(optimizer_configs or {})
        if set(self.optimizer_configs) != {"p11"}:
            raise ValueError("P11-v4 training requires only optimizer_configs.p11")
        self.planner_loss_group_weights = (
            {
                int(key): float(value)
                for key, value in planner_loss_group_weights.items()
            }
            if planner_loss_group_weights
            else None
        )
        self.transfusion_cot_loss_weights = {
            str(key): float(value)
            for key, value in (transfusion_cot_loss_weights or {}).items()
        }
        required = {"flow", "solve", "locality", "owner"}
        if isinstance(model, ScenePlanP11AudioAwarePlanner):
            required.add("delta_control")
        configured = set(self.transfusion_cot_loss_weights)
        if configured != required:
            raise ValueError(
                "P11-v4 requires exact transfusion_cot_loss_weights "
                f"{sorted(required)}"
            )
        if any(value < 0.0 for value in self.transfusion_cot_loss_weights.values()):
            raise ValueError("P11-v4 loss weights must be non-negative")
        if self.transfusion_cot_loss_weights["owner"] <= 0.0:
            raise ValueError("P11-v4 DeltaSketch-owner loss weight must be positive")
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.p11_ema = (
            TrainableParameterEMA(
                self.p11,
                beta=float(ema_beta),
                power=float(ema_power),
                update_every=int(ema_update_every),
                update_after_step=int(ema_update_after_step),
            )
            if bool(use_ema)
            else None
        )

    @staticmethod
    def _metadata(batch: Sequence[Any]) -> list[Mapping[str, Any]]:
        if not isinstance(batch, (list, tuple)) or len(batch) != 2:
            raise TypeError("P11-v4 batch must be [carrier_latents, metadata_list]")
        metadata = list(batch[1])
        if not metadata or not all(isinstance(value, Mapping) for value in metadata):
            raise TypeError("P11-v4 metadata batch must contain mapping rows")
        return metadata

    @staticmethod
    def _optional_tensor(value: Any) -> Optional[Tensor]:
        return None if value is None else torch.as_tensor(value)

    def _shared_step(self, batch, stage: str) -> Tensor:
        metadata = self._metadata(batch)
        tasks = [normalize_p11_task(value["p11_task"]) for value in metadata]
        prompts = [value["p11_prompt"] for value in metadata]
        targets = [value["p11_target_tokens"] for value in metadata]
        input_audio = [
            self._optional_tensor(value.get("p11_input_foa")) for value in metadata
        ]
        input_masks = [
            self._optional_tensor(value.get("p11_input_valid_mask"))
            for value in metadata
        ]
        input_semantic = [
            self._optional_tensor(value.get("p11_input_semantic"))
            for value in metadata
        ]
        input_plans = [value.get("p11_input_sceneplan_tokens") for value in metadata]
        input_lexical = [value.get("p11_input_lexical") for value in metadata]

        for index, (task, value) in enumerate(zip(tasks, metadata)):
            if value.get("p11_v4_data_contract") != P11_V4_DATA_CONTRACT:
                raise ValueError("P11-v4 row has the wrong data contract")
            expected_audio = task is P11Task.UNDERSTANDING
            expected_plan = task is P11Task.EDITING
            if (input_audio[index] is not None) != expected_audio:
                raise ValueError("P11-v4 task/audio truth table changed")
            if (input_masks[index] is not None) != expected_audio:
                raise ValueError("P11-v4 task/audio-mask truth table changed")
            if (input_plans[index] is not None) != expected_plan:
                raise ValueError("P11-v4 task/current-plan truth table changed")
            expected_kind = (
                "delta_sketch" if task is P11Task.EDITING else "scene_sketch"
            )
            if value.get("p11_output_kind") != expected_kind:
                raise ValueError("P11-v4 row uses the wrong discrete target")
            if value.get("p11_scene_thought_contract") is not None:
                raise ValueError("retired core40 supervision leaked into P11-v4")
            if task is not P11Task.UNDERSTANDING and input_lexical[index] is not None:
                raise ValueError("lexical evidence leaked into a non-U row")
            direction_mask = bool(value["p11_v4_control_direction_mask"])
            direction = int(value["p11_v4_control_direction"])
            direction_operation = value.get("p11_v4_control_direction_operation")
            expected_direction_contract = (
                P11_V4_CONTROL_DIRECTION_CONTRACT if direction_mask else None
            )
            if (
                value.get("p11_v4_control_direction_contract")
                != expected_direction_contract
            ):
                raise ValueError("P11-v4 control-direction data contract changed")
            if direction_mask:
                if (
                    direction not in {-1, 1}
                    or direction_operation not in CONTROL_DIRECTION_OPERATIONS
                    or task is not P11Task.EDITING
                ):
                    raise ValueError("P11-v4 active control direction is invalid")
                program = self.p11.delta_sketch_codec.decode(
                    value["p11_target_tokens"]["input_ids"]
                )
                if (
                    program.get("operation") != direction_operation
                    or int(program.get("control_direction", 0)) != direction
                ):
                    raise ValueError(
                        "P11-v4 control-direction metadata disagrees with DeltaSketch"
                    )
            elif direction != 0 or direction_operation is not None:
                raise ValueError(
                    "masked P11-v4 control-direction row carries an endpoint"
                )

        loss, metrics = self.p11.forward_transfusion_cot(
            prompts,
            targets,
            tasks=tasks,
            input_foa=input_audio,
            input_valid_masks=input_masks,
            input_semantic=input_semantic,
            input_plans=input_plans,
            input_lexical=input_lexical,
            target_execution_cores=[
                value["p11_v4_target_execution_core"] for value in metadata
            ],
            input_execution_cores=[
                value["p11_v4_input_execution_core"] for value in metadata
            ],
            delta_execution_cores=[
                value["p11_v4_delta_execution_core"] for value in metadata
            ],
            target_source_masks=[
                value["p11_v4_target_source_mask"] for value in metadata
            ],
            input_source_masks=[
                value["p11_v4_input_source_mask"] for value in metadata
            ],
            loss_group_weights=self.planner_loss_group_weights,
            thought_loss_weights=self.transfusion_cot_loss_weights,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("P11-v4 produced a non-finite loss")
        per_row = metrics["discrete_ce_per_row"]
        if per_row.ndim != 1 or per_row.numel() != len(tasks):
            raise RuntimeError("P11-v4 per-row CE no longer aligns")
        logged = {
            f"{stage}/loss": loss.detach(),
            f"{stage}/discrete_ce": metrics["discrete_ce"],
            f"{stage}/flow": metrics["flow_per_row"].mean(),
            f"{stage}/solve": metrics["solve_per_row"].mean(),
            f"{stage}/locality": metrics["locality_per_row"].mean(),
            f"{stage}/owner": metrics["owner_active_loss"],
            f"{stage}/owner_rows": metrics["owner_row_count"].float(),
            f"{stage}/owner_accuracy": metrics["owner_accuracy"].float(),
            f"{stage}/text_end_ce": metrics["text_end_ce"],
            f"{stage}/text_end_ce_rows": metrics["text_end_count"].float(),
            f"{stage}/scene_eos_ce": metrics["scene_eos_ce"],
            f"{stage}/scene_eos_ce_rows": metrics["scene_eos_count"].float(),
            f"{stage}/u_inventory_ce": metrics["u_inventory_loss"],
            f"{stage}/u_inventory_ce_rows": metrics["u_inventory_rows"].float(),
            f"{stage}/u_source_count_ce": metrics["u_source_count_ce"],
            f"{stage}/u_source_count_ce_rows": metrics[
                "u_source_count_count"
            ].float(),
            f"{stage}/u_source_count_accuracy": metrics[
                "u_source_count_accuracy"
            ].float(),
            f"{stage}/u_room_ce": metrics["u_room_ce"],
            f"{stage}/u_room_ce_rows": metrics["u_room_count"].float(),
            f"{stage}/u_room_accuracy": metrics["u_room_accuracy"].float(),
            f"{stage}/u_kind_ce": metrics["u_kind_ce"],
            f"{stage}/u_kind_ce_rows": metrics["u_kind_count"].float(),
            f"{stage}/u_kind_accuracy": metrics["u_kind_accuracy"].float(),
            f"{stage}/control_direction_ce": metrics["control_direction_ce"],
            # TrainingHealthGateCallback pairs a sparse objective with
            # ``<objective>_rows``.  Keep the short public counter name below
            # for dashboards, and expose this exact alias so direction-token
            # CE is weighted only by active rotate/distance rows.
            f"{stage}/control_direction_ce_rows": metrics[
                "control_direction_count"
            ].float(),
            f"{stage}/control_direction_rows": metrics[
                "control_direction_count"
            ].float(),
            f"{stage}/control_direction_accuracy": metrics[
                "control_direction_accuracy"
            ].float(),
            f"{stage}/discrete_tokens": metrics["discrete_tokens"].float(),
            f"{stage}/thought_tokens": metrics["thought_tokens"].float(),
            f"{stage}/prompt_tokens": metrics["prompt_tokens"].float(),
            f"{stage}/input_audio_tokens": metrics["input_audio_tokens"].float(),
            f"{stage}/input_semantic_tokens": metrics[
                "input_semantic_tokens"
            ].float(),
            f"{stage}/input_lexical_tokens": metrics[
                "input_lexical_tokens"
            ].float(),
            f"{stage}/input_plan_tokens": metrics["input_plan_tokens"].float(),
        }
        for task in P11Task:
            mask = torch.tensor(
                [value is task for value in tasks],
                device=per_row.device,
                dtype=torch.bool,
            )
            logged[f"{stage}/task_{task.value}"] = mask.sum().float()
            logged[f"{stage}/{task.value}_discrete_ce"] = (
                per_row[mask].mean() if bool(mask.any()) else per_row.new_zeros(())
            )
            logged[f"{stage}/{task.value}_flow"] = (
                metrics["flow_per_row"][mask].mean()
                if bool(mask.any())
                else per_row.new_zeros(())
            )
        if stage == "train":
            self._last_step_metrics = logged
            if int(self.global_step) % self.log_every_n_steps == 0:
                self.log_dict(
                    logged,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=self.logger is not None,
                    sync_dist=False,
                )
        else:
            self.log_dict(
                logged,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=self.logger is not None,
                sync_dist=True,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_before_zero_grad(self, *args, **kwargs) -> None:
        if self.p11_ema is not None:
            self.p11_ema.update()

    def ema_scope(self):
        return (
            self.p11_ema.apply_to(self.p11)
            if self.p11_ema is not None
            else nullcontext(self.p11)
        )

    def export_model(self, path: str) -> None:
        with self.ema_scope():
            torch.save({"state_dict": self.p11.state_dict()}, path)

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ) -> None:
        clip_value = float(gradient_clip_val or 0.0)
        if clip_value <= 0.0:
            return
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        algorithm = getattr(
            gradient_clip_algorithm, "value", gradient_clip_algorithm or "norm"
        )
        if str(algorithm).lower() == "norm":
            torch.nn.utils.clip_grad_norm_(parameters, clip_value)
        elif str(algorithm).lower() == "value":
            torch.nn.utils.clip_grad_value_(parameters, clip_value)
        else:
            raise ValueError(f"unsupported gradient clip algorithm {algorithm!r}")

    def configure_optimizers(self):
        config = self.optimizer_configs["p11"]
        parameters = [
            parameter for parameter in self.p11.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("P11-v4 exposes no trainable parameters")
        optimizer = create_optimizer_from_config(config["optimizer"], parameters)
        if "scheduler" not in config:
            return [optimizer]
        scheduler = create_scheduler_from_config(config["scheduler"], optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class ScenePlanP11AudioAwareTrainingWrapper(ScenePlanP11V4TrainingWrapper):
    """Train FOA observation and instruction-conditioned delta in one DAG."""

    def __init__(self, model: ScenePlanP11AudioAwarePlanner, **kwargs: Any) -> None:
        if not isinstance(model, ScenePlanP11AudioAwarePlanner):
            raise TypeError(
                "audio-aware P11 wrapper requires ScenePlanP11AudioAwarePlanner"
            )
        super().__init__(model, **kwargs)

    def _shared_step(self, batch, stage: str) -> Tensor:
        metadata = self._metadata(batch)
        tasks = [normalize_p11_task(value["p11_task"]) for value in metadata]
        input_audio = [
            self._optional_tensor(value.get("p11_input_foa")) for value in metadata
        ]
        input_masks = [
            self._optional_tensor(value.get("p11_input_valid_mask"))
            for value in metadata
        ]
        input_semantic = [
            self._optional_tensor(value.get("p11_input_semantic"))
            for value in metadata
        ]
        input_plans = [value.get("p11_prior_sceneplan_tokens") for value in metadata]
        input_lexical = [value.get("p11_input_lexical") for value in metadata]
        for index, (task, row) in enumerate(zip(tasks, metadata)):
            if (
                row.get("p11_audio_aware_data_contract")
                != P11_AUDIO_AWARE_DATA_CONTRACT
            ):
                raise ValueError("P11 row has the wrong active data contract")
            expects_audio = task in {P11Task.UNDERSTANDING, P11Task.EDITING}
            if (input_audio[index] is not None) != expects_audio:
                raise ValueError("audio-aware task/audio truth table changed")
            if (input_masks[index] is not None) != expects_audio:
                raise ValueError("audio-aware task/audio-mask truth table changed")
            if task is not P11Task.EDITING and input_plans[index] is not None:
                raise ValueError("fallible old-plan evidence leaked outside Editing")
            if task is P11Task.EDITING:
                if (
                    row.get("p11_delta_scene_sketch_tokens") is None
                    or row.get("p11_observed_sceneplan_target") is None
                    or row.get("p11_revised_sceneplan_target") is None
                ):
                    raise ValueError("Editing row lacks observed/delta/revised targets")
            elif row.get("p11_delta_scene_sketch_tokens") is not None:
                raise ValueError("DeltaSketch leaked into a non-Editing row")
            if task is P11Task.GENERATION and input_lexical[index] is not None:
                raise ValueError("lexical audio evidence leaked into Generation")

        loss, metrics = self.p11.forward_audio_aware(
            [value["p11_prompt"] for value in metadata],
            [value["p11_observation_prompt"] for value in metadata],
            [value["p11_observed_scene_sketch_tokens"] for value in metadata],
            [value.get("p11_delta_scene_sketch_tokens") for value in metadata],
            tasks=tasks,
            input_foa=input_audio,
            input_valid_masks=input_masks,
            input_semantic=input_semantic,
            input_plans=input_plans,
            input_lexical=input_lexical,
            observed_execution_cores=[
                value["p11_observed_execution_core"] for value in metadata
            ],
            revised_execution_cores=[
                value["p11_revised_execution_core"] for value in metadata
            ],
            delta_execution_cores=[
                value["p11_delta_execution_core"] for value in metadata
            ],
            observed_source_masks=[
                value["p11_observed_source_mask"] for value in metadata
            ],
            revised_source_masks=[
                value["p11_revised_source_mask"] for value in metadata
            ],
            loss_group_weights=self.planner_loss_group_weights,
            thought_loss_weights=self.transfusion_cot_loss_weights,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("audio-aware P11 produced a non-finite loss")
        per_row = metrics["discrete_ce_per_row"]
        logged = {
            f"{stage}/loss": loss.detach(),
            f"{stage}/discrete_ce": metrics["discrete_ce"],
            f"{stage}/observation_ce": metrics["observation_ce_per_row"].mean(),
            f"{stage}/delta_ce": metrics["delta_ce_per_row"].sum()
            / metrics["editing_rows"].clamp_min(1.0),
            f"{stage}/flow": metrics["flow_per_row"].mean(),
            f"{stage}/observation_solve": metrics[
                "observation_solve_per_row"
            ].mean(),
            f"{stage}/delta_solve": metrics["delta_solve_per_row"].sum()
            / metrics["editing_rows"].clamp_min(1.0),
            f"{stage}/delta_control": metrics["delta_control_loss"],
            f"{stage}/delta_control_raw_rmse": metrics[
                "delta_control_raw_rmse_per_row"
            ].sum()
            / metrics["delta_control_rows"].clamp_min(1.0),
            f"{stage}/delta_control_rows": metrics[
                "delta_control_rows"
            ].float(),
            f"{stage}/locality": metrics["locality_per_row"].sum()
            / metrics["editing_rows"].clamp_min(1.0),
            f"{stage}/owner": metrics["owner_active_loss"],
            f"{stage}/owner_rows": metrics["owner_row_count"].float(),
            f"{stage}/owner_accuracy": metrics["owner_accuracy"].float(),
            f"{stage}/text_end_ce": metrics["text_end_ce"],
            f"{stage}/text_end_ce_rows": metrics["text_end_count"].float(),
            f"{stage}/scene_eos_ce": metrics["scene_eos_ce"],
            f"{stage}/scene_eos_ce_rows": metrics["scene_eos_count"].float(),
            f"{stage}/u_inventory_ce": metrics["u_inventory_loss"],
            f"{stage}/u_inventory_ce_rows": metrics["u_inventory_rows"].float(),
            f"{stage}/u_source_count_ce": metrics["u_source_count_ce"],
            f"{stage}/u_source_count_ce_rows": metrics[
                "u_source_count_count"
            ].float(),
            f"{stage}/u_source_count_accuracy": metrics[
                "u_source_count_accuracy"
            ].float(),
            f"{stage}/u_room_ce": metrics["u_room_ce"],
            f"{stage}/u_room_ce_rows": metrics["u_room_count"].float(),
            f"{stage}/u_room_accuracy": metrics["u_room_accuracy"].float(),
            f"{stage}/u_kind_ce": metrics["u_kind_ce"],
            f"{stage}/u_kind_ce_rows": metrics["u_kind_count"].float(),
            f"{stage}/u_kind_accuracy": metrics["u_kind_accuracy"].float(),
            f"{stage}/discrete_tokens": metrics["discrete_tokens"].float(),
            f"{stage}/thought_tokens": metrics["thought_tokens"].float(),
            f"{stage}/prompt_tokens": metrics["prompt_tokens"].float(),
            f"{stage}/input_audio_tokens": metrics["input_audio_tokens"].float(),
            f"{stage}/input_semantic_tokens": metrics[
                "input_semantic_tokens"
            ].float(),
            f"{stage}/input_lexical_tokens": metrics[
                "input_lexical_tokens"
            ].float(),
            f"{stage}/input_plan_tokens": metrics["input_plan_tokens"].float(),
        }
        for task in P11Task:
            mask = torch.tensor(
                [value is task for value in tasks],
                device=per_row.device,
                dtype=torch.bool,
            )
            logged[f"{stage}/task_{task.value}"] = mask.sum().float()
            logged[f"{stage}/{task.value}_discrete_ce"] = (
                per_row[mask].mean() if bool(mask.any()) else per_row.new_zeros(())
            )
            logged[f"{stage}/{task.value}_flow"] = (
                metrics["flow_per_row"][mask].mean()
                if bool(mask.any())
                else per_row.new_zeros(())
            )
        if stage == "train":
            self._last_step_metrics = logged
            if int(self.global_step) % self.log_every_n_steps == 0:
                self.log_dict(
                    logged,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=self.logger is not None,
                    sync_dist=False,
                )
        else:
            self.log_dict(
                logged,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=self.logger is not None,
                sync_dist=True,
            )
        return loss


__all__ = [
    "ScenePlanP11AudioAwareTrainingWrapper",
    "ScenePlanP11V4TrainingWrapper",
]
