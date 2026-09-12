"""Lightning training loop for the canonical Qwen-backed P11 route."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch import Tensor

from .ema import TrainableParameterEMA
from .utils import create_optimizer_from_config, create_scheduler_from_config
from ..data.sceneplan_edit_patch import PATCH_OUTPUT_CONTRACT
from ..data.sceneplan_p11_dataset import P11_MANIFEST_VERSION
from ..data.sceneplan_p11_single_turn import (
    P11_EDITING_CONTRACT,
    P11_EDITING_INPUT_CONTRACT,
    P11Task,
    normalize_p11_task,
)
from ..models.sceneplan_p11 import ScenePlanP11Planner


class ScenePlanP11TrainingWrapper(pl.LightningModule):
    """Train G/U full plans and E atomic patches with task-routed inputs."""

    def __init__(
        self,
        model: ScenePlanP11Planner,
        *,
        optimizer_configs: Mapping[str, Any],
        planner_loss_group_weights: Optional[Mapping[int | str, float]] = None,
        use_ema: bool = True,
        ema_beta: float = 0.999,
        ema_power: float = 0.75,
        ema_update_every: int = 1,
        ema_update_after_step: int = 1,
        log_every_n_steps: int = 10,
    ) -> None:
        super().__init__()
        self.p11 = model
        self.optimizer_configs = dict(optimizer_configs or {})
        if set(self.optimizer_configs) != {"p11"}:
            raise ValueError("P11 training requires only optimizer_configs.p11")
        self.planner_loss_group_weights = (
            {int(key): float(value) for key, value in planner_loss_group_weights.items()}
            if planner_loss_group_weights
            else None
        )
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
            raise TypeError("P11 batch must be [carrier_latents, metadata_list]")
        metadata = list(batch[1])
        if not metadata or not all(isinstance(value, Mapping) for value in metadata):
            raise TypeError("P11 metadata batch must contain mapping rows")
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
            self._optional_tensor(value.get("p11_input_valid_mask")) for value in metadata
        ]
        semantics = [
            self._optional_tensor(value.get("p11_input_semantic")) for value in metadata
        ]
        input_plans = [value.get("p11_input_sceneplan_tokens") for value in metadata]
        for index, (task, value) in enumerate(zip(tasks, metadata)):
            if int(value.get("p11_manifest_version", -1)) != P11_MANIFEST_VERSION:
                raise ValueError(
                    "canonical P11 accepts only manifest "
                    f"v{P11_MANIFEST_VERSION}"
                )
            expected_audio = task is P11Task.UNDERSTANDING
            expected_plan = task is P11Task.EDITING
            if (input_audio[index] is not None) != expected_audio:
                raise ValueError(f"P11 row {index} has a stale task/audio truth table")
            if (input_masks[index] is not None) != expected_audio:
                raise ValueError(f"P11 row {index} has a stale task/mask truth table")
            if (input_plans[index] is not None) != expected_plan:
                raise ValueError(f"P11 row {index} has a stale task/current-plan truth table")
            if expected_audio and self.p11.require_semantic and semantics[index] is None:
                raise ValueError("canonical P11 Understanding lacks semantic features")
            if not expected_audio and semantics[index] is not None:
                raise ValueError("semantic audio features leaked into a non-U row")
            if value.get("p11_output_contract") != PATCH_OUTPUT_CONTRACT:
                raise ValueError("P11 output protocol changed")
            if task is P11Task.EDITING:
                if value.get("p11_editing_contract") != P11_EDITING_CONTRACT:
                    raise ValueError("P11 Editing lacks the atomic-patch contract")
                if value.get("p11_editing_input_contract") != P11_EDITING_INPUT_CONTRACT:
                    raise ValueError("P11 Editing input contract changed")
                if value.get("p11_output_kind") != "edit_patch":
                    raise ValueError("P11 Editing target is not an edit patch")
            elif value.get("p11_output_kind") != "sceneplan":
                raise ValueError("P11 G/U target is not a ScenePlan")
            if value.get("p11_scene_thought_contract") is not None:
                raise ValueError("retired core40 supervision leaked into discrete D0")

        planner_loss, planner_metrics = self.p11.forward_planner(
            prompts,
            targets,
            tasks=tasks,
            input_foa=input_audio,
            input_valid_masks=input_masks,
            input_semantic=semantics,
            input_plans=input_plans,
            loss_group_weights=self.planner_loss_group_weights,
        )
        if not bool(torch.isfinite(planner_loss)):
            raise RuntimeError("P11 produced a non-finite Planner loss")

        per_row = planner_metrics["planner_ce_per_row"]
        if per_row.ndim != 1 or per_row.numel() != len(tasks):
            raise RuntimeError("P11 per-row CE no longer aligns with its batch")
        task_losses = {}
        task_counts = {}
        for task in P11Task:
            task_mask = torch.tensor(
                [value is task for value in tasks],
                device=per_row.device,
                dtype=torch.bool,
            )
            task_counts[task.value] = int(task_mask.sum())
            task_losses[task.value] = (
                per_row[task_mask].mean() if bool(task_mask.any()) else per_row.new_zeros(())
            )
        metrics = {
            f"{stage}/loss": planner_loss.detach(),
            f"{stage}/planner_ce": planner_metrics["planner_ce"],
            f"{stage}/planner_tokens": planner_metrics["planner_tokens"].float(),
            f"{stage}/sequence_tokens": planner_metrics["sequence_tokens"],
            f"{stage}/sequence_padding_tokens": planner_metrics[
                "sequence_padding_tokens"
            ],
            f"{stage}/prompt_tokens": planner_metrics["prompt_tokens"],
            f"{stage}/input_audio_tokens": planner_metrics["input_audio_tokens"],
            f"{stage}/input_semantic_tokens": planner_metrics["input_semantic_tokens"],
            f"{stage}/input_plan_tokens": planner_metrics["input_plan_tokens"],
            **{
                f"{stage}/task_{name}": planner_loss.new_tensor(float(count))
                for name, count in task_counts.items()
            },
            **{
                f"{stage}/{name}_ce": value for name, value in task_losses.items()
            },
        }
        prompt_views = [str(value.get("p11_prompt_view_id") or "unknown") for value in metadata]
        for view_id in (
            "manifest_exact_v0",
            "exact_v1",
            "semantic_only_v1",
            "semantic_temporal_v1",
            "coarse_scene_v1",
            "audio_evidence_v1",
            "atomic_instruction_v1",
        ):
            view_mask = torch.tensor(
                [value == view_id for value in prompt_views],
                device=per_row.device,
                dtype=torch.bool,
            )
            metrics[f"{stage}/view_{view_id}_ce_present_mean"] = (
                per_row[view_mask].mean()
                if bool(view_mask.any())
                else per_row.new_zeros(())
            )
            metrics[f"{stage}/view_{view_id}_rows"] = view_mask.sum().float()
        if stage == "train":
            self._last_step_metrics = metrics
            if int(self.global_step) % self.log_every_n_steps == 0:
                self.log_dict(
                    metrics,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=self.logger is not None,
                    sync_dist=False,
                )
        else:
            self.log_dict(
                metrics,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=self.logger is not None,
                sync_dist=True,
            )
        return planner_loss

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
        handles_unscaling = bool(getattr(optimizer, "_step_supports_amp_scaling", False))
        if not handles_unscaling or clip_value <= 0.0:
            return super().configure_gradient_clipping(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )
        scaler = getattr(self.trainer.precision_plugin, "scaler", None)
        if scaler is not None:
            raise RuntimeError("fused P11 gradient clipping supports BF16 but not FP16")
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
            raise RuntimeError("P11 exposes no trainable planner/bridge parameters")
        optimizer = create_optimizer_from_config(config["optimizer"], parameters)
        if "scheduler" not in config:
            return [optimizer]
        scheduler = create_scheduler_from_config(config["scheduler"], optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


__all__ = ["ScenePlanP11TrainingWrapper"]
