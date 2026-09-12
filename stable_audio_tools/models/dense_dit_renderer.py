"""Frozen dense-DiT acoustic field reused by the unified FOA renderer.

This is deliberately a plain Python owner rather than an ``nn.Module``.  The
proven dense checkpoint is loaded lazily once per process, stays frozen and is
referenced by immutable path; it is therefore neither copied into DDP buckets
nor duplicated in every Spatial-CoT checkpoint.  Trainable control/residual
adapters belong to the unified model and are checkpointed normally.
"""
from __future__ import annotations

import math
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


def build_trainable_dense_dit_late_blocks(
    config: Mapping[str, Any],
) -> tuple[nn.ModuleList, dict[str, Any]]:
    """Restore an exact, registered copy of selected native Dense blocks.

    The immutable Dense owner remains outside the unified module so its full
    checkpoint is not duplicated in DDP/EMA/checkpoints.  Only the explicitly
    selected late blocks are materialized here.  They start bit-identical to
    the same online/EMA checkpoint branch used by :class:`FrozenDenseDiTRenderer`
    and can therefore replace those native blocks without changing step-zero
    behavior.
    """

    renderer_config = dict(config)
    joint_config = renderer_config.get("joint_late_blocks") or {}
    if not isinstance(joint_config, Mapping):
        raise ValueError("dense joint_late_blocks must be an object")
    if not bool(joint_config.get("enabled", False)):
        raise ValueError("dense joint late-block construction requires enabled=true")

    model_config_path = Path(str(renderer_config.get("model_config", "")))
    checkpoint_path = Path(str(renderer_config.get("checkpoint", "")))
    weights = str(renderer_config.get("weights", "ema")).lower()
    for path, label in (
        (model_config_path, "model_config"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"dense joint late blocks {label}: {path}")
    if weights not in {"ema", "online"}:
        raise ValueError("dense joint late-block weights must be 'ema' or 'online'")

    from stable_audio_tools.configuration import load_config
    from stable_audio_tools.models.dit import DiffusionTransformer
    from stable_audio_tools.models.transformer import ContinuousTransformer
    from stable_audio_tools.models.utils import load_ckpt_state_dict

    model_config = load_config(model_config_path)
    if model_config.get("model_type") != "diffusion_cond":
        raise ValueError("dense joint late blocks require model_type='diffusion_cond'")
    diffusion_config = ((model_config.get("model") or {}).get("diffusion") or {})
    if diffusion_config.get("type") != "dit":
        raise ValueError("dense joint late blocks require a DiT diffusion model")
    objective = diffusion_config.get("diffusion_objective")
    if objective != "rectified_flow":
        raise ValueError("dense joint late blocks require rectified_flow")
    dit_config = dict(diffusion_config.get("config") or {})
    depth = int(dit_config.get("depth", 0))
    start_index = joint_config.get("start_index")
    count = joint_config.get("count")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or start_index < 0
        or count <= 0
        or start_index + count > depth
    ):
        raise ValueError(
            "dense joint late-block range must fit the checkpoint-native "
            f"Transformer depth: start={start_index}, count={count}, depth={depth}"
        )

    # Build the exact native architecture on meta first, then allocate only the
    # selected blocks.  This avoids constructing another complete ~321M model
    # on every rank merely to obtain two late blocks.
    with torch.random.fork_rng(devices=[], enabled=True):
        with torch.device("meta"):
            template = DiffusionTransformer(
                diffusion_objective=objective,
                **dit_config,
            )
    transformer = getattr(template, "transformer", None)
    if not isinstance(transformer, ContinuousTransformer):
        raise ValueError(
            "dense joint late blocks require the native ContinuousTransformer"
        )
    selected = nn.ModuleList(
        [transformer.layers[index] for index in range(start_index, start_index + count)]
    )
    selected.to_empty(device=torch.device("cpu"))

    state = load_ckpt_state_dict(str(checkpoint_path))
    branch_prefix = (
        "diffusion_ema.ema_model.model.transformer.layers."
        if weights == "ema"
        else "diffusion.model.model.transformer.layers."
    )
    loaded_keys = 0
    for offset, layer in enumerate(selected):
        native_index = start_index + offset
        prefix = f"{branch_prefix}{native_index}."
        layer_state = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not layer_state:
            raise RuntimeError(
                "dense checkpoint lacks selected native Transformer block "
                f"{native_index} on the {weights} branch"
            )
        layer.load_state_dict(layer_state, strict=True)
        loaded_keys += len(layer_state)
    del state, template, transformer
    selected.requires_grad_(True)
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "model_config": str(model_config_path.resolve()),
        "weights": weights,
        "start_index": int(start_index),
        "count": int(count),
        "transformer_depth": int(depth),
        "loaded_state_keys": int(loaded_keys),
        "trainable_tensors": sum(1 for _ in selected.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in selected.parameters()
        ),
        "function_preserving_initialization": True,
    }
    return selected, report


class FrozenDenseDiTRenderer:
    """Lazy 300k dense DiT with exact Transfusion RF convention conversion."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.model_config_path = Path(str(self.config.get("model_config", "")))
        self.checkpoint_path = Path(str(self.config.get("checkpoint", "")))
        self.weights = str(self.config.get("weights", "ema")).lower()
        self.cfg_scale = float(self.config.get("cfg_scale", 1.0))
        self.rescale_cfg = bool(self.config.get("rescale_cfg", False))
        for path, label in (
            (self.model_config_path, "model_config"),
            (self.checkpoint_path, "checkpoint"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"dense DiT renderer {label}: {path}")
        if self.weights not in {"ema", "online"}:
            raise ValueError("dense DiT renderer weights must be 'ema' or 'online'")
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 1.0:
            raise ValueError("dense DiT renderer cfg_scale must be finite and >= 1")

        self._device: torch.device | None = None
        self._renderer = None
        self.load_report: dict[str, Any] = {}

    @staticmethod
    def native_integration_points(steps: int) -> int:
        """Translate native DiT Euler intervals to torchdiffeq time points."""

        if isinstance(steps, bool) or int(steps) < 1:
            raise ValueError("dense DiT sampling steps must be a positive integer")
        return int(steps) + 1

    @staticmethod
    def draw_native_noise(
        channels: int,
        frames: int,
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Draw the same channel-first seeded tensor as native DiT sampling."""

        if int(channels) <= 0 or int(frames) <= 0:
            raise ValueError("dense DiT noise dimensions must be positive")
        return torch.randn(
            (int(channels), int(frames)),
            device=device,
            dtype=dtype,
        )

    def native_time_schedule(
        self,
        steps: int,
        frames: int,
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return student times, native DiT sigmas, and exact Euler sizes."""

        renderer = self.ensure_loaded(device)
        from stable_audio_tools.inference.sampling import build_schedule

        dit_times = build_schedule(
            steps=int(steps),
            sigma_max=1.0,
            dist_shift=renderer.sampling_dist_shift,
            effective_seq_len=None,
            fallback_seq_len=int(frames),
            include_endpoint=True,
            device=device,
        ).float()
        student_times = 1.0 - dit_times
        if (
            student_times.ndim != 1
            or student_times.numel() != int(steps) + 1
            or not bool((student_times[1:] > student_times[:-1]).all())
        ):
            raise RuntimeError("dense DiT produced an invalid native time schedule")
        # Compute from the native descending schedule itself. Subtracting the
        # transformed student endpoints loses bits through cancellation and
        # no longer reproduces native Euler updates exactly.
        step_sizes = dit_times[:-1] - dit_times[1:]
        return student_times, dit_times, step_sizes

    @staticmethod
    def _prefix_state(
        state: Mapping[str, Tensor], prefix: str
    ) -> dict[str, Tensor]:
        return {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }

    @staticmethod
    def pad_channel_first(values: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        if not values:
            raise ValueError("cannot pad an empty latent batch")
        channels = int(values[0].shape[0])
        lengths = []
        for value in values:
            if value.ndim != 2 or int(value.shape[0]) != channels:
                raise ValueError(
                    "dense DiT renderer expects channel-first [C,T] latents"
                )
            lengths.append(int(value.shape[-1]))
        maximum = max(lengths)
        batch = values[0].new_zeros(len(values), channels, maximum)
        valid = torch.zeros(
            len(values), maximum, device=values[0].device, dtype=torch.bool
        )
        for index, (value, length) in enumerate(zip(values, lengths)):
            batch[index, :, :length] = value
            valid[index, :length] = True
        return batch, valid

    def ensure_loaded(self, device: torch.device):
        if self._renderer is not None:
            if self._device != device:
                raise RuntimeError(
                    f"dense DiT renderer is on {self._device}, requested {device}"
                )
            return self._renderer

        # This renderer is loaded lazily during Spatial-CoT inference. Model
        # constructors initialize temporary random weights before the exact
        # checkpoint is applied; without this guard, that invisible work moves
        # the caller's RNG cursor and changes the supposedly matched sampling
        # noise. Loading an immutable frozen prior must be RNG-transparent.
        fork_devices = []
        if device.type == "cuda":
            fork_devices = [
                torch.cuda.current_device()
                if device.index is None
                else int(device.index)
            ]
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            return self._load_uncached(device)

    def _load_uncached(self, device: torch.device):
        """Construct and restore the frozen renderer inside an RNG guard."""

        from stable_audio_tools.configuration import load_config
        from stable_audio_tools.models import create_model_from_config
        from stable_audio_tools.models.utils import load_ckpt_state_dict

        model_config = load_config(self.model_config_path)
        if model_config.get("model_type") != "diffusion_cond":
            raise ValueError("dense DiT renderer requires model_type='diffusion_cond'")
        diffusion_config = ((model_config.get("model") or {}).get("diffusion") or {})
        if diffusion_config.get("diffusion_objective") != "rectified_flow":
            raise ValueError("dense DiT renderer requires rectified_flow")

        renderer = create_model_from_config(model_config)
        # Joint training consumes already encoded latents.  The codec belongs
        # to the outer route and is loaded from its exact frozen checkpoint.
        renderer.pretransform = None
        state = load_ckpt_state_dict(str(self.checkpoint_path))
        core_prefix = (
            "diffusion_ema.ema_model."
            if self.weights == "ema"
            else "diffusion.model."
        )
        core_state = self._prefix_state(state, core_prefix)
        conditioner_state = self._prefix_state(state, "diffusion.conditioner.")
        if not core_state or not conditioner_state:
            raise RuntimeError(
                "dense DiT checkpoint lacks the requested core or conditioner"
            )
        renderer.model.load_state_dict(core_state, strict=True)
        renderer.conditioner.load_state_dict(conditioner_state, strict=True)
        del state, core_state, conditioner_state

        renderer.model.eval().requires_grad_(False).to(device)
        renderer.conditioner.eval().requires_grad_(False).to(device)
        self._renderer = renderer
        self._device = device
        self.load_report = {
            "checkpoint": str(self.checkpoint_path.resolve()),
            "model_config": str(self.model_config_path.resolve()),
            "weights": self.weights,
            "cfg_scale": self.cfg_scale,
            "rescale_cfg": self.rescale_cfg,
            "external_frozen_backbone": True,
        }
        print(f"Loaded frozen dense-DiT renderer: {self.load_report}", flush=True)
        return renderer

    @torch.no_grad()
    def prepare_conditioning(
        self,
        conditioning: Sequence[Mapping[str, Any]],
        *,
        device: torch.device,
    ) -> dict[str, Any]:
        renderer = self.ensure_loaded(device)
        dtype = next(renderer.model.parameters()).dtype
        # Native DiT prepares Qwen/number conditioning before entering its
        # BF16 sampler autocast. Spatial-CoT rendering is itself commonly
        # called from an outer autocast context, so make this boundary explicit
        # instead of silently changing the frozen prior's conditioning.
        context = (
            torch.autocast(device_type="cuda", enabled=False)
            if device.type == "cuda"
            else nullcontext()
        )
        with context:
            condition_tensors = renderer.conditioner(list(conditioning), device)
            condition_inputs = renderer.get_conditioning_inputs(condition_tensors)
        return {
            key: (
                value.to(dtype=dtype)
                if isinstance(value, Tensor) and value.is_floating_point()
                else value
            )
            for key, value in condition_inputs.items()
        }

    @torch.no_grad()
    def source_region_membership(
        self,
        conditioning: Sequence[Mapping[str, Any]],
        *,
        prepared_conditioning: Mapping[str, Any],
        max_sources: int,
        device: torch.device,
    ) -> Tensor:
        """Map exact caption spans onto the Dense cross-attention sequence."""

        renderer = self.ensure_loaded(device)
        conditioners = getattr(renderer.conditioner, "conditioners", None)
        if conditioners is None or "prompt" not in conditioners:
            raise RuntimeError("Dense renderer exposes no prompt conditioner")
        prompt_conditioner = conditioners["prompt"]
        tokenizer = getattr(prompt_conditioner, "tokenizer", None)
        max_length = int(getattr(prompt_conditioner, "max_length", 0))
        if tokenizer is None or max_length <= 0:
            raise RuntimeError("Dense prompt conditioner exposes no tokenizer")
        prompts = [row.get("prompt") for row in conditioning]
        source_regions = [row.get("_source_regions") for row in conditioning]
        if not prompts or any(
            not isinstance(prompt, str)
            or not prompt.strip()
            or not isinstance(regions, (list, tuple))
            or not regions
            for prompt, regions in zip(prompts, source_regions)
        ):
            raise ValueError(
                "Dense source-regional condition requires raw prompts plus "
                "exact source-region sidecars"
            )
        texts = [str(value) for value in prompts]
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = torch.as_tensor(encoded["offset_mapping"])
        attention_mask = torch.as_tensor(encoded["attention_mask"]).bool()
        from ..data.text_conditioning import build_source_region_ids

        source_ids = torch.stack(
            [
                (
                    build_source_region_ids(
                        texts[index],
                        offsets[index],
                        attention_mask[index],
                        source_regions[index],
                        max_sources=max_sources,
                    )
                    if source_regions[index]
                    else torch.zeros(offsets.shape[1], dtype=torch.long)
                )
                for index in range(len(prompts))
            ]
        )
        prompt_length = int(attention_mask.sum(dim=1).max().clamp_min(1).item())
        cross_attn = prepared_conditioning.get("cross_attn_cond")
        if not isinstance(cross_attn, Tensor) or cross_attn.ndim != 3:
            raise ValueError("prepared Dense conditioning lacks cross-attention tokens")
        if (
            int(cross_attn.shape[0]) != len(prompts)
            or int(cross_attn.shape[1]) < prompt_length
        ):
            raise ValueError(
                "Dense prompt membership does not align with prepared "
                f"conditioning {tuple(cross_attn.shape)}"
            )
        membership = torch.zeros(
            len(prompts),
            int(cross_attn.shape[1]),
            int(max_sources),
            device=device,
            dtype=cross_attn.dtype,
        )
        source_ids = source_ids[:, :prompt_length].to(device=device)
        for slot in range(int(max_sources)):
            membership[:, :prompt_length, slot] = source_ids.eq(slot + 1)
        return membership

    def compose_source_condition_residual(
        self,
        source_codes: Sequence[Tensor],
        conditioning: Sequence[Mapping[str, Any]],
        *,
        prepared_conditioning: Mapping[str, Any],
    ) -> Tensor:
        """Scatter source-slot codes only onto their matching caption spans."""

        if not source_codes:
            raise ValueError("source condition codes cannot be empty")
        shape = tuple(source_codes[0].shape)
        if (
            len(shape) != 2
            or any(tuple(value.shape) != shape for value in source_codes)
            or len(source_codes) != len(conditioning)
        ):
            raise ValueError(
                "source condition codes must be aligned [batch,sources,dim]"
            )
        codes = torch.stack(list(source_codes), dim=0)
        membership = self.source_region_membership(
            conditioning,
            prepared_conditioning=prepared_conditioning,
            max_sources=shape[0],
            device=codes.device,
        )
        cross_attn = prepared_conditioning.get("cross_attn_cond")
        assert isinstance(cross_attn, Tensor)
        if int(codes.shape[-1]) != int(cross_attn.shape[-1]):
            raise ValueError(
                "source condition code dimension must equal Dense condition "
                f"dimension {cross_attn.shape[-1]}, got {codes.shape[-1]}"
            )
        return torch.einsum(
            "bks,bsd->bkd",
            membership.to(codes),
            codes,
        )

    def predict_velocity(
        self,
        noised: Tensor,
        student_times: Tensor,
        conditioning: Sequence[Mapping[str, Any]] | None,
        valid_mask: Tensor,
        *,
        prepared_conditioning: Mapping[str, Any] | None = None,
        dit_times: Tensor | None = None,
        external_cross_attn_residual: Tensor | None = None,
        external_cross_attn_kv_lora_down: Tensor | None = None,
        external_cross_attn_kv_lora_up: Tensor | None = None,
        external_cross_attn_kv_lora_scale: float = 1.0,
        external_cross_attn_kv_lora_start_index: int = 0,
        external_layer_replacements: nn.ModuleList | None = None,
        external_layer_replacement_start_index: int = 0,
    ) -> Tensor:
        """Return velocity in the Transfusion ``noise -> data`` convention."""

        device = noised.device
        renderer = self.ensure_loaded(device)
        dtype = next(renderer.model.parameters()).dtype
        if prepared_conditioning is None:
            if conditioning is None:
                raise ValueError(
                    "dense DiT renderer requires raw or prepared conditioning"
                )
            condition_inputs = self.prepare_conditioning(
                conditioning, device=device
            )
        else:
            condition_inputs = dict(prepared_conditioning)
        # DiT: x_t=(1-t)data+t*noise, v=noise-data.
        # CoT: x_t=t*data+(1-t)noise, v=data-noise.
        if dit_times is None:
            dit_times = (1.0 - student_times.float()).clamp(0.0, 1.0)
        else:
            dit_times = torch.as_tensor(
                dit_times, device=device, dtype=torch.float32
            ).reshape(student_times.shape)
        if external_cross_attn_residual is not None:
            cross_attn_cond = condition_inputs.get("cross_attn_cond")
            if not isinstance(cross_attn_cond, Tensor):
                raise ValueError(
                    "dense condition-regional input requires cross-attention "
                    "conditioning"
                )
            if tuple(external_cross_attn_residual.shape) != tuple(
                cross_attn_cond.shape
            ):
                raise ValueError(
                    "dense condition-regional input must match prepared "
                    f"conditioning {tuple(cross_attn_cond.shape)}, got "
                    f"{tuple(external_cross_attn_residual.shape)}"
                )
        kv_lora_values = (
            external_cross_attn_kv_lora_down,
            external_cross_attn_kv_lora_up,
        )
        if any(value is not None for value in kv_lora_values):
            if not all(value is not None for value in kv_lora_values):
                raise ValueError(
                    "dense native K/V LoRA requires down and up weights together"
                )
            if (
                external_cross_attn_kv_lora_down.ndim != 3
                or external_cross_attn_kv_lora_up.ndim != 3
                or int(external_cross_attn_kv_lora_down.shape[0]) <= 0
                or int(external_cross_attn_kv_lora_down.shape[0])
                != int(external_cross_attn_kv_lora_up.shape[0])
                or int(external_cross_attn_kv_lora_down.shape[2]) <= 0
                or int(external_cross_attn_kv_lora_down.shape[2])
                != int(external_cross_attn_kv_lora_up.shape[1])
            ):
                raise ValueError(
                    "dense native K/V LoRA weights must be "
                    "[layers,context_dim,rank] and [layers,rank,to_kv_out]"
                )
            if (
                isinstance(external_cross_attn_kv_lora_scale, bool)
                or not isinstance(
                    external_cross_attn_kv_lora_scale, (int, float)
                )
                or not math.isfinite(float(external_cross_attn_kv_lora_scale))
                or float(external_cross_attn_kv_lora_scale) <= 0.0
            ):
                raise ValueError(
                    "dense native K/V LoRA scale must be finite and positive"
                )
            if (
                isinstance(external_cross_attn_kv_lora_start_index, bool)
                or not isinstance(
                    external_cross_attn_kv_lora_start_index, int
                )
                or external_cross_attn_kv_lora_start_index < 0
            ):
                raise ValueError(
                    "dense native K/V LoRA start index must be non-negative"
                )
        if external_layer_replacements is not None:
            if (
                not isinstance(external_layer_replacements, nn.ModuleList)
                or len(external_layer_replacements) <= 0
                or isinstance(external_layer_replacement_start_index, bool)
                or not isinstance(external_layer_replacement_start_index, int)
                or external_layer_replacement_start_index < 0
            ):
                raise ValueError(
                    "dense native layer replacements require a non-empty "
                    "ModuleList and a non-negative start index"
                )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        gradient_context = (
            torch.enable_grad()
            if (
                external_cross_attn_residual is not None
                and external_cross_attn_residual.requires_grad
            )
            or (
                external_cross_attn_kv_lora_down is not None
                and external_cross_attn_kv_lora_down.requires_grad
            )
            or (
                external_cross_attn_kv_lora_up is not None
                and external_cross_attn_kv_lora_up.requires_grad
            )
            or (
                external_layer_replacements is not None
                and any(
                    parameter.requires_grad
                    for parameter in external_layer_replacements.parameters()
                )
            )
            else torch.no_grad()
        )
        with gradient_context, autocast:
            velocity = renderer.model(
                noised.to(dtype=dtype),
                dit_times,
                cfg_scale=self.cfg_scale,
                cfg_dropout_prob=0.0,
                batch_cfg=True,
                rescale_cfg=self.rescale_cfg,
                padding_mask=valid_mask,
                external_cross_attn_residual=external_cross_attn_residual,
                external_cross_attn_kv_lora_down=(
                    external_cross_attn_kv_lora_down
                ),
                external_cross_attn_kv_lora_up=(
                    external_cross_attn_kv_lora_up
                ),
                external_cross_attn_kv_lora_scale=(
                    external_cross_attn_kv_lora_scale
                ),
                external_cross_attn_kv_lora_start_index=(
                    external_cross_attn_kv_lora_start_index
                ),
                external_layer_replacements=external_layer_replacements,
                external_layer_replacement_start_index=(
                    external_layer_replacement_start_index
                ),
                **condition_inputs,
            )
        return -velocity


__all__ = [
    "FrozenDenseDiTRenderer",
    "build_trainable_dense_dit_late_blocks",
]
