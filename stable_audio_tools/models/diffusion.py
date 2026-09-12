import torch
from torch import nn
from torch.nn import functional as F
from functools import partial, reduce
import numpy as np
import typing as tp
import random

from .blocks import ResConvBlock, FourierFeatures, Upsample1d, Upsample1d_2, Downsample1d, Downsample1d_2, SelfAttention1d, SkipBlock, expand_to_planes
from .conditioners import MultiConditioner, create_multi_conditioner_from_conditioning_config
from .dit import DiffusionTransformer
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from .transformer import ContinuousTransformer
from ..inference.generation import generate_diffusion_cond
from ..inference.sampling import FluxDistributionShift, DistributionShift, LogSNRShift, IdentityDistributionShift

from time import time

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep

class DiffusionModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, t, **kwargs):
        raise NotImplementedError()

class DiffusionModelWrapper(nn.Module):
    def __init__(
                self,
                model: DiffusionModel,
                io_channels,
                sample_size,
                sample_rate,
                min_input_length,
                pretransform: tp.Optional[Pretransform] = None,
    ):
        super().__init__()
        self.io_channels = io_channels
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.min_input_length = min_input_length

        self.model = model

        if pretransform is not None:
            self.pretransform = pretransform
        else:
            self.pretransform = None

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

class ConditionedDiffusionModel(nn.Module):
    def __init__(self,
                *args,
                supports_cross_attention: bool = False,
                supports_input_concat: bool = False,
                supports_global_cond: bool = False,
                supports_prepend_cond: bool = False,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond

    def forward(self,
                x: torch.Tensor,
                t: torch.Tensor,
                cross_attn_cond: torch.Tensor = None,
                cross_attn_mask: torch.Tensor = None,
                input_concat_cond: torch.Tensor = None,
                local_add_cond: torch.Tensor = None,
                global_embed: torch.Tensor = None,
                prepend_cond: torch.Tensor = None,
                prepend_cond_mask: torch.Tensor = None,
                cfg_scale: float = 1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                **kwargs):
        raise NotImplementedError()

class ConditionedDiffusionModelWrapper(nn.Module):
    """
    A diffusion model that takes in conditioning
    """
    def __init__(
            self,
            model: ConditionedDiffusionModel,
            conditioner: MultiConditioner,
            io_channels,
            sample_rate,
            min_input_length: int,
            diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
            distribution_shift_options = None,
            sampling_distribution_shift_options = None,
            mask_padding_attention: bool = False,
            use_effective_length_for_schedule: bool = False,
            pretransform: tp.Optional[Pretransform] = None,
            cross_attn_cond_ids: tp.List[str] = [],
            global_cond_ids: tp.List[str] = [],
            input_concat_ids: tp.List[str] = [],
            local_add_cond_ids: tp.List[str] = [],
            modular_local_cond_ids: tp.List[str] = [],
            prepend_cond_ids: tp.List[str] = [],
            gate: bool = False,
            gate_type: tp.Optional[str] = None,
            gate_type_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
            maf_cond_ids: tp.Optional[tp.List[str]] = None,
            ):
        super().__init__()

        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.diffusion_objective = diffusion_objective
        self.pretransform = pretransform
        self.cross_attn_cond_ids = cross_attn_cond_ids
        self.global_cond_ids = global_cond_ids
        self.input_concat_ids = input_concat_ids
        self.local_add_cond_ids = local_add_cond_ids
        self.modular_local_cond_ids = modular_local_cond_ids
        self.prepend_cond_ids = prepend_cond_ids
        self.min_input_length = min_input_length
        self.mask_padding_attention = mask_padding_attention
        self.use_effective_length_for_schedule = use_effective_length_for_schedule
        self.gate = gate
        self.gate_type = gate_type
        self.gate_type_config = gate_type_config or {}
        self.maf_cond_ids = maf_cond_ids or ["video_prompt", "prompt", "audio_prompt"]
        self.maf_block = None
        if gate and gate_type == "MAF":
            from .multimodal_adaptive_fusion import MAF_Block
            cond_dim = self.gate_type_config.get("cond_dim", 768)
            self.maf_block = MAF_Block(
                dim=cond_dim,
                num_experts_per_modality=self.gate_type_config.get("num_experts_per_modality", 64),
                num_heads=self.gate_type_config.get("num_heads", 12),
                num_fusion_layers=self.gate_type_config.get("num_fusion_layers", 4),
            )

        self.dist_shift = None
        if distribution_shift_options is not None:
            self.dist_shift = self._create_dist_shift(distribution_shift_options)

        # Sampling dist_shift: separate config for inference-time schedule
        if sampling_distribution_shift_options is not None:
            self.sampling_dist_shift = self._create_dist_shift(sampling_distribution_shift_options)
        else:
            # Default: seq_len-invariant LogSNR shift matching legacy log_snr_sampling=True
            self.sampling_dist_shift = LogSNRShift(rate=0, anchor_logsnr=-6.2, logsnr_end=2.0)

    def load_pretrained_route_state_dict(
        self,
        state_dict: tp.Dict[str, torch.Tensor],
        *,
        prefer_ema: bool = True,
        source_model_config: tp.Optional[dict] = None,
        source_text_conditioner_ema_names: tp.Optional[tp.Sequence[str]] = None,
        source_conditioner_ema_names: tp.Optional[tp.Sequence[str]] = None,
    ) -> tp.Dict[str, tp.Any]:
        """Warm-start a conditional diffusion route without trainer state.

        Lightning checkpoints prefix online model tensors with ``diffusion.``;
        the DiT EMA and trainable-conditioner EMA use two different layouts.
        Loading by exact target name and shape prevents a superficially
        successful warm-start from silently leaving the model random.  New route
        parameters (for example the ordered transcript projection) intentionally
        retain their destination initialization.
        """

        del source_text_conditioner_ema_names
        nested = state_dict.get("state_dict") if isinstance(state_dict, dict) else None
        if isinstance(nested, dict):
            state_dict = nested
        if not isinstance(state_dict, dict):
            raise TypeError("diffusion route warm-start expects a state_dict mapping")

        conditioner_ema_indices = {
            str(name): index
            for index, name in enumerate(source_conditioner_ema_names or ())
        }
        target_state = self.state_dict()
        loaded_from: tp.Dict[str, int] = {}
        missing: tp.List[str] = []
        shape_mismatches: tp.List[tp.Dict[str, tp.Any]] = []
        partial_expansions: tp.List[tp.Dict[str, tp.Any]] = []
        semantic_role_expansions: tp.List[tp.Dict[str, tp.Any]] = []
        transformer_loaded = 0

        source_diffusion_config: tp.Dict[str, tp.Any] = {}
        if isinstance(source_model_config, dict):
            source_diffusion_config = dict(
                source_model_config.get("model", {}).get("diffusion", {})
            )
        source_dit_config = dict(source_diffusion_config.get("config", {}))
        source_io_channels = int(
            source_dit_config.get(
                "io_channels",
                source_model_config.get("model", {}).get("io_channels", -1)
                if isinstance(source_model_config, dict)
                else -1,
            )
        )
        source_input_concat_dim = int(
            source_dit_config.get("input_concat_dim", 0)
        )
        source_objective = source_diffusion_config.get("diffusion_objective")
        target_input_concat_dim = int(
            getattr(getattr(self.model, "model", None), "input_concat_dim", 0)
        )
        source_total_input_dim = source_io_channels + source_input_concat_dim
        target_total_input_dim = int(self.io_channels) + target_input_concat_dim
        allow_input_suffix_expansion = (
            source_io_channels == int(self.io_channels)
            and source_input_concat_dim >= 0
            and target_input_concat_dim > source_input_concat_dim
            and source_objective == self.diffusion_objective
        )

        def copy_expanded_input_prefix(
            target_name: str,
            target_value: torch.Tensor,
            source_value: torch.Tensor,
        ) -> tp.Optional[str]:
            """Preserve a trained frame input while appending zeroed channels.

            Both the original ScenePlan 4+4 upcycle and Transfusion Editing use
            strict suffix expansion.  P10 Editing changes the per-frame input
            from ``[z_t:64, plan:256]`` to
            ``[z_t:64, plan:256, z_source:64]``.  Copying the complete trained
            prefix and zero-initializing only the suffix keeps the pretrained
            flow field bitwise unchanged at step zero.  No reordering, interior
            insertion, or other shape mismatch is adapted.
            """

            if not allow_input_suffix_expansion:
                return None
            if target_name == "model.model.transformer.project_in.weight":
                if (
                    target_value.ndim == 2
                    and source_value.ndim == 2
                    and int(target_value.shape[0]) == int(source_value.shape[0])
                    and int(source_value.shape[1]) == source_total_input_dim
                    and int(target_value.shape[1]) == target_total_input_dim
                ):
                    target_value.zero_()
                    target_value[:, :source_total_input_dim].copy_(
                        source_value.to(
                            device=target_value.device,
                            dtype=target_value.dtype,
                        )
                    )
                    return "linear_input_trained_prefix"
            if target_name == "model.model.preprocess_conv.weight":
                if (
                    target_value.ndim == 3
                    and source_value.ndim == 3
                    and tuple(source_value.shape)
                    == (source_total_input_dim, source_total_input_dim, 1)
                    and tuple(target_value.shape)
                    == (target_total_input_dim, target_total_input_dim, 1)
                ):
                    target_value.zero_()
                    target_value[
                        :source_total_input_dim, :source_total_input_dim
                    ].copy_(
                        source_value.to(
                            device=target_value.device,
                            dtype=target_value.dtype,
                        )
                    )
                    return "conv1x1_trained_prefix_block"
            return None

        def copy_expanded_semantic_roles(
            target_name: str,
            target_value: torch.Tensor,
        ) -> tp.Optional[tp.Tuple[str, str, str]]:
            """Lift the checkpoint's cue/transcript roles into source-local rows.

            The FOA prior used a three-row caption-region table with raw roles
            ``0=ordinary``, ``1=event/speaker cue`` and ``2=quoted speech``.
            ScenePlan 4+4 keeps the latter two semantics but refines each into
            persistent source ids 1..4.  Replicating the learned cue or speech
            row into those four destination rows preserves the trained text
            signal while allowing the rows to separate during fine-tuning.
            Unknown (-1) and known-non-role (0) deliberately start at zero.
            """

            if not allow_input_suffix_expansion or source_input_concat_dim != 0:
                return None
            role_by_target = {
                "conditioner.conditioners.prompt.event_role_embed.weight": (
                    1,
                    "event_role_rows_from_caption_cue",
                ),
                "conditioner.conditioners.prompt.speech_role_embed.weight": (
                    2,
                    "speech_role_rows_from_caption_quote",
                ),
            }
            role = role_by_target.get(target_name)
            if role is None or target_value.ndim != 2:
                return None
            source_row, mode = role
            source_parameter_name = (
                "conditioners.prompt.caption_region_embed.weight"
            )
            candidates: tp.List[tp.Tuple[str, str]] = []
            source_index = conditioner_ema_indices.get(source_parameter_name)
            if prefer_ema and source_index is not None:
                candidates.append(
                    (
                        "conditioner_ema",
                        f"conditioner_ema.shadow_{source_index:05d}",
                    )
                )
            candidates.extend(
                (
                    (
                        "online",
                        "diffusion.conditioner." + source_parameter_name,
                    ),
                    ("plain", "conditioner." + source_parameter_name),
                )
            )
            for source_kind, source_name in candidates:
                source_value = state_dict.get(source_name)
                if not torch.is_tensor(source_value):
                    continue
                if (
                    source_value.ndim != 2
                    or int(source_value.shape[0]) < 3
                    or int(source_value.shape[1]) != int(target_value.shape[1])
                    or int(target_value.shape[0]) != 6
                ):
                    continue
                target_value.zero_()
                target_value[2:6].copy_(
                    source_value[source_row]
                    .to(device=target_value.device, dtype=target_value.dtype)
                    .unsqueeze(0)
                    .expand(4, -1)
                )
                return source_kind, source_name, mode
            return None

        with torch.no_grad():
            for target_name, target_value in target_state.items():
                candidates: tp.List[tp.Tuple[str, str]] = []
                if prefer_ema and target_name.startswith("model."):
                    candidates.append(
                        (
                            "dit_ema",
                            "diffusion_ema.ema_model."
                            + target_name[len("model.") :],
                        )
                    )
                if prefer_ema and target_name.startswith("conditioner."):
                    conditioner_name = target_name[len("conditioner.") :]
                    shadow_index = conditioner_ema_indices.get(conditioner_name)
                    if shadow_index is not None:
                        candidates.append(
                            (
                                "conditioner_ema",
                                f"conditioner_ema.shadow_{shadow_index:05d}",
                            )
                        )
                candidates.extend(
                    (
                        ("online", "diffusion." + target_name),
                        ("plain", target_name),
                    )
                )

                selected = None
                mismatched_candidates: tp.List[
                    tp.Tuple[str, str, torch.Tensor]
                ] = []
                for source_kind, source_name in candidates:
                    source_value = state_dict.get(source_name)
                    if not torch.is_tensor(source_value):
                        continue
                    if tuple(source_value.shape) != tuple(target_value.shape):
                        shape_mismatches.append(
                            {
                                "target": target_name,
                                "source": source_name,
                                "target_shape": tuple(target_value.shape),
                                "source_shape": tuple(source_value.shape),
                            }
                        )
                        mismatched_candidates.append(
                            (source_kind, source_name, source_value)
                        )
                        continue
                    selected = (source_kind, source_value)
                    break

                if selected is None:
                    role_expansion = copy_expanded_semantic_roles(
                        target_name, target_value
                    )
                    if role_expansion is not None:
                        source_kind, source_name, expansion = role_expansion
                        loaded_key = f"{source_kind}_semantic_role_expansion"
                        loaded_from[loaded_key] = (
                            loaded_from.get(loaded_key, 0) + 1
                        )
                        semantic_role_expansions.append(
                            {
                                "target": target_name,
                                "source": source_name,
                                "mode": expansion,
                                "source_ids": [1, 2, 3, 4],
                            }
                        )
                        continue
                    expanded = None
                    for source_kind, source_name, source_value in mismatched_candidates:
                        expansion = copy_expanded_input_prefix(
                            target_name, target_value, source_value
                        )
                        if expansion is not None:
                            expanded = (source_kind, source_name, expansion)
                            break
                    if expanded is not None:
                        source_kind, source_name, expansion = expanded
                        loaded_key = f"{source_kind}_expanded_input"
                        loaded_from[loaded_key] = loaded_from.get(loaded_key, 0) + 1
                        partial_expansions.append(
                            {
                                "target": target_name,
                                "source": source_name,
                                "mode": expansion,
                                "audio_channels": int(self.io_channels),
                                "source_input_concat_channels": source_input_concat_dim,
                                "target_input_concat_channels": target_input_concat_dim,
                                "copied_prefix_channels": source_total_input_dim,
                                "zero_initialized_suffix_channels": (
                                    target_total_input_dim - source_total_input_dim
                                ),
                            }
                        )
                        if target_name.startswith("model."):
                            transformer_loaded += 1
                        continue
                    missing.append(target_name)
                    continue
                source_kind, source_value = selected
                target_value.copy_(
                    source_value.to(
                        device=target_value.device,
                        dtype=target_value.dtype,
                    )
                )
                loaded_from[source_kind] = loaded_from.get(source_kind, 0) + 1
                if target_name.startswith("model."):
                    transformer_loaded += 1

        if transformer_loaded == 0:
            raise RuntimeError(
                "diffusion warm-start loaded no DiT tensors; checkpoint is incompatible"
            )
        return {
            "loaded": sum(loaded_from.values()),
            "target_total": len(target_state),
            "transformer_loaded": transformer_loaded,
            "destination_route": "conditioned_diffusion",
            "loaded_from": loaded_from,
            "modality_mapping": (
                (
                    "exact_name_and_shape_plus_audio_prefix_input_expansion"
                    if source_input_concat_dim == 0
                    else "exact_name_and_shape_plus_trained_prefix_input_expansion"
                )
                if partial_expansions
                else "exact_name_and_shape"
            ),
            "text_rows_loaded": 0,
            "shape_mismatches": shape_mismatches,
            "partial_expansions": partial_expansions,
            "semantic_role_expansions": semantic_role_expansions,
            "missing": missing,
            "prefer_ema": bool(prefer_ema),
        }

    @staticmethod
    def _create_dist_shift(options: dict):
        """Create a distribution shift object from config options."""
        dist_shift_type = options.get("type", "full")
        dist_shift_kwargs = {k: v for k, v in options.items() if k != "type"}
        if dist_shift_type == "none":
            return IdentityDistributionShift()
        elif dist_shift_type == "flux":
            return FluxDistributionShift(**dist_shift_kwargs)
        elif dist_shift_type == "full":
            return DistributionShift(**dist_shift_kwargs)
        elif dist_shift_type == "logsnr":
            return LogSNRShift(**dist_shift_kwargs)
        else:
            raise ValueError(f"Unknown distribution shift type: {dist_shift_type}. Expected 'none', 'flux', 'full', or 'logsnr'.")     

    def get_conditioning_inputs(self, conditioning_tensors: tp.Dict[str, tp.Any], negative=False):
        cross_attention_input = None
        cross_attention_masks = None
        cross_attention_aux = None
        global_cond = None
        input_concat_cond = None
        input_concat_aux = None
        prepend_cond = None
        prepend_cond_mask = None
        local_add_cond = None
        modular_local_cond = None

        if len(self.cross_attn_cond_ids) > 0:
            # Concatenate all cross-attention inputs over the sequence dimension
            # Assumes that the cross-attention inputs are of shape (batch, seq, channels)
            cross_attention_input = []
            cross_attention_masks = []

            for key in self.cross_attn_cond_ids:
                conditioned = conditioning_tensors[key]
                if not isinstance(conditioned, (list, tuple)) or len(conditioned) not in {
                    2,
                    3,
                }:
                    raise ValueError(
                        f"cross-attention conditioner {key!r} must return "
                        "(embedding, mask[, auxiliary])"
                    )
                cross_attn_in, cross_attn_mask = conditioned[:2]
                auxiliary = conditioned[2] if len(conditioned) == 3 else None
                if auxiliary is not None:
                    if len(self.cross_attn_cond_ids) != 1:
                        raise ValueError(
                            "ScenePlan frame/text auxiliary currently requires "
                            "one cross-attention conditioner"
                        )
                    if not isinstance(auxiliary, dict):
                        raise TypeError(
                            "cross-attention auxiliary must be a dictionary"
                        )
                    cross_attention_aux = auxiliary

                # Add sequence dimension if it's not there
                if len(cross_attn_in.shape) == 2:
                    cross_attn_in = cross_attn_in.unsqueeze(1)
                    cross_attn_mask = cross_attn_mask.unsqueeze(1)

                cross_attention_input.append(cross_attn_in)
                cross_attention_masks.append(cross_attn_mask)

            if self.gate and self.gate_type == "MAF" and self.maf_block is not None:
                by_id = {k: (cross_attention_input[i], cross_attention_masks[i])
                         for i, k in enumerate(self.cross_attn_cond_ids)}
                missing = [k for k in self.maf_cond_ids if k not in by_id]
                if missing:
                    raise ValueError(
                        f"MAF maf_cond_ids {missing} not in cross_attention_cond_ids {self.cross_attn_cond_ids}"
                    )
                v_in, _ = by_id[self.maf_cond_ids[0]]
                t_in, _ = by_id[self.maf_cond_ids[1]]
                a_in, _ = by_id[self.maf_cond_ids[2]]
                refined = self.maf_block(v_in, t_in, a_in)
                cross_attention_input = [
                    refined["video"], refined["text"], refined["audio"],
                ]
                cross_attention_masks = [
                    cross_attention_masks[self.cross_attn_cond_ids.index(k)]
                    for k in self.maf_cond_ids
                ]

            cross_attention_input = torch.cat(cross_attention_input, dim=1)
            cross_attention_masks = torch.cat(cross_attention_masks, dim=1)

        if len(self.global_cond_ids) > 0:
            # Concatenate all global conditioning inputs over the channel dimension
            # Assumes that the global conditioning inputs are of shape (batch, channels)
            global_conds = []
            for key in self.global_cond_ids:
                global_cond_input = conditioning_tensors[key][0]

                global_conds.append(global_cond_input)

            # Concatenate over the channel dimension
            global_cond = torch.cat(global_conds, dim=-1)

            if len(global_cond.shape) == 3:
                global_cond = global_cond.squeeze(1)

        if len(self.input_concat_ids) > 0:
            # Concatenate all input concat conditioning inputs over the channel dimension
            # Assumes that the input concat conditioning inputs are of shape (batch, channels, seq)
            input_concat_values = []
            for key in self.input_concat_ids:
                conditioned = conditioning_tensors[key]
                if not isinstance(conditioned, (list, tuple)) or len(conditioned) not in {
                    2,
                    3,
                }:
                    raise ValueError(
                        f"input-concat conditioner {key!r} must return "
                        "(embedding, mask[, auxiliary])"
                    )
                input_concat_values.append(conditioned[0])
                auxiliary = conditioned[2] if len(conditioned) == 3 else None
                if auxiliary is not None:
                    if len(self.input_concat_ids) != 1:
                        raise ValueError(
                            "ScenePlan frame auxiliary currently requires one "
                            "input-concat conditioner"
                        )
                    if not isinstance(auxiliary, dict):
                        raise TypeError("input-concat auxiliary must be a dictionary")
                    input_concat_aux = auxiliary
            input_concat_cond = torch.cat(input_concat_values, dim=1)

        if len(self.local_add_cond_ids) > 0:
            # Concatenate all local conditioning inputs over the channel dimension
            # Assumes that the local conditioning inputs are of shape (batch, channels, seq)
            local_add_cond = torch.cat([conditioning_tensors[key][0] for key in self.local_add_cond_ids], dim=1)

        if len(self.modular_local_cond_ids) > 0:
            # Keep modular local conditioning as a dict of tensors (not concatenated)
            # Each tensor is of shape (batch, channels, seq)
            modular_local_cond = {}
            for key in self.modular_local_cond_ids:
                if key in conditioning_tensors:
                    modular_local_cond[key] = conditioning_tensors[key][0]
            # Only set if we have any conditioning
            if len(modular_local_cond) == 0:
                modular_local_cond = None

        if len(self.prepend_cond_ids) > 0:
            # Concatenate all prepend conditioning inputs over the sequence dimension
            # Assumes that the prepend conditioning inputs are of shape (batch, seq, channels)
            prepend_conds = []
            prepend_cond_masks = []

            for key in self.prepend_cond_ids:
                prepend_cond_input, prepend_cond_mask = conditioning_tensors[key]
                prepend_conds.append(prepend_cond_input)
                prepend_cond_masks.append(prepend_cond_mask)

            prepend_cond = torch.cat(prepend_conds, dim=1)
            prepend_cond_mask = torch.cat(prepend_cond_masks, dim=1)

        if negative:
            return {
                "negative_cross_attn_cond": cross_attention_input,
                "negative_cross_attn_mask": cross_attention_masks,
                "negative_cross_attn_aux": cross_attention_aux,
                "negative_global_cond": global_cond,
                "negative_input_concat_cond": input_concat_cond,
                "negative_input_concat_aux": input_concat_aux,
            }
        else:
            return {
                "cross_attn_cond": cross_attention_input,
                "cross_attn_mask": cross_attention_masks,
                "cross_attn_aux": cross_attention_aux,
                "global_cond": global_cond,
                "input_concat_cond": input_concat_cond,
                "input_concat_aux": input_concat_aux,
                "local_add_cond": local_add_cond,
                "modular_local_cond": modular_local_cond,
                "prepend_cond": prepend_cond,
                "prepend_cond_mask": prepend_cond_mask
            }

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: tp.Dict[str, tp.Any], **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)

    def generate(self, *args, **kwargs):
        return generate_diffusion_cond(self, *args, **kwargs)

class UNetCFG1DWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=True, supports_input_concat=True)

        from .adp import UNetCFG1d

        self.model = UNetCFG1d(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self,
                x,
                t,
                cross_attn_cond=None,
                cross_attn_mask=None,
                input_concat_cond=None,
                global_cond=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                negative_global_cond=None,
                negative_input_concat_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                local_add_cond=None,
                **kwargs):
        p = Profiler()

        p.tick("start")

        channels_list = None
        if input_concat_cond is not None:
            channels_list = [input_concat_cond]

        outputs = self.model(
            x,
            t,
            embedding=cross_attn_cond,
            embedding_mask=cross_attn_mask,
            features=global_cond,
            channels_list=channels_list,
            embedding_scale=cfg_scale,
            embedding_mask_proba=cfg_dropout_prob,
            batch_cfg=batch_cfg,
            rescale_cfg=rescale_cfg,
            negative_embedding=negative_cross_attn_cond,
            negative_embedding_mask=negative_cross_attn_mask,
            **kwargs)

        p.tick("UNetCFG1D forward")

        #print(f"Profiler: {p}")
        return outputs

class UNet1DCondWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=False, supports_global_cond=True, supports_input_concat=True)

        from .adp import UNet1d

        self.model = UNet1d(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self,
                x,
                t,
                input_concat_cond=None,
                local_add_cond=None,
                global_cond=None,
                cross_attn_cond=None,
                cross_attn_mask=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                negative_global_cond=None,
                negative_input_concat_cond=None,
                **kwargs):

        channels_list = None
        if input_concat_cond is not None:

            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            channels_list = [input_concat_cond]

        outputs = self.model(
            x,
            t,
            features=global_cond,
            channels_list=channels_list,
            **kwargs)

        return outputs

class UNet1DUncondWrapper(DiffusionModel):
    def __init__(
        self,
        in_channels,
        *args,
        **kwargs
    ):
        super().__init__()

        from .adp import UNet1d

        self.model = UNet1d(in_channels=in_channels, *args, **kwargs)

        self.io_channels = in_channels

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

class DAU1DCondWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=False, supports_global_cond=False, supports_input_concat=True)

        self.model = DiffusionAttnUnet1D(*args, **kwargs)

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self,
                x,
                t,
                input_concat_cond=None,
                local_add_cond=None,
                cross_attn_cond=None,
                cross_attn_mask=None,
                global_cond=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = False,
                rescale_cfg: bool = False,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                negative_global_cond=None,
                negative_input_concat_cond=None,
                prepend_cond=None,
                **kwargs):

        return self.model(x, t, cond = input_concat_cond)

class DiffusionAttnUnet1D(nn.Module):
    def __init__(
        self,
        io_channels = 2,
        depth=14,
        n_attn_layers = 6,
        channels = [128, 128, 256, 256] + [512] * 10,
        cond_dim = 0,
        cond_noise_aug = False,
        kernel_size = 5,
        learned_resample = False,
        strides = [2] * 13,
        conv_bias = True,
        use_snake = False
    ):
        super().__init__()

        self.cond_noise_aug = cond_noise_aug

        self.io_channels = io_channels

        if self.cond_noise_aug:
            self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_embed = FourierFeatures(1, 16)

        attn_layer = depth - n_attn_layers

        strides = [1] + strides

        block = nn.Identity()

        conv_block = partial(ResConvBlock, kernel_size=kernel_size, conv_bias = conv_bias, use_snake=use_snake)

        for i in range(depth, 0, -1):
            c = channels[i - 1]
            stride = strides[i-1]
            if stride > 2 and not learned_resample:
                raise ValueError("Must have stride 2 without learned resampling")

            if i > 1:
                c_prev = channels[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d_2(c_prev, c_prev, stride) if (learned_resample or stride == 1) else Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != depth else c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SelfAttention1d(
                        c, c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SelfAttention1d(c_prev, c_prev //
                                    32) if add_attn else nn.Identity(),
                    Upsample1d_2(c_prev, c_prev, stride) if learned_resample else Upsample1d(kernel="cubic")
                )
            else:
                cond_embed_dim = 16 if not self.cond_noise_aug else 32
                block = nn.Sequential(
                    conv_block((io_channels + cond_dim) + cond_embed_dim, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, io_channels, is_last=True),
                )
        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, x, t, cond=None, cond_aug_scale=None):

        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)

        inputs = [x, timestep_embed]

        if cond is not None:
            if cond.shape[2] != x.shape[2]:
                cond = F.interpolate(cond, (x.shape[2], ), mode='linear', align_corners=False)

            if self.cond_noise_aug:
                # Get a random number between 0 and 1, uniformly sampled
                if cond_aug_scale is None:
                    aug_level = self.rng.draw(cond.shape[0])[:, 0].to(cond)
                else:
                    aug_level = torch.tensor([cond_aug_scale]).repeat([cond.shape[0]]).to(cond)

                # Add noise to the conditioning signal
                cond = cond + torch.randn_like(cond) * aug_level[:, None, None]

                # Get embedding for noise cond level, reusing timestamp_embed
                aug_level_embed = expand_to_planes(self.timestep_embed(aug_level[:, None]), x.shape)

                inputs.append(aug_level_embed)

            inputs.append(cond)

        outputs = self.net(torch.cat(inputs, dim=1))

        return outputs

class DiTWrapper(ConditionedDiffusionModel):
    def __init__(
        self,
        diffusion_objective: str,
        *args,
        **kwargs
    ):
        super().__init__(supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False)

        self.diffusion_objective = diffusion_objective

        self.model = DiffusionTransformer(diffusion_objective=diffusion_objective, *args, **kwargs)

    def forward(self,
                x,
                t,
                cross_attn_cond=None,
                cross_attn_mask=None,
                cross_attn_aux=None,
                negative_cross_attn_cond=None,
                negative_cross_attn_mask=None,
                negative_cross_attn_aux=None,
                input_concat_cond=None,
                input_concat_aux=None,
                local_add_cond=None,
                negative_input_concat_cond=None,
                negative_input_concat_aux=None,
                global_cond=None,
                negative_global_cond=None,
                prepend_cond=None,
                prepend_cond_mask=None,
                cfg_scale=1.0,
                cfg_dropout_prob: float = 0.0,
                batch_cfg: bool = True,
                rescale_cfg: bool = False,
                scale_phi: float = 0.0,
                **kwargs):

        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
        #assert negative_input_concat_cond is None, "negative_input_concat_cond is not supported for DiTWrapper"

        # Keep direct DiTWrapper callers consistent with sample_diffusion.  The
        # old wrapper accepted rescale_cfg but discarded it, which made
        # rescale_cfg=True a no-op unless a separate scale_phi happened to be
        # supplied as well.  An explicit non-zero scale_phi still takes
        # precedence.
        if rescale_cfg and scale_phi == 0.0:
            scale_phi = 0.4

        return self.model(
            x,
            t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_mask,
            cross_attn_aux=cross_attn_aux,
            negative_cross_attn_cond=negative_cross_attn_cond,
            negative_cross_attn_mask=negative_cross_attn_mask,
            negative_cross_attn_aux=negative_cross_attn_aux,
            input_concat_cond=input_concat_cond,
            input_concat_aux=input_concat_aux,
            negative_input_concat_cond=negative_input_concat_cond,
            negative_input_concat_aux=negative_input_concat_aux,
            prepend_cond=prepend_cond,
            prepend_cond_mask=prepend_cond_mask,
            cfg_scale=cfg_scale,
            cfg_dropout_prob=cfg_dropout_prob,
            scale_phi=scale_phi,
            global_embed=global_cond,
            local_add_cond=local_add_cond,
            **kwargs)

class DiTUncondWrapper(DiffusionModel):
    def __init__(
        self,
        in_channels,
        *args,
        **kwargs
    ):
        super().__init__()

        self.model = DiffusionTransformer(io_channels=in_channels, *args, **kwargs)

        self.io_channels = in_channels

        with torch.no_grad():
            for param in self.model.parameters():
                param *= 0.5

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

def create_diffusion_uncond_from_config(config: tp.Dict[str, tp.Any]):
    diffusion_uncond_config = config["model"]

    model_type = diffusion_uncond_config.get('type', None)

    diffusion_config = diffusion_uncond_config.get('config', {})

    assert model_type is not None, "Must specify model type in config"

    pretransform = diffusion_uncond_config.get("pretransform", None)

    sample_size = config.get("sample_size", None)
    assert sample_size is not None, "Must specify sample size in config"

    sample_rate = config.get("sample_rate", None)
    assert sample_rate is not None, "Must specify sample rate in config"

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    if model_type == 'DAU1d':

        model = DiffusionAttnUnet1D(
            **diffusion_config
        )
    
    elif model_type == "adp_uncond_1d":

        model = UNet1DUncondWrapper(
            **diffusion_config
        )

    elif model_type == "dit":
        model = DiTUncondWrapper(
            **diffusion_config
        )

    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

    return DiffusionModelWrapper(model,
                                io_channels=model.io_channels,
                                sample_size=sample_size,
                                sample_rate=sample_rate,
                                pretransform=pretransform,
                                min_input_length=min_input_length)

def create_diffusion_cond_from_config(config: tp.Dict[str, tp.Any]):

    model_config = config["model"]

    model_type = config["model_type"]

    diffusion_config = model_config.get('diffusion', None)
    assert diffusion_config is not None, "Must specify diffusion config"

    diffusion_objective = diffusion_config.get('diffusion_objective', 'v')

    diffusion_model_type = diffusion_config.get('type', None)
    assert diffusion_model_type is not None, "Must specify diffusion model type"

    diffusion_model_config = diffusion_config.get('config', None)
    assert diffusion_model_config is not None, "Must specify diffusion model config"

    # Parse modular_local_cond_configs before model creation (needed for DiT)
    modular_local_cond_configs = diffusion_config.get('modular_local_cond_configs', [])

    if diffusion_model_type == 'adp_cfg_1d':
        diffusion_model = UNetCFG1DWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'adp_1d':
        diffusion_model = UNet1DCondWrapper(**diffusion_model_config)
    elif diffusion_model_type == 'dit':
        # Pass modular_local_cond_configs to the DiT model
        diffusion_model = DiTWrapper(
            diffusion_objective=diffusion_objective,
            modular_local_cond_configs=modular_local_cond_configs,
            **diffusion_model_config
        )

    io_channels = model_config.get('io_channels', None)
    assert io_channels is not None, "Must specify io_channels in model config"

    sample_rate = config.get('sample_rate', None)
    assert sample_rate is not None, "Must specify sample_rate in config"


    cross_attention_ids = diffusion_config.get('cross_attention_cond_ids', [])
    gate = diffusion_config.get('gate', False)
    gate_type = diffusion_config.get('gate_type', None)
    gate_type_config = diffusion_config.get('gate_type_config', None)
    maf_cond_ids = diffusion_config.get('maf_cond_ids', None)
    global_cond_ids = diffusion_config.get('global_cond_ids', [])
    input_concat_ids = diffusion_config.get('input_concat_ids', [])
    local_add_cond_ids = diffusion_config.get('local_add_cond_ids', [])
    modular_local_cond_ids = [c["id"] for c in modular_local_cond_configs]
    prepend_cond_ids = diffusion_config.get('prepend_cond_ids', [])

    pretransform = model_config.get("pretransform", None)

    distribution_shift_options = diffusion_config.get("distribution_shift_options", None)
    sampling_distribution_shift_options = diffusion_config.get("sampling_distribution_shift_options", None)
    mask_padding_attention = diffusion_config.get("mask_padding_attention", False)
    use_effective_length_for_schedule = diffusion_config.get("use_effective_length_for_schedule", False)

    if pretransform is not None:
        pretransform = create_pretransform_from_config(pretransform, sample_rate)
        min_input_length = pretransform.downsampling_ratio
    else:
        min_input_length = 1

    conditioning_config = model_config.get('conditioning', None)

    conditioner = None
    if conditioning_config is not None:
        conditioner = create_multi_conditioner_from_conditioning_config(conditioning_config, pretransform=pretransform)

    if diffusion_model_type == "adp_cfg_1d" or diffusion_model_type == "adp_1d":
        min_input_length *= np.prod(diffusion_model_config["factors"])
    elif diffusion_model_type == "dit":
        min_input_length *= diffusion_model.model.patch_size

    # Get the proper wrapper class

    extra_kwargs = {}

    if model_type == "diffusion_cond" or model_type == "diffusion_cond_inpaint":
        wrapper_fn = ConditionedDiffusionModelWrapper

        extra_kwargs["diffusion_objective"] = diffusion_objective
        
    return wrapper_fn(
        diffusion_model,
        conditioner,
        min_input_length=min_input_length,
        sample_rate=sample_rate,
        cross_attn_cond_ids=cross_attention_ids,
        global_cond_ids=global_cond_ids,
        input_concat_ids=input_concat_ids,
        local_add_cond_ids=local_add_cond_ids,
        modular_local_cond_ids=modular_local_cond_ids,
        prepend_cond_ids=prepend_cond_ids,
        pretransform=pretransform,
        io_channels=io_channels,
        distribution_shift_options=distribution_shift_options,
        sampling_distribution_shift_options=sampling_distribution_shift_options,
        mask_padding_attention=mask_padding_attention,
        use_effective_length_for_schedule=use_effective_length_for_schedule,
        gate=gate,
        gate_type=gate_type,
        gate_type_config=gate_type_config,
        maf_cond_ids=maf_cond_ids,
        **extra_kwargs
    )
