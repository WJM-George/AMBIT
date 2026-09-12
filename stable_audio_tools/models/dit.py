import typing as tp
import math
import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .blocks import FourierFeatures, ExpoFourierFeatures
from .transformer import ContinuousTransformer        
from .lora import LoRAParametrization, set_lora_strength, has_lora, enable_lora, disable_lora, filter_lora_layers

class DiffusionTransformer(nn.Module):
    def __init__(self,
        io_channels=32,
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer", "mm_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        timestep_features_type: tp.Literal["learned", "expo"] = "learned",
        timestep_features_dim = 256,
        timestep_features_logsnr: bool = False,
        input_concat_cond_cfg: bool = False,
        joint_cfg_dropout: bool = False,
        require_explicit_negative_input_concat: bool = False,
        sceneplan_frame_text_alignment = None,
        sceneplan_soft_block_attention = None,
        sceneplan_chunk_moe = None,
        modular_local_cond_configs = None,
        activation_checkpointing: bool = True,
        **kwargs):

        super().__init__()

        self.cond_token_dim = cond_token_dim
        self.activation_checkpointing = bool(activation_checkpointing)

        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type
        self.timestep_features_logsnr = timestep_features_logsnr

        timestep_features_dim = timestep_features_dim

        if timestep_features_type == "expo":
            self.timestep_features = ExpoFourierFeatures(timestep_features_dim, 0.5, 10000.0)
        else:
            self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        
        self.diffusion_objective = diffusion_objective

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.input_concat_dim = input_concat_dim
        self.input_concat_cond_cfg = input_concat_cond_cfg
        self.joint_cfg_dropout = bool(joint_cfg_dropout)
        self.require_explicit_negative_input_concat = bool(
            require_explicit_negative_input_concat
        )

        alignment_config = dict(sceneplan_frame_text_alignment or {})
        alignment_enabled = bool(alignment_config.pop("enabled", False))
        self.sceneplan_alignment = None
        if alignment_enabled:
            if cond_embed_dim <= 0:
                raise ValueError(
                    "ScenePlan frame/text alignment requires cross-attention tokens"
                )
            from .sceneplan_alignment import ScenePlanFrameTextAlignment

            self.sceneplan_alignment = ScenePlanFrameTextAlignment(
                token_dim=int(cond_embed_dim), **alignment_config
            )
        elif alignment_config:
            raise ValueError(
                "disabled ScenePlan frame/text alignment has unused settings: "
                f"{sorted(alignment_config)}"
            )

        soft_block_config = dict(sceneplan_soft_block_attention or {})
        soft_block_enabled = bool(soft_block_config.get("enabled", False))
        self.sceneplan_soft_block = None
        if soft_block_enabled:
            if self.sceneplan_alignment is not None:
                raise ValueError(
                    "monotonic alignment and no-align soft-block attention are "
                    "mutually exclusive"
                )
            if cond_embed_dim <= 0:
                raise ValueError(
                    "ScenePlan soft-block attention requires cross-attention tokens"
                )
            from .sceneplan_alignment import ScenePlanSoftBlockAlignment

            self.sceneplan_soft_block = ScenePlanSoftBlockAlignment(
                max_sources=int(soft_block_config.get("max_sources", 4))
            )
        elif set(soft_block_config) - {"enabled"}:
            raise ValueError(
                "disabled ScenePlan soft-block attention has unused settings: "
                f"{sorted(set(soft_block_config) - {'enabled'})}"
            )

        chunk_moe_config = dict(sceneplan_chunk_moe or {})
        self.sceneplan_chunk_moe_enabled = bool(
            chunk_moe_config.get("enabled", False)
        )
        if self.sceneplan_chunk_moe_enabled and self.sceneplan_alignment is not None:
            raise ValueError(
                "the no-align ScenePlan chunk-MoE pilot cannot enable the "
                "legacy monotonic alignment module"
            )
        if not self.sceneplan_chunk_moe_enabled and (
            set(chunk_moe_config) - {"enabled"}
        ):
            raise ValueError(
                "disabled ScenePlan chunk-MoE has unused settings: "
                f"{sorted(set(chunk_moe_config) - {'enabled'})}"
            )
        if self.sceneplan_chunk_moe_enabled and cond_embed_dim <= 0:
            raise ValueError("ScenePlan chunk-MoE requires caption cross-attention")
        self.requires_sceneplan_aux = bool(
            self.sceneplan_alignment is not None
            or self.sceneplan_soft_block is not None
            or self.sceneplan_chunk_moe_enabled
        )

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        transformer_dim_out = io_channels * patch_size

        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=transformer_dim_out,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                modular_local_cond_configs=modular_local_cond_configs,
                sceneplan_soft_block_attention=soft_block_config,
                sceneplan_chunk_moe=chunk_moe_config,
                **kwargs
            )
      
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    # Fixed logsnr normalization range: maps logsnr to [0, 1] preserving direction (t=0→0, t=1→1)
    _LOGSNR_MIN = -12.0
    _LOGSNR_MAX = 5.0
    _LOGSNR_RANGE = _LOGSNR_MAX - _LOGSNR_MIN

    def _t_to_logsnr_cond(self, t: torch.Tensor) -> torch.Tensor:
        """Convert t to normalized logsnr in [0, 1] for timestep conditioning.

        Maps t through logsnr = log((1-t)/t), clamps to fixed range,
        then normalizes to [0, 1] preserving direction (t=0→0, t=1→1).
        """
        t_clamped = t.float().clamp(1e-7, 1 - 1e-7)
        logsnr = torch.log((1 - t_clamped) / t_clamped)
        logsnr = logsnr.clamp(self._LOGSNR_MIN, self._LOGSNR_MAX)
        return ((self._LOGSNR_MAX - logsnr) / self._LOGSNR_RANGE).to(t.dtype)

    def _call_transformer(self, x, *, prepend_inputs=None, cross_attn_cond=None,
                         cross_attn_mask=None,
                         cross_attention_bias=None,
                         cross_attention_event_mask=None,
                         cross_attention_speech_mask=None,
                         mask=None, prepend_mask=None, return_info=False,
                         return_moe_info=False,
                         exit_layer_ix=None, local_add_cond=None,
                         modular_local_cond=None, padding_mask=None,
                         moe_time_condition=None,
                         moe_context_source_ids=None,
                         moe_frame_source_ids=None,
                         extra_args=None, **kwargs):
        """Helper method to call transformer and handle early exit logic."""

        transformer_kwargs = dict(extra_args or {})
        transformer_kwargs.update(kwargs)
        transformer_kwargs.setdefault(
            "use_checkpointing", self.activation_checkpointing
        )
        output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond,
                                    context_mask=cross_attn_mask,
                                    cross_attention_bias=cross_attention_bias,
                                    cross_attention_event_mask=cross_attention_event_mask,
                                    cross_attention_speech_mask=cross_attention_speech_mask,
                                    prepend_mask=prepend_mask,
                                    return_info=return_info,
                                    return_moe_info=return_moe_info,
                                    exit_layer_ix=exit_layer_ix,
                                    local_add_cond=local_add_cond, modular_local_cond=modular_local_cond,
                                    padding_mask=padding_mask,
                                    moe_time_condition=moe_time_condition,
                                    moe_context_source_ids=moe_context_source_ids,
                                    moe_frame_source_ids=moe_frame_source_ids,
                                    **transformer_kwargs)

        if return_info or return_moe_info:
            output, info = output

        # Avoid postprocessing on early exit
        if exit_layer_ix is not None:
            if return_info:
                return output, info
            else:
                return output

        return (
            (output, info)
            if (return_info or return_moe_info) and 'info' in locals()
            else output
        )

    def _forward(
        self,
        x,
        t,
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        cross_attn_aux=None,
        input_concat_cond=None,
        input_concat_aux=None,
        local_add_cond=None,
        modular_local_cond=None,
        external_cross_attn_kv_lora_down=None,
        external_cross_attn_kv_lora_up=None,
        external_cross_attn_kv_lora_scale=1.0,
        external_cross_attn_kv_lora_start_index=0,
        external_layer_replacements=None,
        external_layer_replacement_start_index=0,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        return_info=False,
        return_alignment_info=False,
        return_moe_info=False,
        exit_layer_ix=None,
        **kwargs):

        if sum(bool(value) for value in (
            return_info, return_alignment_info, return_moe_info
        )) > 1:
            raise ValueError(
                "only one Transformer diagnostic return mode may be enabled"
            )
        if exit_layer_ix is not None and (return_alignment_info or return_moe_info):
            raise ValueError("ScenePlan diagnostics do not support early exit")

        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        cross_attention_bias = None
        cross_attention_event_mask = None
        cross_attention_speech_mask = None
        alignment_info = None
        if self.sceneplan_alignment is not None:
            if (
                cross_attn_cond is None
                or cross_attn_cond_mask is None
                or not isinstance(cross_attn_aux, dict)
                or not isinstance(input_concat_aux, dict)
            ):
                raise ValueError(
                    "ScenePlan frame/text alignment requires token and frame "
                    "auxiliary inputs"
                )
            cross_attention_bias, alignment_info = self.sceneplan_alignment(
                cross_attn_cond,
                cross_attn_cond_mask,
                cross_attn_aux,
                input_concat_aux,
                query_frames=int(x.shape[-1]),
            )

        if self.sceneplan_soft_block is not None:
            if (
                cross_attn_cond is None
                or cross_attn_cond_mask is None
                or not isinstance(cross_attn_aux, dict)
                or not isinstance(input_concat_aux, dict)
            ):
                raise ValueError(
                    "ScenePlan soft-block attention requires token and frame "
                    "auxiliary inputs"
                )
            (
                cross_attention_event_mask,
                cross_attention_speech_mask,
                _,
            ) = self.sceneplan_soft_block(
                cross_attn_cond,
                cross_attn_cond_mask,
                cross_attn_aux,
                input_concat_aux,
                query_frames=int(x.shape[-1]),
            )

        moe_context_source_ids = None
        moe_frame_source_ids = None
        if self.sceneplan_chunk_moe_enabled:
            if (
                cross_attn_cond is None
                or cross_attn_cond_mask is None
                or not isinstance(cross_attn_aux, dict)
                or not isinstance(input_concat_aux, dict)
            ):
                raise ValueError(
                    "ScenePlan chunk-MoE requires token and frame auxiliary inputs"
                )
            missing_moe_aux = sorted(
                key
                for key, values in (
                    ("event_source_ids", cross_attn_aux),
                    ("speech_source_ids", cross_attn_aux),
                    ("source_event_frame_ids", input_concat_aux),
                )
                if key not in values
            )
            if missing_moe_aux:
                raise ValueError(
                    "ScenePlan chunk-MoE auxiliaries are missing: "
                    f"{missing_moe_aux}"
                )
            event_source_ids = torch.as_tensor(
                cross_attn_aux["event_source_ids"],
                device=x.device,
                dtype=torch.long,
            )
            speech_source_ids = torch.as_tensor(
                cross_attn_aux["speech_source_ids"],
                device=x.device,
                dtype=torch.long,
            )
            if event_source_ids.shape != speech_source_ids.shape or (
                event_source_ids.shape != cross_attn_cond_mask.shape
            ):
                raise ValueError("chunk-MoE caption source roles do not align")
            moe_context_source_ids = torch.where(
                speech_source_ids.ne(0), speech_source_ids, event_source_ids
            )
            frame_source_ids = torch.as_tensor(
                input_concat_aux["source_event_frame_ids"],
                device=x.device,
                dtype=torch.long,
            )
            if frame_source_ids.ndim != 3 or int(frame_source_ids.shape[0]) != int(
                x.shape[0]
            ):
                raise ValueError("chunk-MoE frame source roles must be [B,4,T]")
            if int(frame_source_ids.shape[-1]) != int(x.shape[-1]):
                frame_source_ids = F.interpolate(
                    frame_source_ids.float(),
                    size=int(x.shape[-1]),
                    mode="nearest",
                ).to(torch.long)
            moe_frame_source_ids = frame_source_ids

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)
            
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

            prepend_length = prepend_cond.shape[1]

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        if local_add_cond is not None:
            local_add_cond = rearrange(local_add_cond, "b c t -> b t c")

        # Rearrange modular_local_cond tensors
        if modular_local_cond is not None:
            modular_local_cond = {
                k: rearrange(v, "b c t -> b t c")
                for k, v in modular_local_cond.items()
            }

        # Get the batch of timestep embeddings
        t_cond = self._t_to_logsnr_cond(t) if self.timestep_features_logsnr else t
        # Convert to model dtype for linear layers (t itself is kept in float32 for precision)
        # x has already been converted to model dtype in the outer forward() method
        t_cond = t_cond.to(x.dtype)
        timestep_embed = self.to_timestep_embed(self.timestep_features(t_cond[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(2).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            if (
                cross_attention_bias is not None
                or cross_attention_event_mask is not None
                or self.sceneplan_chunk_moe_enabled
            ):
                raise ValueError(
                    "ScenePlan aligned attention/MoE currently requires patch_size=1"
                )
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        prefix_queries = int(prepend_length) + int(
            getattr(self.transformer, "num_memory_tokens", 0)
        )
        if prefix_queries:
            if cross_attention_bias is not None:
                cross_attention_bias = F.pad(
                    cross_attention_bias,
                    (0, 0, prefix_queries, 0),
                    value=0.0,
                )
            if cross_attention_event_mask is not None:
                cross_attention_event_mask = F.pad(
                    cross_attention_event_mask,
                    (0, 0, prefix_queries, 0),
                    value=0.0,
                )
            if cross_attention_speech_mask is not None:
                cross_attention_speech_mask = F.pad(
                    cross_attention_speech_mask,
                    (0, 0, prefix_queries, 0),
                    value=0.0,
                )

        result = self._call_transformer(
            x,
            prepend_inputs=prepend_inputs,
            cross_attn_cond=cross_attn_cond,
            cross_attn_mask=cross_attn_cond_mask,
            cross_attention_bias=cross_attention_bias,
            cross_attention_event_mask=cross_attention_event_mask,
            cross_attention_speech_mask=cross_attention_speech_mask,
            mask=mask,
            prepend_mask=prepend_mask,
            return_info=return_info,
            return_moe_info=return_moe_info,
            exit_layer_ix=exit_layer_ix,
            local_add_cond=local_add_cond,
            modular_local_cond=modular_local_cond,
            external_cross_attn_kv_lora_down=(
                external_cross_attn_kv_lora_down
            ),
            external_cross_attn_kv_lora_up=external_cross_attn_kv_lora_up,
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
            padding_mask=padding_mask,
            moe_time_condition=timestep_embed,
            moe_context_source_ids=moe_context_source_ids,
            moe_frame_source_ids=moe_frame_source_ids,
            extra_args=extra_args,
            **kwargs,
        )

        # Handle early exit (result contains both output and info)
        if exit_layer_ix is not None:
            return result

        output = result[0] if (return_info or return_moe_info) else result
        if return_info or return_moe_info:
            info = result[1]

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]       

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        if return_alignment_info:
            assert alignment_info is not None
            return output, alignment_info

        if return_moe_info:
            return output, info

        return output

    def sceneplan_soft_block_gate_metrics(self) -> dict[str, torch.Tensor]:
        """Return compact, detached diagnostics for the per-layer score gates."""

        event_values = []
        speech_values = []
        saturation = []
        for layer in self.transformer.layers:
            maximum = getattr(layer, "sceneplan_soft_block_max_bias", None)
            event_gate = getattr(layer, "sceneplan_event_bias_gate", None)
            speech_gate = getattr(layer, "sceneplan_speech_bias_gate", None)
            if maximum is None or event_gate is None or speech_gate is None:
                continue
            event_value = float(maximum) * torch.tanh(event_gate.detach().float())
            speech_value = float(maximum) * torch.tanh(
                speech_gate.detach().float()
            )
            event_values.append(event_value)
            speech_values.append(speech_value)
            saturation.extend(
                (
                    event_value.abs().div(float(maximum)),
                    speech_value.abs().div(float(maximum)),
                )
            )
        if not event_values:
            return {}
        event = torch.stack(event_values)
        speech = torch.stack(speech_values)
        saturation_values = torch.stack(saturation)
        return {
            "event_bias_mean": event.mean(),
            "event_bias_abs_mean": event.abs().mean(),
            "speech_bias_mean": speech.mean(),
            "speech_bias_abs_mean": speech.abs().mean(),
            "bias_abs_max": torch.cat((event.abs(), speech.abs())).max(),
            "saturation_fraction": saturation_values.ge(0.95).float().mean(),
            "active_layers": event.new_tensor(len(event_values)),
        }

    def apg_project(self, v0, v1, padding_mask=None):
        """
        Project v0 into components parallel and orthogonal to v1.

        Args:
            v0: Tensor to project (B, C, T)
            v1: Reference direction (B, C, T)
            padding_mask: Optional mask (B, T) where True = valid, False = padding.
                          If provided, only valid positions contribute to the projection.
        """
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()

        if padding_mask is not None:
            # Expand mask to match tensor shape: (B, T) -> (B, 1, T)
            mask = padding_mask.unsqueeze(1).double()
            # Zero out padding positions for projection computation
            v0_masked = v0 * mask
            v1_masked = v1 * mask
            # Normalize only over valid positions
            v1_norm = v1_masked.norm(dim=[-1, -2], keepdim=True).clamp(min=1e-8)
            v1_normalized = v1_masked / v1_norm
            # Compute projection using masked values
            v0_parallel = (v0_masked * v1_normalized).sum(dim=[-1, -2], keepdim=True) * v1_normalized
            # Orthogonal component: subtract parallel from original (not masked) v0
            # but apply mask to ensure padding stays zero
            v0_orthogonal = (v0 - (v0 * v1_normalized).sum(dim=[-1, -2], keepdim=True) * v1_normalized) * mask
        else:
            v1 = torch.nn.functional.normalize(v1, dim=[-1, -2])
            v0_parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
            v0_orthogonal = v0 - v0_parallel

        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    @staticmethod
    def _batch_alignment_aux(
        positive,
        negative,
        *,
        padded_last_dim: int | None = None,
    ):
        if positive is None and negative is None:
            return None
        if not isinstance(positive, dict) or not isinstance(negative, dict):
            raise ValueError(
                "ScenePlan CFG requires explicit positive and negative "
                "alignment auxiliaries"
            )
        if set(positive) != set(negative):
            raise ValueError(
                "positive and negative ScenePlan auxiliaries have different fields"
            )
        output = {}
        for key in sorted(positive):
            left = positive[key]
            right = negative[key]
            if not torch.is_tensor(left) or not torch.is_tensor(right):
                raise TypeError("ScenePlan alignment auxiliaries must be tensors")
            if padded_last_dim is not None:
                if left.ndim != 2 or right.ndim != 2:
                    raise ValueError(
                        "token alignment auxiliaries must have shape [B,K]"
                    )
                if int(left.shape[-1]) > padded_last_dim or int(right.shape[-1]) > padded_last_dim:
                    raise ValueError("ScenePlan token auxiliary exceeds padded length")
                left = F.pad(
                    left,
                    (0, padded_last_dim - int(left.shape[-1])),
                    value=0,
                )
                right = F.pad(
                    right,
                    (0, padded_last_dim - int(right.shape[-1])),
                    value=0,
                )
            elif tuple(left.shape[1:]) != tuple(right.shape[1:]):
                raise ValueError(
                    f"ScenePlan frame auxiliary {key!r} shape differs across CFG"
                )
            output[key] = torch.cat([left, right], dim=0)
        return output

    def forward(
        self,
        x,
        t,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        cross_attn_aux=None,
        external_cross_attn_residual=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        negative_cross_attn_aux=None,
        input_concat_cond=None,
        input_concat_aux=None,
        negative_input_concat_cond=None,
        negative_input_concat_aux=None,
        local_add_cond=None,
        modular_local_cond=None,
        external_cross_attn_kv_lora_down=None,
        external_cross_attn_kv_lora_up=None,
        external_cross_attn_kv_lora_scale=1.0,
        external_cross_attn_kv_lora_start_index=0,
        external_layer_replacements=None,
        external_layer_replacement_start_index=0,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_interval = (0, 1),
        lora_interval = (0, 1),
        lora_layer_filter = "",
        lora_configs = None,
        causal=False,
        scale_phi=0.0,
        cfg_norm_threshold=0.0,
        apg_scale=1.0,
        mask=None,
        return_info=False,
        return_alignment_info=False,
        return_moe_info=False,
        exit_layer_ix=None,
        **kwargs):

        assert not causal, "Causal mode is not supported for DiffusionTransformer"
        if return_alignment_info and cfg_scale != 1.0:
            raise ValueError(
                "alignment diagnostics are available only on a single "
                "non-CFG training forward"
            )
        if return_moe_info and cfg_scale != 1.0:
            raise ValueError(
                "chunk-MoE diagnostics are available only on a single "
                "non-CFG training forward"
            )

        model_dtype = next(self.parameters()).dtype

        x = x.to(model_dtype)

        # Keep t in float32: the logsnr transform log((1-t)/t) amplifies bf16
        # quantization error ~380x near t=1, causing catastrophic conditioning errors.
        # t is a 1D batch-size tensor so float32 has zero memory impact.
        t = t.float()

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if external_cross_attn_residual is not None:
            if cross_attn_cond is None:
                raise ValueError(
                    "external_cross_attn_residual requires cross-attention "
                    "conditioning"
                )
            if tuple(external_cross_attn_residual.shape) != tuple(
                cross_attn_cond.shape
            ):
                raise ValueError(
                    "external_cross_attn_residual must match conditioning "
                    f"tokens {tuple(cross_attn_cond.shape)}, got "
                    f"{tuple(external_cross_attn_residual.shape)}"
                )
            # Add before the checkpoint-native condition projection.  A zero
            # residual is therefore bit-exact, while learned source-region
            # evidence is processed by every pretrained cross-attention block.
            cross_attn_cond = cross_attn_cond + external_cross_attn_residual.to(
                device=cross_attn_cond.device,
                dtype=cross_attn_cond.dtype,
            )

        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)

        if negative_input_concat_cond is not None:
            negative_input_concat_cond = negative_input_concat_cond.to(model_dtype)
            if input_concat_cond is None:
                raise ValueError(
                    "negative_input_concat_cond requires positive input_concat_cond"
                )
            if tuple(negative_input_concat_cond.shape) != tuple(
                input_concat_cond.shape
            ):
                raise ValueError(
                    "positive and negative input-concat conditions must have "
                    "equal shapes"
                )

        if local_add_cond is not None:
            local_add_cond = local_add_cond.to(model_dtype)

        if modular_local_cond is not None:
            modular_local_cond = {k: v.to(model_dtype) for k, v in modular_local_cond.items()}

        if external_cross_attn_kv_lora_down is not None:
            external_cross_attn_kv_lora_down = (
                external_cross_attn_kv_lora_down.to(model_dtype)
            )

        if external_cross_attn_kv_lora_up is not None:
            external_cross_attn_kv_lora_up = (
                external_cross_attn_kv_lora_up.to(model_dtype)
            )

        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.to(model_dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()
        if negative_cross_attn_mask is not None:
            negative_cross_attn_mask = negative_cross_attn_mask.bool()

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        # Early exit bypasses CFG processing
        if exit_layer_ix is not None:
            assert self.transformer_type == "continuous_transformer", "exit_layer_ix is only supported for continuous_transformer"
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                cross_attn_aux=cross_attn_aux,
                input_concat_cond=input_concat_cond,
                input_concat_aux=input_concat_aux,
                local_add_cond=local_add_cond,
                modular_local_cond=modular_local_cond,
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
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                padding_mask=padding_mask,
                mask=mask,
                return_info=return_info,
                return_alignment_info=return_alignment_info,
                return_moe_info=return_moe_info,
                exit_layer_ix=exit_layer_ix,
                **kwargs
            )

        # CFG dropout
        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:
            if self.requires_sceneplan_aux:
                raise ValueError(
                    "ScenePlan aligned attention/MoE requires raw independent CFG dropout; "
                    "generic embedding dropout is not allowed"
                )
            shared_dropout_mask = None
            if self.joint_cfg_dropout:
                cfg_tensors = [
                    value
                    for value in (
                        cross_attn_cond,
                        prepend_cond,
                        input_concat_cond if self.input_concat_cond_cfg else None,
                    )
                    if value is not None
                ]
                if cfg_tensors:
                    batch_size = int(cfg_tensors[0].shape[0])
                    if any(int(value.shape[0]) != batch_size for value in cfg_tensors):
                        raise ValueError(
                            "joint CFG dropout requires aligned conditioning batches"
                        )
                    shared_dropout_mask = torch.bernoulli(
                        torch.full(
                            (batch_size, 1, 1),
                            cfg_dropout_prob,
                            device=cfg_tensors[0].device,
                        )
                    ).to(torch.bool)

            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = shared_dropout_mask
                if dropout_mask is None:
                    dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = shared_dropout_mask
                if dropout_mask is None:
                    dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

            if self.input_concat_cond_cfg and input_concat_cond is not None:
                null_embed = torch.zeros_like(input_concat_cond, device=input_concat_cond.device)
                dropout_mask = shared_dropout_mask
                if dropout_mask is None:
                    dropout_mask = torch.bernoulli(torch.full((input_concat_cond.shape[0], 1, 1), cfg_dropout_prob, device=input_concat_cond.device)).to(torch.bool)
                input_concat_cond = torch.where(dropout_mask, null_embed, input_concat_cond)

        if self.diffusion_objective == "v":
            sigma = torch.sin(t * math.pi / 2)
            alpha = torch.cos(t * math.pi / 2)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            sigma = t

        # Ensure sigma is indexable (handle scalar tensors from v-diffusion samplers)
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)

        # LoRA interval
        if has_lora(self):
            if lora_configs is not None:
                # Multi-LoRA: per-LoRA interval and layer filter
                for lora_config in lora_configs:
                    idx = lora_config["lora_index"]
                    interval = lora_config.get("interval", (0, 1))
                    layer_filter = lora_config.get("layer_filter", "")
                    if interval[0] <= sigma[0] <= interval[1]:
                        enable_lora(self, lora_index=idx)
                        filter_lora_layers(self, layer_filter, lora_index=idx)
                    else:
                        disable_lora(self, lora_index=idx)
            else:
                # Legacy single-LoRA path
                if lora_interval[0] <= sigma[0] <= lora_interval[1]:
                    enable_lora(self)
                    filter_lora_layers(self, lora_layer_filter)
                else:
                    disable_lora(self)

        has_cfg_cond = (
            cross_attn_cond is not None
            or prepend_cond is not None
            or (self.input_concat_cond_cfg and input_concat_cond is not None)
        )
        if cfg_scale != 1.0 and has_cfg_cond and (cfg_interval[0] <= sigma[0] <= cfg_interval[1]):

            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if input_concat_cond is not None:
                if self.input_concat_cond_cfg:
                    if (
                        self.require_explicit_negative_input_concat
                        and negative_input_concat_cond is None
                    ):
                        raise ValueError(
                            "this model requires explicit -1 unknown local CFG "
                            "conditioning; a zero tensor means known inactivity"
                        )
                    null_input_concat_cond = (
                        negative_input_concat_cond
                        if negative_input_concat_cond is not None
                        else torch.zeros_like(
                            input_concat_cond, device=input_concat_cond.device
                        )
                    )
                    batch_input_concat_cond = torch.cat([input_concat_cond, null_input_concat_cond], dim=0)
                else:
                    batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            else:
                batch_input_concat_cond = None

            if local_add_cond is not None:
                batch_local_add_cond = torch.cat([local_add_cond, local_add_cond], dim=0)
            else:
                batch_local_add_cond = None

            if modular_local_cond is not None:
                batch_modular_local_cond = {k: torch.cat([v, v], dim=0) for k, v in modular_local_cond.items()}
            else:
                batch_modular_local_cond = None

            batch_cond = None
            batch_cond_masks = None
            batch_cross_attn_aux = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:

                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # Explicit negative captions may be much shorter than positive
                # semantic captions (the ScenePlan unknown branch is one null
                # token). Pad both token sequences and their masks before the
                # batched CFG pass.
                if negative_cross_attn_cond is not None:
                    max_cond_tokens = max(
                        int(cross_attn_cond.shape[1]),
                        int(negative_cross_attn_cond.shape[1]),
                    )
                    positive_cond = F.pad(
                        cross_attn_cond,
                        (0, 0, 0, max_cond_tokens - int(cross_attn_cond.shape[1])),
                    )
                    negative_cond = F.pad(
                        negative_cross_attn_cond,
                        (
                            0,
                            0,
                            0,
                            max_cond_tokens
                            - int(negative_cross_attn_cond.shape[1]),
                        ),
                    )
                    batch_cond = torch.cat([positive_cond, negative_cond], dim=0)
                    batch_cross_attn_aux = self._batch_alignment_aux(
                        cross_attn_aux,
                        negative_cross_attn_aux,
                        padded_last_dim=max_cond_tokens,
                    )

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)
                    if self.requires_sceneplan_aux:
                        raise ValueError(
                            "ScenePlan aligned attention/MoE CFG requires an explicit "
                            "negative caption"
                        )

                if negative_cross_attn_cond is not None:
                    if cross_attn_cond_mask is not None or negative_cross_attn_mask is not None:
                        positive_mask = cross_attn_cond_mask
                        if positive_mask is None:
                            positive_mask = torch.ones(
                                cross_attn_cond.shape[:2],
                                device=cross_attn_cond.device,
                                dtype=torch.bool,
                            )
                        negative_mask = negative_cross_attn_mask
                        if negative_mask is None:
                            negative_mask = torch.ones(
                                negative_cross_attn_cond.shape[:2],
                                device=negative_cross_attn_cond.device,
                                dtype=torch.bool,
                            )
                        positive_mask = F.pad(
                            positive_mask,
                            (0, max_cond_tokens - int(positive_mask.shape[1])),
                            value=False,
                        )
                        negative_mask = F.pad(
                            negative_mask,
                            (0, max_cond_tokens - int(negative_mask.shape[1])),
                            value=False,
                        )
                        batch_cond_masks = torch.cat(
                            [positive_mask, negative_mask], dim=0
                        )
                elif cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
               
            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None:

                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                           
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
         

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None

            if padding_mask is not None:
                batch_padding_mask = torch.cat([padding_mask, padding_mask], dim=0)
            else:
                batch_padding_mask = None

            batch_input_concat_aux = self._batch_alignment_aux(
                input_concat_aux,
                negative_input_concat_aux,
            )

            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                cross_attn_aux=batch_cross_attn_aux,
                mask = batch_masks,
                input_concat_cond = batch_input_concat_cond,
                input_concat_aux=batch_input_concat_aux,
                local_add_cond = batch_local_add_cond,
                modular_local_cond=batch_modular_local_cond,
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
                global_embed = batch_global_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                padding_mask = batch_padding_mask,
                return_info = return_info,
                return_alignment_info=False,
                return_moe_info=False,
                **kwargs)

            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)

            if self.diffusion_objective == "v":
                cond_denoised = x * alpha[:, None, None] - cond_output * sigma[:, None, None]
                uncond_denoised = x * alpha[:, None, None] - uncond_output * sigma[:, None, None]

            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                cond_denoised = x - cond_output * sigma[:, None, None]
                uncond_denoised = x - uncond_output * sigma[:, None, None]

            diff = cond_denoised - uncond_denoised
            
            if cfg_norm_threshold > 0:
                if padding_mask is not None:
                    # Only compute norm over valid positions
                    mask = padding_mask.unsqueeze(1).float()  # (B, 1, T)
                    diff_masked = diff * mask
                    diff_norm = diff_masked.norm(p=2, dim=[-1, -2], keepdim=True)
                else:
                    diff_norm = diff.norm(p=2, dim=[-1, -2], keepdim=True)
                scale_factor = torch.minimum(torch.ones_like(diff), cfg_norm_threshold / diff_norm)
                diff *= scale_factor

            if apg_scale == 0.0:
                # Vanilla CFG: use full diff
                cfg_diff = diff
            elif apg_scale == 1.0:
                # Full APG: use only orthogonal component
                _, diff_orthogonal = self.apg_project(diff, cond_denoised, padding_mask=padding_mask)
                cfg_diff = diff_orthogonal
            else:
                # Blended APG: interpolate between full diff and orthogonal
                diff_parallel, diff_orthogonal = self.apg_project(diff, cond_denoised, padding_mask=padding_mask)
                cfg_diff = apg_scale * diff_orthogonal + (1 - apg_scale) * diff

            cfg_denoised = cond_denoised + (cfg_scale - 1) * cfg_diff
                    
            if self.diffusion_objective == "v":
                output = (x * alpha[:, None, None] - cfg_denoised) / sigma[:, None, None]
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                output = (x - cfg_denoised) / sigma[:, None, None]

            # CFG Rescale
            if scale_phi != 0.0:
                # Match one scale per sample over the complete latent rather
                # than independently rescaling every frame.  Per-frame scales
                # can distort temporal envelopes and become unstable when a
                # frame has near-zero variance.
                reduce_dims = tuple(range(1, output.ndim))
                cond_out_std = cond_output.float().std(
                    dim=reduce_dims, keepdim=True, unbiased=False
                )
                out_cfg_std = output.float().std(
                    dim=reduce_dims, keepdim=True, unbiased=False
                ).clamp_min(torch.finfo(torch.float32).eps)
                rescaled = output * (cond_out_std / out_cfg_std).to(output.dtype)
                output = scale_phi * rescaled + (1 - scale_phi) * output
           
            if return_info:
                info["uncond_output"] = uncond_output
                return output, info

            return output
            
        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                cross_attn_aux=cross_attn_aux,
                input_concat_cond=input_concat_cond,
                input_concat_aux=input_concat_aux,
                local_add_cond=local_add_cond,
                modular_local_cond=modular_local_cond,
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
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                padding_mask=padding_mask,
                mask=mask,
                return_info=return_info,
                return_alignment_info=return_alignment_info,
                return_moe_info=return_moe_info,
                **kwargs
            )
