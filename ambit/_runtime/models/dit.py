import typing as tp
import math
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from .blocks import FourierFeatures, ExpoFourierFeatures
from .transformer import ContinuousTransformer


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
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
        transformer_type: tp.Literal[
            "continuous_transformer", "mm_transformer"
        ] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        timestep_features_type: tp.Literal["learned", "expo"] = "learned",
        timestep_features_dim=256,
        timestep_features_logsnr: bool = False,
        input_concat_cond_cfg: bool = False,
        joint_cfg_dropout: bool = False,
        require_explicit_negative_input_concat: bool = False,
        modular_local_cond_configs=None,
        **kwargs,
    ):
        super().__init__()
        self.cond_token_dim = cond_token_dim
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
            assert timestep_embed_dim is not None, (
                "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            )
            input_concat_dim += timestep_embed_dim
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        self.diffusion_objective = diffusion_objective
        if cond_token_dim > 0:
            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
            )
        else:
            cond_embed_dim = 0
        if global_cond_dim > 0:
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False),
            )
        if prepend_cond_dim > 0:
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False),
            )
        self.input_concat_dim = input_concat_dim
        self.input_concat_cond_cfg = input_concat_cond_cfg
        self.joint_cfg_dropout = bool(joint_cfg_dropout)
        self.require_explicit_negative_input_concat = bool(require_explicit_negative_input_concat)
        self.requires_sceneplan_aux = False
        dim_in = io_channels + self.input_concat_dim
        self.patch_size = patch_size
        self.transformer_type = transformer_type
        self.global_cond_type = global_cond_type
        transformer_dim_out = io_channels * patch_size
        if self.transformer_type == "continuous_transformer":
            global_dim = None
            if self.global_cond_type == "adaLN":
                global_dim = embed_dim
            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=transformer_dim_out,
                cross_attend=cond_token_dim > 0,
                cond_token_dim=cond_embed_dim,
                global_cond_dim=global_dim,
                modular_local_cond_configs=modular_local_cond_configs,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")
        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    _LOGSNR_MIN = -12.0
    _LOGSNR_MAX = 5.0
    _LOGSNR_RANGE = _LOGSNR_MAX - _LOGSNR_MIN

    def _t_to_logsnr_cond(self, t: torch.Tensor) -> torch.Tensor:
        """Convert t to normalized logsnr in [0, 1] for timestep conditioning.

        Maps t through logsnr = log((1-t)/t), clamps to fixed range,
        then normalizes to [0, 1] preserving direction (t=0→0, t=1→1).
        """
        t_clamped = t.float().clamp(1e-07, 1 - 1e-07)
        logsnr = torch.log((1 - t_clamped) / t_clamped)
        logsnr = logsnr.clamp(self._LOGSNR_MIN, self._LOGSNR_MAX)
        return ((self._LOGSNR_MAX - logsnr) / self._LOGSNR_RANGE).to(t.dtype)

    def _call_transformer(
        self,
        x,
        *,
        prepend_inputs=None,
        cross_attn_cond=None,
        cross_attn_mask=None,
        cross_attention_bias=None,
        mask=None,
        prepend_mask=None,
        return_info=False,
        exit_layer_ix=None,
        local_add_cond=None,
        modular_local_cond=None,
        padding_mask=None,
        extra_args=None,
        **kwargs,
    ):
        """Helper method to call transformer and handle early exit logic."""
        transformer_kwargs = dict(extra_args or {})
        transformer_kwargs.update(kwargs)
        output = self.transformer(
            x,
            prepend_embeds=prepend_inputs,
            context=cross_attn_cond,
            context_mask=cross_attn_mask,
            cross_attention_bias=cross_attention_bias,
            prepend_mask=prepend_mask,
            return_info=return_info,
            exit_layer_ix=exit_layer_ix,
            local_add_cond=local_add_cond,
            modular_local_cond=modular_local_cond,
            padding_mask=padding_mask,
            **transformer_kwargs,
        )
        if return_info or False:
            (output, info) = output
        if exit_layer_ix is not None:
            if return_info:
                return (output, info)
            else:
                return output
        return (output, info) if (return_info or False) and "info" in locals() else output

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
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs,
    ):
        if sum((bool(value) for value in (return_info, False, False))) > 1:
            raise ValueError("only one Transformer diagnostic return mode may be enabled")
        if exit_layer_ix is not None and False:
            raise ValueError("ScenePlan diagnostics do not support early exit")
        if cross_attn_cond is not None:
            cross_attn_cond = self.to_cond_embed(cross_attn_cond)
        cross_attention_bias = None
        if global_embed is not None:
            global_embed = self.to_global_embed(global_embed)
        prepend_inputs = None
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            prepend_cond = self.to_prepend_embed(prepend_cond)
            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask
            prepend_length = prepend_cond.shape[1]
        if input_concat_cond is not None:
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2],), mode="nearest")
            x = torch.cat([x, input_concat_cond], dim=1)
        if local_add_cond is not None:
            local_add_cond = rearrange(local_add_cond, "b c t -> b t c")
        if modular_local_cond is not None:
            modular_local_cond = {
                k: rearrange(v, "b c t -> b t c") for (k, v) in modular_local_cond.items()
            }
        t_cond = self._t_to_logsnr_cond(t) if self.timestep_features_logsnr else t
        t_cond = t_cond.to(x.dtype)
        timestep_embed = self.to_timestep_embed(self.timestep_features(t_cond[:, None]))
        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(2).expand(-1, -1, x.shape[2])], dim=1)
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat(
                    [prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)],
                    dim=1,
                )
            prepend_length = prepend_inputs.shape[1]
        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c t -> b t c")
        extra_args = {}
        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed
        if self.patch_size > 1:
            if cross_attention_bias is not None or False or False:
                raise ValueError("ScenePlan aligned attention/MoE currently requires patch_size=1")
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)
        prefix_queries = int(prepend_length) + int(
            getattr(self.transformer, "num_memory_tokens", 0)
        )
        if prefix_queries:
            if cross_attention_bias is not None:
                cross_attention_bias = F.pad(
                    cross_attention_bias, (0, 0, prefix_queries, 0), value=0.0
                )
        result = self._call_transformer(
            x,
            prepend_inputs=prepend_inputs,
            cross_attn_cond=cross_attn_cond,
            cross_attn_mask=cross_attn_cond_mask,
            cross_attention_bias=cross_attention_bias,
            mask=mask,
            prepend_mask=prepend_mask,
            return_info=return_info,
            exit_layer_ix=exit_layer_ix,
            local_add_cond=local_add_cond,
            modular_local_cond=modular_local_cond,
            padding_mask=padding_mask,
            extra_args=extra_args,
            **kwargs,
        )
        if exit_layer_ix is not None:
            return result
        output = result[0] if return_info or False else result
        if return_info or False:
            info = result[1]
        output = rearrange(output, "b t c -> b c t")[:, :, prepend_length:]
        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)
        output = self.postprocess_conv(output) + output
        if return_info:
            return (output, info)
        return output

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
        (v0, v1) = (v0.double(), v1.double())
        if padding_mask is not None:
            mask = padding_mask.unsqueeze(1).double()
            v0_masked = v0 * mask
            v1_masked = v1 * mask
            v1_norm = v1_masked.norm(dim=[-1, -2], keepdim=True).clamp(min=1e-08)
            v1_normalized = v1_masked / v1_norm
            v0_parallel = (v0_masked * v1_normalized).sum(
                dim=[-1, -2], keepdim=True
            ) * v1_normalized
            v0_orthogonal = (
                v0 - (v0 * v1_normalized).sum(dim=[-1, -2], keepdim=True) * v1_normalized
            ) * mask
        else:
            v1 = torch.nn.functional.normalize(v1, dim=[-1, -2])
            v0_parallel = (v0 * v1).sum(dim=[-1, -2], keepdim=True) * v1
            v0_orthogonal = v0 - v0_parallel
        return (v0_parallel.to(dtype), v0_orthogonal.to(dtype))

    @staticmethod
    def _batch_alignment_aux(positive, negative, *, padded_last_dim: int | None = None):
        if positive is None and negative is None:
            return None
        if not isinstance(positive, dict) or not isinstance(negative, dict):
            raise ValueError(
                "ScenePlan CFG requires explicit positive and negative alignment auxiliaries"
            )
        if set(positive) != set(negative):
            raise ValueError("positive and negative ScenePlan auxiliaries have different fields")
        output = {}
        for key in sorted(positive):
            left = positive[key]
            right = negative[key]
            if not torch.is_tensor(left) or not torch.is_tensor(right):
                raise TypeError("ScenePlan alignment auxiliaries must be tensors")
            if padded_last_dim is not None:
                if left.ndim != 2 or right.ndim != 2:
                    raise ValueError("token alignment auxiliaries must have shape [B,K]")
                if int(left.shape[-1]) > padded_last_dim or int(right.shape[-1]) > padded_last_dim:
                    raise ValueError("ScenePlan token auxiliary exceeds padded length")
                left = F.pad(left, (0, padded_last_dim - int(left.shape[-1])), value=0)
                right = F.pad(right, (0, padded_last_dim - int(right.shape[-1])), value=0)
            elif tuple(left.shape[1:]) != tuple(right.shape[1:]):
                raise ValueError(f"ScenePlan frame auxiliary {key!r} shape differs across CFG")
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
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        padding_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_interval=(0, 1),
        lora_interval=(0, 1),
        lora_layer_filter="",
        lora_configs=None,
        causal=False,
        scale_phi=0.0,
        cfg_norm_threshold=0.0,
        apg_scale=1.0,
        mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs,
    ):
        assert not causal, "Causal mode is not supported for DiffusionTransformer"
        model_dtype = next(self.parameters()).dtype
        x = x.to(model_dtype)
        t = t.float()
        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)
        if external_cross_attn_residual is not None:
            if cross_attn_cond is None:
                raise ValueError(
                    "external_cross_attn_residual requires cross-attention conditioning"
                )
            if tuple(external_cross_attn_residual.shape) != tuple(cross_attn_cond.shape):
                raise ValueError(
                    f"external_cross_attn_residual must match conditioning tokens {tuple(cross_attn_cond.shape)}, got {tuple(external_cross_attn_residual.shape)}"
                )
            cross_attn_cond = cross_attn_cond + external_cross_attn_residual.to(
                device=cross_attn_cond.device, dtype=cross_attn_cond.dtype
            )
        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)
        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)
        if negative_input_concat_cond is not None:
            negative_input_concat_cond = negative_input_concat_cond.to(model_dtype)
            if input_concat_cond is None:
                raise ValueError("negative_input_concat_cond requires positive input_concat_cond")
            if tuple(negative_input_concat_cond.shape) != tuple(input_concat_cond.shape):
                raise ValueError(
                    "positive and negative input-concat conditions must have equal shapes"
                )
        if local_add_cond is not None:
            local_add_cond = local_add_cond.to(model_dtype)
        if modular_local_cond is not None:
            modular_local_cond = {k: v.to(model_dtype) for (k, v) in modular_local_cond.items()}
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
        if exit_layer_ix is not None:
            assert self.transformer_type == "continuous_transformer", (
                "exit_layer_ix is only supported for continuous_transformer"
            )
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
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                padding_mask=padding_mask,
                mask=mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
                **kwargs,
            )
        if self.diffusion_objective == "v":
            sigma = torch.sin(t * math.pi / 2)
            alpha = torch.cos(t * math.pi / 2)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            sigma = t
        if sigma.dim() == 0:
            sigma = sigma.unsqueeze(0)
        has_cfg_cond = (
            cross_attn_cond is not None
            or prepend_cond is not None
            or (self.input_concat_cond_cfg and input_concat_cond is not None)
        )
        if cfg_scale != 1.0 and has_cfg_cond and (cfg_interval[0] <= sigma[0] <= cfg_interval[1]):
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
                            "this model requires explicit -1 unknown local CFG conditioning; a zero tensor means known inactivity"
                        )
                    null_input_concat_cond = (
                        negative_input_concat_cond
                        if negative_input_concat_cond is not None
                        else torch.zeros_like(input_concat_cond, device=input_concat_cond.device)
                    )
                    batch_input_concat_cond = torch.cat(
                        [input_concat_cond, null_input_concat_cond], dim=0
                    )
                else:
                    batch_input_concat_cond = torch.cat(
                        [input_concat_cond, input_concat_cond], dim=0
                    )
            else:
                batch_input_concat_cond = None
            if local_add_cond is not None:
                batch_local_add_cond = torch.cat([local_add_cond, local_add_cond], dim=0)
            else:
                batch_local_add_cond = None
            if modular_local_cond is not None:
                batch_modular_local_cond = {
                    k: torch.cat([v, v], dim=0) for (k, v) in modular_local_cond.items()
                }
            else:
                batch_modular_local_cond = None
            batch_cond = None
            batch_cond_masks = None
            batch_cross_attn_aux = None
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                if negative_cross_attn_cond is not None:
                    max_cond_tokens = max(
                        int(cross_attn_cond.shape[1]), int(negative_cross_attn_cond.shape[1])
                    )
                    positive_cond = F.pad(
                        cross_attn_cond, (0, 0, 0, max_cond_tokens - int(cross_attn_cond.shape[1]))
                    )
                    negative_cond = F.pad(
                        negative_cross_attn_cond,
                        (0, 0, 0, max_cond_tokens - int(negative_cross_attn_cond.shape[1])),
                    )
                    batch_cond = torch.cat([positive_cond, negative_cond], dim=0)
                    batch_cross_attn_aux = self._batch_alignment_aux(
                        cross_attn_aux, negative_cross_attn_aux, padded_last_dim=max_cond_tokens
                    )
                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)
                    if self.requires_sceneplan_aux:
                        raise ValueError(
                            "ScenePlan aligned attention/MoE CFG requires an explicit negative caption"
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
                        batch_cond_masks = torch.cat([positive_mask, negative_mask], dim=0)
                elif cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat(
                        [cross_attn_cond_mask, cross_attn_cond_mask], dim=0
                    )
            batch_prepend_cond = None
            batch_prepend_cond_mask = None
            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat(
                        [prepend_cond_mask, prepend_cond_mask], dim=0
                    )
            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None
            if padding_mask is not None:
                batch_padding_mask = torch.cat([padding_mask, padding_mask], dim=0)
            else:
                batch_padding_mask = None
            batch_input_concat_aux = self._batch_alignment_aux(
                input_concat_aux, negative_input_concat_aux
            )
            batch_output = self._forward(
                batch_inputs,
                batch_timestep,
                cross_attn_cond=batch_cond,
                cross_attn_cond_mask=batch_cond_masks,
                cross_attn_aux=batch_cross_attn_aux,
                mask=batch_masks,
                input_concat_cond=batch_input_concat_cond,
                input_concat_aux=batch_input_concat_aux,
                local_add_cond=batch_local_add_cond,
                modular_local_cond=batch_modular_local_cond,
                global_embed=batch_global_cond,
                prepend_cond=batch_prepend_cond,
                prepend_cond_mask=batch_prepend_cond_mask,
                padding_mask=batch_padding_mask,
                return_info=return_info,
                **kwargs,
            )
            if return_info:
                (batch_output, info) = batch_output
            (cond_output, uncond_output) = torch.chunk(batch_output, 2, dim=0)
            if self.diffusion_objective == "v":
                cond_denoised = x * alpha[:, None, None] - cond_output * sigma[:, None, None]
                uncond_denoised = x * alpha[:, None, None] - uncond_output * sigma[:, None, None]
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                cond_denoised = x - cond_output * sigma[:, None, None]
                uncond_denoised = x - uncond_output * sigma[:, None, None]
            diff = cond_denoised - uncond_denoised
            if cfg_norm_threshold > 0:
                if padding_mask is not None:
                    mask = padding_mask.unsqueeze(1).float()
                    diff_masked = diff * mask
                    diff_norm = diff_masked.norm(p=2, dim=[-1, -2], keepdim=True)
                else:
                    diff_norm = diff.norm(p=2, dim=[-1, -2], keepdim=True)
                scale_factor = torch.minimum(torch.ones_like(diff), cfg_norm_threshold / diff_norm)
                diff *= scale_factor
            if apg_scale == 0.0:
                cfg_diff = diff
            elif apg_scale == 1.0:
                (_, diff_orthogonal) = self.apg_project(
                    diff, cond_denoised, padding_mask=padding_mask
                )
                cfg_diff = diff_orthogonal
            else:
                (diff_parallel, diff_orthogonal) = self.apg_project(
                    diff, cond_denoised, padding_mask=padding_mask
                )
                cfg_diff = apg_scale * diff_orthogonal + (1 - apg_scale) * diff
            cfg_denoised = cond_denoised + (cfg_scale - 1) * cfg_diff
            if self.diffusion_objective == "v":
                output = (x * alpha[:, None, None] - cfg_denoised) / sigma[:, None, None]
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                output = (x - cfg_denoised) / sigma[:, None, None]
            if scale_phi != 0.0:
                reduce_dims = tuple(range(1, output.ndim))
                cond_out_std = cond_output.float().std(
                    dim=reduce_dims, keepdim=True, unbiased=False
                )
                out_cfg_std = (
                    output.float()
                    .std(dim=reduce_dims, keepdim=True, unbiased=False)
                    .clamp_min(torch.finfo(torch.float32).eps)
                )
                rescaled = output * (cond_out_std / out_cfg_std).to(output.dtype)
                output = scale_phi * rescaled + (1 - scale_phi) * output
            if return_info:
                info["uncond_output"] = uncond_output
                return (output, info)
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
                global_embed=global_embed,
                prepend_cond=prepend_cond,
                prepend_cond_mask=prepend_cond_mask,
                padding_mask=padding_mask,
                mask=mask,
                return_info=return_info,
                **kwargs,
            )
