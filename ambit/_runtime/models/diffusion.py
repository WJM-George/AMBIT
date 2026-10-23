import torch
from torch import nn
from .dit import DiffusionTransformer


class ConditionedDiffusionModel(nn.Module):
    def __init__(
        self,
        *args,
        supports_cross_attention: bool = False,
        supports_input_concat: bool = False,
        supports_global_cond: bool = False,
        supports_prepend_cond: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.supports_cross_attention = supports_cross_attention
        self.supports_input_concat = supports_input_concat
        self.supports_global_cond = supports_global_cond
        self.supports_prepend_cond = supports_prepend_cond

    def forward(
        self,
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
        **kwargs,
    ):
        raise NotImplementedError()


class DiTWrapper(ConditionedDiffusionModel):
    def __init__(self, diffusion_objective: str, *args, **kwargs):
        super().__init__(
            supports_cross_attention=True, supports_global_cond=False, supports_input_concat=False
        )
        self.diffusion_objective = diffusion_objective
        self.model = DiffusionTransformer(*args, diffusion_objective=diffusion_objective, **kwargs)

    def forward(
        self,
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
        **kwargs,
    ):
        assert batch_cfg, "batch_cfg must be True for DiTWrapper"
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
            **kwargs,
        )


class ConditionedDiffusionModelWrapper(nn.Module):
    """Shared text and native FOA control conditioning."""

    def __init__(
        self,
        model,
        conditioner,
        io_channels,
        sample_rate,
        min_input_length,
        diffusion_objective="rectified_flow",
        mask_padding_attention=True,
        pretransform=None,
        cross_attn_cond_ids=(),
        global_cond_ids=(),
        input_concat_ids=(),
    ):
        super().__init__()
        if pretransform is not None or global_cond_ids or diffusion_objective != "rectified_flow":
            raise ValueError("Unsupported AMBIT renderer configuration")
        self.model = model
        self.conditioner = conditioner
        self.io_channels = io_channels
        self.sample_rate = sample_rate
        self.min_input_length = min_input_length
        self.diffusion_objective = diffusion_objective
        self.mask_padding_attention = mask_padding_attention
        self.pretransform = None
        self.cross_attn_cond_ids = tuple(cross_attn_cond_ids)
        self.input_concat_ids = tuple(input_concat_ids)

    def get_conditioning_inputs(self, conditioning_tensors, negative=False):
        text = [conditioning_tensors[key] for key in self.cross_attn_cond_ids]
        local = [conditioning_tensors[key] for key in self.input_concat_ids]
        if any((len(value) != 2 for value in [*text, *local])):
            raise ValueError("Conditioners must return embedding and mask pairs")
        prefix = "negative_" if negative else ""
        return {
            prefix + "cross_attn_cond": torch.cat([value[0] for value in text], dim=1)
            if text
            else None,
            prefix + "cross_attn_mask": torch.cat([value[1] for value in text], dim=1)
            if text
            else None,
            prefix + "input_concat_cond": torch.cat([value[0] for value in local], dim=1)
            if local
            else None,
        }

    def forward(self, x, t, cond, **kwargs):
        return self.model(x, t, **self.get_conditioning_inputs(cond), **kwargs)
