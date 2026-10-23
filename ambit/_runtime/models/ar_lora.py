from contextvars import ContextVar
import math
import torch
from torch import nn
from torch.nn import functional as F
from ambit._runtime.models.generation_ar import ScenePlanTransfusionGenerationAR


class ARLinearResidual(nn.Module):
    def __init__(self, linear, *, rank, alpha):
        super().__init__()
        self.scale = float(alpha) / rank
        self.down = nn.Parameter(
            torch.empty(rank, linear.in_features, device=linear.weight.device, dtype=torch.float32)
        )
        self.up = nn.Parameter(
            torch.zeros(linear.out_features, rank, device=linear.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, inputs):
        return F.linear(F.linear(inputs, self.down), self.up) * self.scale


class AdaptedGenerationAR(ScenePlanTransfusionGenerationAR):
    def __init__(self, base, codec, *, rank=8, alpha=8.0):
        if not isinstance(rank, int) or not 1 <= rank <= 64 or (not 0 < float(alpha) <= 64):
            raise ValueError("AR LoRA requires rank 1..64 and alpha in (0,64]")
        super().__init__(
            p10_dit=base.p10_dit,
            prompt_conditioner=base.prompt_conditioner,
            pad_id=base.pad_id,
            vocab_size=base.vocab_size,
        )
        self.ar_adapter = base.ar_adapter
        self.lora_rank = rank
        self.lora_alpha = float(alpha)
        self.ar_lora = nn.ModuleDict()
        self._ar_lora_active = ContextVar(f"ar_lora_active_{id(self)}", default=False)
        for index, layer in enumerate(self.shared_transformer.layers):
            for attention_name, projection_names in [
                ("self_attn", ("to_qkv", "to_out")),
                ("cross_attn", ("to_q", "to_kv", "to_out")),
            ]:
                attention = getattr(layer, attention_name)
                for projection_name in projection_names:
                    linear = getattr(attention, projection_name)
                    if not isinstance(linear, nn.Linear):
                        raise TypeError("AR LoRA requires canonical linear attention projections")
                    if getattr(linear, "_generation_ar_lora_installed", False):
                        raise ValueError("AR LoRA already installed")
                    key = f"layer_{index:02d}__{attention_name}__{projection_name}"
                    residual = ARLinearResidual(linear, rank=rank, alpha=alpha)
                    self.ar_lora[key] = residual
                    original = linear.forward

                    def scoped(inputs, *args, _original=original, _residual=residual, **kwargs):
                        output = _original(inputs, *args, **kwargs)
                        if self._ar_lora_active.get():
                            output = output + _residual(inputs).to(dtype=output.dtype)
                        return output

                    linear.forward = scoped
                    linear._generation_ar_lora_installed = True

    def forward(self, plan_input_ids, plan_attention_mask, request_context, request_attention_mask):
        token = self._ar_lora_active.set(True)
        try:
            return super().forward(
                plan_input_ids, plan_attention_mask, request_context, request_attention_mask
            )
        finally:
            self._ar_lora_active.reset(token)

    @torch.no_grad()
    def prepare_decode_cache(
        self, request_context, request_attention_mask, *, max_plan_tokens=1024
    ):
        token = self._ar_lora_active.set(True)
        try:
            return super().prepare_decode_cache(
                request_context, request_attention_mask, max_plan_tokens=max_plan_tokens
            )
        finally:
            self._ar_lora_active.reset(token)

    @torch.no_grad()
    def decode_step(self, token_ids, cache):
        token = self._ar_lora_active.set(True)
        try:
            return super().decode_step(token_ids, cache)
        finally:
            self._ar_lora_active.reset(token)
