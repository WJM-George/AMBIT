"""Explicit FP32 DiT comparison runtime; released weights are unchanged.

Both frozen references and candidates must use this runtime when attributing
an effect to learning. It is not numerically identical to BF16 execution.
"""
from __future__ import annotations

from contextvars import ContextVar

import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend


def configure_event_dit_fp32(diffusion, *, sample_model=None):
    if getattr(diffusion, '_event_dit_fp32_configured', False):
        raise ValueError('FP32 DiT runtime is already installed')
    selected = diffusion.model if sample_model is None else sample_model
    core = selected.model
    if any(p.dtype != torch.float32 for p in selected.parameters()):
        raise ValueError('restore the selected DiT directly with FP32 master parameters')
    if core.sceneplan_alignment is not None:
        raise ValueError('initial FP32 comparison covers the selected no-alignment P10')
    active = ContextVar(f'event_dit_fp32_{id(diffusion)}', default=False)
    linear_count = 0
    seen = set()
    for owner in (selected, diffusion.conditioner):
        for module in owner.modules():
            if not isinstance(module, torch.nn.Linear) or id(module) in seen:
                continue
            seen.add(id(module))
            original = module.forward
            def linear(inputs, *args, _original=original, _module=module, **kwargs):
                if not active.get():
                    return _original(inputs, *args, **kwargs)
                if args or kwargs or inputs.dtype != torch.float32:
                    raise ValueError('FP32 DiT linear projection requires its ordinary FP32 tensor input')
                # The installed F.linear path for rank-three tensors changes
                # its GEMM grouping when weights are frozen. Explicit rank-two
                # projection gives frozen and trainable copies the same math.
                shape = inputs.shape[:-1]
                output = F.linear(inputs.reshape(-1, inputs.shape[-1]), _module.weight, _module.bias)
                return output.reshape(*shape, _module.out_features)
            module.forward = linear
            linear_count += 1
    count = 0
    for layer in core.transformer.layers:
        for module in (layer.self_attn, layer.cross_attn):
            original = module.apply_attn
            def attention(q, k, v, causal=None, *, _original=original, **kwargs):
                if not active.get():
                    return _original(q, k, v, causal=causal, **kwargs)
                if any(value.dtype != torch.float32 for value in (q, k, v)):
                    raise ValueError('FP32 DiT attention received a lower-precision tensor')
                if any(kwargs.get(name) is not None for name in ('attention_bias', 'flex_attention_block_mask',
                        'flex_attention_score_mod', 'flash_attn_sliding_window')):
                    raise ValueError('selected FP32 comparison requires ordinary full attention')
                if q.shape[1] != k.shape[1]:
                    if q.shape[1] % k.shape[1]:
                        raise ValueError('attention query and KV head counts are incompatible')
                    repeats = q.shape[1] // k.shape[1]
                    k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
                padding = kwargs.get('padding_mask')
                packed = kwargs.get('varlen_metadata') is not None
                exact_keys = bool(kwargs.get('mask_padding_logits')) or packed
                mask = None
                if padding is not None:
                    padding = padding.to(device=q.device, dtype=torch.bool)
                    if exact_keys:
                        mask = padding[:, None, None, :].expand(-1, 1, q.shape[-2], -1)
                    else:
                        v = v * padding[:, None, :, None].to(v.dtype)
                if causal and mask is not None:
                    triangular = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool,
                        device=q.device).tril(diagonal=k.shape[-2] - q.shape[-2])
                    mask = mask & triangular[None, None]
                with sdpa_kernel([SDPBackend.MATH]):
                    result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                        is_causal=bool(causal) if mask is None else False, dropout_p=0.)
                if packed and padding is not None:
                    result = result * padding[:, None, :, None].to(result.dtype)
                return result
            module.apply_attn = attention
            count += 1
    original_model = selected.forward
    def model_forward(*args, **kwargs):
        token = active.set(True)
        try:
            with torch.autocast('cuda', enabled=False):
                return original_model(*args, **kwargs)
        finally:
            active.reset(token)
    selected.forward = model_forward
    # EVENT AR calls the prompt conditioner directly, so its released BF16
    # request path is unaffected by this DiT-only multiconditioner wrapper.
    original_conditioner = diffusion.conditioner.forward
    def conditioner_forward(*args, **kwargs):
        token = active.set(True)
        try:
            with torch.autocast('cuda', enabled=False):
                return original_conditioner(*args, **kwargs)
        finally:
            active.reset(token)
    diffusion.conditioner.forward = conditioner_forward
    diffusion._event_dit_fp32_configured = True
    return {'contract': 'event_dit_trainable_conditioning_and_backbone_fp32_v1',
        'attention_modules': count, 'attention': 'FP32 SDPA math with native key/query padding semantics',
        'linear_modules': linear_count, 'linear_projection': 'explicit rank-two FP32 GEMM for frozen/trainable parity',
        'checkpoint_parameters_changed': False, 'frozen_qwen_and_vae_precision_changed': False,
        'ar_forward_precision_changed': False, 'requires_matched_frozen_reference': True}
