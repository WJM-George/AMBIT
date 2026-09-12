"""Explicit frozen-Qwen execution policy for reproducible EVENT/P10 comparisons."""
from __future__ import annotations


def qwen_fla_kernel_receipt():
    from fla.ops.common import chunk_delta_h
    from fla.ops.utils.cache import FLA_CACHE_MODE
    kernel = chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    while getattr(kernel, 'configs', None) is None:
        kernel = getattr(kernel, 'fn', None)
        if kernel is None:
            raise RuntimeError('cannot locate Qwen FLA GatedDeltaRule autotuner')
    def describe(config):
        return {'kwargs': dict(config.kwargs), 'num_warps': int(config.num_warps), 'num_stages': int(config.num_stages)}
    return kernel, {'fla_cache_mode': FLA_CACHE_MODE.value,
        'configs': [describe(c) for c in kernel.configs],
        'selected': {str(key): describe(value) for key, value in getattr(kernel, 'cache', {}).items()}}


def pin_qwen_fla_kernel(*, num_warps=4):
    """Pin the configuration observed in the reproduced native reference.

    The released P10 reference reproduction selected BV32/warps4/stages2. An
    AR-first process can instead autotune to warps2 and change P10 output with
    identical weights. This is an explicit runtime amendment. It must be
    applied equally to native reference, controls and candidates before use.
    """
    if num_warps not in (2, 4):
        raise ValueError('only the measured Qwen FLA kernel configurations are supported')
    kernel, before = qwen_fla_kernel_receipt()
    configs = [config for config in kernel.configs if config.kwargs.get('BV') == 32
        and config.num_warps == num_warps and config.num_stages == 2]
    if len(configs) != 1:
        raise RuntimeError('installed FLA lacks the declared single reproducible kernel configuration')
    kernel.configs = configs
    kernel.cache.clear()
    _, after = qwen_fla_kernel_receipt()
    return {'contract': f'event_p10_qwen_fla_fixed_bv32_w{num_warps}_s2_v1',
        'before': before, 'after': after, 'checkpoint_parameters_changed': False}


def configure_qwen_torch_reference(backbone):
    """The existing repo's canonical comparison backend, scoped to one Qwen.

    This avoids optional FLA autotuning in a shared AR/DiT training process.
    It changes the numerical runtime, so a separate unchanged-weight reference
    must be rendered with the same backend before assessing an RL candidate.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35
    count = 0
    for module in backbone.modules():
        if isinstance(module, qwen35.Qwen3_5GatedDeltaNet):
            module.causal_conv1d_fn = None
            module.causal_conv1d_update = qwen35.torch_causal_conv1d_update
            module.chunk_gated_delta_rule = qwen35.torch_chunk_gated_delta_rule
            module.recurrent_gated_delta_rule = qwen35.torch_recurrent_gated_delta_rule
            count += 1
    if not count:
        raise ValueError('canonical EVENT runtime requires the selected Qwen3.5 backbone')
    backbone.config._attn_implementation = 'eager'
    from .event_qwen_norm_runtime import qwen_fused_norm_receipt
    norm = qwen_fused_norm_receipt(backbone)
    return {'contract': 'event_qwen_torch_reference_v1', 'gated_delta_layers': count,
        'full_attention': 'eager', 'optional_fla_enabled': bool(norm['fused_norm_layers']),
        'optional_fla_gated_delta_enabled': False, 'fused_gated_norm': norm,
        'checkpoint_parameters_changed': False}
