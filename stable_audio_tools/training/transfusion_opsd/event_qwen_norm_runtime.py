"""Explicit process-local layout for the remaining frozen-Qwen FLA norm.

The torch gated-delta reference does not replace FusedRMSNormGated. Its
autotuned layouts can round differently with unchanged weights. A declared
pin applies to teacher, student and frozen copies in this process; it never
edits the installed FLA package or a checkpoint.
"""
from copy import deepcopy
import inspect

from .provenance import sha256_file


_PINNED_SPEC = None
_CONFIG = {'kwargs': {'BT': 16}, 'num_warps': 4, 'num_stages': 3}
_CONTRACT = 'event_qwen_fused_norm_bt16_w4_s3_v1'


def _describe(config):
    return {'kwargs': dict(config.kwargs), 'num_warps': config.num_warps,
        'num_stages': config.num_stages}


def _autotuner(kernel):
    while getattr(kernel, 'configs', None) is None:
        kernel = getattr(kernel, 'fn', None)
        if kernel is None:
            raise ValueError('cannot locate the frozen-Qwen gated-norm autotuner')
    return kernel


def _restrict_kernel(kernel, config):
    """Validate before mutation; remove earlier measurements and their choice."""
    tuner = _autotuner(kernel)
    selected = [value for value in tuner.configs if _describe(value) == config]
    if len(selected) != 1:
        raise ValueError('installed gated norm lacks the declared single layout')
    tuner.configs = selected
    tuner.cache.clear()
    return tuner


def pin_qwen_fused_norm(spec):
    global _PINNED_SPEC
    if (not isinstance(spec, dict) or set(spec) != {'contract', 'source_sha256'}
            or spec['contract'] != _CONTRACT or not isinstance(spec['source_sha256'], str)
            or len(spec['source_sha256']) != 64
            or any(char not in '0123456789abcdef' for char in spec['source_sha256'])):
        raise ValueError('declare the verified frozen-Qwen norm layout and exact FLA source')
    if _PINNED_SPEC is not None and _PINNED_SPEC != spec:
        raise ValueError('cannot switch the declared Qwen norm runtime inside one process')
    from fla.modules import fused_norm_gate
    source = inspect.getfile(fused_norm_gate)
    if sha256_file(source) != spec['source_sha256']:
        raise ValueError('frozen-Qwen gated norm source differs from the protocol')
    _restrict_kernel(fused_norm_gate.layer_norm_gated_fwd_kernel, _CONFIG)
    _PINNED_SPEC = deepcopy(spec)
    return {'spec': deepcopy(spec), 'config': deepcopy(_CONFIG),
        'source': source, 'autotuning_enabled': False,
        'scope': 'current process; all matching fused norms; Qwen head widths at most 512',
        'checkpoint_parameters_changed': False}


def qwen_fused_norm_receipt(backbone):
    norms = [module.norm for module in backbone.modules()
        if type(module).__name__ == 'Qwen3_5GatedDeltaNet']
    fused = [module for module in norms if type(module).__name__ == 'FusedRMSNormGated']
    if _PINNED_SPEC is not None:
        if len(fused) != len(norms) or not norms or any(module.hidden_size > 512 for module in fused):
            raise ValueError('declared Qwen norm pin does not cover this complete backbone')
        from fla.modules import fused_norm_gate
        tuner = _autotuner(fused_norm_gate.layer_norm_gated_fwd_kernel)
        if ([_describe(config) for config in tuner.configs] != [_CONFIG]
                or any(_describe(config) != _CONFIG for config in tuner.cache.values())):
            raise ValueError('frozen-Qwen gated norm lost its declared layout')
    return {'fused_norm_layers': len(fused), 'total_gated_delta_layers': len(norms),
        'pinned_spec': deepcopy(_PINNED_SPEC),
        'autotuning_enabled': bool(fused) and _PINNED_SPEC is None}
