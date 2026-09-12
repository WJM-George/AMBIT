"""Complete typed state fingerprints without duplicate optimizer payload files."""
import hashlib
import json
import math
import torch

STATE_KEYS = ('diffusion_state_dict', 'editing_ar_specific_state_dict', 'optimizer',
              'scheduler', 'rng_states_by_rank', 'global_step', 'epoch', 'next_batch')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    elements = 0; tensors = 0
    def visit(x):
        nonlocal elements, tensors
        if isinstance(x, torch.Tensor):
            v = x.detach().cpu().contiguous()
            elements += v.numel(); tensors += 1
            return {'type': 'torch.Tensor', 'dtype': str(v.dtype), 'shape': list(v.shape),
                    'sha256': hashlib.sha256(v.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        if isinstance(x, dict):
            items = [[visit(k), visit(v)] for k, v in x.items()]
            items.sort(key=lambda item: canonical(item[0]))
            return {'type': 'dict', 'items': items}
        if isinstance(x, (list, tuple)):
            return {'type': type(x).__name__, 'items': [visit(v) for v in x]}
        if isinstance(x, float):
            assert math.isfinite(x)
            return {'type': 'float', 'hex': x.hex()}
        assert x is None or type(x) in (str, int, bool), type(x)
        return {'type': type(x).__name__, 'value': x}
    tree = visit(value)
    return {'schema': 'editing_complete_typed_state_fingerprint_v1', 'tree': tree,
            'tensor_elements': elements, 'tensors': tensors,
            'sha256': hashlib.sha256(canonical(tree).encode()).hexdigest()}


def training_state(payload):
    return {k: payload[k] for k in STATE_KEYS}


def old_prefix_state(snapshot):
    return {**{k: snapshot[k] for k in STATE_KEYS if k != 'rng_states_by_rank'},
            'rng_states_by_rank': [snapshot['rng']]}
