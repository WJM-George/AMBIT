"""Small reproducibility helpers shared by the bounded EVENT experiment."""
from __future__ import annotations

import hashlib
import json

import torch


def tensor_fingerprint(value):
    tensor = value.detach().contiguous().cpu()
    digest = hashlib.sha256(json.dumps([str(tensor.dtype), list(tensor.shape)]).encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_fingerprint(module):
    """Hash all registered tensors once, including buffers, without alias names."""
    digest = hashlib.sha256()
    for kind, iterator in [('parameter', module.named_parameters()), ('buffer', module.named_buffers())]:
        for name, value in sorted(iterator):
            tensor = value.detach().contiguous().cpu()
            digest.update(json.dumps([kind, name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def runtime_receipt():
    properties = torch.cuda.get_device_properties(0)
    return {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'gpu': properties.name,
        'compute_capability': [properties.major, properties.minor],
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}
