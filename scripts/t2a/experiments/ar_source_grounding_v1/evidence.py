"""Small immutable evidence helpers for the bounded AR experiment."""
import copy
import hashlib
import json
import os
from pathlib import Path
import torch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    return copy.deepcopy(value)


def exact(left, right, path='state'):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right), path
        return left.numel()
    assert type(left) is type(right), path
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        return sum(exact(left[k], right[k], f'{path}.{k}') for k in left)
    if isinstance(left, (tuple, list)):
        assert len(left) == len(right), path
        return sum(exact(a, b, f'{path}.{i}') for i, (a, b) in enumerate(zip(left, right)))
    assert left == right, path
    return 0


def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
