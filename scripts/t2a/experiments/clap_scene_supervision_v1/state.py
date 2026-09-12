"""Complete two-clock checkpoints; native encoder history is never relabeled."""
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

SCHEMA = 'editing_clap44_scene_event_two_optimizer_v1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise RuntimeError(f'Cannot overwrite published evidence: {path}')
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    return value


def fingerprint(value):
    digest = hashlib.sha256(); elements = 0

    def visit(item):
        nonlocal elements
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            visit(('tensor', str(tensor.dtype), tuple(tensor.shape)))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            elements += tensor.numel()
        elif isinstance(item, dict):
            digest.update(b'dict{')
            for key in sorted(item, key=lambda x: (type(x).__name__, str(x))):
                visit(key); visit(item[key])
            digest.update(b'}')
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode() + b'[')
            for part in item:
                visit(part)
            digest.update(b']')
        elif item is None or isinstance(item, (int, float, str, bool)):
            digest.update(type(item).__name__.encode() + b':' + json.dumps(item, allow_nan=False).encode() + b';')
        else:
            raise TypeError(f'Unsupported checkpoint fingerprint type: {type(item)}')
    visit(value)
    return {'sha256': digest.hexdigest(), 'tensor_elements': elements}


def expected_cursor(contract, updates):
    data = contract['data']; batches = data['rows'] // contract['world_size'] // data['pairs_per_rank']
    extra_epochs, next_batch = divmod(contract['parent']['next_batch'] + updates, batches)
    return contract['parent']['epoch'] + extra_epochs, next_batch


def parameter_schema(model):
    return [{'name': name, 'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in model.named_parameters()]


def validate_rng(state):
    if not isinstance(state, dict) or set(state) != {'python', 'numpy', 'torch', 'cuda'}:
        raise RuntimeError('Checkpoint requires complete Python/NumPy/Torch/CUDA rank RNG')
    random.Random().setstate(state['python'])
    n = state['numpy']
    if n['name'] != 'MT19937' or tuple(n['keys'].shape) != (624,) or n['keys'].dtype != torch.int64:
        raise RuntimeError('Invalid native NumPy RNG')
    np.random.RandomState(0).set_state((n['name'], n['keys'].cpu().numpy().astype(np.uint32), n['position'], n['has_gauss'], n['cached']))
    for key in ('torch', 'cuda'):
        tensor = state[key]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8 or tensor.ndim != 1 or tensor.numel() < 8:
            raise RuntimeError('Invalid rank generator byte state')
    torch.Generator().set_state(state['torch'].cpu())


def validate_payload(payload, contract):
    if payload.get('schema') != SCHEMA or payload.get('contract') != contract or contract.get('schema') != SCHEMA:
        raise RuntimeError('Scene-event checkpoint contract mismatch')
    if payload.get('quality_gate_passed') is not False:
        raise RuntimeError('Training state cannot declare quality acceptance')
    updates = payload.get('new_updates')
    if type(updates) is not int or not 0 <= updates <= contract['max_new_updates']:
        raise RuntimeError('Invalid new optimizer update count')
    if payload.get('step') != contract['parent']['step'] + updates:
        raise RuntimeError('Encoder and new-head step clocks disagree')
    if any(type(payload.get(key)) is not int for key in ('step', 'epoch', 'next_batch')):
        raise RuntimeError('Checkpoint progress fields must be integers')
    if (payload.get('epoch'), payload.get('next_batch')) != expected_cursor(contract, updates):
        raise RuntimeError('Checkpoint data position disagrees with native sampler progression')
    if contract['physical_gpus'] != [5, 6] or contract['world_size'] != 2:
        raise RuntimeError('This contract is limited to physical GPU5–6, in that rank order')
    if contract['gpu_topology']['physical_indices'] != contract['physical_gpus']:
        raise RuntimeError('GPU topology disagrees with physical allocation')
    mapping = contract['gpu_topology']['mapping']
    if [x['local_rank'] for x in mapping] != [0, 1] or [x['physical_index'] for x in mapping] != [5, 6] or len({x['uuid'] for x in mapping}) != 2:
        raise RuntimeError('GPU rank UUID binding is incomplete')
    for key, completed in (('encoder', payload['step']), ('readout', updates)):
        block = payload[key]; schema = contract['parameter_schema'][key]
        if set(block['model']) != {row['name'] for row in schema}:
            raise RuntimeError(f'{key} parameter coverage changed')
        for row in schema:
            tensor = block['model'][row['name']]
            if tuple(tensor.shape) != tuple(row['shape']) or str(tensor.dtype) != row['dtype'] or not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(f'Invalid {key} model tensor: {row["name"]}')
        groups = block['optimizer']['param_groups']; states = block['optimizer']['state']
        if len(groups) != 1 or groups[0]['params'] != list(range(len(schema))):
            raise RuntimeError(f'{key} optimizer parameter order changed')
        static = {k: v for k, v in groups[0].items() if k not in ('params', 'lr')}
        if json.loads(json.dumps(static)) != contract['optimizer_static_parameters'][key]:
            raise RuntimeError(f'{key} optimizer hyperparameters changed')
        if groups[0]['lr'] != contract['fixed_learning_rates'][key]:
            raise RuntimeError(f'{key} learning rate differs from the bounded continuation contract')
        if set(states) != (set(range(len(schema))) if completed else set()):
            raise RuntimeError(f'{key} optimizer moments are missing or stale')
        for index, value in states.items():
            if set(value) != {'step', 'exp_avg', 'exp_avg_sq'} or float(value['step']) != completed:
                raise RuntimeError(f'{key} optimizer clock mismatch')
            for moment in ('exp_avg', 'exp_avg_sq'):
                tensor = value[moment]
                if tuple(tensor.shape) != tuple(schema[index]['shape']) or tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                    raise RuntimeError(f'{key} optimizer moments invalid')
        scheduler = block['scheduler']
        if scheduler['last_epoch'] != completed or scheduler['_step_count'] != completed + 1:
            raise RuntimeError(f'{key} scheduler clock mismatch')
        if scheduler['_last_lr'] != [group['lr'] for group in groups]:
            raise RuntimeError(f'{key} scheduler and actual optimizer LR differ')
        if {k: v for k, v in scheduler.items() if k not in ('last_epoch', '_step_count', '_last_lr')} != contract['scheduler_static_parameters'][key]:
            raise RuntimeError(f'{key} scheduler configuration changed')
    rng = payload.get('rng_states')
    if not isinstance(rng, list) or len(rng) != contract['world_size']:
        raise RuntimeError('Missing rank RNG state')
    for state in rng:
        validate_rng(state)


def make_payload(model, optimizers, schedulers, updates, contract, rng_states):
    epoch, next_batch = expected_cursor(contract, updates)
    payload = {'schema': SCHEMA, 'contract': contract, 'new_updates': updates,
               'step': contract['parent']['step'] + updates, 'epoch': epoch, 'next_batch': next_batch,
               'rng_states': rng_states, 'quality_gate_passed': False}
    for key in optimizers:
        payload[key] = {'model': getattr(model, key).state_dict(), 'optimizer': optimizers[key].state_dict(),
                        'scheduler': schedulers[key].state_dict()}
    return payload


def manifest(path, payload):
    return {'schema': SCHEMA, 'checkpoint': str(path.resolve()), 'sha256': sha(path),
            'contract_sha256': sha(path.parent / 'TRAIN_CONTRACT.json'),
            **{key: payload[key] for key in ('step', 'new_updates', 'epoch', 'next_batch')},
            'quality_gate_passed': False}


def load_checkpoint(path, contract, *, require_manifest=True):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    if read(path.parent / 'TRAIN_CONTRACT.json') != contract:
        raise RuntimeError('Checkpoint belongs to another directory contract')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    validate_payload(payload, contract)
    if path.name != f'step-{payload["step"]:06d}.pt' or payload['new_updates'] < 1:
        raise RuntimeError('Only finalized nonzero-step states can be restored')
    identity = manifest(path, payload)
    sidecar = path.with_suffix('.manifest.json')
    if sidecar.exists():
        if read(sidecar) != identity:
            raise RuntimeError('Published checkpoint identity changed')
    elif require_manifest:
        raise RuntimeError('Checkpoint publication is incomplete')
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError('Checkpoint changed while reading')
    return payload, identity


def save_checkpoint(path, payload, contract):
    path = Path(path)
    if path.exists() or path.with_suffix('.manifest.json').exists():
        raise RuntimeError('Never overwrite a finalized scene-event checkpoint')
    if read(path.parent / 'TRAIN_CONTRACT.json') != contract:
        raise RuntimeError('Output training contract changed')
    validate_payload(payload, contract)
    if any(int(other.stem.split('-')[1]) >= payload['step'] for other in path.parent.glob('step-*.pt')):
        raise RuntimeError('Checkpoint publication would roll back a run')
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as handle:
        torch.save(cpu_copy(payload), handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    _, identity = load_checkpoint(path, contract, require_manifest=False)
    write(path.with_suffix('.manifest.json'), identity)
    write(path.parent / 'LATEST.json', identity, replace=True)
    return identity


def resolve_resume(directory, contract):
    directory = Path(directory)
    for path, expected in contract['source_sha256'].items():
        if sha(path) != expected:
            raise RuntimeError(f'Training source changed: {path}')
    candidates = sorted(directory.glob('step-*.pt'))
    if not candidates:
        if (directory / 'LATEST.json').exists():
            raise RuntimeError('LATEST points to absent checkpoints')
        return None
    recovered = []
    for path in candidates:
        _, identity = load_checkpoint(path, contract, require_manifest=False)
        if not path.with_suffix('.manifest.json').exists():
            write(path.with_suffix('.manifest.json'), identity); recovered.append(str(path))
    pointer = directory / 'LATEST.json'
    previous = None
    if pointer.exists():
        try:
            previous = read(pointer)
        except json.JSONDecodeError:
            pass
    if previous != identity:
        if pointer.exists():
            backup = directory / f'LATEST.unpublished.{sha(pointer)}.json'
            if not backup.exists():
                backup.write_bytes(pointer.read_bytes())
        write(pointer, identity, replace=True)
        recovered.append(str(pointer))
    if recovered:
        write(directory / 'PUBLICATION_RECOVERY.json', {'recovered': recovered, 'latest': identity}, replace=True)
    return identity
