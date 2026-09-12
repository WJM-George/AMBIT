"""Fresh three-rank optimizer lineage, distinct from native two-rank resume."""
import math
import os
from pathlib import Path

import torch
from scripts.t2a.experiments.clap_scene_supervision_v1.state import (
    sha, read, write, cpu_copy, fingerprint, parameter_schema, validate_rng,
)
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_pretrain import optimizer_and_scheduler

SCHEMA = 'editing_clap44_factual50k_warmstart_three_rank_v1'


def cursor(contract, updates):
    batches = contract['data']['rows'] // 3 // contract['data']['pairs_per_rank']
    return divmod(updates, batches)


def multiplier(step, settings):
    warmup, maximum = settings['warmup_steps'], settings['max_steps']
    if step < warmup:
        return max(1, step + 1) / max(1, warmup)
    return .05 + .95 * (1 + math.cos(math.pi * min(1, (step - warmup) / max(1, maximum - warmup)))) / 2


def build_optimizers(model, contract):
    optimizers, schedulers = {}, {}
    for key in ('encoder', 'readout'):
        optimizers[key], schedulers[key] = optimizer_and_scheduler(getattr(model, key), contract['optimizers'][key])
        if optimizers[key].state:
            raise RuntimeError('Warm start must create empty optimizer moments')
    return optimizers, schedulers


def validate(payload, contract):
    if contract['schema'] != SCHEMA or payload.get('schema') != SCHEMA or payload.get('contract') != contract:
        raise RuntimeError('Factual50k checkpoint contract mismatch')
    if contract['physical_gpus'] != [5, 6, 7] or contract['world_size'] != 3:
        raise RuntimeError('Full factual CLAP is restricted to GPU5-7')
    topology = contract['gpu_topology']
    if topology['physical_indices'] != [5, 6, 7] or [x['local_rank'] for x in topology['mapping']] != [0, 1, 2]:
        raise RuntimeError('Incomplete three-rank topology')
    if [x['physical_index'] for x in topology['mapping']] != [5, 6, 7] or len({x['uuid'] for x in topology['mapping']}) != 3:
        raise RuntimeError('Physical rank mapping changed')
    updates = payload.get('new_updates')
    if type(updates) is not int or not 0 <= updates <= contract['max_new_updates'] == 50000:
        raise RuntimeError('Invalid fresh update count')
    if payload.get('step') != updates or payload.get('initialization_step') != 20000:
        raise RuntimeError('New optimizer clock must not inherit the native 20k clock')
    if (payload.get('epoch'), payload.get('next_batch')) != cursor(contract, updates):
        raise RuntimeError('Data cursor disagrees with full1M three-rank sampler')
    if payload.get('quality_gate_passed') is not False:
        raise RuntimeError('Fit checkpoints cannot claim effect acceptance')
    for key in ('encoder', 'readout'):
        block = payload[key]; schema = contract['parameter_schema'][key]
        if set(block) != {'model', 'optimizer', 'scheduler'} or set(block['model']) != {x['name'] for x in schema}:
            raise RuntimeError('Model/optimizer coverage mismatch')
        for row in schema:
            tensor = block['model'][row['name']]
            if list(tensor.shape) != row['shape'] or str(tensor.dtype) != row['dtype'] or not bool(torch.isfinite(tensor).all()):
                raise RuntimeError(('Invalid model tensor', key, row['name']))
        groups, moments = block['optimizer']['param_groups'], block['optimizer']['state']
        if len(groups) != 1 or groups[0]['params'] != list(range(len(schema))):
            raise RuntimeError('Optimizer parameter order differs')
        static = {k: v for k, v in groups[0].items() if k not in ('params', 'lr')}
        import json
        if json.loads(json.dumps(static)) != contract['optimizer_static_parameters'][key]:
            raise RuntimeError('Optimizer hyperparameters changed')
        wanted_lr = contract['optimizers'][key]['learning_rate'] * multiplier(updates, contract['optimizers'][key])
        if groups[0]['lr'] != wanted_lr:
            raise RuntimeError(('Scheduler and optimizer learning rate differ', key, groups[0]['lr'], wanted_lr))
        if set(moments) != (set(range(len(schema))) if updates else set()):
            raise RuntimeError('Missing or unexpected AdamW moments')
        for i, value in moments.items():
            if set(value) != {'step', 'exp_avg', 'exp_avg_sq'} or float(value['step']) != updates:
                raise RuntimeError('AdamW step did not use the new optimizer clock')
            for name in ('exp_avg', 'exp_avg_sq'):
                v = value[name]
                if list(v.shape) != schema[i]['shape'] or v.dtype != torch.float32 or not bool(torch.isfinite(v).all()):
                    raise RuntimeError('Invalid full AdamW moments')
        schedule = block['scheduler']
        if schedule['last_epoch'] != updates or schedule['_step_count'] != updates + 1 or schedule['_last_lr'] != [wanted_lr]:
            raise RuntimeError('Scheduler clock is incomplete')
        if {k: v for k, v in schedule.items() if k not in ('last_epoch', '_step_count', '_last_lr')} != contract['scheduler_static_parameters'][key]:
            raise RuntimeError('Scheduler configuration changed')
    rng = payload.get('rng_states')
    if not isinstance(rng, list) or len(rng) != 3:
        raise RuntimeError('Three complete RNG states are required')
    for value in rng:
        validate_rng(value)


def make(model, optimizers, schedulers, updates, contract, rng_states):
    epoch, offset = cursor(contract, updates)
    value = {'schema': SCHEMA, 'contract': contract, 'step': updates, 'new_updates': updates,
        'initialization_step': 20000, 'epoch': epoch, 'next_batch': offset,
        'rng_states': rng_states, 'quality_gate_passed': False}
    for key in optimizers:
        value[key] = {'model': getattr(model, key).state_dict(), 'optimizer': optimizers[key].state_dict(),
            'scheduler': schedulers[key].state_dict()}
    return value


def identity(path, payload):
    return {'schema': SCHEMA, 'checkpoint': str(path.resolve()), 'sha256': sha(path),
        'contract_sha256': sha(path.parent / 'TRAIN_CONTRACT.json'),
        **{k: payload[k] for k in ('new_updates', 'step', 'initialization_step', 'epoch', 'next_batch')},
        'quality_gate_passed': False}


def load(path, contract, require_manifest=True):
    path = Path(path).resolve(strict=True); before = path.stat()
    if read(path.parent / 'TRAIN_CONTRACT.json') != contract:
        raise RuntimeError('Checkpoint directory has a different contract')
    value = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    validate(value, contract)
    if path.name != f'update-{value["new_updates"]:06d}.pt' or value['new_updates'] < 1:
        raise RuntimeError('Only finalized nonzero updates may be restored')
    ident = identity(path, value); sidecar = path.with_suffix('.manifest.json')
    if sidecar.exists():
        if read(sidecar) != ident:
            raise RuntimeError('Published checkpoint hash changed')
    elif require_manifest:
        raise RuntimeError('Missing checkpoint manifest')
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise RuntimeError('Checkpoint changed during reading')
    return value, ident


def save(path, value, contract):
    path = Path(path); validate(value, contract)
    if path.exists() or path.with_suffix('.manifest.json').exists():
        raise RuntimeError('Never replace a finalized checkpoint')
    if read(path.parent / 'TRAIN_CONTRACT.json') != contract:
        raise RuntimeError('Output contract changed')
    if any(int(p.stem.split('-')[1]) >= value['new_updates'] for p in path.parent.glob('update-*.pt')):
        raise RuntimeError('Refusing to roll back an existing run')
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('wb') as stream:
        torch.save(cpu_copy(value), stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    _, ident = load(path, contract, require_manifest=False)
    write(path.with_suffix('.manifest.json'), ident)
    write(path.parent / 'LATEST.json', ident, replace=True)
    return ident


def latest(directory, contract):
    directory = Path(directory); paths = sorted(directory.glob('update-*.pt'))
    if not paths:
        if (directory / 'LATEST.json').exists():
            raise RuntimeError('LATEST references an absent checkpoint')
        return None
    # Older immutable states retain their own manifests. The newest binary is
    # independently validated before recovering publication interrupted by exit.
    _, ident = load(paths[-1], contract, require_manifest=False)
    sidecar = paths[-1].with_suffix('.manifest.json')
    if not sidecar.exists(): write(sidecar, ident)
    pointer = directory / 'LATEST.json'
    previous = None
    if pointer.exists():
        import json
        try: previous = read(pointer)
        except json.JSONDecodeError: pass
    if previous != ident:
        if pointer.exists():
            backup = directory / f'LATEST.unpublished.{sha(pointer)}.json'
            if not backup.exists(): backup.write_bytes(pointer.read_bytes())
        write(pointer, ident, replace=True)
        write(directory / 'PUBLICATION_RECOVERY.json', {'latest': ident}, replace=True)
    return ident
