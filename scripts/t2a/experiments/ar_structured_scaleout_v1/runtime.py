"""Resume the existing joint optimizer on six GPUs, keeping each GPU's batch.

The immutable three-GPU implementation supplies the model, objective and loop.
Only allocation, checkpoint lineage and the declared RNG boundary differ here.
"""
from __future__ import annotations

import argparse
import copy
import functools
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from scripts.t2a.experiments.ar_structured_v1 import runtime as base
from scripts.t2a.experiments.ar_structured_scaleout_v1.sampler import ScaleoutSampler, POLICY

GPUS = [2, 3, 4, 5, 6, 7]
SCHEMA = 'structured_joint_AR_EditingDiT_new50k_scaleout_v1'
RNG_POLICY = 'explicit_six_rank_seed_boundary_then_native_exact_resume_v1'


def source_inventory():
    extra = {str(p.relative_to(ROOT)): base.sha(p) for p in Path(__file__).parent.glob('*.py')
             if p.name not in ('qa.py', 'prepare.py')}
    return {**base.source_inventory(), **extra}


def validate_config(cfg):
    transition = cfg['scaleout']
    if base.sha(transition['parent_config']) != transition['parent_config_sha256']:
        raise RuntimeError('The three-GPU recipe changed')
    parent = base.read(transition['parent_config'])
    base.validate_config(parent)
    if base.source_inventory() != parent['training_source_sha256']:
        raise RuntimeError('The original training implementation changed')
    direction = base.read(transition['user_direction'])
    if (base.sha(transition['user_direction']) != transition['user_direction_sha256']
            or direction.get('physical_gpus_after_10000') != GPUS
            or direction.get('switch_at_update') != 10000
            or transition['minimum_step'] != 10000 or transition['rng_policy'] != RNG_POLICY
            or direction.get('keep_per_gpu_batch_sizes') != {'short': 64, 'long': 40}):
        raise ValueError('Six-GPU continuation requires the explicit 10k user instruction')
    expected = copy.deepcopy(parent)
    expected['schema'] = SCHEMA
    expected['physical_gpus'] = GPUS
    expected['numerics'] = {str(rank): parent['numerics'][str(rank // 2)] for rank in range(6)}
    expected['scaleout'] = transition
    expected['training_source_sha256'] = source_inventory()
    if cfg != expected:
        raise ValueError('Scale-out may only expand ranks and declare its checkpoint/data lineage')
    for name in ('short_batch_size', 'long_batch_size'):
        if parent['performance'][name] != cfg['performance'][name]:
            raise ValueError('The user requires unchanged per-GPU batch sizes')
    parent_run = Path(transition['parent_run']).resolve(strict=True)
    if base.sha(parent_run / 'RUN_CONTRACT.json') != transition['parent_contract_sha256']:
        raise ValueError('Scale-out must continue the declared production 10k checkpoint')
    return parent


def validate_parent_payload(payload, identity, cfg, boundary):
    transition = cfg['scaleout']
    contract = payload['run_contract']
    step = int(identity['step'])
    if (identity != boundary['checkpoint']
            or Path(identity['checkpoint']).resolve() != Path(transition['parent_run']).resolve() / f'checkpoints/step-{step:08d}.pt'
            or identity['run_contract_sha256'] != transition['parent_contract_sha256']
            or contract != base.read(Path(transition['parent_run']) / 'RUN_CONTRACT.json')
            or contract['world_size'] != 3 or contract['physical_gpus'] != [5, 6, 7]
            or not transition['minimum_step'] <= step < cfg['new_updates'] - 4
            or step % cfg['recovery_every'] != 0
            or payload['global_step'] != step or payload['scheduler']['last_epoch'] != step
            or payload['epoch'] != boundary['parent_epoch']
            or payload['next_batch'] != boundary['parent_next_batch']):
        raise RuntimeError('Checkpoint is not the authorized three-to-six GPU boundary')
    return identity


def make_contract(cfg, run_dir, topology, provenance, codec, boundary):
    contract = base.make_contract(cfg, run_dir, topology, provenance, codec)
    contract.update(world_size=6, physical_gpus=GPUS, source_sha256=source_inventory())
    contract['continuation'] = {
        'parent_checkpoint': boundary['checkpoint'],
        'optimizer_and_scheduler_transferred': True,
        'data_cursor_policy': POLICY,
        'data_boundary_sha256': base.sha(cfg['scaleout']['boundary_file']),
        'data_boundary_file': cfg['scaleout']['boundary_file'],
        'same_per_gpu_batch_sizes': {'short': 64, 'long': 40},
        'global_batch_sizes': {'short': 384, 'long': 240},
        'already_consumed_parent_examples_replayed': False,
        'dropped_boundary_examples': len(boundary['dropped_incomplete_batch_ordinals']),
        'rng_policy': RNG_POLICY,
        'bitwise_equal_to_three_rank_future': False,
        'new_optimizer_updates_after_parent': cfg['new_updates'] - boundary['checkpoint']['step'],
    }
    return contract


def restore_training_state(module, optimizer, scheduler, payload, *, include_rng):
    module.diffusion.load_state_dict(payload['diffusion_state_dict'], strict=True)
    base.load_ar_specific(module.ar, payload['editing_ar_specific_state_dict'])
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    restored = {'diffusion_state_dict': module.diffusion.state_dict(),
        'editing_ar_specific_state_dict': base.ar_specific_state(module.ar),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        **{key: int(payload[key]) for key in ('global_step', 'epoch', 'next_batch')}}
    if include_rng:
        restored['rng_states_by_rank'] = payload['rng_states_by_rank']
    actual = base.fingerprint(restored)
    if actual != base.fingerprint({k: payload[k] for k in restored}):
        raise RuntimeError('Model, optimizer, scheduler or data cursor did not transfer exactly')
    if module.ar.shared_transformer is not module.diffusion.model.model.transformer:
        raise RuntimeError('AR and DiT lost their shared Transformer')
    return actual


def run(args):
    cfg = base.read(args.config)
    validate_config(cfg)
    if torch.cuda.is_initialized() or os.environ.get('EDITING_GPUS') != '2,3,4,5,6,7':
        raise RuntimeError('Lease and select GPU2–7 before CUDA initialization')
    boundary = base.read(cfg['scaleout']['boundary_file'])
    parent_checkpoint = Path(boundary['checkpoint']['checkpoint']).resolve(strict=True)
    if base.read(parent_checkpoint.with_suffix('.manifest.json')) != boundary['checkpoint']:
        raise RuntimeError('Parent checkpoint publication changed')
    migrating = args.resume.resolve(strict=True) == parent_checkpoint
    if not boundary['checkpoint']['step'] < args.stop_updates <= cfg['new_updates']:
        raise ValueError('Continue the original 50k clock after its 10k boundary')
    run_dir = base.initialization.assert_separate_output(args.run_dir, cfg['protected_DiT50k'])
    if run_dir == Path(cfg['scaleout']['parent_run']).resolve():
        raise ValueError('Six-GPU state needs its own run contract and output directory')
    run_dir.mkdir(parents=True, exist_ok=True)
    if migrating and any((run_dir / 'checkpoints').glob('step-*.pt')):
        raise RuntimeError('Existing six-GPU checkpoints require native six-GPU resume')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rank, local, world, device, topology = base.allocated_runtime.distributed(timeout_seconds=1800)
    if world != 6:
        raise ValueError('All six allocated ranks are required')
    out = run_dir / 'attempts' / args.attempt / f'rank{rank}'
    out.mkdir(parents=True, exist_ok=False)
    base.write(out / 'STARTED.json', {'at': base.now(), 'pid': os.getpid(), 'physical_gpu': GPUS[rank]})
    numerical = cfg['numerics'][str(rank)]
    catalog = Path(numerical['directory']) / 'AUTOTUNE_CATALOG.json'
    if base.sha(catalog) != numerical['sha256']:
        raise RuntimeError('Pinned numerical catalog changed')
    for record in base.read(catalog)['records'].values():
        if base.sha(record['source_file']) != record['source_sha256']:
            raise RuntimeError('Pinned numerical implementation changed')
    finish_numerics = base.install_autotune_observer({'name': 'autotune', 'mode': 'pin',
        'reference_case': numerical['directory']}, out)
    flash_calls, restore_flash = base.original_runtime.install_flash(cfg['base_AR_configuration']['numerical_execution'])
    try:
        base.native._seed_everything(cfg['seed'], 0)
        module, codec, groups, provenance = base.initialization.build(cfg)
        module.diffusion.model.model.activation_checkpointing = cfg['performance']['RF_activation_checkpointing']
        teacher = base.make_teacher(module, cfg)
        module.to(device).train()
        optimizer, scheduler = base.make_optimizer(groups, cfg)
        dataset = base.build_dataset(cfg, codec, module.ar.instruction_conditioner.tokenizer)
        perf = cfg['performance']
        sampler = ScaleoutSampler(dataset, seed=cfg['seed'], rank=rank, boundary=boundary)
        generator = torch.Generator()
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=perf['num_workers'],
            pin_memory=True, persistent_workers=perf['num_workers'] > 0,
            collate_fn=functools.partial(base.data.collate, pad_id=codec.pad_id, joint=True),
            generator=generator, multiprocessing_context='spawn' if perf['num_workers'] else None,
            **({'prefetch_factor': perf['prefetch_factor']} if perf['num_workers'] else {}))
        contract = make_contract(cfg, run_dir, topology, provenance, codec, boundary)
        if rank == 0:
            path = run_dir / 'RUN_CONTRACT.json'
            if path.exists() and base.read(path) != contract:
                raise RuntimeError('Existing six-GPU contract changed')
            if not path.exists():
                base.write(path, contract)
        dist.barrier()
        payload, identity = base.load_joint_checkpoint(args.resume,
            expected_contract=None if migrating else contract, require_latest=not migrating)
        if migrating:
            validate_parent_payload(payload, identity, cfg, boundary)
        actual = restore_training_state(module, optimizer, scheduler, payload, include_rng=not migrating)
        step, epoch, next_batch = (int(payload[k]) for k in ('global_step', 'epoch', 'next_batch'))
        pending_rng = None if migrating else payload['rng_states_by_rank'][rank]
        base.write(out / ('TRANSFER_STATE.json' if migrating else 'RESUME_STATE.json'), actual)
        base.write(out / 'RESUME_LOADED.json', {'identity': identity, 'step': step,
            'migration': migrating, 'optimizer_and_scheduler_transferred': True,
            'shared_transformer_same_object': module.ar.shared_transformer is module.diffusion.model.model.transformer,
            'rng_policy': RNG_POLICY if migrating else 'native_all_six_rank_states'})
        del payload, actual
        if migrating:
            base.write(out / 'DATA_CURSOR_MIGRATION.json', {
                'parent': {'epoch': epoch, 'next_batch': next_batch},
                'six_gpu': {'epoch': boundary['new_epoch'], 'next_batch': 0},
                'boundary_file': cfg['scaleout']['boundary_file'],
                'boundary_sha256': base.sha(cfg['scaleout']['boundary_file'])})
            epoch, next_batch = boundary['new_epoch'], 0
        state = sampler.resumable_state_dict(at_epoch_boundary=True)
        state['resume_epoch'] = epoch
        sampler.load_resumable_state_dict(state)
        if step >= args.stop_updates or next_batch > len(sampler):
            raise ValueError('Invalid resumed step or data cursor')
        if next_batch == len(sampler):
            epoch += 1
            next_batch = 0
        state = sampler.resumable_state_dict(at_epoch_boundary=True)
        state['resume_epoch'] = epoch
        sampler.load_resumable_state_dict(state)
        if next_batch:
            sampler.set_resume_batch_offset(next_batch)
        base.native._seed_everything(cfg['seed'] + (97_000_003 + step if migrating else 0), rank)
        wrapped = DistributedDataParallel(module, device_ids=[local], broadcast_buffers=False,
            find_unused_parameters=True, static_graph=False, gradient_as_bucket_view=True,
            bucket_cap_mb=perf['ddp_bucket_cap_mb'])
        frozen = {'CLAP': base.state_hash(module.ar.source_clap_model),
            'Qwen': base.state_hash(module.ar.instruction_conditioner.model)}
        base.write(out / 'READY.json', {'at': base.now(), 'step': step, 'frozen': frozen,
            'shared_transformer_same_object': True, 'initialization': provenance})
        base.train_loop(wrapped, teacher, optimizer, scheduler, loader, sampler, generator, cfg,
            contract, out, device, rank, world, step, epoch, next_batch, pending_rng,
            args.stop_updates, set(args.evidence_updates))
        finish_numerics()
        if frozen != {'CLAP': base.state_hash(module.ar.source_clap_model),
                'Qwen': base.state_hash(module.ar.instruction_conditioner.model)}:
            raise RuntimeError('Frozen encoder changed during continuation')
        if base.initialization.protected_identity(cfg['protected_DiT50k'], full_hash=rank == 0) != provenance['protected_DiT50k']:
            raise RuntimeError('Protected original checkpoint changed')
        base.write(out / 'COMPLETE.json', {'at': base.now(), 'new_updates': args.stop_updates,
            'joint_AR_and_RF_trained': True, 'shared_transformer_same_object': True,
            'protected_DiT50k_unchanged': True, 'frozen_encoders_unchanged': True,
            'flash_calls': flash_calls, 'quality_gate_passed': False})
    finally:
        restore_flash()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--stop-updates', type=int, required=True)
    parser.add_argument('--resume', type=Path, required=True)
    parser.add_argument('--evidence-updates', type=int, nargs='*', default=[])
    run(parser.parse_args())
