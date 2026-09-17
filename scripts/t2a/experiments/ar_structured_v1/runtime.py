"""Joint new50k runtime: one shared Transformer and immutable parent files."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from datetime import datetime
import functools
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from scripts.t2a.experiments.ar_instruction_t200_v1 import runtime as original_runtime
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint
from scripts.t2a.experiments.ar_instruction_t200_v1.instruction_data import InstructionOverlay, DatasetWithInstructions
from scripts.t2a.experiments.ar_source_grounding_v1 import allocated_runtime
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, state_hash, write
from scripts.t2a.experiments.ar_source_grounding_v1.numerics import install_autotune_observer
from scripts.t2a.train import train_sceneplan_transfusion_editing_ar_clap44 as native
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import (
    _gather_rank_rng_states, _restore_rank_rng_state, _move_joint_batch,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import (
    JOINT44_RUN_SCHEMA, ar_specific_state, ensure_run_identity, load_ar_specific,
    load_joint_checkpoint, save_joint_checkpoint,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_text import FrozenCLAP44TextFeatures
from scripts.t2a.experiments.ar_structured_v1 import data, initialization, losses, model as structured_model


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now().astimezone().isoformat()


def source_inventory():
    paths = [p for p in Path(__file__).parent.glob('*.py') if p.name not in ('qa.py', 'joint_profile.py', 'prepare.py')]
    paths += [Path(data.__file__), Path(initialization.__file__), Path(losses.__file__),
        Path(structured_model.__file__), Path(original_runtime.__file__)]
    paths += [ROOT / 'scripts/t2a/experiments/clap_scene_supervision_v1/events.py',
        ROOT / 'scripts/t2a/experiments/clap_scene_supervision_v1/data.py',
        ROOT / 'scripts/t2a/experiments/ar_factual_clap_v1/integration.py',
        ROOT / 'scripts/t2a/experiments/ar_instruction_t200_v3/policy.py',
        ROOT / 'scripts/t2a/experiments/ar_instruction_t200_v1/instruction_data.py',
        ROOT / 'stable_audio_tools/models/sceneplan_editing_gain_adapter.py',
        ROOT / 'scripts/t2a/experiments/ar_source_grounding_v1/numerics.py']
    return {**native.run_source_inventory(), **{str(p.relative_to(ROOT)): sha(p) for p in set(paths)}}


def validate_config(cfg):
    direction = read(cfg['user_direction'])
    if (cfg['schema'] != 'structured_joint_AR_EditingDiT_new50k_v1'
            or cfg['new_updates'] != 50000 or cfg['physical_gpus'] != [5, 6, 7]
            or direction.get('joint_training_explicitly_selected_by_user') is not True
            or direction.get('original_checkpoint_must_remain_unchanged') is not True):
        raise ValueError('Joint50k requires the current explicit user direction')
    if cfg['quality_gate_passed'] is not False or cfg['independent_test_used'] is not False:
        raise ValueError('Training authorization is not quality acceptance')
    if cfg['save_every'] != 10000 or cfg['recovery_every'] != 2500:
        raise ValueError('Five10k milestones and2500-update recovery points are required')
    if cfg['loss_weights']['AR'] <= 0 or cfg['loss_weights']['RF'] <= 0:
        raise ValueError('Both AR and Editing DiT losses must train the shared Transformer')
    if not cfg['optimizer']['fresh_joint_optimizer']:
        raise ValueError('Multiple weight parents require the declared new joint optimizer')


def build_dataset(cfg, codec, tokenizer, split='train'):
    record = cfg['data'][split]
    base = native.ScenePlanTransfusionEditingDataset(record['native_index_path'],
        tokenizer_spec=(tokenizer, 512), expected_num_samples=record['rows'],
        expected_index_sha256=record['native_index_sha256'], latent_crop_length=648,
        require_frozen=True, verify_tensor_hashes_on_access=True)
    joint = native.ScenePlanTransfusionEditingJointDataset(base, codec=codec)
    overlay = InstructionOverlay(record['path'], expected_sha256=record['sha256'],
        native_index_path=record['native_index_path'], native_index_sha256=record['native_index_sha256'],
        expected_rows=record['rows'], split=split)
    return data.StructuredDataset(DatasetWithInstructions(joint, overlay, joint=True), record['native_index_path'], joint=True)


def make_teacher(module, cfg):
    root = cfg['base_AR_configuration']['clap_dependency']['checkpoint']['path']
    contract = read(Path(root).parent / 'TRAIN_CONTRACT.json')
    text = contract['native_config']['text']
    teacher = FrozenCLAP44TextFeatures(text['model_path'], hidden_dim=1024,
        max_tokens=text['max_tokens'], batch_size=text['batch_size']).eval()
    existing = module.ar.instruction_conditioner.model
    if state_hash(teacher.conditioner.model) != state_hash(existing):
        raise RuntimeError('Content teacher and frozen instruction Qwen weights differ')
    # Share only the immutable pretrained Qwen backbone. The unprojected
    # content-teacher path remains separate from trainable task projections.
    teacher.conditioner.__dict__['model'] = existing
    return teacher


def lr_factor(step, cfg):
    if step < cfg['warmup_updates']:
        return .1 + .9 * step / cfg['warmup_updates']
    progress = min(1., (step - cfg['warmup_updates']) / (cfg['new_updates'] - cfg['warmup_updates']))
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


def make_optimizer(groups, cfg):
    options = cfg['optimizer']
    optimizer = torch.optim.AdamW(groups, betas=tuple(options['betas']),
        weight_decay=options['weight_decay'], fused=True)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, cfg))
    return optimizer, scheduler


def make_contract(cfg, run, topology, provenance, codec):
    identity = [ensure_run_identity(run) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(identity, src=0)
    identity = identity[0]
    return {'schema': JOINT44_RUN_SCHEMA, 'training_mode': 'joint', 'rf_mode': 'editing_reference',
        'run_dir': str(run), 'run_id': identity['run_id'], 'repo_root': str(ROOT),
        'ar_contract': native.EDITING_AR_CLAP44_CONTRACT if hasattr(native, 'EDITING_AR_CLAP44_CONTRACT')
            else 'p10v11_shared_blocks_audio_reference_editing_ar_clap44_v1',
        'variant': 'global_and_sequence', 'm2d_used': False, 'independent_test_used': False,
        'structured_architecture': structured_model.CONTRACT, 'recipe': cfg,
        'world_size': 3, 'physical_gpus': [5, 6, 7], 'gpu_topology': topology,
        'schedule': {'max_steps': cfg['new_updates'], 'save_every': cfg['save_every'],
            'recovery_every': cfg['recovery_every'], **cfg['performance']},
        'runtime_inputs': ['source_foa_latent', 'raw_edit_request'],
        'old_plan_input': False, 'source_caption_input': False, 'target_audio_ar_input': False,
        'RF_training_inputs': ['source_foa_latent', 'ground_truth_target_plan', 'noised_target_latent', 'timestep'],
        'structured_supervision_only': ['source_plan_fields', 'source_identity_correspondence',
            'edit_object_and_operation', 'target_fields', 'frozen_content_teacher'],
        'training_objective': 'AR_CE_PLUS_EDITING_RF_PLUS_STRUCTURED_SOURCE_BINDING_PRESERVATION',
        'loss_normalization': 'global_valid_plan_tokens_RF_values_and_examples_per_update',
        'initialization': provenance, 'codec_fingerprint': codec.fingerprint,
        'source_sha256': source_inventory(), 'quality_gate_passed': False}


def batch_loss(wrapped, teacher, batch, cfg, device, step, rank, world, denominators):
    ar, target, metadata, rf_mask = _move_joint_batch(batch, device)
    supervision = data.to_device(batch['supervision'], device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        source_content, target_content = losses.content_targets(supervision, teacher, device)
        seed = cfg['seed'] + 31_000_001 + step * world + rank
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(target.shape, device=device, dtype=target.dtype, generator=generator)
        times = torch.rand(len(target), device=device, generator=generator)
        noised = (1 - times[:, None, None]) * target + times[:, None, None] * noise
        logits, prediction, outputs = wrapped(source_foa_latent=ar['source_foa_latent'],
            source_attention_mask=ar['source_attention_mask'], plan_input_ids=ar['plan_input_ids'],
            plan_attention_mask=ar['plan_attention_mask'], raw_edit_requests=ar['raw_edit_requests'],
            metadata=metadata, noised_target=noised, timesteps=times, rf_padding_mask=rf_mask)
        ar_sum = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1),
            ar['plan_labels'].flatten(), ignore_index=-100, reduction='sum')
        rf_sum = (((prediction.float() - (noise - target)).square()) * rf_mask[:, None]).sum()
        auxiliary = losses.structured_objective(outputs, supervision, source_content, target_content,
            weights={k: cfg['loss_weights'][k] for k in losses.DEFAULT_LOSS_WEIGHTS})
        loss = world * (cfg['loss_weights']['AR'] * ar_sum / denominators[0]
            + cfg['loss_weights']['RF'] * rf_sum / denominators[1]
            + auxiliary['loss_sum'] / denominators[2])
    sums = torch.stack((ar_sum.detach(), rf_sum.detach(), auxiliary['loss_sum'].detach(),
        *auxiliary['metrics_sums'].values())).double()
    return loss, sums, tuple(auxiliary['metrics_sums']), outputs


def snapshot(module, optimizer, scheduler, *, step, epoch, next_batch, rank, world, device):
    states = _gather_rank_rng_states(rank=rank, world_size=world, device=device)
    state = {'diffusion_state_dict': module.diffusion.state_dict(),
        'editing_ar_specific_state_dict': ar_specific_state(module.ar),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'rng_states_by_rank': states, 'global_step': step, 'epoch': epoch, 'next_batch': next_batch}
    result = fingerprint(state)
    hashes = [None] * world
    dist.all_gather_object(hashes, result['sha256'])
    if len(set(hashes)) != 1:
        raise RuntimeError('Joint model/Adam/scheduler/all-rank RNG state differs between ranks')
    return states, result


def run(args):
    cfg = read(args.config); validate_config(cfg)
    if cfg.get('training_source_sha256') != source_inventory():
        raise RuntimeError('Joint training requires the prepared, unchanged source inventory')
    if os.environ.get('EDITING_GPUS') != '5,6,7' or torch.cuda.is_initialized():
        raise RuntimeError('Allocate GPU5–7 before initializing CUDA')
    run_dir = initialization.assert_separate_output(args.run_dir, cfg['protected_DiT50k'])
    run_dir.mkdir(parents=True, exist_ok=True)
    if not 1 <= args.stop_updates <= cfg['new_updates']:
        raise ValueError('Stop must be a successful new-update count within the50k budget')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    rank, local_rank, world, device, topology = allocated_runtime.distributed(timeout_seconds=1800)
    if world != 3:
        raise ValueError('This recipe requires all three allocated ranks')
    out = run_dir / 'attempts' / args.attempt / f'rank{rank}'
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'STARTED.json', {'at': now(), 'pid': os.getpid(), 'physical_gpu': [5, 6, 7][rank]})
    numerical = cfg['numerics'][str(rank)]
    numerical_path = Path(numerical['directory']) / 'AUTOTUNE_CATALOG.json'
    if sha(numerical_path) != numerical['sha256']:
        raise RuntimeError('Pinned numerical catalog changed')
    for record in read(numerical_path)['records'].values():
        if sha(record['source_file']) != record['source_sha256']:
            raise RuntimeError('Pinned native FLA implementation changed')
    finish_numerics = install_autotune_observer({'name': 'autotune', 'mode': 'pin',
        'reference_case': numerical['directory']}, out)
    flash_calls, restore_flash = original_runtime.install_flash(cfg['base_AR_configuration']['numerical_execution'])
    try:
        native._seed_everything(cfg['seed'], 0)
        module, codec, groups, provenance = initialization.build(cfg)
        module.diffusion.model.model.activation_checkpointing = cfg['performance']['RF_activation_checkpointing']
        teacher = make_teacher(module, cfg)
        module.to(device).train()
        optimizer, scheduler = make_optimizer(groups, cfg)
        dataset = build_dataset(cfg, codec, module.ar.instruction_conditioner.tokenizer)
        perf = cfg['performance']
        sampler = native.DistributedScenePlanBucketBatchSampler(dataset,
            short_batch_size=perf['short_batch_size'], long_batch_size=perf['long_batch_size'],
            num_replicas=world, rank=rank, shuffle=True, seed=cfg['seed'], drop_last=True)
        if args.probe_interface:
            from scripts.t2a.experiments.ar_structured_v1.probe import verify
            probe_sampler = native.DistributedScenePlanBucketBatchSampler(dataset,
                short_batch_size=perf['short_batch_size'], long_batch_size=perf['long_batch_size'],
                num_replicas=world, rank=rank, shuffle=True, seed=cfg['seed'], drop_last=True)
            indices = next(iter(probe_sampler))
            verify(module, data.collate([dataset[i] for i in indices], pad_id=codec.pad_id, joint=True), cfg, device, out)
            del probe_sampler, indices
        generator = torch.Generator()
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=perf['num_workers'],
            pin_memory=True, persistent_workers=perf['num_workers'] > 0,
            collate_fn=functools.partial(data.collate, pad_id=codec.pad_id, joint=True), generator=generator,
            multiprocessing_context='spawn' if perf['num_workers'] else None,
            **({'prefetch_factor': perf['prefetch_factor']} if perf['num_workers'] else {}))
        contract = make_contract(cfg, run_dir, topology, provenance, codec)
        if rank == 0:
            p = run_dir / 'RUN_CONTRACT.json'
            if p.exists() and read(p) != contract:
                raise RuntimeError('Existing joint run has a different contract')
            if not p.exists():
                write(p, contract)
        dist.barrier()
        step = epoch = next_batch = 0
        pending_rng = None
        if args.resume:
            payload, identity = load_joint_checkpoint(args.resume, expected_contract=contract, require_latest=True)
            module.diffusion.load_state_dict(payload['diffusion_state_dict'], strict=True)
            load_ar_specific(module.ar, payload['editing_ar_specific_state_dict'])
            optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
            step, epoch, next_batch = (int(payload[k]) for k in ('global_step', 'epoch', 'next_batch'))
            pending_rng = payload['rng_states_by_rank'][rank]
            restored = {'diffusion_state_dict': module.diffusion.state_dict(),
                'editing_ar_specific_state_dict': ar_specific_state(module.ar),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'rng_states_by_rank': payload['rng_states_by_rank'],
                'global_step': step, 'epoch': epoch, 'next_batch': next_batch}
            loaded_fingerprint = fingerprint(restored)
            expected_fingerprint = fingerprint({k: payload[k] for k in restored})
            if loaded_fingerprint != expected_fingerprint:
                raise RuntimeError('Native joint state did not restore exactly')
            write(out / 'RESUME_STATE.json', loaded_fingerprint)
            write(out / 'RESUME_LOADED.json', {'identity': identity, 'step': step,
                'shared_transformer_same_object': module.ar.shared_transformer is module.diffusion.model.model.transformer})
            del payload, restored, expected_fingerprint, loaded_fingerprint
        elif list((run_dir / 'checkpoints').glob('step-*.pt')):
            raise RuntimeError('Existing joint checkpoint requires an explicit native resume')
        if step >= args.stop_updates:
            raise ValueError('Resume has already reached the requested stop')
        if next_batch == len(sampler):
            epoch += 1; next_batch = 0
        state = sampler.resumable_state_dict(at_epoch_boundary=True)
        state['resume_epoch'] = epoch; sampler.load_resumable_state_dict(state)
        if next_batch:
            sampler.set_resume_batch_offset(next_batch)
        native._seed_everything(cfg['seed'], rank)
        wrapped = DistributedDataParallel(module, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=True, static_graph=False, gradient_as_bucket_view=True,
            bucket_cap_mb=perf['ddp_bucket_cap_mb'])
        frozen_before = {'CLAP': state_hash(module.ar.source_clap_model),
            'Qwen': state_hash(module.ar.instruction_conditioner.model)}
        write(out / 'READY.json', {'at': now(), 'step': step, 'frozen': frozen_before,
            'shared_transformer_same_object': True, 'initialization': provenance})
        train_loop(wrapped, teacher, optimizer, scheduler, loader, sampler, generator, cfg,
            contract, out, device, rank, world, step, epoch, next_batch, pending_rng, args.stop_updates,
            set(args.evidence_updates))
        finish_numerics()
        frozen_after = {'CLAP': state_hash(module.ar.source_clap_model),
            'Qwen': state_hash(module.ar.instruction_conditioner.model)}
        if frozen_after != frozen_before:
            raise RuntimeError('Frozen encoder changed during joint training')
        if initialization.protected_identity(cfg['protected_DiT50k'], full_hash=rank == 0) != provenance['protected_DiT50k']:
            raise RuntimeError('Protected original checkpoint changed')
        write(out / 'COMPLETE.json', {'at': now(), 'new_updates': args.stop_updates,
            'joint_AR_and_RF_trained': True, 'shared_transformer_same_object': True,
            'protected_DiT50k_unchanged': True, 'frozen_encoders_unchanged': True,
            'flash_calls': flash_calls, 'quality_gate_passed': False})
    finally:
        restore_flash()
        if dist.is_initialized():
            dist.destroy_process_group()


def train_loop(wrapped, teacher, optimizer, scheduler, loader, sampler, generator, cfg,
        contract, out, device, rank, world, step, epoch, next_batch, pending_rng, stop_updates, evidence_updates):
    module = wrapped.module
    initial_step = step
    started = last_log = time.perf_counter()
    interval_pairs = 0
    interval_steps = 0
    interval_sums = None
    input_wait = 0.
    while step < stop_updates:
        generator.manual_seed(cfg['seed'] + 900001 + epoch * world + rank)
        iterator = iter(loader)
        if pending_rng is not None:
            _restore_rank_rng_state(pending_rng, rank=rank, device=device); pending_rng = None
        while step < stop_updates:
            waited = time.perf_counter()
            window = list(itertools.islice(iterator, cfg['performance']['gradient_accumulation']))
            input_wait += time.perf_counter() - waited
            if not window:
                break
            denominators = torch.tensor([
                sum(int((b['ar']['plan_labels'] != -100).sum()) for b in window),
                sum(sum(int(m['padding_mask'][0].sum()) * 64 for m in b['metadata']) for b in window),
                sum(len(b['ar']['pair_ids']) for b in window)], device=device, dtype=torch.float64)
            dist.all_reduce(denominators)
            optimizer.zero_grad(set_to_none=True)
            sums = None
            for micro, batch in enumerate(window):
                context = wrapped.no_sync() if micro + 1 < len(window) else nullcontext()
                with context:
                    loss, local_sums, names, _ = batch_loss(wrapped, teacher, batch, cfg, device,
                        step * cfg['performance']['gradient_accumulation'] + micro, rank, world, denominators)
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('Nonfinite joint training loss')
                    loss.backward()
                sums = local_sums if sums is None else sums + local_sums
            missing = [n for n, p in module.named_parameters() if p.requires_grad and p.grad is None]
            leaked = [n for n, p in module.named_parameters() if not p.requires_grad and p.grad is not None]
            if missing or leaked:
                raise RuntimeError(f'Incorrect gradient scope: missing={missing}, frozen={leaked}')
            norms = [float(torch.nn.utils.clip_grad_norm_(g['params'], cfg['optimizer']['gradient_clip'],
                error_if_nonfinite=True)) for g in optimizer.param_groups]
            optimizer.step(); scheduler.step(); step += 1; next_batch += len(window)
            record = {'new_update': step, 'epoch': epoch, 'next_batch': next_batch,
                'pair_ids': [p for b in window for p in b['ar']['pair_ids']],
                'request_sha256': [hashlib.sha256(s.encode()).hexdigest() for b in window for s in b['ar']['raw_edit_requests']],
                'learning_rates': scheduler.get_last_lr()}
            with (out / 'windows.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            interval_pairs += int(denominators[2])
            interval_steps += 1
            normalized = torch.cat((sums[:3] / denominators, sums[3:] / denominators[2]))
            interval_sums = normalized if interval_sums is None else interval_sums + normalized
            if step == initial_step + 1 or step % 10 == 0 or step == stop_updates:
                dist.all_reduce(interval_sums)
                elapsed = time.perf_counter() - last_log
                metrics = {'at': now(), 'new_update': step, 'epoch': epoch,
                    'global_pairs_per_second': interval_pairs / elapsed,
                    'interval_examples': interval_pairs, 'interval_seconds': elapsed,
                    'interval_updates': interval_steps,
                    'data_wait_seconds': input_wait, 'gradient_norms': norms,
                    'loss_sums_over_interval': dict(zip(('AR_CE', 'RF_MSE', 'structured', *names), interval_sums.tolist())),
                    'loss_means': dict(zip(('AR_CE', 'RF_MSE', 'structured', *names), (interval_sums / interval_steps).tolist())),
                    'memory_peak_allocated_GiB': torch.cuda.max_memory_allocated(device) / 2**30,
                    'elapsed_seconds': time.perf_counter() - started}
                if rank == 0:
                    print('STRUCTURED_JOINT_TRAIN=' + json.dumps(metrics), flush=True)
                    with (Path(contract['run_dir']) / 'metrics.jsonl').open('a') as stream:
                        stream.write(json.dumps(metrics) + '\n')
                interval_pairs = 0; interval_steps = 0; interval_sums = None; input_wait = 0.; last_log = time.perf_counter()
            checkpoint_due = step % cfg['recovery_every'] == 0 or step == stop_updates
            if checkpoint_due or step in evidence_updates:
                states, result = snapshot(module, optimizer, scheduler, step=step, epoch=epoch,
                    next_batch=next_batch, rank=rank, world=world, device=device)
                write(out / f'STATE_step{step:06d}.json', result)
                if rank == 0 and checkpoint_due:
                    save_joint_checkpoint(Path(contract['run_dir']) / f'checkpoints/step-{step:08d}.pt',
                        module=module, optimizer=optimizer, scheduler=scheduler, step=step,
                        epoch=epoch, next_batch=next_batch, contract=contract, rng_states=states)
                dist.barrier()
        if step < stop_updates:
            epoch += 1; next_batch = 0
            sampler_state = sampler.resumable_state_dict(at_epoch_boundary=True)
            if sampler_state['resume_epoch'] != epoch:
                raise RuntimeError('Loader and sampler epoch clocks diverged')
            sampler.load_resumable_state_dict(sampler_state)
    ddp = wrapped._get_ddp_logging_data()
    if ddp.get('has_rebuilt_buckets', 0) != 0:
        raise RuntimeError('DDP changed bucket layout; restart arithmetic needs revalidation')
    write(out / 'DDP_REVIEW.json', ddp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--stop-updates', type=int, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--evidence-updates', type=int, nargs='*', default=[])
    parser.add_argument('--probe-interface', action='store_true')
    run(parser.parse_args())
