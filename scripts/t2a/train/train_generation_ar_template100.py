#!/usr/bin/env python3
"""One complete qualitative-caption epoch, with a three-rank technical gate.

No request parser, supplied source count, source labels, or teacher context is
passed to the model. Only the English request and supervised codec tokens are
used. Immutable numerical GT completions are training witnesses, not metrics.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from generation_ar_sampling import ShuffledGlobalBatchSampler
import numpy as np
import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import (
    GenerationARSQLiteDataset, LengthBucketDistributedSampler, collate_generation_ar)
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.models.sceneplan_transfusion_generation_ar import load_p10v11_generation_ar
from stable_audio_tools.models.sceneplan_generation_ar_lora import AdaptedGenerationAR
from stable_audio_tools.inference.sceneplan_generation_ar_precision import configure_float32_ar


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    temp.replace(path)


def tensor_digest(named):
    h = hashlib.sha256()
    for name, value in named:
        h.update(name.encode())
        h.update(value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def rng_state():
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(),
            'numpy': np.random.get_state(), 'python': random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])
    np.random.set_state(state['numpy']); random.setstate(state['python'])


def save_checkpoint(path, model, optimizer, scheduler, step, contract):
    # Every rank contributes its own state; only rank zero writes the artifact.
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, rng_state())
    if dist.get_rank() == 0:
        value = {'global_step': step, 'epoch': 0, 'batch_in_epoch': step, 'run_contract': contract,
                 'ar_adapter': model.trainable_state_dict(), 'ar_lora': model.lora_state_dict(),
                 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                 'rng_states_by_rank': states}
        temp = path.with_name(path.name + '.tmp'); torch.save(value, temp); temp.replace(path)
        atomic(path.parent.parent / 'LATEST_CHECKPOINT.json', {'path': str(path), 'step': step, 'sha256': sha(path)})
    dist.barrier()


def groups(loader, accumulation):
    pending = []
    for batch in loader:
        pending.append(batch)
        if len(pending) == accumulation:
            yield pending; pending = []
    if pending: yield pending


def cosine(step, steps, warmup):
    if warmup and step < warmup: return (step + 1) / warmup
    progress = min(1., max(0., (step - warmup) / max(1, steps - warmup)))
    return .1 + .9 * .5 * (1. + math.cos(math.pi * progress))


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0,1,2'
    assert os.environ.get('WORLD_SIZE') == '3'
    config = json.loads(args.config.read_text()); config_sha = sha(args.config)
    assert config['schema'] == 'generation_ar_template100_training_v1'
    assert config['steps'] == 8334 and config['train_rows'] == 1600000
    assert config['binding_strength'] == 0 and config['global_batch_size'] == 192
    assert config['max_wall_seconds'] <= 14400
    rank = int(os.environ['RANK']); local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local_rank); torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device)
    torch.manual_seed(42 + rank); np.random.seed(42 + rank); random.seed(42 + rank)
    run = args.run_dir.resolve()
    if rank == 0:
        run.mkdir(parents=True, exist_ok=True); (run / 'checkpoints').mkdir(exist_ok=True)
        atomic(run / 'STATUS.json', {'status': 'LOADING', 'pid': os.getpid(), 'mode': 'gate' if args.gate else 'full_train'})
    dist.barrier()
    data = Path(config['data']); init = Path(config['initialize_checkpoint'])
    quality = json.loads((data / 'QUALITY_REPORT.json').read_text())
    assert quality['status'] == 'PASS' and not any(quality['scene_family_overlaps'].values())
    assert sha(data / 'QUALITY_REPORT.json') == config['quality_report_sha256']
    source_hashes = {p: sha(REPO / p) for p in config['source_sha256']}
    assert source_hashes == config['source_sha256']
    if rank == 0:
        assert sha(init) == config['initialize_checkpoint_sha256']
        for split in ('train', 'validation'):
            assert sha(data / (split + '.sqlite')) == quality['splits'][split]['sha256']
        if not args.gate:
            proof = json.loads(args.gate_proof.read_text())
            assert proof['status'] == 'PASS' and proof['training_config_sha256'] == config_sha
            assert proof['source_sha256'] == source_hashes
    dist.barrier()
    codec = ModelScenePlanCodecV4(config['codec'])
    base, p10 = load_p10v11_generation_ar(pad_id=codec.pad_id, verify_sha256=rank == 0, activation_checkpointing=False)
    state = torch.load(init, map_location='cpu', weights_only=False)
    assert p10.as_dict() == state['run_contract']['p10_load']
    assert codec.fingerprint == state['run_contract']['codec_fingerprint']
    base.load_trainable_state_dict(state['ar_adapter'])
    model = AdaptedGenerationAR(base, codec, rank=8, alpha=8., binding_strength=0.)
    model.load_lora_state_dict(state['ar_lora']); del base, state
    model.p10_dit.to(device=device, dtype=torch.float32)
    model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32); model.ar_lora.to(device=device, dtype=torch.float32)
    configure_float32_ar(model); model.train()
    parameter_contract = model.adaptation_contract()
    columns = np.load(data / 'train_columns.npz')
    lengths, counts, raw_lengths = columns['lengths'], columns['source_counts'], columns['request_lengths']
    assert len(lengths) == 1600000
    if args.gate:
        ordinals = []
        for count in range(1, 5):
            candidates = np.flatnonzero(counts == count)
            selected = set(sorted(candidates, key=lambda i: (-int(raw_lengths[i]), int(i)))[:16])
            selected.update(sorted(candidates, key=lambda i: (-int(lengths[i]), int(i)))[:16])
            for i in sorted(candidates, key=lambda i: (-int(lengths[i] + raw_lengths[i]), int(i))):
                if len(selected) == 48: break
                selected.add(i)
            assert len(selected) == 48
            ordinals.extend(sorted(selected))
        ordinals = np.asarray(ordinals)
        dataset = GenerationARSQLiteDataset(data / 'train.sqlite', split='train', row_ordinals=ordinals)
        used_lengths = lengths[ordinals]; steps = 3; warmup = 0
    else:
        ordinals = None; dataset = GenerationARSQLiteDataset(data / 'train.sqlite', split='train')
        used_lengths = lengths; steps = config['steps']; warmup = config['warmup_steps']
    base_sampler = LengthBucketDistributedSampler(used_lengths, num_replicas=3, rank=rank, batch_size=64)
    sampler = ShuffledGlobalBatchSampler(base_sampler); sampler.set_epoch(config['sampler_epoch'])
    micro_size = config['micro_batch_size']; accumulation = 64 // micro_size
    # A dedicated generator keeps worker seeding independent of model RNG on resume.
    worker_rng = torch.Generator().manual_seed(10042 + rank)
    loader = DataLoader(dataset, batch_size=micro_size, sampler=sampler,
        num_workers=2 if args.gate else 6, generator=worker_rng,
        collate_fn=functools.partial(collate_generation_ar, pad_id=codec.pad_id),
        pin_memory=True, drop_last=False, persistent_workers=True)
    optimizer = torch.optim.AdamW([
        {'params': model.ar_adapter.parameters(), 'lr': config['adapter_lr']},
        {'params': model.ar_lora.parameters(), 'lr': config['lora_lr']}], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: cosine(s, steps, warmup))
    contract = {**config, 'mode': 'distributed_correctness_gate' if args.gate else 'one_full_epoch',
        'training_config_sha256': config_sha, 'source_snapshot': str(REPO), 'source_sha256': source_hashes,
        'steps': steps, 'warmup_steps': warmup, 'parameter_contract': parameter_contract,
        'p10_load': p10.as_dict(), 'codec_fingerprint': codec.fingerprint,
        'gate_proof_sha256': sha(args.gate_proof) if args.gate_proof else None,
        'gate_ordinals': ordinals.tolist() if ordinals is not None else [],
        'manifest_sha256': {s: quality['splits'][s]['sha256'] for s in ('train', 'validation')},
        'planned_presentations': 576 if args.gate else 1600000,
        'epoch_sampling': 'All rows once; shuffled global length buckets; uneven 64-row final global batch without duplicates',
        'request_input_policy': 'raw English only; binding bias disabled; no source-count forcing or supplied labels',
        'test_used': False, 'official_selection_modified': False}
    if rank == 0:
        path = run / 'RUN_CONTRACT.json'
        if path.exists(): assert json.loads(path.read_text()) == contract
        else: atomic(path, contract)
    dist.barrier()
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False)
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=False)
        assert state['run_contract'] == contract
        model.load_trainable_state_dict(state['ar_adapter']); model.load_lora_state_dict(state['ar_lora'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        start_step = int(state['global_step']); restore_rng(state['rng_states_by_rank'][rank]); del state
        assert 0 <= start_step < steps
    frozen_modules = [('p10', model.p10_dit), ('conditioner', model.prompt_conditioner)]
    frozen_before = {n: tensor_digest(m.named_parameters()) for n, m in frozen_modules} if args.gate else None
    adapter_before = tensor_digest(model.ar_adapter.named_parameters()) if args.gate else None
    lora_before = tensor_digest(model.ar_lora.named_parameters()) if args.gate else None
    trainable = [p for p in model.parameters() if p.requires_grad]
    started = time.monotonic(); global_step = start_step; local_tokens = local_rows = 0; local_loss = 0.; probe = None
    optimizer.zero_grad(set_to_none=True)
    for repeat in range(steps if args.gate else 1):
        for batch_index, raw_group in enumerate(groups(loader, accumulation)):
            if not args.gate and batch_index < start_step: continue
            denominator = torch.tensor(sum(int((r['plan_labels'] != -100).sum()) for r in raw_group), device=device, dtype=torch.float64)
            dist.all_reduce(denominator)
            for micro, raw in enumerate(raw_group):
                batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in raw.items()}
                with torch.no_grad(): context, mask = model.encode_requests(batch['raw_user_requests'], device=device)
                sync = wrapped.no_sync() if micro < len(raw_group) - 1 else nullcontext()
                with sync:
                    logits = wrapped(batch['plan_input_ids'], batch['plan_attention_mask'], context, mask)
                    loss = F.cross_entropy(logits.reshape(-1, 4096), batch['plan_labels'].reshape(-1), ignore_index=-100, reduction='sum')
                    (loss * (3. / denominator.item())).backward()
                local_loss += float(loss.detach()); local_tokens += int((batch['plan_labels'] != -100).sum())
                local_rows += len(batch['raw_user_requests'])
                if args.gate: probe = (batch['plan_input_ids'], batch['plan_attention_mask'], context, mask)
                del logits, loss
            norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient')
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1
            if global_step % 25 == 0 or global_step == steps or args.gate:
                stats = torch.tensor([local_loss, local_tokens, local_rows, torch.cuda.max_memory_allocated(device)], device=device, dtype=torch.float64)
                dist.all_reduce(stats[:3]); dist.all_reduce(stats[3:], op=dist.ReduceOp.MAX)
                if rank == 0:
                    elapsed = time.monotonic() - started
                    status = {'status': 'RUNNING', 'step': global_step, 'steps': steps, 'elapsed_s': elapsed,
                        'updated_unix': time.time(), 'start_step': start_step, 'global_rows_since_start': int(stats[2]),
                        'global_tokens_per_s': float(stats[1]) / max(elapsed, 1.), 'mean_loss_since_start': float(stats[0] / stats[1]),
                        'learning_rates': scheduler.get_last_lr(), 'peak_gpu_bytes_max_rank': int(stats[3]),
                        'eta_s': (steps - global_step) * elapsed / max(1, global_step - start_step)}
                    atomic(run / 'STATUS.json', status)
                    with (run / 'metrics.jsonl').open('a') as f: f.write(json.dumps(status) + '\n')
                    print(json.dumps(status), flush=True)
            if global_step % config['checkpoint_every'] == 0 or global_step == steps or (args.gate and global_step == 1):
                save_checkpoint(run / 'checkpoints' / f'step_{global_step:08d}.pt', model, optimizer, scheduler, global_step, contract)
            elapsed = torch.tensor(time.monotonic() - started, device=device); dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            if elapsed.item() > (600 if args.gate else config['max_wall_seconds']) and global_step < steps:
                save_checkpoint(run / 'checkpoints' / f'budget_stop_{global_step:08d}.pt', model, optimizer, scheduler, global_step, contract)
                if rank == 0: atomic(run / 'STATUS.json', {'status': 'BUDGET_STOP', 'step': global_step, 'elapsed_s': elapsed.item()})
                dist.destroy_process_group(); return 3
            if global_step == steps: break
        if global_step == steps: break
    assert global_step == steps
    if args.gate:
        for name, module in frozen_modules: assert tensor_digest(module.named_parameters()) == frozen_before[name]
        assert tensor_digest(model.ar_adapter.named_parameters()) != adapter_before
        assert tensor_digest(model.ar_lora.named_parameters()) != lora_before
        model.eval()
        with torch.no_grad(): before = model(*probe)
        saved = torch.load(run / 'checkpoints' / f'step_{steps:08d}.pt', map_location='cpu', weights_only=False)
        model.load_trainable_state_dict(saved['ar_adapter']); model.load_lora_state_dict(saved['ar_lora'])
        with torch.no_grad(): after = model(*probe)
        assert torch.equal(before, after)
        # Exercise rank-specific RNG restoration rather than merely saving it.
        restore_rng(saved['rng_states_by_rank'][rank])
        expected = (torch.rand(4), torch.rand(4, device=device), np.random.rand(4), random.random())
        restore_rng(saved['rng_states_by_rank'][rank])
        actual = (torch.rand(4), torch.rand(4, device=device), np.random.rand(4), random.random())
        assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])
        assert np.array_equal(expected[2], actual[2]) and expected[3] == actual[3]
        if rank == 0: atomic(run / 'GATE.json', {'status': 'PASS', 'training_config_sha256': config_sha,
            'source_sha256': source_hashes, 'frozen_p10_qwen_conditioner_unchanged': True,
            'ar_adapter_and_lora_updated': True, 'checkpoint_reload_exact': True, 'per_rank_rng_restore_exact': True,
            'micro_batch_size': micro_size, 'steps': steps, 'model_request_inputs': 'English raw request only'})
    if rank == 0:
        atomic(run / 'STATUS.json', {'status': 'COMPLETE', 'step': steps, 'elapsed_s': time.monotonic() - started,
            'global_rows_since_start': (576 if args.gate else 1600000) if start_step == 0 else None,
            'goal_complete': False, 'validation_acceptance': 'PENDING'})
    dataset.close(); dist.barrier(); dist.destroy_process_group(); return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--gate', action='store_true'); parser.add_argument('--gate-proof', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    try: raise SystemExit(main(args))
    except Exception as exc:
        if int(os.environ.get('RANK', 0)) == 0:
            args.run_dir.mkdir(parents=True, exist_ok=True)
            atomic(args.run_dir / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'})
        raise
