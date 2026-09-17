"""Bounded joint AR/RF profile with real batches and resident Adam buffers."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from scripts.t2a.experiments.ar_structured_v1 import runtime as rt
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint


def gather(value, world):
    records = [None] * world
    dist.all_gather_object(records, value)
    return records


def run(args):
    cfg = rt.read(args.config); rt.validate_config(cfg)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    rank, local, world, device, topology = rt.allocated_runtime.distributed(timeout_seconds=900)
    out = args.output / f'rank{rank}'; out.mkdir(parents=True, exist_ok=False)
    rt.write(out / 'STARTED.json', {'at': rt.now(), 'physical_gpu': [5, 6, 7][rank]})
    finish = rt.install_autotune_observer({'name': f'rank{rank}', 'mode': 'capture'}, args.output)
    flash, restore = rt.original_runtime.install_flash(cfg['base_AR_configuration']['numerical_execution'])
    try:
        rt.native._seed_everything(cfg['seed'], 0)
        module, codec, groups, provenance = rt.initialization.build(cfg)
        teacher = rt.make_teacher(module, cfg)
        module.to(device).train()
        optimizer, scheduler = rt.make_optimizer(groups, cfg)
        # Reserve the same two moment tensors as real fused AdamW, without
        # taking any optimizer step or changing the inherited model weights.
        for group in optimizer.param_groups:
            for p in group['params']:
                optimizer.state[p] = {'step': torch.zeros((), device=device),
                    'exp_avg': torch.zeros_like(p), 'exp_avg_sq': torch.zeros_like(p)}
        before = fingerprint(dict(module.named_parameters()))['sha256']
        rt.write(out / 'INITIALIZED.json', {'at': rt.now(), 'provenance': provenance,
            'optimizer_updates': 0, 'resident_Adam_state': True})
        dataset = rt.build_dataset(cfg, codec, module.ar.instruction_conditioner.tokenizer)
        sampler = rt.native.DistributedScenePlanBucketBatchSampler(dataset, short_batch_size=64,
            long_batch_size=40, num_replicas=world, rank=rank, shuffle=True, seed=cfg['seed'], drop_last=True)
        samples = {}
        for position, indices in enumerate(sampler):
            bucket = 432 if len(indices) == 64 else 648
            if bucket not in samples:
                started = time.perf_counter(); rows = [dataset[i] for i in indices]
                samples[bucket] = rows
                rt.write(out / f'POPULATION_{bucket}.json', {'indices': indices,
                    'sampler_batch': position, 'read_seconds': time.perf_counter() - started,
                    'pair_ids': [row['model_row'][2]['pair_id'] for row in rows]})
            if len(samples) == 2:
                break
        rt.native._seed_everything(cfg['seed'], rank)
        wrapped = DistributedDataParallel(module, device_ids=[local], broadcast_buffers=False,
            find_unused_parameters=True, static_graph=False, gradient_as_bucket_view=True,
            bucket_cap_mb=cfg['performance']['ddp_bucket_cap_mb'])
        cases = [('checkpoint_both_32_20', True, True, 32, 20, 1),
                 ('checkpoint_AR_32_20', True, False, 32, 20, 1),
                 ('no_checkpoint_16_10', False, False, 16, 10, 2),
                 ('no_checkpoint_32_20', False, False, 32, 20, 1),
                 ('checkpoint_AR_64_40', True, False, 64, 40, 1),
                 ('no_checkpoint_64_40', False, False, 64, 40, 1)]
        results = []
        for name, ar_checkpoint, rf_checkpoint, short, long, accumulation in cases:
            if name in ('no_checkpoint_32_20', 'checkpoint_AR_64_40', 'no_checkpoint_64_40'):
                reference = {'no_checkpoint_32_20': 'no_checkpoint_16_10',
                    'checkpoint_AR_64_40': 'checkpoint_AR_32_20',
                    'no_checkpoint_64_40': 'no_checkpoint_32_20'}[name]
                previous = [x for x in results if x['case'] == reference]
                # Conservative bound: double the whole smaller-batch peak,
                # leaving at least five GiB available on these48GB cards.
                if not previous or max(x['peak_allocated_GiB'] for x in previous) * 2 - 4 > 42:
                    rt.write(out / f'SKIPPED_{name}.json', {'reason': 'measured_memory_headroom',
                        'previous_peaks_GiB': [x['peak_allocated_GiB'] for x in previous]})
                    continue
            module.ar.activation_checkpointing = ar_checkpoint
            module.diffusion.model.model.activation_checkpointing = rf_checkpoint
            for bucket in (432, 648):
                size = short if bucket == 432 else long
                rows = samples[bucket][:(size * accumulation)]
                batches = [rt.data.collate(rows[i:i+size], pad_id=codec.pad_id, joint=True)
                    for i in range(0, len(rows), size)]
                denominators = torch.tensor([
                    sum(int((b['ar']['plan_labels'] != -100).sum()) for b in batches),
                    sum(sum(int(m['padding_mask'][0].sum()) * 64 for m in b['metadata']) for b in batches),
                    len(rows)], device=device, dtype=torch.float64)
                dist.all_reduce(denominators)
                measurements = []
                for repeat in range(5):
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect(); torch.cuda.empty_cache()
                    dist.barrier(); torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter(); sums = None
                    for micro, batch in enumerate(batches):
                        with wrapped.no_sync() if micro + 1 < len(batches) else nullcontext():
                            loss, values, names, outputs = rt.batch_loss(wrapped, teacher, batch, cfg,
                                device, micro, rank, world, denominators)
                            if not bool(torch.isfinite(loss)):
                                raise RuntimeError('Nonfinite joint profile loss')
                            loss.backward()
                        sums = values if sums is None else sums + values
                        del loss, outputs
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                    peak = torch.cuda.max_memory_allocated(device) / 2**30
                    missing = [n for n, p in module.named_parameters() if p.requires_grad and p.grad is None]
                    frozen_grads = [n for n, p in module.named_parameters() if not p.requires_grad and p.grad is not None]
                    if missing or frozen_grads:
                        raise RuntimeError(f'Incorrect gradient scope {missing=} {frozen_grads=}')
                    norm = float(torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']],
                        float('inf'), error_if_nonfinite=True))
                    records = gather({'rank': rank, 'seconds': elapsed, 'peak_allocated_GiB': peak,
                        'gradient_norm': norm, 'loss_sums': sums.tolist()}, world)
                    record = {'case': name, 'bucket': bucket, 'repeat': repeat,
                        'measured': repeat >= 2, 'ranks': records,
                        'global_pairs_per_second': int(denominators[2]) / max(r['seconds'] for r in records)}
                    with (out / 'REPEATS.jsonl').open('a') as stream:
                        stream.write(json.dumps(record) + '\n')
                    if rank == 0:
                        print('JOINT_PROFILE=' + json.dumps(record), flush=True)
                    if repeat >= 2:
                        measurements.append(record)
                result = {'case': name, 'bucket': bucket, 'AR_activation_checkpointing': ar_checkpoint,
                    'RF_activation_checkpointing': rf_checkpoint, 'short_batch_size': short,
                    'long_batch_size': long, 'gradient_accumulation': accumulation,
                    'global_pairs_per_second': statistics.median(x['global_pairs_per_second'] for x in measurements),
                    'peak_allocated_GiB': max(r['peak_allocated_GiB'] for x in measurements for r in x['ranks'])}
                results.append(result); rt.write(out / f'RESULT_{name}_{bucket}.json', result)
                del batches
        optimizer.zero_grad(set_to_none=True)
        if fingerprint(dict(module.named_parameters()))['sha256'] != before:
            raise RuntimeError('A zero-update profile changed model parameters')
        if any(int(state['step']) for state in optimizer.state.values()):
            raise RuntimeError('Profile unexpectedly advanced Adam')
        finish()
        rt.write(out / 'RESULTS.json', results)
        rt.write(out / 'COMPLETE.json', {'at': rt.now(), 'results': results,
            'optimizer_updates': 0, 'same_shared_Transformer': True, 'all_trainable_gradients_present': True,
            'protected_DiT50k': rt.initialization.protected_identity(cfg['protected_DiT50k'], full_hash=rank == 0),
            'flash_calls': flash, 'quality_gate_passed': False})
    finally:
        restore()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
