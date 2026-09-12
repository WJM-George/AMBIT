"""CPU checks for data regrouping, real checkpoint transfer and allocation guards."""
import argparse
import copy
import gc
import hashlib
import itertools
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from scripts.t2a.experiments.ar_structured_scaleout_v1 import runtime as rt
from scripts.t2a.experiments.ar_structured_scaleout_v1.sampler import make_boundary, ScaleoutSampler
from scripts.t2a.experiments.ar_structured_scaleout_v1.coordination import IndexBuckets, command, verify_owner


def reject(fn):
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('An invalid configuration was accepted')


def run(args):
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():
        raise RuntimeError('Migration QA must not use any GPU')
    cfg = rt.base.read(args.config)
    rt.validate_config(cfg)
    plan = rt.base.read(args.output / 'PLAN.json')
    verify_owner(plan)
    bad = copy.deepcopy(cfg)
    bad['performance']['short_batch_size'] = 32
    reject(lambda: rt.validate_config(bad))
    bad = copy.deepcopy(cfg)
    bad['physical_gpus'] = [0, 1, 2, 3, 4, 5]
    reject(lambda: rt.validate_config(bad))
    reject(lambda: rt.base.allocated_runtime.selected_gpus('0,1,2,3,4,5'))
    dataset = IndexBuckets(cfg['data']['train']['native_index_path'])
    cases = []
    identity = rt.base.read(Path(plan['parent_run']) / 'checkpoints/LATEST.json')
    for epoch in (0, 1, 3):
        parent = rt.base.native.DistributedScenePlanBucketBatchSampler(dataset,
            short_batch_size=192, long_batch_size=120, num_replicas=1, rank=0,
            shuffle=True, seed=cfg['seed'], drop_last=True)
        state = parent.resumable_state_dict(at_epoch_boundary=True)
        state['resume_epoch'] = epoch
        parent.load_resumable_state_dict(state)
        original = list(parent)
        for cursor in (0, 2500, len(parent) - 2, len(parent)):
            if cursor > len(parent):
                continue
            boundary = make_boundary(dataset, seed=cfg['seed'], epoch=epoch,
                next_batch=cursor, checkpoint=identity)
            consumed = {i for row in original[:cursor] for i in row}
            remaining = [i for row in original[cursor:] for i in row]
            retained = [i for row in boundary['global_batches'] for i in row]
            dropped = boundary['dropped_incomplete_batch_ordinals']
            assert len(retained) == len(set(retained))
            assert not consumed.intersection(retained)
            assert not set(retained).intersection(dropped)
            assert sorted(remaining) == sorted(retained + dropped)
            assert len(dropped) <= 312
            samplers = [ScaleoutSampler(dataset, seed=cfg['seed'], rank=r, boundary=boundary) for r in range(6)]
            lengths = {len(s) for s in samplers}
            assert len(lengths) == 1
            if boundary['partial_epoch'] is not None:
                for expected, rows in zip(boundary['global_batches'], zip(*(iter(s) for s in samplers))):
                    assert list(itertools.chain.from_iterable(rows)) == expected
                    assert len({len(row) for row in rows}) == 1 and len(rows[0]) in (64, 40)
            # Native resume must skip descriptors without replaying latent reads.
            for rank in (0, 5):
                s = ScaleoutSampler(dataset, seed=cfg['seed'], rank=rank, boundary=boundary)
                stream = list(s)
                offset = min(3, len(stream) - 1)
                resumed = ScaleoutSampler(dataset, seed=cfg['seed'], rank=rank, boundary=boundary)
                resumed.set_resume_batch_offset(offset)
                assert list(resumed) == stream[offset:]
                state = s.resumable_state_dict(at_epoch_boundary=True)
                s.load_resumable_state_dict(state)
                expected = rt.base.native.DistributedScenePlanBucketBatchSampler(dataset,
                    short_batch_size=64, long_batch_size=40, num_replicas=6, rank=rank,
                    shuffle=True, seed=cfg['seed'], drop_last=True)
                native_state = expected.resumable_state_dict(at_epoch_boundary=True)
                native_state['resume_epoch'] = state['resume_epoch']
                expected.load_resumable_state_dict(native_state)
                assert list(s) == list(expected)
            cases.append({'epoch': epoch, 'cursor': cursor, 'retained': len(retained),
                'dropped_by_drop_last': len(dropped), 'six_rank_batch_sizes': [64, 40]})
            print('SAMPLER_QA=' + str(cases[-1]), flush=True)
        del original
    payload, identity = rt.base.load_joint_checkpoint(identity['checkpoint'])
    rt.base.native._seed_everything(cfg['seed'], 0)
    module, codec, groups, provenance = rt.base.initialization.build(cfg)
    optimizer, scheduler = rt.base.make_optimizer(groups, cfg)
    fingerprint = rt.restore_training_state(module, optimizer, scheduler, payload, include_rng=False)
    assert scheduler.last_epoch == payload['global_step']
    assert len(optimizer.state) == sum(len(group['params']) for group in groups)
    assert not torch.cuda.is_initialized()
    rt.base.write(args.output / 'CPU_QA.json', {'at': rt.base.now(), 'passed': True,
        'source_sha256': rt.source_inventory(), 'sampler_cases': cases,
        'no_consumed_parent_samples_replayed_at_boundary': True,
        'all_unconsumed_parent_rows_accounted_for': True,
        'actual_parent_checkpoint': identity,
        'actual_model_optimizer_scheduler_cursor_transfer_sha256': fingerprint['sha256'],
        'actual_transferred_tensor_elements': fingerprint['tensor_elements'],
        'shared_transformer_same_object': True,
        'protected_DiT50k': rt.base.initialization.protected_identity(cfg['protected_DiT50k'], full_hash=True),
        'GPUs_used': [], 'six_gpu_execution_tested': False,
        'six_gpu_native_restart_proof_required_before_continuation': True,
        'quality_gate_passed': False})
    print('CPU_QA_PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
