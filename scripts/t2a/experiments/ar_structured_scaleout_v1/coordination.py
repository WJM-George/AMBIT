"""One-shot 10k checkpoint handoff, six-rank restart proof, then continuation.

This service sends no notifications. It only controls the explicitly recorded
training service and leases GPUs authorized by the user's scale-out request.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.t2a.experiments.ar_structured_scaleout_v1 import runtime as rt
from scripts.t2a.experiments.ar_structured_scaleout_v1.sampler import make_boundary

CHILD = None


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def status(args, phase, **values):
    atomic(args.output / 'STATE.json', {'at': rt.base.now(), 'phase': phase, **values})
    print(json.dumps({'at': rt.base.now(), 'phase': phase, **values}, ensure_ascii=False), flush=True)


class IndexBuckets:
    def __init__(self, path):
        self.buckets = {432: [], 648: []}
        with sqlite3.connect(f'file:{path}?mode=ro&immutable=1', uri=True) as db:
            for ordinal, frames in db.execute('SELECT pair_ordinal,latent_bucket_frames FROM pairs ORDER BY pair_ordinal'):
                self.buckets[int(frames)].append(int(ordinal))

    def __len__(self):
        return sum(map(len, self.buckets.values()))

    def length_bucket_indices(self):
        return self.buckets


def service_state(name):
    text = subprocess.check_output(['systemctl', '--user', 'show', name,
        '-p', 'MainPID', '-p', 'ActiveState'], text=True)
    return dict(row.split('=', 1) for row in text.splitlines())


def verify_owner(plan):
    state = service_state(plan['parent_service'])
    if state != {'MainPID': str(plan['parent_main_pid']), 'ActiveState': 'active'}:
        raise RuntimeError(f'Parent training service ownership changed: {state}')
    command = Path(f"/proc/{plan['parent_main_pid']}/cmdline").read_bytes().replace(b'\0', b' ').decode()
    if plan['parent_run'] not in command or plan['parent_config'] not in command:
        raise RuntimeError('Parent service no longer runs the declared experiment')


def latest_window(path):
    with Path(path).open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 1024 * 1024))
        data = stream.read()
    complete = data[:data.rfind(b'\n')].splitlines()
    return json.loads(complete[-1])


def latest_step(plan):
    paths = [Path(plan['parent_run']) / 'attempts/full50k_initial' / f'rank{r}/windows.jsonl' for r in range(3)]
    return max(latest_window(path)['new_update'] for path in paths)


def command(args, run, attempt, stop, resume, *, evidence=False, fallback=False):
    script = ROOT / 'scripts/t2a/experiments' / ('ar_structured_v1' if fallback else 'ar_structured_scaleout_v1') / 'runtime.py'
    cfg = rt.base.read(args.config)
    result = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
        f'--nproc-per-node={3 if fallback else 6}', str(script), '--config',
        cfg['scaleout']['parent_config'] if fallback else str(args.config),
        '--run-dir', str(run), '--attempt', attempt, '--stop-updates', str(stop), '--resume', str(resume)]
    if evidence:
        boundary_step = rt.base.read(cfg['scaleout']['boundary_file'])['checkpoint']['step']
        result += ['--evidence-updates', str(boundary_step + 2), str(boundary_step + 4)]
    return result


def launch(args, name, cmd, *, timeout=None):
    global CHILD
    rt.base.write(args.output / f'COMMAND_{name}.json', {'at': rt.base.now(), 'command': cmd})
    with (args.output / f'{name}.log').open('x') as log:
        CHILD = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = CHILD.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, cmd)
        except BaseException:
            try:
                os.killpg(CHILD.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                CHILD.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid, signal.SIGKILL)
                CHILD.wait()
            raise
        finally:
            CHILD = None


def review(args, plan, step):
    result = []
    for rank in range(6):
        paths = {'continuous': Path(plan['six_gpu_run']) / f'attempts/migration4/rank{rank}',
            'split': args.output / f'proof_split/attempts/migration2/rank{rank}',
            'resumed': args.output / f'proof_split/attempts/native2/rank{rank}'}
        for path in paths.values():
            done = rt.base.read(path / 'COMPLETE.json')
            if not all(done[k] for k in ('joint_AR_and_RF_trained', 'shared_transformer_same_object',
                    'protected_DiT50k_unchanged', 'frozen_encoders_unchanged')):
                raise RuntimeError('Six-GPU proof failed a model or frozen-weight invariant')
            if rt.base.read(path / 'DDP_REVIEW.json').get('has_rebuilt_buckets', 0):
                raise RuntimeError('Six-GPU DDP buckets changed')
        if rt.base.read(paths['continuous'] / 'TRANSFER_STATE.json') != rt.base.read(paths['split'] / 'TRANSFER_STATE.json'):
            raise RuntimeError('Parent state did not transfer identically in both proof branches')
        for offset, name in ((2, 'split'), (4, 'resumed')):
            left = rt.base.read(paths['continuous'] / f'STATE_step{step + offset:06d}.json')
            right = rt.base.read(paths[name] / f'STATE_step{step + offset:06d}.json')
            if left != right:
                raise RuntimeError(f'Six-GPU native restart differs at {step + offset}, rank {rank}')
            result.append({'rank': rank, 'step': step + offset, 'sha256': left['sha256']})
        if rt.base.read(paths['split'] / f'STATE_step{step + 2:06d}.json') != rt.base.read(paths['resumed'] / 'RESUME_STATE.json'):
            raise RuntimeError('Six-GPU native resume omitted saved state')
        rows = {name: [json.loads(line) for line in (path / 'windows.jsonl').read_text().splitlines()]
                for name, path in paths.items()}
        if rows['continuous'] != rows['split'] + rows['resumed']:
            raise RuntimeError('Six-GPU restart changed samples, instructions or LR clocks')
        if any(len(row['pair_ids']) not in (64, 40) for row in rows['continuous']):
            raise RuntimeError('The requested per-GPU batch was not maintained')
    rt.base.write(args.output / 'SIX_GPU_RESTART_REVIEW.json', {
        'at': rt.base.now(), 'six_rank_native_restart_bitwise_exact': True,
        'parent_optimizer_and_scheduler_transferred': True,
        'per_gpu_batch_sizes': {'short': 64, 'long': 40},
        'global_batch_sizes': {'short': 384, 'long': 240}, 'states': result,
        'quality_gate_passed': False, 'independent_test_used': False})


def operate(args):
    cfg = rt.base.read(args.config)
    rt.validate_config(cfg)
    plan = rt.base.read(args.output / 'PLAN.json')
    if rt.base.sha(args.config) != plan['config_sha256']:
        raise RuntimeError('Prepared six-GPU configuration changed')
    if any(plan[key] != cfg['scaleout'][key] for key in
            ('parent_run', 'parent_config', 'parent_service', 'parent_main_pid', 'six_gpu_run')):
        raise RuntimeError('The handoff plan and authorized configuration disagree')
    qa = rt.base.read(args.output / 'CPU_QA.json')
    if qa.get('passed') is not True or qa.get('source_sha256') != rt.source_inventory():
        raise RuntimeError('CPU migration QA is missing or stale')
    parent_run = Path(plan['parent_run'])
    extra = ExitStack()
    old = ExitStack()
    stopped = False
    proof_passed = False
    old_leases = None
    last_wait_reason = None
    try:
        status(args, 'WAITING_FOR_CHECKPOINT', minimum_step=10000,
            per_gpu_batch_sizes={'short': 64, 'long': 40}, global_batch_sizes={'short': 384, 'long': 240})
        while True:
            verify_owner(plan)
            latest = parent_run / 'checkpoints/LATEST.json'
            identity = rt.base.read(latest) if latest.exists() else None
            eligible = identity and 10000 <= identity['step'] < cfg['new_updates'] - 4
            if not eligible or latest_step(plan) > identity['step'] + 24:
                time.sleep(10)
                continue
            try:
                extra_leases = extra.enter_context(rt.base.allocated_runtime.gpu_lease(
                    rt.base.allocated_runtime.gpu_topology([2, 3, 4])))
            except RuntimeError as error:
                extra.close()
                if str(error) != last_wait_reason:
                    status(args, 'WAITING_FOR_AVAILABLE_GPU2_4', reason=str(error),
                        parent_training_continues=True)
                    last_wait_reason = str(error)
                time.sleep(10)
                continue
            payload, identity = rt.base.load_joint_checkpoint(identity['checkpoint'], require_latest=True)
            bucket_data = IndexBuckets(cfg['data']['train']['native_index_path'])
            boundary = make_boundary(bucket_data, seed=cfg['seed'], epoch=payload['epoch'],
                next_batch=payload['next_batch'], checkpoint=identity)
            rt.validate_parent_payload(payload, identity, cfg, boundary)
            if latest_step(plan) > identity['step'] + 24:
                del payload, bucket_data, boundary
                extra.close()
                time.sleep(10)
                continue
            verify_owner(plan)
            rt.base.write(cfg['scaleout']['boundary_file'], boundary)
            rt.base.write(args.output / 'PARENT_CHECKPOINT_VERIFIED.json', {'at': rt.base.now(),
                'checkpoint': identity, 'scheduler_step': payload['scheduler']['last_epoch'],
                'last_observed_parent_update': latest_step(plan),
                'dropped_incomplete_boundary_examples': len(boundary['dropped_incomplete_batch_ordinals'])})
            del payload, bucket_data
            status(args, 'STOPPING_PARENT_AFTER_VERIFIED_CHECKPOINT', checkpoint=identity)
            subprocess.run(['systemctl', '--user', 'stop', plan['parent_service']], check=True, timeout=150)
            stopped = True
            old_topology = rt.base.allocated_runtime.gpu_topology([5, 6, 7])
            old_leases = old.enter_context(rt.base.allocated_runtime.gpu_lease(old_topology))
            topology = rt.base.allocated_runtime.gpu_topology(rt.GPUS)
            rt.base.allocated_runtime.configure_visibility(topology)
            os.environ['EDITING_GPU_LEASES'] = json.dumps(extra_leases + old_leases)
            step = identity['step']
            break
        status(args, 'SIX_GPU_RESTART_VALIDATION', parent_step=step, gpus=rt.GPUS)
        launch(args, 'migration4', command(args, plan['six_gpu_run'], 'migration4', step + 4,
            identity['checkpoint'], evidence=True), timeout=1800)
        launch(args, 'migration2', command(args, args.output / 'proof_split', 'migration2', step + 2,
            identity['checkpoint'], evidence=True), timeout=1800)
        launch(args, 'native2', command(args, args.output / 'proof_split', 'native2', step + 4,
            args.output / f'proof_split/checkpoints/step-{step + 2:08d}.pt', evidence=True), timeout=1800)
        review(args, plan, step)
        proof_passed = True
        status(args, 'RUNNING_SIX_GPU_CONTINUATION', restored_parent_step=step,
            verified_six_gpu_step=step + 4, target_step=50000, run_directory=plan['six_gpu_run'])
        launch(args, 'full50k_after_scaleout', command(args, plan['six_gpu_run'],
            'full50k_after_scaleout', 50000, Path(plan['six_gpu_run']) / f'checkpoints/step-{step + 4:08d}.pt'))
        status(args, 'COMPLETE_50000', run_directory=plan['six_gpu_run'], quality_gate_passed=False)
    except Exception as error:
        status(args, 'FAILED', error=repr(error), original_service_stopped=stopped,
            six_gpu_proof_passed=proof_passed)
        if stopped and not proof_passed:
            # The parent checkpoint remains valid and the original source/config
            # remain untouched. Return to its proven three-rank continuation.
            extra.close()
            topology = rt.base.allocated_runtime.gpu_topology([5, 6, 7])
            if old_leases is None:
                old_leases = old.enter_context(rt.base.allocated_runtime.gpu_lease(topology))
            rt.base.allocated_runtime.configure_visibility(topology)
            os.environ['EDITING_GPU_LEASES'] = json.dumps(old_leases)
            identity = rt.base.read(cfg['scaleout']['boundary_file'])['checkpoint']
            status(args, 'FALLBACK_RUNNING_THREE_GPUS', reason=repr(error), parent_step=identity['step'])
            launch(args, 'three_gpu_fallback', command(args, parent_run,
                'scaleout_validation_fallback', 50000, identity['checkpoint'], fallback=True))
            status(args, 'COMPLETE_50000_ON_FALLBACK', quality_gate_passed=False)
        else:
            raise
    finally:
        old.close()
        extra.close()


def stop(signum, frame):
    if CHILD is not None:
        try:
            os.killpg(CHILD.pid, signum)
        except ProcessLookupError:
            pass
    raise SystemExit(128 + signum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    operate(args)
