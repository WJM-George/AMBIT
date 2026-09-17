"""Explicit launch of the two Editing OPSD arms; never auto-resume or wait for GPUs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def check_user_hold(directory):
    if (directory / 'USER_HOLD.json').exists():
        raise RuntimeError('Code preparation only: wait for the user\'s explicit go-ahead. No GPU process was started.')


def prepare(directory, *, expand=False, resume=False, resize=False, limit_updates=None):
    """Read configuration and build commands without querying or loading a GPU."""
    if resize and not (resume and expand):
        raise ValueError('--resize requires --resume --expand for the first 2-to-4-rank transition.')
    if limit_updates is not None and limit_updates <= 0:
        raise ValueError('--limit-updates must be a positive total update count.')
    manifest = read(directory / 'PROTOCOL.json')
    for field in ('sources', 'configurations'):
        for path, expected in manifest[field].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError(f'Prepared file changed: {path}')
    jobs = {}
    for arm, port in [('off', 29431), ('on', 29432)]:
        config = directory / f'{arm}{"4" if expand else ""}.json'
        q = read(config)
        world = 4 if expand else 2
        expected_gpus = list(range(world)) if arm == 'off' else list(range(world, 2 * world))
        if q['physical_gpus'] != expected_gpus or q['connected_credit'] != (arm == 'on'):
            raise ValueError(f'Incorrect GPU placement or proxy switch for {arm}.')
        if q['request_rows_per_rank'] * world != q['global_request_batch'] or q['paired_rows_per_rank'] * world != q['global_paired_batch']:
            raise ValueError('Effective batch changes with the requested topology.')
        if limit_updates is not None and limit_updates > q['maximum_updates']:
            raise ValueError('Startup limit exceeds the configured training schedule.')
        command = [sys.executable, '-m', 'torch.distributed.run',
                   '--nnodes=1', f'--nproc_per_node={world}', f'--master_port={port}',
                   str(ROOT / 'scripts/t2a/rl/train_editing_opsd_stream.py'), '--config', str(config)]
        if resume:
            command += ['--resume', str(Path(q['output']) / 'resume_latest.pt')]
        if resize:
            command += ['--resize']
        if limit_updates is not None:
            command += ['--limit-updates', str(limit_updates)]
        jobs[arm] = dict(config=q, command=command)
    ignored = {'arm', 'output', 'physical_gpus', 'connected_credit', 'resize_parent_config', 'resume_config_parent'}
    comparable = [{k: v for k, v in job['config'].items() if k not in ignored} for job in jobs.values()]
    if comparable[0] != comparable[1]:
        raise ValueError('OFF and ON differ beyond the proxy switch and placement.')
    return manifest, jobs


def acquire_leases(pool):
    """Use the existing task coordination locks; any compute process makes a GPU busy."""
    leases = []
    try:
        identity = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'],
                                  check=True, capture_output=True, text=True, timeout=10)
        actual = {int(parts[0]): parts[1] for line in identity.stdout.splitlines()
                  if (parts := [part.strip() for part in line.split(',')]) and len(parts) == 2}
        for gpu in pool:
            if actual.get(gpu['index']) != gpu['uuid']:
                raise RuntimeError(f'GPU identity changed: {gpu["index"]}')
            handle = Path(gpu['lock']).open('rb')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise
            leases.append(handle)
        observation = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                                     check=True, capture_output=True, text=True, timeout=10)
        selected = {gpu['uuid'] for gpu in pool}
        busy = [line for line in observation.stdout.splitlines() if line.split(',')[0].strip() in selected]
        if busy:
            raise RuntimeError(f'GPUs still occupied; no automatic waiting or restart: {busy}')
        return leases
    except BaseException:
        for handle in leases:
            handle.close()
        raise


def stop_owned(workers):
    for worker in workers.values():
        if worker.poll() is None:
            try:
                os.killpg(worker.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 20
    for worker in workers.values():
        try:
            worker.wait(timeout=max(.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait(timeout=5)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--expand', action='store_true', help='OFF GPUs0-3; ON GPUs4-7. All must be free.')
    parser.add_argument('--resize', action='store_true', help='Explicit first migration of a saved 2-rank run to 4 ranks.')
    parser.add_argument('--limit-updates', type=int)
    parser.add_argument('--paired-rows-per-rank', type=int)
    parser.add_argument('--paired-microbatch', type=int)
    parser.add_argument('--plan-audit-every', type=int)
    parser.add_argument('--batch-change-after-update', type=int)
    parser.add_argument('--evaluate-at-updates', type=int, nargs='+', default=[])
    parser.add_argument('--dry-run', action='store_true', help='CPU-only command/configuration validation, allowed during user hold.')
    args = parser.parse_args(argv)
    directory = args.run_dir.resolve()
    if not args.dry_run:
        check_user_hold(directory)
    manifest, jobs = prepare(directory, expand=args.expand, resume=args.resume, resize=args.resize,
                             limit_updates=args.limit_updates)
    overrides = {key: value for key, value in [('paired-rows-per-rank', args.paired_rows_per_rank),
                 ('paired-microbatch', args.paired_microbatch), ('plan-audit-every', args.plan_audit_every),
                 ('batch-change-after-update', args.batch_change_after_update)] if value is not None}
    if any(value < 1 for value in overrides.values()):
        raise ValueError('Execution overrides must be positive.')
    for job in jobs.values():
        for key, value in overrides.items():
            job['command'] += [f'--{key}', str(value)]
        if any(not 0 < step <= job['config']['maximum_updates'] for step in args.evaluate_at_updates):
            raise ValueError('Additional evaluation updates must be within the training schedule.')
        if args.evaluate_at_updates:
            job['command'] += ['--evaluate-at-updates', *map(str, sorted(set(args.evaluate_at_updates)))]
    if args.dry_run:
        print(json.dumps(dict(user_hold=(directory/'USER_HOLD.json').exists(), gpu_processes_started=0,
                              commands={arm: job['command'] for arm, job in jobs.items()}), indent=2))
        return
    for job in jobs.values():
        output = Path(job['config']['output'])
        if (output/'STOP').exists():
            raise RuntimeError(f'STOP is still set: {output}. Resume only after the user gives the go-ahead.')
        checkpoint_exists = (output/'resume_latest.pt').exists()
        if checkpoint_exists != args.resume:
            raise RuntimeError(f'Use explicit --resume with an existing recovery checkpoint: {output}')
    selected = {gpu for job in jobs.values() for gpu in job['config']['physical_gpus']}
    pool = sorted([gpu for gpu in manifest['gpu_pool'] if gpu['index'] in selected], key=lambda gpu: gpu['index'])
    if {gpu['index'] for gpu in pool} != selected:
        raise ValueError('Missing GPU coordination lock definitions.')
    leases, workers, logs = [], {}, []
    status = dict(phase='PREPARING', driver_pid=os.getpid(), started_unix=time.time(),
                  stage='resize' if args.resize else ('resume' if args.resume else 'startup'),
                  physical_gpus={arm: job['config']['physical_gpus'] for arm, job in jobs.items()},
                  limit_updates=args.limit_updates, execution_overrides=overrides,
                  extra_evaluation_updates=sorted(set(args.evaluate_at_updates)))
    try:
        leases = acquire_leases(pool)
        check_user_hold(directory)
        for arm, job in jobs.items():
            check_user_hold(directory)
            q = job['config']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, q['physical_gpus'])),
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            log = (directory/f'{arm}_{status["stage"]}.log').open('a')
            logs.append(log)
            workers[arm] = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True, pass_fds=tuple(handle.fileno() for handle in leases))
        status.update(phase='RUNNING', pids={arm: worker.pid for arm, worker in workers.items()})
        write(directory/'LAUNCH_STATUS.json', status)
        print(json.dumps(status), flush=True)
        while any(worker.poll() is None for worker in workers.values()):
            if (directory/'USER_HOLD.json').exists():
                status['phase'] = 'USER_HOLD'
                stop_owned(workers)
                break
            if any(worker.poll() not in (None, 0) for worker in workers.values()):
                status['phase'] = 'FAILED'
                stop_owned(workers)
                break
            status.update(exit_codes={arm: worker.poll() for arm, worker in workers.items()}, observed_unix=time.time())
            write(directory/'LAUNCH_STATUS.json', status)
            time.sleep(5)
        if status['phase'] == 'RUNNING':
            status['phase'] = 'COMPLETE' if all(worker.returncode == 0 for worker in workers.values()) else 'FAILED'
        status.update(exit_codes={arm: worker.poll() for arm, worker in workers.items()}, finished_unix=time.time())
        write(directory/'LAUNCH_STATUS.json', status)
        if status['phase'] == 'FAILED':
            raise RuntimeError('At least one training arm failed. Inspect logs; automatic restart is disabled.')
    except BaseException as exc:
        stop_owned(workers)
        status.update(phase='USER_HOLD' if (directory/'USER_HOLD.json').exists() else 'FAILED',
                      reason=str(exc), finished_unix=time.time(), exit_codes={arm: worker.poll() for arm, worker in workers.items()})
        write(directory/'LAUNCH_STATUS.json', status)
        raise
    finally:
        for handle in logs + leases:
            handle.close()


if __name__ == '__main__':
    main()
