"""Durable launch of short spatial-OPSD checks or one qualified eight-GPU run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.t2a.rl.launch_editing_opsd_comparison import acquire_leases, stop_owned


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['short', 'full'], required=True)
    parser.add_argument('--arm', choices=['repair', 'repair_spatial'])
    parser.add_argument('--resume-short', action='store_true')
    parser.add_argument('--fresh-from-base', action='store_true',
                        help='Explicit fresh full run from the original checkpoint; never overwrite a recovery state.')
    parser.add_argument('--evaluate-at-updates', type=int, nargs='+', default=[],
                        help='Extra cumulative review updates for the resumed full run.')
    args = parser.parse_args()
    if args.fresh_from_base and (args.stage != 'full' or args.resume_short):
        raise ValueError('A fresh full run cannot also request a short resume.')
    run = args.run_dir.resolve()
    protocol = json.loads((run / 'PROTOCOL.json').read_text())
    for filename, expected in protocol['sources'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Training source changed after preparation: ' + filename)
    arms = ['repair', 'repair_spatial'] if args.stage == 'short' else [args.arm]
    if args.stage == 'full':
        if args.arm is None:
            raise ValueError('Choose the verified short-run arm.')
        review = json.loads((run / 'SHORT_REVIEW.json').read_text())
        if not review['qualified_for_continuation'] or review['selected_arm'] != args.arm:
            raise ValueError('Short real-generation verification has not qualified this arm.')
    jobs = {}
    for index, arm in enumerate(arms):
        config = run / (arm + ('_8gpu' if args.stage == 'full' else '') + '.json')
        if hashlib.sha256(config.read_bytes()).hexdigest() != protocol['configurations'][str(config)]:
            raise ValueError('Configuration changed after preparation.')
        q = json.loads(config.read_text())
        if q['connected_credit']:
            raise ValueError('The spatial mainline must disable STE.')
        world = len(q['physical_gpus'])
        if (args.stage == 'full' and q['physical_gpus'] != list(range(8))) or world not in (4, 8):
            raise ValueError('Unexpected GPU allocation.')
        command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
            f'--nproc_per_node={world}', f'--master_port={29551 + index}',
            str(ROOT / 'scripts/t2a/rl/train_editing_opsd_spatial.py'), '--config', str(config)]
        if args.stage == 'short':
            command += ['--limit-updates', str(protocol['short_updates'])]
            if args.resume_short:
                command += ['--resume', str(Path(q['output']) / 'resume_latest.pt')]
        else:
            recovery = Path(q['output']) / 'resume_latest.pt'
            command += ['--paired-microbatch', str(q['paired_microbatch']),
                        '--plan-audit-every', str(q['native_plan_audit_every'])]
            if args.fresh_from_base:
                if (not protocol.get('fresh_start_authorized') or q.get('initial_overlay') is not None
                        or any(key in q for key in ('resize_parent_config', 'resume_config_parent', 'request_batch_transition'))):
                    raise ValueError('Fresh initialization must explicitly use only the original checkpoint.')
                if recovery.exists() or any(Path(q['output']).glob('UPDATES_rank*.jsonl')):
                    raise ValueError('Existing progress requires resume; fresh mode never overwrites it.')
                saved_step = 0
            else:
                import torch
                state = torch.load(recovery, map_location='cpu', weights_only=False, mmap=True)
                saved_world, saved_step = state['world_size'], state['step']
                del state
                command += ['--resume', str(recovery)]
                if saved_world != world:
                    command += ['--resize']
            if any(not saved_step < step <= q['maximum_updates'] for step in args.evaluate_at_updates):
                raise ValueError('Extra reviews must follow the saved update and stay within the schedule.')
            reviews = sorted({saved_step + 1, 25, 50, 100, *args.evaluate_at_updates})
            command += ['--evaluate-at-updates', *map(str, reviews)]
        jobs[arm] = (q, command)
    selected = {gpu for q, _ in jobs.values() for gpu in q['physical_gpus']}
    pool = [gpu for gpu in protocol['gpu_pool'] if gpu['index'] in selected]
    leases, workers, logs = [], {}, []
    status = dict(stage=args.stage, phase='STARTING', pid=os.getpid(), started_unix=time.time(),
                  commands={arm: command for arm, (_, command) in jobs.items()})
    path = run / ('SHORT_STATUS.json' if args.stage == 'short' else 'TRAINING_STATUS.json')
    write(path, status)
    try:
        leases = acquire_leases(pool)
        for arm, (q, command) in jobs.items():
            output = Path(q['output'])
            if (output / 'STOP').exists():
                raise ValueError('STOP remains set for this arm.')
            if args.stage == 'short' and not args.resume_short and (output / 'resume_latest.pt').exists():
                raise ValueError('Use an explicit short resume for an existing checkpoint.')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, q['physical_gpus'])),
                CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            log = (run / f'{arm}_{args.stage}.log').open('a'); logs.append(log)
            workers[arm] = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        status.update(phase='RUNNING', workers={arm: p.pid for arm, p in workers.items()})
        write(path, status)
        while any(p.poll() is None for p in workers.values()):
            failures = {arm: p.returncode for arm, p in workers.items() if p.poll() is not None and p.returncode != 0}
            if failures:
                raise RuntimeError('A training worker failed: ' + repr(failures))
            status['observed_unix'] = time.time()
            status['exit_codes'] = {arm: p.poll() for arm, p in workers.items()}
            write(path, status)
            time.sleep(10)
        if any(p.returncode != 0 for p in workers.values()):
            raise RuntimeError('Training ended with an error.')
        status.update(phase='COMPLETE', completed_unix=time.time(), exit_codes={arm: p.returncode for arm, p in workers.items()})
        write(path, status)
    except BaseException as exc:
        stop_owned(workers)
        status.update(phase='FAILED', error=repr(exc), observed_unix=time.time(),
                      exit_codes={arm: p.poll() for arm, p in workers.items()})
        write(path, status)
        raise
    finally:
        for handle in logs + leases:
            handle.close()


if __name__ == '__main__':
    main()
