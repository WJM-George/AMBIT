"""Four matched short runs isolate the two positive OPSD objectives."""
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
from scripts.t2a.rl.launch_editing_opsd_comparison import acquire_leases, stop_owned, read, write

ARMS = [('paired_keep', 0., 0.), ('ar_only', 1., 0.), ('dit_only', 0., 1.), ('joint', 1., 1.)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve(); protocol = read(run / 'PROTOCOL.json')
    for group in ('sources', 'configurations'):
        for path, expected in protocol[group].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('Prepared diagnostic file changed: ' + path)
    common, jobs = None, {}
    for i, (arm, ar, dit) in enumerate(ARMS):
        config = run / (arm + '.json'); q = read(config)
        if (q['physical_gpus'] != [2 * i, 2 * i + 1] or q['connected_credit'] or
                q['request_rows_per_rank'] != 8 or q['paired_rows_per_rank'] != 256 or
                q['spatial_recipe']['ar_teacher_weight'] != ar or q['spatial_recipe']['terminal_RF_weight'] != dit):
            raise ValueError('Unexpected arm, batch or positive objective switch.')
        comparable = json.loads(json.dumps(q))
        for key in ('arm', 'output', 'physical_gpus'):
            comparable.pop(key)
        for key in ('ar_teacher_weight', 'terminal_RF_weight'):
            comparable['spatial_recipe'].pop(key)
        if common is not None and comparable != common:
            raise ValueError('Diagnostic arms differ beyond the two positive loss switches.')
        common = comparable
        output = Path(q['output'])
        if (output / 'resume_latest.pt').exists() or (output / 'STOP').exists() or (run / 'USER_HOLD.json').exists():
            raise ValueError('Use a new diagnostic directory; never overwrite or silently resume.')
        command = [sys.executable, '-m', 'torch.distributed.run',
            '--nnodes=1', '--nproc_per_node=2', f'--master_port={29571 + i}',
            str(ROOT / 'scripts/t2a/rl/train_editing_opsd_complete.py'), '--config', str(config),
            '--limit-updates', str(protocol['short_updates'])]
        jobs[arm] = (q, command)
    if args.dry_run:
        print(json.dumps({arm: command for arm, (_, command) in jobs.items()}));return
    path = run / 'STATUS.json'; started = time.time()
    status = dict(phase='STARTING', pid=os.getpid(), started_unix=started,
                  commands={arm: command for arm, (_, command) in jobs.items()})
    write(path, status)
    leases, workers, logs = [], {}, []
    try:
        leases = acquire_leases(protocol['gpu_pool'])
        for arm, (q, command) in jobs.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, q['physical_gpus'])),
                CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            log = (run / (arm + '.log')).open('a');logs.append(log)
            workers[arm] = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        status.update(phase='RUNNING', workers={arm: p.pid for arm,p in workers.items()});write(path,status)
        while any(p.poll() is None for p in workers.values()):
            failures = {arm:p.returncode for arm,p in workers.items() if p.poll() is not None and p.returncode != 0}
            if failures:
                raise RuntimeError('Diagnostic worker failure: ' + repr(failures))
            status.update(observed_unix=time.time(), elapsed_seconds=time.time()-started,
                          exit_codes={arm:p.poll() for arm,p in workers.items()});write(path,status)
            time.sleep(10)
        if any(p.returncode != 0 for p in workers.values()):
            raise RuntimeError('Diagnostic ended with an error.')
        status.update(phase='COMPLETE', completed_unix=time.time(), elapsed_seconds=time.time()-started)
        write(path,status)
    except BaseException as exc:
        stop_owned(workers)
        status.update(phase='FAILED', error=repr(exc), observed_unix=time.time());write(path,status)
        raise
    finally:
        for handle in logs+leases:
            handle.close()


if __name__ == '__main__':
    main()
