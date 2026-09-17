"""Run the revised Editing recipe after fixed step500 evaluation and a short check."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.t2a.rl.launch_editing_opsd_comparison import acquire_leases, stop_owned, write, read


def prepare(run, stage, resume_short=False):
    if resume_short and stage != 'short':
        raise ValueError('--resume-short applies only to the short check.')
    protocol = read(run / 'PROTOCOL.json')
    for group in ('sources', 'configurations'):
        for path, expected in protocol[group].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise ValueError('Prepared file changed: ' + path)
    evaluation = protocol['required_step500_evaluation']
    if hashlib.sha256(Path(evaluation['path']).read_bytes()).hexdigest() != evaluation['sha256']:
        raise ValueError('Required step500 evaluation changed.')
    result = read(evaluation['path'])
    if result['phase'] != 'COMPLETE' or result['rows'] != 1000:
        raise ValueError('Complete the requested matched1000 evaluation before training.')
    candidate = next(r for r in result['results'] if r['id'] == 'OPSD_SPATIAL_OFF500')
    metrics = ['Paired CLAP', 'FD-CLAP', 'FAD', 'FD-PANN', 'KL', 'LSD', 'GCC', 'CRW', 'FSAD']
    if (candidate['rows'] != 1000 or
            any(not math.isfinite(candidate['metrics'][key]) for key in metrics) or
            any(value != 1000 for value in candidate['scalar_coverage'].values())):
        raise ValueError('All nine finite scores and complete sample coverage are required.')
    config = run / 'complete_8gpu.json'
    q = read(config)
    if q['physical_gpus'] != list(range(8)) or q['connected_credit']:
        raise ValueError('This recipe uses all eight GPUs with STE disabled.')
    if (q['global_request_batch'], q['global_paired_batch']) != (16, 512):
        raise ValueError('The revised recipe keeps the authorized16+512 batches.')
    if q['request_rows_per_rank'] * 8 != 16 or q['paired_rows_per_rank'] * 8 != 512:
        raise ValueError('Per-rank and global batch sizes differ.')
    recovery = Path(q['output']) / 'resume_latest.pt'
    command = [sys.executable, '-m', 'torch.distributed.run',
        '--nnodes=1', '--nproc_per_node=8', '--master_port=29561',
        str(ROOT / 'scripts/t2a/rl/train_editing_opsd_complete.py'), '--config', str(config)]
    if stage == 'short':
        command += ['--limit-updates', str(protocol['short_updates'])]
        if recovery.exists() != resume_short:
            raise ValueError('Explicit short resume must match the presence of recovery state.')
        if resume_short:
            command += ['--resume', str(recovery)]
    else:
        review = read(run / 'SHORT_REVIEW.json')
        if (not review['qualified_for_continuation'] or
                review['configuration_sha256'] != protocol['configurations'][str(config)]):
            raise ValueError('The actual recipe has not passed its short training/output review.')
        import torch
        saved = torch.load(recovery, map_location='cpu', weights_only=False, mmap=True)
        if saved['world_size'] != 8 or saved['step'] < protocol['short_updates']:
            raise ValueError('Full run requires the completed eight-GPU short state.')
        command += ['--resume', str(recovery), '--evaluate-at-updates',
                    *map(str, sorted({saved['step'] + 1, 25, 50, 100}))]
        del saved
    return protocol, q, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--stage', choices=['short', 'full'], required=True)
    parser.add_argument('--resume-short', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    protocol, q, command = prepare(run, args.stage, args.resume_short)
    if args.dry_run:
        print(json.dumps(dict(command=command, gpu_processes_started=0), ensure_ascii=False))
        return
    if (run / 'USER_HOLD.json').exists() or (Path(q['output']) / 'STOP').exists():
        raise ValueError('Run remains held; no process started.')
    path = run / ('SHORT_STATUS.json' if args.stage == 'short' else 'TRAINING_STATUS.json')
    status = dict(stage=args.stage, phase='STARTING', pid=os.getpid(), started_unix=time.time(), command=command)
    write(path, status)
    leases, workers, logs = [], {}, []
    try:
        leases = acquire_leases(protocol['gpu_pool'])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
            CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
            HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
        log = (run / f'complete_{args.stage}.log').open('a'); logs.append(log)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        workers['complete'] = process
        status.update(phase='RUNNING', worker_pid=process.pid)
        write(path, status)
        observed_gpu = 0.
        while process.poll() is None:
            status.update(observed_unix=time.time())
            if time.time() - observed_gpu >= 60:
                try:
                    usage = subprocess.run(['nvidia-smi',
                        '--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True,
                        check=True, timeout=10)
                    status['gpu_snapshot_csv'] = usage.stdout.strip().splitlines()
                    status['gpu_snapshot_columns'] = ['index', 'utilization_percent',
                        'memory_used_MiB', 'memory_total_MiB', 'power_W']
                    status['gpu_snapshot_unix'] = time.time()
                except (OSError, subprocess.SubprocessError) as exc:
                    status['gpu_snapshot_error'] = repr(exc)
                observed_gpu = time.time()
            write(path, status)
            time.sleep(10)
        if process.returncode != 0:
            raise RuntimeError('Training worker failed with code ' + str(process.returncode))
        status.update(phase='COMPLETE', completed_unix=time.time(), exit_code=process.returncode)
        write(path, status)
    except BaseException as exc:
        stop_owned(workers)
        status.update(phase='FAILED', error=repr(exc), observed_unix=time.time())
        write(path, status)
        raise
    finally:
        for handle in logs + leases:
            handle.close()


if __name__ == '__main__':
    main()
