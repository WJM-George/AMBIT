"""Run one selected OPSD recipe to100 updates, then its frozen200-request panel."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.t2a.rl.launch_editing_opsd_comparison import acquire_leases, read, stop_owned, write
from scripts.t2a.rl.launch_editing_opsd_selective import sha
from stable_audio_tools.training.transfusion_opsd.top_checkpoints import METRIC_DIRECTIONS, relative_metric_score


def prepare(run):
    p = read(run / 'PROTOCOL.json')
    for group in ('sources', 'configurations', 'evidence'):
        for path, expected in p[group].items():
            if sha(path) != expected:
                raise ValueError('Prepared stability input changed: ' + path)
    q = read(run / 'config.json')
    from scripts.t2a.rl.train_editing_opsd_stability import validate_isolated_resize
    validate_isolated_resize(q, read(q['resize_parent_config']))
    if (p['limit_updates'] != 100 or p['checkpoint']['step'] != 4 or
            {gpu['index'] for gpu in p['gpu_pool']} != {4, 5, 6, 7} or
            Path(q['output']).resolve() != run / 'training'):
        raise ValueError('Unexpected stability boundary or placement.')
    command = [sys.executable, '-m', 'torch.distributed.run',
               '--nnodes=1', '--nproc_per_node=4', '--master_port=29587',
               str(ROOT / 'scripts/t2a/rl/train_editing_opsd_stability.py'),
               '--config', str(run / 'config.json'), '--resume', p['checkpoint']['path'],
               '--resize', '--limit-updates', '100', '--evaluate-at-updates', '5', '25', '50', '100']
    return p, q, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    p, q, command = prepare(run)
    if args.dry_run:
        print(command)
        return
    if (run / 'STATUS.json').exists() or (run / 'training/resume_latest.pt').exists():
        raise ValueError('A stability run must not overwrite or automatically resume existing work.')
    if sha(p['checkpoint']['path']) != p['checkpoint']['sha256']:
        raise ValueError('The selected source checkpoint changed.')
    baseline = read(p['development_baseline'])
    if baseline['step'] != 0 or baseline['requests'] != len(q['validation_ordinals']):
        raise ValueError('Missing the matching original40k development baseline.')
    (run / 'training').mkdir(exist_ok=True)
    write(run / 'training/EVALUATION_step000000.json', baseline)
    workers, logs, leases = {}, [], []
    started = time.time()
    status = dict(phase='STARTING', pid=os.getpid(), started_unix=started,
                  selected_arm=p['selected_arm'], limit_updates=100, physical_gpus=[4, 5, 6, 7])

    def stage(name, jobs):
        if (run / 'USER_HOLD.json').exists() or (run / 'training/STOP').exists():
            raise RuntimeError('The stability run is paused.')
        for arm, (gpus, cmd) in jobs.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            log = (run / f'{arm}_{name}.log').open('a'); logs.append(log)
            workers[arm] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=tuple(handle.fileno() for handle in leases))
        status.update(phase=name, workers={arm: w.pid for arm, w in workers.items()},
                      commands={arm: job[1] for arm, job in jobs.items()})
        while True:
            status.update(observed_unix=time.time(), elapsed_seconds=time.time() - started,
                          exit_codes={arm: w.poll() for arm, w in workers.items()})
            write(run / 'STATUS.json', status)
            if any(w.poll() not in (None, 0) for w in workers.values()):
                raise RuntimeError('A stability worker failed; automatic restart is disabled.')
            if (run / 'USER_HOLD.json').exists():
                raise RuntimeError('The stability run was paused.')
            if all(w.poll() == 0 for w in workers.values()):
                break
            time.sleep(5)
        workers.clear()

    try:
        write(run / 'STATUS.json', status)
        leases = acquire_leases(p['gpu_pool'])
        stage('TRAINING_TO100', {'joint': ([4, 5, 6, 7], command)})
        checkpoint = run / 'training/resume_latest.pt'
        if read(run / 'training/RESUME.json')['step'] != 100:
            raise RuntimeError('Training did not reach its prescribed100-update boundary.')
        identity = dict(path=str(checkpoint), sha256=sha(checkpoint), step=100)
        write(run / 'EVALUATION_CHECKPOINT.json', identity)
        jobs = {}
        for arm, planner, gpus, port in [('original40k', 'old', [4, 5], 29588),
                                       ('candidate100', 'new', [6, 7], 29589)]:
            cmd = [sys.executable, '-m', 'torch.distributed.run',
                   '--nnodes=1', '--nproc_per_node=2', f'--master_port={port}',
                   str(ROOT / 'scripts/t2a/rl/evaluate_editing_opsd_branch_cross.py'),
                   '--config', str(run / 'validation200.json'), '--checkpoint', str(checkpoint),
                   '--checkpoint-sha256', identity['sha256'], '--step', '100',
                   '--planner', planner, '--executor', planner,
                   '--output', str(run / 'validation200' / arm), '--fresh-matched-panel']
            jobs[arm] = (gpus, cmd)
        stage('VALIDATION200', jobs)
        scores = {arm: read(run / 'validation200' / arm / 'EVALUATION_step000100.json')
                  for arm in jobs}
        for score in scores.values():
            if score['requests'] != 200 or score['outputs'] != 400 or score['target_audio_in_inference']:
                raise RuntimeError('Incomplete frozen validation200 evaluation.')
        comparison = relative_metric_score(scores['candidate100']['metrics'], scores['original40k']['metrics'])
        write(run / 'RESULT.json', dict(scope=p['scope'], selected_arm=p['selected_arm'],
              checkpoint=identity, validation200=scores, comparison=comparison,
              automatic_long_training=False))
        lines = ['# OPSD100-update stability validation', '', p['scope'], '',
                 '| Metric | Original40k | Candidate100 | Relative improvement |', '|---|---:|---:|---:|']
        for metric in METRIC_DIRECTIONS:
            lines.append(f"| {metric} | {scores['original40k']['metrics'][metric]:.6f} | "
                         f"{scores['candidate100']['metrics'][metric]:.6f} | "
                         f"{comparison['relative_improvement_percent'][metric]:+.3f}% |")
        lines += ['', 'All200 requests and two noises are included. The panel was frozen before continuation.',
                  'Review operation-level outcomes and severe failures before extending the same recipe.']
        (run / 'TABLE.md').write_text('\n'.join(lines) + '\n')
        status.update(phase='COMPLETE', completed_unix=time.time(), elapsed_seconds=time.time() - started,
                      allocated_GPU_hours=(time.time() - started) * 4 / 3600, automatic_long_training=False)
        write(run / 'STATUS.json', status)
    except BaseException as exc:
        stop_owned(workers)
        status.update(phase='FAILED', error=repr(exc), observed_unix=time.time(),
                      exit_codes={arm: w.poll() for arm, w in workers.items()})
        write(run / 'STATUS.json', status)
        raise
    finally:
        for handle in logs + leases:
            handle.close()


if __name__ == '__main__':
    main()
