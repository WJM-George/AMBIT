"""Run two isolated four-update OPSD changes on physical GPUs4-7.

Reuse the completed low-update joint control, then evaluate both changes on
the same two request/noise panels. A completed experiment never auto-promotes
to a longer training run. Existing run directories are never overwritten.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.t2a.rl.launch_editing_opsd_comparison import acquire_leases, read, stop_owned, write
from stable_audio_tools.training.transfusion_opsd.top_checkpoints import METRIC_DIRECTIONS, relative_metric_score


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check_files(protocol):
    for group in ('sources', 'configurations', 'control_evidence'):
        for path, expected in protocol[group].items():
            if sha(path) != expected:
                raise ValueError('Prepared experiment input changed: ' + path)


def prepare(run):
    p = read(run / 'PROTOCOL.json')
    check_files(p)
    if p['short_updates'] != 4 or {g['index'] for g in p['gpu_pool']} != {4, 5, 6, 7}:
        raise ValueError('This diagnostic is limited to four updates on GPUs4-7.')
    control = read(p['control_config'])
    allowed_differences = {'arm', 'output', 'physical_gpus', 'authorization',
                           'top_checkpoint_policy', 'selective_recipe'}
    jobs = {}
    for arm, gpus, prefix, select, port in (
            ('prefix_only', [4, 5], True, False, 29585),
            ('terminal_only', [6, 7], False, True, 29586)):
        path = run / (arm + '.json')
        q = read(path)
        if {k: v for k, v in q.items() if k not in allowed_differences} != {
                k: v for k, v in control.items() if k not in allowed_differences}:
            raise ValueError('New arm differs from the control beyond the declared changes: ' + arm)
        recipe = q['selective_recipe']
        if (q['physical_gpus'] != gpus or recipe['reference_native_prefix'] != prefix
                or recipe['select_same_plan_improvements'] != select
                or q['connected_credit'] or q['global_request_batch'] != 16
                or q['request_rows_per_rank'] != 8 or q['paired_rows_per_rank'] != 256
                or q['global_paired_batch'] != 512):
            raise ValueError('Unexpected diagnostic topology or recipe: ' + arm)
        output = Path(q['output'])
        if output.resolve() != run / arm or (output / 'resume_latest.pt').exists():
            raise ValueError('Use a fresh experiment output: ' + arm)
        command = [sys.executable, '-m', 'torch.distributed.run',
                   '--nnodes=1', '--nproc_per_node=2', f'--master_port={port}',
                   str(ROOT / 'scripts/t2a/rl/train_editing_opsd_selective.py'),
                   '--config', str(path), '--limit-updates', str(p['short_updates'])]
        jobs[arm] = dict(config=q, port=port, command=command)
    return p, jobs


def report(run, p):
    panels = {}
    for panel, baseline_path, control_path in (
            ('development20', p['development_baseline'], p['development_control']),
            ('confirmation40', p['confirmation_baseline'], p['confirmation_control'])):
        baseline = read(baseline_path)['metrics']
        paths = {'low_update_control': control_path}
        for arm in ('prefix_only', 'terminal_only'):
            folder = run / arm if panel == 'development20' else run / 'confirmation40' / arm
            paths[arm] = str(folder / 'EVALUATION_step000004.json')
        candidates = {arm: dict(metrics=read(path)['metrics'], evaluation=path,
                               **relative_metric_score(read(path)['metrics'], baseline))
                      for arm, path in paths.items()}
        panels[panel] = dict(baseline=baseline, baseline_evaluation=baseline_path,
                            candidates=candidates)
    result = dict(scope=p['scope'], panels=panels, automatic_long_training=False,
                  candidate_selection='Review both panels and per-request failures before promotion.')
    write(run / 'RESULT.json', result)
    lines = ['# OPSD selective-teacher and prefix-retention diagnostic', '', p['scope'], '']
    for name, panel in panels.items():
        lines += ['## ' + name, '', '| Metric | original40k | Low-update control | Prefix only | Terminal only |',
                  '|---|---:|---:|---:|---:|']
        for metric in METRIC_DIRECTIONS:
            values = [panel['baseline'][metric]] + [c['metrics'][metric] for c in panel['candidates'].values()]
            lines.append('| ' + metric + ' | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
        lines += ['', '| Arm | Improved metrics | Mean relative improvement | Worst relative improvement |',
                  '|---|---:|---:|---:|']
        for arm, c in panel['candidates'].items():
            lines.append(f"| {arm} | {c['improved_metrics']}/9 | {c['score']:+.4f}% | {c['worst_relative_improvement_percent']:+.4f}% |")
        lines.append('')
    lines += ['Both are development/confirmation panels, not a final blind test.',
              'No automatic long training or replacement of the protected candidate.']
    (run / 'TABLE.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    p, jobs = prepare(run)
    if args.dry_run:
        print({arm: job['command'] for arm, job in jobs.items()})
        return
    if (run / 'STATUS.json').exists():
        raise ValueError('Do not overwrite or automatically restart an experiment.')
    workers, logs, leases = {}, [], []
    started = time.time()
    status = dict(phase='STARTING', pid=os.getpid(), started_unix=started, physical_gpus=[4, 5, 6, 7])

    def run_stage(stage, commands):
        check_files(p)
        if (run / 'USER_HOLD.json').exists() or any((run / arm / 'STOP').exists() for arm in jobs):
            raise RuntimeError('The experiment is paused.')
        for arm, command in commands.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, jobs[arm]['config']['physical_gpus'])),
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            log = (run / f'{arm}_{stage}.log').open('a'); logs.append(log)
            workers[arm] = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=tuple(handle.fileno() for handle in leases))
        status.update(phase=stage, workers={arm: w.pid for arm, w in workers.items()}, commands=commands)
        while True:
            status.update(observed_unix=time.time(), elapsed_seconds=time.time() - started,
                          exit_codes={arm: w.poll() for arm, w in workers.items()})
            write(run / 'STATUS.json', status)
            if any(w.poll() not in (None, 0) for w in workers.values()):
                raise RuntimeError('A diagnostic worker failed; automatic restart is disabled.')
            if (run / 'USER_HOLD.json').exists():
                raise RuntimeError('The experiment was paused.')
            if all(w.poll() == 0 for w in workers.values()):
                break
            time.sleep(5)

    try:
        write(run / 'STATUS.json', status)
        leases = acquire_leases(p['gpu_pool'])
        run_stage('TRAINING', {arm: job['command'] for arm, job in jobs.items()})
        commands = {}
        for arm, job in jobs.items():
            output = run / arm
            if read(output / 'RESUME.json')['step'] != p['short_updates']:
                raise RuntimeError('Training ended before the prescribed update boundary.')
            checkpoint = output / 'resume_latest.pt'
            identity = dict(path=str(checkpoint), sha256=sha(checkpoint), step=p['short_updates'])
            write(run / (arm + '_EVALUATION_CHECKPOINT.json'), identity)
            commands[arm] = [sys.executable, '-m', 'torch.distributed.run',
                '--nnodes=1', '--nproc_per_node=2', f"--master_port={job['port']}",
                str(ROOT / 'scripts/t2a/rl/evaluate_editing_opsd_branch_cross.py'),
                '--config', str(run / (arm + '_confirmation40.json')),
                '--checkpoint', str(checkpoint), '--checkpoint-sha256', identity['sha256'],
                '--step', str(p['short_updates']), '--planner', 'new', '--executor', 'new',
                '--output', str(run / 'confirmation40' / arm), '--fresh-matched-panel']
        run_stage('CONFIRMATION40', commands)
        report(run, p)
        status.update(phase='COMPLETE', completed_unix=time.time(), elapsed_seconds=time.time() - started,
                      allocated_GPU_hours=(time.time() - started) * 4 / 3600,
                      result=str(run / 'RESULT.json'), automatic_long_training=False)
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
