"""Queue two bounded0915-recipe/LR checks after an identified GPU4-7 job."""
import argparse
import json
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

ARMS = {'legacy_current_lr': [4, 5], 'legacy_original_lr': [6, 7]}
PORTS = {'legacy_current_lr': 29590, 'legacy_original_lr': 29591}


def validate_configuration(q, control, old, arm):
    allowed = {'schema', 'arm', 'output', 'physical_gpus', 'authorization', 'supervision',
               'complete_recipe', 'legacy_recipe', 'top_checkpoint_policy', 'checkpoint_retention',
               'learning_rates'}
    if ({k: v for k, v in q.items() if k not in allowed} !=
            {k: v for k, v in control.items() if k not in allowed}):
        raise ValueError('Undeclared difference from the completed four-update control.')
    expected_lr = control['learning_rates'] if arm == 'legacy_current_lr' else old['learning_rates']
    if (q['learning_rates'] != expected_lr or q['spatial_recipe'] != old['spatial_recipe'] or
            q['physical_gpus'] != ARMS[arm] or q['request_rows_per_rank'] != 8 or
            q['paired_rows_per_rank'] != 256 or q['global_request_batch'] != 16 or
            q['global_paired_batch'] != 512 or q['connected_credit'] or
            any(k in q for k in ('complete_recipe', 'selective_recipe', 'initialization_checkpoint'))):
        raise ValueError('Unexpected0915 recipe, learning rates, scope or16+512 batch.')


def verify_inputs(p):
    for group in ('sources', 'configurations', 'evidence'):
        for path, digest in p[group].items():
            if sha(path) != digest:
                raise ValueError('Prepared experiment input changed: ' + path)


def prepare(run):
    p = read(run / 'PROTOCOL.json')
    verify_inputs(p)
    if p['short_updates'] != 4 or {g['index'] for g in p['gpu_pool']} != {4, 5, 6, 7}:
        raise ValueError('This diagnostic is limited to four updates on GPUs4-7.')
    control, old = read(p['control_config']), read(p['archived_config'])
    jobs = {}
    for arm, gpus in ARMS.items():
        q = read(run / (arm + '.json'))
        validate_configuration(q, control, old, arm)
        if Path(q['output']).resolve() != run / arm:
            raise ValueError('Diagnostic outputs must be isolated.')
        commands = [sys.executable, '-m', 'torch.distributed.run',
                    '--nnodes=1', '--nproc_per_node=2', f'--master_port={PORTS[arm]}',
                    str(ROOT / 'scripts/t2a/rl/train_editing_opsd_0915_recipe.py'),
                    '--config', str(run / (arm + '.json')), '--limit-updates', '4']
        jobs[arm] = dict(config=q, command=commands)
    return p, jobs


def process_active(pid, start_ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and int(fields[19]) == start_ticks
    except FileNotFoundError:
        return False


def audit_training_samples(run, p):
    control = Path(read(p['control_config'])['output'])
    audits, first = [], {}
    for arm in ARMS:
        for rank in range(2):
            filename = f'UPDATES_rank{rank}.jsonl'
            rows = [json.loads(line) for line in (run / arm / filename).read_text().splitlines()]
            previous = [json.loads(line) for line in (control / filename).read_text().splitlines()]
            if len(rows) != 4 or len(previous) != 4:
                raise ValueError('A matched arm must contain exactly four training updates.')
            for row, reference in zip(rows, previous):
                paired = row['paired_ordinals'] == reference['paired_ordinals']
                requested = ([r['ordinal'] for r in row['request_updates']] ==
                             [r['ordinal'] for r in reference['request_updates']])
                audits.append(dict(arm=arm, rank=rank, step=row['step'],
                                   paired_ordinals_equal=paired, request_ordinals_equal=requested))
                if not paired or not requested or row['step'] != reference['step']:
                    raise ValueError('Training data streams differ from the completed control.')
            first[(arm, rank)] = rows[0]
    initial = []
    for rank in range(2):
        low, high = first[('legacy_current_lr', rank)], first[('legacy_original_lr', rank)]
        initial.append(dict(rank=rank,
            first_execution_metrics_equal=([r['execution_metrics'] for r in low['request_updates']] ==
                                           [r['execution_metrics'] for r in high['request_updates']]),
            first_clip_norms_equal=low['clip_norms'] == high['clip_norms'],
            first_paired_loss_equal=low['paired_loss'] == high['paired_loss']))
    write(run / 'MATCHED_TRAINING_AUDIT.json', dict(samples=audits, lr_only_initial_forward=initial,
        note='Two archived arms differ only in learning rates. Teacher rules intentionally differ from the newer control.'))


def report(run, p):
    panels = {}
    for panel, baseline, control in (
            ('development20', p['development_baseline'], p['development_control']),
            ('confirmation40', p['confirmation_baseline'], p['confirmation_control'])):
        reference = read(baseline)['metrics']
        paths = {'current_recipe_current_lr': control}
        for arm in ARMS:
            folder = run / arm if panel == 'development20' else run / 'confirmation40' / arm
            paths[arm] = str(folder / 'EVALUATION_step000004.json')
        scores = {}
        for arm, path in paths.items():
            evaluation = read(path)
            expected = 20 if panel == 'development20' else 40
            if (evaluation['requests'] != expected or evaluation['outputs'] != expected * 2 or
                    evaluation['target_audio_in_inference']):
                raise ValueError('Incomplete or unmatched evaluation: ' + path)
            scores[arm] = dict(evaluation=path, metrics=evaluation['metrics'],
                               **relative_metric_score(evaluation['metrics'], reference))
        panels[panel] = dict(baseline_evaluation=baseline, baseline=reference, candidates=scores,
            lr_effect_within_0915=relative_metric_score(scores['legacy_current_lr']['metrics'],
                                                       scores['legacy_original_lr']['metrics']),
            recipe_effect_at_current_lr=relative_metric_score(scores['legacy_current_lr']['metrics'],
                                                              scores['current_recipe_current_lr']['metrics']))
    write(run / 'RESULT.json', dict(scope=p['scope'], panels=panels, automatic_long_training=False,
        interpretation='Four updates test early direction only. Review tails and then stability before any full run.'))
    lines = ['# 0915配方与学习率对照', '', p['scope'], '']
    for panel, values in panels.items():
        lines += ['## ' + panel, '', '| Metric | original40k | Current recipe/current LR |0915/current LR |0915/original LR |',
                  '|---|---:|---:|---:|---:|']
        for metric in METRIC_DIRECTIONS:
            row = [values['baseline'][metric]] + [s['metrics'][metric] for s in values['candidates'].values()]
            lines.append('| ' + metric + ' | ' + ' | '.join(f'{x:.6f}' for x in row) + ' |')
        lines += ['', 'Four-update evidence only; no automatic long-run promotion.', '']
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
    if (run / 'STATUS.json').exists() or any((run / arm / 'resume_latest.pt').exists() for arm in ARMS):
        raise ValueError('Do not overwrite or automatically restart a diagnostic.')
    workers, logs, leases = {}, [], []
    queued = time.time()
    status = dict(phase='QUEUED_AFTER_STABILITY', pid=os.getpid(), queued_unix=queued,
                  predecessor=p['predecessor'], physical_gpus=[4, 5, 6, 7], allocated_GPUs=0,
                  short_updates=4, automatic_long_training=False)

    def held():
        return (run / 'USER_HOLD.json').exists() or any((run / arm / 'STOP').exists() for arm in ARMS)

    def stage(name, commands):
        verify_inputs(p)
        if held():
            raise RuntimeError('Diagnostic paused before launch.')
        for arm, command in commands.items():
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, ARMS[arm])),
                       CUDA_DEVICE_ORDER='PCI_BUS_ID', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
            log = (run / f'{arm}_{name}.log').open('a'); logs.append(log)
            workers[arm] = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                pass_fds=tuple(handle.fileno() for handle in leases))
        status.update(phase=name, workers={arm: w.pid for arm, w in workers.items()}, commands=commands)
        while True:
            status.update(observed_unix=time.time(), exit_codes={arm: w.poll() for arm, w in workers.items()})
            write(run / 'STATUS.json', status)
            if any(w.poll() not in (None, 0) for w in workers.values()) or held():
                raise RuntimeError('Diagnostic failed or was paused; no automatic restart.')
            if all(w.poll() == 0 for w in workers.values()):
                break
            time.sleep(5)
        workers.clear()

    try:
        previous = p['predecessor']
        while True:
            prior = read(previous['status'])
            if prior['pid'] != previous['pid'] or prior['phase'] == 'FAILED' or held():
                raise RuntimeError('Predecessor changed/failed, or diagnostic paused.')
            active = process_active(previous['pid'], previous['start_ticks'])
            if not active:
                if prior['phase'] != 'COMPLETE':
                    raise RuntimeError('Predecessor exited without completing the prescribed review.')
                break
            status.update(observed_unix=time.time(), predecessor_phase=prior['phase'])
            write(run / 'STATUS.json', status)
            time.sleep(15)
        verify_inputs(p)
        leases = acquire_leases(p['gpu_pool'])
        status.update(started_unix=time.time(), allocated_GPUs=4)
        stage('TRAINING_FOUR_UPDATES', {arm: job['command'] for arm, job in jobs.items()})
        audit_training_samples(run, p)
        commands = {}
        for arm in ARMS:
            checkpoint = run / arm / 'resume_latest.pt'
            if read(run / arm / 'RESUME.json')['step'] != 4:
                raise RuntimeError('Training did not reach four updates.')
            identity = dict(path=str(checkpoint), sha256=sha(checkpoint), step=4)
            write(run / (arm + '_EVALUATION_CHECKPOINT.json'), identity)
            commands[arm] = [sys.executable, '-m', 'torch.distributed.run',
                '--nnodes=1', '--nproc_per_node=2', f'--master_port={PORTS[arm]}',
                str(ROOT / 'scripts/t2a/rl/evaluate_editing_opsd_branch_cross.py'),
                '--config', str(run / (arm + '_confirmation40.json')), '--checkpoint', str(checkpoint),
                '--checkpoint-sha256', identity['sha256'], '--step', '4', '--planner', 'new',
                '--executor', 'new', '--output', str(run / 'confirmation40' / arm), '--fresh-matched-panel']
        stage('CONFIRMATION40', commands)
        report(run, p)
        status.update(phase='COMPLETE', completed_unix=time.time(), allocated_GPUs=0,
            allocated_GPU_hours=(time.time() - status['started_unix']) * 4 / 3600,
            result=str(run / 'RESULT.json'))
        write(run / 'STATUS.json', status)
    except BaseException as exc:
        stop_owned(workers)
        status.update(phase='FAILED', error=repr(exc), observed_unix=time.time())
        write(run / 'STATUS.json', status)
        raise
    finally:
        for handle in logs + leases:
            handle.close()


if __name__ == '__main__':
    main()
