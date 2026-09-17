"""Continue the unchanged100-step joint OPSD candidate to a500-step review.

Wait for its in-progress200-request evaluation, retain the100-step recovery,
resume the same four ranks/configuration, and audit the next sampled update.
"""
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
from scripts.t2a.rl.launch_editing_opsd_0915_lr import process_active, verify_inputs
from stable_audio_tools.training.transfusion_opsd.top_checkpoints import METRIC_DIRECTIONS, relative_metric_score


def validate_checkpoint(q, state, config_sha):
    if (state['step'] != 100 or state['world_size'] != 4 or
            state['config_sha256'] != config_sha or state['original_checkpoint'] != q['base_checkpoint'] or
            state['initial_overlay'] != q['initial_overlay']):
        raise ValueError('Resume must preserve the identified100-step joint model and configuration.')
    if (q['physical_gpus'] != [4, 5, 6, 7] or q['connected_credit'] or
            q['global_request_batch'] != 16 or q['request_rows_per_rank'] != 4 or
            q['global_paired_batch'] != 512 or q['paired_rows_per_rank'] != 128):
        raise ValueError('Resume requires unchanged four-rank16+512, STE-off execution.')
    execution = {k: q[k] for k in ('request_rows_per_rank', 'global_request_batch',
        'paired_rows_per_rank', 'paired_microbatch', 'global_paired_batch', 'native_plan_audit_every')}
    if state['execution'] != execution or state.get('execution_schedule') != q.get('execution_schedule'):
        raise ValueError('Resume cannot silently change sampling or microbatch execution.')
    groups = {group['group_name']: group['lr'] for group in state['optimizer']['param_groups']}
    if groups != q['learning_rates']:
        raise ValueError('Saved Adam learning rates differ from the current recipe.')
    moments = state['optimizer']['state']
    if not moments or any(int(value['step']) != 100 or
                          not {'exp_avg', 'exp_avg_sq'} <= set(value) for value in moments.values()):
        raise ValueError('The complete100-step Adam moments must be retained.')
    if len(state['rank_states']) != 4:
        raise ValueError('Missing rank recovery state.')
    for rank, local in enumerate(state['rank_states']):
        for key, offset in [('request', 1), ('paired', 2)]:
            stream = local[key]
            if (stream['rank'], stream['world'], stream['seed']) != (rank, 4, q['seed'] + offset):
                raise ValueError('Saved sampler identity differs from this rank.')
        if not {'random', 'numpy', 'cpu_rng', 'cuda_rng'} <= set(local):
            raise ValueError('Missing saved random state.')


def prepare(run):
    p = read(run / 'PROTOCOL.json')
    verify_inputs(p)
    q = read(p['configuration'])
    if (p['limit_updates'] != 500 or p['checkpoint']['step'] != 100 or
            {gpu['index'] for gpu in p['gpu_pool']} != {4, 5, 6, 7} or
            q['physical_gpus'] != [4, 5, 6, 7] or q['maximum_updates'] < 500 or
            q['output'] != p['training_output']):
        raise ValueError('Unexpected continuation boundary or placement.')
    command = [sys.executable, '-m', 'torch.distributed.run',
        '--nnodes=1', '--nproc_per_node=4', '--master_port=29592',
        str(ROOT / 'scripts/t2a/rl/train_editing_opsd_throughput.py'),
        '--performance-directory', str(run / 'performance'),
        '--config', p['configuration'], '--resume', p['checkpoint']['path'],
        '--limit-updates', '500', '--evaluate-at-updates', '125', '150', '200', '250', '500']
    return p, q, command


def audit_first_update(run, p, *, required=False):
    expected = read(p['expected_first_update'])
    observed = []
    for rank in range(4):
        filename = Path(p['training_output']) / f'UPDATES_rank{rank}.jsonl'
        with filename.open() as handle:
            handle.seek(p['update_log_offsets'][str(rank)])
            line = handle.readline()
        if not line.endswith('\n'):
            if required:
                raise ValueError('Missing first resumed update from rank ' + str(rank))
            return False
        value = json.loads(line)
        actual = dict(step=value['step'], request_ordinals=[r['ordinal'] for r in value['request_updates']],
                      paired_ordinals=value['paired_ordinals'])
        if actual != expected['ranks'][str(rank)]:
            raise ValueError('Resume repeated, skipped or changed the saved request/paired stream.')
        observed.append(dict(rank=rank, **actual, clip_norms=value['clip_norms']))
    write(run / 'FIRST_RESUMED_UPDATE_AUDIT.json', dict(step=101, samples_match_saved_cursors=True,
        rank_observations=observed, checkpoint=p['checkpoint'], recipe_and_topology_unchanged=True))
    return True


def report(run, p, checkpoint):
    paths = dict(original40k=p['validation200_baseline'], candidate100=p['validation200_candidate100'],
                 candidate500=str(run / 'validation200/candidate500/EVALUATION_step000500.json'))
    evaluations = {name: read(path) for name, path in paths.items()}
    for value in evaluations.values():
        if value['requests'] != 200 or value['outputs'] != 400 or value['target_audio_in_inference']:
            raise ValueError('Incomplete frozen-panel evaluation.')
    comparisons = {name: relative_metric_score(evaluations[name]['metrics'], evaluations['original40k']['metrics'])
                   for name in ('candidate100', 'candidate500')}
    write(run / 'RESULT.json', dict(scope=p['scope'], checkpoint=checkpoint, evaluation_paths=paths,
        evaluations=evaluations, comparisons=comparisons,
        change_from100=relative_metric_score(evaluations['candidate500']['metrics'], evaluations['candidate100']['metrics']),
        automatic_extension_beyond500=False))
    lines = ['# 当前配方100→500步复核', '', p['scope'], '',
             '| 指标 | Original40k | step100 | step500 | step500相对起点改善 |', '|---|---:|---:|---:|---:|']
    for metric in METRIC_DIRECTIONS:
        values = [evaluations[name]['metrics'][metric] for name in paths]
        lines.append('| ' + metric + ' | ' + ' | '.join(f'{v:.6f}' for v in values) +
                     f" | {comparisons['candidate500']['relative_improvement_percent'][metric]:+.4f}% |")
    lines += ['', 'Same frozen200 requests and two noises; validation, not a blind final test.',
              'The protected100-step optimizer/model remains available. Review stability before extending beyond500.']
    (run / 'TABLE.md').write_text('\n'.join(lines) + '\n')


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
    if (run / 'STATUS.json').exists():
        raise ValueError('Do not overwrite or automatically restart a continuation.')
    workers, monitors, logs, leases = {}, {}, [], []
    status = dict(phase='QUEUED_AFTER_VALIDATION200', pid=os.getpid(), queued_unix=time.time(),
                  physical_gpus=[4, 5, 6, 7], allocated_GPUs=0, start_step=100, limit_updates=500,
                  training_output=p['training_output'], checkpoint=p['checkpoint'], predecessor=p['predecessor'])

    def held():
        return (run / 'USER_HOLD.json').exists() or (Path(p['training_output']) / 'STOP').exists()

    def stage(name, cmd):
        verify_inputs(p)
        if held():
            raise RuntimeError('Continuation paused before launch.')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='4,5,6,7', CUDA_DEVICE_ORDER='PCI_BUS_ID',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            PYTHONUNBUFFERED='1')
        if not monitors:
            monitor_log = (run / 'GPU_UTILIZATION.csv').open('a'); logs.append(monitor_log)
            monitors['gpu_monitor'] = subprocess.Popen(['nvidia-smi', '--id=4,5,6,7',
                '--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw',
                '--format=csv,noheader,nounits', '--loop-ms=2000'], stdin=subprocess.DEVNULL,
                stdout=monitor_log, stderr=subprocess.STDOUT, start_new_session=True)
            status['gpu_monitor_pid'] = monitors['gpu_monitor'].pid
        log = (run / (name + '.log')).open('a'); logs.append(log)
        workers['joint'] = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            pass_fds=tuple(handle.fileno() for handle in leases))
        status.update(phase=name, workers={'joint': workers['joint'].pid}, command=cmd)
        audited = False
        while True:
            code = workers['joint'].poll()
            status.update(observed_unix=time.time(), exit_code=code)
            if name == 'TRAINING_TO500':
                current = read(Path(p['training_output']) / 'STATUS_rank0.json')
                status.update(training_step=current.get('step'), training_phase=current.get('phase'))
                if not audited:
                    audited = audit_first_update(run, p)
                status['first_resumed_samples_verified'] = audited
            write(run / 'STATUS.json', status)
            if code not in (None, 0) or held():
                raise RuntimeError('Continuation failed or was paused; no automatic restart.')
            if code == 0:
                if name == 'TRAINING_TO500':
                    audit_first_update(run, p, required=True)
                break
            time.sleep(10)
        workers.clear()

    try:
        previous = p['predecessor']
        while True:
            prior = read(previous['status'])
            if prior['pid'] != previous['pid'] or prior['phase'] == 'FAILED' or held():
                raise RuntimeError('Predecessor changed/failed, or continuation paused.')
            if not process_active(previous['pid'], previous['start_ticks']):
                if prior['phase'] != 'COMPLETE':
                    raise RuntimeError('Predecessor exited before its prescribed validation completed.')
                break
            status.update(observed_unix=time.time(), predecessor_phase=prior['phase'])
            write(run / 'STATUS.json', status)
            time.sleep(15)
        verify_inputs(p)
        if sha(p['checkpoint']['path']) != p['checkpoint']['sha256']:
            raise ValueError('Protected100-step checkpoint changed.')
        if read(Path(p['training_output']) / 'RESUME.json')['step'] != 100:
            raise ValueError('The training directory has advanced outside this continuation.')
        previous_result = read(p['predecessor_result'])
        write(run / 'PRECEDING_VALIDATION200.json', previous_result)
        status['preceding_validation200_comparison'] = previous_result['comparison']
        leases = acquire_leases(p['gpu_pool'])
        status.update(started_unix=time.time(), allocated_GPUs=4)
        stage('TRAINING_TO500', command)
        checkpoint_path = Path(p['training_output']) / 'checkpoints/step-00000500.pt'
        if read(Path(p['training_output']) / 'RESUME.json')['step'] != 500:
            raise ValueError('Continuation stopped before its500-step review boundary.')
        checkpoint = dict(path=str(checkpoint_path), sha256=sha(checkpoint_path), step=500)
        write(run / 'EVALUATION_CHECKPOINT.json', checkpoint)
        evaluation = [sys.executable, '-m', 'torch.distributed.run',
            '--nnodes=1', '--nproc_per_node=4', '--master_port=29593',
            str(ROOT / 'scripts/t2a/rl/evaluate_editing_opsd_branch_cross.py'),
            '--config', p['validation200_config'], '--checkpoint', str(checkpoint_path),
            '--checkpoint-sha256', checkpoint['sha256'], '--step', '500', '--planner', 'new', '--executor', 'new',
            '--output', str(run / 'validation200/candidate500'), '--fresh-matched-panel']
        stage('VALIDATION200_AT500', evaluation)
        report(run, p, checkpoint)
        status.update(phase='COMPLETE', completed_unix=time.time(), allocated_GPUs=0,
            allocated_GPU_hours=(time.time() - status['started_unix']) * 4 / 3600,
            result=str(run / 'RESULT.json'), automatic_extension_beyond500=False)
        write(run / 'STATUS.json', status)
    except BaseException as exc:
        stop_owned(workers)
        status.update(phase='FAILED', observed_unix=time.time(), error=repr(exc))
        write(run / 'STATUS.json', status)
        raise
    finally:
        stop_owned(monitors)
        for handle in logs + leases:
            handle.close()


if __name__ == '__main__':
    main()
