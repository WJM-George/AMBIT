#!/usr/bin/env python3
"""Resume timed-out T100 validation without stopping healthy GPU jobs.

The original frozen workers, prompts, batches and per-invocation budgets remain
unchanged. Completed raw jobs are verified before reuse; partial workers resume
their committed SQLite rows. Only timeout failures receive a bounded retry.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time


def read(path): return json.loads(path.read_text())


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def alive(launch):
    proc = Path('/proc') / str(launch['pid']) / 'cmdline'
    return proc.exists() and str(launch['command'][1]).encode() in proc.read_bytes()


def wait_exit(launch, deadline):
    while alive(launch):
        if time.time() > deadline: raise TimeoutError('Recovery adoption deadline exceeded')
        time.sleep(5)


def timeout_failure(output):
    status = read(output / 'STATUS.json') if (output / 'STATUS.json').exists() else {}
    return status.get('status', '').startswith('FAILED') and 'wall budget exceeded' in status.get('error', '')


def archive(output, label):
    dest = output / 'recovery_history' / label; dest.mkdir(parents=True, exist_ok=False)
    for name in ('STATUS.json', 'LAUNCH.json', 'CONTRACT.json', 'SUMMARY.json'):
        if (output / name).exists(): shutil.copy2(output / name, dest / name)
    return dest


def main(root):
    recovery = root / 'validation_recovery'; recovery.mkdir(exist_ok=True)
    lock = (recovery / 'LOCK').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    deadline = time.time() + 18000
    snapshot = root / 'evaluation_source_snapshot'
    driver = load_module('frozen_t100_validation_driver', snapshot / 'scripts/t2a/diagnostics/evaluate_generation_ar_template100.py')
    original_run = driver.command_run
    cache = load_module('verified_raw_cache', root / 'p10_validation16/source_snapshot/stable_audio_tools/inference/sceneplan_generation_ar_foa_resume.py')
    initial_pipeline = read(root / 'PIPELINE_LAUNCH.json')
    initial_driver = {'pid': read(root / 'STATUS.json')['child_pid'],
        'command': [driver.PYTHON, str(snapshot / 'scripts/t2a/diagnostics/evaluate_generation_ar_template100.py')]}
    output = root / 'validation_after_epoch'

    def state(name, **kw):
        atomic(recovery / 'STATUS.json', {'status': name, 'pid': os.getpid(), 'updated_unix': time.time(),
            'test_used': False, 'goal_complete': False, **kw})
        print(json.dumps({'event': name, **kw}), flush=True)

    def verified_complete(command, where):
        if not (where / 'STATUS.json').exists() or read(where / 'STATUS.json').get('status') != 'COMPLETE': return False
        argv = list(map(str, command)); options = dict(zip(argv[2::2], argv[3::2]))
        worker = Path(argv[1]).name
        if worker == 'generate_sceneplan_from_raw_english.py':
            requests = Path(options['--requests']); checkpoint = Path(options['--checkpoint']); snap = Path(options['--snapshot'])
            contract = read(where / 'CONTRACT.json')
            assert contract['batch_size'] == int(options['--batch-size'])
            assert contract['max_plan_tokens'] == int(options['--max-plan-tokens'])
            assert contract['wall_cap_s'] == int(options['--max-wall-seconds'])
            return cache.read_raw_results(where, requests=read(requests)['requests'], requests_sha256=sha(requests),
                checkpoint_sha256=sha(checkpoint), snapshot_manifest_sha256=sha(snap / 'SOURCE_SNAPSHOT_MANIFEST.json'),
                entry_sha256=sha(Path(argv[1]))) is not None
        if worker == 'evaluate_generation_ar_semantic_pairs.py':
            pairs = Path(options['--pairs']); judge = Path(options['--judge']); contract = read(where / 'CONTRACT.json')
            assert contract['pairs_sha256'] == sha(pairs) and contract['judge_contract_sha256'] == sha(judge / 'CONTRACT.json')
            assert contract['script_sha256'] == sha(Path(argv[1]))
            assert contract['wall_cap_s'] == int(options['--max-wall-seconds'])
            expected = {row['id']: row for row in read(pairs)['pairs']}
            db = sqlite3.connect('file:' + str(where / 'results.sqlite') + '?mode=ro&immutable=1', uri=True)
            rows = {i: json.loads(v) for i, v in db.execute('SELECT id,payload FROM results')}; db.close()
            assert rows.keys() == expected.keys()
            assert all(all(row[k] == expected[i][k] for k in ('id', 'kind', 'reference', 'candidate')) for i, row in rows.items())
            return True
        return False

    def bounded_run(command, gpu, where, cap):
        if verified_complete(command, where):
            print(json.dumps({'event': 'VERIFIED_CACHE_REUSED', 'output': str(where)}), flush=True); return
        # A full retry here is only for a timeout. Each original invocation
        # retains its own budget; completed rows are never regenerated.
        # An already timed-out original worker gets one additional segment;
        # a not-yet-started worker may get two segments in total.
        attempts = 1 if timeout_failure(where) else 2
        for attempt in range(attempts):
            if attempt:
                if not timeout_failure(where): raise RuntimeError('Non-timeout failure requires diagnosis: ' + str(where))
                archive(where, f'bounded_retry_{int(time.time())}')
            try: original_run(command, gpu, where, cap); return
            except RuntimeError:
                if attempt + 1 == attempts or not timeout_failure(where): raise

    def recover_baseline():
        baseline = output / 'baseline_raw'
        launch = read(baseline / 'LAUNCH.json'); wait_exit(launch, deadline)
        if not verified_complete(launch['command'], baseline):
            if not timeout_failure(baseline): raise RuntimeError('Baseline ended with a non-timeout failure')
            archive(baseline, 'initial_timeout')
            state('RESUMING_BASELINE_COMMITTED_ROWS')
            original_run(launch['command'], launch['gpu'], baseline, launch['wall_cap_s'])
            assert verified_complete(launch['command'], baseline)
        # Its original driver may already have left this future on timeout.
        # Prepare its semantics now, while GPU1/2 continue their existing job.
        sem = output / 'semantic_baseline'
        original_run([driver.PYTHON, snapshot / 'scripts/t2a/diagnostics/evaluate_generation_ar_natural_candidates.py',
            'prepare', '--validation', root / 'validation_inputs/baseline_panel_pairs.json',
            '--candidate', 'r3_baseline=' + str(baseline / 'predictions.sqlite'), '--output', sem], '', sem / 'preparation', 300)
        audio = root / 'p10_validation16'; status = read(audio / 'STATUS.json')
        if status.get('status', '').startswith('FAILED') and 'R3 baseline failed' in status.get('error', ''):
            old = read(audio / 'LAUNCH.json'); wait_exit(old, deadline)
            archive(audio, 'baseline_timeout_waiter')
            command = [driver.PYTHON, str(audio / 'wait_and_validate.py')]
            with (audio / 'supervisor.log').open('ab') as log:
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic(audio / 'LAUNCH.json', {'pid': proc.pid, 'command': command, 'started_unix': time.time(),
                'gpu_scope': [0], 'reason': 'Baseline resumed to verified completion; same frozen P10 request panel and inference.'})
        state('BASELINE_READY_P10_QUEUE_RELEASED')

    state('WATCHING_EXISTING_JOBS', recovery_policy='Timeout-only; identical frozen workers; reuse complete outputs and partial SQLite rows.')
    with ThreadPoolExecutor(max_workers=1) as pool:
        baseline_job = pool.submit(recover_baseline)
        wait_exit(initial_driver, deadline)
        wait_exit(initial_pipeline, deadline)
        baseline_job.result()
    if (output / 'RESULT.json').exists():
        state('ORIGINAL_VALIDATION_COMPLETE_NO_MAIN_RETRY'); return
    # Both original supervisors have exited. Preserve their terminal receipts
    # before taking over, so no two drivers write the same state concurrently.
    archive(root, 'initial_validation_driver_timeout')
    shutil.copy2(root / 'PIPELINE_LAUNCH.json', recovery / 'INITIAL_PIPELINE_LAUNCH.json')
    archive(output, 'initial_validation_driver_timeout')
    atomic(root / 'PIPELINE_LAUNCH.json', {'pid': os.getpid(), 'command': [sys.executable, str(Path(__file__).resolve()), '--root', str(root)],
        'started_unix': time.time(), 'recovery': str(recovery), 'training_unchanged': True})
    atomic(root / 'STATUS.json', {'status': 'RUNNING', 'stage': 'validation', 'child_pid': os.getpid(),
        'started_unix': time.time(), 'recovery': str(recovery)})
    state('RESUMING_VALIDATION_AFTER_TIMEOUT')
    driver.command_run = bounded_run
    driver.main(root)
    atomic(root / 'STATUS.json', {'status': 'VALIDATION_STAGE_COMPLETE', 'stage': 'validation',
        'result': str(output / 'RESULT.json'), 'goal_complete': False, 'recovered_committed_results': True})
    state('VALIDATION_RECOVERY_COMPLETE', result=str(output / 'RESULT.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True); args = parser.parse_args()
    try: main(args.root)
    except BaseException as exc:
        recovery = args.root / 'validation_recovery'; recovery.mkdir(exist_ok=True)
        atomic(recovery / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}', 'updated_unix': time.time()})
        raise
