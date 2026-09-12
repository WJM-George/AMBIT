#!/usr/bin/env python3
"""Finish request scoring and append the new audio column without rerenders."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def job(command, output, gpu, cap):
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    with (output / 'launcher.log').open('ab') as log:
        p = subprocess.Popen(['timeout', '--kill-after=30s', str(cap), *command],
               stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
               env=env, start_new_session=True)
    save(output / 'LAUNCH.json', {'pid': p.pid, 'command': command, 'gpu': gpu,
         'started_unix': time.time(), 'wall_cap_s': cap})
    if p.wait():
        raise RuntimeError(f'Evaluation child failed: {output / "launcher.log"}')


def main(root):
    lock = (root / 'METRICS_SUPERVISOR_LOCK').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = read(root / 'METRICS_PROTOCOL.json')
    assert cfg['supervisor_sha256'] == sha(__file__)
    for path, digest in cfg['files_sha256'].items():
        assert sha(path) == digest
    freeze = read(root / 'FREEZE.json')
    py = freeze['python']
    request_root = root / 'request_evaluation'
    request_protocol = read(request_root / 'PROTOCOL.json')

    def status(stage, **extra):
        save(root / 'METRICS_STATUS.json', {'status': stage, 'pid': os.getpid(),
             'updated_unix': time.time(), 'test_used': True, 'goal_complete': False, **extra})

    status('WAITING_FOR_NEW_AR_FOA_ONLY')
    while not (root / 'COMPLETE.json').exists():
        state = read(root / 'STATUS.json')
        if str(state['status']).startswith('FAIL'):
            raise RuntimeError('Audio supervisor needs attention before metrics')
        time.sleep(15)
    assert (root / 'AUDIO_MANIFEST.json').exists()
    while True:
        lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                      '--format=csv,noheader,nounits'], text=True).splitlines()
        memory = {int(line.split(',')[0]): int(line.split(',')[1]) for line in lines
                  if int(line.split(',')[0]) in (0, 1, 2)}
        if len(memory) == 3 and max(memory.values()) < 2048:
            break
        status('WAITING_FOR_ASSIGNED_GPU_RELEASE', memory_mib=memory)
        time.sleep(15)
    status('SCORING_FROZEN_TEST_REQUESTS_AND_NEW_AUDIO')

    def request_scoring():
        semantic = request_root / 'semantic'
        pairs = read(semantic / 'pairs.json')
        assert pairs['test_used'] is True
        output = semantic / 'scoring'
        if pairs['pairs']:
            job([py, str(request_root / 'semantic_worker_test.py'), '--judge', request_protocol['judge'],
                 '--pairs', str(semantic / 'pairs.json'), '--output', str(output), '--max-wall-seconds', '2700'],
                 request_root / 'judge_process', '1,2', 3000)
        else:
            import sqlite3
            output.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(output / 'results.sqlite')
            db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
            db.close()
            save(output / 'CONTRACT.json', {'pairs_sha256': sha(semantic / 'pairs.json'), 'test_used': True,
                 'scope': 'Zero semantic pairs: every relevant semantic edge has an exact-text proof. No fabricated judgments.'})
            save(output / 'STATUS.json', {'status': 'COMPLETE', 'pairs': 0})
        job([py, str(request_root / 'evaluate.py'), '--evaluator-snapshot', request_protocol['evaluator_snapshot'],
             'score', '--output', str(semantic), '--report-tag', 'precompletion'],
             request_root / 'score_process', '', 900)
        report_path = semantic / 'REQUEST_SATISFACTION_REPORT.precompletion.json'
        report = read(report_path)
        groups = report['by_candidate']['event_scope_test8k']['natural']
        minima = {'valid': 1., 'count': .95, 'core_precision': .90, 'core_recall': .90,
                  'motion': .95, 'motion_with_direction_constraints': .95, 'start': .95, 'end': .95,
                  'onset': .95, 'offset': .95, 'speech_words': .95,
                  'request_joint_before_completion_review': .80}
        checks = [{'source_count': int(count), 'metric': metric, 'value': group['metrics'][metric]['rate'],
                   'minimum': minimum, 'pass': group['metrics'][metric]['rate'] is not None and
                   group['metrics'][metric]['rate'] + 1e-9 >= minimum}
                  for count, group in groups.items() for metric, minimum in minima.items()]
        save(root / 'TEST_REQUEST_RESULT.json', {'status': 'PASS_TEST_REQUEST_CONSTRAINTS_BEFORE_COMPLETION_AND_AUDIO_REVIEW'
             if all(row['pass'] for row in checks) else 'FAIL_TEST_REQUEST_CONSTRAINTS', 'checks': checks,
             'report': str(report_path), 'test_used': True, 'model_changed_from_test': False, 'goal_complete': False})

    def metric_job(arm, gpu, shard=None):
        command = [py, cfg['adapter'], '--root', str(root), '--arm', arm, '--num-shards', '3']
        if shard is not None:
            command += ['--shard-index', str(shard)]
        job(command, root / 'metric_jobs' / (arm + (f'_{shard}' if shard is not None else '')),
            str(gpu), 5400)

    # While the text judge occupies1/2, score one speech shard on0. Then each
    # card has one metric lane; no baseline generation is scheduled anywhere.
    with ThreadPoolExecutor(max_workers=2) as pool:
        judge = pool.submit(request_scoring)
        speech0 = pool.submit(metric_job, 'speech', 0, 0)
        judge.result()

        def lane(arm, gpu, shard):
            metric_job(arm, gpu)
            metric_job('speech', gpu, shard)

        with ThreadPoolExecutor(max_workers=2) as parallel:
            other = [parallel.submit(lane, 'vggish', 1, 1), parallel.submit(lane, 'panns', 2, 2)]
            speech0.result()
            metric_job('clap', 0)
            for pending in other:
                pending.result()
    status('MERGING_NEW_COLUMN_WITH_UNCHANGED_EXISTING_RESULTS')
    job([py, cfg['adapter'], '--root', str(root), '--arm', 'merge', '--num-shards', '3'],
        root / 'metric_jobs/merge', '', 900)
    assert (root / 'METRICS_COMPLETE.json').exists()
    status('CONTENT_COMPARISON_COMPLETE', report=str(root / 'COMBINED_CONTENT_METRICS.json'))
    save(root / 'EVALUATION_COMPLETE.json', {'status': 'COMPLETE', 'new_audio_rows': 8000,
         'request_result': str(root / 'TEST_REQUEST_RESULT.json'),
         'content_report': str(root / 'COMBINED_CONTENT_METRICS.json'),
         'spatial_demo_delivery_pending': True, 'goal_complete': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root.resolve())
    except BaseException as exc:
        save(args.root / 'METRICS_STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION',
             'error': f'{type(exc).__name__}: {exc}', 'test_used': True, 'goal_complete': False})
        raise
