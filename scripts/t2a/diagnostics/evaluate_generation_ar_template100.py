#!/usr/bin/env python3
"""Full 32K raw validation plus a fixed 1K R3 comparison, after T100 training."""
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.t2a.diagnostics.evaluate_generation_ar_natural_candidates import metrics
PYTHON = os.environ.get("AMBIT_PYTHON", "python3")
JUDGE = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/judge")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temp.replace(path)


def read_rows(path):
    db = sqlite3.connect('file:' + str(path) + '?mode=ro&immutable=1', uri=True)
    rows = {i: json.loads(v) for i, v in db.execute('SELECT id,payload FROM results')}; db.close(); return rows


def command_run(command, gpu, output, cap):
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    # GNU timeout supervises only this newly created process group.
    with (output / 'launcher.log').open('ab') as log:
        p = subprocess.Popen(['timeout', '--kill-after=20s', str(cap), *map(str, command)], env=env,
            cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    atomic(output / 'LAUNCH.json', {'pid': p.pid, 'command': list(map(str, command)), 'gpu': gpu, 'wall_cap_s': cap, 'started_unix': time.time()})
    rc = p.wait()
    if rc: raise RuntimeError(f'Child exited {rc}: {output / "launcher.log"}')


def main(root):
    root = root.resolve(); inputs = root / 'validation_inputs'; output = root / 'validation_after_epoch'; output.mkdir(exist_ok=True)
    snap = root / 'training_source_snapshot'; config = json.loads((root / 'TRAINING_CONFIG.json').read_text())
    initial = Path(config['initialize_checkpoint']); final = root / 'training/checkpoints/step_00008334.pt'
    assert json.loads((root / 'training/STATUS.json').read_text())['status'] == 'COMPLETE'
    frozen = json.loads((REPO / 'T100_EVALUATION_SNAPSHOT.json').read_text())
    for rel, digest in frozen['files'].items(): assert sha(REPO / rel) == digest
    manifest = json.loads((inputs / 'MANIFEST.json').read_text())
    for rel, digest in manifest['files'].items(): assert sha(inputs / rel) == digest
    inference = snap / 'scripts/t2a/inference/generate_sceneplan_from_raw_english.py'
    scorer = REPO / 'scripts/t2a/diagnostics/evaluate_generation_ar_natural_candidates.py'
    worker = REPO / 'scripts/t2a/diagnostics/evaluate_generation_ar_semantic_pairs.py'
    def state(status, **kw): atomic(output / 'STATUS.json', {'status': status, 'updated_unix': time.time(), 'test_used': False, 'goal_complete': False, **kw})
    def generate(gpu, requests, checkpoint, where, wall=7200):
        command_run([PYTHON, inference, '--snapshot', snap, '--checkpoint', checkpoint, '--requests', requests,
            '--output', where, '--batch-size', '32', '--max-plan-tokens', '512', '--max-wall-seconds', str(wall)], str(gpu), where, wall + 180)
    state('FULL_VALIDATION_RAW_RUNNING')
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(generate, g, inputs / f'validation_requests_gpu{g}.json', final, output / f'raw_gpu{g}') for g in range(3)]
        for job in jobs: job.result()
    merged = output / 'full_raw'; merged.mkdir(exist_ok=True)
    all_rows = {}
    for g in range(3):
        rows = read_rows(output / f'raw_gpu{g}/predictions.sqlite'); assert not (set(rows) & set(all_rows)); all_rows.update(rows)
    refs = {r['id']: r for r in json.loads((inputs / 'validation_pairs.json').read_text())['pairs']}
    assert set(all_rows) == set(refs) and len(all_rows) == 32000
    db = sqlite3.connect(merged / 'predictions.sqlite'); db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    if db.execute('SELECT COUNT(*) FROM results').fetchone()[0] == 0:
        db.executemany('INSERT INTO results VALUES (?,?)', [(k, json.dumps(v, ensure_ascii=False)) for k, v in all_rows.items()]); db.commit()
    assert db.execute('SELECT COUNT(*) FROM results').fetchone()[0] == 32000; db.close()
    atomic(merged / 'CONTRACT.json', {'schema': 'generation_ar_disjoint_raw_merge_v1', 'checkpoint': str(final), 'checkpoint_sha256': sha(final),
        'shard_contracts': {str(g): sha(output / f'raw_gpu{g}/CONTRACT.json') for g in range(3)}, 'raw_only': True, 'test_used': False})
    atomic(merged / 'STATUS.json', {'status': 'COMPLETE', 'rows': 32000})
    counts = {}
    for n in range(1, 5):
        part = [all_rows[k] for k, r in refs.items() if r['source_count'] == n]
        correct = sum(r['prediction'] is not None and len(r['prediction']['sources']) == n for r in part)
        counts[str(n)] = {'correct': correct, 'total': len(part), 'rate': correct / len(part)}
    atomic(output / 'RAW_COUNT_REPORT.json', {'status': 'COMPLETE', 'by_source_count': counts, 'test_used': False, 'core_and_constraints': 'PENDING'})
    semantic = output / 'semantic_full'
    command_run([PYTHON, scorer, 'prepare', '--validation', inputs / 'validation_pairs.json', '--candidate', 't100_full=' + str(merged / 'predictions.sqlite'), '--output', semantic], '', semantic / 'preparation', 600)
    state('FULL_SEMANTIC_JUDGE_AND_BASELINE_RAW_RUNNING')
    baseline = output / 'baseline_raw'; baseline_sem = output / 'semantic_baseline'
    def baseline_job():
        generate(0, inputs / 'baseline_panel_requests.json', initial, baseline, 1200)
        command_run([PYTHON, scorer, 'prepare', '--validation', inputs / 'baseline_panel_pairs.json', '--candidate', 'r3_baseline=' + str(baseline / 'predictions.sqlite'), '--output', baseline_sem], '', baseline_sem / 'preparation', 300)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(baseline_job), pool.submit(command_run,
            [PYTHON, worker, '--judge', JUDGE, '--pairs', semantic / 'pairs.json', '--output', semantic / 'scoring', '--max-wall-seconds', '5400'],
            '1,2', semantic / 'scoring', 5580)]
        for job in jobs: job.result()
    # Reuse only exact pairs already scored by the same frozen calibrated judge.
    cache = read_rows(semantic / 'scoring/results.sqlite')
    pairs = {r['id']: r for r in json.loads((baseline_sem / 'pairs.json').read_text())['pairs']}
    scoring = baseline_sem / 'scoring'; scoring.mkdir(exist_ok=True)
    db = sqlite3.connect(scoring / 'results.sqlite'); db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    reused = 0
    for pid in sorted(set(cache) & set(pairs)):
        row = cache[pid]
        assert all(row[k] == pairs[pid][k] for k in ('id', 'kind', 'reference', 'candidate'))
        db.execute('INSERT OR IGNORE INTO results VALUES (?,?)', (pid, json.dumps(row, ensure_ascii=False))); reused += 1
    db.commit(); db.close()
    atomic(baseline_sem / 'CACHE_RECEIPT.json', {'source_db_sha256': sha(semantic / 'scoring/results.sqlite'),
        'source_judge_contract': sha(JUDGE / 'CONTRACT.json'), 'worker_sha256': sha(worker), 'reused': reused, 'total': len(pairs)})
    state('BASELINE_SEMANTIC_JUDGE_RUNNING', cached_pairs=reused, total_pairs=len(pairs))
    command_run([PYTHON, worker, '--judge', JUDGE, '--pairs', baseline_sem / 'pairs.json', '--output', scoring, '--max-wall-seconds', '1200'], '1,2', scoring, 1380)
    state('REQUEST_CONSTRAINT_SCORING')
    for where in (semantic, baseline_sem):
        command_run([PYTHON, scorer, 'score', '--output', where, '--report-tag', 'precompletion'], '', where / 'precompletion', 600)
    full_scores = json.loads((semantic / 'SCENE_SCORES.precompletion.json').read_text())['rows']
    baseline_scores = json.loads((baseline_sem / 'SCENE_SCORES.precompletion.json').read_text())['rows']
    baseline_ids = {r['id'] for r in baseline_scores}; comparison = {}
    for n in range(1, 5):
        before = [r for r in baseline_scores if r['scored']['requested_count'] == n]
        after = [r for r in full_scores if r['id'] in baseline_ids and r['scored']['requested_count'] == n]
        assert len(before) == len(after) == 256
        comparison[str(n)] = {'r3_initializer_raw': metrics(before), 't100_full_epoch_raw': metrics(after)}
    report = {'status': 'REQUEST_METRICS_COMPLETE_COMPLETION_AND_AUDIO_PENDING', 'test_used': False, 'goal_complete': False,
        'full_validation_rows': 32000, 'matched_baseline_panel_rows': 1024, 'comparison_by_source_count': comparison,
        'full_report': str(semantic / 'REQUEST_SATISFACTION_REPORT.precompletion.json'),
        'full_checkpoint': str(final), 'full_checkpoint_sha256': sha(final), 'raw_input_only': True,
        'completion_review': 'Free numerical values are not compared to GT. Acceptance stays pending where reasonable-completion review is unresolved.',
        'scope': '100/10/5 qualitative template family, not unrestricted English generalization'}
    atomic(output / 'RESULT.json', report); state(report['status'], result=str(output / 'RESULT.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True); args = parser.parse_args()
    try: main(args.root)
    except BaseException as exc:
        out = args.root / 'validation_after_epoch'; out.mkdir(exist_ok=True)
        atomic(out / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}', 'updated_unix': time.time()})
        raise
