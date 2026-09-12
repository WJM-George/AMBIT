#!/usr/bin/env python3
"""Resume frozen semantic scoring of the copy pilot after T100 releases GPUs1/2."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import time

PYTHON = '/mnt/sdc/stable-audio-tools-workspace/.venv/bin/python3'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def main(root):
    root = root.resolve(); base = root.parent / 'template100_rebuild_20260905_v1'
    judge = root.parent / 'semantic_judge_20260905_v1'; snap = root / 'evaluation_source_snapshot'
    semantic = root / 'semantic'; scoring = semantic / 'scoring'
    def status(name, **kw):
        atomic(semantic / 'STATUS.json', {'status': name, 'pid': os.getpid(), 'updated_unix': time.time(),
            'test_used': False, 'goal_complete': False, **kw})
    for rel, digest in json.loads((snap / 'MANIFEST.json').read_text())['files'].items():
        assert sha(snap / rel) == digest, rel
    status('WAITING_T100_VALIDATION_GPU12_RELEASE')
    deadline = time.monotonic() + 21600
    while True:
        if (base / 'validation_after_epoch/RESULT.json').exists():
            usage = subprocess.check_output(['nvidia-smi', '--id=1,2', '--query-gpu=memory.used',
                '--format=csv,noheader,nounits'], text=True, timeout=15)
            if all(int(value) < 128 for value in usage.split()): break
        if time.monotonic() > deadline: raise TimeoutError('Six-hour T100 validation readiness window exceeded')
        time.sleep(30)
    scoring.mkdir(exist_ok=True)
    db = sqlite3.connect(scoring / 'results.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    pairs = {row['id']: row for row in json.loads((semantic / 'pairs.json').read_text())['pairs']}
    receipts = []
    worker = snap / 'scripts/t2a/diagnostics/evaluate_generation_ar_semantic_pairs.py'
    for name in ('semantic_full', 'semantic_baseline'):
        source = base / 'validation_after_epoch' / name / 'scoring'
        contract = json.loads((source / 'CONTRACT.json').read_text())
        assert json.loads((source / 'STATUS.json').read_text())['status'] == 'COMPLETE'
        assert contract['judge_contract_sha256'] == sha(judge / 'CONTRACT.json')
        assert contract['script_sha256'] == sha(worker)
        source_db = sqlite3.connect('file:' + str(source / 'results.sqlite') + '?mode=ro&immutable=1', uri=True)
        reused = 0
        for sid, payload in source_db.execute('SELECT id,payload FROM results'):
            if sid not in pairs: continue
            value = json.loads(payload)
            assert all(value[key] == pairs[sid][key] for key in ('id', 'kind', 'reference', 'candidate'))
            previous = db.execute('SELECT payload FROM results WHERE id=?', (sid,)).fetchone()
            if previous: assert json.loads(previous[0]) == value
            else: db.execute('INSERT INTO results VALUES (?,?)', (sid, payload)); reused += 1
        source_db.close()
        receipts.append({'source_db': str(source / 'results.sqlite'), 'sha256': sha(source / 'results.sqlite'),
            'source_contract_sha256': sha(source / 'CONTRACT.json'), 'newly_reused': reused})
    db.commit(); cached = db.execute('SELECT COUNT(*) FROM results').fetchone()[0]; db.close()
    atomic(semantic / 'CACHE_RECEIPT.json', {'sources': receipts, 'total': len(pairs), 'cached': cached,
        'judge_contract_sha256': sha(judge / 'CONTRACT.json'), 'pairs_sha256': sha(semantic / 'pairs.json')})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1,2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    def run(command, gpu, log_path, cap):
        command = list(map(str, command)); child_env = dict(env, CUDA_VISIBLE_DEVICES=gpu)
        with log_path.open('ab') as log:
            child = subprocess.Popen(['timeout', '--kill-after=20s', str(cap), *command], env=child_env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        atomic(log_path.with_suffix('.launch.json'), {'pid': child.pid, 'command': command,
            'started_unix': time.time(), 'gpu': gpu, 'wall_cap_s': cap})
        rc = child.wait()
        if rc: raise RuntimeError(f'Child exited {rc}; inspect {log_path}')
    status('JUDGING_UNRESOLVED_SEMANTIC_PAIRS', cached=cached, pairs=len(pairs))
    # The unchanged worker verifies all cached pairs and seals its ordinary
    # calibration/model/threshold contract. Its load is bounded even if all
    # pairs happen to be reused.
    run([PYTHON, worker, '--judge', judge, '--pairs', semantic / 'pairs.json', '--output', scoring,
        '--max-wall-seconds', '1200'], '1,2', scoring / 'launcher.log', 1380)
    status('SCORING_COUPLED_REQUEST_CONSTRAINTS')
    scorer = snap / 'scripts/t2a/diagnostics/evaluate_generation_ar_natural_candidates.py'
    run([PYTHON, scorer, 'score', '--output', semantic, '--report-tag', 'precompletion'], '',
        semantic / 'score.log', 600)
    report = json.loads((semantic / 'REQUEST_SATISFACTION_REPORT.precompletion.json').read_text())
    protocol = json.loads((root / 'PROTOCOL.json').read_text()); rules = protocol['pilot_success']
    checks = []; comparison = {}
    for n in map(str, range(1, 5)):
        current = report['by_candidate']['t100_copy']['natural'][n]
        prior = report['by_candidate']['t100_base']['natural'][n]
        comparison[n] = {'t100_copy': current, 't100_base': prior}
        for metric, threshold in [('valid', rules['valid_rate_each_class']),
                ('count', rules['count_accuracy_each_class']), ('speech_words', rules['raw_transcript_exact_rate_each_class'])]:
            value = current['metrics'][metric]['rate']
            checks.append({'count': int(n), 'metric': metric, 'value': value,
                'minimum': threshold, 'pass': value is not None and value >= threshold - 1e-9})
        for metric in ('count', 'motion', 'motion_with_direction_constraints', 'start', 'end', 'onset', 'offset'):
            value = current['metrics'][metric]['rate']; old = prior['metrics'][metric]['rate']
            maximum_drop = rules['maximum_paired_count_drop'] if metric == 'count' else rules['maximum_requested_motion_compass_time_metric_drop']
            if value is not None and old is not None:
                checks.append({'count': int(n), 'metric': metric + '_retention', 'delta': value - old,
                    'minimum': -maximum_drop, 'pass': value - old >= -maximum_drop - 1e-9})
    pending = any(comparison[n]['t100_copy']['pending_semantics'] for n in comparison)
    assert not pending
    passed = all(check['pass'] for check in checks)
    result = {'status': 'PASS_COPY_PILOT_ONLY' if passed else 'FAIL_COPY_PILOT_RULE', 'checks': checks,
        'comparison_by_source_count': comparison, 'test_used': False, 'goal_complete': False, 'promoted': False,
        'checkpoint_sha256': sha(root / 'candidate.pt'), 'protocol_sha256': sha(root / 'PROTOCOL.json'),
        'raw_prediction_sha256': sha(root / 'raw_validation/predictions.sqlite'),
        'full_report': str(semantic / 'REQUEST_SATISFACTION_REPORT.precompletion.json'),
        'completion_and_audio': 'Separate review remains required, irrespective of this small pilot decision.'}
    atomic(root / 'PILOT_RESULT.json', result); status(result['status'], result=str(root / 'PILOT_RESULT.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try: main(args.root)
    except BaseException as exc:
        atomic(args.root / 'semantic/STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION',
            'error': f'{type(exc).__name__}: {exc}', 'updated_unix': time.time()})
        raise
