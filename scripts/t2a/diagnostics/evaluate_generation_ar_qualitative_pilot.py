#!/usr/bin/env python3
"""Actual raw generation and coupled scoring for the selected qualitative head."""
from concurrent.futures import ThreadPoolExecutor
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
    temp = path.with_name(path.name + '.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def rows(path):
    db = sqlite3.connect('file:' + str(path) + '?mode=ro&immutable=1', uri=True)
    result = {sid: json.loads(payload) for sid, payload in db.execute('SELECT id,payload FROM results')}; db.close(); return result


def main(root, existing_shards=0, extra_candidates=()):
    root = root.resolve(); ar = root.parent; base = ar / 'template100_rebuild_20260905_v1'; copy = ar / 'learned_copy_pointer_20260906_v1'
    output = root / 'raw_evaluation'; output.mkdir(exist_ok=True)
    protocol = json.loads((root / 'RAW_VALIDATION_PROTOCOL.json').read_text())
    assert sha(root / 'candidate.pt') == protocol['checkpoint_sha256']
    inference = root / 'inference_source_snapshot'
    for rel, digest in json.loads((inference / 'MANIFEST.json').read_text())['files'].items(): assert sha(inference / rel) == digest
    evaluator = copy / 'evaluation_source_snapshot'
    for rel, digest in json.loads((evaluator / 'MANIFEST.json').read_text())['files'].items(): assert sha(evaluator / rel) == digest
    scorer = evaluator / 'scripts/t2a/diagnostics/evaluate_generation_ar_natural_candidates.py'
    worker = evaluator / 'scripts/t2a/diagnostics/evaluate_generation_ar_semantic_pairs.py'
    def status(name, **kw):
        atomic(output / 'STATUS.json', {'status': name, 'pid': os.getpid(), 'updated_unix': time.time(), 'test_used': False, 'goal_complete': False, **kw})
    def run(command, gpu, folder, cap):
        folder.mkdir(parents=True, exist_ok=True); command = list(map(str, command))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
        with (folder / 'launcher.log').open('ab') as log:
            child = subprocess.Popen(['timeout', '--kill-after=20s', str(cap), *command], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        atomic(folder / 'LAUNCH.json', {'pid': child.pid, 'command': command, 'started_unix': time.time(), 'gpu': gpu, 'wall_cap_s': cap})
        rc = child.wait()
        if rc: raise RuntimeError(f'Child exited {rc}: {folder / "launcher.log"}')
    status('WAITING_FOR_EXISTING_RAW_GENERATION' if existing_shards else 'RAW_GENERATION_GPU0_1_2')
    def generate(gpu):
        request_path = root / 'raw_inputs' / f'gpu{gpu}.json'
        assert sha(request_path) == protocol['raw_shards_sha256'][str(gpu)]
        run([PYTHON, inference / 'scripts/t2a/inference/generate_sceneplan_with_learned_copy.py', '--snapshot', base / 'training_source_snapshot',
            '--checkpoint', root / 'candidate.pt', '--requests', request_path, '--output', output / f'raw_gpu{gpu}',
            '--gate-requests', base / 'p10_validation16/requests.json', '--batch-size', '32', '--max-plan-tokens', '512', '--max-wall-seconds', '600'],
            str(gpu), output / f'raw_gpu{gpu}', 780)
    shard_count = existing_shards or 3
    extra = {}
    for value in extra_candidates:
        name, path = value.split('=', 1)
        assert name not in ('qualitative_execution', 'copy_v2') and name not in extra
        extra[name] = Path(path).resolve()
    if existing_shards:
        pending_folders = [output / f'raw_gpu{gpu}' for gpu in range(shard_count)] + [p.parent for p in extra.values()]
        started_wait = time.monotonic()
        while pending_folders:
            remaining = []
            for folder in pending_folders:
                saved = json.loads((folder / 'STATUS.json').read_text()) if (folder / 'STATUS.json').exists() else {}
                if saved.get('status') == 'COMPLETE':
                    continue
                if saved.get('status', '').startswith('FAILED'):
                    raise RuntimeError(f'Raw generation failed: {folder}: {saved}')
                launch = json.loads((folder / 'LAUNCH.json').read_text())
                cmdline = Path('/proc') / str(launch['pid']) / 'cmdline'
                if not cmdline.exists() or str(launch['command'][1]).encode() not in cmdline.read_bytes():
                    raise RuntimeError(f'Raw process disappeared before completion: {folder}')
                remaining.append(folder)
            if time.monotonic() - started_wait > 1800:
                raise TimeoutError('Existing raw generation did not finish in its collection budget')
            pending_folders = remaining
            if pending_folders: time.sleep(15)
    else:
        assert not extra, 'Extra candidates require the completed-shard collection path'
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(generate, gpu) for gpu in range(3)]
            for future in futures: future.result()
    merged = output / 'raw_validation'; merged.mkdir(exist_ok=True); predictions = {}
    for gpu in range(shard_count):
        folder = output / f'raw_gpu{gpu}'; assert json.loads((folder / 'STATUS.json').read_text())['status'] == 'COMPLETE'
        shard = rows(folder / 'predictions.sqlite'); assert set(shard).isdisjoint(predictions); predictions.update(shard)
    references = json.loads((base / 'validation_inputs/baseline_panel_pairs.json').read_text())['pairs']
    assert len(predictions) == 1024 and set(predictions) == {row['id'] for row in references}
    for ref in references:
        row = predictions[ref['id']]; assert row['request'] == ref['request']
        assert row['model_input_sha256'] == hashlib.sha256(row['request'].encode()).hexdigest()
    db = sqlite3.connect(merged / 'predictions.sqlite'); db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    if db.execute('SELECT COUNT(*) FROM results').fetchone()[0] == 0:
        db.executemany('INSERT INTO results VALUES (?,?)', [(key, json.dumps(value, ensure_ascii=False)) for key, value in predictions.items()]); db.commit()
    db.close(); assert rows(merged / 'predictions.sqlite') == predictions
    atomic(merged / 'CONTRACT.json', {'schema': 'generation_ar_disjoint_raw_merge_v1', 'checkpoint_sha256': protocol['checkpoint_sha256'],
        'shard_contract_sha256': {str(gpu): sha(output / f'raw_gpu{gpu}/CONTRACT.json') for gpu in range(shard_count)},
        'raw_protocol_sha256': sha(root / 'RAW_VALIDATION_PROTOCOL.json'), 'raw_only': True, 'test_used': False})
    summary = {'status': 'COMPLETE', 'rows': len(predictions), 'parsed': sum(row['prediction'] is not None for row in predictions.values()),
        'primary_copy_eos_failures': sum(row.get('decode_recovery') is not None for row in predictions.values()), 'test_used': False}
    atomic(merged / 'SUMMARY.json', summary); atomic(merged / 'STATUS.json', {'status': 'COMPLETE', 'summary': str(merged / 'SUMMARY.json')})
    semantic = output / 'semantic'; status('PREPARING_COUPLED_REQUEST_SCORING')
    extra_arguments = [item for name, path in extra.items() for item in ('--candidate', name + '=' + str(path))]
    run([PYTHON, scorer, 'prepare', '--validation', base / 'validation_inputs/baseline_panel_pairs.json',
        '--candidate', 'qualitative_execution=' + str(merged / 'predictions.sqlite'),
        '--candidate', 'copy_v2=' + str(copy / 'raw_validation_v2/predictions.sqlite'),
        *extra_arguments, '--output', semantic, '--prove-exact-satisfied'], '', semantic / 'preparation', 300)
    pairs = {row['id']: row for row in json.loads((semantic / 'pairs.json').read_text())['pairs']}
    scoring = semantic / 'scoring'; scoring.mkdir(exist_ok=True); cache = {}; receipts = []
    judge = ar / 'semantic_judge_20260905_v1'; judge_sha = sha(judge / 'CONTRACT.json')
    for source in [copy / 'semantic/scoring', base / 'validation_after_epoch/semantic_full/scoring',
            ar / 'learned_qualitative_field_attention_20260906_v1/raw_evaluation/semantic/scoring']:
        if source.resolve() == scoring.resolve() or not (source / 'CONTRACT.json').exists(): continue
        contract = json.loads((source / 'CONTRACT.json').read_text())
        assert contract['judge_contract_sha256'] == judge_sha and contract['script_sha256'] == sha(worker)
        assert json.loads((source / 'STATUS.json').read_text())['status'] == 'COMPLETE'
        source_rows = rows(source / 'results.sqlite'); added = 0
        for sid in set(pairs) & set(source_rows):
            value = source_rows[sid]; assert all(value[key] == pairs[sid][key] for key in ('id', 'kind', 'reference', 'candidate'))
            if sid in cache: assert cache[sid] == value
            else: cache[sid] = value; added += 1
        receipts.append({'source': str(source), 'db_sha256': sha(source / 'results.sqlite'), 'contract_sha256': sha(source / 'CONTRACT.json'), 'added': added})
    db = sqlite3.connect(scoring / 'results.sqlite'); db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    for sid, value in cache.items():
        previous = db.execute('SELECT payload FROM results WHERE id=?', (sid,)).fetchone()
        if previous: assert json.loads(previous[0]) == value
        else: db.execute('INSERT INTO results VALUES (?,?)', (sid, json.dumps(value, ensure_ascii=False)))
    db.commit(); db.close()
    atomic(semantic / 'CACHE_RECEIPT.json', {'sources': receipts, 'cached': len(cache), 'total': len(pairs), 'judge_contract_sha256': judge_sha})
    if len(cache) == len(pairs):
        atomic(scoring / 'CONTRACT.json', {'schema': 'generation_ar_exact_frozen_judge_cache_union_v1', 'sources': receipts,
            'pairs_sha256': sha(semantic / 'pairs.json'), 'judge_contract_sha256': judge_sha, 'new_judge_calls': 0, 'test_used': False})
        atomic(scoring / 'STATUS.json', {'status': 'COMPLETE', 'cache_only': True, 'pairs': len(pairs)})
    else:
        status('JUDGING_NEW_SEMANTIC_PAIRS', new_pairs=len(pairs) - len(cache))
        run([PYTHON, worker, '--judge', judge, '--pairs', semantic / 'pairs.json', '--output', scoring,
            '--max-wall-seconds', '600'], '1,2', scoring, 780)
    status('SCORING_REQUEST_CONSTRAINTS')
    run([PYTHON, scorer, 'score', '--output', semantic, '--report-tag', 'precompletion'], '', semantic / 'score', 300)
    report = json.loads((semantic / 'REQUEST_SATISFACTION_REPORT.precompletion.json').read_text()); checks = []
    for n in map(str, range(1, 5)):
        current = report['by_candidate']['qualitative_execution']['natural'][n]['metrics']
        prior = report['by_candidate']['copy_v2']['natural'][n]['metrics']
        thresholds = {'valid': 1., 'count': .95, 'core_precision': .9, 'core_recall': .9, 'motion': .95,
            'motion_with_direction_constraints': .95, 'start': .95, 'end': .95, 'onset': .95, 'offset': .95,
            'speech_words': .95, 'request_joint_before_completion_review': .8}
        for metric, minimum in thresholds.items():
            value = current[metric]['rate']; checks.append({'source_count': int(n), 'metric': metric, 'value': value,
                'minimum': minimum, 'pass': value is not None and value >= minimum - 1e-9})
        for metric, maximum_drop in [('count', .005), ('core_precision', .02), ('core_recall', .02), ('speech_words', .02)]:
            delta = current[metric]['rate'] - prior[metric]['rate']
            checks.append({'source_count': int(n), 'metric': metric + '_retention', 'delta': delta, 'minimum': -maximum_drop, 'pass': delta >= -maximum_drop - 1e-9})
    result = {'status': 'PASS_RAW_PILOT_BEFORE_COMPLETION_AND_AUDIO' if all(check['pass'] for check in checks) else 'FAIL_RAW_QUALITATIVE_PILOT_RULE',
        'checks': checks, 'full_report': str(semantic / 'REQUEST_SATISFACTION_REPORT.precompletion.json'),
        'checkpoint_sha256': protocol['checkpoint_sha256'], 'raw_protocol_sha256': sha(root / 'RAW_VALIDATION_PROTOCOL.json'),
        'test_used': False, 'goal_complete': False, 'model_promoted': False,
        'completion_and_audio': 'Separate reasonableness and P10 request/audio checks remain required.'}
    atomic(output / 'RESULT.json', result); status(result['status'], result=str(output / 'RESULT.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--existing-shards', type=int, default=0, choices=(0, 1, 2, 3))
    parser.add_argument('--extra-candidate', action='append', default=[]); args = parser.parse_args()
    try: main(args.root, args.existing_shards, args.extra_candidate)
    except BaseException as exc:
        atomic(args.root / 'raw_evaluation/STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}', 'updated_unix': time.time()})
        raise
