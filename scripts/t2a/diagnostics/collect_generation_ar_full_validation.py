#!/usr/bin/env python3
"""Collect frozen raw-validation shards, reuse verified judgments and score them.

This supervisor never supplies scoring annotations to a generation worker. It
also runs the separately frozen P10 speech diagnostic once GPU0 is released.
"""
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
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def rows(path):
    db = sqlite3.connect('file:' + str(path) + '?mode=ro&immutable=1', uri=True)
    try:
        return {key: json.loads(value) for key, value in db.execute('SELECT id,payload FROM results')}
    finally:
        db.close()


def process_alive(launch):
    path = Path('/proc') / str(launch['pid']) / 'cmdline'
    try:
        return str(launch['command'][1]).encode() in path.read_bytes().split(b'\0')
    except FileNotFoundError:
        return False


def run(command, folder, gpu, cap):
    folder.mkdir(parents=True, exist_ok=True)
    command = list(map(str, command))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    with (folder / 'launcher.log').open('ab') as log:
        child = subprocess.Popen(['timeout', '--kill-after=20s', str(cap), *command],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True, env=env)
    atomic(folder / 'LAUNCH.json', {'pid': child.pid, 'command': command,
           'started_unix': time.time(), 'gpu': gpu, 'wall_cap_s': cap})
    code = child.wait()
    if code:
        raise RuntimeError(f'Child exited {code}: {folder / "launcher.log"}')


def wait_raw_shard(folder, expected_rows, cap):
    started = time.monotonic()
    while True:
        saved = read(folder / 'STATUS.json') if (folder / 'STATUS.json').exists() else {}
        launch = read(folder / 'LAUNCH.json')
        alive = process_alive(launch)
        if saved.get('status') == 'COMPLETE' and not alive:
            summary = read(folder / 'SUMMARY.json')
            assert summary['rows'] == expected_rows
            assert read(folder / 'RAW_GPU_GATE.json')['status'] == 'PASS'
            return
        if str(saved.get('status', '')).startswith('FAIL'):
            raise RuntimeError(f'Raw generation failed: {folder}: {saved}')
        if not alive and saved.get('status') != 'COMPLETE':
            raise RuntimeError(f'Raw process disappeared before completion: {folder}')
        if time.monotonic() - started > cap:
            raise TimeoutError(f'Collection exceeded generation budget: {folder}')
        time.sleep(15)


def speech_after_gpu0(root, protocol):
    """Independent diagnostic; raw/full request scoring never waits to launch it."""
    output = Path(protocol['speech_diagnostic']['output'])
    try:
        wait_raw_shard(root / 'raw_evaluation/raw_gpu0', protocol['raw_shard_rows']['0'],
                       protocol['budget']['raw_process_wall_s_per_gpu'] + 120)
        spec = protocol['speech_diagnostic']
        assert sha(spec['script']) == spec['script_sha256']
        assert sha(spec['summary']) == spec['summary_sha256']
        assert sha(spec['witnesses']) == spec['witnesses_sha256']
        if (output / 'STATUS.json').exists() and read(output / 'STATUS.json')['status'] == 'COMPLETE':
            return
        run([PYTHON, spec['script'], '--snapshot', protocol['training_snapshot'],
             '--summary', spec['summary'], '--witnesses', spec['witnesses'], '--output', output],
            output, '0', 1200)
    except BaseException as exc:
        atomic(output / 'SUPERVISOR_ERROR.json', {'status': 'FAILED_NEEDS_ATTENTION',
               'error': f'{type(exc).__name__}: {exc}', 'updated_unix': time.time()})
        raise


def collect_and_score(root, protocol):
    output = root / 'raw_evaluation'
    ar = root.parent
    candidate_name = protocol.get('candidate_name', 'event_scope_full')
    evaluator = Path(protocol['evaluation_snapshot'])
    manifest = evaluator / 'MANIFEST.json'
    assert sha(manifest) == protocol['evaluation_snapshot_manifest_sha256']
    for rel, digest in read(manifest)['files'].items():
        assert sha(evaluator / rel) == digest
    scorer = evaluator / 'scripts/t2a/diagnostics/evaluate_generation_ar_natural_candidates.py'
    worker = evaluator / 'scripts/t2a/diagnostics/evaluate_generation_ar_semantic_pairs.py'

    def status(name, **extra):
        atomic(root / 'STATUS.json', {'status': name, 'pid': os.getpid(),
               'updated_unix': time.time(), 'test_used': False, 'goal_complete': False, **extra})

    status('WAITING_FOR_FROZEN_RAW_SHARDS')
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(wait_raw_shard, output / f'raw_gpu{gpu}',
                protocol['raw_shard_rows'][str(gpu)],
                protocol['budget']['raw_process_wall_s_per_gpu'] + 120) for gpu in range(3)]
        for job in jobs:
            job.result()
    status('VERIFYING_AND_MERGING_RAW_RESULTS')
    assert sha(protocol['validation_pairs']) == protocol['validation_pairs_sha256']
    references = read(protocol['validation_pairs'])['pairs']
    predictions = {}
    for gpu in range(3):
        folder = output / f'raw_gpu{gpu}'
        contract = read(folder / 'CONTRACT.json')
        assert contract['checkpoint_sha256'] == protocol['checkpoint_sha256']
        assert contract['input_sha256'] == protocol['raw_shards_sha256'][str(gpu)]
        assert contract['target_or_annotation_inputs'] is False
        shard = rows(folder / 'predictions.sqlite')
        assert set(shard).isdisjoint(predictions)
        predictions.update(shard)
    assert len(predictions) == protocol['rows'] == len(references)
    assert set(predictions) == {row['id'] for row in references}
    for ref in references:
        pred = predictions[ref['id']]
        assert pred['request'] == ref['request']
        assert pred['model_input_sha256'] == hashlib.sha256(ref['request'].encode()).hexdigest()
    merged = output / 'raw_validation'
    merged.mkdir(exist_ok=True)
    database = merged / 'predictions.sqlite'
    db = sqlite3.connect(database)
    db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    if not db.execute('SELECT COUNT(*) FROM results').fetchone()[0]:
        db.executemany('INSERT INTO results VALUES (?,?)',
                       [(key, json.dumps(value, ensure_ascii=False)) for key, value in predictions.items()])
        db.commit()
    db.close()
    assert rows(database) == predictions
    atomic(merged / 'CONTRACT.json', {'schema': 'generation_ar_disjoint_raw_merge_v1',
           'checkpoint_sha256': protocol['checkpoint_sha256'],
           'shard_contract_sha256': {str(g): sha(output / f'raw_gpu{g}/CONTRACT.json') for g in range(3)},
           'raw_protocol_sha256': sha(root / 'RAW_VALIDATION_PROTOCOL.json'),
           'raw_only': True, 'test_used': False})
    atomic(merged / 'SUMMARY.json', {'status': 'COMPLETE', 'rows': len(predictions),
           'parsed': sum(row['prediction'] is not None for row in predictions.values()),
           'primary_copy_eos_failures': sum(row.get('decode_recovery') is not None for row in predictions.values()),
           'test_used': False})
    atomic(merged / 'STATUS.json', {'status': 'COMPLETE'})
    del predictions

    semantic = output / 'semantic'
    status('PREPARING_COUPLED_REQUEST_SCORING')
    run([PYTHON, scorer, 'prepare', '--validation', protocol['validation_pairs'],
         '--candidate', candidate_name + '=' + str(database), '--output', semantic,
         '--prove-exact-satisfied'], semantic / 'preparation', '', 900)
    pairs = {row['id']: row for row in read(semantic / 'pairs.json')['pairs']}
    scoring = semantic / 'scoring'
    scoring.mkdir(exist_ok=True)
    judge = ar / 'semantic_judge_20260905_v1'
    judge_sha = sha(judge / 'CONTRACT.json')
    assert judge_sha == protocol['judge_contract_sha256']
    cache, receipts = {}, []
    cache_sources = list(protocol['semantic_cache_sources'])
    inherited_unions = []
    for name in protocol.get('semantic_optional_cache_sources', []):
        source = Path(name)
        contract = read(source / 'CONTRACT.json')
        assert contract['judge_contract_sha256'] == judge_sha
        assert read(source / 'STATUS.json')['status'] == 'COMPLETE'
        if contract.get('script_sha256') == sha(worker):
            cache_sources.append(name)
        else:
            assert contract['schema'] == 'generation_ar_exact_frozen_judge_cache_union_v1'
            assert contract['new_judge_calls'] == 0
            for inherited in contract['sources']:
                assert inherited['source'] in protocol['semantic_cache_sources']
                parent = Path(inherited['source'])
                assert sha(parent / 'CONTRACT.json') == inherited['contract_sha256']
                assert sha(parent / 'results.sqlite') == inherited['db_sha256']
            inherited_unions.append({'source': name, 'contract_sha256': sha(source / 'CONTRACT.json'),
                                     'reason': 'Zero new judgments; all parents are already in the verified direct cache sources.'})
    for name in cache_sources:
        source = Path(name)
        contract = read(source / 'CONTRACT.json')
        # These are direct frozen-worker outputs, not unverified cache unions.
        assert contract['judge_contract_sha256'] == judge_sha
        assert contract['script_sha256'] == sha(worker)
        assert read(source / 'STATUS.json')['status'] == 'COMPLETE'
        cached = rows(source / 'results.sqlite')
        added = 0
        for key in sorted(set(pairs) & set(cached)):
            value = cached[key]
            assert all(value[field] == pairs[key][field] for field in ('id', 'kind', 'reference', 'candidate'))
            if key in cache:
                assert cache[key] == value
            else:
                cache[key] = value
                added += 1
        receipts.append({'source': str(source), 'db_sha256': sha(source / 'results.sqlite'),
                         'contract_sha256': sha(source / 'CONTRACT.json'), 'added': added})
    db = sqlite3.connect(scoring / 'results.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    for key, value in cache.items():
        prior = db.execute('SELECT payload FROM results WHERE id=?', (key,)).fetchone()
        if prior:
            assert json.loads(prior[0]) == value
        else:
            db.execute('INSERT INTO results VALUES (?,?)', (key, json.dumps(value, ensure_ascii=False)))
    db.commit()
    db.close()
    atomic(semantic / 'CACHE_RECEIPT.json', {'sources': receipts, 'cached': len(cache),
           'total': len(pairs), 'judge_contract_sha256': judge_sha,
           'redundant_verified_cache_unions': inherited_unions})
    if len(cache) == len(pairs):
        atomic(scoring / 'CONTRACT.json', {'schema': 'generation_ar_exact_frozen_judge_cache_union_v1',
               'sources': receipts, 'pairs_sha256': sha(semantic / 'pairs.json'),
               'judge_contract_sha256': judge_sha, 'new_judge_calls': 0, 'test_used': False})
        atomic(scoring / 'STATUS.json', {'status': 'COMPLETE', 'cache_only': True, 'pairs': len(pairs)})
    else:
        status('JUDGING_NEW_SEMANTIC_PAIRS', new_pairs=len(pairs) - len(cache))
        budget = protocol['budget']['semantic_worker_wall_s']
        run([PYTHON, worker, '--judge', judge, '--pairs', semantic / 'pairs.json',
             '--output', scoring, '--max-wall-seconds', str(budget)], scoring, '1,2', budget + 300)
    status('SCORING_ALL_REQUEST_CONSTRAINTS')
    run([PYTHON, scorer, 'score', '--output', semantic, '--report-tag', 'precompletion'],
        semantic / 'score', '', 900)
    report_path = semantic / 'REQUEST_SATISFACTION_REPORT.precompletion.json'
    current = read(report_path)['by_candidate'][candidate_name]['natural']
    assert sha(protocol['retention_reference']) == protocol['retention_reference_sha256']
    prior = read(protocol['retention_reference'])['by_candidate']['t100_full']['natural']
    checks = []
    for count in map(str, range(1, 5)):
        for metric, minimum in protocol['thresholds'].items():
            value = current[count]['metrics'][metric]['rate']
            checks.append({'source_count': int(count), 'metric': metric, 'value': value,
                           'minimum': minimum, 'pass': value is not None and value >= minimum - 1e-9})
        for metric, maximum_drop in protocol['retention_maximum_drop'].items():
            delta = current[count]['metrics'][metric]['rate'] - prior[count]['metrics'][metric]['rate']
            checks.append({'source_count': int(count), 'metric': metric + '_retention_vs_t100_full',
                           'delta': delta, 'minimum': -maximum_drop, 'pass': delta >= -maximum_drop - 1e-9})
    peer_receipt = None
    if protocol.get('peer_comparison'):
        peer = protocol['peer_comparison']
        assert sha(peer['raw_protocol']) == peer['raw_protocol_sha256']
        peer_result = read(peer['result'])
        assert peer_result['raw_protocol_sha256'] == peer['raw_protocol_sha256']
        assert peer_result['checkpoint_sha256'] == protocol['checkpoint_sha256']
        peer_report_path = Path(peer_result['full_report'])
        peer_metrics = read(peer_report_path)['by_candidate'][peer['candidate_name']]['natural']
        for count in map(str, range(1, 5)):
            for metric in protocol['thresholds']:
                delta = current[count]['metrics'][metric]['rate'] - peer_metrics[count]['metrics'][metric]['rate']
                checks.append({'source_count': int(count), 'metric': metric + '_retention_vs_cached_event',
                               'delta': delta, 'minimum': -peer['maximum_drop'],
                               'pass': delta >= -peer['maximum_drop'] - 1e-9})
        peer_receipt = {'result_sha256': sha(peer['result']), 'report_sha256': sha(peer_report_path)}
    result = {'status': 'PASS_FULL_VALIDATION_BEFORE_COMPLETION_AND_AUDIO' if all(c['pass'] for c in checks)
              else 'FAIL_FULL_VALIDATION_REQUEST_RULE', 'rows': protocol['rows'], 'checks': checks,
              'full_report': str(report_path), 'checkpoint_sha256': protocol['checkpoint_sha256'],
              'raw_protocol_sha256': sha(root / 'RAW_VALIDATION_PROTOCOL.json'),
              'collector_protocol_sha256': sha(root / 'COLLECTOR_PROTOCOL.json'),
              'peer_comparison_receipt': peer_receipt,
              'test_used': False, 'goal_complete': False, 'model_promoted': False,
              'completion_and_audio': 'Separate reasonableness and P10 audio/request checks remain required.'}
    atomic(root / 'RESULT.json', result)
    status(result['status'], result=str(root / 'RESULT.json'))


def main(root):
    root = root.resolve()
    protocol = read(root / 'RAW_VALIDATION_PROTOCOL.json')
    collector = read(root / 'COLLECTOR_PROTOCOL.json')
    assert collector['raw_protocol_sha256'] == sha(root / 'RAW_VALIDATION_PROTOCOL.json')
    assert collector['script_sha256'] == sha(Path(__file__))
    protocol.update(collector)
    if protocol.get('speech_diagnostic') is None:
        collect_and_score(root, protocol)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            speech = pool.submit(speech_after_gpu0, root, protocol)
            collect_and_score(root, protocol)
            speech.result()
    atomic(root / 'COLLECTOR_COMPLETE.json', {'status': 'COMPLETE', 'updated_unix': time.time(),
           'request_result': str(root / 'RESULT.json'),
           'speech_diagnostic': protocol['speech_diagnostic']['output'] if protocol.get('speech_diagnostic') else None,
           'goal_complete': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root)
    except BaseException as exc:
        atomic(args.root / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION',
               'error': f'{type(exc).__name__}: {exc}', 'updated_unix': time.time()})
        raise
