#!/usr/bin/env python3
"""Independent validation-only comparison of completed-run AR checkpoints.

Does not mutate or impersonate the official checkpoint selection/test contract.
Prepare the same stratified panel before decoding; one process owns one GPU.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import zlib

from generation_ar_fidelity import compare_fields, summarize_fields, TOLERANCES

SNAPSHOT = Path('/mnt/sdc/ckpts/transfusion_sceneplan/generation_ar/source_snapshots/p10v11_gen_ar_20260904_10epoch_continuation_v8')
RUN = Path('/mnt/sdc/ckpts/transfusion_sceneplan/generation_ar/p10v11_shared_gen_ar_full_1p6m_10ep_s42_20260904_stage2_v1')
MANIFEST = Path('/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/transfusion_shared_v1/generation_ar/validation.sqlite')
MANIFEST_SHA = '697113f9c38f190cb7e54bf8863c3e3b78dfa77de76daeb9b1ea7c035fa6dd4c'
STEPS = (66672, 70839, 83340)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, obj):
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temporary.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, path)


def load_frozen():
    sys.path.insert(0, str(SNAPSHOT))
    path = SNAPSHOT / 'scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_8k.py'
    spec = importlib.util.spec_from_file_location('frozen_ar_evaluation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(output, per_class):
    if per_class <= 0 or per_class % 4:
        raise ValueError('per-class must be positive and divisible by four templates')
    output.mkdir(parents=True, exist_ok=False)
    if sha(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError('validation manifest hash mismatch')
    selection = json.loads((RUN / 'CHECKPOINT_SELECTION.json').read_text())
    training = json.loads((RUN / 'RUN_CONTRACT.json').read_text())
    candidates = [{k: c[k] for k in ('step', 'checkpoint', 'checkpoint_sha256')}
                  for c in selection['candidates'] if c['step'] in STEPS]
    assert tuple(c['step'] for c in candidates) == STEPS
    for candidate in candidates:
        assert sha(candidate['checkpoint']) == candidate['checkpoint_sha256']
    sources = {str(SNAPSHOT / path): digest for path, digest in training['source_sha256'].items()}
    sources[training['source_snapshot']['path']] = training['source_snapshot']['sha256']
    for path, digest in sources.items():
        assert sha(path) == digest, path
    for path in [Path(__file__), Path(__file__).with_name('generation_ar_fidelity.py'),
                 SNAPSHOT / 'scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_8k.py',
                 SNAPSHOT / 'stable_audio_tools/data/sceneplan_transfusion_generation_ar_evaluation.py',
                 SNAPSHOT / 'stable_audio_tools/data/sceneplan_p11_metrics.py']:
        sources[str(path.resolve())] = sha(path)
    db = sqlite3.connect(f'file:{MANIFEST}?mode=ro&immutable=1', uri=True)
    metadata = dict(db.execute('SELECT key,value FROM metadata'))
    assert metadata['split'] == 'validation'
    groups = defaultdict(list)
    for ordinal, sample_id, count, template in db.execute('SELECT ordinal,sample_id,source_count,template_id FROM rows'):
        groups[(count, template)].append((hashlib.sha256(f'42:{sample_id}'.encode()).hexdigest(), ordinal))
    assert len(groups) == 16 and sum(len(v) for v in groups.values()) == 32000
    ordinals = sorted(ordinal for values in groups.values() for _, ordinal in sorted(values)[:per_class // 4])
    assert len(ordinals) == len(set(ordinals)) == per_class * 4
    panel = sqlite3.connect(output / 'panel.sqlite')
    panel.execute(db.execute("SELECT sql FROM sqlite_master WHERE name='rows'").fetchone()[0])
    panel.execute('CREATE TABLE panel_order(panel_index INTEGER PRIMARY KEY, ordinal INTEGER UNIQUE NOT NULL)')
    for index, ordinal in enumerate(ordinals):
        row = db.execute('SELECT * FROM rows WHERE ordinal=?', (ordinal,)).fetchone()
        panel.execute('INSERT INTO rows VALUES (' + ','.join('?' for _ in row) + ')', row)
        panel.execute('INSERT INTO panel_order VALUES (?,?)', (index, ordinal))
    panel.commit()
    counts = panel.execute('SELECT source_count,template_id,COUNT(*) FROM rows GROUP BY source_count,template_id').fetchall()
    panel.close(); db.close()
    contract = {'schema': 'generation_ar_validation_comparison_v1', 'purpose': 'DIAGNOSTIC_ONLY',
                'evaluation_split': 'validation', 'test_used': False, 'official_selection_modified': False,
                'manifest': str(MANIFEST), 'manifest_sha256': MANIFEST_SHA,
                'panel': str(output / 'panel.sqlite'), 'panel_sha256': sha(output / 'panel.sqlite'),
                'rows': len(ordinals), 'sampling': 'equal source_count/template cells; ascending SHA256(42:sample_id); chosen before decode',
                'panel_cell_counts': counts, 'candidates': candidates,
                'training_contract_path': str(RUN / 'RUN_CONTRACT.json'),
                'training_contract_sha256': sha(RUN / 'RUN_CONTRACT.json'),
                'source_sha256': sources, 'snapshot': str(SNAPSHOT),
                'batch_size': 32, 'max_plan_tokens': 512, 'decode': 'original frozen greedy codec-constrained AR',
                'tolerances_diagnostic_not_acceptance': TOLERANCES,
                'physical_gpu_by_step': {str(step): gpu for gpu, step in enumerate(STEPS)},
                'selection_note': 'This balanced panel diagnoses candidates. Any new production selection requires full validation and a new explicit selection rule.'}
    atomic(output / 'CONTRACT.json', contract)
    print(json.dumps({'event': 'prepared', 'output': str(output), 'rows': len(ordinals), 'steps': STEPS}), flush=True)


def worker(output, step):
    contract_path = output / 'CONTRACT.json'
    contract = json.loads(contract_path.read_text())
    gpu = str(contract['physical_gpu_by_step'][str(step)])
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == gpu and gpu in ('0', '1', '2')
    for path, digest in contract['source_sha256'].items():
        assert sha(path) == digest, path
    assert sha(contract['panel']) == contract['panel_sha256']
    assert sha(contract['training_contract_path']) == contract['training_contract_sha256']
    candidate = next(c for c in contract['candidates'] if c['step'] == step)
    assert sha(candidate['checkpoint']) == candidate['checkpoint_sha256']
    run = output / f'step_{step:08d}'
    run.mkdir(exist_ok=True)
    lock = (run / 'LOCK').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = load_frozen()
    torch = frozen.torch
    torch.manual_seed(42)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    codec = frozen.ModelScenePlanCodecV4(frozen.CODEC_PATH)
    state = torch.load(candidate['checkpoint'], map_location='cpu', weights_only=False)
    training = json.loads(Path(contract['training_contract_path']).read_text())
    assert state['global_step'] == step and state['run_contract'] == training
    assert codec.fingerprint == training['codec_fingerprint']
    print(json.dumps({'event': 'loading_model', 'step': step, 'physical_gpu': gpu}), flush=True)
    model, p10_report = frozen.load_p10v11_generation_ar(pad_id=codec.pad_id, verify_sha256=True, activation_checkpointing=False)
    assert p10_report.as_dict() == training['p10_load']
    model.load_trainable_state_dict(state['ar_adapter'])
    del state
    model.p10_dit.to(device=device, dtype=torch.bfloat16)
    model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32)
    model.eval()
    panel = sqlite3.connect(f"file:{contract['panel']}?mode=ro&immutable=1", uri=True)
    panel.row_factory = sqlite3.Row
    rows = [dict(r) for r in panel.execute('SELECT r.*,p.panel_index FROM rows r JOIN panel_order p USING(ordinal) ORDER BY p.panel_index')]
    panel.close()
    result_db = sqlite3.connect(run / 'predictions.sqlite')
    result_db.execute('PRAGMA journal_mode=WAL')
    result_db.execute('PRAGMA synchronous=FULL')
    result_db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    result_db.execute('CREATE TABLE IF NOT EXISTS results(panel_index INTEGER PRIMARY KEY,ordinal INTEGER NOT NULL,sample_id TEXT NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL)')
    identity = {'contract_sha256': sha(contract_path), 'checkpoint_sha256': candidate['checkpoint_sha256']}
    existing = dict(result_db.execute('SELECT key,value FROM metadata'))
    if existing:
        assert existing == identity
    else:
        result_db.executemany('INSERT INTO metadata VALUES (?,?)', identity.items()); result_db.commit()
    completed = {i: (o, s) for i, o, s in result_db.execute('SELECT panel_index,ordinal,sample_id FROM results')}
    for row in rows:
        if row['panel_index'] in completed:
            assert completed[row['panel_index']] == (row['ordinal'], row['sample_id'])
    pending = [r for r in rows if r['panel_index'] not in completed]
    started = time.monotonic()
    for offset in range(0, len(pending), contract['batch_size']):
        batch = pending[offset:offset + contract['batch_size']]
        generated = frozen._generate_with_fallback(model, batch, codec, device=device, max_plan_tokens=contract['max_plan_tokens'])
        for row, (tokens, error, elapsed) in zip(batch, generated):
            target_bytes = zlib.decompress(row['target_sceneplan_zlib'])
            assert hashlib.sha256(target_bytes).hexdigest() == row['target_sceneplan_sha256']
            target = json.loads(target_bytes)
            prediction = None
            status = 'generation_error' if tokens is None else 'ok'
            if tokens is not None:
                try:
                    prediction = codec.decode(tokens, sample_id=row['sample_id'])
                except Exception as exc:
                    status, error = 'parse_error', repr(exc)
            metrics = frozen.score_parsed_generation(target, prediction) if prediction is not None else {}
            payload = {'panel_index': row['panel_index'], 'ordinal': row['ordinal'], 'sample_id': row['sample_id'],
                       'source_count': row['source_count'], 'template_id': row['template_id'], 'request': row['raw_user_request'],
                       'target_sceneplan_sha256': row['target_sceneplan_sha256'], 'target': target, 'prediction': prediction,
                       'tokens': tokens, 'status': status, 'error': error, 'generation_sec': elapsed,
                       'legacy_metrics': metrics, 'fidelity': compare_fields(target, prediction)}
            result_db.execute('INSERT INTO results VALUES (?,?,?,?,?)', (row['panel_index'], row['ordinal'], row['sample_id'], status, json.dumps(payload, ensure_ascii=False)))
        result_db.commit()
        done = len(completed) + min(offset + len(batch), len(pending))
        atomic(run / 'STATUS.json', {'status': 'RUNNING', 'step': step, 'rows_done': done, 'rows': len(rows), 'elapsed_s': time.monotonic() - started})
        if (offset // contract['batch_size']) % 4 == 0:
            print(json.dumps({'event': 'decode_progress', 'step': step, 'done': done, 'total': len(rows), 'elapsed_s': round(time.monotonic() - started, 1)}), flush=True)
    records = [json.loads(r[0]) for r in result_db.execute('SELECT payload FROM results ORDER BY panel_index')]
    assert [r['panel_index'] for r in records] == list(range(contract['rows']))
    def summarize(part):
        fields = summarize_fields([r['fidelity'] for r in part])
        fields['lexical_description_token_f1_proxy'] = sum(r['legacy_metrics'].get('persistent_semantic_token_f1', 0.) for r in part) / len(part)
        return fields
    summary = {'status': 'COMPLETE', 'purpose': contract['purpose'], 'step': step, 'checkpoint_sha256': candidate['checkpoint_sha256'],
               'contract_sha256': sha(contract_path), 'physical_gpu': gpu,
               'status_counts': dict(Counter(r['status'] for r in records)), 'overall_balanced_panel': summarize(records),
               'by_source_count': {str(k): summarize([r for r in records if r['source_count'] == k]) for k in range(1, 5)},
               'natural_distribution_count_accuracy_estimate': sum(summarize([r for r in records if r['source_count'] == k])['source_count_accuracy'] * weight
                                                                  for k, weight in [(1,.2875),(2,.50625),(3,.1375),(4,.06875)]),
               'elapsed_s': time.monotonic() - started}
    atomic(run / 'SUMMARY.json', summary)
    atomic(run / 'STATUS.json', {'status': 'COMPLETE', 'step': step, 'rows_done': len(records), 'summary': str(run / 'SUMMARY.json')})
    result_db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); result_db.close()
    print(json.dumps({'event': 'complete', 'step': step, 'summary': str(run / 'SUMMARY.json'), 'count_accuracy': summary['overall_balanced_panel']['source_count_accuracy']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--per-class', type=int, default=256)
    parser.add_argument('--step', type=int, choices=STEPS)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.output_dir.resolve(), args.per_class)
    else:
        if args.step is None:
            parser.error('--step is required for a worker')
        worker(args.output_dir.resolve(), args.step)


if __name__ == '__main__':
    main()
