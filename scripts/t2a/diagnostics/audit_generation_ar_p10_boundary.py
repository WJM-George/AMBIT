#!/usr/bin/env python3
"""Audit every raw AR result through the real P10 compiler, without audio/GPU."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import time

CODEC = TOKENIZER = NORMALIZE = FINALIZE = TASK = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def initialize(snapshot, normalizer):
    global CODEC, TOKENIZER, NORMALIZE, FINALIZE, TASK
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, snapshot)
    import torch
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    # The independently frozen P10 adapter is newer than the model snapshot.
    # Load only this explicit, hashed file without changing the frozen package.
    spec = importlib.util.spec_from_file_location('p10_audit_normalizer', normalizer)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from stable_audio_tools.data.sceneplan_p11_single_turn import finalize_sceneplan_for_p10, P11Task
    CODEC = ModelScenePlanCodecV4(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    TOKENIZER = AutoTokenizer.from_pretrained(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B", local_files_only=True)
    NORMALIZE = module.normalize_generated_sceneplan; FINALIZE = finalize_sceneplan_for_p10; TASK = P11Task.GENERATION


def audit(batch):
    rows = []
    for sid, value in batch:
        result = json.loads(value); plan = result['prediction']
        if plan is None:
            rows.append({'id': sid, 'status': 'NO_PREDICTION', 'error': result.get('error')}); continue
        row = {'id': sid, 'source_count': len(plan['sources']), 'duration_sec': plan['duration_sec'],
               'degenerate_linear_sources': [], 'status': 'PENDING'}
        for source in plan['sources']:
            tr = source['trajectory']
            if tr['type'] == 'linear':
                a, b = tr['start'], tr['end']
                delta = abs((b['azimuth_deg'] - a['azimuth_deg'] + 180) % 360 - 180)
                if delta < 1e-8 and all(abs(a[k] - b[k]) < 1e-8 for k in ['distance_m', 'elevation_deg']):
                    row['degenerate_linear_sources'].append(source['source_id'])
        try:
            normalized = NORMALIZE(CODEC, result['tokens'], sample_id=sid, max_tokens=512)
            assert normalized['raw_plan'] == plan
            bundle = FINALIZE(CODEC, normalized['p10_token_ids'], tokenizer=TOKENIZER, task=TASK, sample_id=sid)
            bundle.assert_external_p10_boundary(); assert bundle.sceneplan == normalized['p10_plan']
            assert normalized['non_text_fields_unchanged'] is True
            row.update(status='P10_COMPILER_PASS', whitespace_changes=len(normalized['whitespace_changes']),
                prompt_tokens=int(bundle.p10_metadata['prompt']['attention_mask'].sum()), latent_frames=bundle.latent_frames_valid)
        except Exception as exc:
            row.update(status='P10_COMPILER_FAIL', error=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    return rows


def main(args):
    assert 1 <= args.workers <= 8
    args.output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect('file:' + str(args.predictions) + '?mode=ro&immutable=1', uri=True)
    rows = db.execute('SELECT id,payload FROM results ORDER BY id').fetchall(); db.close()
    assert len(rows) == 32000
    identity = {'prediction_sha256': sha(args.predictions), 'snapshot': str(args.snapshot),
        'snapshot_manifest_sha256': sha(args.snapshot / 'SOURCE_SNAPSHOT_MANIFEST.json'),
        'normalizer': str(args.normalizer), 'normalizer_sha256': sha(args.normalizer),
        'script_sha256': sha(Path(__file__)), 'rows': len(rows), 'test_used': False, 'gpu_used': False}
    atomic(args.output / 'CONTRACT.json', identity)
    started = time.monotonic(); complete = []
    atomic(args.output / 'STATUS.json', {'status': 'RUNNING', 'rows_done': 0, 'rows': len(rows), 'pid': os.getpid()})
    # Fail promptly on setup mistakes before submitting a large process queue.
    initialize(str(args.snapshot), str(args.normalizer))
    smoke = audit(rows[:1]); atomic(args.output / 'STARTUP_CHECK.json', {'rows': smoke})
    if smoke[0]['status'] != 'P10_COMPILER_PASS':
        raise RuntimeError('First-row P10 check failed; review STARTUP_CHECK.json before launching the full audit')
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize, initargs=(str(args.snapshot), str(args.normalizer))) as pool:
        futures = [pool.submit(audit, rows[i:i + 256]) for i in range(0, len(rows), 256)]
        for f in as_completed(futures):
            complete.extend(f.result())
            atomic(args.output / 'STATUS.json', {'status': 'RUNNING', 'rows_done': len(complete), 'rows': len(rows), 'elapsed_s': time.monotonic() - started})
    assert len(complete) == len(rows)
    failures = [r for r in complete if r['status'] != 'P10_COMPILER_PASS']
    with (args.output / 'rows.jsonl').open('w') as f:
        for row in sorted(complete, key=lambda r: r['id']): f.write(json.dumps(row) + '\n')
    result = {**identity, 'status': 'PASS' if not failures else 'FAIL', 'passed': len(complete) - len(failures),
        'failures': failures, 'whitespace_normalized_rows': sum(bool(r.get('whitespace_changes')) for r in complete),
        'degenerate_linear_sources': [{'id': r['id'], 'sources': r['degenerate_linear_sources']} for r in complete if r.get('degenerate_linear_sources')],
        'elapsed_s': time.monotonic() - started, 'scope': 'Real decode/whitespace-only canonicalization/P10 condition compilation. Does not establish source semantics or audible request satisfaction.'}
    atomic(args.output / 'REPORT.json', result); atomic(args.output / 'STATUS.json', {'status': 'COMPLETE', 'result': str(args.output / 'REPORT.json'), 'pass': not failures})
    print(json.dumps({'status': result['status'], 'rows': len(rows), 'failures': len(failures), 'degenerate_linear_sources': len(result['degenerate_linear_sources']), 'elapsed_s': result['elapsed_s']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--predictions', type=Path, required=True); p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--normalizer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--workers', type=int, default=4); args = p.parse_args()
    try: main(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True); atomic(args.output / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'}); raise
