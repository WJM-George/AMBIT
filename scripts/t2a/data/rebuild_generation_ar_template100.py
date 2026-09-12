#!/usr/bin/env python3
"""Rewrite existing GT pairs using 100/10/5 qualitative English templates.

CPU only. Targets are copied byte-for-byte. Each request receives internal
request constraints, and every original plan must satisfy those constraints.
Completed shards can be resumed under the identical build contract.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import sys
import time
import zlib

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from stable_audio_tools.data.sceneplan_generation_template100 import (
    CONTRACT, NUMERIC_CONTRACT, SPLIT_COUNTS, catalog, render_pair, template_for)
from stable_audio_tools.data.sceneplan_generation_ar_natural_constraints import (
    validate_requirements, source_constraint_pass, COMPASS_CENTERS,
    COMPASS_HALF_WIDTH, TIME_PHASE_RANGES)

EXPECTED_ROWS = {'train': 1600000, 'validation': 32000, 'test': 8000}
TOKENIZER = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def read_db(path):
    return sqlite3.connect('file:' + str(path) + '?mode=ro&immutable=1', uri=True)


def initialize_worker(tokenizer_path):
    global TOKENIZER
    os.environ.update(CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false',
                      RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    import torch
    torch.set_num_threads(1)
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)


def build_shard(job):
    split, begin, end, source_path, output, contract_sha = job
    out = Path(output)
    report_path = out.with_suffix('.json')
    if report_path.exists():
        report = json.loads(report_path.read_text())
        assert report['contract_sha256'] == contract_sha and sha(out) == report['sha256']
        return report
    started = time.monotonic()
    source = read_db(source_path)
    columns = [r[1] for r in source.execute('PRAGMA table_info(rows)')]
    index = {k: i for i, k in enumerate(columns)}
    schema = source.execute("SELECT sql FROM sqlite_master WHERE name='rows'").fetchone()[0]
    building = out.with_suffix('.building')
    if building.exists():
        building.unlink()  # Only this same-contract unfinished shard.
    dest = sqlite3.connect(building)
    dest.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;' + schema + ';')
    dest.execute('ALTER TABLE rows ADD COLUMN request_requirements_zlib BLOB NOT NULL DEFAULT X\'\'')
    dest.execute("ALTER TABLE rows ADD COLUMN semantic_scene_sha256 TEXT NOT NULL DEFAULT ''")
    insert = 'INSERT INTO rows VALUES (' + ','.join('?' for _ in range(len(columns) + 2)) + ')'
    templates = Counter(); counts = Counter(); phases = Counter(); max_tokens = 0; done = 0
    targets_before = hashlib.sha256(); targets_after = hashlib.sha256()
    cursor = source.execute('SELECT * FROM rows WHERE ordinal>=? AND ordinal<? ORDER BY ordinal', (begin, end))
    while True:
        batch = cursor.fetchmany(256)
        if not batch:
            break
        prepared = []
        for original in batch:
            row = list(original)
            ordinal = row[index['ordinal']]
            plan = json.loads(zlib.decompress(row[index['target_sceneplan_zlib']]))
            tid = template_for(split, ordinal)
            request, req = render_pair(plan, tid)
            validate_requirements(request, req)
            byid = {s['source_id']: s for s in plan['sources']}
            for ref in req['sources']:
                for constraint in ref['constraints']:
                    if not source_constraint_pass(byid[ref['key']], constraint, scene_duration=plan['duration_sec']):
                        raise ValueError(f'{split}/{ordinal}: GT contradicts {constraint}')
                    if constraint['op'] == 'time_phase':
                        phases[constraint['field'] + '/' + constraint['value']] += 1
            # The sample identifier is not part of a scene-family fingerprint.
            semantic_plan = {k: v for k, v in plan.items() if k != 'sample_id'}
            scene_sha = hashlib.sha256(canonical(semantic_plan).encode()).hexdigest()
            row[index['template_id']] = tid
            row[index['raw_user_request']] = request
            row[index['raw_user_request_sha256']] = hashlib.sha256(request.encode()).hexdigest()
            row.extend([zlib.compress(canonical(req).encode(), level=1), scene_sha])
            for field in ('target_sceneplan_zlib', 'target_token_ids_u16le', 'target_loss_group_ids_i16le'):
                value = original[index[field]]
                assert row[index[field]] == value
                targets_before.update(value); targets_after.update(row[index[field]])
            templates[tid] += 1; counts[len(plan['sources'])] += 1
            prepared.append(row)
        lengths = [len(ids) for ids in TOKENIZER([r[index['raw_user_request']] for r in prepared],
                    add_special_tokens=True, truncation=False)['input_ids']]
        for row, length in zip(prepared, lengths):
            if length > 512:
                raise ValueError(f"Input exceeds 512 tokens: {split}/{row[index['ordinal']]} length={length}")
            row[index['raw_request_qwen_tokens']] = length
        max_tokens = max(max_tokens, max(lengths))
        dest.executemany(insert, prepared); dest.commit(); done += len(prepared)
    assert done == end - begin and targets_before.digest() == targets_after.digest()
    dest.close(); source.close(); building.replace(out)
    result = {'status': 'PASS', 'split': split, 'begin': begin, 'end': end, 'rows': done,
              'path': str(out), 'sha256': sha(out), 'contract_sha256': contract_sha,
              'templates': dict(templates), 'source_counts': dict(counts), 'time_phases': dict(phases),
              'max_request_tokens': max_tokens, 'gt_constraints_all_pass': True,
              'target_bytes_unchanged': True, 'target_stream_sha256': targets_before.hexdigest(),
              'elapsed_s': time.monotonic() - started}
    atomic(report_path, result)
    return result


def merge_split(split, shards, source_path, root, contract_sha, catalog_sha):
    output = root / (split + '.sqlite')
    if output.exists():
        result = json.loads(output.with_suffix('.report.json').read_text())
        assert result['contract_sha256'] == contract_sha and sha(output) == result['sha256']
        return result
    source = read_db(source_path)
    metadata = dict(source.execute('SELECT key,value FROM metadata')); source.close()
    first = read_db(shards[0]['path'])
    schema = first.execute("SELECT sql FROM sqlite_master WHERE name='rows'").fetchone()[0]; first.close()
    building = output.with_suffix('.building')
    if building.exists():
        building.unlink()
    db = sqlite3.connect(building)
    db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;' + schema + '; CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;')
    template_counts = Counter(); source_counts = Counter()
    for shard in sorted(shards, key=lambda r: r['begin']):
        db.execute('ATTACH DATABASE ? AS shard', (shard['path'],))
        db.execute('INSERT INTO rows SELECT * FROM shard.rows ORDER BY ordinal'); db.commit()
        db.execute('DETACH DATABASE shard')
        template_counts.update(shard['templates']); source_counts.update(shard['source_counts'])
    rows = db.execute('SELECT COUNT(*) FROM rows').fetchone()[0]
    assert rows == EXPECTED_ROWS[split]
    assert len(template_counts) == SPLIT_COUNTS[split]
    assert set(template_counts.values()) == {rows // SPLIT_COUNTS[split]}
    metadata.update(raw_request_contract=CONTRACT, raw_request_numeric_contract=NUMERIC_CONTRACT,
                    request_compiler_implementation_sha256=sha(REPO / 'stable_audio_tools/data/sceneplan_generation_template100.py'),
                    builder_implementation_sha256=sha(Path(__file__)), raw_request_qwen_max_tokens=str(max(s['max_request_tokens'] for s in shards)),
                    template_ids_json=canonical(sorted(template_counts)), template_counts_json=canonical(template_counts),
                    template_catalog_sha256=catalog_sha, build_contract_sha256=contract_sha,
                    template_count=str(SPLIT_COUNTS[split]), raw_request_truncation_count='0',
                    raw_request_qwen_token_limit='512', rows=str(rows), is_full_split='true',
                    coordinate_and_activity_numbers_in_request='false', split=split,
                    hidden_gt_numbers_are_unique_answers='false', target_bytes_unchanged='true')
    db.executemany('INSERT INTO metadata VALUES (?,?)', sorted(metadata.items()))
    db.execute('CREATE INDEX source_count_idx ON rows(source_count,ordinal)')
    db.execute('CREATE INDEX semantic_scene_idx ON rows(semantic_scene_sha256)')
    db.execute('CREATE INDEX raw_request_hash_idx ON rows(raw_user_request_sha256)')
    duplicates = db.execute('SELECT SUM(n-1) FROM (SELECT COUNT(*) n FROM rows GROUP BY raw_user_request_sha256 HAVING n>1)').fetchone()[0] or 0
    db.commit(); assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'; db.close()
    building.replace(output)
    result = {'status': 'PASS', 'split': split, 'rows': rows, 'templates': dict(template_counts),
              'source_counts': dict(source_counts), 'path': str(output), 'sha256': sha(output),
              'contract_sha256': contract_sha, 'max_request_tokens': max(s['max_request_tokens'] for s in shards),
              'duplicate_raw_requests': duplicates, 'target_bytes_unchanged': True,
              'all_gt_satisfy_request_constraints': True,
              'duplicate_note': 'Identical qualitative requests can have different valid numerical completions; they are not contradictory labels.'}
    atomic(output.with_suffix('.report.json'), result)
    return result


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'shards').mkdir(exist_ok=True)
    sources = {s: str(args.source / (s + '.sqlite')) for s in EXPECTED_ROWS}
    cat = {'schema': CONTRACT, 'recipes': catalog(), 'compass_centers_internal': COMPASS_CENTERS,
           'compass_half_width_internal': COMPASS_HALF_WIDTH, 'relative_time_ranges_internal': TIME_PHASE_RANGES,
           'model_input': 'Natural English only; no generated coordinate, distance, duration or activity-time numbers. GT descriptions and spoken text are retained.',
           'numbers_inside_gt_text': 'Quoted content and speaker wording are copied; spoken numerical words are not removed.',
           'output_order': 'Model outputs matched by source identity; no extra source IDs/count hints provided to inference.'}
    cat_path = args.output / 'TEMPLATE_CATALOG.json'
    atomic(cat_path, cat)
    contract = {'schema': CONTRACT, 'source_manifests': {s: {'path': p, 'sha256': sha(p)} for s, p in sources.items()},
                'rows': EXPECTED_ROWS, 'template_counts': SPLIT_COUNTS, 'catalog_sha256': sha(cat_path),
                'builder_sha256': sha(Path(__file__)), 'renderer_sha256': sha(REPO / 'stable_audio_tools/data/sceneplan_generation_template100.py'),
                'constraint_module_sha256': sha(REPO / 'stable_audio_tools/data/sceneplan_generation_ar_natural_constraints.py'),
                'seed': 42, 'chunk_rows': args.chunk_rows, 'tokenizer': str(args.tokenizer),
                'template_assignment': 'Seeded permutation within each split-sized template block; exactly balanced template counts.',
                'targets': 'Existing immutable GT target bytes retained; unrequested numerical values are valid witnesses only.',
                'test_generated_predictions_used': False}
    cp = args.output / 'BUILD_CONTRACT.json'
    if cp.exists():
        assert json.loads(cp.read_text()) == contract, 'Build contract changed; use a new output directory'
    else:
        atomic(cp, contract)
    digest = sha(cp); completed = {}; bysplit = {s: [] for s in EXPECTED_ROWS}; reports = {}; start = time.monotonic()
    jobs = [(s, begin, min(begin + args.chunk_rows, EXPECTED_ROWS[s]), sources[s],
             str(args.output / 'shards' / f'{s}_{begin:07d}.sqlite'), digest)
            for s in ('validation', 'test', 'train') for begin in range(0, EXPECTED_ROWS[s], args.chunk_rows)]
    expected_shards = Counter(j[0] for j in jobs)
    atomic(args.output / 'STATUS.json', {'status': 'BUILDING', 'pid': os.getpid(), 'rows_done': 0, 'rows': sum(EXPECTED_ROWS.values())})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker, initargs=(str(args.tokenizer),)) as pool:
        futures = [pool.submit(build_shard, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result(); completed[result['path']] = result; bysplit[result['split']].append(result)
            split = result['split']
            if len(bysplit[split]) == expected_shards[split]:
                reports[split] = merge_split(split, bysplit[split], sources[split], args.output, digest, contract['catalog_sha256'])
            done = sum(r['rows'] for r in completed.values()); elapsed = time.monotonic() - start
            atomic(args.output / 'STATUS.json', {'status': 'BUILDING', 'pid': os.getpid(), 'rows_done': done,
                   'rows': sum(EXPECTED_ROWS.values()), 'elapsed_s': elapsed, 'rows_per_second': done / max(elapsed, 1e-6),
                   'ready_splits': sorted(reports)})
    train = read_db(args.output / 'train.sqlite')
    overlaps = {}
    for split in ('validation', 'test'):
        train.execute('ATTACH DATABASE ? AS other', (str(args.output / (split + '.sqlite')),))
        overlaps['train_' + split] = train.execute('SELECT COUNT(*) FROM other.rows o WHERE EXISTS (SELECT 1 FROM main.rows t WHERE t.semantic_scene_sha256=o.semantic_scene_sha256)').fetchone()[0]
        train.execute('DETACH DATABASE other')
    train.close()
    val = read_db(args.output / 'validation.sqlite'); val.execute('ATTACH DATABASE ? AS other', (str(args.output / 'test.sqlite'),))
    overlaps['validation_test'] = val.execute('SELECT COUNT(*) FROM other.rows o WHERE EXISTS (SELECT 1 FROM main.rows v WHERE v.semantic_scene_sha256=o.semantic_scene_sha256)').fetchone()[0]; val.close()
    assert not any(overlaps.values()), overlaps
    result = {'status': 'PASS', 'splits': reports, 'scene_family_overlaps': overlaps,
              'elapsed_s': time.monotonic() - start, 'contract_sha256': digest,
              'training_started': False, 'test_predictions_used': False}
    atomic(args.output / 'QUALITY_REPORT.json', result)
    atomic(args.output / 'STATUS.json', {'status': 'COMPLETE', 'quality_report': str(args.output / 'QUALITY_REPORT.json'), 'elapsed_s': result['elapsed_s']})
    print(json.dumps({'status': 'COMPLETE', 'elapsed_s': result['elapsed_s']}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, default=Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B"))
    p.add_argument('--workers', type=int, default=8); p.add_argument('--chunk-rows', type=int, default=20000)
    args = p.parse_args()
    assert args.workers in range(1, 17) and args.chunk_rows > 0 and args.chunk_rows % 100 == 0
    try:
        main(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        atomic(args.output / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'})
        raise
