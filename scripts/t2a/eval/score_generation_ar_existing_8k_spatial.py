#!/usr/bin/env python3
"""Stream native FOA diagnostics with the existing benchmark's frozen methods.

AR audio is measured against its own generated plan. Unrequested GT numerical
completions are not an acceptance target. Existing audio/scores remain read-only.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_methods(root, freeze):
    sys.path.insert(0, freeze['model_snapshot'])
    path = root / 'metric_source/scripts/t2a/eval/score_sceneplan_dit_p10_core.py'
    assert sha(path) == freeze['files_sha256'][str(path)]
    name = 'scripts.t2a.eval.score_sceneplan_dit_p10_core'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    return module


def score_audio(methods, audio_path, plan, samples, frames):
    audio, rate = methods.load_foa(audio_path, expected_samples=samples)
    assert rate == 44100 and int(frames) == (int(samples) + 1023) // 1024
    args = dict(model_num_samples=samples, latent_frames=frames)
    return {'generated_doa': methods._doa_metrics(audio, plan, **args),
            'generated_activity': methods._activity_metrics(audio, plan, **args)}


def gate(root, freeze, protocol, methods):
    old_rows = [json.loads(line) for line in Path(protocol['existing_per_output']).read_text().splitlines()]
    mapping = {row['id']: row for row in read(root / 'BENCHMARK_MAPPING.json')['rows']}
    panel = {row['sample_id']: row for row in map(json.loads, Path(protocol['panel']).read_text().splitlines())}
    checks = []
    for count in range(1, 5):
        row = next(row for row in old_rows if row['source_count'] == count)
        mapped = mapping[row['sample_id']]
        metadata = read(mapped['existing_gt_plan_p10_metadata'])
        assert sha(mapped['existing_gt_plan_p10_foa']) == mapped['existing_gt_plan_p10_foa_sha256']
        result = score_audio(methods, mapped['existing_gt_plan_p10_foa'], panel[row['sample_id']]['scene_plan'],
                             metadata['model_num_samples'], metadata['latent_frames'])
        # CPU reduction thread counts can change auxiliary RMS/dB at the last
        # floating-point bits. Request-facing activity metrics stay exact.
        primary = ('expected_active_frames', 'detected_active_frames', 'temporal_iou',
                   'onset_abs_error_sec', 'offset_abs_error_sec')
        aux = ('energy_threshold', 'active_rms', 'inactive_rms', 'active_to_inactive_db')
        activity = result['generated_activity']; previous = row['generated_activity']
        checks.append({'source_count': count, 'id': row['sample_id'],
                       'exact_existing_doa_parity': result['generated_doa'] == row['generated_doa'],
                       'exact_existing_activity_primary_parity': all(activity[key] == previous[key] for key in primary),
                       'auxiliary_reduction_parity_atol_1e_6': all(activity[key] == previous[key] or
                           (activity[key] is not None and previous[key] is not None and
                            math.isclose(activity[key], previous[key], rel_tol=0., abs_tol=1e-6)) for key in aux)})
    ok = all(row['exact_existing_doa_parity'] and row['exact_existing_activity_primary_parity'] and
             row['auxiliary_reduction_parity_atol_1e_6'] for row in checks)
    save(root / 'spatial/CPU_GATE.json', {'status': 'PASS' if ok else 'FAIL', 'checks': checks,
         'protocol_sha256': sha(root / 'spatial/PROTOCOL.json'), 'test_model_changed': False,
         'scope': 'Four existing outputs, one per source-count group; exact DOA/activity primary parity, auxiliary CPU reduction tolerance 1e-6.'})
    assert ok, checks


def aggregate(rows, methods):
    keys = {'generated_plan_doa_error_deg': ('generated_doa', 'spherical_error_mean_deg'),
            'generated_valid_direction_fraction': ('generated_doa', 'valid_direction_fraction'),
            'generated_activity_iou': ('generated_activity', 'temporal_iou'),
            'generated_activity_onset_error_sec': ('generated_activity', 'onset_abs_error_sec'),
            'generated_activity_offset_error_sec': ('generated_activity', 'offset_abs_error_sec')}
    return {'rows': len(rows), **{key: methods.summarize(row[group][field] for row in rows)
                                 for key, (group, field) in keys.items()},
            'exactly_one_active_source_frames': sum(row['generated_doa']['active_frames'] for row in rows),
            'overlap_frames_not_assigned_to_a_source': sum(row['generated_doa']['overlap_frames'] for row in rows),
            'valid_direction_frames': sum(row['generated_doa']['valid_direction_frames'] for row in rows)}


def collect(root, protocol, methods, db):
    rows = [json.loads(payload) for (payload,) in db.execute('SELECT payload FROM results ORDER BY ordinal')]
    mapping = read(root / 'BENCHMARK_MAPPING.json')['rows']
    assert len(rows) == len(mapping) == 8000 and {row['sample_id'] for row in rows} == {row['id'] for row in mapping}
    result = {'schema': 'generation_ar_native_foa_execution_diagnostics_v1', 'status': 'COMPLETE',
              'rows': len(rows), 'system_id': 'generation_ar_p10_event', 'test_used': True,
              'protocol': protocol, 'protocol_sha256': sha(root / 'spatial/PROTOCOL.json'),
              'goal_complete': False, 'domains': {}, 'by_source_count': {}}
    for domain in ('music', 'sound', 'speech'):
        chosen = [row for row in rows if domain in row['source_kinds']]
        result['domains'][domain] = {'strata': {'all': aggregate(chosen, methods), **{
            f'source_{count}': aggregate([row for row in chosen if row['source_count'] == count], methods)
            for count in range(1, 5)}}}
    for count in range(1, 5):
        result['by_source_count'][str(count)] = aggregate([row for row in rows if row['source_count'] == count], methods)
    save(root / 'spatial/NEW_AR_SPATIAL_METRICS.json', result)
    old_path = Path(protocol['existing_metrics'])
    combined = {'schema': 'generation_ar_existing_8k_spatial_comparison_v1', 'status': 'COMPLETE',
                'existing_scores_sha256': sha(old_path), 'test_used': True, 'goal_complete': False,
                'existing_gt_plan_p10_and_reference': copy.deepcopy(read(old_path)), 'generation_ar_p10': result,
                'public_mono_stereo_baselines': 'N/A: these are not native FOA outputs',
                'comparison_scope': protocol['comparison_scope']}
    save(root / 'COMBINED_SPATIAL_METRICS.json', combined)
    output = root / 'spatial/PER_OUTPUT.jsonl'
    temporary = output.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in rows))
    temporary.replace(output)
    lines = ['# Native FOA execution diagnostics', '', protocol['comparison_scope'], '',
             '| Domain | System / plan used for scoring | Scenes | Eligible DOA scenes | Plan DOA error (deg) ↓ | Activity IoU ↑ | Onset error (s) ↓ | Offset error (s) ↓ |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    old = combined['existing_gt_plan_p10_and_reference']
    def number(value):
        return 'N/A' if value is None else f'{value:.4f}'
    for domain in ('music', 'sound', 'speech'):
        new = result['domains'][domain]['strata']['all']
        previous = old['domains'][domain]['strata']['all']
        for name, data in [('Generation AR → P10 / generated plan', new), ('Existing P10 / GT plan', previous)]:
            cells = [data[key]['mean'] for key in ('generated_plan_doa_error_deg', 'generated_activity_iou',
                      'generated_activity_onset_error_sec', 'generated_activity_offset_error_sec')]
            lines.append('| ' + ' | '.join([domain, name, str(data['rows']),
                str(data['generated_plan_doa_error_deg']['count']), *map(number, cells)]) + ' |')
        lines.append('| ' + ' | '.join([domain, 'Original GT FOA / GT plan', str(previous['rows']),
            str(previous['reference_plan_doa_error_deg']['count']), number(previous['reference_plan_doa_error_deg']['mean']),
            number(previous['reference_activity_iou']['mean']), 'N/A', 'N/A']) + ' |')
    lines += ['', 'DOA includes only exactly-one-planned-active-source frames with sufficient energy/coherence; its eligible denominator varies by generated plan. Activity is the scene-level union, not individual-source detection. Confidence intervals and source-count strata are in the JSON reports. These diagnostics do not alone establish per-source audible identity or request satisfaction.', '']
    (root / 'tables').mkdir(exist_ok=True)
    (root / 'tables/spatial.md').write_text('\n'.join(lines))
    save(root / 'spatial/COMPLETE.json', {'status': 'COMPLETE', 'rows': 8000, 'test_used': True,
         'combined': str(root / 'COMBINED_SPATIAL_METRICS.json'), 'goal_complete': False})


def stream(root, freeze, protocol, methods, args):
    assert read(root / 'spatial/CPU_GATE.json')['status'] == 'PASS'
    lock = (root / 'spatial/LOCK').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = sqlite3.connect(root / 'spatial/results.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS contract(payload TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,receipt_sha256 TEXT,payload TEXT NOT NULL)')
    identity = json.dumps({'protocol_sha256': sha(root / 'spatial/PROTOCOL.json')}, sort_keys=True)
    existing = db.execute('SELECT payload FROM contract').fetchone()
    if existing:
        assert existing[0] == identity, 'Changed protocol cannot reuse spatial rows'
    else:
        db.execute('INSERT INTO contract VALUES(?)', (identity,)); db.commit()
    done = {sid: digest for sid, digest in db.execute('SELECT id,receipt_sha256 FROM results')}
    mapping = {row['id']: row for row in read(root / 'BENCHMARK_MAPPING.json')['rows']}
    assert len(mapping) == 8000
    deadline = time.monotonic() + args.wall_cap_seconds
    seen_paths = set()
    for path in root.glob('gpu*/output/*.audio.json'):
        receipt = read(path)
        if receipt.get('id') in done:
            assert sha(path) == done[receipt['id']], f'Committed receipt changed: {path}'
            assert sha(receipt['foa']) == receipt['foa_sha256'], f'Committed audio changed: {path}'
            previous = json.loads(db.execute('SELECT payload FROM results WHERE id=?', (receipt['id'],)).fetchone()[0])
            assert sha(receipt['sceneplan']) == previous['sceneplan_file_sha256'], f'Committed plan changed: {path}'
            seen_paths.add(path)
    def status(stage):
        save(root / 'spatial/STATUS.json', {'status': stage, 'rows_done': len(done), 'rows': 8000,
             'pid': os.getpid(), 'updated_unix': time.time(), 'gpu': None, 'test_used': True, 'goal_complete': False})
    status('STREAMING_COMMITTED_FOA_ON_CPU')
    while len(done) < 8000:
        assert time.monotonic() < deadline, 'Spatial worker wall-time budget exhausted'
        for path in sorted(root.glob('gpu*/output/*.audio.json')):
            if path in seen_paths:
                continue
            receipt = read(path)
            if receipt.get('status') != 'FOA_WRITTEN':
                continue
            sid = receipt['id']; mapped = mapping[sid]
            assert sid not in done, f'Duplicate ID receipt: {sid}'
            assert sha(receipt['foa']) == receipt['foa_sha256']
            assert receipt['channel_order'] == 'WYZX' and receipt['ambisonic_convention'] == 'ACN/SN3D'
            assert receipt['seed'] == mapped['noise_seed'] and receipt['sample_rate'] == 44100
            plan = read(receipt['sceneplan'])
            assert plan['sample_id'] == sid and receipt['render_identity']['p10_plan_sha256'] == plan['p10_plan_sha256']
            assert hashlib.sha256(json.dumps(plan['p10_plan'], ensure_ascii=False, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest() == plan['p10_plan_sha256']
            assert plan['non_text_fields_unchanged'] is True
            scored = score_audio(methods, receipt['foa'], plan['p10_plan'], receipt['model_num_samples'], receipt['latent_frames'])
            row = {'sample_id': sid, 'panel_id': mapped['panel_id'], 'source_count': mapped['source_count'],
                   'source_kinds': mapped['source_kinds'], 'predicted_source_count': len(plan['p10_plan']['sources']),
                   'foa_sha256': receipt['foa_sha256'], 'p10_plan_sha256': plan['p10_plan_sha256'],
                   'sceneplan_file_sha256': sha(receipt['sceneplan']), **scored}
            digest = sha(path)
            db.execute('INSERT INTO results VALUES(?,?,?,?)', (sid, mapped['ordinal'], digest, json.dumps(row)))
            db.commit(); done[sid] = digest; seen_paths.add(path)
            if len(done) % 50 == 0:
                status('STREAMING_COMMITTED_FOA_ON_CPU')
        status('STREAMING_COMMITTED_FOA_ON_CPU' if len(done) == 8000 else 'WAITING_FOR_MORE_COMMITTED_FOA')
        if len(done) < 8000:
            # A render supervisor failure does not discard healthy workers or
            # committed metrics. Its own monitor reports the failure separately.
            time.sleep(min(args.poll_seconds, max(0, deadline - time.monotonic())))
    status('AGGREGATING_SPATIAL_DIAGNOSTICS')
    collect(root, protocol, methods, db)
    status('COMPLETE')
    db.close()


def main(args):
    root = args.root.resolve()
    freeze = read(root / 'FREEZE.json')
    protocol = read(root / 'spatial/PROTOCOL.json')
    assert sha(__file__) == protocol['adapter_sha256']
    for path, digest in protocol['files_sha256'].items():
        assert sha(path) == digest, path
    methods = load_methods(root, freeze)
    if args.mode == 'gate':
        gate(root, freeze, protocol, methods)
    else:
        stream(root, freeze, protocol, methods, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=('gate', 'stream'), required=True)
    parser.add_argument('--wall-cap-seconds', type=int, default=28800)
    parser.add_argument('--poll-seconds', type=int, default=60)
    args = parser.parse_args()
    try:
        main(args)
    except BlockingIOError:
        raise  # An already active worker owns the status file.
    except BaseException as error:
        save(args.root / 'spatial/STATUS.json', {'status': 'FAILED', 'pid': os.getpid(),
             'updated_unix': time.time(), 'error': repr(error), 'test_used': True, 'goal_complete': False})
        raise
