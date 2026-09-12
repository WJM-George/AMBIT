#!/usr/bin/env python3
"""Add one raw-AR/P10 column to the frozen, existing 8k content benchmark.

Existing P10/baseline audio and scores are read-only. Metric implementations
come from the snapshot fixed before this candidate's test inference started.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

SYSTEM_ID = 'generation_ar_p10_event'
DISPLAY_NAME = 'Generation AR → P10 (EVENT)'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def load_frozen_metrics(root, cfg):
    # Model/metric support comes from the already frozen training tree. Load
    # each directly imported benchmark helper from its pretest snapshot.
    sys.path.insert(0, cfg['model_snapshot'])
    prefix = root / 'metric_source/scripts/t2a/eval/baselines'
    for name in ('score_p10_60k_cross_system', 'score_p10_v11_stratified_3000_public_baselines',
                 'score_p10_final_content_benchmark'):
        path = prefix / (name + '.py')
        assert sha(path) == cfg['files_sha256'][str(path)]
        module_name = 'scripts.t2a.eval.baselines.' + name
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        # Older benchmark helpers also use the adjacent module's short name.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def inputs_for_new_column(root, cfg, metrics):
    manifest = read(root / 'AUDIO_MANIFEST.json')
    assert manifest['status'] == 'COMPLETE' and len(manifest['rows']) == 8000
    assert manifest['freeze_sha256'] == sha(root / 'FREEZE.json')
    base = metrics._load_inputs(Path(cfg['existing_public_benchmark']), 'internal8k')
    new_paths = {row['panel_id']: row['new_ar_foa'] for row in manifest['rows']}
    assert set(new_paths) == set(base['panel_meta']) and len(new_paths) == 8000
    base['paths'] = {metrics.REFERENCE_ID: base['paths'][metrics.REFERENCE_ID], metrics.OURS_ID: new_paths}
    base['domain_system_ids'] = {
        domain: {system: set(base['all_domain_ids'][domain]) for system in
                 ((metrics.OURS_ID,) if domain == 'speech' else (metrics.REFERENCE_ID, metrics.OURS_ID))}
        for domain in metrics.DOMAINS}
    base['display_names'] = {metrics.REFERENCE_ID: 'Ground truth', metrics.OURS_ID: DISPLAY_NAME}
    base['conditioning'] = {metrics.REFERENCE_ID: 'reference', metrics.OURS_ID: 'qualitative raw English → learned AR → P10'}
    base['native_foa_system_ids'] = {metrics.REFERENCE_ID, metrics.OURS_ID}
    return base, manifest


def merge(root, cfg, metrics, inputs, shards):
    metrics_root = root / 'new_column_metrics'
    only_new = metrics._merge(inputs, metrics_root, shards)
    old_path = Path(cfg['existing_public_benchmark']) / 'metrics/final_content/CONTENT_METRICS.json'
    assert sha(old_path) == cfg['files_sha256'][str(old_path)]
    combined = copy.deepcopy(read(old_path))
    for domain in ('music', 'sound'):
        assert SYSTEM_ID not in combined['audio_domains'][domain]
        combined['audio_domains'][domain][SYSTEM_ID] = only_new['audio_domains'][domain][metrics.OURS_ID]
    assert SYSTEM_ID not in combined['speech']
    combined['speech'][SYSTEM_ID] = only_new['speech'][metrics.OURS_ID]
    combined.update(schema='generation_ar_existing_8k_combined_content_benchmark_v1',
        existing_scores_sha256=sha(old_path), new_column_freeze_sha256=sha(root / 'FREEZE.json'),
        new_audio_manifest_sha256=sha(root / 'AUDIO_MANIFEST.json'),
        new_metrics_adapter_sha256=sha(__file__), test_used=True, goal_complete=False,
        conditioning_disclosure=cfg['input_disclosure'],
        numerical_scope='Unrequested GT coordinates/times are not unique correct answers; content metrics use the unchanged benchmark reference texts/audio. Spatial numerical differences are separately labelled diagnostics.')
    save(root / 'COMBINED_CONTENT_METRICS.json', combined)
    tables = root / 'tables'
    tables.mkdir(exist_ok=True)
    for domain in ('music', 'sound', 'speech'):
        records = combined['speech'] if domain == 'speech' else combined['audio_domains'][domain]
        cell = metrics._speech_cell if domain == 'speech' else metrics._content_cell
        measure = 'WER ↓ / CER ↓ / UTMOS ↑' if domain == 'speech' else 'CLAP ↑ / FAD-VGGish ↓ / KL-PANN ↓'
        lines = [f'# {domain.title()}: existing 8k benchmark plus Generation AR', '', measure, '',
                 '| System | Input | All | 1 source | 2 sources | 3 sources | 4 sources |',
                 '|---|---|---:|---:|---:|---:|---:|']
        for name in [SYSTEM_ID] + [name for name in records if name != SYSTEM_ID]:
            row = records[name]
            values = [cell(row['strata'].get(key)) for key in ('all', 'source_1', 'source_2', 'source_3', 'source_4')]
            lines.append('| ' + ' | '.join([row['display_name'], row['conditioning'], *values]) + ' |')
        lines += ['', cfg['input_disclosure'], '', 'Existing system scores and audio are reused unchanged. Public mono/stereo systems have no FOA spatial score.', '']
        (tables / f'{domain}.md').write_text('\n'.join(lines))
    save(root / 'METRICS_COMPLETE.json', {'status': 'COMPLETE', 'combined': str(root / 'COMBINED_CONTENT_METRICS.json'),
         'new_system_id': SYSTEM_ID, 'existing_audio_regenerated': False, 'goal_complete': False})


def main(args):
    root = args.root.resolve()
    cfg = read(root / 'FREEZE.json')
    metric = load_frozen_metrics(root, cfg)
    inputs, manifest = inputs_for_new_column(root, cfg, metric)
    output = root / 'new_column_metrics/metrics/final_content/partials'
    output.mkdir(parents=True, exist_ok=True)
    lineage = {'freeze_sha256': sha(root / 'FREEZE.json'), 'audio_manifest_sha256': sha(root / 'AUDIO_MANIFEST.json'),
               'adapter_sha256': sha(__file__), 'only_new_model_and_needed_reference_features': True,
               'test_used': True, 'goal_complete': False}
    if args.arm == 'merge':
        merge(root, cfg, metric, inputs, args.num_shards)
        return
    import torch
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    if args.arm in ('clap', 'vggish', 'panns'):
        result = getattr(metric, '_score_' + args.arm)(inputs, device)
        path = output / (args.arm.upper() + '.json')
    else:
        result = metric._score_speech_shard(inputs, device_index=0,
                          shard_index=args.shard_index, num_shards=args.num_shards)
        path = output / f'SPEECH_SHARD_{args.shard_index:02d}_OF_{args.num_shards:02d}.json'
    result['new_column_lineage'] = lineage
    save(path, result)
    print(json.dumps({'status': 'COMPLETE', 'arm': args.arm, 'output': str(path)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=('clap', 'vggish', 'panns', 'speech', 'merge'), required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=3)
    main(parser.parse_args())
