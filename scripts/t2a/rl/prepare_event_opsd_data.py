#!/usr/bin/env python3
"""Pin a small RL-only split from request metadata before candidate scores exist."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def normalized(value):
    return ' '.join(value.lower().split())


def pool(data, split, source_count):
    connection = sqlite3.connect('file:' + str(data / f'{split}.sqlite') + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    columns_path = data / f'{split}_columns.npz'
    if columns_path.exists():
        columns = np.load(columns_path, allow_pickle=False)
        eligible = np.flatnonzero(columns['source_counts'] == source_count)
    else:
        eligible = np.asarray([r[0] for r in connection.execute('SELECT ordinal FROM rows WHERE source_count=? ORDER BY ordinal', (source_count,))])
    ordinals = eligible[np.linspace(0, len(eligible) - 1, min(4096, len(eligible)), dtype=int)]
    result = []
    for ordinal in ordinals:
        row = connection.execute('SELECT sample_id,raw_user_request,raw_user_request_sha256,template_id,'
            'source_count,request_requirements_zlib,semantic_scene_sha256 FROM rows WHERE ordinal=?', (int(ordinal),)).fetchone()
        requirements = json.loads(zlib.decompress(row['request_requirements_zlib']))
        speech = [s for s in requirements['sources'] if s['kind'] == 'speech']
        transcripts = [c['value'] for s in speech for c in s['constraints'] if c['op'] == 'transcript']
        if any(len(value.split()) > 18 for value in transcripts):
            continue
        motion = {c['value'] for s in requirements['sources'] for c in s['constraints'] if c['op'] == 'motion'}
        group = ('speech' if speech else 'nonspeech') + ('_static' if motion == {'static'} else '_moving')
        clusters = [digest(['literal_core', normalized(s['core'])]) for s in requirements['sources']]
        clusters += [digest(['transcript', normalized(value)]) for value in transcripts]
        request = row['raw_user_request']
        assert hashlib.sha256(request.encode()).hexdigest() == row['raw_user_request_sha256']
        result.append({'sample_id': row['sample_id'], 'request': request, 'requirements': requirements,
            'source_count': source_count, 'group': group, 'semantic_scene_sha256': row['semantic_scene_sha256'],
            'literal_content_clusters': clusters, 'template_id': row['template_id'],
            'origin_split': split, 'origin_ordinal': int(ordinal)})
    connection.close()
    return sorted(result, key=lambda r: digest(['event_opsd_metadata_selection_v1', r['sample_id']]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    contract = json.loads((args.workspace / 'WORKSPACE_CONTRACT.json').read_text())
    bundle = json.loads(Path(contract['release_bundle']).read_text())
    data = Path(bundle['data']['root'])
    output = args.workspace / 'pilot_data_v1'
    assert not output.exists()
    groups = ('nonspeech_static', 'nonspeech_moving', 'speech_static', 'speech_moving')
    used_clusters, used_scenes = set(), set()
    selected = {'train': [], 'validation': []}
    for split, per_group in [('train', 2), ('validation', 4)]:
        candidates = pool(data, split, 1)
        for group in groups:
            remaining = per_group
            for row in candidates:
                if row['group'] != group or used_clusters.intersection(row['literal_content_clusters']) or row['semantic_scene_sha256'] in used_scenes:
                    continue
                selected[split].append({**row, 'split': split, 'panel': 'single_source_spatial'})
                used_clusters.update(row['literal_content_clusters'])
                used_scenes.add(row['semantic_scene_sha256'])
                remaining -= 1
                if not remaining:
                    break
            if remaining:
                raise ValueError(f'insufficient independent metadata-only cases for {split}/{group}')
    for count in (2, 3, 4):
        remaining = 2
        for row in pool(data, 'validation', count):
            if used_clusters.intersection(row['literal_content_clusters']) or row['semantic_scene_sha256'] in used_scenes:
                continue
            selected['validation'].append({**row, 'split': 'validation', 'panel': 'multi_source_retention'})
            used_clusters.update(row['literal_content_clusters'])
            used_scenes.add(row['semantic_scene_sha256'])
            remaining -= 1
            if not remaining:
                break
        if remaining:
            raise ValueError('insufficient independent multi-source retention cases')
    output.mkdir(parents=True, exist_ok=False)
    for split, rows in selected.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    report = {'schema': 'event_opsd_metadata_selected_split_v1', 'source_root': str(data),
        'selection': '4096 evenly spaced ordinals per source count, SHA256 order, metadata-only quotas',
        'rows': {key: len(rows) for key, rows in selected.items()},
        'sha256': {key: hashlib.sha256((output / f'{key}.jsonl').read_bytes()).hexdigest() for key in selected},
        'train_validation_literal_core_transcript_clusters_disjoint': True,
        'all_selected_semantic_scene_hashes_distinct': True, 'underlying_audio_asset_independence': 'NOT_PROVEN_BY_LITERAL_CLUSTER_FILTER',
        'baseline_training_exposure': 'train cases may have been seen in baseline SFT; validation may have informed prior baseline selection',
        'rl_holdout': 'No validation requests or outcomes may enter RL fitting, target search, checkpoint selection or hyperparameter tuning.',
        'selection_uses_model_output_or_audio': False, 'test8k_database_accessed': False,
        'scope': '8 single-source mechanism training cases; 16 single-source holdout + 6 multi-source retention cases; not a powered population efficacy study'}
    (output / 'MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
