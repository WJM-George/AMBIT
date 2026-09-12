#!/usr/bin/env python3
"""Select fresh request metadata, excluding every consumed v1 content cluster."""
import argparse
import hashlib
import json
from pathlib import Path

from prepare_event_opsd_data import digest, normalized, pool


def clusters(requirements):
    values = [digest(['literal_core', normalized(source['core'])]) for source in requirements['sources']]
    values += [digest(['transcript', normalized(constraint['value'])])
        for source in requirements['sources'] for constraint in source['constraints'] if constraint['op'] == 'transcript']
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = json.loads((args.prior_workspace / 'WORKSPACE_CONTRACT.json').read_text())
    bundle = json.loads(Path(contract['release_bundle']).read_text())
    data = Path(bundle['data']['root'])
    consumed = []
    sources = [args.prior_workspace / 'pilot_data_v1' / (name + '.jsonl') for name in ['train', 'validation']]
    for path in sources:
        consumed += [json.loads(line) for line in path.read_text().splitlines()]
    ref = args.prior_workspace / 'REFERENCE_ANNOTATIONS.json'
    consumed += json.loads(ref.read_text())['cases']
    sources.append(ref)
    used_ids = {row.get('sample_id', row.get('id')) for row in consumed}
    used_scenes = {row['semantic_scene_sha256'] for row in consumed}
    used_clusters = {value for row in consumed for value in clusters(row['requirements'])}
    excluded = {'ids': set(used_ids), 'scenes': set(used_scenes), 'clusters': set(used_clusters)}
    selected = []
    def choose(candidates, count, group=None):
        accepted = 0
        for row in candidates:
            if group is not None and row['group'] != group:
                continue
            if (row['sample_id'] in used_ids or row['semantic_scene_sha256'] in used_scenes
                    or used_clusters.intersection(row['literal_content_clusters'])):
                continue
            selected.append({**row, 'split': 'v2_heldout',
                'panel': 'single_source_spatial' if row['source_count'] == 1 else 'multi_source_retention'})
            used_ids.add(row['sample_id'])
            used_scenes.add(row['semantic_scene_sha256'])
            used_clusters.update(row['literal_content_clusters'])
            accepted += 1
            if accepted == count:
                return
        raise ValueError('insufficient fresh metadata-selected cases')
    singles = pool(data, 'validation', 1)
    for group in ['nonspeech_static', 'nonspeech_moving', 'speech_static', 'speech_moving']:
        choose(singles, 8, group)
    for count in [2, 3, 4]:
        choose(pool(data, 'validation', count), 2)
    assert len(selected) == 38
    assert not any(row['sample_id'] in excluded['ids'] or row['semantic_scene_sha256'] in excluded['scenes']
        or excluded['clusters'].intersection(row['literal_content_clusters']) for row in selected)
    args.output.mkdir(parents=True)
    path = args.output / 'validation.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected))
    report = {'contract': 'event_opsd_fresh_metadata_holdout_v2', 'source_root': str(data),
        'selected_count': len(selected), 'single_source_count': 32, 'multi_source_count': 6,
        'excluded_consumed_requests': len(excluded['ids']), 'validation_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'exclusion_sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources],
        'selection_uses_model_output_or_audio': False, 'test8k_accessed': False,
        'literal_content_and_transcript_and_scene_hashes_disjoint': True,
        'underlying_audio_asset_independence': 'NOT_PROVEN',
        'baseline_exposure': 'This split may have informed earlier baseline selection; it is fresh relative to this RL study.',
        'usage': 'Open model outputs only after training decisions and the final analysis protocol are fixed.',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output / 'MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
