#!/usr/bin/env python3
"""Check predicted event scopes and controls using raw-context neural heads only."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main(root):
    import torch
    torch.set_num_threads(4)
    output = root / 'learned_scope_diagnostic'
    atomic(output / 'STATUS.json', {'status': 'LOADING_VERIFIED_RAW_CONTEXTS'})
    snapshot = root / 'source_snapshot'
    inv_module = module('scope_inventory_module', snapshot / 'stable_audio_tools/models/sceneplan_generation_ar_source_inventory.py')
    qual_module = module('scope_qualitative_module', snapshot / 'stable_audio_tools/models/sceneplan_generation_ar_qualitative_head.py')
    copy = module('scope_character_alignment', snapshot / 'stable_audio_tools/models/sceneplan_generation_ar_copy_pointer.py')
    training = module('scope_raw_context_loading', snapshot / 'scripts/t2a/train/train_generation_ar_source_inventory.py')
    decoder = module('scope_raw_inventory_projection', Path(__file__).resolve().parents[3] / 'stable_audio_tools/inference/sceneplan_generation_ar_learned_copy.py')
    heads = {}
    for key, arm, constructor, model_key in (
        ('inventory', 'inventory_events', inv_module.SourceInventoryHead, 'source_inventory'),
        ('qualitative', 'scoped_without_ar_query', qual_module.QualitativeExecutionHead, 'qualitative_head')):
        receipt = json.loads((root / arm / 'training/LATEST_CHECKPOINT.json').read_text())
        assert sha(receipt['path']) == receipt['sha256']
        saved = torch.load(receipt['path'], map_location='cpu', weights_only=False)
        model = constructor(**saved['contract']['model']).eval()
        model.load_state_dict(saved[model_key])
        heads[key] = model
    assert not heads['qualitative'].use_ar_query and heads['qualitative'].source_local_attention
    contract = json.loads((root / 'features/CONTRACT.json').read_text())
    feature = contract['splits']['validation']
    assert sha(feature['labels']) == feature['labels_sha256']
    labels = {r['id']: r for r in json.loads(Path(feature['labels']).read_text())['rows']}
    manifest = json.loads(Path(feature['context_manifest']).read_text())
    records = []
    for shard in manifest['shards']:
        assert sha(shard['path']) == shard['sha256']
        part = torch.load(shard['path'], map_location='cpu', weights_only=False, mmap=True)
        assert [r['id'] for r in part] == shard['ids']
        for row in part:
            records.append({'id': row['id'], 'request': row['request'], 'context': row['context'],
                'alignment': copy.character_alignment([row['request']], [row['offsets']], [[True] * len(row['offsets'])])})
    assert len(records) == 1024
    groups = defaultdict(lambda: {'correct': 0, 'total': 0})
    span_examples = []
    errors = []
    started = time.monotonic()
    def add(count, name, correct):
        group = groups[f'{count}/{name}']
        group['correct'] += bool(correct)
        group['total'] += 1
    with torch.inference_mode():
        for offset in range(0, len(records), 32):
            rows = records[offset:offset + 32]
            batch = len(rows)
            length = max(len(row['context']) for row in rows)
            chars = max(len(row['request']) for row in rows)
            context = torch.zeros(batch, length, 1024)
            mask = torch.zeros(batch, length).bool()
            arrays = {'token_indices': torch.zeros(batch, chars).long(), 'characters': torch.zeros(batch, chars, 3).long(),
                'token_char_offsets': torch.zeros(batch, chars).long(), 'relative_positions': torch.zeros(batch, chars, 2),
                'endpoint_mask': torch.zeros(batch, chars).bool()}
            for i, row in enumerate(rows):
                context[i, :len(row['context'])] = row['context']
                mask[i, :len(row['context'])] = True
                for name, dest in arrays.items(): dest[i, :len(row['request'])] = getattr(row['alignment'], name)[0]
            alignment = copy.CharRequestBatch(**arrays)
            predicted = decoder.predict_source_inventory(heads['inventory'], inv_module, copy,
                [r['request'] for r in rows], context, mask, alignment)
            starts = torch.zeros(batch, 4).long()
            ends = torch.zeros_like(starts)
            fields = torch.zeros_like(starts)
            scopes = torch.zeros(batch, 4, 2).long()
            for i, row in enumerate(predicted):
                for j, source in enumerate(row['sources']):
                    fields[i, j] = 1 if source['kind'] == 'speech' else 0
                    starts[i, j] = alignment.token_indices[i, source['identity']['start']]
                    ends[i, j] = alignment.token_indices[i, source['identity']['end']]
                    scopes[i, j, 0] = min(alignment.token_indices[i, source['event']['start']], starts[i, j])
                    scopes[i, j, 1] = max(alignment.token_indices[i, source['event']['end']], ends[i, j])
            logits = heads['qualitative'](torch.zeros(batch, 4, 1024), fields, starts, ends, context, mask, source_spans=scopes)
            classes = {name: value.argmax(-1).tolist() for name, value in logits.items()}
            # All annotations below are scoring targets; none entered either head.
            for i, (row, prediction) in enumerate(zip(rows, predicted)):
                owned = labels[row['id']]
                assert hashlib.sha256(row['request'].encode()).hexdigest() == owned['raw_sha256']
                entries = sorted(owned['entries'], key=lambda entry: entry['start'])
                count = owned['source_count']
                add(count, 'count', prediction['count'] == count)
                scene = prediction['count'] == count
                for j, entry in enumerate(entries):
                    source = prediction['sources'][j] if j < prediction['count'] else None
                    bound = source is not None and source['kind'] == entry['source_kind'] and source['identity']['text'] == row['request'][entry['start']:entry['end'] + 1]
                    add(count, 'kind_identity', bound)
                    scene = scene and bound
                    for name in qual_module.ATTRIBUTES:
                        correct = bound and classes[name][i][j] == entry['labels'][name]
                        add(count, name, correct)
                        if name == 'radial' and entry['labels'][name] != 0:
                            add(count, 'radial/' + qual_module.ATTRIBUTES[name][entry['labels'][name]], correct)
                        scene = scene and correct
                        if not correct and len(errors) < 24:
                            errors.append({'id': row['id'], 'source': entry['source_key'], 'field': name,
                                'correct_binding': bound, 'expected': entry['labels'][name],
                                'predicted': classes[name][i][j] if source is not None else None})
                    if source is None:
                        add(count, 'scope_covers_own_controls_excludes_neighbors', False)
                        continue
                    low, high = scopes[i, j].tolist()
                    covers = all(low <= a <= b <= high for a, b in entry['evidence_tokens'].values())
                    excludes = all(high < a or low > b for k, other in enumerate(entries) if k != j
                                   for a, b in other['evidence_tokens'].values())
                    add(count, 'scope_covers_own_controls_excludes_neighbors', covers and excludes)
                    exact = [source['event']['start'], source['event']['end']] == entry['event_chars']
                    if not exact and len(span_examples) < 12:
                        span_examples.append({'id': row['id'], 'source': entry['source_key'], 'predicted_event': source['event'],
                            'teacher_event': entry['event_chars'], 'covers_own_controls': covers, 'excludes_neighbors': excludes})
                add(count, 'controls_and_inventory_joint', scene)
    for value in groups.values(): value['rate'] = value['correct'] / value['total']
    result = {'status': 'COMPLETE_DIAGNOSTIC_NOT_RAW_AR_ACCEPTANCE', 'metrics': dict(groups), 'errors': errors,
        'nonexact_event_examples': span_examples, 'elapsed_s': time.monotonic() - started,
        'model_inputs': 'Frozen unchanged raw-request context and generic character alignment. Qualitative anchors and event scopes predicted by the inventory. Zero hidden query is ignored by no-AR-query head. No GT count/kind/anchor/scope/prefix reaches model forward.',
        'limitations': 'CPU raw-context component diagnostic. No AR serialization, numerical completion, source-order-independent semantic matching or P10 audio acceptance from these rates.',
        'test_used': False, 'goal_complete': False}
    atomic(output / 'REPORT.json', result)
    atomic(output / 'STATUS.json', {'status': 'COMPLETE', 'report': str(output / 'REPORT.json')})
    print(json.dumps({key: value for key, value in groups.items() if key.endswith(('joint', 'excludes_neighbors'))}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try: main(args.root)
    except BaseException as exc:
        atomic(args.root / 'learned_scope_diagnostic/STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'})
        raise
