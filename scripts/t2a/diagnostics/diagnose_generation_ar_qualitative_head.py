#!/usr/bin/env python3
"""Compare in-sample and held-out correct-prefix errors before another head fit."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def main(root, device='cuda', model_module=None, validation_only=False):
    if device == 'cuda': assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '1', '2')
    import torch
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    output = root / 'teacher_diagnostic'; atomic(output / 'STATUS.json', {'status': 'LOADING', 'pid': os.getpid()})
    receipt = json.loads((root / 'training/LATEST_CHECKPOINT.json').read_text()); assert sha(receipt['path']) == receipt['sha256']
    saved = torch.load(receipt['path'], map_location='cpu', weights_only=False)
    path = model_module or root / 'source_snapshot/stable_audio_tools/models/sceneplan_generation_ar_qualitative_head.py'
    assert sha(path) == saved['contract']['module_sha256']
    spec = importlib.util.spec_from_file_location('qualitative_diagnostic_module', path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    model = module.QualitativeExecutionHead(**saved['contract']['model']).to(device).eval(); model.load_state_dict(saved['qualitative_head'])
    contract = json.loads((root / 'features/CONTRACT.json').read_text()); names = tuple(module.ATTRIBUTES)
    groups = defaultdict(lambda: {'correct': 0, 'total': 0}); confusions = defaultdict(lambda: defaultdict(int)); details = []; selection = {}
    started = time.monotonic()
    for split, feature in contract['splits'].items():
        if validation_only and split != 'validation': continue
        assert sha(feature['labels']) == feature['labels_sha256']
        labels = {row['id']: row for row in json.loads(Path(feature['labels']).read_text())['rows']}
        ids = set(labels)
        if split == 'train':
            ids = set()
            for n in range(1, 5):
                candidates = [sid for sid, row in labels.items() if row['source_count'] == n]
                candidates.sort(key=lambda sid: hashlib.sha256(('train_diagnostic42/' + sid).encode()).digest())
                ids.update(candidates[:256])
        selection[split] = sorted(ids); assert len(ids) == 1024
        records = []
        for shard in json.loads(Path(feature['context_manifest']).read_text())['shards']:
            if not ids.intersection(shard['ids']): continue
            assert sha(shard['path']) == shard['sha256']
            part = torch.load(shard['path'], map_location='cpu', weights_only=False, mmap=True)
            records.extend(row for row in part if row['id'] in ids)
        assert len(records) == len(ids)
        with torch.inference_mode():
            for offset in range(0, len(records), 32):
                rows = records[offset:offset + 32]; b = len(rows); t = max(len(r['context']) for r in rows); q = max(r['source_count'] for r in rows)
                context = torch.zeros(b, t, 1024); mask = torch.zeros(b, t).bool(); hidden = torch.zeros(b, q, 1024)
                fields = torch.zeros(b, q).long(); starts = torch.zeros_like(fields); ends = torch.zeros_like(fields)
                for i, row in enumerate(rows):
                    entries = labels[row['id']]['entries']; length = len(row['context']); context[i, :length] = row['context']; mask[i, :length] = True
                    for j, entry in enumerate(entries):
                        hidden[i, j] = row['queries'][entry['query_index']]; fields[i, j] = entry['field_type']
                        for key, destination in [('start', starts), ('end', ends)]:
                            destination[i, j] = max(k for k, (a, z) in enumerate(row['offsets']) if a <= entry[key] < z)
                logits = model(*(v.to(device) for v in (hidden, fields, starts, ends, context, mask)))
                scores = {name: value.cpu() for name, value in logits.items()}
                for i, row in enumerate(rows):
                    errors = []
                    for j, entry in enumerate(labels[row['id']]['entries']):
                        for name in names:
                            expected = entry['labels'][name]; predicted = int(scores[name][i, j].argmax()); correct = expected == predicted
                            for key in (f'{split}/{name}', f'{split}/{row["source_count"]}/{name}', f'{split}/{row["template"]}/{name}', f'{split}/{entry["source_kind"]}/{name}'):
                                groups[key]['correct'] += correct; groups[key]['total'] += 1
                            confusions[f'{split}/{name}'][f'{module.ATTRIBUTES[name][expected]} -> {module.ATTRIBUTES[name][predicted]}'] += 1
                            if not correct:
                                errors.append({'source_key': entry['source_key'], 'source_kind': entry['source_kind'], 'field': name,
                                    'expected': module.ATTRIBUTES[name][expected], 'predicted': module.ATTRIBUTES[name][predicted],
                                    'confidence': float(scores[name][i, j].softmax(-1)[predicted])})
                    if errors: details.append({'split': split, 'id': row['id'], 'template': row['template'], 'request': row['request'], 'errors': errors})
        atomic(output / 'STATUS.json', {'status': 'SCORING', 'split_done': split})
    for group in groups.values(): group['rate'] = group['correct'] / group['total']
    result = {'status': 'COMPLETE_DIAGNOSTIC_ONLY', 'metrics': dict(groups), 'confusions': {k: dict(v) for k, v in confusions.items()},
        'checkpoint_sha256': receipt['sha256'], 'script_sha256': sha(Path(__file__)), 'elapsed_s': time.monotonic() - started,
        'test_used': False, 'raw_acceptance_established': False, 'goal_complete': False}
    atomic(output / 'SELECTION.json', selection); atomic(output / 'ERRORS.json', {'rows': details}); atomic(output / 'REPORT.json', result)
    atomic(output / 'STATUS.json', {'status': 'COMPLETE', 'report': str(output / 'REPORT.json')})
    print(json.dumps({split: {name: groups[f'{split}/{name}'] for name in names} for split in selection}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda'); parser.add_argument('--model-module', type=Path)
    parser.add_argument('--validation-only', action='store_true'); args = parser.parse_args()
    try: main(args.root, args.device, args.model_module, args.validation_only)
    except BaseException as exc:
        atomic(args.root / 'teacher_diagnostic/STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'})
        raise
