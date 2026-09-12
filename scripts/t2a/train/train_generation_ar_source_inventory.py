#!/usr/bin/env python3
"""Bounded, resumable source-inventory training on frozen English raw contexts."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_examples(features, copy, inventory, *, include_event_span=False):
    import torch
    data = {}
    for split, feature in features['splits'].items():
        if split not in ('train', 'validation'):
            raise ValueError('Test data is excluded from this training experiment')
        assert sha(feature['labels']) == feature['labels_sha256']
        assert sha(feature['context_manifest']) == feature['context_manifest_sha256']
        labels = {row['id']: row for row in json.loads(Path(feature['labels']).read_text())['rows']}
        manifest = json.loads(Path(feature['context_manifest']).read_text())
        records = []
        for shard in manifest['shards']:
            assert sha(shard['path']) == shard['sha256']
            part = torch.load(shard['path'], map_location='cpu', weights_only=False, mmap=True)
            assert [row['id'] for row in part] == shard['ids']
            for row in part:
                owned = labels[row['id']]
                assert hashlib.sha256(row['request'].encode()).hexdigest() == owned['raw_sha256']
                targets = []
                for entry in sorted(owned['entries'], key=lambda x: x['start']):
                    literal = row['targets'][entry['query_index']]
                    assert (entry['start'], entry['end']) == (literal['start'], literal['end'])
                    assert row['request'][literal['start']:literal['end'] + 1] == literal['text']
                    transcript = None
                    if entry['source_kind'] == 'speech':
                        transcript = row['targets'][entry['query_index'] + 1]
                        assert literal['field'] == 'speaker_description' and transcript['field'] == 'transcript'
                        assert row['request'][transcript['start']:transcript['end'] + 1] == transcript['text']
                    token_ids = [i for i, (a, b) in enumerate(row['offsets'])
                                 if a < b and a <= literal['end'] and b > literal['start']]
                    assert token_ids
                    targets.append({'kind': inventory.KINDS.index(entry['source_kind']),
                        'identity': {key: literal[key] for key in ('start', 'end', 'text')},
                        'transcript': {key: transcript[key] for key in ('start', 'end', 'text')} if transcript else None,
                        'identity_tokens': (min(token_ids), max(token_ids))})
                    if include_event_span:
                        low, high = entry['event_chars']
                        assert low <= literal['start'] <= literal['end'] <= high
                        targets[-1]['event'] = {'start': low, 'end': high, 'text': row['request'][low:high + 1]}
                assert len(targets) == row['source_count'] == owned['source_count']
                records.append({'id': row['id'], 'request': row['request'], 'context': row['context'],
                    'source_count': row['source_count'], 'template': row['template'], 'targets': targets,
                    'alignment': copy.character_alignment([row['request']], [row['offsets']], [[True] * len(row['offsets'])])})
        assert len(records) == feature['rows'] == {'train': 16384, 'validation': 1024}[split]
        data[split] = records
    assert {row['id'] for row in data['train']}.isdisjoint(row['id'] for row in data['validation'])
    return data


def make_collate(copy, device, *, evidence, include_event_span=False):
    import torch
    def collate(rows):
        batch = len(rows)
        tokens = max(len(row['context']) for row in rows)
        chars = max(len(row['request']) for row in rows)
        context = torch.zeros(batch, tokens, 1024)
        mask = torch.zeros(batch, tokens).bool()
        count = torch.zeros(batch).long()
        kind = torch.full((batch, 4), 3).long()
        text_fields = ('identity', 'transcript', 'event') if include_event_span else ('identity', 'transcript')
        starts = torch.zeros(batch, 4, len(text_fields)).long()
        ends = torch.zeros_like(starts)
        valid = torch.zeros_like(starts).bool()
        regions = torch.zeros(batch, 4, tokens).bool() if evidence else None
        arrays = {'token_indices': torch.zeros(batch, chars).long(),
            'characters': torch.zeros(batch, chars, 3).long(), 'token_char_offsets': torch.zeros(batch, chars).long(),
            'relative_positions': torch.zeros(batch, chars, 2), 'endpoint_mask': torch.zeros(batch, chars).bool()}
        for i, row in enumerate(rows):
            nt, nc = len(row['context']), len(row['request'])
            context[i, :nt] = row['context']
            mask[i, :nt] = True
            count[i] = row['source_count'] - 1
            for j, target in enumerate(row['targets']):
                kind[i, j] = target['kind']
                for k, field in enumerate(text_fields):
                    value = target[field]
                    if value is not None:
                        starts[i, j, k], ends[i, j, k] = value['start'], value['end']
                        valid[i, j, k] = True
                if regions is not None:
                    a, b = target['identity_tokens']
                    regions[i, j, a:b + 1] = True
            for key, destination in arrays.items():
                destination[i, :nc] = getattr(row['alignment'], key)[0]
        inputs = (context.to(device), mask.to(device), copy.CharRequestBatch(**arrays).to(device))
        targets = tuple(x.to(device) for x in (count, kind, starts, ends, valid))
        return inputs, targets, regions.to(device) if regions is not None else None
    return collate


def main(root):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '1', '2')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    source = Path(__file__).resolve().parents[3]
    module_path = source / 'stable_audio_tools/models/sceneplan_generation_ar_source_inventory.py'
    copy_path = source / 'stable_audio_tools/models/sceneplan_generation_ar_copy_pointer.py'
    inventory = load_module('inventory_training_model', module_path)
    copy = load_module('inventory_character_alignment', copy_path)
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    budget = protocol['budget']
    variant = protocol['training_variant']
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) == variant['gpu']
    assert budget['formal_updates'] == 2048 and budget['batch_size'] == 32
    weight = float(variant['evidence_attention_weight'])
    assert 0 <= weight <= 1
    output = root / 'training'
    features = json.loads((root / 'features/CONTRACT.json').read_text())
    assert features['protocol_sha256'] == sha(root / 'PROTOCOL.json')
    assert features['inventory_module_sha256'] == sha(module_path)
    contract = {'schema': 'generation_ar_source_inventory_training_v1', 'protocol_sha256': sha(root / 'PROTOCOL.json'),
        'features_sha256': sha(root / 'features/CONTRACT.json'), 'module_sha256': sha(module_path),
        'copy_alignment_sha256': sha(copy_path), 'script_sha256': sha(Path(__file__)),
        'model': protocol['model'], 'training_variant': variant, 'budget': budget,
        'loss': 'Equal count, kind, identity endpoints, speech endpoints CE terms. Optional negative log source identity attention mass. Absent slot kind is supervised; padded spans are excluded.',
        'trainable': 'SourceInventoryHead only. No base AR, encoder, existing copy/qualitative head or P10 model loaded.',
        'teacher_prefix_or_annotations_in_forward': False, 'test_used': False}
    if (output / 'CONTRACT.json').exists():
        assert json.loads((output / 'CONTRACT.json').read_text()) == contract
    else:
        atomic(output / 'CONTRACT.json', contract)
    atomic(output / 'STATUS.json', {'status': 'LOADING_VERIFIED_RAW_CONTEXTS', 'pid': os.getpid()})
    include_event_span = protocol['model'].get('include_event_span', False)
    data = load_examples(features, copy, inventory, include_event_span=include_event_span)
    device = torch.device('cuda:0')
    collate = make_collate(copy, device, evidence=bool(weight), include_event_span=include_event_span)

    def forward(model, batch):
        inputs, target, regions = batch
        if weight:
            logits, attention = model(*inputs, return_attention=True)
        else:
            logits = model(*inputs)
        count, kind, starts, ends, valid = target
        losses = [F.cross_entropy(logits['count'], count),
                  F.cross_entropy(logits['kind'].flatten(0, 1), kind.flatten())]
        for field in range(starts.shape[-1]):
            selected = valid[..., field]
            if bool(selected.any()):
                losses.append((F.cross_entropy(logits['start'][:, :, field][selected], starts[..., field][selected]) +
                    F.cross_entropy(logits['end'][:, :, field][selected], ends[..., field][selected])) / 2)
        loss = sum(losses) / len(losses)
        if weight:
            mass = (attention * regions).sum(-1)[valid[..., 0]]
            loss = loss - weight * mass.clamp_min(1e-8).log().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite source-inventory loss')
        return loss, logits

    def evaluate(model, records):
        groups = defaultdict(lambda: {'correct': 0, 'total': 0})
        errors = []
        model.eval()
        def add(name, count, correct):
            for key in (name, f'{count}/{name}'):
                groups[key]['correct'] += int(correct)
                groups[key]['total'] += 1
        with torch.inference_mode():
            for offset in range(0, len(records), 32):
                rows = records[offset:offset + 32]
                batch = collate(rows)
                logits = model(*batch[0])
                decoded = inventory.decode_inventory(logits, [row['request'] for row in rows],
                    batch[0][2].endpoint_mask, copy.best_ordered_span)
                for row, prediction in zip(rows, decoded):
                    count = row['source_count']
                    correct_count = prediction['count'] == count
                    add('count', count, correct_count)
                    scene = correct_count
                    for index, target in enumerate(row['targets']):
                        predicted = prediction['sources'][index] if index < prediction['count'] else None
                        kind_ok = predicted is not None and predicted['kind'] == inventory.KINDS[target['kind']]
                        identity_ok = predicted is not None and ' '.join(predicted['identity']['text'].split()) == target['identity']['text']
                        for name, correct in (('kind', kind_ok), ('identity', identity_ok), ('kind_identity', kind_ok and identity_ok)):
                            add(name, count, correct)
                        scene = scene and kind_ok and identity_ok
                        transcript_ok = True
                        if target['transcript'] is not None:
                            transcript_ok = kind_ok and ' '.join(predicted['transcript']['text'].split()) == target['transcript']['text']
                            add('transcript', count, transcript_ok)
                            scene = scene and transcript_ok
                        if include_event_span:
                            event_ok = predicted is not None and predicted['event']['start'] == target['event']['start'] and predicted['event']['end'] == target['event']['end']
                            add('event_span', count, event_ok)
                        if (not kind_ok or not identity_ok or not transcript_ok) and len(errors) < 24:
                            errors.append({'id': row['id'], 'slot': index, 'count': prediction['count'],
                                'expected': target, 'predicted': predicted})
                    add('inventory_scene', count, scene)
        for group in groups.values():
            group['rate'] = group['correct'] / group['total']
        return {'metrics': dict(groups), 'errors': errors,
            'scope': 'Frozen raw context only, no GT prefix, count, kind or anchor input. Fixed mention-order diagnostic; end-to-end evaluation uses global source matching.'}

    gate_path = output / 'OVERFIT_GATE.json'
    if not gate_path.exists():
        cells = defaultdict(list)
        for row in data['train']:
            key = (row['source_count'], any(t['transcript'] is not None for t in row['targets']))
            if len(cells[key]) < 4:
                cells[key].append(row)
        tiny = [row for key in sorted(cells) for row in cells[key]]
        assert len(tiny) == 32
        model = inventory.SourceInventoryHead(**protocol['model']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        batch = collate(tiny)
        passed = False
        for step in range(1, 201):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _ = forward(model, batch)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            assert bool(torch.isfinite(norm))
            optimizer.step()
            if step % 20 == 0:
                result = evaluate(model, tiny)
                gate_fields = ('count', 'kind_identity', 'transcript', 'event_span') if include_event_span else ('count', 'kind_identity', 'transcript')
                rates = {name: result['metrics'][name]['rate'] for name in gate_fields}
                atomic(output / 'STATUS.json', {'status': 'TRAINING_ONLY_OVERFIT_GATE', 'step': step, 'rates': rates})
                if min(rates.values()) >= .99:
                    passed = True
                    break
        atomic(gate_path, {'status': 'PASS' if passed else 'FAIL', 'rows': 32, 'updates': step,
            'training_only': True, 'result': result})
        del model, optimizer, batch
    assert json.loads(gate_path.read_text())['status'] == 'PASS', 'Training-only source-inventory gate failed'
    torch.manual_seed(42)
    model = inventory.SourceInventoryHead(**protocol['model']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    def schedule(step):
        if step < 100:
            return max(.001, (step + 1) / 100)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step - 100) / (2048 - 100))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    latest = output / 'LATEST_CHECKPOINT.json'
    initial = 0
    if latest.exists():
        receipt = json.loads(latest.read_text())
        assert sha(receipt['path']) == receipt['sha256']
        saved = torch.load(receipt['path'], map_location='cpu', weights_only=False)
        assert saved['contract'] == contract
        model.load_state_dict(saved['source_inventory'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        initial = saved['step']
    def save(step):
        path = output / 'checkpoints' / f'step_{step:08d}.pt'
        path.parent.mkdir(exist_ok=True)
        temp = path.with_suffix('.tmp')
        torch.save({'source_inventory': {key: value.detach().cpu() for key, value in model.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'step': step,
            'contract': contract}, temp)
        temp.replace(path)
        atomic(latest, {'path': str(path), 'sha256': sha(path), 'step': step})
    started = time.monotonic()
    previous_epoch = None
    losses = []
    model.train()
    for step in range(initial, 2048):
        if time.monotonic() - started > 1800:
            save(step)
            raise TimeoutError('Source inventory budget reached; deterministic resume position saved')
        epoch = step // 512
        if epoch != previous_epoch:
            order = torch.randperm(len(data['train']), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
            previous_epoch = epoch
        index = step % 512 * 32
        batch = collate([data['train'][i] for i in order[index:index + 32]])
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward(model, batch)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert bool(torch.isfinite(norm))
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 64 == 0:
            status = {'status': 'RUNNING', 'step': step + 1, 'steps': 2048,
                'mean_loss': sum(losses[-64:]) / len(losses[-64:]), 'elapsed_s': time.monotonic() - started,
                'trainable_parameters': sum(p.numel() for p in model.parameters())}
            atomic(output / 'STATUS.json', status)
            with (output / 'metrics.jsonl').open('a') as log:
                log.write(json.dumps(status) + '\n')
        if (step + 1) % 512 == 0:
            save(step + 1)
    validation = evaluate(model, data['validation'])
    checks = []
    thresholds = {'count': .95, 'kind_identity': .98, 'transcript': .98}
    if include_event_span: thresholds['event_span'] = .98
    for count in range(1, 5):
        for name, minimum in thresholds.items():
            group = validation['metrics'][f'{count}/{name}']
            checks.append({'source_count': count, 'field': name, **group, 'minimum': minimum, 'pass': group['rate'] >= minimum})
    ready = all(check['pass'] for check in checks)
    result = {'status': 'READY_FOR_RAW_DECODER_GATE' if ready else 'FAIL_RAW_CONTEXT_INVENTORY_GATE',
        'raw_context_validation': validation, 'checks': checks, 'checkpoint': json.loads(latest.read_text()),
        'formal_updates': 2048, 'train_rows': 16384, 'exposures': 65536,
        'elapsed_formal_s': time.monotonic() - started, 'test_used': False, 'goal_complete': False, 'model_promoted': False}
    atomic(output / 'RESULT.json', result)
    atomic(output / 'STATUS.json', {'status': 'COMPLETE', 'ready_for_raw_decoder_gate': ready, 'result': str(output / 'RESULT.json')})
    print(json.dumps({key: value for key, value in result.items() if key not in ('raw_context_validation', 'checks')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    try:
        main(args.root)
    except BaseException as exc:
        atomic(args.root / 'training/STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION',
            'error': f'{type(exc).__name__}: {exc}', 'pid': os.getpid()})
        raise
