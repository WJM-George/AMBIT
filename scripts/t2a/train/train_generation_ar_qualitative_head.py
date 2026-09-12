#!/usr/bin/env python3
"""Fit only the small source-conditioned qualitative execution head."""
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
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


def main(root):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '1', '2')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(4); torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    path = Path(__file__).resolve().parents[3] / 'stable_audio_tools/models/sceneplan_generation_ar_qualitative_head.py'
    spec = importlib.util.spec_from_file_location('qualitative_training_module', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    protocol = json.loads((root / 'PROTOCOL.json').read_text()); budget = protocol['budget']
    assert int(os.environ['CUDA_VISIBLE_DEVICES']) == budget['gpu']
    variant = protocol.get('training_variant', {'use_ar_query': True, 'phase_balanced_loss': False})
    features = json.loads((root / 'features/CONTRACT.json').read_text()); output = root / 'training'
    output.mkdir(parents=True, exist_ok=True)
    assert json.loads((root / 'features/STATUS.json').read_text())['status'] == 'COMPLETE'
    assert features['head_module_sha256'] == sha(path) and features['protocol_sha256'] == sha(root / 'PROTOCOL.json')
    model_config = {'hidden_dim': 1024, 'width': 128, 'heads': 4, 'layers': 2, 'relative_clip': 96,
        'use_ar_query': variant['use_ar_query'], 'attribute_queries': variant.get('attribute_queries', False),
        'source_local_attention': variant.get('source_local_attention', False)}
    attention_weight = float(variant.get('evidence_attention_weight', 0.))
    assert 0 <= attention_weight <= 1 and (not attention_weight or model_config['attribute_queries'])
    contract = {'schema': 'generation_ar_learned_qualitative_head_training_v1', 'protocol_sha256': sha(root / 'PROTOCOL.json'),
        'features_contract_sha256': sha(root / 'features/CONTRACT.json'), 'module_sha256': sha(path),
        'script_sha256': sha(Path(__file__)), 'model': model_config, 'budget': budget, 'training_variant': variant,
        'trainable': 'Only the new qualitative head. No base AR, copy head, encoder or P10 model is loaded.',
        'loss': 'Mean of six request-owned categorical cross-entropies; optional training-only sqrt inverse-frequency phase and radial weights clamped to[0.5,8]. Optional field-evidence loss is negative log attention mass inside the annotated region, weighted by the preregistered training variant. Evidence targets never enter model.forward.',
        'hidden_witness_numbers_used': False, 'test_used': False}
    if (output / 'CONTRACT.json').exists(): assert json.loads((output / 'CONTRACT.json').read_text()) == contract
    else: atomic(output / 'CONTRACT.json', contract)
    atomic(output / 'STATUS.json', {'status': 'LOADING_VERIFIED_CACHED_FEATURES', 'pid': os.getpid()})
    data = {}; names = tuple(module.ATTRIBUTES)
    for split, feature in features['splits'].items():
        assert sha(feature['labels']) == feature['labels_sha256'] and sha(feature['context_manifest']) == feature['context_manifest_sha256']
        labels = {row['id']: row for row in json.loads(Path(feature['labels']).read_text())['rows']}
        manifest = json.loads(Path(feature['context_manifest']).read_text()); records = []
        for shard in manifest['shards']:
            assert sha(shard['path']) == shard['sha256']
            part = torch.load(shard['path'], map_location='cpu', weights_only=False)
            assert [row['id'] for row in part] == shard['ids']
            for row in part:
                owned = labels[row['id']]; entries = owned['entries']
                assert hashlib.sha256(row['request'].encode()).hexdigest() == owned['raw_sha256']
                def token_for_char(position):
                    return max(index for index, (start, end) in enumerate(row['offsets']) if start <= position < end)
                records.append({'id': row['id'], 'source_count': row['source_count'], 'context': row['context'],
                    'query': row['queries'][[entry['query_index'] for entry in entries]],
                    'field': torch.tensor([entry['field_type'] for entry in entries]),
                    'start': torch.tensor([token_for_char(entry['start']) for entry in entries]),
                    'end': torch.tensor([token_for_char(entry['end']) for entry in entries]),
                    'labels': torch.tensor([[entry['labels'][name] for name in names] for entry in entries]),
                    'evidence': torch.tensor([[entry['evidence_tokens'][name] for name in names] for entry in entries]) if attention_weight else None,
                    'event_spans': torch.tensor([entry['event_tokens'] for entry in entries]) if model_config['source_local_attention'] else None,
                    'speech': any(entry['source_kind'] == 'speech' for entry in entries)})
        assert len(records) == feature['rows']; data[split] = records
    assert {row['id'] for row in data['train']}.isdisjoint(row['id'] for row in data['validation'])
    device = torch.device('cuda:0')
    weights = {name: None for name in names}; class_counts = {}
    for index, name in enumerate(names):
        counts = torch.bincount(torch.cat([row['labels'][:, index] for row in data['train']]), minlength=len(module.ATTRIBUTES[name]))
        class_counts[name] = counts.tolist()
        if ((variant['phase_balanced_loss'] and name in ('onset', 'offset')) or
                (variant.get('radial_balanced_loss', False) and name == 'radial')):
            weights[name] = (counts.sum() / (len(counts) * counts.clamp_min(1))).sqrt().clamp(.5, 8.).to(device)
    atomic(output / 'CLASS_WEIGHTS.json', {'training_only_class_counts': class_counts,
        'weights': {name: value.tolist() if value is not None else None for name, value in weights.items()},
        'validation_labels_used_for_weights': False})
    def collate(rows):
        batch = len(rows); length = max(len(row['context']) for row in rows); sources = max(row['source_count'] for row in rows)
        context = torch.zeros(batch, length, 1024); mask = torch.zeros(batch, length).bool()
        query = torch.zeros(batch, sources, 1024); field = torch.zeros(batch, sources).long()
        start = torch.zeros_like(field); end = torch.zeros_like(field); labels = torch.zeros(batch, sources, len(names)).long()
        valid = torch.zeros(batch, sources).bool(); evidence = torch.zeros(batch, sources, len(names), length).bool() if attention_weight else None
        event_spans = torch.zeros(batch, sources, 2).long() if model_config['source_local_attention'] else None
        for index, row in enumerate(rows):
            n, t = row['source_count'], len(row['context'])
            context[index, :t] = row['context']; mask[index, :t] = True
            query[index, :n] = row['query']; field[index, :n] = row['field']
            start[index, :n] = row['start']; end[index, :n] = row['end']
            labels[index, :n] = row['labels']; valid[index, :n] = True
            if event_spans is not None: event_spans[index, :n] = row['event_spans']
            if evidence is not None:
                for source_index in range(n):
                    for attribute in range(len(names)):
                        low, high = row['evidence'][source_index, attribute].tolist()
                        assert 0 <= low <= high < t
                        evidence[index, source_index, attribute, low:high + 1] = True
        result = tuple(value.to(device) for value in (query, field, start, end, context, mask, labels, valid))
        return result + (evidence.to(device) if evidence is not None else None,
                         event_spans.to(device) if event_spans is not None else None)
    def forward(model, batch):
        labels, valid = batch[6:8]
        kwargs = {'source_spans': batch[9]} if model_config['source_local_attention'] else {}
        if attention_weight: result, attention = model(*batch[:6], return_attention=True, **kwargs)
        else: result = model(*batch[:6], **kwargs)
        loss = sum(F.cross_entropy(result[name][valid], labels[..., index][valid], weight=weights[name]) for index, name in enumerate(names)) / len(names)
        if attention_weight:
            # Evidence is a training target only; it never enters model.forward.
            mass = (attention * batch[8]).sum(-1)[valid]
            loss = loss - attention_weight * mass.clamp_min(1e-8).log().mean()
        if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite qualitative-head loss')
        return loss, result
    def evaluate(model, records):
        groups = defaultdict(lambda: {'correct': 0, 'total': 0}); model.eval()
        with torch.inference_mode():
            for offset in range(0, len(records), 32):
                rows = records[offset:offset + 32]; batch = collate(rows); _, logits = forward(model, batch)
                for index, name in enumerate(names):
                    correct = (logits[name].argmax(-1) == batch[6][..., index]).cpu()
                    for i, row in enumerate(rows):
                        n = row['source_count']; count = int(correct[i, :n].sum())
                        for key in ('all', name, f'{n}/{name}'):
                            groups[key]['correct'] += count; groups[key]['total'] += n
                        if name == 'radial':
                            for category, label in enumerate(module.ATTRIBUTES[name]):
                                selected = batch[6][i, :n, index].cpu() == category
                                key = f'{n}/radial/{label}'
                                groups[key]['correct'] += int(correct[i, :n][selected].sum())
                                groups[key]['total'] += int(selected.sum())
        for group in groups.values(): group['rate'] = group['correct'] / group['total'] if group['total'] else None
        return {'metrics': dict(groups), 'scope': 'Correct GT-prefix states and GT literal anchors, diagnostic only; raw own-prefix and learned-anchor execution remains untested.'}
    gate_path = output / 'OVERFIT_GATE.json'
    if not gate_path.exists():
        cells = defaultdict(list)
        for row in data['train']:
            key = (row['source_count'], row['speech'])
            if len(cells[key]) < 4: cells[key].append(row)
        tiny = [row for key in sorted(cells) for row in cells[key]]; assert len(tiny) == 32
        model = module.QualitativeExecutionHead(**model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01); batch = collate(tiny); passed = False
        for step in range(1, 201):
            model.train(); optimizer.zero_grad(set_to_none=True); loss, _ = forward(model, batch); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); assert bool(torch.isfinite(norm)); optimizer.step()
            if step % 20 == 0:
                result = evaluate(model, tiny); rate = result['metrics']['all']['rate']
                atomic(output / 'STATUS.json', {'status': 'TRAINING_ONLY_OVERFIT_GATE', 'step': step, 'field_accuracy': rate})
                if rate >= .99: passed = True; break
        atomic(gate_path, {'status': 'PASS' if passed else 'FAIL', 'rows': 32, 'updates': step, 'training_only': True, 'result': result})
        del model, optimizer, batch
    assert json.loads(gate_path.read_text())['status'] == 'PASS', 'Training-only technical gate failed'
    torch.manual_seed(42); model = module.QualitativeExecutionHead(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    def schedule(step):
        if step < 100: return max(.001, (step + 1) / 100)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step - 100) / (2048 - 100))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule); initial = 0
    latest = output / 'LATEST_CHECKPOINT.json'; checkpoints = output / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    if latest.exists():
        receipt = json.loads(latest.read_text()); assert sha(receipt['path']) == receipt['sha256']
        saved = torch.load(receipt['path'], map_location='cpu', weights_only=False); assert saved['contract'] == contract
        model.load_state_dict(saved['qualitative_head']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); initial = saved['step']
    def save(step):
        path = checkpoints / f'step_{step:08d}.pt'; temp = path.with_suffix('.tmp')
        torch.save({'qualitative_head': {key: value.detach().cpu() for key, value in model.state_dict().items()},
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'step': step, 'contract': contract}, temp); temp.replace(path)
        atomic(latest, {'path': str(path), 'sha256': sha(path), 'step': step})
    started = time.monotonic(); previous_epoch = None; losses = []; model.train()
    for step in range(initial, 2048):
        if time.monotonic() - started > 1800: save(step); raise TimeoutError('Formal head budget reached; checkpoint saved')
        epoch = step // 512
        if epoch != previous_epoch:
            order = torch.randperm(len(data['train']), generator=torch.Generator().manual_seed(42 + epoch)).tolist(); previous_epoch = epoch
        index = step % 512 * 32; batch = collate([data['train'][i] for i in order[index:index + 32]])
        optimizer.zero_grad(set_to_none=True); loss, _ = forward(model, batch); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); assert bool(torch.isfinite(norm)); optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
        if (step + 1) % 64 == 0:
            status = {'status': 'RUNNING', 'step': step + 1, 'steps': 2048, 'mean_loss': sum(losses[-64:]) / len(losses[-64:]),
                'elapsed_s': time.monotonic() - started, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
            atomic(output / 'STATUS.json', status)
            with (output / 'metrics.jsonl').open('a') as log: log.write(json.dumps(status) + '\n')
        if (step + 1) % 512 == 0: save(step + 1)
    validation = evaluate(model, data['validation']); checks = []
    for n in range(1, 5):
        for name in names:
            group = validation['metrics'][f'{n}/{name}']; checks.append({'source_count': n, 'field': name, **group, 'pass': group['rate'] >= .95})
        if variant.get('require_requested_radial_gate', False):
            for label in ('approaching', 'receding'):
                group = validation['metrics'][f'{n}/radial/{label}']
                checks.append({'source_count': n, 'field': f'radial/{label}', **group,
                    'pass': group['rate'] is not None and group['rate'] >= .95})
    ready = all(check['pass'] for check in checks)
    result = {'status': 'READY_FOR_RAW_VALIDATION' if ready else 'FAIL_CORRECT_PREFIX_GATE', 'teacher_prefix_validation': validation,
        'correct_prefix_gate': checks, 'checkpoint': json.loads(latest.read_text()), 'formal_steps': 2048,
        'train_rows': 16384, 'train_exposures': 65536, 'elapsed_formal_s': time.monotonic() - started,
        'test_used': False, 'goal_complete': False, 'model_promoted': False}
    atomic(output / 'RESULT.json', result); atomic(output / 'STATUS.json', {'status': 'COMPLETE', 'result': str(output / 'RESULT.json'), 'ready_for_raw_validation': ready})
    print(json.dumps({key: value for key, value in result.items() if key not in ('teacher_prefix_validation', 'correct_prefix_gate')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True); args = parser.parse_args()
    try: main(args.root)
    except BaseException as exc:
        atomic(args.root / 'training/STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'})
        raise
