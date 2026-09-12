#!/usr/bin/env python3
"""Train only the bounded literal-span head on verified frozen feature caches."""
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
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(4); torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    spec = importlib.util.spec_from_file_location('copy_pointer_training_impl', args.copy_module)
    copy = importlib.util.module_from_spec(spec); sys.modules[spec.name] = copy; spec.loader.exec_module(copy)
    root = args.root; features = root / 'features'; output = root / 'training'; output.mkdir(parents=True, exist_ok=True)
    assert json.loads((features / 'STATUS.json').read_text())['status'] == 'COMPLETE'
    feature_contract = json.loads((features / 'CONTRACT.json').read_text()); assert sha(args.copy_module) == feature_contract['copy_module_sha256']
    identity = {'schema': 'generation_ar_learned_literal_copy_training_v1', 'script_sha256': sha(Path(__file__)),
        'copy_module_sha256': sha(args.copy_module), 'feature_contract_sha256': sha(features / 'CONTRACT.json'),
        'protocol_sha256': sha(root / 'PROTOCOL.json'), 'base_checkpoint_sha256': feature_contract['checkpoint_sha256'],
        'model': {'hidden_dim': 1024, 'width': 128}, 'optimizer': 'AdamW', 'learning_rate': .001, 'weight_decay': .01,
        'formal_updates': 2048, 'batch_size': 32, 'warmup_updates': 100, 'cosine_floor': .1, 'gradient_clip': 1.,
        'loss': 'Equal mean of start and end cross-entropies over actual text-field queries; no padded-query loss.',
        'trainable': 'LiteralCopyPointer only. No P10, Qwen or base AR model is loaded in this training process.',
        'precision': 'FP32, TF32 disabled', 'deterministic_algorithms': True, 'seed': 42, 'test_used': False, 'gpu_scope': [0]}
    if (output / 'CONTRACT.json').exists(): assert json.loads((output / 'CONTRACT.json').read_text()) == identity
    else: atomic(output / 'CONTRACT.json', identity)
    atomic(output / 'STATUS.json', {'status': 'LOADING_VERIFIED_FEATURES', 'pid': os.getpid()})
    data = {}
    for split in ('train', 'validation'):
        manifest = json.loads((features / split / 'MANIFEST.json').read_text()); assert manifest['contract_sha256'] == identity['feature_contract_sha256']
        records = []
        for shard in manifest['shards']:
            assert sha(shard['path']) == shard['sha256']
            part = torch.load(shard['path'], map_location='cpu', weights_only=False)
            assert [r['id'] for r in part] == shard['ids']; records.extend(part)
        assert len(records) == manifest['rows'] == {'train': 16384, 'validation': 1024}[split]
        for row in records:
            row['alignment'] = copy.character_alignment([row['request']], [row['offsets']], [[True] * len(row['offsets'])])
        data[split] = records
    assert {r['id'] for r in data['train']}.isdisjoint(r['id'] for r in data['validation'])
    device = torch.device('cuda:0')

    def collate(rows):
        b = len(rows); t = max(len(r['context']) for r in rows); q = max(len(r['targets']) for r in rows); c = max(len(r['request']) for r in rows)
        context = torch.zeros(b, t, 1024); mask = torch.zeros(b, t).bool(); query = torch.zeros(b, q, 1024)
        kinds = torch.zeros(b, q).long(); starts = torch.zeros(b, q).long(); ends = torch.zeros(b, q).long(); valid = torch.zeros(b, q).bool()
        arrays = {'token_indices': torch.zeros(b, c).long(), 'characters': torch.zeros(b, c, 3).long(),
            'token_char_offsets': torch.zeros(b, c).long(), 'relative_positions': torch.zeros(b, c, 2), 'endpoint_mask': torch.zeros(b, c).bool()}
        for i, row in enumerate(rows):
            nt, nq, nc = len(row['context']), len(row['targets']), len(row['request'])
            context[i, :nt] = row['context']; mask[i, :nt] = True; query[i, :nq] = row['queries']; valid[i, :nq] = True
            kinds[i, :nq] = torch.tensor([t['field_type'] for t in row['targets']]); starts[i, :nq] = torch.tensor([t['start'] for t in row['targets']]); ends[i, :nq] = torch.tensor([t['end'] for t in row['targets']])
            for k, dest in arrays.items(): dest[i, :nc] = getattr(row['alignment'], k)[0]
        return tuple(v.to(device) for v in (query, kinds, context, mask, starts, ends, valid)) + (copy.CharRequestBatch(**arrays).to(device),)

    def forward(head, batch):
        query, kinds, context, mask, starts, ends, valid, alignment = batch
        a, b = head(query, kinds, context, mask, alignment)
        loss = (F.cross_entropy(a[valid], starts[valid]) + F.cross_entropy(b[valid], ends[valid])) / 2
        if not bool(torch.isfinite(loss)): raise RuntimeError('Nonfinite pointer loss')
        return loss, a, b

    def evaluate(head, records):
        counts = defaultdict(lambda: {'correct': 0, 'total': 0}); errors = []; head.eval()
        with torch.inference_mode():
            for offset in range(0, len(records), 32):
                rows = records[offset:offset + 32]; batch = collate(rows); _, a, b = forward(head, batch)
                starts, ends = copy.best_ordered_span(a, b, batch[-1].endpoint_mask)
                starts, ends = starts.cpu().tolist(), ends.cpu().tolist()
                for i, row in enumerate(rows):
                    for j, target in enumerate(row['targets']):
                        predicted = ' '.join(row['request'][starts[i][j]:ends[i][j] + 1].split()); correct = predicted == target['text']
                        for key in ('all', target['field'], f"{row['source_count']}/{target['field']}"):
                            counts[key]['correct'] += int(correct); counts[key]['total'] += 1
                        if not correct and len(errors) < 24: errors.append({'id': row['id'], 'field': target['field'], 'expected': target['text'], 'copied': predicted})
        for group in counts.values(): group['rate'] = group['correct'] / group['total']
        return {'metrics': dict(counts), 'errors': errors, 'scope': 'Correct GT-prefix queries; actual raw AR generation has not yet been tested.'}

    # Fit only training examples, then discard this head and optimizer.
    gate_path = output / 'OVERFIT_GATE.json'
    if not gate_path.exists():
        cells = defaultdict(list)
        for row in data['train']:
            key = (row['source_count'], any(t['field'] == 'transcript' for t in row['targets']))
            if len(cells[key]) < 4: cells[key].append(row)
        tiny = [row for key in sorted(cells) for row in cells[key]]; assert len(tiny) == 32
        head = copy.LiteralCopyPointer().to(device); optim = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
        batch = collate(tiny); gate = None
        for step in range(1, 201):
            head.train(); optim.zero_grad(set_to_none=True); loss, _, _ = forward(head, batch); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); assert bool(torch.isfinite(norm)); optim.step()
            if step % 20 == 0:
                scored = evaluate(head, tiny)
                atomic(output / 'STATUS.json', {'status': 'TECHNICAL_OVERFIT_GATE', 'step': step, 'maximum_steps': 200,
                    'training_only_literal_accuracy': scored['metrics']['all']['rate']})
                if scored['metrics']['all']['rate'] >= .99:
                    gate = {'status': 'PASS', 'updates': step, 'rows': 32, 'training_only': True, 'scored': scored}; break
        if gate is None:
            atomic(gate_path, {'status': 'FAIL', 'updates': 200, 'scored': scored}); raise RuntimeError('Learned span head failed its training-only overfit gate')
        atomic(gate_path, gate); del head, optim, batch
    assert json.loads(gate_path.read_text())['status'] == 'PASS'
    torch.manual_seed(42); head = copy.LiteralCopyPointer().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
    def schedule(step):
        if step < 100: return max(1e-3, (step + 1) / 100)
        return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., (step - 100) / (2048 - 100))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule); start_step = 0
    latest = output / 'LATEST_CHECKPOINT.json'; checkpoints = output / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    if latest.exists():
        receipt = json.loads(latest.read_text()); assert sha(receipt['path']) == receipt['sha256']
        saved = torch.load(receipt['path'], map_location='cpu', weights_only=False); assert saved['contract'] == identity
        head.load_state_dict(saved['copy_pointer']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler']); start_step = saved['step']
    def save(step):
        path = checkpoints / f'step_{step:08d}.pt'; temp = path.with_suffix('.tmp')
        torch.save({'copy_pointer': {k: v.detach().cpu() for k, v in head.state_dict().items()}, 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'step': step, 'contract': identity}, temp); temp.replace(path)
        atomic(latest, {'path': str(path), 'sha256': sha(path), 'step': step})
    started = time.monotonic(); loss_sum = 0.; measured = 0; previous_epoch = None
    head.train()
    for step in range(start_step, 2048):
        if time.monotonic() - started > 1800:
            save(step); raise TimeoutError('Copy head training wall budget exceeded; exact deterministic batch position saved')
        epoch = step // 512
        if epoch != previous_epoch:
            generator = torch.Generator().manual_seed(42 + epoch); order = torch.randperm(len(data['train']), generator=generator).tolist(); previous_epoch = epoch
        index = (step % 512) * 32; rows = [data['train'][i] for i in order[index:index + 32]]; batch = collate(rows)
        optimizer.zero_grad(set_to_none=True); loss, _, _ = forward(head, batch); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.); assert bool(torch.isfinite(norm)); optimizer.step(); scheduler.step()
        loss_sum += float(loss.detach()); measured += 1
        if (step + 1) % 64 == 0:
            status = {'status': 'RUNNING', 'step': step + 1, 'steps': 2048, 'mean_loss': loss_sum / measured,
                'elapsed_s': time.monotonic() - started, 'rows_seen_this_run': measured * 32, 'trainable_parameters': sum(p.numel() for p in head.parameters())}
            atomic(output / 'STATUS.json', status)
            with (output / 'metrics.jsonl').open('a') as log: log.write(json.dumps(status) + '\n')
        if (step + 1) % 512 == 0: save(step + 1)
    scored = evaluate(head, data['validation']); atomic(output / 'TEACHER_PREFIX_VALIDATION.json', scored)
    report = {'status': 'HEAD_TRAINED_RAW_GENERATION_VALIDATION_PENDING', 'steps': 2048, 'train_rows': 16384, 'exposures': 65536,
        'teacher_prefix_validation': scored, 'checkpoint': json.loads(latest.read_text()), 'elapsed_formal_s': time.monotonic() - started,
        'test_used': False, 'goal_complete': False, 'no_base_models_loaded_or_modified': True}
    atomic(output / 'RESULT.json', report); atomic(output / 'STATUS.json', {'status': 'COMPLETE', 'result': str(output / 'RESULT.json')}); print(json.dumps({k: v for k, v in report.items() if k != 'teacher_prefix_validation'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True); p.add_argument('--copy-module', type=Path, required=True); args = p.parse_args()
    try: main(args)
    except BaseException as exc:
        output = args.root / 'training'; output.mkdir(exist_ok=True); atomic(output / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'}); raise
