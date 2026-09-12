#!/usr/bin/env python3
"""Measure witness CE and count/header sensitivity without training a model.

Teacher-forced scores are auxiliary diagnostics, never raw-request acceptance.
Only the six-token native header rollout is generated from raw input. Header
sweeps are controlled interventions on unrequested values, not an inference
policy or a source-count oracle used for delivered generation.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def run(args):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '1', '2'):
        raise ValueError('Exactly one authorized GPU from 0–2 is required')
    contract = json.loads(args.contract.read_text())
    assert sha(Path(__file__)) == contract['script_sha256']
    assert sha(contract['training']) == contract['training_sha256']
    assert sha(contract['annotations']) == contract['annotations_sha256']
    assert sha(contract['request_first']) == contract['request_first_sha256']
    assert sha(Path(contract['snapshot']) / 'SOURCE_SNAPSHOT_MANIFEST.json') == contract['snapshot_manifest_sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def status(stage, **kw):
        atomic(args.output / 'STATUS.json', {'status': stage, 'pid': os.getpid(), 'elapsed_s': time.monotonic() - started, **kw})

    def budget():
        if time.monotonic() - started > contract['wall_cap_seconds']:
            raise TimeoutError('Diagnostic budget exceeded; keep existing results')

    sys.path.insert(0, contract['snapshot'])
    import torch
    from torch.nn import functional as F
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.models.sceneplan_transfusion_generation_ar import load_p10v11_generation_ar
    from stable_audio_tools.models.sceneplan_generation_ar_lora import AdaptedGenerationAR
    from stable_audio_tools.inference.sceneplan_generation_ar_precision import configure_float32_ar
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    training = json.loads(Path(contract['training']).read_text())
    annotation = json.loads(Path(contract['annotations']).read_text())
    annotated = {(x['index'], x['view']): x for x in annotation['rows']}
    categories = annotation['categories']; parents = training['parents']
    batches = [training['schedule'][i] for i in contract['schedule_batch_indices']]

    def text_for(item):
        p = parents[item['index']]
        return p['precise_requests_by_view'][item['view']] if item['mode'] == 'precise' else p['natural_requests'][item['view']]

    records = {r['id']: r for r in json.loads(Path(contract['request_first']).read_text())['records']}
    header_rows = []
    for n in range(1, 5):
        candidates = [p for p in parents if p['route'] == 'request_to_plan' and p['source_count'] == n
                      and not records[p['id']]['requirements'][0]['scene']
                      and all('requested_interval' not in e for e in records[p['id']]['intent']['events'])]
        candidates.sort(key=lambda p: hashlib.sha256(('header-audit-42/' + p['id']).encode()).hexdigest())
        assert len(candidates) >= contract['header_rows_per_count']
        header_rows.extend(candidates[:contract['header_rows_per_count']])
    texts = sorted({text_for(x) for batch in batches for x in batch} | {p['natural_requests'][0] for p in header_rows})
    status('LOADING')
    codec = ModelScenePlanCodecV4(contract['codec'])
    assert codec.fingerprint == contract['codec_fingerprint']
    base, p10 = load_p10v11_generation_ar(pad_id=codec.pad_id, verify_sha256=True, activation_checkpointing=False)
    model = AdaptedGenerationAR(base, codec, rank=8, alpha=8., binding_strength=0.); del base
    model.p10_dit.to(device=device, dtype=torch.float32); model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32); model.ar_lora.to(device=device, dtype=torch.float32)
    configure_float32_ar(model); model.eval()
    cache = {}
    with torch.inference_mode():
        for offset in range(0, len(texts), 32):
            budget(); part = texts[offset:offset + 32]
            ctx, mask = model.encode_requests(part, device=device)
            for i, text in enumerate(part):
                length = int(mask[i].nonzero().flatten()[-1]) + 1
                cache[text] = (ctx[i, :length].cpu(), mask[i, :length].cpu())
    status('CONTEXT_READY', requests=len(texts))

    def contexts(texts):
        rows = [cache[t] for t in texts]
        width = max(x.shape[0] for x, _ in rows)
        ctx = torch.zeros(len(rows), width, rows[0][0].shape[-1], device=device)
        mask = torch.zeros(len(rows), width, dtype=torch.bool, device=device)
        for i, (x, m) in enumerate(rows):
            ctx[i, :len(x)] = x.to(device); mask[i, :len(m)] = m.to(device)
        return ctx, mask

    count_ids = torch.tensor([codec.token_to_id[f'<num_sources_{n}>'] for n in range(1, 5)], device=device)
    all_results = {}
    for candidate in contract['candidates']:
        budget(); assert sha(candidate['checkpoint']) == candidate['sha256']
        saved = torch.load(candidate['checkpoint'], map_location='cpu', weights_only=False)
        assert saved['run_contract']['p10_load'] == p10.as_dict()
        assert saved['run_contract']['codec_fingerprint'] == codec.fingerprint
        model.load_trainable_state_dict(saved['ar_adapter']); model.load_lora_state_dict(saved['ar_lora']); del saved
        model.eval(); stats = defaultdict(lambda: defaultdict(float)); tf_counts = defaultdict(lambda: [0, 0])
        with torch.inference_mode():
            for step, items in enumerate(batches):
                budget()
                token_arrays = [parents[x['index']]['target_token_ids_by_view'][x['view']] for x in items]
                width = max(map(len, token_arrays))
                tokens = torch.full((len(items), width), codec.pad_id, dtype=torch.long, device=device)
                group_ids = torch.full((len(items), width - 1), -1, dtype=torch.long, device=device)
                for i, (item, values) in enumerate(zip(items, token_arrays)):
                    tokens[i, :len(values)] = torch.tensor(values, device=device)
                    key = 'precise' if item['mode'] == 'precise' else 'natural'
                    groups = annotated[item['index'], item['view']][key]
                    assert len(groups) == len(values) - 1
                    group_ids[i, :len(groups)] = torch.tensor(groups, device=device)
                ctx, mask = contexts([text_for(x) for x in items])
                logits = model(tokens[:, :-1], tokens[:, :-1] != codec.pad_id, ctx, mask)
                labels = tokens[:, 1:].clone(); labels[labels == codec.pad_id] = -100
                losses = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction='none')
                logp = logits.log_softmax(-1)
                probabilities = logp.exp()
                p_y = probabilities.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                grad_l1 = 2. * (1. - p_y)
                grad_l2_sq = (probabilities.square().sum(-1) - 2. * p_y + 1.).clamp_min(0.)
                correct = logits.argmax(-1) == labels
                denominator = int((labels != -100).sum())
                for group, category in enumerate(categories):
                    keep = group_ids == group; count = int(keep.sum())
                    if not count:
                        continue
                    row = stats[category]
                    row['tokens'] += count; row['ce_sum'] += float(losses[keep].sum())
                    row['correct'] += int(correct[keep].sum())
                    row['mean_update_ce_contribution'] += float(losses[keep].sum()) / denominator / len(batches)
                    row['mean_update_logit_gradient_l1'] += float(grad_l1[keep].sum()) / denominator / len(batches)
                    row['mean_update_logit_gradient_l2_squared'] += float(grad_l2_sq[keep].sum()) / denominator**2 / len(batches)
                count_prediction = logits[:, 5, count_ids].argmax(-1) + 1
                for i, item in enumerate(items):
                    n = parents[item['index']]['source_count']; key = f"{item['mode']}/{n}"
                    tf_counts[key][0] += int(count_prediction[i] == n); tf_counts[key][1] += 1
                del logits, logp, probabilities, losses
                if step == 0 or (step + 1) % 8 == 0:
                    status('MEASURING_TEACHER_FORCED_LOSS', candidate=candidate['name'], batch=step + 1, batches=len(batches))

            ctx, cmask = contexts([p['natural_requests'][0] for p in header_rows])
            native = torch.full((len(header_rows), 1), codec.bos_id, device=device, dtype=torch.long)
            for _ in range(6):
                budget()
                logits = model(native, torch.ones_like(native, dtype=torch.bool), ctx, cmask)[:, -1]
                allowed = torch.zeros_like(logits, dtype=torch.bool)
                for i, prefix in enumerate(native.tolist()):
                    allowed[i, sorted(codec.allowed_next_ids(prefix))] = True
                value = logits.masked_fill(~allowed, -torch.inf).argmax(-1)
                native = torch.cat([native, value[:, None]], dim=1)
            header_outputs = []
            for i, parent in enumerate(header_rows):
                prefixes = [parent['target_token_ids_by_view'][0][:6], native[i, :6].tolist()]
                labels = ['witness', 'native']
                for duration in contract['sweep_duration_seconds']:
                    frame = codec.frame_ids[codec._frame_from_seconds(duration, mode='nearest')]
                    for room in contract['sweep_rooms']:
                        prefixes.append([codec.bos_id, codec.token_to_id['<duration_frames>'], frame,
                                         codec.token_to_id['<room>'], codec.token_to_id[f'<room_{room}>'],
                                         codec.token_to_id['<num_sources>']])
                        labels.append(f'{duration}s/{room}')
                prefix = torch.tensor(prefixes, device=device)
                c, m = contexts([parent['natural_requests'][0]] * len(prefixes))
                logits = model(prefix, torch.ones_like(prefix, dtype=torch.bool), c, m)[:, -1, count_ids]
                ps = logits.softmax(-1).cpu().tolist(); predicted = (logits.argmax(-1) + 1).cpu().tolist()
                native_count = count_ids.cpu().tolist().index(int(native[i, 6])) + 1
                header_outputs.append({'id': parent['id'], 'request': parent['natural_requests'][0],
                                       'expected_count_for_scoring_only': parent['source_count'],
                                       'native_count': native_count, 'native_prefix': native[i, :6].tolist(),
                                       'sweeps': [{'header': label, 'prediction': pred, 'count_probabilities': p}
                                                  for label, pred, p in zip(labels, predicted, ps)],
                                       'repeat_native_prefix_count_agrees': native_count == predicted[1]})
            total_ce = sum(x['ce_sum'] for x in stats.values())
            for row in stats.values():
                row.update(mean_ce=row['ce_sum'] / row['tokens'], argmax_accuracy=row['correct'] / row['tokens'],
                           ce_share=row['ce_sum'] / total_ce)
            result = {'candidate': candidate, 'teacher_forced_categories': dict(stats),
                      'teacher_forced_count_correct_total': dict(tf_counts), 'header_sensitivity_rows': header_outputs,
                      'header_sweep_changed_count_rows': sum(len({x['prediction'] for x in r['sweeps'][2:]}) > 1 for r in header_outputs),
                      'native_header_count_correct': sum(r['native_count'] == r['expected_count_for_scoring_only'] for r in header_outputs),
                      'header_rows': len(header_outputs), 'elapsed_s': time.monotonic() - started,
                      'limits': contract['limits']}
            atomic(args.output / (candidate['name'] + '.json'), result)
            all_results[candidate['name']] = result
            status('CANDIDATE_COMPLETE', candidate=candidate['name'])
    atomic(args.output / 'REPORT.json', {'status': 'COMPLETE', 'contract_sha256': sha(args.contract), 'candidates': all_results,
                                        'elapsed_s': time.monotonic() - started, 'test_used': False, 'training_updates': 0})
    status('COMPLETE')
    print(json.dumps({'status': 'COMPLETE', 'elapsed_s': time.monotonic() - started, 'candidates': list(all_results)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    try:
        run(args)
    except BaseException as exc:
        # Never replace a completed or conflicting run's receipt.
        if args.output.exists() and not (args.output / 'REPORT.json').exists():
            atomic(args.output / 'FAILURE.json', {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'})
        raise
