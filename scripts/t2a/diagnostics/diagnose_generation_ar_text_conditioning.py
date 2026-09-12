#!/usr/bin/env python3
"""Bounded teacher-forced field diagnostics, plus a mismatched-transcript probe.

This is not a request-acceptance score. GT prefixes are explicit diagnostics;
the independent raw-only generations remain the actual acceptance evidence.
"""
import argparse
from collections import defaultdict
import functools
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def main(root, output):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    snapshot = root / 'training_source_snapshot'; sys.path.insert(0, str(snapshot))
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import GenerationARSQLiteDataset, collate_generation_ar
    from stable_audio_tools.models.sceneplan_transfusion_generation_ar import load_p10v11_generation_ar
    from stable_audio_tools.models.sceneplan_generation_ar_lora import AdaptedGenerationAR
    from stable_audio_tools.inference.sceneplan_generation_ar_precision import configure_float32_ar
    from stable_audio_tools.data import scene_plan
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / 'TRAINING_CONFIG.json').read_text()); checkpoint = root / 'training/checkpoints/step_00008334.pt'
    refs_path = root / 'validation_inputs/baseline_panel_pairs.json'; refs = json.loads(refs_path.read_text())['pairs']
    dbpath = root / 'data_v2/validation.sqlite'; db = sqlite3.connect('file:' + str(dbpath) + '?mode=ro&immutable=1', uri=True)
    byid = {sid: (ordinal, length) for ordinal, sid, length in db.execute('SELECT ordinal,sample_id,target_token_count FROM rows')}; db.close()
    ordinals = [byid[r['id']][0] for r in sorted(refs, key=lambda r: (byid[r['id']][1], r['id']))]
    speech = [(r, c['value']) for r in refs for s in r['requirements']['sources'] for c in s['constraints'] if c['op'] == 'transcript']
    speech.sort(key=lambda item: hashlib.sha256(('text-condition-probe42/' + item[0]['id']).encode()).hexdigest())
    probe = speech[:64]; wrong_requests = {}
    for i, (row, text) in enumerate(probe):
        alternative = probe[(i + 1) % len(probe)][1]
        assert text in row['request'] and alternative != text
        wrong_requests[row['id']] = row['request'].replace(text, alternative, 1)
    identity = {'schema': 'generation_ar_text_conditioning_diagnostic_v1', 'checkpoint_sha256': sha(checkpoint),
        'panel_sha256': sha(refs_path), 'script_sha256': sha(Path(__file__)), 'snapshot_manifest_sha256': sha(snapshot / 'SOURCE_SNAPSHOT_MANIFEST.json'),
        'rows': len(refs), 'batch_size': 16, 'gpu_scope': [0], 'scoring_cap_s': 600,
        'probe': '64 hash-selected validation speech requests, cyclically replace only the quoted transcript with another English transcript; score unchanged GT prefixes. Diagnostic mismatch only, never a training pair or an acceptance input.',
        'hypothesis': 'Identify weak text prediction versus autoregressive error accumulation; measure whether transcript-token likelihood responds to the actual request text.',
        'success': 'Finite complete field/token statistics and signed paired transcript NLL differences; no optimizer or model/data mutation.',
        'test_used': False, 'acceptance_established': False}
    atomic(output / 'CONTRACT.json', identity); atomic(output / 'PROBE_INPUTS.json', {'requests': wrong_requests})
    atomic(output / 'STATUS.json', {'status': 'LOADING', 'pid': os.getpid()})
    torch.set_num_threads(4); torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0'); codec = ModelScenePlanCodecV4(config['codec'])
    base, p10 = load_p10v11_generation_ar(pad_id=codec.pad_id, verify_sha256=True, activation_checkpointing=False)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert p10.as_dict() == saved['run_contract']['p10_load'] and codec.fingerprint == saved['run_contract']['codec_fingerprint']
    base.load_trainable_state_dict(saved['ar_adapter']); model = AdaptedGenerationAR(base, codec, rank=8, alpha=8., binding_strength=0.)
    model.load_lora_state_dict(saved['ar_lora']); del base, saved
    model.p10_dit.to(device=device, dtype=torch.float32); model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32); model.ar_lora.to(device=device, dtype=torch.float32)
    configure_float32_ar(model); model.eval()
    data = GenerationARSQLiteDataset(dbpath, split='validation', row_ordinals=ordinals)
    loader = DataLoader(data, batch_size=16, shuffle=False, num_workers=0, collate_fn=functools.partial(collate_generation_ar, pad_id=codec.pad_id))
    totals = defaultdict(lambda: {'tokens': 0, 'nll_sum': 0., 'top1_correct': 0})
    names = {getattr(scene_plan, k): k.removeprefix('LOSS_').lower() for k in ('LOSS_GRAMMAR', 'LOSS_SEMANTIC', 'LOSS_ROOM', 'LOSS_SPATIAL_METRIC', 'LOSS_MOTION', 'LOSS_SPEECH_CONTENT')}
    probe_rows = []; started = time.monotonic(); complete = 0; max_request_tokens = 0
    with torch.inference_mode():
        for raw in loader:
            if time.monotonic() - started > 600: raise TimeoutError('Text-conditioning diagnostic wall budget exceeded')
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in raw.items()}
            context, mask = model.encode_requests(batch['raw_user_requests'], device=device)
            max_request_tokens = max(max_request_tokens, int(mask.sum(-1).max()))
            logits = model(batch['plan_input_ids'], batch['plan_attention_mask'], context, mask)
            assert bool(torch.isfinite(logits).all())
            loss = F.cross_entropy(logits.transpose(1, 2), batch['plan_labels'], ignore_index=-100, reduction='none')
            correct = logits.argmax(-1).eq(batch['plan_labels']); pieces = (batch['plan_labels'] >= codec.text_offset) & (batch['plan_labels'] < codec.text_offset + codec.text_processor.get_piece_size())
            for gid, name in names.items():
                group = batch['plan_loss_group_ids'].eq(gid)
                for suffix, selected in [('', group), ('_text_pieces', group & pieces)] if gid in (2, 7) else [('', group)]:
                    dest = totals[name + suffix]; dest['tokens'] += int(selected.sum()); dest['nll_sum'] += float(loss[selected].sum()); dest['top1_correct'] += int(correct[selected].sum())
            chosen = [i for i, sid in enumerate(raw['sample_ids']) if sid in wrong_requests]
            if chosen:
                wrong = [wrong_requests[raw['sample_ids'][i]] for i in chosen]
                changed_context, changed_mask = model.encode_requests(wrong, device=device)
                changed_logits = model(batch['plan_input_ids'][chosen], batch['plan_attention_mask'][chosen], changed_context, changed_mask)
                changed_loss = F.cross_entropy(changed_logits.transpose(1, 2), batch['plan_labels'][chosen], ignore_index=-100, reduction='none')
                for j, i in enumerate(chosen):
                    selected = batch['plan_loss_group_ids'][i].eq(7) & pieces[i]
                    a = float(loss[i][selected].mean()); b = float(changed_loss[j][selected].mean())
                    probe_rows.append({'id': raw['sample_ids'][i], 'transcript_tokens': int(selected.sum()), 'actual_request_nll': a, 'wrong_transcript_request_nll': b, 'wrong_minus_actual': b - a})
                del changed_logits, changed_loss, changed_context
            complete += len(raw['sample_ids']); atomic(output / 'STATUS.json', {'status': 'RUNNING', 'rows_done': complete, 'rows': len(refs), 'elapsed_s': time.monotonic() - started})
            del logits, loss, context
    assert complete == len(refs) == 1024 and len(probe_rows) == 64
    for values in totals.values():
        values['mean_nll'] = values['nll_sum'] / values['tokens']; values['top1_accuracy'] = values['top1_correct'] / values['tokens']
    result = {'status': 'COMPLETE_DIAGNOSTIC_ONLY', 'field_groups': dict(totals), 'counterfactual_probe': probe_rows,
        'probe_mean_wrong_minus_actual_nll': sum(r['wrong_minus_actual'] for r in probe_rows) / len(probe_rows),
        'probe_actual_better_rows': sum(r['wrong_minus_actual'] > 0 for r in probe_rows), 'max_actual_request_tokens': max_request_tokens,
        'elapsed_s': time.monotonic() - started, 'test_used': False, 'acceptance_established': False,
        'interpretation': 'Teacher-forced predictions use correct GT prefixes and can overestimate raw AR accuracy. Numeric-token CE is auxiliary and does not require hidden numerical completions to match at acceptance.'}
    atomic(output / 'REPORT.json', result); atomic(output / 'STATUS.json', {'status': 'COMPLETE', 'result': str(output / 'REPORT.json')})
    print(json.dumps({k: v for k, v in result.items() if k not in ('field_groups', 'counterfactual_probe')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    try: main(args.root, args.output)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True); atomic(args.output / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'}); raise
