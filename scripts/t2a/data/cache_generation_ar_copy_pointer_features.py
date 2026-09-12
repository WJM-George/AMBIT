#!/usr/bin/env python3
"""Cache frozen AR features for a bounded learned literal-span pilot on GPU0."""
import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2) + '\n'); temp.replace(path)


def select_training(path):
    pools = defaultdict(list)
    db = sqlite3.connect('file:' + str(path) + '?mode=ro&immutable=1', uri=True)
    for ordinal, sid, count, speech, template in db.execute("SELECT ordinal,sample_id,source_count,instr(target_loss_group_ids_i16le,x'0700')>0,template_id FROM rows"):
        key = (count, bool(speech)); score = int.from_bytes(hashlib.sha256(('copy-pilot42/' + sid).encode()).digest(), 'big')
        value = (-score, ordinal, sid, count, bool(speech), template); pool = pools[key]
        if len(pool) < 2048: heapq.heappush(pool, value)
        elif score < -pool[0][0]: heapq.heapreplace(pool, value)
    db.close(); assert len(pools) == 8 and all(len(v) == 2048 for v in pools.values())
    selected = [{'ordinal': r[1], 'id': r[2], 'source_count': r[3], 'speech': r[4], 'template': r[5]} for key in sorted(pools) for r in sorted(pools[key])]
    random.Random(42).shuffle(selected)
    assert len(selected) == 16384 and len({r['template'] for r in selected}) == 100
    return selected


def main(args):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '0'
    base = args.base_root; snapshot = base / 'training_source_snapshot'; sys.path.insert(0, str(snapshot))
    import torch
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import GenerationARSQLiteDataset, collate_generation_ar
    from stable_audio_tools.models.sceneplan_transfusion_generation_ar import load_p10v11_generation_ar
    from stable_audio_tools.models.sceneplan_generation_ar_lora import AdaptedGenerationAR
    from stable_audio_tools.inference.sceneplan_generation_ar_precision import configure_float32_ar
    spec = importlib.util.spec_from_file_location('frozen_literal_copy_features', args.copy_module)
    copy = importlib.util.module_from_spec(spec); sys.modules[spec.name] = copy; spec.loader.exec_module(copy)
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    atomic(out / 'STATUS.json', {'status': 'SELECTING_AND_CHECKING_LITERAL_SUPERVISION', 'pid': os.getpid()})
    checkpoint = base / 'training/checkpoints/step_00008334.pt'; config = json.loads((base / 'TRAINING_CONFIG.json').read_text())
    identity = {'schema': 'generation_ar_copy_pointer_features_v1', 'checkpoint_sha256': sha(checkpoint),
        'snapshot_manifest_sha256': sha(snapshot / 'SOURCE_SNAPSHOT_MANIFEST.json'), 'script_sha256': sha(Path(__file__)),
        'copy_module_sha256': sha(args.copy_module), 'full_data_quality_report_sha256': sha(base / 'data_v2/QUALITY_REPORT.json'),
        'train_rows': 16384, 'validation_rows': 1024, 'seed': 42, 'batch_size': 16, 'shard_rows': 64,
        'selection': '2048 hash-selected training rows in each source-count x speech-presence cell; all100 templates. Existing fixed1024 validation panel.',
        'feature_precision': 'FP32 frozen request context and FP32 output-normalized AR hidden state at GT <text_begin>. Head training only; GT prefixes are not raw-generation validation.',
        'source_data_unchanged': True, 'test_used': False, 'gpu_scope': [0], 'cache_byte_cap': 24 * (1 << 30)}
    contract = out / 'CONTRACT.json'
    if contract.exists(): assert json.loads(contract.read_text()) == identity
    else: atomic(contract, identity)
    selection_path = out / 'SELECTION.json'
    if selection_path.exists(): selection = json.loads(selection_path.read_text())
    else:
        selected = select_training(base / 'data_v2/train.sqlite')
        validation = json.loads((base / 'validation_inputs/baseline_panel_pairs.json').read_text())['pairs']
        db = sqlite3.connect('file:' + str(base / 'data_v2/validation.sqlite') + '?mode=ro&immutable=1', uri=True)
        indices = {sid: ordinal for ordinal, sid in db.execute('SELECT ordinal,sample_id FROM rows')}; db.close()
        selection = {'train': selected, 'validation': [{'ordinal': indices[r['id']], 'id': r['id'], 'source_count': r['source_count']} for r in validation]}
        atomic(selection_path, selection)
    codec = ModelScenePlanCodecV4(config['codec']); dataset = {}; targets = {}; proof = {}
    for split, selected in selection.items():
        data = GenerationARSQLiteDataset(base / f'data_v2/{split}.sqlite', split=split, row_ordinals=[r['ordinal'] for r in selected])
        dataset[split] = data; labels = {}; counts = Counter(); alternatives = 0
        for i in range(len(data)):
            row = data[i]; assert row['sample_id'] == selected[i]['id']
            fields = copy.literal_field_targets(codec, row['target_token_ids'], row['raw_user_request'])
            labels[row['sample_id']] = fields
            counts.update(t['field'] for t in fields); alternatives += sum(len(t['alternative_spans']) > 1 for t in fields)
        targets[split] = labels; proof[split] = {'rows': len(data), 'fields': dict(counts), 'repeated_literal_fields': alternatives, 'all_literal_targets_valid': True}
    assert set(targets['train']).isdisjoint(targets['validation'])
    atomic(out / 'LITERAL_SUPERVISION_GATE.json', {'status': 'PASS', 'splits': proof, 'selection_sha256': sha(selection_path), 'test_used': False})
    if shutil.disk_usage(out).free < 28 * (1 << 30): raise RuntimeError('Copy feature pilot requires 28GiB free disk budget')
    torch.set_num_threads(4); torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    atomic(out / 'STATUS.json', {'status': 'LOADING_FROZEN_AR', 'pid': os.getpid()})
    model_base, p10 = load_p10v11_generation_ar(pad_id=codec.pad_id, verify_sha256=True, activation_checkpointing=False)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert p10.as_dict() == saved['run_contract']['p10_load'] and codec.fingerprint == saved['run_contract']['codec_fingerprint']
    model_base.load_trainable_state_dict(saved['ar_adapter']); model = AdaptedGenerationAR(model_base, codec, rank=8, alpha=8., binding_strength=0.)
    model.load_lora_state_dict(saved['ar_lora']); del model_base, saved
    device = torch.device('cuda:0')
    model.p10_dit.to(device=device, dtype=torch.float32); model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32); model.ar_lora.to(device=device, dtype=torch.float32); configure_float32_ar(model); model.eval()
    captured = {}
    hook = model.ar_adapter.output_norm.register_forward_hook(lambda _module, _args, value: captured.update(hidden=value.detach()))
    started = time.monotonic(); manifests = {}; bytes_written = 0
    with torch.inference_mode():
        for split in ('validation', 'train'):
            directory = out / split; directory.mkdir(exist_ok=True); manifest = []
            for begin in range(0, len(dataset[split]), 64):
                if time.monotonic() - started > 1200: raise TimeoutError('Copy-feature cache wall budget exceeded')
                path = directory / f'shard_{begin // 64:05d}.pt'; receipt_path = path.with_suffix('.json')
                expected_ids = [r['id'] for r in selection[split][begin:begin + 64]]
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text()); assert receipt['ids'] == expected_ids and receipt['sha256'] == sha(path)
                else:
                    records = []
                    for offset in range(begin, min(begin + 64, len(dataset[split])), 16):
                        raw = [dataset[split][i] for i in range(offset, min(offset + 16, len(dataset[split])))]; batch = collate_generation_ar(raw, pad_id=codec.pad_id)
                        context, mask = model.encode_requests(batch['raw_user_requests'], device=device)
                        offsets = model.prompt_conditioner.tokenizer(batch['raw_user_requests'], add_special_tokens=True, padding=True, truncation=False,
                            return_offsets_mapping=True, return_tensors='pt')
                        assert torch.equal(torch.as_tensor(offsets['attention_mask']).bool(), mask.cpu())
                        logits = model(batch['plan_input_ids'].to(device), batch['plan_attention_mask'].to(device), context, mask)
                        hidden = captured.pop('hidden'); assert bool(torch.isfinite(hidden).all()) and bool(torch.isfinite(context).all())
                        for i, row in enumerate(raw):
                            fields = targets[split][row['sample_id']]; selected = mask[i].cpu()
                            token_offsets = offsets['offset_mapping'][i][selected].tolist()
                            alignment = copy.character_alignment([row['raw_user_request']], [token_offsets], [[True] * len(token_offsets)])
                            assert all(bool(alignment.endpoint_mask[0, t[k]]) for t in fields for k in ('start', 'end'))
                            records.append({'id': row['sample_id'], 'request': row['raw_user_request'], 'ordinal': row['ordinal'], 'source_count': row['source_count'],
                                'template': row['template_id'], 'context': context[i][mask[i]].float().cpu().contiguous(),
                                'queries': hidden[i, [t['query_position'] for t in fields]].float().cpu().contiguous(), 'offsets': token_offsets, 'targets': fields})
                        del logits, hidden, context
                    assert [r['id'] for r in records] == expected_ids
                    temp = path.with_suffix('.pt.tmp'); torch.save(records, temp); temp.replace(path)
                    receipt = {'path': str(path), 'sha256': sha(path), 'rows': len(records), 'ids': expected_ids, 'bytes': path.stat().st_size, 'split': split}
                    atomic(receipt_path, receipt)
                bytes_written += receipt['bytes']; assert bytes_written <= identity['cache_byte_cap']
                manifest.append(receipt)
                atomic(out / 'STATUS.json', {'status': 'CACHING_FROZEN_FEATURES', 'split': split, 'rows_done': min(begin + 64, len(dataset[split])),
                    'rows': len(dataset[split]), 'elapsed_s': time.monotonic() - started, 'cache_bytes': bytes_written})
            atomic(directory / 'MANIFEST.json', {'split': split, 'shards': manifest, 'rows': len(dataset[split]), 'contract_sha256': sha(contract)})
            manifests[split] = str(directory / 'MANIFEST.json')
    hook.remove()
    report = {'status': 'COMPLETE_FEATURE_CACHE_ONLY', 'manifests': manifests, 'cache_bytes': bytes_written,
        'elapsed_s': time.monotonic() - started, 'test_used': False, 'model_trained': False, 'acceptance_established': False}
    atomic(out / 'CACHE_REPORT.json', report); atomic(out / 'STATUS.json', {'status': 'COMPLETE', 'report': str(out / 'CACHE_REPORT.json')}); print(json.dumps(report))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base-root', 'copy-module', 'output'): p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    try: main(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True); atomic(args.output / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'}); raise
