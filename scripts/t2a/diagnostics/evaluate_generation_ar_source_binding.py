#!/usr/bin/env python3
"""Validation-only ablation of explicit count and AR source attention bias.

Prepare freezes the candidate, sources, panel and success rule before decoding.
The existing official test and P10 contracts are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import zlib

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DIAGNOSIS = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/diagnosis")
PROTOTYPE = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/source_snapshots/snapshot")
MODES = {'baseline': (0., False), 'count': (0., True), 'count_bias1': (1., True), 'count_bias2': (2., True)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    os.replace(temp, path)


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if args.checkpoint is None or args.training_contract is None:
        raise ValueError('prepare requires checkpoint and training-contract')
    sources = root / 'evaluation_source'
    sources.mkdir()
    for path in [Path(__file__), HERE / 'generation_ar_fidelity.py',
                 REPO / 'stable_audio_tools/inference/sceneplan_generation_ar_constraints.py',
                 REPO / 'stable_audio_tools/inference/sceneplan_generation_ar_vectorized.py',
                 REPO / 'stable_audio_tools/inference/sceneplan_generation_ar_precision.py']:
        shutil.copy2(path, sources / path.name)
    benchmark_path = Path("reports/generation_ar_vectorized_benchmark.json")
    benchmark = json.loads(benchmark_path.read_text())
    assert benchmark['status'] == 'PASS' and all(r['exact_tokens'] for r in benchmark['measurements'])
    assert benchmark['checkpoint_sha256'] == sha(args.checkpoint)
    assert benchmark['generator_sha256'] == sha(sources / 'sceneplan_generation_ar_vectorized.py')
    precision_proof = Path("reports/generation_ar_fp32_sdpa_cache_gate.json")
    precision_gate = json.loads(precision_proof.read_text())
    assert precision_gate['status'] == 'FP32_SDPA_GATE_PASS'
    assert precision_gate['precision_source_sha256'] == sha(sources / 'sceneplan_generation_ar_precision.py')
    frozen_manifest = PROTOTYPE / 'SOURCE_SNAPSHOT_MANIFEST.json'
    # Full file hashes make independently frozen prototype provenance explicit.
    hashes = {str(p): sha(p) for p in PROTOTYPE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    hashes.update({str(p): sha(p) for p in sources.iterdir()})
    assert str(frozen_manifest) in hashes
    input_db = sqlite3.connect(f'file:{DIAGNOSIS / "panel.sqlite"}?mode=ro&immutable=1', uri=True)
    groups = defaultdict(list)
    for ordinal, sample_id, count, template in input_db.execute('SELECT ordinal,sample_id,source_count,template_id FROM rows'):
        groups[count, template].append((hashlib.sha256(f'42:{sample_id}'.encode()).hexdigest(), ordinal))
    assert len(groups) == 16
    ordinals = sorted(o for group in groups.values() for _, o in sorted(group)[:16])
    assert len(ordinals) == len(set(ordinals)) == 256
    db = sqlite3.connect(root / 'panel.sqlite')
    db.execute(input_db.execute("SELECT sql FROM sqlite_master WHERE name='rows'").fetchone()[0])
    for ordinal in ordinals:
        row = input_db.execute('SELECT * FROM rows WHERE ordinal=?', (ordinal,)).fetchone()
        db.execute('INSERT INTO rows VALUES (' + ','.join('?' for _ in row) + ')', row)
    db.commit(); db.close(); input_db.close()
    contract = {
        'schema': 'generation_ar_source_binding_ablation_v3_fp32_native_zero', 'purpose': 'VALIDATION_DIAGNOSIS_ONLY',
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha(args.checkpoint),
        'training_contract': str(args.training_contract.resolve()), 'training_contract_sha256': sha(args.training_contract),
        'prototype': str(PROTOTYPE), 'source_sha256': hashes,
        'panel': str(root / 'panel.sqlite'), 'panel_sha256': sha(root / 'panel.sqlite'),
        'parent_panel_contract_sha256': sha(DIAGNOSIS / 'CONTRACT.json'),
        'rows': 256, 'source_template_cells': '16 cells x 16 rows; nested D0 SHA256(42:sample_id) order',
        'modes': {name: {'strength': strength, 'declared_count_constraint': count} for name, (strength, count) in MODES.items()},
        'max_plan_tokens': 512, 'batch_size': 32, 'seed': 42, 'test_used': False,
        'decoder': 'batched legal mask, same greedy choice; all four modes use the same implementation',
        'decoder_parity_proof': str(benchmark_path), 'decoder_parity_proof_sha256': sha(benchmark_path),
        'precision': 'FP32 AR including math SDPA self attention; BF16 request encoder; TF32 disabled; native P10 caller precision untouched',
        'precision_gate_proof': str(precision_proof), 'precision_gate_proof_sha256': sha(precision_proof),
        'previous_failed_experiment': os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/previous",
        'previous_failure': 'BF16 cache/full differences failed .25 cap, and source bias changed 2/89 legal decisions. FP32 with native Flash also silently cast to FP16. V2 changes AR precision rather than relaxing semantic or field acceptance.',
        'v3_fix': 'Return None when the source penalty is entirely zero, preserving native attention dispatch for single-source requests and global-only queries. V2 lost 3/64 single-source medium passes with an all-zero additive mask.',
        'hypotheses': ['Explicit request labels can prevent missing/extra sources without reference-plan information.',
                       'A soft penalty on other-source request keys can reduce source binding errors; test separately against count-only.'],
        'count_success': 'count accuracy >= .95 in every class; no class lexical F1 loss > .02 or medium spatiotemporal scene loss > .02 versus baseline',
        'binding_success': 'over count-only, mean of class-3/4 medium joint rate improves >= .03 or lexical F1 improves >= .03, with no class drop > .02 in either metric and all parse',
        'quality_limit': 'Lexical F1 is not semantic correctness; this small panel cannot establish final acceptance.',
        'gpu_scope': [0, 1, 2], 'wall_budget_s_per_mode': 1200, 'max_total_gpu_hours': 1.5,
        'gpu_gate': {'zero_strength_exact': True, 'native_p10_exact': True,
                     'bound_cached_max_abs_cap': .001, 'bound_extra_cached_error_cap': .001,
                     'extra_legal_greedy_disagreement_cap': 0., 'require_both_legal_equal': True},
    }
    previous = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/generation_ar/previous")
    contract['reused_modes'] = {mode: {'summary': str(previous / mode / 'SUMMARY.json'),
                                     'summary_sha256': sha(previous / mode / 'SUMMARY.json'),
                                     'predictions': str(previous / mode / 'predictions.sqlite'),
                                     'predictions_sha256': sha(previous / mode / 'predictions.sqlite'),
                                     'reason': 'strength=0 already returns None before the changed branch; same checkpoint, panel, precision and decode policy'}
                              for mode in ('baseline', 'count')}
    atomic(root / 'CONTRACT.json', contract)
    for p in sources.iterdir(): p.chmod(0o444)
    sources.chmod(0o555)
    print(json.dumps({'status': 'PREPARED', 'contract': str(root / 'CONTRACT.json')}), flush=True)


def gpu_gate(base, codec, binding_class, device, limits, report_path=None):
    import numpy as np
    import torch
    db = sqlite3.connect('file:' + str(Path(os.environ.get("AMBIT_CACHE_ROOT", "cache")) / "generation_ar_manifests" / "train.sqlite") + '?mode=ro&immutable=1', uri=True)
    ordinal, request, blob = db.execute('SELECT ordinal,raw_user_request,target_token_ids_u16le FROM rows WHERE source_count=2 ORDER BY target_token_count,ordinal LIMIT 1').fetchone()
    one_request = db.execute('SELECT raw_user_request FROM rows WHERE source_count=1 ORDER BY target_token_count,ordinal LIMIT 1').fetchone()[0]
    db.close()
    ids = torch.tensor(np.frombuffer(blob, dtype='<u2').astype(np.int64), device=device)[None, :-1]
    mask = torch.ones_like(ids, dtype=torch.bool)
    acoustic = torch.randn(1, 9, 320, device=device, dtype=torch.bfloat16)
    acoustic_mask = torch.ones(1, 9, device=device, dtype=torch.bool)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        context, context_mask = base.encode_requests([request], device=device)
        native = base(ids, mask, context, context_mask)
        p10_before = base.shared_transformer(acoustic, context=context, context_mask=context_mask, padding_mask=acoustic_mask, use_checkpointing=False)
        bound = binding_class(base, codec, strength=0.)
        bound.eval()
        sources = bound.request_source_ids([request], device=device)
        zero = bound(ids, mask, context, context_mask, sources)
        assert torch.equal(native, zero), 'zero-strength changed native AR'
        checks = {}
        for strength in (0., 2.):
            bound.binding_strength = strength
            full = bound(ids, mask, context, context_mask, sources)
            cache = bound.prepare_decode_cache(context, context_mask, max_plan_tokens=ids.shape[1] + 1)
            cache.ar_binding_sources = sources
            cached = torch.stack([bound.decode_step(ids[:, i], cache) for i in range(ids.shape[1])], dim=1)
            disagreement = 0
            prefix = ids[0].tolist()
            for i in range(len(prefix)):
                allowed = torch.tensor(sorted(codec.allowed_next_ids(prefix[:i+1])), device=device)
                disagreement += int(full[0, i, allowed].argmax() != cached[0, i, allowed].argmax())
            checks[str(strength)] = {'max_abs': float((full-cached).abs().max()), 'legal_greedy_disagreement': disagreement / len(prefix)}
        p10_after = bound.shared_transformer(acoustic, context=context, context_mask=context_mask, padding_mask=acoustic_mask, use_checkpointing=False)
        assert torch.equal(p10_before, p10_after), 'native P10 path changed'
    report = {'status': 'MEASURED_AWAITING_ASSERTIONS', 'dtype': 'contract-selected AR precision under external bfloat16 autocast', 'train_ordinal': ordinal,
              'zero_strength_ar_exact': True, 'native_p10_exact': True, 'cache_checks': checks, 'limits': limits, 'quality_measured': False}
    if report_path is not None: atomic(report_path, report)
    assert checks['2.0']['max_abs'] <= limits['bound_cached_max_abs_cap']
    assert checks['2.0']['max_abs'] <= checks['0.0']['max_abs'] + limits['bound_extra_cached_error_cap']
    assert checks['2.0']['legal_greedy_disagreement'] <= checks['0.0']['legal_greedy_disagreement'] + limits['extra_legal_greedy_disagreement_cap']
    if limits.get('require_both_legal_equal'):
        assert all(check['legal_greedy_disagreement'] == 0. for check in checks.values())
    class OneSourceCodec:
        def __getattr__(self,name): return getattr(codec,name)
        def allowed_next_ids(self,prefix): return codec.allowed_next_ids(prefix,min_sources=1,max_sources=1)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        native_tokens = base.generate_constrained([one_request], OneSourceCodec(), device=device, max_plan_tokens=512)
        for strength in (1., 2.):
            bound.binding_strength = strength
            assert bound.generate_constrained([one_request], OneSourceCodec(), device=device, max_plan_tokens=512) == native_tokens
    report['single_source_constrained_tokens_exact_strengths_1_2'] = True
    report['status'] = 'PASS'
    return bound, report


def worker(args):
    root = args.output.resolve()
    contract = json.loads((root / 'CONTRACT.json').read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    assert gpu in ('0', '1', '2'), 'one authorized GPU must be visible'
    run = root / args.mode
    run.mkdir(exist_ok=True)
    lock = (run / 'LOCK').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for path, digest in contract['source_sha256'].items(): assert sha(path) == digest, path
    for name in ('checkpoint', 'training_contract', 'panel'): assert sha(contract[name]) == contract[name + '_sha256']
    sys.path.insert(0, contract['prototype'])
    from stable_audio_tools.models.sceneplan_generation_ar_source_binding import SourceBoundGenerationAR
    from stable_audio_tools.models.sceneplan_transfusion_generation_ar import ScenePlanTransfusionGenerationAR
    frozen = module_at('binding_frozen_evaluation', Path(contract['prototype']) / 'scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_8k.py')
    fields = module_at('binding_fidelity', root / 'evaluation_source/generation_ar_fidelity.py')
    policy = module_at('binding_count_policy', root / 'evaluation_source/sceneplan_generation_ar_constraints.py')
    vector = module_at('binding_vectorized', root / 'evaluation_source/sceneplan_generation_ar_vectorized.py')
    precision = module_at('binding_precision', root / 'evaluation_source/sceneplan_generation_ar_precision.py')
    # Process-local AR method selection. SourceBoundGenerationAR's scoped
    # request-source context still surrounds its super() call. P10 is untouched.
    ScenePlanTransfusionGenerationAR.generate_constrained = vector.generate_constrained_vectorized
    torch = frozen.torch
    torch.set_num_threads(4); torch.manual_seed(contract['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    codec = frozen.ModelScenePlanCodecV4(frozen.CODEC_PATH)
    state = torch.load(contract['checkpoint'], map_location='cpu', weights_only=False)
    training = json.loads(Path(contract['training_contract']).read_text())
    assert state['run_contract'] == training
    model, p10 = frozen.load_p10v11_generation_ar(pad_id=codec.pad_id, verify_sha256=True, activation_checkpointing=False)
    assert codec.fingerprint == training['codec_fingerprint']
    assert p10.as_dict() == training['p10_load']
    model.load_trainable_state_dict(state['ar_adapter']); del state
    model.p10_dit.to(device=device, dtype=torch.float32)
    model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32); model.eval()
    precision.configure_float32_ar(model)
    def binding_factory(base, codec, **kwargs):
        return precision.configure_float32_ar(SourceBoundGenerationAR(base, codec, **kwargs))
    model, gate = gpu_gate(model, codec, binding_factory, device, contract['gpu_gate'], run / 'GPU_GATE.json')
    atomic(run / 'GPU_GATE.json', gate)
    mode = contract['modes'][args.mode]
    model.binding_strength = mode['strength']
    class DecodePolicy:
        def generate_constrained(self, requests, codec, **kwargs):
            if mode['declared_count_constraint']:
                return policy.generate_with_declared_source_count(model, requests, codec, **kwargs)['token_ids']
            return model.generate_constrained(requests, codec, **kwargs)
    source = sqlite3.connect(f'file:{contract["panel"]}?mode=ro&immutable=1', uri=True); source.row_factory = sqlite3.Row
    rows = [dict(r) for r in source.execute('SELECT * FROM rows ORDER BY ordinal')]; source.close()
    db = sqlite3.connect(run / 'predictions.sqlite')
    db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS results(ordinal INTEGER PRIMARY KEY,payload TEXT NOT NULL)')
    identity = {'contract_sha256': sha(root / 'CONTRACT.json'), 'mode': args.mode}
    existing = dict(db.execute('SELECT key,value FROM metadata'))
    if existing: assert existing == identity
    else: db.executemany('INSERT INTO metadata VALUES (?,?)', identity.items()); db.commit()
    done = {r[0] for r in db.execute('SELECT ordinal FROM results')}
    pending = [r for r in rows if r['ordinal'] not in done]
    started = time.monotonic()
    for offset in range(0, len(pending), contract['batch_size']):
        if time.monotonic()-started > contract['wall_budget_s_per_mode']: raise TimeoutError('mode wall budget exceeded')
        batch = pending[offset:offset+contract['batch_size']]
        outputs = frozen._generate_with_fallback(DecodePolicy(), batch, codec, device=device, max_plan_tokens=contract['max_plan_tokens'])
        for row, (tokens, error, elapsed) in zip(batch, outputs):
            raw = zlib.decompress(row['target_sceneplan_zlib'])
            assert hashlib.sha256(raw).hexdigest() == row['target_sceneplan_sha256']
            target = json.loads(raw); prediction = None
            status = 'generation_error' if tokens is None else 'ok'
            if tokens is not None:
                try: prediction = codec.decode(tokens, sample_id=row['sample_id'])
                except Exception as exc: status, error = 'parse_error', repr(exc)
            record = {'ordinal': row['ordinal'], 'sample_id': row['sample_id'], 'source_count': row['source_count'],
                      'template_id': row['template_id'], 'request': row['raw_user_request'], 'target': target,
                      'prediction': prediction, 'tokens': tokens, 'status': status, 'error': error, 'generation_sec': elapsed,
                      'decode_mode': args.mode, 'fidelity': fields.compare_fields(target, prediction),
                      'legacy_metrics': frozen.score_parsed_generation(target, prediction) if prediction is not None else {}}
            db.execute('INSERT INTO results VALUES (?,?)', (row['ordinal'], json.dumps(record, ensure_ascii=False)))
        db.commit()
        atomic(run / 'STATUS.json', {'status': 'RUNNING', 'mode': args.mode, 'rows_done': len(done)+offset+len(batch), 'rows': len(rows), 'elapsed_s': time.monotonic()-started})
    records = [json.loads(r[0]) for r in db.execute('SELECT payload FROM results ORDER BY ordinal')]
    assert [r['ordinal'] for r in records] == [r['ordinal'] for r in rows]
    def summarize(part):
        result = fields.summarize_fields([r['fidelity'] for r in part])
        result['lexical_description_token_f1_proxy'] = sum(r['legacy_metrics'].get('persistent_semantic_token_f1', 0.) for r in part)/len(part)
        return result
    summary = {'status': 'COMPLETE', 'mode': args.mode, 'policy': mode, 'contract_sha256': identity['contract_sha256'],
               'status_counts': dict(Counter(r['status'] for r in records)), 'rows': len(records), 'elapsed_s': time.monotonic()-started,
               'by_source_count': {str(n): summarize([r for r in records if r['source_count']==n]) for n in range(1, 5)}}
    atomic(run / 'SUMMARY.json', summary)
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.close()
    atomic(run / 'STATUS.json', {'status': 'COMPLETE', 'summary': str(run / 'SUMMARY.json')})
    print(json.dumps({'status': 'COMPLETE', 'mode': args.mode, 'summary': str(run / 'SUMMARY.json')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--training-contract', type=Path)
    parser.add_argument('--mode', choices=MODES)
    args = parser.parse_args()
    if args.prepare: prepare(args)
    else:
        if args.mode is None: parser.error('--mode required for worker')
        try: worker(args)
        except BaseException as exc:
            (args.output / args.mode).mkdir(parents=True, exist_ok=True)
            atomic(args.output / args.mode / 'STATUS.json', {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'})
            raise


if __name__ == '__main__': main()
