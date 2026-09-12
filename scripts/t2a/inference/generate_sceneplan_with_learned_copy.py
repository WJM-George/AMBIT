#!/usr/bin/env python3
"""English raw requests → AR ScenePlan with optional learned planning heads."""
import argparse
import fcntl
import hashlib
import importlib.util
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


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value; spec.loader.exec_module(value); return value


def raw_items(path):
    items = json.loads(path.read_text())['requests']
    assert items and all(set(r) == {'id', 'request'} and isinstance(r['request'], str) and r['request'].strip() for r in items)
    assert len({r['id'] for r in items}) == len(items)
    return items


def main(args):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    args.output.mkdir(parents=True, exist_ok=True); lock = (args.output / 'LOCK').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    items = raw_items(args.requests)
    import torch
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    has_execution_head = 'qualitative_head' in state
    has_inventory_head = 'source_inventory' in state
    identity = {'schema': 'generation_ar_direct_raw_learned_planning_heads_v4', 'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': sha(args.checkpoint), 'snapshot': str(args.snapshot), 'snapshot_manifest_sha256': sha(args.snapshot / 'SOURCE_SNAPSHOT_MANIFEST.json'),
        'input_sha256': sha(args.requests), 'script_sha256': sha(Path(__file__)), 'copy_module_sha256': sha(args.copy_module),
        'decoder_sha256': sha(args.decoder), 'batch_size': args.batch_size, 'max_plan_tokens': args.max_plan_tokens, 'seed': 42,
        'decoder_support_sha256': sha(args.decoder.with_name('sceneplan_generation_ar_learned_copy.py')) if args.decoder.name == 'sceneplan_generation_ar_decision_blocks.py' else None,
        'benchmark_decoder_sha256': sha(args.benchmark_decoder) if args.benchmark_decoder else None,
        'benchmark_decoder_support_sha256': sha(args.benchmark_decoder.with_name('sceneplan_generation_ar_learned_copy.py')) if args.benchmark_decoder else None,
        'benchmark_only': args.benchmark_only,
        'binding_strength': 0, 'declared_count_constraint': False, 'target_or_annotation_inputs': False,
        'learned_copy_pointer': 'Learned raw-character span selection from request context and generated-prefix hidden state; no source/template parser.',
        'learned_qualitative_execution': has_execution_head,
        'learned_source_inventory': has_inventory_head,
        'inventory_module_sha256': sha(args.inventory_module) if has_inventory_head else None,
        'inventory_policy': 'When bundled, learned raw-context queries predict count, executable kinds and source-owned character spans. No teacher prefix, count, kind, annotation or template parser input. At most one speech source follows the unchanged codec grammar.',
        'qualitative_module_sha256': sha(args.qualitative_module) if has_execution_head else None,
        'qualitative_policy': 'When bundled, learned categories select seeded executable numerical completions. No evidence annotations or template parser at inference.',
        'precision': 'FP32 AR/math-SDPA and copy head, BF16 frozen request encoder, TF32 off', 'wall_cap_s': args.max_wall_seconds,
        'gate_raw_requests_sha256': sha(args.gate_requests) if args.gate_requests else None,
        'failure_policy': 'Retain finished batch members. Retry only EOS-unfinished rows once with the bundled base AR, same token limit and unchanged raw request; record primary failure and fallback. Overlong proposed literal spans fall back to the base AR text path.',
        'p10_audio_acceptance': 'PENDING'}
    contract = args.output / 'CONTRACT.json'
    if contract.exists(): assert json.loads(contract.read_text()) == identity
    else: atomic(contract, identity)
    db = sqlite3.connect(args.output / 'predictions.sqlite'); db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    done = {i: json.loads(p) for i, p in db.execute('SELECT id,payload FROM results')}; byid = {r['id']: r for r in items}
    assert set(done).issubset(byid)
    for sid, row in done.items():
        assert row['request'] == byid[sid]['request'] and row['model_input_sha256'] == hashlib.sha256(row['request'].encode()).hexdigest()
    pending = [r for r in items if r['id'] not in done]
    if not pending and (args.output / 'SUMMARY.json').exists():
        assert json.loads((args.output / 'STATUS.json').read_text())['status'] == 'COMPLETE'; db.close(); print('Verified complete raw-copy cache reused'); return
    sys.path.insert(0, str(args.snapshot))
    import torch
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.models.sceneplan_transfusion_generation_ar import load_p10v11_generation_ar
    from stable_audio_tools.models.sceneplan_generation_ar_lora import AdaptedGenerationAR
    from stable_audio_tools.inference.sceneplan_generation_ar_precision import configure_float32_ar
    from stable_audio_tools.inference.sceneplan_generation_ar_vectorized import generate_constrained_vectorized
    pointer_module = module('raw_literal_copy_head', args.copy_module); decoder = module('raw_literal_copy_decoder', args.decoder)
    torch.set_num_threads(4); torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    atomic(args.output / 'STATUS.json', {'status': 'LOADING', 'pid': os.getpid()})
    assert state['bundle_schema'] == ('generation_ar_with_learned_source_inventory_v1' if has_inventory_head else
        'generation_ar_with_learned_qualitative_execution_v1' if has_execution_head else 'generation_ar_with_learned_literal_copy_v1')
    assert state['copy_pointer_contract']['copy_module_sha256'] == identity['copy_module_sha256']
    assert state['copy_pointer_contract']['base_checkpoint_sha256'] == state['parent_ar_checkpoint_sha256']
    codec = ModelScenePlanCodecV4(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    base, p10 = load_p10v11_generation_ar(pad_id=codec.pad_id, verify_sha256=True, activation_checkpointing=False)
    assert p10.as_dict() == state['run_contract']['p10_load'] and codec.fingerprint == state['run_contract']['codec_fingerprint']
    base.load_trainable_state_dict(state['ar_adapter']); model = AdaptedGenerationAR(base, codec, rank=8, alpha=8., binding_strength=0.)
    model.load_lora_state_dict(state['ar_lora']); pointer = pointer_module.LiteralCopyPointer(**state['copy_pointer_contract']['model'])
    pointer.load_state_dict(state['copy_pointer']); execution_head = None; execution_module = None
    if has_execution_head:
        assert state['qualitative_contract']['module_sha256'] == identity['qualitative_module_sha256']
        execution_module = module('raw_qualitative_execution_head', args.qualitative_module)
        execution_head = execution_module.QualitativeExecutionHead(**state['qualitative_contract']['model'])
        execution_head.load_state_dict(state['qualitative_head'])
    inventory_head = None; inventory_module = None
    if has_inventory_head:
        assert state['source_inventory_contract']['module_sha256'] == identity['inventory_module_sha256']
        inventory_module = module('raw_learned_source_inventory', args.inventory_module)
        inventory_head = inventory_module.SourceInventoryHead(**state['source_inventory_contract']['model'])
        inventory_head.load_state_dict(state['source_inventory'])
    del state, base
    device = torch.device('cuda:0')
    model.p10_dit.to(device=device, dtype=torch.float32); model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32); model.ar_lora.to(device=device, dtype=torch.float32)
    configure_float32_ar(model); model.eval(); pointer.to(device=device, dtype=torch.float32).eval()
    if execution_head is not None: execution_head.to(device=device, dtype=torch.float32).eval()
    if inventory_head is not None: inventory_head.to(device=device, dtype=torch.float32).eval()
    execution_kwargs = {'execution_head': execution_head, 'execution_module': execution_module,
        'inventory_head': inventory_head, 'inventory_module': inventory_module}
    def versions(): return {k: (id(p), p.data_ptr(), p._version) for k, p in model.named_parameters()}
    before_versions = versions()
    if args.gate_requests and not (args.output / 'RAW_GPU_GATE.json').exists():
        texts = [r['request'] for r in raw_items(args.gate_requests)]
        atomic(args.output / 'STATUS.json', {'status': 'CHECKING_RAW_DECODE_AND_QUERY_PARITY', 'rows': len(texts)})
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            baseline = generate_constrained_vectorized(model, texts, codec, device=device, max_plan_tokens=args.max_plan_tokens)
            observed, traces = decoder.generate_with_learned_copy(model, pointer, pointer_module, texts, codec, device=device,
                max_plan_tokens=args.max_plan_tokens, observe_only=True, **execution_kwargs)
        assert baseline == observed, 'Observing copy queries changed base AR tokens'
        # Check the prefixes actually produced with all learned decisions enabled.
        # These are the model's own outputs, never validation target prefixes.
        if has_execution_head or has_inventory_head:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                baseline, traces = decoder.generate_with_learned_copy(model, pointer, pointer_module, texts, codec,
                    device=device, max_plan_tokens=args.max_plan_tokens, **execution_kwargs)
        maximum = max(map(len, baseline)); inputs = torch.full((len(texts), maximum), codec.pad_id, device=device, dtype=torch.long)
        mask = torch.zeros_like(inputs).bool()
        for i, tokens in enumerate(baseline): inputs[i, :len(tokens)] = torch.tensor(tokens, device=device); mask[i, :len(tokens)] = True
        captured = {}; hook = model.ar_adapter.output_norm.register_forward_hook(lambda _m, _a, value: captured.update(hidden=value.detach()))
        with torch.inference_mode():
            context, context_mask = model.encode_requests(texts, device=device)
            model(inputs, mask, context, context_mask); hidden = captured.pop('hidden'); hook.remove()
            alignment = pointer_module.encode_character_alignment(model.prompt_conditioner.tokenizer, texts, context_mask, device=device)
            prepared = pointer.prepare_keys(context, context_mask, alignment)
            fields = max(map(len, traces)); queries = hidden.new_zeros(len(texts), fields, 1024); kinds = torch.zeros(len(texts), fields, device=device, dtype=torch.long)
            for i, events in enumerate(traces):
                for j, event in enumerate(events): queries[i, j] = hidden[i, event['query_position']]; kinds[i, j] = pointer_module.FIELD_TYPES[event['field']]
            if has_inventory_head:
                inventory = decoder.predict_source_inventory(inventory_head, inventory_module, pointer_module,
                    texts, context, context_mask, alignment)
                starts, ends = [[0] * fields for _ in texts], [[0] * fields for _ in texts]
                for i, events in enumerate(traces):
                    plan = codec.decode(baseline[i])
                    assert len(plan['sources']) == inventory[i]['count']
                    assert [s['kind'] for s in plan['sources']] == [s['kind'] for s in inventory[i]['sources']]
                    for j, event in enumerate(events):
                        source = inventory[i]['sources'][event['generated_source_slot']]
                        span = source['transcript' if event['field'] == 'transcript' else 'identity']
                        starts[i][j], ends[i][j] = span['start'], span['end']
                        assert event['learned_inventory_count'] == inventory[i]['count']
                        assert event['learned_inventory_kind'] == source['kind']
            else:
                a, b = pointer.score_queries(queries, kinds, prepared); starts, ends = pointer_module.best_ordered_span(a, b, alignment.endpoint_mask)
                starts, ends = starts.tolist(), ends.tolist()
            for i, events in enumerate(traces):
                for j, event in enumerate(events): assert texts[i][starts[i][j]:ends[i][j] + 1] == event['text'], 'Full generated-prefix vs cache copy proposal differs'
            execution_queries_checked = 0
            if execution_head is not None:
                token_starts = context_mask.long().argmax(-1, keepdim=True).expand(-1, fields).clone(); token_ends = token_starts.clone()
                for i, events in enumerate(traces):
                    for j, event in enumerate(events):
                        assert (starts[i][j], ends[i][j]) == (event['start'], event['end']), 'Bound content span endpoints differ'
                        token_starts[i, j] = alignment.token_indices[i, starts[i][j]]
                        token_ends[i, j] = alignment.token_indices[i, ends[i][j]]
                execution_options = {}
                if getattr(execution_head, 'source_local_attention', False):
                    last_tokens = (context_mask.long() * torch.arange(context_mask.shape[1], device=device)).amax(-1)
                    source_spans = torch.stack((token_starts, last_tokens[:, None].expand(-1, fields)), -1)
                    for i, events in enumerate(traces):
                        for j, event in enumerate(events):
                            scope = event['learned_event_span']
                            source_spans[i, j, 0] = min(int(alignment.token_indices[i, scope['start']]), int(token_starts[i, j]))
                            source_spans[i, j, 1] = max(int(alignment.token_indices[i, scope['end']]), int(token_ends[i, j]))
                    execution_options['source_spans'] = source_spans
                values = execution_head(queries, kinds, token_starts, token_ends, context, context_mask, **execution_options)
                values = {name: value.cpu().tolist() for name, value in values.items()}
                for i, events in enumerate(traces):
                    for j, event in enumerate(events):
                        if 'qualitative_control' not in event: continue
                        control = execution_module.complete_from_logits({name: value[i][j] for name, value in values.items()},
                            codec.frame_ids.index(baseline[i][2]), seed_key=f'42/{texts[i]}/{event["generated_source_slot"]}',
                            speech=event['field'] == 'speaker_description')
                        assert control == event['qualitative_control'], 'Full generated-prefix vs cached qualitative control differs'
                        execution_queries_checked += 1
        assert versions() == before_versions
        atomic(args.output / 'RAW_GPU_GATE.json', {'status': 'PASS', 'raw_requests': len(texts), 'fields': sum(map(len, traces)),
            'observe_only_base_tokens_exact': True, 'full_generated_prefix_vs_cache_copied_text_exact': True,
            'query_parity_uses_actual_augmented_model_prefixes': has_execution_head or has_inventory_head,
            'learned_inventory_recomputed_from_raw_context_exact': True if has_inventory_head else None,
            'learned_qualitative_control_queries_checked': execution_queries_checked,
            'full_generated_prefix_vs_cache_qualitative_controls_exact': True if has_execution_head else None,
            'all_base_parameter_objects_pointers_and_versions_unchanged': True, 'target_or_annotation_inputs': False,
            'checkpoint_sha256': identity['checkpoint_sha256'], 'contract_sha256': sha(contract)})
    if args.gate_requests:
        gate = json.loads((args.output / 'RAW_GPU_GATE.json').read_text()); assert gate['status'] == 'PASS' and gate['contract_sha256'] == sha(contract)
    if args.benchmark_decoder:
        alternative = module('raw_planning_benchmark_backend', args.benchmark_decoder)
        probes = [row['request'] for row in items[:32]]
        measured = {}
        outputs = {}
        for name, implementation in [('cached_step', decoder), ('decision_blocks', alternative)]:
            torch.cuda.synchronize()
            begin = time.monotonic()
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                tokens, traces = implementation.generate_with_learned_copy(model, pointer, pointer_module,
                    probes, codec, device=device, max_plan_tokens=args.max_plan_tokens, **execution_kwargs)
            torch.cuda.synchronize()
            measured[name] = time.monotonic() - begin
            outputs[name] = (tokens, traces)
        disagreements = [items[i]['id'] for i, (a, b) in enumerate(zip(outputs['cached_step'][0], outputs['decision_blocks'][0])) if a != b]
        assert versions() == before_versions
        result = {'status': 'PASS_TOKEN_PARITY_BENCHMARK' if not disagreements else 'FAIL_TOKEN_PARITY_BENCHMARK',
            'rows': len(probes), 'seconds': measured, 'speedup': measured['cached_step'] / measured['decision_blocks'],
            'token_disagreements': disagreements, 'checkpoint_sha256': identity['checkpoint_sha256'],
            'contract_sha256': sha(contract), 'test_used': False, 'model_promoted': False,
            'scope': 'One paired timing on the same restored model/GPU/batch after the startup gate. Full raw validation remains required.'}
        atomic(args.output / 'BENCHMARK.json', result)
        atomic(args.output / 'BENCHMARK_OUTPUTS.json', {name: {'tokens': tokens, 'traces': traces} for name, (tokens, traces) in outputs.items()})
        if args.benchmark_only:
            db.close(); atomic(args.output / 'STATUS.json', {'status': result['status'], 'benchmark': str(args.output / 'BENCHMARK.json')})
            print(json.dumps(result)); return
    elif args.benchmark_only:
        raise ValueError('--benchmark-only requires --benchmark-decoder')
    started = time.monotonic()
    def generate(rows):
        if time.monotonic() - started > args.max_wall_seconds: raise TimeoutError('raw AR inference wall budget exceeded')
        try:
            tokens, traces = decoder.generate_with_learned_copy(model, pointer, pointer_module, [r['request'] for r in rows], codec,
                device=device, max_plan_tokens=args.max_plan_tokens, **execution_kwargs)
            return [(t, trace, None, None) for t, trace in zip(tokens, traces)]
        except decoder.CopyDecodeLimitError as exc:
            result = []
            for row, prefix, trace, finished in zip(rows, exc.prefixes, exc.traces, exc.finished):
                if finished:
                    result.append((prefix, trace, None, None)); continue
                if time.monotonic() - started > args.max_wall_seconds:
                    raise TimeoutError('raw AR recovery wall budget exceeded')
                recovery = {'policy': 'one_bundled_base_ar_retry', 'primary_error': str(exc),
                    'primary_partial_tokens': prefix, 'primary_copy_trace': trace,
                    'same_raw_request': True, 'same_token_limit': args.max_plan_tokens,
                    'target_or_annotation_inputs': False}
                try:
                    fallback = generate_constrained_vectorized(model, [row['request']], codec,
                        device=device, max_plan_tokens=args.max_plan_tokens)[0]
                    codec.decode(fallback, sample_id=row['id'])
                    recovery['status'] = 'BASE_AR_RETRY_SUCCEEDED'
                    result.append((fallback, [], None, recovery))
                except RuntimeError as error:
                    if 'Generation AR did not emit EOS within' not in str(error): raise
                    recovery.update(status='BASE_AR_RETRY_FAILED', fallback_error=str(error))
                    result.append((None, [], str(error), recovery))
            return result
    for offset in range(0, len(pending), args.batch_size):
        rows = pending[offset:offset + args.batch_size]; before = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16): output = generate(rows)
        for row, (tokens, trace, error, recovery) in zip(rows, output):
            try:
                if tokens is None: raise RuntimeError(error)
                prediction = codec.decode(tokens, sample_id=row['id']); status = 'ok'; error = None
            except Exception as exc: prediction = None; status = 'generation_error' if tokens is None else 'parse_error'; error = repr(exc)
            value = {'id': row['id'], 'request': row['request'], 'model_input_sha256': hashlib.sha256(row['request'].encode()).hexdigest(),
                'prediction': prediction, 'tokens': tokens, 'copy_trace': trace, 'status': status, 'error': error,
                'batch_elapsed_s': time.monotonic() - before, 'decode_recovery': recovery,
                'policy': 'raw request + unchanged schema grammar + learned literal spans + optional neural inventory and qualitative controls; recorded base AR retry only after EOS budget failure'}
            db.execute('INSERT INTO results VALUES (?,?)', (row['id'], json.dumps(value, ensure_ascii=False)))
        db.commit(); atomic(args.output / 'STATUS.json', {'status': 'RUNNING', 'rows_done': len(done) + offset + len(rows), 'rows': len(items)})
    results = [json.loads(p) for (p,) in db.execute('SELECT payload FROM results')]; db.close(); assert len(results) == len(items)
    assert versions() == before_versions
    summary = {'status': 'COMPLETE', 'rows': len(results), 'parsed': sum(r['status'] == 'ok' for r in results),
        'elapsed_generation_s': time.monotonic() - started, 'base_parameter_versions_unchanged': True,
        'copied_fields': sum(sum(t['copied'] for t in r['copy_trace']) for r in results),
        'length_fallbacks': sum(sum(t['fallback_reason'] is not None for t in r['copy_trace']) for r in results),
        'primary_copy_eos_failures': sum(r.get('decode_recovery') is not None for r in results),
        'base_ar_recovery_succeeded': sum((r.get('decode_recovery') or {}).get('status') == 'BASE_AR_RETRY_SUCCEEDED' for r in results),
        'learned_qualitative_controls': sum(sum('qualitative_control' in t for t in r['copy_trace']) for r in results),
        'learned_inventory_fields': sum(sum('learned_inventory_count' in t for t in r['copy_trace']) for r in results),
        'acceptance': 'PENDING_COUPLED_REQUEST_SCORING_AND_P10_AUDIO'}
    atomic(args.output / 'SUMMARY.json', summary); atomic(args.output / 'STATUS.json', {'status': 'COMPLETE', 'summary': str(args.output / 'SUMMARY.json')}); print(json.dumps(summary))


if __name__ == '__main__':
    here = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('requests', 'checkpoint', 'snapshot', 'output'): p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--copy-module', type=Path, default=here / 'stable_audio_tools/models/sceneplan_generation_ar_copy_pointer.py')
    p.add_argument('--decoder', type=Path, default=here / 'stable_audio_tools/inference/sceneplan_generation_ar_learned_copy.py')
    p.add_argument('--qualitative-module', type=Path, default=here / 'stable_audio_tools/models/sceneplan_generation_ar_qualitative_head.py')
    p.add_argument('--inventory-module', type=Path, default=here / 'stable_audio_tools/models/sceneplan_generation_ar_source_inventory.py')
    p.add_argument('--benchmark-decoder', type=Path); p.add_argument('--benchmark-only', action='store_true')
    p.add_argument('--gate-requests', type=Path); p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-plan-tokens', type=int, default=512); p.add_argument('--max-wall-seconds', type=int, default=1200); args = p.parse_args()
    try: main(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True); atomic(args.output / 'STATUS.json', {'status': 'FAILED_NEEDS_ATTENTION', 'error': f'{type(exc).__name__}: {exc}'}); raise
