#!/usr/bin/env python3
"""Direct raw-English AR inference using frozen existing Generation assets.

No target plan, supplied count, source IDs, request parser or planning teacher
is used to produce tokens. The fixed codec grammar enforces legal syntax only.
This entry emits plans; final P10 FOA integration is a separate pending gate.
"""
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
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n'); temporary.replace(path)


def isolate_generation_failures(generate, requests):
    """Retain healthy rows and count individual EOS failures without extending the limit."""
    try:
        return [(tokens, None) for tokens in generate(requests)]
    except RuntimeError as exc:
        if 'Generation AR did not emit EOS within' not in str(exc):
            raise
        if len(requests) == 1:
            return [(None, str(exc))]
        middle = len(requests) // 2
        return isolate_generation_failures(generate, requests[:middle]) + isolate_generation_failures(generate, requests[middle:])


def main(args):
    args.count_expert_checkpoint = getattr(args, 'count_expert_checkpoint', None)
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('0', '1', '2')
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'LOCK').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    items = json.loads(args.requests.read_text())['requests']
    assert len({r['id'] for r in items}) == len(items)
    assert all(isinstance(r['request'], str) and r['request'].strip() for r in items)
    identity = {'schema': 'generation_ar_direct_raw_request_inference_v1', 'checkpoint': str(args.checkpoint),
                'checkpoint_sha256': sha(args.checkpoint), 'snapshot': str(args.snapshot),
                'snapshot_manifest_sha256': sha(args.snapshot / 'SOURCE_SNAPSHOT_MANIFEST.json'),
                'input_sha256': sha(args.requests), 'script_sha256': sha(Path(__file__)), 'batch_size': args.batch_size,
                'max_plan_tokens': args.max_plan_tokens, 'seed': 42, 'binding_strength': 0,
                'declared_count_constraint': False, 'target_or_annotation_inputs': False,
                'precision': 'FP32 AR/math-SDPA, BF16 frozen request encoder, TF32 off',
                'wall_cap_s': args.max_wall_seconds, 'p10_audio_acceptance': 'PENDING',
                'failure_policy': 'bisect failed EOS batches; preserve the token limit and record individual failures; successful batches unchanged'}
    if args.count_expert_checkpoint:
        identity['learned_count_expert'] = {
            'checkpoint': str(args.count_expert_checkpoint),
            'checkpoint_sha256': sha(args.count_expert_checkpoint),
            'module_sha256': sha(args.snapshot / 'stable_audio_tools/inference/sceneplan_generation_ar_count_expert.py'),
            'policy': 'Expert AR logits only at source-count token; base raw-generated header and base request context; no supplied count or annotations.'}
    if (args.output / 'CONTRACT.json').exists(): assert json.loads((args.output / 'CONTRACT.json').read_text()) == identity
    else: atomic(args.output / 'CONTRACT.json', identity)
    sys.path.insert(0, str(args.snapshot))
    import torch
    from stable_audio_tools.models.sceneplan_generation_ar_lora import AdaptedGenerationAR
    from stable_audio_tools.models.sceneplan_transfusion_generation_ar import ScenePlanTransfusionGenerationAR
    from stable_audio_tools.inference.sceneplan_generation_ar_precision import configure_float32_ar
    from stable_audio_tools.inference.sceneplan_generation_ar_vectorized import generate_constrained_vectorized
    spec = importlib.util.spec_from_file_location('frozen_ar_asset_loader', args.snapshot / 'scripts/t2a/test/evaluate_sceneplan_transfusion_generation_ar_8k.py')
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    torch.set_num_threads(4); torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    atomic(args.output / 'STATUS.json', {'status': 'LOADING', 'pid': os.getpid()})
    codec = frozen.ModelScenePlanCodecV4(frozen.CODEC_PATH)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    training = state['run_contract']
    base, p10 = frozen.load_p10v11_generation_ar(pad_id=codec.pad_id, verify_sha256=True, activation_checkpointing=False)
    assert p10.as_dict() == training['p10_load'] and codec.fingerprint == training['codec_fingerprint']
    base.load_trainable_state_dict(state['ar_adapter'])
    model = AdaptedGenerationAR(base, codec, rank=8, alpha=8., binding_strength=0.); del base
    if 'ar_lora' in state: model.load_lora_state_dict(state['ar_lora'])
    del state
    model.p10_dit.to(device=device, dtype=torch.float32); model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32); model.ar_lora.to(device=device, dtype=torch.float32)
    configure_float32_ar(model); model.eval()
    if args.count_expert_checkpoint:
        from stable_audio_tools.inference.sceneplan_generation_ar_count_expert import CountExpertDecoder
        expert = torch.load(args.count_expert_checkpoint, map_location='cpu', weights_only=False)
        assert expert['run_contract']['p10_load'] == training['p10_load']
        assert expert['run_contract']['codec_fingerprint'] == training['codec_fingerprint']
        model = CountExpertDecoder(model, codec, expert['ar_adapter'], expert['ar_lora'])
        del expert
    # Call the base vectorized decoder explicitly: no source-span extraction,
    # count forcing, target count, external planner, or post-hoc field filling.
    db = sqlite3.connect(args.output / 'predictions.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS results(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    done = {r[0] for r in db.execute('SELECT id FROM results')}
    pending = [r for r in items if r['id'] not in done]; started = time.monotonic()
    for offset in range(0, len(pending), args.batch_size):
        if time.monotonic() - started > args.max_wall_seconds: raise TimeoutError('raw AR inference wall budget exceeded')
        batch = pending[offset:offset + args.batch_size]
        requests = [r['request'] for r in batch]
        before = time.monotonic()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            def generate(part):
                if time.monotonic() - started > args.max_wall_seconds: raise TimeoutError('raw AR inference wall budget exceeded')
                return generate_constrained_vectorized(model, part, codec, device=device, max_plan_tokens=args.max_plan_tokens)
            outputs = isolate_generation_failures(generate, requests)
        elapsed = time.monotonic() - before
        for row, (tokens, generation_error) in zip(batch, outputs):
            try:
                if tokens is None: raise RuntimeError(generation_error)
                plan = codec.decode(tokens, sample_id=row['id']); status = 'ok'; error = None
            except Exception as exc:
                plan = None; status = 'generation_error' if tokens is None else 'parse_error'; error = repr(exc)
            value = {'id': row['id'], 'request': row['request'], 'model_input_sha256': hashlib.sha256(row['request'].encode()).hexdigest(),
                     'prediction': plan, 'tokens': tokens, 'status': status, 'error': error,
                     'batch_elapsed_s': elapsed, 'policy': 'raw request + fixed schema grammar only'}
            db.execute('INSERT INTO results VALUES (?,?)', (row['id'], json.dumps(value, ensure_ascii=False)))
        db.commit(); atomic(args.output / 'STATUS.json', {'status': 'RUNNING', 'rows_done': len(done) + offset + len(batch), 'rows': len(items)})
    results = [json.loads(p) for (p,) in db.execute('SELECT payload FROM results')]; db.close()
    if args.count_expert_checkpoint:
        model.assert_restored()
        atomic(args.output / 'COUNT_EXPERT_RESTORATION.json', {
            'status': 'PASS', 'base_adapter_and_lora_bit_exact_after_generation': True,
            'count_forward_calls': model.count_calls, 'p10_weights_loaded_by_expert': False})
    assert len(results) == len(items)
    summary = {'status': 'COMPLETE', 'rows': len(results), 'parsed': sum(r['status'] == 'ok' for r in results),
               'elapsed_generation_s': time.monotonic() - started, 'acceptance': 'PENDING_REQUEST_SCORING_AND_P10',
               'generated_source_counts': {str(n): sum(bool(r['prediction']) and len(r['prediction']['sources']) == n for r in results) for n in range(1, 5)}}
    atomic(args.output / 'SUMMARY.json', summary); atomic(args.output / 'STATUS.json', {'status': 'COMPLETE', 'summary': str(args.output / 'SUMMARY.json')})
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, required=True); parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--count-expert-checkpoint', type=Path)
    parser.add_argument('--snapshot', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16); parser.add_argument('--max-plan-tokens', type=int, default=512)
    parser.add_argument('--max-wall-seconds', type=int, default=1800); args = parser.parse_args()
    try: main(args)
    except BaseException as exc:
        args.output.mkdir(parents=True, exist_ok=True); atomic(args.output / 'STATUS.json', {'status': 'FAILED', 'error': f'{type(exc).__name__}: {exc}'})
        raise
