"""Prepare and execute matched250 adaptation only after actual native restart QA."""
from __future__ import annotations

import argparse
from datetime import datetime
import inspect
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.t2a.experiments.ar_instruction_t200_v3 import policy, runtime, resume_proof
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, write


def prepare(stage, proof):
    review = policy.read(proof / 'REVIEW.json')
    assert review['three_rank_native_resume_exact'] and not review['full_AR_expansion_allowed']
    for path, expected in review['source_sha256'].items():
        assert sha(path) == expected, path
    previous = policy.read(proof / 'SPEC.json')
    cfg = policy.read(proof / 'CONFIG.json')
    protocol = policy.read(stage / 'PROTOCOL.json')
    assert protocol['new_updates_per_arm'] == 250 and protocol['initial_AR_step'] == 5250
    control = policy.read(stage / 'CONTROL_CONFIG.json')
    control['current_AR_policy'] = cfg['current_AR_policy']
    write(stage / 'CONTROL_EXECUTION_CONFIG.json', control)
    for value in (cfg, control): policy.validate_config(value); policy.check_selection(value)
    spec = {key: value for key, value in previous.items() if key != 'phases'}
    spec['fingerprint_steps'] = [5500]
    gate = {'path': str(proof / 'REVIEW.json'), 'sha256': sha(proof / 'REVIEW.json')}
    spec['phases'] = {}
    for name in ('selected', 'control'):
        selected = name == 'selected'
        run = Path(previous['phases']['resumed2']['run_dir']) if selected else stage / 'runs/control'
        spec['phases'][name] = {'stage': 'short_adaptation', 'artifacts': str(stage / name),
            'run_dir': str(run), 'config': str(proof / 'CONFIG.json' if selected else stage / 'CONTROL_EXECUTION_CONFIG.json'),
            'stop_after_prefix': True, 'stop_at_step': 5500, 'checkpoint_at_stop': True,
            'prerequisite': gate, 'resume': str(run / 'checkpoints/step-00005254.pt') if selected else cfg['initial_state_transfer']['parent_checkpoint']['checkpoint']}
    write(stage / 'SPEC.json', spec)
    sources = {str(ROOT / p): value for p, value in runtime.source_inventory().items()}
    for path in (stage / 'SPEC.json', stage / 'PROTOCOL.json', stage / 'CONTROL_EXECUTION_CONFIG.json',
                 proof / 'CONFIG.json', proof / 'REVIEW.json', Path(__file__), Path(resume_proof.__file__),
                 ROOT / 'scripts/t2a/experiments/ar_instruction_t200_v3/fixed_ddp.py',
                 ROOT / 'scripts/t2a/experiments/ar_instruction_t200_v3/short_evaluation.py'):
        sources[str(path)] = sha(path)
    write(stage / 'PLAN.json', {'schema': policy.SCHEMA, 'experiment_spec': str(stage / 'SPEC.json'),
        'source_sha256': sources, 'resume_proof': str(proof)})
    return stage / 'PLAN.json'


def review_training(plan_path):
    plan = policy.read(plan_path)
    spec = policy.read(plan['experiment_spec'])
    proof = Path(plan['resume_proof'])
    results = {}
    refs = {}
    def read_lines(path):
        refs[str(path)] = sha(path)
        return [json.loads(line) for line in path.read_text().splitlines()]
    for rank in range(3):
        windows = {}
        for name in ('selected', 'control'):
            out = Path(spec['phases'][name]['artifacts']) / f'rank{rank}'
            done = policy.read(out / 'RUNTIME_COMPLETE.json')
            refs[str(out / 'RUNTIME_COMPLETE.json')] = sha(out / 'RUNTIME_COMPLETE.json')
            assert done['all_frozen_parameters_exact'] and done['RF_training_calls'] == 0 and done['step'] == 5500
            rows = read_lines(out / 'training_windows.jsonl')
            if name == 'selected':
                rows = (read_lines(proof / f'split2/rank{rank}/training_windows.jsonl') +
                    read_lines(proof / f'resumed2/rank{rank}/training_windows.jsonl') + rows)
            assert [r['step'] for r in rows] == list(range(5251, 5501))
            windows[name] = [{k: row[k] for k in ('step', 'epoch', 'next_batch', 'rank', 'microbatches', 'learning_rates_after_scheduler', 'native_denominators')} for row in rows]
        assert windows['selected'] == windows['control'], f'matched data or LR differs at rank{rank}'
        results[str(rank)] = {'updates_per_arm': 250, 'same_actual_pairs_requests_masks_and_LR': True,
            'pair_exposures_per_arm': sum(len(b['pair_ids']) for row in windows['selected'] for b in row['microbatches'])}
    return {'at': datetime.now().astimezone().isoformat(), 'three_rank_native_resume_exact': True,
        'matched250_training_complete': True, 'ranks': results, 'source_sha256': refs,
        'full_AR_expansion_allowed': False, 'AR_quality_gate_passed': False, 'independent_test_used': False}


def prepare_evaluation(plan_path):
    from scripts.t2a.experiments.ar_instruction_t200_v3 import short_evaluation
    plan = policy.read(plan_path)
    spec = policy.read(plan['experiment_spec'])
    stage = plan_path.parent
    cfg = policy.read(spec['phases']['selected']['config'])
    control = policy.read(spec['phases']['control']['config'])
    selected = cfg['clap_dependency']['checkpoint']
    old = control['clap_dependency']['checkpoint']
    def endpoint(name):
        path = Path(spec['phases'][name]['run_dir']) / 'checkpoints/step-00005500.manifest.json'
        return policy.read(path)
    parent = cfg['initial_state_transfer']['parent_checkpoint']
    arms = [
        {'name': 'selected_start', 'AR_checkpoint': parent, 'encoder': selected},
        {'name': 'selected250', 'AR_checkpoint': endpoint('selected'), 'encoder': selected},
        {'name': 'control_start', 'AR_checkpoint': parent, 'encoder': old},
        {'name': 'control250', 'AR_checkpoint': endpoint('control'), 'encoder': old},
    ]
    sources = dict(plan['source_sha256'])
    sources[str(Path(short_evaluation.__file__))] = sha(short_evaluation.__file__)
    requested = short_evaluation.requested
    for module in (requested, requested.endpoints, short_evaluation.integration):
        sources[str(Path(module.__file__))] = sha(module.__file__)
    for function in (short_evaluation.evaluate_ar, short_evaluation.free_ar_pass):
        path = inspect.getsourcefile(function); sources[path] = sha(path)
    for arm in arms:
        checkpoint = Path(arm['AR_checkpoint']['checkpoint'])
        sources[str(checkpoint.with_suffix('.manifest.json'))] = sha(checkpoint.with_suffix('.manifest.json'))
    result = {'schema': 'AR_matched_short_free_validation_v1', 'training_config': spec['phases']['selected']['config'],
        'protocol': str(stage / 'PROTOCOL.json'), 'protocol_sha256': sha(stage / 'PROTOCOL.json'),
        'arms': arms, 'source_sha256': sources, 'full_AR_expansion_allowed': False}
    target = stage / 'evaluation/PLAN.json'; write(target, result)
    return target


def run(plan_path):
    out = plan_path.parent
    try:
        for name in ('selected', 'control'):
            plan = policy.read(plan_path)
            for path, expected in plan['source_sha256'].items(): assert sha(path) == expected, path
            command = resume_proof.command(plan_path, name)
            index = command.index(str(ROOT / 'scripts/t2a/experiments/ar_instruction_t200_v3/runtime.py'))
            command[index] = str(ROOT / 'scripts/t2a/experiments/ar_instruction_t200_v3/fixed_ddp.py')
            print(json.dumps({'at': datetime.now().astimezone().isoformat(), 'phase': name, 'status': 'starting'}), flush=True)
            with (out / f'{name}.log').open('x') as stream:
                subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        write(out / 'TRAINING_REVIEW.json', review_training(plan_path))
        evaluation = prepare_evaluation(plan_path)
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=3',
            str(ROOT / 'scripts/t2a/experiments/ar_instruction_t200_v3/short_evaluation.py'), '--plan', str(evaluation)]
        with (out / 'evaluation.log').open('x') as stream:
            subprocess.run(command, cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
        write(out / 'AWAITING_EFFECT_REVIEW.json', {'at': datetime.now().astimezone().isoformat(),
            'matched_short_training_and_free_generation_complete': True,
            'full_AR_expansion_allowed': False, 'next': 'Review all matched100 free-plan results against the preregistered screen.'})
    except Exception as error:
        write(out / 'FAILURE.json', {'at': datetime.now().astimezone().isoformat(), 'error': repr(error),
            'full_AR_expansion_allowed': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--run-plan', type=Path)
    args = parser.parse_args()
    if args.run_plan: run(args.run_plan)
    else: print(prepare(args.stage, args.proof))


if __name__ == '__main__': main()
