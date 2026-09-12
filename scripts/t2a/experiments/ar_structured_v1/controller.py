"""Bounded exact joint-restart proof followed by the authorized new50k run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.t2a.experiments.ar_structured_v1 import runtime as rt


def command(config, run, attempt, stop, *, resume=None, proof=False):
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=3',
        str(ROOT/'scripts/t2a/experiments/ar_structured_v1/runtime.py'), '--config', str(config),
        '--run-dir', str(run), '--attempt', attempt, '--stop-updates', str(stop)]
    if resume is not None:
        cmd += ['--resume', str(resume)]
    if proof:
        cmd += ['--evidence-updates', '2', '4']
    if attempt == 'continuous4':
        cmd += ['--probe-interface']
    return cmd


def phase(args, run, name, stop, *, resume=None, proof=False):
    cmd = command(args.config, run, name, stop, resume=resume, proof=proof)
    rt.write(args.output / f'COMMAND_{name}.json', {'at': rt.now(), 'command': cmd})
    print(json.dumps({'at': rt.now(), 'phase': name, 'status': 'started'}), flush=True)
    with (args.output / f'{name}.log').open('x') as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({'at': rt.now(), 'phase': name, 'status': 'complete'}), flush=True)


def review(output):
    rows = []
    for rank in range(3):
        paths = {'continuous4': output / 'continuous/attempts/continuous4' / f'rank{rank}',
            'split2': output / 'split/attempts/split2' / f'rank{rank}',
            'resumed2': output / 'split/attempts/resumed2' / f'rank{rank}'}
        for directory in paths.values():
            done = rt.read(directory/'COMPLETE.json')
            if not all(done[k] for k in ('joint_AR_and_RF_trained','shared_transformer_same_object',
                    'protected_DiT50k_unchanged','frozen_encoders_unchanged')):
                raise RuntimeError('A joint proof phase did not finish with all invariants')
            if rt.read(directory/'DDP_REVIEW.json').get('has_rebuilt_buckets', 0) != 0:
                raise RuntimeError('Joint DDP bucket layout changed')
        probe = rt.read(paths['continuous4']/'SHARED_GRADIENT_AND_INPUT_PROBE.json')
        if min(probe['AR_shared_gradient_norm'], probe['RF_shared_gradient_norm']) <= 0:
            raise RuntimeError('Both losses must train the same backbone')
        for step, name in ((2, 'split2'), (4, 'resumed2')):
            left = rt.read(paths['continuous4']/f'STATE_step{step:06d}.json')
            right = rt.read(paths[name]/f'STATE_step{step:06d}.json')
            if left != right:
                raise RuntimeError(f'Joint restart differs at update{step}, rank{rank}')
            rows.append({'rank': rank, 'new_update': step, 'state_sha256': left['sha256']})
        if rt.read(paths['split2']/'STATE_step000002.json') != rt.read(paths['resumed2']/'RESUME_STATE.json'):
            raise RuntimeError('Joint native resume did not load the complete saved state')
        windows = {k: [json.loads(x) for x in (p/'windows.jsonl').read_text().splitlines()] for k,p in paths.items()}
        if windows['continuous4'] != windows['split2'] + windows['resumed2']:
            raise RuntimeError('Native restart changed examples, requests or learning-rate clocks')
    result = {'at': rt.now(), 'three_rank_joint_restart_bitwise_exact': True,
        'same_examples_and_raw_requests': True,
        'compared': ['all_model_parameters_and_buffers','all_Adam_states','scheduler','all_rank_RNG','epoch','cursor'],
        'both_losses_train_the_same_Transformer': True, 'protected_original_DiT50k_unchanged': True,
        'results': rows, 'quality_gate_passed': False, 'independent_test_used': False}
    rt.write(output/'REVIEW.json', result)
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = rt.read(args.config); rt.validate_config(cfg)
    if rt.source_inventory() != cfg['training_source_sha256']:
        raise RuntimeError('Prepared training source changed')
    try:
        phase(args, args.output/'continuous', 'continuous4', 4, proof=True)
        phase(args, args.output/'split', 'split2', 2, proof=True)
        phase(args, args.output/'split', 'resumed2', 4,
            resume=args.output/'split/checkpoints/step-00000002.pt', proof=True)
        review(args.output)
        if args.production_run is not None:
            if args.production_run.exists() and any(args.production_run.iterdir()):
                raise RuntimeError('Initial full50k launch requires its own empty run directory')
            rt.write(args.output/'FULL50K_LAUNCH_AUTHORIZED.json', {'at': rt.now(),
                'authorization': cfg['user_direction'], 'correctness_review': str(args.output/'REVIEW.json'),
                'new_optimizer_updates': 50000, 'run_directory': str(args.production_run),
                'old_checkpoint_is_read_only': True, 'quality_gate_passed': False})
            phase(args, args.production_run, 'full50k_initial', cfg['new_updates'])
    except Exception as error:
        rt.write(args.output/'FAILURE.json', {'at': rt.now(), 'error': repr(error), 'quality_gate_passed': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--production-run', type=Path)
    run(parser.parse_args())
