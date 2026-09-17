"""Bind measured batch sizes and complete native kernel coverage before training."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from scripts.t2a.experiments.ar_structured_v1 import runtime as rt


def run(args):
    cfg = rt.read(args.draft)
    base = args.draft.parent
    checks = rt.read(base/'CPU_QA.json')
    if checks['label_examples'] != 500 or not all(checks[k] for k in (
            'consistent_slot_permutation_invariant','incorrect_post_edit_identity_penalized',
            'paired_CLAP10k_source_readout_exact','zero_residual_preserves_decoder_exactly')):
        raise RuntimeError('Structured CPU correctness evidence is incomplete')
    initialization = rt.read(base/'CPU_INITIALIZATION.json')
    if not initialization['all201_DiT_parameters_copied_exactly'] or not initialization['shared_transformer_same_object']:
        raise RuntimeError('Editing DiT50k initialization has not passed')
    results = []
    for rank in range(3):
        directory = args.profile/f'rank{rank}'
        done = rt.read(directory/'COMPLETE.json')
        if done['optimizer_updates'] != 0 or not done['same_shared_Transformer'] or not done['all_trainable_gradients_present']:
            raise RuntimeError('Joint performance profile did not preserve its invariants')
        current = rt.read(directory/'RESULTS.json')
        if results and current != results:
            raise RuntimeError('Ranks disagree on the performance measurements')
        results = current
    selected = [r for r in results if r['case'] == 'checkpoint_AR_64_40']
    if {r['bucket'] for r in selected} != {432,648} or max(r['peak_allocated_GiB'] for r in selected) > 42:
        raise RuntimeError('Selected microbatches lack measured memory headroom')
    for key in ('short_batch_size','long_batch_size','gradient_accumulation','AR_activation_checkpointing','RF_activation_checkpointing'):
        if len({r[key] for r in selected}) != 1:
            raise RuntimeError('Selected short and long profiles differ in configuration')
        cfg['performance'][key] = selected[0][key]
    cfg['numerics'] = {}
    for rank in range(3):
        directory = args.numerics/f'rank{rank}'
        done = rt.read(directory/'COMPLETE.json')
        if (done['optimizer_updates'] != 0 or done['frozen_Qwen_sha256'] != initialization['frozen_Qwen_sha256']
                or not set(range(1,65)) <= set(done['batch_sizes'])
                or not set(range(1,9)) <= set(done['normalization_NB'])):
            raise RuntimeError('Native kernel coverage does not cover the declared training bounds')
        cfg['numerics'][str(rank)] = {'directory': str(directory),
            'sha256': rt.sha(directory/'AUTOTUNE_CATALOG.json')}
    cfg['training_source_sha256'] = rt.source_inventory()
    cfg['preparation_evidence'] = {'CPU_QA': {'path': str(base/'CPU_QA.json'), 'sha256': rt.sha(base/'CPU_QA.json')},
        'CPU_initialization': {'path': str(base/'CPU_INITIALIZATION.json'), 'sha256': rt.sha(base/'CPU_INITIALIZATION.json')},
        'performance_results': {'path': str(args.profile/'rank0/RESULTS.json'), 'sha256': rt.sha(args.profile/'rank0/RESULTS.json')},
        'selected_performance': selected, 'effective_global_short_batch': 192, 'effective_global_long_batch': 120,
        'data_policy': 'Full frozen training index; verify every used source/target latent tensor against its stored hash.',
        'optimizer_policy': 'Fresh joint optimizer and scheduler;50000 new updates; multiple read-only weight parents.',
        'user_direction_sha256': rt.sha(cfg['user_direction']), 'quality_gate_passed': False}
    rt.validate_config(cfg)
    rt.write(args.output, cfg)
    print(json.dumps({'prepared': str(args.output), 'performance': cfg['performance'], 'new_updates': cfg['new_updates']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft', type=Path, required=True)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--numerics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
