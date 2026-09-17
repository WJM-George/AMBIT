"""Evaluate a discrete-plan intervention between original and updated Editing.

Only the decoded ScenePlan crosses the boundary. The executor always uses
its own complete shared backbone, conditioners and DiT. No training or new
optimizer checkpoint is performed. The normal nine-metric evaluator is reused.
"""
import argparse
import copy
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import numpy as np

from scripts.t2a.rl.train_editing_opsd_complete import CompleteLearner, base


def finish_runtime(learner, original_autotuner_run, *, evaluation_completed):
    """A fully cached rank did no native execution to audit this invocation."""
    from triton.runtime.autotuner import Autotuner
    try:
        cached_only = False
        if (evaluation_completed and not learner.q.get('evaluation_save_audio', True)
                and not learner.q.get('evaluation_additional_native_execution', False)):
            result = base.read(learner.out / f'eval_step{learner.step:06d}_rank{learner.rank}.json')
            expected = (len(learner.q['validation_ordinals'][learner.rank::learner.world]) *
                        len(learner.q['evaluation_seeds']))
            cached_only = (len(result['rows']) == expected and result['resumed_outputs'] == expected)
        if cached_only:
            # The frozen observer requires at least one native forward when
            # finishing. Here all outputs came from identity/hash-verified
            # records, so restore its process-local hook without claiming a
            # new native-kernel observation. Any new output uses normal finish.
            Autotuner.run = original_autotuner_run
            base.write(learner.runtime_dir / 'CACHED_EVALUATION_ONLY.json', dict(
                step=learner.step, rank=learner.rank, resumed_outputs=expected,
                native_forwards_this_evaluation=0, native_audit='No new execution; reused verified metric records.'))
        else:
            learner.finish_numerics()
    finally:
        learner.restore_flash()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--planner', choices=['old', 'new'], required=True)
    parser.add_argument('--executor', choices=['old', 'new'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fresh-matched-panel', action='store_true',
                        help='Evaluate old/old or new/new on a separately prepared request/noise panel.')
    parser.add_argument('--no-save-audio', action='store_true',
                        help='Score waveforms in memory; keep resumable per-output metric records and features only.')
    parser.add_argument('--full-native-validation', action='store_true',
                        help='Also measure teacher-forced AR/RF/structured losses on the original20000 rows.')
    args = parser.parse_args()
    if args.planner == args.executor and not args.fresh_matched_panel:
        raise ValueError('Use the existing matched old/old and new/new evaluations.')
    if args.full_native_validation and args.planner != args.executor:
        raise ValueError('Native loss diagnostics require one complete old or new model.')
    rank, world = int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.mha.set_fastpath_enabled(False)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    q = copy.deepcopy(base.read(args.config))
    if args.fresh_matched_panel:
        # A new request/noise panel must actually execute its own baseline.
        q.pop('initial_evaluation_reference', None)
    random.seed(q['seed'] + rank)
    np.random.seed(q['seed'] + rank)
    q.update(output=str(args.output), config_path=str(args.config), top_checkpoint_policy=None,
             evaluation_intervention=dict(planner=args.planner, executor=args.executor),
             initialization_checkpoint=dict(path=str(args.checkpoint), sha256=args.checkpoint_sha256,
                                            step=args.step))
    if args.no_save_audio:
        q['evaluation_save_audio'] = False
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    learner = None
    evaluation_completed = False
    from triton.runtime.autotuner import Autotuner
    original_autotuner_run = Autotuner.run
    try:
        learner = CompleteLearner(q, rank, world, args.output)
        learner.step = args.step
        # The constructor copies C0 before loading the candidate into the
        # student. The independent reference is consequently still original40k.
        if args.planner == 'old':
            learner.adapter.native_plan = learner.reference.native_plan
        if args.executor == 'old':
            learner.reference_pipeline.audio_autoencoder = learner.pipeline.audio_autoencoder
            learner.pipeline = learner.reference_pipeline
        if args.full_native_validation:
            from scripts.t2a.rl.validate_editing_opsd_native import validate_native
            q['evaluation_additional_native_execution'] = True
            validate_native(learner, learner.reference if args.executor == 'old' else learner.adapter)
        if rank == 0:
            base.write(args.output / 'INTERVENTION.json', dict(
                planner=args.planner, executor=args.executor, step=args.step,
                base_checkpoint=q['base_checkpoint'], updated_checkpoint=q['initialization_checkpoint'],
                config=dict(path=str(args.config), sha256=base.sha(args.config)),
                source_inputs='Source FOA latent and raw edit request only',
                boundary='Only hard ScenePlan; executor owns all representations and shared Transformer weights',
                actual_outputs=len(q['validation_ordinals']) * len(q['evaluation_seeds']),
                audio_retained=q.get('evaluation_save_audio', True),
                full_native_validation=args.full_native_validation,
                updates=0, new_checkpoints=0, target_audio_in_inference=False))
        learner.evaluate()
        evaluation_completed = True
    finally:
        try:
            if learner is not None:
                finish_runtime(learner, original_autotuner_run, evaluation_completed=evaluation_completed)
            if evaluation_completed:
                dist.barrier()
        finally:
            gc.collect()
            dist.destroy_process_group()
    if rank == 0:
        base.write(args.output / 'COMPLETE.json', dict(
            phase='COMPLETE', elapsed_seconds=time.time() - started, world_size=world,
            planner=args.planner, executor=args.executor, updates=0, new_checkpoints=0))


if __name__ == '__main__':
    main()
