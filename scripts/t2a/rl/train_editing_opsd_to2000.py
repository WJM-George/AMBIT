"""Isolated, exact-state continuation of the four-rank step500 OPSD run.

Only validation, checkpoint retention and artifact placement change. Native
training objectives, optimizer, learning rates, batch and streams stay fixed.
Audio evaluation runs separately against immutable milestone checkpoints.
"""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch

from scripts.t2a.rl.train_editing_opsd_throughput import ThroughputLearner, peek
from scripts.t2a.rl.train_editing_opsd_stability import StabilityLearner, base


CONTROL_FIELDS = {'output', 'validation_ordinals', 'evaluation_seeds', 'initial_evaluation_reference',
                  'top_checkpoint_policy', 'checkpoint_retention', 'evaluate_every_seconds',
                  'continuation_parent_config', 'defer_audio_evaluation', 'evaluation_save_audio',
                  'authorization', 'continuation_policy', 'config_path'}


def validate_configuration(current, parent):
    if {k:v for k,v in current.items() if k not in CONTROL_FIELDS} != {
            k:v for k,v in parent.items() if k not in CONTROL_FIELDS}:
        raise ValueError('Continuation cannot change training data, objectives, optimizer or execution.')
    if (len(current['physical_gpus']) != 4 or current['maximum_updates'] != 2000)
            or current['save_every'] != 500 or current['global_request_batch'] != 16
            or current['global_paired_batch'] != 512 or current['paired_microbatch'] != 48
            or current['connected_credit'] or not current['defer_audio_evaluation']
            or current['evaluation_save_audio'] or current['recovery_every_seconds'] != 600):
        raise ValueError('Unexpected continuation contract.')
    if (len(current['validation_ordinals']) != 500 or len(set(current['validation_ordinals'])) != 500
            or len(current['evaluation_seeds']) != 2 or len(set(current['evaluation_seeds'])) != 2
            or current['top_checkpoint_policy'] != dict(keep=3, ranking='mean_signed_relative_improvement_percent')):
        raise ValueError('Require a frozen500x2 development panel and Top3.')
    if Path(current['output']).resolve() == Path(parent['output']).resolve():
        raise ValueError('Preserve the original500-step training directory.')


def validate_state(q, state, allowed_config_hashes):
    if (not 500 <= state['step'] <= 2000 or state['world_size'] != 4
            or state['config_sha256'] not in allowed_config_hashes
            or state['original_checkpoint'] != q['base_checkpoint'] or state['initial_overlay'] != q['initial_overlay']):
        raise ValueError('Invalid four-rank continuation checkpoint.')
    fields = ('request_rows_per_rank', 'global_request_batch', 'paired_rows_per_rank',
              'paired_microbatch', 'global_paired_batch', 'native_plan_audit_every')
    if state['execution'] != {k:q[k] for k in fields} or state.get('execution_schedule') != q.get('execution_schedule'):
        raise ValueError('The recorded batch/sampling execution changed.')
    groups = state['optimizer']['param_groups']
    if {g['group_name']:g['lr'] for g in groups} != q['learning_rates']:
        raise ValueError('Adam learning rates changed.')
    if any(tuple(g['betas']) != (.9, .95) or g['weight_decay'] != .001 for g in groups):
        raise ValueError('Adam hyperparameters changed.')
    states = state['optimizer']['state']
    if len(states) != len(state['model']) or any(int(v['step']) != state['step'] or
            not {'exp_avg', 'exp_avg_sq'} <= set(v) for v in states.values()):
        raise ValueError('Full Adam moments are required for every trained tensor.')
    if len(state['rank_states']) != 4:
        raise ValueError('Missing rank state.')
    for rank, local in enumerate(state['rank_states']):
        if not {'random', 'numpy', 'cpu_rng', 'cuda_rng'} <= set(local):
            raise ValueError('Missing saved RNG.')
        for key, offset in [('request', 1), ('paired', 2)]:
            stream = local[key]
            if (stream['rank'], stream['world'], stream['seed']) != (rank, 4, q['seed'] + offset):
                raise ValueError('Sampler identity changed.')


class ContinuationLearner(ThroughputLearner):
    def resume(self, path, *, resize=False, state=None):
        if resize or state is None:
            raise ValueError('Use an explicit, unchanged four-rank recovery.')
        parent_path = self.q['continuation_parent_config']
        parent = base.read(parent_path)
        validate_configuration(self.q, parent)
        current_path = self.q['config_path']
        parent_sha, current_sha = base.sha(parent_path), base.sha(current_path)
        validate_state(self.q, state, {parent_sha, current_sha})
        if set(state['model']) != set(self.trainable):
            raise ValueError('The full trainable parameter set changed.')

        def restore():
            # Configuration compatibility was checked field by field above.
            # The original resumer is given the checkpoint's verified config
            # path; only artifact/evaluation metadata is restored afterwards.
            self.q['config_path'] = parent_path if state['config_sha256'] == parent_sha else current_path
            try:
                return StabilityLearner.resume(self, path, resize=False, state=state)
            finally:
                self.q['config_path'] = current_path

        elapsed = restore()
        self.performance_dir = self.out / 'continuation_checks'
        self.performance_dir.mkdir(exist_ok=True)
        # Check the exact upcoming native plans, then discard every probe.
        selected_elision, trials = self.profile_planning()
        if state['step'] == 500:
            from scripts.t2a.rl.validate_editing_opsd_native import validate_native
            validate_native(self, self.adapter, max_batches=2, label='startup_probe')
        restore()
        self.forward_cache.clear()
        self.optimizer.zero_grad(set_to_none=True)
        self.plan_batch_cap = 1
        for fastpath in self.greedy_fastpaths:
            fastpath.enabled = selected_elision
        observed = base.tensor_digest(self.trainable)
        if observed != state['model_sha256']:
            raise RuntimeError('Startup checks changed the resumed model.')
        self.expected_first_update = dict(step=self.step + 1,
            request_ordinals=peek(self.request_stream, self.q['request_rows_per_rank']),
            paired_ordinals=peek(self.pair_stream, self.q['paired_rows_per_rank']))
        base.write(self.performance_dir / f'RESUME_step{self.step:06d}_rank{self.rank}.json', dict(
            source=str(path), step=self.step, model_sha256=observed, complete_Adam_restored=True,
            learning_rates=self.q['learning_rates'], RNG_and_samplers_restored_after_probes=True,
            singleton_forward_elision=selected_elision, paired_microbatch=48,
            native_plan_batch=1, planning_trials=trials, expected_next=self.expected_first_update))
        # Rebind the same model/Adam/RNG to the new artifact contract. The
        # step500 milestone is now eligible for this run's asynchronous Top3.
        self.save(elapsed)
        if self.step % 500 == 0:
            self.evaluate()
        torch.cuda.empty_cache()
        return elapsed

    def train_step(self):
        stats = super().train_step()
        expected = self.expected_first_update
        if expected is not None:
            actual = dict(step=stats['step'],
                          request_ordinals=[r['ordinal'] for r in stats['request_updates']],
                          paired_ordinals=stats['paired_ordinals'])
            if actual != expected:
                raise RuntimeError('The resumed update repeated or skipped saved samples.')
            base.write(self.performance_dir / f'FIRST_UPDATE_{stats["step"]:06d}_rank{self.rank}.json',
                       dict(samples_match_saved_cursors=True, **actual))
            self.expected_first_update = None
        coverage = {}
        for request in stats['request_updates']:
            row = coverage.setdefault(request['requested_operation'], dict(requests=0, qualified_teachers=0,
                request_constraints=0, bound_requests=0))
            row['requests'] += 1
            row['qualified_teachers'] += int(request['enabled'])
            row['request_constraints'] += bool(request['request_constraint_fields'])
            row['bound_requests'] += bool(request['binding']['available'])
        stats['feedback_coverage'] = coverage
        return stats

    def evaluate(self):
        if self.step % 500:
            return
        if self.rank == 0:
            pending = self.out / 'evaluation_pending'
            pending.mkdir(exist_ok=True)
            base.write(pending / f'step-{self.step:08d}.json', dict(
                step=self.step, checkpoint=str(self.out / 'checkpoints' / f'step-{self.step:08d}.pt'),
                configuration_sha256=base.sha(self.q['config_path']), requests=500, seeds=2,
                audio_retained=False, phase='PENDING_EXTERNAL_EVALUATION', created_unix=time.time()))
        # Validation is separate from the optimizer process and cannot change
        # its RNG, sampler cursor, weights, gradient buffers or training time.


def main():
    base.Learner = ContinuationLearner
    base.main()


if __name__ == '__main__':
    main()
