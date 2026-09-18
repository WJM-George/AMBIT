"""Fresh original40k OPSD, with isolated asynchronous validation and Top5."""
import copy
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist

from scripts.t2a.rl.train_editing_opsd_throughput import ThroughputLearner, peek
from scripts.t2a.rl.train_editing_opsd_stability import StabilityLearner, base
from stable_audio_tools.training.transfusion_opsd.request_paired_supervision import feedback_coverage


def validate_configuration(q):
    if (q['initial_overlay'] is not None or q.get('initialization_checkpoint') is not None
            or q['base_checkpoint']['step'] != 40000 or q['maximum_updates'] != 2000
            or q['physical_gpus'] != [4, 5, 6, 7] or q['save_every'] != 500
            or q['global_request_batch'] != 16 or q['global_paired_batch'] != 512
            or q['request_rows_per_rank'] != 4 or q['paired_rows_per_rank'] != 128
            or q['paired_microbatch'] != 48 or q['connected_credit']
            or not q['defer_audio_evaluation'] or q['evaluation_save_audio']):
        raise ValueError('Require a fresh original40k, four-rank 16+512 run to2000.')
    if (len(q['validation_ordinals']) != 500 or len(set(q['validation_ordinals'])) != 500
            or len(q['evaluation_seeds']) != 2 or len(set(q['evaluation_seeds'])) != 2
            or q['top_checkpoint_policy'] != dict(keep=5, ranking='mean_signed_relative_improvement_percent')
            or q['candidate_every'] != 250):
        raise ValueError('Require fixed500x2, candidates every250, and Top5.')
    if q.get('execution_schedule') or q.get('resize_parent_config'):
        raise ValueError('Fresh run cannot inherit an old topology or batch schedule.')
    recipe = q['selective_recipe']
    if not recipe['reference_native_prefix'] or not recipe['select_same_plan_improvements']:
        raise ValueError('Both declared teacher and reference-prefix constraints are required.')


def validate_state(q, state):
    if (not 0 <= state['step'] <= 2000 or state['world_size'] != 4
            or state['config_sha256'] != base.sha(q['config_path'])
            or state['original_checkpoint'] != q['base_checkpoint'] or state['initial_overlay'] is not None):
        raise ValueError('Recovery must belong to this fresh run; old OPSD states are forbidden.')
    fields = ('request_rows_per_rank', 'global_request_batch', 'paired_rows_per_rank',
              'paired_microbatch', 'global_paired_batch', 'native_plan_audit_every')
    if state['execution'] != {k:q[k] for k in fields} or state.get('execution_schedule'):
        raise ValueError('Recovery changed batch execution.')
    groups = state['optimizer']['param_groups']
    if ({g['group_name']:g['lr'] for g in groups} != q['learning_rates']
            or any(tuple(g['betas']) != (.9, .95) or g['weight_decay'] != .001 for g in groups)):
        raise ValueError('Adam configuration changed.')
    states = state['optimizer']['state']
    if state['step'] == 0:
        if states:
            raise ValueError('Fresh optimizer must have no inherited moments.')
    elif len(states) != len(state['model']) or any(int(v['step']) != state['step'] or
            not {'exp_avg', 'exp_avg_sq'} <= set(v) for v in states.values()):
        raise ValueError('Recovery must contain complete Adam moments.')
    if len(state['rank_states']) != 4:
        raise ValueError('Missing rank states.')
    for rank, local in enumerate(state['rank_states']):
        if not {'random', 'numpy', 'cpu_rng', 'cuda_rng'} <= set(local):
            raise ValueError('Missing RNG state.')
        for key, offset in [('request', 1), ('paired', 2)]:
            stream = local[key]
            if (stream['rank'], stream['world'], stream['seed']) != (rank, 4, q['seed'] + offset):
                raise ValueError('Sampler identity changed.')


def freeze_candidate(out, step):
    """Persist an inode before rolling recovery can advance; milestones survive."""
    out = Path(out)
    candidates = out / 'evaluation_candidates'
    candidates.mkdir(exist_ok=True)
    source, target = out / 'resume_latest.pt', candidates / f'step-{step:08d}.pt'
    if not target.exists():
        os.link(source, target)
    return target


class FreshLearner(ThroughputLearner):
    def __init__(self, q, rank, world, out):
        validate_configuration(q)
        super().__init__(q, rank, world, out)
        self.performance_dir = self.out / 'startup_checks'
        self.performance_dir.mkdir(exist_ok=True)
        self.expected_first_update = None
        self.runtime_ready = False

    def prepare_runtime(self, *, fresh):
        before = base.tensor_digest(self.trainable)
        if fresh:
            reference = dict(self.reference.named_parameters())
            original_digest = base.tensor_digest({n:reference[n] for n in self.trainable})
            if before != original_digest:
                raise RuntimeError('Fresh trainable weights must equal the original40k reference.')
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
               torch.cuda.get_rng_state(self.device))
        costs = copy.deepcopy(self.costs)
        streams = (self.request_stream.state_dict(), self.pair_stream.state_dict())
        selected, trials = self.profile_planning()
        if fresh:
            from scripts.t2a.rl.validate_editing_opsd_native import validate_native
            validate_native(self, self.adapter, max_batches=2, label='startup_probe')
            validate_native(self, self.reference, max_batches=2, label='startup_probe_reference')
        self.forward_cache.clear()
        self.optimizer.zero_grad(set_to_none=True)
        random.setstate(rng[0]); np.random.set_state(rng[1])
        torch.set_rng_state(rng[2]); torch.cuda.set_rng_state(rng[3], self.device)
        self.costs = costs
        if streams != (self.request_stream.state_dict(), self.pair_stream.state_dict()):
            raise RuntimeError('A startup probe consumed training samples.')
        if base.tensor_digest(self.trainable) != before:
            raise RuntimeError('A startup probe changed the training weights.')
        if fresh and (self.step != 0 or self.optimizer.state):
            raise RuntimeError('Fresh initialization inherited optimizer state.')
        for fastpath in self.greedy_fastpaths:
            fastpath.enabled = selected
        self.plan_batch_cap = 1
        upcoming = peek(self.request_stream, self.q['request_rows_per_rank'])
        request_ids, used = [], 0
        for j in range(self.q['request_rows_per_rank']):
            index = self.step*self.q['global_request_batch'] + self.rank*self.q['request_rows_per_rank'] + j
            if index < len(self.q['development_ordinals']):
                request_ids.append(self.q['development_ordinals'][index])
            else:
                request_ids.append(upcoming[used]); used += 1
        self.expected_first_update = dict(step=self.step+1, request_ordinals=request_ids,
            paired_ordinals=peek(self.pair_stream, self.q['paired_rows_per_rank']))
        base.write(self.performance_dir/f'PREPARED_step{self.step:06d}_rank{self.rank}.json', dict(
            fresh=fresh, step=self.step, model_sha256=before, optimizer_state_tensors=len(self.optimizer.state),
            original40k_weights_verified=fresh,
            expected_next=self.expected_first_update, singleton_forward_elision=selected,
            planning_trials=trials, paired_microbatch=48, RNG_restored=True, samplers_unchanged=True,
            original_checkpoint=self.q['base_checkpoint']))
        self.runtime_ready = True
        torch.cuda.empty_cache()

    def save(self, elapsed):
        if not self.runtime_ready:
            self.prepare_runtime(fresh=True)
        super().save(elapsed)

    def resume(self, path, *, resize=False, state=None):
        if resize or state is None:
            raise ValueError('Only exact recovery within this fresh run is allowed.')
        validate_state(self.q, state)
        if set(state['model']) != set(self.trainable):
            raise ValueError('Recovery trainable scope changed.')
        elapsed = StabilityLearner.resume(self, path, resize=False, state=state)
        self.prepare_runtime(fresh=state['step'] == 0)
        if base.tensor_digest(self.trainable) != state['model_sha256']:
            raise RuntimeError('Recovered model digest mismatch.')
        return elapsed

    def train_step(self):
        stats = super().train_step()
        if self.expected_first_update is not None:
            actual = dict(step=stats['step'], request_ordinals=[r['ordinal'] for r in stats['request_updates']],
                          paired_ordinals=stats['paired_ordinals'])
            if actual != self.expected_first_update:
                raise RuntimeError('First update did not follow the saved sampler cursors.')
            base.write(self.performance_dir/f'FIRST_UPDATE_{self.step:06d}_rank{self.rank}.json',
                       dict(samples_match_saved_cursors=True, **actual))
            self.expected_first_update = None
        stats['feedback_coverage'] = feedback_coverage(stats['request_updates'],
            require_complete=bool(self.q.get('request_paired_correction')))
        return stats

    def evaluate(self):
        if self.step % self.q['candidate_every']:
            return
        if self.rank == 0:
            checkpoint = freeze_candidate(self.out, self.step)
            pending = self.out / 'evaluation_pending'
            pending.mkdir(exist_ok=True)
            base.write(pending/f'step-{self.step:08d}.json', dict(step=self.step, checkpoint=str(checkpoint),
                configuration_sha256=base.sha(self.q['config_path']), requests=500, seeds=2,
                audio_retained=False, phase='PENDING_EXTERNAL_EVALUATION', created_unix=time.time()))
        dist.barrier()


def main():
    base.Learner = FreshLearner
    base.main()


if __name__ == '__main__':
    main()
