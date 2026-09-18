"""Eight-GPU continuation of the fresh40k experiment at fixed global budgets."""
import copy
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.distributed as dist

from scripts.t2a.rl.train_editing_opsd_fresh2000 import FreshLearner, validate_state as validate_four_state
from scripts.t2a.rl.train_editing_opsd_throughput import ThroughputLearner
from scripts.t2a.rl.train_editing_opsd_stability import StabilityLearner, base
from stable_audio_tools.training.transfusion_opsd.editing_stream import resize_stream_position

EXECUTION = ('request_rows_per_rank', 'global_request_batch', 'paired_rows_per_rank',
             'paired_microbatch', 'global_paired_batch', 'native_plan_audit_every')
LAYOUT = {'physical_gpus', 'request_rows_per_rank', 'paired_rows_per_rank', 'output',
          'resize_parent_config', 'config_path', 'authorization'}


def validate_contract(q, parent):
    if (parent['physical_gpus'] != [4,5,6,7] or q['physical_gpus'] != list(range(8))
            or q['request_rows_per_rank'] != 2 or q['paired_rows_per_rank'] != 64
            or q['global_request_batch'] != 16 or q['global_paired_batch'] != 512
            or q['complete_recipe']['decoded_rows_per_rank'] != 1
            or parent['complete_recipe']['decoded_rows_per_rank'] != 2
            or q['development_ordinals'] or q.get('execution_schedule')
            or Path(q['output']).resolve() == Path(parent['output']).resolve()):
        raise ValueError('Require an isolated4→8 migration preserving global16+512 and8 decoded rows.')
    def objective(value, world):
        result = copy.deepcopy({k:v for k,v in value.items() if k not in LAYOUT})
        result['complete_recipe']['decoded_rows_per_rank'] *= world
        return result
    if objective(q,8) != objective(parent,4):
        raise ValueError('Migration cannot change losses, learning rates, data, validation or retention.')


def validate_state(q, state):
    if (not 0 < state['step'] <= 2000 or state['world_size'] != 8
            or state['config_sha256'] != base.sha(q['config_path'])
            or state['original_checkpoint'] != q['base_checkpoint'] or state['initial_overlay'] is not None
            or state['execution'] != {k:q[k] for k in EXECUTION} or state.get('execution_schedule')):
        raise ValueError('Eight-GPU recovery identity or execution mismatch.')
    groups, states = state['optimizer']['param_groups'], state['optimizer']['state']
    if ({g['group_name']:g['lr'] for g in groups} != q['learning_rates']
            or any(tuple(g['betas']) != (.9,.95) or g['weight_decay'] != .001 for g in groups)
            or len(states) != len(state['model']) or any(int(s['step']) != state['step'] or
                not {'exp_avg','exp_avg_sq'} <= set(s) for s in states.values())):
        raise ValueError('Require complete, unchanged Adam moments.')
    if len(state['rank_states']) != 8:
        raise ValueError('Missing rank states.')
    for rank, row in enumerate(state['rank_states']):
        if not {'random','numpy','cpu_rng','cuda_rng'} <= set(row):
            raise ValueError('Missing RNG state.')
        for key, offset in [('request',1),('paired',2)]:
            s=row[key]
            if (s['rank'],s['world'],s['seed']) != (rank,8,q['seed']+offset):
                raise ValueError('Sampler identity changed.')


def migrated_rank_states(rows):
    result=[]
    for rank in range(8):
        local=copy.deepcopy(rows[rank%4])
        for key in ('request','paired'):
            local[key]=resize_stream_position([r[key] for r in rows],new_world=8,rank=rank)
        keys=set().union(*(r['costs'] for r in rows))
        local['costs']={k:sum(r['costs'].get(k,0) for r in rows)/8 for k in keys}
        result.append(local)
    return result


class EightLearner(FreshLearner):
    def __init__(self,q,rank,world,out):
        validate_contract(q,base.read(q['resize_parent_config']))
        if world != 8:
            raise ValueError('This entrypoint requires all8 authorized GPUs.')
        # Reuse the proven objective/cache implementation, with the new
        # explicit contract replacing FreshLearner's four-rank-only guard.
        ThroughputLearner.__init__(self,q,rank,world,out)
        self.performance_dir=out/'startup_checks'
        self.performance_dir.mkdir(exist_ok=True)
        self.expected_first_update=None
        self.runtime_ready=False

    def resume(self,path,*,resize=False,state=None):
        if state is None:
            raise ValueError('Eight-GPU training must resume an explicit full state.')
        migrating=state['world_size']==4
        if migrating != resize:
            raise ValueError('Use --resize exactly once for the four-rank transition.')
        if migrating:
            parent_path=self.q['resize_parent_config']
            parent=dict(base.read(parent_path),config_path=parent_path)
            validate_four_state(parent,state)
            converted=dict(state,world_size=8,config_sha256=base.sha(self.q['config_path']),
                execution={k:self.q[k] for k in EXECUTION},rank_states=migrated_rank_states(state['rank_states']))
        else:
            converted=state
        validate_state(self.q,converted)
        if set(converted['model']) != set(self.trainable):
            raise ValueError('Full trainable scope changed.')
        elapsed=StabilityLearner.resume(self,path,resize=False,state=converted)
        new_seed=None
        if migrating:
            # A new distributed topology is not bit-identical stochastic
            # replay. Model/Adam and global consumed examples stay intact.
            new_seed=self.q['seed']+self.step*1009+self.rank
            random.seed(new_seed);np.random.seed(new_seed);torch.manual_seed(new_seed)
        self.prepare_runtime(fresh=False)
        if base.tensor_digest(self.trainable) != state['model_sha256']:
            raise RuntimeError('Migration/probes changed resumed model weights.')
        base.write(self.performance_dir/f'RESUME_step{self.step:06d}_rank{self.rank}.json',dict(
            source=str(path),source_world=state['world_size'],current_world=8,step=self.step,
            model_sha256=state['model_sha256'],complete_Adam_preserved=True,
            global_request_batch=16,global_paired_batch=512,global_decoded_rows=8,
            request=self.request_stream.state_dict(),paired=self.pair_stream.state_dict(),
            new_topology_seed=new_seed,bit_exact_old_topology_replay=False if migrating else None,
            expected_first_update=self.expected_first_update))
        if migrating:
            self.save(elapsed)
            if self.rank==0:
                protected=self.out/'migration_start.pt'
                if not protected.exists():os.link(self.out/'resume_latest.pt',protected)
            dist.barrier()
        return elapsed


def main():
    if '--resume' not in sys.argv:
        raise ValueError('Resume the existing fresh40k experiment; never reset its optimizer.')
    base.Learner=EightLearner
    base.main()


if __name__=='__main__':
    main()
