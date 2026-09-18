"""Eight-rank repaired OPSD: original40k or a declared250 weight initialization.

Fresh Adam and sample streams at update0; recovery restores this new run only.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from scripts.t2a.rl.train_editing_opsd_fresh2000 import FreshLearner
from scripts.t2a.rl.train_editing_opsd_eight_gpu import validate_state as validate_eight_state, EXECUTION
from scripts.t2a.rl.train_editing_opsd_throughput import ThroughputLearner
from scripts.t2a.rl.train_editing_opsd_stability import StabilityLearner, base
from stable_audio_tools.training.transfusion_opsd.top_checkpoints import GUARDED_RANKING
from stable_audio_tools.training.transfusion_opsd.editing_stream import OrdinalStream
from stable_audio_tools.training.transfusion_opsd.request_paired_supervision import RECIPE
from stable_audio_tools.training.transfusion_opsd.branch_request_supervision import RECIPE as BRANCH_RECIPE, active as branch_recipe

POLICY=dict(keep=5,ranking=GUARDED_RANKING,maximum_overall_regression_percent=1.,
            maximum_operation_regression_percent=3.)


def validate_configuration(q):
    if q.get('request_paired_correction') is not None:
        if q['request_paired_correction'] not in (RECIPE, BRANCH_RECIPE) or q.get('removal_repair') is not None:
            raise ValueError('Require one declared all-operation fallback, without duplicate removal loss.')
        if q['request_paired_correction'] == BRANCH_RECIPE and any(
                q['spatial_recipe'][key] != 1. for key in ('ar_teacher_weight', 'terminal_RF_weight')):
            raise ValueError('Execution and GT must share the declared unit branch budgets.')
    elif q.get('removal_repair') != dict(version='removal_paired_v1',paired_native_weight=1.):
        raise ValueError('Missing a declared request/removal correction recipe.')
    if (q['physical_gpus']!=list(range(8)) or q['base_checkpoint']['step']!=40000
            or q['initial_overlay'] is not None or q.get('resize_parent_config') or q.get('execution_schedule')
            or q['global_request_batch']!=16 or q['global_paired_batch']!=512
            or q['request_rows_per_rank']!=2 or q['paired_rows_per_rank']!=64
            or q['paired_microbatch']!=48 or q['complete_recipe']['decoded_rows_per_rank']!=1
            or q['maximum_updates']!=2000 or q['candidate_every']!=250 or q['save_every']!=500
            or q['connected_credit'] or q['development_ordinals']
            or not q['defer_audio_evaluation'] or q['evaluation_save_audio']
            or len(q['validation_ordinals'])!=500 or len(set(q['validation_ordinals']))!=500
            or len(set(q['evaluation_seeds']))!=2 or len(q['evaluation_seeds'])!=2
            or q['top_checkpoint_policy']!=POLICY
            or q.get('paired_catalog_scope')!='full_train_1m'):
        raise ValueError('Require the declared fresh eight-rank repaired2000 recipe and fixed validation guardrails.')
    if q.get('initialization_checkpoint') is not None and q['initialization_checkpoint']['step']!=250:
        raise ValueError('Only original40k or the user-selected250 weight initialization is allowed.')


def validate_state(q,state):
    if state['step']>0:
        validate_eight_state(q,state)
        return
    if (state['step']!=0 or state['world_size']!=8 or state['optimizer']['state']
            or state['config_sha256']!=base.sha(q['config_path'])
            or state['original_checkpoint']!=q['base_checkpoint'] or state['initial_overlay'] is not None
            or state['execution']!={k:q[k] for k in EXECUTION} or len(state['rank_states'])!=8):
        raise ValueError('Fresh update0 must have identified weights, empty Adam and all eight rank states.')


class RepairedFreshLearner(FreshLearner):
    def __init__(self,q,rank,world,out):
        validate_configuration(q)
        if world!=8:
            raise ValueError('Use all eight authorized GPUs.')
        ThroughputLearner.__init__(self,q,rank,world,out)
        self.performance_dir=out/'startup_checks';self.performance_dir.mkdir(exist_ok=True)
        self.expected_first_update=None;self.runtime_ready=False
        if len(self.paired)!=1_000_000:
            raise ValueError('The declared full training catalogue must contain1M paired rows.')
        self.pair_stream=OrdinalStream(range(len(self.paired)),seed=q['seed']+2,rank=rank,world=world)
        self.partition=dict(request_stream=len(self.request_stream.ordinals),
            paired_stream=len(self.pair_stream.ordinals),rows=len(self.paired),
            stream_overlap=len(self.request_stream.ordinals),
            request_label_access=('all operations lacking joint execution supervision: native paired backward only'
                if q.get('request_paired_correction') else 'event_removal only: separately logged native paired correction'))

    def prepare_runtime(self,*,fresh):
        if fresh and (self.step!=0 or self.optimizer.state):
            raise ValueError('A new run must start with empty Adam at update0.')
        initialized=self.q.get('initialization_checkpoint')
        FreshLearner.prepare_runtime(self,fresh=fresh and initialized is None)
        if branch_recipe(self.q):
            from scripts.t2a.rl.validate_editing_opsd_branches import validate_runtime
            if fresh:
                validate_runtime(self)
            else:
                report=base.read(self.performance_dir/f'BRANCH_RUNTIME_rank{self.rank}.json')
                if report['phase']!='PASS' or not report['weights_Adam_RNG_and_samplers_unchanged']:
                    raise ValueError('Resume requires completed real branch acceptance.')
                self.request_cache_enabled=report['performance']['selected']
        if fresh and initialized is not None:
            base.write(self.performance_dir/f'WEIGHT_INITIALIZATION_rank{self.rank}.json',dict(
                source=initialized,step=0,optimizer='fresh_AdamW',samplers='fresh',
                frozen_reference=self.q['base_checkpoint']))

    def resume(self,path,*,resize=False,state=None):
        if resize or state is None:
            raise ValueError('Resume only a complete state from this new run.')
        validate_state(self.q,state)
        if set(state['model'])!=set(self.trainable):
            raise ValueError('Recovery changed the trainable scope.')
        elapsed=StabilityLearner.resume(self,path,resize=False,state=state)
        self.prepare_runtime(fresh=state['step']==0)
        if base.tensor_digest(self.trainable)!=state['model_sha256']:
            raise RuntimeError('Recovery/probes changed model weights.')
        return elapsed


if __name__=='__main__':
    base.Learner=RepairedFreshLearner
    base.main()
