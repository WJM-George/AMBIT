"""Explicit, isolated recipe transition for the removal repair.

The parent remains paused. Only the declared loss change is permitted;
weights, Adam moments, RNG and both sample cursors survive the transition.
An explicit --limit-updates is required, including for a bounded smoke run.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.t2a.rl.train_editing_opsd_eight_gpu import validate_state
from scripts.t2a.rl.train_editing_opsd_fresh2000 import FreshLearner
from scripts.t2a.rl.train_editing_opsd_throughput import ThroughputLearner
from scripts.t2a.rl.train_editing_opsd_stability import StabilityLearner, base


def validate_repair(q, parent):
    recipe = q['removal_repair']
    allowed = {'schema', 'output', 'config_path', 'authorization', 'supervision', 'removal_repair'}
    if ({k: v for k, v in q.items() if k not in allowed}
            != {k: v for k, v in parent.items() if k not in allowed}):
        raise ValueError('Removal repair cannot change data, batches, rates, validation or other losses.')
    if (recipe['version'] != 'removal_paired_v1' or recipe['paired_native_weight'] != 1.
            or q['physical_gpus'] != list(range(8)) or parent.get('removal_repair')
            or Path(q['output']).resolve() == Path(parent['output']).resolve()):
        raise ValueError('Require an isolated eight-rank removal correction at native weight1.')
    if base.sha(recipe['parent_config']['path']) != recipe['parent_config']['sha256']:
        raise ValueError('Parent configuration changed.')


def transition_state(q, parent, state, path):
    validate_repair(q, parent)
    if state['config_sha256'] == base.sha(q['config_path']):
        validate_state(q, state)
        return state, False
    record = q['removal_repair']['parent_checkpoint']
    if (Path(path).resolve() != Path(record['path']).resolve() or base.sha(path) != record['sha256']
            or state['step'] != record['step'] or state['model_sha256'] != record['model_sha256']):
        raise ValueError('Use only the protected, identified paused parent state.')
    validate_state(dict(parent, config_path=q['removal_repair']['parent_config']['path']), state)
    converted = dict(state, config_sha256=base.sha(q['config_path']))
    validate_state(q, converted)
    return converted, True


class RemovalRepairLearner(FreshLearner):
    def __init__(self, q, rank, world, out):
        parent = base.read(q['removal_repair']['parent_config']['path'])
        validate_repair(q, parent)
        if world != 8:
            raise ValueError('Preserve the paused eight-rank topology.')
        ThroughputLearner.__init__(self, q, rank, world, out)
        self.performance_dir = out / 'startup_checks'
        self.performance_dir.mkdir(exist_ok=True)
        self.expected_first_update = None
        self.runtime_ready = False
        self.partition = dict(request_stream=len(self.request_stream.ordinals),
            paired_stream=len(self.pair_stream.ordinals), rows=len(self.paired), stream_overlap=0,
            request_label_access='event_removal only: explicit paired correction in backward')

    def resume(self, path, *, resize=False, state=None):
        if resize or state is None:
            raise ValueError('Repair transition requires the full eight-rank paused state.')
        parent = base.read(self.q['removal_repair']['parent_config']['path'])
        converted, transition = transition_state(self.q, parent, state, path)
        if set(converted['model']) != set(self.trainable):
            raise ValueError('Repair changed the trainable scope.')
        elapsed = StabilityLearner.resume(self, path, resize=False, state=converted)
        self.prepare_runtime(fresh=False)
        if base.tensor_digest(self.trainable) != state['model_sha256']:
            raise RuntimeError('Repair startup changed the recovered model.')
        base.write(self.performance_dir / f'REPAIR_RESUME_rank{self.rank}.json', dict(
            recipe_transition=transition, parent=str(path), step=self.step,
            model_sha256=state['model_sha256'], complete_Adam_preserved=True,
            RNG_and_samplers_preserved=True, expected_first_update=self.expected_first_update,
            paired_correction_role='explicit paired labels; not execution feedback',
            original_run_automatically_resumed=False))
        return elapsed


def main():
    if '--resume' not in sys.argv or '--limit-updates' not in sys.argv:
        raise ValueError('Specify the protected resume and an explicit cumulative update cap.')
    base.Learner = RemovalRepairLearner
    base.main()


if __name__ == '__main__':
    main()
