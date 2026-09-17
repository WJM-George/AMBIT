"""Continue a selected two-rank OPSD candidate on GPUs4-7 in a new directory.

Model, Adam moments and global stream positions are resumed. Only rank
topology and artifact storage/retention may change at this boundary.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.t2a.rl.train_editing_opsd_selective import SelectiveLearner, CompleteLearner, base


ARTIFACT_FIELDS = {'output', 'top_checkpoint_policy', 'checkpoint_retention'}
RESIZE_FIELDS = {'physical_gpus', 'paired_rows_per_rank', 'request_rows_per_rank',
                 'resize_parent_config', 'config_path', 'paired_microbatch',
                 'global_paired_batch', 'native_plan_audit_every', 'execution_schedule'}


def validate_isolated_resize(current, parent):
    if current['physical_gpus'] != [4, 5, 6, 7] or len(parent['physical_gpus']) != 2:
        raise ValueError('Stability continuation requires the declared two-to-four-rank migration.')
    if Path(current['output']).resolve() == Path(parent['output']).resolve():
        raise ValueError('Preserve the diagnostic candidate in its original directory.')
    for q in (current, parent):
        world = len(q['physical_gpus'])
        if (q['request_rows_per_rank'] * world != q['global_request_batch'] or
                q['paired_rows_per_rank'] * world != q['global_paired_batch'] or
                (q['global_request_batch'], q['global_paired_batch']) != (16, 512)):
            raise ValueError('The global16+512 batch must remain unchanged.')
    ignored = ARTIFACT_FIELDS | RESIZE_FIELDS
    if ({k: v for k, v in current.items() if k not in ignored} !=
            {k: v for k, v in parent.items() if k not in ignored}):
        raise ValueError('Stability continuation cannot change the recipe, optimizer, schedule or data.')
    if (current.get('top_checkpoint_policy') !=
            dict(keep=1, ranking='mean_signed_relative_improvement_percent') or
            current.get('checkpoint_retention') != dict(rolling_recoveries=1)):
        raise ValueError('Use one top candidate and one rolling recovery for this stability check.')


class StabilityLearner(SelectiveLearner):
    def collect(self, ordinal):
        if 'selective_recipe' in self.q:
            return super().collect(ordinal)
        return CompleteLearner.collect(self, ordinal)

    def backward_self(self, item, *, scale=1.):
        if 'selective_recipe' in self.q:
            return super().backward_self(item, scale=scale)
        return CompleteLearner.backward_self(self, item, scale=scale)

    def resume(self, path, *, resize=False, state=None):
        if not resize:
            return super().resume(path, resize=False, state=state)
        parent = base.read(self.q['resize_parent_config'])
        current = self.q
        validate_isolated_resize(current, parent)
        # The existing resumer verifies the source configuration hash,
        # checkpoint identity, optimizer and sampler migration. It predates
        # isolated output directories, so normalize only artifact metadata
        # after validating the complete configuration above. self.out remains
        # the new directory throughout; the source checkpoint is read-only.
        compatible = dict(current)
        for key in ARTIFACT_FIELDS:
            if key in parent:
                compatible[key] = parent[key]
            else:
                compatible.pop(key, None)
        try:
            self.q = compatible
            elapsed = super().resume(path, resize=True, state=state)
        finally:
            self.q = current
        base.write(self.out / f'ISOLATED_RESUME_rank{self.rank}.json', dict(
            step=self.step, source_checkpoint=str(path), source_configuration=current['resize_parent_config'],
            original_output_preserved=parent['output'], current_output=str(self.out),
            optimizer='restored_AdamW_moments', global_request_batch=16, global_paired_batch=512,
            loss_recipe_unchanged=True, topology='2_to_4', bit_exact_old_topology_replay=False))
        return elapsed


base.Learner = StabilityLearner

if __name__ == '__main__':
    base.main()
