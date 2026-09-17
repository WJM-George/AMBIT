"""Native joint AR/RF/structured FT baseline, without request-side OPSD.

Use the identical model scope, paired dataset, native inference and nine-metric
evaluation. No execution-teacher collection or extra request/decoded losses are
performed. This is distinct from the public-retention ablation.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.t2a.rl import train_editing_opsd_stream as base

# Capture the original learner before the spatial executable installs its
# process-local overrides. Its native paired loss and optimizer are reused.
NativeLearner = base.Learner
from scripts.t2a.rl.train_editing_opsd_spatial import SpatialLearner


class NativeFTLearner(NativeLearner):
    def __init__(self, q, rank, world, out):
        if q['request_rows_per_rank'] != 0 or q['global_request_batch'] != 0:
            raise ValueError('Pure FT must not collect or fit request-only examples.')
        if q['connected_credit'] or q.get('initial_overlay') is not None:
            raise ValueError('The native FT baseline starts from the common original checkpoint.')
        if q.get('execution_schedule'):
            raise ValueError('Use a fixed paired batch for this FT baseline.')
        super().__init__(q, rank, world, out)
        base.write(out / f'FT_OBJECTIVES_rank{rank}.json', dict(
            base_checkpoint=q['base_checkpoint'], native_loss_weights=self.cfg['loss_weights'],
            positive_OPSD=False, request_constraints=False, reference_KL=False,
            extra_decoded_spatial_losses=False, request_teacher_audio_generated=0,
            evaluation='Same native inference and frozen nine metrics as OPSD.',
            information='Only the original paired partition is used; request-only targets remain unread.',
            compute='For an efficiency comparison, match total GPU time, not just update count.'))

    def collect(self, ordinal):
        raise RuntimeError('Request-side feedback is disabled in native FT.')

    def backward_extra(self, batches):
        return dict(native_FT_only=True, extra_objectives=False)

    evaluate = SpatialLearner.evaluate
    record_evaluation = SpatialLearner.record_evaluation


base.Learner = NativeFTLearner

if __name__ == '__main__':
    base.main()
