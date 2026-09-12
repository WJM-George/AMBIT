"""Regroup only unconsumed three-GPU batches at the six-GPU boundary.

Each new batch combines two old batches of the same audio length. The ordinary
drop-last rule applies to at most one unmatched old batch per length bucket.
Following epochs use the ordinary six-rank sampler over the full dataset.
"""
from __future__ import annotations

from stable_audio_tools.data.sceneplan_bucket_sampler import DistributedScenePlanBucketBatchSampler

POLICY = 'unconsumed_parent_batches_paired_by_length_then_full_epochs_v1'


def make_boundary(dataset, *, seed, epoch, next_batch, checkpoint):
    parent = DistributedScenePlanBucketBatchSampler(dataset, short_batch_size=192,
        long_batch_size=120, num_replicas=1, rank=0, shuffle=True, seed=seed, drop_last=True)
    state = parent.resumable_state_dict(at_epoch_boundary=True)
    state['resume_epoch'] = epoch
    parent.load_resumable_state_dict(state)
    if not 0 <= next_batch <= len(parent):
        raise ValueError('Invalid parent sampler cursor')
    batches = []
    pending = {192: None, 120: None}
    remaining_count = 0
    if next_batch < len(parent):
        parent.set_resume_batch_offset(next_batch)
        for indices in parent:
            size = len(indices)
            if size not in pending:
                raise ValueError('Parent batch sizes changed')
            remaining_count += size
            if pending[size] is None:
                pending[size] = indices
            else:
                batches.append(pending[size] + indices)
                pending[size] = None
    dropped = [i for row in pending.values() if row is not None for i in row]
    return {'schema': POLICY, 'checkpoint': checkpoint,
        'parent_epoch': epoch, 'parent_next_batch': next_batch,
        'new_epoch': epoch if batches else epoch + 1, 'new_next_batch': 0,
        'partial_epoch': epoch if batches else None,
        'global_batches': batches, 'dropped_incomplete_batch_ordinals': dropped,
        'remaining_parent_examples': remaining_count,
        'retained_boundary_examples': sum(map(len, batches)),
        'per_gpu_batch_sizes': {'short': 64, 'long': 40},
        'global_batch_sizes': {'short': 384, 'long': 240}}


class ScaleoutSampler:
    """Expose the same cursor API as the immutable original training loop."""
    def __init__(self, dataset, *, seed, rank, boundary):
        expected = make_boundary(dataset, seed=seed, epoch=boundary['parent_epoch'],
            next_batch=boundary['parent_next_batch'], checkpoint=boundary['checkpoint'])
        if expected != boundary:
            raise RuntimeError('Scale-out data boundary differs from the frozen parent stream')
        self.dataset, self.seed, self.rank, self.boundary = dataset, seed, rank, boundary
        self._epoch = boundary['new_epoch']
        self._next_epoch = self._epoch
        self._offset = 0
        self._active = False
        self._full = self._native()

    def _native(self):
        return DistributedScenePlanBucketBatchSampler(self.dataset, short_batch_size=64,
            long_batch_size=40, num_replicas=6, rank=self.rank, shuffle=True,
            seed=self.seed, drop_last=True)

    def __len__(self):
        if self._epoch == self.boundary['partial_epoch']:
            return len(self.boundary['global_batches'])
        return len(self._full)

    def resumable_state_dict(self, *, at_epoch_boundary):
        epoch = self._next_epoch if at_epoch_boundary else self._epoch
        return {'schema': POLICY, 'resume_epoch': epoch, 'rank': self.rank, 'seed': self.seed}

    def load_resumable_state_dict(self, state):
        if (state['schema'] != POLICY or state['rank'] != self.rank or state['seed'] != self.seed
                or state['resume_epoch'] < self.boundary['new_epoch']):
            raise ValueError('Invalid six-rank sampler state')
        self._epoch = self._next_epoch = state['resume_epoch']
        self._offset = 0
        self._active = False

    def set_resume_batch_offset(self, offset):
        if self._active or not 0 <= offset < len(self):
            raise ValueError('Invalid six-rank batch cursor')
        self._offset = offset

    def __iter__(self):
        self._active = True
        self._next_epoch = self._epoch + 1
        offset, self._offset = self._offset, 0
        if self._epoch == self.boundary['partial_epoch']:
            for global_batch in self.boundary['global_batches'][offset:]:
                size = len(global_batch) // 6
                yield global_batch[self.rank * size:(self.rank + 1) * size]
        else:
            sampler = self._native()
            state = sampler.resumable_state_dict(at_epoch_boundary=True)
            state['resume_epoch'] = self._epoch
            sampler.load_resumable_state_dict(state)
            if offset:
                sampler.set_resume_batch_offset(offset)
            yield from sampler
