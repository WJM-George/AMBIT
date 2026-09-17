"""Skip native sampler indices before I/O, preserving its full1M permutation."""
import itertools

from torch.utils.data import BatchSampler, DistributedSampler


class NativeResumeBatches:
    def __init__(self, dataset, *, rank, world, seed, pairs_per_rank, epoch, next_batch, limit=None):
        self.sampler = DistributedSampler(dataset, num_replicas=world, rank=rank,
                                          shuffle=True, seed=seed, drop_last=True)
        self.sampler.set_epoch(epoch)
        self.batches = BatchSampler(self.sampler, batch_size=pairs_per_rank, drop_last=True)
        if not 0 <= next_batch <= len(self.batches):
            raise ValueError('Invalid native batch offset')
        self.next_batch = next_batch
        self.remaining = len(self.batches) - next_batch
        if limit is not None:
            if limit < 0:
                raise ValueError('Invalid batch limit')
            self.remaining = min(self.remaining, limit)

    def __iter__(self):
        return itertools.islice(iter(self.batches), self.next_batch, self.next_batch + self.remaining)

    def __len__(self):
        return self.remaining
