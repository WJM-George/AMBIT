"""Shared Generation AR sampler repair preserving every original global batch."""
import numpy as np


class ShuffledGlobalBatchSampler:
    """Shuffle complete DDP batches; retain legacy membership and uneven tail.

    Every rank uses the same batch permutation. This changes no padding,
    per-epoch sample frequency, or within-batch rank assignment.
    """

    def __init__(self, base_sampler):
        self.base = base_sampler

    def __len__(self):
        return len(self.base)

    def set_epoch(self, epoch):
        self.base.set_epoch(epoch)

    def __iter__(self):
        local = np.fromiter(iter(self.base), dtype=np.int64, count=len(self.base))
        full_local = self.base.full_rows // self.base.num_replicas
        batches = local[:full_local].reshape(-1, self.base.batch_size)
        rng = np.random.default_rng(np.random.SeedSequence([self.base.seed, self.base.epoch, 20260905]))
        for batch in batches[rng.permutation(len(batches))]:
            yield from map(int, batch)
        yield from map(int, local[full_local:])
