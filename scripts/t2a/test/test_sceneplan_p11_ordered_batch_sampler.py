#!/usr/bin/env python3
"""Regression tests for exact, replay-free P11 DDP cursor recovery."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.resumable_dataloader import ResumableDataLoader  # noqa: E402
from stable_audio_tools.data.sceneplan_p11_ordered_sampler import (  # noqa: E402
    DistributedP11OrderedBatchSampler,
)


class CountingDataset(Dataset):
    def __init__(self, size: int):
        self.size = int(size)
        self.reads: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        self.reads.append(int(index))
        return torch.tensor(int(index))


def _loader(
    dataset: Dataset,
    *,
    rank: int,
    dataset_fingerprint: str = f"sha256:{'1' * 64}",
) -> ResumableDataLoader:
    sampler = DistributedP11OrderedBatchSampler(
        dataset,
        batch_size=8,
        num_replicas=8,
        rank=rank,
        dataset_fingerprint=dataset_fingerprint,
    )
    return ResumableDataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        in_order=True,
    )


class DistributedP11OrderedBatchSamplerTests(unittest.TestCase):
    def test_exactly_reproduces_column_major_distributed_sampler_order(self):
        dataset = CountingDataset(1920)
        rank_batches = [list(_loader(dataset, rank=rank).batch_sampler) for rank in range(8)]
        self.assertTrue(all(len(batches) == 30 for batches in rank_batches))
        reconstructed: list[int] = []
        for step in range(30):
            for local_position in range(8):
                for rank in range(8):
                    reconstructed.append(rank_batches[rank][step][local_position])
        self.assertEqual(reconstructed, list(range(1920)))

    def test_rank_zero_checkpoint_restores_rank_three_without_prefix_reads(self):
        source_dataset = CountingDataset(1920)
        source = _loader(source_dataset, rank=0)
        iterator = iter(source)
        for _ in range(17):
            next(iterator)
        state = source.state_dict()
        self.assertEqual(state["batches_yielded"], 17)
        self.assertEqual(
            state["batch_sampler_state"]["checkpoint_writer_rank"], 0
        )

        resumed_dataset = CountingDataset(1920)
        resumed = _loader(resumed_dataset, rank=3)
        resumed.load_state_dict(state)
        resumed.assert_resume_state_loaded()
        observed = next(iter(resumed)).tolist()
        expected = [
            (17 * 8 + local_position) * 8 + 3 for local_position in range(8)
        ]
        self.assertEqual(observed, expected)
        # Only the delivered batch was read; the 17-batch prefix was skipped at
        # the sampler level before DataLoader iteration.
        self.assertEqual(resumed_dataset.reads, expected)

    def test_resume_rejects_changed_dataset_or_world(self):
        source = _loader(CountingDataset(1920), rank=0)
        next(iter(source))
        state = source.state_dict()
        with self.assertRaisesRegex(ValueError, "epoch length changed"):
            _loader(CountingDataset(1984), rank=0).load_state_dict(state)

        sampler = DistributedP11OrderedBatchSampler(
            CountingDataset(1920),
            batch_size=16,
            num_replicas=4,
            rank=0,
            dataset_fingerprint=f"sha256:{'1' * 64}",
        )
        target = ResumableDataLoader(
            sampler.dataset,
            batch_sampler=sampler,
            num_workers=0,
            in_order=True,
        )
        with self.assertRaisesRegex(ValueError, "batch_size changed"):
            target.load_state_dict(state)

    def test_resume_rejects_same_length_changed_content(self):
        source = _loader(
            CountingDataset(1920),
            rank=0,
            dataset_fingerprint=f"sha256:{'a' * 64}",
        )
        next(iter(source))
        state = source.state_dict()

        target = _loader(
            CountingDataset(1920),
            rank=0,
            dataset_fingerprint=f"sha256:{'b' * 64}",
        )
        with self.assertRaisesRegex(ValueError, "dataset_fingerprint changed"):
            target.load_state_dict(state)

    def test_requires_real_sha256_fingerprint_shape(self):
        with self.assertRaisesRegex(ValueError, "sha256:<64 hex>"):
            _loader(
                CountingDataset(1920),
                rank=0,
                dataset_fingerprint="sha256:not-a-content-digest",
            )

    def test_epoch_boundary_checkpoint_starts_next_epoch_at_batch_zero(self):
        source = _loader(CountingDataset(1920), rank=0)
        for _ in source:
            pass
        state = source.state_dict()
        self.assertTrue(state["at_epoch_boundary"])
        self.assertEqual(state["batches_yielded"], 0)
        self.assertEqual(state["batch_sampler_state"]["resume_epoch"], 1)

        resumed_dataset = CountingDataset(1920)
        resumed = _loader(resumed_dataset, rank=5)
        resumed.load_state_dict(state)
        observed = next(iter(resumed)).tolist()
        expected = [local_position * 8 + 5 for local_position in range(8)]
        self.assertEqual(observed, expected)
        self.assertEqual(resumed_dataset.reads, expected)


if __name__ == "__main__":
    unittest.main()
