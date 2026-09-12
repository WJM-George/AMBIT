"""Exact, O(1)-resume batch sampling for ordered P11 DDP curricula."""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any

from torch.utils.data import Sampler


class DistributedP11OrderedBatchSampler(Sampler[list[int]]):
    """Reproduce ``DistributedSampler(shuffle=False)`` without replay on resume.

    The canonical P11 curriculum is already shuffled and emitted column-major:
    physical row ``rank + world_size * local_position`` belongs to ``rank``.
    A regular ``DistributedSampler`` followed by ``BatchSampler`` produces that
    exact order, but cannot skip a restored prefix before DataLoader workers
    start. This sampler emits the same indices and accepts the batch cursor used
    by :class:`ResumableDataLoader`.
    """

    _STATE_SCHEMA = "stable_audio_tools.p11_ordered_batch_sampler"
    _STATE_VERSION = 1

    def __init__(
        self,
        dataset: Sized,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        dataset_fingerprint: str,
    ) -> None:
        self.dataset = dataset
        self.dataset_items = int(len(dataset))
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.dataset_fingerprint = str(dataset_fingerprint).strip().lower()
        if self.dataset_items <= 0:
            raise ValueError("P11 ordered sampler requires a non-empty dataset")
        if self.batch_size <= 0:
            raise ValueError("P11 ordered sampler batch_size must be positive")
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"invalid P11 ordered sampler rank/world: "
                f"{self.rank}/{self.num_replicas}"
            )
        fingerprint_prefix = "sha256:"
        fingerprint_digest = self.dataset_fingerprint.removeprefix(
            fingerprint_prefix
        )
        if (
            not self.dataset_fingerprint.startswith(fingerprint_prefix)
            or len(fingerprint_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in fingerprint_digest
            )
        ):
            raise ValueError(
                "P11 ordered sampler requires dataset_fingerprint=sha256:<64 hex>"
            )
        self.global_batch_size = self.batch_size * self.num_replicas
        if self.dataset_items % self.global_batch_size:
            raise ValueError(
                "P11 ordered sampler requires exact global batches: "
                f"{self.dataset_items} rows are not divisible by "
                f"{self.global_batch_size}"
            )
        self._length = self.dataset_items // self.global_batch_size
        self._resume_batch_offset = 0
        self._active_epoch: int | None = None
        self._next_epoch = 0

    @property
    def shuffle(self) -> bool:
        return False

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self._next_epoch
        self._active_epoch = epoch
        self._next_epoch = epoch + 1
        resume_batch_offset = self._resume_batch_offset
        self._resume_batch_offset = 0
        if not 0 <= resume_batch_offset < self._length:
            raise RuntimeError("P11 ordered sampler resume offset exceeds the epoch")
        for batch_index in range(resume_batch_offset, self._length):
            local_start = batch_index * self.batch_size
            yield [
                (local_start + local_position) * self.num_replicas + self.rank
                for local_position in range(self.batch_size)
            ]

    def set_resume_batch_offset(self, batches_yielded: int) -> None:
        offset = int(batches_yielded)
        if not 0 <= offset < self._length:
            raise ValueError("invalid P11 ordered sampler resume offset")
        if self._active_epoch is not None:
            raise RuntimeError(
                "P11 ordered sampler resume offset must be set before iteration"
            )
        self._resume_batch_offset = offset

    def resumable_state_dict(self, *, at_epoch_boundary: bool) -> dict[str, Any]:
        if self._active_epoch is None:
            resume_epoch = self._next_epoch
        elif at_epoch_boundary:
            resume_epoch = self._next_epoch
        else:
            resume_epoch = self._active_epoch
        return {
            "schema": self._STATE_SCHEMA,
            "version": self._STATE_VERSION,
            "resume_epoch": int(resume_epoch),
            "dataset_items": self.dataset_items,
            "dataset_fingerprint": self.dataset_fingerprint,
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            # Lightning normally writes rank 0's loader state and restores it on
            # every rank. Preserve the writer for diagnostics without replacing
            # this process's rank-local identity.
            "checkpoint_writer_rank": self.rank,
            "ordering_contract": "strided_shuffle_false_drop_last_false_v1",
        }

    def load_resumable_state_dict(self, value: dict[str, Any]) -> None:
        if (
            value.get("schema") != self._STATE_SCHEMA
            or int(value.get("version", -1)) != self._STATE_VERSION
        ):
            raise ValueError("incompatible P11 ordered sampler checkpoint")
        expected = {
            "dataset_items": self.dataset_items,
            "dataset_fingerprint": self.dataset_fingerprint,
            "batch_size": self.batch_size,
            "num_replicas": self.num_replicas,
            "ordering_contract": "strided_shuffle_false_drop_last_false_v1",
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise ValueError(
                    f"P11 ordered sampler {key} changed since checkpoint"
                )
        writer_rank = int(value.get("checkpoint_writer_rank", -1))
        if not 0 <= writer_rank < self.num_replicas:
            raise ValueError("invalid P11 ordered sampler checkpoint writer rank")
        resume_epoch = int(value.get("resume_epoch", -1))
        if resume_epoch < 0:
            raise ValueError("invalid P11 ordered sampler resume epoch")
        self._active_epoch = None
        self._next_epoch = resume_epoch
        self._resume_batch_offset = 0


__all__ = ["DistributedP11OrderedBatchSampler"]
