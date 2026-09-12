"""Small checkpointable DataLoader for deterministic map-style training."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from torch.utils.data import DataLoader


class ResumableDataLoader(DataLoader):
    """Resume an ordered map-style loader at the next unconsumed batch.

    Lightning checkpoints objects exposing ``state_dict`` and
    ``load_state_dict``.  PyTorch's regular ``DataLoader`` exposes neither, so
    an iteration-based run otherwise restores the optimizer while silently
    restarting the current epoch's family permutation.  This subclass stores
    the number of batches delivered to the trainer and consumes that prefix
    when the checkpoint is restored.

    Ordered delivery is required: with ``in_order=False``, worker timing can
    change which prefetched batch lies at the checkpoint boundary.
    """

    _STATE_SCHEMA = "stable_audio_tools.resumable_dataloader"
    _STATE_VERSION = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("in_order", True) is not True:
            raise ValueError("ResumableDataLoader requires in_order=True")
        super().__init__(*args, **kwargs)
        self._batches_yielded = 0
        self._resume_batches = 0
        self._resume_state_loaded = False

    def __iter__(self) -> Iterator[Any]:
        resume_batches = self._resume_batches
        self._resume_batches = 0
        self._batches_yielded = 0

        # ScenePlan's bucket sampler is a pure function of seed, epoch, rank,
        # and bucket membership.  Let it drop the restored prefix before the
        # DataLoader starts workers so resume does not read and collate data
        # that will never reach the model.  Generic samplers retain the
        # conservative consume-and-discard path below.
        sampler_fast_forward = getattr(
            self.batch_sampler, "set_resume_batch_offset", None
        )
        if resume_batches and callable(sampler_fast_forward):
            sampler_fast_forward(resume_batches)
            self._batches_yielded = resume_batches
            resume_batches = 0

        iterator = super().__iter__()

        for _ in range(resume_batches):
            try:
                next(iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "saved DataLoader cursor exceeds the current epoch"
                ) from exc
            self._batches_yielded += 1

        for batch in iterator:
            # Increment before yielding so a checkpoint written from
            # on_train_batch_end records the batch that just completed.
            self._batches_yielded += 1
            yield batch

    def state_dict(self) -> dict[str, Any]:
        epoch_batches = len(self)
        raw_batches_yielded = self._batches_yielded
        at_epoch_boundary = raw_batches_yielded == epoch_batches
        batches_yielded = raw_batches_yielded
        # A checkpoint after the final batch resumes in the next epoch, whose
        # cursor starts at zero.
        if batches_yielded == epoch_batches:
            batches_yielded = 0
        state = {
            "schema": self._STATE_SCHEMA,
            "version": self._STATE_VERSION,
            "batches_yielded": int(batches_yielded),
            "epoch_batches": int(epoch_batches),
            "dataset_items": int(len(self.dataset)),
            "at_epoch_boundary": bool(at_epoch_boundary),
        }
        sampler_state = getattr(self.batch_sampler, "resumable_state_dict", None)
        if callable(sampler_state):
            state["batch_sampler_state"] = sampler_state(
                at_epoch_boundary=at_epoch_boundary
            )
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("schema") != self._STATE_SCHEMA:
            raise ValueError("incompatible ResumableDataLoader checkpoint schema")
        version = int(state_dict.get("version", -1))
        if version not in {1, self._STATE_VERSION}:
            raise ValueError("incompatible ResumableDataLoader checkpoint version")

        epoch_batches = len(self)
        dataset_items = len(self.dataset)
        if int(state_dict.get("epoch_batches", -1)) != epoch_batches:
            raise ValueError("DataLoader epoch length changed since the checkpoint")
        if int(state_dict.get("dataset_items", -1)) != dataset_items:
            raise ValueError("DataLoader dataset length changed since the checkpoint")

        batches_yielded = int(state_dict.get("batches_yielded", -1))
        if not 0 <= batches_yielded < max(1, epoch_batches):
            raise ValueError("invalid DataLoader batch cursor in checkpoint")
        self._batches_yielded = batches_yielded
        self._resume_batches = batches_yielded
        self._resume_state_loaded = True
        sampler_loader = getattr(
            self.batch_sampler, "load_resumable_state_dict", None
        )
        sampler_state = state_dict.get("batch_sampler_state")
        if callable(sampler_loader):
            if version < 2 or not isinstance(sampler_state, dict):
                raise ValueError(
                    "checkpoint predates the resumable ScenePlan bucket sampler"
                )
            sampler_loader(sampler_state)
        elif sampler_state is not None:
            raise ValueError(
                "checkpoint contains a batch-sampler state but the loader does not"
            )

    def assert_resume_state_loaded(self) -> None:
        """Fail closed when a mid-epoch checkpoint predates loader state."""

        if not self._resume_state_loaded:
            raise RuntimeError(
                "mid-epoch checkpoint has no resumable DataLoader cursor; "
                "warm-start a new run or use a checkpoint written by the "
                "resumable loader"
            )


__all__ = ["ResumableDataLoader"]
