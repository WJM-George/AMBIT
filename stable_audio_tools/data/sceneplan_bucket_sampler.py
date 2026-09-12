"""Rank-aware, frame-balanced batches for 432/648-frame ScenePlan training."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Sampler


def _dataset_bucket_indices(dataset: Any) -> dict[int, list[int]]:
    if hasattr(dataset, "length_bucket_indices"):
        return {
            int(bucket): list(indices)
            for bucket, indices in dataset.length_bucket_indices().items()
        }
    if isinstance(dataset, ConcatDataset):
        output: dict[int, list[int]] = {432: [], 648: []}
        offset = 0
        for child in dataset.datasets:
            child_buckets = _dataset_bucket_indices(child)
            for bucket, values in child_buckets.items():
                output.setdefault(bucket, []).extend(offset + value for value in values)
            offset += len(child)
        return {bucket: values for bucket, values in output.items() if values}
    raise TypeError(
        "length-bucket batching requires ScenePlanV2Dataset or ConcatDataset children"
    )


class DistributedScenePlanBucketBatchSampler(Sampler[list[Any]]):
    """Build homogeneous global batches and give each DDP rank one slice.

    Short and long local batches can use different sample counts while keeping
    nearly equal latent-frame budgets (for example 72x432 == 48x648).  Since
    the diffusion loss is averaged over valid audio frames, this preserves a
    frame-balanced objective while avoiding 648-frame padding for the large
    short-audio majority.
    """

    _STATE_SCHEMA = "stable_audio_tools.sceneplan_bucket_batch_sampler"
    _STATE_VERSION = 1

    def __init__(
        self,
        dataset: Any,
        *,
        short_batch_size: int,
        long_batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        semantic_epoch_resume_migration: str = "forbid",
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.semantic_epoch_resume_migration = str(
            semantic_epoch_resume_migration
        )
        if self.semantic_epoch_resume_migration not in {
            "forbid",
            "paired_to_fixed",
        }:
            raise ValueError(
                "semantic_epoch_resume_migration must be forbid or "
                "paired_to_fixed"
            )
        self.include_semantic_epoch = bool(
            getattr(dataset, "semantic_caption_requires_epoch_key", False)
        )
        if isinstance(dataset, ConcatDataset) and any(
            bool(getattr(child, "semantic_caption_requires_epoch_key", False))
            for child in dataset.datasets
        ):
            raise ValueError(
                "epoch-paired semantic captions currently require one frozen index"
            )
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"invalid ScenePlan bucket rank/world: {self.rank}/{self.num_replicas}"
            )
        self.local_batch_sizes = {
            432: int(short_batch_size),
            648: int(long_batch_size),
        }
        if any(value <= 0 for value in self.local_batch_sizes.values()):
            raise ValueError("ScenePlan bucket batch sizes must be positive")
        if not self.drop_last:
            raise ValueError(
                "distributed variable-size ScenePlan batches require drop_last=true"
            )
        raw = _dataset_bucket_indices(dataset)
        unexpected = set(raw) - {432, 648}
        if unexpected:
            raise RuntimeError(f"unsupported ScenePlan length buckets: {unexpected}")
        self.bucket_indices = {
            bucket: np.asarray(values, dtype=np.int64)
            for bucket, values in raw.items()
            if values
        }
        if not self.bucket_indices:
            raise RuntimeError("ScenePlan length-bucket sampler found no examples")
        covered = sum(len(values) for values in self.bucket_indices.values())
        if covered != len(dataset):
            raise RuntimeError(
                f"ScenePlan length buckets cover {covered} of {len(dataset)} rows"
            )
        self._active_epoch: int | None = None
        self._next_epoch = 0
        # A restored ResumableDataLoader can advance this deterministic
        # sampler at the descriptor level.  This avoids opening and decoding
        # every already-consumed latent merely to discard it after a
        # mid-epoch checkpoint restore.
        self._resume_batch_offset = 0
        self._length = sum(
            len(values)
            // (self.local_batch_sizes[bucket] * self.num_replicas)
            for bucket, values in self.bucket_indices.items()
        )
        if self._length <= 0:
            raise RuntimeError("ScenePlan length buckets do not form one global batch")

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[list[Any]]:
        epoch = self._next_epoch
        self._active_epoch = epoch
        self._next_epoch = epoch + 1
        generator = np.random.default_rng(self.seed + epoch)
        orders: dict[int, np.ndarray] = {}
        descriptors: list[tuple[int, int]] = []
        for bucket, original in sorted(self.bucket_indices.items()):
            order = generator.permutation(original) if self.shuffle else original
            orders[bucket] = order
            global_batch = self.local_batch_sizes[bucket] * self.num_replicas
            usable = len(order) - (len(order) % global_batch)
            descriptors.extend((bucket, start) for start in range(0, usable, global_batch))
        if self.shuffle:
            generator.shuffle(descriptors)
        resume_batch_offset = self._resume_batch_offset
        self._resume_batch_offset = 0
        if not 0 <= resume_batch_offset < max(1, len(descriptors)):
            raise RuntimeError(
                "ScenePlan bucket sampler resume offset exceeds the epoch"
            )
        if resume_batch_offset:
            descriptors = descriptors[resume_batch_offset:]
        local_offsets = {
            bucket: self.rank * size
            for bucket, size in self.local_batch_sizes.items()
        }
        for bucket, global_start in descriptors:
            local_size = self.local_batch_sizes[bucket]
            start = global_start + local_offsets[bucket]
            batch = orders[bucket][start : start + local_size]
            if len(batch) != local_size:
                raise RuntimeError("ScenePlan rank slice produced a partial batch")
            indices = batch.tolist()
            if self.include_semantic_epoch:
                yield [(int(index), int(epoch)) for index in indices]
            else:
                yield indices

    def set_resume_batch_offset(self, batches_yielded: int) -> None:
        """Skip a restored deterministic prefix without loading its samples."""

        offset = int(batches_yielded)
        if not 0 <= offset < max(1, self._length):
            raise ValueError("invalid ScenePlan bucket sampler resume offset")
        if self._active_epoch is not None:
            raise RuntimeError(
                "ScenePlan bucket sampler resume offset must be set before iteration"
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
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "rank": self.rank,
            "local_batch_sizes": dict(self.local_batch_sizes),
            "include_semantic_epoch": self.include_semantic_epoch,
            "bucket_counts": {
                str(bucket): len(values)
                for bucket, values in self.bucket_indices.items()
            },
        }

    def load_resumable_state_dict(self, value: dict[str, Any]) -> None:
        if (
            value.get("schema") != self._STATE_SCHEMA
            or int(value.get("version", -1)) != self._STATE_VERSION
        ):
            raise ValueError("incompatible ScenePlan bucket sampler checkpoint")
        expected = {
            "seed": self.seed,
            "num_replicas": self.num_replicas,
            "local_batch_sizes": dict(self.local_batch_sizes),
            "bucket_counts": {
                str(bucket): len(rows)
                for bucket, rows in self.bucket_indices.items()
            },
        }
        for key, expected_value in expected.items():
            if value.get(key) != expected_value:
                raise ValueError(
                    f"ScenePlan bucket sampler {key} changed since checkpoint"
                )
        # Revision-6 source checkpoints predate paired prompt augmentation and
        # legitimately lack this field. New continuation checkpoints persist
        # it so later restarts fail closed if the prompt curriculum changes.
        if "include_semantic_epoch" in value:
            checkpoint_semantic_epoch = bool(value["include_semantic_epoch"])
            semantic_epoch_changed = (
                checkpoint_semantic_epoch != self.include_semantic_epoch
            )
            allowed_paired_to_fixed = (
                self.semantic_epoch_resume_migration == "paired_to_fixed"
                and checkpoint_semantic_epoch
                and not self.include_semantic_epoch
            )
            if semantic_epoch_changed and not allowed_paired_to_fixed:
                raise ValueError(
                    "ScenePlan bucket sampler semantic-epoch mode changed since "
                    "checkpoint"
                )
        # Lightning writes one global checkpoint (normally from rank 0) and
        # broadcasts that loader state to every rank on restore.  The saved
        # rank therefore identifies the checkpoint writer; it must not replace
        # or equal the rank-local sampler identity created for this process.
        checkpoint_rank = int(value.get("rank", -1))
        if not 0 <= checkpoint_rank < self.num_replicas:
            raise ValueError("invalid ScenePlan bucket sampler checkpoint rank")
        epoch = int(value.get("resume_epoch", -1))
        if epoch < 0:
            raise ValueError("invalid ScenePlan bucket sampler resume epoch")
        self._active_epoch = None
        self._next_epoch = epoch
        self._resume_batch_offset = 0


def sceneplan_bucket_collation(
    samples: Sequence[tuple[torch.Tensor, dict[str, Any]]],
) -> list[Any]:
    """Trim immutable 648-padding to the homogeneous batch bucket."""

    if not samples:
        raise ValueError("cannot collate an empty ScenePlan batch")
    buckets = {int(metadata["latent_bucket_frames"]) for _, metadata in samples}
    if len(buckets) != 1:
        raise RuntimeError(f"mixed ScenePlan length buckets in one batch: {buckets}")
    bucket = buckets.pop()
    if bucket not in {432, 648}:
        raise RuntimeError(f"unsupported ScenePlan batch bucket: {bucket}")
    trimmed: list[tuple[torch.Tensor, dict[str, Any]]] = []
    for audio, original_metadata in samples:
        if audio.ndim != 2 or audio.shape[-1] < bucket:
            raise RuntimeError("ScenePlan latent is shorter than its batch bucket")
        metadata = dict(original_metadata)
        metadata["audio"] = original_metadata["audio"][..., :bucket]
        metadata["padding_mask"] = [
            original_metadata["padding_mask"][0][..., :bucket]
        ]
        controls = dict(original_metadata["sceneplan_44"])
        controls["source_event_frame_ids"] = controls[
            "source_event_frame_ids"
        ][..., :bucket]
        controls["source_trajectory_features"] = controls[
            "source_trajectory_features"
        ][..., :bucket, :]
        controls["frame_valid_mask"] = controls["frame_valid_mask"][..., :bucket]
        controls["speech_active_frame_mask"] = controls[
            "speech_active_frame_mask"
        ][..., :bucket]
        metadata["sceneplan_44"] = controls
        if "source_foa_latent" in original_metadata:
            source_latent = original_metadata["source_foa_latent"]
            if (
                not isinstance(source_latent, torch.Tensor)
                or source_latent.ndim != 2
                or source_latent.shape[0] != 64
                or source_latent.shape[-1] < bucket
            ):
                raise RuntimeError(
                    "Transfusion Editing source latent is not aligned [64,T]"
                )
            metadata["source_foa_latent"] = source_latent[..., :bucket]
        metadata["latent_crop_length"] = bucket
        metadata["batch_bucket_frames"] = bucket
        trimmed.append((audio[..., :bucket], metadata))
    # Import locally to avoid a dataset.py -> sampler -> dataset.py cycle.
    from .dataset import collation_fn

    return collation_fn(trimmed)


__all__ = [
    "DistributedScenePlanBucketBatchSampler",
    "sceneplan_bucket_collation",
]
