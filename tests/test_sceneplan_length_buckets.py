import copy

import pytest
import torch

from stable_audio_tools.data.model_sceneplan import (
    MAX_DURATION_SEC,
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    compile_model_44_controls,
    validate_model_sceneplan,
)
from stable_audio_tools.data.resumable_dataloader import ResumableDataLoader
from stable_audio_tools.data.sceneplan_bucket_sampler import (
    DistributedScenePlanBucketBatchSampler,
    sceneplan_bucket_collation,
)


class _BucketDataset(torch.utils.data.Dataset):
    def __init__(self, short: int, long: int):
        self.short = short
        self.long = long

    def __len__(self):
        return self.short + self.long

    def length_bucket_indices(self):
        return {
            432: tuple(range(self.short)),
            648: tuple(range(self.short, self.short + self.long)),
        }

    def __getitem__(self, index):
        bucket = 432 if index < self.short else 648
        audio = torch.full((64, 648), float(index), dtype=torch.float16)
        valid = torch.ones(648, dtype=torch.bool)
        metadata = {
            "sample_id": str(index),
            "audio": audio,
            "padding_mask": [valid],
            "latent_bucket_frames": bucket,
            "latent_crop_length": 648,
            "sceneplan_44": {
                "source_event_frame_ids": torch.zeros(4, 648, dtype=torch.int8),
                "source_trajectory_features": torch.zeros(4, 648, 5),
                "frame_valid_mask": valid,
                "speech_active_frame_mask": torch.zeros(648, dtype=torch.bool),
            },
        }
        return audio, metadata


class _EpochPairedBucketDataset(_BucketDataset):
    semantic_caption_requires_epoch_key = True

    def __getitem__(self, key):
        index, epoch = key
        audio, metadata = super().__getitem__(index)
        metadata["semantic_caption_epoch"] = int(epoch)
        return audio, metadata


def _sampler(dataset, rank=0, *, semantic_epoch_resume_migration="forbid"):
    return DistributedScenePlanBucketBatchSampler(
        dataset,
        short_batch_size=3,
        long_batch_size=2,
        num_replicas=2,
        rank=rank,
        shuffle=True,
        seed=19,
        drop_last=True,
        semantic_epoch_resume_migration=semantic_epoch_resume_migration,
    )


def _loader(dataset, rank=0):
    return ResumableDataLoader(
        dataset,
        batch_sampler=_sampler(dataset, rank),
        collate_fn=sceneplan_bucket_collation,
        num_workers=0,
        in_order=True,
    )


def _batch_ids(batch):
    return tuple(int(row["sample_id"]) for row in batch[1])


def test_rank_slices_are_disjoint_and_shapes_are_homogeneous():
    dataset = _BucketDataset(short=24, long=12)
    left = list(_loader(dataset, rank=0))
    right = list(_loader(dataset, rank=1))
    assert len(left) == len(right) == 7
    for left_batch, right_batch in zip(left, right):
        assert left_batch[0].shape[-1] == right_batch[0].shape[-1]
        bucket = left_batch[0].shape[-1]
        assert bucket in {432, 648}
        assert set(_batch_ids(left_batch)).isdisjoint(_batch_ids(right_batch))
        expected_size = 3 if bucket == 432 else 2
        assert left_batch[0].shape == (expected_size, 64, bucket)
        for row in left_batch[1]:
            assert row["sceneplan_44"]["source_event_frame_ids"].shape == (
                4,
                bucket,
            )
            assert row["sceneplan_44"]["source_trajectory_features"].shape == (
                4,
                bucket,
                5,
            )


def test_bucket_sampler_resume_replays_current_epoch_then_advances():
    dataset = _BucketDataset(short=24, long=12)
    original = _loader(dataset)
    iterator = iter(original)
    first = [_batch_ids(next(iterator)), _batch_ids(next(iterator))]
    state = original.state_dict()
    remaining = [_batch_ids(batch) for batch in iterator]

    restored = _loader(dataset)
    restored.load_state_dict(state)
    assert [_batch_ids(batch) for batch in restored] == remaining

    completed_state = restored.state_dict()
    assert completed_state["at_epoch_boundary"] is True
    next_epoch = _loader(dataset)
    next_epoch.load_state_dict(completed_state)
    assert [_batch_ids(batch) for batch in next_epoch] != first + remaining


def test_rank0_checkpoint_broadcast_restores_each_rank_local_slice():
    dataset = _BucketDataset(short=24, long=12)
    writer = _loader(dataset, rank=0)
    writer_iterator = iter(writer)
    next(writer_iterator)
    next(writer_iterator)
    state = writer.state_dict()
    assert state["batch_sampler_state"]["rank"] == 0

    rank1_full_epoch = [_batch_ids(batch) for batch in _loader(dataset, rank=1)]
    restored_rank1 = _loader(dataset, rank=1)
    restored_rank1.load_state_dict(state)

    assert [_batch_ids(batch) for batch in restored_rank1] == rank1_full_epoch[2:]


def test_bucket_sampler_attaches_resume_stable_epoch_to_paired_prompts():
    dataset = _EpochPairedBucketDataset(short=24, long=12)
    loader = _loader(dataset)
    iterator = iter(loader)
    first_batch = next(iterator)
    assert {row["semantic_caption_epoch"] for row in first_batch[1]} == {0}
    state = loader.state_dict()

    restored = _loader(dataset)
    restored.load_state_dict(state)
    remaining = list(restored)
    assert remaining
    assert all(
        {row["semantic_caption_epoch"] for row in batch[1]} == {0}
        for batch in remaining
    )

    next_epoch = list(restored)
    assert next_epoch
    assert all(
        {row["semantic_caption_epoch"] for row in batch[1]} == {1}
        for batch in next_epoch
    )


def test_bucket_sampler_rejects_uncontracted_semantic_mode_change():
    paired = _loader(_EpochPairedBucketDataset(short=24, long=12))
    iterator = iter(paired)
    next(iterator)
    state = paired.state_dict()

    fixed = _loader(_BucketDataset(short=24, long=12))
    with pytest.raises(ValueError, match="semantic-epoch mode changed"):
        fixed.load_state_dict(state)


def test_bucket_sampler_migrates_paired_cursor_to_fixed_v2_explicitly():
    paired = _loader(_EpochPairedBucketDataset(short=24, long=12))
    paired_iterator = iter(paired)
    next(paired_iterator)
    next(paired_iterator)
    state = paired.state_dict()

    dataset = _BucketDataset(short=24, long=12)
    fixed_full_epoch = [_batch_ids(batch) for batch in _loader(dataset)]
    fixed = ResumableDataLoader(
        dataset,
        batch_sampler=_sampler(
            dataset,
            semantic_epoch_resume_migration="paired_to_fixed",
        ),
        collate_fn=sceneplan_bucket_collation,
        num_workers=0,
        in_order=True,
    )
    fixed.load_state_dict(state)

    assert [_batch_ids(batch) for batch in fixed] == fixed_full_epoch[2:]


def test_bucket_sampler_never_migrates_fixed_cursor_to_paired():
    fixed = _loader(_BucketDataset(short=24, long=12))
    iterator = iter(fixed)
    next(iterator)
    state = fixed.state_dict()

    paired_dataset = _EpochPairedBucketDataset(short=24, long=12)
    paired = ResumableDataLoader(
        paired_dataset,
        batch_sampler=_sampler(
            paired_dataset,
            semantic_epoch_resume_migration="paired_to_fixed",
        ),
        collate_fn=sceneplan_bucket_collation,
        num_workers=0,
        in_order=True,
    )
    with pytest.raises(ValueError, match="semantic-epoch mode changed"):
        paired.load_state_dict(state)


def _plan(duration):
    return {
        "sample_id": "long_15s",
        "duration_sec": duration,
        "room": {"type": "moderate"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "speech",
                "speaker_description": "An adult narrator with a clear measured voice.",
                "transcript": "This complete utterance tests the longer model envelope.",
                "activity": {"onset_sec": 0.0, "offset_sec": duration},
                "trajectory": {
                    "type": "linear",
                    "start": {
                        "azimuth_deg": -90.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.0,
                    },
                    "end": {
                        "azimuth_deg": 90.0,
                        "elevation_deg": 0.0,
                        "distance_m": 2.0,
                    },
                },
                "gain_db": 0.0,
            }
        ],
    }


def test_sceneplan_accepts_exact_648_frame_envelope_and_rejects_more():
    plan = _plan(MAX_DURATION_SEC)
    validate_model_sceneplan(plan)
    controls = compile_model_44_controls(
        plan,
        model_num_samples=648 * 1024,
        latent_frames_valid=648,
    )
    assert controls["source_event_frame_ids"].shape == (4, 648)
    assert controls["source_trajectory_features"].shape == (4, 648, 5)

    too_long = copy.deepcopy(plan)
    too_long["duration_sec"] = MAX_DURATION_SEC + 1.0e-4
    too_long["sources"][0]["activity"]["offset_sec"] = too_long["duration_sec"]
    with pytest.raises(ValueError, match="outside the model envelope"):
        validate_model_sceneplan(too_long)


def test_six_decimal_duration_keeps_exact_sample_envelope_strict() -> None:
    serialized_boundary = round(MAX_MODEL_SAMPLES / MODEL_SAMPLE_RATE, 6)
    boundary = _plan(serialized_boundary)
    validate_model_sceneplan(boundary)
    controls = compile_model_44_controls(
        boundary,
        model_num_samples=MAX_MODEL_SAMPLES,
        latent_frames_valid=648,
    )
    assert controls["source_event_frame_ids"].shape == (4, 648)

    one_sample_over = round((MAX_MODEL_SAMPLES + 1) / MODEL_SAMPLE_RATE, 6)
    invalid = _plan(one_sample_over)
    with pytest.raises(ValueError, match="outside the model envelope"):
        validate_model_sceneplan(invalid)
