from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import (
    GenerationARSQLiteDataset,
    LengthBucketDistributedSampler,
    collate_generation_ar,
    select_tiny_overfit_ordinals,
)


MANIFEST = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/transfusion_shared_v1/"
    "generation_ar/validation.sqlite"
)
CODEC = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)


@pytest.fixture(scope="module")
def codec() -> ModelScenePlanCodecV4:
    if not MANIFEST.is_file() or not CODEC.is_dir():
        pytest.skip("Generation AR validation artifacts are unavailable")
    return ModelScenePlanCodecV4(CODEC)


def test_sqlite_dataset_and_shifted_collate(codec: ModelScenePlanCodecV4) -> None:
    dataset = GenerationARSQLiteDataset(
        MANIFEST, split="validation", row_ordinals=(0, 10000, 20000, 31999)
    )
    rows = [dataset[index] for index in range(len(dataset))]
    batch = collate_generation_ar(rows, pad_id=codec.pad_id)

    assert batch["plan_input_ids"].shape == batch["plan_labels"].shape
    assert batch["plan_attention_mask"].shape == batch["plan_labels"].shape
    for index, row in enumerate(rows):
        length = int(row["target_token_ids"].numel()) - 1
        assert batch["plan_input_ids"][index, :length].equal(
            row["target_token_ids"][:-1]
        )
        assert batch["plan_labels"][index, :length].equal(
            row["target_token_ids"][1:]
        )
        assert int(batch["plan_attention_mask"][index].sum()) == length
        assert row["sample_id"] not in row["raw_user_request"]
    dataset.close()


def test_tiny_selector_spans_source_counts(codec: ModelScenePlanCodecV4) -> None:
    ordinals = select_tiny_overfit_ordinals(MANIFEST, rows=8)
    dataset = GenerationARSQLiteDataset(
        MANIFEST, split="validation", row_ordinals=ordinals
    )
    assert {dataset[index]["source_count"] for index in range(8)} == {1, 2, 3, 4}
    dataset.close()


def test_length_bucket_sampler_is_disjoint_reproducible_and_exact() -> None:
    lengths = np.arange(103, dtype=np.int32)
    samplers = [
        LengthBucketDistributedSampler(
            lengths,
            num_replicas=3,
            rank=rank,
            batch_size=4,
            seed=42,
            bucket_batches=2,
        )
        for rank in range(3)
    ]
    first = [list(sampler) for sampler in samplers]
    assert first[0] == list(samplers[0])
    assert [len(values) for values in first] == [35, 34, 34]
    assert not set(first[0]) & set(first[1])
    assert not set(first[0]) & set(first[2])
    assert not set(first[1]) & set(first[2])
    combined = [value for rank_values in first for value in rank_values]
    assert len(combined) == 103
    assert len(set(combined)) == 103
    assert set(combined) == set(range(103))
    samplers[0].set_epoch(1)
    assert first[0] != list(samplers[0])
