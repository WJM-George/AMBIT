from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.data.sceneplan_transfusion_generation import (
    TEMPLATE_IDS_BY_SPLIT,
    build_generation_request,
    canonical_generation_target,
    generation_request_missing_facts,
    generation_request_numeric_reversibility_errors,
    render_generation_request,
)


CODEC_PATH = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)


@pytest.fixture(scope="module")
def codec() -> ModelScenePlanCodecV4:
    if not CODEC_PATH.is_dir():
        pytest.skip("frozen codec-v4 artifact is unavailable")
    return ModelScenePlanCodecV4(CODEC_PATH)


def _plan() -> dict:
    return {
        "sample_id": "must_not_leak_0007",
        "duration_sec": 4.2,
        "room": {"type": "moderate"},
        "sources": [
            {
                "source_id": "source_1",
                "kind": "sound",
                "description": "a small metal bell ringing twice",
                "activity": {"onset_sec": 1.0, "offset_sec": 2.7},
                "trajectory": {
                    "type": "static",
                    "position": {
                        "azimuth_deg": -90.4,
                        "elevation_deg": 2.2,
                        "distance_m": 1.25,
                    },
                },
                "gain_db": 0.0,
            },
            {
                "source_id": "source_3",
                "kind": "speech",
                "speaker_description": "a calm adult narrator",
                "transcript": 'She said, "go now."',
                "activity": {"onset_sec": 0.2, "offset_sec": 3.8},
                "trajectory": {
                    "type": "linear",
                    "start": {
                        "azimuth_deg": 45.1,
                        "elevation_deg": -4.8,
                        "distance_m": 2.1,
                    },
                    "end": {
                        "azimuth_deg": 135.2,
                        "elevation_deg": 10.4,
                        "distance_m": 1.1,
                    },
                },
                "gain_db": 0.0,
            },
        ],
    }


def test_target_is_codec_projected_and_observably_canonical(
    codec: ModelScenePlanCodecV4,
) -> None:
    target = canonical_generation_target(_plan(), codec)
    assert [source["source_id"] for source in target["sources"]] == [
        "source_0",
        "source_1",
    ]
    assert target["sources"][0]["kind"] == "speech"
    encoded = codec.encode(target)["input_ids"]
    decoded = codec.decode(encoded, sample_id="roundtrip")
    reencoded = codec.encode(decoded)["input_ids"]
    assert torch.equal(encoded, reencoded)


@pytest.mark.parametrize("split", ("train", "validation", "test"))
def test_request_is_exact_deterministic_and_has_no_identifier_leakage(
    codec: ModelScenePlanCodecV4, split: str
) -> None:
    first = build_generation_request(_plan(), codec, split=split)
    second = build_generation_request(_plan(), codec, split=split)
    assert first == second
    assert "must_not_leak_0007" not in first.text
    assert generation_request_missing_facts(first.text, first.target_sceneplan) == []
    assert generation_request_numeric_reversibility_errors(
        first.target_sceneplan, codec
    ) == []
    assert "distance millimeters" in first.text
    assert first.target_sceneplan["sources"][0]["transcript"] in first.text
    assert first.template_id.startswith(f"gen_ar_exact_v1/{split}/")


def test_template_partitions_are_disjoint() -> None:
    train = set(TEMPLATE_IDS_BY_SPLIT["train"])
    validation = set(TEMPLATE_IDS_BY_SPLIT["validation"])
    test = set(TEMPLATE_IDS_BY_SPLIT["test"])
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert len(train | validation | test) == 32


def test_explicit_cross_split_template_is_rejected(
    codec: ModelScenePlanCodecV4,
) -> None:
    target = canonical_generation_target(_plan(), codec)
    with pytest.raises(ValueError, match="not reserved"):
        render_generation_request(target, split="test", template_number=0)
