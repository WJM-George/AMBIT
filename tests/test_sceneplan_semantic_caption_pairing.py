import inspect

import pytest

from stable_audio_tools.data.model_sceneplan import (
    compile_model_semantic_caption,
    compile_model_semantic_caption_v2,
)
from stable_audio_tools.data.sceneplan_v2_dataset import ScenePlanV2Dataset


def _speech_plan():
    return {
        "sample_id": "paired_speech",
        "duration_sec": 3.0,
        "room": {"type": "dry"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "speech",
                "speaker_description": "an adult narrator with a clear measured voice",
                "transcript": "This exact sentence must remain the only spoken text.",
                "activity": {"onset_sec": 0.0, "offset_sec": 3.0},
                "trajectory": {
                    "type": "static",
                    "position": {
                        "azimuth_deg": -30.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.2,
                    },
                },
                "gain_db": 0.0,
            }
        ],
    }


def _paired_selector():
    dataset = object.__new__(ScenePlanV2Dataset)
    dataset.semantic_caption_mode = "paired_v1_v2"
    dataset.semantic_caption_v2_probability = 0.5
    dataset.semantic_caption_seed = 20260830
    return dataset


def test_sceneplan_dataset_defaults_to_canonical_v2_only():
    parameters = inspect.signature(ScenePlanV2Dataset.__init__).parameters
    assert parameters["semantic_caption_mode"].default == "v2"
    assert parameters["semantic_caption_v2_probability"].default == 1.0


def test_same_sceneplan_flips_semantic_surface_form_in_adjacent_epochs():
    dataset = _paired_selector()
    plan = _speech_plan()
    first = dataset._semantic_caption(
        plan, sample_id=plan["sample_id"], semantic_epoch=32
    )
    second = dataset._semantic_caption(
        plan, sample_id=plan["sample_id"], semantic_epoch=33
    )
    assert {first["compiler_version"], second["compiler_version"]} == {1, 2}
    expected = {
        compile_model_semantic_caption(plan)["text"],
        compile_model_semantic_caption_v2(plan)["text"],
    }
    assert {first["text"], second["text"]} == expected


def test_paired_semantic_surface_form_is_resume_stable():
    dataset = _paired_selector()
    plan = _speech_plan()
    left = dataset._semantic_caption(
        plan, sample_id=plan["sample_id"], semantic_epoch=41
    )
    right = dataset._semantic_caption(
        plan, sample_id=plan["sample_id"], semantic_epoch=41
    )
    assert left == right


def test_paired_semantic_surface_form_requires_sampler_epoch():
    dataset = _paired_selector()
    plan = _speech_plan()
    with pytest.raises(RuntimeError, match="requires the sampler"):
        dataset._semantic_caption(plan, sample_id=plan["sample_id"])
