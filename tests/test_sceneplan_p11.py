from __future__ import annotations

import os
import copy

import pytest
import torch

from stable_audio_tools.data.model_sceneplan_codec import load_model_sceneplan_codec
from stable_audio_tools.data.sceneplan_edit_patch import ScenePlanEditPatchCodec
from stable_audio_tools.data.sceneplan_p11_dataset import P11_MANIFEST_VERSION
from stable_audio_tools.data.sceneplan_p11_metrics import (
    score_p11_prediction,
    score_sceneplan_fields,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (
    canonicalize_sceneplan_source_ids,
    compile_p10_aligned_target_conditions,
    validate_runtime_audio_spans,
)


CODEC_PATH = (
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)


def _plan() -> dict:
    return {
        "sample_id": "p11_test",
        "duration_sec": 2.0,
        "room": {"type": "dry"},
        "sources": [
            {
                "source_id": "source_1",
                "kind": "sound",
                "description": "a short metal bell",
                "activity": {"onset_sec": 1.0, "offset_sec": 1.5},
                "trajectory": {
                    "type": "static",
                    "position": {
                        "azimuth_deg": 45.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.0,
                    },
                },
                "gain_db": 0.0,
            },
            {
                "source_id": "source_3",
                "kind": "music",
                "description": "a sustained cello note",
                "activity": {"onset_sec": 0.0, "offset_sec": 2.0},
                "trajectory": {
                    "type": "static",
                    "position": {
                        "azimuth_deg": -45.0,
                        "elevation_deg": 0.0,
                        "distance_m": 2.0,
                    },
                },
                "gain_db": 0.0,
            },
        ],
    }


@pytest.fixture(scope="module")
def patch_codec() -> ScenePlanEditPatchCodec:
    return ScenePlanEditPatchCodec(load_model_sceneplan_codec(CODEC_PATH))


def test_canonical_source_ids_ignore_frozen_slot_labels() -> None:
    canonical = canonicalize_sceneplan_source_ids(_plan())
    assert [source["source_id"] for source in canonical["sources"]] == [
        "source_0",
        "source_1",
    ]
    assert canonical["sources"][0]["description"] == "a sustained cello note"


def test_permutation_invariant_metric_does_not_penalize_label_swap() -> None:
    target = _plan()
    prediction = copy.deepcopy(target)
    left, right = prediction["sources"]
    left["source_id"], right["source_id"] = right["source_id"], left["source_id"]
    prediction["sources"].sort(key=lambda source: source["source_id"])
    assert score_sceneplan_fields(
        target, prediction, source_matching="permutation_invariant"
    )["scene_exact"] == 1.0


@pytest.mark.parametrize(
    ("spec", "expected_tokens"),
    [
        ({"operation": "no_op"}, 3),
        ({"operation": "room_change", "new_room": "outdoor"}, 4),
        (
            {
                "operation": "rotate_source",
                "source_id": "source_0",
                "delta_azimuth_deg": 45,
            },
            5,
        ),
        (
            {
                "operation": "distance_source",
                "source_id": "source_0",
                "distance_factor": 0.75,
            },
            5,
        ),
        (
            {
                "operation": "retime_source",
                "source_id": "source_0",
                "new_interval_frames": [1, 40],
            },
            6,
        ),
        ({"operation": "remove_source", "source_id": "source_0"}, 4),
    ],
)
def test_patch_roundtrip_and_executor(
    patch_codec: ScenePlanEditPatchCodec,
    spec: dict,
    expected_tokens: int,
) -> None:
    current = canonicalize_sceneplan_source_ids(
        patch_codec.plan_codec.project_plan(_plan())
    )
    encoded = patch_codec.encode(spec)["input_ids"]
    assert len(encoded) == expected_tokens
    decoded = patch_codec.decode(encoded)
    assert torch.equal(patch_codec.canonicalize(encoded)["input_ids"], encoded)
    revised = patch_codec.apply(current, decoded)
    if spec["operation"] == "no_op":
        assert patch_codec.plan_codec.encode(revised)["input_ids"].equal(
            patch_codec.plan_codec.encode(current)["input_ids"]
        )


def test_patch_grammar_rejects_removing_the_only_source(
    patch_codec: ScenePlanEditPatchCodec,
) -> None:
    current = canonicalize_sceneplan_source_ids(_plan())
    current["sources"] = current["sources"][:1]
    prefix = torch.tensor(
        [
            patch_codec.bos_id,
            patch_codec.token_to_id["<op_remove_source>"],
        ]
    )
    with pytest.raises(Exception):
        patch_codec.allowed_next_ids(prefix, input_sceneplan=current)


def test_edit_score_is_measured_after_patch_application(
    patch_codec: ScenePlanEditPatchCodec,
) -> None:
    current = canonicalize_sceneplan_source_ids(
        patch_codec.plan_codec.project_plan(_plan())
    )
    target = patch_codec.apply(
        current,
        {
            "operation": "rotate_source",
            "source_id": "source_0",
            "delta_azimuth_deg": 45,
        },
    )
    correct = score_p11_prediction(
        task="editing",
        target_plan=target,
        prediction=target,
        input_plan=current,
        editing_score_version="patch_applied_v3",
    )
    copied = score_p11_prediction(
        task="editing",
        target_plan=target,
        prediction=current,
        input_plan=current,
        editing_score_version="patch_applied_v3",
    )
    assert correct["task_score"] == 1.0
    assert copied["task_score"] == 0.0


def test_audio_aware_editing_requires_an_audio_span() -> None:
    validate_runtime_audio_spans("editing", input_foa=torch.zeros(64, 10))
    with pytest.raises(ValueError, match="editing requires input FOA latent=True"):
        validate_runtime_audio_spans("editing", input_foa=None)


def test_p11_handoff_uses_canonical_unquoted_p10_speech_caption() -> None:
    plan = _plan()
    plan["sources"] = [
        {
            "source_id": "source_0",
            "kind": "speech",
            "speaker_description": "an adult narrator with a clear measured voice",
            "transcript": "Only these exact words should be spoken.",
            "activity": {"onset_sec": 0.0, "offset_sec": 2.0},
            "trajectory": {
                "type": "static",
                "position": {
                    "azimuth_deg": 0.0,
                    "elevation_deg": 0.0,
                    "distance_m": 1.0,
                },
            },
            "gain_db": 0.0,
        }
    ]
    semantic = compile_p10_aligned_target_conditions(plan)["semantic_caption"]
    assert semantic["compiler_version"] == 2
    assert "who says: Only these exact words should be spoken." in semantic["text"]
    assert '\"Only these exact words should be spoken.\"' not in semantic["text"]


def test_canonical_p11_manifest_contract_is_audio_aware_v7() -> None:
    assert P11_MANIFEST_VERSION == 7
