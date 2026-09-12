from __future__ import annotations

import copy

import pytest
import torch

from stable_audio_tools.data.model_sceneplan_codec import load_model_sceneplan_codec
from stable_audio_tools.data.sceneplan_edit_patch import (
    ACTIVE_OPERATION_TOKENS,
    RETIME_FRAME_LEVELS,
    RETIME_MODES,
    RETIME_POLICY,
    ScenePlanEditPatchCodec,
    make_deterministic_edit,
)
from stable_audio_tools.data.scene_sketch_v1 import (
    AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT,
    EXECUTION_FEATURE_NAMES,
    AudioAwareDeltaSceneSketchCodec,
    audio_aware_delta_control_mask,
    compile_audio_aware_delta_sketch,
    compile_execution_state,
    execution_delta_core,
    project_audio_aware_delta_to_atomic_patch,
)
from stable_audio_tools.models.sceneplan_p11_v4 import (
    _audio_aware_delta_control_objective,
)
from stable_audio_tools.data.sceneplan_p11_dataset import P11_DATA_CONTRACT
from stable_audio_tools.data.sceneplan_p11_metrics import score_p11_prediction
from stable_audio_tools.data.sceneplan_p11_single_turn import (
    P11_EDITING_INPUT_CONTRACT,
    P11_EDITING_OUTPUT_CONTRACT,
    P11_MODEL_CONTRACT,
    single_turn_layout,
    validate_runtime_audio_spans,
    validate_single_turn_record,
)


CODEC_PATH = (
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)


def _plan(sample_id: str = "audio_aware_test") -> dict:
    return {
        "sample_id": sample_id,
        "duration_sec": 2.0,
        "room": {"type": "dry"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "sound",
                "description": "a short metal bell",
                "activity": {"onset_sec": 0.0, "offset_sec": 2.0},
                "trajectory": {
                    "type": "static",
                    "position": {
                        "azimuth_deg": -45.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.0,
                    },
                },
                "gain_db": 0.0,
            }
        ],
    }


def _two_source_plan() -> dict:
    plan = _plan("audio_aware_two_source")
    plan["sources"].append(
        {
            "source_id": "source_1",
            "kind": "speech",
            "speaker_description": "a calm narrator",
            "transcript": "The station is quiet tonight.",
            "activity": {"onset_sec": 0.25, "offset_sec": 1.75},
            "trajectory": {
                "type": "linear",
                "start": {
                    "azimuth_deg": -90.0,
                    "elevation_deg": 0.0,
                    "distance_m": 2.0,
                },
                "end": {
                    "azimuth_deg": 45.0,
                    "elevation_deg": 15.0,
                    "distance_m": 1.0,
                },
            },
            "gain_db": 0.0,
        }
    )
    return plan


def _editing_record(mode: str) -> dict:
    observed = _plan()
    revised = copy.deepcopy(observed)
    revised["room"] = {"type": "moderate"}
    record = {
        "task": "editing",
        "sample_id": "audio_aware_test",
        "edit_instruction": "Change the room to moderate.",
        "input_foa_ref": "/frozen/input.flac",
        "input_foa_latent_ref": "/frozen/input.npy",
        "observed_sceneplan_target": observed,
        "edit_patch_target": {
            "operation": "room_change",
            "new_room": "moderate",
        },
        "target_sceneplan": revised,
        "editing_evidence_mode": mode,
    }
    if mode != "no_plan":
        record["input_sceneplan"] = copy.deepcopy(observed)
        if mode == "corrupt_plan":
            record["input_sceneplan"]["room"] = {"type": "outdoor"}
    return record


def test_audio_aware_contract_identity_and_layout() -> None:
    assert P11_MODEL_CONTRACT == "sceneplan_p11_audio_aware_v1"
    assert P11_EDITING_INPUT_CONTRACT == (
        "input_foa_required_old_sceneplan_optional_v1"
    )
    assert P11_EDITING_OUTPUT_CONTRACT == (
        "observed_plan_atomic_patch_revised_plan_v1"
    )
    assert P11_DATA_CONTRACT == "audio_aware_sketch_first_transfusion_cot_v2"
    assert single_turn_layout("generation").num_input_audio_spans == 0
    assert single_turn_layout("understanding").num_input_audio_spans == 1
    assert single_turn_layout("editing").num_input_audio_spans == 1


@pytest.mark.parametrize("mode", ["no_plan", "correct_plan", "corrupt_plan"])
def test_audio_aware_editing_accepts_all_old_plan_evidence_modes(mode: str) -> None:
    validated = validate_single_turn_record(_editing_record(mode))
    assert validated["editing_input_contract"] == P11_EDITING_INPUT_CONTRACT
    assert validated["editing_output_contract"] == P11_EDITING_OUTPUT_CONTRACT


def test_audio_aware_editing_fails_without_input_audio() -> None:
    record = _editing_record("no_plan")
    record.pop("input_foa_ref")
    record.pop("input_foa_latent_ref")
    with pytest.raises(ValueError, match="editing requires input FOA span=True"):
        validate_single_turn_record(record)


def test_audio_aware_editing_rejects_evidence_mode_plan_mismatch() -> None:
    record = _editing_record("correct_plan")
    record.pop("input_sceneplan")
    with pytest.raises(ValueError, match="evidence mode disagrees"):
        validate_single_turn_record(record)


@pytest.mark.parametrize(
    "forbidden", ["target_foa", "target_foa_ref", "target_foa_latent_ref"]
)
def test_audio_aware_records_still_forbid_target_audio(forbidden: str) -> None:
    record = _editing_record("no_plan")
    record[forbidden] = "forbidden"
    with pytest.raises(ValueError, match="must not carry target FOA"):
        validate_single_turn_record(record)


def test_audio_aware_edit_metrics_use_observed_plan_not_fallible_prior() -> None:
    observed = _plan()
    target = copy.deepcopy(observed)
    target["room"] = {"type": "moderate"}
    corrupt_prior = copy.deepcopy(observed)
    corrupt_prior["room"] = {"type": "outdoor"}
    metrics = score_p11_prediction(
        task="editing",
        target_plan=target,
        prediction=target,
        input_plan=corrupt_prior,
        observed_plan=observed,
        source_matching="persistent_id",
        editing_score_version="observed_patch_revised_v1",
    )
    assert metrics["edit_success"] == 1.0
    assert metrics["task_score"] == 1.0


def test_audio_aware_edit_metrics_fail_without_observed_plan() -> None:
    observed = _plan()
    with pytest.raises(ValueError, match="require the observed ScenePlan"):
        score_p11_prediction(
            task="editing",
            target_plan=observed,
            prediction=observed,
            input_plan=observed,
            editing_score_version="observed_patch_revised_v1",
        )


def test_runtime_audio_truth_table_is_generation_vs_understanding_editing() -> None:
    latent = torch.zeros(64, 16)
    validate_runtime_audio_spans("generation", input_foa=None)
    validate_runtime_audio_spans("understanding", input_foa=latent)
    validate_runtime_audio_spans("editing", input_foa=latent)
    with pytest.raises(ValueError):
        validate_runtime_audio_spans("generation", input_foa=latent)
    with pytest.raises(ValueError):
        validate_runtime_audio_spans("understanding", input_foa=None)
    with pytest.raises(ValueError):
        validate_runtime_audio_spans("editing", input_foa=None)


def test_active_patch_v2_add_remove_preserves_existing_source_ids() -> None:
    codec = load_model_sceneplan_codec(CODEC_PATH)
    patch_codec = ScenePlanEditPatchCodec(codec)
    observed = codec.project_plan(_plan())
    added_source = {
        "source_id": "source_1",
        "kind": "music",
        "description": "a gentle cello phrase",
        "activity": {"onset_sec": 0.0, "offset_sec": observed["duration_sec"]},
        "trajectory": {
            "type": "linear",
            "start": {
                "azimuth_deg": -90.0,
                "elevation_deg": 0.0,
                "distance_m": 2.0,
            },
            "end": {
                "azimuth_deg": 90.0,
                "elevation_deg": 15.0,
                "distance_m": 1.0,
            },
        },
        "gain_db": 0.0,
    }
    add = {"operation": "add_source", "source": added_source}
    add_tokens = patch_codec.encode(add)["input_ids"]
    assert torch.equal(patch_codec.canonicalize(add_tokens)["input_ids"], add_tokens)
    revised = patch_codec.apply(observed, add_tokens)
    assert [source["source_id"] for source in revised["sources"]] == [
        "source_0",
        "source_1",
    ]
    removed = patch_codec.apply(
        revised, {"operation": "remove_source", "source_id": "source_0"}
    )
    assert [source["source_id"] for source in removed["sources"]] == ["source_1"]


def test_active_patch_v2_text_and_motion_roundtrip() -> None:
    codec = load_model_sceneplan_codec(CODEC_PATH)
    patch_codec = ScenePlanEditPatchCodec(codec)
    observed = _plan()
    observed["sources"][0] = {
        "source_id": "source_0",
        "kind": "speech",
        "speaker_description": "a calm narrator",
        "transcript": "The first exact sentence.",
        "activity": {"onset_sec": 0.0, "offset_sec": 2.0},
        "trajectory": {
            "type": "static",
            "position": {
                "azimuth_deg": -45.0,
                "elevation_deg": 0.0,
                "distance_m": 1.0,
            },
        },
        "gain_db": 0.0,
    }
    observed = codec.project_plan(observed)
    transcript_patch = {
        "operation": "change_transcript",
        "source_id": "source_0",
        "new_transcript": "The revised exact sentence.",
    }
    transcript_tokens = patch_codec.encode(transcript_patch)["input_ids"]
    changed = patch_codec.apply(observed, transcript_tokens)
    assert changed["sources"][0]["transcript"] == "The revised exact sentence."
    move_patch = {
        "operation": "move_source",
        "source_id": "source_0",
        "trajectory": {
            "type": "static",
            "position": {
                "azimuth_deg": 90.0,
                "elevation_deg": 20.0,
                "distance_m": 3.0,
            },
        },
    }
    move_tokens = patch_codec.encode(move_patch)["input_ids"]
    moved = patch_codec.apply(observed, move_tokens)
    assert moved["sources"][0]["trajectory"]["position"]["azimuth_deg"] == 90.0


def test_active_patch_decode_excludes_retired_relative_operations() -> None:
    codec = load_model_sceneplan_codec(CODEC_PATH)
    patch_codec = ScenePlanEditPatchCodec(codec)
    observed = codec.project_plan(_plan())
    allowed = patch_codec.allowed_next_ids(
        [patch_codec.bos_id], input_sceneplan=observed
    )
    assert patch_codec.token_to_id["<op_move_source>"] in allowed
    assert patch_codec.token_to_id["<op_rotate_source>"] not in allowed
    assert patch_codec.token_to_id["<op_scale_distance>"] not in allowed
    assert set(ACTIVE_OPERATION_TOKENS) == {
        "no_op",
        "add_source",
        "remove_source",
        "replace_source",
        "move_source",
        "retime_source",
        "room_change",
        "change_speech_description",
        "change_transcript",
    }


@pytest.mark.parametrize("operation", tuple(ACTIVE_OPERATION_TOKENS))
def test_audio_aware_delta_keeps_numeric_values_out_of_discrete_program(
    operation: str,
) -> None:
    codec = load_model_sceneplan_codec(CODEC_PATH)
    patch_codec = ScenePlanEditPatchCodec(codec)
    observed = codec.project_plan(_two_source_plan())
    revised, _instruction, generated_operation, patch = make_deterministic_edit(
        codec,
        observed,
        ordinal=4,
        seed=42,
        requested_operation=operation,
    )
    assert generated_operation == operation
    program = compile_audio_aware_delta_sketch(
        observed, revised, patch, codec, patch_codec
    )
    delta_codec = AudioAwareDeltaSceneSketchCodec(codec, patch_codec)
    tokens = delta_codec.encode(program)["input_ids"]
    assert torch.equal(delta_codec.canonicalize(tokens)["input_ids"], tokens)
    assert not ({"trajectory", "new_interval_frames", "source", "new_source"} & set(program))

    observed_state = compile_execution_state(observed, codec)
    revised_state = compile_execution_state(revised, codec)
    projected = project_audio_aware_delta_to_atomic_patch(
        observed,
        program,
        execution_delta_core(observed_state, revised_state),
        codec,
        patch_codec,
    )
    assert torch.equal(
        codec.encode(projected["revised_sceneplan"])["input_ids"],
        codec.encode(revised)["input_ids"],
    )
    patch_codec.assert_target(
        observed, projected["patch_spec"], projected["revised_sceneplan"]
    )


@pytest.mark.parametrize(
    ("operation", "expected_coordinates"),
    [
        ("no_op", 0),
        ("add_source", 14),
        ("remove_source", 0),
        ("replace_source", 0),
        ("move_source", 12),
        ("retime_source", 2),
        ("room_change", 0),
        ("change_speech_description", 0),
        ("change_transcript", 0),
    ],
)
def test_audio_aware_delta_control_mask_matches_assembler_authority(
    operation: str, expected_coordinates: int
) -> None:
    codec = load_model_sceneplan_codec(CODEC_PATH)
    patch_codec = ScenePlanEditPatchCodec(codec)
    observed = codec.project_plan(_two_source_plan())
    revised, _instruction, generated_operation, patch = make_deterministic_edit(
        codec,
        observed,
        ordinal=4,
        seed=42,
        requested_operation=operation,
    )
    assert generated_operation == operation
    program = compile_audio_aware_delta_sketch(
        observed, revised, patch, codec, patch_codec
    )
    mask = audio_aware_delta_control_mask(program)
    assert mask.shape == (5, 15)
    assert int(mask.sum()) == expected_coordinates


def test_audio_aware_retime_is_multiscale_and_p10_frame_aligned() -> None:
    codec = load_model_sceneplan_codec(CODEC_PATH)
    observed = codec.project_plan(_two_source_plan())
    seen_magnitudes: set[int] = set()
    seen_modes: set[str] = set()
    for ordinal in range(80):
        revised, _instruction, operation, spec = make_deterministic_edit(
            codec,
            observed,
            ordinal=ordinal,
            seed=42,
            requested_operation="retime_source",
        )
        assert operation == "retime_source"
        assert spec["retime_policy"] == RETIME_POLICY
        magnitude = int(spec["retime_magnitude_frames"])
        mode = str(spec["retime_mode"])
        assert magnitude in RETIME_FRAME_LEVELS
        assert mode in RETIME_MODES
        assert magnitude >= 4
        source_id = str(spec["source_id"])
        before = next(
            source for source in observed["sources"] if source["source_id"] == source_id
        )
        after = next(
            source for source in revised["sources"] if source["source_id"] == source_id
        )
        before_frames = tuple(
            codec._frame_from_seconds(before["activity"][key], mode="nearest")
            for key in ("onset_sec", "offset_sec")
        )
        after_frames = tuple(
            codec._frame_from_seconds(after["activity"][key], mode="nearest")
            for key in ("onset_sec", "offset_sec")
        )
        assert max(abs(a - b) for a, b in zip(after_frames, before_frames)) == magnitude
        seen_magnitudes.add(magnitude)
        seen_modes.add(mode)
    assert seen_magnitudes == set(RETIME_FRAME_LEVELS)
    assert len(seen_modes) >= 4


def test_delta_control_retime_loss_is_normalized_in_p10_frame_units() -> None:
    durations = (320, 640)
    observed = torch.zeros(2, 5, 15)
    predicted = torch.zeros_like(observed)
    target = torch.zeros_like(observed)
    duration_index = EXECUTION_FEATURE_NAMES.index("duration_frames_norm")
    onset_index = EXECUTION_FEATURE_NAMES.index("onset_frame_norm")
    offset_index = EXECUTION_FEATURE_NAMES.index("offset_frame_norm")
    for row, duration in enumerate(durations):
        observed[row, 0, duration_index] = duration / 648
        target[row, 1, onset_index] = 32 / duration
        target[row, 1, offset_index] = 32 / duration
    predicted[:, 4, -1] = 100.0  # an unowned coordinate must be ignored
    per_row, active, raw_rmse = _audio_aware_delta_control_objective(
        predicted,
        target,
        observed,
        [
            {
                "contract": AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT,
                "operation": "retime_source",
                "source_id": "source_0",
            },
            {
                "contract": AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT,
                "operation": "retime_source",
                "source_id": "source_0",
            },
        ],
        max_frames=648,
        retime_frame_unit=32,
        beta=1.0,
    )
    assert active.tolist() == [True, True]
    assert torch.allclose(per_row, torch.full((2,), 0.5), atol=1.0e-6)
    assert torch.allclose(
        raw_rmse,
        torch.tensor([32 / durations[0], 32 / durations[1]]),
        atol=1.0e-6,
    )
