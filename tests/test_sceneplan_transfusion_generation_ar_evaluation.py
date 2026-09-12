from __future__ import annotations

import copy

import pytest

from stable_audio_tools.data.sceneplan_transfusion_generation_ar_evaluation import (
    score_parsed_generation,
    score_token_sequence,
    summarize_prediction_records,
)


class _TinyCodec:
    bos_id = 1
    eos_id = 2

    @staticmethod
    def allowed_next_ids(prefix):
        transitions = {(1,): {3}, (1, 3): {2}}
        return transitions.get(tuple(prefix), set())


def _plan() -> dict:
    return {
        "sample_id": "eval_test",
        "duration_sec": 2.0,
        "room": {"type": "dry"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "sound",
                "description": "a short bell ring",
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


def test_token_metrics_replay_the_grammar() -> None:
    codec = _TinyCodec()
    exact = score_token_sequence([1, 3, 2], [1, 3, 2], codec)
    assert exact["token_sequence_exact"] == 1.0
    assert exact["grammar_legal"] == 1.0
    illegal = score_token_sequence([1, 3, 2], [1, 2], codec)
    assert illegal["token_sequence_exact"] == 0.0
    assert illegal["grammar_legal"] == 0.0


def test_parsed_metrics_expose_both_source_matching_modes() -> None:
    target = _plan()
    prediction = copy.deepcopy(target)
    metrics = score_parsed_generation(target, prediction)
    assert metrics["persistent_scene_exact"] == 1.0
    assert metrics["permutation_scene_exact"] == 1.0


def test_ordered_metrics_keep_fixed_zero_penalties_for_wrong_source_slot() -> None:
    target = _plan()
    target["sources"][0].pop("description")
    target["sources"][0].update(
        {
            "kind": "speech",
            "speaker_description": "an adult speaker",
            "transcript": "test phrase",
        }
    )
    prediction = copy.deepcopy(target)
    prediction["sources"][0]["source_id"] = "source_1"

    metrics = score_parsed_generation(target, prediction)
    for name in (
        "ordered_kind_exact",
        "ordered_description_exact",
        "ordered_activity_onset_exact",
        "ordered_activity_offset_exact",
        "ordered_trajectory_type_exact",
        "ordered_gain_exact",
        "ordered_start_position_exact",
        "ordered_end_position_exact",
        "ordered_transcript_exact",
    ):
        assert name in metrics
        assert metrics[name] == 0.0
    assert metrics["ordered_start_azimuth_mae_deg"] == 180.0
    assert metrics["ordered_end_azimuth_mae_deg"] == 180.0
    assert metrics["ordered_start_elevation_mae_deg"] == 90.0
    assert metrics["ordered_end_elevation_mae_deg"] == 90.0
    assert metrics["ordered_start_distance_mae_m"] == 50.0
    assert metrics["ordered_end_distance_mae_m"] == 50.0
    assert metrics["ordered_source_count_exact"] == 1.0


def test_summary_penalizes_failed_decodes_and_rejects_duplicates() -> None:
    records = [
        {
            "ordinal": 0,
            "status": "ok",
            "source_count": 1,
            "template_id": "a",
            "metrics": {"token_sequence_exact": 1.0, "azimuth_mae_deg": 2.0},
        },
        {
            "ordinal": 1,
            "status": "generation_error",
            "source_count": 2,
            "template_id": "b",
            "metrics": {},
        },
    ]
    summary = summarize_prediction_records(records, expected_rows=2)
    assert summary["generation_success_rate"] == 0.5
    assert summary["means_on_successful_decodes"]["token_sequence_exact"] == 1.0
    assert summary["failure_penalized_means"]["token_sequence_exact"] == 0.5
    assert "azimuth_mae_deg" not in summary["failure_penalized_means"]

    duplicate = [records[0], dict(records[0])]
    with pytest.raises(ValueError, match="duplicate"):
        summarize_prediction_records(duplicate, expected_rows=2)


def test_summary_separates_token_and_parsed_domains() -> None:
    records = [
        {
            "ordinal": 0,
            "status": "ok",
            "source_count": 1,
            "template_id": "a",
            "metrics": {
                "token_sequence_exact": 1.0,
                "grammar_legal": 1.0,
                "token_count_abs_error": 0.0,
                "parse_rate": 1.0,
                "persistent_scene_exact": 1.0,
            },
        },
        {
            "ordinal": 1,
            "status": "parse_error",
            "source_count": 1,
            "template_id": "a",
            "metrics": {
                "token_sequence_exact": 0.0,
                "grammar_legal": 1.0,
                "token_count_abs_error": 3.0,
                "parse_rate": 0.0,
            },
        },
        {
            "ordinal": 2,
            "status": "generation_error",
            "source_count": 2,
            "template_id": "b",
            "metrics": {},
        },
    ]

    summary = summarize_prediction_records(records, expected_rows=3)
    assert summary["status_counts"] == {
        "generation_error": 1,
        "ok": 1,
        "parse_error": 1,
    }
    assert summary["generated_rows"] == 2
    assert summary["parsed_rows"] == 1
    assert summary["generation_success_rate"] == pytest.approx(2 / 3)
    assert summary["parse_success_rate"] == pytest.approx(1 / 3)
    assert summary["coverage"]["parse_success_rate_given_generation"] == 0.5

    token = summary["metric_domains"]["token_sequence"]
    assert token["rows_with_metrics"] == 2
    assert token["means_on_generated_sequences"]["grammar_legal"] == 1.0
    assert token["means_on_generated_sequences"]["token_count_abs_error"] == 1.5
    assert token["failure_penalized_means"]["grammar_legal"] == pytest.approx(
        2 / 3
    )
    assert token["failure_penalized_means"]["token_sequence_exact"] == pytest.approx(
        1 / 3
    )

    parsed = summary["metric_domains"]["parsed_sceneplan"]
    assert parsed["rows_with_metrics"] == 1
    assert parsed["means_on_parsed_sceneplans"]["persistent_scene_exact"] == 1.0
    assert parsed["failure_penalized_means"][
        "persistent_scene_exact"
    ] == pytest.approx(1 / 3)
    assert summary["failure_penalized_means"]["grammar_legal"] == pytest.approx(
        2 / 3
    )
    assert summary["failure_penalized_means"]["parse_rate"] == pytest.approx(1 / 3)

    grouped = summary["by_template_id"]["a"]
    assert grouped["generated_rows"] == 2
    assert grouped["parsed_rows"] == 1
    assert grouped["generation_success_rate"] == 1.0
    assert grouped["parse_success_rate"] == 0.5


def test_all_generation_errors_have_explicit_zero_penalties() -> None:
    summary = summarize_prediction_records(
        [
            {
                "ordinal": 0,
                "status": "generation_error",
                "source_count": 1,
                "template_id": "a",
                "metrics": {},
            }
        ],
        expected_rows=1,
    )
    assert summary["coverage"] == {
        "expected_rows": 1,
        "ordinal_coverage_exact": True,
        "rows": 1,
        "generated_rows": 0,
        "parsed_rows": 0,
        "generation_error_rows": 1,
        "parse_error_rows": 0,
        "other_status_rows": 0,
        "generation_coverage_rate": 0.0,
        "parse_success_rate_over_all_rows": 0.0,
        "parse_success_rate_given_generation": 0.0,
    }
    token_penalties = summary["metric_domains"]["token_sequence"][
        "failure_penalized_means"
    ]
    assert token_penalties["token_sequence_exact"] == 0.0
    assert token_penalties["grammar_legal"] == 0.0
    parsed_penalties = summary["metric_domains"]["parsed_sceneplan"][
        "failure_penalized_means"
    ]
    assert parsed_penalties["persistent_scene_exact"] == 0.0
    assert parsed_penalties["ordered_kind_exact"] == 0.0
    assert summary["failure_penalized_means"]["parse_rate"] == 0.0
