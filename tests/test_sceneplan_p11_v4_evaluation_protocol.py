import pytest

from scripts.t2a.eval.evaluate_sceneplan_p11_v4_p10_closure import (
    _render_length_integrity,
)
from scripts.t2a.eval.summarize_sceneplan_p11_v4_lexical_ab import (
    _causal_gate_values,
)
from scripts.t2a.eval.evaluate_sceneplan_p11_v4_sketch_exposure import (
    _summarize_exposure_rows,
)
from scripts.t2a.eval.summarize_sceneplan_p11_v4_u_inventory import (
    _summarize as _summarize_u_inventory,
)


def _length_row(predicted_frames: int, target_frames: int) -> dict:
    return {
        "predicted_shape": [4, predicted_frames * 1024],
        "target_shape": [4, target_frames * 1024],
        "predicted_model_num_samples": predicted_frames * 1024,
        "target_model_num_samples": target_frames * 1024,
        "predicted_latent_frames_valid": predicted_frames,
        "target_latent_frames_valid": target_frames,
    }


def test_p10_closure_accepts_different_legal_sceneplan_durations() -> None:
    gates = _render_length_integrity([_length_row(621, 620)])
    assert all(gates.values())


def test_p10_closure_rejects_render_not_matching_its_own_sceneplan() -> None:
    row = _length_row(621, 620)
    row["predicted_shape"][-1] -= 1024
    gates = _render_length_integrity([row])
    assert not gates["all_predicted_render_lengths_match_own_plan"]
    assert gates["all_target_render_lengths_match_own_plan"]


def test_p10_closure_rejects_duration_above_frozen_15s_capability() -> None:
    gates = _render_length_integrity([_length_row(649, 648)])
    assert not gates["all_plan_lengths_within_p10_capability"]


def test_reliable_asr_uses_material_mean_and_90_percent_positive_gate() -> None:
    # One difficult row may regress; a 19/20 positive, materially positive
    # paired effect is stronger and more meaningful than requiring 20/20.
    deltas = [0.05] * 19 + [-0.01]
    gates = _causal_gate_values(
        on_valid_rate=1.0,
        drop_valid_rate=1.0,
        applied_deltas=deltas,
        unapplied_exact=[True, True],
        non_u_exact=[],
        drop_authority_applied=[False] * 22,
        understanding_on_mean=0.70,
        understanding_drop_mean=0.64,
        arm="flow",
        numeric_states_equal=[False] * 22,
    )
    assert all(gates.values())


def test_reliable_asr_still_requires_exact_unapplied_rows() -> None:
    gates = _causal_gate_values(
        on_valid_rate=1.0,
        drop_valid_rate=1.0,
        applied_deltas=[0.05] * 20,
        unapplied_exact=[True, False],
        non_u_exact=[],
        drop_authority_applied=[False] * 22,
        understanding_on_mean=0.70,
        understanding_drop_mean=0.64,
        arm="flow",
        numeric_states_equal=[False] * 22,
    )
    assert not gates["unapplied_u_rows_exactly_unchanged"]


def test_reliable_asr_rejects_authority_in_drop_arm() -> None:
    gates = _causal_gate_values(
        on_valid_rate=1.0,
        drop_valid_rate=1.0,
        applied_deltas=[0.05] * 20,
        unapplied_exact=[True, True],
        non_u_exact=[],
        drop_authority_applied=[False] * 21 + [True],
        understanding_on_mean=0.70,
        understanding_drop_mean=0.64,
        arm="flow",
        numeric_states_equal=[False] * 22,
    )
    assert not gates["drop_authority_never_applied"]


def test_sketch_exposure_summary_separates_matched_and_mismatched_rows() -> None:
    rows = [
        {
            "task": "generation",
            "status": "PASS",
            "discrete_exact": True,
            "matched_context_exact_parity": True,
            "deployment": {
                "task_score": 0.8,
                "task_metrics": {"source_count_accuracy": 1.0},
            },
            "teacher_context": {
                "task_score": 0.8,
                "task_metrics": {"source_count_accuracy": 1.0},
            },
            "task_score_lift": 0.0,
            "deployment_continuous_rmse": 0.2,
            "teacher_context_continuous_rmse": 0.2,
            "continuous_rmse_reduction": 0.0,
        },
        {
            "task": "understanding",
            "status": "PASS",
            "discrete_exact": False,
            "matched_context_exact_parity": False,
            "deployment": {
                "task_score": 0.5,
                "task_metrics": {"source_count_accuracy": 0.0},
            },
            "teacher_context": {
                "task_score": 0.7,
                "task_metrics": {"source_count_accuracy": 1.0},
            },
            "task_score_lift": 0.2,
            "deployment_continuous_rmse": 0.4,
            "teacher_context_continuous_rmse": 0.3,
            "continuous_rmse_reduction": 0.1,
        },
    ]
    summary = _summarize_exposure_rows(rows)
    assert summary["overall"]["discrete_exact_rate"] == 0.5
    assert summary["overall"]["continuous_rmse_reduction"] == pytest.approx(0.05)
    assert summary["overall"]["mismatched_continuous_rmse_reduction"] == 0.1
    assert summary["overall"]["matched_context_exact_parity_rate"] == 1.0
    source_count = summary["overall"]["task_metric_means"][
        "source_count_accuracy"
    ]
    assert source_count["deployment"] == 0.5
    assert source_count["teacher_context"] == 1.0
    assert source_count["teacher_minus_deployment"] == 0.5


def test_sketch_exposure_localizes_structure_vs_surface_mismatch() -> None:
    def row(*, count: float, kind: float, room: float) -> dict[str, object]:
        deployment_metrics = {
            "source_count_accuracy": count,
            "kind_accuracy": kind,
            "room_accuracy": room,
        }
        teacher_metrics = {
            "source_count_accuracy": 1.0,
            "kind_accuracy": 1.0,
            "room_accuracy": 1.0,
        }
        return {
            "task": "understanding",
            "status": "PASS",
            "discrete_exact": False,
            "matched_context_exact_parity": False,
            "deployment": {
                "task_score": 0.5,
                "task_metrics": deployment_metrics,
            },
            "teacher_context": {
                "task_score": 0.7,
                "task_metrics": teacher_metrics,
            },
            "task_score_lift": 0.2,
            "deployment_continuous_rmse": 0.4,
            "teacher_context_continuous_rmse": 0.3,
            "continuous_rmse_reduction": 0.1,
        }

    summary = _summarize_exposure_rows(
        [
            row(count=1.0, kind=1.0, room=1.0),
            row(count=0.0, kind=1.0, room=1.0),
        ]
    )["by_task"]["understanding"]["mismatch_attribution"]
    assert summary["structure_exact_surface_or_lexical_mismatch"]["rows"] == 1
    assert summary["room_count_or_kind_mismatch"]["rows"] == 1


def test_u_inventory_audit_stratifies_target_count_and_structure() -> None:
    rows = [
        {
            "ordinal": 4,
            "scored": [
                {
                    "valid": True,
                    "task_score": 0.8,
                    "reference_anchor_nearest_core_rmse": 0.1,
                    "sceneplan": {"sources": [{}, {}]},
                    "task_metrics": {
                        "room_accuracy": 1.0,
                        "source_count_accuracy": 1.0,
                        "kind_accuracy": 1.0,
                    },
                }
            ],
        },
        {
            "ordinal": 9,
            "scored": [
                {
                    "valid": True,
                    "task_score": 0.5,
                    "reference_anchor_nearest_core_rmse": 0.4,
                    "sceneplan": {"sources": [{}, {}, {}, {}]},
                    "task_metrics": {
                        "room_accuracy": 1.0,
                        "source_count_accuracy": 0.0,
                        "kind_accuracy": 1.0,
                    },
                }
            ],
        },
    ]
    summary = _summarize_u_inventory(rows, {4: 2, 9: 3})
    assert summary["source_count_accuracy"] == 0.5
    assert summary["structure_exact_rate"] == 0.5
    assert summary["by_target_source_count"]["2"]["source_count_accuracy"] == 1.0
    assert summary["by_target_source_count"]["3"][
        "predicted_count_distribution"
    ] == {"4": 1}
