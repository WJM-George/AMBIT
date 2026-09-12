from scripts.t2a.eval.diagnose_sceneplan_p11_v4_train_pair_causality import (
    _score_executable_axis_pair,
)


def test_executable_axis_pair_scores_valid_signed_pair():
    result = _score_executable_axis_pair(
        {
            -1: {"selected_executable_axis_score": -1.5},
            1: {"selected_executable_axis_score": 2.0},
        }
    )

    assert result == {
        "executable_axis_scores": {"-1": -1.5, "1": 2.0},
        "executable_axis_scores_present": True,
        "executable_axis_score_gap": 3.5,
        "executable_axis_both_signs_correct": True,
    }


def test_executable_axis_pair_missing_score_is_explicit_failure():
    result = _score_executable_axis_pair(
        {
            -1: {"selected_executable_axis_score": None},
            1: {"selected_executable_axis_score": 2.0},
        }
    )

    assert result == {
        "executable_axis_scores": {"-1": None, "1": 2.0},
        "executable_axis_scores_present": False,
        "executable_axis_score_gap": None,
        "executable_axis_both_signs_correct": False,
    }
