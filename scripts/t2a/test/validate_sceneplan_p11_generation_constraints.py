#!/usr/bin/env python3
"""Regression checks for prompt-conditional P11 Generation scoring."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_p11_metrics import (  # noqa: E402
    P11_GENERATION_CONSTRAINT_CONTRACT,
    score_generation_constraints,
)


def _plan() -> dict:
    return {
        "sample_id": "generation_constraint_metric",
        "duration_sec": 10.0,
        "room": {"type": "dry"},
        "sources": [
            {
                "source_id": "source_0",
                "kind": "music",
                "description": "soft solo piano",
                "activity": {"onset_sec": 0.0, "offset_sec": 10.0},
                "trajectory": {
                    "type": "static",
                    "position": {
                        "azimuth_deg": -45.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.0,
                    },
                },
                "gain_db": 0.0,
            },
            {
                "source_id": "source_1",
                "kind": "sound",
                "description": "steady evening rain",
                "activity": {"onset_sec": 6.0, "offset_sec": 9.0},
                "trajectory": {
                    "type": "linear",
                    "start": {
                        "azimuth_deg": -90.0,
                        "elevation_deg": 0.0,
                        "distance_m": 3.0,
                    },
                    "end": {
                        "azimuth_deg": 90.0,
                        "elevation_deg": 0.0,
                        "distance_m": 1.5,
                    },
                },
                "gain_db": 0.0,
            },
        ],
    }


def _score(target: dict, prediction: dict, groups: list[str]) -> float:
    return score_generation_constraints(
        target,
        prediction,
        known_field_groups=groups,
        source_matching="permutation_invariant",
    )["task_score"]


def main() -> None:
    target = _plan()

    semantic_only = copy.deepcopy(target)
    semantic_only["room"] = {"type": "reverberant"}
    semantic_only["sources"][0]["activity"] = {
        "onset_sec": 1.0,
        "offset_sec": 8.0,
    }
    semantic_only["sources"][0]["trajectory"]["position"] = {
        "azimuth_deg": 135.0,
        "elevation_deg": 15.0,
        "distance_m": 3.0,
    }
    semantic_groups = ["duration", "source_semantics"]
    semantic_score = _score(target, semantic_only, semantic_groups)
    if semantic_score != 1.0:
        raise AssertionError(
            "unspecified room/timing/geometry changed semantic-only score"
        )

    wrong_semantic = copy.deepcopy(semantic_only)
    wrong_semantic["sources"][0]["description"] = "distorted electric guitar"
    wrong_semantic_score = _score(target, wrong_semantic, semantic_groups)
    if not wrong_semantic_score < semantic_score:
        raise AssertionError("a stated semantic error did not lower the score")

    semantic_temporal_groups = [
        "duration",
        "source_semantics",
        "temporal",
    ]
    semantic_temporal = copy.deepcopy(target)
    semantic_temporal["room"] = {"type": "outdoor"}
    semantic_temporal["sources"][0]["trajectory"]["position"][
        "azimuth_deg"
    ] = 90.0
    temporal_score = _score(
        target, semantic_temporal, semantic_temporal_groups
    )
    if temporal_score != 1.0:
        raise AssertionError("unspecified room/geometry changed temporal score")
    wrong_temporal = copy.deepcopy(semantic_temporal)
    wrong_temporal["sources"][1]["activity"] = {
        "onset_sec": 0.0,
        "offset_sec": 2.0,
    }
    if not _score(target, wrong_temporal, semantic_temporal_groups) < temporal_score:
        raise AssertionError("a stated exact-timing error did not lower the score")

    coarse_groups = [
        "room",
        "source_semantics",
        "coarse_temporal",
        "coarse_geometry",
    ]
    coarse_equivalent = copy.deepcopy(target)
    coarse_equivalent["sources"][1]["activity"] = {
        "onset_sec": 7.0,
        "offset_sec": 9.5,
    }
    coarse_equivalent["sources"][0]["trajectory"]["position"].update(
        {"azimuth_deg": -60.0, "distance_m": 1.1}
    )
    coarse_equivalent["sources"][1]["trajectory"]["start"].update(
        {"azimuth_deg": -100.0, "distance_m": 4.0}
    )
    coarse_equivalent["sources"][1]["trajectory"]["end"].update(
        {"azimuth_deg": 100.0, "distance_m": 1.8}
    )
    coarse_score = _score(target, coarse_equivalent, coarse_groups)
    if coarse_score != 1.0:
        raise AssertionError("within-bin coarse completion did not score perfectly")
    wrong_coarse = copy.deepcopy(coarse_equivalent)
    wrong_coarse["sources"][0]["trajectory"]["position"][
        "azimuth_deg"
    ] = 90.0
    wrong_coarse_score = _score(target, wrong_coarse, coarse_groups)
    if not wrong_coarse_score < coarse_score:
        raise AssertionError("a stated coarse-geometry error did not lower the score")

    exact_groups = [
        "duration",
        "room",
        "source_semantics",
        "temporal",
        "geometry",
        "gain",
    ]
    exact_score = _score(target, target, exact_groups)
    if exact_score != 1.0:
        raise AssertionError("the exact target did not receive a perfect score")

    print(
        json.dumps(
            {
                "status": "PASS",
                "contract": P11_GENERATION_CONSTRAINT_CONTRACT,
                "semantic_only_unspecified_mutation": semantic_score,
                "semantic_only_wrong_semantic": wrong_semantic_score,
                "semantic_temporal_unspecified_mutation": temporal_score,
                "coarse_equivalent": coarse_score,
                "coarse_wrong_geometry": wrong_coarse_score,
                "exact": exact_score,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
