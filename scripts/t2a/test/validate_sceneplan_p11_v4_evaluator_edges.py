#!/usr/bin/env python3
"""Regression gate for unified P11-v4 evaluator failure aggregation."""

from __future__ import annotations
import os

import json
import sys
from types import SimpleNamespace
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.evaluate_sceneplan_p11_v4_challenge import (  # noqa: E402
    _aggregate,
    _decode_flow,
    _ids,
    _summarize_prefix,
)
from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.sceneplan_edit_patch import (  # noqa: E402
    ScenePlanEditPatchCodec,
)


def main() -> None:
    codec = load_model_sceneplan_codec(
        os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
        "model_sceneplan_codec_v4"
    )
    patch_codec = ScenePlanEditPatchCodec(codec)
    predicted_patch = {
        "operation": "distance_source",
        "source_id": "source_1",
        "distance_factor": 0.75,
    }
    audited_target_patch = {
        **predicted_patch,
        "contract": "same_scene_atomic_patch_v1",
        "changed_paths": ["sources.source_1.trajectory.distance"],
        "preserve_all_unspecified_fields": True,
        "input_sceneplan_tokens_required": True,
        "input_foa_required": False,
    }
    canonical_predicted = patch_codec.decode(
        _ids(patch_codec.encode(predicted_patch))
    )
    canonical_target = patch_codec.decode(
        _ids(patch_codec.encode(audited_target_patch))
    )
    if canonical_predicted != canonical_target:
        raise RuntimeError("canonical patch equality retained audit-only fields")
    scored = [
        {
            "valid": False,
            "parse": False,
            "roundtrip": False,
            "finite": False,
            "task_score": 0.0,
            "error": f"synthetic invalid draw {index}",
        }
        for index in range(4)
    ]
    metadata = {"p11_task": "generation"}
    prefix = _summarize_prefix(scored, metadata, k=4)
    expected = {
        "valid_rate": 0.0,
        "semantic_immutability": False,
        "unique_numeric_count": 0,
        "constraint_pass_rate": 0.0,
        "constraint_pass_any": False,
        "reference_anchor_nearest_core_rmse_mean": None,
        "invalid_draws": 4,
    }
    for key, value in expected.items():
        if prefix.get(key) != value:
            raise RuntimeError(
                f"all-invalid prefix {key}={prefix.get(key)!r}, expected {value!r}"
            )
    aggregate = _aggregate(
        [
            {
                "family": "generation_numeric_posterior",
                "task": "generation",
                "prefixes": {"4": prefix},
            }
        ],
        [4],
    )
    summary = aggregate["all"]["4"]
    if summary["valid_rate"] != 0.0 or summary["task_score_mean"] != 0.0:
        raise RuntimeError("all-invalid rows were not retained as zero-score failures")
    if summary["reference_anchor_nearest_core_rmse"] is not None:
        raise RuntimeError("undefined all-invalid reference RMSE became a numeric claim")

    class FakePlanner:
        def __init__(self, *, editing_uses_direct: bool) -> None:
            self.plan_embedding = torch.nn.Embedding(1, 1)
            self.execution_reasoner = SimpleNamespace(
                editing_uses_direct=editing_uses_direct
            )
            self.direct_calls = 0
            self.sample_calls = 0
            self.last_input_lexical = "unset"

        def decode_transfusion_cot(self, *args, **kwargs):
            self.direct_calls += 1
            if kwargs.get("noise_seed") is not None or kwargs.get("noise") is not None:
                raise RuntimeError("deterministic evaluator route leaked Flow noise")
            return {"diagnostics": {"thought_noise_source": None}}

        def decode_transfusion_cot_samples(self, *args, **kwargs):
            self.sample_calls += 1
            self.last_input_lexical = kwargs.get("input_lexical")
            seeds = list(kwargs["noise_seeds"])
            return [
                {
                    "diagnostics": {
                        "posterior_draw_index": index,
                        "thought_noise_seeds": [seed],
                    }
                }
                for index, seed in enumerate(seeds)
            ]

    editing_metadata = {
        "p11_prompt": "move source one farther",
        "p11_task": "editing",
        "p11_challenge_id": "evaluator_edge_edit",
        "p11_target_sample_id": "evaluator_edge_edit",
        "p11_input_sceneplan": {"duration_sec": 15.0},
    }
    hybrid = FakePlanner(editing_uses_direct=True)
    editing_outputs, editing_rng = _decode_flow(
        hybrid,
        editing_metadata,
        draws=4,
        root_seed=20260901,
        discrete_decode_mode="prefix_recompute",
    )
    if hybrid.direct_calls != 1 or hybrid.sample_calls != 0:
        raise RuntimeError("hybrid Flow evaluator did not use deterministic E route")
    if len(editing_outputs) != 4 or not all(editing_rng):
        raise RuntimeError("hybrid Flow evaluator did not retain the requested K budget")
    if any(
        output["diagnostics"].get("posterior_draw_is_stochastic") is not False
        or output["diagnostics"].get("thought_noise_source") is not None
        for output in editing_outputs
    ):
        raise RuntimeError("deterministic E repeats were mislabeled as posterior draws")

    generation_metadata = {
        "p11_prompt": "a short rain scene",
        "p11_prompt_text": "Create a 15-second FOA scene with rain.",
        "p11_task": "generation",
        "p11_challenge_id": "evaluator_edge_generation",
        "p11_target_sample_id": "evaluator_edge_generation",
    }
    stochastic = FakePlanner(editing_uses_direct=True)
    generation_outputs, generation_rng = _decode_flow(
        stochastic,
        generation_metadata,
        draws=4,
        root_seed=20260901,
        discrete_decode_mode="prefix_recompute",
    )
    if stochastic.direct_calls != 0 or stochastic.sample_calls != 1:
        raise RuntimeError("Flow G route did not use explicit posterior samples")
    generation_seeds = [
        output["diagnostics"]["thought_noise_seeds"][0]
        for output in generation_outputs
    ]
    if len(set(generation_seeds)) != 4 or not all(generation_rng):
        raise RuntimeError("Flow G route did not retain four isolated noise draws")
    if any(
        output["diagnostics"].get("posterior_draw_is_stochastic") is not True
        for output in generation_outputs
    ):
        raise RuntimeError("Flow G posterior draws were not labeled stochastic")

    lexical_metadata = {
        **generation_metadata,
        "p11_input_lexical": {
            "lexical_authority": {
                "transcript": "frozen input evidence",
            }
        },
    }
    lexical_counterfactual = FakePlanner(editing_uses_direct=True)
    _decode_flow(
        lexical_counterfactual,
        lexical_metadata,
        draws=1,
        root_seed=20260901,
        discrete_decode_mode="prefix_recompute",
        lexical_authority_intervention="drop_reliable_asr",
    )
    if lexical_counterfactual.last_input_lexical is not None:
        raise RuntimeError("ASR-drop counterfactual crossed the model boundary")
    print(
        json.dumps(
            {
                "status": "PASS",
                "gate": "p11_v4_unified_evaluator_all_invalid_regression_v1",
                "invalid_draws_retained": 4,
                "canonical_patch_audit_fields_ignored": True,
                "hybrid_flow_gu_direct_e_route": True,
                "lexical_authority_drop_before_model": True,
                "aggregate_valid_rate": summary["valid_rate"],
                "aggregate_task_score": summary["task_score_mean"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
