#!/usr/bin/env python3
"""CPU gate for the shared reliable-ASR boundary across D0/Direct/Flow."""

from __future__ import annotations
import os

import json
import sqlite3
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.evaluate_sceneplan_p11_v4_challenge import (  # noqa: E402
    _normalize_d0_output,
)
from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.scene_sketch_v1 import (  # noqa: E402
    compile_execution_state,
    execution_state_core,
)
from stable_audio_tools.data.sceneplan_p11_lexical_cache import (  # noqa: E402
    P11_LEXICAL_AUTHORITY_CONTRACT,
)


CODEC = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
CHALLENGE = REPO_ROOT / (
    "artifacts/sceneplan_p11/challenges/"
    "p11_v4_heldout_challenge_v1_20260901.sqlite"
)


def _non_speech_plan() -> dict:
    connection = sqlite3.connect(f"file:{CHALLENGE}?mode=ro&immutable=1", uri=True)
    try:
        rows = connection.execute(
            "SELECT target_sceneplan_zlib FROM rows WHERE task='understanding'"
        )
        for (payload,) in rows:
            plan = json.loads(zlib.decompress(payload))
            if len(plan["sources"]) >= 2 and all(
                source["kind"] != "speech" for source in plan["sources"]
            ):
                return plan
    finally:
        connection.close()
    raise RuntimeError("challenge lacks a multi-source non-speech U plan")


def main() -> None:
    codec = load_model_sceneplan_codec(CODEC)
    plan = codec.project_plan(_non_speech_plan())
    tokens = codec.encode(plan)["input_ids"]
    source_ids = [str(source["source_id"]) for source in plan["sources"]]
    chosen = source_ids[-1]
    scores = []
    for index, source_id in enumerate(source_ids):
        wins = source_id == chosen
        scores.append(
            {
                "source_id": source_id,
                "music_probability": 0.45,
                "sound_probability": 0.45,
                "speech_probability": 0.90 if wins else 0.10,
                "speech_margin": 2.0 if wins else -2.0 - index,
            }
        )
    planner = SimpleNamespace(
        plan_codec=codec,
        plan_max_tokens=768,
        patch_codec=None,
    )
    lexical = {
        "lexical_authority": {
            "contract": P11_LEXICAL_AUTHORITY_CONTRACT,
            "transcript": "input only frozen asr hypothesis",
            "confidence": 0.91,
            "language": "en",
            "target_transcript_access": False,
        }
    }
    before = execution_state_core(compile_execution_state(plan, codec)).copy()
    output = _normalize_d0_output(
        planner,
        tokens,
        {"source_kind_scores": scores},
        metadata={
            "p11_task": "understanding",
            "p11_target_sample_id": str(plan["sample_id"]),
        },
        input_lexical=lexical,
    )
    after = execution_state_core(
        compile_execution_state(output["sceneplan"], codec)
    )
    if not np.array_equal(before, after):
        raise RuntimeError("D0 reliable-ASR boundary changed numeric execution")
    selected = next(
        source
        for source in output["sceneplan"]["sources"]
        if source["source_id"] == chosen
    )
    if selected["kind"] != "speech" or selected["transcript"] != (
        "input only frozen asr hypothesis"
    ):
        raise RuntimeError("D0 reliable-ASR boundary did not promote its argmax owner")
    if not output["diagnostics"]["lexical_authority_applied"]:
        raise RuntimeError("D0 reliable-ASR application was not audited")
    if output["diagnostics"]["lexical_authority_source_id"] != chosen:
        raise RuntimeError("D0 reliable-ASR owner audit is incorrect")
    if output["diagnostics"]["lexical_authority_action"] != (
        "promote_argmax_speech_likelihood_source"
    ):
        raise RuntimeError("D0 reliable-ASR promotion action changed")

    dropped = _normalize_d0_output(
        planner,
        tokens,
        {"source_kind_scores": scores},
        metadata={
            "p11_task": "understanding",
            "p11_target_sample_id": str(plan["sample_id"]),
        },
        input_lexical=None,
    )
    if dropped["diagnostics"]["lexical_authority_applied"]:
        raise RuntimeError("ASR-drop D0 counterfactual retained lexical authority")
    if dropped["sceneplan"] != plan:
        raise RuntimeError("ASR-drop D0 counterfactual changed the autoregressive plan")

    print(
        json.dumps(
            {
                "status": "PASS",
                "gate": "p11_v4_fair_input_only_lexical_boundary_v1",
                "d0_post_assembly": True,
                "shared_argmax_owner": chosen,
                "numeric_execution_bitwise_immutable": True,
                "drop_counterfactual_identity": True,
                "target_transcript_access": "forbidden",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
