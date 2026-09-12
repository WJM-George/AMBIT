#!/usr/bin/env python3
"""Strict CPU gate for active audio-aware P11 -> frozen P10.

The gate exercises every held-out G/U target and every E operation without
loading a planner checkpoint or touching a GPU.  It proves exact round trips,
grammar termination, semantic/numeric separation, source-local P10 effects,
and fail-closed capability limits.
"""

from __future__ import annotations

import argparse
import copy
import json
import sqlite3
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    ModelScenePlanCodecError,
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.scene_sketch_v1 import (  # noqa: E402
    AUDIO_AWARE_DELTA_OPERATIONS,
    EXECUTION_FEATURE_NAMES,
    AudioAwareDeltaSceneSketchCodec,
    SceneSketchCodec,
    assemble_sceneplan,
    compile_audio_aware_delta_sketch,
    compile_execution_state,
    compile_p10_from_contract,
    compile_scene_sketch,
    execution_delta_core,
    project_audio_aware_delta_to_atomic_patch,
    scene_sketch_semantic_state,
    validate_execution_state,
    validate_scene_sketch,
)
from stable_audio_tools.data.sceneplan_edit_patch import (  # noqa: E402
    ScenePlanEditPatchCodec,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    P10_CANONICAL_CHECKPOINT,
    P10_CANONICAL_CHECKPOINT_SHA256,
    P10_CANONICAL_CHECKPOINT_STEP,
)
from stable_audio_tools.data.scene_sketch_v1 import sha256_json  # noqa: E402


DEFAULT_MANIFEST = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/"
    "p11_audio_aware_edit_v2/manifests/p11_train_pilot90_v8_seed42.sqlite"
)
DEFAULT_CODEC = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/"
    "p11_single_turn_15s_v2/model_sceneplan_codec_v4"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/audio_aware_v1/"
    "pilot90_static_contract_seed42.json"
)


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _plan(payload: bytes) -> dict[str, Any]:
    value = json.loads(zlib.decompress(payload))
    if not isinstance(value, dict):
        raise RuntimeError("compressed ScenePlan is not an object")
    return value


def _positions(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    trajectory = source["trajectory"]
    motion = str(trajectory["type"])
    if motion == "static":
        return [trajectory["position"]]
    if motion == "linear":
        return [trajectory["start"], trajectory["end"]]
    return [item["position"] for item in trajectory["keyframes"]]


def _rotated_plan(
    plan: Mapping[str, Any], source_id: str, codec: Any
) -> dict[str, Any]:
    value = copy.deepcopy(dict(plan))
    source = next(
        item for item in value["sources"] if str(item["source_id"]) == source_id
    )
    for position in _positions(source):
        position["azimuth_deg"] = float(
            ((int(round(float(position["azimuth_deg"]))) + 45 + 180) % 360) - 180
        )
    return codec.project_plan(value)


def _p10_arrays(compiled: Mapping[str, Any]) -> dict[str, Any]:
    controls = compiled["sceneplan_44"]
    return {
        "semantic_text": str(compiled["semantic_caption"]["text"]),
        "present": np.asarray(controls["source_present_mask"]),
        "kind": np.asarray(controls["source_kind_ids"]),
        "event": np.asarray(controls["source_event_frame_ids"]),
        "trajectory": np.asarray(controls["source_trajectory_features"]),
    }


def _changed_tracks(left: np.ndarray, right: np.ndarray) -> list[int]:
    if left.shape != right.shape:
        return list(range(4))
    return [
        index for index in range(4) if not np.array_equal(left[index], right[index])
    ]


def _p10_delta(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    left = _p10_arrays(baseline)
    right = _p10_arrays(candidate)
    return {
        "semantic": left["semantic_text"] != right["semantic_text"],
        "present": _changed_tracks(left["present"], right["present"]),
        "kind": _changed_tracks(left["kind"], right["kind"]),
        "event": _changed_tracks(left["event"], right["event"]),
        "trajectory": _changed_tracks(left["trajectory"], right["trajectory"]),
    }


def _assert_subset(values: list[int], allowed: set[int], label: str) -> None:
    if not set(values) <= allowed:
        raise AssertionError(f"{label} spilled to tracks {values}, allowed={sorted(allowed)}")


def _assert_exact_roundtrip(
    plan: Mapping[str, Any], codec: Any, sketch_codec: SceneSketchCodec
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    projected = codec.project_plan(plan)
    sketch = compile_scene_sketch(projected, codec)
    execution = compile_execution_state(projected, codec)
    assembled = assemble_sceneplan(
        sketch, execution, codec, require_target_hash=True
    )
    if assembled != projected:
        raise AssertionError("SceneSketch + ExecutionState changed the target ScenePlan")
    tokens = sketch_codec.encode(sketch)["input_ids"]
    decoded = sketch_codec.decode(
        tokens, sceneplan_sha256=sketch["sceneplan_sha256"]
    )
    if decoded != sketch:
        raise AssertionError("SceneSketch token round-trip changed semantics")
    canonical = sketch_codec.canonicalize(tokens)["input_ids"]
    if not torch.equal(tokens, canonical):
        raise AssertionError("SceneSketch token canonicalization is not idempotent")
    for prefix_length, expected in enumerate(tokens.tolist()):
        allowed = sketch_codec.allowed_next_ids(
            tokens[:prefix_length], max_text_tokens=192
        )
        if int(expected) not in allowed:
            raise AssertionError("SceneSketch grammar rejected its target token")
    if sketch_codec.allowed_next_ids(tokens, max_text_tokens=192):
        raise AssertionError("SceneSketch grammar did not terminate")
    p10 = compile_p10_from_contract(
        sketch, execution, codec, require_target_hash=True
    )
    return sketch, execution, p10


def _assert_numeric_locality(
    sketch: Mapping[str, Any],
    execution: Mapping[str, Any],
    baseline_p10: Mapping[str, Any],
    codec: Any,
) -> None:
    source_id = str(sketch["sources"][0]["source_id"])
    source_index = int(source_id[7:])
    baseline_plan = assemble_sceneplan(sketch, execution, codec)
    rotated = _rotated_plan(baseline_plan, source_id, codec)
    rotated_execution = compile_execution_state(rotated, codec)
    candidate_execution = copy.deepcopy(dict(execution))
    candidate_execution["sceneplan_sha256"] = None
    candidate_execution["core_features"][source_index + 1] = copy.deepcopy(
        rotated_execution["core_features"][source_index + 1]
    )
    candidate_p10 = compile_p10_from_contract(
        sketch, candidate_execution, codec
    )
    delta = _p10_delta(baseline_p10, candidate_p10)
    if delta["semantic"]:
        raise AssertionError("numeric execution intervention changed P10 semantics")
    for key in ("present", "kind", "event"):
        if delta[key]:
            raise AssertionError(f"numeric geometry intervention changed P10 {key}")
    if delta["trajectory"] != [source_index]:
        raise AssertionError(
            "numeric geometry intervention was not exactly source-local: "
            f"{delta['trajectory']}"
        )
    if scene_sketch_semantic_state(sketch) != scene_sketch_semantic_state(
        compile_scene_sketch(candidate_p10["sceneplan"], codec)
    ):
        raise AssertionError("numeric execution intervention changed SceneSketch state")


def _assert_semantic_isolation(
    sketch: Mapping[str, Any],
    execution: Mapping[str, Any],
    baseline_p10: Mapping[str, Any],
    codec: Any,
) -> None:
    changed_sketch = copy.deepcopy(dict(sketch))
    changed_sketch["sceneplan_sha256"] = None
    source = changed_sketch["sources"][0]
    if source["kind"] == "speech":
        source["speaker_description"] += " clearly audible"
    else:
        source["description"] += " clearly audible"
    validate_scene_sketch(changed_sketch, codec=codec)
    candidate_p10 = compile_p10_from_contract(changed_sketch, execution, codec)
    delta = _p10_delta(baseline_p10, candidate_p10)
    if not delta["semantic"]:
        raise AssertionError("SceneSketch semantic intervention had no P10 effect")
    for key in ("present", "kind", "event", "trajectory"):
        if delta[key]:
            raise AssertionError(f"semantic intervention changed numeric P10 {key}")


def _assert_edit_locality(
    operation: str,
    spec: Mapping[str, Any],
    baseline: Mapping[str, Any],
    target: Mapping[str, Any],
) -> None:
    delta = _p10_delta(baseline, target)
    numeric_keys = ("present", "kind", "event", "trajectory")
    if operation == "no_op":
        if delta["semantic"] or any(delta[key] for key in numeric_keys):
            raise AssertionError("no-op changed P10 conditioning")
        return
    if operation in {
        "room_change",
        "change_speech_description",
        "change_transcript",
    }:
        if not delta["semantic"] or any(delta[key] for key in numeric_keys):
            raise AssertionError(f"{operation} escaped the semantic-only route")
        return
    source_id = (
        str(spec["source"]["source_id"])
        if operation == "add_source"
        else str(spec["source_id"])
    )
    source_index = int(source_id[7:])
    owner = {source_index}
    if operation == "move_source":
        if delta["semantic"] or delta["present"] or delta["kind"] or delta["event"]:
            raise AssertionError("move_source changed non-geometry P10 state")
        if delta["trajectory"] != [source_index]:
            raise AssertionError("move_source was not exactly source-local")
    elif operation == "retime_source":
        if delta["semantic"] or delta["present"] or delta["kind"]:
            raise AssertionError("retime_source changed semantic/inventory P10 state")
        if source_index not in delta["event"]:
            raise AssertionError("retime_source did not change its event track")
        _assert_subset(delta["event"], owner, "retime event")
        _assert_subset(delta["trajectory"], owner, "retime trajectory")
    elif operation == "remove_source":
        if not delta["semantic"]:
            raise AssertionError("remove_source did not update P10 semantic inventory")
        for key in numeric_keys:
            _assert_subset(delta[key], owner, f"remove {key}")
        if source_index not in delta["present"]:
            raise AssertionError("remove_source did not clear its presence track")
    elif operation == "add_source":
        if not delta["semantic"]:
            raise AssertionError("add_source did not update P10 semantic inventory")
        for key in numeric_keys:
            _assert_subset(delta[key], owner, f"add {key}")
        if delta["present"] != [source_index]:
            raise AssertionError("add_source did not activate exactly its new slot")
    elif operation == "replace_source":
        if not delta["semantic"]:
            raise AssertionError("replace_source did not update P10 semantics")
        if delta["present"]:
            raise AssertionError("replace_source changed source presence")
        for key in ("kind", "event", "trajectory"):
            _assert_subset(delta[key], owner, f"replace {key}")
        if not any(delta[key] for key in ("kind", "event", "trajectory")):
            raise AssertionError("replace_source changed no executable owner track")
    else:
        raise AssertionError(f"unhandled edit operation {operation!r}")


def _expect_rejected(label: str, callback: Callable[[], Any]) -> str:
    try:
        callback()
    except (ValueError, ModelScenePlanCodecError) as error:
        return f"{label}: {error}"
    raise AssertionError(f"negative gate {label!r} was accepted")


def _negative_gates(
    base_plan: Mapping[str, Any], codec: Any
) -> list[str]:
    sketch = compile_scene_sketch(base_plan, codec)
    execution = compile_execution_state(base_plan, codec)
    failures: list[str] = []

    nan_state = copy.deepcopy(execution)
    nan_state["core_features"][0][0] = float("nan")
    failures.append(
        _expect_rejected(
            "non_finite_execution",
            lambda: validate_execution_state(nan_state, codec=codec),
        )
    )

    mismatched = copy.deepcopy(execution)
    absent = next(
        (index for index, present in enumerate(mismatched["source_present_mask"]) if not present),
        None,
    )
    if absent is None:
        mismatched["source_present_mask"][3] = False
        mismatched["core_features"][4] = [0.0] * len(EXECUTION_FEATURE_NAMES)
    else:
        mismatched["source_present_mask"][absent] = True
        mismatched["core_features"][absent + 1] = copy.deepcopy(
            mismatched["core_features"][1]
        )
    failures.append(
        _expect_rejected(
            "inventory_mask_mismatch",
            lambda: assemble_sceneplan(sketch, mismatched, codec),
        )
    )

    polluted = copy.deepcopy(execution)
    polluted["semantic_text"] = "continuous state must never own this"
    failures.append(
        _expect_rejected(
            "semantic_field_in_execution",
            lambda: validate_execution_state(polluted, codec=codec),
        )
    )

    reordered = copy.deepcopy(sketch)
    reordered["sources"] = list(reversed(reordered["sources"]))
    if len(reordered["sources"]) > 1:
        failures.append(
            _expect_rejected(
                "unordered_sketch_sources",
                lambda: validate_scene_sketch(reordered, codec=codec),
            )
        )

    nonzero_gain = copy.deepcopy(dict(base_plan))
    nonzero_gain["sources"][0]["gain_db"] = 1.0
    failures.append(
        _expect_rejected(
            "nonzero_gain",
            lambda: compile_execution_state(nonzero_gain, codec),
        )
    )

    keyframed = copy.deepcopy(dict(base_plan))
    source = keyframed["sources"][0]
    positions = _positions(source)
    source["trajectory"] = {
        "type": "keyframed",
        "keyframes": [
            {
                "time_sec": source["activity"]["onset_sec"],
                "position": copy.deepcopy(positions[0]),
            },
            {
                "time_sec": source["activity"]["offset_sec"],
                "position": copy.deepcopy(positions[-1]),
            },
        ],
    }
    failures.append(
        _expect_rejected(
            "untrained_keyframed_motion",
            lambda: compile_execution_state(keyframed, codec),
        )
    )
    forbidden = {"room", "kind", "semantic", "transcript", "gain", "source_count"}
    for name in EXECUTION_FEATURE_NAMES:
        if any(fragment in name for fragment in forbidden):
            raise AssertionError(f"execution feature {name!r} owns a semantic field")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument(
        "--base-scenes",
        type=int,
        default=300,
        help="Held-out triplets to validate; 0 means all rows in the manifest.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.base_scenes < 0:
        raise ValueError("base-scenes must be non-negative")

    manifest = args.manifest.expanduser().resolve(strict=True)
    codec_path = args.codec.expanduser().resolve(strict=True)
    codec = load_model_sceneplan_codec(codec_path)
    sketch_codec = SceneSketchCodec(codec)
    patch_codec = ScenePlanEditPatchCodec(codec)
    delta_codec = AudioAwareDeltaSceneSketchCodec(codec, patch_codec)
    connection = _readonly(manifest)
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    total = int(metadata["base_samples"])
    selected = total if args.base_scenes == 0 else min(total, args.base_scenes)
    if selected <= 0:
        raise RuntimeError("P11-v4 validator selected no held-out scenes")

    edit_counts: Counter[str] = Counter()
    source_counts: Counter[int] = Counter()
    sketch_lengths: list[int] = []
    grammar_transitions = 0
    gu_locality = Counter()
    edit_locality = 0
    first_plan: dict[str, Any] | None = None
    for base_index in range(selected):
        rows = connection.execute(
            """
            SELECT ordinal,task,observed_sceneplan_zlib,target_sceneplan_zlib,
                   edit_kind,edit_spec_json
            FROM rows WHERE ordinal BETWEEN ? AND ? ORDER BY ordinal
            """,
            (base_index * 3, base_index * 3 + 2),
        ).fetchall()
        if len(rows) != 3 or [row[1] for row in rows] != [
            "generation",
            "understanding",
            "editing",
        ]:
            raise RuntimeError(f"held-out triplet {base_index} is incomplete")
        generation = _plan(rows[0][3])
        understanding = _plan(rows[1][3])
        if generation != understanding:
            raise AssertionError("G/U target ScenePlans diverged")
        if _plan(rows[1][2]) != understanding:
            raise AssertionError("U observed/target ScenePlans diverged")
        if first_plan is None:
            first_plan = generation
        source_counts[len(generation["sources"])] += 1
        for task, plan in (("generation", generation), ("understanding", understanding)):
            sketch, execution, p10 = _assert_exact_roundtrip(
                plan, codec, sketch_codec
            )
            tokens = sketch_codec.encode(sketch)["input_ids"]
            sketch_lengths.append(int(tokens.numel()))
            grammar_transitions += int(tokens.numel())
            _assert_numeric_locality(sketch, execution, p10, codec)
            _assert_semantic_isolation(sketch, execution, p10, codec)
            gu_locality[task] += 1

        edit_observed = _plan(rows[2][2])
        edit_target = _plan(rows[2][3])
        if edit_observed != understanding:
            raise AssertionError("E source FOA is not bound to the U-observed scene")
        spec = json.loads(str(rows[2][5]))
        operation = str(rows[2][4])
        if str(spec["operation"]) != operation:
            raise AssertionError("edit manifest operation/spec mismatch")
        target_sketch, target_execution, target_p10 = _assert_exact_roundtrip(
            edit_target, codec, sketch_codec
        )
        observed_execution = compile_execution_state(edit_observed, codec)
        delta_program = compile_audio_aware_delta_sketch(
            edit_observed, edit_target, spec, codec, patch_codec
        )
        delta_tokens = delta_codec.encode(delta_program)["input_ids"]
        if delta_codec.decode(delta_tokens) != delta_program:
            raise AssertionError("audio-aware DeltaSketch token round-trip changed")
        for prefix_length, expected in enumerate(delta_tokens.tolist()):
            allowed = delta_codec.allowed_next_ids(
                delta_tokens[:prefix_length],
                observed_sceneplan=edit_observed,
                max_text_tokens=192,
            )
            if int(expected) not in allowed:
                raise AssertionError("audio-aware DeltaSketch grammar rejected target")
        if delta_codec.allowed_next_ids(
            delta_tokens,
            observed_sceneplan=edit_observed,
            max_text_tokens=192,
        ):
            raise AssertionError("audio-aware DeltaSketch grammar did not terminate")
        grammar_transitions += int(delta_tokens.numel())
        projected = project_audio_aware_delta_to_atomic_patch(
            edit_observed,
            delta_program,
            execution_delta_core(observed_execution, target_execution),
            codec,
            patch_codec,
        )
        if projected["revised_sceneplan"] != codec.project_plan(edit_target):
            raise AssertionError("audio-aware delta projection changed revised plan")
        if not torch.equal(
            patch_codec.encode(projected["patch_spec"])["input_ids"],
            patch_codec.encode(spec)["input_ids"],
        ):
            raise AssertionError("derived patch differs from canonical edit program")
        baseline_p10 = compile_p10_from_contract(
            compile_scene_sketch(edit_observed, codec),
            observed_execution,
            codec,
            require_target_hash=True,
        )
        _assert_edit_locality(
            operation, projected["patch_spec"], baseline_p10, target_p10
        )
        edit_counts[operation] += 1
        edit_locality += 1

    if first_plan is None:
        raise RuntimeError("P11-v4 contract gate did not retain a reference plan")
    first_sketch = compile_scene_sketch(first_plan, codec)
    first_tokens = sketch_codec.encode(first_sketch)["input_ids"].tolist()
    text_begin = codec._tid("<text_begin>")
    text_end = codec._tid("<text_end>")
    begin_index = first_tokens.index(text_begin)
    first_piece = int(first_tokens[begin_index + 1])
    text_prefix = first_tokens[: begin_index + 1]
    before_ceiling = text_prefix + [first_piece] * 191
    at_ceiling = text_prefix + [first_piece] * 192
    if text_end not in sketch_codec.allowed_next_ids(
        before_ceiling, max_text_tokens=192
    ):
        raise AssertionError("SceneSketch text may not terminate before its ceiling")
    if sketch_codec.allowed_next_ids(
        at_ceiling, max_text_tokens=192
    ) != {text_end}:
        raise AssertionError("SceneSketch text ceiling did not force <text_end>")
    connection.close()

    required_operations = {
        "no_op",
        "add_source",
        "replace_source",
        "room_change",
        "move_source",
        "retime_source",
        "remove_source",
        "change_speech_description",
        "change_transcript",
    }
    if required_operations != set(AUDIO_AWARE_DELTA_OPERATIONS):
        raise AssertionError("validator operation inventory drifted from active codec")
    if set(edit_counts) != required_operations:
        raise AssertionError(
            f"held-out slice lacks edit families: {sorted(required_operations-set(edit_counts))}"
        )
    assert first_plan is not None
    negative_gates = _negative_gates(first_plan, codec)
    report = {
        "schema": "stable_audio_tools.p11_audio_aware_contract_validation",
        "schema_version": 1,
        "status": "PASS",
        "contract": "audio_observation_plus_atomic_delta_to_p10_v1",
        "manifest": str(manifest),
        "manifest_schema_version": int(metadata["schema_version"]),
        "codec": str(codec_path),
        "codec_fingerprint": codec.fingerprint,
        "scene_sketch_codec_fingerprint": sketch_codec.fingerprint,
        "p10": {
            "checkpoint": P10_CANONICAL_CHECKPOINT,
            "checkpoint_step": P10_CANONICAL_CHECKPOINT_STEP,
            "checkpoint_sha256": P10_CANONICAL_CHECKPOINT_SHA256,
            "boundary": "ScenePlan -> DiT -> FOA",
        },
        "coverage": {
            "base_scenes": selected,
            "generation": int(gu_locality["generation"]),
            "understanding": int(gu_locality["understanding"]),
            "editing": edit_locality,
            "edit_operations": dict(sorted(edit_counts.items())),
            "source_counts": {
                str(key): value for key, value in sorted(source_counts.items())
            },
            "grammar_transitions": grammar_transitions,
        },
        "scene_sketch_tokens": {
            "minimum": min(sketch_lengths),
            "maximum": max(sketch_lengths),
            "mean": float(np.mean(sketch_lengths)),
        },
        "hard_invariants": {
            "sceneplan_roundtrip": "100%",
            "scene_sketch_token_roundtrip": "100%",
            "scene_sketch_grammar": "100%",
            "numeric_intervention_semantic_immutability": "100%",
            "semantic_intervention_numeric_immutability": "100%",
            "source_local_p10_controls": "100%",
            "atomic_edit_reconstruction": "100%",
            "editing_source_foa_observation": "100%",
            "delta_sketch_semantics_only": "100%",
            "non_finite": "rejected",
            "untrained_keyframed_motion": "rejected",
            "nonzero_gain": "rejected",
        },
        "execution_features": list(EXECUTION_FEATURE_NAMES),
        "negative_gates": negative_gates,
        "decision": (
            "CPU contract is ready for the audio-aware pilot; this report does not "
            "authorize two-GPU medium-scale or full-corpus training."
        ),
        "report_sha256_without_self": None,
    }
    report["report_sha256_without_self"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_sha256_without_self"}
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
