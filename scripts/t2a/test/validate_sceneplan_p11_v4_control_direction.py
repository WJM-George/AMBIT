#!/usr/bin/env python3
"""CPU/data gate for P10-aligned DeltaSketch control-direction forcing."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config, validate_training_configs
from stable_audio_tools.data.model_sceneplan_codec import load_model_sceneplan_codec
from stable_audio_tools.data.scene_sketch_v1 import (
    CONTROL_DIRECTION_CONTRACT,
    CONTROL_DIRECTION_OPERATIONS,
    DELTA_SCENE_SKETCH_CONTRACT,
    DELTA_SCENE_SKETCH_SCHEMA,
    DELTA_SCENE_SKETCH_VERSION,
    DELTA_SKETCH_TOKEN_CONTRACT,
    DeltaSceneSketchCodec,
    control_direction_from_patch,
)


DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
)
DEFAULT_DATASET_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_pilot90_curriculum_pair_aware_transfusion_cot_v4_reliable_asr_v1.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/p11_v4_control_direction_cpu_gate_20260902.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _delta_audit(edit_spec: dict[str, Any]) -> dict[str, Any]:
    operation = str(edit_spec["operation"])
    source_id = str(edit_spec["source_id"])
    return {
        "schema": DELTA_SCENE_SKETCH_SCHEMA,
        "schema_version": DELTA_SCENE_SKETCH_VERSION,
        "contract": DELTA_SCENE_SKETCH_CONTRACT,
        "operation": operation,
        "changed_source_ids": [source_id],
        "changed_field_groups": [
            "geometry" if operation == "rotate_source" else "distance"
        ],
        "added_source_ids": [],
        "removed_source_ids": [],
    }


def _curriculum_audit(
    path: Path, codec: DeltaSceneSketchCodec, *, batch_size: int
) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    rows = connection.execute(
        "SELECT ordinal,task,pair_id,pair_label,edit_spec_json "
        "FROM rows ORDER BY ordinal"
    ).fetchall()
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    connection.close()
    full_rows = len(rows) // batch_size * batch_size
    direction_rows = 0
    token_round_trips = 0
    operation_counts: dict[str, int] = defaultdict(int)
    pair_directions: dict[str, set[int]] = defaultdict(set)
    token_ids: dict[str, dict[str, int]] = defaultdict(dict)
    for _, task, pair_id, pair_label, edit_json in rows[:full_rows]:
        if task != "editing" or edit_json is None:
            continue
        edit_spec = json.loads(str(edit_json))
        operation = str(edit_spec.get("operation") or "")
        if operation not in CONTROL_DIRECTION_OPERATIONS:
            continue
        direction = control_direction_from_patch(edit_spec)
        encoded = codec.encode(_delta_audit(edit_spec), edit_spec)["input_ids"]
        decoded = codec.decode(encoded)
        if int(encoded.numel()) != 5:
            raise RuntimeError("binary P10 DeltaSketch must contain exactly five tokens")
        if (
            decoded["contract"] != DELTA_SKETCH_TOKEN_CONTRACT
            or decoded["operation"] != operation
            or decoded["source_id"] != str(edit_spec["source_id"])
            or int(decoded["control_direction"]) != direction
        ):
            raise RuntimeError("DeltaSketch direction token failed round-trip")
        direction_rows += 1
        token_round_trips += 1
        operation_counts[operation] += 1
        token_ids[operation][str(direction)] = int(encoded[3])
        if pair_id is not None:
            pair_directions[str(pair_id)].add(direction)
    complete_pairs = sum(values == {-1, 1} for values in pair_directions.values())
    if direction_rows <= 0 or token_round_trips != direction_rows:
        raise RuntimeError("curriculum has no valid control-direction rows")
    population_balanced = all(
        set(token_ids[operation]) == {"-1", "1"}
        for operation in CONTROL_DIRECTION_OPERATIONS
    )
    # The small pilot deliberately places exact counterfactual pairs in one
    # batch.  The 10k population screen instead has one matched E target per
    # scene and balances both directions across the frozen population.
    if complete_pairs <= 0 and metadata.get("contract") != (
        "p10_v11_matched_10k_gue_screening_v1"
    ):
        raise RuntimeError("curriculum has no negative/positive control pair")
    if not population_balanced:
        raise RuntimeError("curriculum lacks population coverage of both directions")
    for operation in CONTROL_DIRECTION_OPERATIONS:
        if operation_counts[operation] <= 0:
            raise RuntimeError(f"curriculum lacks {operation} direction rows")
        ids = token_ids[operation]
        if set(ids) != {"-1", "1"} or ids["-1"] == ids["1"]:
            raise RuntimeError(f"{operation} direction tokens are not distinct")
    all_ids = [value for values in token_ids.values() for value in values.values()]
    if len(set(all_ids)) != 4:
        raise RuntimeError("rotate/distance must use four operation-specific tokens")
    return {
        "rows": len(rows),
        "batch_size": batch_size,
        "full_rows": full_rows,
        "dropped_tail_rows": len(rows) - full_rows,
        "direction_rows": direction_rows,
        "round_trip_rows": token_round_trips,
        "operation_counts": dict(sorted(operation_counts.items())),
        "complete_pair_ids": complete_pairs,
        "population_has_both_directions_per_operation": population_balanced,
        "operation_specific_token_ids": {
            key: dict(sorted(value.items())) for key, value in sorted(token_ids.items())
        },
        "ordering_contract": metadata.get("ordering_contract"),
        "ordering_batch_size": metadata.get("ordering_batch_size"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size != 8:
        raise ValueError("P11 control-direction gate is frozen at batch size 8")
    model_path = args.model_config.expanduser().resolve(strict=True)
    dataset_path = args.dataset_config.expanduser().resolve(strict=True)
    model_config = load_config(model_path)
    dataset_config = load_config(dataset_path)
    validate_training_configs(model_config, dataset_config)
    transfusion = model_config["model"]["transfusion_cot"]
    control = transfusion["control_direction"]
    if control["contract"] != CONTROL_DIRECTION_CONTRACT:
        raise RuntimeError("candidate config lacks canonical control-direction forcing")
    if "executable_axis" in transfusion["thought"]:
        raise RuntimeError("retired executable-axis head remains configured")
    plan_codec = load_model_sceneplan_codec(
        model_config["model"]["text"]["plan_codec_path"]
    )
    delta_codec = DeltaSceneSketchCodec(plan_codec)
    curriculum = Path(dataset_config["p11_v4_curriculum_path"]).resolve(strict=True)
    coverage = _curriculum_audit(
        curriculum, delta_codec, batch_size=args.batch_size
    )
    report = {
        "schema": "stable_audio_tools.p11_v4_control_direction_gate",
        "schema_version": 1,
        "status": "PASS",
        "model_config": str(model_path),
        "model_config_sha256": _sha256_file(model_path),
        "dataset_config": str(dataset_path),
        "dataset_config_sha256": _sha256_file(dataset_path),
        "curriculum": str(curriculum),
        "curriculum_sha256": _sha256_file(curriculum),
        "delta_token_contract": DELTA_SKETCH_TOKEN_CONTRACT,
        "control_direction_contract": CONTROL_DIRECTION_CONTRACT,
        "continuous_parallel_head": False,
        "coverage": coverage,
        "decision": (
            "CPU/data contract is ready for one single-GPU batch-8 falsification; "
            "this does not authorize held-out, 8-GPU, or full-corpus training."
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
