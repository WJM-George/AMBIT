#!/usr/bin/env python3
"""Audit whether P11-v4 Flow diversity follows evidence uncertainty.

This is a P11-only analysis over an existing unified challenge report.  It
never renders P10 and never treats hidden exact targets as deployable posterior
selection.  Diversity is decomposed into the frozen P10 ExecutionState fields.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sqlite3
import sys
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.evaluate_sceneplan_p11_v4_challenge import (  # noqa: E402
    _json_sha256,
    _mean,
    _sha256_file,
)
from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.scene_sketch_v1 import (  # noqa: E402
    EXECUTION_FEATURE_NAMES,
    compile_execution_state,
    execution_state_core,
)


DEFAULT_CODEC = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
SCHEMA = "stable_audio_tools.p11_v4_posterior_specificity"
SCHEMA_VERSION = 1
K = 8

FEATURE_GROUPS = {
    "temporal": ("onset_frame_norm", "offset_frame_norm"),
    "motion": ("motion_static", "motion_linear"),
    "azimuth": (
        "start_sin_azimuth",
        "start_cos_azimuth",
        "end_sin_azimuth",
        "end_cos_azimuth",
    ),
    "elevation": (
        "start_sin_elevation",
        "start_cos_elevation",
        "end_sin_elevation",
        "end_cos_elevation",
    ),
    "distance": ("start_log_distance_norm", "end_log_distance_norm"),
}
FEATURE_GROUPS["geometry"] = tuple(
    name
    for group in ("motion", "azimuth", "elevation", "distance")
    for name in FEATURE_GROUPS[group]
)
FEATURE_GROUPS["all_source_numeric"] = tuple(
    name for name in EXECUTION_FEATURE_NAMES if name != "duration_frames_norm"
)


def _base_id(challenge_id: str) -> str:
    for marker in ("/g/", "/u/", "/e/"):
        if marker in challenge_id:
            return challenge_id.split(marker, 1)[0]
    raise ValueError(f"challenge id lacks task marker: {challenge_id}")


def _decode_target(payload: bytes) -> dict[str, Any]:
    value = json.loads(zlib.decompress(payload).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("challenge target ScenePlan is not an object")
    return value


def _challenge_targets(path: Path) -> dict[int, dict[str, Any]]:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False
    )
    try:
        rows = connection.execute(
            "SELECT ordinal,target_sceneplan_zlib,known_field_groups_json "
            "FROM rows ORDER BY ordinal"
        ).fetchall()
    finally:
        connection.close()
    return {
        int(ordinal): {
            "target_sceneplan": _decode_target(payload),
            "known_field_groups": json.loads(str(known)),
        }
        for ordinal, payload, known in rows
    }


def _source_slots(plan: Mapping[str, Any]) -> tuple[int, ...]:
    slots = []
    for source in plan["sources"]:
        source_id = str(source["source_id"])
        if not source_id.startswith("source_"):
            raise ValueError(f"non-canonical source id: {source_id}")
        slot = int(source_id.removeprefix("source_")) + 1
        if not 1 <= slot <= 4:
            raise ValueError(f"source slot outside P10 capacity: {source_id}")
        slots.append(slot)
    return tuple(sorted(slots))


def _group_mask(
    group: str,
    *,
    target_plan: Mapping[str, Any],
) -> np.ndarray:
    mask = np.zeros((5, len(EXECUTION_FEATURE_NAMES)), dtype=bool)
    if group == "duration":
        mask[0, EXECUTION_FEATURE_NAMES.index("duration_frames_norm")] = True
        return mask
    names = FEATURE_GROUPS[group]
    indices = [EXECUTION_FEATURE_NAMES.index(name) for name in names]
    for slot in _source_slots(target_plan):
        mask[slot, indices] = True
    return mask


def _pairwise_rmse(values: np.ndarray, mask: np.ndarray) -> float:
    if values.ndim != 3 or values.shape[1:] != mask.shape:
        raise ValueError("posterior cores do not align with field mask")
    if not bool(mask.any()):
        return 0.0
    distances = [
        float(np.sqrt(np.mean(np.square(values[left][mask] - values[right][mask]))))
        for left, right in itertools.combinations(range(values.shape[0]), 2)
    ]
    return float(np.mean(distances)) if distances else 0.0


def _semantic_exact(score: Mapping[str, Any]) -> bool:
    value = (score.get("task_metrics") or {}).get("semantic_exact")
    return value is not None and math.isclose(float(value), 1.0, abs_tol=1.0e-12)


def _row_specificity(
    row: Mapping[str, Any],
    *,
    target: Mapping[str, Any],
    codec: Any,
) -> dict[str, Any]:
    scored = list(row["scored"][:K])
    if len(scored) != K or not all(value.get("valid") is True for value in scored):
        raise ValueError("specificity requires eight valid posterior draws")
    cores = np.stack(
        [
            execution_state_core(
                compile_execution_state(value["sceneplan"], codec)
            )
            for value in scored
        ]
    )
    groups = {
        group: _pairwise_rmse(
            cores,
            _group_mask(group, target_plan=target["target_sceneplan"]),
        )
        for group in ("duration", *FEATURE_GROUPS)
    }
    prefix = row["prefixes"][str(K)]
    return {
        "ordinal": int(row["ordinal"]),
        "challenge_id": str(row["challenge_id"]),
        "base_id": _base_id(str(row["challenge_id"])),
        "task": str(row["task"]),
        "family": str(row["family"]),
        "view_id": str(row["view_id"]),
        "source_count": len(target["target_sceneplan"]["sources"]),
        "known_field_groups": list(target["known_field_groups"]),
        "semantic_exact": _semantic_exact(scored[0]),
        "semantic_sha256": str(scored[0]["semantic_sha256"]),
        "semantic_immutable": bool(prefix["semantic_immutability"]),
        "task_score_mean": float(prefix["task_score_mean"]),
        "oracle_best_of_k_task_score": float(
            prefix["oracle_best_of_k_task_score"]
        ),
        "oracle_lift": float(prefix["oracle_best_of_k_task_score"])
        - float(prefix["task_score_mean"]),
        "field_pairwise_rmse": groups,
    }


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["semantic_exact"]]
    groups = ("duration", *FEATURE_GROUPS)
    return {
        "rows": len(rows),
        "semantic_exact_rows": len(eligible),
        "semantic_exact_rate": _mean(
            [float(bool(row["semantic_exact"])) for row in rows]
        ),
        "semantic_immutability_rate": _mean(
            [float(bool(row["semantic_immutable"])) for row in rows]
        ),
        "task_score_mean": _mean([float(row["task_score_mean"]) for row in rows]),
        "oracle_lift": _mean([float(row["oracle_lift"]) for row in rows]),
        "field_pairwise_rmse_all_valid": {
            group: _mean(
                [float(row["field_pairwise_rmse"][group]) for row in rows]
            )
            for group in groups
        },
        "field_pairwise_rmse_target_semantic_exact": {
            group: _mean(
                [float(row["field_pairwise_rmse"][group]) for row in eligible]
            )
            for group in groups
        },
    }


def _paired_response(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_view: str,
    changed_view: str,
    semantic_pairing: str,
) -> dict[str, Any]:
    if semantic_pairing not in {"target_exact", "same_decoded"}:
        raise ValueError("unknown semantic pairing rule")
    by_key = {(row["base_id"], row["view_id"]): row for row in rows}
    pairs = []
    candidate_pairs = 0
    semantic_rejections = 0
    for (base_id, view), reference in by_key.items():
        if view != reference_view:
            continue
        changed = by_key.get((base_id, changed_view))
        if changed is None:
            continue
        candidate_pairs += 1
        semantic_match = (
            reference["semantic_exact"] and changed["semantic_exact"]
            if semantic_pairing == "target_exact"
            else reference["semantic_sha256"] == changed["semantic_sha256"]
        )
        if not semantic_match:
            semantic_rejections += 1
            continue
        delta = {
            group: float(changed["field_pairwise_rmse"][group])
            - float(reference["field_pairwise_rmse"][group])
            for group in ("duration", *FEATURE_GROUPS)
        }
        pairs.append({"base_id": base_id, "delta": delta})
    return {
        "reference_view": reference_view,
        "changed_view": changed_view,
        "semantic_pairing": semantic_pairing,
        "candidate_pairs": candidate_pairs,
        "semantic_matched_pairs": len(pairs),
        "semantic_rejections": semantic_rejections,
        "mean_spread_delta": {
            group: _mean([pair["delta"][group] for pair in pairs])
            for group in ("duration", *FEATURE_GROUPS)
        },
        "spread_increase_rate": {
            group: _mean(
                [float(pair["delta"][group] > 0.0) for pair in pairs]
            )
            for group in ("duration", *FEATURE_GROUPS)
        },
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--challenge", type=Path)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report_path = args.report.expanduser().resolve(strict=True)
    source = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        source.get("schema")
        != "stable_audio_tools.p11_v4_unified_challenge_eval"
        or int(source.get("schema_version", -1)) != 10
        or source.get("status") != "PASS"
        or source.get("arm") != "flow"
        or int(source.get("draws", 0)) < K
    ):
        raise ValueError("specificity requires a passing Flow evaluator-v10 K8 report")
    challenge_path = (
        Path(str(source["challenge"]))
        if args.challenge is None
        else args.challenge
    ).expanduser().resolve(strict=True)
    if _sha256_file(challenge_path) != source["challenge_sha256"]:
        raise RuntimeError("challenge changed since source evaluation")
    codec_path = args.codec.expanduser().resolve(strict=True)
    codec = load_model_sceneplan_codec(codec_path)
    targets = _challenge_targets(challenge_path)
    rows = [
        _row_specificity(
            row,
            target=targets[int(row["ordinal"])],
            codec=codec,
        )
        for row in source["aggregate_rows"]
        if str(row["task"]) in {"generation", "understanding"}
    ]
    by_view: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_view[str(row["view_id"])].append(row)

    generation_layout = _paired_response(
        rows,
        reference_view="exact_v1",
        changed_view="numeric_layout_v1",
        semantic_pairing="target_exact",
    )
    generation_coarse = _paired_response(
        rows,
        reference_view="exact_v1",
        changed_view="coarse_numeric_v1",
        semantic_pairing="target_exact",
    )
    understanding = {
        view: _paired_response(
            rows,
            reference_view="audio_evidence_exact_v1",
            changed_view=view,
            semantic_pairing="same_decoded",
        )
        for view in (
            "foa_time_mask_12p5_v1",
            "clap_channel_mask_25p0_v1",
            "combined_foa25_clap50_v1",
        )
    }
    output = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "scope": (
            "P11 Flow posterior specificity in frozen P10 ExecutionState space; "
            "no P10 rendering and no oracle-based deployment selection"
        ),
        "source_report": {
            "path": str(report_path),
            "file_sha256": _sha256_file(report_path),
            "report_sha256_without_self": source["report_sha256_without_self"],
        },
        "challenge": {
            "path": str(challenge_path),
            "sha256": source["challenge_sha256"],
        },
        "codec": {
            "path": str(codec_path),
            "fingerprint": codec.fingerprint,
        },
        "posterior_draws": K,
        "interpretation_contract": {
            "semantic_exact_filter_for_cross_view_field_comparison": True,
            "understanding_cross_view_semantic_filter": (
                "same decoded SceneSketch hash; exact target wording is not required"
            ),
            "positive_spread_delta_means_more_continuous_diversity": True,
            "desired_generation_numeric_layout_response": (
                "geometry spread increases more than temporal spread"
            ),
            "desired_understanding_degradation_response": (
                "continuous spread increases when usable FOA evidence is degraded"
            ),
            "thresholds_frozen_before_observation": False,
            "therefore_this_report_is_diagnostic_not_a_promotion_gate": True,
        },
        "by_view": {
            view: _group_summary(values) for view, values in sorted(by_view.items())
        },
        "paired_response": {
            "generation_numeric_layout_vs_exact": generation_layout,
            "generation_coarse_vs_exact": generation_coarse,
            "understanding_degraded_vs_exact": understanding,
        },
        "rows": rows,
    }
    output["report_sha256_without_self"] = _json_sha256(output)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact_paired = {
        "generation_numeric_layout_vs_exact": {
            key: value
            for key, value in generation_layout.items()
            if key != "pairs"
        },
        "generation_coarse_vs_exact": {
            key: value
            for key, value in generation_coarse.items()
            if key != "pairs"
        },
        "understanding_degraded_vs_exact": {
            view: {key: value for key, value in value.items() if key != "pairs"}
            for view, value in understanding.items()
        },
    }
    print(
        json.dumps(
            {
                "status": output["status"],
                "output": str(output_path),
                "by_view": output["by_view"],
                "paired_response": compact_paired,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
