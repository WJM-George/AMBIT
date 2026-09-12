#!/usr/bin/env python3
"""Pure validation and comparison helpers for Spatial-CoT text gates."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


PANEL_SCHEMA = "stable_audio_tools.spatial_cot_text_panel"
RESULT_SCHEMA = "stable_audio_tools.spatial_cot_text_evaluation"
COMPARISON_SCHEMA = "stable_audio_tools.spatial_cot_text_comparison"
EVALUATOR_VERSION = 2
COMPARISON_VERSION = 2


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_text_panel(
    path: Path, *, latent_root: Path | None = None
) -> tuple[dict[str, Any], tuple[int, ...]]:
    path = path.expanduser().resolve()
    panel = read_json(path)
    if panel.get("schema") != PANEL_SCHEMA or panel.get("schema_version") != 1:
        raise ValueError(f"unsupported Spatial-CoT text panel: {path}")
    ranks = panel.get("family_ranks")
    if (
        not isinstance(ranks, list)
        or not ranks
        or any(
            isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
            for rank in ranks
        )
        or len(ranks) != len(set(ranks))
    ):
        raise ValueError("text panel family_ranks must be unique non-negative integers")
    declared_root = panel.get("latent_root")
    if not isinstance(declared_root, str):
        raise ValueError("text panel latent_root must be a path string")
    if latent_root is not None and (
        Path(declared_root).expanduser().resolve()
        != latent_root.expanduser().resolve()
    ):
        raise ValueError("text panel latent_root does not match evaluated latent store")
    families = panel.get("families")
    if not isinstance(families, list):
        raise ValueError("text panel must contain immutable family ids")
    described = {
        int(item["family_rank"]): str(item["family_id"])
        for item in families
        if isinstance(item, dict)
        and isinstance(item.get("family_rank"), int)
        and not isinstance(item.get("family_rank"), bool)
        and isinstance(item.get("family_id"), str)
    }
    if set(described) != set(ranks):
        raise ValueError("text panel families do not match family_ranks")
    turns = panel.get("turns")
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
        raise ValueError("text panel turns must be a positive integer")
    shift = panel.get("counterfactual_shift")
    if (
        isinstance(shift, bool)
        or not isinstance(shift, int)
        or shift < 1
        or shift >= len(ranks)
    ):
        raise ValueError("counterfactual_shift must be in [1, family_count)")
    free_cases = panel.get("free_decode_cases")
    if not isinstance(free_cases, list) or not free_cases:
        raise ValueError("text panel must contain free_decode_cases")
    seen_cases: set[tuple[int, int]] = set()
    for item in free_cases:
        if not isinstance(item, dict):
            raise ValueError("free_decode_cases entries must be objects")
        case = (item.get("family_rank"), item.get("turn"))
        if (
            isinstance(case[0], bool)
            or not isinstance(case[0], int)
            or case[0] not in ranks
            or isinstance(case[1], bool)
            or not isinstance(case[1], int)
            or not 0 <= case[1] < turns
            or case in seen_cases
        ):
            raise ValueError(f"invalid or duplicate free decode case: {item}")
        seen_cases.add(case)
    return panel, tuple(ranks)


def verify_panel_families(
    panel: dict[str, Any], observed: Sequence[dict[str, Any]]
) -> None:
    expected = {
        int(item["family_rank"]): str(item["family_id"])
        for item in panel["families"]
    }
    actual = {
        int(item["family_rank"]): str(item["family_id"])
        for item in observed
    }
    if expected != actual:
        mismatches = {
            rank: {"expected": family_id, "observed": actual.get(rank)}
            for rank, family_id in expected.items()
            if actual.get(rank) != family_id
        }
        raise RuntimeError(f"text panel family ids changed: {mismatches}")


def rotate(values: Sequence[Any], shift: int) -> list[Any]:
    if not values or not 0 < shift < len(values):
        raise ValueError("rotation requires 0 < shift < len(values)")
    return [values[(index + shift) % len(values)] for index in range(len(values))]


def _finite_metric(mapping: dict[str, Any], key: str) -> float:
    value = float(mapping[key])
    if not math.isfinite(value):
        raise ValueError(f"metric {key!r} is not finite")
    return value


def _validate_pair(
    baseline: dict[str, Any], candidate: dict[str, Any], *, mode: str
) -> None:
    for label, result in (("baseline", baseline), ("candidate", candidate)):
        if result.get("schema") != RESULT_SCHEMA or result.get("schema_version") != 1:
            raise ValueError(f"{label} is not a Spatial-CoT text result")
        if result.get("evaluator_version") != EVALUATOR_VERSION:
            raise ValueError(f"{label} uses an obsolete text evaluator")
        if result.get("mode") != mode:
            raise ValueError(f"{label} mode is not {mode!r}")
        if result.get("status") != "DIAGNOSTIC":
            raise ValueError(f"{label} evaluation is incomplete")
    identity_keys = (
        "evaluator_version",
        "panel_sha256",
        "panel_name",
        "family_ranks",
    )
    for key in identity_keys:
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"baseline/candidate {key} mismatch")
    if int(baseline.get("case_count", -1)) != int(candidate.get("case_count", -2)):
        raise ValueError("baseline/candidate case count mismatch")


def _check(
    checks: list[dict[str, Any]],
    failures: list[str],
    *,
    name: str,
    value: float,
    operator: str,
    threshold: float,
) -> None:
    passed = value <= threshold if operator == "<=" else value >= threshold
    checks.append(
        {
            "name": name,
            "value": value,
            "operator": operator,
            "threshold": threshold,
            "passed": passed,
        }
    )
    if not passed:
        failures.append(f"{name}: {value:.8g} {operator} {threshold:.8g} failed")


def compare_teacher_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    mode = "teacher_forced_counterfactual"
    _validate_pair(baseline, candidate, mode=mode)
    max_ce_ratio = float(thresholds["max_aligned_ce_ratio"])
    ce_tolerance = float(thresholds["aligned_ce_absolute_tolerance"])
    max_accuracy_drop = float(thresholds["max_token_accuracy_drop"])
    max_ambiguous_accuracy_drop = float(
        thresholds["max_ambiguous_token_accuracy_drop"]
    )
    max_exact_drop = float(thresholds["max_exact_fraction_drop"])
    gap_retention = float(thresholds["min_counterfactual_gap_retention"])
    relative_gap_tolerance = float(thresholds["counterfactual_gap_tolerance"])
    min_gap = float(thresholds["min_counterfactual_gap"])
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    variants = {
        "state_planner": ("instruction", "state", "full"),
        "understanding": ("audio",),
    }
    for objective, counterfactual_variants in variants.items():
        base = baseline["metrics"][objective]
        cand = candidate["metrics"][objective]
        base_ce = _finite_metric(base, "aligned_ce")
        cand_ce = _finite_metric(cand, "aligned_ce")
        _check(
            checks,
            failures,
            name=f"{objective}.aligned_ce",
            value=cand_ce,
            operator="<=",
            threshold=base_ce * max_ce_ratio + ce_tolerance,
        )
        _check(
            checks,
            failures,
            name=f"{objective}.token_accuracy",
            value=_finite_metric(cand, "token_accuracy"),
            operator=">=",
            threshold=max(0.0, _finite_metric(base, "token_accuracy") - max_accuracy_drop),
        )
        _check(
            checks,
            failures,
            name=f"{objective}.ambiguous_token_accuracy",
            value=_finite_metric(cand, "ambiguous_token_accuracy"),
            operator=">=",
            threshold=max(
                0.0,
                _finite_metric(base, "ambiguous_token_accuracy")
                - max_ambiguous_accuracy_drop,
            ),
        )
        _check(
            checks,
            failures,
            name=f"{objective}.exact_fraction",
            value=_finite_metric(cand, "exact_fraction"),
            operator=">=",
            threshold=max(0.0, _finite_metric(base, "exact_fraction") - max_exact_drop),
        )
        for variant in counterfactual_variants:
            base_counterfactual = base["counterfactual"][variant]
            cand_counterfactual = cand["counterfactual"][variant]
            base_gap = _finite_metric(base_counterfactual, "ce_gap")
            cand_gap = _finite_metric(cand_counterfactual, "ce_gap")
            base_reference = _finite_metric(
                base_counterfactual, "aligned_reference_ce"
            )
            cand_reference = _finite_metric(
                cand_counterfactual, "aligned_reference_ce"
            )
            if base_reference <= 0.0 or cand_reference <= 0.0:
                raise ValueError("counterfactual aligned_reference_ce must be positive")
            base_relative_gap = base_gap / base_reference
            cand_relative_gap = cand_gap / cand_reference
            _check(
                checks,
                failures,
                name=f"{objective}.{variant}_counterfactual_ce_gap",
                value=cand_gap,
                operator=">=",
                threshold=min_gap,
            )
            _check(
                checks,
                failures,
                name=(
                    f"{objective}.{variant}_counterfactual_relative_ce_gap"
                ),
                value=cand_relative_gap,
                operator=">=",
                threshold=max(
                    0.0,
                    base_relative_gap * gap_retention
                    - relative_gap_tolerance,
                ),
            )
    return _comparison_report(
        mode=mode,
        baseline=baseline,
        candidate=candidate,
        checks=checks,
        failures=failures,
    )


def compare_free_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    mode = "free_decode"
    _validate_pair(baseline, candidate, mode=mode)
    checks: list[dict[str, Any]] = []
    failures: list[str] = []
    metric_drops = {
        "token_accuracy": float(thresholds["max_token_accuracy_drop"]),
        "source_component_accuracy": float(
            thresholds["max_source_component_accuracy_drop"]
        ),
        "decoded_exact_fraction": float(
            thresholds["max_decoded_exact_fraction_drop"]
        ),
    }
    max_failure_count = float(thresholds["max_failure_count"])
    for objective in ("state_planner", "understanding"):
        base = baseline["metrics"][objective]
        cand = candidate["metrics"][objective]
        for metric, allowed_drop in metric_drops.items():
            _check(
                checks,
                failures,
                name=f"{objective}.{metric}",
                value=_finite_metric(cand, metric),
                operator=">=",
                threshold=max(0.0, _finite_metric(base, metric) - allowed_drop),
            )
        _check(
            checks,
            failures,
            name=f"{objective}.failure_count",
            value=float(cand["failure_count"]),
            operator="<=",
            threshold=float(base["failure_count"]),
        )
        _check(
            checks,
            failures,
            name=f"{objective}.absolute_failure_count",
            value=float(cand["failure_count"]),
            operator="<=",
            threshold=max_failure_count,
        )
        if objective == "state_planner":
            max_edit_drop = float(
                thresholds.get("max_edit_field_accuracy_drop", 0.0)
            )
            min_edit_accuracy = float(
                thresholds.get("min_edit_field_accuracy", 0.0)
            )
            min_edit_exact = float(
                thresholds.get("min_edit_exact_fraction", 0.0)
            )
            _check(
                checks,
                failures,
                name="state_planner.edit_field_accuracy",
                value=_finite_metric(cand, "edit_field_accuracy"),
                operator=">=",
                threshold=max(
                    min_edit_accuracy,
                    _finite_metric(base, "edit_field_accuracy") - max_edit_drop,
                ),
            )
            _check(
                checks,
                failures,
                name="state_planner.edit_exact_fraction",
                value=_finite_metric(cand, "edit_exact_fraction"),
                operator=">=",
                threshold=min_edit_exact,
            )
    return _comparison_report(
        mode=mode,
        baseline=baseline,
        candidate=candidate,
        checks=checks,
        failures=failures,
    )


def _comparison_report(
    *,
    mode: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    checks: list[dict[str, Any]],
    failures: list[str],
) -> dict[str, Any]:
    return {
        "schema": COMPARISON_SCHEMA,
        "schema_version": 1,
        "evaluator_version": candidate["evaluator_version"],
        "comparison_version": COMPARISON_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "mode": mode,
        "panel_name": candidate["panel_name"],
        "panel_sha256": candidate["panel_sha256"],
        "baseline_checkpoint": baseline["checkpoint"],
        "candidate_checkpoint": candidate["checkpoint"],
        "checks": checks,
        "failures": failures,
    }
