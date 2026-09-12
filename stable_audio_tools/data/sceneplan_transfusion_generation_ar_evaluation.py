"""Auditable metrics for raw-request to ScenePlan Generation AR evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

from .sceneplan_p11_metrics import score_sceneplan_fields


GENERATION_AR_EVALUATION_CONTRACT = (
    "p10v11_shared_generation_ar_8k_evaluation_v2"
)

TOKEN_SEQUENCE_METRIC_NAMES = frozenset(
    {
        "token_sequence_exact",
        "token_aligned_target_accuracy",
        "token_aligned_symmetric_accuracy",
        "token_common_prefix_fraction",
        "token_count_abs_error",
        "grammar_legal",
    }
)

_ORDERED_SOURCE_METRIC_NAMES = (
    "kind_exact",
    "description_exact",
    "activity_onset_exact",
    "activity_offset_exact",
    "trajectory_type_exact",
    "gain_exact",
    "start_azimuth_mae_deg",
    "start_elevation_mae_deg",
    "start_distance_mae_m",
    "start_position_exact",
    "end_azimuth_mae_deg",
    "end_elevation_mae_deg",
    "end_distance_mae_m",
    "end_position_exact",
)

_TOKEN_FAILURE_PENALIZED_METRIC_NAMES = frozenset(
    TOKEN_SEQUENCE_METRIC_NAMES - {"token_count_abs_error"}
)
_SCENEPLAN_FIELD_BOUNDED_METRIC_NAMES = frozenset(
    {
        "source_count_accuracy",
        "source_id_f1",
        "source_assignment_f1",
        "room_accuracy",
        "duration_accuracy",
        "kind_accuracy",
        "semantic_exact",
        "semantic_token_f1",
        "transcript_exact",
        "transcript_word_accuracy",
        "transcript_char_accuracy",
        "activity_exact",
        "activity_iou",
        "motion_type_accuracy",
        "position_bin_exact",
        "spatial_soft_score",
        "scene_score",
        "persistent_scene_exact",
        "permutation_scene_exact",
        "scene_exact",
    }
)
_PARSED_FAILURE_PENALIZED_METRIC_NAMES = frozenset(
    {
        f"{matching}_{name}"
        for matching in ("persistent", "permutation")
        for name in _SCENEPLAN_FIELD_BOUNDED_METRIC_NAMES
    }
    | {
        f"ordered_{name}"
        for name in _ORDERED_SOURCE_METRIC_NAMES
        if name.endswith("_exact")
    }
    | {"ordered_transcript_exact", "ordered_source_count_exact"}
)


def score_token_sequence(
    target: Sequence[int], prediction: Sequence[int], codec
) -> dict[str, float]:
    """Score free-decoded ids and independently replay the codec grammar."""

    reference = [int(value) for value in target]
    hypothesis = [int(value) for value in prediction]
    if not reference:
        raise ValueError("Generation AR target token sequence cannot be empty")
    aligned = sum(left == right for left, right in zip(reference, hypothesis))
    common_prefix = 0
    for left, right in zip(reference, hypothesis):
        if left != right:
            break
        common_prefix += 1

    legal = bool(hypothesis and hypothesis[0] == int(codec.bos_id))
    prefix = hypothesis[:1]
    if legal:
        for token_id in hypothesis[1:]:
            allowed = codec.allowed_next_ids(prefix)
            if int(token_id) not in allowed:
                legal = False
                break
            prefix.append(int(token_id))
    legal = legal and hypothesis[-1] == int(codec.eos_id)
    return {
        "token_sequence_exact": float(reference == hypothesis),
        "token_aligned_target_accuracy": aligned / len(reference),
        "token_aligned_symmetric_accuracy": aligned
        / max(len(reference), len(hypothesis), 1),
        "token_common_prefix_fraction": common_prefix / len(reference),
        "token_count_abs_error": float(abs(len(reference) - len(hypothesis))),
        "grammar_legal": float(legal),
    }


def score_parsed_generation(
    target: Mapping[str, Any], prediction: Mapping[str, Any]
) -> dict[str, float]:
    """Expose persistent-order and id-free assignment metrics side by side."""

    persistent = score_sceneplan_fields(
        target, prediction, source_matching="persistent_id"
    )
    permutation = score_sceneplan_fields(
        target, prediction, source_matching="permutation_invariant"
    )
    return {
        **{f"persistent_{key}": float(value) for key, value in persistent.items()},
        **{f"permutation_{key}": float(value) for key, value in permutation.items()},
        **_ordered_field_metrics(target, prediction),
    }


def _endpoints(source: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    trajectory = source["trajectory"]
    motion = str(trajectory["type"])
    if motion == "static":
        return trajectory["position"], trajectory["position"]
    if motion == "linear":
        return trajectory["start"], trajectory["end"]
    keyframes = trajectory["keyframes"]
    return keyframes[0]["position"], keyframes[-1]["position"]


def _azimuth_error(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _ordered_field_metrics(
    target: Mapping[str, Any], prediction: Mapping[str, Any]
) -> dict[str, float]:
    """Score every requested ordered field, including separate path endpoints."""

    references = list(target["sources"])
    hypotheses = {
        str(source["source_id"]): source for source in prediction["sources"]
    }
    denominator = max(1, len(references))
    # A fixed schema is important for macro aggregation.  If every predicted
    # source uses the wrong persistent slot, no source pair is visited below;
    # absent exact keys would otherwise make that catastrophic row disappear
    # from per-metric means instead of contributing zeros.
    sums = {name: 0.0 for name in _ORDERED_SOURCE_METRIC_NAMES}
    transcript_exact_sum = 0.0
    speech_count = sum(str(source["kind"]) == "speech" for source in references)

    for reference in references:
        hypothesis = hypotheses.get(str(reference["source_id"]))
        if hypothesis is None:
            sums["start_azimuth_mae_deg"] += 180.0
            sums["end_azimuth_mae_deg"] += 180.0
            sums["start_elevation_mae_deg"] += 90.0
            sums["end_elevation_mae_deg"] += 90.0
            sums["start_distance_mae_m"] += 50.0
            sums["end_distance_mae_m"] += 50.0
            continue
        ref_semantic = str(
            reference.get("speaker_description")
            or reference.get("description")
            or ""
        )
        hyp_semantic = str(
            hypothesis.get("speaker_description")
            or hypothesis.get("description")
            or ""
        )
        sums["kind_exact"] += float(reference["kind"] == hypothesis["kind"])
        sums["description_exact"] += float(ref_semantic == hyp_semantic)
        sums["activity_onset_exact"] += float(
            reference["activity"]["onset_sec"]
            == hypothesis["activity"]["onset_sec"]
        )
        sums["activity_offset_exact"] += float(
            reference["activity"]["offset_sec"]
            == hypothesis["activity"]["offset_sec"]
        )
        sums["trajectory_type_exact"] += float(
            reference["trajectory"]["type"]
            == hypothesis["trajectory"]["type"]
        )
        sums["gain_exact"] += float(
            float(reference.get("gain_db", 0.0))
            == float(hypothesis.get("gain_db", 0.0))
        )
        if str(reference["kind"]) == "speech":
            transcript_exact_sum += float(
                reference.get("transcript") == hypothesis.get("transcript")
            )

        ref_start, ref_end = _endpoints(reference)
        hyp_start, hyp_end = _endpoints(hypothesis)
        for label, left, right in (
            ("start", ref_start, hyp_start),
            ("end", ref_end, hyp_end),
        ):
            azimuth = _azimuth_error(left["azimuth_deg"], right["azimuth_deg"])
            elevation = abs(
                float(left["elevation_deg"]) - float(right["elevation_deg"])
            )
            distance = abs(float(left["distance_m"]) - float(right["distance_m"]))
            sums[f"{label}_azimuth_mae_deg"] += azimuth
            sums[f"{label}_elevation_mae_deg"] += elevation
            sums[f"{label}_distance_mae_m"] += distance
            sums[f"{label}_position_exact"] += float(
                math.isclose(azimuth, 0.0)
                and math.isclose(elevation, 0.0)
                and math.isclose(distance, 0.0)
            )

    metrics = {
        f"ordered_{key}": float(value) / denominator
        for key, value in sums.items()
    }
    metrics["ordered_transcript_exact"] = (
        float(transcript_exact_sum) / speech_count
        if speech_count
        else 1.0
    )
    metrics["ordered_source_count_exact"] = float(
        len(references) == len(prediction["sources"])
    )
    return metrics


def _is_failure_penalized_metric(name: str) -> bool:
    bounded_markers = (
        "accuracy",
        "exact",
        "_f1",
        "_iou",
        "_rate",
        "_score",
        "grammar_legal",
        "fraction",
    )
    return any(marker in name for marker in bounded_markers) and not any(
        marker in name
        for marker in ("mae", "abs_error", "relative_mae")
    )


def _row_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    output = {
        str(key): float(value)
        for key, value in dict(row.get("metrics") or {}).items()
    }
    non_finite = [key for key, value in output.items() if not math.isfinite(value)]
    if non_finite:
        raise ValueError(
            "Generation AR evaluation metrics must be finite: "
            f"{non_finite[:8]}"
        )
    return output


def _metric_aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    include: Callable[[str], bool],
    denominator_rows: int,
    failure_metric_names: Iterable[str] = (),
) -> dict[str, Any]:
    sums: defaultdict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()
    rows_with_metrics = 0
    for row in rows:
        selected = {
            key: value
            for key, value in _row_metrics(row).items()
            if include(key)
        }
        if selected:
            rows_with_metrics += 1
        for key, value in selected.items():
            sums[key] += value
            counts[key] += 1
    failure_keys = {
        key for key in sums if _is_failure_penalized_metric(key)
    } | {str(key) for key in failure_metric_names}
    return {
        "rows_with_metrics": rows_with_metrics,
        "metric_counts": {key: int(counts[key]) for key in sorted(counts)},
        "means": {
            key: sums[key] / counts[key]
            for key in sorted(sums)
        },
        "failure_penalized_means": {
            key: sums[key] / max(1, int(denominator_rows))
            for key in sorted(failure_keys)
        },
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    status_counts = Counter(str(row["status"]) for row in rows)
    parsed_rows = [row for row in rows if str(row["status"]) == "ok"]
    generated_rows = [
        row
        for row in rows
        if any(
            key in TOKEN_SEQUENCE_METRIC_NAMES
            for key in _row_metrics(row)
        )
    ]

    token = _metric_aggregate(
        generated_rows,
        include=lambda key: key in TOKEN_SEQUENCE_METRIC_NAMES,
        denominator_rows=total,
        failure_metric_names=_TOKEN_FAILURE_PENALIZED_METRIC_NAMES,
    )
    parsed = _metric_aggregate(
        parsed_rows,
        include=lambda key: (
            key not in TOKEN_SEQUENCE_METRIC_NAMES and key != "parse_rate"
        ),
        denominator_rows=total,
        failure_metric_names=_PARSED_FAILURE_PENALIZED_METRIC_NAMES,
    )

    # Parsing is a coverage transition between the two metric domains.  Derive
    # it from the durable status rather than trusting a duplicated metric
    # scalar, and keep it out of field means where no ScenePlan exists.
    parse_rate = len(parsed_rows) / max(1, total)

    successful_sums: defaultdict[str, float] = defaultdict(float)
    successful_counts: Counter[str] = Counter()
    for row in parsed_rows:
        for key, value in _row_metrics(row).items():
            successful_sums[key] += value
            successful_counts[key] += 1
    successful_means = {
        key: successful_sums[key] / successful_counts[key]
        for key in sorted(successful_sums)
    }

    failure_penalized = {
        **token["failure_penalized_means"],
        **parsed["failure_penalized_means"],
    }
    failure_penalized["parse_rate"] = parse_rate

    generated_count = len(generated_rows)
    parsed_count = len(parsed_rows)
    coverage = {
        "rows": total,
        "generated_rows": generated_count,
        "parsed_rows": parsed_count,
        "generation_error_rows": int(status_counts.get("generation_error", 0)),
        "parse_error_rows": int(status_counts.get("parse_error", 0)),
        "other_status_rows": int(
            total
            - status_counts.get("ok", 0)
            - status_counts.get("generation_error", 0)
            - status_counts.get("parse_error", 0)
        ),
        "generation_coverage_rate": generated_count / max(1, total),
        "parse_success_rate_over_all_rows": parsed_count / max(1, total),
        "parse_success_rate_given_generation": (
            parsed_count / generated_count if generated_count else 0.0
        ),
    }
    return {
        "successful_rows": parsed_count,
        "generated_rows": generated_count,
        "parsed_rows": parsed_count,
        "generation_success_rate": coverage["generation_coverage_rate"],
        "parse_success_rate": coverage["parse_success_rate_over_all_rows"],
        "status_counts": dict(sorted(status_counts.items())),
        "coverage": coverage,
        "metric_domains": {
            "token_sequence": {
                "rows_with_metrics": token["rows_with_metrics"],
                "metric_counts": token["metric_counts"],
                "means_on_generated_sequences": token["means"],
                "failure_penalized_means": token["failure_penalized_means"],
            },
            "parsed_sceneplan": {
                "rows_with_metrics": parsed["rows_with_metrics"],
                "metric_counts": parsed["metric_counts"],
                "means_on_parsed_sceneplans": parsed["means"],
                "failure_penalized_means": parsed["failure_penalized_means"],
            },
        },
        # Compatibility views used by the first evaluator version.  Token
        # metrics here remain conditioned on fully parsed rows, while the
        # merged penalized view now correctly includes parse-error token rows.
        "means_on_successful_decodes": successful_means,
        "failure_penalized_means": dict(sorted(failure_penalized.items())),
    }


def summarize_prediction_records(
    records: Iterable[Mapping[str, Any]], *, expected_rows: int
) -> dict[str, Any]:
    """Aggregate durable prediction rows without hiding decode failures."""

    rows = list(records)
    if len(rows) != int(expected_rows):
        raise ValueError(
            f"expected {expected_rows} prediction rows, found {len(rows)}"
        )
    ordinals = [int(row["ordinal"]) for row in rows]
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("duplicate Generation AR evaluation ordinals")
    if sorted(ordinals) != list(range(int(expected_rows))):
        raise ValueError("Generation AR evaluation ordinals do not cover expected rows")

    aggregate = _aggregate_rows(rows)
    aggregate["coverage"] = {
        "expected_rows": int(expected_rows),
        "ordinal_coverage_exact": True,
        **aggregate["coverage"],
    }

    def grouped(field: str) -> dict[str, Any]:
        values: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            values[str(row[field])].append(row)
        output = {}
        for key, group in sorted(values.items()):
            group_aggregate = _aggregate_rows(group)
            output[key] = {
                "rows": len(group),
                **group_aggregate,
            }
        return output

    return {
        "rows": len(rows),
        **aggregate,
        "by_source_count": grouped("source_count"),
        "by_template_id": grouped("template_id"),
    }


__all__ = [
    "GENERATION_AR_EVALUATION_CONTRACT",
    "TOKEN_SEQUENCE_METRIC_NAMES",
    "score_parsed_generation",
    "score_token_sequence",
    "summarize_prediction_records",
]
