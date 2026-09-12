"""Field-level held-out metrics for P11 ScenePlan generation and editing."""

from __future__ import annotations

import math
import re
from collections import Counter
from itertools import permutations
from typing import Any, Mapping, Sequence

from .model_sceneplan import validate_model_sceneplan


_WORD = re.compile(r"\w+", flags=re.UNICODE)
_MISSING = object()
P11_FIELD_METRIC_CONTRACT = "gue_prompt_constraint_v3"
P11_GENERATION_CONSTRAINT_CONTRACT = "prompt_known_field_groups_v1"


def _words(value: Any) -> list[str]:
    return _WORD.findall(" ".join(str(value or "").lower().split()))


def _edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, 1):
        current = [row]
        for column, right_value in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _sequence_accuracy(reference: Sequence[Any], hypothesis: Sequence[Any]) -> float:
    if not reference:
        return 1.0 if not hypothesis else 0.0
    return max(0.0, 1.0 - _edit_distance(reference, hypothesis) / len(reference))


def _token_f1(reference: Any, hypothesis: Any) -> float:
    reference_tokens = Counter(_words(reference))
    hypothesis_tokens = Counter(_words(hypothesis))
    if not reference_tokens:
        return 1.0 if not hypothesis_tokens else 0.0
    overlap = sum((reference_tokens & hypothesis_tokens).values())
    if not hypothesis_tokens:
        return 0.0
    precision = overlap / sum(hypothesis_tokens.values())
    recall = overlap / sum(reference_tokens.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _source_map(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(source["source_id"]): source for source in plan["sources"]}


def _set_f1(reference: set[str], hypothesis: set[str]) -> float:
    if not reference:
        return 1.0 if not hypothesis else 0.0
    overlap = len(reference & hypothesis)
    if not hypothesis:
        return 0.0
    precision = overlap / len(hypothesis)
    recall = overlap / len(reference)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _positions(source: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    trajectory = source["trajectory"]
    motion = str(trajectory["type"])
    if motion == "static":
        return {"position": trajectory["position"]}
    if motion == "linear":
        return {"start": trajectory["start"], "end": trajectory["end"]}
    return {
        f"keyframe_{index}": item["position"]
        for index, item in enumerate(trajectory["keyframes"])
    }


def _circular_error(left: float, right: float) -> float:
    return abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)


def _source_semantic(source: Mapping[str, Any]) -> str:
    return str(
        source.get("speaker_description")
        or source.get("description")
        or ""
    )


def _pair_activity_iou(
    reference: Mapping[str, Any], hypothesis: Mapping[str, Any]
) -> float:
    left = reference["activity"]
    right = hypothesis["activity"]
    intersection = max(
        0.0,
        min(float(left["offset_sec"]), float(right["offset_sec"]))
        - max(float(left["onset_sec"]), float(right["onset_sec"])),
    )
    union = max(float(left["offset_sec"]), float(right["offset_sec"])) - min(
        float(left["onset_sec"]), float(right["onset_sec"])
    )
    return 1.0 if union <= 0.0 else intersection / union


def _source_pair_similarity(
    reference: Mapping[str, Any], hypothesis: Mapping[str, Any]
) -> float:
    """Similarity used only to find an id-free one-to-one source assignment."""

    kind_exact = float(reference["kind"] == hypothesis["kind"])
    semantic = _token_f1(
        _source_semantic(reference), _source_semantic(hypothesis)
    )
    transcript = _sequence_accuracy(
        _words(reference.get("transcript")),
        _words(hypothesis.get("transcript")),
    )
    activity = _pair_activity_iou(reference, hypothesis)
    motion = float(
        reference["trajectory"]["type"] == hypothesis["trajectory"]["type"]
    )
    spatial_values: list[float] = []
    reference_positions = _positions(reference)
    hypothesis_positions = _positions(hypothesis)
    for label in sorted(set(reference_positions) & set(hypothesis_positions)):
        left = reference_positions[label]
        right = hypothesis_positions[label]
        azimuth = max(
            0.0,
            1.0
            - _circular_error(left["azimuth_deg"], right["azimuth_deg"]) / 180.0,
        )
        elevation = max(
            0.0,
            1.0
            - abs(
                float(left["elevation_deg"])
                - float(right["elevation_deg"])
            )
            / 90.0,
        )
        distance = min(
            float(left["distance_m"]), float(right["distance_m"])
        ) / max(float(left["distance_m"]), float(right["distance_m"]), 1e-9)
        spatial_values.append((azimuth + elevation + distance) / 3.0)
    spatial = _mean(spatial_values)
    return (
        3.0 * kind_exact
        + 4.0 * semantic
        + transcript
        + 2.0 * activity
        + motion
        + spatial
    )


def _match_sources(
    target_sources: Mapping[str, Mapping[str, Any]],
    predicted_sources: Mapping[str, Mapping[str, Any]],
    *,
    source_matching: str,
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if source_matching == "persistent_id":
        return [
            (target_sources[source_id], predicted_sources[source_id])
            for source_id in sorted(set(target_sources) & set(predicted_sources))
        ]
    if source_matching != "permutation_invariant":
        raise ValueError(
            "source_matching must be persistent_id or permutation_invariant"
        )

    references = list(target_sources.values())
    hypotheses = list(predicted_sources.values())
    if not references or not hypotheses:
        return []
    best_pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    best_score = -math.inf
    if len(references) <= len(hypotheses):
        for hypothesis_indices in permutations(
            range(len(hypotheses)), len(references)
        ):
            pairs = [
                (reference, hypotheses[hypothesis_index])
                for reference, hypothesis_index in zip(
                    references, hypothesis_indices
                )
            ]
            score = sum(_source_pair_similarity(*pair) for pair in pairs)
            if score > best_score:
                best_score, best_pairs = score, pairs
    else:
        for reference_indices in permutations(
            range(len(references)), len(hypotheses)
        ):
            pairs = [
                (references[reference_index], hypothesis)
                for reference_index, hypothesis in zip(
                    reference_indices, hypotheses
                )
            ]
            score = sum(_source_pair_similarity(*pair) for pair in pairs)
            if score > best_score:
                best_score, best_pairs = score, pairs
    return best_pairs


def _count_f1(reference_count: int, hypothesis_count: int, matched: int) -> float:
    if reference_count <= 0:
        return float(hypothesis_count <= 0)
    if hypothesis_count <= 0:
        return 0.0
    precision = matched / hypothesis_count
    recall = matched / reference_count
    return 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)


def _source_content(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in source.items()
        if key not in {"source_id", "gain_db"}
    }


def _mean(values: Sequence[float], *, default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else float(default)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten plans by persistent source id so removal is directly scored."""

    output: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"sample_id", "gain_db"}:
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            if key == "sources" and isinstance(item, Sequence):
                for source in item:
                    source_id = str(source["source_id"])
                    source_path = f"{path}.{source_id}"
                    output[f"{source_path}.__present__"] = True
                    output.update(_flatten(source, source_path))
            else:
                output.update(_flatten(item, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            output.update(_flatten(item, f"{prefix}.{index}"))
    else:
        output[prefix] = value
    return output


def score_sceneplan_fields(
    target: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    source_matching: str = "persistent_id",
) -> dict[str, float]:
    """Score one parsed prediction without allowing averages to hide fields."""

    target = validate_model_sceneplan(target)
    prediction = validate_model_sceneplan(prediction)
    target_sources = _source_map(target)
    predicted_sources = _source_map(prediction)
    target_ids = set(target_sources)
    predicted_ids = set(predicted_sources)
    matched_sources = _match_sources(
        target_sources,
        predicted_sources,
        source_matching=source_matching,
    )
    target_speech_count = sum(
        source["kind"] == "speech" for source in target_sources.values()
    )

    kind_exact: list[float] = []
    semantic_exact: list[float] = []
    semantic_f1: list[float] = []
    transcript_exact: list[float] = []
    transcript_word_accuracy: list[float] = []
    transcript_char_accuracy: list[float] = []
    activity_exact: list[float] = []
    activity_iou: list[float] = []
    onset_frame_abs: list[float] = []
    offset_frame_abs: list[float] = []
    motion_exact: list[float] = []
    position_exact: list[float] = []
    azimuth_abs: list[float] = []
    elevation_abs: list[float] = []
    distance_abs: list[float] = []
    distance_relative_abs: list[float] = []

    for reference, hypothesis in matched_sources:
        kind_matches = str(reference["kind"]) == str(hypothesis["kind"])
        kind_exact.append(float(kind_matches))
        if kind_matches and reference["kind"] == "speech":
            ref_semantic = reference["speaker_description"]
            hyp_semantic = hypothesis["speaker_description"]
            ref_transcript = reference["transcript"]
            hyp_transcript = hypothesis["transcript"]
            transcript_exact.append(float(ref_transcript == hyp_transcript))
            transcript_word_accuracy.append(
                _sequence_accuracy(_words(ref_transcript), _words(hyp_transcript))
            )
            transcript_char_accuracy.append(
                _sequence_accuracy(list(ref_transcript), list(hyp_transcript))
            )
        elif kind_matches:
            ref_semantic = reference["description"]
            hyp_semantic = hypothesis["description"]
        else:
            ref_semantic = (
                reference.get("speaker_description") or reference.get("description")
            )
            hyp_semantic = (
                hypothesis.get("speaker_description") or hypothesis.get("description")
            )
        semantic_exact.append(float(ref_semantic == hyp_semantic))
        semantic_f1.append(_token_f1(ref_semantic, hyp_semantic))

        ref_activity = reference["activity"]
        hyp_activity = hypothesis["activity"]
        ref_onset = float(ref_activity["onset_sec"])
        ref_offset = float(ref_activity["offset_sec"])
        hyp_onset = float(hyp_activity["onset_sec"])
        hyp_offset = float(hyp_activity["offset_sec"])
        activity_exact.append(float(ref_activity == hyp_activity))
        intersection = max(0.0, min(ref_offset, hyp_offset) - max(ref_onset, hyp_onset))
        union = max(ref_offset, hyp_offset) - min(ref_onset, hyp_onset)
        activity_iou.append(1.0 if union <= 0 else intersection / union)
        hop_seconds = 1024.0 / 44_100.0
        onset_frame_abs.append(abs(ref_onset - hyp_onset) / hop_seconds)
        offset_frame_abs.append(abs(ref_offset - hyp_offset) / hop_seconds)

        ref_motion = str(reference["trajectory"]["type"])
        hyp_motion = str(hypothesis["trajectory"]["type"])
        motion_exact.append(float(ref_motion == hyp_motion))
        ref_positions = _positions(reference)
        hyp_positions = _positions(hypothesis)
        for label in sorted(set(ref_positions) & set(hyp_positions)):
            ref_position = ref_positions[label]
            hyp_position = hyp_positions[label]
            az_error = _circular_error(
                ref_position["azimuth_deg"], hyp_position["azimuth_deg"]
            )
            el_error = abs(
                float(ref_position["elevation_deg"])
                - float(hyp_position["elevation_deg"])
            )
            dist_error = abs(
                float(ref_position["distance_m"])
                - float(hyp_position["distance_m"])
            )
            azimuth_abs.append(az_error)
            elevation_abs.append(el_error)
            distance_abs.append(dist_error)
            distance_relative_abs.append(
                dist_error / max(float(ref_position["distance_m"]), 1e-9)
            )
            position_exact.append(
                float(az_error == 0.0 and el_error == 0.0 and dist_error == 0.0)
            )

    source_recall_denominator = max(1, len(target_ids))
    matched_fraction = len(matched_sources) / source_recall_denominator
    # Per-source metrics are recall-weighted so a missing source cannot vanish
    # from an average over only matched ids.
    def source_metric(values: Sequence[float]) -> float:
        return _mean(values) * matched_fraction

    # Transcript lists contain entries only for target speech sources that
    # were actually assigned to a predicted speech source.  Dividing their
    # mean by the all-source matched fraction would let a missing or
    # wrong-kind speech source disappear whenever another transcript matched.
    # Normalize the accumulated scores by the complete target speech count so
    # every missing/wrong-kind speech source contributes an explicit zero.
    def transcript_metric(values: Sequence[float]) -> float:
        if target_speech_count == 0:
            return 1.0
        return float(sum(values) / target_speech_count)

    spatial_soft = _mean(
        [
            max(0.0, 1.0 - az / 30.0)
            * max(0.0, 1.0 - el / 15.0)
            * max(0.0, 1.0 - rel / 0.25)
            for az, el, rel in zip(
                azimuth_abs, elevation_abs, distance_relative_abs
            )
        ]
    ) * matched_fraction
    components = {
        "source_count_accuracy": float(len(target_ids) == len(predicted_ids)),
        "source_id_f1": _set_f1(target_ids, predicted_ids),
        "source_assignment_f1": _count_f1(
            len(target_ids), len(predicted_ids), len(matched_sources)
        ),
        "room_accuracy": float(target["room"] == prediction["room"]),
        "duration_accuracy": float(target["duration_sec"] == prediction["duration_sec"]),
        "kind_accuracy": source_metric(kind_exact),
        "semantic_exact": source_metric(semantic_exact),
        "semantic_token_f1": source_metric(semantic_f1),
        "transcript_exact": transcript_metric(transcript_exact),
        "transcript_word_accuracy": transcript_metric(
            transcript_word_accuracy
        ),
        "transcript_char_accuracy": transcript_metric(
            transcript_char_accuracy
        ),
        "activity_exact": source_metric(activity_exact),
        "activity_iou": source_metric(activity_iou),
        "motion_type_accuracy": source_metric(motion_exact),
        "position_bin_exact": source_metric(position_exact),
        "spatial_soft_score": spatial_soft,
        "azimuth_mae_deg": _mean(azimuth_abs, default=180.0),
        "elevation_mae_deg": _mean(elevation_abs, default=90.0),
        "distance_mae_m": _mean(distance_abs, default=50.0),
        "distance_relative_mae": _mean(distance_relative_abs, default=1.0),
        "onset_mae_frames": _mean(onset_frame_abs, default=432.0),
        "offset_mae_frames": _mean(offset_frame_abs, default=432.0),
    }
    scene_components = [
        components["source_count_accuracy"],
        components["source_assignment_f1"],
        components["room_accuracy"],
        components["duration_accuracy"],
        components["kind_accuracy"],
        components["semantic_token_f1"],
        components["transcript_word_accuracy"],
        components["activity_iou"],
        components["motion_type_accuracy"],
        components["spatial_soft_score"],
    ]
    components["scene_score"] = _mean(scene_components)
    persistent_scene_exact = float(_flatten(target) == _flatten(prediction))
    permutation_scene_exact = float(
        len(target_sources) == len(predicted_sources)
        and target["room"] == prediction["room"]
        and target["duration_sec"] == prediction["duration_sec"]
        and all(
            _source_content(reference) == _source_content(hypothesis)
            for reference, hypothesis in matched_sources
        )
    )
    components["persistent_scene_exact"] = persistent_scene_exact
    components["permutation_scene_exact"] = permutation_scene_exact
    components["scene_exact"] = (
        permutation_scene_exact
        if source_matching == "permutation_invariant"
        else persistent_scene_exact
    )
    return components


def score_sceneplan_edit(
    input_plan: Mapping[str, Any],
    target_plan: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> dict[str, float]:
    """Measure requested deltas and preservation of every unchanged scalar."""

    input_flat = _flatten(validate_model_sceneplan(input_plan))
    target_flat = _flatten(validate_model_sceneplan(target_plan))
    prediction_flat = _flatten(validate_model_sceneplan(prediction))
    universe = set(input_flat) | set(target_flat)
    changed = {
        key
        for key in universe
        if input_flat.get(key, _MISSING) != target_flat.get(key, _MISSING)
    }
    unchanged = universe - changed

    def matches_target(key: str) -> bool:
        return prediction_flat.get(key, _MISSING) == target_flat.get(key, _MISSING)

    changed_accuracy = (
        _mean([float(matches_target(key)) for key in sorted(changed)])
        if changed
        else float(prediction_flat == target_flat)
    )
    preservation = (
        _mean([float(matches_target(key)) for key in sorted(unchanged)], default=1.0)
    )
    return {
        "edit_changed_scalars": float(len(changed)),
        "edit_preserved_scalars": float(len(unchanged)),
        "edit_is_noop": float(not changed),
        "edit_success": changed_accuracy,
        "preservation_accuracy": preservation,
        "edit_exact": float(changed_accuracy == 1.0),
        "preservation_exact": float(preservation == 1.0),
    }


def _harmonic_score(values: Sequence[float]) -> float:
    if not values or any(float(value) <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / float(value) for value in values)


def _coarse_activity_label(source: Mapping[str, Any], duration_sec: float) -> str:
    activity = source["activity"]
    duration = max(float(duration_sec), 1.0e-9)
    onset_fraction = float(activity["onset_sec"]) / duration
    offset_fraction = float(activity["offset_sec"]) / duration
    if onset_fraction < 0.05 and offset_fraction > 0.95:
        return "throughout"
    if offset_fraction <= 0.5:
        return "early"
    if onset_fraction >= 0.5:
        return "late"
    return "middle"


def _coarse_octant(azimuth_deg: float) -> int:
    normalized = (float(azimuth_deg) + 360.0) % 360.0
    return int(((normalized + 22.5) % 360.0) // 45.0)


def _coarse_distance_band(distance_m: float) -> int:
    value = float(distance_m)
    return 0 if value < 1.2 else 1 if value < 2.0 else 2


def _coarse_geometry_signature(source: Mapping[str, Any]) -> tuple[Any, ...]:
    trajectory = source["trajectory"]
    motion = str(trajectory["type"])
    if motion == "static":
        position = trajectory["position"]
        return (
            "static",
            _coarse_octant(position["azimuth_deg"]),
            _coarse_distance_band(position["distance_m"]),
        )
    if motion == "linear":
        start, end = trajectory["start"], trajectory["end"]
    else:
        start = trajectory["keyframes"][0]["position"]
        end = trajectory["keyframes"][-1]["position"]
    return (
        "moving",
        _coarse_octant(start["azimuth_deg"]),
        _coarse_octant(end["azimuth_deg"]),
        _coarse_distance_band(start["distance_m"]),
        _coarse_distance_band(end["distance_m"]),
    )


def _matched_source_group_score(
    target: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    source_matching: str,
    scorer,
) -> float:
    target_sources = _source_map(target)
    predicted_sources = _source_map(prediction)
    pairs = _match_sources(
        target_sources,
        predicted_sources,
        source_matching=source_matching,
    )
    if not target_sources:
        return float(not predicted_sources)
    return sum(float(scorer(left, right)) for left, right in pairs) / len(
        target_sources
    )


def score_generation_constraints(
    target: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    known_field_groups: Sequence[str],
    source_matching: str = "permutation_invariant",
) -> dict[str, float]:
    """Score only fields stated by a Generation prompt.

    Underspecified prompt views deliberately permit multiple valid complete
    ScenePlans.  Their hidden synthetic room, exact timing, or exact geometry
    must therefore remain diagnostics rather than supervised evaluation
    targets.  The sidecar's frozen ``known_field_groups`` is the authority.
    """

    target = validate_model_sceneplan(target)
    prediction = validate_model_sceneplan(prediction)
    fields = score_sceneplan_fields(
        target,
        prediction,
        source_matching=source_matching,
    )
    groups = tuple(dict.fromkeys(str(value) for value in known_field_groups))
    allowed = {
        "duration",
        "room",
        "source_semantics",
        "temporal",
        "geometry",
        "gain",
        "coarse_temporal",
        "coarse_geometry",
    }
    unknown = sorted(set(groups) - allowed)
    if unknown:
        raise ValueError(f"unsupported Generation known-field groups: {unknown}")

    group_scores: dict[str, float] = {}
    if "duration" in groups:
        group_scores["duration"] = fields["duration_accuracy"]
    if "room" in groups:
        group_scores["room"] = fields["room_accuracy"]
    if "source_semantics" in groups:
        group_scores["source_semantics"] = _mean(
            [
                fields["source_count_accuracy"],
                fields["source_assignment_f1"],
                fields["kind_accuracy"],
                fields["semantic_token_f1"],
                fields["transcript_word_accuracy"],
            ]
        )
    if "temporal" in groups:
        group_scores["temporal"] = fields["activity_iou"]
    if "geometry" in groups:
        group_scores["geometry"] = _mean(
            [fields["motion_type_accuracy"], fields["spatial_soft_score"]]
        )
    if "gain" in groups:
        group_scores["gain"] = _matched_source_group_score(
            target,
            prediction,
            source_matching=source_matching,
            scorer=lambda left, right: float(
                float(left.get("gain_db", 0.0))
                == float(right.get("gain_db", 0.0))
            ),
        )
    if "coarse_temporal" in groups:
        target_duration = float(target["duration_sec"])
        predicted_duration = float(prediction["duration_sec"])
        group_scores["coarse_temporal"] = _matched_source_group_score(
            target,
            prediction,
            source_matching=source_matching,
            scorer=lambda left, right: float(
                _coarse_activity_label(left, target_duration)
                == _coarse_activity_label(right, predicted_duration)
            ),
        )
    if "coarse_geometry" in groups:
        group_scores["coarse_geometry"] = _matched_source_group_score(
            target,
            prediction,
            source_matching=source_matching,
            scorer=lambda left, right: float(
                _coarse_geometry_signature(left)
                == _coarse_geometry_signature(right)
            ),
        )

    # Old exact-only manifests did not carry known groups. Preserve their
    # historical behavior without allowing new weighted prompt views to leak
    # hidden target fields into model selection.
    constraint_score = (
        _mean(list(group_scores.values()))
        if group_scores
        else float(fields["scene_score"])
    )
    return {
        **fields,
        "oracle_full_scene_score": float(fields["scene_score"]),
        "generation_constraint_score": constraint_score,
        "generation_known_group_count": float(len(group_scores)),
        **{
            f"constraint_{name}_score": float(value)
            for name, value in sorted(group_scores.items())
        },
        "task_score": constraint_score,
    }


def _editing_task_score_v2(
    fields: Mapping[str, float], edit: Mapping[str, float]
) -> float:
    """Gate scene quality by the requested delta instead of rewarding copies."""

    if bool(edit["edit_is_noop"]):
        # A no-op is correct only when the complete input state is reproduced.
        return float(fields["scene_score"]) * float(edit["edit_success"])
    delta_and_preservation = _harmonic_score(
        [float(edit["edit_success"]), float(edit["preservation_accuracy"])]
    )
    return float(fields["scene_score"]) * delta_and_preservation


def score_p11_prediction(
    *,
    task: str,
    target_plan: Mapping[str, Any],
    prediction: Mapping[str, Any],
    input_plan: Mapping[str, Any] | None = None,
    observed_plan: Mapping[str, Any] | None = None,
    source_matching: str = "persistent_id",
    editing_score_version: str = "patch_applied_v3",
    known_field_groups: Sequence[str] | None = None,
) -> dict[str, float]:
    if str(task) == "generation" and known_field_groups is not None:
        return score_generation_constraints(
            target_plan,
            prediction,
            known_field_groups=known_field_groups,
            source_matching=source_matching,
        )
    metrics = score_sceneplan_fields(
        target_plan,
        prediction,
        source_matching=source_matching,
    )
    if str(task) == "editing":
        if editing_score_version == "observed_patch_revised_v1":
            if observed_plan is None:
                raise ValueError(
                    "audio-aware editing metrics require the observed ScenePlan"
                )
            edit_base = observed_plan
        elif editing_score_version == "patch_applied_v3":
            if input_plan is None:
                raise ValueError("editing metrics require the input ScenePlan")
            edit_base = input_plan
        else:
            raise ValueError(
                "unsupported P11 editing_score_version="
                f"{editing_score_version!r}"
            )
        edit = score_sceneplan_edit(edit_base, target_plan, prediction)
        metrics.update(edit)
        metrics["task_score"] = _editing_task_score_v2(metrics, edit)
        copy_fields = score_sceneplan_fields(
            target_plan, edit_base,
            source_matching="persistent_id",
        )
        copy_edit = score_sceneplan_edit(edit_base, target_plan, edit_base)
        copy_score = _editing_task_score_v2(copy_fields, copy_edit)
        metrics["copy_input_task_score"] = copy_score
        metrics["task_score_lift_over_copy"] = float(metrics["task_score"]) - copy_score
        if copy_score >= 1.0:
            metrics["task_score_normalized_lift"] = (
                1.0
                if metrics["task_score"] >= copy_score
                else float(metrics["task_score"]) - copy_score
            )
        else:
            metrics["task_score_normalized_lift"] = (
                float(metrics["task_score"] - copy_score) / (1.0 - copy_score)
            )
    else:
        metrics["task_score"] = metrics["scene_score"]
    return metrics


__all__ = [
    "P11_FIELD_METRIC_CONTRACT",
    "P11_GENERATION_CONSTRAINT_CONTRACT",
    "score_p11_prediction",
    "score_generation_constraints",
    "score_sceneplan_edit",
    "score_sceneplan_fields",
]
