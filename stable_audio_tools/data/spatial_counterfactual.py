"""Matched source-binding counterfactuals for fixed-window Spatial-CoT.

The intervention keeps source identity and semantic content fixed, while
permuting complete renderer-control bundles between two persistent source
slots.  Because activity, motion, and gain move together, the multiset of
per-source controls (and therefore the Plan-derived global field anchor) is
unchanged.  The target FOA changes because each dry source is rendered through
the other source's execution track.

ScenePlan and render recipes intentionally use different motion time bases, so
they are transformed separately.  Recipe playback windows are re-resolved
after an activity swap; retaining a stale playback window would silently
truncate or pad the wrong dry-source segment and turn the intervention into an
ill-defined renderer bug.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, MutableMapping, Sequence

from .spatial_edit_recipe import (
    _resolved_playback_window,
    is_speech_source,
    recipe_fingerprint,
)


SCENE_PLAN_CONTROL_KEYS = ("activity", "motion", "acoustics")
RECIPE_CONTROL_KEYS = ("activity", "motion", "gain_db")
_LOUDNESS_POLICY_KEYS = (
    "policy",
    "policy_version",
    "target_mono_rms",
    "peak_ceiling",
    "max_gain_db",
    "minimum_input_rms",
)


def _sources_by_id(
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for source in sources:
        source_id = str(source.get("source_id") or "")
        if not source_id or source_id in result:
            raise ValueError("source IDs must be present and unique")
        result[source_id] = source
    return result


def _mutable_sources_by_id(
    sources: Sequence[MutableMapping[str, Any]],
) -> dict[str, MutableMapping[str, Any]]:
    return {
        source_id: source  # type: ignore[dict-item]
        for source_id, source in _sources_by_id(sources).items()
    }


def _swap_optional_fields(
    left: MutableMapping[str, Any],
    right: MutableMapping[str, Any],
    keys: Sequence[str],
) -> None:
    for key in keys:
        left_value = copy.deepcopy(left.get(key))
        right_value = copy.deepcopy(right.get(key))
        if right_value is None:
            left.pop(key, None)
        else:
            left[key] = right_value
        if left_value is None:
            right.pop(key, None)
        else:
            right[key] = left_value


def _checked_pair(
    sources: Mapping[str, Mapping[str, Any]], source_a: str, source_b: str
) -> tuple[str, str]:
    left, right = str(source_a), str(source_b)
    if left == right or left not in sources or right not in sources:
        raise ValueError(f"invalid source pair: {left!r}, {right!r}")
    return left, right


def swap_scene_plan_source_controls(
    plan: Mapping[str, Any], source_a: str, source_b: str
) -> dict[str, Any]:
    """Return a ScenePlan with two complete source-control bundles swapped."""

    swapped = copy.deepcopy(dict(plan))
    sources = ((swapped.get("scene") or {}).get("sources") or [])
    if not isinstance(sources, list) or not all(
        isinstance(source, MutableMapping) for source in sources
    ):
        raise ValueError("ScenePlan sources must be a list of mappings")
    by_id = _mutable_sources_by_id(sources)
    left, right = _checked_pair(by_id, source_a, source_b)
    _swap_optional_fields(
        by_id[left], by_id[right], SCENE_PLAN_CONTROL_KEYS
    )
    return swapped


def _source_loudness_policy(source: Mapping[str, Any]) -> dict[str, Any]:
    loudness = ((source.get("playback") or {}).get("loudness") or {})
    missing = [key for key in _LOUDNESS_POLICY_KEYS if key not in loudness]
    if missing:
        raise ValueError(
            f"source {source.get('source_id')!r} lacks loudness policy: {missing}"
        )
    return {key: copy.deepcopy(loudness[key]) for key in _LOUDNESS_POLICY_KEYS}


def _refresh_playback(source: MutableMapping[str, Any]) -> None:
    activity = source.get("activity") or {}
    onset = float(activity.get("onset_sec", 0.0))
    offset = float(activity.get("offset_sec", 0.0))
    duration = offset - onset
    if duration <= 0.0:
        raise ValueError(
            f"source {source.get('source_id')!r} has a non-positive activity window"
        )
    source["playback"] = _resolved_playback_window(
        source["dry_audio"],
        activity_duration_sec=duration,
        preserve_source_start=is_speech_source(source),
        source_loudness=_source_loudness_policy(source),
    )


def swap_recipe_source_controls(
    recipe: Mapping[str, Any], source_a: str, source_b: str
) -> dict[str, Any]:
    """Swap the render controls for two dry sources in an immutable recipe.

    Dry audio, event/content metadata, and persistent IDs remain attached to
    their original sources.  Activity, activity-relative motion, and gain move
    as one execution bundle.  Playback is then deterministically re-resolved
    from each unchanged dry crop for its newly assigned activity duration.
    """

    swapped = copy.deepcopy(dict(recipe))
    sources = swapped.get("sources") or []
    if not isinstance(sources, list) or not all(
        isinstance(source, MutableMapping) for source in sources
    ):
        raise ValueError("recipe sources must be a list of mappings")
    by_id = _mutable_sources_by_id(sources)
    left, right = _checked_pair(by_id, source_a, source_b)
    _swap_optional_fields(by_id[left], by_id[right], RECIPE_CONTROL_KEYS)
    _refresh_playback(by_id[left])
    _refresh_playback(by_id[right])
    swapped.pop("recipe_fingerprint", None)
    swapped["recipe_fingerprint"] = recipe_fingerprint(swapped)
    return swapped


__all__ = [
    "RECIPE_CONTROL_KEYS",
    "SCENE_PLAN_CONTROL_KEYS",
    "swap_recipe_source_controls",
    "swap_scene_plan_source_controls",
]
