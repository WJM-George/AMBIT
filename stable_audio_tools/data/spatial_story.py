"""Conversation and edit lineage helpers for Spatial-CoT datasets.

``ScenePlan`` remains the canonical description of one rendered FOA clip.  This
module adds the missing *relationship* layer: stable source identities within a
conversation, before/after references, and explicit add/change/remove diffs.

The helpers deliberately distinguish facts present in an existing artifact
from facts that would be needed to reproduce it.  In particular, a source path
and a room description do not imply that the original dry-audio crop, exact
microphone, RIR order, or pre-mix source track is available.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


CONVERSATION_SCHEMA = "stable_audio_tools.spatial_cot_conversation"
CONVERSATION_SCHEMA_VERSION = "1.0"
TRACK_SCHEMA = "stable_audio_tools.scene_plan_source_tracks"
TRACK_SCHEMA_VERSION = "1.0"

SOURCE_TRACK_FEATURES = (
    "activity",
    "direction_x",
    "direction_y",
    "direction_z",
    "inverse_distance",
    "gain_linear",
    "geometry_confidence",
    "activity_confidence",
)

_REPRODUCIBLE_RECIPE_FIELDS = (
    "renderer.seed",
    "renderer.version",
    "renderer.dynamic_mode",
    "room.microphone_xyz_m",
    "room.max_order",
    "mix.normalization.type",
)


class SpatialStoryError(ValueError):
    """Raised when a conversation, diff, or source-track request is invalid."""


def _digest(*values: Any, size: int = 10) -> str:
    payload = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=size).hexdigest()


def conversation_id_for_plan(plan: Mapping[str, Any]) -> str:
    sample_id = str(plan.get("sample_id") or "")
    if not sample_id:
        audio_path = (plan.get("audio") or {}).get("path")
        if not audio_path:
            raise SpatialStoryError("ScenePlan requires sample_id or audio.path")
        sample_id = _digest(str(audio_path), size=12)
    return f"spcot_{sample_id}"


def persistent_source_id(
    conversation_id: str,
    source: Mapping[str, Any],
    source_index: int,
) -> str:
    """Return an ID that remains stable across cloned/edited conversation turns.

    The ID is conversation-local by design.  A dry file reused in two unrelated
    scenes is not automatically the same acoustic event, while a source cloned
    from one turn to the next keeps the same ID.
    """

    content = source.get("content") or {}
    event = source.get("event") or {}
    identity = (
        content.get("source_audio_id"),
        content.get("source_audio_path"),
        source.get("source_id"),
        event.get("source_dataset"),
        event.get("label"),
        int(source_index),
    )
    return f"src_{_digest(conversation_id, identity)}"


def with_persistent_source_ids(
    plan: Mapping[str, Any],
    *,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    """Copy a plan and assign stable, model-friendly conversation source slots.

    ``source_0`` ... ``source_N`` is the ID exposed to the compact plan LM and
    edit instructions. A hashed ``source_uid`` is retained as non-generative
    lineage metadata; the LM must never be asked to predict random hashes.
    """

    result = copy.deepcopy(dict(plan))
    conversation_id = conversation_id or conversation_id_for_plan(result)
    sources = ((result.get("scene") or {}).get("sources") or [])
    seen: set[str] = set()
    for index, source in enumerate(sources):
        legacy_id = source.get("source_id")
        source_uid = persistent_source_id(conversation_id, source, index)
        source_id = f"source_{index}"
        if source_id in seen:
            raise SpatialStoryError(f"duplicate persistent source ID: {source_id}")
        seen.add(source_id)
        if legacy_id is not None and legacy_id != source_id:
            source["legacy_source_id"] = str(legacy_id)
        source["source_id"] = source_id
        source["source_uid"] = source_uid
    return result


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key in sorted(value):
            if key in {"source_id", "legacy_source_id"}:
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(value[key], child))
        return flattened
    if isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            flattened.update(_flatten(item, child))
        if not value:
            flattened[prefix] = []
        return flattened
    flattened[prefix] = value
    return flattened


def _same_value(first: Any, second: Any) -> bool:
    if isinstance(first, (int, float)) and isinstance(second, (int, float)):
        return math.isclose(float(first), float(second), rel_tol=1e-6, abs_tol=1e-6)
    return first == second


def diff_scene_plans(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute an AudioChat-style source diff between two ScenePlans."""

    before_sources = {
        str(source["source_id"]): source
        for source in (((before or {}).get("scene") or {}).get("sources") or [])
    }
    after_sources = {
        str(source["source_id"]): source
        for source in ((after.get("scene") or {}).get("sources") or [])
    }
    if len(before_sources) != len((((before or {}).get("scene") or {}).get("sources") or [])):
        raise SpatialStoryError("before plan contains duplicate/missing source_id")
    if len(after_sources) != len(((after.get("scene") or {}).get("sources") or [])):
        raise SpatialStoryError("after plan contains duplicate/missing source_id")

    before_ids = set(before_sources)
    after_ids = set(after_sources)
    changed = []
    for source_id in sorted(before_ids & after_ids):
        left = _flatten(before_sources[source_id])
        right = _flatten(after_sources[source_id])
        fields = sorted(
            key
            for key in set(left) | set(right)
            if key not in left or key not in right or not _same_value(left[key], right[key])
        )
        if fields:
            changed.append({"source_id": source_id, "fields": fields})
    return {
        "added": sorted(after_ids - before_ids),
        "changed": changed,
        "removed": sorted(before_ids - after_ids),
    }


def _nested_present(value: Mapping[str, Any], dotted: str) -> bool:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or current.get(part) is None:
            return False
        current = current[part]
    return True


def plan_readiness(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Classify which supervision routes are honest for one existing plan."""

    audio = plan.get("audio") or {}
    scene = plan.get("scene") or {}
    sources = scene.get("sources") or []
    source_ids = [source.get("source_id") for source in sources]
    source_paths = [
        (source.get("content") or {}).get("source_audio_path") for source in sources
    ]
    activity = [source.get("activity") or {} for source in sources]
    exact_activity = all(
        item.get("onset_sec") is not None and item.get("offset_sec") is not None
        for item in activity
    ) if sources else False
    recipe = plan.get("render_recipe") or {}
    recipe_missing = [
        field for field in _REPRODUCIBLE_RECIPE_FIELDS if not _nested_present(recipe, field)
    ]
    has_tracks = all(
        bool((source.get("render") or {}).get("source_track_ref")) for source in sources
    ) if sources else False
    persistent_ids = (
        bool(sources)
        and len(source_ids) == len(set(source_ids))
        and all(
            isinstance(value, str) and value.startswith("source_")
            for value in source_ids
        )
        and all(
            isinstance(source.get("source_uid"), str)
            and source["source_uid"].startswith("src_")
            for source in sources
        )
    )
    all_dry_sources = bool(sources) and all(bool(path) for path in source_paths)
    return {
        "generation_ready": bool(audio.get("path") and sources),
        "understanding_ready": bool(audio.get("path") and sources),
        "source_plan_ready": bool(sources and persistent_ids),
        "multi_source": len(sources) > 1,
        "activity_timing_ready": exact_activity,
        "all_dry_sources_referenced": all_dry_sources,
        "source_tracks_available": has_tracks,
        "exact_rerender_ready": not recipe_missing and all_dry_sources,
        "paired_edit_ready": False,
        "missing_render_recipe_fields": recipe_missing,
        "notes": (
            "Existing audio plus its plan supports generation/understanding. "
            "Paired editing requires a separately rendered before/after pair; "
            "a synthetic diff must never be labelled as paired audio supervision."
        ),
    }


def plan_reference(plan: Mapping[str, Any], store_root: str | Path) -> dict[str, Any]:
    audio = plan.get("audio") or {}
    return {
        "store": str(Path(store_root).expanduser().resolve()),
        "sample_id": str(plan["sample_id"]),
        "audio_path": audio.get("path"),
    }


def seed_conversation(
    plan: Mapping[str, Any],
    *,
    scene_plan_store: str | Path,
) -> dict[str, Any]:
    """Create a non-fabricated initial generation turn for an existing clip."""

    conversation_id = conversation_id_for_plan(plan)
    persistent_plan = with_persistent_source_ids(plan, conversation_id=conversation_id)
    source_ids = [
        str(source["source_id"])
        for source in ((persistent_plan.get("scene") or {}).get("sources") or [])
    ]
    source_id_map = [
        {
            "legacy_source_id": source.get("legacy_source_id"),
            "source_id": source.get("source_id"),
            "source_uid": source.get("source_uid"),
        }
        for source in ((persistent_plan.get("scene") or {}).get("sources") or [])
    ]
    reference = plan_reference(plan, scene_plan_store)
    readiness = plan_readiness(persistent_plan)
    return {
        "schema": CONVERSATION_SCHEMA,
        "schema_version": CONVERSATION_SCHEMA_VERSION,
        "conversation_id": conversation_id,
        "dataset_id": plan.get("dataset_id"),
        "source_sample_id": plan.get("sample_id"),
        "source_id_map": source_id_map,
        "capabilities": readiness,
        "turns": [
            {
                "turn_id": "turn_000",
                "parent_turn_id": None,
                "task": "text_to_spatial_audio",
                "instruction": plan.get("caption"),
                "before": {"audio_ref": None, "scene_plan_ref": None},
                "after": {
                    "audio_ref": (plan.get("audio") or {}).get("path"),
                    "scene_plan_ref": reference,
                },
                "diff": {"added": source_ids, "changed": [], "removed": []},
                "supervision": {
                    "planner": True,
                    "renderer": True,
                    "audio_understanding_view": True,
                    "paired_edit": False,
                },
            }
        ],
    }


def validate_conversation(value: Mapping[str, Any]) -> None:
    if value.get("schema") != CONVERSATION_SCHEMA:
        raise SpatialStoryError(f"unexpected conversation schema: {value.get('schema')!r}")
    if str(value.get("schema_version")) != CONVERSATION_SCHEMA_VERSION:
        raise SpatialStoryError(
            f"unexpected conversation schema version: {value.get('schema_version')!r}"
        )
    turns = value.get("turns")
    if not isinstance(turns, list) or not turns:
        raise SpatialStoryError("conversation requires at least one turn")
    seen_turns: set[str] = set()
    current_sources: set[str] = set()
    for turn in turns:
        turn_id = str(turn.get("turn_id") or "")
        if not turn_id or turn_id in seen_turns:
            raise SpatialStoryError(f"invalid/duplicate turn_id: {turn_id!r}")
        parent = turn.get("parent_turn_id")
        if parent is not None and parent not in seen_turns:
            raise SpatialStoryError(f"turn {turn_id} refers to unknown parent {parent!r}")
        diff = turn.get("diff") or {}
        added = set(map(str, diff.get("added") or []))
        removed = set(map(str, diff.get("removed") or []))
        changed_entries = diff.get("changed") or []
        changed = {str(item.get("source_id")) for item in changed_entries}
        if added & removed or added & changed or removed & changed:
            raise SpatialStoryError(f"turn {turn_id} has overlapping diff operations")
        if removed - current_sources or changed - current_sources:
            raise SpatialStoryError(f"turn {turn_id} edits a source absent before the turn")
        current_sources.difference_update(removed)
        current_sources.update(added)
        seen_turns.add(turn_id)


def _interpolate_angle(start: float, end: float, fraction: float) -> float:
    delta = ((float(end) - float(start) + 180.0) % 360.0) - 180.0
    return ((float(start) + delta * fraction + 180.0) % 360.0) - 180.0


def _metric_position(position: Mapping[str, Any]) -> tuple[float, float, float, float]:
    azimuth = position.get("azimuth_deg")
    elevation = position.get("elevation_deg")
    distance = position.get("distance_m")
    if azimuth is not None and elevation is not None:
        return float(azimuth), float(elevation), float(distance or 1.0), 1.0
    direction = str(position.get("direction") or "front")
    elevation_label = str(position.get("elevation") or "level")
    azimuth = {
        "front": 0.0,
        "front-left": 45.0,
        "left": 90.0,
        "rear-left": 135.0,
        "behind": 180.0,
        "rear-right": -135.0,
        "right": -90.0,
        "front-right": -45.0,
    }.get(direction, 0.0)
    elevation = {"level": 0.0, "above": 30.0, "below": -30.0}.get(
        elevation_label, 0.0
    )
    return float(azimuth), float(elevation), float(distance or 1.0), 0.5


def _position_at(source: Mapping[str, Any], fraction: float) -> tuple[float, float, float, float]:
    motion = source.get("motion") or {}
    keyframes = sorted(
        motion.get("keyframes") or [], key=lambda item: float(item.get("t_norm", 0.0))
    )
    if not keyframes:
        return 0.0, 0.0, 1.0, 0.0
    if len(keyframes) == 1 or fraction <= float(keyframes[0].get("t_norm", 0.0)):
        return _metric_position(keyframes[0].get("position") or {})
    if fraction >= float(keyframes[-1].get("t_norm", 1.0)):
        return _metric_position(keyframes[-1].get("position") or {})
    for left, right in zip(keyframes, keyframes[1:]):
        left_t = float(left.get("t_norm", 0.0))
        right_t = float(right.get("t_norm", 1.0))
        if fraction > right_t:
            continue
        local = 0.0 if math.isclose(left_t, right_t) else (fraction - left_t) / (right_t - left_t)
        a0, e0, d0, q0 = _metric_position(left.get("position") or {})
        a1, e1, d1, q1 = _metric_position(right.get("position") or {})
        return (
            _interpolate_angle(a0, a1, local),
            e0 + (e1 - e0) * local,
            d0 + (d1 - d0) * local,
            min(q0, q1),
        )
    return _metric_position(keyframes[-1].get("position") or {})


def compile_source_tracks(
    plan: Mapping[str, Any],
    *,
    num_frames: int,
    max_sources: int = 4,
    max_distance_m: float = 20.0,
) -> dict[str, Any]:
    """Compile a ScenePlan into deterministic per-source, per-frame controls.

    Event semantics remain in the plan tokens.  These tracks carry only the
    time-aligned controls needed by the renderer, so a predicted plan can be
    edited and recompiled without asking a flow model to rediscover geometry.
    """

    if num_frames <= 0 or max_sources <= 0:
        raise SpatialStoryError("num_frames and max_sources must be positive")
    sources = ((plan.get("scene") or {}).get("sources") or [])
    if len(sources) > max_sources:
        raise SpatialStoryError(
            f"ScenePlan has {len(sources)} sources, exceeding max_sources={max_sources}"
        )
    duration = float((plan.get("audio") or {}).get("duration_sec") or 0.0)
    if duration <= 0:
        raise SpatialStoryError("ScenePlan audio.duration_sec must be positive")

    tracks = torch.zeros(max_sources, len(SOURCE_TRACK_FEATURES), num_frames)
    source_mask = torch.zeros(max_sources, dtype=torch.bool)
    source_ids: list[str | None] = [None] * max_sources
    frame_times = (torch.arange(num_frames, dtype=torch.float32) + 0.5) * duration / num_frames

    occupied_slots: set[int] = set()
    for source_index, source in enumerate(sources):
        source_id = str(source.get("source_id") or f"source_{source_index}")
        suffix = None
        if source_id.startswith("source_") and source_id[7:].isdigit():
            suffix = int(source_id[7:])
        elif source_id.startswith("s") and source_id[1:].isdigit():
            suffix = int(source_id[1:])
        slot = source_index if suffix is None else suffix
        if not 0 <= slot < max_sources:
            raise SpatialStoryError(
                f"source {source_id!r} maps outside max_sources={max_sources}"
            )
        if slot in occupied_slots:
            raise SpatialStoryError(f"duplicate source-control slot: {slot}")
        occupied_slots.add(slot)
        source_mask[slot] = True
        source_ids[slot] = source_id
        activity = source.get("activity") or {}
        onset = activity.get("onset_sec")
        offset = activity.get("offset_sec")
        quality_name = str(activity.get("quality") or "")
        activity_quality = {
            "source_annotation": 1.0,
            "paired_edit_exact": 1.0,
            "foa_energy_derived_single_source": 0.7,
            "resolved_from_dry_crop": 0.8,
            "inactive_in_crop": 0.7,
        }.get(quality_name, 0.25)
        explicitly_inactive = (
            activity.get("active_in_crop") is False
            or quality_name == "inactive_in_crop"
        )
        if explicitly_inactive:
            onset, offset = 0.0, 0.0
        elif onset is None or offset is None or float(offset) <= float(onset):
            render_window = activity.get("render_window_sec") or [0.0, duration]
            onset, offset = float(render_window[0]), float(render_window[1])
            activity_quality = 0.25
        onset = max(0.0, float(onset))
        offset = min(duration, float(offset))
        active = (frame_times >= onset) & (frame_times < offset)
        raw_gain_db = (source.get("acoustics") or {}).get("gain_db")
        gain_db = 0.0 if raw_gain_db is None else float(raw_gain_db)
        gain_linear = 10.0 ** (gain_db / 20.0)
        time_basis = str((source.get("motion") or {}).get("time_basis") or "full_clip")

        for frame in range(num_frames):
            t = float(frame_times[frame])
            if time_basis == "activity_window":
                denominator = max(1e-6, offset - onset)
                fraction = min(1.0, max(0.0, (t - onset) / denominator))
            else:
                fraction = min(1.0, max(0.0, t / duration))
            azimuth, elevation, distance, geometry_quality = _position_at(source, fraction)
            azimuth_rad = math.radians(azimuth)
            elevation_rad = math.radians(elevation)
            cos_el = math.cos(elevation_rad)
            active_value = float(active[frame])
            tracks[slot, 0, frame] = active_value
            # Geometry and acoustic controls have no renderer meaning outside
            # the source's activity window. Keeping a direction/gain alive in
            # silent frames made the old control prefix internally
            # contradictory (activity=0 while every other feature still
            # described an emitting source), and encouraged pre-onset/tail
            # leakage. Activity confidence intentionally remains populated so
            # zero here means a known inactive interval, not missing metadata.
            tracks[slot, 1, frame] = (
                math.cos(azimuth_rad) * cos_el * active_value
            )
            tracks[slot, 2, frame] = (
                math.sin(azimuth_rad) * cos_el * active_value
            )
            tracks[slot, 3, frame] = math.sin(elevation_rad) * active_value
            tracks[slot, 4, frame] = (
                1.0 / max(1.0, min(max_distance_m, distance)) * active_value
            )
            tracks[slot, 5, frame] = gain_linear * active_value
            tracks[slot, 6, frame] = geometry_quality * active_value
            tracks[slot, 7, frame] = activity_quality

    return {
        "schema": TRACK_SCHEMA,
        "schema_version": TRACK_SCHEMA_VERSION,
        "features": list(SOURCE_TRACK_FEATURES),
        "tracks": tracks,
        "source_mask": source_mask,
        "source_ids": source_ids,
        "aligned": True,
    }


def source_tracks_to_mixture_trajectory(
    tracks: torch.Tensor,
    *,
    fourth_component: str = "geometric_dispersion",
) -> torch.Tensor:
    """Collapse deterministic source tracks to a 4-D geometric anchor.

    ``tracks`` is ``[source, feature, time]`` and the result is ``[time, 4]``
    with ``[mx, my, mz, fourth]``.  Each direction is weighted by the expected
    received energy ``(gain / distance)^2``.  Dividing the vector sum by total
    energy returns the mixture moment and damps it by geometric concentration.

    ``fourth_component='geometric_dispersion'`` preserves the original
    ``1 - ||m||`` contract.  ``'received_level'`` instead returns
    ``sqrt(sum energy)`` clipped to ``[0, 1]``.  The latter is non-redundant,
    distinguishes silence from equal-energy directional cancellation, and is
    the preferred control for a frame-aligned Renderer adapter.

    This is deliberately deterministic and content-independent. ScenePlan
    tokens still carry source identity and event semantics.  Geometric
    dispersion is not physical DirAC diffuseness: it cannot model room
    reflections, phase, spectrum, or source-level dry-audio energy.
    """

    value = torch.as_tensor(tracks)
    if value.ndim != 3 or value.shape[1] != len(SOURCE_TRACK_FEATURES):
        raise ValueError(
            "source tracks must be [source, feature, time] with "
            f"{len(SOURCE_TRACK_FEATURES)} features, got {tuple(value.shape)}"
        )
    if not value.is_floating_point():
        value = value.float()
    if fourth_component not in {"geometric_dispersion", "received_level"}:
        raise ValueError(
            "fourth_component must be 'geometric_dispersion' or "
            f"'received_level', got {fourth_component!r}"
        )

    activity = value[:, 0].clamp(0.0, 1.0)
    direction = value[:, 1:4]
    inverse_distance = value[:, 4].clamp_min(0.0)
    gain = value[:, 5].clamp_min(0.0)
    received_amplitude = activity * inverse_distance * gain
    energy = received_amplitude.square()
    total_energy = energy.sum(dim=0)
    vector = (energy[:, None] * direction).sum(dim=0)
    vector_norm = vector.norm(dim=0)
    denominator = total_energy.clamp_min(torch.finfo(value.dtype).eps)
    damped_direction = (vector / denominator[None]).transpose(0, 1)
    coherence = (vector_norm / denominator).clamp(0.0, 1.0)
    active = total_energy > 0
    damped_direction = torch.where(
        active[:, None], damped_direction, torch.zeros_like(damped_direction)
    )
    if fourth_component == "geometric_dispersion":
        fourth = torch.where(
            active,
            1.0 - coherence,
            torch.ones_like(coherence),
        )
    else:
        fourth = total_energy.sqrt().clamp(0.0, 1.0)
    return torch.cat((damped_direction, fourth[:, None]), dim=-1)


__all__ = [
    "CONVERSATION_SCHEMA",
    "CONVERSATION_SCHEMA_VERSION",
    "SOURCE_TRACK_FEATURES",
    "SpatialStoryError",
    "compile_source_tracks",
    "conversation_id_for_plan",
    "diff_scene_plans",
    "persistent_source_id",
    "plan_readiness",
    "seed_conversation",
    "source_tracks_to_mixture_trajectory",
    "validate_conversation",
    "with_persistent_source_ids",
]
