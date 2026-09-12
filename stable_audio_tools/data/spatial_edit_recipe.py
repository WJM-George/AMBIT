"""Deterministic render recipes for high-fidelity Spatial-CoT edit pairs.

AudioChat-style conversation JSON describes *what* is in every turn.  Exact
paired audio additionally needs a reproducible render recipe: dry crop, room,
microphone, motion, activity, gain, renderer version, cached source tracks, and
mix mastering.  This module keeps those concerns explicit and non-destructive.

The invariant for an edit family is simple:

* every turn is synthesized from dry-source tracks, never from the previous
  turn's FOA waveform;
* unchanged source render signatures are byte-for-byte reusable;
* gain-only edits reuse the same source track and change only the mix gain;
* spatial/timing edits rerender only the changed source;
* semantic edits use a different dry source;
* every turn is mixed with one family-level master gain.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from .scene_plan import semantic_caption_render_from_sources
from .spatial_caption_templates import SemanticCaptionRender
from .spatial_story import (
    SpatialStoryError,
    diff_scene_plans,
    with_persistent_source_ids,
)


RECIPE_SCHEMA = "stable_audio_tools.spatial_edit_family"
RECIPE_VERSION = "1.5"
RENDERER_NAME = "stable_audio_tools.foa_edit_renderer"
RENDERER_VERSION = "1.3"
FOA_LAYOUT = "WYZX_ACN_SN3D"
STATE_RENDER_INPUT = "dry_source_tracks"
MIN_PLAYBACK_RMS = 1e-4
DEFAULT_SOURCE_LOUDNESS = {
    "policy": "rms_target_peak_limited",
    "policy_version": 1,
    "target_mono_rms": 0.05,
    "peak_ceiling": 0.95,
    "max_gain_db": 30.0,
    "minimum_input_rms": MIN_PLAYBACK_RMS,
}
_SPEECH_LABEL = re.compile(
    r"\b(?:speech|speaking|conversation|narration|monologue|dialogue|"
    r"whisper|whispering|babble|babbling|chatter|talk|talking)\b",
    flags=re.IGNORECASE,
)
_SILENT_EVENT_NAMES = {
    "no audible sound",
    "no sound",
    "no sounds",
    "silence",
    "silent",
}

SUPPORTED_EDIT_TYPES = {
    "move_source",
    "change_gain",
    "change_activity",
    "remove_source",
    "add_source",
    "replace_source",
}


def is_speech_source(source: Mapping[str, Any]) -> bool:
    """Conservatively detect any source that contains linguistic speech.

    AudioSet-style labels are often composites such as ``Speech, Music`` or
    ``Male speech, man speaking``.  Exact equality with ``speech`` would let
    those clips enter a non-speech donor pool and create acoustic
    speech+speech mixtures even though the structured source count looked
    valid.  Singing and non-linguistic vocalizations remain music/sound unless
    the label also explicitly names speech.
    """

    if str((source.get("content") or {}).get("transcript") or "").strip():
        return True
    event = source.get("event") or {}
    text = " ".join(
        str(event.get(field) or "") for field in ("category", "label")
    )
    return bool(_SPEECH_LABEL.search(text))


def is_silent_source(source: Mapping[str, Any]) -> bool:
    """Detect labels that explicitly describe a no-sound source.

    AudioSet stores multiple ontology labels as one comma-separated string, so
    ``Silence, Engine`` must be rejected just like the exact ``Silence`` class.
    Free-form captions such as ``silent for a moment, followed by a crash`` are
    intentionally retained: they describe audio with an audible event rather
    than declaring the whole source silent.
    """

    event = source.get("event") or {}
    category = " ".join(
        str(event.get("category") or "").strip().lower().split()
    )
    label = " ".join(str(event.get("label") or "").strip().lower().split())
    if category in _SILENT_EVENT_NAMES or label in _SILENT_EVENT_NAMES:
        return True
    dataset = str(event.get("source_dataset") or "").strip().lower()
    if dataset == "audioset":
        labels = {
            " ".join(token.strip().lower().split())
            for token in label.split(",")
            if token.strip()
        }
        return bool(labels & _SILENT_EVENT_NAMES)
    return False


def _source_loudness_policy(
    value: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    policy = dict(DEFAULT_SOURCE_LOUDNESS if value is None else value)
    if policy.get("policy") != "rms_target_peak_limited":
        raise SpatialStoryError(
            f"unsupported source loudness policy: {policy.get('policy')}"
        )
    if int(policy.get("policy_version", -1)) != 1:
        raise SpatialStoryError("unsupported source loudness policy version")
    for key in (
        "target_mono_rms",
        "peak_ceiling",
        "max_gain_db",
        "minimum_input_rms",
    ):
        policy[key] = float(policy[key])
        if not math.isfinite(policy[key]):
            raise SpatialStoryError(f"source loudness {key} must be finite")
    if not 0.0 < policy["target_mono_rms"] < 1.0:
        raise SpatialStoryError("source loudness target_mono_rms must lie in (0,1)")
    if not 0.0 < policy["peak_ceiling"] <= 1.0:
        raise SpatialStoryError("source loudness peak_ceiling must lie in (0,1]")
    if policy["max_gain_db"] < 0.0:
        raise SpatialStoryError("source loudness max_gain_db must be non-negative")
    if policy["minimum_input_rms"] < MIN_PLAYBACK_RMS:
        raise SpatialStoryError(
            f"source loudness minimum_input_rms must be >= {MIN_PLAYBACK_RMS}"
        )
    return policy


def _source_loudness(
    *,
    mono_peak: float,
    mono_rms: float,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = _source_loudness_policy(policy)
    if mono_rms < float(normalized["minimum_input_rms"]):
        raise SpatialStoryError(
            f"dry playback window is effectively silent: rms={mono_rms:.3g}"
        )
    gain_cap = 10.0 ** (float(normalized["max_gain_db"]) / 20.0)
    rms_gain = float(normalized["target_mono_rms"]) / mono_rms
    peak_gain = float(normalized["peak_ceiling"]) / max(mono_peak, 1.0e-12)
    gain = min(rms_gain, peak_gain, gain_cap)
    if not math.isfinite(gain) or gain <= 0.0:
        raise SpatialStoryError("source loudness produced an invalid gain")
    return {
        **normalized,
        "gain_linear": gain,
        "gain_db": 20.0 * math.log10(gain),
        "post_gain_mono_peak": mono_peak * gain,
        "post_gain_mono_rms": mono_rms * gain,
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_digest(*values: Any, size: int = 12) -> str:
    return hashlib.blake2b(_canonical_json(values), digest_size=size).hexdigest()


def recipe_fingerprint(value: Mapping[str, Any]) -> str:
    """Hash render-affecting recipe content, excluding materialized outputs."""

    normalized = copy.deepcopy(dict(value))
    normalized.pop("outputs", None)
    normalized.pop("render_status", None)
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def _source_seed(family_seed: int, source_id: str) -> int:
    return int(stable_digest(family_seed, source_id, size=8), 16) & 0x7FFFFFFF


def _resolved_dry_crop(
    path: str,
    *,
    duration_sec: float,
    seed: int,
    preserve_source_start: bool = False,
) -> dict[str, Any]:
    """Resolve a deterministic crop in native samples without loading audio."""

    import soundfile as sf

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    info = sf.info(str(source_path))
    source_stat = source_path.stat()
    native_rate = int(info.samplerate)
    native_frames = int(info.frames)
    requested = max(1, int(round(float(duration_sec) * native_rate)))
    if native_frames > requested:
        start = (
            0
            if preserve_source_start
            else random.Random(int(seed)).randint(0, native_frames - requested)
        )
        count = requested
    else:
        start = 0
        count = native_frames
    return {
        "path": str(source_path),
        "native_sample_rate": native_rate,
        "native_num_frames": native_frames,
        "crop_start_native_sample": start,
        "crop_num_native_samples": count,
        "requested_native_samples": requested,
        "pad_after_native_samples": max(0, requested - count),
        "channels_in_file": int(info.channels),
        "file_size_bytes": int(source_stat.st_size),
        "file_mtime_ns": int(source_stat.st_mtime_ns),
        "crop_seed": int(seed),
    }


def _resolved_playback_window(
    dry: Mapping[str, Any],
    *,
    activity_duration_sec: float,
    preserve_source_start: bool,
    source_loudness: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the exact source-internal segment placed in an activity window.

    ``activity`` says *when* a source is audible in the generated scene.  It
    does not say which part of a sparse 10-second dry clip should be played.
    Keeping that second offset implicit used to turn valid event clips into
    silence whenever their energy occurred after the requested activity
    duration.  The selected offset is therefore first-class recipe state.

    Speech starts at the beginning to preserve transcript order.  Sound and
    music use the contiguous window with maximum mono energy inside the
    deterministic dry crop.  Ties resolve to the earliest sample via argmax.
    """

    import soundfile as sf

    native_rate = int(dry["native_sample_rate"])
    crop_count = int(dry["crop_num_native_samples"])
    requested = max(1, int(round(float(activity_duration_sec) * native_rate)))
    playback_count = min(crop_count, requested)
    if playback_count <= 0:
        raise SpatialStoryError("dry crop has no playable samples")

    audio, actual_rate = sf.read(
        str(dry["path"]),
        start=int(dry["crop_start_native_sample"]),
        stop=int(dry["crop_start_native_sample"]) + crop_count,
        always_2d=True,
        dtype="float32",
    )
    if int(actual_rate) != native_rate:
        raise SpatialStoryError(
            f"dry source sample-rate changed: {actual_rate} != {native_rate}"
        )
    if len(audio) != crop_count:
        raise SpatialStoryError(
            f"dry source crop changed: read {len(audio)} != expected {crop_count}"
        )
    mono = audio.mean(axis=1, dtype=np.float32)

    if preserve_source_start or playback_count == crop_count:
        offset = 0
        policy = "source_start" if preserve_source_start else "complete_crop"
    else:
        squared = np.square(mono, dtype=np.float64)
        cumulative = np.empty(len(squared) + 1, dtype=np.float64)
        cumulative[0] = 0.0
        np.cumsum(squared, out=cumulative[1:])
        energies = cumulative[playback_count:] - cumulative[:-playback_count]
        offset = int(np.argmax(energies))
        policy = "maximum_rms_window"

    selected = mono[offset : offset + playback_count]
    peak = float(np.max(np.abs(selected)))
    rms = float(np.sqrt(np.mean(np.square(selected, dtype=np.float64))))
    if not math.isfinite(peak) or not math.isfinite(rms):
        raise SpatialStoryError("dry playback window contains non-finite samples")
    if rms < float(_source_loudness_policy(source_loudness)["minimum_input_rms"]):
        raise SpatialStoryError(
            f"dry playback window is effectively silent: rms={rms:.3g}, "
            f"path={dry['path']}"
        )
    return {
        "offset_in_crop_native_sample": offset,
        "num_native_samples": playback_count,
        "requested_native_samples": requested,
        "pad_after_native_samples": max(0, requested - playback_count),
        "selection_policy": policy,
        "selection_version": 1,
        "mono_peak": peak,
        "mono_rms": rms,
        "loudness": _source_loudness(
            mono_peak=peak,
            mono_rms=rms,
            policy=source_loudness,
        ),
    }


def _safe_room(plan: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    """Complete missing runtime room fields deterministically.

    Existing dimensions/RT60/class are retained.  The old rendered FOA did not
    persist the microphone or image-source order, so the regenerated branch
    deliberately creates and records a new deterministic choice.
    """

    room = copy.deepcopy(((plan.get("scene") or {}).get("room") or {}))
    dimensions = room.get("dimensions_m")
    if not isinstance(dimensions, Sequence) or len(dimensions) != 3:
        dimensions = [6.0, 5.0, 3.0]
    dimensions = [max(1.5, float(value)) for value in dimensions]
    rng = random.Random(int(seed))
    microphone = [
        rng.uniform(0.45 * dimensions[0], 0.55 * dimensions[0]),
        rng.uniform(0.45 * dimensions[1], 0.55 * dimensions[1]),
        rng.uniform(1.2, max(1.21, min(1.8, dimensions[2] - 0.35))),
    ]
    free_field = bool(room.get("free_field") or room.get("type") == "outdoor")
    rt60 = float(room.get("rt60_s") or (0.35 if free_field else 0.5))
    if free_field:
        max_order = 0
    else:
        # Bounded, deterministic and affordable.  The exact value is part of
        # the recipe, unlike the legacy manifest.
        max_order = max(3, min(10, int(round(3.0 + 6.0 * min(rt60, 1.5) / 1.5))))
    return {
        "type": room.get("type") or "shoebox",
        "description": room.get("description"),
        "dimensions_m": dimensions,
        "rt60_s": rt60,
        "free_field": free_field,
        "microphone_xyz_m": microphone,
        "max_order": max_order,
        "quality": "regenerated_exact_recipe",
    }


def _source_activity(
    source: Mapping[str, Any],
    *,
    duration_sec: float,
    dry_duration_sec: float,
) -> dict[str, Any]:
    activity = source.get("activity") or {}
    onset = activity.get("onset_sec")
    offset = activity.get("offset_sec")
    annotated = (
        onset is not None
        and offset is not None
        and float(offset) > float(onset)
    )
    if not annotated:
        onset = 0.0
        offset = min(float(duration_sec), float(dry_duration_sec))
    onset = min(float(duration_sec), max(0.0, float(onset)))
    offset = min(float(duration_sec), max(onset, float(offset)))
    return {
        "onset_sec": onset,
        "offset_sec": offset,
        "quality": (
            str(activity.get("quality") or "source_annotation")
            if annotated
            else "resolved_from_dry_crop"
        ),
    }


def _motion(source: Mapping[str, Any]) -> dict[str, Any]:
    motion = copy.deepcopy(source.get("motion") or {})
    keyframes = sorted(
        motion.get("keyframes") or [],
        key=lambda item: float(item.get("t_norm", 0.0)),
    )
    if not keyframes:
        keyframes = [
            {
                "t_norm": 0.0,
                "position": {
                    "azimuth_deg": 0.0,
                    "elevation_deg": 0.0,
                    "distance_m": 1.0,
                    "geometry_quality": "fallback_front",
                },
            }
        ]
    motion_type = "static" if len(keyframes) == 1 else (
        "linear" if len(keyframes) == 2 else "keyframed"
    )
    return {
        "type": motion_type,
        "time_basis": "activity_window",
        "interpolation": "linear_shortest_azimuth_arc",
        "keyframes": keyframes,
    }


def _source_recipe(
    source: Mapping[str, Any],
    *,
    family_seed: int,
    duration_sec: float,
    source_loudness: Mapping[str, Any],
) -> dict[str, Any]:
    if is_silent_source(source):
        event = source.get("event") or {}
        raise SpatialStoryError(
            f"source {source.get('source_id')} is explicitly silent: "
            f"{event.get('label') or event.get('category')}"
        )
    source_id = str(source["source_id"])
    source_uid = str(source.get("source_uid") or source_id)
    content = copy.deepcopy(source.get("content") or {})
    dry_path = content.get("source_audio_path")
    if not isinstance(dry_path, str) or not dry_path:
        raise SpatialStoryError(f"source {source_id} has no direct dry-audio path")
    seed = _source_seed(family_seed, source_uid)
    original_transcript = str(content.get("transcript") or "").strip() or None
    dry = _resolved_dry_crop(
        dry_path,
        duration_sec=duration_sec,
        seed=seed,
        preserve_source_start=is_speech_source(source),
    )
    transcript_was_truncated = bool(
        original_transcript
        and int(dry["native_num_frames"]) > int(dry["crop_num_native_samples"])
    )
    conditioning_transcript = None if transcript_was_truncated else original_transcript
    dry_duration = float(dry["crop_num_native_samples"]) / float(
        dry["native_sample_rate"]
    )
    activity = _source_activity(
        source,
        duration_sec=duration_sec,
        dry_duration_sec=dry_duration,
    )
    preserve_source_start = is_speech_source(source)
    playback = _resolved_playback_window(
        dry,
        activity_duration_sec=float(activity["offset_sec"])
        - float(activity["onset_sec"]),
        preserve_source_start=preserve_source_start,
        source_loudness=source_loudness,
    )
    return {
        "source_id": source_id,
        "source_uid": source_uid,
        "event": copy.deepcopy(source.get("event") or {}),
        "content": {
            "transcript": conditioning_transcript,
            "transcript_quality": (
                "omitted_unaligned_long_source"
                if transcript_was_truncated
                else "exact_full_source"
                if conditioning_transcript
                else "not_available"
            ),
            "speaker_id": content.get("speaker_id"),
            "source_audio_id": content.get("source_audio_id"),
            "source_audio_sha256": content.get("source_audio_sha256"),
        },
        "dry_audio": dry,
        "playback": playback,
        "activity": activity,
        "motion": _motion(source),
        "gain_db": float((source.get("acoustics") or {}).get("gain_db", 0.0)),
    }


def source_render_signature(
    recipe: Mapping[str, Any], source: Mapping[str, Any]
) -> str:
    """Hash fields that require a distinct FOA pre-mix track.

    ``gain_db`` is intentionally excluded: a gain-only edit reuses the cached
    track.  Semantic changes necessarily change ``dry_audio``.
    """

    dry = source["dry_audio"]
    playback = source["playback"]
    activity = source["activity"]
    keyframes = (source.get("motion") or {}).get("keyframes") or []
    room = recipe["room"]
    payload = {
        "renderer": recipe["renderer"],
        "sample_rate": recipe["audio"]["sample_rate"],
        "num_samples": recipe["audio"]["num_samples"],
        # Hash the values actually consumed by the renderer. Annotation-only
        # quality strings must not create a redundant, acoustically identical
        # source track.
        "room": {
            "dimensions_m": room["dimensions_m"],
            "rt60_s": room["rt60_s"],
            "microphone_xyz_m": room["microphone_xyz_m"],
            "max_order": room["max_order"],
        },
        "dry_audio": {
            key: dry.get(key)
            for key in (
                "path",
                "native_sample_rate",
                "crop_start_native_sample",
                "crop_num_native_samples",
                "file_size_bytes",
                "file_mtime_ns",
            )
        },
        "playback": {
            "offset_in_crop_native_sample": playback[
                "offset_in_crop_native_sample"
            ],
            "num_native_samples": playback["num_native_samples"],
            "loudness": playback["loudness"],
        },
        "activity": {
            "onset_sec": activity["onset_sec"],
            "offset_sec": activity["offset_sec"],
        },
        "keyframes": [
            {
                "t_norm": keyframe.get("t_norm"),
                "position": {
                    key: (keyframe.get("position") or {}).get(key)
                    for key in ("azimuth_deg", "elevation_deg", "distance_m")
                },
            }
            for keyframe in keyframes
        ],
    }
    return f"trk_{stable_digest(payload, size=16)}"


def _position_summary(position: Mapping[str, Any]) -> str:
    def number(name: str, default: float) -> float:
        value = position.get(name)
        return default if value is None else float(value)

    return (
        f"az={number('azimuth_deg', 0.0):.1f}deg,"
        f"el={number('elevation_deg', 0.0):.1f}deg,"
        f"d={max(0.0, number('distance_m', 1.0)):.2f}m"
    )


def _describe_recipe(recipe: Mapping[str, Any]) -> str:
    """Return a compact full-state description, not a stale edit caption."""

    room = recipe.get("room") or {}
    room_name = room.get("description") or room.get("type") or "room"
    clauses = [f"Room: {room_name}."]
    for source in recipe.get("sources") or []:
        source_id = str(source.get("source_id") or "source")
        event = source.get("event") or {}
        label = event.get("label") or event.get("category") or "audio event"
        activity = source.get("activity") or {}
        onset = float(activity.get("onset_sec") or 0.0)
        offset = float(activity.get("offset_sec") or onset)
        keyframes = (source.get("motion") or {}).get("keyframes") or []
        if len(keyframes) <= 1:
            position = (keyframes[0].get("position") if keyframes else {}) or {}
            motion = f"static at {_position_summary(position)}"
        else:
            start = (keyframes[0].get("position") or {})
            end = (keyframes[-1].get("position") or {})
            motion = (
                f"moves from {_position_summary(start)} to "
                f"{_position_summary(end)}"
            )
        transcript = (source.get("content") or {}).get("transcript")
        semantic = str(label)
        if transcript and str(label).strip().lower() == "speech":
            semantic = f"speech saying {json.dumps(str(transcript), ensure_ascii=False)}"
        clauses.append(
            f"{source_id}: {semantic}; active {onset:.2f}-{offset:.2f}s; "
            f"{motion}; gain={float(source.get('gain_db', 0.0)):.1f}dB."
        )
    return " ".join(clauses)


def describe_recipe(recipe: Mapping[str, Any]) -> str:
    """Public full-state prompt renderer for exact or intervened recipes."""

    return _describe_recipe(recipe)


def _semantic_caption(recipe: Mapping[str, Any]) -> SemanticCaptionRender:
    """Describe audible content without duplicating ScenePlan controls.

    Spatial position, activity time, gain, room geometry, and trajectory are
    deliberately absent.  They are authoritative in the discrete ScenePlan
    and its deterministic tracks.  This separation lets an edit instruction
    change *where/when* while the frozen-Qwen prefix continues to identify
    *what* should be rendered.
    """

    return semantic_caption_render_from_sources(
        recipe.get("sources") or [],
        # One family keeps one linguistic style through all four turns.  Edits
        # therefore change source content/state rather than surface wording.
        template_key=recipe.get("family_id"),
    )


def _scene_plan_motion(
    source: Mapping[str, Any], *, duration_sec: float
) -> dict[str, Any]:
    """Normalize recipe-local activity-relative motion to clip time.

    SpatialPlan codec v1 has one canonical time basis: keyframe ``t_norm`` is
    relative to the full clip. Render recipes may use the more convenient
    activity-relative basis internally; this conversion preserves the same
    start/end instants without adding a second ambiguous grammar.
    """

    motion = copy.deepcopy(source.get("motion") or {})
    if str(motion.get("time_basis") or "full_clip") != "activity_window":
        motion["time_basis"] = "full_clip"
        return motion
    activity = source.get("activity") or {}
    onset = max(0.0, float(activity.get("onset_sec") or 0.0))
    offset = min(
        float(duration_sec),
        max(onset, float(activity.get("offset_sec") or onset)),
    )
    start_norm = onset / max(1e-6, float(duration_sec))
    stop_norm = offset / max(1e-6, float(duration_sec))
    for keyframe in motion.get("keyframes") or []:
        local = min(1.0, max(0.0, float(keyframe.get("t_norm") or 0.0)))
        keyframe["t_norm"] = start_norm + local * (stop_norm - start_norm)
    motion["time_basis"] = "full_clip"
    motion["timing_quality"] = "paired_edit_exact_clip_time"
    return motion


def _turn_scene_plan(recipe: Mapping[str, Any]) -> dict[str, Any]:
    sources = []
    for source in recipe["sources"]:
        sources.append(
            {
                "source_id": source["source_id"],
                "event": copy.deepcopy(source["event"]),
                "content": copy.deepcopy(source["content"]),
                "activity": copy.deepcopy(source["activity"]),
                "motion": _scene_plan_motion(
                    source,
                    duration_sec=float(recipe["audio"]["duration_sec"]),
                ),
                "acoustics": {"gain_db": source["gain_db"]},
            }
        )
    return {
        "schema": "stable_audio_tools.spatial_scene_plan",
        "schema_version": "1.2-paired",
        "sample_id": f"{recipe['family_id']}_{recipe['turn_id']}",
        "caption": recipe["scene_description"],
        "audio": {
            "duration_sec": recipe["audio"]["duration_sec"],
            "sample_rate": recipe["audio"]["sample_rate"],
            "spatial_format": "foa",
            "channel_layout": FOA_LAYOUT,
        },
        "mix": {
            "type": "single" if len(sources) == 1 else "mixture",
            "num_sources": len(sources),
        },
        "scene": {"room": copy.deepcopy(recipe["room"]), "sources": sources},
    }


def _refresh_turn(recipe: dict[str, Any]) -> dict[str, Any]:
    recipe["scene_description"] = _describe_recipe(recipe)
    caption = _semantic_caption(recipe)
    recipe["semantic_caption"] = caption.text
    recipe["semantic_caption_metadata"] = caption.metadata()
    if str(recipe.get("task")) == "text_to_spatial_audio":
        # The creation prompt must describe the actual executable recipe.  In
        # particular, a long speech source whose unaligned transcript was
        # honestly omitted must not retain that stale full transcript here.
        recipe["instruction"] = recipe["scene_description"]
    recipe["planner_prompt"] = str(
        recipe.get("instruction") or recipe["scene_description"]
    )
    recipe["understanding_prompt"] = (
        "Recover the complete source state, activity, and metric 3-D "
        "trajectories from this FOA audio."
    )
    recipe["scene_plan"] = _turn_scene_plan(recipe)
    recipe["recipe_fingerprint"] = recipe_fingerprint(recipe)
    return recipe


def build_base_recipe(
    plan: Mapping[str, Any],
    *,
    seed: int,
    sample_rate: int = 44_100,
    max_sources: int = 4,
    family_id: Optional[str] = None,
    source_loudness: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Create a fully resolved generation recipe from a direct-dry ScenePlan."""

    family_id = str(
        family_id or f"editfam_{stable_digest(plan.get('sample_id'), seed)}"
    )
    persistent = with_persistent_source_ids(plan, conversation_id=family_id)
    plan_sources = ((persistent.get("scene") or {}).get("sources") or [])
    if not 1 <= len(plan_sources) <= max_sources:
        raise SpatialStoryError(
            f"base plan requires 1..{max_sources} sources, found {len(plan_sources)}"
        )
    duration = float((persistent.get("audio") or {}).get("duration_sec") or 0.0)
    if duration <= 0:
        raise SpatialStoryError("base plan requires a positive audio duration")
    num_samples = max(1, int(round(duration * int(sample_rate))))
    duration = num_samples / float(sample_rate)
    room_seed = int(stable_digest(seed, family_id, "room", size=8), 16)
    loudness = _source_loudness_policy(source_loudness)
    recipe = {
        "schema": RECIPE_SCHEMA,
        "schema_version": RECIPE_VERSION,
        "family_id": family_id,
        "turn_id": "turn_000",
        "parent_turn_id": None,
        "task": "text_to_spatial_audio",
        "instruction": persistent.get("caption"),
        "reference_caption": persistent.get("caption"),
        "scene_description": "",
        "reference_audio": (persistent.get("audio") or {}).get("path"),
        "seed": int(seed),
        "audio": {
            "sample_rate": int(sample_rate),
            "num_samples": num_samples,
            "duration_sec": duration,
            "spatial_format": "foa",
            "channel_layout": FOA_LAYOUT,
        },
        "renderer": {
            "name": RENDERER_NAME,
            "version": RENDERER_VERSION,
            "backend": "pyroomacoustics",
            "dynamic_mode": "rir_keyframe_crossfade",
            "dry_resampler": "scipy_resample_poly",
        },
        "render_contract": {
            "state_input": STATE_RENDER_INPUT,
            "uses_previous_foa": False,
            "independent_state_mix": True,
            "unchanged_track_cache": "deterministic_exact_reuse",
            "source_loudness": loudness,
        },
        "room": _safe_room(persistent, seed=room_seed),
        "mix": {
            "family_peak_target": 0.9,
            "family_master_gain_linear": None,
            "limiter": "none",
            "normalization_scope": "edit_family",
        },
        "sources": [
            _source_recipe(
                source,
                family_seed=int(seed),
                duration_sec=duration,
                source_loudness=loudness,
            )
            for source in plan_sources
        ],
        "edit": {
            "type": "create",
            "target_source_id": None,
            "changed_fields": [],
        },
    }
    return _refresh_turn(recipe)


def _rotate_motion(source: dict[str, Any], delta_deg: float) -> None:
    for keyframe in source["motion"]["keyframes"]:
        position = keyframe.setdefault("position", {})
        azimuth = float(position.get("azimuth_deg") or 0.0)
        position["azimuth_deg"] = (
            (azimuth + float(delta_deg) + 180.0) % 360.0
        ) - 180.0
        position["geometry_quality"] = "paired_edit_exact"


def make_edit_turn(
    previous: Mapping[str, Any],
    *,
    turn_index: int,
    edit_type: str,
    seed: int,
    donor_source: Optional[Mapping[str, Any]] = None,
    max_sources: int = 4,
) -> dict[str, Any]:
    """Apply one deterministic edit while preserving source identity/crops."""

    if edit_type not in SUPPORTED_EDIT_TYPES:
        raise SpatialStoryError(f"unsupported edit type: {edit_type}")
    if turn_index <= 0:
        raise SpatialStoryError("edit turn_index must be positive")
    result = copy.deepcopy(dict(previous))
    result.pop("scene_plan", None)
    result.pop("recipe_fingerprint", None)
    result["turn_id"] = f"turn_{turn_index:03d}"
    result["parent_turn_id"] = str(previous["turn_id"])
    result["task"] = "spatial_audio_edit"
    rng = random.Random(int(seed))
    sources = result["sources"]
    if not sources:
        raise SpatialStoryError("cannot edit an empty source list")
    if edit_type == "change_activity":
        scene_duration = float(result["audio"]["duration_sec"])
        minimum_shift = min(1.0, max(0.25, 0.1 * scene_duration))
        shiftable = []
        for index, candidate in enumerate(sources):
            activity = candidate["activity"]
            left_slack = float(activity["onset_sec"])
            right_slack = scene_duration - float(activity["offset_sec"])
            if max(left_slack, right_slack) >= minimum_shift:
                shiftable.append((index, left_slack, right_slack))
        if not shiftable:
            raise SpatialStoryError(
                "no source activity can be shifted without truncating its content"
            )
        target_index, left_slack, right_slack = rng.choice(shiftable)
    elif edit_type == "replace_source":
        if donor_source is None:
            raise SpatialStoryError("replace_source requires a resolved donor source")

        def semantic_key(source: Mapping[str, Any]) -> tuple[str, str, str]:
            event = source.get("event") or {}
            content = source.get("content") or {}
            return (
                str(event.get("category") or "").strip().lower(),
                str(event.get("label") or "").strip().lower(),
                str(content.get("transcript") or "").strip(),
            )

        donor_key = semantic_key(donor_source)
        donor_is_speech = is_speech_source(donor_source)
        existing_speech = [
            index
            for index, candidate in enumerate(sources)
            if is_speech_source(candidate)
        ]
        # A speech donor is legal.  If the current state already contains
        # speech, it must replace that persistent speech source rather than a
        # sound/music source; otherwise the next state would contain two
        # overlapping speakers.  With no current speech, any semantically
        # distinct source can be replaced.
        candidate_indices = (
            existing_speech
            if donor_is_speech and existing_speech
            else range(len(sources))
        )
        replaceable = [
            index
            for index in candidate_indices
            if semantic_key(sources[index]) != donor_key
        ]
        if not replaceable:
            raise SpatialStoryError(
                "replace_source donor would be a planner-visible semantic no-op"
            )
        target_index = rng.choice(replaceable)
    else:
        target_index = rng.randrange(len(sources))
    target = sources[target_index]
    target_id = str(target["source_id"])
    changed_fields: list[str]

    if edit_type == "move_source":
        delta = rng.choice((-1.0, 1.0)) * rng.uniform(45.0, 120.0)
        _rotate_motion(target, delta)
        result["instruction"] = (
            f"Move {target['event'].get('label') or target_id} by "
            f"{abs(delta):.0f} degrees {'left' if delta > 0 else 'right'}."
        )
        changed_fields = ["motion.keyframes"]
    elif edit_type == "change_gain":
        delta_db = rng.choice((-6.0, 6.0))
        target["gain_db"] = float(target.get("gain_db", 0.0)) + delta_db
        result["instruction"] = (
            f"Make {target['event'].get('label') or target_id} "
            f"{'louder' if delta_db > 0 else 'quieter'}."
        )
        changed_fields = ["gain_db"]
    elif edit_type == "change_activity":
        activity = target["activity"]
        old_onset = float(activity["onset_sec"])
        old_offset = float(activity["offset_sec"])
        directions = []
        if right_slack >= minimum_shift:
            directions.append((1.0, right_slack, "later"))
        if left_slack >= minimum_shift:
            directions.append((-1.0, left_slack, "earlier"))
        sign, available, direction = rng.choice(directions)
        shift = min(max(minimum_shift, 0.15 * scene_duration), available)
        activity["onset_sec"] = old_onset + sign * shift
        activity["offset_sec"] = old_offset + sign * shift
        activity["quality"] = "paired_edit_exact"
        result["instruction"] = (
            f"Make {target['event'].get('label') or target_id} start "
            f"{shift:.2f} seconds {direction}."
        )
        changed_fields = ["activity.onset_sec", "activity.offset_sec"]
    elif edit_type == "remove_source":
        if len(sources) < 2:
            raise SpatialStoryError("remove_source requires at least two sources")
        removed = sources.pop(target_index)
        result["instruction"] = (
            f"Remove {removed['event'].get('label') or target_id} from the scene."
        )
        changed_fields = ["sources"]
    elif edit_type == "add_source":
        if len(sources) >= int(max_sources):
            raise SpatialStoryError(
                f"add_source requires fewer than {max_sources} active sources"
            )
        if donor_source is None:
            raise SpatialStoryError("add_source requires a resolved donor source")
        donor = copy.deepcopy(dict(donor_source))
        if is_speech_source(donor) and any(
            is_speech_source(source) for source in sources
        ):
            raise SpatialStoryError(
                "cannot add speech to a state that already contains speech"
            )
        occupied = {str(source["source_id"]) for source in sources}
        next_slot = next(
            (
                f"source_{index}"
                for index in range(int(max_sources))
                if f"source_{index}" not in occupied
            ),
            None,
        )
        if next_slot is None:
            raise SpatialStoryError("no free persistent source slot remains")
        donor["source_id"] = next_slot
        donor["source_uid"] = f"src_{stable_digest(result['family_id'], next_slot)}"
        sources.append(donor)
        label = donor.get("event", {}).get("label") or next_slot
        result["instruction"] = f"Add {label} to the scene."
        target_id = next_slot
        changed_fields = ["sources"]
    else:  # replace_source
        if donor_source is None:
            raise SpatialStoryError("replace_source requires a resolved donor source")
        donor = copy.deepcopy(dict(donor_source))
        if is_speech_source(donor) and any(
            index != target_index and is_speech_source(source)
            for index, source in enumerate(sources)
        ):
            raise SpatialStoryError(
                "speech donor must replace the existing speech source"
            )
        previous_label = target.get("event", {}).get("label") or target_id
        replacement_label = donor.get("event", {}).get("label") or "the new sound"
        # The conversation entity and its spatial/timing controls remain stable;
        # only semantic content and the dry asset are replaced.
        for key in ("event", "content", "dry_audio"):
            target[key] = copy.deepcopy(donor[key])
        category = str((target.get("event") or {}).get("category") or "").lower()
        target["playback"] = _resolved_playback_window(
            target["dry_audio"],
            activity_duration_sec=float(target["activity"]["offset_sec"])
            - float(target["activity"]["onset_sec"]),
            preserve_source_start=(
                category == "speech"
                or bool((target.get("content") or {}).get("transcript"))
            ),
            source_loudness=result["render_contract"]["source_loudness"],
        )
        result["instruction"] = (
            f"Replace {previous_label} with {replacement_label}, keeping its "
            "position and timing."
        )
        changed_fields = ["event", "content", "dry_audio", "playback"]

    if sum(is_speech_source(source) for source in sources) > 1:
        raise SpatialStoryError(
            "an edit may not create more than one speech source in a state"
        )

    result["edit"] = {
        "type": edit_type,
        "target_source_id": target_id,
        "changed_fields": changed_fields,
    }
    result["seed"] = int(seed)
    return _refresh_turn(result)


def build_edit_family(
    plan: Mapping[str, Any],
    *,
    seed: int,
    turns: int = 3,
    sample_rate: int = 44_100,
    max_sources: int = 4,
    edit_types: Optional[Sequence[str]] = None,
    donor_sources: Optional[Sequence[Mapping[str, Any]]] = None,
    family_id: Optional[str] = None,
    source_loudness: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build one cumulative AudioChat-like conversation of render recipes."""

    if turns < 1:
        raise SpatialStoryError("turns must be positive")
    base = build_base_recipe(
        plan,
        seed=seed,
        sample_rate=sample_rate,
        max_sources=max_sources,
        family_id=family_id,
        source_loudness=source_loudness,
    )
    recipes = [base]
    operation_cycle = list(
        edit_types or ("move_source", "change_activity", "change_gain")
    )
    if not operation_cycle:
        raise SpatialStoryError("edit_types must not be empty")
    unknown = set(operation_cycle) - SUPPORTED_EDIT_TYPES
    if unknown:
        raise SpatialStoryError(f"unsupported edit types: {sorted(unknown)}")
    donors = []
    for donor_index, donor in enumerate(donor_sources or ()):
        normalized = copy.deepcopy(dict(donor))
        normalized["source_id"] = f"donor_{donor_index}"
        normalized["source_uid"] = f"donor_{stable_digest(seed, donor_index)}"
        donors.append(
            _source_recipe(
                normalized,
                family_seed=int(seed),
                duration_sec=float(base["audio"]["duration_sec"]),
                source_loudness=base["render_contract"]["source_loudness"],
            )
        )
    donor_cursor = 0
    for turn_index in range(1, turns):
        operation = operation_cycle[(turn_index - 1) % len(operation_cycle)]
        if operation == "remove_source" and len(recipes[-1]["sources"]) < 2:
            operation = "move_source"
        if operation == "add_source" and len(recipes[-1]["sources"]) >= max_sources:
            operation = "replace_source"
        donor = None
        if operation in {"add_source", "replace_source"}:
            if donor_cursor >= len(donors):
                raise SpatialStoryError(f"{operation} requires donor_sources")
            donor = donors[donor_cursor]
            donor_cursor += 1
        turn_seed = int(stable_digest(seed, turn_index, operation, size=8), 16)
        try:
            next_recipe = make_edit_turn(
                recipes[-1],
                turn_index=turn_index,
                edit_type=operation,
                seed=turn_seed,
                donor_source=donor,
                max_sources=max_sources,
            )
        except SpatialStoryError:
            if operation != "change_activity":
                raise
            # Full-duration sources (most commonly long speech) cannot move in
            # time without contradicting their transcript.  Keep the family
            # usable and fall back to an exact spatial edit.
            operation = "move_source"
            turn_seed = int(stable_digest(seed, turn_index, operation, size=8), 16)
            next_recipe = make_edit_turn(
                recipes[-1],
                turn_index=turn_index,
                edit_type=operation,
                seed=turn_seed,
                max_sources=max_sources,
            )
        recipes.append(next_recipe)

    if donor_cursor != len(donors):
        raise SpatialStoryError(
            f"unused donor sources: consumed={donor_cursor} provided={len(donors)}"
        )

    conversation_turns = []
    for index, recipe in enumerate(recipes):
        before_plan = recipes[index - 1]["scene_plan"] if index else None
        after_plan = recipe["scene_plan"]
        conversation_turns.append(
            {
                "turn_id": recipe["turn_id"],
                "parent_turn_id": recipe["parent_turn_id"],
                "task": recipe["task"],
                "instruction": recipe["instruction"],
                "planner_prompt": recipe["planner_prompt"],
                "semantic_caption": recipe["semantic_caption"],
                "semantic_caption_metadata": copy.deepcopy(
                    recipe["semantic_caption_metadata"]
                ),
                "understanding_prompt": recipe["understanding_prompt"],
                "before": {
                    "recipe_fingerprint": (
                        recipes[index - 1]["recipe_fingerprint"] if index else None
                    ),
                    "audio_ref": None,
                    "scene_plan": copy.deepcopy(before_plan),
                },
                "after": {
                    "recipe_fingerprint": recipe["recipe_fingerprint"],
                    "audio_ref": None,
                    "scene_plan": copy.deepcopy(after_plan),
                },
                "diff": diff_scene_plans(before_plan, after_plan),
                "edit": copy.deepcopy(recipe["edit"]),
            }
        )
    family = {
        "schema": RECIPE_SCHEMA,
        "schema_version": RECIPE_VERSION,
        "family_id": base["family_id"],
        "source_sample_id": plan.get("sample_id"),
        "reference_audio": base.get("reference_audio"),
        "render_status": "planned",
        "turns": conversation_turns,
        "recipes": recipes,
    }
    validate_edit_family(family, require_outputs=False)
    return family


def validate_edit_family(
    family: Mapping[str, Any],
    *,
    require_outputs: bool = False,
) -> None:
    if family.get("schema") != RECIPE_SCHEMA:
        raise SpatialStoryError(f"unexpected edit family schema: {family.get('schema')}")
    if str(family.get("schema_version")) != RECIPE_VERSION:
        raise SpatialStoryError("unsupported edit family schema version")
    recipes = family.get("recipes")
    turns = family.get("turns")
    if not isinstance(recipes, list) or not recipes or len(recipes) != len(turns or []):
        raise SpatialStoryError("edit family recipes/turns are missing or misaligned")
    seen_turns = set()
    for index, recipe in enumerate(recipes):
        turn_id = str(recipe.get("turn_id") or "")
        if not turn_id or turn_id in seen_turns:
            raise SpatialStoryError(f"invalid/duplicate turn id: {turn_id!r}")
        if index and recipe.get("parent_turn_id") != recipes[index - 1].get("turn_id"):
            raise SpatialStoryError("edit recipes must form one cumulative chain")
        if recipe.get("audio", {}).get("channel_layout") != FOA_LAYOUT:
            raise SpatialStoryError("edit recipe must use WYZX ACN/SN3D FOA")
        render_contract = recipe.get("render_contract") or {}
        if (
            render_contract.get("state_input") != STATE_RENDER_INPUT
            or render_contract.get("uses_previous_foa") is not False
            or render_contract.get("independent_state_mix") is not True
        ):
            raise SpatialStoryError(
                "every turn must be independently mixed from dry-source tracks"
            )
        loudness_policy = _source_loudness_policy(
            render_contract.get("source_loudness")
        )
        if not recipe.get("sources"):
            raise SpatialStoryError("every turn must retain at least one source")
        speech_sources = [
            source for source in recipe["sources"] if is_speech_source(source)
        ]
        if len(speech_sources) > 1:
            raise SpatialStoryError(
                "a Spatial-CoT turn may contain at most one speech source"
            )
        if index and str((recipe.get("edit") or {}).get("type")) == "replace_source":
            target_id = str((recipe.get("edit") or {}).get("target_source_id") or "")
            previous_sources = {
                str(source["source_id"]): source for source in recipes[index - 1]["sources"]
            }
            current_sources = {
                str(source["source_id"]): source for source in recipe["sources"]
            }
            if target_id not in previous_sources or target_id not in current_sources:
                raise SpatialStoryError("replace_source target is not persistent")

            def semantic_key(source: Mapping[str, Any]) -> tuple[str, str, str]:
                event = source.get("event") or {}
                content = source.get("content") or {}
                return (
                    str(event.get("category") or "").strip().lower(),
                    str(event.get("label") or "").strip().lower(),
                    str(content.get("transcript") or "").strip(),
                )

            if semantic_key(previous_sources[target_id]) == semantic_key(
                current_sources[target_id]
            ):
                raise SpatialStoryError(
                    "replace_source must change planner-visible semantic state"
                )
            diff = (turns[index].get("diff") or {}) if turns else {}
            changed_ids = {
                str(item.get("source_id")) for item in diff.get("changed") or []
            }
            if target_id not in changed_ids:
                raise SpatialStoryError(
                    "replace_source target is absent from the persisted ScenePlan diff"
                )
        if index and str((recipe.get("edit") or {}).get("type")) == "move_source":
            target_id = str((recipe.get("edit") or {}).get("target_source_id") or "")
            previous_sources = {
                str(source["source_id"]): source for source in recipes[index - 1]["sources"]
            }
            current_sources = {
                str(source["source_id"]): source for source in recipe["sources"]
            }
            if target_id not in previous_sources or target_id not in current_sources:
                raise SpatialStoryError("move_source target is not persistent")
            before = previous_sources[target_id]
            after = current_sources[target_id]
            for field in ("event", "content", "dry_audio", "activity"):
                if before.get(field) != after.get(field):
                    raise SpatialStoryError(
                        f"move_source unexpectedly changed {field}"
                    )
            if before.get("motion") == after.get("motion"):
                raise SpatialStoryError("move_source must change the trajectory")
        for source in recipe["sources"]:
            if is_silent_source(source):
                raise SpatialStoryError(
                    f"source {source.get('source_id')} is explicitly silent"
                )
            dry = source.get("dry_audio") or {}
            playback = source.get("playback") or {}
            content = source.get("content") or {}
            required_dry = {
                "path",
                "native_sample_rate",
                "crop_start_native_sample",
                "crop_num_native_samples",
            }
            if not required_dry.issubset(dry):
                raise SpatialStoryError(
                    f"source {source.get('source_id')} has an unresolved dry crop"
                )
            transcript = str(content.get("transcript") or "").strip()
            transcript_quality = str(content.get("transcript_quality") or "")
            source_was_cropped = int(dry["native_num_frames"]) > int(
                dry["crop_num_native_samples"]
            )
            if transcript and source_was_cropped:
                raise SpatialStoryError(
                    "a cropped long speech source cannot retain an unaligned transcript"
                )
            if transcript and int(dry["crop_start_native_sample"]) != 0:
                raise SpatialStoryError(
                    "a transcript-bearing speech source must preserve source start"
                )
            if source_was_cropped and is_speech_source(source):
                if transcript_quality not in {
                    "omitted_unaligned_long_source",
                    "not_available",
                }:
                    raise SpatialStoryError(
                        "cropped speech requires explicit transcript omission quality"
                    )
            required_playback = {
                "offset_in_crop_native_sample",
                "num_native_samples",
                "mono_peak",
                "mono_rms",
                "loudness",
            }
            if not required_playback.issubset(playback):
                raise SpatialStoryError(
                    f"source {source.get('source_id')} has no resolved playback window"
                )
            playback_offset = int(playback["offset_in_crop_native_sample"])
            playback_count = int(playback["num_native_samples"])
            if (
                playback_offset < 0
                or playback_count <= 0
                or playback_offset + playback_count
                > int(dry["crop_num_native_samples"])
            ):
                raise SpatialStoryError(
                    f"source {source.get('source_id')} playback exceeds its dry crop"
                )
            mono_peak = float(playback["mono_peak"])
            mono_rms = float(playback["mono_rms"])
            if (
                not math.isfinite(mono_peak)
                or not math.isfinite(mono_rms)
                or mono_rms < float(loudness_policy["minimum_input_rms"])
            ):
                raise SpatialStoryError(
                    f"source {source.get('source_id')} playback failed energy QC"
                )
            expected_loudness = _source_loudness(
                mono_peak=mono_peak,
                mono_rms=mono_rms,
                policy=loudness_policy,
            )
            actual_loudness = playback["loudness"]
            if set(actual_loudness) != set(expected_loudness) or any(
                not math.isclose(
                    float(actual_loudness[key]),
                    float(expected_loudness[key]),
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-12,
                )
                if isinstance(expected_loudness[key], (int, float))
                else actual_loudness[key] != expected_loudness[key]
                for key in expected_loudness
            ):
                raise SpatialStoryError(
                    f"source {source.get('source_id')} loudness contract changed"
                )
            source_render_signature(recipe, source)
        if require_outputs:
            output = recipe.get("outputs") or {}
            if not output.get("foa_path"):
                raise SpatialStoryError(f"turn {turn_id} has no rendered FOA output")
        seen_turns.add(turn_id)


def unique_source_render_jobs(
    family: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return one job per unique pre-mix track required by a family."""

    jobs: dict[str, dict[str, Any]] = {}
    for recipe in family.get("recipes") or []:
        for source in recipe.get("sources") or []:
            signature = source_render_signature(recipe, source)
            jobs.setdefault(
                signature,
                {
                    "track_id": signature,
                    "recipe": recipe,
                    "source": source,
                },
            )
    return [jobs[key] for key in sorted(jobs)]


__all__ = [
    "FOA_LAYOUT",
    "RECIPE_SCHEMA",
    "RECIPE_VERSION",
    "RENDERER_NAME",
    "RENDERER_VERSION",
    "STATE_RENDER_INPUT",
    "SUPPORTED_EDIT_TYPES",
    "DEFAULT_SOURCE_LOUDNESS",
    "build_base_recipe",
    "build_edit_family",
    "describe_recipe",
    "is_speech_source",
    "is_silent_source",
    "make_edit_turn",
    "recipe_fingerprint",
    "source_render_signature",
    "stable_digest",
    "unique_source_render_jobs",
    "validate_edit_family",
]
