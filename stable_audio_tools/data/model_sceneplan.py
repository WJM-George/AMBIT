"""Model-facing ScenePlan v1 validation and deterministic P10 conditioning.

This module is intentionally separate from :mod:`sceneplan_v2`.  Revision-4
renderer records remain immutable for P0--P6, while P7.5 stores only semantic
and controllable model state.  Asset locators and renderer execution lineage
live in a separate render-recipe artifact.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


MAX_SOURCES = 4
MODEL_SAMPLE_RATE = 44_100
VAE_HOP_SAMPLES = 1024
MAX_LATENT_FRAMES = 648
MAX_MODEL_SAMPLES = MAX_LATENT_FRAMES * VAE_HOP_SAMPLES
MAX_DURATION_SEC = MAX_MODEL_SAMPLES / MODEL_SAMPLE_RATE
# Model-facing ScenePlans serialize times to six decimal places.  A legal
# exact-sample boundary can therefore round upward by at most half a
# microsecond; the validator must accept that representation while still
# rejecting even one additional 44.1-kHz sample (~22.7 microseconds).
SERIALIZED_TIME_TOLERANCE_SEC = 0.51e-6
ROOM_TYPES = {"dry", "moderate", "reverberant", "outdoor"}
SOURCE_KINDS = {"speech", "music", "sound"}
KIND_IDS = {"empty": 0, "speech": 1, "music": 2, "sound": 3}
SOURCE_NAMES = ("one", "two", "three", "four")


def _renderer_semantic_fragment(
    value: Any, *, label: str = "source description"
) -> str:
    """Return the frozen deterministic text view without importing legacy 4+4+2.

    Revision-5 renderer-caption integrity still needs the exact historic
    normalization, but the P10 model path must not depend on the retired
    ScenePlan 4+4+2 module.
    """

    text = (
        " ".join(str(value or "").split())
        .strip()
        .strip('"“”')
        .rstrip(" .;,:")
    )
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text.replace('"', "'").replace("“", "'").replace("”", "'")


def _finite_number(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _complete_text(value: Any, *, label: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text


def _source_slot(source_id: Any) -> int:
    value = str(source_id or "")
    if not value.startswith("source_") or not value[7:].isdigit():
        raise ValueError(f"invalid persistent source_id: {value!r}")
    slot = int(value[7:])
    if not 0 <= slot < MAX_SOURCES:
        raise ValueError(f"source_id is outside four slots: {value!r}")
    return slot


def _validate_position(position: Mapping[str, Any], *, label: str) -> dict[str, float]:
    if set(position) != {"azimuth_deg", "elevation_deg", "distance_m"}:
        raise ValueError(f"{label} must contain only azimuth/elevation/distance")
    azimuth = _finite_number(position["azimuth_deg"], label=f"{label}.azimuth_deg")
    elevation = _finite_number(position["elevation_deg"], label=f"{label}.elevation_deg")
    distance = _finite_number(position["distance_m"], label=f"{label}.distance_m")
    if not -180.0 <= azimuth <= 180.0:
        raise ValueError(f"{label}.azimuth_deg is outside [-180,180]")
    if not -90.0 <= elevation <= 90.0:
        raise ValueError(f"{label}.elevation_deg is outside [-90,90]")
    if distance <= 0.0:
        raise ValueError(f"{label}.distance_m must be positive")
    return {
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
        "distance_m": distance,
    }


def _trajectory_keyframes(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    activity = source["activity"]
    onset = float(activity["onset_sec"])
    offset = float(activity["offset_sec"])
    trajectory = source["trajectory"]
    motion_type = str(trajectory["type"])
    if motion_type == "static":
        return [
            {
                "time_sec": onset,
                "position": _validate_position(
                    trajectory["position"], label=f"{source['source_id']}.position"
                ),
            }
        ]
    if motion_type == "linear":
        return [
            {
                "time_sec": onset,
                "position": _validate_position(
                    trajectory["start"], label=f"{source['source_id']}.start"
                ),
            },
            {
                "time_sec": offset,
                "position": _validate_position(
                    trajectory["end"], label=f"{source['source_id']}.end"
                ),
            },
        ]
    if motion_type != "keyframed":
        raise ValueError(f"unsupported trajectory type: {motion_type!r}")
    raw = trajectory.get("keyframes")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not 2 <= len(raw) <= 8:
        raise ValueError("keyframed trajectory requires 2--8 keyframes")
    output = []
    for index, item in enumerate(raw):
        time_sec = _finite_number(
            item["time_sec"], label=f"{source['source_id']}.keyframes[{index}].time_sec"
        )
        output.append(
            {
                "time_sec": time_sec,
                "position": _validate_position(
                    item["position"],
                    label=f"{source['source_id']}.keyframes[{index}].position",
                ),
            }
        )
    times = [item["time_sec"] for item in output]
    if any(left >= right for left, right in zip(times, times[1:])):
        raise ValueError("keyframe times must be strictly increasing")
    if times[0] < onset - 1e-6 or times[-1] > offset + 1e-6:
        raise ValueError("keyframed trajectory lies outside source activity")
    return output


def validate_model_sceneplan(scene_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the compact model-facing state without rewriting it."""

    if set(scene_plan) != {"sample_id", "duration_sec", "room", "sources"}:
        raise ValueError("model ScenePlan has missing or non-model top-level fields")
    _complete_text(scene_plan["sample_id"], label="sample_id")
    duration = _finite_number(scene_plan["duration_sec"], label="duration_sec")
    if not 0.0 < duration <= MAX_DURATION_SEC + SERIALIZED_TIME_TOLERANCE_SEC:
        raise ValueError("duration_sec is outside the model envelope")
    room = scene_plan["room"]
    if not isinstance(room, Mapping) or set(room) != {"type"} or room["type"] not in ROOM_TYPES:
        raise ValueError("room must contain exactly one supported type")
    sources = scene_plan["sources"]
    if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
        raise ValueError("sources must be an array")
    if not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError("model ScenePlan requires one to four actual sources")
    slots = [_source_slot(source.get("source_id")) for source in sources]
    if slots != sorted(slots) or len(set(slots)) != len(slots):
        raise ValueError("sources must have unique persistent IDs in slot order")
    speech_count = 0
    for source in sources:
        kind = str(source.get("kind") or "")
        if kind not in SOURCE_KINDS:
            raise ValueError(f"unsupported source kind: {kind!r}")
        common = {"source_id", "kind", "activity", "trajectory", "gain_db"}
        semantic = (
            {"speaker_description", "transcript"}
            if kind == "speech"
            else {"description"}
        )
        if set(source) != common | semantic:
            raise ValueError(f"{source['source_id']}: source fields do not match kind={kind}")
        if kind == "speech":
            speech_count += 1
            _complete_text(source["speaker_description"], label="speaker_description")
            _complete_text(source["transcript"], label="transcript")
        else:
            _complete_text(source["description"], label="description")
        activity = source["activity"]
        if not isinstance(activity, Mapping) or set(activity) != {"onset_sec", "offset_sec"}:
            raise ValueError(f"{source['source_id']}: activity must be one interval")
        onset = _finite_number(activity["onset_sec"], label="activity.onset_sec")
        offset = _finite_number(activity["offset_sec"], label="activity.offset_sec")
        if not 0.0 <= onset < offset <= duration + 1e-6:
            raise ValueError(f"{source['source_id']}: activity lies outside scene")
        gain = _finite_number(source["gain_db"], label="gain_db")
        if not -24.0 <= gain <= 12.0:
            raise ValueError(f"{source['source_id']}: gain_db is outside [-24,12]")
        _trajectory_keyframes(source)
    if speech_count > 1:
        raise ValueError("a model ScenePlan permits at most one formal speech source")
    return dict(scene_plan)


def _position_words(position: Mapping[str, Any]) -> str:
    azimuth = float(position["azimuth_deg"])
    elevation = float(position["elevation_deg"])
    distance = float(position["distance_m"])
    directions = (
        "front",
        "front-left",
        "left",
        "rear-left",
        "behind",
        "rear-right",
        "right",
        "front-right",
    )
    normalized = (azimuth + 360.0) % 360.0
    direction = directions[int(((normalized + 22.5) % 360.0) // 45.0)]
    vertical = " above" if elevation > 20.0 else " below" if elevation < -20.0 else ""
    return f"{direction}{vertical} at {distance:.1f} m"


def _room_prefix(room_type: str) -> str:
    return {
        "dry": "In a dry acoustic environment, ",
        "moderate": "In a moderately reverberant room, ",
        "reverberant": "In a reverberant room, ",
        "outdoor": "In an outdoor acoustic environment, ",
    }[room_type]


def _append_span(
    parts: list[str],
    regions: list[dict[str, Any]],
    text: str,
    source: Mapping[str, Any],
    role: str,
) -> tuple[int, int]:
    start = sum(len(value) for value in parts)
    parts.append(text)
    end = start + len(text)
    regions.append(
        {
            "source_id": str(source["source_id"]),
            "source_slot": _source_slot(source["source_id"]),
            "start": start,
            "end": end,
            "role": role,
        }
    )
    return start, end


def _motion_activity_gain_text(source: Mapping[str, Any]) -> str:
    activity = source["activity"]
    keyframes = _trajectory_keyframes(source)
    start = _position_words(keyframes[0]["position"])
    if source["trajectory"]["type"] == "static":
        spatial = f"stationary {start}"
    else:
        spatial = f"moving from {start} to {_position_words(keyframes[-1]['position'])}"
    return (
        f"active {float(activity['onset_sec']):.2f}-{float(activity['offset_sec']):.2f} s, "
        f"{spatial}, mixed at {float(source['gain_db']):+.2f} dB"
    )


def compile_model_renderer_caption(scene_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Recompile the frozen revision-5 renderer caption for integrity checks.

    This artifact is retained only to validate the immutable P9 index.  The
    model-facing text path is :func:`compile_model_semantic_caption` below and
    deliberately contains no timing, geometry, motion, or gain language.
    """

    validate_model_sceneplan(scene_plan)
    parts = [_room_prefix(str(scene_plan["room"]["type"]))]
    clause_regions: list[dict[str, Any]] = []
    semantic_regions: list[dict[str, Any]] = []
    motion_regions: list[dict[str, Any]] = []
    speaker_regions: list[dict[str, Any]] = []
    transcript_regions: list[dict[str, Any]] = []
    for source_index, source in enumerate(scene_plan["sources"]):
        if source_index:
            parts.append("; ")
        clause_start = sum(len(value) for value in parts)
        if source["kind"] == "speech":
            semantic = _renderer_semantic_fragment(
                source["speaker_description"], label="speech speaker description"
            )
            semantic_start, semantic_end = _append_span(
                parts, semantic_regions, semantic, source, "source_semantic"
            )
            speaker_regions.append(
                {
                    "source_id": source["source_id"],
                    "source_slot": _source_slot(source["source_id"]),
                    "start": semantic_start,
                    "end": semantic_end,
                    "role": "speaker_info",
                }
            )
            parts.append(' says "')
            _append_span(
                parts,
                transcript_regions,
                _complete_text(source["transcript"], label="transcript"),
                source,
                "transcript",
            )
            parts.append('"')
        else:
            semantic = _renderer_semantic_fragment(
                source["description"], label=f"{source['kind']} description"
            )
            _append_span(parts, semantic_regions, semantic, source, "source_semantic")
        parts.append(", ")
        _append_span(
            parts,
            motion_regions,
            _motion_activity_gain_text(source),
            source,
            "motion_activity",
        )
        clause_regions.append(
            {
                "source_id": source["source_id"],
                "source_slot": _source_slot(source["source_id"]),
                "start": clause_start,
                "end": sum(len(value) for value in parts),
                "role": "source_clause",
            }
        )
    parts.append(".")
    return {
        "compiler": "sceneplan_renderer_caption",
        "compiler_version": 5,
        "text": "".join(parts),
        "source_clause_regions": clause_regions,
        "source_semantic_regions": semantic_regions,
        "source_motion_activity_regions": motion_regions,
        "speaker_info_regions": speaker_regions,
        "transcript_regions": transcript_regions,
    }


def _semantic_fragment(value: Any, *, label: str) -> str:
    """Return one clean source phrase without changing its audible meaning."""

    text = _renderer_semantic_fragment(value, label=label).strip()
    # A terminal mark before ``who says`` produces an accidental sentence
    # boundary.  The source registry remains unchanged; this is only a natural
    # language compiler for the Qwen cross-attention input.
    return text.rstrip(" .!?;:")


def _compile_model_semantic_caption(
    scene_plan: Mapping[str, Any], *, compiler_version: int
) -> dict[str, Any]:
    """Compile one versioned semantic caption without changing its meaning."""

    if int(compiler_version) not in {1, 2}:
        raise ValueError("semantic caption compiler_version must be 1 or 2")

    validate_model_sceneplan(scene_plan)
    parts = [_room_prefix(str(scene_plan["room"]["type"])).rstrip(), " "]
    event_regions: list[dict[str, Any]] = []
    speech_regions: list[dict[str, Any]] = []
    final_transcript_has_terminal_mark = False

    for source_index, source in enumerate(scene_plan["sources"]):
        if source_index:
            parts.append("; ")
        slot = _source_slot(source["source_id"])
        source_label = slot + 1
        event_start = sum(len(value) for value in parts)
        if source["kind"] == "speech":
            parts.append(f"Source {SOURCE_NAMES[slot]} is ")
            parts.append(
                _semantic_fragment(
                    source["speaker_description"],
                    label="speech speaker description",
                )
            )
            # Version 1 surrounded the exact spoken text with double quotes.
            # The quotes were always raw role 0, but their visual resemblance
            # to literary quotation marks made the data contract ambiguous.
            # Version 2 ends the event at ``who says`` and uses an ordinary
            # role-0 colon as the separator.  Only the explicit speech region
            # below determines which tokens must be spoken.
            parts.append(" who says " if int(compiler_version) == 1 else " who says")
        else:
            parts.append(
                f"Source {SOURCE_NAMES[slot]} contains {source['kind']}: "
            )
            parts.append(
                _semantic_fragment(
                    source["description"],
                    label=f"{source['kind']} description",
                )
            )
        event_end = sum(len(value) for value in parts)
        event_regions.append(
            {
                "source_id": str(source["source_id"]),
                "source_slot": slot,
                "source_label": source_label,
                "start": event_start,
                "end": event_end,
                "role": "event",
            }
        )
        if source["kind"] == "speech":
            parts.append('"' if int(compiler_version) == 1 else ": ")
            transcript_start = sum(len(value) for value in parts)
            transcript_text = _complete_text(
                source["transcript"], label="transcript"
            )
            parts.append(transcript_text)
            transcript_end = sum(len(value) for value in parts)
            if int(compiler_version) == 1:
                parts.append('"')
            final_transcript_has_terminal_mark = transcript_text.endswith(
                (".", ",", "!", "?", ";", ":")
            )
            speech_regions.append(
                {
                    "source_id": str(source["source_id"]),
                    "source_slot": slot,
                    "source_label": source_label,
                    "start": transcript_start,
                    "end": transcript_end,
                    "role": "speech",
                }
            )
        else:
            final_transcript_has_terminal_mark = False
    if not final_transcript_has_terminal_mark:
        parts.append(".")
    text = "".join(parts)
    return {
        "compiler": "sceneplan_semantic_caption",
        "compiler_version": int(compiler_version),
        "text": text,
        "event_regions": event_regions,
        "speech_regions": speech_regions,
    }


def compile_model_semantic_caption(scene_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Compile the frozen v1 semantic caption used by existing checkpoints.

    Event regions contain the complete audible source description.  For a
    formal speech source this includes the speaker description and the words
    ``who says`` as one event; only the exact spoken transcript is assigned to
    the independent speech role.  The legacy outer double quotes are ordinary
    role-0 punctuation and never define the speech span.
    """

    return _compile_model_semantic_caption(scene_plan, compiler_version=1)


def compile_model_semantic_caption_v2(
    scene_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile an unambiguous semantic caption with no protocol quotes.

    A formal speech clause is serialized as ``<speaker> who says: <words>``.
    ``event_source_ids`` cover the source clause through ``who says``;
    ``speech_source_ids`` alone cover the exact spoken text.  The role-0 colon
    is merely readable punctuation and carries no protocol meaning.
    """

    return _compile_model_semantic_caption(scene_plan, compiler_version=2)


def _compile_one_role_source_ids(
    regions: Sequence[Mapping[str, Any]],
    offsets: np.ndarray,
    attention_mask: np.ndarray,
    *,
    role: str,
) -> np.ndarray:
    labels = np.zeros(len(offsets), dtype=np.int8)
    for token_index, ((start, end), valid) in enumerate(
        zip(offsets.tolist(), attention_mask.tolist())
    ):
        if not valid or int(end) <= int(start):
            continue
        hits = {
            int(region["source_label"])
            for region in regions
            if max(int(start), int(region["start"]))
            < min(int(end), int(region["end"]))
        }
        if len(hits) > 1:
            raise ValueError(
                f"one Qwen token overlaps multiple {role} sources: token={token_index}, "
                f"offset=({start},{end}), labels={sorted(hits)}"
            )
        if hits:
            label = hits.pop()
            if not 1 <= label <= MAX_SOURCES:
                raise ValueError(f"{role} source label is outside [1,4]: {label}")
            labels[token_index] = label
    return labels


def tokenize_model_semantic_caption(
    caption: Mapping[str, Any], tokenizer: Any, *, max_length: int = 512
) -> dict[str, np.ndarray]:
    """Tokenize a complete semantic caption and assign raw ``0..4`` roles.

    ``-1`` is intentionally absent here: it is introduced only by classifier-
    free conditioning dropout.  Padding is represented solely by
    ``attention_mask`` and always carries role id 0.
    """

    text = str(caption["text"])
    encoded = tokenizer(
        text,
        truncation=False,
        padding="max_length",
        max_length=int(max_length),
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    attention = np.asarray(encoded["attention_mask"], dtype=np.uint8)
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    if input_ids.shape != (int(max_length),):
        raise ValueError("Qwen tokenizer did not return the requested padded length")
    complete = tokenizer(
        text,
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )
    if len(complete["input_ids"]) > int(max_length):
        raise ValueError(
            "complete semantic caption requires "
            f"{len(complete['input_ids'])} tokens > {int(max_length)}"
        )
    event_ids = _compile_one_role_source_ids(
        caption.get("event_regions") or (),
        offsets,
        attention,
        role="event",
    )
    speech_ids = _compile_one_role_source_ids(
        caption.get("speech_regions") or (),
        offsets,
        attention,
        role="speech",
    )
    # Decoder tokenizers can merge the closing event-space/opening quote with
    # the transcript's first apostrophe (for example ``who says "'tis``).  The
    # exact-speech role wins for that indivisible token; the preceding event
    # tokens still carry the complete speaker cue.  A token is never duplicated
    # across both roles.
    event_ids[speech_ids != 0] = 0
    speech_lexical = np.zeros(len(offsets), dtype=np.uint8)
    for token_index, ((start, end), valid, speech_id) in enumerate(
        zip(offsets.tolist(), attention.tolist(), speech_ids.tolist())
    ):
        if not valid or int(speech_id) <= 0 or int(end) <= int(start):
            continue
        # Punctuation remains part of the exact-speech role, but it receives no
        # forced-alignment duration.  This deterministic lexical mask is
        # available at both training and inference and therefore cannot leak a
        # teacher timestamp into generation.
        if any(character.isalnum() for character in text[int(start) : int(end)]):
            speech_lexical[token_index] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "offset_mapping": offsets,
        "event_source_ids": event_ids,
        "speech_source_ids": speech_ids,
        "speech_lexical_mask": speech_lexical,
    }


def make_sceneplan_cfg_dropout_metadata(
    metadata: Mapping[str, Any],
    *,
    caption_unknown: bool,
    structured_unknown: bool,
) -> dict[str, Any]:
    """Apply independent raw CFG states without mutating a positive row.

    ``caption_unknown`` controls only the Qwen semantic/caption branch.
    ``structured_unknown`` controls only the framewise 4+4 branch.  Keeping
    these flags separate is the model-facing contract that prevents training
    from coupling semantic and spatial/activity dropout by accident.
    """

    if not isinstance(metadata.get("prompt"), Mapping):
        raise ValueError("ScenePlan CFG metadata requires a tokenized prompt")
    if not isinstance(metadata.get("sceneplan_44"), Mapping):
        raise ValueError("ScenePlan CFG metadata requires sceneplan_44 controls")
    if not isinstance(caption_unknown, bool) or not isinstance(
        structured_unknown, bool
    ):
        raise TypeError("ScenePlan CFG branch states must be booleans")
    if "cfg_unknown" in metadata["prompt"] or "cfg_unknown" in metadata["sceneplan_44"]:
        raise ValueError("positive ScenePlan metadata already contains cfg_unknown")
    output = dict(metadata)
    if caption_unknown:
        prompt = dict(metadata["prompt"])
        prompt["cfg_unknown"] = True
        output["prompt"] = prompt
    if structured_unknown:
        controls = dict(metadata["sceneplan_44"])
        controls["cfg_unknown"] = True
        output["sceneplan_44"] = controls
    return output


def make_sceneplan_cfg_unknown_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Build the fully unconditional negative used by standard CFG inference."""

    return make_sceneplan_cfg_dropout_metadata(
        metadata,
        caption_unknown=True,
        structured_unknown=True,
    )


def _shortest_angle_radians(start: float, stop: float, fraction: np.ndarray) -> np.ndarray:
    delta = (stop - start + math.pi) % (2.0 * math.pi) - math.pi
    return start + delta * fraction


def _interpolate_trajectory(
    keyframes: Sequence[Mapping[str, Any]], seconds: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([float(item["time_sec"]) for item in keyframes], dtype=np.float64)
    azimuth = np.radians([float(item["position"]["azimuth_deg"]) for item in keyframes])
    elevation = np.radians([float(item["position"]["elevation_deg"]) for item in keyframes])
    distance = np.asarray([float(item["position"]["distance_m"]) for item in keyframes])
    if len(times) == 1:
        return (
            np.full_like(seconds, azimuth[0]),
            np.full_like(seconds, elevation[0]),
            np.full_like(seconds, distance[0]),
        )
    segment = np.clip(np.searchsorted(times, seconds, side="right") - 1, 0, len(times) - 2)
    left, right = times[segment], times[segment + 1]
    fraction = np.clip((seconds - left) / np.maximum(right - left, 1e-9), 0.0, 1.0)
    az = _shortest_angle_radians(azimuth[segment], azimuth[segment + 1], fraction)
    el = elevation[segment] + (elevation[segment + 1] - elevation[segment]) * fraction
    dist = distance[segment] + (distance[segment + 1] - distance[segment]) * fraction
    az[seconds <= times[0]] = azimuth[0]
    el[seconds <= times[0]] = elevation[0]
    dist[seconds <= times[0]] = distance[0]
    az[seconds >= times[-1]] = azimuth[-1]
    el[seconds >= times[-1]] = elevation[-1]
    dist[seconds >= times[-1]] = distance[-1]
    return az, el, dist


def compile_model_44_controls(
    scene_plan: Mapping[str, Any],
    *,
    model_num_samples: int | None = None,
    latent_frames_valid: int | None = None,
) -> dict[str, np.ndarray]:
    """Compile four event tracks plus four full geometric trajectory tracks.

    The source-level ``gain_db`` remains part of the frozen ScenePlan contract,
    but it is deliberately not a model condition.  Distance attenuation is
    already present in the rendered FOA target; distance itself remains in the
    trajectory so the DiT can learn that acoustic relation.
    """

    validate_model_sceneplan(scene_plan)
    duration = float(scene_plan["duration_sec"])
    samples = (
        int(model_num_samples)
        if model_num_samples is not None
        else int(round(duration * MODEL_SAMPLE_RATE))
    )
    if samples <= 0 or not math.isclose(
        duration, samples / MODEL_SAMPLE_RATE, abs_tol=1.1e-6
    ):
        raise ValueError("duration_sec and model_num_samples disagree")
    frames = (
        int(latent_frames_valid)
        if latent_frames_valid is not None
        else math.ceil(samples / VAE_HOP_SAMPLES)
    )
    if (
        samples > MAX_MODEL_SAMPLES
        or frames != math.ceil(samples / VAE_HOP_SAMPLES)
        or not 1 <= frames <= MAX_LATENT_FRAMES
    ):
        raise ValueError("latent_frames_valid disagrees with model duration")
    frame_start = np.arange(frames, dtype=np.float64) * VAE_HOP_SAMPLES / MODEL_SAMPLE_RATE
    frame_end = np.minimum(
        (np.arange(frames, dtype=np.float64) + 1.0)
        * VAE_HOP_SAMPLES
        / MODEL_SAMPLE_RATE,
        duration,
    )
    seconds = (frame_start + frame_end) * 0.5
    present = np.zeros(MAX_SOURCES, dtype=np.uint8)
    kind_ids = np.zeros(MAX_SOURCES, dtype=np.int64)
    slot_ids = np.arange(1, MAX_SOURCES + 1, dtype=np.int64)
    event_ids = np.zeros((MAX_SOURCES, frames), dtype=np.int8)
    # sin/cos azimuth, sin/cos elevation, log1p distance.  Activity is encoded
    # by event_ids and therefore is not duplicated as a continuous feature.
    trajectory = np.zeros((MAX_SOURCES, frames, 5), dtype=np.float32)
    speech_active = np.zeros(frames, dtype=np.uint8)
    for source in scene_plan["sources"]:
        slot = _source_slot(source["source_id"])
        present[slot] = 1
        kind_ids[slot] = KIND_IDS[str(source["kind"])]
        interval = source["activity"]
        onset = float(interval["onset_sec"])
        offset = float(interval["offset_sec"])
        # A latent frame is active whenever its waveform support overlaps the
        # source interval.  This cannot silently lose a sub-hop event.
        active_bool = (frame_start < offset) & (frame_end > onset)
        active = active_bool.astype(np.float32)
        event_ids[slot, active_bool] = slot + 1
        azimuth, elevation, distance = _interpolate_trajectory(
            _trajectory_keyframes(source), np.clip(seconds, onset, offset)
        )
        trajectory[slot, :, 0] = np.sin(azimuth) * active
        trajectory[slot, :, 1] = np.cos(azimuth) * active
        trajectory[slot, :, 2] = np.sin(elevation) * active
        trajectory[slot, :, 3] = np.cos(elevation) * active
        trajectory[slot, :, 4] = np.log1p(np.maximum(distance, 0.0)) * active
        if str(source["kind"]) == "speech":
            speech_active |= active_bool.astype(np.uint8)
    return {
        "source_present_mask": present,
        "source_kind_ids": kind_ids,
        "source_slot_ids": slot_ids,
        "source_event_frame_ids": event_ids,
        "source_trajectory_features": trajectory,
        "speech_active_frame_mask": speech_active,
    }


__all__ = [
    "compile_model_44_controls",
    "compile_model_renderer_caption",
    "compile_model_semantic_caption",
    "compile_model_semantic_caption_v2",
    "make_sceneplan_cfg_unknown_metadata",
    "make_sceneplan_cfg_dropout_metadata",
    "tokenize_model_semantic_caption",
    "validate_model_sceneplan",
]
