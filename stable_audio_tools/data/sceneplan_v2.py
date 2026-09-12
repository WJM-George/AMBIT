"""Frozen P0--P6 renderer-contract compiler (retired from model training).

The 4+4+2 helpers remain solely to reproduce and audit immutable historical
artifacts.  P10 imports no code from this module; its model-facing contract is
semantic cross-attention plus direct 4+4 controls in :mod:`model_sceneplan`.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

import numpy as np


MAX_SOURCES = 4
MODEL_SAMPLE_RATE = 44_100
VAE_HOP_SAMPLES = 1024
KIND_IDS = {"empty": 0, "speech": 1, "music": 2, "sound": 3}
_QUOTE_CHARS = {'"', "“", "”"}
_NON_SEMANTIC_TOKEN_CHARS = set('"“”;,.:!?()[]{}')


def _clean_fragment(value: Any, *, label: str) -> str:
    text = (
        " ".join(str(value or "").split())
        .strip()
        .strip('"“”')
        .rstrip(" .;,:")
    )
    if not text:
        raise ValueError(f"{label} must be non-empty")
    return text.replace('"', "'").replace("“", "'").replace("”", "'")


def renderer_semantic_fragment(value: Any, *, label: str = "source description") -> str:
    """Return the deterministic caption view without mutating ScenePlan data."""
    return _clean_fragment(value, label=label)


def _complete_transcript(value: Any) -> str:
    """Preserve complete spoken text, including transcript-internal quotes.

    Character spans, rather than quote searching, define the outer spoken
    region.  Therefore nested quotation punctuation is unambiguous and must
    not be rewritten merely to make the wrapper visually simpler.
    """

    text = " ".join(str(value or "").split())
    if not text:
        raise ValueError("speech transcript must be non-empty")
    return text


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


def _motion_activity_text(source: Mapping[str, Any]) -> str:
    activity = source.get("activity") or []
    if not activity:
        raise ValueError(f"present source {source.get('source_id')} has no activity")
    onset = min(float(item["onset_sec"]) for item in activity)
    offset = max(float(item["offset_sec"]) for item in activity)
    motion = source.get("motion") or {}
    keyframes = motion.get("keyframes") or []
    if not keyframes:
        raise ValueError(f"present source {source.get('source_id')} has no trajectory")
    start = _position_words(keyframes[0]["position"])
    if str(motion.get("type")) == "static" or len(keyframes) == 1:
        spatial = f"stationary {start}"
    else:
        stop = _position_words(keyframes[-1]["position"])
        spatial = f"moving from {start} to {stop}"
    return f"active {onset:.2f}-{offset:.2f} s, {spatial}"


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
            "source_slot": int(source["slot"]),
            "start": start,
            "end": end,
            "role": role,
        }
    )
    return start, end


def compile_renderer_caption(scene_plan: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one natural caption and exact connector-free character spans."""

    sources = scene_plan.get("sources")
    if not isinstance(sources, Sequence) or len(sources) != MAX_SOURCES:
        raise ValueError("ScenePlan v2 requires exactly four persistent source slots")
    parts: list[str] = []
    clause_regions: list[dict[str, Any]] = []
    semantic_regions: list[dict[str, Any]] = []
    motion_regions: list[dict[str, Any]] = []
    speaker_regions: list[dict[str, Any]] = []
    transcript_regions: list[dict[str, Any]] = []
    present_sources = [source for source in sources if bool(source.get("present"))]
    if not present_sources:
        raise ValueError("a ScenePlan must contain at least one source")
    speech_sources = [source for source in present_sources if source.get("kind") == "speech"]
    if len(speech_sources) > 1:
        raise ValueError("revision-4 scenes allow at most one speech source")

    for source_index, source in enumerate(present_sources):
        slot = int(source["slot"])
        if source.get("source_id") != f"source_{slot}" or not 0 <= slot < MAX_SOURCES:
            raise ValueError("source id and persistent slot do not agree")
        if source_index:
            # This separator is deliberately outside every mask.  There is no
            # lexical conjunction such as 'and' or 'while' to leak source
            # identity between slots.
            parts.append("; ")
        clause_start = sum(len(value) for value in parts)
        kind = str(source.get("kind"))
        if kind == "speech":
            speech = source.get("speech") or {}
            semantic = renderer_semantic_fragment(
                speech.get("speaker_description") or source.get("description"),
                label="speech speaker description",
            )
            semantic_start, semantic_end = _append_span(
                parts, semantic_regions, semantic, source, "source_semantic"
            )
            speaker_regions.append(
                {
                    "source_id": str(source["source_id"]),
                    "source_slot": slot,
                    "start": semantic_start,
                    "end": semantic_end,
                    "role": "speaker_info",
                }
            )
            parts.append(' says "')
            transcript = _complete_transcript(speech.get("transcript"))
            _append_span(parts, transcript_regions, transcript, source, "transcript")
            parts.append('"')
        elif kind in {"music", "sound"}:
            semantic = renderer_semantic_fragment(
                source.get("description"), label=f"{kind} description"
            )
            _append_span(parts, semantic_regions, semantic, source, "source_semantic")
        else:
            raise ValueError(f"present source has unsupported kind {kind!r}")
        parts.append(", ")
        motion_text = _motion_activity_text(source)
        _append_span(parts, motion_regions, motion_text, source, "motion_activity")
        clause_end = sum(len(value) for value in parts)
        clause_regions.append(
            {
                "source_id": str(source["source_id"]),
                "source_slot": slot,
                "start": clause_start,
                "end": clause_end,
                "role": "source_clause",
            }
        )
    parts.append(".")
    text = "".join(parts)
    return {
        "compiler": "sceneplan_renderer_caption",
        "compiler_version": 4,
        "text": text,
        "source_clause_regions": clause_regions,
        "source_semantic_regions": semantic_regions,
        "source_motion_activity_regions": motion_regions,
        "speaker_info_regions": speaker_regions,
        "transcript_regions": transcript_regions,
    }


def _validate_region(text: str, region: Mapping[str, Any]) -> tuple[int, int, int]:
    start, end, slot = int(region["start"]), int(region["end"]), int(region["source_slot"])
    if not 0 <= start < end <= len(text):
        raise ValueError(f"invalid caption region [{start},{end})")
    if not 0 <= slot < MAX_SOURCES or region.get("source_id") != f"source_{slot}":
        raise ValueError("caption region source id/slot mismatch")
    return start, end, slot


def _token_belongs_to_region(
    text: str,
    token_start: int,
    token_end: int,
    region_start: int,
    region_end: int,
) -> bool:
    significant = [
        index
        for index in range(max(0, token_start), min(len(text), token_end))
        if not text[index].isspace() and text[index] not in _NON_SEMANTIC_TOKEN_CHARS
    ]
    return bool(significant) and all(region_start <= index < region_end for index in significant)


def compile_442_token_masks(
    caption: Mapping[str, Any],
    offsets: Sequence[Sequence[int]],
    attention_mask: Sequence[int | bool],
) -> dict[str, np.ndarray]:
    """Map explicit caption regions to ten token masks (4+4+2)."""

    text = str(caption["text"])
    offsets_array = np.asarray(offsets, dtype=np.int64)
    attention = np.asarray(attention_mask, dtype=np.bool_)
    if offsets_array.ndim != 2 or offsets_array.shape[1] != 2:
        raise ValueError("offset mapping must have shape [L,2]")
    if attention.shape != offsets_array.shape[:1]:
        raise ValueError("attention mask must align with offsets")
    length = len(offsets_array)
    semantic = np.zeros((MAX_SOURCES, length), dtype=np.uint8)
    motion = np.zeros((MAX_SOURCES, length), dtype=np.uint8)
    speaker = np.zeros(length, dtype=np.uint8)
    transcript = np.zeros(length, dtype=np.uint8)

    def assign(regions: Sequence[Mapping[str, Any]], output: np.ndarray, per_slot: bool) -> None:
        for region in regions:
            start, end, slot = _validate_region(text, region)
            assigned = False
            for token_index, (token_start, token_end) in enumerate(offsets_array.tolist()):
                if not attention[token_index] or token_end <= token_start:
                    continue
                if _token_belongs_to_region(text, token_start, token_end, start, end):
                    if per_slot:
                        output[slot, token_index] = 1
                    else:
                        output[token_index] = 1
                    assigned = True
            if not assigned:
                raise ValueError(f"caption region [{start},{end}) did not align to any token")

    assign(caption.get("source_semantic_regions") or (), semantic, True)
    assign(caption.get("source_motion_activity_regions") or (), motion, True)
    assign(caption.get("speaker_info_regions") or (), speaker, False)
    assign(caption.get("transcript_regions") or (), transcript, False)

    # Connector punctuation and lexical wrappers must remain outside all ten
    # masks. Nested semantic/speaker masks are intentional for the one speaker.
    if np.any(speaker & transcript):
        raise ValueError("speaker-info and quoted-transcript token masks overlap")
    return {
        "source_semantic_token_masks": semantic,
        "source_motion_activity_token_masks": motion,
        "speaker_info_token_mask": speaker,
        "quoted_transcript_token_mask": transcript,
    }


def tokenize_renderer_caption(
    caption: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int = 512,
) -> dict[str, np.ndarray]:
    """Tokenize without silent truncation and compile the aligned 4+4+2 masks."""

    text = str(caption["text"])
    encoded = tokenizer(
        text,
        truncation=False,
        padding=False,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    attention = np.asarray(encoded["attention_mask"], dtype=np.uint8)
    offsets = np.asarray(encoded["offset_mapping"], dtype=np.int64)
    if len(input_ids) > int(max_length):
        raise ValueError(
            f"complete renderer caption requires {len(input_ids)} tokens > max_length={max_length}; "
            "repair non-speech descriptions, never truncate the transcript"
        )
    masks = compile_442_token_masks(caption, offsets, attention)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "offset_mapping": offsets,
        **masks,
    }


def _shortest_angle_radians(start: float, stop: float, fraction: np.ndarray) -> np.ndarray:
    delta = (stop - start + math.pi) % (2.0 * math.pi) - math.pi
    return start + delta * fraction


def _interpolate_trajectory(
    keyframes: Sequence[Mapping[str, Any]],
    seconds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray([float(item["time_sec"]) for item in keyframes], dtype=np.float64)
    if len(times) == 0 or (len(times) > 1 and np.any(np.diff(times) <= 0)):
        raise ValueError("trajectory keyframes must be non-empty and strictly ordered")
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


def compile_structured_source_controls(scene_plan: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Compile four event ids and four time-aligned position/activity streams."""

    audio = scene_plan["audio"]
    frames = int(audio["latent_frames_valid"])
    sample_rate = int(audio["model_sample_rate_hz"])
    hop = int(audio["vae_hop_samples"])
    if frames != math.ceil(int(audio["model_num_samples"]) / hop):
        raise ValueError("latent_frames_valid does not match model_num_samples")
    seconds = (np.arange(frames, dtype=np.float64) + 0.5) * hop / sample_rate
    sources = scene_plan["sources"]
    if len(sources) != MAX_SOURCES:
        raise ValueError("structured controls require four source slots")
    present = np.zeros(MAX_SOURCES, dtype=np.uint8)
    kind_ids = np.zeros(MAX_SOURCES, dtype=np.int64)
    slot_ids = np.arange(1, MAX_SOURCES + 1, dtype=np.int64)
    activity = np.zeros((MAX_SOURCES, frames), dtype=np.uint8)
    # active, sin/cos azimuth, sin/cos elevation, log1p distance,
    # normalized scene time, dynamic flag
    position = np.zeros((MAX_SOURCES, frames, 8), dtype=np.float32)
    scene_duration = int(audio["model_num_samples"]) / sample_rate
    speech_count = 0
    for slot, source in enumerate(sources):
        if int(source["slot"]) != slot or source["source_id"] != f"source_{slot}":
            raise ValueError("source array order must equal persistent slot order")
        kind = str(source["kind"])
        if kind not in KIND_IDS:
            raise ValueError(f"unknown source kind {kind!r}")
        kind_ids[slot] = KIND_IDS[kind]
        if not bool(source["present"]):
            if kind != "empty":
                raise ValueError("absent source slot must have kind=empty")
            continue
        if kind == "empty":
            raise ValueError("present source cannot have kind=empty")
        present[slot] = 1
        speech_count += int(kind == "speech")
        intervals = source.get("activity") or []
        for interval in intervals:
            start = int(interval["model_onset_sample"]) / sample_rate
            stop = int(interval["model_offset_sample"]) / sample_rate
            activity[slot] |= ((seconds >= start) & (seconds < stop)).astype(np.uint8)
        motion = source.get("motion") or {}
        keyframes = motion.get("keyframes") or []
        azimuth, elevation, distance = _interpolate_trajectory(keyframes, seconds)
        active = activity[slot].astype(np.float32)
        position[slot, :, 0] = active
        position[slot, :, 1] = np.sin(azimuth) * active
        position[slot, :, 2] = np.cos(azimuth) * active
        position[slot, :, 3] = np.sin(elevation) * active
        position[slot, :, 4] = np.cos(elevation) * active
        position[slot, :, 5] = np.log1p(np.maximum(distance, 0.0)) * active
        position[slot, :, 6] = np.clip(seconds / max(scene_duration, 1e-9), 0.0, 1.0) * active
        position[slot, :, 7] = float(str(motion.get("type")) != "static") * active
    if speech_count > 1:
        raise ValueError("revision-4 structured controls permit at most one speech source")
    return {
        "source_present_mask": present,
        "source_kind_ids": kind_ids,
        "source_slot_ids": slot_ids,
        "source_activity_frame_masks": activity,
        "source_position_activity_features": position,
    }


def connector_character_mask(caption: Mapping[str, Any]) -> np.ndarray:
    """Return caption characters outside every one of the 4+4+2 regions."""

    text = str(caption["text"])
    covered = np.zeros(len(text), dtype=np.uint8)
    for key in (
        "source_semantic_regions",
        "source_motion_activity_regions",
        "speaker_info_regions",
        "transcript_regions",
    ):
        for region in caption.get(key) or ():
            start, end, _ = _validate_region(text, region)
            covered[start:end] = 1
    return 1 - covered
