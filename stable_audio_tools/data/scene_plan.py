"""Canonical Spatial ScenePlan helpers and the versioned SpatialDSL view."""
from __future__ import annotations

import copy
import json
import math
from typing import Any, Iterable, Mapping

import torch

from stable_audio_tools.data.spatial_caption_templates import (
    SemanticCaptionRender,
    render_semantic_caption,
)


DSL_VERSION = "spatial_cot_dsl_v1"

# Token-level loss groups used by the optional Qwen cache.
LOSS_IGNORE = 0
LOSS_GRAMMAR = 1
LOSS_SEMANTIC = 2
LOSS_ROOM = 3
LOSS_SPATIAL_CATEGORICAL = 4
LOSS_SPATIAL_METRIC = 5
LOSS_MOTION = 6
LOSS_SPEECH_CONTENT = 7


def semantic_caption_render_from_sources(
    sources: Iterable[Mapping[str, Any]],
    *,
    template_key: Any = None,
    template_id: int | None = None,
) -> SemanticCaptionRender:
    """Describe audible content and retain exact persistent-source regions."""

    return render_semantic_caption(
        sources,
        template_key=template_key,
        template_id=template_id,
    )


def semantic_caption_from_sources(
    sources: Iterable[Mapping[str, Any]],
    *,
    template_key: Any = None,
    template_id: int | None = None,
) -> str:
    """Backward-compatible text-only view of the diverse caption renderer."""

    return semantic_caption_render_from_sources(
        sources,
        template_key=template_key,
        template_id=template_id,
    ).text


def semantic_caption_render_from_scene_plan(
    plan: Mapping[str, Any],
    *,
    template_key: Any = None,
    template_id: int | None = None,
) -> SemanticCaptionRender:
    """Compile source-aligned renderer semantics from a ScenePlan."""

    scene = plan.get("scene") or {}
    if template_key is None and template_id is None:
        template_key = plan.get("conversation_id") or plan.get("sample_id")
    return semantic_caption_render_from_sources(
        scene.get("sources") or [],
        template_key=template_key,
        template_id=template_id,
    )


def semantic_caption_from_scene_plan(plan: Mapping[str, Any]) -> str:
    """Backward-compatible text-only renderer prefix from a ScenePlan."""

    return semantic_caption_render_from_scene_plan(plan).text


def _quote(value: Any) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "unknown"
    number = float(value)
    text = f"{number:.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _position_text(position: dict[str, Any]) -> tuple[str, int]:
    if position.get("azimuth_deg") is not None and position.get("elevation_deg") is not None:
        return (
            "az="
            + _number(position["azimuth_deg"], 1)
            + ", el="
            + _number(position["elevation_deg"], 1)
            + ", d="
            + _number(position.get("distance_m"), 2),
            LOSS_SPATIAL_METRIC,
        )
    return (
        "direction="
        + _quote(position.get("direction", "unknown"))
        + ", elevation="
        + _quote(position.get("elevation", "unknown"))
        + ", d="
        + _number(position.get("distance_m"), 2),
        LOSS_SPATIAL_CATEGORICAL,
    )


def serialize_spatial_dsl_segments(
    plan: dict[str, Any],
) -> list[tuple[str, int]]:
    """Serialize a plan into deterministic text segments with loss-group IDs."""

    scene = plan["scene"]
    room = scene.get("room") or {}
    segments: list[tuple[str, int]] = [
        (f"scene<{DSL_VERSION}> {{\n", LOSS_GRAMMAR),
        (
            "  audio(format=foa, duration="
            + _number(plan.get("audio", {}).get("duration_sec"), 3)
            + ");\n",
            LOSS_GRAMMAR,
        ),
    ]
    room_parts = []
    for key, label in (
        ("type", "type"),
        ("description", "description"),
        ("reverb_label", "reverb"),
    ):
        if room.get(key) is not None:
            room_parts.append(f"{label}={_quote(room[key])}")
    if room.get("rt60_s") is not None:
        room_parts.append(f"rt60={_number(room['rt60_s'], 3)}")
    if room.get("dimensions_m") is not None:
        dims = ",".join(_number(value, 2) for value in room["dimensions_m"])
        room_parts.append(f"dimensions=[{dims}]")
    if room_parts:
        segments.append(("  room(" + ", ".join(room_parts) + ");\n", LOSS_ROOM))

    for source in scene.get("sources", []):
        source_id = source.get("source_id", "source_0")
        event = source.get("event") or {}
        segments.append((f"  source(id={_quote(source_id)}) {{\n", LOSS_GRAMMAR))
        event_parts = []
        if event.get("label") is not None:
            event_parts.append(f"label={_quote(event['label'])}")
        if event.get("category") is not None:
            event_parts.append(f"category={_quote(event['category'])}")
        if event_parts:
            segments.append(("    event(" + ", ".join(event_parts) + ");\n", LOSS_SEMANTIC))
        content = source.get("content") or {}
        transcript = content.get("transcript")
        if transcript:
            segments.append(
                (f"    utterance(text={_quote(transcript)});\n", LOSS_SPEECH_CONTENT)
            )

        activity = source.get("activity") or {}
        if activity.get("onset_sec") is not None and activity.get("offset_sec") is not None:
            segments.append(
                (
                    "    activity(onset="
                    + _number(activity["onset_sec"], 3)
                    + ", offset="
                    + _number(activity["offset_sec"], 3)
                    + ");\n",
                    LOSS_MOTION,
                )
            )

        motion = source.get("motion") or {}
        keyframes = motion.get("keyframes") or []
        if motion.get("type") == "linear" and len(keyframes) == 2:
            p0, group0 = _position_text(keyframes[0]["position"])
            p1, group1 = _position_text(keyframes[-1]["position"])
            group = LOSS_MOTION if group0 == group1 else max(group0, group1)
            segments.append(
                (
                    "    move(t0=("
                    + p0
                    + "), t1=("
                    + p1
                    + "));\n",
                    group,
                )
            )
        elif len(keyframes) > 1:
            parts = []
            groups = []
            for keyframe in keyframes:
                position, group = _position_text(keyframe["position"])
                parts.append(
                    "kf(t=" + _number(keyframe.get("t_norm"), 3) + ", " + position + ")"
                )
                groups.append(group)
            segments.append(
                ("    path(" + ", ".join(parts) + ");\n", max([LOSS_MOTION, *groups]))
            )
        elif keyframes:
            position, group = _position_text(keyframes[0]["position"])
            segments.append((f"    static({position});\n", group))
        segments.append(("  }\n", LOSS_GRAMMAR))
    segments.append(("}\n", LOSS_GRAMMAR))
    return segments


def serialize_spatial_dsl(plan: dict[str, Any]) -> str:
    return "".join(text for text, _ in serialize_spatial_dsl_segments(plan))


def _interpolate_angle(start: float, end: float, fraction: float) -> float:
    delta = ((float(end) - float(start) + 180.0) % 360.0) - 180.0
    value = float(start) + delta * fraction
    return ((value + 180.0) % 360.0) - 180.0


def _interpolate_position(
    start: dict[str, Any], end: dict[str, Any], fraction: float
) -> dict[str, Any]:
    result = copy.deepcopy(start if fraction <= 0.5 else end)
    metric = all(
        start.get(key) is not None and end.get(key) is not None
        for key in ("azimuth_deg", "elevation_deg", "distance_m")
    )
    if metric:
        result["azimuth_deg"] = _interpolate_angle(
            start["azimuth_deg"], end["azimuth_deg"], fraction
        )
        result["elevation_deg"] = float(start["elevation_deg"]) + (
            float(end["elevation_deg"]) - float(start["elevation_deg"])
        ) * fraction
        result["distance_m"] = float(start["distance_m"]) + (
            float(end["distance_m"]) - float(start["distance_m"])
        ) * fraction
        result["crop_interpolation"] = "metric_shortest_azimuth_arc"
    else:
        result["crop_interpolation"] = "categorical_nearest_endpoint"
    return result


def _position_at_keyframes(
    keyframes: list[dict[str, Any]], fraction: float
) -> dict[str, Any]:
    """Piecewise-linear position using shortest-arc azimuth interpolation."""

    ordered = sorted(keyframes, key=lambda item: float(item.get("t_norm", 0.0)))
    if not ordered:
        return {}
    if fraction <= float(ordered[0].get("t_norm", 0.0)):
        return copy.deepcopy(ordered[0].get("position") or {})
    if fraction >= float(ordered[-1].get("t_norm", 1.0)):
        return copy.deepcopy(ordered[-1].get("position") or {})
    for first, second in zip(ordered, ordered[1:]):
        first_t = float(first.get("t_norm", 0.0))
        second_t = float(second.get("t_norm", 1.0))
        if fraction > second_t:
            continue
        local = 0.0 if math.isclose(first_t, second_t) else (fraction - first_t) / (second_t - first_t)
        return _interpolate_position(
            first.get("position") or {}, second.get("position") or {}, local
        )
    return copy.deepcopy(ordered[-1].get("position") or {})


def crop_scene_plan(
    plan: dict[str, Any], timestamps: Iterable[float] | None
) -> dict[str, Any]:
    """Project full-clip motion supervision into a normalized training crop."""

    cropped = copy.deepcopy(plan)
    if timestamps is None:
        return cropped
    values = list(timestamps)
    if len(values) != 2:
        raise ValueError(f"timestamps must contain two values, got {values}")
    start, end = map(float, values)
    if not (0.0 <= start < end <= 1.0 + 1e-6):
        raise ValueError(f"timestamps must satisfy 0 <= start < end <= 1, got {values}")
    end = min(end, 1.0)
    full_duration = cropped.get("audio", {}).get("duration_sec")
    cropped["training_crop"] = {
        "source_timestamps": [start, end],
        "duration_sec": (
            float(full_duration) * (end - start) if full_duration is not None else None
        ),
    }
    crop_duration = float(full_duration) * (end - start) if full_duration is not None else None
    if crop_duration is not None:
        cropped["audio"]["duration_sec"] = crop_duration

    if math.isclose(start, 0.0) and math.isclose(end, 1.0):
        return cropped
    for source in cropped.get("scene", {}).get("sources", []):
        activity = source.get("activity") or {}
        if full_duration is not None:
            crop_start_sec = float(full_duration) * start
            crop_end_sec = float(full_duration) * end
            onset = activity.get("onset_sec")
            offset = activity.get("offset_sec")
            if onset is not None and offset is not None:
                overlap_start = max(float(onset), crop_start_sec)
                overlap_end = min(float(offset), crop_end_sec)
                if overlap_end > overlap_start:
                    activity["onset_sec"] = overlap_start - crop_start_sec
                    activity["offset_sec"] = overlap_end - crop_start_sec
                    activity["active_in_crop"] = True
                else:
                    # Keep the source identity in the structured plan but make
                    # inactivity explicit and codec-roundtrippable. ``None``
                    # means unknown activity elsewhere in the schema and would
                    # incorrectly compile to a full-window fallback.
                    activity["source_activity_quality"] = activity.get("quality")
                    activity["onset_sec"] = 0.0
                    activity["offset_sec"] = 0.0
                    activity["quality"] = "inactive_in_crop"
                    activity["active_in_crop"] = False
            render_window = activity.get("render_window_sec")
            if isinstance(render_window, (list, tuple)) and len(render_window) == 2:
                window_start = max(float(render_window[0]), crop_start_sec)
                window_end = min(float(render_window[1]), crop_end_sec)
                activity["render_window_sec"] = (
                    [window_start - crop_start_sec, window_end - crop_start_sec]
                    if window_end > window_start
                    else None
                )
            source["activity"] = activity

        motion = source.get("motion") or {}
        keyframes = motion.get("keyframes") or []
        if motion.get("type") not in {"linear", "keyframed"} or len(keyframes) < 2:
            continue
        projected = [
            {"t_norm": 0.0, "position": _position_at_keyframes(keyframes, start)}
        ]
        for keyframe in sorted(keyframes, key=lambda item: float(item.get("t_norm", 0.0))):
            time = float(keyframe.get("t_norm", 0.0))
            if start < time < end:
                projected.append(
                    {
                        "t_norm": (time - start) / (end - start),
                        "position": copy.deepcopy(keyframe.get("position") or {}),
                    }
                )
        projected.append(
            {"t_norm": 1.0, "position": _position_at_keyframes(keyframes, end)}
        )
        motion["keyframes"] = projected
        motion["type"] = "linear" if len(projected) == 2 else "keyframed"
        motion["crop_projected"] = True
    return cropped


def tokenize_spatial_cot(
    caption: str,
    plan: dict[str, Any],
    tokenizer,
    *,
    max_tokens: int = 1024,
    system_prompt: str = (
        "Convert the requested audio into SpatialDSL v1. "
        "Preserve event, speech, room, position, and motion information. "
        "Output only the DSL."
    ),
) -> dict[str, torch.Tensor]:
    """Build Qwen SFT tensors plus a token-level semantic loss-group mask."""

    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("Spatial-CoT tokenization requires a non-empty caption")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": caption.strip()},
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    # Qwen3.5's tokenizer returns BatchEncoding here, while older Qwen and
    # standard HF chat templates commonly return a plain list.
    if hasattr(prompt_ids, "get") and prompt_ids.get("input_ids") is not None:
        prompt_ids = prompt_ids["input_ids"]
    if isinstance(prompt_ids, torch.Tensor):
        prompt_ids = prompt_ids.flatten().tolist()
    else:
        prompt_ids = list(prompt_ids)

    target_ids: list[int] = []
    target_groups: list[int] = []
    for text, group in serialize_spatial_dsl_segments(plan):
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        target_ids.extend(int(token) for token in encoded)
        target_groups.extend([int(group)] * len(encoded))
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        target_ids.append(int(eos_id))
        target_groups.append(LOSS_GRAMMAR)

    input_ids = prompt_ids + target_ids
    if len(input_ids) > int(max_tokens):
        raise ValueError(
            f"Spatial-CoT sequence has {len(input_ids)} tokens, exceeding max_tokens={max_tokens}"
        )
    labels = [-100] * len(prompt_ids) + target_ids
    loss_groups = [LOSS_IGNORE] * len(prompt_ids) + target_groups
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "loss_group_ids": torch.tensor(loss_groups, dtype=torch.long),
    }


__all__ = [
    "DSL_VERSION",
    "LOSS_GRAMMAR",
    "LOSS_IGNORE",
    "crop_scene_plan",
    "semantic_caption_from_scene_plan",
    "semantic_caption_render_from_scene_plan",
    "semantic_caption_render_from_sources",
    "semantic_caption_from_sources",
    "serialize_spatial_dsl",
    "serialize_spatial_dsl_segments",
    "tokenize_spatial_cot",
]
