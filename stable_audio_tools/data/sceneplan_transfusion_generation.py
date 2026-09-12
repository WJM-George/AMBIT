"""Exact, natural-language supervision for Transfusion Generation AR.

The raw request is derived only from model-visible ScenePlan state.  Renderer
lineage, sample identifiers, latent locations, and source asset identifiers are
never placed in the request.  Targets are projected to the frozen codec-v4
grid and receive acoustically observable, contiguous source identifiers before
the request is rendered.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .model_sceneplan import validate_model_sceneplan
from .model_sceneplan_codec_v4 import ModelScenePlanCodecV4


GENERATION_RAW_REQUEST_CONTRACT = "sceneplan_transfusion_generation_request_v1"
GENERATION_REQUEST_NUMERIC_CONTRACT = "codec_v4_reversible_seconds_ms_mm_v1"
GENERATION_TARGET_CONTRACT = "codec_v4_projected_contiguous_observable_sources_v1"
GENERATION_REQUEST_SEED = 42
GENERATION_REQUEST_MAX_QWEN_TOKENS = 512

# Exact template surfaces are split-disjoint.  This makes the frozen 8K test a
# genuine language-surface generalization test in addition to being content
# disjoint from training.
TRAIN_TEMPLATE_IDS = tuple(range(0, 24))
VALIDATION_TEMPLATE_IDS = tuple(range(24, 28))
TEST_TEMPLATE_IDS = tuple(range(28, 32))
TEMPLATE_IDS_BY_SPLIT = {
    "train": TRAIN_TEMPLATE_IDS,
    "validation": VALIDATION_TEMPLATE_IDS,
    "test": TEST_TEMPLATE_IDS,
}

_ROOM_TEXT = {
    "dry": "dry",
    "moderate": "moderate",
    "reverberant": "reverberant",
    "outdoor": "outdoor",
}

_ORDINAL_WORDS = ("first", "second", "third", "fourth")

_OPENERS = (
    "Create {duration}s of FOA audio; room type={room}.",
    "Generate a {duration}s first-order Ambisonic scene; room type={room}.",
    "I need {duration}s of spatial FOA audio; room type={room}.",
    "Render a {duration}s FOA scene; room type={room}.",
)

_SOURCE_LEADS = (
    "Source {number}",
    "{ordinal_cap} source",
)

_CLOSERS = (
    "Use this order, fixed 0 dB gains, and return the complete ScenePlan.",
    "Return one complete ScenePlan in this order with every gain at 0 dB.",
    "Output the full ScenePlan; keep this order and set every gain to 0 dB.",
    "Preserve the listed order and canonical 0 dB gains in one complete ScenePlan.",
)

_POSITION_GUIDE = (
    "Position triples mean (azimuth degrees, elevation degrees, distance millimeters)."
)


@dataclass(frozen=True)
class GenerationRequest:
    """One deterministic raw request plus its exact codec-v4 target."""

    text: str
    template_id: str
    target_sceneplan: dict[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _source_key(source: Mapping[str, Any]) -> tuple[Any, ...]:
    """Order only by facts exposed in the generated request."""

    kind = str(source["kind"])
    semantic = (
        str(source["speaker_description"])
        if kind == "speech"
        else str(source["description"])
    )
    activity = source["activity"]
    return (
        float(activity["onset_sec"]),
        float(activity["offset_sec"]),
        {"speech": 0, "music": 1, "sound": 2}[kind],
        " ".join(semantic.casefold().split()),
        " ".join(str(source.get("transcript") or "").casefold().split()),
        _canonical_json(source["trajectory"]),
    )


def canonical_generation_target(
    sceneplan: Mapping[str, Any], codec: ModelScenePlanCodecV4
) -> dict[str, Any]:
    """Project a P10 row and replace renderer-only sparse slot labels."""

    validate_model_sceneplan(sceneplan)
    projected = codec.project_plan(sceneplan)
    target = copy.deepcopy(projected)
    target["sources"] = sorted(target["sources"], key=_source_key)
    for index, source in enumerate(target["sources"]):
        source["source_id"] = f"source_{index}"
    validate_model_sceneplan(target)
    return target


def _number(value: Any) -> str:
    number = float(value)
    text = f"{number:.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _duration_number(value: Any) -> str:
    """Compact duration that still selects the identical codec-v4 frame."""

    text = f"{float(value):.5f}".rstrip("0").rstrip(".")
    return "0" if text in {"-0", ""} else text


def _milliseconds(value: Any) -> str:
    """Human-scale integer time with sub-bin codec-v4 precision."""

    return str(int(round(float(value) * 1000.0)))


def _millimeters(value: Any) -> str:
    """Human-scale integer distance with sub-bin codec-v4 precision."""

    return str(int(round(float(value) * 1000.0)))


def _quoted(value: Any) -> str:
    # Keep the payload byte-for-byte visible even when it contains ordinary
    # ASCII quote marks.  JSON string escaping is reversible but looks like a
    # protocol serialization rather than a natural user request.
    return "“" + " ".join(str(value).split()) + "”"


def _position_text(position: Mapping[str, Any]) -> str:
    return "(" + ", ".join(
        (
            _number(position["azimuth_deg"]),
            _number(position["elevation_deg"]),
            _millimeters(position["distance_m"]),
        )
    ) + ")"


def _source_text(
    source: Mapping[str, Any], *, source_index: int, source_style: int
) -> str:
    lead = _SOURCE_LEADS[source_style].format(
        ordinal_cap=_ORDINAL_WORDS[source_index].capitalize(),
        number=source_index + 1,
    )
    kind = str(source["kind"])
    if kind == "speech":
        identity = (
            f"speech; speaker={_quoted(source['speaker_description'])}; "
            f"transcript={_quoted(source['transcript'])}"
        )
    elif kind == "music":
        identity = f"music; description={_quoted(source['description'])}"
    elif kind == "sound":
        identity = f"sound; description={_quoted(source['description'])}"
    else:  # validate_model_sceneplan should already make this unreachable.
        raise ValueError(f"unsupported source kind {kind!r}")

    activity = source["activity"]
    timing = (
        f"active from {_milliseconds(activity['onset_sec'])} to "
        f"{_milliseconds(activity['offset_sec'])}ms"
    )
    trajectory = source["trajectory"]
    motion = str(trajectory["type"])
    if motion == "static":
        spatial = "static at " + _position_text(trajectory["position"])
    elif motion == "linear":
        spatial = (
            "linear from "
            + _position_text(trajectory["start"])
            + " to "
            + _position_text(trajectory["end"])
        )
    else:
        raise ValueError(
            "Generation Transfusion v1 supports only static or linear motion, "
            f"got {motion!r}"
        )
    return f"{lead}: {identity}; {timing}; {spatial}."


def _template_number(sample_id: str, split: str, seed: int) -> int:
    try:
        allowed = TEMPLATE_IDS_BY_SPLIT[str(split)]
    except KeyError as error:
        raise ValueError(f"unknown Generation split {split!r}") from error
    digest = hashlib.blake2b(
        f"{int(seed)}:{split}:{sample_id}".encode("utf-8"),
        digest_size=8,
        person=b"gen-ar-v1",
    ).digest()
    return allowed[int.from_bytes(digest, "big") % len(allowed)]


def render_generation_request(
    target_sceneplan: Mapping[str, Any],
    *,
    split: str,
    seed: int = GENERATION_REQUEST_SEED,
    template_number: int | None = None,
) -> tuple[str, str]:
    """Render an exact natural request without renderer/provenance leakage."""

    validate_model_sceneplan(target_sceneplan)
    if int(seed) != GENERATION_REQUEST_SEED:
        raise ValueError("Generation Transfusion v1 uses frozen seed 42")
    allowed = TEMPLATE_IDS_BY_SPLIT.get(str(split))
    if allowed is None:
        raise ValueError(f"unknown Generation split {split!r}")
    selected = (
        _template_number(str(target_sceneplan["sample_id"]), str(split), int(seed))
        if template_number is None
        else int(template_number)
    )
    if selected not in allowed:
        raise ValueError(
            f"template {selected} is not reserved for the {split} split"
        )
    opener = _OPENERS[selected % len(_OPENERS)].format(
        duration=_duration_number(target_sceneplan["duration_sec"]),
        room=_ROOM_TEXT[str(target_sceneplan["room"]["type"])],
    )
    source_style = (selected // len(_OPENERS)) % len(_SOURCE_LEADS)
    closer = _CLOSERS[(selected // (len(_OPENERS) * len(_SOURCE_LEADS))) % len(_CLOSERS)]
    clauses = [
        _source_text(source, source_index=index, source_style=source_style)
        for index, source in enumerate(target_sceneplan["sources"])
    ]
    text = " ".join((opener, _POSITION_GUIDE, *clauses, closer))
    if str(target_sceneplan["sample_id"]) in text:
        raise RuntimeError("sample_id leaked into Generation raw request")
    return text, f"gen_ar_exact_v1/{split}/{selected:02d}"


def generation_request_missing_facts(
    text: str, target_sceneplan: Mapping[str, Any]
) -> list[str]:
    """Return exact target facts absent from the rendered surface."""

    required: list[tuple[str, str]] = [
        ("duration_sec", _duration_number(target_sceneplan["duration_sec"])),
        ("room.type", str(target_sceneplan["room"]["type"])),
    ]
    for index, source in enumerate(target_sceneplan["sources"]):
        prefix = f"sources[{index}]"
        required.extend(
            [
                (f"{prefix}.kind", str(source["kind"])),
                (
                    f"{prefix}.activity.onset_sec",
                    _milliseconds(source["activity"]["onset_sec"]),
                ),
                (
                    f"{prefix}.activity.offset_sec",
                    _milliseconds(source["activity"]["offset_sec"]),
                ),
            ]
        )
        if source["kind"] == "speech":
            required.extend(
                [
                    (
                        f"{prefix}.speaker_description",
                        _quoted(source["speaker_description"]),
                    ),
                    (f"{prefix}.transcript", _quoted(source["transcript"])),
                ]
            )
        else:
            required.append(
                (f"{prefix}.description", _quoted(source["description"]))
            )
        trajectory = source["trajectory"]
        positions: Sequence[tuple[str, Mapping[str, Any]]]
        if trajectory["type"] == "static":
            positions = (("position", trajectory["position"]),)
        else:
            positions = (("start", trajectory["start"]), ("end", trajectory["end"]))
        for position_name, position in positions:
            for coordinate in ("azimuth_deg", "elevation_deg", "distance_m"):
                surface = (
                    _millimeters(position[coordinate])
                    if coordinate == "distance_m"
                    else _number(position[coordinate])
                )
                required.append(
                    (
                        f"{prefix}.trajectory.{position_name}.{coordinate}",
                        surface,
                    )
                )
    return [path for path, surface in required if surface not in text]


def generation_request_numeric_reversibility_errors(
    target_sceneplan: Mapping[str, Any], codec: ModelScenePlanCodecV4
) -> list[str]:
    """Check that compact human units recover every target codec frame/bin."""

    errors: list[str] = []
    target_duration_frame = codec._frame_from_seconds(  # noqa: SLF001
        target_sceneplan["duration_sec"], mode="nearest"
    )
    request_duration_frame = codec._frame_from_seconds(  # noqa: SLF001
        float(_duration_number(target_sceneplan["duration_sec"])), mode="nearest"
    )
    if request_duration_frame != target_duration_frame:
        errors.append("duration_sec")

    for index, source in enumerate(target_sceneplan["sources"]):
        prefix = f"sources[{index}]"
        for field in ("onset_sec", "offset_sec"):
            target_frame = codec._frame_from_seconds(  # noqa: SLF001
                source["activity"][field], mode="nearest"
            )
            request_frame = codec._frame_from_seconds(  # noqa: SLF001
                int(_milliseconds(source["activity"][field])) / 1000.0,
                mode="nearest",
            )
            if request_frame != target_frame:
                errors.append(f"{prefix}.activity.{field}")

        trajectory = source["trajectory"]
        positions: Sequence[tuple[str, Mapping[str, Any]]]
        if trajectory["type"] == "static":
            positions = (("position", trajectory["position"]),)
        else:
            positions = (("start", trajectory["start"]), ("end", trajectory["end"]))
        for position_name, position in positions:
            requested_distance_m = int(_millimeters(position["distance_m"])) / 1000.0
            if codec._distance_index(requested_distance_m) != codec._distance_index(  # noqa: SLF001
                position["distance_m"]
            ):
                errors.append(
                    f"{prefix}.trajectory.{position_name}.distance_m"
                )
            for field, projector in (
                ("azimuth_deg", codec._azimuth_value),  # noqa: SLF001
                ("elevation_deg", codec._elevation_value),  # noqa: SLF001
            ):
                if projector(float(_number(position[field]))) != projector(
                    position[field]
                ):
                    errors.append(
                        f"{prefix}.trajectory.{position_name}.{field}"
                    )
    return errors


def build_generation_request(
    sceneplan: Mapping[str, Any],
    codec: ModelScenePlanCodecV4,
    *,
    split: str,
    seed: int = GENERATION_REQUEST_SEED,
) -> GenerationRequest:
    target = canonical_generation_target(sceneplan, codec)
    text, template_id = render_generation_request(
        target, split=split, seed=seed
    )
    missing = generation_request_missing_facts(text, target)
    if missing:
        raise RuntimeError(f"Generation request omitted target facts: {missing}")
    numeric_errors = generation_request_numeric_reversibility_errors(target, codec)
    if numeric_errors:
        raise RuntimeError(
            "Generation request numeric fields are not codec-v4 reversible: "
            f"{numeric_errors}"
        )
    return GenerationRequest(
        text=text,
        template_id=template_id,
        target_sceneplan=target,
    )


__all__ = [
    "GENERATION_RAW_REQUEST_CONTRACT",
    "GENERATION_REQUEST_MAX_QWEN_TOKENS",
    "GENERATION_REQUEST_NUMERIC_CONTRACT",
    "GENERATION_REQUEST_SEED",
    "GENERATION_TARGET_CONTRACT",
    "GenerationRequest",
    "TEMPLATE_IDS_BY_SPLIT",
    "build_generation_request",
    "canonical_generation_target",
    "generation_request_missing_facts",
    "generation_request_numeric_reversibility_errors",
    "render_generation_request",
]
