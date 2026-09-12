"""Paired ScenePlan construction for the Transfusion Editing route.

This module deliberately does *not* use the retired Atomic-Patch machinery.
It creates a complete target ScenePlan and a matched renderer recipe for one
source ScenePlan.  The source recipe remains renderer-only provenance; asset
paths, hashes, and latent references are never exposed in the natural edit
instruction consumed by Editing AR.  The source ScenePlan is never an Editing
AR input; only the clean source FOA latent and that instruction are inputs.

The audio editor is trained separately on ``source latent + complete target
ScenePlan + noisy target latent``.  These helpers only define auditable pair
truth and do not treat a ScenePlan mutation as the audio-editing operation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .model_sceneplan import MODEL_SAMPLE_RATE, validate_model_sceneplan


EDITING_PAIR_CONTRACT = "sceneplan_transfusion_paired_editing_v1"
EDITING_INSTRUCTION_CONTRACT = "sceneplan_transfusion_edit_instruction_v1"
EDITING_PAIR_SEED = 42

OP_EVENT_ADD = "event_addition"
OP_EVENT_REMOVE = "event_removal"
OP_RELOCATE = "stationary_spatial_relocation"
OP_STATIC_TO_LINEAR = "static_to_linear"
OP_LINEAR_TO_STATIC = "linear_to_static"

EDIT_OPERATIONS = (
    OP_EVENT_ADD,
    OP_EVENT_REMOVE,
    OP_RELOCATE,
    OP_STATIC_TO_LINEAR,
    OP_LINEAR_TO_STATIC,
)
OPERATION_FAMILY = {
    OP_EVENT_ADD: "event_add_remove",
    OP_EVENT_REMOVE: "event_add_remove",
    OP_RELOCATE: "stationary_azimuth_change",
    OP_STATIC_TO_LINEAR: "static_linear_toggle",
    OP_LINEAR_TO_STATIC: "static_linear_toggle",
}

# Actual language surfaces, rather than only template labels, are held out.
_INSTRUCTION_STYLE_IDS = {
    "train": tuple(range(0, 24)),
    "validation": tuple(range(24, 28)),
    "test": tuple(range(28, 32)),
}
_OPENERS = {
    "train": (
        "Please edit the scene as follows:",
        "Make this change to the existing spatial audio:",
        "Apply the following edit while preserving everything else:",
        "Revise the current scene with this instruction:",
        "For the edited version,",
        "In the current FOA scene,",
    ),
    "validation": (
        "For the revised scene, make this exact change:",
        "Apply this held-out editing request:",
    ),
    "test": (
        "Carry out this spatial-audio edit:",
        "Produce an edited scene with this change:",
    ),
}
_CLOSERS = {
    "train": (
        "Keep every unmentioned source, its timing, and the room unchanged.",
        "Preserve all other events and acoustic properties exactly.",
        "Nothing else in the scene should change.",
        "Leave the remaining content, timing, and acoustics intact.",
    ),
    "validation": (
        "All non-target sources and room properties must remain intact.",
        "Retain the rest of the existing scene without alteration.",
    ),
    "test": (
        "Every aspect outside this requested edit must be preserved.",
        "Do not disturb any other source or acoustic setting.",
    ),
}

_DIRECTION_ANCHORS = (
    ("in front", 0.0),
    ("on the left", 90.0),
    ("behind the listener", -180.0),
    ("on the right", -90.0),
)


@dataclass(frozen=True)
class EditingPairMutation:
    """Complete, renderer-executable truth for one paired edit."""

    pair_id: str
    split: str
    operation_family: str
    operation: str
    instruction: str
    instruction_template_id: str
    old_sceneplan: dict[str, Any]
    new_sceneplan: dict[str, Any]
    source_render_recipe: dict[str, Any]
    target_render_recipe: dict[str, Any]
    edited_source_ids: tuple[str, ...]
    unchanged_source_ids: tuple[str, ...]
    source_members: tuple[dict[str, Any], ...]
    target_members: tuple[dict[str, Any], ...]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def deterministic_u64(*parts: Any, person: bytes = b"edit-pair-v1") -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8, person=person).digest(), "big"
    )


def make_pair_id(
    split: str,
    source_sample_id: str,
    operation: str,
    *,
    nonce: int = 0,
    seed: int = EDITING_PAIR_SEED,
) -> str:
    _validate_split(split)
    _validate_operation(operation)
    if int(seed) != EDITING_PAIR_SEED:
        raise ValueError("Transfusion Editing v1 uses frozen seed 42")
    digest = hashlib.blake2b(
        f"{seed}:{split}:{source_sample_id}:{operation}:{nonce}".encode("utf-8"),
        digest_size=12,
        person=b"edit-id-v1",
    ).hexdigest()
    return f"speditv1_{split}_{digest}"


def target_sample_id(pair_id: str) -> str:
    return f"{pair_id}_target"


def _validate_split(split: str) -> None:
    if split not in _INSTRUCTION_STYLE_IDS:
        raise ValueError(f"unknown Editing split {split!r}")


def _validate_operation(operation: str) -> None:
    if operation not in EDIT_OPERATIONS:
        raise ValueError(f"unknown Editing operation {operation!r}")


def _source_by_id(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(source["source_id"]): source for source in plan["sources"]}


def _recipe_source_by_id(
    recipe: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {str(source["source_id"]): source for source in recipe["sources"]}


def validate_render_recipe_binding(
    plan: Mapping[str, Any], recipe: Mapping[str, Any]
) -> None:
    """Check the exact model/renderer join without loading any source audio."""

    validate_model_sceneplan(plan)
    sample_id = str(plan["sample_id"])
    if recipe.get("sample_id") != sample_id:
        raise ValueError("render recipe sample_id does not match ScenePlan")
    audio = recipe.get("audio_execution")
    if not isinstance(audio, Mapping):
        raise ValueError("render recipe has no audio_execution")
    model_num_samples = int(audio["model_num_samples"])
    frames = int(audio["latent_frames_valid"])
    if frames != math.ceil(model_num_samples / 1024):
        raise ValueError("render recipe latent geometry is inconsistent")
    if not math.isclose(
        float(plan["duration_sec"]),
        model_num_samples / MODEL_SAMPLE_RATE,
        rel_tol=0.0,
        abs_tol=1.1e-6,
    ):
        raise ValueError("render recipe duration differs from ScenePlan")
    plan_sources = _source_by_id(plan)
    recipe_sources = _recipe_source_by_id(recipe)
    if set(plan_sources) != set(recipe_sources):
        raise ValueError("render recipe source ids differ from ScenePlan")
    for source_id, source in plan_sources.items():
        renderer_source = recipe_sources[source_id]
        if renderer_source.get("kind") != source.get("kind"):
            raise ValueError(f"{source_id}: recipe kind differs from ScenePlan")
        window = renderer_source.get("exact_source_sample_window")
        if not isinstance(window, Mapping):
            raise ValueError(f"{source_id}: exact source window is absent")
        onset = int(window["model_onset_sample"])
        offset = int(window["model_offset_sample"])
        dry_start = int(window["dry_start_sample"])
        dry_end = int(window["dry_end_sample"])
        if dry_start != 0 or offset - onset != dry_end - dry_start:
            raise ValueError(f"{source_id}: renderer window is not a complete member")
        activity = source["activity"]
        if not (
            math.isclose(
                float(activity["onset_sec"]),
                onset / MODEL_SAMPLE_RATE,
                rel_tol=0.0,
                abs_tol=1.1e-6,
            )
            and math.isclose(
                float(activity["offset_sec"]),
                offset / MODEL_SAMPLE_RATE,
                rel_tol=0.0,
                abs_tol=1.1e-6,
            )
        ):
            raise ValueError(f"{source_id}: activity and exact window differ")
        asset = renderer_source.get("asset_ref")
        if not isinstance(asset, Mapping) or not str(asset.get("identity_hash") or ""):
            raise ValueError(f"{source_id}: dry-source identity hash is absent")


def _source_semantic_text(source: Mapping[str, Any]) -> str:
    if source["kind"] == "speech":
        return (
            f"the speech saying {json.dumps(str(source['transcript']), ensure_ascii=False)} "
            f"in the voice described as "
            f"{json.dumps(str(source['speaker_description']), ensure_ascii=False)}"
        )
    noun = "music" if source["kind"] == "music" else "sound"
    return f"the {noun} described as {json.dumps(str(source['description']), ensure_ascii=False)}"


def _number(value: Any) -> str:
    text = f"{float(value):.9f}".rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


def _position_text(position: Mapping[str, Any]) -> str:
    return (
        f"azimuth {_number(position['azimuth_deg'])} degrees, "
        f"elevation {_number(position['elevation_deg'])} degrees, and "
        f"distance {_number(position['distance_m'])} meters"
    )


def _trajectory_text(trajectory: Mapping[str, Any]) -> str:
    if trajectory["type"] == "static":
        return "stationary at " + _position_text(trajectory["position"])
    if trajectory["type"] == "linear":
        return (
            "moving in a straight line from "
            + _position_text(trajectory["start"])
            + " to "
            + _position_text(trajectory["end"])
        )
    raise ValueError("Transfusion Editing v1 supports static and linear motion")


def _instruction_style(pair_id: str, split: str) -> tuple[str, str, str]:
    _validate_split(split)
    allowed = _INSTRUCTION_STYLE_IDS[split]
    value = deterministic_u64(EDITING_PAIR_SEED, split, pair_id, person=b"edit-tmpl-v1")
    selected = allowed[value % len(allowed)]
    opener = _OPENERS[split][selected % len(_OPENERS[split])]
    closer = _CLOSERS[split][
        (selected // len(_OPENERS[split])) % len(_CLOSERS[split])
    ]
    return opener, closer, f"edit_ar_exact_v1/{split}/{selected:02d}"


def _render_instruction(
    *,
    pair_id: str,
    split: str,
    operation: str,
    old_source: Mapping[str, Any] | None = None,
    new_source: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    opener, closer, template_id = _instruction_style(pair_id, split)
    if operation == OP_EVENT_REMOVE:
        assert old_source is not None
        body = f"remove {_source_semantic_text(old_source)} completely."
    elif operation == OP_EVENT_ADD:
        assert new_source is not None
        activity = new_source["activity"]
        body = (
            f"add {_source_semantic_text(new_source)}, active from exactly "
            f"{_number(activity['onset_sec'])} seconds through exactly "
            f"{_number(activity['offset_sec'])} seconds and "
            f"{_trajectory_text(new_source['trajectory'])}, at the canonical "
            "zero-decibel source gain."
        )
    elif operation == OP_RELOCATE:
        assert old_source is not None and new_source is not None
        body = (
            f"move {_source_semantic_text(old_source)} to remain "
            f"{_trajectory_text(new_source['trajectory'])}; keep its content "
            "and active interval unchanged."
        )
    elif operation == OP_STATIC_TO_LINEAR:
        assert old_source is not None and new_source is not None
        body = (
            f"change {_source_semantic_text(old_source)} from stationary to "
            f"{_trajectory_text(new_source['trajectory'])}; keep its content "
            "and active interval unchanged."
        )
    elif operation == OP_LINEAR_TO_STATIC:
        assert old_source is not None and new_source is not None
        body = (
            f"stop the motion of {_source_semantic_text(old_source)} and make it "
            f"{_trajectory_text(new_source['trajectory'])}; keep its content "
            "and active interval unchanged."
        )
    else:
        _validate_operation(operation)
        raise AssertionError("unreachable")
    return " ".join((opener, body, closer)), template_id


def _angular_distance(left: float, right: float) -> float:
    return abs(((float(left) - float(right) + 180.0) % 360.0) - 180.0)


def _choose_direction(
    current_azimuth: float, *, pair_id: str, purpose: str
) -> tuple[str, float]:
    eligible = [
        item
        for item in _DIRECTION_ANCHORS
        if _angular_distance(current_azimuth, item[1]) >= 60.0
    ]
    if not eligible:
        raise ValueError("no non-trivial direction anchor is available")
    value = deterministic_u64(pair_id, purpose, person=b"edit-dir-v1")
    return eligible[value % len(eligible)]


def _copy_position(position: Mapping[str, Any]) -> dict[str, float]:
    return {
        "azimuth_deg": float(position["azimuth_deg"]),
        "elevation_deg": float(position["elevation_deg"]),
        "distance_m": float(position["distance_m"]),
    }


def _linear_midpoint(trajectory: Mapping[str, Any]) -> dict[str, float]:
    start = trajectory["start"]
    end = trajectory["end"]
    delta = ((float(end["azimuth_deg"]) - float(start["azimuth_deg"]) + 180.0) % 360.0) - 180.0
    azimuth = float(start["azimuth_deg"]) + 0.5 * delta
    azimuth = ((azimuth + 180.0) % 360.0) - 180.0
    return {
        "azimuth_deg": round(azimuth, 6),
        "elevation_deg": round(
            0.5 * (float(start["elevation_deg"]) + float(end["elevation_deg"])),
            6,
        ),
        "distance_m": round(
            math.sqrt(float(start["distance_m"]) * float(end["distance_m"])),
            6,
        ),
    }


def _empty_source_id(plan: Mapping[str, Any]) -> str:
    occupied = {int(str(source["source_id"])[7:]) for source in plan["sources"]}
    for slot in range(4):
        if slot not in occupied:
            return f"source_{slot}"
    raise ValueError("event addition requires a free persistent source slot")


def _member_record(
    plan_source: Mapping[str, Any], recipe_source: Mapping[str, Any]
) -> dict[str, Any]:
    asset = copy.deepcopy(recipe_source["asset_ref"])
    return {
        "source_id": str(plan_source["source_id"]),
        "kind": str(plan_source["kind"]),
        "asset_id": str(asset["asset_id"]),
        "identity_hash": str(asset["identity_hash"]),
        "asset_ref": asset,
        "exact_source_sample_window": copy.deepcopy(
            recipe_source["exact_source_sample_window"]
        ),
        "activity": copy.deepcopy(plan_source["activity"]),
        "trajectory": copy.deepcopy(plan_source["trajectory"]),
        "gain_db": float(plan_source["gain_db"]),
    }


def render_members(
    plan: Mapping[str, Any], recipe: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    validate_render_recipe_binding(plan, recipe)
    recipe_by_id = _recipe_source_by_id(recipe)
    return tuple(
        _member_record(source, recipe_by_id[str(source["source_id"])])
        for source in plan["sources"]
    )


def _target_recipe(
    source_recipe: Mapping[str, Any],
    new_plan: Mapping[str, Any],
    target_recipe_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    recipe = copy.deepcopy(source_recipe)
    recipe["sample_id"] = str(new_plan["sample_id"])
    recipe["model_sceneplan_sha256"] = sha256_json(new_plan)
    recipe["sources"] = sorted(
        (copy.deepcopy(source) for source in target_recipe_sources),
        key=lambda item: int(str(item["source_id"])[7:]),
    )
    kinds = {str(source["kind"]) for source in new_plan["sources"]}
    has_speech = "speech" in kinds
    has_background = bool(kinds - {"speech"})
    recipe["mixing"] = {
        "speech_background_mode": (
            "overlap_calibrated" if has_speech and has_background else "not_applicable"
        )
    }
    validate_render_recipe_binding(new_plan, recipe)
    return recipe


def _finish_mutation(
    *,
    pair_id: str,
    split: str,
    operation: str,
    instruction: str,
    template_id: str,
    old_plan: Mapping[str, Any],
    new_plan: Mapping[str, Any],
    source_recipe: Mapping[str, Any],
    target_recipe: Mapping[str, Any],
    edited_ids: Sequence[str],
) -> EditingPairMutation:
    old_ids = {str(source["source_id"]) for source in old_plan["sources"]}
    new_ids = {str(source["source_id"]) for source in new_plan["sources"]}
    edited = tuple(sorted(map(str, edited_ids)))
    unchanged = tuple(sorted((old_ids & new_ids) - set(edited)))
    value = EditingPairMutation(
        pair_id=pair_id,
        split=split,
        operation_family=OPERATION_FAMILY[operation],
        operation=operation,
        instruction=instruction,
        instruction_template_id=template_id,
        old_sceneplan=copy.deepcopy(dict(old_plan)),
        new_sceneplan=copy.deepcopy(dict(new_plan)),
        source_render_recipe=copy.deepcopy(dict(source_recipe)),
        target_render_recipe=copy.deepcopy(dict(target_recipe)),
        edited_source_ids=edited,
        unchanged_source_ids=unchanged,
        source_members=render_members(old_plan, source_recipe),
        target_members=render_members(new_plan, target_recipe),
    )
    validate_editing_pair_mutation(value)
    return value


def build_event_removal(
    old_plan: Mapping[str, Any],
    source_recipe: Mapping[str, Any],
    *,
    split: str,
    source_id: str,
    pair_id: str,
) -> EditingPairMutation:
    validate_render_recipe_binding(old_plan, source_recipe)
    sources = _source_by_id(old_plan)
    if len(sources) < 2:
        raise ValueError("event removal cannot leave an empty ScenePlan")
    if source_id not in sources:
        raise ValueError(f"removal source {source_id!r} is absent")
    new_plan = copy.deepcopy(dict(old_plan))
    new_plan["sample_id"] = target_sample_id(pair_id)
    new_plan["sources"] = [
        source
        for source in new_plan["sources"]
        if str(source["source_id"]) != source_id
    ]
    validate_model_sceneplan(new_plan)
    recipe_sources = [
        source
        for source in source_recipe["sources"]
        if str(source["source_id"]) != source_id
    ]
    target_recipe = _target_recipe(source_recipe, new_plan, recipe_sources)
    instruction, template_id = _render_instruction(
        pair_id=pair_id,
        split=split,
        operation=OP_EVENT_REMOVE,
        old_source=sources[source_id],
    )
    return _finish_mutation(
        pair_id=pair_id,
        split=split,
        operation=OP_EVENT_REMOVE,
        instruction=instruction,
        template_id=template_id,
        old_plan=old_plan,
        new_plan=new_plan,
        source_recipe=source_recipe,
        target_recipe=target_recipe,
        edited_ids=(source_id,),
    )


def build_event_addition(
    old_plan: Mapping[str, Any],
    source_recipe: Mapping[str, Any],
    donor_plan_source: Mapping[str, Any],
    donor_recipe_source: Mapping[str, Any],
    *,
    split: str,
    pair_id: str,
) -> EditingPairMutation:
    validate_render_recipe_binding(old_plan, source_recipe)
    if len(old_plan["sources"]) >= 4:
        raise ValueError("event addition requires fewer than four old sources")
    old_has_speech = any(source["kind"] == "speech" for source in old_plan["sources"])
    if old_has_speech and donor_plan_source["kind"] == "speech":
        raise ValueError("a target ScenePlan cannot contain two speech sources")
    if donor_plan_source["trajectory"]["type"] not in {"static", "linear"}:
        raise ValueError("addition donor must use static or linear motion")
    old_identity_hashes = {
        str(source["asset_ref"]["identity_hash"])
        for source in source_recipe["sources"]
    }
    donor_identity = str(donor_recipe_source["asset_ref"]["identity_hash"])
    if donor_identity in old_identity_hashes:
        raise ValueError("addition donor duplicates an existing dry member")

    audio = source_recipe["audio_execution"]
    model_num_samples = int(audio["model_num_samples"])
    donor_window = donor_recipe_source["exact_source_sample_window"]
    donor_samples = int(donor_window["dry_end_sample"]) - int(
        donor_window["dry_start_sample"]
    )
    if donor_samples <= 0 or donor_samples > model_num_samples:
        raise ValueError("addition donor does not fit the source scene duration")
    available = model_num_samples - donor_samples
    onset = (
        deterministic_u64(pair_id, donor_identity, person=b"edit-add-v1")
        % (available + 1)
    )
    offset = onset + donor_samples
    new_source_id = _empty_source_id(old_plan)
    new_source = copy.deepcopy(dict(donor_plan_source))
    new_source["source_id"] = new_source_id
    new_source["gain_db"] = 0.0
    new_source["activity"] = {
        "onset_sec": round(onset / MODEL_SAMPLE_RATE, 6),
        "offset_sec": round(offset / MODEL_SAMPLE_RATE, 6),
    }
    new_recipe_source = copy.deepcopy(dict(donor_recipe_source))
    new_recipe_source["source_id"] = new_source_id
    new_recipe_source["exact_source_sample_window"] = {
        "dry_start_sample": int(donor_window["dry_start_sample"]),
        "dry_end_sample": int(donor_window["dry_end_sample"]),
        "model_onset_sample": onset,
        "model_offset_sample": offset,
    }

    new_plan = copy.deepcopy(dict(old_plan))
    new_plan["sample_id"] = target_sample_id(pair_id)
    new_plan["sources"].append(new_source)
    new_plan["sources"].sort(key=lambda item: int(str(item["source_id"])[7:]))
    validate_model_sceneplan(new_plan)
    target_recipe = _target_recipe(
        source_recipe,
        new_plan,
        [*source_recipe["sources"], new_recipe_source],
    )
    instruction, template_id = _render_instruction(
        pair_id=pair_id,
        split=split,
        operation=OP_EVENT_ADD,
        new_source=new_source,
    )
    return _finish_mutation(
        pair_id=pair_id,
        split=split,
        operation=OP_EVENT_ADD,
        instruction=instruction,
        template_id=template_id,
        old_plan=old_plan,
        new_plan=new_plan,
        source_recipe=source_recipe,
        target_recipe=target_recipe,
        edited_ids=(new_source_id,),
    )


def build_stationary_relocation(
    old_plan: Mapping[str, Any],
    source_recipe: Mapping[str, Any],
    *,
    split: str,
    source_id: str,
    pair_id: str,
) -> EditingPairMutation:
    validate_render_recipe_binding(old_plan, source_recipe)
    sources = _source_by_id(old_plan)
    if source_id not in sources:
        raise ValueError(f"relocation source {source_id!r} is absent")
    old_source = sources[source_id]
    if old_source["trajectory"]["type"] != "static":
        raise ValueError("stationary relocation requires a static source")
    old_position = old_source["trajectory"]["position"]
    _, azimuth = _choose_direction(
        float(old_position["azimuth_deg"]), pair_id=pair_id, purpose="relocate"
    )
    new_plan = copy.deepcopy(dict(old_plan))
    new_plan["sample_id"] = target_sample_id(pair_id)
    new_source = _source_by_id(new_plan)[source_id]
    new_position = _copy_position(old_position)
    new_position["azimuth_deg"] = azimuth
    new_source["trajectory"] = {"type": "static", "position": new_position}
    validate_model_sceneplan(new_plan)
    target_recipe = _target_recipe(source_recipe, new_plan, source_recipe["sources"])
    instruction, template_id = _render_instruction(
        pair_id=pair_id,
        split=split,
        operation=OP_RELOCATE,
        old_source=old_source,
        new_source=new_source,
    )
    return _finish_mutation(
        pair_id=pair_id,
        split=split,
        operation=OP_RELOCATE,
        instruction=instruction,
        template_id=template_id,
        old_plan=old_plan,
        new_plan=new_plan,
        source_recipe=source_recipe,
        target_recipe=target_recipe,
        edited_ids=(source_id,),
    )


def build_static_to_linear(
    old_plan: Mapping[str, Any],
    source_recipe: Mapping[str, Any],
    *,
    split: str,
    source_id: str,
    pair_id: str,
) -> EditingPairMutation:
    validate_render_recipe_binding(old_plan, source_recipe)
    sources = _source_by_id(old_plan)
    if source_id not in sources:
        raise ValueError(f"motion source {source_id!r} is absent")
    old_source = sources[source_id]
    if old_source["trajectory"]["type"] != "static":
        raise ValueError("static-to-linear requires a static source")
    start = _copy_position(old_source["trajectory"]["position"])
    _, azimuth = _choose_direction(
        start["azimuth_deg"], pair_id=pair_id, purpose="static-to-linear"
    )
    end = _copy_position(start)
    end["azimuth_deg"] = azimuth
    new_plan = copy.deepcopy(dict(old_plan))
    new_plan["sample_id"] = target_sample_id(pair_id)
    new_source = _source_by_id(new_plan)[source_id]
    new_source["trajectory"] = {"type": "linear", "start": start, "end": end}
    validate_model_sceneplan(new_plan)
    target_recipe = _target_recipe(source_recipe, new_plan, source_recipe["sources"])
    instruction, template_id = _render_instruction(
        pair_id=pair_id,
        split=split,
        operation=OP_STATIC_TO_LINEAR,
        old_source=old_source,
        new_source=new_source,
    )
    return _finish_mutation(
        pair_id=pair_id,
        split=split,
        operation=OP_STATIC_TO_LINEAR,
        instruction=instruction,
        template_id=template_id,
        old_plan=old_plan,
        new_plan=new_plan,
        source_recipe=source_recipe,
        target_recipe=target_recipe,
        edited_ids=(source_id,),
    )


def build_linear_to_static(
    old_plan: Mapping[str, Any],
    source_recipe: Mapping[str, Any],
    *,
    split: str,
    source_id: str,
    pair_id: str,
) -> EditingPairMutation:
    validate_render_recipe_binding(old_plan, source_recipe)
    sources = _source_by_id(old_plan)
    if source_id not in sources:
        raise ValueError(f"motion source {source_id!r} is absent")
    old_source = sources[source_id]
    if old_source["trajectory"]["type"] != "linear":
        raise ValueError("linear-to-static requires a linear source")
    position = _linear_midpoint(old_source["trajectory"])
    new_plan = copy.deepcopy(dict(old_plan))
    new_plan["sample_id"] = target_sample_id(pair_id)
    new_source = _source_by_id(new_plan)[source_id]
    new_source["trajectory"] = {"type": "static", "position": position}
    validate_model_sceneplan(new_plan)
    target_recipe = _target_recipe(source_recipe, new_plan, source_recipe["sources"])
    instruction, template_id = _render_instruction(
        pair_id=pair_id,
        split=split,
        operation=OP_LINEAR_TO_STATIC,
        old_source=old_source,
        new_source=new_source,
    )
    return _finish_mutation(
        pair_id=pair_id,
        split=split,
        operation=OP_LINEAR_TO_STATIC,
        instruction=instruction,
        template_id=template_id,
        old_plan=old_plan,
        new_plan=new_plan,
        source_recipe=source_recipe,
        target_recipe=target_recipe,
        edited_ids=(source_id,),
    )


def _instruction_forbidden_values(recipe: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for source in recipe["sources"]:
        asset = source["asset_ref"]
        for key in (
            "asset_id",
            "identity_hash",
            "dry_audio_path",
            "parquet_path",
            "parent_asset_id",
        ):
            value = asset.get(key)
            if value:
                values.append(str(value))
    return values


def validate_editing_pair_mutation(pair: EditingPairMutation) -> None:
    """Fail closed on locality, provenance, and Editing-AR leakage."""

    _validate_split(pair.split)
    _validate_operation(pair.operation)
    if pair.operation_family != OPERATION_FAMILY[pair.operation]:
        raise ValueError("operation family does not match operation")
    validate_render_recipe_binding(pair.old_sceneplan, pair.source_render_recipe)
    validate_render_recipe_binding(pair.new_sceneplan, pair.target_render_recipe)
    source_audio = pair.source_render_recipe["audio_execution"]
    target_audio = pair.target_render_recipe["audio_execution"]
    for key in ("model_num_samples", "latent_frames_valid", "vae_padded_num_samples"):
        if int(source_audio[key]) != int(target_audio[key]):
            raise ValueError(f"source/target audio geometry differs at {key}")
    if pair.old_sceneplan["duration_sec"] != pair.new_sceneplan["duration_sec"]:
        raise ValueError("source and target ScenePlan durations differ")
    if pair.old_sceneplan["room"] != pair.new_sceneplan["room"]:
        raise ValueError("editing changed the model-visible room")
    if pair.source_render_recipe["resolved_room"] != pair.target_render_recipe["resolved_room"]:
        raise ValueError("editing changed resolved room acoustics")

    old_plan_sources = _source_by_id(pair.old_sceneplan)
    new_plan_sources = _source_by_id(pair.new_sceneplan)
    old_recipe_sources = _recipe_source_by_id(pair.source_render_recipe)
    new_recipe_sources = _recipe_source_by_id(pair.target_render_recipe)
    for source_id in pair.unchanged_source_ids:
        if old_plan_sources[source_id] != new_plan_sources[source_id]:
            raise ValueError(f"{source_id}: unchanged ScenePlan member drifted")
        if old_recipe_sources[source_id] != new_recipe_sources[source_id]:
            raise ValueError(f"{source_id}: unchanged renderer member drifted")

    old_ids = set(old_plan_sources)
    new_ids = set(new_plan_sources)
    edited = set(pair.edited_source_ids)
    if pair.operation == OP_EVENT_ADD:
        if not (len(new_ids) == len(old_ids) + 1 and new_ids - old_ids == edited):
            raise ValueError("event addition source-id delta is invalid")
    elif pair.operation == OP_EVENT_REMOVE:
        if not (len(new_ids) + 1 == len(old_ids) and old_ids - new_ids == edited):
            raise ValueError("event removal source-id delta is invalid")
    else:
        if old_ids != new_ids or len(edited) != 1:
            raise ValueError("control edit changed the source inventory")
        source_id = next(iter(edited))
        old_source = old_plan_sources[source_id]
        new_source = new_plan_sources[source_id]
        for key in set(old_source) | set(new_source):
            if key == "trajectory":
                continue
            if old_source.get(key) != new_source.get(key):
                raise ValueError(f"{source_id}: control edit changed non-trajectory {key}")
        if old_recipe_sources[source_id] != new_recipe_sources[source_id]:
            raise ValueError(f"{source_id}: control edit changed dry member provenance")

    forbidden = {
        str(pair.old_sceneplan["sample_id"]),
        str(pair.new_sceneplan["sample_id"]),
        *_instruction_forbidden_values(pair.source_render_recipe),
        *_instruction_forbidden_values(pair.target_render_recipe),
    }
    leaked = sorted(value for value in forbidden if value and value in pair.instruction)
    if leaked:
        raise ValueError("renderer/provenance value leaked into edit instruction")
    if not pair.instruction.strip():
        raise ValueError("empty edit instruction")


def eligible_source_ids(plan: Mapping[str, Any], operation: str) -> tuple[str, ...]:
    """Return deterministic legal edited-source choices for an operation."""

    validate_model_sceneplan(plan)
    _validate_operation(operation)
    if operation == OP_EVENT_ADD:
        return ("__new_source__",) if len(plan["sources"]) < 4 else ()
    if operation == OP_EVENT_REMOVE:
        return (
            tuple(str(source["source_id"]) for source in plan["sources"])
            if len(plan["sources"]) > 1
            else ()
        )
    motion = "static" if operation in {OP_RELOCATE, OP_STATIC_TO_LINEAR} else "linear"
    return tuple(
        str(source["source_id"])
        for source in plan["sources"]
        if source["trajectory"]["type"] == motion
    )


__all__ = [
    "EDITING_INSTRUCTION_CONTRACT",
    "EDITING_PAIR_CONTRACT",
    "EDITING_PAIR_SEED",
    "EDIT_OPERATIONS",
    "EditingPairMutation",
    "OPERATION_FAMILY",
    "OP_EVENT_ADD",
    "OP_EVENT_REMOVE",
    "OP_LINEAR_TO_STATIC",
    "OP_RELOCATE",
    "OP_STATIC_TO_LINEAR",
    "build_event_addition",
    "build_event_removal",
    "build_linear_to_static",
    "build_static_to_linear",
    "build_stationary_relocation",
    "canonical_json",
    "deterministic_u64",
    "eligible_source_ids",
    "make_pair_id",
    "render_members",
    "sha256_json",
    "target_sample_id",
    "validate_editing_pair_mutation",
    "validate_render_recipe_binding",
]
