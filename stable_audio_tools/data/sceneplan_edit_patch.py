"""Versioned atomic ScenePlan edits for the active audio-aware P11 route.

The observed ScenePlan is the only state to which a patch may be applied.
The deterministic executor below is consequently the sole producer of the
revised ScenePlan.  Existing source slots are persistent, removal leaves a
hole, and addition consumes the first free slot.

The v2 grammar is a compatibility extension over the frozen codec-v4
vocabulary; the codec-v4 artifact itself is never modified.  Two relative v1
operations remain decodable solely for retired-checkpoint audit, but active
constrained decoding and new data generation cannot emit them.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch

from .model_sceneplan import (
    MAX_SOURCES,
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
    validate_model_sceneplan,
)
from .model_sceneplan_codec import ModelScenePlanCodecError
from .model_sceneplan_codec_v3 import (
    KIND_TOKENS,
    MOTION_TOKENS,
    ROOM_TOKENS,
    SOURCE_SLOT_TOKENS,
    ModelScenePlanCodecV3,
)
from .scene_plan import (
    LOSS_GRAMMAR,
    LOSS_MOTION,
    LOSS_ROOM,
    LOSS_SEMANTIC,
    LOSS_SPATIAL_METRIC,
    LOSS_SPEECH_CONTENT,
)


PATCH_CODEC_NAME = "sceneplan_edit_patch_v2"
PATCH_OUTPUT_CONTRACT = "observed_plan_atomic_patch_revised_plan_v1"
PATCH_MAX_TOKENS = 512
PATCH_TEXT_FIELD_MAX_TOKENS = 192
ROOM_ORDER = ("dry", "moderate", "reverberant", "outdoor")
RETIME_POLICY = "p10_frame_multiscale_v1"
RETIME_FRAME_LEVELS = (4, 8, 16, 32, 64)
RETIME_MODES = (
    "shift_later",
    "shift_earlier",
    "trim_start",
    "trim_end",
    "extend_earlier",
    "extend_later",
)

# Keep the original twelve ids in their historical order.  New ids are an
# append-only compatibility extension in codec-v4's unused reserve.
PATCH_TOKENS = (
    "<patch_bos>",
    "<patch_eos>",
    "<op_keep>",
    "<op_set_room>",
    "<op_rotate_source>",
    "<op_scale_distance>",
    "<op_set_activity>",
    "<op_remove_source>",
    "<delta_azimuth_minus_045>",
    "<delta_azimuth_plus_045>",
    "<distance_scale_075>",
    "<distance_scale_125>",
    "<op_add_source>",
    "<op_replace_source>",
    "<op_move_source>",
    "<op_change_speech_description>",
    "<op_change_transcript>",
)

ACTIVE_OPERATION_TOKENS = {
    "no_op": "<op_keep>",
    "add_source": "<op_add_source>",
    "remove_source": "<op_remove_source>",
    "replace_source": "<op_replace_source>",
    "move_source": "<op_move_source>",
    "retime_source": "<op_set_activity>",
    "room_change": "<op_set_room>",
    "change_speech_description": "<op_change_speech_description>",
    "change_transcript": "<op_change_transcript>",
}

# Read-only compatibility for frozen v4 evidence.  These names are excluded
# from active allowed-next-token sets and from make_deterministic_edit().
LEGACY_OPERATION_TOKENS = {
    "rotate_source": "<op_rotate_source>",
    "distance_source": "<op_scale_distance>",
}
OPERATION_TOKENS = {**ACTIVE_OPERATION_TOKENS, **LEGACY_OPERATION_TOKENS}


class _NeedPatchToken(Exception):
    def __init__(self, allowed: set[int]):
        super().__init__()
        self.allowed = allowed


def _source_slot(source_id: Any) -> int:
    text = str(source_id)
    if text not in {f"source_{index}" for index in range(MAX_SOURCES)}:
        raise ModelScenePlanCodecError(f"invalid patch source id {text!r}")
    return int(text[7:])


def _positions(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    trajectory = source["trajectory"]
    motion = str(trajectory["type"])
    if motion == "static":
        return [trajectory["position"]]
    if motion == "linear":
        return [trajectory["start"], trajectory["end"]]
    raise ModelScenePlanCodecError("P11 editing supports static/linear motion only")


def _frame(seconds: Any) -> int:
    return int(round(float(seconds) * MODEL_SAMPLE_RATE / VAE_HOP_SAMPLES))


def _seconds(frame: int) -> float:
    return float(int(frame) * VAE_HOP_SAMPLES / MODEL_SAMPLE_RATE)


def _first_free_source_id(plan: Mapping[str, Any]) -> str:
    occupied = {_source_slot(source["source_id"]) for source in plan["sources"]}
    for slot in range(MAX_SOURCES):
        if slot not in occupied:
            return f"source_{slot}"
    raise ModelScenePlanCodecError("ADD_SOURCE requires one free P10 source slot")


def _shift_position(position: Mapping[str, Any], *, ordinal: int) -> dict[str, float]:
    azimuth_delta = 45 if int(ordinal) % 2 == 0 else -45
    elevation_delta = 15 if (int(ordinal) // 2) % 2 == 0 else -15
    factor = 1.25 if (int(ordinal) // 4) % 2 == 0 else 0.75
    return {
        "azimuth_deg": float(
            ((int(round(float(position["azimuth_deg"]))) + azimuth_delta + 180) % 360)
            - 180
        ),
        "elevation_deg": float(
            max(-90, min(90, int(round(float(position["elevation_deg"]))) + elevation_delta))
        ),
        "distance_m": float(position["distance_m"]) * factor,
    }


def _retime_source(
    plan: dict[str, Any],
    source: dict[str, Any],
    *,
    ordinal: int,
    seed: int,
) -> tuple[str, dict[str, Any]] | None:
    """Apply a deterministic, multi-scale edit on P10's native frame grid.

    The retired data route moved every source by only one 1024-sample VAE
    frame.  In the normalized ExecutionState that target is roughly 0.002,
    making retime supervision two orders of magnitude smaller than add/move.
    This policy deliberately spans 4--64 frames and rotates across translation,
    trimming, and extension while preserving a legal half-open interval.
    """

    duration = _frame(plan["duration_sec"])
    onset = _frame(source["activity"]["onset_sec"])
    offset = _frame(source["activity"]["offset_sec"])
    level_index = (int(ordinal) * 11 + int(seed)) % len(RETIME_FRAME_LEVELS)
    requested_magnitude = RETIME_FRAME_LEVELS[level_index]
    magnitudes = [
        requested_magnitude,
        *reversed(RETIME_FRAME_LEVELS[:level_index]),
    ]
    mode_index = (int(ordinal) * 7 + int(seed)) % len(RETIME_MODES)
    modes = RETIME_MODES[mode_index:] + RETIME_MODES[:mode_index]

    selected: tuple[int, int, str, int] | None = None
    for magnitude in magnitudes:
        for mode in modes:
            if mode == "shift_later":
                candidate = (onset + magnitude, offset + magnitude)
            elif mode == "shift_earlier":
                candidate = (onset - magnitude, offset - magnitude)
            elif mode == "trim_start":
                candidate = (onset + magnitude, offset)
            elif mode == "trim_end":
                candidate = (onset, offset - magnitude)
            elif mode == "extend_earlier":
                candidate = (onset - magnitude, offset)
            else:
                candidate = (onset, offset + magnitude)
            new_onset, new_offset = candidate
            if (
                0 <= new_onset < new_offset <= duration
                and (new_onset, new_offset) != (onset, offset)
            ):
                selected = (new_onset, new_offset, mode, magnitude)
                break
        if selected is not None:
            break
    if selected is None:
        return None
    new_onset, new_offset, mode, magnitude = selected
    source["activity"] = {
        "onset_sec": _seconds(new_onset),
        "offset_sec": _seconds(new_offset),
    }
    return (
        f"Set {source['source_id']} activity to exact frame interval "
        f"[{new_onset},{new_offset}).",
        {
            "source_id": source["source_id"],
            "new_interval_frames": [new_onset, new_offset],
            "retime_policy": RETIME_POLICY,
            "retime_mode": mode,
            "retime_magnitude_frames": magnitude,
            "changed_paths": [f"sources.{source['source_id']}.activity"],
        },
    )


def _replacement_source(source: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    output = copy.deepcopy(dict(source))
    if str(source["kind"]) == "speech":
        output.pop("speaker_description")
        output.pop("transcript")
        output["kind"] = "sound"
        output["description"] = "a clear wooden knock"
    else:
        output["kind"] = "music" if str(source["kind"]) == "sound" else "sound"
        output["description"] = (
            "a gentle cello phrase" if output["kind"] == "music" else "a soft glass chime"
        )
    output["gain_db"] = 0.0
    return output


def _added_source(plan: Mapping[str, Any], *, ordinal: int) -> dict[str, Any]:
    duration = _frame(plan["duration_sec"])
    slot = _first_free_source_id(plan)
    onset = min(max(0, duration // 4), duration - 1)
    offset = max(onset + 1, min(duration, max(1, 3 * duration // 4)))
    position = {
        "azimuth_deg": float((-135, -45, 45, 135)[int(ordinal) % 4]),
        "elevation_deg": float((-15, 0, 15)[int(ordinal) % 3]),
        "distance_m": float((0.75, 1.5, 3.0)[int(ordinal) % 3]),
    }
    kind = "music" if int(ordinal) % 2 == 0 else "sound"
    return {
        "source_id": slot,
        "kind": kind,
        "description": (
            "a gentle cello phrase" if kind == "music" else "a soft glass chime"
        ),
        "activity": {"onset_sec": _seconds(onset), "offset_sec": _seconds(offset)},
        "trajectory": {"type": "static", "position": position},
        "gain_db": 0.0,
    }


def make_deterministic_edit(
    codec: ModelScenePlanCodecV3,
    base_plan: Mapping[str, Any],
    *,
    ordinal: int,
    seed: int,
    requested_operation: str | None = None,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    """Create one deterministic v2 edit whose target is executable by P10."""

    original = codec.project_plan(base_plan)
    if any(str(source["trajectory"]["type"]) not in {"static", "linear"} for source in original["sources"]):
        raise ModelScenePlanCodecError("audio-aware P11 edits require static/linear sources")
    candidates = [
        "room_change",
        "move_source",
        "retime_source",
        "replace_source",
        "no_op",
    ]
    if len(original["sources"]) > 1:
        candidates.append("remove_source")
    if len(original["sources"]) < MAX_SOURCES:
        candidates.append("add_source")
    if any(str(source["kind"]) == "speech" for source in original["sources"]):
        candidates.extend(["change_speech_description", "change_transcript"])
    if requested_operation is None:
        operation = candidates[(int(ordinal) * 17 + int(seed)) % len(candidates)]
    else:
        operation = str(requested_operation)
        if operation not in candidates:
            raise ModelScenePlanCodecError(
                f"requested operation {operation!r} is invalid for this observed scene"
            )
    target = copy.deepcopy(original)
    source_index = (int(ordinal) * 13 + int(seed)) % len(target["sources"])
    source = target["sources"][source_index]

    if operation == "room_change":
        old_room = str(target["room"]["type"])
        new_room = ROOM_ORDER[(ROOM_ORDER.index(old_room) + 1 + (ordinal + seed) % 3) % len(ROOM_ORDER)]
        target["room"] = {"type": new_room}
        instruction = f"Change only the acoustic room from {old_room} to {new_room}."
        spec = {"new_room": new_room, "changed_paths": ["room.type"]}
    elif operation == "move_source":
        trajectory = copy.deepcopy(source["trajectory"])
        if trajectory["type"] == "static":
            trajectory["position"] = _shift_position(trajectory["position"], ordinal=ordinal + seed)
        else:
            trajectory["start"] = _shift_position(trajectory["start"], ordinal=ordinal + seed)
            trajectory["end"] = _shift_position(trajectory["end"], ordinal=ordinal + seed + 1)
        source["trajectory"] = trajectory
        instruction = (
            f"Move only {source['source_id']} to this exact executable "
            f"trajectory: {trajectory}."
        )
        spec = {
            "source_id": source["source_id"],
            "trajectory": trajectory,
            "changed_paths": [f"sources.{source['source_id']}.trajectory"],
        }
    elif operation == "retime_source":
        retimed = _retime_source(
            target,
            source,
            ordinal=ordinal,
            seed=seed,
        )
        if retimed is None:
            operation = "move_source"
            trajectory = copy.deepcopy(source["trajectory"])
            for position in _positions({"trajectory": trajectory}):
                position.update(_shift_position(position, ordinal=ordinal + seed))
            source["trajectory"] = trajectory
            instruction = (
                f"Move only {source['source_id']} to this exact executable "
                f"trajectory: {trajectory}."
            )
            spec = {
                "source_id": source["source_id"],
                "trajectory": trajectory,
                "changed_paths": [f"sources.{source['source_id']}.trajectory"],
                "fallback_from": "retime_source",
            }
        else:
            instruction, spec = retimed
    elif operation == "replace_source":
        replacement = _replacement_source(source, ordinal=ordinal + seed)
        target["sources"][source_index] = replacement
        instruction = (
            f"Replace only {source['source_id']} with {replacement['description']}; "
            "keep its executable timing and trajectory."
        )
        spec = {
            "source_id": source["source_id"],
            "new_source": replacement,
            "changed_paths": [f"sources.{source['source_id']}"],
        }
    elif operation == "remove_source":
        removed = target["sources"].pop(source_index)
        instruction = f"Remove {removed['source_id']} completely from the scene."
        spec = {
            "source_id": removed["source_id"],
            "changed_paths": [f"sources.{removed['source_id']}"],
        }
    elif operation == "add_source":
        added = _added_source(target, ordinal=ordinal + seed)
        target["sources"].append(added)
        target["sources"].sort(key=lambda item: _source_slot(item["source_id"]))
        instruction = (
            f"Add {added['description']} as {added['source_id']}; use activity "
            f"{added['activity']} and exact trajectory {added['trajectory']}."
        )
        spec = {
            "source": added,
            "changed_paths": [f"sources.{added['source_id']}"],
        }
    elif operation == "change_speech_description":
        source = next(value for value in target["sources"] if value["kind"] == "speech")
        new_text = "a calm close-miked narrator"
        source["speaker_description"] = new_text
        instruction = f"Change only {source['source_id']} speaker description to: {new_text}."
        spec = {
            "source_id": source["source_id"],
            "new_speaker_description": new_text,
            "changed_paths": [f"sources.{source['source_id']}.speaker_description"],
        }
    elif operation == "change_transcript":
        source = next(value for value in target["sources"] if value["kind"] == "speech")
        new_text = "Please meet me beside the quiet station."
        source["transcript"] = new_text
        instruction = f"Change only {source['source_id']} exact transcript to: {new_text}"
        spec = {
            "source_id": source["source_id"],
            "new_transcript": new_text,
            "changed_paths": [f"sources.{source['source_id']}.transcript"],
        }
    else:
        instruction = "Make no ScenePlan change; keep the observed scene exactly as it is."
        spec = {"changed_paths": []}

    target = codec.project_plan(target)
    spec = {
        "contract": "audio_aware_atomic_patch_v1",
        "operation": operation,
        "preserve_all_unspecified_fields": True,
        **spec,
    }
    return target, instruction, operation, spec


class ScenePlanEditPatchCodec:
    """Encode, constrain, decode, and execute one audio-aware atomic patch."""

    def __init__(self, plan_codec: ModelScenePlanCodecV3) -> None:
        if not isinstance(plan_codec, ModelScenePlanCodecV3):
            raise TypeError("the patch codec requires ModelScenePlanCodecV3/v4")
        self.plan_codec = plan_codec
        offset = int(plan_codec.details["used_vocab_size"])
        if offset + len(PATCH_TOKENS) > int(plan_codec.vocab_size):
            raise ModelScenePlanCodecError("the ScenePlan vocabulary has no patch-id reserve")
        self.token_to_id = {token: offset + index for index, token in enumerate(PATCH_TOKENS)}
        self.id_to_token = {value: key for key, value in self.token_to_id.items()}
        self.vocab_size = int(plan_codec.vocab_size)
        self.used_token_ids = frozenset(self.token_to_id.values())
        self.nonempty_first_text_ids = {
            plan_codec.text_offset + piece_id
            for piece_id in range(plan_codec.text_vocab_size)
            if plan_codec.text_processor.decode([piece_id]).strip()
        }

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<patch_bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<patch_eos>"]

    @property
    def fingerprint(self) -> str:
        return f"{PATCH_CODEC_NAME}:{self.plan_codec.fingerprint}"

    def _source_id(self, value: Any) -> int:
        return self.plan_codec._tid(SOURCE_SLOT_TOKENS[_source_slot(value)])

    def _room_id(self, value: Any) -> int:
        text = str(value)
        if text not in ROOM_TOKENS:
            raise ModelScenePlanCodecError(f"invalid patch room {text!r}")
        return self.plan_codec._tid(ROOM_TOKENS[text])

    def _position_ids(self, value: Mapping[str, Any]) -> list[int]:
        if not isinstance(value, Mapping) or set(value) != {
            "azimuth_deg", "elevation_deg", "distance_m"
        }:
            raise ModelScenePlanCodecError("patch position fields changed")
        return [
            self.plan_codec._tid("<azimuth_bin>"),
            self.plan_codec.azimuth_ids[self.plan_codec._azimuth_value(value["azimuth_deg"]) + 180],
            self.plan_codec._tid("<elevation_bin>"),
            self.plan_codec.elevation_ids[self.plan_codec._elevation_value(value["elevation_deg"]) + 90],
            self.plan_codec._tid("<distance_bin>"),
            self.plan_codec.distance_ids[self.plan_codec._distance_index(value["distance_m"])],
        ]

    def _text_ids(self, value: Any) -> list[int]:
        ids = self.plan_codec._text(value)
        if len(ids) - 2 > PATCH_TEXT_FIELD_MAX_TOKENS:
            raise ModelScenePlanCodecError(
                f"patch text needs {len(ids) - 2} pieces > {PATCH_TEXT_FIELD_MAX_TOKENS}"
            )
        return ids

    def _trajectory_ids(self, trajectory: Mapping[str, Any]) -> list[int]:
        if not isinstance(trajectory, Mapping):
            raise ModelScenePlanCodecError("patch trajectory must be an object")
        motion = str(trajectory.get("type") or "")
        if motion not in {"static", "linear"}:
            raise ModelScenePlanCodecError("patch motion must be static or linear")
        ids = [self.plan_codec._tid("<trajectory_begin>"), self.plan_codec._tid(MOTION_TOKENS[motion])]
        if motion == "static":
            if set(trajectory) != {"type", "position"}:
                raise ModelScenePlanCodecError("static patch trajectory fields changed")
            ids.extend([self.plan_codec._tid("<position>"), *self._position_ids(trajectory["position"])])
        else:
            if set(trajectory) != {"type", "start", "end"}:
                raise ModelScenePlanCodecError("linear patch trajectory fields changed")
            ids.extend([self.plan_codec._tid("<start>"), *self._position_ids(trajectory["start"])])
            ids.extend([self.plan_codec._tid("<end>"), *self._position_ids(trajectory["end"])])
        ids.append(self.plan_codec._tid("<trajectory_end>"))
        return ids

    def _source_ids(self, source: Mapping[str, Any]) -> list[int]:
        if not isinstance(source, Mapping):
            raise ModelScenePlanCodecError("patch source must be an object")
        kind = str(source.get("kind") or "")
        common = {"source_id", "kind", "activity", "trajectory", "gain_db"}
        semantic = {"speaker_description", "transcript"} if kind == "speech" else {"description"}
        if kind not in KIND_TOKENS or set(source) != common | semantic:
            raise ModelScenePlanCodecError("patch source fields do not match its kind")
        activity = source["activity"]
        if not isinstance(activity, Mapping) or set(activity) != {"onset_sec", "offset_sec"}:
            raise ModelScenePlanCodecError("patch source activity fields changed")
        onset = _frame(activity["onset_sec"])
        offset = _frame(activity["offset_sec"])
        if not 0 <= onset < offset <= self.plan_codec.max_frames:
            raise ModelScenePlanCodecError("patch source activity is outside P10 grid")
        ids = [
            self.plan_codec._tid("<source_begin>"),
            self._source_id(source["source_id"]),
            self.plan_codec._tid("<kind>"),
            self.plan_codec._tid(KIND_TOKENS[kind]),
        ]
        if kind == "speech":
            ids.extend([self.plan_codec._tid("<speaker_description>"), *self._text_ids(source["speaker_description"])])
            ids.extend([self.plan_codec._tid("<transcript>"), *self._text_ids(source["transcript"])])
        else:
            ids.extend([self.plan_codec._tid("<description>"), *self._text_ids(source["description"])])
        ids.extend(
            [
                self.plan_codec._tid("<activity_begin>"),
                self.plan_codec._tid("<onset_frame>"),
                self.plan_codec.frame_ids[onset],
                self.plan_codec._tid("<offset_frame>"),
                self.plan_codec.frame_ids[offset],
                self.plan_codec._tid("<activity_end>"),
                *self._trajectory_ids(source["trajectory"]),
                self.plan_codec._tid("<source_end>"),
            ]
        )
        return ids

    def encode(self, spec: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        operation = str(spec.get("operation") or "")
        if operation not in OPERATION_TOKENS:
            raise ModelScenePlanCodecError(f"unsupported edit operation {operation!r}")
        ids = [self.bos_id, self.token_to_id[OPERATION_TOKENS[operation]]]
        groups = [LOSS_GRAMMAR, LOSS_GRAMMAR]

        def append(values: Sequence[int], group: int) -> None:
            ids.extend(int(value) for value in values)
            groups.extend([int(group)] * len(values))

        if operation == "room_change":
            append([self._room_id(spec["new_room"])], LOSS_ROOM)
        elif operation in {"remove_source", "change_speech_description", "change_transcript"}:
            append([self._source_id(spec["source_id"])], LOSS_SEMANTIC)
            if operation == "change_speech_description":
                append([self.plan_codec._tid("<speaker_description>")], LOSS_SEMANTIC)
                append(self._text_ids(spec["new_speaker_description"]), LOSS_SEMANTIC)
            elif operation == "change_transcript":
                append([self.plan_codec._tid("<transcript>")], LOSS_SPEECH_CONTENT)
                append(self._text_ids(spec["new_transcript"]), LOSS_SPEECH_CONTENT)
        elif operation in {"add_source", "replace_source"}:
            source = spec["source"] if operation == "add_source" else spec["new_source"]
            if operation == "replace_source" and str(source["source_id"]) != str(spec["source_id"]):
                raise ModelScenePlanCodecError("replacement must retain its source id")
            append(self._source_ids(source), LOSS_SEMANTIC)
        elif operation == "move_source":
            append([self._source_id(spec["source_id"])], LOSS_SEMANTIC)
            append(self._trajectory_ids(spec["trajectory"]), LOSS_MOTION)
        elif operation == "retime_source":
            append([self._source_id(spec["source_id"])], LOSS_SEMANTIC)
            interval = spec.get("new_interval_frames")
            if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)) or len(interval) != 2:
                raise ModelScenePlanCodecError("activity patch requires two frame bounds")
            onset, offset = int(interval[0]), int(interval[1])
            if not 0 <= onset < offset <= self.plan_codec.max_frames:
                raise ModelScenePlanCodecError("activity patch lies outside the P10 frame grid")
            append([self.plan_codec.frame_ids[onset], self.plan_codec.frame_ids[offset]], LOSS_MOTION)
        elif operation == "rotate_source":
            append([self._source_id(spec["source_id"])], LOSS_SEMANTIC)
            token = {-45: "<delta_azimuth_minus_045>", 45: "<delta_azimuth_plus_045>"}.get(int(spec["delta_azimuth_deg"]))
            if token is None:
                raise ModelScenePlanCodecError("legacy rotation must be exactly +/-45 degrees")
            append([self.token_to_id[token]], LOSS_SPATIAL_METRIC)
        elif operation == "distance_source":
            append([self._source_id(spec["source_id"])], LOSS_SEMANTIC)
            token = {0.75: "<distance_scale_075>", 1.25: "<distance_scale_125>"}.get(round(float(spec["distance_factor"]), 2))
            if token is None:
                raise ModelScenePlanCodecError("legacy distance scale must be 0.75 or 1.25")
            append([self.token_to_id[token]], LOSS_SPATIAL_METRIC)
        ids.append(self.eos_id)
        groups.append(LOSS_GRAMMAR)
        if len(ids) > PATCH_MAX_TOKENS:
            raise ModelScenePlanCodecError("patch codec exceeded its fixed token ceiling")
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool),
            "loss_group_ids": torch.tensor(groups, dtype=torch.long),
        }

    def _decode_position(self, cur: ModelScenePlanCodecV3._Decoder, token: str) -> dict[str, float]:
        cur.expect(token)
        cur.expect("<azimuth_bin>")
        azimuth_id = cur.take()
        cur.expect("<elevation_bin>")
        elevation_id = cur.take()
        cur.expect("<distance_bin>")
        distance_id = cur.take()
        try:
            return {
                "azimuth_deg": float(self.plan_codec.azimuth_ids.index(azimuth_id) - 180),
                "elevation_deg": float(self.plan_codec.elevation_ids.index(elevation_id) - 90),
                "distance_m": self.plan_codec._distance_value(self.plan_codec.distance_ids.index(distance_id)),
            }
        except ValueError as error:
            raise ModelScenePlanCodecError("patch position contains an invalid bin") from error

    def _decode_trajectory(self, cur: ModelScenePlanCodecV3._Decoder) -> dict[str, Any]:
        cur.expect("<trajectory_begin>")
        motion_id = cur.take()
        if motion_id == self.plan_codec._tid(MOTION_TOKENS["static"]):
            result = {"type": "static", "position": self._decode_position(cur, "<position>")}
        elif motion_id == self.plan_codec._tid(MOTION_TOKENS["linear"]):
            result = {
                "type": "linear",
                "start": self._decode_position(cur, "<start>"),
                "end": self._decode_position(cur, "<end>"),
            }
        else:
            raise ModelScenePlanCodecError("patch motion must be static or linear")
        cur.expect("<trajectory_end>")
        return result

    def _decode_source(self, cur: ModelScenePlanCodecV3._Decoder) -> dict[str, Any]:
        cur.expect("<source_begin>")
        source_lookup = {
            self.plan_codec._tid(token): f"source_{index}"
            for index, token in enumerate(SOURCE_SLOT_TOKENS)
        }
        source_id_token = cur.take()
        if source_id_token not in source_lookup:
            raise ModelScenePlanCodecError("patch source has invalid slot")
        cur.expect("<kind>")
        kind_lookup = {self.plan_codec._tid(token): kind for kind, token in KIND_TOKENS.items()}
        kind_token = cur.take()
        if kind_token not in kind_lookup:
            raise ModelScenePlanCodecError("patch source has invalid kind")
        kind = kind_lookup[kind_token]
        result: dict[str, Any] = {"source_id": source_lookup[source_id_token], "kind": kind}
        if kind == "speech":
            cur.expect("<speaker_description>")
            result["speaker_description"] = cur.text()
            cur.expect("<transcript>")
            result["transcript"] = cur.text()
        else:
            cur.expect("<description>")
            result["description"] = cur.text()
        cur.expect("<activity_begin>")
        cur.expect("<onset_frame>")
        onset = cur.frame()
        cur.expect("<offset_frame>")
        offset = cur.frame()
        if not 0 <= onset < offset <= self.plan_codec.max_frames:
            raise ModelScenePlanCodecError("patch source activity is invalid")
        cur.expect("<activity_end>")
        result["activity"] = {"onset_sec": _seconds(onset), "offset_sec": _seconds(offset)}
        result["trajectory"] = self._decode_trajectory(cur)
        result["gain_db"] = 0.0
        cur.expect("<source_end>")
        return result

    def decode(self, token_ids: Sequence[int] | torch.Tensor) -> dict[str, Any]:
        values = torch.as_tensor(token_ids, dtype=torch.long).flatten().tolist()
        if len(values) < 3 or len(values) > PATCH_MAX_TOKENS or values[0] != self.bos_id or values[-1] != self.eos_id:
            raise ModelScenePlanCodecError("patch must contain BOS, one operation, and EOS")
        operations = {
            self.token_to_id[token]: operation for operation, token in OPERATION_TOKENS.items()
        }
        operation = operations.get(int(values[1]))
        if operation is None:
            raise ModelScenePlanCodecError("patch contains an unknown operation")
        cur = self.plan_codec._Decoder(self.plan_codec, values[2:-1])
        source_lookup = {
            self.plan_codec._tid(token): f"source_{index}"
            for index, token in enumerate(SOURCE_SLOT_TOKENS)
        }

        def source_id() -> str:
            token = cur.take()
            if token not in source_lookup:
                raise ModelScenePlanCodecError("patch contains an invalid source owner")
            return source_lookup[token]

        result: dict[str, Any] = {"operation": operation}
        if operation == "no_op":
            pass
        elif operation == "room_change":
            room_lookup = {self.plan_codec._tid(token): room for room, token in ROOM_TOKENS.items()}
            token = cur.take()
            if token not in room_lookup:
                raise ModelScenePlanCodecError("SET_ROOM patch requires one room")
            result["new_room"] = room_lookup[token]
        elif operation == "remove_source":
            result["source_id"] = source_id()
        elif operation == "add_source":
            result["source"] = self._decode_source(cur)
        elif operation == "replace_source":
            result["new_source"] = self._decode_source(cur)
            result["source_id"] = result["new_source"]["source_id"]
        elif operation == "move_source":
            result["source_id"] = source_id()
            result["trajectory"] = self._decode_trajectory(cur)
        elif operation == "retime_source":
            result["source_id"] = source_id()
            onset, offset = cur.frame(), cur.frame()
            if not 0 <= onset < offset <= self.plan_codec.max_frames:
                raise ModelScenePlanCodecError("SET_ACTIVITY frame interval is invalid")
            result["new_interval_frames"] = [onset, offset]
        elif operation in {"change_speech_description", "change_transcript"}:
            result["source_id"] = source_id()
            marker = "<speaker_description>" if operation == "change_speech_description" else "<transcript>"
            cur.expect(marker)
            result[
                "new_speaker_description" if operation == "change_speech_description" else "new_transcript"
            ] = cur.text()
        elif operation == "rotate_source":
            result["source_id"] = source_id()
            lookup = {
                self.token_to_id["<delta_azimuth_minus_045>"]: -45,
                self.token_to_id["<delta_azimuth_plus_045>"]: 45,
            }
            token = cur.take()
            if token not in lookup:
                raise ModelScenePlanCodecError("legacy ROTATE patch has invalid delta")
            result["delta_azimuth_deg"] = lookup[token]
        elif operation == "distance_source":
            result["source_id"] = source_id()
            lookup = {
                self.token_to_id["<distance_scale_075>"]: 0.75,
                self.token_to_id["<distance_scale_125>"]: 1.25,
            }
            token = cur.take()
            if token not in lookup:
                raise ModelScenePlanCodecError("legacy DISTANCE patch has invalid scale")
            result["distance_factor"] = lookup[token]
        if cur.index != len(cur.values):
            raise ModelScenePlanCodecError("tokens remain after a complete patch")
        return result

    def canonicalize(self, token_ids: Sequence[int] | torch.Tensor) -> dict[str, torch.Tensor]:
        return self.encode(self.decode(token_ids))

    def _active_operations(self, plan: Mapping[str, Any]) -> set[str]:
        operations = set(ACTIVE_OPERATION_TOKENS)
        if len(plan["sources"]) == 1:
            operations.remove("remove_source")
        if len(plan["sources"]) == MAX_SOURCES:
            operations.remove("add_source")
        if not any(str(source["kind"]) == "speech" for source in plan["sources"]):
            operations.remove("change_speech_description")
            operations.remove("change_transcript")
        return operations

    def allowed_next_ids(
        self,
        prefix: Sequence[int] | torch.Tensor,
        *,
        input_sceneplan: Mapping[str, Any],
    ) -> set[int]:
        """Return active v2 grammar choices conditioned on the observed plan."""

        plan = self.plan_codec.project_plan(validate_model_sceneplan(input_sceneplan))
        values = torch.as_tensor(prefix, dtype=torch.long).flatten().tolist()
        index = 0

        def choice(allowed: set[int]) -> int:
            nonlocal index
            if index >= len(values):
                raise _NeedPatchToken(set(allowed))
            value = int(values[index])
            if value not in allowed:
                raise ModelScenePlanCodecError(f"invalid patch prefix token {value} at {index}")
            index += 1
            return value

        def exact(value: int) -> None:
            choice({int(value)})

        def text_value() -> None:
            exact(self.plan_codec._tid("<text_begin>"))
            emitted = 0
            while True:
                allowed = set(self.nonempty_first_text_ids if emitted == 0 else self.plan_codec.text_ids)
                if emitted:
                    allowed.add(self.plan_codec._tid("<text_end>"))
                if emitted >= PATCH_TEXT_FIELD_MAX_TOKENS:
                    allowed = {self.plan_codec._tid("<text_end>")}
                token = choice(allowed)
                if token == self.plan_codec._tid("<text_end>"):
                    return
                emitted += 1

        def position(marker: str) -> None:
            exact(self.plan_codec._tid(marker))
            exact(self.plan_codec._tid("<azimuth_bin>"))
            choice(set(self.plan_codec.azimuth_ids))
            exact(self.plan_codec._tid("<elevation_bin>"))
            choice(set(self.plan_codec.elevation_ids))
            exact(self.plan_codec._tid("<distance_bin>"))
            choice(set(self.plan_codec.distance_ids))

        def trajectory() -> None:
            exact(self.plan_codec._tid("<trajectory_begin>"))
            motion = choice(
                {
                    self.plan_codec._tid(MOTION_TOKENS["static"]),
                    self.plan_codec._tid(MOTION_TOKENS["linear"]),
                }
            )
            if motion == self.plan_codec._tid(MOTION_TOKENS["static"]):
                position("<position>")
            else:
                position("<start>")
                position("<end>")
            exact(self.plan_codec._tid("<trajectory_end>"))

        existing_ids = {
            self._source_id(source["source_id"]) for source in plan["sources"]
        }
        speech_ids = {
            self._source_id(source["source_id"])
            for source in plan["sources"]
            if source["kind"] == "speech"
        }
        duration = self.plan_codec._duration_frame(plan["duration_sec"])

        def source_payload(*, add: bool) -> None:
            exact(self.plan_codec._tid("<source_begin>"))
            source_token = choice(
                {self._source_id(_first_free_source_id(plan))} if add else existing_ids
            )
            replaced_speech = source_token in speech_ids
            exact(self.plan_codec._tid("<kind>"))
            kinds = {self.plan_codec._tid(token) for token in KIND_TOKENS.values()}
            if speech_ids and not replaced_speech:
                kinds.remove(self.plan_codec._tid(KIND_TOKENS["speech"]))
            kind = choice(kinds)
            if kind == self.plan_codec._tid(KIND_TOKENS["speech"]):
                exact(self.plan_codec._tid("<speaker_description>"))
                text_value()
                exact(self.plan_codec._tid("<transcript>"))
                text_value()
            else:
                exact(self.plan_codec._tid("<description>"))
                text_value()
            exact(self.plan_codec._tid("<activity_begin>"))
            exact(self.plan_codec._tid("<onset_frame>"))
            onset_id = choice(set(self.plan_codec.frame_ids[:duration]))
            onset = self.plan_codec.frame_ids.index(onset_id)
            exact(self.plan_codec._tid("<offset_frame>"))
            choice(set(self.plan_codec.frame_ids[onset + 1 : duration + 1]))
            exact(self.plan_codec._tid("<activity_end>"))
            trajectory()
            exact(self.plan_codec._tid("<source_end>"))

        try:
            exact(self.bos_id)
            operation_ids = {
                self.token_to_id[ACTIVE_OPERATION_TOKENS[operation]]
                for operation in self._active_operations(plan)
            }
            operation_id = choice(operation_ids)
            operation = next(
                operation
                for operation, token in ACTIVE_OPERATION_TOKENS.items()
                if operation_id == self.token_to_id[token]
            )
            if operation == "room_change":
                rooms = {self.plan_codec._tid(token) for token in ROOM_TOKENS.values()}
                rooms.discard(self._room_id(plan["room"]["type"]))
                choice(rooms)
            elif operation == "remove_source":
                choice(existing_ids)
            elif operation == "add_source":
                source_payload(add=True)
            elif operation == "replace_source":
                source_payload(add=False)
            elif operation == "move_source":
                choice(existing_ids)
                trajectory()
            elif operation == "retime_source":
                choice(existing_ids)
                onset_id = choice(set(self.plan_codec.frame_ids[:duration]))
                onset = self.plan_codec.frame_ids.index(onset_id)
                choice(set(self.plan_codec.frame_ids[onset + 1 : duration + 1]))
            elif operation in {"change_speech_description", "change_transcript"}:
                choice(speech_ids)
                exact(
                    self.plan_codec._tid(
                        "<speaker_description>" if operation == "change_speech_description" else "<transcript>"
                    )
                )
                text_value()
            exact(self.eos_id)
            if index != len(values):
                raise ModelScenePlanCodecError("tokens remain after a complete patch")
            return set()
        except _NeedPatchToken as need:
            return need.allowed

    def apply(
        self,
        input_sceneplan: Mapping[str, Any],
        patch: Mapping[str, Any] | Sequence[int] | torch.Tensor,
    ) -> dict[str, Any]:
        """Apply one patch to the observed plan and preserve unspecified state."""

        current = self.plan_codec.project_plan(validate_model_sceneplan(input_sceneplan))
        program = self.decode(self.encode(patch)["input_ids"]) if isinstance(patch, Mapping) else self.decode(patch)
        operation = str(program["operation"])
        output = copy.deepcopy(current)
        by_id = {str(source["source_id"]): source for source in output["sources"]}

        if operation == "no_op":
            pass
        elif operation == "room_change":
            room = str(program["new_room"])
            if room == str(current["room"]["type"]):
                raise ModelScenePlanCodecError("SET_ROOM must change the room")
            output["room"] = {"type": room}
        elif operation == "add_source":
            if len(output["sources"]) >= MAX_SOURCES:
                raise ModelScenePlanCodecError("ADD_SOURCE exceeds P10 source capacity")
            source = copy.deepcopy(program["source"])
            if str(source["source_id"]) != _first_free_source_id(current):
                raise ModelScenePlanCodecError("ADD_SOURCE must use the first free source slot")
            output["sources"].append(source)
            output["sources"].sort(key=lambda item: _source_slot(item["source_id"]))
        elif operation == "replace_source":
            source_id = str(program["source_id"])
            if source_id not in by_id:
                raise ModelScenePlanCodecError(f"replacement names absent source {source_id!r}")
            replacement = copy.deepcopy(program["new_source"])
            if str(replacement["source_id"]) != source_id:
                raise ModelScenePlanCodecError("replacement changed persistent source id")
            output["sources"] = [
                replacement if str(source["source_id"]) == source_id else source
                for source in output["sources"]
            ]
        elif operation in {
            "remove_source",
            "move_source",
            "retime_source",
            "change_speech_description",
            "change_transcript",
            "rotate_source",
            "distance_source",
        }:
            source_id = str(program["source_id"])
            if source_id not in by_id:
                raise ModelScenePlanCodecError(f"patch names absent source {source_id!r}")
            source = by_id[source_id]
            if operation == "remove_source":
                if len(output["sources"]) <= 1:
                    raise ModelScenePlanCodecError("REMOVE_SOURCE cannot create an empty scene")
                output["sources"] = [
                    value for value in output["sources"] if value["source_id"] != source_id
                ]
            elif operation == "move_source":
                source["trajectory"] = copy.deepcopy(program["trajectory"])
            elif operation == "retime_source":
                onset, offset = map(int, program["new_interval_frames"])
                duration = self.plan_codec._duration_frame(output["duration_sec"])
                if not 0 <= onset < offset <= duration:
                    raise ModelScenePlanCodecError("SET_ACTIVITY exceeds scene duration")
                source["activity"] = {"onset_sec": _seconds(onset), "offset_sec": _seconds(offset)}
            elif operation in {"change_speech_description", "change_transcript"}:
                if str(source["kind"]) != "speech":
                    raise ModelScenePlanCodecError("speech text patch names a non-speech source")
                key = "speaker_description" if operation == "change_speech_description" else "transcript"
                value_key = "new_speaker_description" if operation == "change_speech_description" else "new_transcript"
                source[key] = str(program[value_key])
            elif operation == "rotate_source":
                delta = int(program["delta_azimuth_deg"])
                for position in _positions(source):
                    position["azimuth_deg"] = float(
                        ((int(round(float(position["azimuth_deg"]))) + delta + 180) % 360) - 180
                    )
            else:
                factor = float(program["distance_factor"])
                for position in _positions(source):
                    position["distance_m"] = float(position["distance_m"]) * factor
        return self.plan_codec.project_plan(validate_model_sceneplan(output))

    def assert_target(
        self,
        input_sceneplan: Mapping[str, Any],
        spec: Mapping[str, Any],
        target_sceneplan: Mapping[str, Any],
    ) -> None:
        encoded = self.encode(spec)["input_ids"]
        applied = self.apply(input_sceneplan, encoded)
        target = self.plan_codec.project_plan(target_sceneplan)
        if not torch.equal(
            self.plan_codec.encode(applied)["input_ids"],
            self.plan_codec.encode(target)["input_ids"],
        ):
            raise ModelScenePlanCodecError(
                "edit spec does not deterministically reproduce its revised ScenePlan"
            )


__all__ = [
    "ACTIVE_OPERATION_TOKENS",
    "LEGACY_OPERATION_TOKENS",
    "OPERATION_TOKENS",
    "PATCH_CODEC_NAME",
    "PATCH_MAX_TOKENS",
    "PATCH_OUTPUT_CONTRACT",
    "PATCH_TEXT_FIELD_MAX_TOKENS",
    "PATCH_TOKENS",
    "RETIME_FRAME_LEVELS",
    "RETIME_MODES",
    "RETIME_POLICY",
    "ScenePlanEditPatchCodec",
    "make_deterministic_edit",
]
