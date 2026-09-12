"""Sketch-first P11-v4 contract aligned to the frozen P10 executor.

The old P11 route autoregressively interleaved semantic text and executable
numeric controls in one ScenePlan token stream.  A control intervention could
therefore change the causal cache used to decode later semantic text.  This
module makes that failure impossible by construction:

* :class:`SceneSketchCodec` owns only discrete room/source identity and text;
* ``P10ExecutionState`` owns only duration, activity, and trajectory numbers;
* :func:`assemble_sceneplan` is the sole deterministic join point;
* Editing is reduced to the existing short atomic patch grammar.

The continuous state deliberately omits room, source kind, source count, and
semantic-presence features.  Those fields are authoritative in SceneSketch
and cannot be changed by a flow/diffusion intervention.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .model_sceneplan import MODEL_SAMPLE_RATE, VAE_HOP_SAMPLES, validate_model_sceneplan
from .model_sceneplan_codec import ModelScenePlanCodecError
from .model_sceneplan_codec_v3 import (
    DISTANCE_MAX_M,
    DISTANCE_MIN_M,
    KIND_TOKENS,
    ROOM_TOKENS,
    SOURCE_COUNT_TOKENS,
    SOURCE_SLOT_TOKENS,
    ModelScenePlanCodecV3,
)
from .scene_plan import (
    LOSS_GRAMMAR,
    LOSS_ROOM,
    LOSS_SEMANTIC,
    LOSS_SPATIAL_METRIC,
    LOSS_SPEECH_CONTENT,
)
from .sceneplan_edit_patch import (
    ACTIVE_OPERATION_TOKENS,
    OPERATION_TOKENS,
    ScenePlanEditPatchCodec,
)
from .sceneplan_p11_single_turn import (
    compile_p10_aligned_target_conditions,
    validate_p11_executor_profile,
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value using the canonical P11 hashing convention."""

    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


SCENE_SKETCH_SCHEMA = "stable_audio_tools.scene_sketch"
SCENE_SKETCH_VERSION = 1
SCENE_SKETCH_CONTRACT = "p10_bound_discrete_cot_v1"
SCENE_SKETCH_CODEC_NAME = "scene_sketch_codec_v1"

DELTA_SCENE_SKETCH_SCHEMA = "stable_audio_tools.delta_scene_sketch"
DELTA_SCENE_SKETCH_VERSION = 1
DELTA_SCENE_SKETCH_CONTRACT = "p10_bound_atomic_delta_cot_v1"

EXECUTION_STATE_SCHEMA = "stable_audio_tools.p10_execution_state"
EXECUTION_STATE_VERSION = 1
EXECUTION_STATE_CONTRACT = "p10_numeric_execution_state_v1"
EXECUTION_FEATURE_SCHEMA = "p10_numeric_execution_core15_v1"

EXECUTION_SLOT_ROLES = (
    "scene_global",
    "source_0",
    "source_1",
    "source_2",
    "source_3",
)
EXECUTION_SLOT_COUNT = len(EXECUTION_SLOT_ROLES)
EXECUTION_FEATURE_NAMES = (
    "duration_frames_norm",
    "onset_frame_norm",
    "offset_frame_norm",
    "motion_static",
    "motion_linear",
    "start_sin_azimuth",
    "start_cos_azimuth",
    "start_sin_elevation",
    "start_cos_elevation",
    "start_log_distance_norm",
    "end_sin_azimuth",
    "end_cos_azimuth",
    "end_sin_elevation",
    "end_cos_elevation",
    "end_log_distance_norm",
)
EXECUTION_FEATURE_DIM = len(EXECUTION_FEATURE_NAMES)
_EXECUTION_INDEX = {
    name: index for index, name in enumerate(EXECUTION_FEATURE_NAMES)
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ROOM_ORDER = tuple(ROOM_TOKENS)
_KIND_ORDER = tuple(KIND_TOKENS)

DELTA_SKETCH_TOKEN_CONTRACT = "atomic_delta_owner_direction_program_v2"
AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT = "audio_aware_atomic_delta_v1"
AUDIO_AWARE_DELTA_OPERATIONS = tuple(ACTIVE_OPERATION_TOKENS)
CONTROL_DIRECTION_CONTRACT = "p10_atomic_edit_control_direction_v1"

# P10 exposes exactly two legal endpoints for each of these atomic controls.
# DeltaSketch owns the categorical branch; DeltaThought remains responsible
# for continuous state reasoning (and for the full retime delta).  This avoids
# asking a second prompt-only regression head to rediscover a decision that is
# already explicit in the edit instruction and in P10's patch grammar.
CONTROL_DIRECTION_OPERATIONS = (
    "rotate_source",
    "distance_source",
)
CONTROL_DIRECTION_TOKENS = {
    "rotate_source": {
        -1: "<delta_azimuth_minus_045>",
        1: "<delta_azimuth_plus_045>",
    },
    "distance_source": {
        -1: "<distance_scale_075>",
        1: "<distance_scale_125>",
    },
}


def control_direction_from_patch(patch_spec: Mapping[str, Any] | None) -> int:
    """Return the P10 endpoint branch encoded by an atomic edit.

    ``-1`` and ``+1`` are categorical directions, not a shared physical unit.
    Non-binary operations return zero because their value is supplied by the
    room/owner token or the continuous DeltaThought.
    """

    if patch_spec is None:
        return 0
    operation = str(patch_spec.get("operation") or "")
    explicit = patch_spec.get("control_direction")
    if explicit is not None:
        direction = int(explicit)
        if operation not in CONTROL_DIRECTION_OPERATIONS or direction not in {-1, 1}:
            raise ModelScenePlanCodecError(
                "control_direction is legal only as +/-1 for rotate/distance"
            )
        return direction
    if operation == "rotate_source":
        value = int(patch_spec["delta_azimuth_deg"])
        if value not in {-45, 45}:
            raise ModelScenePlanCodecError("rotation direction requires exactly +/-45")
        return -1 if value < 0 else 1
    if operation == "distance_source":
        value = round(float(patch_spec["distance_factor"]), 2)
        if value not in {0.75, 1.25}:
            raise ModelScenePlanCodecError(
                "distance direction requires factor 0.75 or 1.25"
            )
        return -1 if value < 1.0 else 1
    return 0


class _NeedSketchToken(Exception):
    def __init__(self, allowed: set[int]):
        super().__init__()
        self.allowed = allowed


def _normalized_text(value: Any, *, label: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        raise ModelScenePlanCodecError(f"{label} must be non-empty")
    return text


def _source_slot(source_id: Any) -> int:
    text = str(source_id)
    if text not in {f"source_{index}" for index in range(4)}:
        raise ModelScenePlanCodecError(f"invalid source id {text!r}")
    return int(text[7:])


def _sketch_source(source: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(source["kind"])
    result: dict[str, Any] = {
        "source_id": str(source["source_id"]),
        "kind": kind,
    }
    if kind == "speech":
        result["speaker_description"] = _normalized_text(
            source["speaker_description"], label="speaker_description"
        )
        result["transcript"] = _normalized_text(
            source["transcript"], label="transcript"
        )
    else:
        result["description"] = _normalized_text(
            source["description"], label="description"
        )
    return result


def validate_scene_sketch(
    sketch: Mapping[str, Any],
    *,
    codec: ModelScenePlanCodecV3 | None = None,
) -> dict[str, Any]:
    """Validate the discrete semantic authority used by P11-v4."""

    required = {
        "schema",
        "schema_version",
        "contract",
        "codec_fingerprint",
        "sceneplan_sha256",
        "room_intent",
        "sources",
    }
    if set(sketch) != required:
        raise ModelScenePlanCodecError("SceneSketch top-level fields changed")
    if (
        sketch["schema"] != SCENE_SKETCH_SCHEMA
        or int(sketch["schema_version"]) != SCENE_SKETCH_VERSION
        or sketch["contract"] != SCENE_SKETCH_CONTRACT
    ):
        raise ModelScenePlanCodecError("SceneSketch contract/version mismatch")
    if codec is not None and sketch["codec_fingerprint"] != codec.fingerprint:
        raise ModelScenePlanCodecError("SceneSketch codec fingerprint mismatch")
    digest = sketch["sceneplan_sha256"]
    if digest is not None and (
        not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
    ):
        raise ModelScenePlanCodecError("SceneSketch ScenePlan hash is invalid")
    room = str(sketch["room_intent"])
    if room not in ROOM_TOKENS:
        raise ModelScenePlanCodecError(f"unsupported SceneSketch room {room!r}")
    sources = sketch["sources"]
    if not isinstance(sources, list) or not 1 <= len(sources) <= 4:
        raise ModelScenePlanCodecError("SceneSketch requires one to four sources")
    previous_slot = -1
    speech_count = 0
    for source in sources:
        if not isinstance(source, Mapping):
            raise ModelScenePlanCodecError("SceneSketch source must be an object")
        kind = str(source.get("kind") or "")
        expected = {"source_id", "kind"}
        if kind == "speech":
            expected |= {"speaker_description", "transcript"}
            speech_count += 1
        elif kind in {"music", "sound"}:
            expected.add("description")
        else:
            raise ModelScenePlanCodecError(f"unsupported SceneSketch kind {kind!r}")
        if set(source) != expected:
            raise ModelScenePlanCodecError(
                f"SceneSketch {kind or 'unknown'} source fields changed"
            )
        slot = _source_slot(source["source_id"])
        if slot <= previous_slot:
            raise ModelScenePlanCodecError(
                "SceneSketch source slots must be strictly increasing"
            )
        previous_slot = slot
        if kind == "speech":
            _normalized_text(
                source["speaker_description"], label="speaker_description"
            )
            _normalized_text(source["transcript"], label="transcript")
        else:
            _normalized_text(source["description"], label="description")
    if speech_count > 1:
        raise ModelScenePlanCodecError(
            "SceneSketch exceeds P10's one-formal-speech-source profile"
        )
    return copy.deepcopy(dict(sketch))


def compile_scene_sketch(
    sceneplan: Mapping[str, Any], codec: ModelScenePlanCodecV3
) -> dict[str, Any]:
    """Project one P10-executable ScenePlan into immutable discrete semantics."""

    # Validate before projection as well as after it.  ``project_plan``
    # intentionally canonicalizes legacy gain to zero; the v4 boundary must
    # fail closed instead of silently accepting a field P10 cannot execute.
    validate_p11_executor_profile(validate_model_sceneplan(sceneplan))
    projected = validate_p11_executor_profile(codec.project_plan(sceneplan))
    result = {
        "schema": SCENE_SKETCH_SCHEMA,
        "schema_version": SCENE_SKETCH_VERSION,
        "contract": SCENE_SKETCH_CONTRACT,
        "codec_fingerprint": codec.fingerprint,
        "sceneplan_sha256": sha256_json(projected),
        "room_intent": str(projected["room"]["type"]),
        "sources": [_sketch_source(source) for source in projected["sources"]],
    }
    return validate_scene_sketch(result, codec=codec)


def scene_sketch_semantic_state(sketch: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact fields that may influence P10 semantic conditioning."""

    value = validate_scene_sketch(sketch)
    return {
        "room": value["room_intent"],
        "sources": copy.deepcopy(value["sources"]),
    }


def apply_reliable_lexical_authority_to_sketch(
    sketch: Mapping[str, Any],
    *,
    transcript: str,
    source_id: str,
    codec: ModelScenePlanCodecV3,
) -> dict[str, Any]:
    """Apply input-only ASR evidence inside the discrete semantic boundary.

    P10 supports at most one formal speech source, so one reliable mixed-FOA
    transcript has an unambiguous lexical owner after a source slot is chosen.
    The function never touches duration, activity, geometry, or any continuous
    ExecutionState value.  If the acoustic decoder missed speech, its existing
    description is retained as the speaker description instead of inventing
    target-only semantics.
    """

    value = validate_scene_sketch(sketch, codec=codec)
    output = copy.deepcopy(value)
    normalized_transcript = _normalized_text(transcript, label="ASR transcript")
    selected = next(
        (
            source
            for source in output["sources"]
            if str(source["source_id"]) == str(source_id)
        ),
        None,
    )
    if selected is None:
        raise ModelScenePlanCodecError(
            f"lexical authority selected absent source {source_id!r}"
        )
    existing_speech = [
        source for source in output["sources"] if source["kind"] == "speech"
    ]
    if existing_speech and existing_speech[0] is not selected:
        raise ModelScenePlanCodecError(
            "lexical authority cannot create a second formal speech source"
        )
    if selected["kind"] == "speech":
        selected["transcript"] = normalized_transcript
    else:
        description = _normalized_text(
            selected.pop("description"), label="speech owner description"
        )
        selected["kind"] = "speech"
        selected["speaker_description"] = description
        selected["transcript"] = normalized_transcript
    output["sceneplan_sha256"] = None
    return validate_scene_sketch(output, codec=codec)


def select_reliable_lexical_source_owner(
    sketch: Mapping[str, Any],
    *,
    source_kind_scores: Sequence[Mapping[str, Any]],
    codec: ModelScenePlanCodecV3,
) -> tuple[str, str]:
    """Select the one P10 speech owner without target-side information.

    Existing decoded speech has semantic priority.  If the discrete decoder
    missed speech, choose the present source with the largest *pre-constraint*
    speech margin, then probability, then lowest stable source slot.  This is
    shared by sketch-first Direct/Flow and the full-ScenePlan D0 evaluator so
    reliable ASR cannot receive a different or target-informed owner rule.
    """

    value = validate_scene_sketch(sketch, codec=codec)
    speech_sources = [
        source for source in value["sources"] if source["kind"] == "speech"
    ]
    if speech_sources:
        return (
            str(speech_sources[0]["source_id"]),
            "copy_transcript_to_detected_speech",
        )

    available = {str(source["source_id"]) for source in value["sources"]}
    score_rows = [
        dict(row)
        for row in source_kind_scores
        if str(row.get("source_id")) in available
    ]
    if len(score_rows) != len(available) or {
        str(row.get("source_id")) for row in score_rows
    } != available:
        raise RuntimeError(
            "lexical assembler lacks one unique speech likelihood per source"
        )

    def owner_key(row: Mapping[str, Any]) -> tuple[float, float, int]:
        source_id = str(row["source_id"])
        margin = float(row["speech_margin"])
        probability = float(row["speech_probability"])
        if not math.isfinite(margin) or not math.isfinite(probability):
            raise RuntimeError("lexical assembler source likelihood is non-finite")
        return margin, probability, -int(source_id.rsplit("_", 1)[1])

    selected = str(max(score_rows, key=owner_key)["source_id"])
    return selected, "promote_argmax_speech_likelihood_source"


class SceneSketchCodec:
    """Encode SceneSketch with a strict subset of the frozen ScenePlan vocab.

    No embedding resize is needed: the grammar reuses the room, source, kind,
    text, BOS, and EOS ids already present in codec-v4.  Numeric frame and
    geometry ids are never legal in this output stream.
    """

    def __init__(self, plan_codec: ModelScenePlanCodecV3) -> None:
        if not isinstance(plan_codec, ModelScenePlanCodecV3):
            raise TypeError("SceneSketchCodec requires ModelScenePlanCodecV3/v4")
        self.plan_codec = plan_codec
        self.vocab_size = int(plan_codec.vocab_size)
        # A syntactically present SentencePiece token can still decode to only
        # whitespace (notably the standalone word-boundary piece). P10 rejects
        # such descriptions, so the constrained grammar must reject them in
        # the first text position too. Later pieces may contain whitespace
        # because a non-empty prefix is already guaranteed.
        self.nonempty_first_text_ids = {
            plan_codec.text_offset + piece_id
            for piece_id in range(plan_codec.text_vocab_size)
            if plan_codec.text_processor.decode([piece_id]).strip()
        }
        if not self.nonempty_first_text_ids:
            raise ModelScenePlanCodecError(
                "SceneSketch codec has no non-empty first text pieces"
            )

    @property
    def bos_id(self) -> int:
        return self.plan_codec.bos_id

    @property
    def eos_id(self) -> int:
        return self.plan_codec.eos_id

    @property
    def pad_id(self) -> int:
        return self.plan_codec.pad_id

    @property
    def fingerprint(self) -> str:
        payload = canonical_json_bytes(
            {
                "codec": SCENE_SKETCH_CODEC_NAME,
                "version": SCENE_SKETCH_VERSION,
                "plan_codec_fingerprint": self.plan_codec.fingerprint,
                "grammar": "room_count_ordered_source_kind_text_only",
            }
        )
        return hashlib.sha256(payload).hexdigest()

    def encode(
        self, sketch: Mapping[str, Any], *, max_tokens: int | None = None
    ) -> dict[str, torch.Tensor]:
        value = validate_scene_sketch(sketch, codec=self.plan_codec)
        ids: list[int] = []
        groups: list[int] = []

        def emit(token: str, group: int = LOSS_GRAMMAR) -> None:
            ids.append(self.plan_codec._tid(token))
            groups.append(int(group))

        def text_field(token: str, text: Any, group: int) -> None:
            emit(token, group)
            encoded = self.plan_codec._text(text)
            ids.extend(encoded)
            groups.extend([int(group)] * len(encoded))

        emit("<plan_bos>")
        emit("<room>", LOSS_ROOM)
        emit(ROOM_TOKENS[str(value["room_intent"])], LOSS_ROOM)
        emit("<num_sources>")
        emit(SOURCE_COUNT_TOKENS[len(value["sources"]) - 1])
        for source in value["sources"]:
            slot = _source_slot(source["source_id"])
            kind = str(source["kind"])
            emit("<source_begin>")
            emit(SOURCE_SLOT_TOKENS[slot], LOSS_SEMANTIC)
            emit("<kind>", LOSS_SEMANTIC)
            emit(KIND_TOKENS[kind], LOSS_SEMANTIC)
            if kind == "speech":
                text_field(
                    "<speaker_description>",
                    source["speaker_description"],
                    LOSS_SEMANTIC,
                )
                text_field("<transcript>", source["transcript"], LOSS_SPEECH_CONTENT)
            else:
                text_field("<description>", source["description"], LOSS_SEMANTIC)
            emit("<source_end>")
        emit("<plan_eos>")
        if max_tokens is not None and len(ids) > int(max_tokens):
            raise ModelScenePlanCodecError(
                f"SceneSketch requires {len(ids)} tokens > max_tokens={max_tokens}"
            )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool),
            "loss_group_ids": torch.tensor(groups, dtype=torch.long),
        }

    def decode(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        sceneplan_sha256: str | None = None,
    ) -> dict[str, Any]:
        values = torch.as_tensor(token_ids, dtype=torch.long).flatten().tolist()
        cur = self.plan_codec._Decoder(self.plan_codec, values)
        cur.expect("<plan_bos>")
        cur.expect("<room>")
        room_id = cur.take()
        room_lookup = {
            self.plan_codec._tid(token): room for room, token in ROOM_TOKENS.items()
        }
        if room_id not in room_lookup:
            raise ModelScenePlanCodecError("SceneSketch has an invalid room token")
        cur.expect("<num_sources>")
        count_id = cur.take()
        count_lookup = {
            self.plan_codec._tid(token): count
            for count, token in enumerate(SOURCE_COUNT_TOKENS, 1)
        }
        if count_id not in count_lookup:
            raise ModelScenePlanCodecError("SceneSketch has an invalid source count")
        slot_lookup = {
            self.plan_codec._tid(token): index
            for index, token in enumerate(SOURCE_SLOT_TOKENS)
        }
        kind_lookup = {
            self.plan_codec._tid(token): kind for kind, token in KIND_TOKENS.items()
        }
        sources: list[dict[str, Any]] = []
        speech_seen = False
        for _ in range(count_lookup[count_id]):
            cur.expect("<source_begin>")
            slot_id = cur.take()
            if slot_id not in slot_lookup:
                raise ModelScenePlanCodecError("SceneSketch has an invalid source slot")
            slot = slot_lookup[slot_id]
            if sources and slot <= _source_slot(sources[-1]["source_id"]):
                raise ModelScenePlanCodecError(
                    "SceneSketch source slots are not strictly ordered"
                )
            cur.expect("<kind>")
            kind_id = cur.take()
            if kind_id not in kind_lookup:
                raise ModelScenePlanCodecError("SceneSketch has an invalid source kind")
            kind = kind_lookup[kind_id]
            source: dict[str, Any] = {
                "source_id": f"source_{slot}",
                "kind": kind,
            }
            if kind == "speech":
                if speech_seen:
                    raise ModelScenePlanCodecError(
                        "SceneSketch contains more than one speech source"
                    )
                speech_seen = True
                cur.expect("<speaker_description>")
                source["speaker_description"] = cur.text()
                cur.expect("<transcript>")
                source["transcript"] = cur.text()
            else:
                cur.expect("<description>")
                source["description"] = cur.text()
            cur.expect("<source_end>")
            sources.append(source)
        cur.expect("<plan_eos>")
        if cur.index != len(cur.values):
            raise ModelScenePlanCodecError("tokens remain after SceneSketch EOS")
        result = {
            "schema": SCENE_SKETCH_SCHEMA,
            "schema_version": SCENE_SKETCH_VERSION,
            "contract": SCENE_SKETCH_CONTRACT,
            "codec_fingerprint": self.plan_codec.fingerprint,
            "sceneplan_sha256": sceneplan_sha256,
            "room_intent": room_lookup[room_id],
            "sources": sources,
        }
        return validate_scene_sketch(result, codec=self.plan_codec)

    def canonicalize(
        self, token_ids: Sequence[int] | torch.Tensor, *, max_tokens: int | None = None
    ) -> dict[str, torch.Tensor]:
        return self.encode(self.decode(token_ids), max_tokens=max_tokens)

    def allowed_next_ids(
        self,
        prefix: Sequence[int] | torch.Tensor,
        *,
        min_sources: int = 1,
        max_sources: int = 4,
        max_text_tokens: int | None = None,
    ) -> set[int]:
        """Return exact grammar choices for constrained SceneSketch decoding."""

        if not 1 <= int(min_sources) <= int(max_sources) <= 4:
            raise ValueError("source limits must satisfy 1 <= min <= max <= 4")
        if max_text_tokens is not None and int(max_text_tokens) < 1:
            raise ValueError("SceneSketch max_text_tokens must be positive")
        text_ceiling = (
            None if max_text_tokens is None else int(max_text_tokens)
        )
        values = torch.as_tensor(prefix, dtype=torch.long).flatten().tolist()
        index = 0

        def choice(allowed: set[int], label: str) -> int:
            nonlocal index
            if index >= len(values):
                raise _NeedSketchToken(set(allowed))
            value = int(values[index])
            if value not in allowed:
                raise ModelScenePlanCodecError(
                    f"invalid SceneSketch prefix at {index}: expected {label}, found {value}"
                )
            index += 1
            return value

        def exact(token: str) -> None:
            choice({self.plan_codec._tid(token)}, token)

        def text_value() -> None:
            exact("<text_begin>")
            emitted = 0
            while True:
                allowed = (
                    set(self.plan_codec.text_ids)
                    if emitted
                    else set(self.nonempty_first_text_ids)
                )
                if emitted:
                    allowed.add(self.plan_codec._tid("<text_end>"))
                if text_ceiling is not None and emitted >= text_ceiling:
                    allowed = {self.plan_codec._tid("<text_end>")}
                value = choice(allowed, "text piece or <text_end>")
                if value == self.plan_codec._tid("<text_end>"):
                    return
                emitted += 1

        def text_field(token: str) -> None:
            exact(token)
            text_value()

        try:
            exact("<plan_bos>")
            exact("<room>")
            choice(
                {self.plan_codec._tid(token) for token in ROOM_TOKENS.values()},
                "room",
            )
            exact("<num_sources>")
            count_id = choice(
                {
                    self.plan_codec._tid(SOURCE_COUNT_TOKENS[count - 1])
                    for count in range(int(min_sources), int(max_sources) + 1)
                },
                "source count",
            )
            source_count = next(
                count
                for count, token in enumerate(SOURCE_COUNT_TOKENS, 1)
                if count_id == self.plan_codec._tid(token)
            )
            previous_slot = -1
            speech_seen = False
            for source_index in range(source_count):
                exact("<source_begin>")
                remaining = source_count - source_index - 1
                maximum_slot = 3 - remaining
                slot_id = choice(
                    {
                        self.plan_codec._tid(SOURCE_SLOT_TOKENS[slot])
                        for slot in range(previous_slot + 1, maximum_slot + 1)
                    },
                    "increasing source slot",
                )
                previous_slot = next(
                    slot
                    for slot, token in enumerate(SOURCE_SLOT_TOKENS)
                    if slot_id == self.plan_codec._tid(token)
                )
                exact("<kind>")
                kinds = {
                    self.plan_codec._tid(KIND_TOKENS["music"]),
                    self.plan_codec._tid(KIND_TOKENS["sound"]),
                }
                if not speech_seen:
                    kinds.add(self.plan_codec._tid(KIND_TOKENS["speech"]))
                kind_id = choice(kinds, "source kind")
                if kind_id == self.plan_codec._tid(KIND_TOKENS["speech"]):
                    speech_seen = True
                    text_field("<speaker_description>")
                    text_field("<transcript>")
                else:
                    text_field("<description>")
                exact("<source_end>")
            exact("<plan_eos>")
            if index != len(values):
                raise ModelScenePlanCodecError(
                    "tokens remain after a complete SceneSketch"
                )
            return set()
        except _NeedSketchToken as need:
            return need.allowed


class DeltaSceneSketchCodec:
    """Discrete semantic/ownership program that precedes ``<DELTA_THOUGHT>``.

    Rotate/distance carry one operation-specific negative/positive control
    token because those are categorical branches in the frozen P10 executor.
    Arbitrary numeric values are still forbidden.  Retime remains continuous;
    room identity and removal ownership remain discrete because they change
    SceneSketch rather than P10's numeric execution tensor.
    """

    def __init__(
        self,
        plan_codec: ModelScenePlanCodecV3,
        patch_codec: ScenePlanEditPatchCodec | None = None,
    ) -> None:
        self.plan_codec = plan_codec
        self.patch_codec = patch_codec or ScenePlanEditPatchCodec(plan_codec)
        self.vocab_size = int(plan_codec.vocab_size)

    @property
    def bos_id(self) -> int:
        return self.patch_codec.bos_id

    @property
    def eos_id(self) -> int:
        return self.patch_codec.eos_id

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "contract": DELTA_SKETCH_TOKEN_CONTRACT,
                    "patch_codec": self.patch_codec.fingerprint,
                }
            )
        ).hexdigest()

    def encode(
        self,
        delta_sketch: Mapping[str, Any],
        patch_spec: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        delta = validate_delta_scene_sketch(delta_sketch)
        operation = str(delta["operation"])
        if operation != str(patch_spec.get("operation") or ""):
            raise ModelScenePlanCodecError(
                "DeltaSceneSketch and atomic patch operation disagree"
            )
        ids = [
            self.bos_id,
            self.patch_codec.token_to_id[OPERATION_TOKENS[operation]],
        ]
        groups = [LOSS_GRAMMAR, LOSS_GRAMMAR]
        if operation == "room_change":
            room = str(patch_spec["new_room"])
            if room not in ROOM_TOKENS:
                raise ModelScenePlanCodecError(f"invalid delta room {room!r}")
            ids.append(self.plan_codec._tid(ROOM_TOKENS[room]))
            groups.append(LOSS_ROOM)
        elif operation in {
            "rotate_source",
            "distance_source",
            "retime_source",
            "remove_source",
        }:
            source_id = str(patch_spec["source_id"])
            slot = _source_slot(source_id)
            expected_owner = (
                delta["removed_source_ids"][0]
                if operation == "remove_source"
                else delta["changed_source_ids"][0]
            )
            if source_id != expected_owner:
                raise ModelScenePlanCodecError(
                    "DeltaSceneSketch token owner does not match its audit object"
                )
            ids.append(self.plan_codec._tid(SOURCE_SLOT_TOKENS[slot]))
            groups.append(LOSS_SEMANTIC)
            if operation in CONTROL_DIRECTION_OPERATIONS:
                direction = control_direction_from_patch(patch_spec)
                ids.append(
                    self.patch_codec.token_to_id[
                        CONTROL_DIRECTION_TOKENS[operation][direction]
                    ]
                )
                groups.append(LOSS_SPATIAL_METRIC)
        ids.append(self.eos_id)
        groups.append(LOSS_GRAMMAR)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool),
            "loss_group_ids": torch.tensor(groups, dtype=torch.long),
        }

    def decode(
        self, token_ids: Sequence[int] | torch.Tensor
    ) -> dict[str, Any]:
        values = torch.as_tensor(token_ids, dtype=torch.long).flatten().tolist()
        if len(values) not in {3, 4, 5} or values[0] != self.bos_id or values[-1] != self.eos_id:
            raise ModelScenePlanCodecError(
                "DeltaSketch program must contain BOS, operation, optional owner, EOS"
            )
        operations = {
            self.patch_codec.token_to_id[token]: operation
            for operation, token in OPERATION_TOKENS.items()
        }
        operation = operations.get(int(values[1]))
        if operation is None:
            raise ModelScenePlanCodecError("DeltaSketch has an unknown operation")
        payload = values[2:-1]
        result: dict[str, Any] = {
            "contract": DELTA_SKETCH_TOKEN_CONTRACT,
            "operation": operation,
        }
        if operation == "no_op":
            if payload:
                raise ModelScenePlanCodecError("no-op DeltaSketch has arguments")
        elif operation == "room_change":
            room_lookup = {
                self.plan_codec._tid(token): room
                for room, token in ROOM_TOKENS.items()
            }
            if len(payload) != 1 or payload[0] not in room_lookup:
                raise ModelScenePlanCodecError(
                    "room DeltaSketch requires one room argument"
                )
            result["new_room"] = room_lookup[payload[0]]
        elif operation in CONTROL_DIRECTION_OPERATIONS:
            source_lookup = {
                self.plan_codec._tid(token): f"source_{index}"
                for index, token in enumerate(SOURCE_SLOT_TOKENS)
            }
            direction_lookup = {
                self.patch_codec.token_to_id[token]: direction
                for direction, token in CONTROL_DIRECTION_TOKENS[operation].items()
            }
            if (
                len(payload) != 2
                or payload[0] not in source_lookup
                or payload[1] not in direction_lookup
            ):
                raise ModelScenePlanCodecError(
                    f"{operation} DeltaSketch requires source owner and direction"
                )
            result["source_id"] = source_lookup[payload[0]]
            result["control_direction"] = direction_lookup[payload[1]]
        else:
            source_lookup = {
                self.plan_codec._tid(token): f"source_{index}"
                for index, token in enumerate(SOURCE_SLOT_TOKENS)
            }
            if len(payload) != 1 or payload[0] not in source_lookup:
                raise ModelScenePlanCodecError(
                    f"{operation} DeltaSketch requires one source owner"
                )
            result["source_id"] = source_lookup[payload[0]]
        return result

    def canonicalize(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        delta_sketch: Mapping[str, Any],
    ) -> dict[str, torch.Tensor]:
        program = self.decode(token_ids)
        return self.encode(delta_sketch, program)

    def allowed_next_ids(
        self,
        prefix: Sequence[int] | torch.Tensor,
        *,
        input_sceneplan: Mapping[str, Any],
    ) -> set[int]:
        current = validate_p11_executor_profile(
            self.plan_codec.project_plan(input_sceneplan)
        )
        values = torch.as_tensor(prefix, dtype=torch.long).flatten().tolist()
        index = 0

        def choice(allowed: set[int]) -> int:
            nonlocal index
            if index >= len(values):
                raise _NeedSketchToken(set(allowed))
            value = int(values[index])
            if value not in allowed:
                raise ModelScenePlanCodecError(
                    f"invalid DeltaSketch prefix token {value} at {index}"
                )
            index += 1
            return value

        def exact(value: int) -> None:
            choice({int(value)})

        operation_ids = {
            self.patch_codec.token_to_id[token]
            for token in OPERATION_TOKENS.values()
        }
        if len(current["sources"]) == 1:
            operation_ids.remove(
                self.patch_codec.token_to_id[OPERATION_TOKENS["remove_source"]]
            )
        try:
            exact(self.bos_id)
            operation_id = choice(operation_ids)
            operation = next(
                operation
                for operation, token in OPERATION_TOKENS.items()
                if operation_id == self.patch_codec.token_to_id[token]
            )
            if operation == "room_change":
                choices = {
                    self.plan_codec._tid(token) for token in ROOM_TOKENS.values()
                }
                choices.discard(
                    self.plan_codec._tid(
                        ROOM_TOKENS[str(current["room"]["type"])]
                    )
                )
                choice(choices)
            elif operation != "no_op":
                choice(
                    {
                        self.plan_codec._tid(
                            SOURCE_SLOT_TOKENS[_source_slot(source["source_id"])]
                        )
                        for source in current["sources"]
                    }
                )
                if operation in CONTROL_DIRECTION_OPERATIONS:
                    choice(
                        {
                            self.patch_codec.token_to_id[token]
                            for token in CONTROL_DIRECTION_TOKENS[operation].values()
                        }
                    )
            exact(self.eos_id)
            if index != len(values):
                raise ModelScenePlanCodecError(
                    "tokens remain after complete DeltaSketch program"
                )
            return set()
        except _NeedSketchToken as need:
            return need.allowed


class AudioAwareDeltaSceneSketchCodec:
    """Semantics-only edit program for the active audio-aware P11 route.

    Operation, persistent source owner, room, kind, and text are discrete.
    Activity and trajectory values are intentionally absent: they are owned
    exclusively by ``DeltaExecutionState`` and materialized into the final
    atomic patch by :func:`project_audio_aware_delta_to_atomic_patch`.
    """

    def __init__(
        self,
        plan_codec: ModelScenePlanCodecV3,
        patch_codec: ScenePlanEditPatchCodec | None = None,
    ) -> None:
        self.plan_codec = plan_codec
        self.patch_codec = patch_codec or ScenePlanEditPatchCodec(plan_codec)
        self.vocab_size = int(plan_codec.vocab_size)
        self.nonempty_first_text_ids = {
            plan_codec.text_offset + piece_id
            for piece_id in range(plan_codec.text_vocab_size)
            if plan_codec.text_processor.decode([piece_id]).strip()
        }

    @property
    def bos_id(self) -> int:
        return self.patch_codec.bos_id

    @property
    def eos_id(self) -> int:
        return self.patch_codec.eos_id

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "contract": AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT,
                    "patch_codec": self.patch_codec.fingerprint,
                    "numeric_payload": "forbidden",
                }
            )
        ).hexdigest()

    def _source_id(self, value: Any) -> int:
        return self.plan_codec._tid(SOURCE_SLOT_TOKENS[_source_slot(value)])

    def _semantic_source(self, value: Mapping[str, Any]) -> dict[str, Any]:
        source = _sketch_source(value) if "activity" in value else copy.deepcopy(dict(value))
        probe = {
            "schema": SCENE_SKETCH_SCHEMA,
            "schema_version": SCENE_SKETCH_VERSION,
            "contract": SCENE_SKETCH_CONTRACT,
            "codec_fingerprint": self.plan_codec.fingerprint,
            "sceneplan_sha256": None,
            "room_intent": "dry",
            "sources": [source],
        }
        return validate_scene_sketch(probe, codec=self.plan_codec)["sources"][0]

    def _normalize_program(self, value: Mapping[str, Any]) -> dict[str, Any]:
        operation = str(value.get("operation") or "")
        if operation not in ACTIVE_OPERATION_TOKENS:
            raise ModelScenePlanCodecError(
                f"unsupported audio-aware delta operation {operation!r}"
            )
        result: dict[str, Any] = {
            "contract": AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT,
            "operation": operation,
        }
        if operation == "room_change":
            room = str(value.get("new_room") or "")
            if room not in ROOM_TOKENS:
                raise ModelScenePlanCodecError("audio-aware room delta is invalid")
            result["new_room"] = room
        elif operation in {
            "remove_source",
            "move_source",
            "retime_source",
            "change_speech_description",
            "change_transcript",
        }:
            source_id = str(value.get("source_id") or "")
            _source_slot(source_id)
            result["source_id"] = source_id
            if operation == "change_speech_description":
                result["new_speaker_description"] = _normalized_text(
                    value.get("new_speaker_description"),
                    label="new speaker description",
                )
            elif operation == "change_transcript":
                result["new_transcript"] = _normalized_text(
                    value.get("new_transcript"), label="new transcript"
                )
        elif operation in {"add_source", "replace_source"}:
            raw = value.get("source_semantic")
            if raw is None:
                raw = value.get("source" if operation == "add_source" else "new_source")
            if not isinstance(raw, Mapping):
                raise ModelScenePlanCodecError(
                    f"{operation} requires a semantic source payload"
                )
            source = self._semantic_source(raw)
            if operation == "replace_source" and value.get("source_id") is not None:
                if str(value["source_id"]) != str(source["source_id"]):
                    raise ModelScenePlanCodecError(
                        "replacement semantic payload changed its persistent source id"
                    )
            result["source_semantic"] = source
        return result

    def encode(self, program: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        value = self._normalize_program(program)
        operation = value["operation"]
        ids = [
            self.bos_id,
            self.patch_codec.token_to_id[ACTIVE_OPERATION_TOKENS[operation]],
        ]
        groups = [LOSS_GRAMMAR, LOSS_GRAMMAR]

        def emit(token_id: int, group: int) -> None:
            ids.append(int(token_id))
            groups.append(int(group))

        def text(marker: str, content: str, group: int) -> None:
            emit(self.plan_codec._tid(marker), group)
            pieces = self.plan_codec._text(content)
            ids.extend(pieces)
            groups.extend([int(group)] * len(pieces))

        def semantic_source(source: Mapping[str, Any]) -> None:
            emit(self.plan_codec._tid("<source_begin>"), LOSS_GRAMMAR)
            emit(self._source_id(source["source_id"]), LOSS_SEMANTIC)
            emit(self.plan_codec._tid("<kind>"), LOSS_SEMANTIC)
            emit(self.plan_codec._tid(KIND_TOKENS[str(source["kind"])]), LOSS_SEMANTIC)
            if source["kind"] == "speech":
                text("<speaker_description>", source["speaker_description"], LOSS_SEMANTIC)
                text("<transcript>", source["transcript"], LOSS_SPEECH_CONTENT)
            else:
                text("<description>", source["description"], LOSS_SEMANTIC)
            emit(self.plan_codec._tid("<source_end>"), LOSS_GRAMMAR)

        if operation == "room_change":
            emit(self.plan_codec._tid(ROOM_TOKENS[value["new_room"]]), LOSS_ROOM)
        elif operation in {
            "remove_source",
            "move_source",
            "retime_source",
        }:
            emit(self._source_id(value["source_id"]), LOSS_SEMANTIC)
        elif operation in {"add_source", "replace_source"}:
            semantic_source(value["source_semantic"])
        elif operation == "change_speech_description":
            emit(self._source_id(value["source_id"]), LOSS_SEMANTIC)
            text(
                "<speaker_description>",
                value["new_speaker_description"],
                LOSS_SEMANTIC,
            )
        elif operation == "change_transcript":
            emit(self._source_id(value["source_id"]), LOSS_SEMANTIC)
            text("<transcript>", value["new_transcript"], LOSS_SPEECH_CONTENT)
        emit(self.eos_id, LOSS_GRAMMAR)
        if len(ids) > 512:
            raise ModelScenePlanCodecError(
                "audio-aware DeltaSketch exceeds its 512-token ceiling"
            )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool),
            "loss_group_ids": torch.tensor(groups, dtype=torch.long),
        }

    def decode(self, token_ids: Sequence[int] | torch.Tensor) -> dict[str, Any]:
        values = torch.as_tensor(token_ids, dtype=torch.long).flatten().tolist()
        if (
            len(values) < 3
            or len(values) > 512
            or values[0] != self.bos_id
            or values[-1] != self.eos_id
        ):
            raise ModelScenePlanCodecError(
                "audio-aware DeltaSketch must contain BOS, operation, and EOS"
            )
        operation_lookup = {
            self.patch_codec.token_to_id[token]: operation
            for operation, token in ACTIVE_OPERATION_TOKENS.items()
        }
        operation = operation_lookup.get(int(values[1]))
        if operation is None:
            raise ModelScenePlanCodecError(
                "audio-aware DeltaSketch contains an inactive operation"
            )
        cur = self.plan_codec._Decoder(self.plan_codec, values[2:-1])
        source_lookup = {
            self.plan_codec._tid(token): f"source_{index}"
            for index, token in enumerate(SOURCE_SLOT_TOKENS)
        }
        kind_lookup = {
            self.plan_codec._tid(token): kind for kind, token in KIND_TOKENS.items()
        }

        def source_id() -> str:
            token = cur.take()
            if token not in source_lookup:
                raise ModelScenePlanCodecError(
                    "audio-aware DeltaSketch contains an invalid source owner"
                )
            return source_lookup[token]

        def semantic_source() -> dict[str, Any]:
            cur.expect("<source_begin>")
            owner = source_id()
            cur.expect("<kind>")
            kind_id = cur.take()
            if kind_id not in kind_lookup:
                raise ModelScenePlanCodecError(
                    "audio-aware DeltaSketch contains an invalid source kind"
                )
            kind = kind_lookup[kind_id]
            source: dict[str, Any] = {"source_id": owner, "kind": kind}
            if kind == "speech":
                cur.expect("<speaker_description>")
                source["speaker_description"] = cur.text()
                cur.expect("<transcript>")
                source["transcript"] = cur.text()
            else:
                cur.expect("<description>")
                source["description"] = cur.text()
            cur.expect("<source_end>")
            return self._semantic_source(source)

        result: dict[str, Any] = {
            "contract": AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT,
            "operation": operation,
        }
        if operation == "room_change":
            room_lookup = {
                self.plan_codec._tid(token): room for room, token in ROOM_TOKENS.items()
            }
            room_id = cur.take()
            if room_id not in room_lookup:
                raise ModelScenePlanCodecError("audio-aware room delta is invalid")
            result["new_room"] = room_lookup[room_id]
        elif operation in {"remove_source", "move_source", "retime_source"}:
            result["source_id"] = source_id()
        elif operation in {"add_source", "replace_source"}:
            result["source_semantic"] = semantic_source()
        elif operation == "change_speech_description":
            result["source_id"] = source_id()
            cur.expect("<speaker_description>")
            result["new_speaker_description"] = cur.text()
        elif operation == "change_transcript":
            result["source_id"] = source_id()
            cur.expect("<transcript>")
            result["new_transcript"] = cur.text()
        if cur.index != len(cur.values):
            raise ModelScenePlanCodecError(
                "tokens remain after complete audio-aware DeltaSketch"
            )
        return self._normalize_program(result)

    def canonicalize(
        self, token_ids: Sequence[int] | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return self.encode(self.decode(token_ids))

    def allowed_next_ids(
        self,
        prefix: Sequence[int] | torch.Tensor,
        *,
        observed_sceneplan: Mapping[str, Any] | None = None,
        input_sceneplan: Mapping[str, Any] | None = None,
        max_text_tokens: int = 192,
    ) -> set[int]:
        """Return the exact active semantics-only grammar."""

        if observed_sceneplan is None:
            observed_sceneplan = input_sceneplan
        if observed_sceneplan is None:
            raise ValueError(
                "audio-aware DeltaSketch decoding requires the observed ScenePlan"
            )
        observed = validate_p11_executor_profile(
            self.plan_codec.project_plan(observed_sceneplan)
        )
        values = torch.as_tensor(prefix, dtype=torch.long).flatten().tolist()
        index = 0

        def choice(allowed: set[int]) -> int:
            nonlocal index
            if index >= len(values):
                raise _NeedSketchToken(set(allowed))
            value = int(values[index])
            if value not in allowed:
                raise ModelScenePlanCodecError(
                    f"invalid audio-aware DeltaSketch token {value} at {index}"
                )
            index += 1
            return value

        def exact(token: str | int) -> None:
            choice(
                {int(token)}
                if isinstance(token, int)
                else {self.plan_codec._tid(token)}
            )

        def text_value() -> None:
            exact("<text_begin>")
            emitted = 0
            while True:
                allowed = set(
                    self.nonempty_first_text_ids
                    if emitted == 0
                    else self.plan_codec.text_ids
                )
                if emitted:
                    allowed.add(self.plan_codec._tid("<text_end>"))
                if emitted >= int(max_text_tokens):
                    allowed = {self.plan_codec._tid("<text_end>")}
                token = choice(allowed)
                if token == self.plan_codec._tid("<text_end>"):
                    return
                emitted += 1

        existing = {
            self._source_id(source["source_id"]) for source in observed["sources"]
        }
        speech = {
            self._source_id(source["source_id"])
            for source in observed["sources"]
            if source["kind"] == "speech"
        }
        free = next(
            (
                self._source_id(f"source_{slot}")
                for slot in range(4)
                if self._source_id(f"source_{slot}") not in existing
            ),
            None,
        )

        def semantic_source(*, add: bool) -> None:
            exact("<source_begin>")
            owner = choice({free} if add and free is not None else existing)
            exact("<kind>")
            kinds = {self.plan_codec._tid(token) for token in KIND_TOKENS.values()}
            if speech and (add or owner not in speech):
                kinds.discard(self.plan_codec._tid(KIND_TOKENS["speech"]))
            kind = choice(kinds)
            if kind == self.plan_codec._tid(KIND_TOKENS["speech"]):
                exact("<speaker_description>")
                text_value()
                exact("<transcript>")
                text_value()
            else:
                exact("<description>")
                text_value()
            exact("<source_end>")

        operations = set(AUDIO_AWARE_DELTA_OPERATIONS)
        if len(observed["sources"]) == 1:
            operations.discard("remove_source")
        if len(observed["sources"]) == 4:
            operations.discard("add_source")
        if not speech:
            operations.discard("change_speech_description")
            operations.discard("change_transcript")
        try:
            exact(self.bos_id)
            operation_id = choice(
                {
                    self.patch_codec.token_to_id[ACTIVE_OPERATION_TOKENS[name]]
                    for name in operations
                }
            )
            operation = next(
                name
                for name in operations
                if operation_id
                == self.patch_codec.token_to_id[ACTIVE_OPERATION_TOKENS[name]]
            )
            if operation == "room_change":
                rooms = {
                    self.plan_codec._tid(token) for token in ROOM_TOKENS.values()
                }
                rooms.discard(
                    self.plan_codec._tid(ROOM_TOKENS[observed["room"]["type"]])
                )
                choice(rooms)
            elif operation in {"remove_source", "move_source", "retime_source"}:
                choice(existing)
            elif operation == "add_source":
                semantic_source(add=True)
            elif operation == "replace_source":
                semantic_source(add=False)
            elif operation in {"change_speech_description", "change_transcript"}:
                choice(speech)
                exact(
                    "<speaker_description>"
                    if operation == "change_speech_description"
                    else "<transcript>"
                )
                text_value()
            exact(self.eos_id)
            if index != len(values):
                raise ModelScenePlanCodecError(
                    "tokens remain after complete audio-aware DeltaSketch"
                )
            return set()
        except _NeedSketchToken as need:
            return need.allowed


def _frame_from_seconds(value: Any, *, max_frames: int) -> int:
    frame = int(round(float(value) * MODEL_SAMPLE_RATE / VAE_HOP_SAMPLES))
    if not 0 <= frame <= int(max_frames):
        raise ModelScenePlanCodecError(
            f"projected frame lies outside [0,{int(max_frames)}]"
        )
    return frame


def _seconds_from_frame(frame: int) -> float:
    return float(int(frame) * VAE_HOP_SAMPLES / MODEL_SAMPLE_RATE)


def _distance_norm(distance_m: Any) -> float:
    distance = min(DISTANCE_MAX_M, max(DISTANCE_MIN_M, float(distance_m)))
    return (
        (math.log(distance) - math.log(DISTANCE_MIN_M))
        / (math.log(DISTANCE_MAX_M) - math.log(DISTANCE_MIN_M))
    )


def _distance_from_norm(value: Any) -> float:
    clipped = min(1.0, max(0.0, float(value)))
    return math.exp(
        math.log(DISTANCE_MIN_M)
        + clipped * (math.log(DISTANCE_MAX_M) - math.log(DISTANCE_MIN_M))
    )


def _write_position(vector: np.ndarray, prefix: str, position: Mapping[str, Any]) -> None:
    azimuth = math.radians(float(position["azimuth_deg"]))
    elevation = math.radians(float(position["elevation_deg"]))
    vector[_EXECUTION_INDEX[f"{prefix}_sin_azimuth"]] = math.sin(azimuth)
    vector[_EXECUTION_INDEX[f"{prefix}_cos_azimuth"]] = math.cos(azimuth)
    vector[_EXECUTION_INDEX[f"{prefix}_sin_elevation"]] = math.sin(elevation)
    vector[_EXECUTION_INDEX[f"{prefix}_cos_elevation"]] = math.cos(elevation)
    vector[_EXECUTION_INDEX[f"{prefix}_log_distance_norm"]] = _distance_norm(
        position["distance_m"]
    )


def _read_position(
    vector: np.ndarray, prefix: str, codec: ModelScenePlanCodecV3
) -> dict[str, float]:
    azimuth = math.degrees(
        math.atan2(
            float(vector[_EXECUTION_INDEX[f"{prefix}_sin_azimuth"]]),
            float(vector[_EXECUTION_INDEX[f"{prefix}_cos_azimuth"]]),
        )
    )
    elevation = math.degrees(
        math.atan2(
            float(vector[_EXECUTION_INDEX[f"{prefix}_sin_elevation"]]),
            float(vector[_EXECUTION_INDEX[f"{prefix}_cos_elevation"]]),
        )
    )
    distance = _distance_from_norm(
        vector[_EXECUTION_INDEX[f"{prefix}_log_distance_norm"]]
    )
    return {
        "azimuth_deg": float(codec._azimuth_value(azimuth)),
        "elevation_deg": float(codec._elevation_value(elevation)),
        "distance_m": codec._distance_value(codec._distance_index(distance)),
    }


def _trajectory_endpoints(
    trajectory: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    motion = str(trajectory["type"])
    if motion == "static":
        return trajectory["position"], trajectory["position"]
    if motion == "linear":
        return trajectory["start"], trajectory["end"]
    raise ModelScenePlanCodecError(
        "P10ExecutionState supports only the trained static/linear profile"
    )


def compile_execution_state(
    sceneplan: Mapping[str, Any], codec: ModelScenePlanCodecV3
) -> dict[str, Any]:
    """Compile only the numeric controls P10 can execute."""

    validate_p11_executor_profile(validate_model_sceneplan(sceneplan))
    projected = validate_p11_executor_profile(codec.project_plan(sceneplan))
    duration_frames = _frame_from_seconds(
        projected["duration_sec"], max_frames=codec.max_frames
    )
    core = np.zeros(
        (EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM), dtype=np.float32
    )
    core[0, _EXECUTION_INDEX["duration_frames_norm"]] = (
        duration_frames / codec.max_frames
    )
    source_present = [False] * 4
    for source in projected["sources"]:
        source_index = _source_slot(source["source_id"])
        source_present[source_index] = True
        vector = core[source_index + 1]
        onset = _frame_from_seconds(
            source["activity"]["onset_sec"], max_frames=codec.max_frames
        )
        offset = _frame_from_seconds(
            source["activity"]["offset_sec"], max_frames=codec.max_frames
        )
        vector[_EXECUTION_INDEX["onset_frame_norm"]] = onset / duration_frames
        vector[_EXECUTION_INDEX["offset_frame_norm"]] = offset / duration_frames
        motion = str(source["trajectory"]["type"])
        vector[_EXECUTION_INDEX[f"motion_{motion}"]] = 1.0
        start, end = _trajectory_endpoints(source["trajectory"])
        _write_position(vector, "start", start)
        _write_position(vector, "end", end)
    result = {
        "schema": EXECUTION_STATE_SCHEMA,
        "schema_version": EXECUTION_STATE_VERSION,
        "contract": EXECUTION_STATE_CONTRACT,
        "codec_fingerprint": codec.fingerprint,
        "sceneplan_sha256": sha256_json(projected),
        "sample_id": str(projected["sample_id"]),
        "slot_count": EXECUTION_SLOT_COUNT,
        "feature_schema": EXECUTION_FEATURE_SCHEMA,
        "feature_dim": EXECUTION_FEATURE_DIM,
        "max_frames": int(codec.max_frames),
        "source_present_mask": source_present,
        "core_features": core.tolist(),
    }
    return validate_execution_state(result, codec=codec)


def execution_state_core(state: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray(state["core_features"], dtype=np.float32)
    if values.shape != (EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM):
        raise ModelScenePlanCodecError(
            f"P10ExecutionState core must be [{EXECUTION_SLOT_COUNT},{EXECUTION_FEATURE_DIM}]"
        )
    if not np.isfinite(values).all():
        raise ModelScenePlanCodecError("P10ExecutionState contains non-finite values")
    return values


def validate_execution_state(
    state: Mapping[str, Any],
    *,
    codec: ModelScenePlanCodecV3 | None = None,
) -> dict[str, Any]:
    """Validate a finite, semantics-free P10 numeric state."""

    required = {
        "schema",
        "schema_version",
        "contract",
        "codec_fingerprint",
        "sceneplan_sha256",
        "sample_id",
        "slot_count",
        "feature_schema",
        "feature_dim",
        "max_frames",
        "source_present_mask",
        "core_features",
    }
    if set(state) != required:
        raise ModelScenePlanCodecError("P10ExecutionState top-level fields changed")
    if (
        state["schema"] != EXECUTION_STATE_SCHEMA
        or int(state["schema_version"]) != EXECUTION_STATE_VERSION
        or state["contract"] != EXECUTION_STATE_CONTRACT
        or int(state["slot_count"]) != EXECUTION_SLOT_COUNT
        or state["feature_schema"] != EXECUTION_FEATURE_SCHEMA
        or int(state["feature_dim"]) != EXECUTION_FEATURE_DIM
    ):
        raise ModelScenePlanCodecError("P10ExecutionState contract/version mismatch")
    if not str(state["sample_id"]):
        raise ModelScenePlanCodecError("P10ExecutionState sample_id is empty")
    if codec is not None:
        if state["codec_fingerprint"] != codec.fingerprint:
            raise ModelScenePlanCodecError(
                "P10ExecutionState codec fingerprint mismatch"
            )
        if int(state["max_frames"]) != int(codec.max_frames):
            raise ModelScenePlanCodecError("P10ExecutionState frame ceiling mismatch")
    digest = state["sceneplan_sha256"]
    if digest is not None and (
        not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
    ):
        raise ModelScenePlanCodecError("P10ExecutionState ScenePlan hash is invalid")
    mask = state["source_present_mask"]
    if (
        not isinstance(mask, list)
        or len(mask) != 4
        or any(not isinstance(value, bool) for value in mask)
        or not any(mask)
    ):
        raise ModelScenePlanCodecError(
            "P10ExecutionState source_present_mask must contain one to four sources"
        )
    core = execution_state_core(state)
    if not np.allclose(
        np.delete(core[0], _EXECUTION_INDEX["duration_frames_norm"]),
        0.0,
        atol=1e-7,
        rtol=0.0,
    ):
        raise ModelScenePlanCodecError(
            "P10ExecutionState global slot contains source controls"
        )
    duration_norm = float(core[0, _EXECUTION_INDEX["duration_frames_norm"]])
    if not 0.0 < duration_norm <= 1.0:
        raise ModelScenePlanCodecError("P10ExecutionState duration is out of range")
    duration_frames = int(round(duration_norm * int(state["max_frames"])))
    if not 1 <= duration_frames <= int(state["max_frames"]):
        raise ModelScenePlanCodecError("P10ExecutionState duration quantizes invalidly")
    for source_index, present in enumerate(mask):
        vector = core[source_index + 1]
        if not present:
            if not np.allclose(vector, 0.0, atol=1e-7, rtol=0.0):
                raise ModelScenePlanCodecError(
                    f"inactive source_{source_index} execution slot is non-zero"
                )
            continue
        if abs(float(vector[_EXECUTION_INDEX["duration_frames_norm"]])) > 1e-7:
            raise ModelScenePlanCodecError(
                f"source_{source_index} illegally owns global duration"
            )
        onset_norm = float(vector[_EXECUTION_INDEX["onset_frame_norm"]])
        offset_norm = float(vector[_EXECUTION_INDEX["offset_frame_norm"]])
        onset = int(round(onset_norm * duration_frames))
        offset = int(round(offset_norm * duration_frames))
        if not 0.0 <= onset_norm <= 1.0 or not 0.0 <= offset_norm <= 1.0:
            raise ModelScenePlanCodecError(
                f"source_{source_index} activity normalization is out of range"
            )
        if not 0 <= onset < offset <= duration_frames:
            raise ModelScenePlanCodecError(
                f"source_{source_index} activity interval is invalid"
            )
        motion = vector[
            [_EXECUTION_INDEX["motion_static"], _EXECUTION_INDEX["motion_linear"]]
        ]
        if float(np.max(motion)) <= 0.0:
            raise ModelScenePlanCodecError(
                f"source_{source_index} has no executable motion class"
            )
        for prefix in ("start", "end"):
            azimuth_norm = math.hypot(
                float(vector[_EXECUTION_INDEX[f"{prefix}_sin_azimuth"]]),
                float(vector[_EXECUTION_INDEX[f"{prefix}_cos_azimuth"]]),
            )
            elevation_norm = math.hypot(
                float(vector[_EXECUTION_INDEX[f"{prefix}_sin_elevation"]]),
                float(vector[_EXECUTION_INDEX[f"{prefix}_cos_elevation"]]),
            )
            distance = float(
                vector[_EXECUTION_INDEX[f"{prefix}_log_distance_norm"]]
            )
            if azimuth_norm < 1e-4 or elevation_norm < 1e-4:
                raise ModelScenePlanCodecError(
                    f"source_{source_index} {prefix} angle vector is degenerate"
                )
            if not 0.0 <= distance <= 1.0:
                raise ModelScenePlanCodecError(
                    f"source_{source_index} {prefix} distance is out of range"
                )
    return copy.deepcopy(dict(state))


def execution_state_from_core(
    core_features: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    sketch: Mapping[str, Any],
    codec: ModelScenePlanCodecV3,
    *,
    sample_id: str,
    duration_frames: int | None = None,
    sceneplan_sha256: str | None = None,
) -> dict[str, Any]:
    """Project unconstrained model output onto the finite P10 numeric domain.

    This is an explicit inference projection, not hidden repair inside the
    assembler.  It never creates semantic fields or source inventory; both are
    copied from the already-decoded SceneSketch.
    """

    semantic = validate_scene_sketch(sketch, codec=codec)
    if isinstance(core_features, torch.Tensor):
        raw = core_features.detach().float().cpu().numpy()
    else:
        raw = np.asarray(core_features, dtype=np.float32)
    if raw.shape != (EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM):
        raise ModelScenePlanCodecError(
            f"predicted execution core must be [{EXECUTION_SLOT_COUNT},{EXECUTION_FEATURE_DIM}]"
        )
    if not np.isfinite(raw).all():
        raise ModelScenePlanCodecError("predicted execution core is non-finite")
    core = np.zeros_like(raw, dtype=np.float32)
    if duration_frames is None:
        duration_norm = float(
            np.clip(raw[0, _EXECUTION_INDEX["duration_frames_norm"]], 1.0 / codec.max_frames, 1.0)
        )
        duration_frames = int(round(duration_norm * codec.max_frames))
    duration_frames = max(1, min(codec.max_frames, int(duration_frames)))
    core[0, _EXECUTION_INDEX["duration_frames_norm"]] = (
        duration_frames / codec.max_frames
    )
    mask = _sketch_present_mask(semantic)
    for source_index, present in enumerate(mask):
        if not present:
            continue
        source = raw[source_index + 1]
        output = core[source_index + 1]
        onset = int(
            round(
                float(np.clip(source[_EXECUTION_INDEX["onset_frame_norm"]], 0.0, 1.0))
                * duration_frames
            )
        )
        offset = int(
            round(
                float(np.clip(source[_EXECUTION_INDEX["offset_frame_norm"]], 0.0, 1.0))
                * duration_frames
            )
        )
        onset = max(0, min(duration_frames - 1, onset))
        offset = max(onset + 1, min(duration_frames, offset))
        output[_EXECUTION_INDEX["onset_frame_norm"]] = onset / duration_frames
        output[_EXECUTION_INDEX["offset_frame_norm"]] = offset / duration_frames
        motion_values = source[
            [_EXECUTION_INDEX["motion_static"], _EXECUTION_INDEX["motion_linear"]]
        ]
        motion_index = int(np.argmax(motion_values))
        output[_EXECUTION_INDEX["motion_static"]] = float(motion_index == 0)
        output[_EXECUTION_INDEX["motion_linear"]] = float(motion_index == 1)
        for prefix in ("start", "end"):
            for axis in ("azimuth", "elevation"):
                sin_index = _EXECUTION_INDEX[f"{prefix}_sin_{axis}"]
                cos_index = _EXECUTION_INDEX[f"{prefix}_cos_{axis}"]
                sine = float(source[sin_index])
                cosine = float(source[cos_index])
                norm = math.hypot(sine, cosine)
                if norm < 1.0e-6:
                    sine, cosine, norm = 0.0, 1.0, 1.0
                output[sin_index] = sine / norm
                output[cos_index] = cosine / norm
            distance_index = _EXECUTION_INDEX[f"{prefix}_log_distance_norm"]
            output[distance_index] = float(
                np.clip(source[distance_index], 0.0, 1.0)
            )
    result = {
        "schema": EXECUTION_STATE_SCHEMA,
        "schema_version": EXECUTION_STATE_VERSION,
        "contract": EXECUTION_STATE_CONTRACT,
        "codec_fingerprint": codec.fingerprint,
        "sceneplan_sha256": sceneplan_sha256,
        "sample_id": str(sample_id),
        "slot_count": EXECUTION_SLOT_COUNT,
        "feature_schema": EXECUTION_FEATURE_SCHEMA,
        "feature_dim": EXECUTION_FEATURE_DIM,
        "max_frames": int(codec.max_frames),
        "source_present_mask": mask,
        "core_features": core.tolist(),
    }
    return validate_execution_state(result, codec=codec)


def execution_delta_core(
    input_state: Mapping[str, Any], target_state: Mapping[str, Any]
) -> np.ndarray:
    """Return the exact finite target-minus-input numeric delta."""

    before = execution_state_core(validate_execution_state(input_state))
    after = execution_state_core(validate_execution_state(target_state))
    return (after - before).astype(np.float32)


def audio_aware_delta_control_mask(
    delta_program: Mapping[str, Any],
) -> np.ndarray:
    """Return the numeric coordinates owned by an audio-aware edit program.

    This mask mirrors :func:`project_audio_aware_delta_to_atomic_patch`: only
    add/move/retime consume values from ``DeltaExecutionState``.  Normalizing
    their regression loss over these coordinates prevents a sparse retime
    target (two coordinates) from being diluted by every preserved source
    coordinate in a multi-source scene.
    """

    if delta_program.get("contract") != AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT:
        raise ModelScenePlanCodecError("wrong audio-aware DeltaSketch contract")
    operation = str(delta_program.get("operation") or "")
    if operation not in AUDIO_AWARE_DELTA_OPERATIONS:
        raise ModelScenePlanCodecError(
            f"unsupported audio-aware delta operation {operation!r}"
        )
    mask = np.zeros(
        (EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM), dtype=np.bool_
    )
    if operation == "add_source":
        semantic = delta_program.get("source_semantic")
        if not isinstance(semantic, Mapping):
            raise ModelScenePlanCodecError("add_source lacks source semantics")
        row = _source_slot(str(semantic.get("source_id") or "")) + 1
        # Source slots never own the scene-global duration coordinate.
        mask[row, 1:] = True
    elif operation == "move_source":
        row = _source_slot(str(delta_program.get("source_id") or "")) + 1
        mask[row, _EXECUTION_INDEX["motion_static"] :] = True
    elif operation == "retime_source":
        row = _source_slot(str(delta_program.get("source_id") or "")) + 1
        mask[
            row,
            _EXECUTION_INDEX["onset_frame_norm"] :
            _EXECUTION_INDEX["offset_frame_norm"] + 1,
        ] = True
    return mask


def compile_audio_aware_delta_sketch(
    observed_sceneplan: Mapping[str, Any],
    revised_sceneplan: Mapping[str, Any],
    patch_spec: Mapping[str, Any],
    codec: ModelScenePlanCodecV3,
    patch_codec: ScenePlanEditPatchCodec | None = None,
) -> dict[str, Any]:
    """Compile the discrete, semantics-only part of an audio-aware edit."""

    patch_codec = patch_codec or ScenePlanEditPatchCodec(codec)
    observed = validate_p11_executor_profile(codec.project_plan(observed_sceneplan))
    revised = validate_p11_executor_profile(codec.project_plan(revised_sceneplan))
    patch_codec.assert_target(observed, patch_spec, revised)
    delta_codec = AudioAwareDeltaSceneSketchCodec(codec, patch_codec)
    # Round-tripping through the active grammar strips numeric payloads from
    # move/retime/add and leaves them exclusively to DeltaExecutionState.
    return delta_codec.decode(delta_codec.encode(patch_spec)["input_ids"])


def apply_audio_aware_delta_program_to_sketch(
    observed_sketch: Mapping[str, Any],
    delta_program: Mapping[str, Any],
    codec: ModelScenePlanCodecV3,
    patch_codec: ScenePlanEditPatchCodec | None = None,
) -> dict[str, Any]:
    """Apply only discrete semantic/inventory changes to SceneSketch."""

    delta_codec = AudioAwareDeltaSceneSketchCodec(codec, patch_codec)
    program = delta_codec.decode(delta_codec.encode(delta_program)["input_ids"])
    output = copy.deepcopy(validate_scene_sketch(observed_sketch, codec=codec))
    output["sceneplan_sha256"] = None
    operation = str(program["operation"])
    by_id = {
        str(source["source_id"]): source for source in output["sources"]
    }
    if operation == "no_op":
        pass
    elif operation == "room_change":
        room = str(program["new_room"])
        if room == str(output["room_intent"]):
            raise ModelScenePlanCodecError("room_change must select a new room")
        output["room_intent"] = room
    elif operation == "remove_source":
        source_id = str(program["source_id"])
        if source_id not in by_id or len(output["sources"]) <= 1:
            raise ModelScenePlanCodecError("remove_source has no legal owner")
        output["sources"] = [
            source
            for source in output["sources"]
            if str(source["source_id"]) != source_id
        ]
    elif operation == "add_source":
        source = copy.deepcopy(program["source_semantic"])
        occupied = {_source_slot(value["source_id"]) for value in output["sources"]}
        free = next((slot for slot in range(4) if slot not in occupied), None)
        if free is None or str(source["source_id"]) != f"source_{free}":
            raise ModelScenePlanCodecError(
                "add_source must consume the first free persistent slot"
            )
        output["sources"].append(source)
        output["sources"].sort(key=lambda value: _source_slot(value["source_id"]))
    elif operation == "replace_source":
        source = copy.deepcopy(program["source_semantic"])
        source_id = str(source["source_id"])
        if source_id not in by_id:
            raise ModelScenePlanCodecError("replace_source owner is absent")
        output["sources"] = [
            source if str(value["source_id"]) == source_id else value
            for value in output["sources"]
        ]
    elif operation in {"change_speech_description", "change_transcript"}:
        source_id = str(program["source_id"])
        source = by_id.get(source_id)
        if source is None or source["kind"] != "speech":
            raise ModelScenePlanCodecError(
                f"{operation} requires an observed speech source"
            )
        if operation == "change_speech_description":
            source["speaker_description"] = program["new_speaker_description"]
        else:
            source["transcript"] = program["new_transcript"]
    elif operation in {"move_source", "retime_source"}:
        if str(program["source_id"]) not in by_id:
            raise ModelScenePlanCodecError(f"{operation} owner is absent")
    else:  # pragma: no cover - normalized grammar makes this unreachable.
        raise ModelScenePlanCodecError(f"unsupported operation {operation!r}")
    return validate_scene_sketch(output, codec=codec)


def project_audio_aware_delta_to_atomic_patch(
    observed_sceneplan: Mapping[str, Any],
    delta_program: Mapping[str, Any],
    predicted_delta_core: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    codec: ModelScenePlanCodecV3,
    patch_codec: ScenePlanEditPatchCodec | None = None,
) -> dict[str, Any]:
    """Materialize one executable patch with exactly one authority per field.

    The observed ScenePlan is audio-derived.  ``delta_program`` owns discrete
    semantics and source identity; ``predicted_delta_core`` owns only numeric
    activity/trajectory changes.  Unspecified fields are copied exactly.
    """

    patch_codec = patch_codec or ScenePlanEditPatchCodec(codec)
    delta_codec = AudioAwareDeltaSceneSketchCodec(codec, patch_codec)
    program = delta_codec.decode(delta_codec.encode(delta_program)["input_ids"])
    observed = validate_p11_executor_profile(codec.project_plan(observed_sceneplan))
    observed_sketch = compile_scene_sketch(observed, codec)
    revised_sketch = apply_audio_aware_delta_program_to_sketch(
        observed_sketch, program, codec, patch_codec
    )
    observed_state = compile_execution_state(observed, codec)
    observed_core = execution_state_core(observed_state)
    if isinstance(predicted_delta_core, torch.Tensor):
        predicted = predicted_delta_core.detach().float().cpu().numpy()
    else:
        predicted = np.asarray(predicted_delta_core, dtype=np.float32)
    if predicted.shape != (EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM):
        raise ModelScenePlanCodecError(
            f"DeltaExecutionState must be [{EXECUTION_SLOT_COUNT},{EXECUTION_FEATURE_DIM}]"
        )
    if not np.isfinite(predicted).all():
        raise ModelScenePlanCodecError("DeltaExecutionState is non-finite")

    operation = str(program["operation"])
    raw_target = observed_core + predicted
    projected_core = observed_core.copy()
    if operation == "add_source":
        slot = _source_slot(program["source_semantic"]["source_id"])
        projected_core[slot + 1] = raw_target[slot + 1]
    elif operation == "remove_source":
        slot = _source_slot(program["source_id"])
        projected_core[slot + 1] = 0.0
    elif operation == "move_source":
        slot = _source_slot(program["source_id"])
        # Motion class and both endpoints are the trajectory authority.
        projected_core[slot + 1, 3:] = raw_target[slot + 1, 3:]
    elif operation == "retime_source":
        slot = _source_slot(program["source_id"])
        projected_core[slot + 1, 1:3] = raw_target[slot + 1, 1:3]

    duration_frames = _frame_from_seconds(
        observed["duration_sec"], max_frames=codec.max_frames
    )
    revised_state = execution_state_from_core(
        projected_core,
        revised_sketch,
        codec,
        sample_id=str(observed["sample_id"]),
        duration_frames=duration_frames,
    )
    revised = assemble_sceneplan(revised_sketch, revised_state, codec)
    by_id = {
        str(source["source_id"]): source for source in revised["sources"]
    }
    if operation == "no_op":
        patch: dict[str, Any] = {"operation": operation}
    elif operation == "room_change":
        patch = {"operation": operation, "new_room": program["new_room"]}
    elif operation == "remove_source":
        patch = {"operation": operation, "source_id": program["source_id"]}
    elif operation == "add_source":
        source_id = str(program["source_semantic"]["source_id"])
        patch = {"operation": operation, "source": copy.deepcopy(by_id[source_id])}
    elif operation == "replace_source":
        source_id = str(program["source_semantic"]["source_id"])
        patch = {
            "operation": operation,
            "source_id": source_id,
            "new_source": copy.deepcopy(by_id[source_id]),
        }
    elif operation == "move_source":
        source_id = str(program["source_id"])
        patch = {
            "operation": operation,
            "source_id": source_id,
            "trajectory": copy.deepcopy(by_id[source_id]["trajectory"]),
        }
    elif operation == "retime_source":
        source_id = str(program["source_id"])
        activity = by_id[source_id]["activity"]
        patch = {
            "operation": operation,
            "source_id": source_id,
            "new_interval_frames": [
                _frame_from_seconds(activity["onset_sec"], max_frames=codec.max_frames),
                _frame_from_seconds(activity["offset_sec"], max_frames=codec.max_frames),
            ],
        }
    elif operation == "change_speech_description":
        patch = {
            "operation": operation,
            "source_id": program["source_id"],
            "new_speaker_description": program["new_speaker_description"],
        }
    else:
        patch = {
            "operation": operation,
            "source_id": program["source_id"],
            "new_transcript": program["new_transcript"],
        }
    patch_tokens = patch_codec.encode(patch)["input_ids"]
    patch = patch_codec.decode(patch_tokens)
    applied = patch_codec.apply(observed, patch)
    if not _same_plan_tokens(applied, revised, codec):
        raise ModelScenePlanCodecError(
            "audio-aware deterministic patch and revised ScenePlan disagree"
        )
    return {
        "delta_program": program,
        "observed_scene_sketch": observed_sketch,
        "observed_execution_state": observed_state,
        "revised_scene_sketch": revised_sketch,
        "revised_execution_state": revised_state,
        "patch_spec": patch,
        "patch_tokens": patch_tokens,
        "revised_sceneplan": revised,
    }


def apply_delta_program_to_sketch(
    input_sketch: Mapping[str, Any],
    program: Mapping[str, Any],
    codec: ModelScenePlanCodecV3,
) -> dict[str, Any]:
    """Apply only the semantic/inventory part of an edit program."""

    sketch = validate_scene_sketch(input_sketch, codec=codec)
    if program.get("contract") != DELTA_SKETCH_TOKEN_CONTRACT:
        raise ModelScenePlanCodecError("wrong DeltaSketch token contract")
    operation = str(program.get("operation") or "")
    if operation not in OPERATION_TOKENS:
        raise ModelScenePlanCodecError(f"unsupported delta operation {operation!r}")
    output = copy.deepcopy(sketch)
    output["sceneplan_sha256"] = None
    if operation == "room_change":
        room = str(program["new_room"])
        if room == output["room_intent"]:
            raise ModelScenePlanCodecError("room_change must select a new room")
        output["room_intent"] = room
    elif operation == "remove_source":
        source_id = str(program["source_id"])
        if len(output["sources"]) <= 1:
            raise ModelScenePlanCodecError("remove_source cannot empty SceneSketch")
        retained = [
            source
            for source in output["sources"]
            if str(source["source_id"]) != source_id
        ]
        if len(retained) == len(output["sources"]):
            raise ModelScenePlanCodecError(
                f"remove_source names absent source {source_id!r}"
            )
        output["sources"] = retained
    elif operation not in {"no_op", "rotate_source", "distance_source", "retime_source"}:
        raise ModelScenePlanCodecError(f"unsupported semantic delta {operation!r}")
    return validate_scene_sketch(output, codec=codec)


def _sketch_present_mask(sketch: Mapping[str, Any]) -> list[bool]:
    mask = [False] * 4
    for source in sketch["sources"]:
        mask[_source_slot(source["source_id"])] = True
    return mask


def assemble_sceneplan(
    sketch: Mapping[str, Any],
    execution_state: Mapping[str, Any],
    codec: ModelScenePlanCodecV3,
    *,
    require_target_hash: bool = False,
) -> dict[str, Any]:
    """Deterministically join immutable semantics and numeric P10 controls."""

    semantic = validate_scene_sketch(sketch, codec=codec)
    execution = validate_execution_state(execution_state, codec=codec)
    sketch_mask = _sketch_present_mask(semantic)
    if sketch_mask != list(execution["source_present_mask"]):
        raise ModelScenePlanCodecError(
            "SceneSketch inventory and P10ExecutionState mask disagree"
        )
    core = execution_state_core(execution)
    duration_frames = int(
        round(
            float(core[0, _EXECUTION_INDEX["duration_frames_norm"]])
            * codec.max_frames
        )
    )
    sources: list[dict[str, Any]] = []
    for source_semantic in semantic["sources"]:
        source_index = _source_slot(source_semantic["source_id"])
        vector = core[source_index + 1]
        onset = int(
            round(
                float(vector[_EXECUTION_INDEX["onset_frame_norm"]])
                * duration_frames
            )
        )
        offset = int(
            round(
                float(vector[_EXECUTION_INDEX["offset_frame_norm"]])
                * duration_frames
            )
        )
        motion_index = int(
            np.argmax(
                vector[
                    [
                        _EXECUTION_INDEX["motion_static"],
                        _EXECUTION_INDEX["motion_linear"],
                    ]
                ]
            )
        )
        start = _read_position(vector, "start", codec)
        end = _read_position(vector, "end", codec)
        trajectory: dict[str, Any]
        if motion_index == 0:
            trajectory = {"type": "static", "position": start}
        else:
            trajectory = {"type": "linear", "start": start, "end": end}
        source = {
            **copy.deepcopy(source_semantic),
            "activity": {
                "onset_sec": _seconds_from_frame(onset),
                "offset_sec": _seconds_from_frame(offset),
            },
            "trajectory": trajectory,
            "gain_db": 0.0,
        }
        sources.append(source)
    assembled = validate_p11_executor_profile(
        codec.project_plan(
            validate_model_sceneplan(
                {
                    "sample_id": str(execution["sample_id"]),
                    "duration_sec": _seconds_from_frame(duration_frames),
                    "room": {"type": str(semantic["room_intent"])},
                    "sources": sources,
                }
            )
        )
    )
    if require_target_hash:
        digest = sha256_json(assembled)
        hashes = {
            value
            for value in (
                semantic["sceneplan_sha256"],
                execution["sceneplan_sha256"],
            )
            if value is not None
        }
        if not hashes or hashes != {digest}:
            raise ModelScenePlanCodecError(
                "assembled ScenePlan does not match the bound target hash"
            )
    return assembled


def compile_p10_from_contract(
    sketch: Mapping[str, Any],
    execution_state: Mapping[str, Any],
    codec: ModelScenePlanCodecV3,
    *,
    require_target_hash: bool = False,
) -> dict[str, Any]:
    """Assemble and compile the exact semantic + 4+4 P10 conditions."""

    plan = assemble_sceneplan(
        sketch,
        execution_state,
        codec,
        require_target_hash=require_target_hash,
    )
    frames = _frame_from_seconds(plan["duration_sec"], max_frames=codec.max_frames)
    compiled = compile_p10_aligned_target_conditions(
        plan,
        model_num_samples=frames * VAE_HOP_SAMPLES,
        latent_frames_valid=frames,
    )
    return {"sceneplan": plan, **compiled}


_DELTA_GROUP = {
    "no_op": [],
    "room_change": ["room"],
    "rotate_source": ["geometry"],
    "distance_source": ["distance"],
    "retime_source": ["temporal"],
    "remove_source": ["source_inventory"],
}


def compile_delta_scene_sketch(
    input_sceneplan: Mapping[str, Any],
    target_sceneplan: Mapping[str, Any],
    patch_spec: Mapping[str, Any],
    codec: ModelScenePlanCodecV3,
) -> dict[str, Any]:
    """Compile the auditable discrete thought preceding an atomic edit patch."""

    current = validate_p11_executor_profile(codec.project_plan(input_sceneplan))
    target = validate_p11_executor_profile(codec.project_plan(target_sceneplan))
    operation = str(patch_spec.get("operation") or "")
    if operation not in OPERATION_TOKENS:
        raise ModelScenePlanCodecError(f"unsupported edit operation {operation!r}")
    before_ids = {str(source["source_id"]) for source in current["sources"]}
    after_ids = {str(source["source_id"]) for source in target["sources"]}
    if operation == "room_change":
        changed_source_ids = ["scene_global"]
    elif operation in {"rotate_source", "distance_source", "retime_source"}:
        changed_source_ids = [str(patch_spec["source_id"])]
    else:
        changed_source_ids = []
    result = {
        "schema": DELTA_SCENE_SKETCH_SCHEMA,
        "schema_version": DELTA_SCENE_SKETCH_VERSION,
        "contract": DELTA_SCENE_SKETCH_CONTRACT,
        "operation": operation,
        "changed_source_ids": changed_source_ids,
        "changed_field_groups": list(_DELTA_GROUP[operation]),
        "added_source_ids": sorted(after_ids - before_ids),
        "removed_source_ids": sorted(before_ids - after_ids),
    }
    return validate_delta_scene_sketch(
        result, input_sceneplan=current, target_sceneplan=target, codec=codec
    )


def validate_delta_scene_sketch(
    delta: Mapping[str, Any],
    *,
    input_sceneplan: Mapping[str, Any] | None = None,
    target_sceneplan: Mapping[str, Any] | None = None,
    codec: ModelScenePlanCodecV3 | None = None,
) -> dict[str, Any]:
    required = {
        "schema",
        "schema_version",
        "contract",
        "operation",
        "changed_source_ids",
        "changed_field_groups",
        "added_source_ids",
        "removed_source_ids",
    }
    if set(delta) != required:
        raise ModelScenePlanCodecError("DeltaSceneSketch fields changed")
    operation = str(delta["operation"])
    if (
        delta["schema"] != DELTA_SCENE_SKETCH_SCHEMA
        or int(delta["schema_version"]) != DELTA_SCENE_SKETCH_VERSION
        or delta["contract"] != DELTA_SCENE_SKETCH_CONTRACT
        or operation not in OPERATION_TOKENS
    ):
        raise ModelScenePlanCodecError("DeltaSceneSketch contract/version mismatch")
    if list(delta["changed_field_groups"]) != _DELTA_GROUP[operation]:
        raise ModelScenePlanCodecError("DeltaSceneSketch field-group mismatch")
    for key in ("changed_source_ids", "added_source_ids", "removed_source_ids"):
        values = delta[key]
        if not isinstance(values, list) or len(values) != len(set(values)):
            raise ModelScenePlanCodecError(f"DeltaSceneSketch {key} is invalid")
        for source_id in values:
            if source_id != "scene_global":
                _source_slot(source_id)
    expected_changed = (
        ["scene_global"]
        if operation == "room_change"
        else []
        if operation in {"no_op", "remove_source"}
        else list(delta["changed_source_ids"])
    )
    if operation in {"rotate_source", "distance_source", "retime_source"}:
        if len(expected_changed) != 1:
            raise ModelScenePlanCodecError(
                "source-local DeltaSceneSketch requires exactly one owner"
            )
    elif list(delta["changed_source_ids"]) != expected_changed:
        raise ModelScenePlanCodecError("DeltaSceneSketch changed-owner mismatch")
    if operation == "remove_source":
        if len(delta["removed_source_ids"]) != 1 or delta["added_source_ids"]:
            raise ModelScenePlanCodecError(
                "remove_source DeltaSceneSketch must remove exactly one source"
            )
    elif delta["added_source_ids"] or delta["removed_source_ids"]:
        raise ModelScenePlanCodecError(
            "non-removal DeltaSceneSketch changed source inventory"
        )
    if input_sceneplan is not None or target_sceneplan is not None:
        if input_sceneplan is None or target_sceneplan is None or codec is None:
            raise ValueError("delta pair validation requires input, target, and codec")
        current = validate_p11_executor_profile(codec.project_plan(input_sceneplan))
        target = validate_p11_executor_profile(codec.project_plan(target_sceneplan))
        before_ids = {str(source["source_id"]) for source in current["sources"]}
        after_ids = {str(source["source_id"]) for source in target["sources"]}
        if sorted(after_ids - before_ids) != list(delta["added_source_ids"]):
            raise ModelScenePlanCodecError("DeltaSceneSketch added-source mismatch")
        if sorted(before_ids - after_ids) != list(delta["removed_source_ids"]):
            raise ModelScenePlanCodecError("DeltaSceneSketch removed-source mismatch")
    return copy.deepcopy(dict(delta))


def _same_plan_tokens(
    left: Mapping[str, Any], right: Mapping[str, Any], codec: ModelScenePlanCodecV3
) -> bool:
    return bool(
        torch.equal(
            codec.encode(codec.project_plan(left))["input_ids"],
            codec.encode(codec.project_plan(right))["input_ids"],
        )
    )


def derive_atomic_patch(
    input_sceneplan: Mapping[str, Any],
    target_sceneplan: Mapping[str, Any],
    delta_sketch: Mapping[str, Any],
    codec: ModelScenePlanCodecV3,
    patch_codec: ScenePlanEditPatchCodec,
) -> dict[str, Any]:
    """Recover the unique supported atomic patch from a target contract pair."""

    current = validate_p11_executor_profile(codec.project_plan(input_sceneplan))
    target = validate_p11_executor_profile(codec.project_plan(target_sceneplan))
    delta = validate_delta_scene_sketch(
        delta_sketch,
        input_sceneplan=current,
        target_sceneplan=target,
        codec=codec,
    )
    operation = str(delta["operation"])
    candidates: list[dict[str, Any]] = []
    if operation == "no_op":
        candidates = [{"operation": operation}]
    elif operation == "room_change":
        candidates = [
            {"operation": operation, "new_room": str(target["room"]["type"])}
        ]
    elif operation == "remove_source":
        candidates = [
            {
                "operation": operation,
                "source_id": str(delta["removed_source_ids"][0]),
            }
        ]
    else:
        source_id = str(delta["changed_source_ids"][0])
        if operation == "rotate_source":
            candidates = [
                {
                    "operation": operation,
                    "source_id": source_id,
                    "delta_azimuth_deg": value,
                }
                for value in (-45, 45)
            ]
        elif operation == "distance_source":
            candidates = [
                {
                    "operation": operation,
                    "source_id": source_id,
                    "distance_factor": value,
                }
                for value in (0.75, 1.25)
            ]
        else:
            target_source = next(
                source
                for source in target["sources"]
                if str(source["source_id"]) == source_id
            )
            candidates = [
                {
                    "operation": operation,
                    "source_id": source_id,
                    "new_interval_frames": [
                        _frame_from_seconds(
                            target_source["activity"]["onset_sec"],
                            max_frames=codec.max_frames,
                        ),
                        _frame_from_seconds(
                            target_source["activity"]["offset_sec"],
                            max_frames=codec.max_frames,
                        ),
                    ],
                }
            ]
    matches = [
        candidate
        for candidate in candidates
        if _same_plan_tokens(patch_codec.apply(current, candidate), target, codec)
    ]
    if len(matches) != 1:
        raise ModelScenePlanCodecError(
            f"DeltaSceneSketch did not identify one atomic patch: matches={len(matches)}"
        )
    encoded = patch_codec.encode(matches[0])["input_ids"]
    decoded = patch_codec.decode(encoded)
    patch_codec.assert_target(current, decoded, target)
    return decoded


def project_delta_thought_to_atomic_patch(
    input_sceneplan: Mapping[str, Any],
    delta_program: Mapping[str, Any],
    predicted_delta_core: Sequence[Sequence[float]] | np.ndarray | torch.Tensor,
    codec: ModelScenePlanCodecV3,
    patch_codec: ScenePlanEditPatchCodec,
) -> dict[str, Any]:
    """Project a continuous ``<DELTA_THOUGHT>`` onto one legal atomic edit.

    The discrete program owns operation, source identity, room identity,
    removal, and P10's categorical rotate/distance endpoint.  The continuous
    state owns the new activity interval and remains the executable numeric
    state for G/U.  This preserves one authority per control and a total,
    grammar-valid executor.
    """

    current = validate_p11_executor_profile(codec.project_plan(input_sceneplan))
    if delta_program.get("contract") != DELTA_SKETCH_TOKEN_CONTRACT:
        raise ModelScenePlanCodecError("wrong DeltaSketch token contract")
    operation = str(delta_program.get("operation") or "")
    if operation not in OPERATION_TOKENS:
        raise ModelScenePlanCodecError(f"unsupported delta operation {operation!r}")
    if isinstance(predicted_delta_core, torch.Tensor):
        predicted = predicted_delta_core.detach().float().cpu().numpy()
    else:
        predicted = np.asarray(predicted_delta_core, dtype=np.float32)
    if predicted.shape != (EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM):
        raise ModelScenePlanCodecError(
            f"DeltaThought core must be [{EXECUTION_SLOT_COUNT},{EXECUTION_FEATURE_DIM}]"
        )
    if not np.isfinite(predicted).all():
        raise ModelScenePlanCodecError("DeltaThought core is non-finite")
    current_state = compile_execution_state(current, codec)
    current_core = execution_state_core(current_state)

    candidates: list[dict[str, Any]]
    if operation == "no_op":
        candidates = [{"operation": "no_op"}]
    elif operation == "room_change":
        candidates = [
            {"operation": operation, "new_room": str(delta_program["new_room"])}
        ]
    elif operation == "remove_source":
        candidates = [
            {"operation": operation, "source_id": str(delta_program["source_id"])}
        ]
    elif operation == "rotate_source":
        source_id = str(delta_program["source_id"])
        candidates = [
            {
                "operation": operation,
                "source_id": source_id,
                "delta_azimuth_deg": value,
            }
            for value in (-45, 45)
        ]
    elif operation == "distance_source":
        source_id = str(delta_program["source_id"])
        candidates = [
            {
                "operation": operation,
                "source_id": source_id,
                "distance_factor": value,
            }
            for value in (0.75, 1.25)
        ]
    else:
        source_id = str(delta_program["source_id"])
        source_index = _source_slot(source_id)
        raw_target = current_core + predicted
        duration_frames = _frame_from_seconds(
            current["duration_sec"], max_frames=codec.max_frames
        )
        onset = int(
            round(
                float(
                    np.clip(
                        raw_target[
                            source_index + 1,
                            _EXECUTION_INDEX["onset_frame_norm"],
                        ],
                        0.0,
                        1.0,
                    )
                )
                * duration_frames
            )
        )
        offset = int(
            round(
                float(
                    np.clip(
                        raw_target[
                            source_index + 1,
                            _EXECUTION_INDEX["offset_frame_norm"],
                        ],
                        0.0,
                        1.0,
                    )
                )
                * duration_frames
            )
        )
        onset = max(0, min(duration_frames - 1, onset))
        offset = max(onset + 1, min(duration_frames, offset))
        current_source = next(
            source
            for source in current["sources"]
            if str(source["source_id"]) == source_id
        )
        old = [
            _frame_from_seconds(
                current_source["activity"]["onset_sec"],
                max_frames=codec.max_frames,
            ),
            _frame_from_seconds(
                current_source["activity"]["offset_sec"],
                max_frames=codec.max_frames,
            ),
        ]
        if [onset, offset] == old:
            if offset < duration_frames:
                onset, offset = onset + 1, offset + 1
            elif onset > 0:
                onset, offset = onset - 1, offset - 1
            elif offset - onset > 1:
                offset -= 1
            else:
                raise ModelScenePlanCodecError(
                    "DeltaThought cannot produce a non-trivial retime"
                )
        candidates = [
            {
                "operation": operation,
                "source_id": source_id,
                "new_interval_frames": [onset, offset],
            }
        ]

    selected_control_direction: int | None = None
    if len(candidates) == 1:
        selected = candidates[0]
    elif operation in {"rotate_source", "distance_source"}:
        selected_control_direction = int(delta_program.get("control_direction", 0))
        if selected_control_direction not in {-1, 1}:
            raise ModelScenePlanCodecError(
                f"{operation} DeltaSketch lacks a legal control_direction"
            )
        # Candidate order is frozen as negative, positive and exactly matches
        # the operation-specific token grammar.
        selected = candidates[0 if selected_control_direction < 0 else 1]
    else:
        source_index = _source_slot(str(delta_program["source_id"]))
        target_delta = predicted[source_index + 1]
        scored: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            candidate_plan = patch_codec.apply(current, candidate)
            candidate_core = execution_state_core(
                compile_execution_state(candidate_plan, codec)
            )
            candidate_delta = candidate_core[source_index + 1] - current_core[
                source_index + 1
            ]
            scored.append(
                (
                    float(np.mean(np.square(candidate_delta - target_delta))),
                    candidate,
                )
            )
        scored.sort(key=lambda item: (item[0], json.dumps(item[1], sort_keys=True)))
        selected = scored[0][1]
    tokens = patch_codec.encode(selected)["input_ids"]
    decoded = patch_codec.decode(tokens)
    target = patch_codec.apply(current, decoded)
    return {
        "patch_spec": decoded,
        "patch_tokens": tokens,
        "target_sceneplan": target,
        "target_scene_sketch": compile_scene_sketch(target, codec),
        "target_execution_state": compile_execution_state(target, codec),
        "control_direction": selected_control_direction,
    }


def assemble_atomic_patch(
    input_sceneplan: Mapping[str, Any],
    delta_sketch: Mapping[str, Any],
    target_sketch: Mapping[str, Any],
    target_execution_state: Mapping[str, Any],
    codec: ModelScenePlanCodecV3,
    patch_codec: ScenePlanEditPatchCodec,
    *,
    require_target_hash: bool = True,
) -> dict[str, Any]:
    """Assemble an E target, derive its patch, and prove exact application."""

    target = assemble_sceneplan(
        target_sketch,
        target_execution_state,
        codec,
        require_target_hash=require_target_hash,
    )
    spec = derive_atomic_patch(
        input_sceneplan, target, delta_sketch, codec, patch_codec
    )
    tokens = patch_codec.encode(spec)["input_ids"]
    applied = patch_codec.apply(input_sceneplan, tokens)
    if not _same_plan_tokens(applied, target, codec):
        raise AssertionError("deterministic atomic patch failed to reproduce target")
    return {
        "patch_spec": spec,
        "patch_tokens": tokens,
        "target_sceneplan": target,
        "applied_sceneplan": applied,
    }


__all__ = [
    "AUDIO_AWARE_DELTA_OPERATIONS",
    "AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT",
    "DELTA_SCENE_SKETCH_CONTRACT",
    "DELTA_SCENE_SKETCH_SCHEMA",
    "DELTA_SCENE_SKETCH_VERSION",
    "DELTA_SKETCH_TOKEN_CONTRACT",
    "CONTROL_DIRECTION_CONTRACT",
    "CONTROL_DIRECTION_OPERATIONS",
    "CONTROL_DIRECTION_TOKENS",
    "EXECUTION_FEATURE_DIM",
    "EXECUTION_FEATURE_NAMES",
    "EXECUTION_FEATURE_SCHEMA",
    "EXECUTION_SLOT_COUNT",
    "EXECUTION_SLOT_ROLES",
    "EXECUTION_STATE_CONTRACT",
    "EXECUTION_STATE_SCHEMA",
    "EXECUTION_STATE_VERSION",
    "SCENE_SKETCH_CODEC_NAME",
    "SCENE_SKETCH_CONTRACT",
    "SCENE_SKETCH_SCHEMA",
    "SCENE_SKETCH_VERSION",
    "AudioAwareDeltaSceneSketchCodec",
    "DeltaSceneSketchCodec",
    "SceneSketchCodec",
    "apply_audio_aware_delta_program_to_sketch",
    "apply_reliable_lexical_authority_to_sketch",
    "apply_delta_program_to_sketch",
    "assemble_atomic_patch",
    "assemble_sceneplan",
    "compile_audio_aware_delta_sketch",
    "compile_delta_scene_sketch",
    "compile_execution_state",
    "compile_p10_from_contract",
    "compile_scene_sketch",
    "control_direction_from_patch",
    "derive_atomic_patch",
    "execution_delta_core",
    "audio_aware_delta_control_mask",
    "execution_state_from_core",
    "execution_state_core",
    "project_audio_aware_delta_to_atomic_patch",
    "project_delta_thought_to_atomic_patch",
    "scene_sketch_semantic_state",
    "validate_delta_scene_sketch",
    "validate_execution_state",
    "validate_scene_sketch",
]
