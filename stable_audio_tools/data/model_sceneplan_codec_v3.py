"""Frame-aligned autoregressive ScenePlan codec for P11-v1 trials.

Version 3 projects the frozen P9 ScenePlan onto the state that P10 can
actually execute.  Time is represented by atomic VAE-frame ids, spatial
coordinates use atomic bounded bins, source count is explicit, and gain is
not predicted because P10 does not consume it.  Decoding restores a valid
model-facing ScenePlan with deterministic zero gain.

The projection is intentional and measurable: callers should compare a
prediction against :meth:`ModelScenePlanCodecV3.project_plan`, not against
the sub-frame/sub-degree source JSON.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .model_sceneplan import MODEL_SAMPLE_RATE, VAE_HOP_SAMPLES, validate_model_sceneplan
from .model_sceneplan_codec import (
    DEFAULT_TEXT_CORPUS,
    ModelScenePlanCodecError,
    iter_model_sceneplan_text_fields,
    train_model_sceneplan_text_bpe,
)
from .scene_plan import (
    LOSS_GRAMMAR,
    LOSS_MOTION,
    LOSS_ROOM,
    LOSS_SEMANTIC,
    LOSS_SPATIAL_METRIC,
    LOSS_SPEECH_CONTENT,
)


CODEC_NAME = "model_sceneplan_codec_v3"
CODEC_SCHEMA = "stable_audio_tools.model_sceneplan_codec"
CODEC_VERSION = 3
MAX_FRAMES = 432
DISTANCE_BINS = 256
DISTANCE_MIN_M = 0.1
DISTANCE_MAX_M = 50.0

SOURCE_SLOT_TOKENS = tuple(f"<source_slot_{index}>" for index in range(4))
SOURCE_COUNT_TOKENS = tuple(f"<num_sources_{count}>" for count in range(1, 5))
ROOM_TOKENS = {
    "dry": "<room_dry>",
    "moderate": "<room_moderate>",
    "reverberant": "<room_reverberant>",
    "outdoor": "<room_outdoor>",
}
KIND_TOKENS = {
    "speech": "<kind_speech>",
    "music": "<kind_music>",
    "sound": "<kind_sound>",
}
MOTION_TOKENS = {
    "static": "<motion_static>",
    "linear": "<motion_linear>",
    "keyframed": "<motion_keyframed>",
}

STRUCTURAL_TOKENS = (
    "<pad>",
    "<plan_bos>",
    "<plan_eos>",
    "<text_begin>",
    "<text_end>",
    "<duration_frames>",
    "<room>",
    *ROOM_TOKENS.values(),
    "<num_sources>",
    *SOURCE_COUNT_TOKENS,
    "<source_begin>",
    "<source_end>",
    *SOURCE_SLOT_TOKENS,
    "<kind>",
    *KIND_TOKENS.values(),
    "<description>",
    "<speaker_description>",
    "<transcript>",
    "<activity_begin>",
    "<activity_end>",
    "<onset_frame>",
    "<offset_frame>",
    "<trajectory_begin>",
    "<trajectory_end>",
    *MOTION_TOKENS.values(),
    "<position>",
    "<start>",
    "<end>",
    "<keyframe_begin>",
    "<keyframe_end>",
    "<time_frame>",
    "<azimuth_bin>",
    "<elevation_bin>",
    "<distance_bin>",
)
def _frame_tokens(max_frames: int) -> tuple[str, ...]:
    return tuple(f"<frame_{index:03d}>" for index in range(int(max_frames) + 1))


FRAME_TOKENS = _frame_tokens(MAX_FRAMES)
AZIMUTH_TOKENS = tuple(f"<azimuth_{value:+04d}>" for value in range(-180, 180))
ELEVATION_TOKENS = tuple(f"<elevation_{value:+03d}>" for value in range(-90, 91))
DISTANCE_TOKENS = tuple(f"<distance_{index:03d}>" for index in range(DISTANCE_BINS))
GRAMMAR_TOKENS = (
    *STRUCTURAL_TOKENS,
    *FRAME_TOKENS,
    *AZIMUTH_TOKENS,
    *ELEVATION_TOKENS,
    *DISTANCE_TOKENS,
)


def _grammar_tokens(max_frames: int) -> tuple[str, ...]:
    return (
        *STRUCTURAL_TOKENS,
        *_frame_tokens(max_frames),
        *AZIMUTH_TOKENS,
        *ELEVATION_TOKENS,
        *DISTANCE_TOKENS,
    )


class _NeedToken(Exception):
    def __init__(self, allowed: set[int]):
        super().__init__()
        self.allowed = allowed


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _seconds(frame: int) -> float:
    # Keep the exact Python float produced by the same arithmetic used by
    # ``compile_model_44_controls``.  Decimal rounding can move an activity
    # boundary microscopically across a frame edge and change its event mask.
    return float(int(frame) * VAE_HOP_SAMPLES / MODEL_SAMPLE_RATE)


def _layout(
    text_vocab_size: int,
    vocab_size: int,
    *,
    max_frames: int = MAX_FRAMES,
) -> dict[str, Any]:
    frame_tokens = _frame_tokens(max_frames)
    grammar_tokens = _grammar_tokens(max_frames)
    required = len(grammar_tokens) + int(text_vocab_size)
    if required > int(vocab_size):
        raise ModelScenePlanCodecError(
            f"v3 codec requires {required} ids but vocab_size={vocab_size}"
        )
    return {
        "structural_tokens": list(STRUCTURAL_TOKENS),
        "grammar_tokens": list(grammar_tokens),
        "frame_token_offset": len(STRUCTURAL_TOKENS),
        "frame_token_count": len(frame_tokens),
        "azimuth_token_offset": len(STRUCTURAL_TOKENS) + len(frame_tokens),
        "azimuth_token_count": len(AZIMUTH_TOKENS),
        "elevation_token_offset": (
            len(STRUCTURAL_TOKENS) + len(frame_tokens) + len(AZIMUTH_TOKENS)
        ),
        "elevation_token_count": len(ELEVATION_TOKENS),
        "distance_token_offset": (
            len(STRUCTURAL_TOKENS)
            + len(frame_tokens)
            + len(AZIMUTH_TOKENS)
            + len(ELEVATION_TOKENS)
        ),
        "distance_token_count": len(DISTANCE_TOKENS),
        "text_piece_offset": len(grammar_tokens),
        "text_vocab_size": int(text_vocab_size),
        "used_vocab_size": required,
        "vocab_size": int(vocab_size),
        "sample_rate": MODEL_SAMPLE_RATE,
        "vae_hop_samples": VAE_HOP_SAMPLES,
        "max_frames": int(max_frames),
        "distance_quantizer": {
            "type": "log_uniform",
            "minimum_m": DISTANCE_MIN_M,
            "maximum_m": DISTANCE_MAX_M,
            "bins": DISTANCE_BINS,
        },
        "gain_policy": "decode_constant_0db_not_predicted",
        "projection_contract": "p10_latent_grid_v1",
    }


def _create_model_sceneplan_frame_codec_artifact(
    output_dir: str | os.PathLike[str],
    texts: Iterable[str] = DEFAULT_TEXT_CORPUS,
    *,
    codec_name: str,
    codec_version: int,
    max_frames: int,
    vocab_size: int = 4096,
    text_vocab_size: int = 1536,
    overwrite: bool = False,
) -> Path:
    """Create one immutable frame-token codec using an atomic final rename."""

    if int(max_frames) <= 0:
        raise ValueError("max_frames must be positive")

    output = Path(output_dir).expanduser().resolve()
    if (output / "READY").is_file() and not overwrite:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        model_bytes = train_model_sceneplan_text_bpe(
            texts, text_vocab_size=text_vocab_size
        )
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor(model_proto=model_bytes)
        details = {
            "schema": CODEC_SCHEMA,
            "schema_version": int(codec_version),
            "codec_name": str(codec_name),
            "sentencepiece_sha256": _sha256(model_bytes),
            "text_backend": "sentencepiece_bpe_byte_fallback",
            **_layout(
                processor.get_piece_size(),
                int(vocab_size),
                max_frames=int(max_frames),
            ),
        }
        canonical = json.dumps(
            details, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        details["codec_fingerprint"] = _sha256(canonical)
        (temporary / "sentencepiece.model").write_bytes(model_bytes)
        (temporary / "codec.json").write_text(
            json.dumps(details, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / "READY").write_text(
            json.dumps(
                {
                    "schema": CODEC_SCHEMA,
                    "schema_version": int(codec_version),
                    "codec_name": str(codec_name),
                    "codec_fingerprint": details["codec_fingerprint"],
                    "vocab_size": int(vocab_size),
                    "used_vocab_size": int(details["used_vocab_size"]),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if output.exists():
            if not overwrite:
                raise FileExistsError(output)
            shutil.rmtree(output)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def create_model_sceneplan_codec_v3_artifact(
    output_dir: str | os.PathLike[str],
    texts: Iterable[str] = DEFAULT_TEXT_CORPUS,
    *,
    vocab_size: int = 4096,
    text_vocab_size: int = 1536,
    overwrite: bool = False,
) -> Path:
    """Create one immutable 432-frame v3 artifact."""

    return _create_model_sceneplan_frame_codec_artifact(
        output_dir,
        texts,
        codec_name=CODEC_NAME,
        codec_version=CODEC_VERSION,
        max_frames=MAX_FRAMES,
        vocab_size=vocab_size,
        text_vocab_size=text_vocab_size,
        overwrite=overwrite,
    )


class ModelScenePlanCodecV3:
    """Encode, decode, project, and grammar-constrain P11 v3 plans."""

    codec_name = CODEC_NAME
    codec_version = CODEC_VERSION
    max_frames = MAX_FRAMES
    codec_label = "v3"

    def __init__(self, artifact_dir: str | os.PathLike[str]):
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        details_path = self.artifact_dir / "codec.json"
        model_path = self.artifact_dir / "sentencepiece.model"
        if not details_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"incomplete P11 codec: {self.artifact_dir}")
        self.details = json.loads(details_path.read_text(encoding="utf-8"))
        expected_layout = _layout(
            int(self.details.get("text_vocab_size", -1)),
            int(self.details.get("vocab_size", -1)),
            max_frames=self.max_frames,
        )
        if (
            self.details.get("schema") != CODEC_SCHEMA
            or self.details.get("codec_name") != self.codec_name
            or int(self.details.get("schema_version", -1)) != self.codec_version
            or any(self.details.get(key) != value for key, value in expected_layout.items())
        ):
            raise ModelScenePlanCodecError(
                f"P11 {self.codec_label} codec schema/grammar mismatch"
            )
        model_bytes = model_path.read_bytes()
        if _sha256(model_bytes) != self.details.get("sentencepiece_sha256"):
            raise ModelScenePlanCodecError(
                f"P11 {self.codec_label} SentencePiece checksum mismatch"
            )
        import sentencepiece as spm

        self.text_processor = spm.SentencePieceProcessor(model_proto=model_bytes)
        grammar_tokens = _grammar_tokens(self.max_frames)
        frame_tokens = _frame_tokens(self.max_frames)
        self.token_to_id = {token: index for index, token in enumerate(grammar_tokens)}
        self.text_offset = int(self.details["text_piece_offset"])
        self.text_vocab_size = int(self.details["text_vocab_size"])
        self.vocab_size = int(self.details["vocab_size"])
        if self.text_processor.get_piece_size() != self.text_vocab_size:
            raise ModelScenePlanCodecError(
                f"P11 {self.codec_label} text vocabulary mismatch"
            )
        self.text_ids = set(range(self.text_offset, self.text_offset + self.text_vocab_size))
        self.frame_ids = tuple(self._tid(token) for token in frame_tokens)
        self.azimuth_ids = tuple(self._tid(token) for token in AZIMUTH_TOKENS)
        self.elevation_ids = tuple(self._tid(token) for token in ELEVATION_TOKENS)
        self.distance_ids = tuple(self._tid(token) for token in DISTANCE_TOKENS)

    @property
    def bos_id(self) -> int:
        return self._tid("<plan_bos>")

    @property
    def eos_id(self) -> int:
        return self._tid("<plan_eos>")

    @property
    def pad_id(self) -> int:
        return self._tid("<pad>")

    @property
    def fingerprint(self) -> str:
        return str(self.details["codec_fingerprint"])

    def _tid(self, token: str) -> int:
        return self.token_to_id[token]

    @staticmethod
    def _frame_from_seconds(value: Any, *, mode: str) -> int:
        raw = float(value) * MODEL_SAMPLE_RATE / VAE_HOP_SAMPLES
        if mode == "ceil":
            # ``_seconds`` is persisted to nanoseconds.  Converting that
            # repeating decimal back to frames can land a few 1e-8 above an
            # integer, so use a tolerance far below one audio sample.
            return int(math.ceil(raw - 1e-6))
        if mode == "floor":
            return int(math.floor(raw + 1e-6))
        if mode == "nearest":
            return int(round(raw))
        raise ValueError(mode)

    def _duration_frame(self, value: Any) -> int:
        samples = int(round(float(value) * MODEL_SAMPLE_RATE))
        return max(1, min(self.max_frames, math.ceil(samples / VAE_HOP_SAMPLES)))

    @staticmethod
    def _azimuth_value(value: Any) -> int:
        rounded = int(round(float(value)))
        return ((rounded + 180) % 360) - 180

    @staticmethod
    def _elevation_value(value: Any) -> int:
        return max(-90, min(90, int(round(float(value)))))

    @staticmethod
    def _distance_index(value: Any) -> int:
        distance = max(DISTANCE_MIN_M, min(DISTANCE_MAX_M, float(value)))
        fraction = (
            (math.log(distance) - math.log(DISTANCE_MIN_M))
            / (math.log(DISTANCE_MAX_M) - math.log(DISTANCE_MIN_M))
        )
        return max(0, min(DISTANCE_BINS - 1, int(round(fraction * (DISTANCE_BINS - 1)))))

    @staticmethod
    def _distance_value(index: int) -> float:
        fraction = int(index) / (DISTANCE_BINS - 1)
        value = math.exp(
            math.log(DISTANCE_MIN_M)
            + fraction * (math.log(DISTANCE_MAX_M) - math.log(DISTANCE_MIN_M))
        )
        return float(round(value, 6))

    def snap_numeric_to_grid(self, kind: str, raw: Any) -> float:
        """Compatibility helper used by the external P10 handoff."""

        if kind == "seconds":
            frame = max(
                0,
                min(self.max_frames, self._frame_from_seconds(raw, mode="nearest")),
            )
            return _seconds(frame)
        if kind == "azimuth":
            return float(self._azimuth_value(raw))
        if kind == "elevation":
            return float(self._elevation_value(raw))
        if kind == "distance":
            return self._distance_value(self._distance_index(raw))
        if kind == "gain_db":
            return 0.0
        raise ModelScenePlanCodecError(f"unknown v3 numeric field {kind!r}")

    def _project_position(self, value: Mapping[str, Any]) -> dict[str, float]:
        return {
            "azimuth_deg": float(self._azimuth_value(value["azimuth_deg"])),
            "elevation_deg": float(self._elevation_value(value["elevation_deg"])),
            "distance_m": self._distance_value(self._distance_index(value["distance_m"])),
        }

    def project_plan(
        self, plan: Mapping[str, Any], *, sample_id: str | None = None
    ) -> dict[str, Any]:
        """Project a frozen P9 plan onto P10's executable latent grid."""

        source_plan = validate_model_sceneplan(plan)
        output = copy.deepcopy(source_plan)
        if sample_id is not None:
            output["sample_id"] = str(sample_id)
        # P9 durations are sample-aligned.  Recover that integer first so an
        # almost-exact hop boundary cannot acquire an extra latent frame.
        duration_frame = self._duration_frame(source_plan["duration_sec"])
        output["duration_sec"] = _seconds(duration_frame)
        for source in output["sources"]:
            original_activity = source["activity"]
            onset_frame = max(
                0,
                min(
                    duration_frame - 1,
                    self._frame_from_seconds(original_activity["onset_sec"], mode="floor"),
                ),
            )
            offset_frame = max(
                onset_frame + 1,
                min(
                    duration_frame,
                    self._frame_from_seconds(original_activity["offset_sec"], mode="ceil"),
                ),
            )
            source["activity"] = {
                "onset_sec": _seconds(onset_frame),
                "offset_sec": _seconds(offset_frame),
            }
            trajectory = source["trajectory"]
            motion = str(trajectory["type"])
            if motion == "static":
                trajectory["position"] = self._project_position(trajectory["position"])
            elif motion == "linear":
                trajectory["start"] = self._project_position(trajectory["start"])
                trajectory["end"] = self._project_position(trajectory["end"])
            else:
                projected: list[dict[str, Any]] = []
                for keyframe in trajectory["keyframes"]:
                    frame = max(
                        onset_frame,
                        min(
                            offset_frame,
                            self._frame_from_seconds(keyframe["time_sec"], mode="nearest"),
                        ),
                    )
                    item = {
                        "time_sec": _seconds(frame),
                        "position": self._project_position(keyframe["position"]),
                    }
                    if projected and item["time_sec"] <= projected[-1]["time_sec"]:
                        projected[-1] = item
                    else:
                        projected.append(item)
                if len(projected) < 2:
                    first = self._project_position(trajectory["keyframes"][0]["position"])
                    last = self._project_position(trajectory["keyframes"][-1]["position"])
                    trajectory.clear()
                    trajectory.update({"type": "linear", "start": first, "end": last})
                else:
                    trajectory["keyframes"] = projected[:8]
            source["gain_db"] = 0.0
        return validate_model_sceneplan(output)

    def _text(self, value: Any) -> list[int]:
        text = " ".join(str(value or "").split())
        if not text:
            raise ModelScenePlanCodecError("ScenePlan text fields must be non-empty")
        pieces = self.text_processor.encode(text, out_type=int)
        if not pieces:
            raise ModelScenePlanCodecError("SentencePiece produced an empty field")
        return [
            self._tid("<text_begin>"),
            *(self.text_offset + int(piece) for piece in pieces),
            self._tid("<text_end>"),
        ]

    def encode(
        self, plan: Mapping[str, Any], *, max_tokens: int | None = None
    ) -> dict[str, torch.Tensor]:
        plan = self.project_plan(plan)
        ids: list[int] = []
        groups: list[int] = []

        def emit(token: str, group: int = LOSS_GRAMMAR) -> None:
            ids.append(self._tid(token))
            groups.append(int(group))

        def atomic(token: str, value_id: int, group: int) -> None:
            emit(token, group)
            ids.append(int(value_id))
            groups.append(int(group))

        def text_field(token: str, value: Any, group: int) -> None:
            emit(token, group)
            encoded = self._text(value)
            ids.extend(encoded)
            groups.extend([int(group)] * len(encoded))

        def frame_field(token: str, seconds: Any, group: int) -> None:
            frame = self._frame_from_seconds(seconds, mode="nearest")
            atomic(token, self.frame_ids[frame], group)

        def position(token: str, value: Mapping[str, Any]) -> None:
            emit(token, LOSS_MOTION)
            atomic(
                "<azimuth_bin>",
                self.azimuth_ids[self._azimuth_value(value["azimuth_deg"]) + 180],
                LOSS_SPATIAL_METRIC,
            )
            atomic(
                "<elevation_bin>",
                self.elevation_ids[self._elevation_value(value["elevation_deg"]) + 90],
                LOSS_SPATIAL_METRIC,
            )
            atomic(
                "<distance_bin>",
                self.distance_ids[self._distance_index(value["distance_m"])],
                LOSS_SPATIAL_METRIC,
            )

        emit("<plan_bos>")
        frame_field("<duration_frames>", plan["duration_sec"], LOSS_GRAMMAR)
        emit("<room>", LOSS_ROOM)
        emit(ROOM_TOKENS[str(plan["room"]["type"])], LOSS_ROOM)
        emit("<num_sources>")
        emit(SOURCE_COUNT_TOKENS[len(plan["sources"]) - 1])
        for source in plan["sources"]:
            slot = int(str(source["source_id"])[7:])
            kind = str(source["kind"])
            emit("<source_begin>")
            emit(SOURCE_SLOT_TOKENS[slot], LOSS_SEMANTIC)
            emit("<kind>", LOSS_SEMANTIC)
            emit(KIND_TOKENS[kind], LOSS_SEMANTIC)
            if kind == "speech":
                text_field("<speaker_description>", source["speaker_description"], LOSS_SEMANTIC)
                text_field("<transcript>", source["transcript"], LOSS_SPEECH_CONTENT)
            else:
                text_field("<description>", source["description"], LOSS_SEMANTIC)
            emit("<activity_begin>", LOSS_MOTION)
            frame_field("<onset_frame>", source["activity"]["onset_sec"], LOSS_MOTION)
            frame_field("<offset_frame>", source["activity"]["offset_sec"], LOSS_MOTION)
            emit("<activity_end>", LOSS_MOTION)
            emit("<trajectory_begin>", LOSS_MOTION)
            motion = str(source["trajectory"]["type"])
            emit(MOTION_TOKENS[motion], LOSS_MOTION)
            trajectory = source["trajectory"]
            if motion == "static":
                position("<position>", trajectory["position"])
            elif motion == "linear":
                position("<start>", trajectory["start"])
                position("<end>", trajectory["end"])
            else:
                for keyframe in trajectory["keyframes"]:
                    emit("<keyframe_begin>", LOSS_MOTION)
                    frame_field("<time_frame>", keyframe["time_sec"], LOSS_MOTION)
                    position("<position>", keyframe["position"])
                    emit("<keyframe_end>", LOSS_MOTION)
            emit("<trajectory_end>", LOSS_MOTION)
            emit("<source_end>")
        emit("<plan_eos>")
        if max_tokens is not None and len(ids) > int(max_tokens):
            raise ModelScenePlanCodecError(
                f"ScenePlan requires {len(ids)} tokens > max_tokens={max_tokens}"
            )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.bool),
            "loss_group_ids": torch.tensor(groups, dtype=torch.long),
        }

    class _Decoder:
        def __init__(self, codec: "ModelScenePlanCodecV3", values: Sequence[int]):
            self.codec = codec
            self.values = [int(value) for value in values]
            self.index = 0

        def take(self) -> int:
            if self.index >= len(self.values):
                raise ModelScenePlanCodecError("unexpected end of ScenePlan tokens")
            value = self.values[self.index]
            self.index += 1
            return value

        def expect(self, token: str) -> None:
            expected = self.codec._tid(token)
            found = self.take()
            if found != expected:
                raise ModelScenePlanCodecError(f"expected {token} ({expected}), found {found}")

        def frame(self) -> int:
            value = self.take()
            try:
                return self.codec.frame_ids.index(value)
            except ValueError as error:
                raise ModelScenePlanCodecError("invalid frame token") from error

        def text(self) -> str:
            self.expect("<text_begin>")
            pieces: list[int] = []
            while True:
                token = self.take()
                if token == self.codec._tid("<text_end>"):
                    break
                piece = token - self.codec.text_offset
                if not 0 <= piece < self.codec.text_vocab_size:
                    raise ModelScenePlanCodecError(f"invalid text token {token}")
                pieces.append(piece)
            if not pieces:
                raise ModelScenePlanCodecError("empty ScenePlan text field")
            value = self.codec.text_processor.decode(pieces)
            if not value.strip():
                raise ModelScenePlanCodecError("decoded empty ScenePlan text")
            return value

    def decode(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        sample_id: str = "generated",
    ) -> dict[str, Any]:
        values = (
            token_ids.detach().cpu().flatten().tolist()
            if isinstance(token_ids, torch.Tensor)
            else list(token_ids)
        )
        cur = self._Decoder(self, values)
        cur.expect("<plan_bos>")
        cur.expect("<duration_frames>")
        duration_frame = cur.frame()
        if not 1 <= duration_frame <= self.max_frames:
            raise ModelScenePlanCodecError(
                f"duration frame is outside [1,{self.max_frames}]"
            )
        cur.expect("<room>")
        room_id = cur.take()
        inverse_rooms = {self._tid(token): key for key, token in ROOM_TOKENS.items()}
        if room_id not in inverse_rooms:
            raise ModelScenePlanCodecError("invalid room token")
        cur.expect("<num_sources>")
        count_id = cur.take()
        count_lookup = {
            self._tid(token): count for count, token in enumerate(SOURCE_COUNT_TOKENS, 1)
        }
        if count_id not in count_lookup:
            raise ModelScenePlanCodecError("invalid source-count token")
        expected_sources = count_lookup[count_id]
        sources: list[dict[str, Any]] = []
        seen_speech = False

        def read_position(token: str) -> dict[str, float]:
            cur.expect(token)
            cur.expect("<azimuth_bin>")
            azimuth_id = cur.take()
            cur.expect("<elevation_bin>")
            elevation_id = cur.take()
            cur.expect("<distance_bin>")
            distance_id = cur.take()
            try:
                azimuth = self.azimuth_ids.index(azimuth_id) - 180
                elevation = self.elevation_ids.index(elevation_id) - 90
                distance = self._distance_value(self.distance_ids.index(distance_id))
            except ValueError as error:
                raise ModelScenePlanCodecError("invalid position bin token") from error
            return {
                "azimuth_deg": float(azimuth),
                "elevation_deg": float(elevation),
                "distance_m": distance,
            }

        for _ in range(expected_sources):
            cur.expect("<source_begin>")
            slot_id = cur.take()
            slot_lookup = {self._tid(token): index for index, token in enumerate(SOURCE_SLOT_TOKENS)}
            if slot_id not in slot_lookup:
                raise ModelScenePlanCodecError("invalid source slot token")
            slot = slot_lookup[slot_id]
            if sources and slot <= int(sources[-1]["source_id"][7:]):
                raise ModelScenePlanCodecError("source slots are not strictly ordered")
            cur.expect("<kind>")
            kind_id = cur.take()
            kind_lookup = {self._tid(token): key for key, token in KIND_TOKENS.items()}
            if kind_id not in kind_lookup:
                raise ModelScenePlanCodecError("invalid source kind token")
            kind = kind_lookup[kind_id]
            source: dict[str, Any] = {"source_id": f"source_{slot}", "kind": kind}
            if kind == "speech":
                if seen_speech:
                    raise ModelScenePlanCodecError("more than one formal speech source")
                seen_speech = True
                cur.expect("<speaker_description>")
                source["speaker_description"] = cur.text()
                cur.expect("<transcript>")
                source["transcript"] = cur.text()
            else:
                cur.expect("<description>")
                source["description"] = cur.text()
            cur.expect("<activity_begin>")
            cur.expect("<onset_frame>")
            onset_frame = cur.frame()
            cur.expect("<offset_frame>")
            offset_frame = cur.frame()
            if not 0 <= onset_frame < offset_frame <= duration_frame:
                raise ModelScenePlanCodecError("activity frames are outside duration")
            cur.expect("<activity_end>")
            source["activity"] = {
                "onset_sec": _seconds(onset_frame),
                "offset_sec": _seconds(offset_frame),
            }
            cur.expect("<trajectory_begin>")
            motion_id = cur.take()
            motion_lookup = {self._tid(token): key for key, token in MOTION_TOKENS.items()}
            if motion_id not in motion_lookup:
                raise ModelScenePlanCodecError("invalid motion token")
            motion = motion_lookup[motion_id]
            if motion == "static":
                trajectory: dict[str, Any] = {"type": motion, "position": read_position("<position>")}
            elif motion == "linear":
                trajectory = {
                    "type": motion,
                    "start": read_position("<start>"),
                    "end": read_position("<end>"),
                }
            else:
                keyframes = []
                while cur.index < len(cur.values) and cur.values[cur.index] == self._tid("<keyframe_begin>"):
                    cur.expect("<keyframe_begin>")
                    cur.expect("<time_frame>")
                    frame = cur.frame()
                    position = read_position("<position>")
                    cur.expect("<keyframe_end>")
                    keyframes.append({"time_sec": _seconds(frame), "position": position})
                if not 2 <= len(keyframes) <= 8:
                    raise ModelScenePlanCodecError("keyframed motion requires 2--8 keyframes")
                trajectory = {"type": motion, "keyframes": keyframes}
            cur.expect("<trajectory_end>")
            source["trajectory"] = trajectory
            source["gain_db"] = 0.0
            cur.expect("<source_end>")
            sources.append(source)
        cur.expect("<plan_eos>")
        if cur.index != len(cur.values):
            raise ModelScenePlanCodecError("tokens remain after <plan_eos>")
        return validate_model_sceneplan(
            {
                "sample_id": str(sample_id),
                "duration_sec": _seconds(duration_frame),
                "room": {"type": inverse_rooms[room_id]},
                "sources": sources,
            }
        )

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
        fixed_duration_sec: float | None = None,
    ) -> set[int]:
        """Return exact grammar-allowed ids after one v3 prefix."""

        if not 1 <= int(min_sources) <= int(max_sources) <= 4:
            raise ValueError("source limits must satisfy 1 <= min <= max <= 4")
        values = (
            prefix.detach().cpu().flatten().tolist()
            if isinstance(prefix, torch.Tensor)
            else [int(value) for value in prefix]
        )
        index = 0

        def choice(allowed: set[int], label: str) -> int:
            nonlocal index
            if index >= len(values):
                raise _NeedToken(set(allowed))
            value = values[index]
            if value not in allowed:
                raise ModelScenePlanCodecError(
                    f"invalid prefix at {index}: expected {label}, found {value}"
                )
            index += 1
            return value

        def exact(token: str) -> None:
            choice({self._tid(token)}, token)

        def frame_value(*, minimum: int = 0, maximum: int | None = None) -> int:
            if maximum is None:
                maximum = self.max_frames
            if minimum > maximum:
                raise ModelScenePlanCodecError(
                    f"empty frame range in {self.codec_label} grammar"
                )
            token = choice(set(self.frame_ids[minimum : maximum + 1]), "frame")
            return self.frame_ids.index(token)

        def text_value() -> None:
            exact("<text_begin>")
            emitted = False
            while True:
                allowed = set(self.text_ids)
                if emitted:
                    allowed.add(self._tid("<text_end>"))
                value = choice(allowed, "text piece or <text_end>")
                if value == self._tid("<text_end>"):
                    return
                emitted = True

        def text_field(token: str) -> None:
            exact(token)
            text_value()

        def position(token: str) -> None:
            exact(token)
            exact("<azimuth_bin>")
            choice(set(self.azimuth_ids), "azimuth bin")
            exact("<elevation_bin>")
            choice(set(self.elevation_ids), "elevation bin")
            exact("<distance_bin>")
            choice(set(self.distance_ids), "distance bin")

        try:
            exact("<plan_bos>")
            exact("<duration_frames>")
            if fixed_duration_sec is None:
                duration_frame = frame_value(minimum=1, maximum=self.max_frames)
            else:
                duration_frame = self._duration_frame(fixed_duration_sec)
                choice({self.frame_ids[duration_frame]}, "fixed duration frame")
            exact("<room>")
            choice({self._tid(token) for token in ROOM_TOKENS.values()}, "room")
            exact("<num_sources>")
            source_count_id = choice(
                {
                    self._tid(SOURCE_COUNT_TOKENS[count - 1])
                    for count in range(int(min_sources), int(max_sources) + 1)
                },
                "source count",
            )
            source_count = next(
                count
                for count, token in enumerate(SOURCE_COUNT_TOKENS, 1)
                if source_count_id == self._tid(token)
            )
            last_slot = -1
            speech_seen = False
            for source_index in range(source_count):
                exact("<source_begin>")
                remaining = source_count - source_index - 1
                maximum_slot = 3 - remaining
                slot_options = {
                    self._tid(SOURCE_SLOT_TOKENS[slot])
                    for slot in range(last_slot + 1, maximum_slot + 1)
                }
                slot_id = choice(slot_options, "increasing source slot")
                last_slot = next(
                    slot
                    for slot, token in enumerate(SOURCE_SLOT_TOKENS)
                    if slot_id == self._tid(token)
                )
                exact("<kind>")
                kind_options = {
                    self._tid(KIND_TOKENS["music"]),
                    self._tid(KIND_TOKENS["sound"]),
                }
                if not speech_seen:
                    kind_options.add(self._tid(KIND_TOKENS["speech"]))
                kind_id = choice(kind_options, "source kind")
                if kind_id == self._tid(KIND_TOKENS["speech"]):
                    speech_seen = True
                    text_field("<speaker_description>")
                    text_field("<transcript>")
                else:
                    text_field("<description>")
                exact("<activity_begin>")
                exact("<onset_frame>")
                onset_frame = frame_value(minimum=0, maximum=duration_frame - 1)
                exact("<offset_frame>")
                offset_frame = frame_value(minimum=onset_frame + 1, maximum=duration_frame)
                exact("<activity_end>")
                exact("<trajectory_begin>")
                motion_id = choice(
                    {self._tid(token) for token in MOTION_TOKENS.values()},
                    "motion type",
                )
                if motion_id == self._tid(MOTION_TOKENS["static"]):
                    position("<position>")
                elif motion_id == self._tid(MOTION_TOKENS["linear"]):
                    position("<start>")
                    position("<end>")
                else:
                    keyframes = 0
                    previous = onset_frame - 1
                    while True:
                        allowed: set[int] = set()
                        if keyframes < 8 and previous < offset_frame:
                            allowed.add(self._tid("<keyframe_begin>"))
                        if keyframes >= 2:
                            allowed.add(self._tid("<trajectory_end>"))
                        selected = choice(allowed, "keyframe or trajectory end")
                        if selected == self._tid("<trajectory_end>"):
                            break
                        exact("<time_frame>")
                        previous = frame_value(
                            minimum=max(onset_frame, previous + 1),
                            maximum=offset_frame,
                        )
                        position("<position>")
                        exact("<keyframe_end>")
                        keyframes += 1
                if motion_id != self._tid(MOTION_TOKENS["keyframed"]):
                    exact("<trajectory_end>")
                exact("<source_end>")
            exact("<plan_eos>")
            if index != len(values):
                raise ModelScenePlanCodecError("tokens remain after valid prefix")
            return set()
        except _NeedToken as need:
            return need.allowed


__all__ = [
    "CODEC_NAME",
    "CODEC_SCHEMA",
    "CODEC_VERSION",
    "ModelScenePlanCodecV3",
    "create_model_sceneplan_codec_v3_artifact",
    "iter_model_sceneplan_text_fields",
]
