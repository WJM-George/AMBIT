"""Compact autoregressive codec for the P10 model-facing ScenePlan.

The P11 planner predicts the *same* compact ScenePlan consumed by P10.  Asset
references and renderer provenance are intentionally absent.  Grammar and
metric values use atomic ids, while audible source descriptions and exact
transcripts use a byte-fallback SentencePiece vocabulary.  The codec is fully
reversible up to the declared metric quantization and exposes an exact prefix
constraint for inference.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from .model_sceneplan import validate_model_sceneplan
from .scene_plan import (
    LOSS_GRAMMAR,
    LOSS_MOTION,
    LOSS_ROOM,
    LOSS_SEMANTIC,
    LOSS_SPATIAL_METRIC,
    LOSS_SPEECH_CONTENT,
)


CODEC_NAME = "model_sceneplan_codec_v2"
CODEC_SCHEMA = "stable_audio_tools.model_sceneplan_codec"
CODEC_VERSION = 2

SOURCE_SLOT_TOKENS = tuple(f"<source_slot_{index}>" for index in range(4))
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

# Numeric payload ids are shared by every metric field.  Three big-endian
# byte tokens are enough for the frozen fixed-point ranges below, while
# avoiding the lossy one-token grids used by the retired v1 codec.
BYTE_TOKENS = tuple(f"<byte_{value:03d}>" for value in range(256))

# Token order is checkpoint state.  Additions require a new codec version.
STRUCTURAL_TOKENS = (
    "<pad>",
    "<plan_bos>",
    "<plan_eos>",
    "<text_begin>",
    "<text_end>",
    "<duration>",
    "<room>",
    *ROOM_TOKENS.values(),
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
    "<onset>",
    "<offset>",
    "<trajectory_begin>",
    "<trajectory_end>",
    *MOTION_TOKENS.values(),
    "<position>",
    "<start>",
    "<end>",
    "<keyframe_begin>",
    "<keyframe_end>",
    "<time>",
    "<azimuth>",
    "<elevation>",
    "<distance>",
    "<gain_db>",
    *BYTE_TOKENS,
)

# These fixed-point scales are the native precision of the frozen P9 JSONL:
# seconds are stored to microseconds, angles to 0.001 degree, distance to
# 0.0001 m, and gain to 0.0001 dB.  Each value is serialized with exactly
# three shared byte tokens.  Encoding fails rather than silently quantizing a
# value that exceeds the declared precision.
NUMERIC_SPECS = (
    ("seconds", 0.0, 10.031021, 1_000_000, 3),
    ("azimuth", -180.0, 180.0, 1_000, 3),
    ("elevation", -90.0, 90.0, 1_000, 3),
    ("distance", 0.0001, 50.0, 10_000, 3),
    ("gain_db", -24.0, 12.0, 10_000, 3),
)

DEFAULT_TEXT_CORPUS = (
    "adult narrator with a calm clear voice",
    "engine accelerating with a bright mechanical whine",
    "electronic music with pulsing synthesizers and a driving beat",
    "speech music sound source",
    "hello world",
)


class ModelScenePlanCodecError(ValueError):
    """Raised when a codec artifact, prefix, or ScenePlan is invalid."""


class _NeedToken(Exception):
    def __init__(self, allowed: set[int]):
        super().__init__()
        self.allowed = allowed


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _layout(text_vocab_size: int, vocab_size: int) -> dict[str, Any]:
    cursor = len(STRUCTURAL_TOKENS)
    numeric: dict[str, dict[str, Any]] = {}
    for name, minimum, maximum, scale, num_bytes in NUMERIC_SPECS:
        max_index = int(round((maximum - minimum) * scale))
        if max_index >= 256 ** int(num_bytes):
            raise ModelScenePlanCodecError(
                f"{name} fixed-point range does not fit in {num_bytes} bytes"
            )
        numeric[name] = {
            "minimum": minimum,
            "maximum": maximum,
            "scale": int(scale),
            "num_bytes": int(num_bytes),
            "max_index": max_index,
        }
    required = cursor + int(text_vocab_size)
    if required > int(vocab_size):
        raise ModelScenePlanCodecError(
            f"codec requires {required} ids but vocab_size={vocab_size}"
        )
    return {
        "structural_tokens": list(STRUCTURAL_TOKENS),
        "numeric": numeric,
        "byte_token_offset": STRUCTURAL_TOKENS.index(BYTE_TOKENS[0]),
        "byte_token_count": len(BYTE_TOKENS),
        "text_piece_offset": cursor,
        "text_vocab_size": int(text_vocab_size),
        "used_vocab_size": required,
        "vocab_size": int(vocab_size),
    }


def train_model_sceneplan_text_bpe(
    texts: Iterable[str], *, text_vocab_size: int = 1536
) -> bytes:
    """Train a deterministic byte-fallback BPE for audible text fields."""

    import sentencepiece as spm

    def normalized() -> Iterator[str]:
        emitted = False
        for value in texts:
            if isinstance(value, str) and value.strip():
                emitted = True
                yield value.replace("\r", " ").replace("\n", " ")
        if not emitted:
            yield from DEFAULT_TEXT_CORPUS

    writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=normalized(),
        model_writer=writer,
        model_type="bpe",
        vocab_size=int(text_vocab_size),
        hard_vocab_limit=False,
        byte_fallback=True,
        character_coverage=1.0,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        split_by_whitespace=True,
        shuffle_input_sentence=False,
        input_sentence_size=0,
        num_threads=1,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        minloglevel=2,
    )
    return writer.getvalue()


def create_model_sceneplan_codec_artifact(
    output_dir: str | os.PathLike[str],
    texts: Iterable[str] = DEFAULT_TEXT_CORPUS,
    *,
    vocab_size: int = 4096,
    text_vocab_size: int = 1536,
    overwrite: bool = False,
) -> Path:
    """Create one immutable codec artifact with an atomic final rename."""

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
            "schema_version": CODEC_VERSION,
            "codec_name": CODEC_NAME,
            "sentencepiece_sha256": _sha256(model_bytes),
            "text_backend": "sentencepiece_bpe_byte_fallback",
            **_layout(processor.get_piece_size(), int(vocab_size)),
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
                    "schema_version": CODEC_VERSION,
                    "codec_name": CODEC_NAME,
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


def iter_model_sceneplan_text_fields(plan: Mapping[str, Any]) -> Iterator[str]:
    """Yield only the semantic strings predicted by P11."""

    validate_model_sceneplan(plan)
    for source in plan["sources"]:
        if source["kind"] == "speech":
            yield str(source["speaker_description"])
            yield str(source["transcript"])
        else:
            yield str(source["description"])


class ModelScenePlanCodec:
    """Encode/decode compact P10 ScenePlans and constrain P11 decoding."""

    def __init__(self, artifact_dir: str | os.PathLike[str]):
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        details_path = self.artifact_dir / "codec.json"
        model_path = self.artifact_dir / "sentencepiece.model"
        if not details_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(f"incomplete P11 codec: {self.artifact_dir}")
        self.details = json.loads(details_path.read_text(encoding="utf-8"))
        if (
            self.details.get("schema") != CODEC_SCHEMA
            or self.details.get("codec_name") != CODEC_NAME
            or int(self.details.get("schema_version", -1)) != CODEC_VERSION
            or self.details.get("structural_tokens") != list(STRUCTURAL_TOKENS)
        ):
            raise ModelScenePlanCodecError("P11 codec schema/grammar mismatch")
        model_bytes = model_path.read_bytes()
        if _sha256(model_bytes) != self.details.get("sentencepiece_sha256"):
            raise ModelScenePlanCodecError("P11 SentencePiece checksum mismatch")
        import sentencepiece as spm

        self.text_processor = spm.SentencePieceProcessor(model_proto=model_bytes)
        self.token_to_id = {
            token: index for index, token in enumerate(STRUCTURAL_TOKENS)
        }
        self.numeric = dict(self.details["numeric"])
        self.text_offset = int(self.details["text_piece_offset"])
        self.text_vocab_size = int(self.details["text_vocab_size"])
        self.vocab_size = int(self.details["vocab_size"])
        if self.text_processor.get_piece_size() != self.text_vocab_size:
            raise ModelScenePlanCodecError("P11 text vocabulary mismatch")
        self.text_ids = set(
            range(self.text_offset, self.text_offset + self.text_vocab_size)
        )
        self.byte_ids = tuple(self._tid(token) for token in BYTE_TOKENS)
        self.byte_id_set = set(self.byte_ids)

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

    def snap_numeric_to_grid(self, kind: str, raw: Any) -> float:
        """Return the nearest representable value for a codec numeric field.

        Strict plan encoding still rejects off-grid inputs. This explicit
        helper is for values derived from exact sample/frame arithmetic at
        runtime, where a repeating decimal must be projected onto the frozen
        fixed-point lattice before it is used as a grammar constraint.
        """

        if kind not in self.numeric:
            raise ModelScenePlanCodecError(f"unknown numeric field {kind!r}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ModelScenePlanCodecError(f"{kind} must be numeric") from error
        spec = self.numeric[kind]
        minimum, maximum, scale = (
            float(spec["minimum"]),
            float(spec["maximum"]),
            int(spec["scale"]),
        )
        if (
            not math.isfinite(value)
            or value < minimum - 1e-8
            or value > maximum + 1e-8
        ):
            raise ModelScenePlanCodecError(
                f"{kind}={value} outside codec range [{minimum},{maximum}]"
            )
        index = int(
            round((min(maximum, max(minimum, value)) - minimum) * scale)
        )
        if not 0 <= index <= int(spec["max_index"]):
            raise ModelScenePlanCodecError(f"{kind} fixed-point index is out of range")
        return float(round(minimum + index / scale, 9))

    def _quantize(self, kind: str, raw: Any) -> list[int]:
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ModelScenePlanCodecError(f"{kind} must be numeric") from error
        spec = self.numeric[kind]
        minimum, maximum, scale = (
            float(spec["minimum"]),
            float(spec["maximum"]),
            int(spec["scale"]),
        )
        if not math.isfinite(value) or value < minimum - 1e-8 or value > maximum + 1e-8:
            raise ModelScenePlanCodecError(
                f"{kind}={value} outside codec range [{minimum},{maximum}]"
            )
        recovered = self.snap_numeric_to_grid(kind, value)
        if not math.isclose(value, recovered, rel_tol=0.0, abs_tol=5e-10):
            raise ModelScenePlanCodecError(
                f"{kind}={value} exceeds the v2 fixed-point precision"
            )
        index = int(round((recovered - minimum) * scale))
        if not 0 <= index <= int(spec["max_index"]):
            raise ModelScenePlanCodecError(f"{kind} fixed-point index is out of range")
        num_bytes = int(spec["num_bytes"])
        payload = index.to_bytes(num_bytes, byteorder="big", signed=False)
        return [self.byte_ids[value] for value in payload]

    def _dequantize(self, kind: str, token_ids: Sequence[int]) -> float:
        spec = self.numeric[kind]
        values = [int(value) for value in token_ids]
        if len(values) != int(spec["num_bytes"]):
            raise ModelScenePlanCodecError(
                f"{kind} requires exactly {spec['num_bytes']} byte tokens"
            )
        try:
            payload = bytes(self.byte_ids.index(value) for value in values)
        except ValueError as error:
            raise ModelScenePlanCodecError(
                f"{kind} contains a non-byte payload token"
            ) from error
        index = int.from_bytes(payload, byteorder="big", signed=False)
        if index > int(spec["max_index"]):
            raise ModelScenePlanCodecError(f"{kind} fixed-point payload is out of range")
        value = float(spec["minimum"]) + index / int(spec["scale"])
        return float(round(value, 9))

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

    @staticmethod
    def _position_fields(source: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
        trajectory = source["trajectory"]
        kind = trajectory["type"]
        if kind == "static":
            return [("<position>", trajectory["position"])]
        if kind == "linear":
            return [("<start>", trajectory["start"]), ("<end>", trajectory["end"])]
        return []

    def encode(
        self, plan: Mapping[str, Any], *, max_tokens: int | None = None
    ) -> dict[str, torch.Tensor]:
        plan = validate_model_sceneplan(plan)
        ids: list[int] = []
        groups: list[int] = []

        def emit(token: str, group: int = LOSS_GRAMMAR) -> None:
            ids.append(self._tid(token))
            groups.append(int(group))

        def number(token: str, kind: str, value: Any, group: int) -> None:
            emit(token, group)
            payload = self._quantize(kind, value)
            ids.extend(payload)
            groups.extend([int(group)] * len(payload))

        def text(token: str, value: Any, group: int) -> None:
            emit(token, group)
            encoded = self._text(value)
            ids.extend(encoded)
            groups.extend([int(group)] * len(encoded))

        def position(token: str, value: Mapping[str, Any]) -> None:
            emit(token, LOSS_MOTION)
            number("<azimuth>", "azimuth", value["azimuth_deg"], LOSS_SPATIAL_METRIC)
            number("<elevation>", "elevation", value["elevation_deg"], LOSS_SPATIAL_METRIC)
            number("<distance>", "distance", value["distance_m"], LOSS_SPATIAL_METRIC)

        emit("<plan_bos>")
        number("<duration>", "seconds", plan["duration_sec"], LOSS_GRAMMAR)
        emit("<room>", LOSS_ROOM)
        emit(ROOM_TOKENS[str(plan["room"]["type"])], LOSS_ROOM)
        for source in plan["sources"]:
            slot = int(str(source["source_id"])[7:])
            kind = str(source["kind"])
            emit("<source_begin>")
            emit(SOURCE_SLOT_TOKENS[slot], LOSS_SEMANTIC)
            emit("<kind>", LOSS_SEMANTIC)
            emit(KIND_TOKENS[kind], LOSS_SEMANTIC)
            if kind == "speech":
                text(
                    "<speaker_description>",
                    source["speaker_description"],
                    LOSS_SEMANTIC,
                )
                text("<transcript>", source["transcript"], LOSS_SPEECH_CONTENT)
            else:
                text("<description>", source["description"], LOSS_SEMANTIC)
            emit("<activity_begin>", LOSS_MOTION)
            number("<onset>", "seconds", source["activity"]["onset_sec"], LOSS_MOTION)
            number("<offset>", "seconds", source["activity"]["offset_sec"], LOSS_MOTION)
            emit("<activity_end>", LOSS_MOTION)
            emit("<trajectory_begin>", LOSS_MOTION)
            motion = str(source["trajectory"]["type"])
            emit(MOTION_TOKENS[motion], LOSS_MOTION)
            if motion == "keyframed":
                for keyframe in source["trajectory"]["keyframes"]:
                    emit("<keyframe_begin>", LOSS_MOTION)
                    number("<time>", "seconds", keyframe["time_sec"], LOSS_MOTION)
                    position("<position>", keyframe["position"])
                    emit("<keyframe_end>", LOSS_MOTION)
            else:
                for token, value in self._position_fields(source):
                    position(token, value)
            emit("<trajectory_end>", LOSS_MOTION)
            number("<gain_db>", "gain_db", source["gain_db"], LOSS_SPATIAL_METRIC)
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
        def __init__(self, codec: "ModelScenePlanCodec", values: Sequence[int]):
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
                raise ModelScenePlanCodecError(
                    f"expected {token} ({expected}), found {found}"
                )

        def number(self, kind: str) -> float:
            count = int(self.codec.numeric[kind]["num_bytes"])
            return self.codec._dequantize(
                kind, [self.take() for _ in range(count)]
            )

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
        cur.expect("<duration>")
        duration = cur.number("seconds")
        cur.expect("<room>")
        room_id = cur.take()
        inverse_rooms = {self._tid(token): key for key, token in ROOM_TOKENS.items()}
        if room_id not in inverse_rooms:
            raise ModelScenePlanCodecError("invalid room token")
        sources: list[dict[str, Any]] = []
        seen_speech = False
        while cur.index < len(cur.values) and cur.values[cur.index] == self._tid("<source_begin>"):
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
            cur.expect("<onset>")
            onset = cur.number("seconds")
            cur.expect("<offset>")
            offset = cur.number("seconds")
            cur.expect("<activity_end>")
            source["activity"] = {"onset_sec": onset, "offset_sec": offset}
            cur.expect("<trajectory_begin>")
            motion_id = cur.take()
            motion_lookup = {self._tid(token): key for key, token in MOTION_TOKENS.items()}
            if motion_id not in motion_lookup:
                raise ModelScenePlanCodecError("invalid motion token")
            motion = motion_lookup[motion_id]

            def read_position(token: str) -> dict[str, float]:
                cur.expect(token)
                cur.expect("<azimuth>")
                azimuth = cur.number("azimuth")
                cur.expect("<elevation>")
                elevation = cur.number("elevation")
                cur.expect("<distance>")
                distance = cur.number("distance")
                return {
                    "azimuth_deg": azimuth,
                    "elevation_deg": elevation,
                    "distance_m": distance,
                }

            if motion == "static":
                trajectory: dict[str, Any] = {
                    "type": motion,
                    "position": read_position("<position>"),
                }
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
                    cur.expect("<time>")
                    time_sec = cur.number("seconds")
                    position = read_position("<position>")
                    cur.expect("<keyframe_end>")
                    keyframes.append({"time_sec": time_sec, "position": position})
                if not 2 <= len(keyframes) <= 8:
                    raise ModelScenePlanCodecError("keyframed motion requires 2--8 keyframes")
                trajectory = {"type": motion, "keyframes": keyframes}
            cur.expect("<trajectory_end>")
            source["trajectory"] = trajectory
            cur.expect("<gain_db>")
            source["gain_db"] = cur.number("gain_db")
            cur.expect("<source_end>")
            sources.append(source)
        cur.expect("<plan_eos>")
        if cur.index != len(cur.values):
            raise ModelScenePlanCodecError("tokens remain after <plan_eos>")
        plan = {
            "sample_id": str(sample_id),
            "duration_sec": duration,
            "room": {"type": inverse_rooms[room_id]},
            "sources": sources,
        }
        return validate_model_sceneplan(plan)

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
        """Return exact grammar-allowed ids after one prefix."""

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

        def number(
            kind: str,
            fixed: float | None = None,
            *,
            minimum: float | None = None,
            maximum: float | None = None,
            minimum_exclusive: bool = False,
        ) -> float:
            spec = self.numeric[kind]
            if fixed is not None:
                payload = self._quantize(kind, fixed)
                for token_id in payload:
                    choice({token_id}, kind)
                return float(fixed)

            scale = int(spec["scale"])
            spec_minimum = float(spec["minimum"])
            low = 0
            high = int(spec["max_index"])
            if minimum is not None:
                raw = (float(minimum) - spec_minimum) * scale
                low = max(
                    low,
                    int(math.floor(raw + 1e-9)) + 1
                    if minimum_exclusive
                    else int(math.ceil(raw - 1e-9)),
                )
            if maximum is not None:
                raw = (float(maximum) - spec_minimum) * scale
                high = min(high, int(math.floor(raw + 1e-9)))
            if low > high:
                raise ModelScenePlanCodecError(
                    f"no valid {kind} value remains inside [{minimum},{maximum}]"
                )

            prefix_value = 0
            byte_count = int(spec["num_bytes"])
            for byte_index in range(byte_count):
                remaining = byte_count - byte_index - 1
                suffix_width = 256 ** remaining
                allowed_bytes = {
                    byte
                    for byte in range(256)
                    if (prefix_value * 256 + byte) * suffix_width <= high
                    and (
                        (prefix_value * 256 + byte) * suffix_width
                        + suffix_width
                        - 1
                    )
                    >= low
                }
                token_id = choice(
                    {self.byte_ids[byte] for byte in allowed_bytes}, kind
                )
                prefix_value = prefix_value * 256 + self.byte_ids.index(token_id)
            value = spec_minimum + prefix_value / scale
            return float(round(value, 9))

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

        def numeric_field(
            token: str,
            kind: str,
            fixed: float | None = None,
            **bounds: Any,
        ) -> float:
            exact(token)
            return number(kind, fixed, **bounds)

        def position(token: str) -> None:
            exact(token)
            numeric_field("<azimuth>", "azimuth")
            numeric_field("<elevation>", "elevation")
            numeric_field("<distance>", "distance")

        try:
            exact("<plan_bos>")
            duration = numeric_field(
                "<duration>",
                "seconds",
                fixed_duration_sec,
                minimum=0.000001,
                maximum=10.031021,
            )
            exact("<room>")
            choice({self._tid(value) for value in ROOM_TOKENS.values()}, "room")
            source_count = 0
            last_slot = -1
            speech_seen = False
            while True:
                can_end = source_count >= int(min_sources)
                can_add = source_count < int(max_sources) and last_slot < 3
                boundary: set[int] = set()
                if can_end:
                    boundary.add(self._tid("<plan_eos>"))
                if can_add:
                    boundary.add(self._tid("<source_begin>"))
                selected = choice(boundary, "source or plan end")
                if selected == self._tid("<plan_eos>"):
                    break
                slot_options = {
                    self._tid(SOURCE_SLOT_TOKENS[slot])
                    for slot in range(last_slot + 1, 4)
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
                onset = numeric_field(
                    "<onset>",
                    "seconds",
                    minimum=0.0,
                    maximum=duration - 0.000001,
                )
                offset = numeric_field(
                    "<offset>",
                    "seconds",
                    minimum=onset,
                    maximum=duration,
                    minimum_exclusive=True,
                )
                exact("<activity_end>")
                exact("<trajectory_begin>")
                motion_id = choice(
                    {self._tid(value) for value in MOTION_TOKENS.values()},
                    "motion type",
                )
                if motion_id == self._tid(MOTION_TOKENS["static"]):
                    position("<position>")
                elif motion_id == self._tid(MOTION_TOKENS["linear"]):
                    position("<start>")
                    position("<end>")
                else:
                    keyframes = 0
                    previous_time: float | None = None
                    while True:
                        allowed = (
                            {self._tid("<keyframe_begin>")}
                            if keyframes < 8
                            and (previous_time is None or previous_time < offset)
                            else set()
                        )
                        if keyframes >= 2:
                            allowed.add(self._tid("<trajectory_end>"))
                        selected = choice(allowed, "keyframe or trajectory end")
                        if selected == self._tid("<trajectory_end>"):
                            break
                        maximum_time = (
                            offset - 0.000001 if keyframes == 0 else offset
                        )
                        previous_time = numeric_field(
                            "<time>",
                            "seconds",
                            minimum=(
                                onset if previous_time is None else previous_time
                            ),
                            maximum=maximum_time,
                            minimum_exclusive=previous_time is not None,
                        )
                        keyframes += 1
                        position("<position>")
                        exact("<keyframe_end>")
                if motion_id != self._tid(MOTION_TOKENS["keyframed"]):
                    exact("<trajectory_end>")
                numeric_field("<gain_db>", "gain_db")
                exact("<source_end>")
                source_count += 1
            if index != len(values):
                raise ModelScenePlanCodecError("tokens remain after valid prefix")
            return set()
        except _NeedToken as need:
            return need.allowed


def load_model_sceneplan_codec(
    artifact_dir: str | os.PathLike[str],
) -> "ModelScenePlanCodec":
    """Load a frozen codec artifact without guessing its schema version."""

    root = Path(artifact_dir).expanduser().resolve()
    details_path = root / "codec.json"
    if not details_path.is_file():
        raise FileNotFoundError(f"incomplete P11 codec: {root}")
    details = json.loads(details_path.read_text(encoding="utf-8"))
    version = int(details.get("schema_version", -1))
    name = str(details.get("codec_name", ""))
    if version == CODEC_VERSION and name == CODEC_NAME:
        return ModelScenePlanCodec(root)
    if version == 3 and name == "model_sceneplan_codec_v3":
        from .model_sceneplan_codec_v3 import ModelScenePlanCodecV3

        return ModelScenePlanCodecV3(root)  # type: ignore[return-value]
    if version == 4 and name == "model_sceneplan_codec_v4":
        from .model_sceneplan_codec_v4 import ModelScenePlanCodecV4

        return ModelScenePlanCodecV4(root)  # type: ignore[return-value]
    raise ModelScenePlanCodecError(
        f"unsupported P11 codec artifact: name={name!r}, version={version}"
    )


__all__ = [
    "CODEC_NAME",
    "CODEC_SCHEMA",
    "CODEC_VERSION",
    "ModelScenePlanCodec",
    "ModelScenePlanCodecError",
    "create_model_sceneplan_codec_artifact",
    "iter_model_sceneplan_text_fields",
    "load_model_sceneplan_codec",
    "train_model_sceneplan_text_bpe",
]
