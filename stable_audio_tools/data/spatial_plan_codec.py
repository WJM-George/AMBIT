"""Compact, reversible token codec for the Spatial-CoT ScenePlan.

The codec deliberately separates *control* vocabulary from free text:

* grammar, fields, motion types, and quantized numeric values are atomic ids;
* human-readable strings use a small byte-fallback SentencePiece BPE;
* the complete vocabulary is capped (4096 by default), so tying the input
  embedding and LM head adds only a few million parameters at width 1024;
* :meth:`allowed_next_ids` implements a grammar constraint for decoding.

Encoding happens after ``crop_scene_plan`` in the DataLoader worker.  This is
important: a pre-tokenized full-clip target would silently disagree with a
randomly cropped trajectory and audio latent.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

import torch

from .scene_plan import (
    LOSS_GRAMMAR,
    LOSS_MOTION,
    LOSS_ROOM,
    LOSS_SEMANTIC,
    LOSS_SPATIAL_CATEGORICAL,
    LOSS_SPATIAL_METRIC,
    LOSS_SPEECH_CONTENT,
)


CODEC_NAME = "spatial_plan_codec_v2"
CODEC_SCHEMA = "stable_audio_tools.spatial_plan_codec"
CODEC_VERSION = 2
LEGACY_CODEC_NAME = "spatial_plan_codec_v1"
LEGACY_CODEC_VERSION = 1

SOURCE_SLOT_TOKENS = (
    "<source_slot_0>",
    "<source_slot_1>",
    "<source_slot_2>",
    "<source_slot_3>",
)

# These ids are intentionally stable. New grammar must get a new codec version
# rather than reordering this tuple under an existing checkpoint.
LEGACY_STRUCTURAL_TOKENS = (
    "<pad>",
    "<plan_bos>",
    "<plan_eos>",
    "<unknown>",
    "<text_begin>",
    "<text_end>",
    "<audio>",
    "<duration>",
    "<mix>",
    "<mix_type>",
    "<room_begin>",
    "<room_end>",
    "<room_type>",
    "<room_description>",
    "<room_reverb>",
    "<room_rt60>",
    "<room_dimensions>",
    "<room_quality>",
    "<source_begin>",
    "<source_end>",
    "<source_id>",
    "<event_begin>",
    "<event_end>",
    "<event_label>",
    "<event_category>",
    "<content_begin>",
    "<content_end>",
    "<transcript>",
    "<speaker_id>",
    "<activity_begin>",
    "<activity_end>",
    "<activity_onset>",
    "<activity_offset>",
    "<activity_quality>",
    "<motion_begin>",
    "<motion_end>",
    "<motion_type>",
    "<motion_static>",
    "<motion_linear>",
    "<motion_keyframed>",
    "<motion_unknown>",
    "<timing_quality>",
    "<keyframe_begin>",
    "<keyframe_end>",
    "<time_norm>",
    "<azimuth>",
    "<elevation>",
    "<distance>",
    "<direction>",
    "<elevation_label>",
    "<distance_label>",
    "<geometry_quality>",
) + SOURCE_SLOT_TOKENS

# V2 appends fields instead of reordering V1 ids.  Gain is part of the
# renderer control state and must survive AR encode/decode; otherwise training
# tracks contain the annotated gain while inference tracks silently use 0 dB.
STRUCTURAL_TOKENS = LEGACY_STRUCTURAL_TOKENS + (
    "<acoustics_begin>",
    "<acoustics_end>",
    "<gain_db>",
)

# Shared numeric ranges keep the grammar compact. Values outside a range are
# clipped and therefore decode to the exact quantized value used for training.
NUMERIC_SPECS = (
    ("seconds", 0.0, 16.0, 0.05),
    ("rt60", 0.0, 3.0, 0.025),
    ("dimension", 0.0, 50.0, 0.1),
    ("time_norm", 0.0, 1.0, 0.01),
    ("azimuth", -180.0, 175.0, 5.0),
    ("elevation", -90.0, 90.0, 5.0),
    ("distance", 0.0, 50.0, 0.1),
    ("gain_db", -24.0, 12.0, 0.25),
)

DEFAULT_TEXT_CORPUS = (
    "speech",
    "audio",
    "single",
    "mixture",
    "static",
    "linear",
    "keyframed",
    "front",
    "front-left",
    "left",
    "rear-left",
    "behind",
    "rear-right",
    "right",
    "front-right",
    "level",
    "above",
    "below",
    "nearby",
    "far away",
    "dry room",
    "reverberant room",
    "renderer_defined",
    "source_annotation",
    "unknown",
)


class CodecError(ValueError):
    """Raised for an invalid codec artifact or token sequence."""


class _NeedToken(Exception):
    def __init__(self, allowed: set[int]):
        super().__init__()
        self.allowed = allowed


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _numeric_count(minimum: float, maximum: float, step: float) -> int:
    return int(round((maximum - minimum) / step)) + 1


def _build_layout(text_vocab_size: int, vocab_size: int) -> dict[str, Any]:
    cursor = len(STRUCTURAL_TOKENS)
    numeric: dict[str, dict[str, Any]] = {}
    for name, minimum, maximum, step in NUMERIC_SPECS:
        count = _numeric_count(minimum, maximum, step)
        numeric[name] = {
            "offset": cursor,
            "count": count,
            "minimum": minimum,
            "maximum": maximum,
            "step": step,
        }
        cursor += count
    text_offset = cursor
    required = text_offset + int(text_vocab_size)
    if required > int(vocab_size):
        raise CodecError(
            f"codec layout needs {required} ids but vocab_size={vocab_size}; "
            "reduce the BPE vocabulary or increase num_text_tokens"
        )
    return {
        "structural_tokens": list(STRUCTURAL_TOKENS),
        "numeric": numeric,
        "text_piece_offset": text_offset,
        "text_vocab_size": int(text_vocab_size),
        "used_vocab_size": required,
        "vocab_size": int(vocab_size),
    }


def train_text_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int = 2048,
) -> bytes:
    """Train exact-roundtrip byte-fallback BPE and return its model bytes."""

    import sentencepiece as spm

    def normalized() -> Iterator[str]:
        emitted = False
        for value in texts:
            if not isinstance(value, str) or not value:
                continue
            emitted = True
            # SentencePiece consumes one training sentence per iterator item.
            # Newlines are field separators here, not content semantics.
            yield value.replace("\r", " ").replace("\n", " ")
        if not emitted:
            yield from DEFAULT_TEXT_CORPUS

    writer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=normalized(),
        model_writer=writer,
        model_type="bpe",
        vocab_size=int(vocab_size),
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


def create_codec_artifact(
    output_dir: str | os.PathLike[str],
    texts: Iterable[str] = DEFAULT_TEXT_CORPUS,
    *,
    vocab_size: int = 4096,
    text_vocab_size: int = 2048,
    overwrite: bool = False,
) -> Path:
    """Build one immutable codec directory with an atomic final rename."""

    output = Path(output_dir).expanduser().resolve()
    if (output / "READY").is_file() and not overwrite:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent))
    )
    try:
        model_bytes = train_text_bpe(texts, vocab_size=text_vocab_size)
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor(model_proto=model_bytes)
        layout = _build_layout(processor.get_piece_size(), vocab_size)
        details = {
            "schema": CODEC_SCHEMA,
            "schema_version": CODEC_VERSION,
            "codec_name": CODEC_NAME,
            "text_backend": "sentencepiece_bpe_byte_fallback",
            "sentencepiece_sha256": _sha256_bytes(model_bytes),
            **layout,
        }
        canonical = json.dumps(
            details, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        details["codec_fingerprint"] = _sha256_bytes(canonical)
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
                    "used_vocab_size": details["used_vocab_size"],
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


def iter_plan_text_fields(plan: dict[str, Any]) -> Iterator[str]:
    """Yield only generative ScenePlan strings (never paths/provenance)."""

    mix = plan.get("mix") or {}
    room = (plan.get("scene") or {}).get("room") or {}
    for value in (
        mix.get("type"),
        room.get("type"),
        room.get("description"),
        room.get("reverb_label"),
        room.get("quality"),
    ):
        if isinstance(value, str) and value:
            yield value
    for source in (plan.get("scene") or {}).get("sources") or []:
        event = source.get("event") or {}
        content = source.get("content") or {}
        activity = source.get("activity") or {}
        motion = source.get("motion") or {}
        for value in (
            source.get("source_id"),
            event.get("label"),
            event.get("category"),
            content.get("transcript"),
            content.get("speaker_id"),
            activity.get("quality"),
            motion.get("timing_quality"),
        ):
            if isinstance(value, str) and value:
                yield value
        for keyframe in motion.get("keyframes") or []:
            position = keyframe.get("position") or {}
            for value in (
                position.get("direction"),
                position.get("elevation"),
                position.get("distance_label"),
                position.get("geometry_quality"),
            ):
                if isinstance(value, str) and value:
                    yield value


class SpatialPlanCodec:
    """Encode/decode quantized ScenePlans and constrain autoregressive output."""

    def __init__(self, artifact_dir: str | os.PathLike[str]):
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        details_path = self.artifact_dir / "codec.json"
        model_path = self.artifact_dir / "sentencepiece.model"
        if not details_path.is_file() or not model_path.is_file():
            raise FileNotFoundError(
                f"incomplete {CODEC_NAME} artifact: {self.artifact_dir}"
            )
        self.details = json.loads(details_path.read_text(encoding="utf-8"))
        artifact_name = str(self.details.get("codec_name") or "")
        artifact_version = int(self.details.get("schema_version", -1))
        if artifact_name == CODEC_NAME and artifact_version == CODEC_VERSION:
            expected_tokens = STRUCTURAL_TOKENS
            self.has_gain = True
        elif (
            artifact_name == LEGACY_CODEC_NAME
            and artifact_version == LEGACY_CODEC_VERSION
        ):
            expected_tokens = LEGACY_STRUCTURAL_TOKENS
            self.has_gain = False
        else:
            raise CodecError(
                f"unsupported codec {artifact_name!r} version {artifact_version} "
                f"in {details_path}"
            )
        self.codec_name = artifact_name
        self.codec_version = artifact_version
        if self.details.get("structural_tokens") != list(expected_tokens):
            raise CodecError(
                f"codec grammar mismatch in {details_path}; rebuild {artifact_name}"
            )
        model_bytes = model_path.read_bytes()
        if _sha256_bytes(model_bytes) != self.details.get("sentencepiece_sha256"):
            raise CodecError(f"SentencePiece checksum mismatch in {self.artifact_dir}")
        import sentencepiece as spm

        self.text_processor = spm.SentencePieceProcessor(model_proto=model_bytes)
        self.token_to_id = {
            token: index for index, token in enumerate(self.details["structural_tokens"])
        }
        self.id_to_token = tuple(self.details["structural_tokens"])
        self.numeric = self.details["numeric"]
        self.text_offset = int(self.details["text_piece_offset"])
        self.text_vocab_size = int(self.details["text_vocab_size"])
        self.vocab_size = int(self.details["vocab_size"])
        if self.text_processor.get_piece_size() != self.text_vocab_size:
            raise CodecError("SentencePiece vocabulary does not match codec.json")
        self.text_ids = set(range(self.text_offset, self.text_offset + self.text_vocab_size))
        self.numeric_ids = {
            name: set(range(int(spec["offset"]), int(spec["offset"]) + int(spec["count"])))
            for name, spec in self.numeric.items()
        }

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<plan_bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<plan_eos>"]

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def fingerprint(self) -> str:
        return str(self.details["codec_fingerprint"])

    def _tid(self, name: str) -> int:
        return self.token_to_id[name]

    def _quantize(self, name: str, value: Any) -> int:
        spec = self.numeric[name]
        number = float(value)
        minimum = float(spec["minimum"])
        maximum = float(spec["maximum"])
        step = float(spec["step"])
        if name == "azimuth":
            number = ((number + 180.0) % 360.0) - 180.0
        number = min(maximum, max(minimum, number))
        index = int(round((number - minimum) / step))
        return int(spec["offset"]) + index

    def _dequantize(self, name: str, token_id: int) -> Optional[float]:
        if token_id == self._tid("<unknown>"):
            return None
        spec = self.numeric[name]
        index = int(token_id) - int(spec["offset"])
        if not (0 <= index < int(spec["count"])):
            raise CodecError(f"token {token_id} is not a {name} value")
        value = float(spec["minimum"]) + index * float(spec["step"])
        return float(round(value, 6))

    def _text_ids_for_value(self, value: Any) -> list[int]:
        if value is None or not str(value):
            return [self._tid("<unknown>")]
        pieces = self.text_processor.encode(str(value), out_type=int)
        return [
            self._tid("<text_begin>"),
            *(self.text_offset + int(piece) for piece in pieces),
            self._tid("<text_end>"),
        ]

    def encode(self, plan: dict[str, Any], *, max_tokens: Optional[int] = None) -> dict[str, torch.Tensor]:
        ids: list[int] = []
        groups: list[int] = []

        def emit(token: str, group: int = LOSS_GRAMMAR) -> None:
            ids.append(self._tid(token))
            groups.append(int(group))

        def text_field(token: str, value: Any, group: int) -> None:
            emit(token, group)
            encoded = self._text_ids_for_value(value)
            ids.extend(encoded)
            groups.extend([int(group)] * len(encoded))

        def number_field(token: str, kind: str, value: Any, group: int) -> None:
            emit(token, group)
            ids.append(
                self._tid("<unknown>") if value is None else self._quantize(kind, value)
            )
            groups.append(int(group))

        seen_source_slots: set[int] = set()

        def source_id_field(value: Any) -> None:
            emit("<source_id>", LOSS_SEMANTIC)
            normalized = str(value or "")
            if normalized.startswith("s") and normalized[1:].isdigit():
                normalized = f"source_{normalized[1:]}"
            slot = None
            if normalized.startswith("source_") and normalized[7:].isdigit():
                index = int(normalized[7:])
                if 0 <= index < len(SOURCE_SLOT_TOKENS):
                    slot = SOURCE_SLOT_TOKENS[index]
            if slot is None:
                raise CodecError(
                    f"source_id must be source_0..source_3, found {value!r}"
                )
            slot_index = SOURCE_SLOT_TOKENS.index(slot)
            if slot_index in seen_source_slots:
                raise CodecError(
                    f"duplicate ScenePlan source slot: source_{slot_index}"
                )
            seen_source_slots.add(slot_index)
            emit(slot, LOSS_SEMANTIC)

        audio = plan.get("audio") or {}
        mix = plan.get("mix") or {}
        scene = plan.get("scene") or {}
        room = scene.get("room") or {}

        emit("<plan_bos>")
        emit("<audio>")
        number_field("<duration>", "seconds", audio.get("duration_sec"), LOSS_GRAMMAR)
        emit("<mix>")
        text_field("<mix_type>", mix.get("type"), LOSS_SEMANTIC)

        emit("<room_begin>", LOSS_ROOM)
        text_field("<room_type>", room.get("type"), LOSS_ROOM)
        text_field("<room_description>", room.get("description"), LOSS_ROOM)
        text_field("<room_reverb>", room.get("reverb_label"), LOSS_ROOM)
        number_field("<room_rt60>", "rt60", room.get("rt60_s"), LOSS_ROOM)
        emit("<room_dimensions>", LOSS_ROOM)
        dimensions = room.get("dimensions_m")
        if not isinstance(dimensions, Sequence) or isinstance(dimensions, (str, bytes)):
            dimensions = (None, None, None)
        dimensions = list(dimensions)[:3] + [None] * max(0, 3 - len(dimensions))
        for value in dimensions[:3]:
            ids.append(
                self._tid("<unknown>")
                if value is None
                else self._quantize("dimension", value)
            )
            groups.append(LOSS_ROOM)
        text_field("<room_quality>", room.get("quality"), LOSS_ROOM)
        emit("<room_end>", LOSS_ROOM)

        for source_index, source in enumerate(scene.get("sources") or []):
            if source_index >= len(SOURCE_SLOT_TOKENS):
                raise CodecError(
                    f"ScenePlan exceeds {len(SOURCE_SLOT_TOKENS)} source slots"
                )
            event = source.get("event") or {}
            content = source.get("content") or {}
            activity = source.get("activity") or {}
            motion = source.get("motion") or {}
            emit("<source_begin>")
            source_id_field(source.get("source_id"))
            emit("<event_begin>", LOSS_SEMANTIC)
            text_field("<event_label>", event.get("label"), LOSS_SEMANTIC)
            text_field("<event_category>", event.get("category"), LOSS_SEMANTIC)
            emit("<event_end>", LOSS_SEMANTIC)
            emit("<content_begin>", LOSS_SPEECH_CONTENT)
            text_field("<transcript>", content.get("transcript"), LOSS_SPEECH_CONTENT)
            text_field("<speaker_id>", content.get("speaker_id"), LOSS_SPEECH_CONTENT)
            emit("<content_end>", LOSS_SPEECH_CONTENT)
            emit("<activity_begin>", LOSS_MOTION)
            number_field("<activity_onset>", "seconds", activity.get("onset_sec"), LOSS_MOTION)
            number_field("<activity_offset>", "seconds", activity.get("offset_sec"), LOSS_MOTION)
            text_field("<activity_quality>", activity.get("quality"), LOSS_MOTION)
            emit("<activity_end>", LOSS_MOTION)
            if self.has_gain:
                acoustics = source.get("acoustics") or {}
                emit("<acoustics_begin>", LOSS_SPATIAL_METRIC)
                number_field(
                    "<gain_db>",
                    "gain_db",
                    acoustics.get("gain_db"),
                    LOSS_SPATIAL_METRIC,
                )
                emit("<acoustics_end>", LOSS_SPATIAL_METRIC)
            emit("<motion_begin>", LOSS_MOTION)
            emit("<motion_type>", LOSS_MOTION)
            motion_type = str(motion.get("type") or "unknown").lower()
            motion_token = {
                "static": "<motion_static>",
                "linear": "<motion_linear>",
                "keyframed": "<motion_keyframed>",
            }.get(motion_type, "<motion_unknown>")
            emit(motion_token, LOSS_MOTION)
            text_field("<timing_quality>", motion.get("timing_quality"), LOSS_MOTION)
            keyframes = motion.get("keyframes") or [
                {"t_norm": 0.0, "position": {}}
            ]
            for keyframe in keyframes:
                position = keyframe.get("position") or {}
                emit("<keyframe_begin>", LOSS_MOTION)
                number_field("<time_norm>", "time_norm", keyframe.get("t_norm"), LOSS_MOTION)
                number_field("<azimuth>", "azimuth", position.get("azimuth_deg"), LOSS_SPATIAL_METRIC)
                number_field("<elevation>", "elevation", position.get("elevation_deg"), LOSS_SPATIAL_METRIC)
                number_field("<distance>", "distance", position.get("distance_m"), LOSS_SPATIAL_METRIC)
                text_field("<direction>", position.get("direction"), LOSS_SPATIAL_CATEGORICAL)
                text_field("<elevation_label>", position.get("elevation"), LOSS_SPATIAL_CATEGORICAL)
                text_field("<distance_label>", position.get("distance_label"), LOSS_SPATIAL_CATEGORICAL)
                text_field("<geometry_quality>", position.get("geometry_quality"), LOSS_SPATIAL_CATEGORICAL)
                emit("<keyframe_end>", LOSS_MOTION)
            emit("<motion_end>", LOSS_MOTION)
            emit("<source_end>")
        emit("<plan_eos>")

        if max_tokens is not None and len(ids) > int(max_tokens):
            raise CodecError(
                f"encoded ScenePlan has {len(ids)} tokens, exceeding max_tokens={max_tokens}"
            )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "loss_group_ids": torch.tensor(groups, dtype=torch.long),
        }

    @staticmethod
    def edit_token_mask(
        previous: Mapping[str, Any] | Sequence[int] | torch.Tensor,
        target: Mapping[str, Any] | Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Mark target tokens introduced or replaced by one persistent edit.

        ScenePlan edit turns copy hundreds of unchanged state tokens while a
        gain/activity edit may alter only one to four values.  A plain token
        average consequently gives the actual edit almost no gradient.  This
        alignment mask lets the Planner add a separately normalized delta CE
        without changing the codec or the full-state generation contract.

        ``SequenceMatcher`` is appropriate here because plans are short,
        canonical token sequences and metadata compilation already runs in
        DataLoader workers.  Deletions have no target span, so the first token
        after the removed span is marked: predicting that boundary is the
        target-side supervision for omitting the deleted state.
        """

        def input_ids(value):
            if isinstance(value, Mapping):
                value = value.get("input_ids")
            tensor = torch.as_tensor(value, dtype=torch.long).detach().flatten()
            if tensor.numel() < 2:
                raise ValueError("edit token alignment requires BOS/EOS plans")
            return tensor

        previous_ids = input_ids(previous)
        target_ids = input_ids(target)
        mask = torch.zeros_like(target_ids, dtype=torch.bool)
        matcher = SequenceMatcher(
            None,
            previous_ids.cpu().tolist(),
            target_ids.cpu().tolist(),
            autojunk=False,
        )
        for opcode, _previous_start, _previous_stop, target_start, target_stop in (
            matcher.get_opcodes()
        ):
            if opcode in {"replace", "insert"}:
                mask[target_start:target_stop] = True
            elif opcode == "delete":
                boundary = min(target_start, int(target_ids.numel()) - 1)
                mask[boundary] = True
        return mask

    def canonicalize(
        self,
        token_ids: Sequence[int] | torch.Tensor,
        *,
        max_tokens: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Decode and re-encode one valid plan into canonical token form.

        The grammar permits any valid SentencePiece segmentation inside text
        fields, while ``SentencePieceProcessor.encode`` selects one canonical
        segmentation.  Normalizing generated plans here prevents a valid but
        non-canonical previous-turn plan from becoming out-of-distribution
        context on the next edit turn.
        """

        return self.encode(self.decode(token_ids), max_tokens=max_tokens)

    class _Decoder:
        def __init__(self, codec: "SpatialPlanCodec", ids: Sequence[int]):
            self.codec = codec
            self.ids = [int(value) for value in ids]
            self.position = 0

        def take(self) -> int:
            if self.position >= len(self.ids):
                raise CodecError("unexpected end of SpatialPlan token sequence")
            value = self.ids[self.position]
            self.position += 1
            return value

        def expect(self, name: str) -> None:
            expected = self.codec._tid(name)
            found = self.take()
            if found != expected:
                raise CodecError(f"expected {name} ({expected}), found token {found}")

        def text(self) -> Optional[str]:
            first = self.take()
            if first == self.codec._tid("<unknown>"):
                return None
            if first != self.codec._tid("<text_begin>"):
                raise CodecError(f"expected text value, found token {first}")
            pieces: list[int] = []
            while True:
                token = self.take()
                if token == self.codec._tid("<text_end>"):
                    break
                piece = token - self.codec.text_offset
                if not (0 <= piece < self.codec.text_vocab_size):
                    raise CodecError(f"token {token} is not a text piece")
                pieces.append(piece)
            return self.codec.text_processor.decode(pieces)

        def number(self, kind: str) -> Optional[float]:
            return self.codec._dequantize(kind, self.take())

        def source_id(self) -> Optional[str]:
            token = self.take()
            for index, name in enumerate(SOURCE_SLOT_TOKENS):
                if token == self.codec._tid(name):
                    return f"source_{index}"
            raise CodecError(f"expected atomic source slot, found token {token}")

    def decode(self, token_ids: Sequence[int] | torch.Tensor) -> dict[str, Any]:
        ids = token_ids.detach().cpu().flatten().tolist() if isinstance(token_ids, torch.Tensor) else list(token_ids)
        cur = self._Decoder(self, ids)
        cur.expect("<plan_bos>")
        cur.expect("<audio>")
        cur.expect("<duration>")
        duration = cur.number("seconds")
        cur.expect("<mix>")
        cur.expect("<mix_type>")
        mix_type = cur.text()
        cur.expect("<room_begin>")
        room: dict[str, Any] = {}
        for field, key in (
            ("<room_type>", "type"),
            ("<room_description>", "description"),
            ("<room_reverb>", "reverb_label"),
        ):
            cur.expect(field)
            room[key] = cur.text()
        cur.expect("<room_rt60>")
        room["rt60_s"] = cur.number("rt60")
        cur.expect("<room_dimensions>")
        dimensions = [cur.number("dimension") for _ in range(3)]
        room["dimensions_m"] = None if all(value is None for value in dimensions) else dimensions
        cur.expect("<room_quality>")
        room["quality"] = cur.text()
        cur.expect("<room_end>")

        sources: list[dict[str, Any]] = []
        while cur.position < len(cur.ids) and cur.ids[cur.position] == self._tid("<source_begin>"):
            cur.expect("<source_begin>")
            cur.expect("<source_id>")
            source_id = cur.source_id()
            if any(source.get("source_id") == source_id for source in sources):
                raise CodecError(
                    f"duplicate ScenePlan source slot: {source_id!r}"
                )
            cur.expect("<event_begin>")
            cur.expect("<event_label>")
            event_label = cur.text()
            cur.expect("<event_category>")
            event_category = cur.text()
            cur.expect("<event_end>")
            cur.expect("<content_begin>")
            cur.expect("<transcript>")
            transcript = cur.text()
            cur.expect("<speaker_id>")
            speaker_id = cur.text()
            cur.expect("<content_end>")
            cur.expect("<activity_begin>")
            cur.expect("<activity_onset>")
            onset = cur.number("seconds")
            cur.expect("<activity_offset>")
            offset = cur.number("seconds")
            cur.expect("<activity_quality>")
            activity_quality = cur.text()
            cur.expect("<activity_end>")
            gain_db = None
            if self.has_gain:
                cur.expect("<acoustics_begin>")
                cur.expect("<gain_db>")
                gain_db = cur.number("gain_db")
                cur.expect("<acoustics_end>")
            cur.expect("<motion_begin>")
            cur.expect("<motion_type>")
            motion_id = cur.take()
            motion_type = {
                self._tid("<motion_static>"): "static",
                self._tid("<motion_linear>"): "linear",
                self._tid("<motion_keyframed>"): "keyframed",
                self._tid("<motion_unknown>"): "unknown",
            }.get(motion_id)
            if motion_type is None:
                raise CodecError(f"invalid motion token {motion_id}")
            cur.expect("<timing_quality>")
            timing_quality = cur.text()
            keyframes = []
            while cur.position < len(cur.ids) and cur.ids[cur.position] == self._tid("<keyframe_begin>"):
                cur.expect("<keyframe_begin>")
                cur.expect("<time_norm>")
                t_norm = cur.number("time_norm")
                cur.expect("<azimuth>")
                azimuth = cur.number("azimuth")
                cur.expect("<elevation>")
                elevation = cur.number("elevation")
                cur.expect("<distance>")
                distance = cur.number("distance")
                cur.expect("<direction>")
                direction = cur.text()
                cur.expect("<elevation_label>")
                elevation_label = cur.text()
                cur.expect("<distance_label>")
                distance_label = cur.text()
                cur.expect("<geometry_quality>")
                geometry_quality = cur.text()
                cur.expect("<keyframe_end>")
                keyframes.append(
                    {
                        "t_norm": t_norm,
                        "position": {
                            "azimuth_deg": azimuth,
                            "elevation_deg": elevation,
                            "distance_m": distance,
                            "direction": direction,
                            "elevation": elevation_label,
                            "distance_label": distance_label,
                            "geometry_quality": geometry_quality,
                        },
                    }
                )
            if not keyframes:
                raise CodecError("motion must contain at least one keyframe")
            cur.expect("<motion_end>")
            cur.expect("<source_end>")
            sources.append(
                {
                    "source_id": source_id,
                    "event": {"label": event_label, "category": event_category},
                    "content": {"transcript": transcript, "speaker_id": speaker_id},
                    "activity": {
                        "onset_sec": onset,
                        "offset_sec": offset,
                        "quality": activity_quality,
                    },
                    "acoustics": {"gain_db": gain_db},
                    "motion": {
                        "type": motion_type,
                        "time_basis": "full_clip",
                        "timing_quality": timing_quality,
                        "keyframes": keyframes,
                    },
                }
            )
        cur.expect("<plan_eos>")
        if cur.position != len(cur.ids):
            raise CodecError("tokens remain after <plan_eos>")
        return {
            "schema": "stable_audio_tools.quantized_spatial_scene_plan",
            "schema_version": 1,
            "codec": self.codec_name,
            "codec_fingerprint": self.fingerprint,
            "audio": {"duration_sec": duration, "spatial_format": "foa"},
            "mix": {"type": mix_type, "num_sources": len(sources)},
            "scene": {"room": room, "sources": sources},
        }

    def allowed_next_ids(
        self,
        prefix: Sequence[int] | torch.Tensor,
        *,
        min_sources: int = 0,
        max_sources: Optional[int] = None,
        fixed_duration_sec: Optional[float] = None,
    ) -> set[int]:
        """Return the exact FSM-allowed ids after ``prefix``.

        The recognizer parses the deterministic grammar from the beginning and
        raises a private ``_NeedToken`` at the first incomplete production.  It
        accepts arbitrary SentencePiece contents only inside explicit text spans.
        """

        if min_sources < 0 or (max_sources is not None and max_sources <= 0):
            raise ValueError("min_sources must be non-negative and max_sources positive")
        if max_sources is not None and min_sources > max_sources:
            raise ValueError("min_sources must not exceed max_sources")
        fixed_duration_id = None
        if fixed_duration_sec is not None:
            fixed_duration_sec = float(fixed_duration_sec)
            if not math.isfinite(fixed_duration_sec) or fixed_duration_sec <= 0.0:
                raise ValueError("fixed_duration_sec must be finite and positive")
            fixed_duration_id = self._quantize("seconds", fixed_duration_sec)

        values = prefix.detach().cpu().flatten().tolist() if isinstance(prefix, torch.Tensor) else [int(value) for value in prefix]
        position = 0
        unknown = self._tid("<unknown>")

        def peek() -> Optional[int]:
            return values[position] if position < len(values) else None

        def exact(name: str) -> None:
            nonlocal position
            expected = self._tid(name)
            if position >= len(values):
                raise _NeedToken({expected})
            if values[position] != expected:
                raise CodecError(
                    f"invalid prefix at {position}: expected {name} ({expected}), found {values[position]}"
                )
            position += 1

        def choice(allowed: set[int], label: str) -> int:
            nonlocal position
            if position >= len(values):
                raise _NeedToken(set(allowed))
            token = values[position]
            if token not in allowed:
                raise CodecError(
                    f"invalid prefix at {position}: expected {label}, found {token}"
                )
            position += 1
            return token

        def text_value() -> None:
            nonlocal position
            start = self._tid("<text_begin>")
            end = self._tid("<text_end>")
            first = choice({unknown, start}, "text or <unknown>")
            if first == unknown:
                return
            while True:
                if position >= len(values):
                    raise _NeedToken(set(self.text_ids) | {end})
                token = values[position]
                if token == end:
                    position += 1
                    return
                if token not in self.text_ids:
                    raise CodecError(
                        f"invalid prefix at {position}: expected BPE piece or <text_end>, found {token}"
                    )
                position += 1

        def number(kind: str, *, fixed_id: Optional[int] = None) -> None:
            allowed = (
                {int(fixed_id)}
                if fixed_id is not None
                else self.numeric_ids[kind] | {unknown}
            )
            choice(allowed, f"{kind} value")

        def text_field(name: str) -> None:
            exact(name)
            text_value()

        def source_id_field(used_slots: set[int]) -> None:
            exact("<source_id>")
            allowed = {
                self._tid(name)
                for index, name in enumerate(SOURCE_SLOT_TOKENS)
                if index not in used_slots
            }
            token = choice(allowed, "unused source slot")
            used_slots.add(
                next(
                    index
                    for index, name in enumerate(SOURCE_SLOT_TOKENS)
                    if token == self._tid(name)
                )
            )

        def number_field(
            name: str,
            kind: str,
            *,
            fixed_id: Optional[int] = None,
        ) -> None:
            exact(name)
            number(kind, fixed_id=fixed_id)

        def keyframe() -> None:
            exact("<keyframe_begin>")
            number_field("<time_norm>", "time_norm")
            number_field("<azimuth>", "azimuth")
            number_field("<elevation>", "elevation")
            number_field("<distance>", "distance")
            text_field("<direction>")
            text_field("<elevation_label>")
            text_field("<distance_label>")
            text_field("<geometry_quality>")
            exact("<keyframe_end>")

        def source(used_slots: set[int]) -> None:
            exact("<source_begin>")
            source_id_field(used_slots)
            exact("<event_begin>")
            text_field("<event_label>")
            text_field("<event_category>")
            exact("<event_end>")
            exact("<content_begin>")
            text_field("<transcript>")
            text_field("<speaker_id>")
            exact("<content_end>")
            exact("<activity_begin>")
            number_field("<activity_onset>", "seconds")
            number_field("<activity_offset>", "seconds")
            text_field("<activity_quality>")
            exact("<activity_end>")
            if self.has_gain:
                exact("<acoustics_begin>")
                number_field("<gain_db>", "gain_db")
                exact("<acoustics_end>")
            exact("<motion_begin>")
            exact("<motion_type>")
            choice(
                {
                    self._tid("<motion_static>"),
                    self._tid("<motion_linear>"),
                    self._tid("<motion_keyframed>"),
                    self._tid("<motion_unknown>"),
                },
                "motion type",
            )
            text_field("<timing_quality>")
            keyframe()
            while True:
                token = peek()
                if token is None:
                    raise _NeedToken(
                        {self._tid("<keyframe_begin>"), self._tid("<motion_end>")}
                    )
                if token != self._tid("<keyframe_begin>"):
                    break
                keyframe()
            exact("<motion_end>")
            exact("<source_end>")

        try:
            exact("<plan_bos>")
            exact("<audio>")
            number_field(
                "<duration>",
                "seconds",
                fixed_id=fixed_duration_id,
            )
            exact("<mix>")
            text_field("<mix_type>")
            exact("<room_begin>")
            text_field("<room_type>")
            text_field("<room_description>")
            text_field("<room_reverb>")
            number_field("<room_rt60>", "rt60")
            exact("<room_dimensions>")
            number("dimension")
            number("dimension")
            number("dimension")
            text_field("<room_quality>")
            exact("<room_end>")
            source_count = 0
            used_slots: set[int] = set()
            while True:
                allowed_boundary = {self._tid("<plan_eos>")}
                if max_sources is None or source_count < max_sources:
                    allowed_boundary.add(self._tid("<source_begin>"))
                if source_count < min_sources:
                    allowed_boundary.discard(self._tid("<plan_eos>"))
                token = peek()
                if token is None:
                    raise _NeedToken(allowed_boundary)
                if token not in allowed_boundary:
                    raise CodecError(
                        f"invalid source boundary at {position}: found {token}; "
                        f"source_count={source_count}, allowed={sorted(allowed_boundary)}"
                    )
                if token == self._tid("<plan_eos>"):
                    break
                source(used_slots)
                source_count += 1
            exact("<plan_eos>")
            if position != len(values):
                raise CodecError(f"tokens remain after <plan_eos> at position {position}")
            return set()
        except _NeedToken as need:
            return need.allowed


__all__ = [
    "CODEC_NAME",
    "CODEC_SCHEMA",
    "CODEC_VERSION",
    "LEGACY_CODEC_NAME",
    "CodecError",
    "SOURCE_SLOT_TOKENS",
    "SpatialPlanCodec",
    "create_codec_artifact",
    "iter_plan_text_fields",
    "train_text_bpe",
]
