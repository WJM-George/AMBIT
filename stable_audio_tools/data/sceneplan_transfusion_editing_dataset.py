"""Frozen paired-latent dataset for the Transfusion Editing DiT.

The model-facing contract is intentionally narrow:

``target latent`` is the rectified-flow training target, while
``sceneplan_44`` is compiled from the complete *new* ScenePlan and
``source_foa_latent`` is the clean, time-aligned source reference.  The raw
instruction and this same reference latent are the future Editing-AR inputs.
The old ScenePlan remains in the frozen pair index for construction/audit and
offline metrics, but this training reader neither selects nor returns its
compressed payload.  It is not an Editing-AR or Editing-DiT input.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open

from .model_sceneplan import (
    MODEL_SAMPLE_RATE,
    compile_model_44_controls,
    compile_model_semantic_caption_v2,
    make_sceneplan_cfg_unknown_metadata,
    tokenize_model_semantic_caption,
    validate_model_sceneplan,
)
from .sceneplan_transfusion_editing import sha256_json
from .sceneplan_transfusion_editing_index import sha256_file


EDITING_INDEX_SCHEMA = "sceneplan_transfusion_editing_training_index"
EDITING_INDEX_SCHEMA_VERSION = "1"
EDITING_INDEX_STATE = "materialized_complete_frozen"
EDITING_PAIR_CONTRACT = "sceneplan_transfusion_paired_editing_v1"
EDITING_INSTRUCTION_CONTRACT = "sceneplan_transfusion_edit_instruction_v1"
EDITING_GAIN_POLICY = (
    "fixed_member_corrections_nonboosting_peak_safe_target_master_v1"
)
EDITING_DIT_RUNTIME_SELECT_COLUMNS = (
    "pair_id",
    "split",
    "source_sample_id",
    "target_sample_id",
    "operation_family",
    "operation",
    "raw_edit_request",
    "instruction_template_id",
    "instruction_sha256",
    "source_count",
    "target_count",
    "latent_bucket_frames",
    "model_num_samples",
    "latent_frames_valid",
    "new_sceneplan_zlib",
    # The old-plan payload is deliberately absent.  Its digest remains part
    # of the immutable pair-record chain without exposing the plan itself.
    "old_sceneplan_sha256",
    "new_sceneplan_sha256",
    "source_render_recipe_sha256",
    "target_render_recipe_sha256",
    "source_render_result_sha256",
    "source_members_sha256",
    "target_members_sha256",
    "edited_source_ids_json",
    "unchanged_source_ids_json",
    "source_latent_path",
    "source_latent_key",
    "source_latent_ref",
    "source_latent_tensor_sha256",
    "source_latent_shard_sha256",
    "target_latent_path",
    "target_latent_key",
    "target_latent_ref",
    "target_latent_tensor_sha256",
    "target_latent_shard_sha256",
    "target_foa_sha256",
    "target_render_result_sha256",
    "pair_gain_policy",
    "pair_record_sha256",
    "target_materialized_manifest_sha256",
    "materialized_record_sha256",
)


def _verify_editing_latent_shards(
    index_path: str | Path, *, role: str
) -> dict[str, Any]:
    """Hash every distinct external latent shard once at a formal gate."""

    if role not in {"source", "target"}:
        raise ValueError("Editing latent shard role must be source or target")

    index = Path(index_path).expanduser().resolve(strict=True)
    connection = sqlite3.connect(
        f"file:{index}?mode=ro&immutable=1", uri=True, check_same_thread=False
    )
    try:
        rows = connection.execute(
            f"SELECT {role}_latent_path,{role}_latent_shard_sha256,COUNT(*) "
            f"FROM pairs GROUP BY {role}_latent_path,{role}_latent_shard_sha256 "
            f"ORDER BY {role}_latent_path"
        ).fetchall()
        pair_rows = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    finally:
        connection.close()
    if not rows or sum(int(row[2]) for row in rows) != pair_rows:
        raise RuntimeError(f"{role} latent shard inventory is incomplete")
    paths: set[Path] = set()
    inventory = []
    for raw_path, expected_sha, count in rows:
        path = Path(str(raw_path)).expanduser().resolve(strict=True)
        expected = str(expected_sha)
        if path in paths:
            raise RuntimeError(
                f"one {role} latent shard has conflicting identities"
            )
        paths.add(path)
        if len(expected) != 64 or sha256_file(path) != expected:
            raise RuntimeError(f"{role} latent shard SHA256 changed: {path}")
        inventory.append(
            {"path": str(path), "sha256": expected, "pair_rows": int(count)}
        )
    return {
        f"{role}_latent_shards": len(inventory),
        f"{role}_pair_rows": pair_rows,
        f"{role}_latent_shard_inventory_sha256": sha256_json(inventory),
        f"{role}_latent_shards_exhaustively_verified": True,
    }


def verify_editing_source_latent_shards(index_path: str | Path) -> dict[str, Any]:
    """Hash all source shards without per-item hashing in the hot loader."""

    return _verify_editing_latent_shards(index_path, role="source")


def verify_editing_target_latent_shards(index_path: str | Path) -> dict[str, Any]:
    """Hash all target shards before training, selection, or evaluation."""

    return _verify_editing_latent_shards(index_path, role="target")


def compile_editing_dit_plan_condition(
    model_sceneplan: Mapping[str, Any],
    *,
    tokenizer: Any,
    model_num_samples: int,
    latent_frames_valid: int,
    latent_crop_length: int,
    caption_max_tokens: int = 512,
) -> dict[str, Any]:
    """Compile one complete new ScenePlan into the exact Editing-DiT condition."""

    validate_model_sceneplan(model_sceneplan)
    valid_frames = int(latent_frames_valid)
    crop_length = int(latent_crop_length)
    if crop_length not in (432, 648) or not 1 <= valid_frames <= crop_length:
        raise ValueError("Editing-DiT plan condition has invalid latent geometry")
    semantic_caption = compile_model_semantic_caption_v2(model_sceneplan)
    tokenized = tokenize_model_semantic_caption(
        semantic_caption, tokenizer, max_length=int(caption_max_tokens)
    )
    prompt = {
        "input_ids": torch.as_tensor(tokenized["input_ids"], dtype=torch.long),
        "attention_mask": torch.as_tensor(
            tokenized["attention_mask"], dtype=torch.bool
        ),
        "event_source_ids": torch.as_tensor(
            tokenized["event_source_ids"], dtype=torch.int8
        ),
        "speech_source_ids": torch.as_tensor(
            tokenized["speech_source_ids"], dtype=torch.int8
        ),
        "speech_lexical_mask": torch.as_tensor(
            tokenized["speech_lexical_mask"], dtype=torch.bool
        ),
    }
    structured = compile_model_44_controls(
        model_sceneplan,
        model_num_samples=int(model_num_samples),
        latent_frames_valid=valid_frames,
    )
    if structured["source_trajectory_features"].shape != (
        4,
        valid_frames,
        5,
    ):
        raise RuntimeError("compiled Editing-DiT trajectory shape changed")
    trajectory = np.zeros((4, crop_length, 5), dtype=np.float32)
    trajectory[:, :valid_frames] = structured["source_trajectory_features"]
    event_ids = np.zeros((4, crop_length), dtype=np.int8)
    event_ids[:, :valid_frames] = structured["source_event_frame_ids"]
    frame_valid = torch.zeros(crop_length, dtype=torch.bool)
    frame_valid[:valid_frames] = True
    speech_active = np.zeros(crop_length, dtype=np.uint8)
    speech_active[:valid_frames] = structured["speech_active_frame_mask"]
    controls = {
        "source_event_frame_ids": torch.as_tensor(event_ids, dtype=torch.int8),
        "source_trajectory_features": torch.as_tensor(trajectory),
        "frame_valid_mask": frame_valid,
        "speech_active_frame_mask": torch.as_tensor(
            speech_active, dtype=torch.bool
        ),
    }
    return {
        "model_sceneplan": model_sceneplan,
        "prompt": prompt,
        "prompt_text": semantic_caption["text"],
        "semantic_caption_compiler_version": int(
            semantic_caption["compiler_version"]
        ),
        "sceneplan_44": controls,
    }


def make_editing_dit_cfg_unknown_metadata(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Unknown the new-plan branches while retaining the exact clean source.

    Editing guidance contrasts ``[new plan, source]`` with
    ``[unknown plan, same source]``.  Dropping the source in the negative
    branch would turn CFG into an edit-vs-generation contrast and undermine
    waveform preservation.
    """

    source = metadata.get("source_foa_latent")
    if not isinstance(source, torch.Tensor) or source.ndim != 2 or source.shape[0] != 64:
        raise ValueError("Editing CFG requires a clean aligned [64,T] source latent")
    negative = make_sceneplan_cfg_unknown_metadata(metadata)
    if negative.get("source_foa_latent") is not source:
        raise RuntimeError("Editing CFG changed or copied the source latent")
    return negative


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _unpack_json(value: bytes, *, field: str, pair_id: str) -> Any:
    try:
        return json.loads(zlib.decompress(value))
    except (TypeError, ValueError, zlib.error) as error:
        raise RuntimeError(f"{pair_id}: invalid compressed {field}") from error


class ScenePlanTransfusionEditingDataset(torch.utils.data.Dataset):
    """Load exact aligned source/target FOA latents and new-plan controls."""

    pair_contract = EDITING_PAIR_CONTRACT
    instruction_contract = EDITING_INSTRUCTION_CONTRACT
    operation_count_deltas = {
        "event_addition": 1,
        "event_removal": -1,
        "linear_to_static": 0,
        "static_to_linear": 0,
        "stationary_spatial_relocation": 0,
    }

    def __init__(
        self,
        index_path: str | Path,
        *,
        tokenizer_spec: Any,
        expected_num_samples: int,
        index_num_samples: int | None = None,
        expected_index_sha256: str | None = None,
        sample_ordinals: Sequence[int] | None = None,
        ordinal_start: int | None = None,
        ordinal_stop: int | None = None,
        latent_crop_length: int = 648,
        caption_max_tokens: int = 512,
        random_crop: bool = False,
        require_frozen: bool = True,
        verify_tensor_hashes_on_access: bool = False,
    ) -> None:
        super().__init__()
        self.index_path = Path(index_path).expanduser().resolve(strict=True)
        if self.index_path.suffix != ".sqlite":
            raise ValueError("Editing training index must be a SQLite file")
        if random_crop:
            raise ValueError("aligned Transfusion Editing forbids random crops")
        if int(latent_crop_length) not in (432, 648):
            raise ValueError("Editing latent envelope must be 432 or 648 frames")
        if int(caption_max_tokens) != 512:
            raise ValueError("Editing semantic captions require 512 tokens")
        if not isinstance(tokenizer_spec, (tuple, list)) or len(tokenizer_spec) not in (
            2,
            3,
        ):
            raise ValueError("Editing requires the P10 Qwen prompt tokenizer spec")

        self.tokenizer = tokenizer_spec[0]
        if int(tokenizer_spec[1]) != int(caption_max_tokens):
            raise ValueError("tokenizer and Editing caption limits differ")
        self.latent_crop_length = int(latent_crop_length)
        self.caption_max_tokens = int(caption_max_tokens)
        self.verify_tensor_hashes_on_access = bool(verify_tensor_hashes_on_access)
        self.structured_feature_dim = 5
        self.semantic_caption_requires_epoch_key = False
        self.sample_weights: list[float] = []
        self._connection: sqlite3.Connection | None = None
        self._length_bucket_indices_cache: dict[int, tuple[int, ...]] | None = None

        actual_index_sha = sha256_file(self.index_path)
        if expected_index_sha256 is not None and (
            len(str(expected_index_sha256)) != 64
            or actual_index_sha != str(expected_index_sha256)
        ):
            raise RuntimeError("Editing training-index SHA256 mismatch")
        if require_frozen:
            marker_path = self.index_path.with_suffix(
                self.index_path.suffix + ".frozen.json"
            )
            if not marker_path.is_file():
                raise RuntimeError(f"Editing frozen marker is absent: {marker_path}")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            expected_marker = {
                "schema": EDITING_INDEX_SCHEMA,
                "schema_version": 1,
                "state": EDITING_INDEX_STATE,
            }
            for key, expected in expected_marker.items():
                if marker.get(key) != expected:
                    raise RuntimeError(
                        f"Editing frozen marker {key} changed: "
                        f"{marker.get(key)!r} != {expected!r}"
                    )
            if Path(marker.get("index_path", "")).resolve() != self.index_path:
                raise RuntimeError("Editing frozen marker names another index")
            if marker.get("index_sha256") != actual_index_sha:
                raise RuntimeError("Editing frozen marker index SHA256 mismatch")

        connection = self._open_connection()
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        required_metadata = {
            "schema": EDITING_INDEX_SCHEMA,
            "schema_version": EDITING_INDEX_SCHEMA_VERSION,
            "state": EDITING_INDEX_STATE,
            "editing_pair_contract": self.pair_contract,
            "editing_instruction_contract": self.instruction_contract,
            "pair_gain_policy": EDITING_GAIN_POLICY,
            "target_latents_exhaustively_reopened": "true",
            "target_tensor_hashes_exhaustively_verified": "true",
        }
        for key, expected in required_metadata.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"Editing index metadata {key} changed: "
                    f"{metadata.get(key)!r} != {expected!r}"
                )
        self.split = str(metadata.get("split") or "")
        if self.split not in {"train", "validation", "test"}:
            raise RuntimeError("Editing index has an invalid split")
        count, minimum, maximum, distinct = connection.execute(
            """
            SELECT COUNT(*),MIN(pair_ordinal),MAX(pair_ordinal),
                   COUNT(DISTINCT pair_ordinal)
            FROM pairs
            """
        ).fetchone()
        count = int(count)
        if (
            count <= 0
            or int(minimum) != 0
            or int(maximum) != count - 1
            or int(distinct) != count
            or int(metadata.get("rows", -1)) != count
        ):
            raise RuntimeError("Editing index pair ordinals are not dense and complete")
        if require_frozen and int(marker.get("rows", -1)) != count:
            raise RuntimeError("Editing frozen marker row count mismatch")
        expected_index_rows = int(
            index_num_samples if index_num_samples is not None else expected_num_samples
        )
        if count != expected_index_rows:
            raise RuntimeError(
                f"Editing index has {count} rows, expected {expected_index_rows}"
            )
        incomplete = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM pairs
                WHERE split != ? OR materialization_status != 'encoded'
                   OR target_latent_tensor_sha256 IS NULL
                   OR target_latent_shard_sha256 IS NULL
                   OR source_latent_tensor_sha256 IS NULL
                   OR source_latent_shard_sha256 IS NULL
                   OR materialized_record_sha256 IS NULL
                """,
                (self.split,),
            ).fetchone()[0]
        )
        if incomplete:
            raise RuntimeError(f"Editing index contains {incomplete} incomplete rows")

        has_range = ordinal_start is not None or ordinal_stop is not None
        if has_range and (ordinal_start is None or ordinal_stop is None):
            raise ValueError("Editing ordinal range needs both start and stop")
        if has_range and sample_ordinals is not None:
            raise ValueError("Editing ordinal range and explicit ordinals conflict")
        self._ordinal_start = 0
        if sample_ordinals is None and not has_range:
            if int(expected_num_samples) != count:
                raise RuntimeError(
                    "expected_num_samples differs from the full Editing index "
                    "without an ordinal selection"
                )
            self._ordinals: tuple[int, ...] | None = None
            self._length = count
        elif has_range:
            start, stop = int(ordinal_start), int(ordinal_stop)
            if not 0 <= start < stop <= count:
                raise RuntimeError("Editing ordinal range is outside the index")
            if stop - start != int(expected_num_samples):
                raise RuntimeError("Editing ordinal range length differs from expected")
            self._ordinals = None
            self._ordinal_start = start
            self._length = stop - start
        else:
            ordinals = tuple(int(value) for value in sample_ordinals or ())
            if (
                len(ordinals) != int(expected_num_samples)
                or len(set(ordinals)) != len(ordinals)
                or any(value < 0 or value >= count for value in ordinals)
            ):
                raise RuntimeError("Editing explicit ordinals are invalid")
            self._ordinals = ordinals
            self._length = len(ordinals)
        connection.close()

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.index_path}?mode=ro&immutable=1",
            uri=True,
            check_same_thread=False,
        )
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = self._open_connection()
        return self._connection

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        return state

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()

    def __len__(self) -> int:
        return self._length

    def length_bucket_indices(self) -> dict[int, tuple[int, ...]]:
        if self._length_bucket_indices_cache is not None:
            return self._length_bucket_indices_cache
        buckets: dict[int, list[int]] = {432: [], 648: []}
        connection = self._open_connection()
        try:
            if self._ordinals is None:
                start = self._ordinal_start
                stop = start + self._length
                rows = connection.execute(
                    """
                    SELECT pair_ordinal,latent_bucket_frames,latent_frames_valid
                    FROM pairs WHERE pair_ordinal >= ? AND pair_ordinal < ?
                    ORDER BY pair_ordinal
                    """,
                    (start, stop),
                )
                observed = 0
                for ordinal, bucket, valid in rows:
                    bucket = int(bucket)
                    if bucket not in (432, 648) or int(valid) > min(
                        bucket, self.latent_crop_length
                    ):
                        raise RuntimeError(f"pair ordinal {ordinal}: invalid length bucket")
                    buckets[bucket].append(int(ordinal) - start)
                    observed += 1
                if observed != self._length:
                    raise RuntimeError("Editing bucket scan is incomplete")
            else:
                wanted = {ordinal: local for local, ordinal in enumerate(self._ordinals)}
                observed: set[int] = set()
                for ordinal, bucket, valid in connection.execute(
                    "SELECT pair_ordinal,latent_bucket_frames,latent_frames_valid "
                    "FROM pairs ORDER BY pair_ordinal"
                ):
                    local = wanted.get(int(ordinal))
                    if local is None:
                        continue
                    bucket = int(bucket)
                    if bucket not in (432, 648) or int(valid) > min(
                        bucket, self.latent_crop_length
                    ):
                        raise RuntimeError(f"pair ordinal {ordinal}: invalid length bucket")
                    buckets[bucket].append(local)
                    observed.add(int(ordinal))
                if len(observed) != self._length:
                    raise RuntimeError("Editing subset bucket scan is incomplete")
        finally:
            connection.close()
        self._length_bucket_indices_cache = {
            bucket: tuple(values) for bucket, values in buckets.items() if values
        }
        return self._length_bucket_indices_cache

    def _tokenize(self, caption: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        tokenized = tokenize_model_semantic_caption(
            caption, self.tokenizer, max_length=self.caption_max_tokens
        )
        return {
            "input_ids": torch.as_tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.as_tensor(
                tokenized["attention_mask"], dtype=torch.bool
            ),
            "event_source_ids": torch.as_tensor(
                tokenized["event_source_ids"], dtype=torch.int8
            ),
            "speech_source_ids": torch.as_tensor(
                tokenized["speech_source_ids"], dtype=torch.int8
            ),
            "speech_lexical_mask": torch.as_tensor(
                tokenized["speech_lexical_mask"], dtype=torch.bool
            ),
        }

    def _load_latent(
        self,
        path: str,
        key: str,
        *,
        expected_frames: int,
        expected_sha256: str,
        pair_id: str,
        role: str,
    ) -> torch.Tensor:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if key not in handle.keys():
                raise RuntimeError(f"{pair_id}: {role} latent key is absent")
            value = handle.get_tensor(key).clone()
        if value.dtype != torch.float16 or tuple(value.shape) != (
            64,
            int(expected_frames),
        ):
            raise RuntimeError(f"{pair_id}: {role} latent geometry changed")
        if not torch.isfinite(value).all().item():
            raise RuntimeError(f"{pair_id}: {role} latent is non-finite")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise RuntimeError(f"{pair_id}: {role} latent checksum is invalid")
        if self.verify_tensor_hashes_on_access and _tensor_sha256(value) != expected_sha256:
            raise RuntimeError(f"{pair_id}: {role} latent checksum changed")
        return value

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        if not 0 <= int(index) < self._length:
            raise IndexError(index)
        ordinal = (
            self._ordinal_start + int(index)
            if self._ordinals is None
            else self._ordinals[int(index)]
        )
        select = ",".join(EDITING_DIT_RUNTIME_SELECT_COLUMNS)
        row = self._db().execute(
            f"SELECT {select} FROM pairs WHERE pair_ordinal=?",
            (int(ordinal),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"Editing index has no pair ordinal {ordinal}")
        (
            pair_id,
            split,
            source_sample_id,
            target_sample_id,
            operation_family,
            operation,
            instruction,
            instruction_template_id,
            instruction_sha,
            source_count,
            target_count,
            latent_bucket_frames,
            model_num_samples,
            valid_frames,
            new_plan_blob,
            old_plan_sha,
            new_plan_sha,
            source_recipe_sha,
            target_recipe_sha,
            source_result_sha,
            source_members_sha,
            target_members_sha,
            edited_ids_json,
            unchanged_ids_json,
            source_latent_path,
            source_latent_key,
            source_latent_ref,
            source_tensor_sha,
            source_shard_sha,
            target_latent_path,
            target_latent_key,
            target_latent_ref,
            target_tensor_sha,
            target_shard_sha,
            target_foa_sha,
            target_result_sha,
            pair_gain_policy,
            pair_record_sha,
            target_manifest_sha,
            materialized_record_sha,
        ) = row
        pair_id = str(pair_id)
        if str(split) != self.split:
            raise RuntimeError(f"{pair_id}: split changed")
        if _text_sha256(str(instruction)) != str(instruction_sha):
            raise RuntimeError(f"{pair_id}: edit instruction changed")
        new_plan = _unpack_json(new_plan_blob, field="new ScenePlan", pair_id=pair_id)
        if sha256_json(new_plan) != str(new_plan_sha):
            raise RuntimeError(f"{pair_id}: new ScenePlan checksum changed")
        validate_model_sceneplan(new_plan)
        if (
            new_plan.get("sample_id") != target_sample_id
            or len(new_plan["sources"]) != int(target_count)
        ):
            raise RuntimeError(f"{pair_id}: new ScenePlan identity/count changed")
        expected_count_delta = self.operation_count_deltas.get(str(operation))
        if (
            expected_count_delta is None
            or not 1 <= int(source_count) <= 4
            or not 1 <= int(target_count) <= 4
            or int(target_count) - int(source_count) != expected_count_delta
        ):
            raise RuntimeError(f"{pair_id}: source/target count contract changed")
        model_num_samples = int(model_num_samples)
        valid_frames = int(valid_frames)
        if (
            not math.isclose(
                float(new_plan["duration_sec"]),
                model_num_samples / MODEL_SAMPLE_RATE,
                rel_tol=0.0,
                abs_tol=1.1e-6,
            )
            or math.ceil(model_num_samples / 1024) != valid_frames
            or not 0 < valid_frames <= self.latent_crop_length
        ):
            raise RuntimeError(f"{pair_id}: aligned time geometry changed")
        expected_bucket = 432 if valid_frames <= 432 else 648
        if int(latent_bucket_frames) != expected_bucket:
            raise RuntimeError(f"{pair_id}: latent bucket changed")
        if str(source_latent_ref) != f"{source_latent_path}#{source_latent_key}":
            raise RuntimeError(f"{pair_id}: source latent reference changed")
        if not isinstance(source_shard_sha, str) or len(source_shard_sha) != 64:
            raise RuntimeError(f"{pair_id}: source latent shard checksum changed")
        if (
            str(target_latent_ref) != f"{target_latent_path}#{target_latent_key}"
            or str(target_latent_key) != str(target_sample_id)
        ):
            raise RuntimeError(f"{pair_id}: target latent reference changed")
        expected_pair_record = sha256_json(
            {
                "pair_id": pair_id,
                "split": self.split,
                "operation": str(operation),
                "instruction_sha256": str(instruction_sha),
                "old_sceneplan_sha256": str(old_plan_sha),
                "new_sceneplan_sha256": str(new_plan_sha),
                "source_render_recipe_sha256": str(source_recipe_sha),
                "target_render_recipe_sha256": str(target_recipe_sha),
                "source_render_result_sha256": str(source_result_sha),
                "source_members_sha256": str(source_members_sha),
                "target_members_sha256": str(target_members_sha),
                "source_latent_ref": str(source_latent_ref),
                "source_latent_tensor_sha256": str(source_tensor_sha),
                "target_latent_ref": str(target_latent_ref),
                "pair_gain_policy": str(pair_gain_policy),
            }
        )
        if expected_pair_record != str(pair_record_sha):
            raise RuntimeError(f"{pair_id}: planned pair record changed")
        expected_materialized_record = sha256_json(
            {
                "pair_record_sha256": str(pair_record_sha),
                "target_foa_sha256": str(target_foa_sha),
                "target_latent_ref": str(target_latent_ref),
                "target_latent_tensor_sha256": str(target_tensor_sha),
                "target_latent_shard_sha256": str(target_shard_sha),
                "target_render_result_sha256": str(target_result_sha),
                "target_materialized_manifest_sha256": str(target_manifest_sha),
            }
        )
        if expected_materialized_record != str(materialized_record_sha):
            raise RuntimeError(f"{pair_id}: materialized pair record changed")

        source = self._load_latent(
            str(source_latent_path),
            str(source_latent_key),
            expected_frames=valid_frames,
            expected_sha256=str(source_tensor_sha),
            pair_id=pair_id,
            role="source",
        )
        target = self._load_latent(
            str(target_latent_path),
            str(target_latent_key),
            expected_frames=valid_frames,
            expected_sha256=str(target_tensor_sha),
            pair_id=pair_id,
            role="target",
        )
        source_padded = torch.zeros(
            (64, self.latent_crop_length), dtype=torch.float16
        )
        target_padded = torch.zeros_like(source_padded)
        source_padded[:, :valid_frames] = source
        target_padded[:, :valid_frames] = target
        padding_mask = torch.zeros(self.latent_crop_length, dtype=torch.bool)
        padding_mask[:valid_frames] = True

        plan_condition = compile_editing_dit_plan_condition(
            new_plan,
            tokenizer=self.tokenizer,
            model_num_samples=model_num_samples,
            latent_frames_valid=valid_frames,
            latent_crop_length=self.latent_crop_length,
            caption_max_tokens=self.caption_max_tokens,
        )
        edited_ids = tuple(json.loads(str(edited_ids_json)))
        unchanged_ids = tuple(json.loads(str(unchanged_ids_json)))
        metadata = {
            "pair_ordinal": ordinal,
            "sample_id": str(target_sample_id),
            "pair_id": pair_id,
            "source_sample_id": str(source_sample_id),
            "target_sample_id": str(target_sample_id),
            "editing_split": self.split,
            "operation_family": str(operation_family),
            "operation": str(operation),
            # AR condition/target truth. The reference audio is provided below.
            "raw_edit_request": str(instruction),
            "instruction_template_id": str(instruction_template_id),
            "editing_ar_target_model_sceneplan": new_plan,
            "edited_source_ids": edited_ids,
            "unchanged_source_ids": unchanged_ids,
            # DiT truth: complete new plan plus clean aligned source latent.
            "model_num_samples": model_num_samples,
            **plan_condition,
            "source_foa_latent": source_padded,
            "source_foa_latent_tensor_sha256": str(source_tensor_sha),
            "source_foa_latent_shard_sha256": str(source_shard_sha),
            "target_foa_latent_tensor_sha256": str(target_tensor_sha),
            "padding_mask": [padding_mask],
            "seconds_start": 0.0,
            "seconds_total": model_num_samples / MODEL_SAMPLE_RATE,
            "latent_stored_length": valid_frames,
            "latent_frames_valid": valid_frames,
            "latent_crop_length": self.latent_crop_length,
            "latent_bucket_frames": expected_bucket,
            "latent_crop_start": 0,
            "latent_tensor_sha256": str(target_tensor_sha),
            "audio": target_padded,
        }
        return target_padded, metadata


__all__ = [
    "EDITING_DIT_RUNTIME_SELECT_COLUMNS",
    "ScenePlanTransfusionEditingDataset",
    "compile_editing_dit_plan_condition",
    "make_editing_dit_cfg_unknown_metadata",
    "verify_editing_source_latent_shards",
    "verify_editing_target_latent_shards",
]
