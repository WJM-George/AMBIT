"""Audio-reference dataset for Editing AR.

This reader intentionally never selects the stored old/source ScenePlan.  Its
model-facing row consists only of the clean source FOA latent, the raw edit
instruction, and the complete new ScenePlan teacher target.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence
import zlib

import torch
from safetensors import safe_open

from .model_sceneplan import validate_model_sceneplan
from .model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from .sceneplan_transfusion_editing import sha256_json
from .sceneplan_transfusion_editing_dataset import (
    EDITING_INDEX_SCHEMA,
    EDITING_INDEX_SCHEMA_VERSION,
    EDITING_INDEX_STATE,
)
from .sceneplan_transfusion_editing_index import sha256_file
from .sceneplan_transfusion_editing_plan import canonicalize_editing_plan


EDITING_AR_DATASET_CONTRACT = "audio_reference_instruction_to_canonical_new_sceneplan_v3"
EDITING_AR_SELECT_COLUMNS = (
    "pair_ordinal",
    "pair_id",
    "operation",
    "raw_edit_request",
    "new_sceneplan_zlib",
    "new_sceneplan_sha256",
    "source_latent_path",
    "source_latent_key",
    "source_latent_tensor_sha256",
    "latent_frames_valid",
    "latent_bucket_frames",
)
FORBIDDEN_AR_PLAN_KEYS = {
    "old_sceneplan",
    "old_plan",
    "source_sceneplan",
    "source_plan",
    "previous_sceneplan",
    "previous_plan",
}


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


class ScenePlanTransfusionEditingARDataset(torch.utils.data.Dataset):
    """Read exactly ``reference latent + instruction -> new plan`` rows."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        codec: ModelScenePlanCodecV4,
        expected_num_samples: int,
        expected_index_sha256: str | None = None,
        row_ordinals: Sequence[int] | None = None,
        latent_crop_length: int = 648,
        max_plan_tokens: int = 1024,
        verify_tensor_hashes_on_access: bool = False,
    ) -> None:
        super().__init__()
        self.index_path = Path(index_path).expanduser().resolve(strict=True)
        self.codec = codec
        self.latent_crop_length = int(latent_crop_length)
        self.max_plan_tokens = int(max_plan_tokens)
        self.verify_tensor_hashes_on_access = bool(verify_tensor_hashes_on_access)
        if self.latent_crop_length != 648:
            raise ValueError("Editing AR uses the complete <=648-frame reference")
        if self.max_plan_tokens <= 1:
            raise ValueError("Editing AR max_plan_tokens must exceed one")

        actual_sha = sha256_file(self.index_path)
        if expected_index_sha256 is not None and actual_sha != str(
            expected_index_sha256
        ):
            raise RuntimeError("Editing AR source-index SHA256 mismatch")
        marker_path = self.index_path.with_suffix(
            self.index_path.suffix + ".frozen.json"
        )
        if not marker_path.is_file():
            raise RuntimeError("Editing AR requires a frozen Editing index")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if (
            marker.get("schema") != EDITING_INDEX_SCHEMA
            or int(marker.get("schema_version", -1)) != int(
                EDITING_INDEX_SCHEMA_VERSION
            )
            or marker.get("state") != EDITING_INDEX_STATE
            or marker.get("index_sha256") != actual_sha
        ):
            raise RuntimeError("Editing AR frozen-index marker mismatch")

        connection = _readonly_connection(self.index_path)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            stored_rows = int(
                connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]
            )
        finally:
            connection.close()
        if (
            metadata.get("schema") != EDITING_INDEX_SCHEMA
            or metadata.get("schema_version") != EDITING_INDEX_SCHEMA_VERSION
            or metadata.get("state") != EDITING_INDEX_STATE
            or int(metadata.get("rows", -1)) != stored_rows
        ):
            raise RuntimeError("Editing AR source index contract mismatch")

        if row_ordinals is None:
            if int(expected_num_samples) != stored_rows:
                raise RuntimeError("Editing AR expected row count changed")
            self.row_ordinals: tuple[int, ...] | None = None
            self._length = stored_rows
        else:
            ordinals = tuple(int(value) for value in row_ordinals)
            if (
                len(ordinals) != int(expected_num_samples)
                or len(set(ordinals)) != len(ordinals)
                or any(value < 0 or value >= stored_rows for value in ordinals)
            ):
                raise ValueError("Editing AR row ordinals are invalid")
            self.row_ordinals = ordinals
            self._length = len(ordinals)
        self._connection: sqlite3.Connection | None = None

    def __len__(self) -> int:
        return self._length

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = _readonly_connection(self.index_path)
        return self._connection

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        return state

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= int(index) < self._length:
            raise IndexError(index)
        ordinal = (
            int(index)
            if self.row_ordinals is None
            else self.row_ordinals[int(index)]
        )
        select = ",".join(EDITING_AR_SELECT_COLUMNS)
        row = self._db().execute(
            f"SELECT {select} FROM pairs WHERE pair_ordinal = ?", (ordinal,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing Editing AR pair ordinal {ordinal}")
        values = dict(zip(EDITING_AR_SELECT_COLUMNS, row))
        pair_id = str(values["pair_id"])
        try:
            new_plan = json.loads(zlib.decompress(values["new_sceneplan_zlib"]))
        except (TypeError, ValueError, zlib.error) as error:
            raise RuntimeError(f"{pair_id}: invalid new ScenePlan target") from error
        validate_model_sceneplan(new_plan)
        if sha256_json(new_plan) != str(values["new_sceneplan_sha256"]):
            raise RuntimeError(f"{pair_id}: new ScenePlan target hash mismatch")
        model_plan, _ = canonicalize_editing_plan(new_plan, codec=self.codec)
        encoded = self.codec.encode(model_plan, max_tokens=self.max_plan_tokens)

        valid_frames = int(values["latent_frames_valid"])
        bucket = int(values["latent_bucket_frames"])
        if bucket not in (432, 648) or not 1 <= valid_frames <= bucket:
            raise RuntimeError(f"{pair_id}: invalid source latent frame geometry")
        with safe_open(
            str(values["source_latent_path"]), framework="pt", device="cpu"
        ) as handle:
            key = str(values["source_latent_key"])
            if key not in handle.keys():
                raise RuntimeError(f"{pair_id}: source latent key is absent")
            source = handle.get_tensor(key).clone()
        if source.dtype != torch.float16 or tuple(source.shape) != (
            64,
            valid_frames,
        ):
            raise RuntimeError(f"{pair_id}: source latent shape/dtype changed")
        if not torch.isfinite(source).all().item():
            raise RuntimeError(f"{pair_id}: source latent contains non-finite values")
        if self.verify_tensor_hashes_on_access and _tensor_sha256(source) != str(
            values["source_latent_tensor_sha256"]
        ):
            raise RuntimeError(f"{pair_id}: source latent tensor hash mismatch")
        padded = torch.zeros(64, self.latent_crop_length, dtype=source.dtype)
        padded[:, :valid_frames] = source
        source_mask = torch.zeros(self.latent_crop_length, dtype=torch.bool)
        source_mask[:valid_frames] = True
        return {
            "pair_ordinal": ordinal,
            "pair_id": pair_id,
            "operation": str(values["operation"]),
            "raw_edit_request": str(values["raw_edit_request"]),
            "source_foa_latent": padded,
            "source_attention_mask": source_mask,
            "target_token_ids": encoded["input_ids"],
            "target_loss_group_ids": encoded["loss_group_ids"],
            "source_valid_frames": valid_frames,
            # Keep plan-token absolute/rotary positions identical across the
            # AR-only reader, joint AR+RF training, and inference.
            "source_prefix_frames": bucket,
        }


def collate_editing_ar(
    rows: Sequence[dict[str, Any]], *, pad_id: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty Editing AR batch")
    for row in rows:
        normalized = {
            str(key).lower().replace("-", "_") for key in row
        }
        leaked = normalized & FORBIDDEN_AR_PLAN_KEYS
        if leaked:
            raise RuntimeError(
                "old ScenePlan is forbidden in Editing AR rows "
                "(including source/previous aliases): "
                f"{sorted(leaked)}"
            )
    plan_lengths = [int(row["target_token_ids"].numel()) - 1 for row in rows]
    maximum = max(plan_lengths)
    batch = len(rows)
    source_prefix_frames = [
        int(row.get("source_prefix_frames", row["source_valid_frames"]))
        for row in rows
    ]
    for row, prefix_frames in zip(rows, source_prefix_frames):
        valid_frames = int(row["source_valid_frames"])
        source = row["source_foa_latent"]
        source_mask = row["source_attention_mask"]
        if (
            not valid_frames <= prefix_frames <= int(source.shape[-1])
            or int(source_mask.shape[-1]) < prefix_frames
        ):
            raise RuntimeError("Editing AR source prefix envelope is invalid")
    maximum_source_frames = max(source_prefix_frames)
    input_ids = torch.full((batch, maximum), int(pad_id), dtype=torch.long)
    labels = torch.full((batch, maximum), -100, dtype=torch.long)
    loss_groups = torch.full((batch, maximum), -1, dtype=torch.long)
    plan_mask = torch.zeros((batch, maximum), dtype=torch.bool)
    for index, (row, length) in enumerate(zip(rows, plan_lengths)):
        tokens = row["target_token_ids"].to(torch.long)
        groups = row["target_loss_group_ids"].to(torch.long)
        input_ids[index, :length] = tokens[:-1]
        labels[index, :length] = tokens[1:]
        loss_groups[index, :length] = groups[1:]
        plan_mask[index, :length] = True
    return {
        "source_foa_latent": torch.stack(
            [row["source_foa_latent"] for row in rows]
        )[:, :, :maximum_source_frames],
        "source_attention_mask": torch.stack(
            [row["source_attention_mask"] for row in rows]
        )[:, :maximum_source_frames],
        "raw_edit_requests": [row["raw_edit_request"] for row in rows],
        "plan_input_ids": input_ids,
        "plan_labels": labels,
        "plan_loss_group_ids": loss_groups,
        "plan_attention_mask": plan_mask,
        "pair_ids": [row["pair_id"] for row in rows],
        "pair_ordinals": torch.tensor(
            [int(row["pair_ordinal"]) for row in rows], dtype=torch.long
        ),
        "source_valid_frames": torch.tensor(
            [int(row["source_valid_frames"]) for row in rows], dtype=torch.long
        ),
        "source_prefix_frames": torch.tensor(
            source_prefix_frames, dtype=torch.long
        ),
        "target_lengths": torch.tensor(plan_lengths, dtype=torch.long),
    }


__all__ = [
    "EDITING_AR_DATASET_CONTRACT",
    "EDITING_AR_SELECT_COLUMNS",
    "FORBIDDEN_AR_PLAN_KEYS",
    "ScenePlanTransfusionEditingARDataset",
    "collate_editing_ar",
]
