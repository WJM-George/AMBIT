"""SQLite-backed exact targets for P10-v11 shared-block Generation AR."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


GENERATION_AR_MANIFEST_SCHEMA = (
    "stable_audio_tools.sceneplan_transfusion_generation_ar"
)
GENERATION_AR_MANIFEST_SCHEMA_VERSION = "1"
GENERATION_AR_CODEC_FINGERPRINT = (
    "b512c31b96c6775af6e46e9a5cdf3d513658dcab61b6d357853a019ac16e3874"
)
GENERATION_AR_SOURCE_INDEX_SHA256 = {
    "train": "ca65d8bd80f7060ea21a3c12135c0bfc2f3a21afacf35ebda87e09d9da4c033d",
    "validation": "8cfc9a21f11e3d271b87319c0258923a91c8219febb1dcd47c2b051612dd2ba1",
    "test": "d90b00c4f28fd3395d1b04dbe46ea7e87a08ca72e8e2491aaf8c69227b7ae90c",
}


def _readonly_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


class GenerationARSQLiteDataset(Dataset):
    """Lazy per-worker access without copying the immutable manifest."""

    def __init__(
        self,
        path: str | Path,
        *,
        split: str,
        row_ordinals: Sequence[int] | None = None,
    ):
        self.path = Path(path).expanduser().resolve(strict=True)
        self.split = str(split)
        connection = _readonly_connection(self.path)
        try:
            self.metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            stored_rows = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
        finally:
            connection.close()
        numeric_contract = self.metadata.get("raw_request_numeric_contract")
        supported_request = numeric_contract == "codec_v4_reversible_seconds_ms_mm_v1"
        if numeric_contract == "request_qualitative_compass_time_v3":
            # Qualitative captions deliberately omit GT coordinates and seconds.
            # Require the complete audited rebuild rather than accepting an
            # arbitrary manifest merely because it names this contract.
            supported_request = (
                self.metadata.get("raw_request_contract")
                == "sceneplan_generation_templated_qualitative_v3"
                and self.metadata.get("coordinate_and_activity_numbers_in_request") == "false"
                and self.metadata.get("hidden_gt_numbers_are_unique_answers") == "false"
                and self.metadata.get("target_bytes_unchanged") == "true"
                and self.metadata.get("template_count")
                == {"train": "100", "validation": "10", "test": "5"}.get(self.split)
                and stored_rows == {"train": 1600000, "validation": 32000, "test": 8000}.get(self.split)
                and len(self.metadata.get("template_catalog_sha256", "")) == 64
                and len(self.metadata.get("build_contract_sha256", "")) == 64
            )
        if (
            self.metadata.get("schema") != GENERATION_AR_MANIFEST_SCHEMA
            or self.metadata.get("schema_version")
            != GENERATION_AR_MANIFEST_SCHEMA_VERSION
            or self.metadata.get("split") != self.split
            or self.metadata.get("is_full_split") != "true"
            or self.metadata.get("raw_request_truncation_count") != "0"
            or self.metadata.get("seed") != "42"
            or self.metadata.get("codec_fingerprint")
            != GENERATION_AR_CODEC_FINGERPRINT
            or self.metadata.get("source_index_sha256")
            != GENERATION_AR_SOURCE_INDEX_SHA256.get(self.split)
            or self.metadata.get("raw_request_qwen_token_limit") != "512"
            or not supported_request
            or int(self.metadata.get("rows", -1)) != stored_rows
        ):
            raise RuntimeError(
                f"Generation AR manifest contract mismatch: {self.path}"
            )
        if row_ordinals is None:
            self.row_ordinals: np.ndarray | None = None
            self._length = stored_rows
        else:
            values = np.asarray(tuple(int(value) for value in row_ordinals), dtype=np.int64)
            if (
                values.ndim != 1
                or len(values) == 0
                or int(values.min()) < 0
                or int(values.max()) >= stored_rows
                or len(set(values.tolist())) != len(values)
            ):
                raise ValueError("row_ordinals must be unique valid output ordinals")
            self.row_ordinals = values
            self._length = len(values)
        self._connection: sqlite3.Connection | None = None

    def __len__(self) -> int:
        return self._length

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = _readonly_connection(self.path)
        return self._connection

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        return state

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def __del__(self):
        self.close()

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= int(index) < self._length:
            raise IndexError(index)
        ordinal = (
            int(index)
            if self.row_ordinals is None
            else int(self.row_ordinals[int(index)])
        )
        row = self._connect().execute(
            """
            SELECT ordinal, sample_id, raw_user_request,
                   raw_request_qwen_tokens, target_token_ids_u16le,
                   target_loss_group_ids_i16le, target_token_count,
                   template_id, source_count
            FROM rows WHERE ordinal = ?
            """,
            (ordinal,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing Generation AR row ordinal {ordinal}")
        (
            stored_ordinal,
            sample_id,
            request,
            request_token_count,
            token_blob,
            group_blob,
            target_token_count,
            template_id,
            source_count,
        ) = row
        token_ids = np.frombuffer(token_blob, dtype="<u2").astype(np.int64)
        loss_groups = np.frombuffer(group_blob, dtype="<i2").astype(np.int64)
        if (
            int(stored_ordinal) != ordinal
            or len(token_ids) != int(target_token_count)
            or len(loss_groups) != int(target_token_count)
            or len(token_ids) < 2
        ):
            raise RuntimeError(f"malformed Generation AR row ordinal {ordinal}")
        return {
            "ordinal": ordinal,
            "sample_id": str(sample_id),
            "raw_user_request": str(request),
            "raw_request_qwen_tokens": int(request_token_count),
            "target_token_ids": torch.from_numpy(token_ids),
            "target_loss_group_ids": torch.from_numpy(loss_groups),
            "template_id": str(template_id),
            "source_count": int(source_count),
        }


def collate_generation_ar(
    rows: Sequence[dict[str, Any]], *, pad_id: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty Generation AR batch")
    lengths = [int(row["target_token_ids"].numel()) - 1 for row in rows]
    maximum = max(lengths)
    batch = len(rows)
    input_ids = torch.full((batch, maximum), int(pad_id), dtype=torch.long)
    labels = torch.full((batch, maximum), -100, dtype=torch.long)
    loss_group_ids = torch.full((batch, maximum), -1, dtype=torch.long)
    attention_mask = torch.zeros((batch, maximum), dtype=torch.bool)
    for index, (row, length) in enumerate(zip(rows, lengths)):
        tokens = row["target_token_ids"].to(torch.long)
        groups = row["target_loss_group_ids"].to(torch.long)
        input_ids[index, :length] = tokens[:-1]
        labels[index, :length] = tokens[1:]
        loss_group_ids[index, :length] = groups[1:]
        attention_mask[index, :length] = True
    return {
        "plan_input_ids": input_ids,
        "plan_labels": labels,
        "plan_loss_group_ids": loss_group_ids,
        "plan_attention_mask": attention_mask,
        "raw_user_requests": [row["raw_user_request"] for row in rows],
        "sample_ids": [row["sample_id"] for row in rows],
        "ordinals": torch.tensor([row["ordinal"] for row in rows]),
        "source_counts": torch.tensor([row["source_count"] for row in rows]),
        "target_lengths": torch.tensor(lengths),
    }


def select_tiny_overfit_ordinals(
    path: str | Path, *, rows: int
) -> tuple[int, ...]:
    """Choose a deterministic hard set spanning all available source counts."""

    requested = int(rows)
    if requested < 4:
        raise ValueError("tiny overfit requires at least four rows")
    manifest = Path(path).expanduser().resolve(strict=True)
    connection = _readonly_connection(manifest)
    selected: list[int] = []
    try:
        source_counts = [
            int(value[0])
            for value in connection.execute(
                "SELECT DISTINCT source_count FROM rows ORDER BY source_count"
            )
        ]
        for source_count in source_counts:
            selected.extend(
                int(value[0])
                for value in connection.execute(
                    """
                    SELECT ordinal FROM rows
                    WHERE source_count = ?
                    ORDER BY raw_request_qwen_tokens DESC, target_token_count DESC,
                             ordinal
                    LIMIT ?
                    """,
                    (source_count, max(1, requested // len(source_counts))),
                )
            )
        if len(selected) < requested:
            placeholders = ",".join("?" for _ in selected)
            query = (
                "SELECT ordinal FROM rows "
                + (f"WHERE ordinal NOT IN ({placeholders}) " if selected else "")
                + "ORDER BY raw_request_qwen_tokens DESC, target_token_count DESC, ordinal "
                + "LIMIT ?"
            )
            selected.extend(
                int(value[0])
                for value in connection.execute(
                    query, (*selected, requested - len(selected))
                )
            )
    finally:
        connection.close()
    result = tuple(selected[:requested])
    if len(result) != requested or len(set(result)) != requested:
        raise RuntimeError("could not select the requested unique tiny rows")
    return result


def load_target_token_lengths(path: str | Path) -> np.ndarray:
    manifest = Path(path).expanduser().resolve(strict=True)
    connection = _readonly_connection(manifest)
    try:
        row_count = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
        lengths = np.empty(row_count, dtype=np.int32)
        for expected, (ordinal, target_count) in enumerate(
            connection.execute(
                "SELECT ordinal,target_token_count FROM rows ORDER BY ordinal"
            )
        ):
            if int(ordinal) != expected:
                raise RuntimeError("Generation AR output ordinals are not contiguous")
            lengths[expected] = int(target_count) - 1
    finally:
        connection.close()
    if len(lengths) == 0 or int(lengths.min()) <= 0:
        raise RuntimeError("Generation AR manifest has invalid target lengths")
    return lengths


class LengthBucketDistributedSampler(Sampler[int]):
    """Deterministic, disjoint DDP batches with exact row coverage.

    Full global batches are length bucketed as before.  A non-divisible tail is
    distributed across ranks without padding or duplication; all ranks must
    receive at least one tail row so that they execute the same number of DDP
    steps.  Local final-batch sizes may differ by one, which DDP supports.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        *,
        num_replicas: int,
        rank: int,
        batch_size: int,
        seed: int = 42,
        bucket_batches: int = 64,
    ):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.bucket_batches = int(bucket_batches)
        if (
            self.lengths.ndim != 1
            or len(self.lengths) == 0
            or self.num_replicas <= 0
            or not 0 <= self.rank < self.num_replicas
            or self.batch_size <= 0
            or self.bucket_batches <= 0
        ):
            raise ValueError("invalid length-bucket distributed sampler settings")
        self.global_batch = self.num_replicas * self.batch_size
        self.full_rows = (len(self.lengths) // self.global_batch) * self.global_batch
        self.tail_rows = len(self.lengths) - self.full_rows
        if self.full_rows == 0:
            raise ValueError("dataset is smaller than one distributed batch")
        if 0 < self.tail_rows < self.num_replicas:
            raise ValueError(
                "non-divisible dataset tail must contain at least one row per rank"
            )
        self.sorted_indices = np.argsort(self.lengths, kind="stable")
        self.samples_for_rank = (
            self.full_rows // self.num_replicas
            + self.tail_rows // self.num_replicas
            + int(self.rank < self.tail_rows % self.num_replicas)
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_for_rank

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        ordered = self.sorted_indices
        tail = np.empty(0, dtype=ordered.dtype)
        if self.tail_rows:
            tail_positions = rng.choice(
                len(ordered), size=self.tail_rows, replace=False
            )
            keep = np.ones(len(ordered), dtype=np.bool_)
            keep[tail_positions] = False
            tail = ordered[tail_positions]
            ordered = ordered[keep]
        bucket_size = self.global_batch * self.bucket_batches
        buckets = [
            ordered[start : start + bucket_size]
            for start in range(0, len(ordered), bucket_size)
        ]
        for bucket_index in rng.permutation(len(buckets)):
            bucket = rng.permutation(buckets[int(bucket_index)])
            global_batches = bucket.reshape(-1, self.global_batch)
            for global_batch in global_batches:
                start = self.rank * self.batch_size
                stop = start + self.batch_size
                for index in global_batch[start:stop]:
                    yield int(index)
        if self.tail_rows:
            for index in rng.permutation(tail)[self.rank :: self.num_replicas]:
                yield int(index)


def manifest_summary(path: str | Path) -> dict[str, Any]:
    manifest = Path(path).expanduser().resolve(strict=True)
    connection = _readonly_connection(manifest)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        distributions = {
            "source_count": dict(
                connection.execute(
                    "SELECT source_count,COUNT(*) FROM rows GROUP BY source_count"
                )
            ),
            "template": dict(
                connection.execute(
                    "SELECT template_id,COUNT(*) FROM rows GROUP BY template_id"
                )
            ),
        }
    finally:
        connection.close()
    return {
        "path": str(manifest),
        "metadata": metadata,
        "distributions": json.loads(json.dumps(distributions)),
    }


__all__ = [
    "GENERATION_AR_MANIFEST_SCHEMA",
    "GenerationARSQLiteDataset",
    "LengthBucketDistributedSampler",
    "collate_generation_ar",
    "load_target_token_lengths",
    "manifest_summary",
    "select_tiny_overfit_ordinals",
]
