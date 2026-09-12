"""Versioned, sharded artifact helpers for the T2A data pipeline.

The canonical identity is derived from the normalized source-audio path rather
than the rank/batch-dependent latent filename.  JSONL remains the portable audit
format; SQLite is the worker-friendly random-access index; tensors live in a
small number of safetensors shards rather than millions of tiny NumPy files.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from safetensors import safe_open


def normalize_audio_path(value: str | os.PathLike[str]) -> str:
    """Return the stable path spelling used by all T2A artifact indexes."""

    text = os.fspath(value)
    if not text:
        raise ValueError("audio path must not be empty")
    return os.path.normpath(os.path.abspath(os.path.expanduser(text)))


def stable_sample_id(audio_path: str | os.PathLike[str]) -> str:
    """Content-independent stable ID shared by trajectory and ScenePlan stores."""

    normalized = normalize_audio_path(audio_path)
    digest = hashlib.blake2b(normalized.encode("utf-8"), digest_size=12).hexdigest()
    return f"t2a_{digest}"


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(
    path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]
) -> tuple[int, str]:
    """Atomically write JSONL and return ``(row_count, sha256)``."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, text=True
    )
    temporary = Path(temporary_name)
    count = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                line = json.dumps(
                    row, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return count, digest.hexdigest()


class ShardedTensorStore:
    """Read tensors from a safetensors-shard store indexed by SQLite.

    Runtime resources are opened lazily in each DataLoader worker and discarded
    when the provider is pickled.  A small LRU avoids reopening shard files for
    nearby samples while bounding file descriptors during globally shuffled
    training.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        require_ready: bool = True,
        max_open_shards: int = 16,
    ):
        self.root = Path(root).expanduser().resolve()
        self.index_path = self.root / "index.sqlite"
        self.ready_path = self.root / "READY"
        self.max_open_shards = max(1, int(max_open_shards))
        if require_ready and not self.ready_path.is_file():
            raise RuntimeError(f"artifact store is not finalized: missing {self.ready_path}")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"artifact index does not exist: {self.index_path}")
        self._connection: Optional[sqlite3.Connection] = None
        self._shards: OrderedDict[str, Any] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_shards"] = OrderedDict()
        return state

    def close(self) -> None:
        """Release worker-local SQLite and shard resources deterministically."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None
        # ``safe_open`` releases its mmap when the last Python reference is
        # dropped; unlike regular file objects it does not expose ``close``.
        self._shards.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Interpreter shutdown may already have torn down extension types.
            pass

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            uri = f"file:{self.index_path.as_posix()}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        return self._connection

    def _open_shard(self, relative_path: str):
        handle = self._shards.pop(relative_path, None)
        if handle is None:
            shard_path = self.root / relative_path
            if not shard_path.is_file():
                raise FileNotFoundError(f"artifact shard does not exist: {shard_path}")
            handle = safe_open(str(shard_path), framework="pt", device="cpu")
        self._shards[relative_path] = handle
        while len(self._shards) > self.max_open_shards:
            self._shards.popitem(last=False)
        return handle

    def lookup(self, audio_path: str | os.PathLike[str]) -> dict[str, Any]:
        normalized = normalize_audio_path(audio_path)
        row = self._connect().execute(
            "SELECT sample_id, shard, tensor_key, num_frames, channels, dtype "
            "FROM samples WHERE audio_path = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(f"audio path is absent from artifact index: {normalized}")
        return {
            "sample_id": row[0],
            "shard": row[1],
            "tensor_key": row[2],
            "num_frames": int(row[3]),
            "channels": int(row[4]),
            "dtype": row[5],
        }

    def get_tensor(
        self,
        audio_path: str | os.PathLike[str],
        *,
        expected_frames: Optional[int] = None,
        expected_channels: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        entry = self.lookup(audio_path)
        if expected_frames is not None and entry["num_frames"] != int(expected_frames):
            raise RuntimeError(
                f"artifact frame mismatch for {audio_path}: index={entry['num_frames']} "
                f"expected={int(expected_frames)}"
            )
        if expected_channels is not None and entry["channels"] != int(expected_channels):
            raise RuntimeError(
                f"artifact channel mismatch for {audio_path}: index={entry['channels']} "
                f"expected={int(expected_channels)}"
            )
        tensor = self._open_shard(entry["shard"]).get_tensor(entry["tensor_key"])
        if tensor.ndim != 2 or tuple(tensor.shape) != (
            entry["num_frames"],
            entry["channels"],
        ):
            raise RuntimeError(
                f"corrupt artifact tensor {entry['tensor_key']} in {entry['shard']}: "
                f"shape={tuple(tensor.shape)} index={(entry['num_frames'], entry['channels'])}"
            )
        return tensor, entry


class IndexedJsonlStore:
    """Random-access reader for canonical JSONL shards with a SQLite index."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        require_ready: bool = True,
        max_open_shards: int = 8,
    ):
        self.root = Path(root).expanduser().resolve()
        self.index_path = self.root / "index.sqlite"
        self.ready_path = self.root / "READY"
        self.max_open_shards = max(1, int(max_open_shards))
        if require_ready and not self.ready_path.is_file():
            raise RuntimeError(f"JSONL store is not finalized: missing {self.ready_path}")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"JSONL index does not exist: {self.index_path}")
        self._connection: Optional[sqlite3.Connection] = None
        self._shards: OrderedDict[str, Any] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_shards"] = OrderedDict()
        return state

    def close(self) -> None:
        """Release worker-local SQLite connections and JSONL file handles."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None
        for handle in self._shards.values():
            handle.close()
        self._shards.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Be tolerant of partially initialized objects/interpreter teardown.
            pass

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            uri = f"file:{self.index_path.as_posix()}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        return self._connection

    def _open_shard(self, relative_path: str):
        handle = self._shards.pop(relative_path, None)
        if handle is None:
            shard_path = self.root / relative_path
            if not shard_path.is_file():
                raise FileNotFoundError(f"JSONL shard does not exist: {shard_path}")
            handle = shard_path.open("rb")
        self._shards[relative_path] = handle
        while len(self._shards) > self.max_open_shards:
            _, old_handle = self._shards.popitem(last=False)
            old_handle.close()
        return handle

    def lookup(self, audio_path: str | os.PathLike[str]) -> dict[str, Any]:
        normalized = normalize_audio_path(audio_path)
        row = self._connect().execute(
            "SELECT sample_id, shard, byte_offset, byte_length "
            "FROM samples WHERE audio_path = ?",
            (normalized,),
        ).fetchone()
        if row is None:
            raise KeyError(f"audio path is absent from JSONL index: {normalized}")
        return {
            "sample_id": row[0],
            "shard": row[1],
            "byte_offset": int(row[2]),
            "byte_length": int(row[3]),
        }

    def get(self, audio_path: str | os.PathLike[str]) -> dict[str, Any]:
        entry = self.lookup(audio_path)
        handle = self._open_shard(entry["shard"])
        handle.seek(entry["byte_offset"])
        payload = handle.read(entry["byte_length"])
        if len(payload) != entry["byte_length"]:
            raise RuntimeError(
                f"short JSONL read in {entry['shard']}: "
                f"expected={entry['byte_length']} got={len(payload)}"
            )
        record = json.loads(payload)
        if record.get("sample_id") != entry["sample_id"]:
            raise RuntimeError(
                f"JSONL index mismatch in {entry['shard']}: "
                f"index={entry['sample_id']} record={record.get('sample_id')}"
            )
        return record


__all__ = [
    "IndexedJsonlStore",
    "ShardedTensorStore",
    "atomic_write_json",
    "atomic_write_jsonl",
    "normalize_audio_path",
    "sha256_file",
    "stable_sample_id",
]
