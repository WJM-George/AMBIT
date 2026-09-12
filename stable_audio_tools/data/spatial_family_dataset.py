"""Efficient sharded reader for multi-turn Spatial-CoT latent families."""
from __future__ import annotations

import json
import sqlite3
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch
from safetensors import safe_open

from stable_audio_tools.data.spatial_caption_templates import (
    SEMANTIC_CAPTION_TEMPLATE_VERSION,
    validate_semantic_caption_metadata,
)


class _CaptionOverlay:
    """Indexed, versioned semantic captions layered over immutable latents."""

    def __init__(self, path: str | Path, *, require_ready: bool, max_open_shards: int):
        self.path = Path(path).expanduser().resolve()
        ready_path = self.path / "READY"
        index_path = self.path / "index.sqlite"
        if require_ready and not ready_path.is_file():
            raise RuntimeError(f"Spatial caption overlay is not READY: {self.path}")
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if ready.get("template_version") != SEMANTIC_CAPTION_TEMPLATE_VERSION:
            raise RuntimeError(
                "Spatial caption overlay/template version mismatch: "
                f"{ready.get('template_version')!r} != "
                f"{SEMANTIC_CAPTION_TEMPLATE_VERSION!r}"
            )
        self.index_path = index_path
        self.max_open_shards = max(1, int(max_open_shards))
        self._connection: Optional[sqlite3.Connection] = None
        self._shards: OrderedDict[str, Any] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_shards"] = OrderedDict()
        return state

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        for handle in self._shards.values():
            handle.close()
        self._shards.clear()

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            uri = f"file:{self.index_path.as_posix()}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        return self._connection

    def _shard(self, relative: str):
        handle = self._shards.pop(relative, None)
        if handle is None:
            path = self.path / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            handle = path.open("rb")
        self._shards[relative] = handle
        while len(self._shards) > self.max_open_shards:
            _, old = self._shards.popitem(last=False)
            old.close()
        return handle

    def apply(self, family: dict[str, Any]) -> None:
        family_id = str(family.get("family_id") or "")
        row = self._connect().execute(
            "SELECT caption_shard, caption_offset, caption_length, num_turns "
            "FROM captions WHERE family_id = ?",
            (family_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"caption overlay has no family {family_id!r}")
        shard, offset, length, num_turns = row
        handle = self._shard(str(shard))
        handle.seek(int(offset))
        payload = handle.read(int(length))
        if len(payload) != int(length):
            raise RuntimeError(f"short caption-overlay read for {family_id}")
        overlay = json.loads(payload)
        if (
            overlay.get("template_version") != SEMANTIC_CAPTION_TEMPLATE_VERSION
            or str(overlay.get("family_id")) != family_id
        ):
            raise RuntimeError(f"caption-overlay identity mismatch for {family_id}")
        family_turns = family.get("turns")
        overlay_turns = overlay.get("turns")
        if (
            not isinstance(family_turns, list)
            or not isinstance(overlay_turns, list)
            or len(family_turns) != int(num_turns)
            or len(overlay_turns) != len(family_turns)
        ):
            raise RuntimeError(f"caption-overlay turn mismatch for {family_id}")
        for turn_index, (turn, replacement) in enumerate(
            zip(family_turns, overlay_turns)
        ):
            if str(turn.get("turn_id")) != str(replacement.get("turn_id")):
                raise RuntimeError(
                    f"caption-overlay turn ID mismatch for {family_id} turn {turn_index}"
                )
            caption = str(replacement.get("semantic_caption") or "")
            metadata = replacement.get("semantic_caption_metadata")
            if not caption or not isinstance(metadata, dict):
                raise RuntimeError(
                    f"caption-overlay content missing for {family_id} turn {turn_index}"
                )
            sources = (
                ((((turn.get("after") or {}).get("scene_plan") or {}).get("scene") or {}))
                .get("sources")
                or []
            )
            expected_source_ids = [
                str(source.get("source_id") or f"source_{index}")
                for index, source in enumerate(sources)
            ]
            validate_semantic_caption_metadata(
                caption,
                metadata,
                expected_source_ids=expected_source_ids,
            )
            turn["semantic_caption"] = caption
            turn["semantic_caption_metadata"] = metadata
        family["semantic_caption_overlay"] = {
            "path": str(self.path),
            "template_version": SEMANTIC_CAPTION_TEMPLATE_VERSION,
            "template_id": int(overlay["template_id"]),
        }


class _FamilyRoot:
    def __init__(
        self,
        path: str | Path,
        *,
        family_ranks: Optional[Sequence[int]],
        custom_metadata_fn: Optional[Callable],
        caption_overlay_path: Optional[str | Path],
        require_ready: bool,
        max_open_shards: int,
    ):
        self.path = Path(path).expanduser().resolve()
        ready_path = self.path / "READY"
        index_path = self.path / "index.sqlite"
        if require_ready and not ready_path.is_file():
            raise RuntimeError(f"Spatial family store is not READY: {self.path}")
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        ready = json.loads(ready_path.read_text(encoding="utf-8")) if ready_path.is_file() else {}
        stored_count = int(ready.get("families", 0))
        if stored_count <= 0:
            with sqlite3.connect(index_path) as connection:
                stored_count = int(
                    connection.execute("SELECT COUNT(*) FROM families").fetchone()[0]
                )
        if stored_count <= 0:
            raise RuntimeError(f"Spatial family store contains no rows: {self.path}")
        self.family_ranks: Optional[tuple[int, ...]] = None
        if family_ranks is not None:
            if isinstance(family_ranks, (str, bytes)) or not isinstance(
                family_ranks, Sequence
            ):
                raise TypeError("family_ranks must be a sequence of integers")
            selected: list[int] = []
            for rank in family_ranks:
                if isinstance(rank, bool) or not isinstance(rank, int):
                    raise TypeError("family_ranks must contain only integers")
                if rank < 0:
                    raise ValueError("family_ranks must be non-negative")
                selected.append(rank)
            if not selected:
                raise ValueError("family_ranks must not be empty")
            if len(set(selected)) != len(selected):
                raise ValueError("family_ranks must not contain duplicates")
            with sqlite3.connect(index_path) as connection:
                missing = [
                    rank
                    for rank in selected
                    if connection.execute(
                        "SELECT 1 FROM families WHERE family_rank = ?", (rank,)
                    ).fetchone()
                    is None
                ]
            if missing:
                raise IndexError(
                    f"family_ranks do not exist in {self.path}: {missing}"
                )
            self.family_ranks = tuple(selected)
            self.count = len(selected)
        else:
            self.count = stored_count
        self.index_path = index_path
        self.custom_metadata_fn = custom_metadata_fn
        self.max_open_shards = max(1, int(max_open_shards))
        self.caption_overlay = (
            _CaptionOverlay(
                caption_overlay_path,
                require_ready=require_ready,
                max_open_shards=max_open_shards,
            )
            if caption_overlay_path is not None
            else None
        )
        self._connection: Optional[sqlite3.Connection] = None
        self._tensor_shards: OrderedDict[str, Any] = OrderedDict()
        self._metadata_shards: OrderedDict[str, Any] = OrderedDict()

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_tensor_shards"] = OrderedDict()
        state["_metadata_shards"] = OrderedDict()
        return state

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._tensor_shards.clear()
        for handle in self._metadata_shards.values():
            handle.close()
        self._metadata_shards.clear()
        if self.caption_overlay is not None:
            self.caption_overlay.close()

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            uri = f"file:{self.index_path.as_posix()}?mode=ro&immutable=1"
            self._connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
        return self._connection

    def _tensor_shard(self, relative: str):
        handle = self._tensor_shards.pop(relative, None)
        if handle is None:
            path = self.path / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            handle = safe_open(str(path), framework="pt", device="cpu")
        self._tensor_shards[relative] = handle
        while len(self._tensor_shards) > self.max_open_shards:
            self._tensor_shards.popitem(last=False)
        return handle

    def _metadata_shard(self, relative: str):
        handle = self._metadata_shards.pop(relative, None)
        if handle is None:
            path = self.path / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            handle = path.open("rb")
        self._metadata_shards[relative] = handle
        while len(self._metadata_shards) > self.max_open_shards:
            _, old = self._metadata_shards.popitem(last=False)
            old.close()
        return handle

    def get(self, rank: int) -> tuple[torch.Tensor, dict[str, Any]]:
        local_rank = int(rank)
        if not 0 <= local_rank < self.count:
            raise IndexError(rank)
        stored_rank = (
            local_rank
            if self.family_ranks is None
            else self.family_ranks[local_rank]
        )
        row = self._connect().execute(
            "SELECT family_id, tensor_shard, tensor_key, metadata_shard, "
            "metadata_offset, metadata_length, num_turns, channels, frames, dtype "
            "FROM families WHERE family_rank = ?",
            (stored_rank,),
        ).fetchone()
        if row is None:
            raise IndexError(rank)
        (
            family_id,
            tensor_shard,
            tensor_key,
            metadata_shard,
            metadata_offset,
            metadata_length,
            num_turns,
            channels,
            frames,
            dtype,
        ) = row
        tensor = self._tensor_shard(str(tensor_shard)).get_tensor(str(tensor_key))
        expected_shape = (int(num_turns), int(channels), int(frames))
        if tensor.ndim != 3 or tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"corrupt family tensor {family_id}: {tuple(tensor.shape)} != {expected_shape}"
            )
        if str(tensor.dtype).removeprefix("torch.") != str(dtype):
            raise RuntimeError(
                f"family tensor dtype mismatch for {family_id}: {tensor.dtype} != {dtype}"
            )
        handle = self._metadata_shard(str(metadata_shard))
        handle.seek(int(metadata_offset))
        payload = handle.read(int(metadata_length))
        if len(payload) != int(metadata_length):
            raise RuntimeError(f"short family metadata read for {family_id}")
        record = json.loads(payload)
        if str(record.get("family_id")) != str(family_id):
            raise RuntimeError(
                f"family index/metadata mismatch: {family_id} != {record.get('family_id')}"
            )
        if self.caption_overlay is not None:
            self.caption_overlay.apply(record)
        info: dict[str, Any] = {
            "family_id": str(family_id),
            "path": f"{self.path.as_posix()}/{family_id}",
            "spatial_family": record,
        }
        if self.custom_metadata_fn is not None:
            output = self.custom_metadata_fn(info, tensor)
            if not isinstance(output, dict):
                raise TypeError("Spatial family metadata provider must return a dict")
            info.update(output)
        return tensor, info


class SpatialFamilyDataset(torch.utils.data.Dataset):
    """Map-style union of one or more immutable family stores."""

    def __init__(
        self,
        stores: Sequence[dict[str, Any]],
        *,
        require_ready: bool = True,
        max_open_shards: int = 4,
    ):
        if not stores:
            raise ValueError("SpatialFamilyDataset requires at least one store")
        self.roots = [
            _FamilyRoot(
                item["path"],
                family_ranks=item.get("family_ranks"),
                custom_metadata_fn=item.get("custom_metadata_fn"),
                caption_overlay_path=item.get("caption_overlay_path"),
                require_ready=require_ready,
                max_open_shards=max_open_shards,
            )
            for item in stores
        ]
        self.ends = []
        total = 0
        for root in self.roots:
            total += root.count
            self.ends.append(total)
        self.count = total

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        root_index = bisect_right(self.ends, index)
        start = 0 if root_index == 0 else self.ends[root_index - 1]
        return self.roots[root_index].get(index - start)


__all__ = ["SpatialFamilyDataset"]
