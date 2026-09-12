"""Stable dry-source resolution for reproducible Spatial-CoT rendering.

Sound and music plans normally point at direct WAV files.  The TTS branch was
rendered from Hugging Face Parquet rows and its legacy absolute cache paths are
no longer valid.  This module gives both forms one interface and materializes
selected Parquet audio bytes into an immutable, content-checked source cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping


class SourceAssetError(ValueError):
    """Raised when a dry source cannot be resolved without guessing."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not cleaned:
        cleaned = hashlib.blake2b(value.encode(), digest_size=10).hexdigest()
    return cleaned[:180]


class ParquetAudioSourceIndex:
    """Resolve ``(source_dataset, source_id)`` to embedded Parquet audio bytes."""

    def __init__(self, index_path: str | Path, *, max_open_parquets: int = 4):
        self.index_path = Path(index_path).expanduser().resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(self.index_path)
        if max_open_parquets <= 0:
            raise ValueError("max_open_parquets must be positive")
        self.max_open_parquets = int(max_open_parquets)
        self._rows: dict[tuple[str, str], dict[str, Any]] | None = None
        self._parquets: OrderedDict[str, Any] = OrderedDict()
        self._database: sqlite3.Connection | None = None

    @property
    def _uses_sqlite(self) -> bool:
        return self.index_path.suffix.lower() in {".sqlite", ".sqlite3", ".db"}

    def _db(self) -> sqlite3.Connection:
        if not self._uses_sqlite:
            raise SourceAssetError(f"not a SQLite source index: {self.index_path}")
        if self._database is None:
            self._database = sqlite3.connect(
                f"file:{self.index_path.as_posix()}?mode=ro&immutable=1",
                uri=True,
            )
        return self._database

    def _load(self) -> dict[tuple[str, str], dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["source_dataset"]), str(row["source_id"]))
                if key in rows:
                    raise SourceAssetError(
                        f"duplicate source locator {key} at {self.index_path}:{line_number}"
                    )
                rows[key] = row
        self._rows = rows
        return rows

    def lookup(self, source_dataset: str, source_id: str) -> dict[str, Any]:
        key = (str(source_dataset), str(source_id))
        if self._uses_sqlite:
            row = self._db().execute(
                "SELECT locator_json FROM sources "
                "WHERE source_dataset=? AND source_id=?",
                key,
            ).fetchone()
            if row is None:
                raise SourceAssetError(
                    f"source locator is absent from SQLite index: {key}"
                )
            return json.loads(row[0])
        try:
            return copy.deepcopy(self._load()[key])
        except KeyError as exc:
            raise SourceAssetError(f"source locator is absent from Parquet index: {key}") from exc

    def close(self) -> None:
        if self._database is not None:
            self._database.close()
            self._database = None
        for parquet in self._parquets.values():
            close = getattr(parquet, "close", None)
            if callable(close):
                close()
        self._parquets.clear()

    def _parquet(self, path: str):
        resolved = str(Path(path).expanduser().resolve())
        cached = self._parquets.pop(resolved, None)
        if cached is None:
            import pyarrow.parquet as pq

            if not Path(resolved).is_file():
                raise FileNotFoundError(resolved)
            cached = pq.ParquetFile(resolved)
        self._parquets[resolved] = cached
        while len(self._parquets) > self.max_open_parquets:
            self._parquets.popitem(last=False)
        return cached

    def read_audio(self, locator: Mapping[str, Any]) -> tuple[bytes, str]:
        parquet = self._parquet(str(locator["parquet_path"]))
        row_group = int(locator["row_group"])
        row_in_group = int(locator["row_in_group"])
        table = parquet.read_row_group(row_group, columns=["audio"])
        if not 0 <= row_in_group < table.num_rows:
            raise SourceAssetError(
                f"row_in_group={row_in_group} outside row group with {table.num_rows} rows"
            )
        audio = table.column("audio")[row_in_group].as_py()
        if not isinstance(audio, dict) or not isinstance(audio.get("bytes"), bytes):
            raise SourceAssetError("Parquet audio row does not contain embedded bytes")
        path = str(audio.get("path") or locator.get("audio_path_in_parquet") or "audio.flac")
        suffix = Path(path).suffix.lower()
        if suffix not in {".wav", ".flac", ".ogg"}:
            suffix = ".flac"
        return audio["bytes"], suffix

    def _cache_paths(
        self,
        *,
        source_dataset: str,
        source_id: str,
        cache_root: str | Path,
        locator: Mapping[str, Any],
        suffix: str | None = None,
    ) -> tuple[Path, Path]:
        if suffix is None:
            raw_suffix = Path(
                str(locator.get("audio_path_in_parquet") or "audio.flac")
            ).suffix.lower()
            suffix = raw_suffix if raw_suffix in {".wav", ".flac", ".ogg"} else ".flac"
        key_digest = hashlib.blake2b(
            f"{source_dataset}\0{source_id}".encode(), digest_size=12
        ).hexdigest()
        relative = (
            Path(_safe_name(str(source_dataset)))
            / key_digest[:2]
            / f"{_safe_name(str(source_id))}_{key_digest}{suffix}"
        )
        destination = Path(cache_root).expanduser().resolve() / relative
        return destination, destination.with_suffix(destination.suffix + ".source.json")

    def cached_asset(
        self,
        *,
        source_dataset: str,
        source_id: str,
        cache_root: str | Path,
        locator: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Return a fully verified cache hit without reading Parquet audio."""

        locator = dict(locator or self.lookup(source_dataset, source_id))
        destination, metadata_path = self._cache_paths(
            source_dataset=source_dataset,
            source_id=source_id,
            cache_root=cache_root,
            locator=locator,
        )
        if not metadata_path.is_file():
            return None
        try:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not (
                cached.get("source_dataset") == str(source_dataset)
                and cached.get("source_id") == str(source_id)
                and Path(str(cached.get("path"))).resolve() == destination
                and destination.is_file()
                and destination.stat().st_size == int(cached["size_bytes"])
            ):
                return None
            import soundfile as sf

            audio_info = sf.info(str(destination))
            if not (
                int(audio_info.samplerate) == int(cached["native_sample_rate"])
                and int(audio_info.frames) == int(cached["native_num_frames"])
                and int(audio_info.channels) == int(cached["channels"])
            ):
                return None
            cached["locator"] = locator
            return cached
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return None

    def materialize_payload(
        self,
        *,
        source_dataset: str,
        source_id: str,
        cache_root: str | Path,
        locator: Mapping[str, Any],
        payload: bytes,
        payload_suffix: str,
    ) -> dict[str, Any]:
        """Atomically publish already-read Parquet bytes into the source cache."""

        suffix = payload_suffix if payload_suffix in {".wav", ".flac", ".ogg"} else ".flac"
        destination, metadata_path = self._cache_paths(
            source_dataset=source_dataset,
            source_id=source_id,
            cache_root=cache_root,
            locator=locator,
            suffix=suffix,
        )
        digest = _sha256_bytes(payload)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = destination.read_bytes()
            if _sha256_bytes(existing) != digest:
                raise SourceAssetError(f"source cache hash mismatch: {destination}")
        else:
            temporary = destination.with_name(
                f".{destination.name}.tmp.{os.getpid()}"
            )
            temporary.write_bytes(payload)
            if _sha256_bytes(temporary.read_bytes()) != digest:
                temporary.unlink(missing_ok=True)
                raise SourceAssetError(f"failed to verify materialized source: {temporary}")
            os.replace(temporary, destination)

        import soundfile as sf

        audio_info = sf.info(str(destination))
        asset = {
            "path": str(destination),
            "sha256": digest,
            "size_bytes": len(payload),
            "native_sample_rate": int(audio_info.samplerate),
            "native_num_frames": int(audio_info.frames),
            "channels": int(audio_info.channels),
            "source_dataset": str(source_dataset),
            "source_id": str(source_id),
            "locator": dict(locator),
        }
        temporary_metadata = metadata_path.with_name(
            f".{metadata_path.name}.tmp.{os.getpid()}"
        )
        temporary_metadata.write_text(
            json.dumps(asset, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_metadata, metadata_path)
        return asset

    def materialize(
        self,
        *,
        source_dataset: str,
        source_id: str,
        cache_root: str | Path,
    ) -> dict[str, Any]:
        locator = self.lookup(source_dataset, source_id)
        cached = self.cached_asset(
            source_dataset=source_dataset,
            source_id=source_id,
            cache_root=cache_root,
            locator=locator,
        )
        if cached is not None:
            return cached

        payload, payload_suffix = self.read_audio(locator)
        return self.materialize_payload(
            source_dataset=source_dataset,
            source_id=source_id,
            cache_root=cache_root,
            locator=locator,
            payload=payload,
            payload_suffix=payload_suffix,
        )


def resolve_scene_plan_assets(
    plan: Mapping[str, Any],
    *,
    parquet_index: ParquetAudioSourceIndex | None = None,
    speech_cache_root: str | Path | None = None,
) -> dict[str, Any]:
    """Copy a ScenePlan and replace stable locators with direct asset paths.

    Existing direct paths are preserved.  Missing paths are materialized only
    from an explicit ``parquet_source_id`` locator; this function never searches
    by transcript or filename and therefore cannot silently choose the wrong
    speaker utterance.
    """

    resolved = copy.deepcopy(dict(plan))
    sources = ((resolved.get("scene") or {}).get("sources") or [])
    for source in sources:
        content = source.setdefault("content", {})
        direct = content.get("source_audio_path")
        if isinstance(direct, str) and direct and Path(direct).expanduser().is_file():
            content["source_audio_path"] = str(Path(direct).expanduser().resolve())
            continue
        locator = content.get("source_locator") or {}
        if locator.get("type") != "parquet_source_id":
            raise SourceAssetError(
                f"source {source.get('source_id')} has neither a direct asset nor "
                "a parquet_source_id locator"
            )
        if parquet_index is None or speech_cache_root is None:
            raise SourceAssetError(
                "Parquet source resolution requires an index and speech_cache_root"
            )
        dataset = locator.get("source_dataset") or (source.get("event") or {}).get(
            "source_dataset"
        )
        source_id = locator.get("source_id") or content.get("source_audio_id")
        if not dataset or not source_id:
            raise SourceAssetError("incomplete parquet_source_id locator")
        asset = parquet_index.materialize(
            source_dataset=str(dataset),
            source_id=str(source_id),
            cache_root=speech_cache_root,
        )
        content["source_audio_path"] = asset["path"]
        content["source_audio_sha256"] = asset["sha256"]
        content["source_asset"] = asset
    return resolved


__all__ = [
    "ParquetAudioSourceIndex",
    "SourceAssetError",
    "resolve_scene_plan_assets",
]
