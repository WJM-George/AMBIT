#!/usr/bin/env python3
"""Build a versioned full-family caption overlay without rewriting FOA latents."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.spatial_caption_templates import (  # noqa: E402
    SEMANTIC_CAPTION_TEMPLATE_COUNT,
    SEMANTIC_CAPTION_TEMPLATE_VERSION,
    render_semantic_caption,
    validate_semantic_caption_metadata,
)


OVERLAY_SCHEMA = "stable_audio_tools.spatial_caption_overlay"
OVERLAY_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _turn_sources(turn: dict[str, Any]) -> list[dict[str, Any]]:
    scene_plan = ((turn.get("after") or {}).get("scene_plan") or {})
    sources = ((scene_plan.get("scene") or {}).get("sources") or [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("caption overlay turn has no target ScenePlan sources")
    return sources


def _process_shard(
    source_value: str,
    output_value: str,
    split: str,
) -> dict[str, Any]:
    source = Path(source_value)
    output_root = Path(output_value)
    suffix = source.stem.removeprefix("families-")
    caption_path = output_root / "shards" / f"captions-{suffix}.jsonl"
    index_path = output_root / "shard_indexes" / f"captions-{suffix}.jsonl"
    done_path = output_root / "work_done" / f"captions-{suffix}.json"
    source_hash = _sha256(source)
    if caption_path.is_file() and index_path.is_file() and done_path.is_file():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if (
            done.get("source_metadata_sha256") == source_hash
            and done.get("template_version") == SEMANTIC_CAPTION_TEMPLATE_VERSION
            and done.get("caption_sha256") == _sha256(caption_path)
            and done.get("index_sha256") == _sha256(index_path)
        ):
            return done
        raise RuntimeError(f"incompatible partial caption shard: {caption_path}")

    caption_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    done_path.parent.mkdir(parents=True, exist_ok=True)
    caption_tmp = caption_path.with_name(
        f".{caption_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    index_tmp = index_path.with_name(
        f".{index_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    count = 0
    turn_count = 0
    char_count = 0
    max_chars = 0
    template_counts = [0] * SEMANTIC_CAPTION_TEMPLATE_COUNT
    caption_digest = hashlib.sha256()
    index_digest = hashlib.sha256()
    try:
        with (
            source.open("r", encoding="utf-8") as source_handle,
            caption_tmp.open("wb") as caption_handle,
            index_tmp.open("wb") as index_handle,
        ):
            offset = 0
            for line_number, line in enumerate(source_handle, start=1):
                if not line.strip():
                    continue
                family = json.loads(line)
                family_id = str(family.get("family_id") or "")
                family_rank = int(family.get("family_rank", -1))
                turns = family.get("turns")
                if not family_id or family_rank < 0 or not isinstance(turns, list):
                    raise ValueError(
                        f"invalid family metadata {source}:{line_number}"
                    )
                overlay_turns = []
                family_template_id: int | None = None
                for turn_index, turn in enumerate(turns):
                    if not isinstance(turn, dict):
                        raise TypeError(
                            f"invalid turn {turn_index} in family {family_id}"
                        )
                    sources = _turn_sources(turn)
                    rendered = render_semantic_caption(
                        sources,
                        template_key=family_id,
                    )
                    metadata = rendered.metadata()
                    expected_ids = [
                        str(source_item.get("source_id") or f"source_{index}")
                        for index, source_item in enumerate(sources)
                    ]
                    validate_semantic_caption_metadata(
                        rendered.text,
                        metadata,
                        expected_source_ids=expected_ids,
                    )
                    if family_template_id is None:
                        family_template_id = rendered.template_id
                    elif family_template_id != rendered.template_id:
                        raise RuntimeError(
                            f"family {family_id} changed template across turns"
                        )
                    overlay_turns.append(
                        {
                            "turn_id": str(
                                turn.get("turn_id") or f"turn_{turn_index:03d}"
                            ),
                            "semantic_caption": rendered.text,
                            "semantic_caption_metadata": metadata,
                        }
                    )
                    turn_count += 1
                    char_count += len(rendered.text)
                    max_chars = max(max_chars, len(rendered.text))
                assert family_template_id is not None
                template_counts[family_template_id] += 1
                overlay = {
                    "schema": OVERLAY_SCHEMA,
                    "schema_version": OVERLAY_VERSION,
                    "template_version": SEMANTIC_CAPTION_TEMPLATE_VERSION,
                    "family_id": family_id,
                    "family_rank": family_rank,
                    "split": split,
                    "template_id": family_template_id,
                    "turns": overlay_turns,
                }
                payload = (
                    json.dumps(overlay, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                caption_handle.write(payload)
                caption_digest.update(payload)
                row = {
                    "family_id": family_id,
                    "family_rank": family_rank,
                    "caption_shard": f"shards/{caption_path.name}",
                    "caption_offset": offset,
                    "caption_length": len(payload),
                    "template_id": family_template_id,
                    "num_turns": len(overlay_turns),
                }
                index_payload = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                index_handle.write(index_payload)
                index_digest.update(index_payload)
                offset += len(payload)
                count += 1
            caption_handle.flush()
            index_handle.flush()
            os.fsync(caption_handle.fileno())
            os.fsync(index_handle.fileno())
        os.replace(caption_tmp, caption_path)
        os.replace(index_tmp, index_path)
    except BaseException:
        caption_tmp.unlink(missing_ok=True)
        index_tmp.unlink(missing_ok=True)
        raise

    done = {
        "schema": OVERLAY_SCHEMA,
        "schema_version": OVERLAY_VERSION,
        "template_version": SEMANTIC_CAPTION_TEMPLATE_VERSION,
        "split": split,
        "source_metadata_shard": str(source),
        "source_metadata_sha256": source_hash,
        "caption_shard": str(caption_path),
        "caption_sha256": caption_digest.hexdigest(),
        "index_shard": str(index_path),
        "index_sha256": index_digest.hexdigest(),
        "families": count,
        "turns": turn_count,
        "caption_characters": char_count,
        "max_caption_characters": max_chars,
        "template_counts": template_counts,
    }
    _atomic_json(done_path, done)
    return done


def _index_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _finalize(
    *,
    latent_root: Path,
    output_root: Path,
    split: str,
    expected_families: int,
    source_shards: list[Path],
    done_rows: list[dict[str, Any]],
) -> None:
    done_by_source = {
        str(Path(row["source_metadata_shard"]).resolve()): row for row in done_rows
    }
    ordered_done = [done_by_source[str(path.resolve())] for path in source_shards]
    total_families = sum(int(row["families"]) for row in ordered_done)
    total_turns = sum(int(row["turns"]) for row in ordered_done)
    if total_families != expected_families:
        raise RuntimeError(
            f"caption overlay found {total_families} families, expected {expected_families}"
        )

    global_index = output_root / "index.jsonl"
    index_tmp = global_index.with_name(
        f".{global_index.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    sqlite_path = output_root / "index.sqlite"
    sqlite_tmp = sqlite_path.with_name(
        f".{sqlite_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    index_digest = hashlib.sha256()
    expected_rank = 0
    connection = sqlite3.connect(sqlite_tmp)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE captions("
            "family_id TEXT PRIMARY KEY, family_rank INTEGER NOT NULL UNIQUE, "
            "caption_shard TEXT NOT NULL, caption_offset INTEGER NOT NULL, "
            "caption_length INTEGER NOT NULL, template_id INTEGER NOT NULL, "
            "num_turns INTEGER NOT NULL) WITHOUT ROWID"
        )
        with index_tmp.open("wb") as output:
            for done in ordered_done:
                index_path = Path(done["index_shard"])
                batch = []
                for row in _index_rows(index_path):
                    rank = int(row["family_rank"])
                    if rank != expected_rank:
                        raise RuntimeError(
                            f"caption family ranks are not contiguous: "
                            f"expected {expected_rank}, got {rank}"
                        )
                    expected_rank += 1
                    payload = (
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    ).encode("utf-8")
                    output.write(payload)
                    index_digest.update(payload)
                    batch.append(
                        (
                            row["family_id"],
                            rank,
                            row["caption_shard"],
                            int(row["caption_offset"]),
                            int(row["caption_length"]),
                            int(row["template_id"]),
                            int(row["num_turns"]),
                        )
                    )
                connection.executemany("INSERT INTO captions VALUES(?,?,?,?,?,?,?)", batch)
            output.flush()
            os.fsync(output.fileno())
        connection.commit()
        count = int(connection.execute("SELECT COUNT(*) FROM captions").fetchone()[0])
        if count != expected_families:
            raise RuntimeError(f"caption SQLite count mismatch: {count}")
    except BaseException:
        connection.close()
        index_tmp.unlink(missing_ok=True)
        sqlite_tmp.unlink(missing_ok=True)
        raise
    connection.close()
    os.replace(index_tmp, global_index)
    os.replace(sqlite_tmp, sqlite_path)

    aggregate_templates = [0] * SEMANTIC_CAPTION_TEMPLATE_COUNT
    for done in ordered_done:
        for index, count in enumerate(done["template_counts"]):
            aggregate_templates[index] += int(count)
    latent_details = json.loads(
        (latent_root / "details.json").read_text(encoding="utf-8")
    )
    details = {
        "schema": OVERLAY_SCHEMA,
        "schema_version": OVERLAY_VERSION,
        "template_version": SEMANTIC_CAPTION_TEMPLATE_VERSION,
        "template_count": SEMANTIC_CAPTION_TEMPLATE_COUNT,
        "split": split,
        "families": total_families,
        "turns": total_turns,
        "source_latent_root": str(latent_root),
        "source_latent_ready_sha256": _sha256(latent_root / "READY"),
        "source_vae_checkpoint": latent_details.get("vae_checkpoint"),
        "source_vae_checkpoint_sha256": latent_details.get("vae_checkpoint_sha256"),
        "caption_characters": sum(
            int(row["caption_characters"]) for row in ordered_done
        ),
        "max_caption_characters": max(
            int(row["max_caption_characters"]) for row in ordered_done
        ),
        "distinct_templates": sum(1 for count in aggregate_templates if count),
        "template_counts": aggregate_templates,
        "work_shards": len(source_shards),
    }
    _atomic_json(output_root / "details.json", details)
    _atomic_json(
        output_root / "READY",
        {
            "schema": OVERLAY_SCHEMA,
            "schema_version": OVERLAY_VERSION,
            "template_version": SEMANTIC_CAPTION_TEMPLATE_VERSION,
            "template_count": SEMANTIC_CAPTION_TEMPLATE_COUNT,
            "split": split,
            "families": total_families,
            "turns": total_turns,
            "work_shards": len(source_shards),
            "index_sha256": index_digest.hexdigest(),
            "sqlite_sha256": _sha256(sqlite_path),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-families", type=int)
    parser.add_argument("--max-shards", type=int)
    args = parser.parse_args()

    latent_root = args.latent_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not (latent_root / "READY").is_file():
        raise SystemExit(f"source latent store is not READY: {latent_root}")
    if (output_root / "READY").is_file():
        raise SystemExit(f"caption overlay is already READY: {output_root}")
    latent_ready = json.loads((latent_root / "READY").read_text(encoding="utf-8"))
    split = str(latent_ready.get("split") or latent_root.name)
    source_shards = sorted((latent_root / "metadata").glob("families-*.jsonl"))
    if args.max_shards is not None:
        source_shards = source_shards[: int(args.max_shards)]
    if not source_shards:
        raise SystemExit(f"no metadata shards found under {latent_root}")
    expected_families = int(
        args.expected_families
        if args.expected_families is not None
        else latent_ready.get("families", 0)
    )
    if args.max_shards is not None and args.expected_families is None:
        raise SystemExit("--max-shards requires --expected-families")
    output_root.mkdir(parents=True, exist_ok=True)

    done_rows = []
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {
            executor.submit(
                _process_shard,
                str(path),
                str(output_root),
                split,
            ): path
            for path in source_shards
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            done_rows.append(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"caption shards {completed}/{len(futures)} "
                    f"(latest={path.name})",
                    flush=True,
                )

    _finalize(
        latent_root=latent_root,
        output_root=output_root,
        split=split,
        expected_families=expected_families,
        source_shards=source_shards,
        done_rows=done_rows,
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "output_root": str(output_root),
                "split": split,
                "families": expected_families,
                "templates": SEMANTIC_CAPTION_TEMPLATE_COUNT,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
