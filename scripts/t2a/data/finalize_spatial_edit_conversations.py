#!/usr/bin/env python3
"""Finalize rendered edit families into AudioChat-style training-turn JSONL.

The heavy family branches may live on several volumes.  This script only reads
their validated ``family.json`` files and writes one small, centralized indexed
view.  Each row carries previous full state/audio, instruction, next full
state/audio, source diff, and explicit independent audio-span noise groups.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.build_scene_plan_store import (  # noqa: E402
    _build_sqlite,
    _iter_index_parts,
    _write_indexed_shard,
)
from stable_audio_tools.data.spatial_edit_recipe import (  # noqa: E402
    validate_edit_family,
)
from stable_audio_tools.data.t2a_artifacts import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
)


SCHEMA = "stable_audio_tools.spatial_cot_training_turn"
SCHEMA_VERSION = "1.0"


def _family_manifests(roots: list[Path]) -> Iterator[Path]:
    for root in sorted(roots, key=str):
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        yield from sorted(root.glob("families/*/family.json"))


def _turn_row(family: dict[str, Any], index: int) -> dict[str, Any]:
    recipe = family["recipes"][index]
    turn = family["turns"][index]
    previous = family["recipes"][index - 1] if index else None
    after_audio = (recipe.get("outputs") or {}).get("foa_path")
    before_audio = (
        (previous.get("outputs") or {}).get("foa_path") if previous else None
    )
    if not after_audio:
        raise ValueError(f"rendered turn {recipe.get('turn_id')} lacks target FOA")
    time_groups = []
    if before_audio:
        time_groups.append(
            {
                "id": "before_foa",
                "role": "clean_context_at_inference",
                "sample_noise_time_independently": True,
            }
        )
    time_groups.append(
        {
            "id": "after_foa",
            "role": "flow_target",
            "sample_noise_time_independently": True,
        }
    )
    family_id = str(family["family_id"])
    turn_id = str(recipe["turn_id"])
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "sample_id": f"{family_id}_{turn_id}",
        "audio_path": str(after_audio),
        "conversation_id": family_id,
        "turn_id": turn_id,
        "turn_index": index,
        "parent_turn_id": recipe.get("parent_turn_id"),
        "task": recipe.get("task"),
        "instruction": recipe.get("instruction"),
        "planner_prompt": recipe.get("planner_prompt") or recipe.get("instruction"),
        "semantic_caption": recipe.get("semantic_caption"),
        "semantic_caption_metadata": recipe.get("semantic_caption_metadata"),
        "understanding_prompt": recipe.get("understanding_prompt"),
        "reference_audio": family.get("reference_audio"),
        "before": {
            "audio_path": before_audio,
            "scene_plan": previous.get("scene_plan") if previous else None,
        },
        "after": {
            "audio_path": str(after_audio),
            "scene_plan": recipe["scene_plan"],
            "signal_stats": (recipe.get("outputs") or {}).get("signal_stats"),
            "source_track_refs": (recipe.get("outputs") or {}).get(
                "source_track_refs", []
            ),
            "render_contract": {
                "state_input": (recipe.get("outputs") or {}).get("state_input"),
                "uses_previous_foa": (recipe.get("outputs") or {}).get(
                    "uses_previous_foa"
                ),
            },
        },
        "diff": turn["diff"],
        "edit": recipe["edit"],
        "audio_time_groups": time_groups,
        "supervision": {
            "planner": True,
            "renderer": True,
            "understanding": bool(before_audio),
            "paired_edit": bool(before_audio),
            "unchanged_source_consistency": bool(before_audio),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=10_000)
    parser.add_argument("--limit-families", type=int, default=None)
    args = parser.parse_args()
    if args.shard_size <= 0 or (
        args.limit_families is not None and args.limit_families <= 0
    ):
        raise SystemExit("shard-size/limit-families must be positive")
    output_root = args.output_root.expanduser().resolve()
    if (output_root / "READY").exists():
        raise SystemExit(f"training-turn store is already READY: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    family_ids: set[str] = set()
    buffer: list[dict[str, Any]] = []
    shard_count = 0
    stats: dict[str, Any] = {
        "families": 0,
        "turns": 0,
        "generation_turns": 0,
        "paired_edit_turns": 0,
        "edit_types": Counter(),
    }

    def flush() -> None:
        nonlocal buffer, shard_count
        if not buffer:
            return
        relative = Path("shards") / f"training-turns-{shard_count:05d}.jsonl"
        rows = _write_indexed_shard(output_root, relative, buffer)
        atomic_write_jsonl(
            output_root / "shard_indexes" / f"index-{shard_count:05d}.jsonl",
            rows,
        )
        buffer = []
        shard_count += 1

    for manifest in _family_manifests(args.render_root):
        family = json.loads(manifest.read_text(encoding="utf-8"))
        validate_edit_family(family, require_outputs=True)
        family_id = str(family["family_id"])
        if family_id in family_ids:
            raise RuntimeError(f"duplicate rendered family across roots: {family_id}")
        family_ids.add(family_id)
        for index in range(len(family["recipes"])):
            row = _turn_row(family, index)
            buffer.append(row)
            stats["turns"] += 1
            stats["generation_turns"] += int(index == 0)
            stats["paired_edit_turns"] += int(index > 0)
            stats["edit_types"][row["edit"]["type"]] += 1
            if len(buffer) >= args.shard_size:
                flush()
        stats["families"] += 1
        if (
            args.limit_families is not None
            and stats["families"] >= args.limit_families
        ):
            break
    flush()
    if not stats["families"]:
        raise SystemExit("no rendered family.json files found")

    index_count, index_sha = atomic_write_jsonl(
        output_root / "index.jsonl", _iter_index_parts(output_root, shard_count)
    )
    if index_count != stats["turns"]:
        raise RuntimeError("training-turn index count mismatch")
    if _build_sqlite(
        output_root, _iter_index_parts(output_root, shard_count)
    ) != stats["turns"]:
        raise RuntimeError("training-turn SQLite count mismatch")
    serializable = {
        key: dict(value) if isinstance(value, Counter) else value
        for key, value in stats.items()
    }
    atomic_write_json(output_root / "stats.json", serializable)
    atomic_write_json(
        output_root / "details.json",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "render_roots": [str(path.expanduser().resolve()) for path in args.render_root],
            "contract": (
                "previous and target audio spans have independent training noise "
                "times; paired_edit is true only for materialized before/after FOA"
            ),
        },
    )
    atomic_write_json(
        output_root / "READY",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "families": stats["families"],
            "turns": stats["turns"],
            "shards": shard_count,
            "index_sha256": index_sha,
        },
    )
    print(
        json.dumps(
            {"status": "READY", "output_root": str(output_root), **serializable},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
