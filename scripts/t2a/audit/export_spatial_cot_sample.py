#!/usr/bin/env python3
"""Export one real Spatial-CoT family for human format inspection.

The production store intentionally separates the persistent edit recipe,
loader metadata, and latent tensor index.  This command copies the exact JSONL
records and also writes one pretty joined view; it never duplicates the latent
tensor or changes the training format.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.t2a_artifacts import atomic_write_json  # noqa: E402


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload.rstrip(b"\n") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _jsonl_record(path: Path, *, family_rank: int) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            record = json.loads(raw)
            if int(record.get("family_rank", -1)) == family_rank:
                return record, raw.rstrip(b"\n")
    raise KeyError(f"family rank {family_rank} was not found in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--family-rank", type=int, required=True)
    parser.add_argument("--families-per-work-shard", type=int, default=256)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.family_rank < 0 or args.families_per_work_shard <= 0:
        raise SystemExit("family-rank must be non-negative and shard size positive")

    view_root = args.view_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    work_shard = args.family_rank // args.families_per_work_shard
    recipe_path = (
        view_root
        / "recipes"
        / args.split
        / f"work-{work_shard:05d}"
        / "shards"
        / f"recipes-{args.split}-{work_shard:05d}.jsonl"
    )
    latent_root = view_root / "latents" / args.split
    metadata_path = latent_root / "metadata" / f"families-{work_shard:05d}.jsonl"
    index_path = latent_root / "index.jsonl"

    recipe, recipe_raw = _jsonl_record(recipe_path, family_rank=args.family_rank)
    metadata, metadata_raw = _jsonl_record(
        metadata_path, family_rank=args.family_rank
    )
    index, index_raw = _jsonl_record(index_path, family_rank=args.family_rank)
    family_ids = {
        str(recipe.get("family_id")),
        str(metadata.get("family_id")),
        str(index.get("family_id")),
    }
    if len(family_ids) != 1:
        raise RuntimeError(f"sample branches disagree on family_id: {family_ids}")

    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(output_root / "recipe.jsonl", recipe_raw)
    _atomic_bytes(output_root / "latent_metadata.jsonl", metadata_raw)
    _atomic_bytes(output_root / "latent_index.jsonl", index_raw)
    atomic_write_json(
        output_root / "joined.pretty.json",
        {
            "schema": "stable_audio_tools.spatial_cot_inspection_sample",
            "schema_version": 1,
            "note": (
                "Human inspection view only. Production keeps these three "
                "records separate and stores the [4,64,432] float16 tensor in "
                "the safetensors shard referenced by latent_index."
            ),
            "family_id": next(iter(family_ids)),
            "family_rank": args.family_rank,
            "work_shard": work_shard,
            "recipe": recipe,
            "latent_metadata": metadata,
            "latent_index": index,
        },
    )
    atomic_write_json(
        output_root / "README.json",
        {
            "family_id": next(iter(family_ids)),
            "family_rank": args.family_rank,
            "source_view": str(view_root),
            "recipe_source": str(recipe_path),
            "metadata_source": str(metadata_path),
            "index_source": str(index_path),
            "files": {
                "recipe.jsonl": "exact persistent family recipe JSONL record",
                "latent_metadata.jsonl": "exact loader metadata JSONL record",
                "latent_index.jsonl": "exact tensor/index JSONL record",
                "joined.pretty.json": "pretty joined inspection-only view",
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "READY",
                "output_root": str(output_root),
                "family_id": next(iter(family_ids)),
                "family_rank": args.family_rank,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
