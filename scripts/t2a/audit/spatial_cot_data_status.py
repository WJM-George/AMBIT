#!/usr/bin/env python3
"""Report resumable production progress without starting any workers."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = (
    REPO_ROOT
    / "stable_audio_tools/configs/dataset_configs/construct_dataset/"
    "spatial_cot_v1_1m_families.json"
)


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _marker_totals(root: Path, marker: str) -> tuple[int, int, list[str]]:
    count = 0
    families = 0
    invalid: list[str] = []
    if not root.is_dir():
        return count, families, invalid
    for path in sorted(root.glob(f"work-*/{marker}")):
        payload = _read(path)
        if payload is None or int(payload.get("families", -1)) < 0:
            invalid.append(str(path))
            continue
        count += 1
        families += int(payload["families"])
    return count, families, invalid


def _done_totals(
    root: Path,
    *,
    split: str,
) -> tuple[int, int, list[str]]:
    count = 0
    families = 0
    invalid: list[str] = []
    done_root = root / "work_done"
    if not done_root.is_dir():
        return count, families, invalid
    for path in sorted(done_root.glob("work-*.json")):
        payload = _read(path)
        try:
            shard = int(path.stem.split("-")[-1])
            tensor = Path(str(payload["tensor_shard"])) if payload else Path()
            metadata = root / "metadata" / f"families-{shard:05d}.jsonl"
            index = root / "shard_indexes" / f"families-{shard:05d}.jsonl"
            valid = bool(
                payload
                and payload.get("split") == split
                and int(payload.get("work_shard", -1)) == shard
                and int(payload.get("families", 0)) > 0
                and tensor.is_file()
                and metadata.is_file()
                and index.is_file()
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            invalid.append(str(path))
            continue
        count += 1
        families += int(payload["families"])
    return count, families, invalid


def _split_status(spec: dict[str, Any], split: str) -> dict[str, Any]:
    storage = spec["storage"]
    expected = int(spec["splits"][split]["families"])
    per_shard = int(spec["sharding"]["families_per_latent_shard"])
    expected_shards = math.ceil(expected / per_shard)
    catalog = Path(storage["catalog_root"])
    recipe_root = catalog / "recipes" / split
    render_base = Path(
        storage[
            "render_staging_root"
            if split == "train"
            else "retained_eval_render_root"
        ]
    )
    render_root = render_base / split
    latent_root = Path(storage["latent_root"]) / split
    recipe_shards, recipe_families, recipe_invalid = _marker_totals(
        recipe_root, "READY"
    )
    render_shards, render_families, render_invalid = _marker_totals(
        render_root, "render/READY"
    )
    done_shards, done_families, done_invalid = _done_totals(
        latent_root, split=split
    )
    ready = _read(latent_root / "READY")
    final_ready = bool(
        ready
        and ready.get("split") == split
        and int(ready.get("families", -1)) == expected
    )
    quality_path = render_root / "QUALITY.json" if split != "train" else None
    quality = _read(quality_path) if quality_path is not None else None
    quality_pass = bool(
        split == "train"
        or (
            quality
            and quality.get("status") == "PASS"
            and quality.get("split") == split
            and int(quality.get("families", -1)) == expected
        )
    )
    invalid = recipe_invalid + render_invalid + done_invalid
    if quality_path is not None and quality_path.is_file() and not quality_pass:
        invalid.append(str(quality_path))
    state = (
        "ERROR"
        if invalid
        else "READY"
        if final_ready and quality_pass
        else "READY_UNAUDITED"
        if final_ready
        else "IN_PROGRESS"
        if recipe_shards or render_shards or done_shards
        else "NOT_STARTED"
    )
    return {
        "state": state,
        "expected_families": expected,
        "expected_work_shards": expected_shards,
        "recipe_ready_shards": recipe_shards,
        "recipe_families": recipe_families,
        "render_ready_shards_currently_present": render_shards,
        "render_families_currently_present": render_families,
        "done_shards": done_shards,
        "done_families": done_families,
        "done_percent": round(100.0 * done_families / expected, 4),
        "final_ready": final_ready,
        "quality_pass": quality_pass,
        "quality_marker": str(quality_path) if quality_path is not None else None,
        "invalid_markers": invalid,
        "recipe_root": str(recipe_root),
        "render_root": str(render_root),
        "latent_root": str(latent_root),
        "resume_command": (
            {
                "validation": "scripts/t2a/data/run_spatial_cot_data.sh audit-validation10k",
                "test": "scripts/t2a/data/run_spatial_cot_data.sh audit-test2k",
            }[split]
            if split != "train" and final_ready and not quality_pass
            else {
                "train": "scripts/t2a/data/run_spatial_cot_data.sh train1m",
                "validation": "scripts/t2a/data/run_spatial_cot_data.sh validation10k",
                "test": "scripts/t2a/data/run_spatial_cot_data.sh test2k",
            }[split]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        action="append",
        default=None,
    )
    args = parser.parse_args()
    spec_path = args.build_spec.expanduser().resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    splits = args.split or ["test", "validation", "train"]
    report = {
        "schema": "stable_audio_tools.spatial_cot_pipeline_status",
        "schema_version": 1,
        "build_spec": str(spec_path),
        "resume_policy": (
            "rerun the same command; immutable READY/DONE shards are reused, "
            "and only incomplete transient renders are regenerated"
        ),
        "splits": {split: _split_status(spec, split) for split in splits},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if any(
        value["state"] == "ERROR" for value in report["splits"].values()
    ) else 0


if __name__ == "__main__":
    raise SystemExit(main())
