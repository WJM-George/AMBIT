#!/usr/bin/env python3
"""Re-render one cleaned Spatial-CoT family and verify exact recoverability.

The train pipeline intentionally deletes rendered FOA and source tracks after
the VAE shard is durable.  This check starts only from the persistent recipe
JSONL plus the latent metadata index, renders into a temporary directory, and
requires every regenerated FOA checksum and source-track ID to match the values
recorded before cleanup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTHESIS_ROOT = REPO_ROOT / "dataset" / "synthesis"
for root in (REPO_ROOT, SYNTHESIS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from render_spatial_edit_families import (  # noqa: E402
    RETAINED_PROFILE,
    render_family,
)
from stable_audio_tools.data.spatial_edit_recipe import (  # noqa: E402
    STATE_RENDER_INPUT,
    is_speech_source,
    validate_edit_family,
)
from stable_audio_tools.data.t2a_artifacts import atomic_write_json  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _find(rows: Iterator[dict[str, Any]], family_rank: int) -> dict[str, Any]:
    for row in rows:
        if int(row.get("family_rank", -1)) == family_rank:
            return row
    raise KeyError(f"family rank {family_rank} was not found")


def _speech_count(sources: list[Mapping[str, Any]]) -> int:
    return sum(is_speech_source(source) for source in sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--work-shard", type=int, default=0)
    parser.add_argument("--family-rank", type=int, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--merge-audit-json",
        type=Path,
        default=None,
        help=(
            "Merge this recovery check into an existing smoke/pilot audit so "
            "the audit directory needs no separate recovery files."
        ),
    )
    args = parser.parse_args()
    if args.output_json is None and args.merge_audit_json is None:
        raise SystemExit("provide --output-json and/or --merge-audit-json")

    view_root = args.view_root.expanduser().resolve()
    work_shard = int(args.work_shard)
    recipe_root = (
        view_root / "recipes" / args.split / f"work-{work_shard:05d}"
    )
    recipe_files = sorted((recipe_root / "shards").glob("*.jsonl"))
    if len(recipe_files) != 1:
        raise RuntimeError(f"expected one recipe shard under {recipe_root}")
    recipe_path = recipe_files[0]
    metadata_path = (
        view_root
        / "latents"
        / args.split
        / "metadata"
        / f"families-{work_shard:05d}.jsonl"
    )
    family = _find(_jsonl(recipe_path), args.family_rank)
    metadata = _find(_jsonl(metadata_path), args.family_rank)
    family_id = str(family["family_id"])
    if str(metadata.get("family_id")) != family_id:
        raise RuntimeError("recipe and latent metadata family IDs disagree")
    if metadata.get("recipe_jsonl_sha256") != _sha256(recipe_path):
        raise RuntimeError("persistent recipe JSONL checksum does not match metadata")
    validate_edit_family(family, require_outputs=False)
    provenance = metadata.get("render_provenance") or {}
    if (
        provenance.get("state_input") != STATE_RENDER_INPUT
        or provenance.get("uses_previous_foa") is not False
        or provenance.get("independent_state_mix") is not True
    ):
        raise RuntimeError("metadata does not prove independent dry-source rendering")
    storage_profile = str(
        provenance.get("storage_profile") or RETAINED_PROFILE
    )

    for recipe in family["recipes"]:
        if _speech_count(recipe["sources"]) > 1:
            raise RuntimeError(f"speech+speech state in recovery input: {family_id}")
        for source in recipe["sources"]:
            dry = source["dry_audio"]
            path = Path(dry["path"])
            stat = path.stat()
            if (
                int(stat.st_size) != int(dry["file_size_bytes"])
                or int(stat.st_mtime_ns) != int(dry["file_mtime_ns"])
            ):
                raise RuntimeError(f"dry source identity changed: {path}")

    expected_hashes = [
        str(turn["after"]["foa_sha256"]) for turn in metadata["turns"]
    ]
    expected_track_ids = [
        {
            str(reference["source_id"]): str(reference["track_id"])
            for reference in turn["after"]["source_track_refs"]
        }
        for turn in metadata["turns"]
    ]
    args.scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"recover_{family_id}_", dir=args.scratch_root
    ) as temporary:
        rendered = render_family(
            family,
            Path(temporary) / "render",
            storage_profile=storage_profile,
        )
        validate_edit_family(rendered, require_outputs=True)
        summary = rendered.get("render_summary") or {}
        if (
            summary.get("state_input") != STATE_RENDER_INPUT
            or summary.get("uses_previous_foa") is not False
            or summary.get("independent_state_mix") is not True
        ):
            raise RuntimeError("recovery renderer violated the dry-source state contract")
        actual_hashes = []
        actual_track_ids = []
        for recipe in rendered["recipes"]:
            output = Path(recipe["outputs"]["foa_path"])
            actual = _sha256(output)
            if actual != recipe["outputs"]["foa_sha256"]:
                raise RuntimeError(f"renderer recorded a stale FOA hash: {output}")
            actual_hashes.append(actual)
            actual_track_ids.append(
                {
                    str(reference["source_id"]): str(reference["track_id"])
                    for reference in recipe["outputs"]["source_track_refs"]
                }
            )
        if actual_hashes != expected_hashes:
            raise RuntimeError(
                f"regenerated FOA checksums differ: expected={expected_hashes} "
                f"actual={actual_hashes}"
            )
        if actual_track_ids != expected_track_ids:
            raise RuntimeError("regenerated source-track IDs differ from metadata")
        expected_gain = float(
            metadata["render_provenance"]["family_master_gain_linear"]
        )
        actual_gain = float(rendered["render_summary"]["family_master_gain_linear"])
        if not math.isclose(actual_gain, expected_gain, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"family master gain changed: {actual_gain} != {expected_gain}"
            )

    report = {
        "schema": "stable_audio_tools.spatial_cot_recovery_audit",
        "schema_version": 1,
        "status": "PASS",
        "family_id": family_id,
        "family_rank": int(args.family_rank),
        "split": args.split,
        "work_shard": work_shard,
        "states": len(expected_hashes),
        "speech_sources_per_state": [
            _speech_count(recipe["sources"]) for recipe in family["recipes"]
        ],
        "source_counts_per_state": [
            len(recipe["sources"]) for recipe in family["recipes"]
        ],
        "foa_sha256": expected_hashes,
        "family_master_gain_linear": expected_gain,
        "state_input": STATE_RENDER_INPUT,
        "uses_previous_foa": False,
        "independent_state_mix": True,
        "storage_profile": storage_profile,
        "recipe_jsonl": str(recipe_path),
        "recipe_jsonl_sha256": _sha256(recipe_path),
        "temporary_render_removed": True,
    }
    if args.output_json is not None:
        atomic_write_json(args.output_json.expanduser().resolve(), report)
    if args.merge_audit_json is not None:
        audit_path = args.merge_audit_json.expanduser().resolve()
        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS":
            raise RuntimeError(f"cannot merge into a non-PASS audit: {audit_path}")
        recovery = audit.setdefault(
            "recovery",
            {"status": "PASS", "checks": []},
        )
        checks = [
            item
            for item in recovery.get("checks") or []
            if not (
                str(item.get("split")) == str(report["split"])
                and int(item.get("family_rank", -1)) == int(report["family_rank"])
            )
        ]
        checks.append(report)
        checks.sort(key=lambda item: (str(item["split"]), int(item["family_rank"])))
        recovery["status"] = "PASS"
        recovery["checks"] = checks
        atomic_write_json(audit_path, audit)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
