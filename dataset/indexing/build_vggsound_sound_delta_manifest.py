#!/usr/bin/env python3
"""Build a leak-auditable VGGSound train-only Sound extraction manifest.

This is a delta source-pool builder.  It never mutates the frozen P0--P9
catalogs and it does not assume that an official split alone proves isolation.
Candidates must be absent from the existing WAV cache, classify as non-vocal
Sound, and be lineage-disjoint from both P10 train and the external benchmark.
Exact waveform hashing is intentionally deferred until extraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluation.build_p10_public_core_candidates import (
    classify_vggsound,
    load_train_lineage,
    overlapping_assets,
)


SCHEMA = "stable_audio_tools.vggsound_sound_delta_selection"
SCHEMA_VERSION = 1


def canonical_json(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_rank(stem: str) -> str:
    return hashlib.sha256(
        f"sceneplan-vggsound-sound-delta-v1\0{stem}".encode("utf-8")
    ).hexdigest()


def external_youtube_ids(root: Path) -> set[str]:
    result: set[str] = set()
    for name in ("candidate_pool.jsonl", "candidate_exclusions.jsonl"):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                youtube_id = json.loads(line).get("youtube_id")
                if youtube_id:
                    result.add(str(youtube_id))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vggsound-csv",
        type=Path,
        default=Path("/mnt/sdc/audio_dataset/datasets/vggsound/snapshot/vggsound.csv"),
    )
    parser.add_argument(
        "--existing-audio-root",
        type=Path,
        default=Path("/mnt/sdc/audio_dataset/datasets/vggsound/extracted/audio"),
    )
    parser.add_argument(
        "--sceneplan-root",
        type=Path,
        default=Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m"),
    )
    parser.add_argument(
        "--audiocaps-full-root",
        type=Path,
        default=Path("/mnt/sdd/audio_dataset/datasets/audiocaps/snapshot/data"),
    )
    parser.add_argument(
        "--external-manifest-root",
        type=Path,
        default=Path(
            "/mnt/sdb/audio_dataset/evaluation_benchmark/"
            "p10_evaluation_benchmark_v1/manifests"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/supplements/"
            "vggsound_sound_delta_v1/source_selection"
        ),
    )
    parser.add_argument(
        "--pilot-selection-rows",
        type=int,
        default=25_000,
        help="Oversubscribed extraction set; the post-extraction gate freezes 20k.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = args.vggsound_csv.expanduser().resolve(strict=True)
    existing_root = args.existing_audio_root.expanduser().resolve(strict=True)
    sceneplan_root = args.sceneplan_root.expanduser().resolve(strict=True)
    external_root = args.external_manifest_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if args.pilot_selection_rows <= 0:
        raise ValueError("pilot-selection-rows must be positive")
    try:
        output_root.relative_to(Path("/mnt/sdb"))
    except ValueError as error:
        raise ValueError("delta manifests must live on /mnt/sdb") from error

    (
        train_assets,
        youtube_whole_clip,
        vgg_intervals,
        audiocaps_intervals,
        _train_hashes,
    ) = load_train_lineage(sceneplan_root, args.audiocaps_full_root)
    benchmark_youtube = external_youtube_ids(external_root)
    existing_stems = {path.stem for path in existing_root.glob("*.wav")}

    raw_rows: list[tuple[str, int, str, str]] = []
    official_test_youtube: set[str] = set()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) != 4:
                raise ValueError(f"unexpected VGGSound row: {row!r}")
            youtube_id, start, label, split = row
            parsed = (youtube_id, int(start), label.strip(), split.strip())
            raw_rows.append(parsed)
            if split.strip() == "test":
                official_test_youtube.add(youtube_id)

    candidates: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    reasons = Counter()
    for youtube_id, start, label, split in raw_rows:
        stem = f"{youtube_id}_{start:06d}"
        asset_id = f"sound:vggsound:vggsound_{stem}"
        row_reasons: list[str] = []
        modality = classify_vggsound(label)
        if split != "train":
            row_reasons.append("not_official_train")
        if modality != "sound":
            row_reasons.append(
                "speech_or_vocal" if modality is None else "routes_to_music"
            )
        if stem in existing_stems:
            row_reasons.append("already_extracted")
        if asset_id in train_assets:
            row_reasons.append("already_used_by_p10_train")
        if youtube_id in youtube_whole_clip:
            row_reasons.append("youtube_id_used_by_audioset_or_musiccaps_train")
        if overlapping_assets(
            vgg_intervals, youtube_id, float(start), float(start) + 10.0
        ):
            row_reasons.append("overlaps_existing_vggsound_train_interval")
        if overlapping_assets(
            audiocaps_intervals, youtube_id, float(start), float(start) + 10.0
        ):
            row_reasons.append("overlaps_existing_audiocaps_train_interval")
        if youtube_id in official_test_youtube:
            row_reasons.append("youtube_id_occurs_in_vggsound_test")
        if youtube_id in benchmark_youtube:
            row_reasons.append("youtube_id_reserved_by_external_benchmark")

        record = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "candidate_id": f"vggsound:{youtube_id}:{start}",
            "asset_id": asset_id,
            "source_dataset": "vggsound",
            "official_split": split,
            "youtube_id": youtube_id,
            "start_time_sec": float(start),
            "duration_sec": 10.0,
            "stem": stem,
            "label": label,
            "modality": modality,
            "selection_rank": stable_rank(stem),
            "lineage_gate_status": "pass" if not row_reasons else "fail",
            "lineage_exclusion_reasons": sorted(set(row_reasons)),
            "exact_hash_gate_status": "pending",
        }
        if row_reasons:
            reasons.update(set(row_reasons))
            exclusions.append(record)
        else:
            candidates.append(record)

    candidates.sort(key=lambda row: (row["selection_rank"], row["stem"]))
    exclusions.sort(key=lambda row: (row["selection_rank"], row["stem"]))
    pilot = candidates[: args.pilot_selection_rows]
    atomic_jsonl(output_root / "candidate_pool.jsonl", candidates)
    atomic_jsonl(output_root / "candidate_exclusions.jsonl", exclusions)
    atomic_jsonl(output_root / "pilot_extraction_selection.jsonl", pilot)
    summary = {
        "schema": "stable_audio_tools.vggsound_sound_delta_selection_summary",
        "schema_version": 1,
        "status": "PASS" if len(pilot) == args.pilot_selection_rows else "FAIL",
        "inputs": {
            "vggsound_csv": str(csv_path),
            "vggsound_csv_sha256": sha256_file(csv_path),
            "sceneplan_root": str(sceneplan_root),
            "external_manifest_root": str(external_root),
        },
        "counts": {
            "raw_rows": len(raw_rows),
            "clean_train_sound_candidates": len(candidates),
            "excluded_rows": len(exclusions),
            "pilot_extraction_rows": len(pilot),
            "unique_pilot_youtube_ids": len({row["youtube_id"] for row in pilot}),
        },
        "exclusion_reason_counts": dict(sorted(reasons.items())),
        "contracts": {
            "official_train_only": True,
            "non_vocal_sound_only": True,
            "existing_p10_train_lineage_disjoint": True,
            "external_benchmark_lineage_disjoint": True,
            "exact_audio_hash_gate_after_extraction": True,
        },
        "artifacts": {
            "candidate_pool": str(output_root / "candidate_pool.jsonl"),
            "candidate_exclusions": str(output_root / "candidate_exclusions.jsonl"),
            "pilot_extraction_selection": str(
                output_root / "pilot_extraction_selection.jsonl"
            ),
        },
    }
    atomic_json(output_root / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
