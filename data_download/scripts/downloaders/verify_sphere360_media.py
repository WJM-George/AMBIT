#!/usr/bin/env python3
"""Validate downloaded Sphere360 media against the split manifest.

This is the ONLY "cleaning" you need when re-downloading the *published*
Sphere360: the split lists (dataset/split/{train,test}.txt) are already the
final output of the authors' Silent/Static/AV-match/Voice cleaning pipeline
(103,596 clips x 10s == 288 h, matching the paper). So we do not re-run those
filters; we only verify each downloaded clip is intact and is genuine FOA:

  * has a video stream (equirectangular 360),
  * has a 4-channel ambisonic (FOA) audio stream  -> itag 338, ACN/SN3D,
  * is at least --min-duration seconds long and ffprobe-readable.

It cross-references the split manifest so you can see coverage (present / missing
/ broken) and emit a re-download list for the failures.

Examples
--------
# Validate the test split media we already have:
python3 scripts/downloaders/verify_sphere360_media.py --split test

# Validate train, write a re-download list, and delete broken files so a rerun
# of the downloader re-fetches them:
python3 scripts/downloaders/verify_sphere360_media.py --split train --delete-bad
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATASET_ROOT = Path(os.environ.get("AUDIO_DATASET_SECONDARY_ROOT", "/mnt/sdb/audio_dataset"))
DEFAULT_SPHERE360_ROOT = DEFAULT_DATASET_ROOT / "datasets" / "sphere360"
DEFAULT_SNAPSHOT = DEFAULT_SPHERE360_ROOT / "snapshot"
DEFAULT_MEDIA_ROOT = DEFAULT_SPHERE360_ROOT / "media"
EXPECTED_CHANNELS = 4  # first-order ambisonics W,Y,Z,X


@dataclass
class ClipReport:
    clip_id: str
    path: str
    status: str  # ok | not_4ch | no_video | too_short | unreadable | missing
    channels: int | None = None
    duration: float | None = None
    detail: str = ""


def ffprobe(path: Path) -> dict:
    """Return ffprobe JSON for a media file, or raise on hard failure."""
    proc = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_streams", "-show_format", str(path),
        ],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffprobe failed")
    return json.loads(proc.stdout or "{}")


def inspect_clip(clip_id: str, path: Path, min_duration: float) -> ClipReport:
    if not path.exists():
        return ClipReport(clip_id, str(path), "missing")
    try:
        info = ffprobe(path)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the scan
        return ClipReport(clip_id, str(path), "unreadable", detail=str(exc)[:200])

    streams = info.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    try:
        duration = float(info.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0
    channels = int(audio["channels"]) if audio and "channels" in audio else None

    if audio is None or channels != EXPECTED_CHANNELS:
        return ClipReport(clip_id, str(path), "not_4ch", channels, duration,
                          f"audio channels={channels} (need {EXPECTED_CHANNELS})")
    if video is None:
        return ClipReport(clip_id, str(path), "no_video", channels, duration, "no video stream")
    if duration < min_duration:
        return ClipReport(clip_id, str(path), "too_short", channels, duration,
                          f"duration {duration:.2f}s < {min_duration}s")
    return ClipReport(clip_id, str(path), "ok", channels, duration)


def read_manifest(split_file: Path) -> list[str]:
    ids: list[str] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(line)
    return ids


def run_split(split: str, snapshot: Path, media_root: Path, min_duration: float,
              jobs: int, delete_bad: bool) -> dict:
    split_file = snapshot / "dataset" / "split" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split manifest not found: {split_file}")
    out_dir = media_root / split
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_ids = read_manifest(split_file)
    logging.info("%s: manifest lists %d clips", split, len(clip_ids))

    reports: list[ClipReport] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(inspect_clip, cid, out_dir / f"{cid}.webm", min_duration): cid
            for cid in clip_ids
        }
        for i, fut in enumerate(as_completed(futures), 1):
            reports.append(fut.result())
            if i % 2000 == 0:
                logging.info("  inspected %d/%d", i, len(clip_ids))

    by_status: dict[str, list[ClipReport]] = {}
    for r in reports:
        by_status.setdefault(r.status, []).append(r)

    # Write lists next to the split media.
    def write_list(name: str, rows: list[ClipReport]) -> None:
        p = media_root / f"{split}_{name}.txt"
        p.write_text("".join(f"{r.clip_id}\n" for r in sorted(rows, key=lambda x: x.clip_id)),
                     encoding="utf-8")

    ok = by_status.get("ok", [])
    redo = [r for s in ("missing", "not_4ch", "no_video", "too_short", "unreadable")
            for r in by_status.get(s, [])]
    write_list("verified_ok", ok)
    write_list("redownload", redo)

    if delete_bad:
        removed = 0
        for r in redo:
            if r.status != "missing":
                try:
                    Path(r.path).unlink(missing_ok=True)
                    removed += 1
                except OSError as exc:  # noqa: BLE001
                    logging.warning("could not delete %s: %s", r.path, exc)
        logging.info("%s: deleted %d broken files (will be re-fetched on next download run)", split, removed)

    summary = {s: len(v) for s, v in sorted(by_status.items())}
    logging.info("%s SUMMARY: %s", split, summary)
    logging.info("%s: %d/%d OK (%.1f%%); redownload list -> %s",
                 split, len(ok), len(clip_ids),
                 100.0 * len(ok) / max(1, len(clip_ids)),
                 media_root / f"{split}_redownload.txt")
    return {"split": split, "expected": len(clip_ids), **summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--min-duration", type=float, default=9.5,
                        help="Minimum acceptable clip duration in seconds (keyframe-aligned cuts are >= 10s).")
    parser.add_argument("--jobs", type=int, default=8, help="Parallel ffprobe workers.")
    parser.add_argument("--delete-bad", action="store_true",
                        help="Delete broken/short/non-FOA files so the downloader re-fetches them.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    splits = ["test", "train"] if args.split == "both" else [args.split]
    results = []
    for split in splits:
        try:
            results.append(run_split(split, args.snapshot.resolve(), args.media_root.resolve(),
                                     args.min_duration, args.jobs, args.delete_bad))
        except FileNotFoundError as exc:
            logging.error("%s", exc)
    print("\n=== Sphere360 media verification ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
