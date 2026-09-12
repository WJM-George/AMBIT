#!/usr/bin/env python3
"""Trim Sphere360 .webm clips to exactly TIME_INTERVAL seconds (default 10s).

Because --force-keyframes-at-cuts was removed (libopus can't re-encode
4ch ambisonics), yt-dlp snaps the end of each section to the nearest GOP
boundary (~20s for many 360° YouTube videos). The first 10s of every file is
the correct content; the rest is a bonus GOP that we discard here.

Trimming is done with ffmpeg stream-copy (-c copy), so:
  - No re-encoding: 4ch ambisonic Opus stream is preserved bit-for-bit.
  - Fast: typically ~0.1-0.3s per file on a 1440p webm.
  - Clips already ≤ (TIME_INTERVAL + TOLERANCE) seconds are skipped.

Usage
-----
# Trim all test clips in-place:
python3 scripts/downloaders/trim_sphere360_clips.py --split test

# Dry-run first (shows what would be trimmed):
python3 scripts/downloaders/trim_sphere360_clips.py --split test --dry-run

# Both splits, 32 workers:
python3 scripts/downloaders/trim_sphere360_clips.py --split train test --jobs 32
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_DATASET_ROOT = Path(os.environ.get("AUDIO_DATASET_SECONDARY_ROOT", os.environ.get("AMBIT_DATA_ROOT", "data")))
DEFAULT_MEDIA_ROOT = DEFAULT_DATASET_ROOT / "datasets" / "sphere360" / "media"
TIME_INTERVAL = 10.0   # seconds each Sphere360 clip should contain
TOLERANCE     = 0.5    # skip files already within this margin of TIME_INTERVAL


def ffprobe_duration(path: Path) -> float | None:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def trim_to_interval(path: Path, interval: float, dry_run: bool) -> str:
    """Trim *path* to *interval* seconds in-place. Returns a status string."""
    dur = ffprobe_duration(path)
    if dur is None:
        return f"SKIP (probe failed): {path.name}"
    if dur <= interval + TOLERANCE:
        return f"ok ({dur:.2f}s): {path.name}"
    if dry_run:
        return f"WOULD TRIM ({dur:.2f}s → {interval:.0f}s): {path.name}"

    # Write to a temp file in the same directory, then atomically replace.
    tmp = path.with_suffix(".tmp.webm")
    try:
        r = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(path),
                "-t", str(interval),
                "-c", "copy",          # stream-copy: no re-encode, 4ch Opus preserved
                str(tmp),
            ],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            return f"ERROR ({r.stderr.strip()[:120]}): {path.name}"
        tmp.replace(path)
        return f"TRIMMED ({dur:.2f}s → {interval:.0f}s): {path.name}"
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return f"ERROR ({exc}): {path.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", nargs="+", choices=["train", "test"], default=["test"],
                        help="Which split(s) to process.")
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT,
                        help="Root containing train/ and test/ subdirs.")
    parser.add_argument("--interval", type=float, default=TIME_INTERVAL,
                        help="Target clip length in seconds (default 10).")
    parser.add_argument("--jobs", type=int, default=32,
                        help="Parallel workers (ffmpeg is I/O-bound; 32 is safe).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be trimmed without doing it.")
    args = parser.parse_args()

    if not args.dry_run and (not subprocess.run(["which", "ffmpeg"], capture_output=True).returncode == 0):
        sys.exit("ffmpeg not found on PATH")

    for split in args.split:
        split_dir = args.media_root / split
        if not split_dir.is_dir():
            print(f"[trim] {split_dir} not found, skipping {split}")
            continue

        files = list(split_dir.glob("*.webm"))
        print(f"[trim] {split}: {len(files)} .webm files found in {split_dir}")
        if not files:
            continue

        n_trimmed = n_ok = n_err = 0
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(trim_to_interval, f, args.interval, args.dry_run): f for f in files}
            for i, fut in enumerate(as_completed(futures), 1):
                msg = fut.result()
                if msg.startswith("TRIMMED"):
                    n_trimmed += 1
                elif msg.startswith("ok"):
                    n_ok += 1
                else:
                    n_err += 1
                if i % 200 == 0 or i == len(files):
                    print(f"  [{i}/{len(files)}] trimmed={n_trimmed} ok={n_ok} err={n_err}")
                if msg.startswith("ERROR") or msg.startswith("SKIP"):
                    print(f"  {msg}")

        print(f"[trim] {split} done: {n_trimmed} trimmed, {n_ok} already ok, {n_err} errors")


if __name__ == "__main__":
    main()
