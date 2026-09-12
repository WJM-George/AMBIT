#!/usr/bin/env python3
"""Split downloaded Sphere360 .webm clips into pure video + pure 4-channel FOA audio.

Each downloaded clip is a single .webm holding BOTH streams:
    stream 0 = VP9 equirectangular 360 video
    stream 1 = Opus 4-channel First-Order Ambisonics (ACN/SN3D, order [W, Y, Z, X])

Most players cannot render 4-channel ambisonic Opus, so the raw .webm seems silent
(the audio is fine - it just needs an ambisonic decoder). This tool demuxes each clip,
losslessly, into separate folders while KEEPING the original webm folder intact:

    media/<split>/<id>.webm          # original (kept, untouched)
    media/<split>_video/<id>.webm    # video only, stream-copied (no audio)
    media/<split>_audio/<id>.flac    # 4-channel FOA, lossless FLAC (opens in Reaper)
    media/<split>_preview/<id>.wav   # OPTIONAL 2-ch downmix you can actually hear (L=W+Y, R=W-Y)

Examples
--------
# Split the test split into <split>_video + <split>_audio (FLAC), keep webm:
python3 scripts/downloaders/split_sphere360_av.py --split test --jobs 8

# Also write a listenable stereo preview for spot-checking:
python3 scripts/downloaders/split_sphere360_av.py --split test --jobs 8 --preview

# Both splits:
python3 scripts/downloaders/split_sphere360_av.py --split both --jobs 8
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DEFAULT_DATASET_ROOT = Path(os.environ.get("AUDIO_DATASET_SECONDARY_ROOT", os.environ.get("AMBIT_DATA_ROOT", "data")))
DEFAULT_MEDIA_ROOT = DEFAULT_DATASET_ROOT / "datasets" / "sphere360" / "media"


def run_ffmpeg(cmd: list[str], final_path: Path) -> None:
    """Run ffmpeg writing to a .part file, then atomically rename on success."""
    part = final_path.with_suffix(final_path.suffix + ".part")
    part.unlink(missing_ok=True)
    full = cmd[:1] + ["-v", "error", "-y"] + cmd[1:] + [str(part)]
    proc = subprocess.run(full, capture_output=True, text=True)
    if proc.returncode != 0 or not part.exists() or part.stat().st_size < 1024:
        part.unlink(missing_ok=True)
        raise RuntimeError(proc.stderr.strip()[:300] or "ffmpeg produced no output")
    part.replace(final_path)


def split_clip(webm: Path, video_dir: Path, audio_dir: Path,
               preview_dir: Path | None, overwrite: bool) -> tuple[str, str]:
    clip_id = webm.stem
    video_out = video_dir / f"{clip_id}.webm"
    audio_out = audio_dir / f"{clip_id}.flac"
    preview_out = (preview_dir / f"{clip_id}.wav") if preview_dir else None

    try:
        # 1) video-only, stream copy (lossless, fast)
        if overwrite or not video_out.exists():
            run_ffmpeg(["ffmpeg", "-i", str(webm), "-map", "0:v:0", "-an",
                        "-c:v", "copy", "-f", "webm"], video_out)
        # 2) 4-channel FOA -> FLAC (plain encode; keeps [W,Y,Z,X] sample order).
        #    NB: ffmpeg warns "layout not supported by FLAC" - harmless, data order is intact.
        if overwrite or not audio_out.exists():
            run_ffmpeg(["ffmpeg", "-i", str(webm), "-map", "0:a:0",
                        "-c:a", "flac", "-f", "flac"], audio_out)
        # 3) optional stereo preview (so a human can hear it without an ambisonic decoder)
        if preview_out is not None and (overwrite or not preview_out.exists()):
            run_ffmpeg(["ffmpeg", "-i", str(webm), "-map", "0:a:0",
                        "-filter:a", "pan=stereo|c0=c0+0.6*c1|c1=c0-0.6*c1",
                        "-c:a", "pcm_s16le", "-f", "wav"], preview_out)
        return clip_id, "ok"
    except Exception as exc:  # noqa: BLE001 - record and continue
        logging.warning("FAILED %s: %s", clip_id, exc)
        return clip_id, f"fail: {exc}"


def run_split(split: str, media_root: Path, jobs: int, preview: bool, overwrite: bool) -> dict:
    src_dir = media_root / split
    if not src_dir.is_dir():
        logging.error("%s: source dir not found: %s", split, src_dir)
        return {"split": split, "clips": 0, "ok": 0, "fail": 0}

    video_dir = media_root / f"{split}_video"
    audio_dir = media_root / f"{split}_audio"
    preview_dir = (media_root / f"{split}_preview") if preview else None
    for d in (video_dir, audio_dir, preview_dir):
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    clips = sorted(src_dir.glob("*.webm"))
    logging.info("%s: %d source clips -> video=%s audio=%s preview=%s",
                 split, len(clips), video_dir, audio_dir, bool(preview))

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(split_clip, w, video_dir, audio_dir, preview_dir, overwrite)
                   for w in clips]
        for i, fut in enumerate(as_completed(futures), 1):
            _, status = fut.result()
            if status == "ok":
                ok += 1
            else:
                fail += 1
            if i % 500 == 0:
                logging.info("  %s: %d/%d done (ok=%d fail=%d)", split, i, len(clips), ok, fail)

    logging.info("%s DONE: ok=%d fail=%d (video->%s, audio->%s)", split, ok, fail, video_dir, audio_dir)
    return {"split": split, "clips": len(clips), "ok": ok, "fail": fail}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    parser.add_argument("--media-root", type=Path, default=DEFAULT_MEDIA_ROOT)
    parser.add_argument("--jobs", type=int, default=8, help="Parallel ffmpeg workers.")
    parser.add_argument("--preview", action="store_true",
                        help="Also write a 2-channel listenable stereo downmix (L=W+Y, R=W-Y).")
    parser.add_argument("--overwrite", action="store_true", help="Re-create existing outputs.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not shutil_which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH.")
    splits = ["test", "train"] if args.split == "both" else [args.split]
    results = [run_split(s, args.media_root.resolve(), args.jobs, args.preview, args.overwrite)
               for s in splits]
    print("\n=== Sphere360 A/V split ===")
    for r in results:
        print(r)


def shutil_which(name: str) -> str | None:
    import shutil
    return shutil.which(name)


if __name__ == "__main__":
    main()
