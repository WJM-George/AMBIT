#!/usr/bin/env python3
"""Extract a SUBSET of VGGSound to mono wav sources for spatial synthesis.

VGGSound ships as tar.gz of MP4 *video* (entries look like
``scratch/.../VGGSound_final/video/<ytid>_<start6>.mp4``). Our synthesis pipeline
only needs the audio track as a mono wav, so this script:

  * untars mp4 from the first tarballs into <out>/video/ (KEPT, not deleted),
  * pulls one mono wav per clip via ffmpeg into <out>/audio/,
  * stops once --max-clips clips are available (subset), and is resumable
    (existing wavs are skipped; partial/failed wavs are removed).

Disk: video stays (~17 GB/tarball), audio is small (~0.2 MB/clip). 80k clips
=> ~8 tarballs of mp4 kept + ~16 GB wav.

Run:
    cd ./stable-audio-tools
    uv run python dataset/indexing/extract_vggsound.py \
        --snapshot ${AMBIT_DATA_ROOT}/datasets/vggsound/snapshot \
        --out ${AMBIT_DATA_ROOT}/datasets/vggsound/extracted \
        --max-clips 80000 --sr 48000 --jobs 16
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

LOG = logging.getLogger("vggsound_extract")

# Tar entries are scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/<f>.mp4
# Strip the 6 leading dirs so files land flat under <out>/video/<f>.mp4.
_STRIP_COMPONENTS = 6


def _extract_audio(mp4: str, wav: str, sr: int) -> bool:
    """ffmpeg mono wav from an mp4's first audio stream. Returns True on success."""
    if os.path.exists(wav) and os.path.getsize(wav) > 1024:
        return True
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", mp4,
           "-map", "0:a:0", "-ac", "1", "-ar", str(sr), wav]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = r.returncode == 0 and os.path.exists(wav) and os.path.getsize(wav) > 1024
    except Exception:  # noqa: BLE001
        ok = False
    if not ok and os.path.exists(wav):
        try:
            os.remove(wav)
        except OSError:
            pass
    return ok


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path,
                    default=Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/vggsound/snapshot"))
    ap.add_argument("--out", type=Path,
                    default=Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/datasets/vggsound/extracted"))
    ap.add_argument("--max-clips", type=int, default=80000)
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--jobs", type=int, default=16)
    args = ap.parse_args()

    video_dir = args.out / "video"
    audio_dir = args.out / "audio"
    video_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    tarballs = sorted(args.snapshot.glob("vggsound_*.tar.gz"))
    if not tarballs:
        LOG.error("no vggsound_*.tar.gz under %s", args.snapshot)
        sys.exit(1)

    # 1) Untar mp4 until we have >= max_clips on disk (kept).
    for tb in tarballs:
        have = sum(1 for _ in video_dir.glob("*.mp4"))
        if have >= args.max_clips:
            break
        LOG.info("untar %s (have %d/%d mp4)", tb.name, have, args.max_clips)
        subprocess.run(["tar", "xzf", str(tb), "-C", str(args.out),
                        f"--strip-components={_STRIP_COMPONENTS}"], check=False)

    mp4s = sorted(video_dir.glob("*.mp4"))[: args.max_clips]
    LOG.info("have %d mp4; extracting mono wav @ %d Hz -> %s", len(mp4s), args.sr, audio_dir)

    # 2) ffmpeg audio in parallel (skip already-done).
    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_extract_audio, str(m), str(audio_dir / (m.stem + ".wav")), args.sr): m
                for m in mp4s}
        for n, fut in enumerate(as_completed(futs), 1):
            if fut.result():
                ok += 1
            else:
                fail += 1
            if n % 500 == 0 or n == len(mp4s):
                LOG.info("  %d/%d ok=%d fail=%d", n, len(mp4s), ok, fail)

    LOG.info("DONE wav ok=%d fail=%d -> %s", ok, fail, audio_dir)


if __name__ == "__main__":
    main()
