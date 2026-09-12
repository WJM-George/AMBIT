#!/usr/bin/env python3
"""Download the actual MusicCaps audio clips from YouTube.

The Hugging Face repo `google/MusicCaps` ships only `musiccaps-public.csv`
(YouTube IDs + 10s start/end stamps + captions); the music itself is NOT
redistributed because it is copyrighted. This script reads that CSV and uses
yt-dlp + ffmpeg to fetch and trim each 10-second clip to a wav file.

Prereqs (already present in this repo's setup):
    - yt-dlp           (uv run pip show yt-dlp / on PATH)
    - ffmpeg           (on PATH)
    - deno (optional)  solves YouTube's n-challenge; without it many formats
                       are skipped. Installed at ~/.deno/bin on this box.

Run the metadata downloader first so the CSV exists:
    cd . && uv sync
    uv run python scripts/downloaders/download_musiccaps.py

Then fetch the audio (resumable; already-downloaded clips are skipped):
    uv run python scripts/downloaders/download_musiccaps_audio.py
    uv run python scripts/downloaders/download_musiccaps_audio.py --limit 20   # smoke test
    uv run python scripts/downloaders/download_musiccaps_audio.py --eval-only --workers 4

YouTube blocks most datacenter IPs. If you see "Sign in to confirm you're not
a bot" or near-100% failures, export browser cookies to a Netscape cookies.txt
and pass --cookies /path/to/cookies.txt (or set MUSICCAPS_COOKIE).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get("AUDIO_DATASET_ROOT", os.environ.get("AMBIT_DATA_ROOT", "data")))
DATASET_DIR = DEFAULT_ROOT / "datasets" / "musiccaps"
DEFAULT_CSV = DATASET_DIR / "snapshot" / "musiccaps-public.csv"
DEFAULT_OUTPUT = DATASET_DIR / "audio"
TMP_ROOT = Path(os.environ.get("AUDIO_DATASET_TMP", os.environ.get("AMBIT_CACHE_ROOT", "cache/tmp")))
URL_BASE = "https://www.youtube.com/watch?v="


def discover_cookie() -> str | None:
    """Locate a YouTube cookies.txt. Env vars win, then common on-disk spots.

    YouTube rejects datacenter IPs without a logged-in session, so a valid
    Netscape cookies.txt is effectively required here. The previous default only
    looked under the dataset dir, which silently resolved to None when the file
    actually lived in the shared tmp dir -- yielding the "Sign in to confirm
    you're not a bot" failures. We now search the obvious locations so a present
    cookie file is picked up without remembering the exact flag.
    """
    for env_var in ("MUSICCAPS_COOKIE", "YOUTUBE_COOKIE", "SPHERE360_COOKIE"):
        value = os.environ.get(env_var)
        if value and Path(value).exists():
            return value
    candidates = [
        TMP_ROOT / "youtube_cookies_2.txt",
        TMP_ROOT / "youtube_cookies.txt",
        DATASET_DIR / "youtube_cookies.txt",
        Path.home() / "youtube_cookies.txt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def ensure_js_runtime_on_path() -> None:
    """Prepend ~/.deno/bin so yt-dlp subprocesses can solve YouTube's n-challenge.

    deno installs to ~/.deno/bin, which is not on PATH for non-interactive
    shells; without a JS runtime yt-dlp skips most real formats.
    """
    deno_bin = Path.home() / ".deno" / "bin"
    if deno_bin.is_dir():
        os.environ["PATH"] = f"{deno_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def setup_logging() -> Path:
    log_dir = DEFAULT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"musiccaps-audio-{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_path


def load_rows(csv_path: Path, balanced_only: bool, eval_only: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if balanced_only and row.get("is_balanced_subset", "").strip().lower() != "true":
                continue
            if eval_only and row.get("is_audioset_eval", "").strip().lower() != "true":
                continue
            rows.append(row)
    return rows


def build_command(
    ytid: str,
    start: str,
    end: str,
    output_template: str,
    sample_rate: int,
    cookies: str | None,
    proxy: str | None,
    remote_components: str | None,
) -> list[str]:
    cmd = ["yt-dlp", "--quiet", "--no-playlist"]
    # YouTube's n-challenge now needs an EJS "challenge solver" component (fetched
    # from GitHub) IN ADDITION to a JS runtime (deno). yt-dlp skips that download
    # by default, which yields "Only images are available" -> "Requested format
    # is not available". Enabling it (and deno on PATH) restores real audio
    # formats. The component is cached after the first fetch.
    if remote_components:
        cmd += ["--remote-components", remote_components]
    cmd += [
        "--force-keyframes-at-cuts",
        "--retries", "5",
        "--fragment-retries", "5",
        # Be gentle: YouTube throttles bursts from one account/IP and then
        # returns degraded player responses. A small inter-request sleep helps.
        "--sleep-requests", "1",
        "-x",
        "--audio-format", "wav",
        # Fall back to a combined stream when no pure audio format is exposed;
        # -x still extracts the audio track.
        "-f", "bestaudio/best",
        "--download-sections", f"*{start}-{end}",
    ]
    if sample_rate > 0:
        cmd += ["--postprocessor-args", f"ffmpeg:-ar {sample_rate}"]
    cmd += ["-o", output_template, f"{URL_BASE}{ytid}"]
    if cookies:
        cmd += ["--cookies", cookies]
    if proxy:
        cmd += ["--proxy", proxy]
    return cmd


def download_clip(
    row: dict[str, str],
    output_dir: Path,
    sample_rate: int,
    cookies: str | None,
    proxy: str | None,
    num_attempts: int,
    remote_components: str | None,
) -> tuple[str, bool, str]:
    ytid = row["ytid"]
    out_path = output_dir / f"{ytid}.wav"
    if out_path.exists() and out_path.stat().st_size > 0:
        return ytid, True, "exists"

    output_template = str(output_dir / "%(id)s.%(ext)s")
    cmd = build_command(
        ytid, row["start_s"], row["end_s"], output_template,
        sample_rate, cookies, proxy, remote_components,
    )

    last_err = ""
    for attempt in range(1, num_attempts + 1):
        if attempt > 1:
            # Back off with jitter so throttled/format errors get a cooldown
            # instead of hammering YouTube with identical immediate retries.
            time.sleep(min(30.0, 4.0 * (attempt - 1)) + random.uniform(0, 2))
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as err:
            tail = (err.output or "").strip().splitlines()[-1:] or [""]
            last_err = tail[0][:300]
            continue
        if out_path.exists() and out_path.stat().st_size > 0:
            return ytid, True, f"ok (attempt {attempt})"
        last_err = "yt-dlp exited 0 but no wav produced"
    return ytid, False, last_err or "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to musiccaps-public.csv.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directory for wav clips.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel yt-dlp jobs. Higher risks YouTube rate-limits.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N rows (smoke test).")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Resample wav to this rate (0 = keep source).")
    parser.add_argument("--attempts", type=int, default=3, help="Per-clip yt-dlp retry attempts.")
    parser.add_argument("--balanced-only", action="store_true", help="Only the is_balanced_subset rows.")
    parser.add_argument("--eval-only", action="store_true", help="Only the is_audioset_eval rows.")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy URL passed to yt-dlp.")
    parser.add_argument(
        "--remote-components",
        type=str,
        default="ejs:github",
        help=(
            "yt-dlp --remote-components value for YouTube's JS challenge solver "
            "(needs deno on PATH). Default 'ejs:github'; use 'ejs:npm' as an "
            "alternative, or '' / 'none' to disable."
        ),
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=discover_cookie(),
        help=(
            "Netscape cookies.txt for YouTube (required on server IPs). Defaults to "
            "$MUSICCAPS_COOKIE / a youtube_cookies*.txt under $AUDIO_DATASET_TMP "
            f"({TMP_ROOT}) or the dataset dir."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = setup_logging()

    if shutil.which("yt-dlp") is None:
        sys.exit("yt-dlp is not installed. Install with: uv run pip install -U yt-dlp")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not installed. Install with: sudo apt-get install -y ffmpeg")

    ensure_js_runtime_on_path()
    remote_components = args.remote_components
    if remote_components and remote_components.lower() in {"none", "off", ""}:
        remote_components = None
    if shutil.which("deno") is None:
        logging.warning(
            "No 'deno' JS runtime on PATH. YouTube's n-challenge will fail and "
            "every clip will report 'Requested format is not available'. "
            "Install: curl -fsSL https://deno.land/install.sh | sh"
        )

    if not args.csv.exists():
        sys.exit(
            f"CSV not found: {args.csv}\n"
            "Run the metadata downloader first:\n"
            "  uv run python scripts/downloaders/download_musiccaps.py"
        )

    rows = load_rows(args.csv, args.balanced_only, args.eval_only)
    if args.limit is not None:
        rows = rows[: args.limit]

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cookies and not Path(args.cookies).exists():
        logging.warning("Cookies path does not exist, ignoring: %s", args.cookies)
        args.cookies = None
    if not args.cookies:
        logging.warning(
            "No YouTube cookies found. On a datacenter IP yt-dlp will almost "
            "certainly hit 'Sign in to confirm you're not a bot' and every clip "
            "will fail. Pass --cookies /path/to/cookies.txt (Netscape format) or "
            "set MUSICCAPS_COOKIE / place youtube_cookies.txt under %s.",
            TMP_ROOT,
        )

    logging.info("Log file: %s", log_path)
    logging.info("CSV: %s", args.csv)
    logging.info("Output: %s", output_dir)
    logging.info("Clips to process: %d (workers=%d, sample_rate=%s)", len(rows), args.workers, args.sample_rate)
    logging.info("Cookies: %s", args.cookies or "(none - YouTube will reject datacenter IPs)")
    logging.info("JS challenge solver (--remote-components): %s", remote_components or "(disabled)")

    ok = skipped = failed = 0
    fail_list = output_dir.parent / "musiccaps_audio_fail_list.txt"
    failures: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_clip, row, output_dir, args.sample_rate, args.cookies,
                args.proxy, args.attempts, remote_components,
            ): row["ytid"]
            for row in rows
        }
        done = 0
        for future in as_completed(futures):
            ytid, success, message = future.result()
            done += 1
            if success and message == "exists":
                skipped += 1
            elif success:
                ok += 1
            else:
                failed += 1
                failures.append(f"{ytid}\t{message}")
                logging.warning("FAIL %s: %s", ytid, message)
            if done % 50 == 0 or done == len(rows):
                logging.info("Progress %d/%d (ok=%d skipped=%d failed=%d)", done, len(rows), ok, skipped, failed)

    if failures:
        fail_list.write_text("\n".join(failures) + "\n", encoding="utf-8")
        logging.info("Wrote %d failures to %s", len(failures), fail_list)

    logging.info("Done. ok=%d skipped=%d failed=%d total=%d", ok, skipped, failed, len(rows))
    logging.info("Audio dir: %s", output_dir)


if __name__ == "__main__":
    main()
