#!/usr/bin/env python3
"""Download Sphere360 media (360 video + 4ch FOA audio) from YouTube.

The Hugging Face repo `omniaudio/Sphere360` only ships split lists + tooling,
not the media. This wrapper drives the repo's own `download_list_360()` to fetch
the actual clips with yt-dlp/ffmpeg.

It does NOT start downloading on import; run it explicitly with arguments.

Examples
--------
# 1) Probe the first 5 test clips (recommended first test; YouTube often requires cookies):
python3 scripts/downloaders/download_sphere360_media.py --split test --mode single --end-index 5 --jobs 1 --cookie /path/to/cookies.txt

# 2) Full test split (small, ~3k clips), 8 parallel jobs:
python3 scripts/downloaders/download_sphere360_media.py --split test --jobs 8 --cookie /path/to/cookies.txt

# 3) Train split in a grouped chunk (range indexes grouped videos, not clip rows):
python3 scripts/downloaders/download_sphere360_media.py --split train --start-index 0 --end-index 250 --jobs 8 --cookie /path/to/cookies.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

DEFAULT_DATASET_ROOT = Path(os.environ.get("AUDIO_DATASET_SECONDARY_ROOT", os.environ.get("AMBIT_DATA_ROOT", "data")))
DEFAULT_SPHERE360_ROOT = DEFAULT_DATASET_ROOT / "datasets" / "sphere360"
DEFAULT_SNAPSHOT = DEFAULT_SPHERE360_ROOT / "snapshot"
DEFAULT_MEDIA_ROOT = DEFAULT_SPHERE360_ROOT / "media"
DEFAULT_COOKIE = Path(os.environ.get("SPHERE360_COOKIE", str(DEFAULT_SPHERE360_ROOT / "youtube_cookies.txt")))
TIME_INTERVAL = 10  # each Sphere360 clip is 10 seconds


def load_repo_downloader(snapshot: Path):
    """Import the repo's self-contained core download_list module by file path."""
    module_path = snapshot / "toolset" / "crawl" / "core" / "download" / "download_list.py"
    if not module_path.exists():
        raise FileNotFoundError(
            f"Could not find Sphere360 downloader at {module_path}. "
            "Make sure the snapshot has been downloaded first "
            "(python3 scripts/downloaders/download_sphere360.py)."
        )
    spec = importlib.util.spec_from_file_location("sphere360_download_list", module_path)
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so multiprocessing can resolve the
    # module by name. download_list_360() runs a Pool that pickles its worker
    # functions by module reference; without this the workers raise
    # PicklingError: Can't pickle <function download_360_segments_process>.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_grouped_file(split_file: Path, grouped_file: Path) -> int:
    """Convert `{video_id}_{start}` lines into `{video_id} {s1,s2,...}` lines.

    Grouping by video means yt-dlp is invoked once per video (with multiple
    --download-sections) instead of once per clip, which is far more efficient.
    Returns the number of unique videos.
    """
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            video_id, _, start = line.rpartition("_")
            if not video_id or not start.isdigit():
                continue
            if video_id not in groups:
                groups[video_id] = []
                order.append(video_id)
            groups[video_id].append(int(start))

    grouped_file.parent.mkdir(parents=True, exist_ok=True)
    with grouped_file.open("w", encoding="utf-8") as handle:
        for video_id in order:
            starts = ",".join(str(s) for s in sorted(set(groups[video_id])))
            handle.write(f"{video_id} {starts}\n")
    return len(order)


def ensure_js_runtime_on_path() -> None:
    """Make a JS runtime (deno) discoverable by the yt-dlp subprocesses.

    yt-dlp needs deno to solve YouTube's n-challenge; without it every real
    (non-image) format is skipped and the 4-channel ambisonic stream cannot be
    fetched. Deno installs to ~/.deno/bin, which is not on PATH for
    non-interactive shells, so prepend it here.
    """
    deno_bin = Path.home() / ".deno" / "bin"
    if deno_bin.is_dir():
        os.environ["PATH"] = f"{deno_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Which split list to download.")
    parser.add_argument(
        "--mode",
        choices=["grouped", "single"],
        default="grouped",
        help="grouped: one yt-dlp call per video (efficient, default). single: one call per clip.",
    )
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT, help="Path to the downloaded HF snapshot.")
    parser.add_argument("--output", type=Path, default=None, help="Output dir (default: <media_root>/<split>).")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Parallel videos. This box is 128-core; 16-24 is usually safe (watch fails).")
    parser.add_argument("--client", type=str, default="tv,web_embedded",
                        help="yt-dlp youtube:player_client. 'tv,web_embedded' is ~2.5x faster than 'all' "
                             "and still exposes the 4ch ambisonic (itag 338). Use 'all' for max coverage.")
    parser.add_argument("--height", type=int, default=1440,
                        help="Max video height. Lower (e.g. 720) is much faster/smaller; audio (FOA) is unchanged.")
    parser.add_argument("--start-index", type=int, default=None, help="Start index into the (grouped) list.")
    parser.add_argument("--end-index", type=int, default=None, help="End index (exclusive) into the (grouped) list.")
    parser.add_argument("--proxy", type=str, default=None, help="Proxy URL passed to yt-dlp.")
    parser.add_argument(
        "--cookie",
        type=str,
        default=(str(DEFAULT_COOKIE) if DEFAULT_COOKIE.exists() else None),
        help=(
            "Path to a Netscape cookies.txt. Required for YouTube on server IPs. "
            f"Defaults to {DEFAULT_COOKIE} when present, or set SPHERE360_COOKIE."
        ),
    )
    parser.add_argument(
        "--list-prefix",
        type=str,
        default=None,
        help=(
            "Prefix for success/fail list files under the media root. "
            "Defaults to a split/mode/range-specific name so chunked runs do not overwrite each other."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if shutil.which("yt-dlp") is None:
        sys.exit("yt-dlp is not installed. Install it with: python3 -m pip install --upgrade yt-dlp")
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is not installed. Install it with: sudo apt-get install -y ffmpeg")

    ensure_js_runtime_on_path()
    if shutil.which("deno") is None:
        print(
            "[sphere360][WARN] No 'deno' JS runtime on PATH. yt-dlp needs it to solve "
            "YouTube's n-challenge; without it the 4ch ambisonic format is skipped and "
            "downloads fail. Install: curl -fsSL https://deno.land/install.sh | sh"
        )

    # Tune speed/coverage via env, read by the toolset's download_360_segments().
    os.environ["SPHERE360_PLAYER_CLIENT"] = args.client
    os.environ["SPHERE360_FORMAT"] = f"bv[height<={args.height}]+ba[audio_channels=4]"
    print(f"[sphere360] cookie={args.cookie}")
    print(f"[sphere360] player_client={args.client} format={os.environ['SPHERE360_FORMAT']}")

    snapshot = args.snapshot.resolve()
    downloader = load_repo_downloader(snapshot)

    split_file = snapshot / "dataset" / "split" / f"{args.split}.txt"
    if not split_file.exists():
        sys.exit(f"Split file not found: {split_file}")

    output_dir = (args.output or (DEFAULT_MEDIA_ROOT / args.split)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    range_label = f"{args.start_index or 0}-{args.end_index or 'end'}"
    list_prefix = args.list_prefix or f"{args.split}_{args.mode}_{range_label}"
    success_list = str(output_dir.parent / f"{list_prefix}_success_list.txt")
    fail_list = str(output_dir.parent / f"{list_prefix}_fail_list.txt")

    if args.mode == "grouped":
        grouped_file = output_dir.parent / f"{args.split}_grouped.txt"
        unique_videos = build_grouped_file(split_file, grouped_file)
        print(f"[sphere360] {args.split}: grouped into {unique_videos} unique videos -> {grouped_file}")
        input_file = str(grouped_file)
        specify_start = "multiple"
    else:
        input_file = str(split_file)
        specify_start = "single"

    print(f"[sphere360] input={input_file}")
    print(f"[sphere360] output={output_dir}")
    print(f"[sphere360] mode={args.mode} jobs={args.jobs} range=[{args.start_index}:{args.end_index}]")

    downloader.download_list_360(
        input_file=input_file,
        output_folder=str(output_dir),
        start_index=args.start_index,
        end_index=args.end_index,
        specify_start=specify_start,
        proxy=args.proxy,
        time_interval=TIME_INTERVAL,
        fail_list_name=fail_list,
        success_list_name=success_list,
        jobs=args.jobs,
        cookie=args.cookie,
    )
    print(f"[sphere360] done. success -> {success_list}, fail -> {fail_list}")


if __name__ == "__main__":
    main()
