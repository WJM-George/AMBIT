#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_dataset_download.catalog import DATASETS


DEFAULT_ORDER = [
    "mrsdrama",
    "bewo_1m",
    "mrsaudio",
    "sphere360",
    "audiox_ifcaps",
    "audiocaps",
    "audio_flan",
    "spatial_librispeech",
    "yt_ambient",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download all configured datasets.")
    parser.add_argument("--include-gated", action="store_true", help="Include gated datasets that require HF_TOKEN/auth.")
    parser.add_argument("--include-github", action="store_true", help="Clone GitHub project repositories such as YT-Ambient.")
    parser.add_argument("--dataset-workers", type=int, default=1, help="Number of datasets to download in parallel.")
    parser.add_argument("--snapshot-workers", type=int, default=8, help="Workers for Hugging Face snapshot downloads.")
    parser.add_argument(
        "--parquet-only",
        action="store_true",
        help="Download Hugging Face auto-converted parquet files instead of full repo snapshots.",
    )
    parser.add_argument(
        "--sls-count",
        type=int,
        help="Spatial LibriSpeech sample count. Omit to download metadata only.",
    )
    parser.add_argument("--sls-all", action="store_true", help="Download every Spatial LibriSpeech sample listed in metadata.")
    parser.add_argument("--sls-include-noise", action="store_true", help="Also download Spatial LibriSpeech noise samples.")
    parser.add_argument("--sls-workers", type=int, default=8, help="Parallel workers for Spatial LibriSpeech file downloads.")
    return parser.parse_args()


def command_for_dataset(key: str, mode: str, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "scripts/download_dataset.py",
        key,
        "--mode",
        mode,
        "--max-workers",
        str(args.snapshot_workers),
    ]
    if key == "spatial_librispeech":
        if args.sls_count is not None:
            command.extend(["--sls-count", str(args.sls_count)])
        if args.sls_all:
            command.append("--sls-all")
        if args.sls_include_noise:
            command.append("--sls-include-noise")
        command.extend(["--sls-workers", str(args.sls_workers)])
    return command


def main() -> None:
    args = parse_args()
    commands: list[list[str]] = []
    for key in DEFAULT_ORDER:
        spec = DATASETS[key]
        if spec.gated and not args.include_gated:
            print(f"Skipping gated dataset {spec.name}; pass --include-gated after login/access approval.")
            continue
        if spec.kind == "github_repo" and not args.include_github:
            print(f"Skipping GitHub repo {spec.name}; pass --include-github to clone it.")
            continue

        mode = "auto"
        if args.parquet_only and spec.kind == "hf_snapshot":
            mode = "hf_parquet"

        commands.append(command_for_dataset(key, mode, args))

    def run_command(command: list[str]) -> None:
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    with ThreadPoolExecutor(max_workers=args.dataset_workers) as executor:
        futures = [executor.submit(run_command, command) for command in commands]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
