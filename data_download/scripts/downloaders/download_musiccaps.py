#!/usr/bin/env python3
"""Download MusicCaps metadata (csv, ~3 MB) to ${AMBIT_DATA_ROOT}/datasets/musiccaps/.

    cd . && uv sync
    uv run python scripts/downloaders/download_musiccaps.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio_dataset_download.downloader import main


if __name__ == "__main__":
    main(["musiccaps", *sys.argv[1:]])
