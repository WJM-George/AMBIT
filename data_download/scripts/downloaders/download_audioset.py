#!/usr/bin/env python3
"""Download AudioSet wav snapshot (~2.4 TB) to /mnt/sdb/audio_dataset/datasets/audioset/.

    cd /home/tanhe/dataset_storage && uv sync
    uv run python scripts/downloaders/download_audioset.py --max-workers 8
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio_dataset_download.downloader import main


if __name__ == "__main__":
    main(["audioset", *sys.argv[1:]])
