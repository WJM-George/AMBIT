#!/usr/bin/env python3
"""Download VGGSound (~338 GB) to ${AMBIT_DATA_ROOT}/datasets/vggsound/.

    cd . && uv sync
    uv run python scripts/downloaders/download_vggsound.py --max-workers 8
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio_dataset_download.downloader import main


if __name__ == "__main__":
    main(["vggsound", *sys.argv[1:]])
