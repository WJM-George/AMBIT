#!/usr/bin/env python3
"""Download PicoAudio (~912 MB) to ${AMBIT_DATA_ROOT}/datasets/picoaudio/.

    uv run python scripts/downloaders/download_picoaudio.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio_dataset_download.downloader import main


if __name__ == "__main__":
    main(["picoaudio", *sys.argv[1:]])
