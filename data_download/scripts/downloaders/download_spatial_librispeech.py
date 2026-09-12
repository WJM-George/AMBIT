#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio_dataset_download.downloader import main


if __name__ == "__main__":
    main(["spatial_librispeech", *sys.argv[1:]])
