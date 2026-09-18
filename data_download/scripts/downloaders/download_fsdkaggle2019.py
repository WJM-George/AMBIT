#!/usr/bin/env python3
"""Download FSDKaggle2019 (DCASE 2019 Freesound Audio Tagging) from Zenodo.

Dataset: 29,266 clips, 80 AudioSet ontology labels, ~26.9 GB compressed.
Zenodo: https://zenodo.org/records/3612637

Default storage layout (matches project convention):
  ${AMBIT_DATA_ROOT}/datasets/fsdkaggle2019/
    archives/     # raw .zip / split parts from Zenodo
    extracted/    # optional, after --extract

Planned disk layout for the full VAE corpus:
  ${AMBIT_DATA_ROOT}/datasets/audioset/      # ~2.4 TB (AUDIO_DATASET_SECONDARY_ROOT)
  ${AMBIT_DATA_ROOT}/datasets/
    vggsound/      # ~338 GB
    musiccaps/     # ~3 MB metadata (wav via YouTube separately)
    picoaudio/     # ~912 MB
    fsdkaggle2019/ # ~27 GB  <-- this script

Setup
-----
    uv sync

Examples
--------
# Full download (resumable) to ${AMBIT_DATA_ROOT}/datasets/fsdkaggle2019/archives/
uv run python scripts/downloaders/download_fsdkaggle2019.py

# Download + extract wav/labels (needs unzip; split noisy train needs 7z or zip)
uv run python scripts/downloaders/download_fsdkaggle2019.py --extract

# Only metadata + curated train first (smoke test)
uv run python scripts/downloaders/download_fsdkaggle2019.py --zenodo-file meta --zenodo-file curated
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audio_dataset_download.downloader import main


if __name__ == "__main__":
    main(["fsdkaggle2019", *sys.argv[1:]])
