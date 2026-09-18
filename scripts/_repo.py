"""Locate the AMBIT repository root from any nested scripts/ path."""
from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__)).resolve()
    if p.is_file():
        p = p.parent
    for cand in [p, *p.parents]:
        if (cand / "stable_audio_tools").is_dir() and (cand / "train_4ch.py").is_file():
            return cand
    raise RuntimeError(f"Could not find AMBIT repository root from {start}")
