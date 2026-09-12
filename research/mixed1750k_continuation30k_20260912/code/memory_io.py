"""Compatibility entry for shared in-memory rendering helpers."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stable_audio_tools.data.editing_memory_io import memoized_sources, memory_audio_reader
