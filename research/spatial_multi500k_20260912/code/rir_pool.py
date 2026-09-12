"""Bind the shared RIR helper to this workflow's frozen predecessor."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from common import PREVIOUS as ROOT
from stable_audio_tools.data.editing_rir_pool import FrozenRIRPool

_pool = FrozenRIRPool(ROOT)
native_module = _pool.native_module
bounded_rir_pool = _pool.bounded_rir_pool
