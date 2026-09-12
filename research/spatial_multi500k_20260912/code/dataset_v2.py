"""Compatibility entry for the shared compound editing reader."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import *
from stable_audio_tools.data.sceneplan_compound_editing_dataset import (
    EDITING_PAIR_CONTRACT,
    EDITING_INSTRUCTION_CONTRACT,
    ScenePlanTransfusionEditingDataset,
)
