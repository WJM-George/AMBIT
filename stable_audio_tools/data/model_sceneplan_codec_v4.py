"""Fifteen-second frame-aligned ScenePlan codec for the canonical P11.

Version 4 keeps the executable codec-v3 grammar and quantizers but extends
the atomic VAE-frame vocabulary from 432 to 648 frames.  A distinct codec
version is required because adding frame ids shifts every subsequent token
range and therefore changes checkpoint state.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .model_sceneplan_codec import DEFAULT_TEXT_CORPUS
from .model_sceneplan_codec_v3 import (
    CODEC_SCHEMA,
    ModelScenePlanCodecV3,
    _create_model_sceneplan_frame_codec_artifact,
    iter_model_sceneplan_text_fields,
)


CODEC_NAME = "model_sceneplan_codec_v4"
CODEC_VERSION = 4
MAX_FRAMES = 648


class ModelScenePlanCodecV4(ModelScenePlanCodecV3):
    """Encode and constrain plans across P10's complete 648-frame envelope."""

    codec_name = CODEC_NAME
    codec_version = CODEC_VERSION
    max_frames = MAX_FRAMES
    codec_label = "v4"


def create_model_sceneplan_codec_v4_artifact(
    output_dir: str | os.PathLike[str],
    texts: Iterable[str] = DEFAULT_TEXT_CORPUS,
    *,
    vocab_size: int = 4096,
    text_vocab_size: int = 1536,
    overwrite: bool = False,
) -> Path:
    """Create one immutable 648-frame v4 artifact."""

    return _create_model_sceneplan_frame_codec_artifact(
        output_dir,
        texts,
        codec_name=CODEC_NAME,
        codec_version=CODEC_VERSION,
        max_frames=MAX_FRAMES,
        vocab_size=vocab_size,
        text_vocab_size=text_vocab_size,
        overwrite=overwrite,
    )


__all__ = [
    "CODEC_NAME",
    "CODEC_SCHEMA",
    "CODEC_VERSION",
    "MAX_FRAMES",
    "ModelScenePlanCodecV4",
    "create_model_sceneplan_codec_v4_artifact",
    "iter_model_sceneplan_text_fields",
]
