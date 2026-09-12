"""Metadata provider compiling crop-aligned ScenePlan source-track controls."""
from __future__ import annotations

from typing import Any

import torch

from stable_audio_tools.data.spatial_story import (
    SOURCE_TRACK_FEATURES,
    compile_source_tracks,
)


class ScenePlanTrackMetadata:
    """Turn a canonical plan into latent-rate per-source controls.

    The preceding ``scene_plan`` provider must already have applied the same
    random crop as the VAE latent.  For naturally short clips, controls are
    compiled only over valid latent frames and zero-padded afterward, matching
    the dataset padding mask rather than stretching a 3-second event to 10 s.

    ``packed_source_features`` is the production layout: one Transfusion token
    per latent frame with ``max_sources * len(SOURCE_TRACK_FEATURES)`` channels.
    Keeping source slots inside the feature dimension avoids turning a
    four-source, 432-frame plan into 1,728 attention tokens.  The explicit 3-D
    layout remains available for geometry/debugging ablations.
    """

    def __init__(self, config: dict[str, Any]):
        self.plan_key = str(config.get("plan_key", "scene_plan"))
        self.output_key = str(config.get("output_key", "source_tracks"))
        self.max_sources = int(config.get("max_sources", 4))
        self.max_distance_m = float(config.get("max_distance_m", 20.0))
        self.layout = str(config.get("layout", "packed_source_features"))
        if self.max_sources <= 0 or self.max_distance_m <= 0:
            raise ValueError("max_sources and max_distance_m must be positive")
        if self.layout not in {
            "packed_source_features",
            "feature_source_time",
        }:
            raise ValueError(
                "scene_plan_tracks.layout must be packed_source_features or "
                "feature_source_time"
            )

    @staticmethod
    def _valid_frames(info: dict[str, Any], total_frames: int) -> int:
        value = info.get("padding_mask")
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if value is None:
            return total_frames
        mask = torch.as_tensor(value, dtype=torch.bool).flatten()
        if mask.numel() != total_frames:
            raise ValueError(
                f"padding mask has {mask.numel()} frames, expected {total_frames}"
            )
        valid = int(mask.sum())
        if valid <= 0:
            raise ValueError("source-track compiler received an all-padding sample")
        return valid

    def __call__(self, info: dict[str, Any], latents: torch.Tensor) -> dict[str, Any]:
        if self.plan_key not in info:
            raise KeyError(
                f"ScenePlanTrackMetadata requires '{self.plan_key}'; put the "
                "scene_plan provider before scene_plan_tracks"
            )
        if latents.ndim != 2:
            raise ValueError(f"expected latent [C,T], got {tuple(latents.shape)}")
        total_frames = int(latents.shape[-1])
        valid_frames = self._valid_frames(info, total_frames)
        compiled = compile_source_tracks(
            info[self.plan_key],
            num_frames=valid_frames,
            max_sources=self.max_sources,
            max_distance_m=self.max_distance_m,
        )
        # Compiler layout is [source, feature, time].  The packed production
        # representation preserves source-major slot order in the channel axis.
        if self.layout == "packed_source_features":
            tracks = compiled["tracks"].flatten(0, 1).contiguous()
        else:
            tracks = compiled["tracks"].permute(1, 0, 2).contiguous()
        if valid_frames < total_frames:
            tracks = torch.nn.functional.pad(tracks, (0, total_frames - valid_frames))
        return {
            self.output_key: tracks,
            f"{self.output_key}_aligned": True,
            f"{self.output_key}_features": list(SOURCE_TRACK_FEATURES),
            f"{self.output_key}_layout": self.layout,
            f"{self.output_key}_max_sources": self.max_sources,
            f"{self.output_key}_source_mask": compiled["source_mask"],
            f"{self.output_key}_source_ids": compiled["source_ids"],
            f"{self.output_key}_valid_frames": valid_frames,
        }


def create_custom_metadata(config: dict[str, Any]):
    return ScenePlanTrackMetadata(config)


__all__ = ["ScenePlanTrackMetadata", "create_custom_metadata"]
