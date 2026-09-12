"""Direct four-event plus four-trajectory ScenePlan conditioning."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _assert_tensor_true(condition: Tensor, message: str) -> None:
    """Fail closed without synchronizing the CPU on CUDA hot paths."""

    condition = condition.reshape(())
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class ScenePlan44Conditioner(nn.Module):
    """Encode four source-local event and trajectory tracks without text pooling.

    Raw categorical ids have a semantic contract shared with caption roles:

    * ``-1``: unknown because classifier-free conditioning was requested;
    * ``0``: known inactive/absent at this frame;
    * ``1..4``: active persistent source id.

    Each source owns a fixed 64-channel block: a 32-channel categorical event
    embedding and a 32-channel geometric trajectory embedding.  Concatenating
    four blocks produces the 256 channels attached directly to the noisy FOA
    latent.  No Qwen state, transcript state, gain, or derived loudness enters
    this path.
    """

    def __init__(
        self,
        *,
        max_sources: int = 4,
        trajectory_feature_dim: int = 5,
        event_embedding_dim: int = 32,
        trajectory_embedding_dim: int = 32,
        output_dim: int = 256,
    ) -> None:
        super().__init__()
        self.max_sources = int(max_sources)
        self.trajectory_feature_dim = int(trajectory_feature_dim)
        self.event_embedding_dim = int(event_embedding_dim)
        self.trajectory_embedding_dim = int(trajectory_embedding_dim)
        self.per_source_dim = (
            self.event_embedding_dim + self.trajectory_embedding_dim
        )
        self.output_dim = int(output_dim)
        if self.max_sources != 4:
            raise ValueError("ScenePlan 4+4 requires exactly four persistent slots")
        if self.trajectory_feature_dim != 5:
            raise ValueError(
                "ScenePlan trajectory features must be sin/cos azimuth, "
                "sin/cos elevation, and log1p distance"
            )
        if self.output_dim != self.max_sources * self.per_source_dim:
            raise ValueError(
                "output_dim must equal four source blocks: "
                f"{self.output_dim} != {self.max_sources} * {self.per_source_dim}"
            )

        # Raw ids -1..4 map to embedding rows 0..5.  Raw 0 therefore maps to
        # padding_idx=1 and is held at exact zero; raw -1 remains trainable.
        self.event_embedding = nn.Embedding(
            self.max_sources + 2,
            self.event_embedding_dim,
            padding_idx=1,
        )
        nn.init.normal_(self.event_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.event_embedding.weight[1].zero_()

        self.trajectory_encoder = nn.Sequential(
            nn.Linear(self.trajectory_feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, self.trajectory_embedding_dim),
        )
        self.unknown_trajectory_embedding = nn.Parameter(
            torch.empty(self.trajectory_embedding_dim)
        )
        nn.init.normal_(self.unknown_trajectory_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        *,
        source_event_frame_ids: Tensor,
        source_trajectory_features: Tensor,
    ) -> Tensor:
        if source_event_frame_ids.ndim != 3:
            raise ValueError("source event ids must have shape [B,4,T]")
        if source_trajectory_features.ndim != 4:
            raise ValueError("source trajectories must have shape [B,4,T,5]")
        batch, sources, frames = source_event_frame_ids.shape
        if sources != self.max_sources:
            raise ValueError("source event ids must contain exactly four tracks")
        if tuple(source_trajectory_features.shape) != (
            batch,
            self.max_sources,
            frames,
            self.trajectory_feature_dim,
        ):
            raise ValueError("source event and trajectory tracks do not align")
        event_ids = source_event_frame_ids.to(torch.long)
        _assert_tensor_true(
            ((event_ids >= -1) & (event_ids <= self.max_sources)).all(),
            "source event ids must be in {-1,0,1,2,3,4}",
        )

        # Positive rows are slot-locked.  This catches a swapped-source bug at
        # the conditioner boundary rather than silently teaching wrong motion.
        expected = torch.arange(
            1,
            self.max_sources + 1,
            device=event_ids.device,
            dtype=event_ids.dtype,
        ).view(1, self.max_sources, 1)
        invalid_positive = (event_ids > 0) & event_ids.ne(expected)
        _assert_tensor_true(
            ~invalid_positive.any(),
            "an active event id is attached to the wrong source slot",
        )

        mapped_ids = event_ids + 1
        event = self.event_embedding(mapped_ids)

        trajectory_values = source_trajectory_features.to(
            dtype=self.trajectory_encoder[0].weight.dtype
        )
        _assert_tensor_true(
            torch.isfinite(trajectory_values).all(),
            "source trajectories contain non-finite values",
        )
        active = event_ids.gt(0).unsqueeze(-1)
        unknown = event_ids.eq(-1).unsqueeze(-1)
        known_inactive = event_ids.eq(0).unsqueeze(-1)
        _assert_tensor_true(
            ~(known_inactive & trajectory_values.ne(0)).any(),
            "known inactive frames must carry zero trajectory values",
        )
        encoded_trajectory = self.trajectory_encoder(trajectory_values)
        encoded_trajectory = torch.where(
            active,
            encoded_trajectory,
            torch.zeros_like(encoded_trajectory),
        )
        encoded_trajectory = torch.where(
            unknown,
            self.unknown_trajectory_embedding.view(1, 1, 1, -1).to(
                encoded_trajectory.dtype
            ),
            encoded_trajectory,
        )

        source_blocks = torch.cat(
            [event.to(encoded_trajectory.dtype), encoded_trajectory], dim=-1
        )
        # [B,4,T,64] -> [B,256,T].  No projection mixes persistent slots.
        return source_blocks.permute(0, 1, 3, 2).reshape(
            batch, self.output_dim, frames
        )


__all__ = ["ScenePlan44Conditioner"]
