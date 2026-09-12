from __future__ import annotations

import types

import torch

from stable_audio_tools.data.spatial_conversation_metadata import (
    SpatialFamilyMetadata,
)


def _metadata(*, independent_turns: bool) -> SpatialFamilyMetadata:
    metadata = SpatialFamilyMetadata.__new__(SpatialFamilyMetadata)
    metadata.family_record_key = "spatial_family"
    metadata.turns_key = "turns"
    metadata.independent_turns = independent_turns

    def compile_record(self, info, target, record, *, previous_override):
        previous, present = previous_override
        return {
            "target": target.clone(),
            "previous": previous.clone(),
            "present": present,
            "turn": record["turn"],
        }

    metadata._compile_record = types.MethodType(compile_record, metadata)
    return metadata


def _family() -> tuple[dict, torch.Tensor]:
    latents = torch.stack(
        [torch.full((2, 3), float(index + 1)) for index in range(4)]
    )
    info = {
        "family_id": "family",
        "spatial_family": {
            "family_id": "family",
            "split": "train",
            "turns": [{"turn": index} for index in range(4)],
        },
    }
    return info, latents


def test_spatial_family_metadata_preserves_sequential_previous_state():
    info, latents = _family()
    output = _metadata(independent_turns=False)(info, latents)

    turns = output["family_turn_metadata"]
    assert [turn["present"] for turn in turns] == [False, True, True, True]
    assert torch.count_nonzero(turns[0]["previous"]) == 0
    for index in range(1, 4):
        assert torch.equal(turns[index]["previous"], latents[index - 1].float())


def test_spatial_family_metadata_can_compile_independent_creation_turns():
    info, latents = _family()
    output = _metadata(independent_turns=True)(info, latents)

    turns = output["family_turn_metadata"]
    assert [turn["present"] for turn in turns] == [False, False, False, False]
    assert all(torch.count_nonzero(turn["previous"]) == 0 for turn in turns)
