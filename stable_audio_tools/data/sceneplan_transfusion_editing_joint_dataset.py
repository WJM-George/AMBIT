"""Joint AR/RF batches for audio-reference Transfusion Editing.

The underlying frozen pair row is loaded once.  Its clean source latent and
raw instruction feed Editing AR, its complete new ScenePlan is the AR target
and DiT control, and its aligned target latent supplies the rectified-flow
target.  No old/source ScenePlan is returned by this module.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from .model_sceneplan import validate_model_sceneplan
from .model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from .sceneplan_bucket_sampler import sceneplan_bucket_collation
from .sceneplan_transfusion_editing_ar_dataset import collate_editing_ar
from .sceneplan_transfusion_editing_dataset import (
    ScenePlanTransfusionEditingDataset,
    compile_editing_dit_plan_condition,
)
from .sceneplan_transfusion_editing_plan import (
    EDITING_AR_PLAN_POLICY,
    canonicalize_editing_plan,
)
from .sceneplan_transfusion_editing_m2d_clap import (
    ScenePlanTransfusionEditingM2DCLAPCache,
)


EDITING_JOINT_DATASET_CONTRACT = (
    "source_audio_instruction_to_canonical_new_plan_plus_aligned_rf_v2"
)


def _reject_old_sceneplan_keys(value: dict[str, Any], *, where: str) -> None:
    forbidden = {
        "old_sceneplan",
        "old_plan",
        "source_sceneplan",
        "source_plan",
        "previous_sceneplan",
        "previous_plan",
    }
    leaked = sorted(
        key
        for key in value
        if str(key).lower().replace("-", "_") in forbidden
    )
    if leaked:
        raise RuntimeError(
            f"old ScenePlan is forbidden in Editing AR {where}: {leaked}"
        )


class ScenePlanTransfusionEditingJointDataset(torch.utils.data.Dataset):
    """Add codec-v4 new-plan labels to one aligned Editing-DiT row."""

    def __init__(
        self,
        dataset: ScenePlanTransfusionEditingDataset,
        *,
        codec: ModelScenePlanCodecV4,
        max_plan_tokens: int = 1024,
        source_semantic_cache: ScenePlanTransfusionEditingM2DCLAPCache | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(dataset, ScenePlanTransfusionEditingDataset):
            raise TypeError("joint Editing requires the frozen Editing-DiT dataset")
        if int(max_plan_tokens) <= 1:
            raise ValueError("Editing AR max_plan_tokens must exceed one")
        self.dataset = dataset
        self.codec = codec
        self.max_plan_tokens = int(max_plan_tokens)
        self.source_semantic_cache = source_semantic_cache
        self.semantic_caption_requires_epoch_key = False

    def __len__(self) -> int:
        return len(self.dataset)

    def length_bucket_indices(self) -> dict[int, tuple[int, ...]]:
        return self.dataset.length_bucket_indices()

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        target, metadata = self.dataset[index]
        if not isinstance(metadata, dict):
            raise TypeError("Editing-DiT metadata row must be a dictionary")
        _reject_old_sceneplan_keys(metadata, where="metadata")
        required = {
            "pair_id",
            "pair_ordinal",
            "raw_edit_request",
            "editing_ar_target_model_sceneplan",
            "model_sceneplan",
            "source_foa_latent",
            "padding_mask",
            "latent_frames_valid",
            "latent_bucket_frames",
            "latent_crop_length",
            "model_num_samples",
            "edited_source_ids",
            "unchanged_source_ids",
        }
        missing = sorted(required - set(metadata))
        if missing:
            raise RuntimeError(f"joint Editing metadata is missing {missing}")
        new_plan = metadata["editing_ar_target_model_sceneplan"]
        if new_plan != metadata["model_sceneplan"]:
            raise RuntimeError("AR target and DiT new ScenePlan controls diverged")
        validate_model_sceneplan(new_plan)
        # The base dataset retains the frozen renderer's arbitrary slots.
        # AR labels and RF controls must use the same observable model view.
        model_plan, id_map = canonicalize_editing_plan(new_plan, codec=self.codec)
        metadata = dict(metadata)
        metadata["editing_persistent_target_sceneplan"] = new_plan
        metadata["editing_persistent_edited_source_ids"] = tuple(metadata["edited_source_ids"])
        metadata["editing_persistent_unchanged_source_ids"] = tuple(metadata["unchanged_source_ids"])
        metadata["editing_model_source_id_map"] = id_map
        metadata["editing_ar_plan_policy"] = EDITING_AR_PLAN_POLICY
        # A removed source has no target-model slot; retain its id only in
        # offline provenance, never alias it to a surviving canonical source.
        metadata["edited_source_ids"] = tuple(
            id_map[value] for value in metadata["edited_source_ids"] if value in id_map
        )
        metadata["unchanged_source_ids"] = tuple(
            id_map[value] for value in metadata["unchanged_source_ids"]
        )
        metadata.update(compile_editing_dit_plan_condition(
            model_plan,
            tokenizer=self.dataset.tokenizer,
            model_num_samples=int(metadata["model_num_samples"]),
            latent_frames_valid=int(metadata["latent_frames_valid"]),
            latent_crop_length=int(metadata["latent_crop_length"]),
            caption_max_tokens=self.dataset.caption_max_tokens,
        ))
        metadata["editing_ar_target_model_sceneplan"] = model_plan
        new_plan = model_plan
        encoded = self.codec.encode(new_plan, max_tokens=self.max_plan_tokens)
        source = metadata["source_foa_latent"]
        padding = metadata["padding_mask"]
        if (
            not isinstance(source, torch.Tensor)
            or source.ndim != 2
            or int(source.shape[0]) != 64
            or not isinstance(padding, list)
            or len(padding) != 1
            or not isinstance(padding[0], torch.Tensor)
        ):
            raise RuntimeError("joint Editing clean source/mask contract changed")
        ar_row = {
            "pair_ordinal": int(metadata["pair_ordinal"]),
            "pair_id": str(metadata["pair_id"]),
            "operation": str(metadata["operation"]),
            "raw_edit_request": str(metadata["raw_edit_request"]),
            "source_foa_latent": source,
            "source_attention_mask": padding[0],
            "target_token_ids": encoded["input_ids"],
            "target_loss_group_ids": encoded["loss_group_ids"],
            "source_valid_frames": int(metadata["latent_frames_valid"]),
            # A fixed bucket envelope makes plan-token positions independent
            # of batch composition and matches Editing-DiT/inference geometry.
            "source_prefix_frames": int(metadata["latent_bucket_frames"]),
        }
        if self.source_semantic_cache is not None:
            ar_row.update(
                self.source_semantic_cache.get(
                    int(metadata["pair_ordinal"]),
                    pair_id=str(metadata["pair_id"]),
                    source_sample_id=str(metadata["source_sample_id"]),
                    source_latent_tensor_sha256=str(
                        metadata["source_foa_latent_tensor_sha256"]
                    ),
                )
            )
        _reject_old_sceneplan_keys(ar_row, where="row")
        return target, metadata, ar_row


def collate_editing_joint(
    samples: Sequence[tuple[torch.Tensor, dict[str, Any], dict[str, Any]]],
    *,
    pad_id: int,
) -> dict[str, Any]:
    """Create aligned AR and RF views without loading a pair twice."""

    if not samples:
        raise ValueError("cannot collate an empty joint Editing batch")
    diffusion_samples = [(target, metadata) for target, metadata, _ in samples]
    target_batch, metadata_batch = sceneplan_bucket_collation(diffusion_samples)
    ar_rows = []
    for metadata, (_, _, original_ar) in zip(metadata_batch, samples):
        _reject_old_sceneplan_keys(metadata, where="collated metadata")
        ar_row = dict(original_ar)
        # Use the exact source tensors after homogeneous 432/648 bucket trim.
        ar_row["source_foa_latent"] = metadata["source_foa_latent"]
        ar_row["source_attention_mask"] = metadata["padding_mask"][0]
        _reject_old_sceneplan_keys(ar_row, where="collated row")
        ar_rows.append(ar_row)
    ar_batch = collate_editing_ar(ar_rows, pad_id=int(pad_id))
    # Preserve the homogeneous 432/648 bucket envelope. The training module
    # injects this exact tensor into both AR and DiT.
    ar_batch["source_foa_latent"] = torch.stack(
        [row["source_foa_latent"] for row in metadata_batch]
    )
    ar_batch["source_attention_mask"] = torch.stack(
        [row["padding_mask"][0] for row in metadata_batch]
    )
    semantic_presence = [
        "source_m2d_audio_embedding" in row for row in ar_rows
    ]
    if any(semantic_presence) and not all(semantic_presence):
        raise RuntimeError("joint Editing batch mixed cached and uncached M2D rows")
    if all(semantic_presence):
        required_semantic = {
            "source_m2d_audio_embedding",
            "source_caption_m2d_embedding",
            "source_caption_sha256",
            "source_caption_group_ids",
            "source_semantic_group_ids",
        }
        if any(not required_semantic.issubset(row) for row in ar_rows):
            raise RuntimeError("joint Editing M2D row is incomplete")
        ar_batch.update(
            {
                "source_m2d_audio_embedding": torch.stack(
                    [row["source_m2d_audio_embedding"] for row in ar_rows]
                ),
                "source_caption_m2d_embedding": torch.stack(
                    [row["source_caption_m2d_embedding"] for row in ar_rows]
                ),
                "source_caption_sha256s": [
                    str(row["source_caption_sha256"]) for row in ar_rows
                ],
                "source_caption_group_ids": torch.stack(
                    [row["source_caption_group_ids"] for row in ar_rows]
                ),
                "source_semantic_group_ids": torch.stack(
                    [row["source_semantic_group_ids"] for row in ar_rows]
                ),
            }
        )
    if ar_batch["pair_ids"] != [str(row["pair_id"]) for row in metadata_batch]:
        raise RuntimeError("joint Editing AR/RF pair order diverged")
    return {
        "target_foa_latent": target_batch,
        "metadata": metadata_batch,
        "ar": ar_batch,
    }


__all__ = [
    "EDITING_JOINT_DATASET_CONTRACT",
    "ScenePlanTransfusionEditingJointDataset",
    "collate_editing_joint",
]
