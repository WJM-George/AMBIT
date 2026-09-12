"""Runtime metadata for unified Spatial-CoT generation/understanding/editing.

The heavy latent cache keeps one target ``.npy`` per conversation turn.  Full
ScenePlans and diffs live in an indexed JSONL store, avoiding millions of large
duplicated sidecar JSON files.  This provider resolves one turn, compiles its
compact plan tokens and 32-D source tracks, and loads the previous-turn FOA
latent as an aligned context modality.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from stable_audio_tools.data.scene_plan import crop_scene_plan
from stable_audio_tools.data.spatial_plan_codec import SpatialPlanCodec
from stable_audio_tools.data.spatial_story_metadata import ScenePlanTrackMetadata
from stable_audio_tools.data.t2a_artifacts import IndexedJsonlStore


def _nested(value: Mapping[str, Any], dotted_key: str, default=None):
    current: Any = value
    for component in dotted_key.split("."):
        if not isinstance(current, Mapping) or component not in current:
            return default
        current = current[component]
    return current


class SpatialConversationMetadata:
    """Compile one persistent-state training turn into model metadata."""

    def __init__(self, config: dict[str, Any]):
        store_dir = config.get("turn_store_dir")
        self.store: Optional[IndexedJsonlStore] = (
            IndexedJsonlStore(
                store_dir,
                require_ready=bool(config.get("require_ready", True)),
                max_open_shards=int(config.get("max_open_shards", 8)),
            )
            if store_dir
            else None
        )
        codec_path = config.get("codec_path")
        if not codec_path:
            raise ValueError("spatial_conversation requires codec_path")
        self.codec_path = str(codec_path)
        self._codec: Optional[SpatialPlanCodec] = None
        self.max_tokens = int(config.get("max_tokens", 1024))
        self.record_key = str(config.get("embedded_record_key", "conversation_turn"))
        self.lookup_key = str(config.get("lookup_key", "path"))
        self.target_plan_path = str(config.get("target_plan_path", "after.scene_plan"))
        self.previous_plan_path = str(
            config.get("previous_plan_path", "before.scene_plan")
        )
        self.previous_latent_path = str(
            config.get("previous_latent_path", "before.latent_path")
        )
        self.plan_output_key = str(
            config.get("plan_output_key", "spatial_plan_tokens")
        )
        self.previous_plan_output_key = str(
            config.get("previous_plan_output_key", "previous_plan_tokens")
        )
        self.previous_foa_key = str(config.get("previous_foa_key", "previous_foa"))
        self.presence_key = str(
            config.get("context_presence_key", "previous_foa_present")
        )
        # Creation recipes historically named only the room type/description
        # while supervising exact RT60 and dimensions.  That makes the Planner
        # target under-specified and encourages an arbitrary room-mode guess.
        # Opt in explicitly so older routes retain byte-for-byte prompts.
        self.include_planner_room_metrics = bool(
            config.get("include_planner_room_metrics", False)
        )
        self.track_provider = ScenePlanTrackMetadata(
            {
                "plan_key": "scene_plan",
                "output_key": str(config.get("tracks_output_key", "source_tracks")),
                "layout": str(config.get("tracks_layout", "packed_source_features")),
                "max_sources": int(config.get("max_sources", 4)),
                "max_distance_m": float(config.get("max_distance_m", 20.0)),
            }
        )

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_codec"] = None
        return state

    def _get_codec(self) -> SpatialPlanCodec:
        if self._codec is None:
            self._codec = SpatialPlanCodec(self.codec_path)
        return self._codec

    def _record(self, info: dict[str, Any]) -> dict[str, Any]:
        embedded = info.get(self.record_key)
        if isinstance(embedded, Mapping):
            return dict(embedded)
        if self.store is None:
            raise KeyError(
                f"metadata has no embedded {self.record_key!r} and no turn_store_dir"
            )
        lookup = info.get(self.lookup_key)
        if not isinstance(lookup, str) or not lookup:
            raise KeyError(
                f"spatial_conversation lookup key {self.lookup_key!r} is absent"
            )
        return self.store.get(lookup)

    @staticmethod
    def _aligned_previous_latent(
        path: Optional[str],
        target: torch.Tensor,
        *,
        crop_start: int,
    ) -> tuple[torch.Tensor, bool]:
        if not path:
            return torch.zeros_like(target, dtype=torch.float32), False
        source_path = Path(path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"previous-turn latent is missing: {source_path}")
        array = np.load(source_path, allow_pickle=False)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.floating):
            raise ValueError(
                f"previous-turn latent must be floating [C,T], got {array.shape} {array.dtype}"
            )
        if array.shape[0] != target.shape[0] or not np.isfinite(array).all():
            raise ValueError(
                f"previous/target latent mismatch: {array.shape} vs {tuple(target.shape)}"
            )
        desired = int(target.shape[-1])
        start = max(0, int(crop_start))
        value = torch.from_numpy(array[:, start : start + desired]).float()
        if value.shape[-1] < desired:
            value = torch.nn.functional.pad(value, (0, desired - value.shape[-1]))
        return value, True

    @staticmethod
    def _required_text(record: Mapping[str, Any], key: str, fallback: str) -> str:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        return fallback

    @staticmethod
    def _planner_prompt_with_room_metrics(
        prompt: str, canonical_plan: Mapping[str, Any]
    ) -> str:
        room = _nested(canonical_plan, "scene.room")
        if not isinstance(room, Mapping):
            return prompt
        details = []
        rt60 = room.get("rt60_s")
        if isinstance(rt60, (int, float)):
            details.append(f"RT60={float(rt60):.3f}s")
        dimensions = room.get("dimensions_m")
        if (
            isinstance(dimensions, (list, tuple))
            and len(dimensions) == 3
            and all(isinstance(value, (int, float)) for value in dimensions)
        ):
            details.append(
                "dimensions="
                + "x".join(f"{float(value):.1f}" for value in dimensions)
                + "m"
            )
        if not details:
            return prompt
        contract = "; ".join(details)
        # Creation prompts start with ``Room: description.``. Insert metrics in
        # that first sentence so Qwen's 128-token truncation cannot drop them.
        if prompt.startswith("Room: ") and "." in prompt:
            room_clause, separator, remainder = prompt.partition(".")
            return f"{room_clause}; {contract}{separator}{remainder}"
        return f"Room metrics: {contract}. {prompt}"

    def _compile_record(
        self,
        info: dict[str, Any],
        latents: torch.Tensor,
        record: Mapping[str, Any],
        *,
        previous_override: Optional[tuple[torch.Tensor, bool]] = None,
    ) -> dict[str, Any]:
        target_plan = _nested(record, self.target_plan_path)
        if not isinstance(target_plan, Mapping):
            raise KeyError(
                f"conversation turn lacks target plan at {self.target_plan_path!r}"
            )
        previous_plan = _nested(record, self.previous_plan_path)
        timestamps = info.get("timestamps")
        target_plan = crop_scene_plan(dict(target_plan), timestamps)
        if isinstance(previous_plan, Mapping):
            previous_plan = crop_scene_plan(dict(previous_plan), timestamps)
        else:
            previous_plan = None

        codec = self._get_codec()
        target_tokens = codec.encode(target_plan, max_tokens=self.max_tokens)
        # Renderer controls must be compiled from the same quantized state the
        # AR planner can emit.  Compiling tracks from unquantized JSON while
        # sampling from decoded tokens creates a subtle train/inference mismatch
        # for coordinates, activity times, distance, and gain.
        canonical_target_plan = codec.decode(target_tokens["input_ids"])
        previous_tokens = (
            codec.encode(previous_plan, max_tokens=self.max_tokens)
            if previous_plan is not None
            else None
        )
        target_tokens["edit_token_mask"] = (
            codec.edit_token_mask(previous_tokens, target_tokens)
            if previous_tokens is not None
            else torch.zeros_like(target_tokens["input_ids"], dtype=torch.bool)
        )
        canonical_previous_plan = (
            codec.decode(previous_tokens["input_ids"])
            if previous_tokens is not None
            else None
        )
        if previous_override is None:
            previous_path = _nested(record, self.previous_latent_path)
            previous_foa, context_present = self._aligned_previous_latent(
                previous_path if isinstance(previous_path, str) else None,
                latents,
                crop_start=int(info.get("latent_crop_start", 0)),
            )
        else:
            previous_foa, context_present = previous_override
            previous_foa = previous_foa.to(dtype=torch.float32)
            if tuple(previous_foa.shape) != tuple(latents.shape):
                raise ValueError(
                    "family previous/target latent shapes differ: "
                    f"{tuple(previous_foa.shape)} != {tuple(latents.shape)}"
                )

        task = str(record.get("task") or "text_to_spatial_audio")
        instruction = str(record.get("instruction") or record.get("prompt") or "")
        semantic_caption = self._required_text(
            record,
            "semantic_caption",
            instruction or "spatial audio scene",
        )
        semantic_caption_metadata = record.get("semantic_caption_metadata")
        planner_prompt = self._required_text(
            record,
            "planner_prompt",
            instruction or semantic_caption,
        )
        if self.include_planner_room_metrics and task == "text_to_spatial_audio":
            planner_prompt = self._planner_prompt_with_room_metrics(
                planner_prompt, canonical_target_plan
            )
        understanding_prompt = self._required_text(
            record,
            "understanding_prompt",
            "Describe every audible source, its activity, and its metric 3-D trajectory.",
        )
        output: dict[str, Any] = {
            "scene_plan": canonical_target_plan,
            "previous_scene_plan": canonical_previous_plan,
            self.plan_output_key: target_tokens,
            self.previous_plan_output_key: previous_tokens,
            self.previous_foa_key: previous_foa,
            f"{self.previous_foa_key}_aligned": True,
            self.presence_key: context_present,
            "planner_prompt": planner_prompt,
            "semantic_caption": semantic_caption,
            "semantic_caption_metadata": semantic_caption_metadata,
            "understanding_prompt": understanding_prompt,
            "prompt": planner_prompt,
            "task": task,
            "conversation_id": record.get("conversation_id"),
            "turn_id": record.get("turn_id"),
            "turn_index": record.get("turn_index"),
            "edit": record.get("edit"),
            "diff": record.get("diff"),
        }
        track_info = dict(info)
        track_info.update(output)
        output.update(self.track_provider(track_info, latents))
        return output

    def __call__(self, info: dict[str, Any], latents: torch.Tensor) -> dict[str, Any]:
        if latents.ndim != 2:
            raise ValueError(f"expected target latent [C,T], got {tuple(latents.shape)}")
        return self._compile_record(info, latents, self._record(info))


class SpatialFamilyMetadata(SpatialConversationMetadata):
    """Compile all four states loaded from one sharded family tensor.

    Keeping the full family in one safetensors entry avoids four million tiny
    ``.npy/.json`` pairs and avoids duplicating previous latents.  The training
    wrapper expands the returned turn metadata into ordinary per-turn examples
    after the DataLoader has performed one efficient family read.
    """

    def __init__(self, config: dict[str, Any]):
        config = dict(config)
        config.pop("turn_store_dir", None)
        super().__init__(config)
        self.family_record_key = str(
            config.get("family_record_key", "spatial_family")
        )
        self.turns_key = str(config.get("turns_key", "turns"))
        self.independent_turns = bool(config.get("independent_turns", False))

    def __call__(self, info: dict[str, Any], latents: torch.Tensor) -> dict[str, Any]:
        if latents.ndim != 3:
            raise ValueError(
                f"expected family latent [turn,C,T], got {tuple(latents.shape)}"
            )
        family = info.get(self.family_record_key)
        if not isinstance(family, Mapping):
            raise KeyError(
                f"family metadata lacks {self.family_record_key!r} record"
            )
        turns = family.get(self.turns_key)
        if not isinstance(turns, list) or len(turns) != latents.shape[0]:
            raise ValueError(
                "family turn records do not match latent states: "
                f"{len(turns) if isinstance(turns, list) else None} != "
                f"{latents.shape[0]}"
            )
        compiled_turns = []
        family_id = str(family.get("family_id") or info.get("family_id") or "family")
        curriculum_kind = family.get("curriculum_kind")
        for index, (record, target) in enumerate(zip(turns, latents.unbind(0))):
            if not isinstance(record, Mapping):
                raise TypeError(f"family turn {index} must be an object")
            turn_info = {
                "path": f"{family_id}/turn_{index:03d}",
                "family_id": family_id,
                "family_turn_index": index,
                "padding_mask": [
                    torch.ones(target.shape[-1], dtype=torch.bool)
                ],
                "latent_crop_start": 0,
                "timestamps": [0.0, 1.0],
                "seconds_start": 0.0,
                "seconds_total": float(
                    _nested(record, "after.scene_plan.audio.duration_sec", 0.0)
                    or 0.0
                ),
            }
            if curriculum_kind is not None:
                turn_info["curriculum_kind"] = str(curriculum_kind)
            turn_curriculum = record.get("curriculum")
            if isinstance(turn_curriculum, Mapping):
                # Keep pair structure machine-readable after the family loader
                # expands four stored states into ordinary training examples.
                # The complete object remains in the immutable family record;
                # only stable scalar routing fields are copied into the hot
                # metadata path used by auxiliary objectives.
                for source_key, output_key in (
                    ("kind", "curriculum_turn_kind"),
                    ("role", "curriculum_role"),
                    ("pair_id", "curriculum_pair_id"),
                    ("pair_index", "curriculum_pair_index"),
                    ("source_family_id", "curriculum_source_family_id"),
                    ("source_family_rank", "curriculum_source_family_rank"),
                    ("source_turn_index", "curriculum_source_turn_index"),
                ):
                    value = turn_curriculum.get(source_key)
                    if value is not None:
                        turn_info[output_key] = value
            previous = (
                (torch.zeros_like(target), False)
                if self.independent_turns or index == 0
                else (latents[index - 1], True)
            )
            compiled = self._compile_record(
                turn_info,
                target,
                record,
                previous_override=previous,
            )
            turn_info.update(compiled)
            compiled_turns.append(turn_info)
        return {
            "family_turn_metadata": compiled_turns,
            "family_id": family_id,
            "split": family.get("split"),
        }


def create_custom_metadata(config: dict[str, Any]):
    return SpatialConversationMetadata(config)


__all__ = [
    "SpatialConversationMetadata",
    "SpatialFamilyMetadata",
    "create_custom_metadata",
]


def evaluation_metadata_provider(codec_root: Path) -> SpatialFamilyMetadata:
    return SpatialFamilyMetadata(
        {
            "codec_path": str(codec_root),
            "family_record_key": "spatial_family",
            "turns_key": "turns",
            "target_plan_path": "after.scene_plan",
            "previous_plan_path": "before.scene_plan",
            "plan_output_key": "spatial_plan_tokens",
            "previous_plan_output_key": "previous_plan_tokens",
            "previous_foa_key": "previous_foa",
            "context_presence_key": "previous_foa_present",
            "include_planner_room_metrics": True,
            "tracks_output_key": "source_tracks",
            "tracks_layout": "packed_source_features",
            "max_sources": 4,
            "max_distance_m": 20.0,
            "max_tokens": 1024,
        }
    )

