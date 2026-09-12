"""Active sketch-first runtime view for audio-aware P11.

Generation and Understanding supervise one SceneSketch plus Flow-R1
ExecutionState.  Editing first supervises the same audio-derived observation,
then a semantics-only DeltaSketch plus deterministic DeltaExecutionState.
The optional old ScenePlan remains context evidence and is never the patch
application authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .scene_sketch_v1 import (
    AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT,
    EXECUTION_STATE_CONTRACT,
    SCENE_SKETCH_CONTRACT,
    AudioAwareDeltaSceneSketchCodec,
    SceneSketchCodec,
    compile_audio_aware_delta_sketch,
    compile_execution_state,
    compile_scene_sketch,
    execution_delta_core,
    execution_state_core,
)
from .sceneplan_p11_single_turn import P11Task, normalize_p11_task
from .sceneplan_p11_lexical_cache import (
    P11_LEXICAL_AUTHORITY_CONTRACT,
    P11_LEXICAL_EVIDENCE_CONTRACT,
    P11LexicalEvidenceCache,
)
from .sceneplan_p11_dataset import (
    ScenePlanP11Dataset,
    tokenize_p11_task_prompt,
)


P11_AUDIO_AWARE_DATA_CONTRACT = "audio_aware_sketch_first_transfusion_cot_v2"
P11_AUDIO_AWARE_SEQUENCE_CONTRACT = (
    "audio_observation_then_atomic_delta_then_deterministic_revised_v1"
)
# Frozen names retained only for historical import/checkpoint metadata.
P11_V4_DATA_CONTRACT = "sketch_first_transfusion_cot_v4"
P11_V4_SEQUENCE_CONTRACT = (
    "task_evidence_then_discrete_sketch_then_continuous_thought_then_assembler_v1"
)
P11_V4_CONTROL_DIRECTION_CONTRACT = "p10_atomic_edit_control_direction_v1"


def compile_control_direction_target(
    edit_spec: dict[str, Any] | None,
) -> tuple[float, bool]:
    """Compatibility stub: active absolute move edits have no binary branch."""

    return 0.0, False


class ScenePlanP11AudioAwareDataset(ScenePlanP11Dataset):
    """Add observed/delta/revised supervision to manifest-v8 rows."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        lexical_evidence_mode: str = "none",
        lexical_max_tokens: int = 128,
        lexical_cache_path: str | Path | None = None,
        lexical_encoder_revision: str | None = None,
        lexical_confidence_threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(manifest_path, **kwargs)
        self.scene_sketch_codec = SceneSketchCodec(self.codec)
        self.delta_sketch_codec = AudioAwareDeltaSceneSketchCodec(
            self.codec, self.patch_codec
        )
        self.lexical_evidence_mode = str(lexical_evidence_mode)
        if self.lexical_evidence_mode not in {
            "none",
            "frozen_asr_cache_v1",
            "oracle_transcript_wiring_v1",
        }:
            raise ValueError(
                "P11-v4 lexical_evidence_mode must be none, "
                "frozen_asr_cache_v1, or oracle_transcript_wiring_v1"
            )
        self.lexical_max_tokens = int(lexical_max_tokens)
        if not 8 <= self.lexical_max_tokens <= 256:
            raise ValueError("P11-v4 lexical token ceiling must be within [8,256]")
        if self.lexical_evidence_mode == "frozen_asr_cache_v1":
            if (
                lexical_cache_path is None
                or not str(lexical_encoder_revision or "")
                or lexical_confidence_threshold is None
            ):
                raise ValueError(
                    "P11-v4 frozen lexical evidence requires cache path, ASR "
                    "revision, and a train-calibrated confidence threshold"
                )
            self.lexical_confidence_threshold = float(
                lexical_confidence_threshold
            )
            if not 0.0 <= self.lexical_confidence_threshold <= 1.0:
                raise ValueError(
                    "P11-v4 lexical confidence threshold must be within [0,1]"
                )
            connection = self._open_connection(self.manifest_path)
            try:
                expected_ordinals = [
                    int(row[0])
                    for row in connection.execute(
                        "SELECT DISTINCT source_ordinal FROM rows ORDER BY source_ordinal"
                    )
                ]
            finally:
                connection.close()
            self.lexical_cache = P11LexicalEvidenceCache(
                lexical_cache_path,
                source_manifest=self.manifest_path,
                source_index=self.index_path,
                encoder_revision=str(lexical_encoder_revision),
                expected_ordinals=expected_ordinals,
            )
        else:
            if (
                lexical_cache_path is not None
                or lexical_encoder_revision is not None
                or lexical_confidence_threshold is not None
            ):
                raise ValueError(
                    "P11-v4 lexical cache/threshold fields are legal only in "
                    "frozen_asr_cache_v1"
                )
            self.lexical_cache = None
            self.lexical_confidence_threshold = None

    def _lexical_evidence(
        self,
        task: P11Task,
        observed_plan: dict[str, Any],
        source_ordinal: int,
    ) -> dict[str, Any] | None:
        if (
            task not in {P11Task.UNDERSTANDING, P11Task.EDITING}
            or self.lexical_evidence_mode == "none"
        ):
            return None
        authority: dict[str, Any] | None = None
        if self.lexical_evidence_mode == "frozen_asr_cache_v1":
            if self.lexical_cache is None:
                raise RuntimeError("P11-v4 frozen lexical cache was not initialized")
            hypothesis = self.lexical_cache.row(source_ordinal)
            reliable = bool(hypothesis["has_speech"]) and float(
                hypothesis["confidence"]
            ) >= float(self.lexical_confidence_threshold)
            if not reliable:
                # CLAP and FOA remain present for every U row.  An unreliable
                # ASR hypothesis contributes no token at all; even a textual
                # "no speech" marker would let ASR influence non-speech rows.
                return None
            text = (
                "Frozen ASR hypothesis "
                f"(language={hypothesis['language']}, "
                f"confidence={hypothesis['confidence']:.3f}): "
                + hypothesis["text"]
            )
            authority = {
                "contract": P11_LEXICAL_AUTHORITY_CONTRACT,
                "transcript": str(hypothesis["text"]),
                "confidence": float(hypothesis["confidence"]),
                "language": str(hypothesis["language"]),
                "target_transcript_access": False,
            }
        else:
            # Wiring-only diagnostic.  This mode is explicitly non-canonical
            # because it reads the target transcript and therefore leaks U.
            clauses = [
                f"{source['source_id']}: {source['transcript']}"
                for source in observed_plan["sources"]
                if source["kind"] == "speech"
            ]
            text = (
                "Oracle transcript wiring: " + " ; ".join(clauses)
                if clauses
                else "Oracle transcript wiring: no speech target."
            )
        tokens = dict(tokenize_p11_task_prompt(
            text, self.tokenizer, max_length=self.lexical_max_tokens
        ))
        if authority is not None:
            tokens["lexical_authority"] = authority
        return tokens

    def __del__(self) -> None:
        cache = getattr(self, "lexical_cache", None)
        if cache is not None:
            cache.close()
        super().__del__()

    def __getitem__(self, index: int):
        carrier, metadata = super().__getitem__(index)
        task = normalize_p11_task(metadata["p11_task"])
        revised_plan = metadata["p11_target_sceneplan"]
        observed_plan = (
            revised_plan
            if task is P11Task.GENERATION
            else metadata["p11_observed_sceneplan_target"]
        )
        if observed_plan is None:
            raise RuntimeError(f"P11 {task.value} lacks its observed ScenePlan target")
        observed_sketch = compile_scene_sketch(observed_plan, self.codec)
        observed_execution = compile_execution_state(observed_plan, self.codec)
        observed_sketch_tokens = self.scene_sketch_codec.encode(observed_sketch)
        observed_plan_tokens = self.codec.encode(observed_plan)
        revised_sketch = compile_scene_sketch(revised_plan, self.codec)
        revised_execution = compile_execution_state(revised_plan, self.codec)
        revised_sketch_tokens = self.scene_sketch_codec.encode(revised_sketch)
        revised_plan_tokens = self.codec.encode(revised_plan)
        prior_plan = metadata["p11_prior_sceneplan"]
        prior_sketch = (
            None if prior_plan is None else compile_scene_sketch(prior_plan, self.codec)
        )
        prior_execution = (
            None
            if prior_plan is None
            else compile_execution_state(prior_plan, self.codec)
        )

        delta_sketch = None
        delta_sketch_tokens = None
        atomic_patch_tokens = None
        if task is P11Task.EDITING:
            edit_spec = metadata["p11_edit_spec"]
            if edit_spec is None:
                raise RuntimeError("audio-aware P11 Editing lacks its edit spec")
            atomic_patch_tokens = metadata["p11_target_tokens"]
            delta_sketch = compile_audio_aware_delta_sketch(
                observed_plan,
                revised_plan,
                edit_spec,
                self.codec,
                self.patch_codec,
            )
            delta_sketch_tokens = self.delta_sketch_codec.encode(delta_sketch)
            discrete_target = delta_sketch_tokens
            output_kind = "delta_sketch"
            delta_core = torch.from_numpy(
                execution_delta_core(observed_execution, revised_execution)
            )
            input_core = torch.from_numpy(execution_state_core(observed_execution))
        else:
            discrete_target = observed_sketch_tokens
            output_kind = "scene_sketch"
            delta_core = torch.zeros_like(
                torch.from_numpy(execution_state_core(observed_execution))
            )
            input_core = torch.zeros_like(delta_core)

        observed_core = torch.from_numpy(execution_state_core(observed_execution))
        revised_core = torch.from_numpy(execution_state_core(revised_execution))
        observed_mask = torch.tensor(
            observed_execution["source_present_mask"], dtype=torch.bool
        )
        revised_mask = torch.tensor(
            revised_execution["source_present_mask"],
            dtype=torch.bool,
        )
        metadata.update(
            {
                "p11_observation_prompt": (
                    tokenize_p11_task_prompt(
                        "Observe the input FOA and recover its complete current ScenePlan.",
                        self.tokenizer,
                    )
                    if task is P11Task.EDITING
                    else metadata["p11_prompt"]
                ),
                "p11_target_tokens": discrete_target,
                "p11_output_kind": output_kind,
                "p11_data_contract": P11_AUDIO_AWARE_DATA_CONTRACT,
                "p11_audio_aware_data_contract": P11_AUDIO_AWARE_DATA_CONTRACT,
                "p11_audio_aware_sequence_contract": P11_AUDIO_AWARE_SEQUENCE_CONTRACT,
                "p11_observed_scene_sketch": observed_sketch,
                "p11_observed_scene_sketch_tokens": observed_sketch_tokens,
                "p11_observed_execution_state": observed_execution,
                "p11_observed_execution_core": observed_core,
                "p11_observed_source_mask": observed_mask,
                "p11_observed_sceneplan_tokens": observed_plan_tokens,
                "p11_delta_scene_sketch": delta_sketch,
                "p11_delta_scene_sketch_tokens": delta_sketch_tokens,
                "p11_delta_execution_core": delta_core,
                "p11_revised_scene_sketch": revised_sketch,
                "p11_revised_scene_sketch_tokens": revised_sketch_tokens,
                "p11_revised_execution_state": revised_execution,
                "p11_revised_execution_core": revised_core,
                "p11_revised_source_mask": revised_mask,
                "p11_revised_sceneplan_tokens": revised_plan_tokens,
                "p11_prior_scene_sketch": prior_sketch,
                "p11_prior_execution_state": prior_execution,
                "p11_atomic_patch_tokens": atomic_patch_tokens,
                "p11_delta_token_contract": (
                    AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT
                    if task is P11Task.EDITING
                    else None
                ),
                # Transitional aliases for the shared training code while it
                # is migrated in place.  Importantly, E's input state here is
                # the audio-derived observed target, never the optional prior.
                "p11_v4_data_contract": P11_AUDIO_AWARE_DATA_CONTRACT,
                "p11_v4_sequence_contract": P11_AUDIO_AWARE_SEQUENCE_CONTRACT,
                "p11_v4_scene_sketch_contract": SCENE_SKETCH_CONTRACT,
                "p11_v4_execution_state_contract": EXECUTION_STATE_CONTRACT,
                "p11_v4_delta_scene_sketch_contract": (
                    AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT
                    if task is P11Task.EDITING
                    else None
                ),
                "p11_v4_delta_token_contract": (
                    AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT
                    if task is P11Task.EDITING
                    else None
                ),
                "p11_v4_target_scene_sketch": revised_sketch,
                "p11_v4_target_scene_sketch_tokens": revised_sketch_tokens,
                "p11_v4_target_execution_state": revised_execution,
                "p11_v4_target_execution_core": revised_core,
                "p11_v4_target_source_mask": revised_mask,
                "p11_v4_input_scene_sketch": (
                    observed_sketch if task is P11Task.EDITING else None
                ),
                "p11_v4_input_execution_state": (
                    observed_execution if task is P11Task.EDITING else None
                ),
                "p11_v4_input_execution_core": input_core,
                "p11_v4_input_source_mask": (
                    observed_mask
                    if task is P11Task.EDITING
                    else torch.zeros(4, dtype=torch.bool)
                ),
                "p11_v4_delta_scene_sketch": delta_sketch,
                "p11_v4_delta_scene_sketch_tokens": delta_sketch_tokens,
                "p11_v4_delta_execution_core": delta_core,
                "p11_v4_control_direction_contract": None,
                "p11_v4_control_direction": torch.tensor(0.0),
                "p11_v4_control_direction_operation": None,
                "p11_v4_control_direction_mask": torch.tensor(False),
                "p11_v4_final_sceneplan_tokens": revised_plan_tokens,
                "p11_v4_atomic_patch_tokens": atomic_patch_tokens,
                "p11_input_lexical": self._lexical_evidence(
                    task, observed_plan, int(metadata["p11_source_ordinal"])
                ),
                "p11_lexical_evidence_mode": self.lexical_evidence_mode,
                "p11_lexical_evidence_contract": (
                    P11_LEXICAL_EVIDENCE_CONTRACT
                    if self.lexical_evidence_mode == "frozen_asr_cache_v1"
                    else None
                ),
                # Explicitly retire the old free-form core40 representation.
                "p11_scene_thought_contract": None,
                "p11_target_scene_thought_core": None,
                "p11_input_scene_thought_core": None,
                "p11_scene_thought_delta_core": None,
                "p11_scene_thought_round_slot_masks": None,
                "p11_scene_thought_round_feature_masks": None,
                "p11_scene_thought_changed_slot_mask": None,
                "p11_scene_thought_changed_feature_mask": None,
            }
        )
        return carrier, metadata


# Keep the old import spelling as a temporary source-compatible alias; its
# data contract is active audio-aware v1, so a frozen v4 config cannot launch.
ScenePlanP11V4Dataset = ScenePlanP11AudioAwareDataset


__all__ = [
    "P11_AUDIO_AWARE_DATA_CONTRACT",
    "P11_AUDIO_AWARE_SEQUENCE_CONTRACT",
    "P11_V4_DATA_CONTRACT",
    "P11_V4_CONTROL_DIRECTION_CONTRACT",
    "P11_V4_SEQUENCE_CONTRACT",
    "ScenePlanP11AudioAwareDataset",
    "ScenePlanP11V4Dataset",
    "compile_control_direction_target",
]
