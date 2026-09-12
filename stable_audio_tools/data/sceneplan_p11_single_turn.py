"""P10-aligned contract for the canonical P11 planner/executor route.

P11 owns language/audio understanding and ScenePlan prediction.  It does not
own an FOA renderer.  Generation and editing finish by handing one validated,
fully numeric ScenePlan to the external P10 ScenePlan-DiT executor.  P10 then
compiles the semantic caption and frame-aligned 4+4 controls and synthesizes a
new FOA waveform.

The three P11 sequences are therefore deliberately auditable:

``generation``
    rough user request -> ScenePlan tokens
``understanding``
    clean input FOA span -> ScenePlan tokens
``editing``
    input FOA + edit instruction + optional current ScenePlan
    -> observed ScenePlan -> atomic edit patch -> revised ScenePlan

The edit patch is applied by a deterministic executor before the resulting
complete ScenePlan crosses the P10 boundary.  P10 still resynthesizes the full
scene, so this state-preserving protocol does not claim waveform preservation.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor

from .model_sceneplan import (
    MAX_SOURCES,
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
    compile_model_44_controls,
    compile_model_renderer_caption,
    compile_model_semantic_caption_v2,
    tokenize_model_semantic_caption,
    validate_model_sceneplan,
)
from .model_sceneplan_codec import ModelScenePlanCodec


P11_CONTRACT_VERSION = 7
P10_CONDITIONING_CONTRACT_REVISION = 2
P10_SCENEPLAN_CONTRACT_REVISION = 5
P10_DATASET_CONTRACT_REVISION = 6
P10_SEMANTIC_CAPTION_CONTRACT = "p10_semantic_caption_v2_only_20260830"
P10_SEMANTIC_CAPTION_COMPILER_VERSION = 2
P10_SEMANTIC_CAPTION_SURFACE = (
    "<speaker description> who says: <exact transcript>"
)
P10_TRANSCRIPT_STATE_AUTHORITY = "sceneplan.source.transcript"
P10_CANONICAL_EXECUTOR_FAMILY = "sceneplan_dit_v11_semantic_v2_15s_300m"
P10_CANONICAL_MODEL_CONFIG = (
    "/mnt/sdc/stable-audio-tools-workspace/stable_audio_tools/configs/"
    "model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s_"
    "resume_cosine_40k.json"
)
P10_CANONICAL_MODEL_CONFIG_SHA256 = (
    "3ebcd2b6b3c9a8b78b9160243f6509fb9eb11a32959b2eeddb86f48bb0b44827"
)
P10_CANONICAL_CHECKPOINT_STEP = 150_000
P10_CANONICAL_CHECKPOINT = (
    "/mnt/sdc/ckpts/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
    "checkpoints/epoch=48-step=150000.ckpt"
)
P10_CANONICAL_CHECKPOINT_SHA256 = (
    "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
)
P11_EXECUTION_CONTRACT = "external_p10_sceneplan_executor_v1"
P11_MODEL_CONTRACT = "sceneplan_p11_audio_aware_v1"
P11_EDITING_CONTRACT = "audio_aware_observed_patch_revised_v1"
P11_EDITING_INPUT_CONTRACT = "input_foa_required_old_sceneplan_optional_v1"
P11_EDITING_OUTPUT_CONTRACT = "observed_plan_atomic_patch_revised_plan_v1"
P11_SOURCE_IDENTITY_CONTRACT = "audio_observed_ids_with_optional_prior_anchor_v1"
P11_GENERATION_NATURAL_CONTRACT = "natural_renderer_caption_v1"
P11_EDIT_EVIDENCE_MODES = ("no_plan", "correct_plan", "corrupt_plan")
FOA_LATENT_CHANNELS = 64
P10_MAX_LATENT_FRAMES = 648
P11_MAX_LATENT_FRAMES = 648
# Canonical P11 now spans the same temporal envelope as the current P10 DiT.
MAX_LATENT_FRAMES = P11_MAX_LATENT_FRAMES
P11_SUPPORTED_MOTION_TYPES = ("static", "linear")
P11_GAIN_POLICY = "constant_0db_not_conditioned"
SEMANTIC_CAPTION_MAX_TOKENS = 512
SCENEPLAN_PLAN_MAX_TOKENS = 1024


class P11Task(str, Enum):
    GENERATION = "generation"
    UNDERSTANDING = "understanding"
    EDITING = "editing"


_TASK_ALIASES = {
    "generation": P11Task.GENERATION,
    "generate": P11Task.GENERATION,
    "text_to_audio": P11Task.GENERATION,
    "understanding": P11Task.UNDERSTANDING,
    "understand": P11Task.UNDERSTANDING,
    "audio_to_sceneplan": P11Task.UNDERSTANDING,
    "editing": P11Task.EDITING,
    "edit": P11Task.EDITING,
    "audio_edit": P11Task.EDITING,
}


def normalize_p11_task(value: Any) -> P11Task:
    """Return one canonical task id and reject multi-turn aliases."""

    if isinstance(value, P11Task):
        return value
    key = str(value or "").strip().lower()
    try:
        return _TASK_ALIASES[key]
    except KeyError as error:
        raise ValueError(
            "P11 supports only single-turn generation, understanding, or editing; "
            f"got {value!r}"
        ) from error


def _canonical_source_key(source: Mapping[str, Any]) -> tuple[Any, ...]:
    """Order sources only by information recoverable from the scene itself.

    Frozen source-index slot ids are renderer bookkeeping and have no acoustic
    meaning. Canonical P11 assigns new contiguous ids after sorting by audible
    timing/content and then by the physical trajectory.  The original id is
    deliberately absent from this key.
    """

    kind = str(source["kind"])
    semantic = (
        str(source["speaker_description"])
        if kind == "speech"
        else str(source["description"])
    )
    transcript = str(source.get("transcript") or "")
    activity = source["activity"]
    trajectory = json.dumps(
        source["trajectory"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        float(activity["onset_sec"]),
        float(activity["offset_sec"]),
        {"speech": 0, "music": 1, "sound": 2}[kind],
        " ".join(semantic.lower().split()),
        " ".join(transcript.lower().split()),
        trajectory,
    )


def canonicalize_sceneplan_source_ids(
    sceneplan: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an acoustically equivalent plan with observable canonical ids."""

    validate_model_sceneplan(sceneplan)
    output = copy.deepcopy(dict(sceneplan))
    output["sources"] = sorted(output["sources"], key=_canonical_source_key)
    for index, source in enumerate(output["sources"]):
        source["source_id"] = f"source_{index}"
    validate_model_sceneplan(output)
    return output


@dataclass(frozen=True)
class SingleTurnLayout:
    task: P11Task
    input_audio_span_roles: tuple[str, ...]
    supervises_sceneplan: bool
    requires_external_p10_execution: bool

    @property
    def num_input_audio_spans(self) -> int:
        return len(self.input_audio_span_roles)


_LAYOUTS = {
    P11Task.GENERATION: SingleTurnLayout(
        P11Task.GENERATION, (), True, True
    ),
    P11Task.UNDERSTANDING: SingleTurnLayout(
        P11Task.UNDERSTANDING, ("input",), True, False
    ),
    P11Task.EDITING: SingleTurnLayout(
        P11Task.EDITING, ("input",), True, True
    ),
}


def single_turn_layout(task: P11Task | str) -> SingleTurnLayout:
    return _LAYOUTS[normalize_p11_task(task)]


def validate_single_turn_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one audio-aware P11 supervision record without rewriting it."""

    if not isinstance(record, Mapping):
        raise TypeError("a P11 record must be an object")
    task = normalize_p11_task(record.get("task"))
    target_plan = record.get("target_sceneplan")
    if not isinstance(target_plan, Mapping):
        raise ValueError("every P11 task requires target_sceneplan")
    validate_model_sceneplan(target_plan)

    if task is P11Task.GENERATION:
        prompt = record.get("user_text")
    elif task is P11Task.UNDERSTANDING:
        prompt = record.get(
            "understanding_prompt", "Describe the spatial audio scene."
        )
    else:
        prompt = record.get("edit_instruction")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{task.value} requires a non-empty task prompt")

    # A target FOA is a P10 concern.  Keeping it in the P11 record would make
    # it too easy to accidentally resurrect the retired internal renderer.
    forbidden_target_audio = sorted(
        key
        for key in (
            "target_foa",
            "target_foa_ref",
            "target_foa_latent",
            "target_foa_latent_ref",
            "target_audio",
            "target_audio_ref",
        )
        if key in record
    )
    if forbidden_target_audio:
        raise ValueError(
            "P11 records must not carry target FOA supervision: "
            f"{forbidden_target_audio}"
        )
    input_audio_fields = (
        "input_foa",
        "input_foa_waveform",
        "input_foa_latent",
        "input_foa_ref",
        "input_foa_latent_ref",
    )
    has_input = any(record.get(key) is not None for key in input_audio_fields)
    expected_input = task in {P11Task.UNDERSTANDING, P11Task.EDITING}
    if has_input != expected_input:
        raise ValueError(
            f"{task.value} requires input FOA span={expected_input}, got {has_input}"
        )

    output = dict(record)
    output["task"] = task.value
    if task is P11Task.EDITING:
        input_plan = record.get("input_sceneplan")
        if input_plan is not None:
            if not isinstance(input_plan, Mapping):
                raise ValueError("editing input_sceneplan must be an object or null")
            validate_model_sceneplan(input_plan)
        observed = record.get("observed_sceneplan_target")
        if not isinstance(observed, Mapping):
            raise ValueError("editing requires observed_sceneplan_target")
        validate_model_sceneplan(observed)
        patch = record.get("edit_patch_target")
        if not isinstance(patch, Mapping):
            raise ValueError("editing requires edit_patch_target")
        evidence_mode = str(record.get("editing_evidence_mode") or "")
        if evidence_mode not in P11_EDIT_EVIDENCE_MODES:
            raise ValueError(
                "editing_evidence_mode must be no_plan, correct_plan, or corrupt_plan"
            )
        if (evidence_mode == "no_plan") != (input_plan is None):
            raise ValueError("editing evidence mode disagrees with input_sceneplan")
        output["editing_contract"] = P11_EDITING_CONTRACT
        output["editing_input_contract"] = P11_EDITING_INPUT_CONTRACT
        output["editing_output_contract"] = P11_EDITING_OUTPUT_CONTRACT
        output["preserves_unedited_waveform"] = False
    return output


def _validate_latent(value: Tensor, *, label: str) -> None:
    if value.ndim != 2 or int(value.shape[0]) != FOA_LATENT_CHANNELS:
        raise ValueError(f"{label} must be [64,T], got {tuple(value.shape)}")
    if not 1 <= int(value.shape[1]) <= P11_MAX_LATENT_FRAMES:
        raise ValueError(
            f"{label} exceeds the P11-v2 {P11_MAX_LATENT_FRAMES}-frame profile: "
            f"{value.shape[1]}"
        )
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains non-finite values")


def validate_runtime_audio_spans(
    task: P11Task | str,
    *,
    input_foa: Tensor | None,
    target_foa: Tensor | None = None,
) -> None:
    """Enforce that P11 reads at most one clean input and predicts no audio."""

    task = normalize_p11_task(task)
    if target_foa is not None:
        raise ValueError(
            "P11 no longer accepts a target FOA span; P10 is the only renderer"
        )
    if input_foa is not None:
        _validate_latent(input_foa, label="input_foa")
    expected_input = task in {P11Task.UNDERSTANDING, P11Task.EDITING}
    if (input_foa is not None) != expected_input:
        raise ValueError(
            f"{task.value} requires input FOA latent={expected_input}, "
            f"got {input_foa is not None}"
        )


def compile_p10_aligned_target_conditions(
    sceneplan: Mapping[str, Any],
    *,
    model_num_samples: int | None = None,
    latent_frames_valid: int | None = None,
) -> dict[str, Any]:
    """Compile the exact semantic + 4+4 target conditions used by P10."""

    plan = validate_model_sceneplan(sceneplan)
    # P11 hands structured state to P10 through one canonical, unambiguous
    # surface form.  Quote marks never define speech ownership; the explicit
    # speech_source_ids role map does.
    semantic = compile_model_semantic_caption_v2(plan)
    if (
        int(semantic.get("compiler_version", -1))
        != P10_SEMANTIC_CAPTION_COMPILER_VERSION
    ):
        raise RuntimeError("P11 did not compile the frozen P10 semantic-caption v2 surface")
    controls = compile_model_44_controls(
        plan,
        model_num_samples=model_num_samples,
        latent_frames_valid=latent_frames_valid,
    )
    event_ids = controls["source_event_frame_ids"]
    trajectories = controls["source_trajectory_features"]
    if event_ids.ndim != 2 or event_ids.shape[0] != MAX_SOURCES:
        raise RuntimeError("P10 compiler returned a non-4-track event tensor")
    if trajectories.shape != (*event_ids.shape, 5):
        raise RuntimeError("P10 compiler returned a non-4x5 trajectory tensor")
    if np.any((event_ids < 0) | (event_ids > MAX_SOURCES)):
        raise RuntimeError("positive P10 controls unexpectedly contain CFG unknown ids")
    return {
        "semantic_caption": semantic,
        "renderer_caption": compile_model_renderer_caption(plan),
        "sceneplan_44": controls,
        "alignment": p10_p11_alignment_contract(),
    }


def validate_p11_executor_profile(
    sceneplan: Mapping[str, Any],
) -> dict[str, Any]:
    """Reject fields that the current P11 profile cannot safely hand to P10.

    Canonical P11 spans the complete 648-frame P10 temporal envelope.  Motion
    remains static/linear because that is the latest P10 training support;
    compiler-only keyframed interpolation is not exposed as a learned claim.
    """

    plan = validate_model_sceneplan(sceneplan)
    model_num_samples = int(round(float(plan["duration_sec"]) * MODEL_SAMPLE_RATE))
    latent_frames = int(math.ceil(model_num_samples / VAE_HOP_SAMPLES))
    if not 1 <= latent_frames <= P11_MAX_LATENT_FRAMES:
        raise ValueError(
            "ScenePlan lies outside the P11-v2 capability profile: "
            f"{latent_frames} frames not in [1,{P11_MAX_LATENT_FRAMES}]"
        )
    for source in plan["sources"]:
        motion = str(source["trajectory"]["type"])
        if motion not in P11_SUPPORTED_MOTION_TYPES:
            raise ValueError(
                f"{source['source_id']}: motion {motion!r} is representable by the "
                "P10 compiler but is outside the trained P11-v2 profile"
            )
        if not math.isclose(float(source["gain_db"]), 0.0, abs_tol=1.0e-12):
            raise ValueError(
                f"{source['source_id']}: gain_db is not a P10 condition; "
                "P11 must canonicalize it to 0 dB"
            )
    return plan


class ScenePlanResolver(Protocol):
    """Resolve decoded P11 output into P10's fully numeric ScenePlan state."""

    def resolve(
        self,
        sceneplan: Mapping[str, Any],
        *,
        task: P11Task,
        sample_id: str,
    ) -> Mapping[str, Any]: ...


class StrictNumericScenePlanResolver:
    """Current v1 resolver: accept only codec-decoded, fully numeric plans."""

    def resolve(
        self,
        sceneplan: Mapping[str, Any],
        *,
        task: P11Task,
        sample_id: str,
    ) -> Mapping[str, Any]:
        del task
        plan = validate_model_sceneplan(sceneplan)
        if str(plan["sample_id"]) != str(sample_id):
            raise ValueError("resolver changed the requested ScenePlan sample_id")
        return plan


@dataclass(frozen=True)
class ScenePlanExecutionBundle:
    """The only object allowed to cross from P11 into the P10 executor."""

    task: P11Task
    sample_id: str
    plan_token_ids: Tensor
    sceneplan: Mapping[str, Any]
    renderer_caption: Mapping[str, Any]
    p10_metadata: Mapping[str, Any]
    model_num_samples: int
    latent_frames_valid: int
    execution_contract: str = P11_EXECUTION_CONTRACT
    editing_contract: str | None = None
    preserves_unedited_waveform: bool = False
    localized_editing: bool = False
    edit_patch_token_ids: Tensor | None = None
    edit_patch: Mapping[str, Any] | None = None

    @property
    def requires_p10_render(self) -> bool:
        return self.task in {P11Task.GENERATION, P11Task.EDITING}

    def assert_external_p10_boundary(self) -> None:
        if self.execution_contract != P11_EXECUTION_CONTRACT:
            raise RuntimeError("P11/P10 execution contract changed")
        forbidden = {"input_foa", "input_audio", "source_waveform"}
        leaked = sorted(forbidden.intersection(self.p10_metadata))
        if leaked:
            raise RuntimeError(f"input audio leaked across the P10 boundary: {leaked}")
        if (
            int(self.p10_metadata.get("semantic_caption_compiler_version", -1))
            != P10_SEMANTIC_CAPTION_COMPILER_VERSION
            or self.p10_metadata.get("semantic_caption_contract")
            != P10_SEMANTIC_CAPTION_CONTRACT
        ):
            raise RuntimeError("P11/P10 semantic-caption surface contract changed")
        if self.task is P11Task.EDITING:
            if self.editing_contract != P11_EDITING_CONTRACT:
                raise RuntimeError("editing contract is not atomic ScenePlan patch")
            if self.preserves_unedited_waveform or self.localized_editing:
                raise RuntimeError("current editing contract cannot claim local preservation")
            if self.edit_patch_token_ids is None or self.edit_patch is None:
                raise RuntimeError("editing bundle does not carry its applied atomic patch")
        elif self.edit_patch_token_ids is not None or self.edit_patch is not None:
            raise RuntimeError("non-editing bundle unexpectedly carries an edit patch")


@dataclass(frozen=True)
class AudioAwareEditPlanningBundle:
    """Auditable P11 Editing result before any future reference-aware renderer.

    Input-audio references stay at the planner boundary.  Only the nested
    ``execution_bundle`` may cross into today's frozen P10, and that bundle
    contains the revised ScenePlan but no source waveform or latent.
    """

    sample_id: str
    observed_plan_token_ids: Tensor
    observed_sceneplan: Mapping[str, Any]
    edit_patch_token_ids: Tensor
    edit_patch: Mapping[str, Any]
    revised_plan_token_ids: Tensor
    revised_sceneplan: Mapping[str, Any]
    source_to_target_slot_map: Mapping[str, str | None]
    execution_bundle: ScenePlanExecutionBundle
    input_foa_ref: str | None = None
    input_foa_latent_ref: str | None = None
    model_contract: str = P11_MODEL_CONTRACT
    editing_contract: str = P11_EDITING_CONTRACT
    editing_input_contract: str = P11_EDITING_INPUT_CONTRACT
    editing_output_contract: str = P11_EDITING_OUTPUT_CONTRACT
    requires_reference_conditioned_p10: bool = True
    localized_editing: bool = False
    preserves_unedited_waveform: bool = False

    def assert_consistent(self, *, codec: Any, patch_codec: Any) -> None:
        if self.model_contract != P11_MODEL_CONTRACT:
            raise RuntimeError("audio-aware P11 model contract changed")
        if (
            self.editing_contract != P11_EDITING_CONTRACT
            or self.editing_input_contract != P11_EDITING_INPUT_CONTRACT
            or self.editing_output_contract != P11_EDITING_OUTPUT_CONTRACT
        ):
            raise RuntimeError("audio-aware Editing contract changed")
        if not self.requires_reference_conditioned_p10:
            raise RuntimeError("audio-aware edit bundle lost its future renderer flag")
        if self.localized_editing or self.preserves_unedited_waveform:
            raise RuntimeError("current P10 cannot claim waveform-local editing")

        observed = validate_p11_executor_profile(self.observed_sceneplan)
        revised = validate_p11_executor_profile(self.revised_sceneplan)
        applied = patch_codec.apply(observed, self.edit_patch)
        applied_ids = codec.encode(applied)["input_ids"]
        revised_ids = codec.encode(revised)["input_ids"]
        if not torch.equal(applied_ids, revised_ids):
            raise RuntimeError("apply(edit_patch, observed) != revised ScenePlan")
        if not torch.equal(
            codec.canonicalize(self.observed_plan_token_ids)["input_ids"],
            codec.encode(observed)["input_ids"],
        ):
            raise RuntimeError("observed ScenePlan tokens disagree with observed plan")
        if not torch.equal(
            codec.canonicalize(self.revised_plan_token_ids)["input_ids"],
            revised_ids,
        ):
            raise RuntimeError("revised ScenePlan tokens disagree with revised plan")
        canonical_patch = patch_codec.canonicalize(self.edit_patch_token_ids)[
            "input_ids"
        ]
        if not torch.equal(canonical_patch, patch_codec.encode(self.edit_patch)["input_ids"]):
            raise RuntimeError("edit patch tokens disagree with decoded patch")
        if (
            self.execution_bundle.task is not P11Task.EDITING
            or self.execution_bundle.sceneplan != revised
            or self.execution_bundle.edit_patch != self.edit_patch
        ):
            raise RuntimeError("P10 execution bundle is not bound to the revised plan")
        self.execution_bundle.assert_external_p10_boundary()

        observed_ids = {str(source["source_id"]) for source in observed["sources"]}
        revised_ids_set = {str(source["source_id"]) for source in revised["sources"]}
        if set(self.source_to_target_slot_map) != observed_ids:
            raise RuntimeError("source-slot map does not cover every observed source")
        mapped = {
            str(value)
            for value in self.source_to_target_slot_map.values()
            if value is not None
        }
        if not mapped.issubset(revised_ids_set):
            raise RuntimeError("source-slot map names an absent revised source")


def finalize_sceneplan_for_p10(
    codec: ModelScenePlanCodec,
    plan_token_ids: Mapping[str, Tensor] | Tensor | Sequence[int],
    *,
    tokenizer: Any,
    task: P11Task | str,
    sample_id: str,
    resolver: ScenePlanResolver | None = None,
    edit_patch_token_ids: Tensor | None = None,
    edit_patch: Mapping[str, Any] | None = None,
) -> ScenePlanExecutionBundle:
    """Decode, resolve, validate, and deterministically compile one P11 plan."""

    normalized_task = normalize_p11_task(task)
    raw_ids = (
        plan_token_ids.get("input_ids")
        if isinstance(plan_token_ids, Mapping)
        else plan_token_ids
    )
    if raw_ids is None:
        raise ValueError("P11 handoff has no ScenePlan token ids")
    canonical = codec.canonicalize(
        torch.as_tensor(raw_ids, dtype=torch.long).detach().cpu().flatten(),
        max_tokens=SCENEPLAN_PLAN_MAX_TOKENS,
    )["input_ids"].detach().cpu().contiguous()
    decoded = codec.decode(canonical, sample_id=str(sample_id))
    active_resolver = resolver or StrictNumericScenePlanResolver()
    resolved = active_resolver.resolve(
        decoded, task=normalized_task, sample_id=str(sample_id)
    )
    plan = validate_p11_executor_profile(resolved)

    model_num_samples = int(
        round(float(plan["duration_sec"]) * MODEL_SAMPLE_RATE)
    )
    latent_frames_valid = int(math.ceil(model_num_samples / VAE_HOP_SAMPLES))
    if not 1 <= latent_frames_valid <= P11_MAX_LATENT_FRAMES:
        raise ValueError("resolved ScenePlan lies outside the P11-v2 frame profile")
    compiled = compile_p10_aligned_target_conditions(
        plan,
        model_num_samples=model_num_samples,
        latent_frames_valid=latent_frames_valid,
    )
    tokenized = tokenize_model_semantic_caption(
        compiled["semantic_caption"],
        tokenizer,
        max_length=SEMANTIC_CAPTION_MAX_TOKENS,
    )
    prompt = {
        "input_ids": torch.as_tensor(tokenized["input_ids"], dtype=torch.long),
        "attention_mask": torch.as_tensor(
            tokenized["attention_mask"], dtype=torch.bool
        ),
        "event_source_ids": torch.as_tensor(
            tokenized["event_source_ids"], dtype=torch.int8
        ),
        "speech_source_ids": torch.as_tensor(
            tokenized["speech_source_ids"], dtype=torch.int8
        ),
        "speech_lexical_mask": torch.as_tensor(
            tokenized["speech_lexical_mask"], dtype=torch.bool
        ),
    }
    source_controls = compiled["sceneplan_44"]
    valid_mask = torch.ones(latent_frames_valid, dtype=torch.bool)
    controls = {
        "source_event_frame_ids": torch.as_tensor(
            source_controls["source_event_frame_ids"], dtype=torch.int8
        ),
        "source_trajectory_features": torch.as_tensor(
            source_controls["source_trajectory_features"], dtype=torch.float32
        ),
        "frame_valid_mask": valid_mask.clone(),
        # Supervision-only in P10 training; harmless and useful for audits.
        "speech_active_frame_mask": torch.as_tensor(
            source_controls["speech_active_frame_mask"], dtype=torch.bool
        ),
    }
    p10_metadata = {
        "sample_id": str(sample_id),
        "model_sceneplan": plan,
        "model_num_samples": model_num_samples,
        "prompt": prompt,
        "prompt_text": compiled["semantic_caption"]["text"],
        "semantic_caption_compiler_version": P10_SEMANTIC_CAPTION_COMPILER_VERSION,
        "semantic_caption_contract": P10_SEMANTIC_CAPTION_CONTRACT,
        "transcript_state_authority": P10_TRANSCRIPT_STATE_AUTHORITY,
        "sceneplan_44": controls,
        "padding_mask": [valid_mask],
        "seconds_start": 0.0,
        "seconds_total": model_num_samples / MODEL_SAMPLE_RATE,
        "latent_stored_length": latent_frames_valid,
        "latent_crop_length": latent_frames_valid,
        "latent_crop_start": 0,
        "p11_execution_contract": P11_EXECUTION_CONTRACT,
    }
    editing_contract = (
        P11_EDITING_CONTRACT if normalized_task is P11Task.EDITING else None
    )
    bundle = ScenePlanExecutionBundle(
        task=normalized_task,
        sample_id=str(sample_id),
        plan_token_ids=canonical,
        sceneplan=plan,
        renderer_caption=compiled["renderer_caption"],
        p10_metadata=p10_metadata,
        model_num_samples=model_num_samples,
        latent_frames_valid=latent_frames_valid,
        editing_contract=editing_contract,
        preserves_unedited_waveform=False,
        localized_editing=False,
        edit_patch_token_ids=(
            None
            if edit_patch_token_ids is None
            else torch.as_tensor(edit_patch_token_ids, dtype=torch.long)
            .detach()
            .cpu()
            .contiguous()
        ),
        edit_patch=None if edit_patch is None else copy.deepcopy(dict(edit_patch)),
    )
    bundle.assert_external_p10_boundary()
    return bundle


def p10_p11_alignment_contract() -> dict[str, Any]:
    """Return the immutable system boundary shared by P10 and P11."""

    return {
        "p11_contract_version": P11_CONTRACT_VERSION,
        "p11_model_contract": P11_MODEL_CONTRACT,
        "p11_responsibility": (
            "predict_observed_sceneplan_atomic_patch_and_revised_sceneplan"
        ),
        "p10_responsibility": "sceneplan_to_foa_only",
        "execution_contract": P11_EXECUTION_CONTRACT,
        "editing_contract": P11_EDITING_CONTRACT,
        "editing_input_contract": P11_EDITING_INPUT_CONTRACT,
        "editing_output_contract": P11_EDITING_OUTPUT_CONTRACT,
        "editing_requires_input_foa": True,
        "editing_old_sceneplan_role": "optional_prior",
        "editing_requires_reference_conditioned_p10": True,
        "current_p10_is_reference_conditioned": False,
        "editing_preserves_unedited_waveform": False,
        "localized_editing": False,
        "p10_sceneplan_contract_revision": P10_SCENEPLAN_CONTRACT_REVISION,
        "p10_conditioning_contract_revision": P10_CONDITIONING_CONTRACT_REVISION,
        "p10_dataset_contract_revision": P10_DATASET_CONTRACT_REVISION,
        "p10_semantic_caption_contract": P10_SEMANTIC_CAPTION_CONTRACT,
        "p10_semantic_caption_compiler_version": (
            P10_SEMANTIC_CAPTION_COMPILER_VERSION
        ),
        "p10_semantic_caption_surface": P10_SEMANTIC_CAPTION_SURFACE,
        "transcript_state_authority": P10_TRANSCRIPT_STATE_AUTHORITY,
        "sceneplan_schema": "stable_audio_tools.model_sceneplan.v1",
        "sample_rate": MODEL_SAMPLE_RATE,
        "vae_hop_samples": VAE_HOP_SAMPLES,
        "foa_latent_channels": FOA_LATENT_CHANNELS,
        "p10_max_latent_frames": P10_MAX_LATENT_FRAMES,
        "p11_profile_max_latent_frames": P11_MAX_LATENT_FRAMES,
        "p11_duration_matches_p10": True,
        "p11_is_strict_p10_subset": False,
        "p11_supported_motion_types": list(P11_SUPPORTED_MOTION_TYPES),
        "p11_gain_policy": P11_GAIN_POLICY,
        "word_level_timing_supported": False,
        "semantic_caption_max_tokens": SEMANTIC_CAPTION_MAX_TOKENS,
        "sceneplan_plan_max_tokens": SCENEPLAN_PLAN_MAX_TOKENS,
        "caption_roles": {
            "event_source_ids": [-1, 0, 1, 2, 3, 4],
            "speech_source_ids": [-1, 0, 1, 2, 3, 4],
            "positive_roles_are_disjoint": True,
        },
        "structured_control": {
            "max_sources": 4,
            "event_tracks": 4,
            "trajectory_tracks": 4,
            "trajectory_features": [
                "sin_azimuth",
                "cos_azimuth",
                "sin_elevation",
                "cos_elevation",
                "log1p_distance_m",
            ],
            "gain_db_is_not_a_model_condition": True,
            "target_span_only": True,
        },
        "semantic_control": {
            "route": "frozen_qwen_cross_attention",
            "fields": [
                "room.type",
                "source.kind",
                "source.description",
                "source.speaker_description",
                "source.transcript",
            ],
            "word_level_timestamps": False,
        },
        "unsupported_claims": [
            "nonzero_source_gain_control",
            "word_level_speech_timing",
            "reference_conditioned_p10",
            "localized_waveform_editing",
            "unedited_waveform_preservation",
            "more_than_four_sources",
        ],
    }


__all__ = [
    "FOA_LATENT_CHANNELS",
    "MAX_LATENT_FRAMES",
    "P10_CONDITIONING_CONTRACT_REVISION",
    "P10_DATASET_CONTRACT_REVISION",
    "P10_MAX_LATENT_FRAMES",
    "P10_SEMANTIC_CAPTION_COMPILER_VERSION",
    "P10_SEMANTIC_CAPTION_CONTRACT",
    "P10_SEMANTIC_CAPTION_SURFACE",
    "P10_SCENEPLAN_CONTRACT_REVISION",
    "P10_TRANSCRIPT_STATE_AUTHORITY",
    "P11_CONTRACT_VERSION",
    "P11_EDIT_EVIDENCE_MODES",
    "P11_EDITING_CONTRACT",
    "P11_EDITING_INPUT_CONTRACT",
    "P11_EDITING_OUTPUT_CONTRACT",
    "P11_EXECUTION_CONTRACT",
    "P11_GENERATION_NATURAL_CONTRACT",
    "P11_GAIN_POLICY",
    "P11_MAX_LATENT_FRAMES",
    "P11_MODEL_CONTRACT",
    "P11_SOURCE_IDENTITY_CONTRACT",
    "P11_SUPPORTED_MOTION_TYPES",
    "P11Task",
    "SCENEPLAN_PLAN_MAX_TOKENS",
    "SEMANTIC_CAPTION_MAX_TOKENS",
    "AudioAwareEditPlanningBundle",
    "ScenePlanExecutionBundle",
    "ScenePlanResolver",
    "SingleTurnLayout",
    "StrictNumericScenePlanResolver",
    "canonicalize_sceneplan_source_ids",
    "compile_p10_aligned_target_conditions",
    "finalize_sceneplan_for_p10",
    "normalize_p11_task",
    "p10_p11_alignment_contract",
    "single_turn_layout",
    "validate_runtime_audio_spans",
    "validate_single_turn_record",
    "validate_p11_executor_profile",
]
