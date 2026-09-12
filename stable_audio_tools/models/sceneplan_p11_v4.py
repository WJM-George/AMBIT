"""End-to-end sketch-first Transfusion-CoT planner for P11-v4.

Internal causal order is deliberately one-way::

    task/evidence -> discrete SceneSketch/DeltaSketch -> continuous Thought
                  -> deterministic ScenePlan/Atomic Patch -> frozen P10

The final user-visible trace still exposes ``<SCENE_THOUGHT> ... <PLAN>`` or
``<DELTA_THOUGHT> ... <PATCH>``.  The earlier discrete sketch is an internal
semantic authority, decoded before the continuous state, so a thought
intervention cannot rewrite room/source/kind/text.
"""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from ..data.model_sceneplan_codec_v3 import (
    KIND_TOKENS,
    ROOM_TOKENS,
    SOURCE_COUNT_TOKENS,
    SOURCE_SLOT_TOKENS,
)
from ..data.sceneplan_edit_patch import OPERATION_TOKENS
from ..data.scene_sketch_v1 import (
    AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT,
    CONTROL_DIRECTION_CONTRACT,
    CONTROL_DIRECTION_OPERATIONS,
    CONTROL_DIRECTION_TOKENS,
    DELTA_SKETCH_TOKEN_CONTRACT,
    EXECUTION_FEATURE_DIM,
    EXECUTION_FEATURE_NAMES,
    EXECUTION_SLOT_COUNT,
    SCENE_SKETCH_CONTRACT,
    AudioAwareDeltaSceneSketchCodec,
    DeltaSceneSketchCodec,
    SceneSketchCodec,
    apply_reliable_lexical_authority_to_sketch,
    audio_aware_delta_control_mask,
    assemble_sceneplan,
    compile_execution_state,
    compile_p10_from_contract,
    execution_state_core,
    execution_state_from_core,
    project_audio_aware_delta_to_atomic_patch,
    project_delta_thought_to_atomic_patch,
    select_reliable_lexical_source_owner,
)
from ..data.sceneplan_p11_lexical_cache import (
    P11_LEXICAL_ASSEMBLER_POLICY,
    P11_LEXICAL_TOKEN_CONCAT_POLICY,
    parse_reliable_lexical_authority,
)
from ..data.sceneplan_p11_single_turn import (
    AudioAwareEditPlanningBundle,
    MAX_LATENT_FRAMES,
    P11Task,
    ScenePlanExecutionBundle,
    ScenePlanResolver,
    normalize_p11_task,
)
from ..data.sceneplan_p11_v4_dataset import (
    P11_AUDIO_AWARE_DATA_CONTRACT,
    P11_AUDIO_AWARE_SEQUENCE_CONTRACT,
    P11_V4_DATA_CONTRACT,
    P11_V4_SEQUENCE_CONTRACT,
)
from .scene_thought_p11_v4 import (
    P11_V4_THOUGHT_CONTRACT,
    SketchFirstExecutionReasoner,
)
from .sceneplan_p11 import ScenePlanP11Planner


P11_V4_MODEL_CONTRACT = "p11_sketch_first_transfusion_cot_v4"
P11_V4_OUTPUT_CONTRACT = "deterministic_sceneplan_or_atomic_patch_v1"
P11_V4_DELTA_OWNER_CONTRACT = "delta_sketch_decoder_owned_source_slot_v1"
P11_V4_DELTA_OWNER_OBJECTIVE = (
    "legal_source_ce_plus_hardest_negative_margin_v1"
)
P11_V4_DISCRETE_BOUNDARY_CONTRACT = (
    "scene_sketch_finite_field_boundary_supervision_v1"
)
P11_V4_U_INVENTORY_CONTRACT = "u_same_decoder_finite_inventory_aux_v1"
P11_V4_U_INVENTORY_OBJECTIVE = "grammar_candidate_ce_v1"
P11_AUDIO_AWARE_MODEL_CONTRACT = "sceneplan_p11_audio_aware_v1"
P11_AUDIO_AWARE_OUTPUT_CONTRACT = "observed_plan_atomic_patch_revised_plan_v1"
P11_AUDIO_AWARE_DELTA_CONTROL_CONTRACT = (
    "operation_owned_p10_native_smooth_l1_v1"
)


class P11DiscreteDecodeError(RuntimeError):
    """Fail-closed decode error retaining a CPU copy for offline diagnosis."""

    def __init__(self, message: str, partial_tokens: Tensor) -> None:
        super().__init__(message)
        self.partial_tokens = (
            torch.as_tensor(partial_tokens, dtype=torch.long).detach().cpu().flatten()
        )


def _audio_aware_delta_control_objective(
    predicted: Tensor,
    target: Tensor,
    observed: Tensor,
    programs: Sequence[Mapping[str, Any]],
    *,
    max_frames: int,
    retime_frame_unit: float,
    beta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return row-balanced, operation-owned P10-native delta supervision.

    Spatial and add-source coordinates already live at useful scales in the
    ExecutionState.  Activity coordinates are normalized by scene duration,
    so retime residuals are first converted to units of ``retime_frame_unit``
    P10 latent frames.  Smooth-L1 then bounds large early gradients without
    erasing small but executable timing changes.
    """

    if predicted.shape != target.shape or predicted.shape != observed.shape:
        raise ValueError("audio-aware delta-control tensors must have equal shape")
    if predicted.ndim != 3 or tuple(predicted.shape[1:]) != (
        EXECUTION_SLOT_COUNT,
        EXECUTION_FEATURE_DIM,
    ):
        raise ValueError("audio-aware delta-control core shape changed")
    if len(programs) != int(predicted.shape[0]):
        raise ValueError("audio-aware delta programs do not align with rows")
    if int(max_frames) <= 0 or float(retime_frame_unit) <= 0.0 or float(beta) <= 0.0:
        raise ValueError("audio-aware delta-control scales must be positive")
    if not (
        torch.isfinite(predicted).all()
        and torch.isfinite(target).all()
        and torch.isfinite(observed).all()
    ):
        raise ValueError("audio-aware delta-control tensors must be finite")

    device = predicted.device
    mask = torch.stack(
        [
            torch.from_numpy(audio_aware_delta_control_mask(program))
            for program in programs
        ]
    ).to(device=device, dtype=torch.bool)
    scale = torch.ones_like(predicted, dtype=torch.float32)
    duration_index = EXECUTION_FEATURE_NAMES.index("duration_frames_norm")
    onset_index = EXECUTION_FEATURE_NAMES.index("onset_frame_norm")
    offset_index = EXECUTION_FEATURE_NAMES.index("offset_frame_norm")
    duration_frames = (
        observed[:, 0, duration_index].float() * float(max_frames)
    ).round()
    if bool((duration_frames < 1.0).any()):
        raise ValueError("audio-aware retime rows require a positive scene duration")
    for row, program in enumerate(programs):
        if str(program.get("operation") or "") == "retime_source":
            scale[row, :, onset_index : offset_index + 1] = (
                duration_frames[row] / float(retime_frame_unit)
            )

    error = (predicted.float() - target.float()) * scale
    penalties = F.smooth_l1_loss(
        error,
        torch.zeros_like(error),
        reduction="none",
        beta=float(beta),
    ) * mask.float()
    denominator = mask.float().sum(dim=(1, 2)).clamp_min(1.0)
    per_row = penalties.sum(dim=(1, 2)) / denominator
    active = mask.any(dim=2).any(dim=1)
    raw_squared = (predicted.float() - target.float()).square() * mask.float()
    raw_rmse = torch.sqrt(raw_squared.sum(dim=(1, 2)) / denominator)
    return per_row, active, raw_rmse


def _finite_field_token_objective(
    logits: Tensor,
    labels: Tensor,
    active_mask: Tensor,
    candidate_token_ids: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Score one grammar-constrained field through the existing LM head.

    The candidate logits are vocabulary rows from ``plan_embedding.weight``;
    no auxiliary classifier or second SceneSketch authority is introduced.
    This exactly matches constrained decoding, where room, source count, and
    source kind are selected only from their respective finite token sets.
    """

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("P11-v4 finite-field logits/labels do not align")
    mask = torch.as_tensor(
        active_mask, device=logits.device, dtype=torch.bool
    )
    if mask.shape != labels.shape:
        raise ValueError("P11-v4 finite-field mask does not align with labels")
    candidates = torch.as_tensor(
        candidate_token_ids, device=logits.device, dtype=torch.long
    ).flatten()
    if candidates.numel() < 2 or int(candidates.unique().numel()) != int(
        candidates.numel()
    ):
        raise ValueError("P11-v4 finite-field candidates must be unique")

    count = mask.sum()
    if not bool(mask.any()):
        zero = logits.sum() * 0.0
        return zero, count, zero.detach()

    active_labels = labels[mask]
    matches = active_labels[:, None].eq(candidates[None, :])
    if not bool(matches.sum(dim=1).eq(1).all()):
        raise ValueError("P11-v4 finite-field target is outside its candidates")
    target_classes = matches.to(torch.long).argmax(dim=1)
    candidate_logits = logits[mask].index_select(1, candidates)
    loss = F.cross_entropy(candidate_logits, target_classes)
    accuracy = candidate_logits.argmax(dim=1).eq(target_classes).float().mean()
    return loss, count, accuracy


def _delta_owner_objective(
    source_logits: Tensor,
    target_slot: int,
    legal_source_mask: Tensor,
    *,
    margin: float,
    margin_weight: float,
) -> tuple[Tensor, Tensor]:
    """Train the exact source-slot decision used by DeltaSketch decoding.

    ``source_logits`` must be the four vocabulary logits at the autoregressive
    owner-token position.  The legal mask is derived from the current
    ScenePlan, exactly like :meth:`DeltaSceneSketchCodec.allowed_next_ids`.
    Consequently this objective introduces no second source-owner authority.
    """

    if source_logits.ndim != 1 or int(source_logits.numel()) != len(
        SOURCE_SLOT_TOKENS
    ):
        raise ValueError("P11-v4 DeltaSketch owner logits must contain four slots")
    target_slot = int(target_slot)
    if not 0 <= target_slot < len(SOURCE_SLOT_TOKENS):
        raise ValueError("P11-v4 DeltaSketch owner target is outside [0,4)")
    legal = torch.as_tensor(
        legal_source_mask, device=source_logits.device, dtype=torch.bool
    ).flatten()
    if tuple(legal.shape) != (len(SOURCE_SLOT_TOKENS),):
        raise ValueError("P11-v4 DeltaSketch legal-source mask must be [4]")
    if not bool(legal[target_slot]):
        raise ValueError("P11-v4 DeltaSketch owner target is absent from input ScenePlan")
    if not math.isfinite(float(margin)) or float(margin) <= 0.0:
        raise ValueError("P11-v4 DeltaSketch owner margin must be positive and finite")
    if not math.isfinite(float(margin_weight)) or float(margin_weight) <= 0.0:
        raise ValueError(
            "P11-v4 DeltaSketch owner margin weight must be positive and finite"
        )

    legal_indices = torch.nonzero(legal, as_tuple=False).flatten()
    legal_logits = source_logits.index_select(0, legal_indices)
    predicted_slot = legal_indices[legal_logits.argmax()]
    correct = predicted_slot.eq(target_slot).to(source_logits.dtype)
    if int(legal_indices.numel()) == 1:
        # There is no source-owner decision to learn in a one-source scene.
        return source_logits.sum() * 0.0, correct

    target_position = torch.nonzero(
        legal_indices.eq(target_slot), as_tuple=False
    ).flatten()
    if int(target_position.numel()) != 1:
        raise RuntimeError("P11-v4 DeltaSketch owner target is not uniquely legal")
    cross_entropy = F.cross_entropy(
        legal_logits.view(1, -1), target_position.view(1)
    )
    hardest_negative = source_logits.index_select(
        0, legal_indices[legal_indices.ne(target_slot)]
    ).max()
    target_logit = source_logits[target_slot]
    ranking = F.softplus(source_logits.new_tensor(float(margin)) - (
        target_logit - hardest_negative
    ))
    return cross_entropy + float(margin_weight) * ranking, correct


class ScenePlanP11V4Planner(ScenePlanP11Planner):
    """Shared-Qwen mixed discrete/continuous planner with a hard P10 boundary."""

    def __init__(self, model_config: Mapping[str, Any]) -> None:
        original = copy.deepcopy(dict(model_config))
        base = copy.deepcopy(original)
        super().__init__(base)
        self.model_config = original

        config = dict((original.get("model") or {}).get("transfusion_cot") or {})
        if config.get("contract") != P11_V4_MODEL_CONTRACT:
            raise ValueError(
                f"P11-v4 transfusion_cot.contract must be {P11_V4_MODEL_CONTRACT!r}"
            )
        expected = {
            "sequence_contract": P11_V4_SEQUENCE_CONTRACT,
            "scene_sketch_contract": SCENE_SKETCH_CONTRACT,
            "thought_contract": P11_V4_THOUGHT_CONTRACT,
            "delta_token_contract": DELTA_SKETCH_TOKEN_CONTRACT,
            "output_contract": P11_V4_OUTPUT_CONTRACT,
            "assembler": "deterministic_sketch_execution_assembler_v1",
            "semantic_from_execution_state": False,
            "numeric_from_scene_sketch": False,
            "editing_render_seed_policy": "same_each_turn",
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(
                    f"P11-v4 transfusion_cot.{key}={config.get(key)!r}, expected {value!r}"
                )

        delta_owner = dict(config.get("delta_owner") or {})
        expected_delta_owner = {
            "contract": P11_V4_DELTA_OWNER_CONTRACT,
            "objective": P11_V4_DELTA_OWNER_OBJECTIVE,
            "authority": "delta_sketch_owner_token_v1",
            "context": "edit_instruction_plus_current_sceneplan_v1",
            "inference": "grammar_constrained_argmax_v1",
            "legal_source_inventory": "input_sceneplan_source_mask_v1",
            "loss_normalization": "active_owner_rows_v1",
        }
        for key, value in expected_delta_owner.items():
            if delta_owner.get(key) != value:
                raise ValueError(
                    "P11-v4 transfusion_cot.delta_owner."
                    f"{key}={delta_owner.get(key)!r}, expected {value!r}"
                )
        self.delta_owner_margin = float(delta_owner.get("margin", 0.0))
        self.delta_owner_margin_weight = float(
            delta_owner.get("margin_weight", 0.0)
        )
        if not math.isfinite(self.delta_owner_margin) or self.delta_owner_margin <= 0.0:
            raise ValueError("P11-v4 DeltaSketch owner margin must be positive")
        if (
            not math.isfinite(self.delta_owner_margin_weight)
            or self.delta_owner_margin_weight <= 0.0
        ):
            raise ValueError("P11-v4 DeltaSketch owner margin weight must be positive")
        self.delta_owner_contract = P11_V4_DELTA_OWNER_CONTRACT
        self.delta_owner_objective = P11_V4_DELTA_OWNER_OBJECTIVE
        self.register_buffer(
            "delta_owner_source_token_ids",
            torch.tensor(
                [self.plan_codec._tid(token) for token in SOURCE_SLOT_TOKENS],
                dtype=torch.long,
            ),
            persistent=False,
        )

        self.scene_sketch_codec = SceneSketchCodec(self.plan_codec)
        self.delta_sketch_codec = DeltaSceneSketchCodec(
            self.plan_codec, self.patch_codec
        )
        discrete_supervision = dict(config.get("discrete_supervision") or {})
        if (
            discrete_supervision.get("contract")
            != P11_V4_DISCRETE_BOUNDARY_CONTRACT
        ):
            raise ValueError(
                "P11-v4 discrete supervision requires "
                f"{P11_V4_DISCRETE_BOUNDARY_CONTRACT!r}"
            )
        self.text_end_loss_weight = float(
            discrete_supervision.get("text_end_weight", 0.0)
        )
        self.scene_eos_loss_weight = float(
            discrete_supervision.get("scene_eos_weight", 0.0)
        )
        for name, value in (
            ("text_end_weight", self.text_end_loss_weight),
            ("scene_eos_weight", self.scene_eos_loss_weight),
        ):
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(
                    f"P11-v4 discrete supervision {name} must be finite and >=1"
                )
        understanding_inventory = dict(
            discrete_supervision.get("understanding_inventory") or {}
        )
        expected_inventory = {
            "contract": P11_V4_U_INVENTORY_CONTRACT,
            "objective": P11_V4_U_INVENTORY_OBJECTIVE,
            "authority": "scene_sketch_autoregressive_logits_v1",
            "normalization": "weighted_mean_of_field_ce_v1",
            "fields": ["source_count", "room", "kind"],
        }
        for name, value in expected_inventory.items():
            if understanding_inventory.get(name) != value:
                raise ValueError(
                    "P11-v4 understanding inventory supervision "
                    f"{name}={understanding_inventory.get(name)!r}, "
                    f"expected {value!r}"
                )
        self.u_inventory_loss_weight = float(
            understanding_inventory.get("loss_weight", 0.0)
        )
        self.u_inventory_field_weights = {
            "source_count": float(
                understanding_inventory.get("source_count_weight", 0.0)
            ),
            "room": float(understanding_inventory.get("room_weight", 0.0)),
            "kind": float(understanding_inventory.get("kind_weight", 0.0)),
        }
        for name, value in {
            "loss_weight": self.u_inventory_loss_weight,
            **{
                f"{field}_weight": weight
                for field, weight in self.u_inventory_field_weights.items()
            },
        }.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "P11-v4 understanding inventory supervision "
                    f"{name} must be positive and finite"
                )
        self.u_inventory_contract = P11_V4_U_INVENTORY_CONTRACT
        self.scene_text_end_id = self.plan_codec._tid("<text_end>")
        self.scene_eos_id = self.scene_sketch_codec.eos_id
        self.register_buffer(
            "scene_room_token_ids",
            torch.tensor(
                [self.plan_codec._tid(token) for token in ROOM_TOKENS.values()],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "scene_source_count_token_ids",
            torch.tensor(
                [self.plan_codec._tid(token) for token in SOURCE_COUNT_TOKENS],
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "scene_kind_token_ids",
            torch.tensor(
                [self.plan_codec._tid(token) for token in KIND_TOKENS.values()],
                dtype=torch.long,
            ),
            persistent=False,
        )
        thought = dict(config.get("thought") or {})
        self.execution_reasoner = SketchFirstExecutionReasoner(
            thought, output_dim=self.hidden_dim
        )
        if not self.sct_enabled:
            raise ValueError("P11-v4 requires the true two-stage SCT backbone")
        control_direction = dict(config.get("control_direction") or {})
        expected_control_direction = {
            "contract": CONTROL_DIRECTION_CONTRACT,
            "authority": "delta_sketch_operation_specific_token_v1",
            "operations": list(CONTROL_DIRECTION_OPERATIONS),
            "decode": "exact_negative_positive_lookup_v1",
            "continuous_parallel_head": False,
            "conditions_delta_thought": True,
        }
        if control_direction != expected_control_direction:
            raise ValueError(
                "P11-v4 control_direction must be the exact P10 token-forcing contract"
            )
        self.control_direction_contract = CONTROL_DIRECTION_CONTRACT
        self.transfusion_cot_enabled = True
        self.transfusion_cot_arm = self.execution_reasoner.arm

        self.discrete_boundary = nn.Parameter(torch.empty(2, self.hidden_dim))
        self.v4_thought_boundary = nn.Parameter(torch.empty(2, self.hidden_dim))
        self.lexical_boundary = nn.Parameter(torch.empty(2, self.hidden_dim))
        self.trace_type_embedding = nn.Parameter(torch.empty(2, self.hidden_dim))
        for value in (
            self.discrete_boundary,
            self.v4_thought_boundary,
            self.lexical_boundary,
            self.trace_type_embedding,
        ):
            nn.init.normal_(value, std=0.02)

        self.sketch_max_tokens = int(config.get("scene_sketch_max_tokens", 512))
        self.delta_sketch_max_tokens = int(config.get("delta_sketch_max_tokens", 5))
        if not 16 <= self.sketch_max_tokens <= 768:
            raise ValueError("P11-v4 SceneSketch ceiling must be within [16,768]")
        if self.delta_sketch_max_tokens != 5:
            raise ValueError("P11-v4 DeltaSketch ceiling must be five tokens")
        lexical = dict(config.get("lexical_evidence") or {})
        self.lexical_evidence_required = bool(lexical.get("required", False))
        self.lexical_max_tokens = int(lexical.get("max_tokens", 128))
        if not 8 <= self.lexical_max_tokens <= 256:
            raise ValueError("P11-v4 lexical ceiling must be within [8,256]")
        self.lexical_injection_policy = str(
            lexical.get("injection_policy") or "disabled_v1"
        )
        if self.lexical_injection_policy not in {
            "disabled_v1",
            P11_LEXICAL_TOKEN_CONCAT_POLICY,
            P11_LEXICAL_ASSEMBLER_POLICY,
        }:
            raise ValueError("P11-v4 lexical injection policy is unsupported")

        sketch_mask = torch.zeros(self.plan_codec.vocab_size, dtype=torch.bool)
        sketch_tokens = {
            "<plan_bos>",
            "<plan_eos>",
            "<text_begin>",
            "<text_end>",
            "<room>",
            "<num_sources>",
            "<source_begin>",
            "<source_end>",
            "<kind>",
            "<description>",
            "<speaker_description>",
            "<transcript>",
        }
        sketch_tokens.update(
            token
            for token in self.plan_codec.token_to_id
            if token.startswith("<room_")
            or token.startswith("<num_sources_")
            or token.startswith("<source_slot_")
            or token.startswith("<kind_")
        )
        sketch_mask[
            [self.plan_codec._tid(token) for token in sketch_tokens]
        ] = True
        sketch_mask[list(self.plan_codec.text_ids)] = True

        delta_mask = torch.zeros_like(sketch_mask)
        delta_ids = set(self.patch_codec.token_to_id.values())
        # Operation-specific endpoint tokens are categorical P10 controls, not
        # free numeric values. Frame ids remain absent, so retime stays in the
        # continuous DeltaThought.
        delta_ids.update(
            self.plan_codec._tid(token)
            for token in self.plan_codec.token_to_id
            if token.startswith("<room_") or token.startswith("<source_slot_")
        )
        delta_mask[list(delta_ids)] = True
        self.register_buffer("scene_sketch_vocab_mask", sketch_mask, persistent=False)
        self.register_buffer("delta_sketch_vocab_mask", delta_mask, persistent=False)
        self.register_buffer(
            "control_direction_token_ids",
            torch.tensor(
                [
                    self.patch_codec.token_to_id[token]
                    for operation in CONTROL_DIRECTION_OPERATIONS
                    for token in CONTROL_DIRECTION_TOKENS[operation].values()
                ],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _lexical_ids(self, value: Mapping[str, Any] | Tensor) -> Tensor:
        if isinstance(value, Mapping):
            ids = torch.as_tensor(value["input_ids"], dtype=torch.long).flatten()
            mask = torch.as_tensor(value["attention_mask"], dtype=torch.bool).flatten()
            if ids.shape != mask.shape or ids.numel() != self.lexical_max_tokens:
                raise ValueError("P11-v4 lexical tokens violate their fixed ceiling")
            length = int(mask.sum())
            if length <= 0 or not torch.equal(
                mask, torch.arange(mask.numel(), device=mask.device) < length
            ):
                raise ValueError("P11-v4 lexical mask must be one non-empty prefix")
            return ids[:length]
        ids = torch.as_tensor(value, dtype=torch.long).flatten()
        if not 1 <= ids.numel() <= self.lexical_max_tokens:
            raise ValueError("P11-v4 lexical token count is invalid")
        return ids

    @staticmethod
    def _lexical_authority(value: Mapping[str, Any] | Tensor) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(
                "deterministic lexical assembly requires structured ASR evidence"
            )
        return parse_reliable_lexical_authority(value)

    def _v4_context_embeddings(
        self,
        *,
        task: P11Task,
        prompt: str | Mapping[str, Any],
        input_foa: Tensor | None,
        input_valid_mask: Tensor | None,
        input_semantic: Tensor | None,
        input_plan: Mapping[str, Tensor] | Tensor | None,
        input_lexical: Mapping[str, Any] | Tensor | None,
        device: torch.device,
        qwen_dtype: torch.dtype,
    ) -> tuple[Tensor, dict[str, int]]:
        context, metrics = self._context_embeddings(
            task=task,
            prompt=prompt,
            input_foa=input_foa,
            input_valid_mask=input_valid_mask,
            input_semantic=input_semantic,
            input_plan=input_plan,
            device=device,
            qwen_dtype=qwen_dtype,
        )
        lexical_tasks = (
            {P11Task.UNDERSTANDING, P11Task.EDITING}
            if self.audio_aware_editing
            else {P11Task.UNDERSTANDING}
        )
        if input_lexical is not None and task not in lexical_tasks:
            raise ValueError(
                "speech lexical evidence is legal only for audio-observation tasks"
            )
        if (
            task in lexical_tasks
            and self.lexical_evidence_required
            and input_lexical is None
        ):
            raise ValueError("canonical P11 audio observation lacks lexical evidence")
        metrics["input_lexical_tokens"] = 0
        metrics["input_lexical_context_tokens"] = 0
        metrics["lexical_authority_present"] = 0
        if input_lexical is not None:
            ids = self._lexical_ids(input_lexical).to(device)
            metrics["input_lexical_tokens"] = int(ids.numel())
            if self.lexical_injection_policy == P11_LEXICAL_TOKEN_CONCAT_POLICY:
                with torch.no_grad():
                    embeddings = self.qwen_backbone.embed_tokens(ids)
                context = torch.cat(
                    [
                        context,
                        self.lexical_boundary[0:1].to(qwen_dtype),
                        embeddings.to(qwen_dtype),
                        self.lexical_boundary[1:2].to(qwen_dtype),
                    ],
                    dim=0,
                )
                metrics["input_lexical_context_tokens"] = int(ids.numel())
            elif self.lexical_injection_policy == P11_LEXICAL_ASSEMBLER_POLICY:
                self._lexical_authority(input_lexical)
                metrics["lexical_authority_present"] = 1
            else:
                raise ValueError(
                    "lexical evidence was supplied while lexical input is disabled"
                )
        return context, metrics

    def _run_execution_slots(
        self, contexts: Sequence[Tensor], slot_tokens: Tensor, tasks: Sequence[P11Task]
    ) -> Tensor:
        if slot_tokens.ndim != 3 or tuple(slot_tokens.shape[1:]) != (
            EXECUTION_SLOT_COUNT,
            self.hidden_dim,
        ):
            raise ValueError(
                f"P11-v4 thought tokens must be [B,{EXECUTION_SLOT_COUNT},H]"
            )
        qwen_dtype = self.qwen_backbone.embed_tokens.weight.dtype
        rows: list[Tensor] = []
        starts: list[int] = []
        for index, (context, task) in enumerate(zip(contexts, tasks)):
            starts.append(int(context.shape[0]) + 1)
            thought_type = 1 if task is P11Task.EDITING else 0
            rows.append(
                torch.cat(
                    [
                        context,
                        self.v4_thought_boundary[0:1].to(qwen_dtype)
                        + self.trace_type_embedding[thought_type : thought_type + 1].to(qwen_dtype),
                        slot_tokens[index].to(qwen_dtype),
                        self.v4_thought_boundary[1:2].to(qwen_dtype),
                    ],
                    dim=0,
                )
            )
        padded = pad_sequence(rows, batch_first=True)
        if padded.shape[1] > self.sequence_length:
            raise ValueError(
                f"P11-v4 thought sequence needs {padded.shape[1]} > {self.sequence_length} tokens"
            )
        lengths = torch.tensor([row.shape[0] for row in rows], device=padded.device)
        attention = torch.arange(padded.shape[1], device=padded.device)[None] < lengths[:, None]
        output = self._run_sct_generation_backbone(
            padded, attention, use_cache=False
        )
        return torch.stack(
            [
                output.last_hidden_state[index, start : start + EXECUTION_SLOT_COUNT].float()
                for index, start in enumerate(starts)
            ]
        )

    def _teacher_discrete_ce(
        self,
        contexts: Sequence[Tensor],
        targets: Sequence[Mapping[str, Tensor] | Tensor],
        tasks: Sequence[P11Task],
        input_source_masks: Sequence[Tensor],
        *,
        loss_group_weights: Optional[Mapping[int, float]],
    ) -> tuple[
        Tensor,
        Tensor,
        list[Tensor],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        dict[str, Tensor],
    ]:
        if len(input_source_masks) != len(contexts):
            raise ValueError("P11-v4 source masks do not align with contexts")
        device = self.plan_embedding.weight.device
        qwen_dtype = self.qwen_backbone.embed_tokens.weight.dtype
        labels: list[Tensor] = []
        groups: list[Tensor] = []
        rows: list[Tensor] = []
        starts: list[int] = []
        for context, target, task in zip(contexts, targets, tasks):
            ids = self._ids(target).to(device)
            expected_bos = (
                self.delta_sketch_codec.bos_id
                if task is P11Task.EDITING
                else self.scene_sketch_codec.bos_id
            )
            expected_eos = (
                self.delta_sketch_codec.eos_id
                if task is P11Task.EDITING
                else self.scene_sketch_codec.eos_id
            )
            if int(ids[0]) != expected_bos or int(ids[-1]) != expected_eos:
                raise ValueError("P11-v4 discrete target uses the wrong grammar")
            ceiling = (
                self.delta_sketch_max_tokens
                if task is P11Task.EDITING
                else self.sketch_max_tokens
            )
            if ids.numel() > ceiling:
                raise ValueError("P11-v4 discrete target exceeds its ceiling")
            prefix = torch.cat(
                [context, self.discrete_boundary[0:1].to(qwen_dtype)], dim=0
            )
            starts.append(int(prefix.shape[0]))
            rows.append(
                torch.cat(
                    [
                        prefix,
                        self.output_start.view(1, -1).to(qwen_dtype),
                        self.plan_embedding(ids[:-1]).to(qwen_dtype),
                    ],
                    dim=0,
                )
            )
            labels.append(ids)
            if isinstance(target, Mapping) and target.get("loss_group_ids") is not None:
                one_groups = torch.as_tensor(
                    target["loss_group_ids"], device=device, dtype=torch.long
                ).flatten()
            else:
                one_groups = torch.ones_like(ids)
            if one_groups.shape != ids.shape:
                raise ValueError("P11-v4 discrete loss groups do not align")
            groups.append(one_groups)

        padded = pad_sequence(rows, batch_first=True)
        if padded.shape[1] > self.sequence_length:
            raise ValueError(
                f"P11-v4 discrete sequence needs {padded.shape[1]} > {self.sequence_length} tokens"
            )
        lengths = torch.tensor([row.shape[0] for row in rows], device=device)
        attention = torch.arange(padded.shape[1], device=device)[None] < lengths[:, None]
        output = self._run_sct_understanding_backbone(
            padded,
            attention,
            use_cache=False,
        )
        logits_rows: list[Tensor] = []
        for index, (start, ids, task) in enumerate(zip(starts, labels, tasks)):
            hidden = output.last_hidden_state[index, start : start + ids.numel()].float()
            logits = F.linear(hidden, self.plan_embedding.weight.float(), self.output_bias)
            mask = (
                self.delta_sketch_vocab_mask
                if task is P11Task.EDITING
                else self.scene_sketch_vocab_mask
            )
            logits_rows.append(
                logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
            )

        owner_token_to_slot = {
            int(token_id): slot
            for slot, token_id in enumerate(
                self.delta_owner_source_token_ids.detach().cpu().tolist()
            )
        }
        owner_losses: list[Tensor] = []
        owner_correct: list[Tensor] = []
        owner_active: list[bool] = []
        for logits, ids, task, source_mask in zip(
            logits_rows, labels, tasks, input_source_masks
        ):
            legal_source_mask = torch.as_tensor(
                source_mask, device=device, dtype=torch.bool
            ).flatten()
            if tuple(legal_source_mask.shape) != (len(SOURCE_SLOT_TOKENS),):
                raise ValueError("P11-v4 input source mask must be [4]")
            owner_positions = [
                position
                for position, token in enumerate(ids.detach().cpu().tolist())
                if int(token) in owner_token_to_slot
            ]
            if task is P11Task.EDITING and len(owner_positions) > 1:
                raise ValueError("DeltaSketch contains more than one source owner")
            owner_position = owner_positions[0] if owner_positions else None
            target_slot = (
                None
                if owner_position is None
                else owner_token_to_slot[int(ids[owner_position])]
            )
            active = task is P11Task.EDITING and target_slot is not None
            owner_active.append(active)
            if active:
                source_logits = logits[owner_position].index_select(
                    0, self.delta_owner_source_token_ids
                )
                loss, correct = _delta_owner_objective(
                    source_logits,
                    target_slot,
                    legal_source_mask,
                    margin=self.delta_owner_margin,
                    margin_weight=self.delta_owner_margin_weight,
                )
            else:
                # Keep the metric on the same decoder graph without creating
                # a second owner head or granting any target-only information.
                loss = logits[0, self.delta_owner_source_token_ids[0]] * 0.0
                correct = loss.detach()
            owner_losses.append(loss)
            owner_correct.append(correct)
        padded_logits = pad_sequence(logits_rows, batch_first=True)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        padded_groups = pad_sequence(groups, batch_first=True, padding_value=0)
        token_loss = F.cross_entropy(
            padded_logits.transpose(1, 2),
            padded_labels,
            ignore_index=-100,
            reduction="none",
        )
        valid = padded_labels.ne(-100)
        weights = valid.to(token_loss)
        if loss_group_weights:
            weights.zero_()
            for group_id, weight in loss_group_weights.items():
                weights = torch.where(
                    padded_groups.eq(int(group_id)),
                    token_loss.new_tensor(float(weight)),
                    weights,
                )
            weights *= valid
        scene_rows = torch.tensor(
            [task is not P11Task.EDITING for task in tasks],
            device=device,
            dtype=torch.bool,
        )[:, None]
        text_end_mask = (
            valid & scene_rows & padded_labels.eq(self.scene_text_end_id)
        )
        scene_eos_mask = (
            valid & scene_rows & padded_labels.eq(self.scene_eos_id)
        )
        direction_mask = valid & torch.isin(
            padded_labels, self.control_direction_token_ids
        )
        text_end_count = text_end_mask.sum()
        scene_eos_count = scene_eos_mask.sum()
        text_end_ce = (
            token_loss[text_end_mask].mean()
            if bool(text_end_mask.any())
            else token_loss.new_zeros(())
        )
        scene_eos_ce = (
            token_loss[scene_eos_mask].mean()
            if bool(scene_eos_mask.any())
            else token_loss.new_zeros(())
        )
        control_direction_count = direction_mask.sum()
        control_direction_ce = (
            token_loss[direction_mask].mean()
            if bool(direction_mask.any())
            else token_loss.new_zeros(())
        )
        direction_correct: list[Tensor] = []
        operation_id_to_direction_ids = {
            self.patch_codec.token_to_id[OPERATION_TOKENS[operation]]: torch.tensor(
                [
                    self.patch_codec.token_to_id[token]
                    for token in CONTROL_DIRECTION_TOKENS[operation].values()
                ],
                device=device,
                dtype=torch.long,
            )
            for operation in CONTROL_DIRECTION_OPERATIONS
        }
        for logits, ids in zip(logits_rows, labels):
            if int(ids.numel()) != 5 or int(ids[1]) not in operation_id_to_direction_ids:
                continue
            candidates = operation_id_to_direction_ids[int(ids[1])]
            predicted = candidates[logits[3].index_select(0, candidates).argmax()]
            direction_correct.append(predicted.eq(ids[3]).to(token_loss.dtype))
        if len(direction_correct) != int(control_direction_count):
            raise RuntimeError("P11-v4 control-direction token accounting changed")
        control_direction_accuracy = (
            torch.stack(direction_correct).mean()
            if direction_correct
            else token_loss.new_zeros(())
        )
        understanding_rows = torch.tensor(
            [task is P11Task.UNDERSTANDING for task in tasks],
            device=device,
            dtype=torch.bool,
        )[:, None]
        inventory_specs = {
            "source_count": self.scene_source_count_token_ids,
            "room": self.scene_room_token_ids,
            "kind": self.scene_kind_token_ids,
        }
        inventory_metrics: dict[str, Tensor] = {}
        weighted_inventory_losses: list[Tensor] = []
        active_inventory_weights: list[float] = []
        for field, candidates in inventory_specs.items():
            field_mask = (
                valid
                & understanding_rows
                & torch.isin(padded_labels, candidates)
            )
            field_loss, field_count, field_accuracy = (
                _finite_field_token_objective(
                    padded_logits,
                    padded_labels,
                    field_mask,
                    candidates,
                )
            )
            inventory_metrics[f"u_{field}_ce"] = field_loss
            inventory_metrics[f"u_{field}_count"] = field_count
            inventory_metrics[f"u_{field}_accuracy"] = field_accuracy
            if bool(field_mask.any()):
                weight = self.u_inventory_field_weights[field]
                weighted_inventory_losses.append(field_loss * weight)
                active_inventory_weights.append(weight)
        if weighted_inventory_losses:
            inventory_loss = torch.stack(weighted_inventory_losses).sum() / sum(
                active_inventory_weights
            )
        else:
            inventory_loss = padded_logits.sum() * 0.0
        inventory_metrics["u_inventory_loss"] = inventory_loss
        inventory_metrics["u_inventory_rows"] = understanding_rows.sum()
        weights = torch.where(
            text_end_mask,
            weights * self.text_end_loss_weight,
            weights,
        )
        weights = torch.where(
            scene_eos_mask,
            weights * self.scene_eos_loss_weight,
            weights,
        )
        per_row = (token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        owner_per_row = torch.stack(owner_losses)
        owner_active_mask = torch.tensor(
            owner_active, device=device, dtype=torch.bool
        )
        owner_row_count = owner_active_mask.sum()
        owner_denominator = owner_row_count.clamp_min(1).to(owner_per_row.dtype)
        owner_active_loss = (
            owner_per_row * owner_active_mask.to(owner_per_row.dtype)
        ).sum() / owner_denominator
        owner_accuracy = (
            torch.stack(owner_correct)
            * owner_active_mask.to(owner_per_row.dtype)
        ).sum() / owner_denominator
        return (
            per_row.mean(),
            per_row,
            labels,
            owner_per_row,
            owner_active_loss,
            owner_row_count,
            owner_accuracy,
            control_direction_ce,
            control_direction_count,
            control_direction_accuracy,
            text_end_ce,
            text_end_count,
            scene_eos_ce,
            scene_eos_count,
            inventory_metrics,
        )

    def forward_transfusion_cot(
        self,
        prompts: Sequence[Mapping[str, Any]],
        discrete_targets: Sequence[Mapping[str, Tensor] | Tensor],
        *,
        tasks: Sequence[P11Task | str],
        input_foa: Sequence[Optional[Tensor]],
        input_valid_masks: Sequence[Optional[Tensor]],
        input_semantic: Sequence[Optional[Tensor]],
        input_plans: Sequence[Optional[Mapping[str, Tensor] | Tensor]],
        input_lexical: Sequence[Optional[Mapping[str, Any] | Tensor]],
        target_execution_cores: Sequence[Tensor],
        input_execution_cores: Sequence[Tensor],
        delta_execution_cores: Sequence[Tensor],
        target_source_masks: Sequence[Tensor],
        input_source_masks: Sequence[Tensor],
        loss_group_weights: Optional[Mapping[int, float]] = None,
        thought_loss_weights: Optional[Mapping[str, float]] = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        batch = len(prompts)
        fields = (
            discrete_targets,
            tasks,
            input_foa,
            input_valid_masks,
            input_semantic,
            input_plans,
            input_lexical,
            target_execution_cores,
            input_execution_cores,
            delta_execution_cores,
            target_source_masks,
            input_source_masks,
        )
        if batch <= 0 or any(len(value) != batch for value in fields):
            raise ValueError("P11-v4 batch fields must be non-empty and aligned")
        normalized = [normalize_p11_task(task) for task in tasks]
        device = self.plan_embedding.weight.device
        qwen_dtype = self._ensure_qwen_device(device)
        contexts: list[Tensor] = []
        context_metrics: list[dict[str, int]] = []
        for prompt, task, audio, mask, semantic, plan, lexical in zip(
            prompts,
            normalized,
            input_foa,
            input_valid_masks,
            input_semantic,
            input_plans,
            input_lexical,
        ):
            context, metrics = self._v4_context_embeddings(
                task=task,
                prompt=prompt,
                input_foa=audio,
                input_valid_mask=mask,
                input_semantic=semantic,
                input_plan=plan,
                input_lexical=lexical,
                device=device,
                qwen_dtype=qwen_dtype,
            )
            contexts.append(context)
            context_metrics.append(metrics)

        (
            discrete_ce,
            ce_per_row,
            target_ids,
            delta_owner_per_row,
            delta_owner_active_loss,
            delta_owner_row_count,
            delta_owner_accuracy,
            control_direction_ce,
            control_direction_count,
            control_direction_accuracy,
            text_end_ce,
            text_end_count,
            scene_eos_ce,
            scene_eos_count,
            inventory_metrics,
        ) = self._teacher_discrete_ce(
            contexts,
            discrete_targets,
            normalized,
            input_source_masks,
            loss_group_weights=loss_group_weights,
        )
        thought_contexts = [
            torch.cat(
                [
                    context,
                    self.discrete_boundary[0:1].to(qwen_dtype),
                    self.plan_embedding(ids).to(qwen_dtype),
                    self.discrete_boundary[1:2].to(qwen_dtype),
                ],
                dim=0,
            )
            for context, ids in zip(contexts, target_ids)
        ]
        task_ids = torch.tensor(
            [list(P11Task).index(task) for task in normalized],
            device=device,
            dtype=torch.long,
        )
        def stack(values: Sequence[Tensor], dtype: torch.dtype) -> Tensor:
            return torch.stack(
                [torch.as_tensor(value, device=device, dtype=dtype) for value in values]
            )

        thought = self.execution_reasoner.forward_train(
            run_slots=lambda value: self._run_execution_slots(
                thought_contexts, value, normalized
            ),
            task_ids=task_ids,
            target_core=stack(target_execution_cores, torch.float32),
            input_core=stack(input_execution_cores, torch.float32),
            delta_core=stack(delta_execution_cores, torch.float32),
            target_source_mask=stack(target_source_masks, torch.bool),
            input_source_mask=stack(input_source_masks, torch.bool),
        )
        weights = {
            "flow": 1.0,
            "solve": 1.0,
            "locality": 0.25,
            "owner": 0.1,
            **{
                str(key): float(value)
                for key, value in (thought_loss_weights or {}).items()
            },
        }
        total = discrete_ce
        total = total + self.u_inventory_loss_weight * inventory_metrics[
            "u_inventory_loss"
        ]
        total = total + weights["flow"] * thought.flow_loss_per_row.mean()
        total = total + weights["solve"] * thought.solve_loss_per_row.mean()
        total = total + weights["locality"] * thought.locality_loss_per_row.mean()
        total = total + weights["owner"] * delta_owner_active_loss
        total = total + self._trainable_graph_anchor(total)
        return total, {
            "total_loss": total.detach(),
            "discrete_ce": discrete_ce.detach(),
            "discrete_ce_per_row": ce_per_row.detach(),
            "flow_per_row": thought.flow_loss_per_row.detach(),
            "solve_per_row": thought.solve_loss_per_row.detach(),
            "locality_per_row": thought.locality_loss_per_row.detach(),
            "owner_per_row": delta_owner_per_row.detach(),
            "owner_active_loss": delta_owner_active_loss.detach(),
            "owner_row_count": delta_owner_row_count.detach(),
            "owner_accuracy": delta_owner_accuracy.detach(),
            "text_end_ce": text_end_ce.detach(),
            "text_end_count": text_end_count.detach(),
            "scene_eos_ce": scene_eos_ce.detach(),
            "scene_eos_count": scene_eos_count.detach(),
            **{
                key: value.detach()
                for key, value in inventory_metrics.items()
            },
            "control_direction_ce": control_direction_ce.detach(),
            "control_direction_count": control_direction_count.detach(),
            "control_direction_accuracy": control_direction_accuracy.detach(),
            "discrete_tokens": total.new_tensor(sum(ids.numel() for ids in target_ids)),
            "thought_tokens": total.new_tensor(batch * EXECUTION_SLOT_COUNT),
            "prompt_tokens": total.new_tensor(
                sum(value["prompt_tokens"] for value in context_metrics)
            ),
            "input_audio_tokens": total.new_tensor(
                sum(value["input_audio_tokens"] for value in context_metrics)
            ),
            "input_semantic_tokens": total.new_tensor(
                sum(value["input_semantic_tokens"] for value in context_metrics)
            ),
            "input_lexical_tokens": total.new_tensor(
                sum(value["input_lexical_tokens"] for value in context_metrics)
            ),
            "input_plan_tokens": total.new_tensor(
                sum(value["input_plan_tokens"] for value in context_metrics)
            ),
        }

    @torch.no_grad()
    def _decode_v4_discrete(
        self,
        context: Tensor,
        *,
        task: P11Task,
        input_sceneplan: Mapping[str, Any] | None,
        temperature: float,
        sampling_seed: int | None,
        discrete_decode_mode: str,
    ) -> tuple[Tensor, dict[str, Any]]:
        if discrete_decode_mode not in {"cached", "prefix_recompute"}:
            raise ValueError(
                "P11-v4 discrete_decode_mode must be 'cached' or 'prefix_recompute'"
        )
        device = context.device
        sampling_generator = None
        if sampling_seed is not None:
            sampling_seed = int(sampling_seed)
            if not 0 <= sampling_seed < 2**63:
                raise ValueError(
                    "P11-v4 discrete sampling seed must be within [0,2**63)"
                )
            sampling_generator = torch.Generator(device=device)
            sampling_generator.manual_seed(sampling_seed)
        qwen_dtype = self.qwen_backbone.embed_tokens.weight.dtype
        codec = (
            self.delta_sketch_codec
            if task is P11Task.EDITING
            else self.scene_sketch_codec
        )
        ceiling = (
            self.delta_sketch_max_tokens
            if task is P11Task.EDITING
            else self.sketch_max_tokens
        )

        def allowed_next(values: Sequence[int]) -> set[int]:
            if task is P11Task.EDITING:
                if input_sceneplan is None:
                    raise ValueError("P11-v4 Editing requires input ScenePlan")
                editing_kwargs: dict[str, Any] = {
                    "input_sceneplan": input_sceneplan
                }
                if self.audio_aware_editing:
                    editing_kwargs["max_text_tokens"] = (
                        self.plan_text_field_max_tokens
                    )
                return codec.allowed_next_ids(values, **editing_kwargs)
            return codec.allowed_next_ids(
                values,
                max_text_tokens=self.plan_text_field_max_tokens,
            )

        text_end_id = self.plan_codec._tid("<text_end>")

        def shortest_valid_completion(values: Sequence[int]) -> list[int]:
            """Construct a deterministic short grammar suffix.

            Text-token loops are collapsed to one representative piece and
            closed immediately.  Structural branches are deterministic at the
            point this guard is used in practice; when more than one remains,
            prefer a non-speech kind and the smallest legal source count so the
            suffix stays conservative.  This is validity forcing, not target
            forcing: it has no access to a reference ScenePlan.
            """

            suffix: list[int] = []
            for _ in range(64):
                allowed = allowed_next([*values, *suffix])
                if not allowed:
                    return suffix
                if text_end_id in allowed:
                    token = text_end_id
                else:
                    text_candidates = sorted(allowed & self.plan_codec.text_ids)
                    grammar_candidates = sorted(allowed - self.plan_codec.text_ids)
                    if text_candidates:
                        token = text_candidates[0]
                    else:
                        speech_id = kind_token_ids.get("speech")
                        non_speech = [
                            value
                            for value in grammar_candidates
                            if value != speech_id
                        ]
                        token = (non_speech or grammar_candidates)[0]
                suffix.append(int(token))
            raise RuntimeError("P11 grammar completion exceeds 64 tokens")

        prefix = torch.cat(
            [
                context,
                self.discrete_boundary[0:1].to(qwen_dtype),
                self.output_start.view(1, -1).to(qwen_dtype),
            ],
            dim=0,
        ).unsqueeze(0)
        attention = torch.ones((1, prefix.shape[1]), device=device, dtype=torch.bool)
        cache = None
        hidden = None
        if discrete_decode_mode == "cached":
            output = self._run_sct_understanding_backbone(
                prefix,
                attention,
                use_cache=True,
            )
            cache = output.past_key_values
            hidden = output.last_hidden_state[0, -1].float()
        generated: list[int] = []
        grammar_interventions = 0
        budget_forced_tokens = 0
        budget_forcing_started_at: int | None = None
        source_kind_scores: list[dict[str, Any]] = []
        kind_token_ids = {
            kind: self.plan_codec._tid(token)
            for kind, token in KIND_TOKENS.items()
        }
        source_slot_by_token = {
            self.plan_codec._tid(token): f"source_{slot}"
            for slot, token in enumerate(SOURCE_SLOT_TOKENS)
        }
        kind_marker_id = self.plan_codec._tid("<kind>")
        output_ceiling = min(
            int(ceiling), max(0, self.sequence_length - int(prefix.shape[1]))
        )
        if output_ceiling <= 0:
            raise RuntimeError("P11 context leaves no room for discrete output")
        for _ in range(output_ceiling):
            if discrete_decode_mode == "prefix_recompute":
                step_embeddings = prefix
                if generated:
                    step_embeddings = torch.cat(
                        [
                            prefix,
                            self.plan_embedding(
                                torch.tensor(
                                    [generated], device=device, dtype=torch.long
                                )
                            ).to(qwen_dtype),
                        ],
                        dim=1,
                    )
                if step_embeddings.shape[1] > self.sequence_length:
                    break
                attention = torch.ones(
                    (1, step_embeddings.shape[1]),
                    device=device,
                    dtype=torch.bool,
                )
                output = self._run_sct_understanding_backbone(
                    step_embeddings,
                    attention,
                    use_cache=False,
                )
                hidden = output.last_hidden_state[0, -1].float()
            if hidden is None:
                raise RuntimeError(
                    "P11-v4 discrete decoder did not produce a hidden state"
                )
            allowed = allowed_next(generated)
            if not allowed:
                break
            remaining = output_ceiling - len(generated)
            if remaining <= 64:
                completion = shortest_valid_completion(generated)
                if len(completion) >= remaining:
                    # Reserve exactly enough output positions to close all
                    # currently open text/structural fields and emit EOS.
                    # The intervention is surfaced in diagnostics and should
                    # trend to zero with rollout-aware training.
                    allowed = {int(completion[0])}
                    budget_forced_tokens += 1
                    if budget_forcing_started_at is None:
                        budget_forcing_started_at = len(generated)
            logits = F.linear(
                hidden, self.plan_embedding.weight.float(), self.output_bias
            )
            if (
                task is not P11Task.EDITING
                and generated
                and generated[-1] == kind_marker_id
            ):
                if len(generated) < 2 or generated[-2] not in source_slot_by_token:
                    raise RuntimeError("SceneSketch kind step lost its source owner")
                ordered_kinds = ("music", "sound", "speech")
                kind_logits = torch.stack(
                    [logits[kind_token_ids[kind]] for kind in ordered_kinds]
                )
                probabilities = torch.softmax(kind_logits, dim=0)
                speech_index = ordered_kinds.index("speech")
                acoustic_max = torch.stack(
                    [
                        kind_logits[ordered_kinds.index("music")],
                        kind_logits[ordered_kinds.index("sound")],
                    ]
                ).max()
                source_kind_scores.append(
                    {
                        "source_id": source_slot_by_token[generated[-2]],
                        "music_probability": float(
                            probabilities[ordered_kinds.index("music")]
                        ),
                        "sound_probability": float(
                            probabilities[ordered_kinds.index("sound")]
                        ),
                        "speech_probability": float(probabilities[speech_index]),
                        "speech_margin": float(
                            kind_logits[speech_index] - acoustic_max
                        ),
                    }
                )
            valid = torch.tensor(sorted(allowed), device=device, dtype=torch.long)
            all_mask = (
                self.delta_sketch_vocab_mask
                if task is P11Task.EDITING
                else self.scene_sketch_vocab_mask
            )
            raw_ids = torch.nonzero(all_mask, as_tuple=False).flatten()
            raw_token = int(raw_ids[int(logits.index_select(0, raw_ids).argmax())])
            if raw_token not in allowed:
                grammar_interventions += 1
            candidates = logits.index_select(0, valid)
            if temperature <= 0.0 or valid.numel() == 1:
                selected = int(candidates.argmax())
            else:
                probabilities = torch.softmax(candidates / float(temperature), dim=-1)
                selected = int(
                    torch.multinomial(
                        probabilities,
                        1,
                        generator=sampling_generator,
                    )
                )
            token = int(valid[selected])
            generated.append(token)
            if token == codec.eos_id:
                break
            if discrete_decode_mode == "prefix_recompute":
                continue
            embedding = self.plan_embedding(
                torch.tensor([[token]], device=device)
            ).to(qwen_dtype)
            attention = torch.ones(
                (1, attention.shape[1] + 1), device=device, dtype=torch.bool
            )
            output = self._run_sct_understanding_backbone(
                embedding,
                attention,
                use_cache=True,
                past_key_values=cache,
            )
            cache = output.past_key_values
            hidden = output.last_hidden_state[0, -1].float()
        result = torch.tensor(generated, device=device, dtype=torch.long)
        terminated = bool(generated and generated[-1] == codec.eos_id)
        if not terminated:
            raise P11DiscreteDecodeError(
                "P11-v4 discrete sketch did not terminate", result
            )
        if task is not P11Task.EDITING:
            result = codec.canonicalize(result.detach().cpu())["input_ids"].to(device)
        else:
            codec.decode(result)
        return result, {
            "terminated": True,
            "grammar_interventions": grammar_interventions,
            "budget_forced_tokens": budget_forced_tokens,
            "budget_forcing_started_at": budget_forcing_started_at,
            "output_token_ceiling": output_ceiling,
            "generated_tokens": int(result.numel()),
            "source_kind_scores": source_kind_scores,
            "discrete_sampling_seed": sampling_seed,
            "discrete_sampling_rng": (
                "local_explicit_generator"
                if sampling_seed is not None
                else "process_global_generator"
            ),
            "discrete_decode_mode": discrete_decode_mode,
        }

    def _apply_reliable_lexical_authority(
        self,
        discrete_tokens: Tensor,
        *,
        task: P11Task,
        input_lexical: Mapping[str, Any] | Tensor | None,
        diagnostics: Mapping[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """Resolve reliable ASR inside SceneSketch before continuous thought."""

        result = {
            "lexical_injection_policy": self.lexical_injection_policy,
            "lexical_authority_applied": False,
            "lexical_authority_action": None,
            "lexical_authority_source_id": None,
            "lexical_authority_confidence": None,
            "lexical_authority_transcript_sha256": None,
        }
        if self.lexical_injection_policy != P11_LEXICAL_ASSEMBLER_POLICY:
            return discrete_tokens, result
        if input_lexical is None:
            return discrete_tokens, result
        if task is not P11Task.UNDERSTANDING:
            raise ValueError("lexical authority is legal only for Understanding")

        authority = self._lexical_authority(input_lexical)
        sketch = self.scene_sketch_codec.decode(discrete_tokens.detach().cpu())
        selected_source_id, action = select_reliable_lexical_source_owner(
            sketch,
            source_kind_scores=diagnostics.get("source_kind_scores", []),
            codec=self.plan_codec,
        )
        sketch = apply_reliable_lexical_authority_to_sketch(
            sketch,
            transcript=authority["transcript"],
            source_id=selected_source_id,
            codec=self.plan_codec,
        )
        forced = self.scene_sketch_codec.encode(
            sketch, max_tokens=self.sketch_max_tokens
        )["input_ids"].to(discrete_tokens.device)
        result.update(
            {
                "lexical_authority_applied": True,
                "lexical_authority_action": action,
                "lexical_authority_source_id": selected_source_id,
                "lexical_authority_confidence": float(authority["confidence"]),
                "lexical_authority_transcript_sha256": hashlib.sha256(
                    authority["transcript"].encode("utf-8")
                ).hexdigest(),
            }
        )
        return forced, result

    @staticmethod
    def _validate_v4_intervention(value: str | None) -> None:
        allowed = {
            None,
            "zero_thought",
            "shuffle_source_slots",
            "swap_source_1_2",
            "replace_one_control_field",
            "flip_control_direction",
            "flip_retime_delta",
        }
        if value not in allowed:
            raise ValueError("unsupported P11-v4 thought intervention")

    @staticmethod
    def _apply_v4_intervention(core: Tensor, value: str | None) -> Tensor:
        core = core.clone()
        if value == "zero_thought":
            core.zero_()
        elif value == "shuffle_source_slots":
            core[1:5] = core[1:5].roll(1, dims=0)
        elif value == "swap_source_1_2":
            core[[1, 2]] = core[[2, 1]]
        elif value == "replace_one_control_field":
            core[1, 5] = 1.0
            core[1, 6] = 0.0
        return core

    @staticmethod
    def _apply_control_direction_intervention(
        program: Mapping[str, Any], value: str | None
    ) -> dict[str, Any]:
        output = dict(program)
        if value == "flip_control_direction":
            direction = int(output.get("control_direction", 0))
            if direction not in {-1, 1}:
                raise ValueError(
                    "flip_control_direction requires a rotate/distance DeltaSketch"
                )
            output["control_direction"] = -direction
        return output

    @staticmethod
    def _apply_delta_thought_intervention(
        core: Tensor,
        program: Mapping[str, Any],
        value: str | None,
    ) -> Tensor:
        """Apply an owner-local counterfactual to continuous Editing state.

        Rotate/distance direction is categorical DeltaSketch authority and is
        handled by :meth:`_apply_control_direction_intervention`.  Retime is
        the one active P10 edit whose value remains continuous DeltaThought
        authority, so its causal intervention negates only the owning source's
        onset/offset delta.  No semantic or foreign-source field is touched.
        """

        output = core.clone()
        if value != "flip_retime_delta":
            return output
        if str(program.get("operation") or "") != "retime_source":
            raise ValueError("flip_retime_delta requires a retime DeltaSketch")
        source_id = str(program.get("source_id") or "")
        source_ids = tuple(f"source_{index}" for index in range(EXECUTION_SLOT_COUNT - 1))
        if source_id not in source_ids:
            raise ValueError("flip_retime_delta requires a legal source owner")
        source_index = source_ids.index(source_id) + 1
        onset_index = EXECUTION_FEATURE_NAMES.index("onset_frame_norm")
        offset_index = EXECUTION_FEATURE_NAMES.index("offset_frame_norm")
        output[source_index, onset_index] = -output[source_index, onset_index]
        output[source_index, offset_index] = -output[source_index, offset_index]
        return output

    @staticmethod
    def _v4_noise_sha256(noise: Tensor | None) -> str | None:
        if noise is None:
            return None
        payload = noise.detach().float().cpu().contiguous().numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _v4_token_sha256(tokens: Tensor) -> str:
        payload = (
            torch.as_tensor(tokens, dtype=torch.int64)
            .detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )
        return hashlib.sha256(payload).hexdigest()

    def _validate_v4_diagnostic_discrete_tokens(
        self,
        value: Mapping[str, Tensor] | Tensor,
        *,
        task: P11Task,
        device: torch.device,
    ) -> Tensor:
        """Validate an explicit teacher-context override for evaluator use.

        Normal inference must always decode its own SceneSketch/DeltaSketch.
        This narrowly named path exists only to measure the train/deploy
        exposure gap while holding evidence, Flow noise, and the checkpoint
        fixed.  It is fail-closed and cannot accept a non-canonical token span.
        """

        ids = self._ids(value).detach().to(device=device, dtype=torch.long)
        if task is P11Task.EDITING:
            self.delta_sketch_codec.decode(ids.cpu())
        else:
            canonical = self.scene_sketch_codec.canonicalize(ids.cpu())["input_ids"]
            if not torch.equal(ids.cpu(), canonical):
                raise ValueError(
                    "P11-v4 diagnostic SceneSketch override must be canonical"
                )
        return ids

    def _prepare_v4_decode(
        self,
        prompt: str | Mapping[str, Any],
        *,
        task: P11Task,
        input_foa: Tensor | None,
        input_valid_mask: Tensor | None,
        input_semantic: Tensor | None,
        input_lexical: Mapping[str, Any] | Tensor | None,
        input_sceneplan: Mapping[str, Any] | None,
        temperature: float,
        discrete_seed: int | None,
        discrete_decode_mode: str,
        diagnostic_discrete_tokens: Mapping[str, Tensor] | Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        dict[str, int],
        dict[str, Any],
    ]:
        input_plan_tokens = (
            None
            if input_sceneplan is None
            else self.plan_codec.encode(input_sceneplan)
        )
        device = self.plan_embedding.weight.device
        qwen_dtype = self._ensure_qwen_device(device)
        context, context_metrics = self._v4_context_embeddings(
            task=task,
            prompt=prompt,
            input_foa=input_foa,
            input_valid_mask=input_valid_mask,
            input_semantic=input_semantic,
            input_plan=input_plan_tokens,
            input_lexical=input_lexical,
            device=device,
            qwen_dtype=qwen_dtype,
        )
        discrete_tokens, discrete_diagnostics = self._decode_v4_discrete(
            context,
            task=task,
            input_sceneplan=input_sceneplan,
            temperature=temperature,
            sampling_seed=discrete_seed,
            discrete_decode_mode=discrete_decode_mode,
        )
        discrete_tokens, lexical_diagnostics = (
            self._apply_reliable_lexical_authority(
                discrete_tokens,
                task=task,
                input_lexical=input_lexical,
                diagnostics=discrete_diagnostics,
            )
        )
        discrete_diagnostics.update(lexical_diagnostics)
        model_decoded_tokens = discrete_tokens
        if diagnostic_discrete_tokens is None:
            discrete_diagnostics.update(
                {
                    "discrete_authority_mode": "deployment_decoded",
                    "model_decoded_discrete_sha256": self._v4_token_sha256(
                        model_decoded_tokens
                    ),
                    "diagnostic_discrete_override_sha256": None,
                    "diagnostic_override_matches_model_decode": None,
                }
            )
        else:
            discrete_tokens = self._validate_v4_diagnostic_discrete_tokens(
                diagnostic_discrete_tokens,
                task=task,
                device=device,
            )
            discrete_diagnostics.update(
                {
                    "discrete_authority_mode": "diagnostic_teacher_context_override",
                    "model_decoded_discrete_sha256": self._v4_token_sha256(
                        model_decoded_tokens
                    ),
                    "diagnostic_discrete_override_sha256": self._v4_token_sha256(
                        discrete_tokens
                    ),
                    "diagnostic_override_matches_model_decode": bool(
                        torch.equal(model_decoded_tokens, discrete_tokens)
                    ),
                    "diagnostic_override_after_lexical_authority": True,
                    "diagnostic_target_access": "explicit_evaluator_only",
                }
            )
        thought_context = torch.cat(
            [
                context,
                self.discrete_boundary[0:1].to(qwen_dtype),
                self.plan_embedding(discrete_tokens).to(qwen_dtype),
                self.discrete_boundary[1:2].to(qwen_dtype),
            ],
            dim=0,
        )
        previous_core = torch.zeros(
            (1, EXECUTION_SLOT_COUNT, EXECUTION_FEATURE_DIM),
            device=device,
            dtype=torch.float32,
        )
        if input_sceneplan is not None:
            previous_core[0] = torch.from_numpy(
                execution_state_core(
                    compile_execution_state(input_sceneplan, self.plan_codec)
                )
            ).to(device)
        return (
            discrete_tokens,
            thought_context,
            previous_core,
            context_metrics,
            discrete_diagnostics,
        )

    def _assemble_v4_core(
        self,
        *,
        task: P11Task,
        discrete_tokens: Tensor,
        core: Tensor,
        input_sceneplan: Mapping[str, Any] | None,
        duration_sec: float | None,
        sample_id: str,
        scene_thought_intervention: str | None = None,
    ) -> dict[str, Any]:
        device = discrete_tokens.device
        if task is P11Task.EDITING:
            if input_sceneplan is None:
                raise ValueError("P11-v4 Editing requires current ScenePlan")
            decoded_program = self.delta_sketch_codec.decode(discrete_tokens)
            program = self._apply_control_direction_intervention(
                decoded_program, scene_thought_intervention
            )
            core = self._apply_delta_thought_intervention(
                core, program, scene_thought_intervention
            )
            projected = project_delta_thought_to_atomic_patch(
                input_sceneplan,
                program,
                core,
                self.plan_codec,
                self.patch_codec,
            )
            final_plan = projected["target_sceneplan"]
            patch_tokens = projected["patch_tokens"].to(device)
            plan_tokens = self.plan_codec.encode(final_plan)["input_ids"].to(device)
            output_tokens = patch_tokens
            scene_sketch = projected["target_scene_sketch"]
            execution_state = projected["target_execution_state"]
            patch = projected["patch_spec"]
            control_direction = projected["control_direction"]
            discrete_object = program
            visible_trace = (
                "<E> evidence <DELTA_THOUGHT> continuous core </DELTA_THOUGHT> "
                "<PATCH> atomic patch </PATCH> -> ScenePlan -> P10(same seed) -> FOA"
            )
        else:
            scene_sketch = self.scene_sketch_codec.decode(discrete_tokens)
            fixed_frames = (
                None
                if duration_sec is None
                else self.plan_codec._duration_frame(float(duration_sec))
            )
            execution_state = execution_state_from_core(
                core,
                scene_sketch,
                self.plan_codec,
                sample_id=sample_id,
                duration_frames=fixed_frames,
            )
            final_plan = assemble_sceneplan(
                scene_sketch, execution_state, self.plan_codec
            )
            plan_tokens = self.plan_codec.encode(final_plan)["input_ids"].to(device)
            patch_tokens = None
            patch = None
            control_direction = None
            output_tokens = plan_tokens
            discrete_object = scene_sketch
            tag = "G" if task is P11Task.GENERATION else "U"
            visible_trace = (
                f"<{tag}> evidence <SCENE_THOUGHT> continuous core "
                "</SCENE_THOUGHT> <PLAN> ScenePlan tokens </PLAN> -> P10 -> FOA"
            )
        return {
            "visible_trace": visible_trace,
            "discrete_object": discrete_object,
            "thought_core": core,
            "control_direction": control_direction,
            "scene_sketch": scene_sketch,
            "execution_state": execution_state,
            "sceneplan": final_plan,
            "plan_tokens": plan_tokens,
            "patch": patch,
            "patch_tokens": patch_tokens,
            "output_tokens": output_tokens,
            "p10_conditions": compile_p10_from_contract(
                scene_sketch, execution_state, self.plan_codec
            ),
        }

    @torch.no_grad()
    def decode_transfusion_cot(
        self,
        prompt: str | Mapping[str, Any],
        *,
        task: P11Task | str,
        input_foa: Tensor | None = None,
        input_valid_mask: Tensor | None = None,
        input_semantic: Tensor | None = None,
        input_lexical: Mapping[str, Any] | Tensor | None = None,
        input_sceneplan: Mapping[str, Any] | None = None,
        duration_sec: float | None = None,
        sample_id: str = "p11_v4_generated",
        temperature: float = 0.0,
        discrete_seed: int | None = None,
        discrete_decode_mode: str | None = None,
        scene_thought_intervention: str | None = None,
        noise_seed: int | Sequence[int] | Tensor | None = None,
        noise: Tensor | None = None,
        diagnostic_discrete_tokens: Mapping[str, Tensor] | Tensor | None = None,
    ) -> dict[str, Any]:
        """Decode the canonical planner, optionally under a diagnostic teacher context.

        ``diagnostic_discrete_tokens`` is deliberately explicit: it substitutes
        the discrete authority only after normal deployment decoding so an
        evaluator can quantify teacher-context exposure bias.  Supplying it is
        target access and must never be used for deployable quality metrics.
        """

        task = normalize_p11_task(task)
        discrete_decode_mode = (
            self.discrete_decode_mode
            if discrete_decode_mode is None
            else str(discrete_decode_mode)
        )
        if task is P11Task.EDITING and input_sceneplan is None:
            raise ValueError("P11-v4 Editing requires current ScenePlan")
        self._validate_v4_intervention(scene_thought_intervention)
        if scene_thought_intervention in {
            "flip_control_direction",
            "flip_retime_delta",
        } and task is not P11Task.EDITING:
            raise ValueError("delta intervention is legal only for Editing")
        (
            discrete_tokens,
            thought_context,
            previous_core,
            context_metrics,
            discrete_diagnostics,
        ) = self._prepare_v4_decode(
            prompt,
            task=task,
            input_foa=input_foa,
            input_valid_mask=input_valid_mask,
            input_semantic=input_semantic,
            input_lexical=input_lexical,
            input_sceneplan=input_sceneplan,
            temperature=temperature,
            discrete_seed=discrete_seed,
            discrete_decode_mode=discrete_decode_mode,
            diagnostic_discrete_tokens=diagnostic_discrete_tokens,
        )
        device = discrete_tokens.device
        task_ids = torch.tensor(
            [list(P11Task).index(task)], device=device, dtype=torch.long
        )
        thought = self.execution_reasoner.infer(
            run_slots=lambda value: self._run_execution_slots(
                [thought_context], value, [task]
            ),
            task_ids=task_ids,
            previous_core=previous_core,
            noise_seed=noise_seed,
            noise=noise,
        )
        core = self._apply_v4_intervention(
            thought.core[0], scene_thought_intervention
        )
        assembled = self._assemble_v4_core(
            task=task,
            discrete_tokens=discrete_tokens,
            core=core,
            input_sceneplan=input_sceneplan,
            duration_sec=duration_sec,
            sample_id=sample_id,
            scene_thought_intervention=scene_thought_intervention,
        )
        return {
            "task": task.value,
            "model_contract": P11_V4_MODEL_CONTRACT,
            "sequence_contract": P11_V4_SEQUENCE_CONTRACT,
            **assembled,
            "internal_causal_order": (
                "evidence -> discrete semantic/control authority -> continuous numeric "
                "thought -> deterministic assembler"
            ),
            "discrete_tokens": discrete_tokens,
            "p10_edit_seed_policy": (
                "same_each_turn" if task is P11Task.EDITING else None
            ),
            "diagnostics": {
                **discrete_diagnostics,
                **context_metrics,
                "scene_thought_intervention": scene_thought_intervention,
                "thought_slots": EXECUTION_SLOT_COUNT,
                "thought_core_dim": EXECUTION_FEATURE_DIM,
                "editing_continuous_objective": (
                    self.execution_reasoner.editing_continuous_objective
                ),
                "editing_output_head": (
                    self.execution_reasoner.editing_output_head
                ),
                "control_direction": assembled["control_direction"],
                "control_direction_contract": self.control_direction_contract,
                "control_direction_parallel_head": False,
                "semantic_from_execution_state": False,
                "thought_noise_source": thought.inference_noise_source,
                "thought_noise_seeds": (
                    None
                    if thought.inference_noise_seeds is None
                    else list(thought.inference_noise_seeds)
                ),
                "thought_noise_sha256": self._v4_noise_sha256(
                    None
                    if thought.inference_noise is None
                    else thought.inference_noise[0]
                ),
            },
        }

    @torch.no_grad()
    def decode_transfusion_cot_samples(
        self,
        prompt: str | Mapping[str, Any],
        *,
        task: P11Task | str,
        noise_seeds: Sequence[int] | Tensor | None = None,
        noise: Tensor | None = None,
        input_foa: Tensor | None = None,
        input_valid_mask: Tensor | None = None,
        input_semantic: Tensor | None = None,
        input_lexical: Mapping[str, Any] | Tensor | None = None,
        input_sceneplan: Mapping[str, Any] | None = None,
        duration_sec: float | None = None,
        sample_id: str = "p11_v4_generated",
        temperature: float = 0.0,
        discrete_seed: int | None = None,
        discrete_decode_mode: str | None = None,
        scene_thought_intervention: str | None = None,
    ) -> list[dict[str, Any]]:
        """Decode one discrete authority and K continuous posterior draws.

        The shared SceneSketch/DeltaSketch is decoded exactly once.  Only the
        continuous execution state is batched over noise, making semantic
        immutability structural rather than an evaluator convention.
        """

        task = normalize_p11_task(task)
        discrete_decode_mode = (
            self.discrete_decode_mode
            if discrete_decode_mode is None
            else str(discrete_decode_mode)
        )
        if task is P11Task.EDITING and input_sceneplan is None:
            raise ValueError("P11-v4 Editing requires current ScenePlan")
        self._validate_v4_intervention(scene_thought_intervention)
        if scene_thought_intervention in {
            "flip_control_direction",
            "flip_retime_delta",
        } and task is not P11Task.EDITING:
            raise ValueError("delta intervention is legal only for Editing")
        if noise_seeds is not None and noise is not None:
            raise ValueError("P11-v4 accepts either noise_seeds or noise, not both")
        if noise is not None:
            noise = torch.as_tensor(noise)
            if noise.ndim == 2:
                noise = noise.unsqueeze(0)
            draw_count = int(noise.shape[0])
            normalized_seeds = None
        else:
            if noise_seeds is None:
                raise ValueError("P11-v4 K-sample decoding requires explicit noise_seeds")
            normalized_seeds = [
                int(value)
                for value in torch.as_tensor(noise_seeds, dtype=torch.long)
                .flatten()
                .tolist()
            ]
            draw_count = len(normalized_seeds)
        if not 1 <= draw_count <= 32:
            raise ValueError("P11-v4 posterior draw count must be within [1,32]")

        (
            discrete_tokens,
            thought_context,
            previous_core,
            context_metrics,
            discrete_diagnostics,
        ) = self._prepare_v4_decode(
            prompt,
            task=task,
            input_foa=input_foa,
            input_valid_mask=input_valid_mask,
            input_semantic=input_semantic,
            input_lexical=input_lexical,
            input_sceneplan=input_sceneplan,
            temperature=temperature,
            discrete_seed=discrete_seed,
            discrete_decode_mode=discrete_decode_mode,
        )
        device = discrete_tokens.device
        task_ids = torch.full(
            (draw_count,),
            list(P11Task).index(task),
            device=device,
            dtype=torch.long,
        )
        previous_batch = previous_core.expand(draw_count, -1, -1).clone()
        thought = self.execution_reasoner.infer(
            run_slots=lambda value: self._run_execution_slots(
                [thought_context] * draw_count, value, [task] * draw_count
            ),
            task_ids=task_ids,
            previous_core=previous_batch,
            noise_seed=normalized_seeds,
            noise=noise,
        )
        outputs = []
        for draw_index in range(draw_count):
            core = self._apply_v4_intervention(
                thought.core[draw_index], scene_thought_intervention
            )
            assembled = self._assemble_v4_core(
                task=task,
                discrete_tokens=discrete_tokens,
                core=core,
                input_sceneplan=input_sceneplan,
                duration_sec=duration_sec,
                sample_id=sample_id,
                scene_thought_intervention=scene_thought_intervention,
            )
            one_seed = (
                None
                if thought.inference_noise_seeds is None
                else int(thought.inference_noise_seeds[draw_index])
            )
            outputs.append(
                {
                    "task": task.value,
                    "model_contract": P11_V4_MODEL_CONTRACT,
                    "sequence_contract": P11_V4_SEQUENCE_CONTRACT,
                    **assembled,
                    "internal_causal_order": (
                        "evidence -> discrete semantic/control authority -> continuous "
                        "numeric thought -> deterministic assembler"
                    ),
                    "discrete_tokens": discrete_tokens,
                    "p10_edit_seed_policy": (
                        "same_each_turn" if task is P11Task.EDITING else None
                    ),
                    "diagnostics": {
                        **discrete_diagnostics,
                        **context_metrics,
                        "scene_thought_intervention": scene_thought_intervention,
                        "thought_slots": EXECUTION_SLOT_COUNT,
                        "thought_core_dim": EXECUTION_FEATURE_DIM,
                        "control_direction": assembled["control_direction"],
                        "control_direction_contract": self.control_direction_contract,
                        "control_direction_parallel_head": False,
                        "semantic_from_execution_state": False,
                        "thought_noise_source": thought.inference_noise_source,
                        "thought_noise_seeds": None if one_seed is None else [one_seed],
                        "thought_noise_sha256": self._v4_noise_sha256(
                            None
                            if thought.inference_noise is None
                            else thought.inference_noise[draw_index]
                        ),
                        "posterior_draw_index": draw_index,
                        "posterior_draw_count": draw_count,
                        "discrete_authority_shared_across_draws": True,
                    },
                }
            )
        return outputs

    @torch.no_grad()
    def decode_output_tokens(self, prompt, **kwargs):
        return_scene_thought_core = bool(kwargs.pop("return_scene_thought_core", False))
        kwargs.pop("max_tokens", None)
        constrained = kwargs.pop("constrained", True)
        if not constrained:
            raise ValueError("P11-v4 exposes only fail-closed constrained decoding")
        result = self.decode_transfusion_cot(prompt, **kwargs)
        diagnostics = {
            **result["diagnostics"],
            "output_kind": (
                "edit_patch" if result["task"] == P11Task.EDITING.value else "sceneplan"
            ),
            "decode_output_contract": P11_V4_OUTPUT_CONTRACT,
            "terminated": True,
            "transfusion_cot_arm": self.transfusion_cot_arm,
            "visible_trace": result["visible_trace"],
            "internal_causal_order": result["internal_causal_order"],
            "p10_edit_seed_policy": result["p10_edit_seed_policy"],
        }
        if return_scene_thought_core:
            diagnostics["scene_thought_core"] = (
                result["thought_core"].detach().float().cpu().tolist()
            )
        return result["output_tokens"], diagnostics

    @torch.no_grad()
    def plan_generation(
        self,
        user_text: str,
        *,
        duration_sec: float | None = None,
        sample_id: str = "p11_generation",
        resolver: ScenePlanResolver | None = None,
        temperature: float = 0.0,
        noise_seed: int | None = None,
    ) -> ScenePlanExecutionBundle:
        duration = self.default_plan_duration_sec if duration_sec is None else float(duration_sec)
        result = self.decode_transfusion_cot(
            user_text,
            task=P11Task.GENERATION,
            duration_sec=duration,
            sample_id=sample_id,
            temperature=temperature,
            noise_seed=noise_seed,
        )
        return self.finalize_plan(
            result["plan_tokens"],
            task=P11Task.GENERATION,
            sample_id=sample_id,
            resolver=resolver,
        )

    @torch.no_grad()
    def plan_understanding(
        self,
        input_foa: Tensor,
        *,
        input_semantic: Tensor | None = None,
        input_lexical: Mapping[str, Any] | Tensor | None = None,
        prompt: str = "Transcribe the input FOA into its complete ScenePlan.",
        sample_id: str = "p11_understanding",
        resolver: ScenePlanResolver | None = None,
        temperature: float = 0.0,
        noise_seed: int | None = None,
    ) -> ScenePlanExecutionBundle:
        duration = min(
            MAX_LATENT_FRAMES * self.downsampling_ratio / self.sample_rate,
            float(input_foa.shape[-1]) * self.downsampling_ratio / self.sample_rate,
        )
        result = self.decode_transfusion_cot(
            prompt,
            task=P11Task.UNDERSTANDING,
            input_foa=input_foa,
            input_semantic=input_semantic,
            input_lexical=input_lexical,
            duration_sec=duration,
            sample_id=sample_id,
            temperature=temperature,
            noise_seed=noise_seed,
        )
        return self.finalize_plan(
            result["plan_tokens"],
            task=P11Task.UNDERSTANDING,
            sample_id=sample_id,
            resolver=resolver,
        )

    @torch.no_grad()
    def plan_editing(
        self,
        edit_instruction: str,
        *,
        input_sceneplan: Mapping[str, Any],
        sample_id: str = "p11_editing",
        resolver: ScenePlanResolver | None = None,
        temperature: float = 0.0,
        noise_seed: int | None = None,
    ) -> ScenePlanExecutionBundle:
        result = self.decode_transfusion_cot(
            edit_instruction,
            task=P11Task.EDITING,
            input_sceneplan=input_sceneplan,
            sample_id=sample_id,
            temperature=temperature,
            noise_seed=noise_seed,
        )
        return self.finalize_plan(
            result["plan_tokens"],
            task=P11Task.EDITING,
            sample_id=sample_id,
            resolver=resolver,
            edit_patch_token_ids=result["patch_tokens"],
            edit_patch=result["patch"],
        )


class ScenePlanP11AudioAwarePlanner(ScenePlanP11V4Planner):
    """Active P11: shared FOA observation followed by one executable delta."""

    ROUTE_ID = "sceneplan_p11_audio_aware_v1"

    def __init__(self, model_config: Mapping[str, Any]) -> None:
        original = copy.deepcopy(dict(model_config))
        active = dict((original.get("model") or {}).get("transfusion_cot") or {})
        expected = {
            "contract": P11_AUDIO_AWARE_MODEL_CONTRACT,
            "sequence_contract": P11_AUDIO_AWARE_SEQUENCE_CONTRACT,
            "output_contract": P11_AUDIO_AWARE_OUTPUT_CONTRACT,
            "assembler": "observed_plus_atomic_patch_only_v1",
            "editing_audio_required": True,
            "old_sceneplan_role": "optional_fallible_prior",
            "target_audio_supervision": False,
            "delta_token_contract": AUDIO_AWARE_DELTA_SKETCH_TOKEN_CONTRACT,
            "scene_sketch_max_tokens": 512,
            "delta_sketch_max_tokens": 512,
        }
        for key, value in expected.items():
            if active.get(key) != value:
                raise ValueError(
                    f"audio-aware P11 transfusion_cot.{key}="
                    f"{active.get(key)!r}, expected {value!r}"
                )
        if active.get("observation") != {
            "shared_with_understanding": True,
            "discrete_authority": "scene_sketch",
            "continuous_authority": "flow_r1_execution_state",
        }:
            raise ValueError("audio-aware P11 observation authority changed")
        if active.get("editing") != {
            "discrete_authority": "atomic_delta_sketch",
            "continuous_authority": "deterministic_delta_execution_state",
            "revised_plan_decoder": False,
            "revised_plan_authority": "deterministic_apply_patch_to_observed",
        }:
            raise ValueError("audio-aware P11 revised-plan authority changed")
        if active.get("control_direction") is not None:
            raise ValueError(
                "audio-aware absolute move edits must not retain v4 direction forcing"
            )
        if active.get("delta_owner") != {
            "contract": P11_V4_DELTA_OWNER_CONTRACT,
            "objective": P11_V4_DELTA_OWNER_OBJECTIVE,
            "authority": "delta_sketch_owner_token_v1",
            "context": "edit_instruction_plus_audio_observed_sceneplan_v1",
            "inference": "grammar_constrained_argmax_v1",
            "legal_source_inventory": "observed_or_revised_source_mask_v1",
            "loss_normalization": "active_owner_rows_v1",
            "margin": 2.0,
            "margin_weight": 1.0,
        }:
            raise ValueError("audio-aware DeltaSketch owner contract changed")
        delta_control = dict(active.get("delta_control") or {})
        if delta_control != {
            "contract": P11_AUDIO_AWARE_DELTA_CONTROL_CONTRACT,
            "objective": "smooth_l1",
            "beta": 1.0,
            "retime_frame_unit": 32,
            "row_normalization": "owned_coordinates_then_active_rows_v1",
        }:
            raise ValueError("audio-aware delta-control objective changed")
        active_thought = dict(active.get("thought") or {})
        if (
            active_thought.get("arm") != "audio_aware_flow_r1_v1"
            or active_thought.get("continuous_objective") != "rectified_flow"
            or active_thought.get("editing_continuous_objective") != "direct_mse"
            or active_thought.get("editing_output_head") != "dedicated_delta_v1"
            or int(active_thought.get("inference_noise_seed", -1)) != 42
        ):
            raise ValueError("audio-aware Flow-R1/Direct-Delta contract changed")

        # Reuse the already validated v4 SCT/Flow implementation without
        # maintaining a parallel model tree.  Only constructor-time contract
        # values are adapted; the instance is switched to the active codec and
        # two-stage forward/decode methods immediately below.
        compatibility = copy.deepcopy(original)
        compat = compatibility["model"]["transfusion_cot"]
        compat.update(
            {
                "contract": P11_V4_MODEL_CONTRACT,
                "sequence_contract": P11_V4_SEQUENCE_CONTRACT,
                "delta_token_contract": DELTA_SKETCH_TOKEN_CONTRACT,
                "output_contract": P11_V4_OUTPUT_CONTRACT,
                "assembler": "deterministic_sketch_execution_assembler_v1",
                "scene_sketch_max_tokens": 512,
                "delta_sketch_max_tokens": 5,
            }
        )
        compat["delta_owner"] = {
            "contract": P11_V4_DELTA_OWNER_CONTRACT,
            "objective": P11_V4_DELTA_OWNER_OBJECTIVE,
            "authority": "delta_sketch_owner_token_v1",
            "context": "edit_instruction_plus_current_sceneplan_v1",
            "inference": "grammar_constrained_argmax_v1",
            "legal_source_inventory": "input_sceneplan_source_mask_v1",
            "loss_normalization": "active_owner_rows_v1",
            "margin": float(active["delta_owner"]["margin"]),
            "margin_weight": float(active["delta_owner"]["margin_weight"]),
        }
        compat["control_direction"] = {
            "contract": CONTROL_DIRECTION_CONTRACT,
            "authority": "delta_sketch_operation_specific_token_v1",
            "operations": list(CONTROL_DIRECTION_OPERATIONS),
            "decode": "exact_negative_positive_lookup_v1",
            "continuous_parallel_head": False,
            "conditions_delta_thought": True,
        }
        compat["thought"] = copy.deepcopy(compat["thought"])
        compat["thought"]["arm"] = "sketch_first_transfusion_cot_v4"
        compat["thought"]["inference_noise_seed"] = 42
        super().__init__(compatibility)
        if not self.audio_aware_editing or self.requires_input_sceneplan:
            raise RuntimeError("audio-aware Editing truth table was not activated")
        self.model_config = original
        self.delta_sketch_codec = AudioAwareDeltaSceneSketchCodec(
            self.plan_codec, self.patch_codec
        )
        self.delta_sketch_max_tokens = 512
        delta_mask = self.scene_sketch_vocab_mask.clone()
        delta_mask[list(self.patch_codec.used_token_ids)] = True
        self.delta_sketch_vocab_mask = delta_mask
        self.control_direction_contract = None
        self.delta_control_contract = str(delta_control["contract"])
        self.delta_control_beta = float(delta_control["beta"])
        self.delta_control_retime_frame_unit = float(
            delta_control["retime_frame_unit"]
        )
        self.execution_reasoner.arm = "audio_aware_flow_r1_v1"
        self.transfusion_cot_arm = "audio_aware_flow_r1_v1"
        self.active_model_contract = P11_AUDIO_AWARE_MODEL_CONTRACT
        self.active_data_contract = P11_AUDIO_AWARE_DATA_CONTRACT
        self.active_sequence_contract = P11_AUDIO_AWARE_SEQUENCE_CONTRACT

    def _append_observed_state(
        self,
        context: Tensor,
        observed_tokens: Tensor,
        observed_thought_tokens: Tensor,
    ) -> Tensor:
        """Append the model-visible observed state before DeltaSketch decode."""

        qwen_dtype = self.qwen_backbone.embed_tokens.weight.dtype
        ids = torch.as_tensor(
            observed_tokens,
            device=self.plan_embedding.weight.device,
            dtype=torch.long,
        ).flatten()
        return torch.cat(
            [
                context,
                self.discrete_boundary[0:1].to(qwen_dtype),
                self.plan_embedding(ids).to(qwen_dtype),
                self.discrete_boundary[1:2].to(qwen_dtype),
                self.v4_thought_boundary[0:1].to(qwen_dtype)
                + self.trace_type_embedding[0:1].to(qwen_dtype),
                observed_thought_tokens.to(qwen_dtype),
                self.v4_thought_boundary[1:2].to(qwen_dtype),
            ],
            dim=0,
        )

    @staticmethod
    def _zero_core_like(values: Sequence[Tensor], *, device: torch.device) -> Tensor:
        return torch.stack(
            [torch.zeros_like(torch.as_tensor(value, device=device).float()) for value in values]
        )

    def forward_audio_aware(
        self,
        prompts: Sequence[Mapping[str, Any]],
        observation_prompts: Sequence[Mapping[str, Any]],
        observed_discrete_targets: Sequence[Mapping[str, Tensor] | Tensor],
        delta_discrete_targets: Sequence[Optional[Mapping[str, Tensor] | Tensor]],
        *,
        tasks: Sequence[P11Task | str],
        input_foa: Sequence[Optional[Tensor]],
        input_valid_masks: Sequence[Optional[Tensor]],
        input_semantic: Sequence[Optional[Tensor]],
        input_plans: Sequence[Optional[Mapping[str, Tensor] | Tensor]],
        input_lexical: Sequence[Optional[Mapping[str, Any] | Tensor]],
        observed_execution_cores: Sequence[Tensor],
        revised_execution_cores: Sequence[Tensor],
        delta_execution_cores: Sequence[Tensor],
        observed_source_masks: Sequence[Tensor],
        revised_source_masks: Sequence[Tensor],
        loss_group_weights: Optional[Mapping[int, float]] = None,
        thought_loss_weights: Optional[Mapping[str, float]] = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Train the exact observation -> delta -> deterministic-revision DAG."""

        batch = len(prompts)
        fields = (
            observation_prompts,
            observed_discrete_targets,
            delta_discrete_targets,
            tasks,
            input_foa,
            input_valid_masks,
            input_semantic,
            input_plans,
            input_lexical,
            observed_execution_cores,
            revised_execution_cores,
            delta_execution_cores,
            observed_source_masks,
            revised_source_masks,
        )
        if batch <= 0 or any(len(value) != batch for value in fields):
            raise ValueError("audio-aware P11 batch fields are not aligned")
        normalized = [normalize_p11_task(task) for task in tasks]
        edit_indices = [
            index for index, task in enumerate(normalized) if task is P11Task.EDITING
        ]
        for index, task in enumerate(normalized):
            expects_audio = task in {P11Task.UNDERSTANDING, P11Task.EDITING}
            if (input_foa[index] is not None) != expects_audio:
                raise ValueError("audio-aware P11 task/audio truth table changed")
            if (input_valid_masks[index] is not None) != expects_audio:
                raise ValueError("audio-aware P11 task/audio-mask truth table changed")
            if task is not P11Task.EDITING and input_plans[index] is not None:
                raise ValueError("only Editing may carry a fallible plan prior")
            if (delta_discrete_targets[index] is not None) != (
                task is P11Task.EDITING
            ):
                raise ValueError("audio-aware delta target truth table changed")

        device = self.plan_embedding.weight.device
        qwen_dtype = self._ensure_qwen_device(device)

        def stack(values: Sequence[Tensor], dtype: torch.dtype) -> Tensor:
            return torch.stack(
                [torch.as_tensor(value, device=device, dtype=dtype) for value in values]
            )

        observed_cores = stack(observed_execution_cores, torch.float32)
        revised_cores = stack(revised_execution_cores, torch.float32)
        delta_cores = stack(delta_execution_cores, torch.float32)
        observed_masks = stack(observed_source_masks, torch.bool)
        revised_masks = stack(revised_source_masks, torch.bool)
        zero_cores = torch.zeros_like(observed_cores)
        zero_masks = torch.zeros_like(observed_masks)

        # Stage 1 is literally the Understanding route for E as well: it sees
        # input FOA/CLAP/(reliable ASR), but neither the edit instruction nor
        # the optional plan prior can become its semantic authority.
        observation_tasks = [
            P11Task.UNDERSTANDING if task is P11Task.EDITING else task
            for task in normalized
        ]
        observation_contexts: list[Tensor] = []
        observation_context_metrics: list[dict[str, int]] = []
        for prompt, task, audio, mask, semantic, lexical in zip(
            observation_prompts,
            observation_tasks,
            input_foa,
            input_valid_masks,
            input_semantic,
            input_lexical,
        ):
            context, metrics = self._v4_context_embeddings(
                task=task,
                prompt=prompt,
                input_foa=audio,
                input_valid_mask=mask,
                input_semantic=semantic,
                input_plan=None,
                input_lexical=lexical,
                device=device,
                qwen_dtype=qwen_dtype,
            )
            observation_contexts.append(context)
            observation_context_metrics.append(metrics)

        observation_ce = self._teacher_discrete_ce(
            observation_contexts,
            observed_discrete_targets,
            observation_tasks,
            [mask for mask in zero_masks],
            loss_group_weights=loss_group_weights,
        )
        observation_ids = observation_ce[2]
        observation_thought_contexts = [
            torch.cat(
                [
                    context,
                    self.discrete_boundary[0:1].to(qwen_dtype),
                    self.plan_embedding(ids).to(qwen_dtype),
                    self.discrete_boundary[1:2].to(qwen_dtype),
                ],
                dim=0,
            )
            for context, ids in zip(observation_contexts, observation_ids)
        ]
        observation_task_ids = torch.tensor(
            [list(P11Task).index(task) for task in observation_tasks],
            device=device,
            dtype=torch.long,
        )
        observation_thought = self.execution_reasoner.forward_train(
            run_slots=lambda value: self._run_execution_slots(
                observation_thought_contexts, value, observation_tasks
            ),
            task_ids=observation_task_ids,
            target_core=observed_cores,
            input_core=zero_cores,
            delta_core=zero_cores,
            target_source_mask=observed_masks,
            input_source_mask=zero_masks,
        )

        zero_per_row = observation_ce[1].new_zeros((batch,))
        delta_ce_per_row = zero_per_row.clone()
        delta_solve_per_row = zero_per_row.clone()
        delta_locality_per_row = zero_per_row.clone()
        delta_control_per_row = zero_per_row.clone()
        delta_control_active_per_row = zero_per_row.clone()
        delta_control_raw_rmse_per_row = zero_per_row.clone()
        owner_per_row = zero_per_row.clone()
        delta_owner_loss = zero_per_row.sum()
        delta_owner_count = torch.zeros((), device=device, dtype=torch.long)
        delta_owner_accuracy = zero_per_row.sum()
        delta_token_count = torch.zeros((), device=device)
        delta_context_metrics: list[dict[str, int]] = []

        if edit_indices:
            delta_contexts: list[Tensor] = []
            delta_targets: list[Mapping[str, Tensor] | Tensor] = []
            delta_tasks = [P11Task.EDITING] * len(edit_indices)
            for row_index in edit_indices:
                context, metrics = self._v4_context_embeddings(
                    task=P11Task.EDITING,
                    prompt=prompts[row_index],
                    input_foa=input_foa[row_index],
                    input_valid_mask=input_valid_masks[row_index],
                    input_semantic=input_semantic[row_index],
                    input_plan=input_plans[row_index],
                    input_lexical=input_lexical[row_index],
                    device=device,
                    qwen_dtype=qwen_dtype,
                )
                delta_contexts.append(
                    self._append_observed_state(
                        context,
                        observation_ids[row_index],
                        observation_thought.thought_tokens[row_index],
                    )
                )
                delta_context_metrics.append(metrics)
                target = delta_discrete_targets[row_index]
                if target is None:
                    raise RuntimeError("Editing row lost its DeltaSketch target")
                delta_targets.append(target)
            legal_owner_masks = [
                observed_masks[index] | revised_masks[index]
                for index in edit_indices
            ]
            delta_ce = self._teacher_discrete_ce(
                delta_contexts,
                delta_targets,
                delta_tasks,
                legal_owner_masks,
                loss_group_weights=loss_group_weights,
            )
            delta_ids = delta_ce[2]
            delta_thought_contexts = [
                torch.cat(
                    [
                        context,
                        self.discrete_boundary[0:1].to(qwen_dtype),
                        self.plan_embedding(ids).to(qwen_dtype),
                        self.discrete_boundary[1:2].to(qwen_dtype),
                    ],
                    dim=0,
                )
                for context, ids in zip(delta_contexts, delta_ids)
            ]
            edit_observed_cores = observed_cores[edit_indices]
            edit_revised_cores = revised_cores[edit_indices]
            edit_delta_cores = delta_cores[edit_indices]
            edit_observed_masks = observed_masks[edit_indices]
            edit_revised_masks = revised_masks[edit_indices]
            edit_task_ids = torch.full(
                (len(edit_indices),),
                list(P11Task).index(P11Task.EDITING),
                device=device,
                dtype=torch.long,
            )
            delta_thought = self.execution_reasoner.forward_train(
                run_slots=lambda value: self._run_execution_slots(
                    delta_thought_contexts, value, delta_tasks
                ),
                task_ids=edit_task_ids,
                target_core=edit_revised_cores,
                input_core=edit_observed_cores,
                delta_core=edit_delta_cores,
                target_source_mask=edit_revised_masks,
                input_source_mask=edit_observed_masks,
            )
            delta_programs = [
                self.delta_sketch_codec.decode(ids) for ids in delta_ids
            ]
            (
                edit_delta_control,
                edit_delta_control_active,
                edit_delta_control_raw_rmse,
            ) = _audio_aware_delta_control_objective(
                delta_thought.core,
                edit_delta_cores,
                edit_observed_cores,
                delta_programs,
                max_frames=int(self.plan_codec.max_frames),
                retime_frame_unit=self.delta_control_retime_frame_unit,
                beta=self.delta_control_beta,
            )
            index_tensor = torch.tensor(edit_indices, device=device, dtype=torch.long)
            delta_ce_per_row = delta_ce_per_row.index_copy(
                0,
                index_tensor,
                delta_ce[1].to(dtype=delta_ce_per_row.dtype),
            )
            delta_solve_per_row = delta_solve_per_row.index_copy(
                0,
                index_tensor,
                delta_thought.solve_loss_per_row.to(
                    dtype=delta_solve_per_row.dtype
                ),
            )
            delta_locality_per_row = delta_locality_per_row.index_copy(
                0,
                index_tensor,
                delta_thought.locality_loss_per_row.to(
                    dtype=delta_locality_per_row.dtype
                ),
            )
            delta_control_per_row = delta_control_per_row.index_copy(
                0,
                index_tensor,
                edit_delta_control.to(dtype=delta_control_per_row.dtype),
            )
            delta_control_active_per_row = delta_control_active_per_row.index_copy(
                0,
                index_tensor,
                edit_delta_control_active.to(
                    dtype=delta_control_active_per_row.dtype
                ),
            )
            delta_control_raw_rmse_per_row = delta_control_raw_rmse_per_row.index_copy(
                0,
                index_tensor,
                edit_delta_control_raw_rmse.to(dtype=zero_per_row.dtype),
            )
            owner_per_row = owner_per_row.index_copy(
                0,
                index_tensor,
                delta_ce[3].to(dtype=owner_per_row.dtype),
            )
            delta_owner_loss = delta_ce[4]
            delta_owner_count = delta_ce[5]
            delta_owner_accuracy = delta_ce[6]
            delta_token_count = observation_ce[0].new_tensor(
                sum(ids.numel() for ids in delta_ids)
            )

        weights = {
            "flow": 1.0,
            "solve": 1.0,
            "locality": 0.25,
            "owner": 0.1,
            "delta_control": 1.0,
            **{
                str(key): float(value)
                for key, value in (thought_loss_weights or {}).items()
            },
        }
        discrete_per_row = observation_ce[1] + delta_ce_per_row
        solve_per_row = (
            observation_thought.solve_loss_per_row + delta_solve_per_row
        )
        total = discrete_per_row.mean()
        total = total + self.u_inventory_loss_weight * observation_ce[14][
            "u_inventory_loss"
        ]
        total = total + weights["flow"] * observation_thought.flow_loss_per_row.mean()
        total = total + weights["solve"] * solve_per_row.mean()
        total = total + weights["locality"] * delta_locality_per_row.mean()
        delta_control_rows = delta_control_active_per_row.sum()
        delta_control_loss = delta_control_per_row.sum() / delta_control_rows.clamp_min(
            1.0
        )
        total = total + weights["delta_control"] * delta_control_loss
        total = total + weights["owner"] * delta_owner_loss
        total = total + self._trainable_graph_anchor(total)

        all_context_metrics = observation_context_metrics + delta_context_metrics
        return total, {
            "total_loss": total.detach(),
            "discrete_ce": discrete_per_row.mean().detach(),
            "discrete_ce_per_row": discrete_per_row.detach(),
            "observation_ce_per_row": observation_ce[1].detach(),
            "delta_ce_per_row": delta_ce_per_row.detach(),
            "flow_per_row": observation_thought.flow_loss_per_row.detach(),
            "solve_per_row": solve_per_row.detach(),
            "observation_solve_per_row": observation_thought.solve_loss_per_row.detach(),
            "delta_solve_per_row": delta_solve_per_row.detach(),
            "delta_control_per_row": delta_control_per_row.detach(),
            "delta_control_raw_rmse_per_row": delta_control_raw_rmse_per_row.detach(),
            "delta_control_loss": delta_control_loss.detach(),
            "delta_control_rows": delta_control_rows.detach(),
            "locality_per_row": delta_locality_per_row.detach(),
            "owner_per_row": owner_per_row.detach(),
            "owner_active_loss": delta_owner_loss.detach(),
            "owner_row_count": delta_owner_count.detach(),
            "owner_accuracy": delta_owner_accuracy.detach(),
            "text_end_ce": observation_ce[10].detach(),
            "text_end_count": observation_ce[11].detach(),
            "scene_eos_ce": observation_ce[12].detach(),
            "scene_eos_count": observation_ce[13].detach(),
            **{key: value.detach() for key, value in observation_ce[14].items()},
            "control_direction_ce": zero_per_row.sum().detach(),
            "control_direction_count": torch.zeros(
                (), device=device, dtype=torch.long
            ),
            "control_direction_accuracy": zero_per_row.sum().detach(),
            "discrete_tokens": total.new_tensor(
                sum(ids.numel() for ids in observation_ids)
            )
            + delta_token_count,
            "thought_tokens": total.new_tensor(
                batch * EXECUTION_SLOT_COUNT
                + len(edit_indices) * EXECUTION_SLOT_COUNT
            ),
            "prompt_tokens": total.new_tensor(
                sum(value["prompt_tokens"] for value in all_context_metrics)
            ),
            "input_audio_tokens": total.new_tensor(
                sum(value["input_audio_tokens"] for value in all_context_metrics)
            ),
            "input_semantic_tokens": total.new_tensor(
                sum(value["input_semantic_tokens"] for value in all_context_metrics)
            ),
            "input_lexical_tokens": total.new_tensor(
                sum(value["input_lexical_tokens"] for value in all_context_metrics)
            ),
            "input_plan_tokens": total.new_tensor(
                sum(value["input_plan_tokens"] for value in all_context_metrics)
            ),
            "editing_rows": total.new_tensor(len(edit_indices)),
        }

    @torch.no_grad()
    def decode_transfusion_cot(
        self,
        prompt: str | Mapping[str, Any],
        *,
        task: P11Task | str,
        input_foa: Tensor | None = None,
        input_valid_mask: Tensor | None = None,
        input_semantic: Tensor | None = None,
        input_lexical: Mapping[str, Any] | Tensor | None = None,
        input_sceneplan: Mapping[str, Any] | None = None,
        duration_sec: float | None = None,
        sample_id: str = "p11_audio_aware_generated",
        temperature: float = 0.0,
        discrete_seed: int | None = None,
        discrete_decode_mode: str | None = None,
        scene_thought_intervention: str | None = None,
        observation_thought_intervention: str | None = None,
        delta_thought_intervention: str | None = None,
        noise_seed: int | Sequence[int] | Tensor | None = None,
        noise: Tensor | None = None,
        diagnostic_discrete_tokens: Mapping[str, Tensor] | Tensor | None = None,
    ) -> dict[str, Any]:
        task = normalize_p11_task(task)
        if task is not P11Task.EDITING:
            if (
                observation_thought_intervention is not None
                or delta_thought_intervention is not None
            ):
                raise ValueError(
                    "stage-specific thought interventions are legal only for "
                    "audio-aware Editing"
                )
            result = super().decode_transfusion_cot(
                prompt,
                task=task,
                input_foa=input_foa,
                input_valid_mask=input_valid_mask,
                input_semantic=input_semantic,
                input_lexical=input_lexical,
                input_sceneplan=None,
                duration_sec=duration_sec,
                sample_id=sample_id,
                temperature=temperature,
                discrete_seed=discrete_seed,
                discrete_decode_mode=discrete_decode_mode,
                scene_thought_intervention=scene_thought_intervention,
                noise_seed=noise_seed,
                noise=noise,
                diagnostic_discrete_tokens=diagnostic_discrete_tokens,
            )
            result["model_contract"] = P11_AUDIO_AWARE_MODEL_CONTRACT
            result["sequence_contract"] = P11_AUDIO_AWARE_SEQUENCE_CONTRACT
            return result

        if input_foa is None:
            raise ValueError("audio-aware P11 Editing requires input FOA latent")
        if diagnostic_discrete_tokens is not None:
            raise ValueError(
                "audio-aware Editing diagnostic overrides must name observation "
                "and delta stages explicitly"
            )
        if scene_thought_intervention is not None:
            raise ValueError(
                "audio-aware Editing interventions must target observation or delta "
                "explicitly; the legacy shared switch is disabled"
            )
        self._validate_v4_intervention(observation_thought_intervention)
        self._validate_v4_intervention(delta_thought_intervention)
        observation_allowed = {
            None,
            "zero_thought",
            "shuffle_source_slots",
            "swap_source_1_2",
            "replace_one_control_field",
        }
        delta_allowed = {None, "zero_thought", "flip_retime_delta"}
        if observation_thought_intervention not in observation_allowed:
            raise ValueError(
                "audio-aware Editing observation intervention must alter only "
                "the absolute observed ExecutionState"
            )
        if delta_thought_intervention not in delta_allowed:
            raise ValueError(
                "audio-aware Editing delta intervention must be zero_thought or "
                "the owner-local retime counterfactual"
            )
        if noise is not None and noise_seed is not None:
            raise ValueError("provide either observation noise or noise seed, not both")
        discrete_decode_mode = (
            self.discrete_decode_mode
            if discrete_decode_mode is None
            else str(discrete_decode_mode)
        )
        effective_duration = (
            min(
                MAX_LATENT_FRAMES * self.downsampling_ratio / self.sample_rate,
                float(input_foa.shape[-1])
                * self.downsampling_ratio
                / self.sample_rate,
            )
            if duration_sec is None
            else float(duration_sec)
        )

        # Observation uses the exact U route and a task-independent prompt, so
        # the requested edit cannot leak into the recovered current scene.
        (
            observed_tokens,
            observed_thought_context,
            observed_previous_core,
            observed_context_metrics,
            observed_discrete_diagnostics,
        ) = self._prepare_v4_decode(
            "Observe the input FOA and recover its complete current ScenePlan.",
            task=P11Task.UNDERSTANDING,
            input_foa=input_foa,
            input_valid_mask=input_valid_mask,
            input_semantic=input_semantic,
            input_lexical=input_lexical,
            input_sceneplan=None,
            temperature=temperature,
            discrete_seed=discrete_seed,
            discrete_decode_mode=discrete_decode_mode,
        )
        device = observed_tokens.device
        observed_task_ids = torch.tensor(
            [list(P11Task).index(P11Task.UNDERSTANDING)],
            device=device,
            dtype=torch.long,
        )
        observed_thought = self.execution_reasoner.infer(
            run_slots=lambda value: self._run_execution_slots(
                [observed_thought_context], value, [P11Task.UNDERSTANDING]
            ),
            task_ids=observed_task_ids,
            previous_core=observed_previous_core,
            noise_seed=noise_seed,
            noise=noise,
        )
        observed_raw_core = observed_thought.core[0]
        observed_effective_core = self._apply_v4_intervention(
            observed_raw_core, observation_thought_intervention
        )
        observed_result = self._assemble_v4_core(
            task=P11Task.UNDERSTANDING,
            discrete_tokens=observed_tokens,
            core=observed_effective_core,
            input_sceneplan=None,
            duration_sec=effective_duration,
            sample_id=sample_id,
        )
        observed_plan = observed_result["sceneplan"]

        prior_tokens = (
            None
            if input_sceneplan is None
            else self.plan_codec.encode(input_sceneplan)
        )
        qwen_dtype = self._ensure_qwen_device(device)
        delta_base_context, delta_context_metrics = self._v4_context_embeddings(
            task=P11Task.EDITING,
            prompt=prompt,
            input_foa=input_foa,
            input_valid_mask=input_valid_mask,
            input_semantic=input_semantic,
            input_plan=prior_tokens,
            input_lexical=input_lexical,
            device=device,
            qwen_dtype=qwen_dtype,
        )
        delta_context = self._append_observed_state(
            delta_base_context,
            observed_tokens,
            observed_thought.thought_tokens[0],
        )
        delta_tokens, delta_discrete_diagnostics = self._decode_v4_discrete(
            delta_context,
            task=P11Task.EDITING,
            input_sceneplan=observed_plan,
            temperature=temperature,
            sampling_seed=discrete_seed,
            discrete_decode_mode=discrete_decode_mode,
        )
        delta_thought_context = torch.cat(
            [
                delta_context,
                self.discrete_boundary[0:1].to(qwen_dtype),
                self.plan_embedding(delta_tokens).to(qwen_dtype),
                self.discrete_boundary[1:2].to(qwen_dtype),
            ],
            dim=0,
        )
        observed_execution = compile_execution_state(observed_plan, self.plan_codec)
        previous_core = torch.from_numpy(
            execution_state_core(observed_execution)
        ).to(device=device, dtype=torch.float32).unsqueeze(0)
        edit_task_ids = torch.tensor(
            [list(P11Task).index(P11Task.EDITING)],
            device=device,
            dtype=torch.long,
        )
        delta_thought = self.execution_reasoner.infer(
            run_slots=lambda value: self._run_execution_slots(
                [delta_thought_context], value, [P11Task.EDITING]
            ),
            task_ids=edit_task_ids,
            previous_core=previous_core,
        )
        delta_program = self.delta_sketch_codec.decode(delta_tokens)
        delta_raw_core = delta_thought.core[0]
        delta_effective_core = self._apply_v4_intervention(
            delta_raw_core, delta_thought_intervention
        )
        delta_effective_core = self._apply_delta_thought_intervention(
            delta_effective_core,
            delta_program,
            delta_thought_intervention,
        )
        projected = project_audio_aware_delta_to_atomic_patch(
            observed_plan,
            delta_program,
            delta_effective_core,
            self.plan_codec,
            self.patch_codec,
        )
        revised_plan = projected["revised_sceneplan"]
        revised_tokens = self.plan_codec.encode(revised_plan)["input_ids"].to(device)
        patch_tokens = projected["patch_tokens"].to(device)
        return {
            "task": P11Task.EDITING.value,
            "model_contract": P11_AUDIO_AWARE_MODEL_CONTRACT,
            "sequence_contract": P11_AUDIO_AWARE_SEQUENCE_CONTRACT,
            "visible_trace": (
                "<E> FOA evidence <SCENE_THOUGHT> observed state </SCENE_THOUGHT> "
                "<DELTA_THOUGHT> executable delta </DELTA_THOUGHT> "
                "<PATCH> atomic patch </PATCH> -> revised ScenePlan -> P10(same seed)"
            ),
            "internal_causal_order": (
                "FOA -> observed SceneSketch/Flow-R1 state -> instruction-conditioned "
                "DeltaSketch/Direct-MSE state -> deterministic patch application"
            ),
            "observed_sceneplan": observed_plan,
            "observed_plan_tokens": observed_result["plan_tokens"],
            "observed_scene_sketch": observed_result["scene_sketch"],
            "observed_execution_state": observed_result["execution_state"],
            "observed_raw_thought_core": observed_raw_core,
            "observed_thought_core": observed_effective_core,
            "delta_program": delta_program,
            "delta_tokens": delta_tokens,
            "delta_raw_thought_core": delta_raw_core,
            "delta_thought_core": delta_effective_core,
            "patch": projected["patch_spec"],
            "patch_tokens": patch_tokens,
            "sceneplan": revised_plan,
            "revised_sceneplan": revised_plan,
            "plan_tokens": revised_tokens,
            "output_tokens": patch_tokens,
            "p10_conditions": compile_p10_from_contract(
                projected["revised_scene_sketch"],
                projected["revised_execution_state"],
                self.plan_codec,
            ),
            "p10_edit_seed_policy": "same_each_turn",
            "diagnostics": {
                "observed": {
                    **observed_discrete_diagnostics,
                    **observed_context_metrics,
                    "thought_noise_source": observed_thought.inference_noise_source,
                    "thought_noise_seeds": observed_thought.inference_noise_seeds,
                    "thought_intervention": observation_thought_intervention,
                },
                "delta": {
                    **delta_discrete_diagnostics,
                    **delta_context_metrics,
                    "continuous_objective": (
                        self.execution_reasoner.editing_continuous_objective
                    ),
                    "thought_intervention": delta_thought_intervention,
                },
                "old_sceneplan_present": input_sceneplan is not None,
                "old_sceneplan_role": "optional_fallible_prior",
                "revised_plan_decoder": False,
                "target_audio_supervision": False,
            },
        }

    @torch.no_grad()
    def decode_output_tokens(self, prompt, **kwargs):
        return_scene_thought_core = bool(
            kwargs.pop("return_scene_thought_core", False)
        )
        kwargs.pop("max_tokens", None)
        constrained = kwargs.pop("constrained", True)
        if not constrained:
            raise ValueError("audio-aware P11 exposes only constrained decoding")
        result = self.decode_transfusion_cot(prompt, **kwargs)
        diagnostics = {
            **result["diagnostics"],
            "output_kind": (
                "edit_patch"
                if result["task"] == P11Task.EDITING.value
                else "sceneplan"
            ),
            "decode_output_contract": P11_AUDIO_AWARE_OUTPUT_CONTRACT,
            "terminated": True,
            "transfusion_cot_arm": self.transfusion_cot_arm,
            "visible_trace": result["visible_trace"],
            "internal_causal_order": result["internal_causal_order"],
            "p10_edit_seed_policy": result["p10_edit_seed_policy"],
        }
        if return_scene_thought_core:
            key = (
                "delta_thought_core"
                if result["task"] == P11Task.EDITING.value
                else "thought_core"
            )
            diagnostics["scene_thought_core"] = (
                result[key].detach().float().cpu().tolist()
            )
        return result["output_tokens"], diagnostics

    @torch.no_grad()
    def plan_editing_audio_aware(
        self,
        edit_instruction: str,
        *,
        input_foa_latent: Tensor,
        input_semantic: Tensor,
        input_sceneplan: Mapping[str, Any] | None = None,
        input_valid_mask: Tensor | None = None,
        input_lexical: Mapping[str, Any] | Tensor | None = None,
        input_foa_ref: str | None = None,
        input_foa_latent_ref: str | None = None,
        sample_id: str = "p11_audio_aware_editing",
        resolver: ScenePlanResolver | None = None,
        temperature: float = 0.0,
        noise_seed: int | None = None,
        observation_thought_intervention: str | None = None,
        delta_thought_intervention: str | None = None,
    ) -> AudioAwareEditPlanningBundle:
        result = self.decode_transfusion_cot(
            edit_instruction,
            task=P11Task.EDITING,
            input_foa=input_foa_latent,
            input_valid_mask=input_valid_mask,
            input_semantic=input_semantic,
            input_lexical=input_lexical,
            input_sceneplan=input_sceneplan,
            sample_id=sample_id,
            temperature=temperature,
            noise_seed=noise_seed,
            observation_thought_intervention=observation_thought_intervention,
            delta_thought_intervention=delta_thought_intervention,
        )
        execution = self.finalize_plan(
            result["plan_tokens"],
            task=P11Task.EDITING,
            sample_id=sample_id,
            resolver=resolver,
            edit_patch_token_ids=result["patch_tokens"],
            edit_patch=result["patch"],
        )
        revised_ids = {
            str(source["source_id"]) for source in result["revised_sceneplan"]["sources"]
        }
        bundle = AudioAwareEditPlanningBundle(
            sample_id=sample_id,
            observed_plan_token_ids=result["observed_plan_tokens"].detach().cpu(),
            observed_sceneplan=result["observed_sceneplan"],
            edit_patch_token_ids=result["patch_tokens"].detach().cpu(),
            edit_patch=result["patch"],
            revised_plan_token_ids=result["plan_tokens"].detach().cpu(),
            revised_sceneplan=result["revised_sceneplan"],
            source_to_target_slot_map={
                str(source["source_id"]): (
                    str(source["source_id"])
                    if str(source["source_id"]) in revised_ids
                    else None
                )
                for source in result["observed_sceneplan"]["sources"]
            },
            execution_bundle=execution,
            input_foa_ref=input_foa_ref,
            input_foa_latent_ref=input_foa_latent_ref,
        )
        bundle.assert_consistent(codec=self.plan_codec, patch_codec=self.patch_codec)
        return bundle


def create_sceneplan_p11_v4_from_config(
    model_config: Mapping[str, Any],
) -> ScenePlanP11V4Planner:
    return ScenePlanP11V4Planner(model_config)


def create_sceneplan_p11_audio_aware_from_config(
    model_config: Mapping[str, Any],
) -> ScenePlanP11AudioAwarePlanner:
    return ScenePlanP11AudioAwarePlanner(model_config)


__all__ = [
    "P11_AUDIO_AWARE_DELTA_CONTROL_CONTRACT",
    "P11_AUDIO_AWARE_MODEL_CONTRACT",
    "P11_AUDIO_AWARE_OUTPUT_CONTRACT",
    "P11_V4_MODEL_CONTRACT",
    "P11_V4_OUTPUT_CONTRACT",
    "ScenePlanP11AudioAwarePlanner",
    "ScenePlanP11V4Planner",
    "create_sceneplan_p11_audio_aware_from_config",
    "create_sceneplan_p11_v4_from_config",
]
