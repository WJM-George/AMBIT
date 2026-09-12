"""Held-out challenge overlay for sketch-first P11-v4 evaluation.

The canonical manifest-v6 remains immutable.  This module overlays only
evaluation-time prompt/evidence/target variants while deriving every v4
SceneSketch and P10ExecutionState field through the canonical compilers.

Challenge rows are deliberately split into four roles:

* exact compatibility rows;
* underspecified Generation rows with non-exhaustive legal completions;
* synthetic representation-degradation rows for Understanding;
* paired counterfactual atomic edits for Editing.

The overlay is evaluation-only.  In particular, a hidden exact Generation
target is retained for diagnostics, but model selection must use its frozen
``known_field_groups`` and reference-completion contract instead.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .scene_sketch_v1 import (
    DELTA_SCENE_SKETCH_CONTRACT,
    DELTA_SKETCH_TOKEN_CONTRACT,
    EXECUTION_STATE_CONTRACT,
    SCENE_SKETCH_CONTRACT,
    compile_delta_scene_sketch,
    compile_execution_state,
    compile_scene_sketch,
    execution_delta_core,
    execution_state_core,
)
from .sceneplan_p11_single_turn import P11Task, normalize_p11_task
from .sceneplan_p11_v4_dataset import (
    P11_V4_CONTROL_DIRECTION_CONTRACT,
    P11_V4_DATA_CONTRACT,
    P11_V4_SEQUENCE_CONTRACT,
    ScenePlanP11V4Dataset,
    compile_control_direction_target,
)
from .sceneplan_p11_dataset import tokenize_p11_task_prompt


P11_V4_CHALLENGE_SCHEMA = "stable_audio_tools.sceneplan_p11_v4_challenge"
P11_V4_CHALLENGE_VERSION = 1
P11_V4_CHALLENGE_CONTRACT = "p10_v11_15s_gue_falsification_v1"
P11_V4_REFERENCE_SET_CONTRACT = "non_exhaustive_p10_legal_completions_v1"
P11_V4_EVIDENCE_TRANSFORM_CONTRACT = "synthetic_representation_stress_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_json(payload: bytes) -> Any:
    return json.loads(zlib.decompress(payload))


def _tensor_sha256(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def apply_p11_v4_evidence_transform(
    metadata: dict[str, Any], spec: Mapping[str, Any]
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply one deterministic U-only latent/CLAP representation stressor."""

    contract = str(spec.get("contract") or "")
    if contract != P11_V4_EVIDENCE_TRANSFORM_CONTRACT:
        raise ValueError(f"unsupported P11-v4 evidence transform {contract!r}")
    task = normalize_p11_task(metadata["p11_task"])
    transform_id = str(spec.get("transform_id") or "")
    if not transform_id:
        raise ValueError("P11-v4 evidence transform lacks transform_id")
    if task is not P11Task.UNDERSTANDING and transform_id != "identity_v1":
        raise ValueError("only Understanding may carry degraded evidence")

    latent_value = metadata.get("p11_input_foa")
    semantic_value = metadata.get("p11_input_semantic")
    latent = None if latent_value is None else torch.as_tensor(latent_value).clone()
    semantic = (
        None if semantic_value is None else torch.as_tensor(semantic_value).clone()
    )
    before = {
        "foa_sha256": _tensor_sha256(latent),
        "semantic_sha256": _tensor_sha256(semantic),
    }

    time_mask = spec.get("foa_time_mask")
    if time_mask is not None:
        if latent is None or task is not P11Task.UNDERSTANDING:
            raise ValueError("FOA masking requires Understanding latent evidence")
        if not isinstance(time_mask, list) or len(time_mask) != 2:
            raise ValueError("foa_time_mask must be [start,end]")
        start, end = (int(time_mask[0]), int(time_mask[1]))
        valid = int(torch.as_tensor(metadata["p11_input_valid_mask"]).sum())
        if not 0 <= start < end <= valid:
            raise ValueError("FOA mask lies outside the valid latent prefix")
        latent[..., start:end] = 0

    channel_mask = spec.get("semantic_channel_mask")
    if channel_mask is not None:
        if semantic is None or task is not P11Task.UNDERSTANDING:
            raise ValueError("semantic masking requires Understanding CLAP evidence")
        if not isinstance(channel_mask, list) or len(channel_mask) != 2:
            raise ValueError("semantic_channel_mask must be [start,end]")
        start, end = (int(channel_mask[0]), int(channel_mask[1]))
        if not 0 <= start < end <= int(semantic.shape[-1]):
            raise ValueError("semantic mask lies outside the CLAP feature axis")
        semantic[..., start:end] = 0

    if latent is not None and not bool(torch.isfinite(latent).all()):
        raise ValueError("evidence transform produced non-finite FOA latent")
    if semantic is not None and not bool(torch.isfinite(semantic).all()):
        raise ValueError("evidence transform produced non-finite semantic evidence")
    metadata["p11_input_foa"] = latent
    metadata["p11_input_semantic"] = semantic
    diagnostics = {
        "contract": contract,
        "transform_id": transform_id,
        "synthetic_representation_stress": transform_id != "identity_v1",
        "not_a_real_acoustic_corruption_benchmark": True,
        "foa_time_mask": time_mask,
        "semantic_channel_mask": channel_mask,
        "before": before,
        "after": {
            "foa_sha256": _tensor_sha256(latent),
            "semantic_sha256": _tensor_sha256(semantic),
        },
    }
    metadata["p11_challenge_evidence_transform"] = diagnostics
    carrier = (
        latent
        if latent is not None
        else torch.zeros((64, 648), dtype=torch.float16)
    )
    return carrier, metadata


class ScenePlanP11V4ChallengeDataset(torch.utils.data.Dataset):
    """Read-only evaluation overlay on a validated P11-v4 base dataset."""

    def __init__(
        self,
        base: ScenePlanP11V4Dataset,
        challenge_path: str | Path,
    ) -> None:
        super().__init__()
        if not isinstance(base, ScenePlanP11V4Dataset):
            raise TypeError("P11-v4 challenge overlay requires ScenePlanP11V4Dataset")
        self.base = base
        self.challenge_path = Path(challenge_path).expanduser().resolve(strict=True)
        self._connection: sqlite3.Connection | None = None
        connection = self._open_connection()
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        required = {
            "schema": P11_V4_CHALLENGE_SCHEMA,
            "schema_version": str(P11_V4_CHALLENGE_VERSION),
            "contract": P11_V4_CHALLENGE_CONTRACT,
            "source_manifest": str(base.manifest_path),
            "source_manifest_sha256": _sha256_file(base.manifest_path),
            "source_index": str(base.index_path),
            "source_index_sha256": _sha256_file(base.index_path),
            "codec_fingerprint": base.codec.fingerprint,
            "p11_v4_data_contract": P11_V4_DATA_CONTRACT,
            "p11_v4_sequence_contract": P11_V4_SEQUENCE_CONTRACT,
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"P11-v4 challenge metadata {key}={metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )
        count = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
        bounds = connection.execute("SELECT MIN(ordinal),MAX(ordinal) FROM rows").fetchone()
        if count != int(metadata.get("rows", -1)) or bounds != (0, count - 1):
            raise RuntimeError("P11-v4 challenge rows are incomplete or non-contiguous")
        connection.close()
        self.metadata = metadata
        self._length = count

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.challenge_path}?mode=ro&immutable=1",
            uri=True,
            check_same_thread=False,
        )
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = self._open_connection()
        return self._connection

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        return state

    def __del__(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        if not 0 <= int(index) < self._length:
            raise IndexError(index)
        row = self._db().execute(
            """
            SELECT challenge_id,task,family,view_id,template_id,
                   base_manifest_ordinal,base_target_ordinal,sample_id,prompt,
                   known_field_groups_json,target_sceneplan_zlib,edit_spec_json,
                   evidence_transform_json,reference_sceneplans_zlib,
                   reference_set_contract,pair_id,pair_label,selection_role
            FROM rows WHERE ordinal=?
            """,
            (int(index),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"P11-v4 challenge lacks ordinal {index}")
        (
            challenge_id,
            task_text,
            family,
            view_id,
            template_id,
            base_ordinal,
            target_ordinal,
            sample_id,
            prompt,
            known_json,
            target_payload,
            edit_json,
            evidence_json,
            references_payload,
            reference_contract,
            pair_id,
            pair_label,
            selection_role,
        ) = row
        carrier, metadata = self.base[int(base_ordinal)]
        metadata = dict(metadata)
        task = normalize_p11_task(task_text)
        if metadata["p11_task"] != task.value:
            raise RuntimeError("challenge/base task mismatch")
        if int(metadata["p11_target_ordinal"]) != int(target_ordinal):
            raise RuntimeError("challenge/base target ordinal mismatch")
        if str(metadata["p11_target_sample_id"]) != str(sample_id):
            raise RuntimeError("challenge/base sample identity mismatch")

        target_plan = self.base.codec.project_plan(
            _decode_json(target_payload), sample_id=str(sample_id)
        )
        target_sketch = compile_scene_sketch(target_plan, self.base.codec)
        target_execution = compile_execution_state(target_plan, self.base.codec)
        target_sketch_tokens = self.base.scene_sketch_codec.encode(target_sketch)
        target_core = torch.from_numpy(execution_state_core(target_execution))
        target_mask = torch.tensor(
            target_execution["source_present_mask"], dtype=torch.bool
        )
        edit_spec = None if edit_json is None else json.loads(str(edit_json))

        if task is P11Task.EDITING:
            input_plan = metadata.get("p11_input_sceneplan")
            if input_plan is None or edit_spec is None:
                raise RuntimeError("counterfactual E row lacks input plan/edit spec")
            self.base.patch_codec.assert_target(input_plan, edit_spec, target_plan)
            input_execution = compile_execution_state(input_plan, self.base.codec)
            delta_sketch = compile_delta_scene_sketch(
                input_plan, target_plan, edit_spec, self.base.codec
            )
            delta_tokens = self.base.delta_sketch_codec.encode(delta_sketch, edit_spec)
            patch_tokens = self.base.patch_codec.encode(edit_spec)
            control_direction, control_direction_mask = (
                compile_control_direction_target(edit_spec)
            )
            metadata.update(
                {
                    "p11_target_tokens": delta_tokens,
                    "p11_output_kind": "delta_sketch",
                    "p11_edit_kind": str(edit_spec["operation"]),
                    "p11_edit_spec": edit_spec,
                    "p11_v4_delta_scene_sketch": delta_sketch,
                    "p11_v4_delta_scene_sketch_tokens": delta_tokens,
                    "p11_v4_delta_execution_core": torch.from_numpy(
                        execution_delta_core(input_execution, target_execution)
                    ),
                    "p11_v4_atomic_patch_tokens": patch_tokens,
                    "p11_v4_control_direction_contract": (
                        P11_V4_CONTROL_DIRECTION_CONTRACT
                        if control_direction_mask
                        else None
                    ),
                    "p11_v4_control_direction": torch.tensor(
                        control_direction, dtype=torch.float32
                    ),
                    "p11_v4_control_direction_operation": (
                        str(edit_spec["operation"])
                        if control_direction_mask
                        else None
                    ),
                    "p11_v4_control_direction_mask": torch.tensor(
                        control_direction_mask, dtype=torch.bool
                    ),
                }
            )
        else:
            metadata.update(
                {
                    "p11_target_tokens": target_sketch_tokens,
                    "p11_output_kind": "scene_sketch",
                    "p11_edit_kind": None,
                    "p11_edit_spec": None,
                    "p11_v4_delta_scene_sketch": None,
                    "p11_v4_delta_scene_sketch_tokens": None,
                    "p11_v4_delta_execution_core": torch.zeros_like(target_core),
                    "p11_v4_atomic_patch_tokens": None,
                }
            )

        known_groups = json.loads(str(known_json))
        evidence_spec = json.loads(str(evidence_json))
        references = _decode_json(references_payload)
        metadata.update(
            {
                "p11_prompt": tokenize_p11_task_prompt(
                    str(prompt), self.base.tokenizer
                ),
                "p11_prompt_text": str(prompt),
                "p11_prompt_view_id": str(view_id),
                "p11_prompt_known_field_groups": known_groups,
                "p11_target_sceneplan": target_plan,
                "p11_v4_data_contract": P11_V4_DATA_CONTRACT,
                "p11_v4_sequence_contract": P11_V4_SEQUENCE_CONTRACT,
                "p11_v4_scene_sketch_contract": SCENE_SKETCH_CONTRACT,
                "p11_v4_execution_state_contract": EXECUTION_STATE_CONTRACT,
                "p11_v4_delta_scene_sketch_contract": (
                    DELTA_SCENE_SKETCH_CONTRACT
                    if task is P11Task.EDITING
                    else None
                ),
                "p11_v4_delta_token_contract": (
                    DELTA_SKETCH_TOKEN_CONTRACT
                    if task is P11Task.EDITING
                    else None
                ),
                "p11_v4_target_scene_sketch": target_sketch,
                "p11_v4_target_scene_sketch_tokens": target_sketch_tokens,
                "p11_v4_target_execution_state": target_execution,
                "p11_v4_target_execution_core": target_core,
                "p11_v4_target_source_mask": target_mask,
                "p11_v4_final_sceneplan_tokens": self.base.codec.encode(target_plan),
                "p11_challenge_contract": P11_V4_CHALLENGE_CONTRACT,
                "p11_challenge_eval_only": True,
                "p11_challenge_id": str(challenge_id),
                "p11_challenge_family": str(family),
                "p11_challenge_template_id": str(template_id),
                "p11_challenge_selection_role": str(selection_role),
                "p11_challenge_pair_id": None if pair_id is None else str(pair_id),
                "p11_challenge_pair_label": (
                    None if pair_label is None else str(pair_label)
                ),
                "p11_challenge_reference_set_contract": str(reference_contract),
                "p11_challenge_reference_sceneplans": references,
                "p11_challenge_hidden_exact_target_is_diagnostic_only": (
                    task is P11Task.GENERATION
                    and str(reference_contract)
                    == P11_V4_REFERENCE_SET_CONTRACT
                ),
            }
        )
        carrier, metadata = apply_p11_v4_evidence_transform(
            metadata, evidence_spec
        )
        return carrier, metadata


__all__ = [
    "P11_V4_CHALLENGE_CONTRACT",
    "P11_V4_CHALLENGE_SCHEMA",
    "P11_V4_CHALLENGE_VERSION",
    "P11_V4_EVIDENCE_TRANSFORM_CONTRACT",
    "P11_V4_REFERENCE_SET_CONTRACT",
    "ScenePlanP11V4ChallengeDataset",
    "apply_p11_v4_evidence_transform",
]
