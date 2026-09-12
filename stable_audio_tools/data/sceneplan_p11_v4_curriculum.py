"""Train-only prompt/target overlay for P11-v4 idea validation.

The canonical manifest-v6 and semantic cache stay immutable.  This wrapper
projects a small, versioned curriculum over their rows so all matched arms see
the same prompts, evidence stressors, and legal P10 targets:

* underspecified G prompts are paired with several legal numeric completions;
* U rows receive deterministic train-only latent/CLAP masks;
* E rows share operation/owner while varying a discrete P10 control direction.

The wrapper supports both the discrete D0 dataset and the sketch-first v4
dataset.  It changes only runtime supervision; it never writes through to the
frozen source manifest, index, or semantic cache.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from pathlib import Path
from typing import Any

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
from .sceneplan_p11_single_turn import (
    SCENEPLAN_PLAN_MAX_TOKENS,
    P11Task,
    normalize_p11_task,
)
from .sceneplan_p11_v4_challenge import apply_p11_v4_evidence_transform
from .sceneplan_p11_v4_dataset import (
    P11_V4_CONTROL_DIRECTION_CONTRACT,
    P11_V4_DATA_CONTRACT,
    P11_V4_SEQUENCE_CONTRACT,
    ScenePlanP11V4Dataset,
    compile_control_direction_target,
)
from .sceneplan_p11_dataset import (
    ScenePlanP11Dataset,
    tokenize_p11_task_prompt,
)


P11_V4_CURRICULUM_SCHEMA = "stable_audio_tools.sceneplan_p11_v4_curriculum"
P11_V4_CURRICULUM_VERSION = 1
P11_V4_CURRICULUM_CONTRACT = "p10_v11_train_only_gue_multitarget_v1"
P11_V4_CURRICULUM_PARTITION = "train_curriculum_reserved_v1"
P11_V4_SCREENING_CONTRACT = "p10_v11_matched_10k_gue_screening_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_json(payload: bytes) -> Any:
    return json.loads(zlib.decompress(payload))


class ScenePlanP11V4CurriculumDataset(torch.utils.data.Dataset):
    """Read-only train curriculum over one validated canonical P11 dataset."""

    def __init__(
        self,
        base: ScenePlanP11Dataset,
        curriculum_path: str | Path,
        *,
        expected_rows: int | None = None,
        expected_contract: str = P11_V4_CURRICULUM_CONTRACT,
        expected_ordering_contract: str | None = None,
        expected_ordering_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base, ScenePlanP11Dataset):
            raise TypeError("P11 curriculum requires a canonical P11 base dataset")
        self.base = base
        self.is_v4 = isinstance(base, ScenePlanP11V4Dataset)
        self.curriculum_path = (
            Path(curriculum_path).expanduser().resolve(strict=True)
        )
        self._connection: sqlite3.Connection | None = None
        connection = self._open_connection()
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        required = {
            "schema": P11_V4_CURRICULUM_SCHEMA,
            "schema_version": str(P11_V4_CURRICULUM_VERSION),
            "contract": str(expected_contract),
            "partition": P11_V4_CURRICULUM_PARTITION,
            "source_manifest": str(base.manifest_path),
            "source_manifest_sha256": _sha256_file(base.manifest_path),
            "source_index": str(base.index_path),
            "source_index_sha256": _sha256_file(base.index_path),
            "codec_fingerprint": base.codec.fingerprint,
            "eval_reserved_templates_present": "false",
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"P11 curriculum metadata {key}={metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if (expected_ordering_contract is None) != (
            expected_ordering_batch_size is None
        ):
            raise ValueError(
                "P11 expected ordering contract and batch size must be paired"
            )
        if expected_ordering_contract is not None:
            expected_ordering = {
                "ordering_contract": str(expected_ordering_contract),
                "ordering_batch_size": str(int(expected_ordering_batch_size)),
            }
            for key, expected in expected_ordering.items():
                if metadata.get(key) != expected:
                    raise RuntimeError(
                        f"P11 curriculum metadata {key}={metadata.get(key)!r}, "
                        f"expected {expected!r}"
                    )
        count = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
        bounds = connection.execute(
            "SELECT MIN(ordinal),MAX(ordinal) FROM rows"
        ).fetchone()
        if count != int(metadata.get("rows", -1)) or bounds != (0, count - 1):
            raise RuntimeError("P11 curriculum rows are incomplete or non-contiguous")
        if expected_rows is not None and count != int(expected_rows):
            raise RuntimeError(
                f"P11 curriculum has {count} rows, expected {int(expected_rows)}"
            )
        forbidden = int(
            connection.execute(
                "SELECT COUNT(*) FROM rows WHERE template_id LIKE 'eval_reserved/%'"
            ).fetchone()[0]
        )
        if forbidden:
            raise RuntimeError("held-out eval templates leaked into P11 training")
        connection.close()
        self.metadata = metadata
        self._length = count

    def _open_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{self.curriculum_path}?mode=ro&immutable=1",
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
            SELECT curriculum_id,task,family,view_id,template_id,
                   base_manifest_ordinal,base_target_ordinal,sample_id,prompt,
                   known_field_groups_json,target_sceneplan_zlib,edit_spec_json,
                   evidence_transform_json,target_variant,pair_id,pair_label
            FROM rows WHERE ordinal=?
            """,
            (int(index),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"P11 curriculum lacks ordinal {index}")
        (
            curriculum_id,
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
            target_variant,
            pair_id,
            pair_label,
        ) = row
        carrier, metadata = self.base[int(base_ordinal)]
        metadata = dict(metadata)
        task = normalize_p11_task(task_text)
        if metadata["p11_task"] != task.value:
            raise RuntimeError("curriculum/base task mismatch")
        if int(metadata["p11_target_ordinal"]) != int(target_ordinal):
            raise RuntimeError("curriculum/base target ordinal mismatch")
        if str(metadata["p11_target_sample_id"]) != str(sample_id):
            raise RuntimeError("curriculum/base sample identity mismatch")

        target_plan = self.base.codec.project_plan(
            _decode_json(target_payload), sample_id=str(sample_id)
        )
        edit_spec = None if edit_json is None else json.loads(str(edit_json))
        if task is P11Task.EDITING:
            input_plan = metadata.get("p11_input_sceneplan")
            if input_plan is None or edit_spec is None:
                raise RuntimeError("curriculum E row lacks current plan/edit spec")
            self.base.patch_codec.assert_target(input_plan, edit_spec, target_plan)
        elif edit_spec is not None:
            raise RuntimeError("non-E curriculum row unexpectedly carries an edit")

        metadata.update(
            {
                "p11_prompt": tokenize_p11_task_prompt(
                    str(prompt), self.base.tokenizer
                ),
                "p11_prompt_text": str(prompt),
                "p11_prompt_view_id": str(view_id),
                "p11_prompt_known_field_groups": json.loads(str(known_json)),
                "p11_target_sceneplan": target_plan,
                "p11_edit_kind": (
                    None if edit_spec is None else str(edit_spec["operation"])
                ),
                "p11_edit_spec": edit_spec,
                # Preserve the contract that was already checked against the
                # caller's expected value in __init__.  Hard-coding the pilot
                # contract here would make screening checkpoints claim the
                # wrong training-data provenance.
                "p11_curriculum_contract": str(self.metadata["contract"]),
                "p11_curriculum_train_only": True,
                "p11_curriculum_id": str(curriculum_id),
                "p11_row_identity": f"curriculum_row:{curriculum_id}",
                "p11_curriculum_family": str(family),
                "p11_curriculum_template_id": str(template_id),
                "p11_curriculum_target_variant": int(target_variant),
                "p11_curriculum_pair_id": (
                    None if pair_id is None else str(pair_id)
                ),
                "p11_curriculum_pair_label": (
                    None if pair_label is None else str(pair_label)
                ),
            }
        )

        if not self.is_v4:
            if task is P11Task.EDITING:
                metadata["p11_target_tokens"] = self.base.patch_codec.encode(
                    edit_spec
                )
                metadata["p11_output_kind"] = "edit_patch"
            else:
                metadata["p11_target_tokens"] = self.base.codec.encode(
                    target_plan, max_tokens=SCENEPLAN_PLAN_MAX_TOKENS
                )
                metadata["p11_output_kind"] = "sceneplan"
        else:
            target_sketch = compile_scene_sketch(target_plan, self.base.codec)
            target_execution = compile_execution_state(
                target_plan, self.base.codec
            )
            target_sketch_tokens = self.base.scene_sketch_codec.encode(
                target_sketch
            )
            target_core = torch.from_numpy(
                execution_state_core(target_execution)
            )
            target_mask = torch.tensor(
                target_execution["source_present_mask"], dtype=torch.bool
            )
            if task is P11Task.EDITING:
                input_plan = metadata["p11_input_sceneplan"]
                input_execution = compile_execution_state(
                    input_plan, self.base.codec
                )
                delta_sketch = compile_delta_scene_sketch(
                    input_plan, target_plan, edit_spec, self.base.codec
                )
                delta_tokens = self.base.delta_sketch_codec.encode(
                    delta_sketch, edit_spec
                )
                control_direction, control_direction_mask = (
                    compile_control_direction_target(edit_spec)
                )
                metadata.update(
                    {
                        "p11_target_tokens": delta_tokens,
                        "p11_output_kind": "delta_sketch",
                        "p11_v4_delta_scene_sketch": delta_sketch,
                        "p11_v4_delta_scene_sketch_tokens": delta_tokens,
                        "p11_v4_delta_execution_core": torch.from_numpy(
                            execution_delta_core(
                                input_execution, target_execution
                            )
                        ),
                        "p11_v4_control_direction_contract": (
                            P11_V4_CONTROL_DIRECTION_CONTRACT
                            if control_direction_mask
                            else None
                        ),
                        "p11_v4_control_direction": torch.tensor(
                            control_direction, dtype=torch.float32
                        ),
                        "p11_v4_control_direction_operation": str(
                            edit_spec["operation"]
                        ) if control_direction_mask else None,
                        "p11_v4_control_direction_mask": torch.tensor(
                            control_direction_mask, dtype=torch.bool
                        ),
                        "p11_v4_atomic_patch_tokens": (
                            self.base.patch_codec.encode(edit_spec)
                        ),
                    }
                )
            else:
                metadata.update(
                    {
                        "p11_target_tokens": target_sketch_tokens,
                        "p11_output_kind": "scene_sketch",
                        "p11_v4_delta_scene_sketch": None,
                        "p11_v4_delta_scene_sketch_tokens": None,
                        "p11_v4_delta_execution_core": torch.zeros_like(
                            target_core
                        ),
                        "p11_v4_atomic_patch_tokens": None,
                    }
                )
            metadata.update(
                {
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
                    "p11_v4_final_sceneplan_tokens": self.base.codec.encode(
                        target_plan
                    ),
                }
            )

        evidence_spec = json.loads(str(evidence_json))
        carrier, metadata = apply_p11_v4_evidence_transform(
            metadata, evidence_spec
        )
        metadata["p11_curriculum_evidence_transform"] = metadata.pop(
            "p11_challenge_evidence_transform"
        )
        return carrier, metadata


__all__ = [
    "P11_V4_CURRICULUM_CONTRACT",
    "P11_V4_CURRICULUM_PARTITION",
    "P11_V4_CURRICULUM_SCHEMA",
    "P11_V4_CURRICULUM_VERSION",
    "P11_V4_SCREENING_CONTRACT",
    "ScenePlanP11V4CurriculumDataset",
]
