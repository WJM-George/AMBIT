"""Canonical manifest-v8 dataset for audio-aware P11 Generation/U/E.

Editing reads the same frozen FOA/VAE latent and CLAP evidence as
Understanding.  Its optional old ScenePlan is a fallible prior; patch targets
are always applied to the audio-derived observed target, never to that prior.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .model_sceneplan_codec import load_model_sceneplan_codec
from .model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from .sceneplan_edit_patch import (
    PATCH_OUTPUT_CONTRACT,
    RETIME_FRAME_LEVELS,
    RETIME_MODES,
    RETIME_POLICY,
    ScenePlanEditPatchCodec,
)
from .sceneplan_p11_single_turn import (
    MAX_LATENT_FRAMES,
    P11_EDITING_CONTRACT,
    P11_EDITING_INPUT_CONTRACT,
    P11_EDITING_OUTPUT_CONTRACT,
    P11_EDIT_EVIDENCE_MODES,
    P11_GENERATION_NATURAL_CONTRACT,
    P11_MODEL_CONTRACT,
    P11_SOURCE_IDENTITY_CONTRACT,
    P11Task,
    SCENEPLAN_PLAN_MAX_TOKENS,
    SEMANTIC_CAPTION_MAX_TOKENS,
    canonicalize_sceneplan_source_ids,
    normalize_p11_task,
    validate_runtime_audio_spans,
)
from .sceneplan_v2_dataset import ScenePlanV2Dataset


P11_MANIFEST_SCHEMA = "stable_audio_tools.sceneplan_p11_manifest"
P11_MANIFEST_VERSION = 8
P11_DATA_CONTRACT = "audio_aware_sketch_first_transfusion_cot_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize_p11_task_prompt(
    text: str,
    tokenizer: Any,
    *,
    max_length: int = SEMANTIC_CAPTION_MAX_TOKENS,
) -> dict[str, torch.Tensor]:
    text = " ".join(str(text or "").split())
    if not text:
        raise ValueError("P11 task prompt must be non-empty")
    complete = tokenizer(text, truncation=False, padding=False, add_special_tokens=True)
    if len(complete["input_ids"]) > int(max_length):
        raise ValueError(
            f"P11 prompt requires {len(complete['input_ids'])} tokens > {max_length}"
        )
    encoded = tokenizer(
        text,
        truncation=False,
        padding="max_length",
        max_length=int(max_length),
        add_special_tokens=True,
    )
    input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    attention = torch.as_tensor(encoded["attention_mask"], dtype=torch.bool)
    if tuple(input_ids.shape) != (int(max_length),) or attention.shape != input_ids.shape:
        raise ValueError("P11 tokenizer did not return one fixed-size prompt")
    return {"input_ids": input_ids, "attention_mask": attention}


class ScenePlanP11Dataset(torch.utils.data.Dataset):
    """Materialize audio-aware G/U/E rows from one immutable FOA index."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        index_path: str | Path,
        codec_path: str | Path,
        tokenizer_spec: Any,
        expected_num_samples: int,
        index_num_samples: int,
        require_frozen: bool = True,
        semantic_cache_path: str | Path,
        semantic_dim: int = 512,
        semantic_encoder_revision: str,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
        self.index_path = Path(index_path).expanduser().resolve(strict=True)
        self.semantic_cache_path = (
            Path(semantic_cache_path).expanduser().resolve(strict=True)
        )
        if self.manifest_path.suffix != ".sqlite":
            raise ValueError("P11 manifest must be SQLite")
        if not isinstance(tokenizer_spec, (tuple, list)) or len(tokenizer_spec) not in (2, 3):
            raise ValueError("P11 requires the model Qwen tokenizer spec")
        self.tokenizer = tokenizer_spec[0]
        if int(tokenizer_spec[1]) != SEMANTIC_CAPTION_MAX_TOKENS:
            raise ValueError("P11 Qwen prompt limit must equal 512")
        codec = load_model_sceneplan_codec(codec_path)
        if not isinstance(codec, ModelScenePlanCodecV4):
            raise ValueError("canonical P11 requires the 648-frame ModelScenePlanCodecV4")
        self.codec = codec
        self.patch_codec = ScenePlanEditPatchCodec(codec)
        self.semantic_dim = int(semantic_dim)
        self.semantic_encoder_revision = str(semantic_encoder_revision)
        if self.semantic_dim <= 0 or not self.semantic_encoder_revision:
            raise ValueError("P11 semantic cache contract is incomplete")
        self._connection: sqlite3.Connection | None = None
        self._index_connection: sqlite3.Connection | None = None
        self._semantic_connection: sqlite3.Connection | None = None

        connection = self._open_connection(self.manifest_path)
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        from . import model_sceneplan_codec_v3, model_sceneplan_codec_v4, sceneplan_edit_patch

        required = {
            "schema": P11_MANIFEST_SCHEMA,
            "schema_version": str(P11_MANIFEST_VERSION),
            "p10_sceneplan_contract_revision": "5",
            "p10_conditioning_contract_revision": "2",
            "p10_dataset_contract_revision": "6",
            "planner_max_latent_frames": "648",
            "single_turn_only": "true",
            "max_input_audio_spans": "1",
            "p11_target_audio_spans": "0",
            "execution_contract": "external_p10_sceneplan_executor_v1",
            "codec_fingerprint": self.codec.fingerprint,
            "codec_implementation_sha256": _sha256_file(
                Path(model_sceneplan_codec_v4.__file__).resolve(strict=True)
            ),
            "codec_base_implementation_sha256": _sha256_file(
                Path(model_sceneplan_codec_v3.__file__).resolve(strict=True)
            ),
            "patch_codec_fingerprint": self.patch_codec.fingerprint,
            "patch_codec_implementation_sha256": _sha256_file(
                Path(sceneplan_edit_patch.__file__).resolve(strict=True)
            ),
            "source_index": str(self.index_path),
            "source_index_rows": str(int(index_num_samples)),
            "task_order": "generation,understanding,editing",
            "model_contract": P11_MODEL_CONTRACT,
            "data_contract": P11_DATA_CONTRACT,
            "output_contract": PATCH_OUTPUT_CONTRACT,
            "editing_contract": P11_EDITING_CONTRACT,
            "editing_input_contract": P11_EDITING_INPUT_CONTRACT,
            "editing_output_contract": P11_EDITING_OUTPUT_CONTRACT,
            "source_identity_contract": P11_SOURCE_IDENTITY_CONTRACT,
            "generation_prompt_contract": P11_GENERATION_NATURAL_CONTRACT,
            "understanding_audio_bridge": "hybrid_temporal_semantic_v1",
            "editing_audio_bridge": "hybrid_temporal_semantic_v1",
            "editing_input_audio_required": "true",
            "editing_old_sceneplan_role": "optional_fallible_prior",
            "editing_revised_authority": "deterministic_patch_applied_to_observed",
            "target_audio_supervision": "forbidden",
            "edit_evidence_modes": ",".join(P11_EDIT_EVIDENCE_MODES),
            "retime_policy": RETIME_POLICY,
            "retime_frame_levels": ",".join(map(str, RETIME_FRAME_LEVELS)),
            "retime_modes": ",".join(RETIME_MODES),
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"P11 manifest metadata {key!r}={metadata.get(key)!r}, expected {expected!r}"
                )
        count = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
        if count != int(expected_num_samples) or int(metadata.get("rows", -1)) != count:
            raise RuntimeError(
                f"P11 manifest row count mismatch: table={count}, expected={expected_num_samples}"
            )
        bounds = connection.execute("SELECT MIN(ordinal), MAX(ordinal) FROM rows").fetchone()
        if bounds != (0, count - 1):
            raise RuntimeError(f"P11 manifest ordinals are not contiguous: {bounds}")
        invalid = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM rows
                WHERE task NOT IN ('generation','understanding','editing')
                   OR source_ordinal < 0 OR source_ordinal >= ?
                   OR (task = 'generation' AND input_audio_ordinal IS NOT NULL)
                   OR (task IN ('understanding','editing')
                       AND input_audio_ordinal != source_ordinal)
                   OR target_sceneplan_zlib IS NULL
                   OR (task = 'generation' AND observed_sceneplan_zlib IS NOT NULL)
                   OR (task IN ('understanding','editing')
                       AND observed_sceneplan_zlib IS NULL)
                   OR (task != 'editing' AND input_sceneplan_zlib IS NOT NULL)
                   OR (task = 'editing' AND editing_evidence_mode NOT IN
                       ('no_plan','correct_plan','corrupt_plan'))
                   OR (task != 'editing' AND editing_evidence_mode IS NOT NULL)
                   OR (task = 'editing' AND editing_evidence_mode = 'no_plan'
                       AND input_sceneplan_zlib IS NOT NULL)
                   OR (task = 'editing' AND editing_evidence_mode IN
                       ('correct_plan','corrupt_plan')
                       AND input_sceneplan_zlib IS NULL)
                   OR (task = 'editing' AND (edit_kind IS NULL OR trim(edit_kind) = ''))
                   OR (task != 'editing' AND edit_kind IS NOT NULL)
                   OR (task = 'editing' AND (edit_spec_json IS NULL OR trim(edit_spec_json) = ''))
                   OR (task != 'editing' AND edit_spec_json IS NOT NULL)
                   OR (task = 'editing' AND editing_evidence_mode = 'corrupt_plan'
                       AND (prior_corruption_json IS NULL
                            OR trim(prior_corruption_json) = ''))
                   OR ((task != 'editing' OR editing_evidence_mode != 'corrupt_plan')
                       AND prior_corruption_json IS NOT NULL)
                   OR (ordinal % 3 = 0 AND task != 'generation')
                   OR (ordinal % 3 = 1 AND task != 'understanding')
                   OR (ordinal % 3 = 2 AND task != 'editing')
                   OR prompt IS NULL OR trim(prompt) = ''
                """,
                (int(index_num_samples),),
            ).fetchone()[0]
        )
        if invalid:
            raise RuntimeError(f"P11 manifest contains {invalid} invalid rows")
        base_samples = int(metadata.get("base_samples", -1))
        counts = {
            str(task): int(value)
            for task, value in connection.execute(
                "SELECT task, COUNT(*) FROM rows GROUP BY task"
            )
        }
        expected_counts = {task.value: base_samples for task in P11Task}
        if base_samples <= 0 or counts != expected_counts:
            raise RuntimeError(f"P11 G/U/E views are not balanced: {counts}")
        connection.close()

        semantic = self._open_connection(self.semantic_cache_path)
        semantic_metadata = dict(semantic.execute("SELECT key, value FROM metadata"))
        semantic_version = int(semantic_metadata.get("schema_version", -1))
        if semantic_metadata.get("schema") != "stable_audio_tools.p11_semantic_cache":
            raise RuntimeError("P11 semantic cache has the wrong schema")
        if semantic_version != 2:
            raise RuntimeError("canonical P11 semantic cache must be version 2")
        semantic_required = {
            "source_index": str(self.index_path),
            "source_manifest": str(self.manifest_path),
            "dimension": str(self.semantic_dim),
            "dtype": "float16",
            "encoder_revision": self.semantic_encoder_revision,
            "sample_rate": "48000",
            "normalization": "per_clip_peak_minus_1db",
            "representation": "windowed_clap_pooler_output",
            "window_sec": "5.000000",
            "hop_sec": "5.000000",
        }
        for key, expected in semantic_required.items():
            if semantic_metadata.get(key) != expected:
                raise RuntimeError(
                    f"P11 semantic cache {key}={semantic_metadata.get(key)!r}, expected {expected!r}"
                )
        manifest_connection = self._open_connection(self.manifest_path)
        try:
            manifest_ordinals = sorted(
                int(row[0])
                for row in manifest_connection.execute(
                    "SELECT DISTINCT source_ordinal FROM rows"
                )
            )
        finally:
            manifest_connection.close()
        feature_ordinals = [
            int(row[0]) for row in semantic.execute("SELECT ordinal FROM features ORDER BY ordinal")
        ]
        semantic.close()
        if feature_ordinals != manifest_ordinals or len(feature_ordinals) != base_samples:
            raise RuntimeError("P11 semantic cache does not exactly cover the manifest scenes")
        self.semantic_cache_version = semantic_version
        self.manifest_version = P11_MANIFEST_VERSION
        self.manifest_metadata = metadata
        self._length = count
        self.base = ScenePlanV2Dataset(
            self.index_path,
            tokenizer_spec=tokenizer_spec,
            expected_num_samples=int(index_num_samples),
            index_num_samples=int(index_num_samples),
            latent_crop_length=MAX_LATENT_FRAMES,
            caption_max_tokens=SEMANTIC_CAPTION_MAX_TOKENS,
            random_crop=False,
            require_frozen=bool(require_frozen),
        )

    @staticmethod
    def _open_connection(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False
        )
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = self._open_connection(self.manifest_path)
        return self._connection

    def _index_db(self) -> sqlite3.Connection:
        if self._index_connection is None:
            self._index_connection = self._open_connection(self.index_path)
        return self._index_connection

    def _semantic_db(self) -> sqlite3.Connection:
        if self._semantic_connection is None:
            self._semantic_connection = self._open_connection(self.semantic_cache_path)
        return self._semantic_connection

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_connection"] = None
        state["_index_connection"] = None
        state["_semantic_connection"] = None
        return state

    def __del__(self) -> None:
        for name in ("_connection", "_index_connection", "_semantic_connection"):
            connection = getattr(self, name, None)
            if connection is not None:
                connection.close()

    def __len__(self) -> int:
        return self._length

    def _state(
        self,
        ordinal: int,
        *,
        sceneplan_zlib: bytes | None = None,
        canonical_ids: bool = False,
    ) -> dict[str, Any]:
        row = self._index_db().execute(
            """
            SELECT sample_id, model_num_samples, latent_frames_valid, scene_plan_zlib
            FROM samples WHERE ordinal = ?
            """,
            (int(ordinal),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"frozen source index lacks ordinal {ordinal}")
        sample_id, model_num_samples, valid_frames, base_payload = row
        plan = json.loads(zlib.decompress(base_payload if sceneplan_zlib is None else sceneplan_zlib))
        if str(plan.get("sample_id")) != str(sample_id):
            raise RuntimeError("P11 ScenePlan/sample_id mismatch")
        plan = self.codec.project_plan(plan, sample_id=str(sample_id))
        if canonical_ids:
            plan = canonicalize_sceneplan_source_ids(plan)
        projected_samples = int(round(float(plan["duration_sec"]) * 44_100))
        projected_frames = math.ceil(projected_samples / 1024)
        if projected_frames != int(valid_frames):
            raise RuntimeError("P11 projection changed the frozen latent envelope")
        return {
            "sample_id": str(sample_id),
            "model_num_samples": projected_samples,
            "latent_frames_valid": projected_frames,
            "model_sceneplan": plan,
        }

    def _semantic_features(self, ordinal: int) -> torch.Tensor:
        row = self._semantic_db().execute(
            "SELECT embedding, windows FROM features WHERE ordinal = ?", (int(ordinal),)
        ).fetchone()
        windows = None if row is None else int(row[1])
        if row is None:
            raise RuntimeError(f"P11 semantic cache lacks ordinal {ordinal}")
        array = np.frombuffer(row[0], dtype=np.float16)
        if windows is None or windows <= 0 or array.size != windows * self.semantic_dim:
            raise RuntimeError("P11 semantic cache row has a stale shape")
        output = torch.from_numpy(array.copy()).float().reshape(windows, self.semantic_dim)
        if not bool(torch.isfinite(output).all()):
            raise RuntimeError("P11 semantic cache contains non-finite features")
        return output.squeeze(0) if windows == 1 else output

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        if not 0 <= int(index) < self._length:
            raise IndexError(index)
        row = self._db().execute(
            """
            SELECT task, source_ordinal, input_audio_ordinal, prompt,
                   observed_sceneplan_zlib, input_sceneplan_zlib,
                   target_sceneplan_zlib, editing_evidence_mode,
                   edit_kind, edit_spec_json, prior_corruption_json
            FROM rows WHERE ordinal = ?
            """,
            (int(index),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"P11 manifest lacks ordinal {index}")
        task = normalize_p11_task(row[0])
        source_ordinal = int(row[1])
        input_audio_ordinal = None if row[2] is None else int(row[2])
        prompt_text = str(row[3])
        observed = (
            None
            if row[4] is None
            else self._state(
                source_ordinal,
                sceneplan_zlib=row[4],
                canonical_ids=False,
            )
        )
        input_plan = (
            None
            if row[5] is None
            else self._state(
                source_ordinal,
                sceneplan_zlib=row[5],
                canonical_ids=False,
            )["model_sceneplan"]
        )
        target = self._state(
            source_ordinal,
            sceneplan_zlib=row[6],
            canonical_ids=False,
        )
        target_plan = target["model_sceneplan"]
        observed_plan = None if observed is None else observed["model_sceneplan"]
        evidence_mode = None if row[7] is None else str(row[7])
        edit_kind = None if row[8] is None else str(row[8])
        edit_spec = None if row[9] is None else json.loads(str(row[9]))
        prior_corruption = None if row[10] is None else json.loads(str(row[10]))

        input_latent = None
        input_valid = None
        input_plan_tokens = None
        input_semantic = None
        if task in {P11Task.UNDERSTANDING, P11Task.EDITING}:
            if input_audio_ordinal != source_ordinal:
                raise RuntimeError("P11 U/E row is not bound to its observed FOA")
            input_latent, input_metadata = self.base[source_ordinal]
            input_valid = torch.as_tensor(input_metadata["padding_mask"][0], dtype=torch.bool)
            input_semantic = self._semantic_features(source_ordinal)
        if input_plan is not None:
            input_plan_tokens = self.codec.encode(
                input_plan, max_tokens=SCENEPLAN_PLAN_MAX_TOKENS
            )

        if task is P11Task.UNDERSTANDING:
            if observed_plan is None:
                raise RuntimeError("P11 Understanding lacks observed ScenePlan target")
            if not torch.equal(
                self.codec.encode(observed_plan)["input_ids"],
                self.codec.encode(target_plan)["input_ids"],
            ):
                raise RuntimeError("P11 Understanding observed/target ScenePlans diverged")
        elif task is P11Task.EDITING:
            if observed_plan is None or edit_spec is None or evidence_mode is None:
                raise RuntimeError("audio-aware Editing row lacks observed/patch evidence")
            self.patch_codec.assert_target(observed_plan, edit_spec, target_plan)

        validate_runtime_audio_spans(task, input_foa=input_latent)
        if task is P11Task.EDITING:
            target_tokens = self.patch_codec.encode(edit_spec)
            output_kind = "edit_patch"
        else:
            target_tokens = self.codec.encode(
                target_plan, max_tokens=SCENEPLAN_PLAN_MAX_TOKENS
            )
            output_kind = "sceneplan"
        carrier = (
            input_latent
            if input_latent is not None
            else torch.zeros((64, MAX_LATENT_FRAMES), dtype=torch.float16)
        )
        carrier_mask = (
            input_valid
            if input_valid is not None
            else torch.zeros(MAX_LATENT_FRAMES, dtype=torch.bool)
        )
        metadata = {
            # Stable, target-independent identity used to prove that DDP ranks
            # receive disjoint manifest rows.  This is deliberately generic:
            # the active audio-aware route no longer pretends every manifest
            # is a legacy v4 curriculum.
            "p11_row_identity": f"manifest_row:{int(index)}",
            "p11_task": task.value,
            "p11_prompt": tokenize_p11_task_prompt(prompt_text, self.tokenizer),
            "p11_prompt_text": prompt_text,
            "p11_prompt_view_id": "manifest_exact_v0",
            "p11_prompt_known_field_groups": [],
            "p11_target_tokens": target_tokens,
            "p11_output_kind": output_kind,
            "p11_output_contract": PATCH_OUTPUT_CONTRACT,
            "p11_model_contract": P11_MODEL_CONTRACT,
            "p11_data_contract": P11_DATA_CONTRACT,
            "p11_target_sceneplan": dict(target_plan),
            "p11_observed_sceneplan_target": (
                None if observed_plan is None else dict(observed_plan)
            ),
            "p11_revised_sceneplan_target": (
                dict(target_plan) if task is P11Task.EDITING else None
            ),
            "p11_prior_sceneplan": input_plan,
            "p11_prior_sceneplan_tokens": input_plan_tokens,
            "p11_input_sceneplan": input_plan,
            "p11_input_sceneplan_tokens": input_plan_tokens,
            "p11_input_foa": input_latent,
            "p11_input_valid_mask": input_valid,
            "p11_input_semantic": input_semantic,
            "p11_source_ordinal": source_ordinal,
            "p11_input_audio_ordinal": input_audio_ordinal,
            # Transitional aliases consumed by the shared pre-migration
            # trainer.  Both now mean the immutable observed-audio ordinal.
            "p11_target_ordinal": source_ordinal,
            "p11_input_ordinal": input_audio_ordinal,
            "p11_target_sample_id": target["sample_id"],
            "p11_target_model_num_samples": target["model_num_samples"],
            "p11_target_latent_frames_valid": target["latent_frames_valid"],
            "p11_manifest_version": P11_MANIFEST_VERSION,
            "p11_source_matching": (
                "permutation_invariant"
                if task in {P11Task.GENERATION, P11Task.UNDERSTANDING}
                else "persistent_id"
            ),
            "p11_editing_contract": (
                P11_EDITING_CONTRACT if task is P11Task.EDITING else None
            ),
            "p11_editing_input_contract": (
                P11_EDITING_INPUT_CONTRACT if task is P11Task.EDITING else None
            ),
            "p11_editing_output_contract": (
                P11_EDITING_OUTPUT_CONTRACT if task is P11Task.EDITING else None
            ),
            "p11_editing_evidence_mode": evidence_mode,
            "p11_prior_corruption": prior_corruption,
            "p11_editing_score_version": "observed_patch_revised_v1",
            "p11_edit_kind": edit_kind,
            "p11_edit_spec": edit_spec,
            "p11_preserves_unedited_waveform": False,
            "p11_localized_editing": False,
            "p11_scene_thought_contract": None,
            "padding_mask": [carrier_mask],
            "audio": carrier,
        }
        return carrier, metadata


__all__ = [
    "P11_DATA_CONTRACT",
    "P11_MANIFEST_SCHEMA",
    "P11_MANIFEST_VERSION",
    "ScenePlanP11Dataset",
    "tokenize_p11_task_prompt",
]
