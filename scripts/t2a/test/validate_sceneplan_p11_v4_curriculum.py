#!/usr/bin/env python3
"""Strict structural, leakage, ordering, and runtime gate for P11-v4 curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.scene_sketch_v1 import (  # noqa: E402
    DeltaSceneSketchCodec,
    SceneSketchCodec,
    compile_delta_scene_sketch,
    compile_execution_state,
    compile_scene_sketch,
    execution_state_core,
)
from stable_audio_tools.data.sceneplan_edit_patch import (  # noqa: E402
    ScenePlanEditPatchCodec,
)
from stable_audio_tools.data.sceneplan_p11_metrics import (  # noqa: E402
    score_generation_constraints,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    canonicalize_sceneplan_source_ids,
    validate_p11_executor_profile,
)
from stable_audio_tools.data.sceneplan_p11_v4_curriculum import (  # noqa: E402
    P11_V4_CURRICULUM_CONTRACT,
    P11_V4_CURRICULUM_PARTITION,
    P11_V4_CURRICULUM_SCHEMA,
    P11_V4_CURRICULUM_VERSION,
    ScenePlanP11V4CurriculumDataset,
)
from stable_audio_tools.data.sceneplan_p11_v4_dataset import (  # noqa: E402
    ScenePlanP11V4Dataset,
)
from stable_audio_tools.data.sceneplan_p11_dataset import (  # noqa: E402
    ScenePlanP11Dataset,
)


DEFAULT_CURRICULUM = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "p11_v4_curriculum/"
    "p11_train_pilot90_curriculum_v8_pair_aware_interleaved_batch8_seed42.sqlite"
)
DEFAULT_V4_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_pilot90_curriculum_pair_aware_transfusion_cot_v4_reliable_asr_v1.json"
)
DEFAULT_D0_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_pilot90_curriculum_pair_aware_discrete_d0.json"
)
DEFAULT_MODEL = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/p11_canonical_pair_aware_curriculum_validation_20260901.json"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode(payload: bytes) -> Any:
    return json.loads(zlib.decompress(payload))


def _json_sha(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro&immutable=1", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def _edit_direction(edit_json: str) -> int:
    spec = json.loads(edit_json)
    operation = str(spec.get("operation") or "")
    if operation == "rotate_source":
        return -1 if float(spec["delta_azimuth_deg"]) < 0.0 else 1
    if operation == "distance_source":
        return -1 if float(spec["distance_factor"]) < 1.0 else 1
    raise ValueError(f"unsupported paired E operation {operation!r}")


def _pair_paraphrase(pair_label: str) -> str:
    if "_p" not in pair_label:
        raise ValueError(f"paired E label lacks paraphrase: {pair_label!r}")
    value = "p" + pair_label.rsplit("_p", 1)[1]
    if value not in {"p0", "p1"}:
        raise ValueError(f"paired E paraphrase is invalid: {value!r}")
    return value


def _base_dataset(
    cls: type[ScenePlanP11Dataset],
    config: dict[str, Any],
    tokenizer: Any,
) -> ScenePlanP11Dataset:
    source = config["datasets"][0]
    kwargs: dict[str, Any] = {
        "index_path": source["path"],
        "codec_path": config["codec_path"],
        "tokenizer_spec": (tokenizer, 512, None),
        "expected_num_samples": int(config["expected_num_samples"]),
        "index_num_samples": int(config["index_num_samples"]),
        "require_frozen": bool(config.get("require_complete", True)),
        "semantic_cache_path": config["semantic_cache_path"],
        "semantic_dim": int(config.get("semantic_dim", 512)),
        "semantic_encoder_revision": config["semantic_encoder_revision"],
    }
    if cls is ScenePlanP11V4Dataset:
        kwargs.update(
            {
                "lexical_evidence_mode": str(
                    config.get("lexical_evidence_mode", "none")
                ),
                "lexical_max_tokens": int(
                    config.get("lexical_max_tokens", 128)
                ),
                "lexical_cache_path": config.get("lexical_cache_path"),
                "lexical_encoder_revision": config.get(
                    "lexical_encoder_revision"
                ),
                "lexical_confidence_threshold": config.get(
                    "lexical_confidence_threshold"
                ),
            }
        )
    return cls(config["manifest_path"], **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curriculum", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument("--v4-dataset-config", type=Path, default=DEFAULT_V4_DATASET)
    parser.add_argument("--d0-dataset-config", type=Path, default=DEFAULT_D0_DATASET)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runtime-scenes", type=int, default=2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.runtime_scenes <= 0:
        raise ValueError("--runtime-scenes must be positive")

    curriculum_path = args.curriculum.expanduser().resolve(strict=True)
    v4_config = load_config(args.v4_dataset_config.expanduser().resolve(strict=True))
    d0_config = load_config(args.d0_dataset_config.expanduser().resolve(strict=True))
    model_config = load_config(args.model_config.expanduser().resolve(strict=True))
    connection = _readonly(curriculum_path)
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    base_scenes = int(metadata.get("base_scenes", 0))
    expected_rows = base_scenes * 27
    expected_task_rows = base_scenes * 9
    if base_scenes <= 0:
        raise RuntimeError("curriculum lacks a positive base_scenes count")
    for config in (v4_config, d0_config):
        if Path(config["p11_v4_curriculum_path"]).resolve() != curriculum_path:
            raise RuntimeError("dataset config points at a different curriculum")
        if int(config["p11_v4_curriculum_expected_rows"]) != expected_rows:
            raise RuntimeError("dataset config curriculum row count changed")

    required = {
        "schema": P11_V4_CURRICULUM_SCHEMA,
        "schema_version": str(P11_V4_CURRICULUM_VERSION),
        "contract": P11_V4_CURRICULUM_CONTRACT,
        "partition": P11_V4_CURRICULUM_PARTITION,
        "rows": str(expected_rows),
        "base_scenes": str(base_scenes),
        "rows_per_base_scene": "27",
        "rows_per_task_per_base_scene": "9",
        "generation_targets_per_underspecified_prompt": "4",
        "generation_single_target_for_underspecified_prompt": "false",
        "editing_uses_complete_p10_atomic_numeric_vocabulary": "true",
        "editing_instruction_surface_contract": (
            "paired_train_only_multisurface_semantic_direction_exact_p10_numeric_v3"
        ),
        "editing_instruction_surface_count": "8",
        "editing_instruction_surfaces_per_pair_identity": "2",
        "heldout_edit_prompt_exact_overlap": "0",
        "eval_reserved_templates_present": "false",
        "heldout_sample_overlap": "0",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"curriculum metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    ordering_contract = metadata.get("ordering_contract")
    if ordering_contract not in {
        None,
        "p11_v4_pair_aware_interleaved_batch8_diverse_instruction_surface_v6",
        "p11_v4_ddp8_rank_balanced_pair_aware_batch8_v7",
    }:
        raise RuntimeError(f"unknown curriculum ordering contract {ordering_contract!r}")
    if ordering_contract is not None:
        if metadata.get("ordering_batch_size") != "8":
            raise RuntimeError("pair-aware curriculum is not frozen at batch size 8")
        for config in (v4_config, d0_config):
            if config.get("p11_v4_curriculum_ordering_contract") != ordering_contract:
                raise RuntimeError("dataset config ordering contract disagrees with SQLite")
            if int(config.get("p11_v4_curriculum_ordering_batch_size", -1)) != 8:
                raise RuntimeError("dataset config ordering batch size changed")
    for key in (
        "source_manifest",
        "source_index",
        "codec_path",
        "heldout_challenge",
        "builder",
        "p10_checkpoint",
    ):
        path = Path(metadata[key]).resolve(strict=True)
        expected_hash = metadata.get(f"{key}_sha256")
        if expected_hash is not None and _sha256_file(path) != expected_hash:
            raise RuntimeError(f"curriculum provenance hash changed for {key}")

    count, minimum, maximum = connection.execute(
        "SELECT COUNT(*),MIN(ordinal),MAX(ordinal) FROM rows"
    ).fetchone()
    if (int(count), int(minimum), int(maximum)) != (
        expected_rows,
        0,
        expected_rows - 1,
    ):
        raise RuntimeError("curriculum ordinals are incomplete")
    duplicate_ids = int(
        connection.execute(
            "SELECT COUNT(*)-COUNT(DISTINCT curriculum_id) FROM rows"
        ).fetchone()[0]
    )
    if duplicate_ids:
        raise RuntimeError("curriculum IDs are not unique")
    forbidden_templates = int(
        connection.execute(
            "SELECT COUNT(*) FROM rows WHERE template_id LIKE 'eval_reserved/%'"
        ).fetchone()[0]
    )
    if forbidden_templates:
        raise RuntimeError("held-out template ID entered training")

    source_manifest = _readonly(Path(metadata["source_manifest"]))
    source_index = _readonly(Path(metadata["source_index"]))
    heldout = _readonly(Path(metadata["heldout_challenge"]))
    heldout_ids = {
        str(row[0]) for row in heldout.execute("SELECT DISTINCT sample_id FROM rows")
    }
    heldout_plan_hashes = {
        _json_sha(_decode(row[0]))
        for row in heldout.execute("SELECT target_sceneplan_zlib FROM rows")
    }
    eval_template_ids = set(
        json.loads(
            dict(heldout.execute("SELECT key,value FROM metadata"))[
                "eval_reserved_template_ids_json"
            ]
        )
    )
    codec = load_model_sceneplan_codec(metadata["codec_path"])
    sketch_codec = SceneSketchCodec(codec)
    patch_codec = ScenePlanEditPatchCodec(codec)
    delta_codec = DeltaSceneSketchCodec(codec, patch_codec)

    rows = connection.execute(
        """
        SELECT ordinal,curriculum_id,task,family,view_id,template_id,
               base_manifest_ordinal,base_target_ordinal,sample_id,prompt,
               known_field_groups_json,target_sceneplan_zlib,edit_spec_json,
               evidence_transform_json,target_variant,pair_id,pair_label
        FROM rows ORDER BY ordinal
        """
    ).fetchall()
    pair_ordering_audit: dict[str, Any] | None = None
    if ordering_contract == "p11_v4_ddp8_rank_balanced_pair_aware_batch8_v7":
        world_size = int(metadata.get("ordering_world_size", 0))
        local_batch_size = int(metadata.get("ordering_batch_size", 0))
        global_batch_size = int(metadata.get("ordering_global_batch_size", 0))
        if (world_size, local_batch_size, global_batch_size) != (8, 8, 64):
            raise RuntimeError("DDP-aware curriculum is not frozen at 8x8=64")
        if len(rows) % global_batch_size:
            raise RuntimeError("DDP-aware curriculum has a partial optimizer step")

        source_curriculum_path = Path(metadata["source_curriculum"]).resolve(
            strict=True
        )
        if _sha256_file(source_curriculum_path) != metadata.get(
            "source_curriculum_sha256"
        ):
            raise RuntimeError("DDP-aware source curriculum hash changed")
        source_curriculum = _readonly(source_curriculum_path)
        source_rows = source_curriculum.execute(
            """
            SELECT ordinal,curriculum_id,task,family,view_id,template_id,
                   base_manifest_ordinal,base_target_ordinal,sample_id,prompt,
                   known_field_groups_json,target_sceneplan_zlib,edit_spec_json,
                   evidence_transform_json,target_variant,pair_id,pair_label
            FROM rows ORDER BY ordinal
            """
        ).fetchall()
        source_curriculum.close()
        source_by_id = {str(row[1]): row[1:] for row in source_rows}
        candidate_by_id = {str(row[1]): row[1:] for row in rows}
        if source_by_id != candidate_by_id:
            raise RuntimeError(
                "DDP-aware curriculum changed data instead of only reordering it"
            )

        expected_instances = {
            f"{row[15]}/{_pair_paraphrase(str(row[16]))}"
            for row in rows
            if str(row[2]) == "editing" and row[15] is not None
        }
        covered_instances: set[str] = set()
        rank_task_counts = [Counter() for _ in range(world_size)]
        local_task_min = {
            task: local_batch_size
            for task in ("generation", "understanding", "editing")
        }
        local_task_max = {task: 0 for task in local_task_min}
        complete_pair_min = local_batch_size
        complete_pair_max = 0
        local_batches_with_all_tasks = 0
        local_batches_with_complete_pair = 0
        global_steps = len(rows) // global_batch_size
        for global_step in range(global_steps):
            block_start = global_step * global_batch_size
            block = rows[block_start : block_start + global_batch_size]
            for rank in range(world_size):
                # Exact index sequence emitted by DistributedSampler with
                # shuffle=False: rank, rank+8, ..., rank+56 in this block.
                local = [
                    block[rank + world_size * position]
                    for position in range(local_batch_size)
                ]
                counts = Counter(str(row[2]) for row in local)
                if set(counts) != {"generation", "understanding", "editing"}:
                    raise RuntimeError(
                        f"global step {global_step} rank {rank} lacks a task: {counts}"
                    )
                local_batches_with_all_tasks += 1
                rank_task_counts[rank].update(counts)
                for task in local_task_min:
                    local_task_min[task] = min(local_task_min[task], counts[task])
                    local_task_max[task] = max(local_task_max[task], counts[task])

                grouped: dict[tuple[str, str], set[int]] = defaultdict(set)
                paired_rows = 0
                for row in local:
                    if str(row[2]) != "editing" or row[15] is None:
                        continue
                    paired_rows += 1
                    identity = (str(row[15]), _pair_paraphrase(str(row[16])))
                    grouped[identity].add(_edit_direction(str(row[12])))
                complete = [
                    identity for identity, signs in grouped.items()
                    if signs == {-1, 1}
                ]
                if 2 * len(complete) != paired_rows:
                    raise RuntimeError(
                        f"global step {global_step} rank {rank} split an E pair"
                    )
                if not complete:
                    raise RuntimeError(
                        f"global step {global_step} rank {rank} has no complete E pair"
                    )
                local_batches_with_complete_pair += 1
                complete_pair_min = min(complete_pair_min, len(complete))
                complete_pair_max = max(complete_pair_max, len(complete))
                covered_instances.update(
                    f"{pair_id}/{paraphrase}"
                    for pair_id, paraphrase in complete
                )

        expected_pair_instances = base_scenes * 4
        expected_rank_task_count = expected_task_rows // world_size
        expected_rank_counts = {
            "generation": expected_rank_task_count,
            "understanding": expected_rank_task_count,
            "editing": expected_rank_task_count,
        }
        if expected_task_rows % world_size:
            raise RuntimeError("task rows are not rank-divisible")
        if (
            len(expected_instances) != expected_pair_instances
            or covered_instances != expected_instances
        ):
            raise RuntimeError("DDP rank layout does not cover every E pair instance")
        if any(dict(counts) != expected_rank_counts for counts in rank_task_counts):
            raise RuntimeError(
                "DDP rank task counts are not identical and balanced: "
                f"{rank_task_counts}"
            )
        local_batches = global_steps * world_size
        metadata_expectations = {
            "global_optimizer_steps": str(global_steps),
            "ddp_local_batches": str(local_batches),
            "ddp_local_batches_with_all_tasks": str(local_batches),
            "ddp_local_batches_with_complete_pair": str(local_batches),
            "ddp_local_complete_pair_instances_min": str(complete_pair_min),
            "ddp_local_complete_pair_instances_max": str(complete_pair_max),
            "ddp_rank_task_counts_identical": "true",
            "distributed_sampler_contract": (
                "strided_shuffle_false_drop_last_false_v1"
            ),
        }
        for key, expected in metadata_expectations.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"DDP-aware metadata {key}={metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )
        recorded_rank_counts = json.loads(metadata["ddp_rank_task_counts_json"])
        normalized_rank_counts = [dict(counts) for counts in rank_task_counts]
        if recorded_rank_counts != normalized_rank_counts:
            raise RuntimeError("DDP rank-count metadata disagrees with row layout")
        pair_ordering_audit = {
            "contract": ordering_contract,
            "world_size": world_size,
            "local_batch_size": local_batch_size,
            "global_batch_size": global_batch_size,
            "global_optimizer_steps": global_steps,
            "local_batches": local_batches,
            "local_batches_with_all_tasks": local_batches_with_all_tasks,
            "local_batches_with_complete_pair": (
                local_batches_with_complete_pair
            ),
            "local_task_count_min": local_task_min,
            "local_task_count_max": local_task_max,
            "local_complete_pair_instances_min": complete_pair_min,
            "local_complete_pair_instances_max": complete_pair_max,
            "rank_task_counts": normalized_rank_counts,
            "pair_paraphrase_instances": len(covered_instances),
            "source_row_content_exactly_preserved": True,
            "distributed_sampler_contract": (
                "strided_shuffle_false_drop_last_false_v1"
            ),
        }
    if ordering_contract == (
        "p11_v4_pair_aware_interleaved_batch8_diverse_instruction_surface_v6"
    ):
        source_curriculum_path = Path(metadata["source_curriculum"]).resolve(strict=True)
        if _sha256_file(source_curriculum_path) != metadata.get(
            "source_curriculum_sha256"
        ):
            raise RuntimeError("pair-aware source curriculum hash changed")
        source_curriculum = _readonly(source_curriculum_path)
        source_rows = source_curriculum.execute(
            """
            SELECT ordinal,curriculum_id,task,family,view_id,template_id,
                   base_manifest_ordinal,base_target_ordinal,sample_id,prompt,
                   known_field_groups_json,target_sceneplan_zlib,edit_spec_json,
                   evidence_transform_json,target_variant,pair_id,pair_label
            FROM rows ORDER BY ordinal
            """
        ).fetchall()
        source_curriculum.close()
        source_by_id = {str(row[1]): row[1:] for row in source_rows}
        candidate_by_id = {str(row[1]): row[1:] for row in rows}
        if source_by_id != candidate_by_id:
            raise RuntimeError(
                "pair-aware curriculum changed data instead of only reordering it"
            )

        expected_instances: set[str] = set()
        for row in rows:
            if str(row[2]) == "editing" and row[15] is not None:
                expected_instances.add(
                    f"{row[15]}/{_pair_paraphrase(str(row[16]))}"
                )
        covered_instances: set[str] = set()
        pair_batches = 0
        consecutive_batches_without_pair = 0
        max_consecutive_batches_without_pair = 0
        last_pair_batch_index = -1
        for start in range(0, len(rows) // 8 * 8, 8):
            batch = rows[start : start + 8]
            grouped: dict[tuple[str, str], set[int]] = defaultdict(set)
            for row in batch:
                if str(row[2]) != "editing" or row[15] is None:
                    continue
                identity = (
                    str(row[15]), _pair_paraphrase(str(row[16]))
                )
                grouped[identity].add(_edit_direction(str(row[12])))
            usable = [identity for identity, signs in grouped.items() if signs == {-1, 1}]
            if usable:
                pair_batches += 1
                consecutive_batches_without_pair = 0
                last_pair_batch_index = start // 8
                if len(usable) != 2 or Counter(str(row[2]) for row in batch) != {
                    "generation": 2,
                    "understanding": 2,
                    "editing": 4,
                }:
                    raise RuntimeError("pair-aware full batch composition changed")
                covered_instances.update(
                    f"{pair_id}/{paraphrase}"
                    for pair_id, paraphrase in usable
                )
            else:
                consecutive_batches_without_pair += 1
                max_consecutive_batches_without_pair = max(
                    max_consecutive_batches_without_pair,
                    consecutive_batches_without_pair,
                )
        tail = rows[len(rows) // 8 * 8 :]
        if any(str(row[2]) == "editing" and row[15] is not None for row in tail):
            raise RuntimeError("paired E row entered the dropped batch tail")
        expected_pair_instances = base_scenes * 4
        expected_pair_batches = base_scenes * 2
        expected_full_batches = expected_rows // 8
        expected_remainder_batches = expected_full_batches - expected_pair_batches
        expected_last_pair_batch = expected_full_batches - 1
        if (
            len(expected_instances) != expected_pair_instances
            or covered_instances != expected_instances
        ):
            raise RuntimeError(
                "pair-aware order does not cover every E pair/paraphrase instance"
            )
        if pair_batches != expected_pair_batches:
            raise RuntimeError(f"pair-aware batch count changed: {pair_batches}")
        if max_consecutive_batches_without_pair != 1:
            raise RuntimeError(
                "pair-aware interleave permits consecutive full batches without "
                "paired Editing supervision"
            )
        if last_pair_batch_index != expected_last_pair_batch:
            raise RuntimeError("pair-aware interleave no longer reaches epoch end")
        metadata_expectations = {
            "pair_aware_full_batches": str(expected_pair_batches),
            "remainder_full_batches": str(expected_remainder_batches),
            "full_batches": str(expected_full_batches),
            "max_consecutive_full_batches_without_paired_edit": "1",
            "last_pair_aware_full_batch_index": str(expected_last_pair_batch),
        }
        for key, expected in metadata_expectations.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"pair-aware metadata {key}={metadata.get(key)!r}, "
                    f"expected {expected!r}"
                )
        pair_ordering_audit = {
            "contract": ordering_contract,
            "batch_size": 8,
            "pair_aware_full_batches": pair_batches,
            "pair_identities": len(
                {value.rsplit("/", 1)[0] for value in covered_instances}
            ),
            "pair_paraphrase_instances": len(covered_instances),
            "remainder_full_batches": expected_remainder_batches,
            "full_batches": expected_full_batches,
            "max_consecutive_full_batches_without_paired_edit": (
                max_consecutive_batches_without_pair
            ),
            "last_pair_aware_full_batch_index": last_pair_batch_index,
            "paired_rows_in_dropped_tail": 0,
            "source_row_content_exactly_preserved": True,
        }
    task_order = ("generation", "understanding", "editing")
    task_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    prompt_token_max = 0
    train_plan_hashes: set[str] = set()
    train_ids: set[str] = set()
    g_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    e_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    u_transforms: Counter[str] = Counter()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"]["text"]["model_path"],
        local_files_only=True,
        use_fast=True,
    )
    for row in rows:
        (
            ordinal,
            curriculum_id,
            task,
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
        if ordering_contract is None and task != task_order[int(ordinal) % 3]:
            raise RuntimeError("curriculum G/U/E interleave changed")
        if str(template_id) in eval_template_ids:
            raise RuntimeError("held-out reserved template entered training")
        task_counts[str(task)] += 1
        family_counts[f"{task}:{family}"] += 1
        train_ids.add(str(sample_id))
        if str(sample_id) in heldout_ids:
            raise RuntimeError("held-out sample entered training")
        encoded_prompt = tokenizer(
            str(prompt), truncation=False, padding=False, add_special_tokens=True
        )
        prompt_token_max = max(prompt_token_max, len(encoded_prompt["input_ids"]))
        if len(encoded_prompt["input_ids"]) > 512:
            raise RuntimeError("curriculum prompt exceeds the no-truncation ceiling")
        target = codec.project_plan(_decode(target_payload), sample_id=str(sample_id))
        validate_p11_executor_profile(target)
        plan_tokens = codec.encode(target)["input_ids"]
        if codec.decode(plan_tokens, sample_id=str(sample_id)) != target:
            raise RuntimeError("curriculum target failed codec round-trip")
        target_hash = _json_sha(target)
        train_plan_hashes.add(target_hash)
        known_groups = json.loads(str(known_json))
        evidence = json.loads(str(evidence_json))
        if str(evidence.get("contract")) != (
            "synthetic_representation_stress_v1"
        ):
            raise RuntimeError("curriculum evidence contract changed")
        if task == "generation" and family == "generation_multitarget_posterior":
            if pair_id is None or edit_json is not None:
                raise RuntimeError("G multitarget row has malformed authority")
            g_groups[str(pair_id)].append(
                {
                    "prompt": str(prompt),
                    "template": str(template_id),
                    "known": known_groups,
                    "target": target,
                    "sketch_tokens": sketch_codec.encode(
                        compile_scene_sketch(target, codec)
                    )["input_ids"],
                    "core": execution_state_core(
                        compile_execution_state(target, codec)
                    ),
                    "variant": int(target_variant),
                }
            )
        elif task == "understanding":
            transform_id = str(evidence.get("transform_id"))
            u_transforms[transform_id] += 1
            valid_frames = int(
                source_index.execute(
                    "SELECT latent_frames_valid FROM samples WHERE ordinal=?",
                    (int(target_ordinal),),
                ).fetchone()[0]
            )
            time_mask = evidence.get("foa_time_mask")
            if time_mask is not None and not (
                0 <= int(time_mask[0]) < int(time_mask[1]) <= valid_frames
            ):
                raise RuntimeError("U train FOA mask exceeds valid evidence")
            channel_mask = evidence.get("semantic_channel_mask")
            if channel_mask is not None and not (
                0 <= int(channel_mask[0]) < int(channel_mask[1]) <= 512
            ):
                raise RuntimeError("U train CLAP mask exceeds feature width")
            if transform_id.startswith(("foa_time_mask_", "clap_channel_mask_", "combined_foa")):
                raise RuntimeError("held-out U transform ID entered training")
        elif task == "editing":
            if edit_json is None:
                raise RuntimeError("E curriculum row lacks edit spec")
            edit_spec = json.loads(str(edit_json))
            manifest_task = source_manifest.execute(
                "SELECT task,target_ordinal FROM rows WHERE ordinal=?",
                (int(base_ordinal),),
            ).fetchone()
            if manifest_task != ("editing", int(target_ordinal)):
                raise RuntimeError("E curriculum base row changed")
            source_payload = source_index.execute(
                "SELECT scene_plan_zlib FROM samples WHERE ordinal=?",
                (int(target_ordinal),),
            ).fetchone()[0]
            current = canonicalize_sceneplan_source_ids(
                codec.project_plan(_decode(source_payload), sample_id=str(sample_id))
            )
            patch_codec.assert_target(current, edit_spec, target)
            if family == "editing_numeric_delta_curriculum":
                if pair_id is None:
                    raise RuntimeError("numeric E curriculum lacks pair ID")
                operation = str(edit_spec["operation"])
                if operation == "rotate_source" and abs(
                    int(edit_spec["delta_azimuth_deg"])
                ) != 45:
                    raise RuntimeError("E rotation escaped P10 atomic vocabulary")
                if operation == "distance_source" and float(
                    edit_spec["distance_factor"]
                ) not in {0.75, 1.25}:
                    raise RuntimeError("E distance escaped P10 atomic vocabulary")
                delta = compile_delta_scene_sketch(
                    current, target, edit_spec, codec
                )
                e_groups[str(pair_id)].append(
                    {
                        "prompt": str(prompt),
                        "target": target,
                        "spec": edit_spec,
                        "delta_tokens": delta_codec.encode(delta, edit_spec)[
                            "input_ids"
                        ],
                        "variant": int(target_variant),
                        "label": str(pair_label),
                    }
                )

    if task_counts != {task: expected_task_rows for task in task_order}:
        raise RuntimeError(f"curriculum task balance changed: {task_counts}")
    if train_ids & heldout_ids:
        raise RuntimeError("train/heldout sample leakage detected")
    if train_plan_hashes & heldout_plan_hashes:
        raise RuntimeError("train/heldout target ScenePlan leakage detected")
    if len(g_groups) != base_scenes * 2:
        raise RuntimeError("curriculum must have two G posterior groups per scene")
    g_numeric_pairwise = []
    for pair_id, values in g_groups.items():
        if len(values) != 4 or {value["variant"] for value in values} != set(range(4)):
            raise RuntimeError(f"G group {pair_id} lacks four target variants")
        if len({value["prompt"] for value in values}) != 1:
            raise RuntimeError("G multi-target group changed its prompt")
        if len({value["template"] for value in values}) != 1:
            raise RuntimeError("G multi-target group changed its template")
        first_tokens = values[0]["sketch_tokens"]
        if not all(torch.equal(first_tokens, value["sketch_tokens"]) for value in values[1:]):
            raise RuntimeError("G numeric targets changed discrete SceneSketch")
        if len({_json_sha(value["target"]) for value in values}) != 4:
            raise RuntimeError("G multi-target plans are not unique")
        for left_index in range(4):
            for right_index in range(left_index + 1, 4):
                rmse = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                values[left_index]["core"]
                                - values[right_index]["core"]
                            )
                        )
                    )
                )
                g_numeric_pairwise.append(rmse)
                if not rmse > 0.0:
                    raise RuntimeError("G multi-target numeric cores collapsed")
        # Pair-aware ordering deliberately scatters the four completions across
        # the SQLite ordinal stream, so list position is not target authority.
        # Variant zero is the immutable prompt anchor from which the other
        # legal completions were constructed.
        hidden = next(
            value["target"] for value in values if value["variant"] == 0
        )
        for value in values:
            score = score_generation_constraints(
                hidden,
                value["target"],
                known_field_groups=value["known"],
                source_matching="permutation_invariant",
            )
            if not math.isclose(float(score["task_score"]), 1.0, abs_tol=1e-12):
                raise RuntimeError(
                    "G completion violates its prompt constraints: "
                    f"pair={pair_id}, variant={value['variant']}, "
                    f"known={value['known']}, score={score['task_score']}"
                )

    if len(e_groups) != base_scenes * 2:
        raise RuntimeError("curriculum must have two E numeric groups per scene")
    for pair_id, values in e_groups.items():
        if len(values) != 4 or {value["variant"] for value in values} != set(range(4)):
            raise RuntimeError(f"E group {pair_id} lacks four curriculum variants")
        direction_groups: dict[int, list[torch.Tensor]] = defaultdict(list)
        for value in values:
            program = delta_codec.decode(value["delta_tokens"])
            direction_groups[int(program.get("control_direction", 0))].append(
                value["delta_tokens"]
            )
        if set(direction_groups) != {-1, 1}:
            raise RuntimeError("E group lacks both DeltaSketch control directions")
        for direction, tokens in direction_groups.items():
            if len(tokens) != 2 or not torch.equal(tokens[0], tokens[1]):
                raise RuntimeError(
                    f"E direction {direction:+d} changed across paraphrases"
                )
        negative = direction_groups[-1][0]
        positive = direction_groups[1][0]
        if not (
            int(negative.numel()) == 5
            and torch.equal(negative[:3], positive[:3])
            and int(negative[3]) != int(positive[3])
            and torch.equal(negative[4:], positive[4:])
        ):
            raise RuntimeError(
                "E pair must differ only at the operation-specific direction token"
            )
        if len({_json_sha(value["target"]) for value in values}) != 2:
            raise RuntimeError("E group must contain two legal numeric targets")
        spec_counts = Counter(_json_sha(value["spec"]) for value in values)
        if sorted(spec_counts.values()) != [2, 2]:
            raise RuntimeError("E legal target lacks two train-only paraphrases")

    # Materialize both graph families through the same overlay.  Two complete
    # 27-row scenes exercise every curriculum family without turning this gate
    # into a training run.
    runtime_rows = min(expected_rows, args.runtime_scenes * 27)
    runtime: dict[str, Any] = {}
    for name, cls, config in (
        ("v4", ScenePlanP11V4Dataset, v4_config),
        ("d0", ScenePlanP11Dataset, d0_config),
    ):
        base = _base_dataset(cls, config, tokenizer)
        overlay = ScenePlanP11V4CurriculumDataset(
            base,
            curriculum_path,
            expected_rows=expected_rows,
            expected_ordering_contract=config.get(
                "p11_v4_curriculum_ordering_contract"
            ),
            expected_ordering_batch_size=config.get(
                "p11_v4_curriculum_ordering_batch_size"
            ),
        )
        output_kinds: Counter[str] = Counter()
        evidence_changes = 0
        for index in range(runtime_rows):
            carrier, row = overlay[index]
            if tuple(carrier.shape) != (64, 648) or not bool(torch.isfinite(carrier).all()):
                raise RuntimeError(f"{name} runtime carrier is invalid")
            output_kinds[str(row["p11_output_kind"])] += 1
            transform = row["p11_curriculum_evidence_transform"]
            if transform["synthetic_representation_stress"]:
                evidence_changes += int(transform["before"] != transform["after"])
        runtime[name] = {
            "rows": runtime_rows,
            "output_kinds": dict(sorted(output_kinds.items())),
            "degraded_rows_with_changed_evidence": evidence_changes,
        }

    report = {
        "schema": "stable_audio_tools.p11_v4_curriculum_validation",
        "schema_version": 1,
        "status": "PASS",
        "curriculum": str(curriculum_path),
        "curriculum_sha256": _sha256_file(curriculum_path),
        "contract": P11_V4_CURRICULUM_CONTRACT,
        "rows": expected_rows,
        "base_scenes": base_scenes,
        "task_counts": dict(sorted(task_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "prompt_max_tokens": prompt_token_max,
        "truncation": 0,
        "generation": {
            "multitarget_groups": len(g_groups),
            "targets_per_group": 4,
            "min_numeric_pairwise_rmse": min(g_numeric_pairwise),
            "single_target_underspecified_prompts": 0,
            "discrete_semantic_authority_identical_within_group": True,
        },
        "understanding": {
            "transform_ids": dict(sorted(u_transforms.items())),
            "scope": "synthetic representation stress; not real acoustic corruption",
        },
        "editing": {
            "numeric_groups": len(e_groups),
            "rows_per_group": 4,
            "unique_targets_per_group": 2,
            "complete_p10_atomic_numeric_vocabulary": True,
            "delta_sketch_operation_owner_identical_within_group": True,
            "delta_sketch_opposite_control_directions": True,
            "direction_is_the_only_changed_delta_token": True,
        },
        "ordering": pair_ordering_audit,
        "leakage": {
            "heldout_sample_overlap": 0,
            "heldout_target_sceneplan_overlap": 0,
            "heldout_reserved_template_overlap": 0,
        },
        "runtime": runtime,
        "p10_alignment": {
            "checkpoint": metadata["p10_checkpoint"],
            "checkpoint_sha256": metadata["p10_checkpoint_sha256"],
            "max_latent_frames": 648,
            "source_count": "1-4",
            "motion": ["static", "linear"],
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
