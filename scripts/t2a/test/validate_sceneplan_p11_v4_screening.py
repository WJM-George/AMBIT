#!/usr/bin/env python3
"""Strict gate for the canonical P11-v4 matched 10k-scene screening data."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import zlib
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
    EXECUTION_STATE_CONTRACT,
    SceneSketchCodec,
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
    P11_V4_CURRICULUM_PARTITION,
    P11_V4_CURRICULUM_SCHEMA,
    P11_V4_CURRICULUM_VERSION,
    P11_V4_SCREENING_CONTRACT,
    ScenePlanP11V4CurriculumDataset,
)
from stable_audio_tools.data.sceneplan_p11_v4_dataset import (  # noqa: E402
    P11_V4_DATA_CONTRACT,
    P11_V4_SEQUENCE_CONTRACT,
    ScenePlanP11V4Dataset,
)
from stable_audio_tools.data.sceneplan_p11_dataset import (  # noqa: E402
    ScenePlanP11Dataset,
)


ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2")
DEFAULT_CURRICULUM = (
    ROOT / "p11_v4_curriculum/p11_train_trial30k_matched_screening_v1.sqlite"
)
DEFAULT_V4_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_trial30k_matched_screening_v1_transfusion_cot_v4_reliable_asr.json"
)
DEFAULT_D0_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_trial30k_matched_screening_v1_discrete_d0.json"
)
DEFAULT_MODEL = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/p11_v4_matched_screening30k_validation_20260902.json"
)
ORDERING_CONTRACT = "p11_v4_matched_triplet_interleaved_batch8_v1"
P10_CHECKPOINT = Path(
    "/mnt/sdc/ckpts/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/"
    "checkpoints/epoch=48-step=150000.ckpt"
)
P10_CHECKPOINT_SHA256 = (
    "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
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


def _normalized(value: str) -> str:
    return " ".join(str(value).split())


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


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
                "lexical_max_tokens": int(config.get("lexical_max_tokens", 128)),
                "lexical_cache_path": config.get("lexical_cache_path"),
                "lexical_encoder_revision": config.get("lexical_encoder_revision"),
                "lexical_confidence_threshold": config.get(
                    "lexical_confidence_threshold"
                ),
            }
        )
    return cls(config["manifest_path"], **kwargs)


def _edit_direction(spec: dict[str, Any]) -> int | None:
    if spec["operation"] == "rotate_source":
        return -1 if int(spec["delta_azimuth_deg"]) < 0 else 1
    if spec["operation"] == "distance_source":
        return -1 if float(spec["distance_factor"]) < 1.0 else 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, default=DEFAULT_CURRICULUM)
    parser.add_argument("--v4-dataset-config", type=Path, default=DEFAULT_V4_DATASET)
    parser.add_argument("--d0-dataset-config", type=Path, default=DEFAULT_D0_DATASET)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runtime-scenes", type=int, default=3)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Run structural checks before the frozen-ASR cache exists.",
    )
    args = parser.parse_args()
    if args.runtime_scenes <= 0:
        raise ValueError("--runtime-scenes must be positive")

    curriculum_path = args.curriculum.expanduser().resolve(strict=True)
    v4_config = load_config(args.v4_dataset_config.expanduser().resolve(strict=True))
    d0_config = load_config(args.d0_dataset_config.expanduser().resolve(strict=True))
    model_config = load_config(args.model_config.expanduser().resolve(strict=True))
    for config in (v4_config, d0_config):
        expected = {
            "p11_v4_curriculum_path": str(curriculum_path),
            "p11_v4_curriculum_expected_rows": 30_000,
            "p11_v4_curriculum_contract": P11_V4_SCREENING_CONTRACT,
            "p11_v4_curriculum_ordering_contract": ORDERING_CONTRACT,
            "p11_v4_curriculum_ordering_batch_size": 8,
            "expected_num_samples": 30_000,
            "drop_last": True,
            "require_exact_batch_size": True,
            "shuffle": False,
            "in_order": True,
        }
        for key, value in expected.items():
            actual = config.get(key)
            if key == "p11_v4_curriculum_path":
                actual = str(Path(str(actual)).resolve())
            if actual != value:
                raise RuntimeError(
                    f"dataset config {key}={actual!r}, expected {value!r}"
                )
    if d0_config.get("p11_contract") != "discrete_d0_v1":
        raise RuntimeError("D0 config changed architecture")
    if v4_config.get("p11_contract") != P11_V4_DATA_CONTRACT:
        raise RuntimeError("v4 config changed architecture")
    if v4_config.get("lexical_evidence_mode") != "frozen_asr_cache_v1":
        raise RuntimeError("v4 screening must use reliable frozen ASR")
    if float(v4_config.get("lexical_confidence_threshold", -1.0)) != 0.85:
        raise RuntimeError("v4 screening frozen-ASR threshold must remain 0.85")
    if int(v4_config.get("lexical_max_tokens", -1)) != 128:
        raise RuntimeError("v4 screening lexical ceiling must remain 128 tokens")

    connection = _readonly(curriculum_path)
    if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise RuntimeError("screening SQLite integrity check failed")
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    required_metadata = {
        "schema": P11_V4_CURRICULUM_SCHEMA,
        "schema_version": str(P11_V4_CURRICULUM_VERSION),
        "contract": P11_V4_SCREENING_CONTRACT,
        "partition": P11_V4_CURRICULUM_PARTITION,
        "rows": "30000",
        "base_scenes": "10000",
        "rows_per_base_scene": "3",
        "rows_per_task_per_base_scene": "1",
        "ordering_contract": ORDERING_CONTRACT,
        "ordering_batch_size": "8",
        "drop_last_rows": "0",
        "core40_sidecar_used": "false",
        "runtime_execution_state": "p10_execution_state_core15_v1_[5,15]",
        "eval_reserved_templates_present": "false",
        "heldout_sample_overlap": "0",
        "heldout_prompt_exact_overlap": "0",
        "p10_checkpoint": str(P10_CHECKPOINT),
        "p10_checkpoint_sha256": P10_CHECKPOINT_SHA256,
        "p10_max_latent_frames": "648",
        "p10_duration_limit_sec": "15.0465",
        "p10_source_count": "1-4",
        "p10_motion_profile": "static,linear",
        "p11_v4_data_contract": P11_V4_DATA_CONTRACT,
        "p11_v4_sequence_contract": P11_V4_SEQUENCE_CONTRACT,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"screening metadata {key}={metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
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
            raise RuntimeError(f"screening provenance hash changed for {key}")

    count, minimum, maximum = connection.execute(
        "SELECT COUNT(*),MIN(ordinal),MAX(ordinal) FROM rows"
    ).fetchone()
    if (int(count), int(minimum), int(maximum)) != (30_000, 0, 29_999):
        raise RuntimeError("screening rows are incomplete or non-contiguous")
    duplicate_ids = int(
        connection.execute(
            "SELECT COUNT(*)-COUNT(DISTINCT curriculum_id) FROM rows"
        ).fetchone()[0]
    )
    if duplicate_ids:
        raise RuntimeError("screening curriculum IDs are not unique")

    source_manifest = _readonly(Path(metadata["source_manifest"]))
    source_index = _readonly(Path(metadata["source_index"]))
    heldout = _readonly(Path(metadata["heldout_challenge"]))
    heldout_ids = {
        str(row[0]) for row in heldout.execute("SELECT DISTINCT sample_id FROM rows")
    }
    heldout_prompts = {
        _normalized(row[0]) for row in heldout.execute("SELECT prompt FROM rows")
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

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model"]["text"]["model_path"],
        local_files_only=True,
        use_fast=True,
    )

    rows = connection.execute(
        """
        SELECT ordinal,curriculum_id,task,family,view_id,template_id,
               base_manifest_ordinal,base_target_ordinal,sample_id,prompt,
               known_field_groups_json,target_sceneplan_zlib,edit_spec_json,
               evidence_transform_json,target_variant,pair_id,pair_label
        FROM rows ORDER BY ordinal
        """
    ).fetchall()
    task_order = ("generation", "understanding", "editing")
    task_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    g_views: Counter[str] = Counter()
    u_transforms: Counter[str] = Counter()
    edit_operations: Counter[str] = Counter()
    edit_owners: Counter[str] = Counter()
    edit_directions: Counter[str] = Counter()
    source_counts: Counter[int] = Counter()
    prompt_token_max = 0
    train_plan_hashes: set[str] = set()
    train_ids: set[str] = set()
    current_cache: dict[int, tuple[str, int, dict[str, Any]]] = {}
    triplet_identity: dict[int, tuple[str, int]] = {}

    for row in rows:
        (
            ordinal,
            _curriculum_id,
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
        ordinal = int(ordinal)
        target_ordinal = int(target_ordinal)
        task = str(task)
        sample_id = str(sample_id)
        if task != task_order[ordinal % 3]:
            raise RuntimeError(f"G/U/E interleave changed at row {ordinal}")
        if int(base_ordinal) != ordinal:
            raise RuntimeError("screening no longer maps 1:1 onto source manifest rows")
        manifest_row = source_manifest.execute(
            "SELECT task,target_ordinal FROM rows WHERE ordinal=?", (ordinal,)
        ).fetchone()
        if manifest_row != (task, target_ordinal):
            raise RuntimeError("screening/source manifest row provenance diverged")
        scene_position = ordinal // 3
        previous_identity = triplet_identity.setdefault(
            scene_position, (sample_id, target_ordinal)
        )
        if previous_identity != (sample_id, target_ordinal):
            raise RuntimeError("matched G/U/E triplet changed base scene")

        if target_ordinal not in current_cache:
            index_row = source_index.execute(
                """
                SELECT sample_id,latent_frames_valid,scene_plan_zlib
                FROM samples WHERE ordinal=?
                """,
                (target_ordinal,),
            ).fetchone()
            if index_row is None:
                raise RuntimeError("screening target ordinal absent from source index")
            current_cache[target_ordinal] = (
                str(index_row[0]),
                int(index_row[1]),
                canonicalize_sceneplan_source_ids(
                    codec.project_plan(_decode(index_row[2]), sample_id=sample_id)
                ),
            )
        indexed_id, valid_frames, current = current_cache[target_ordinal]
        if indexed_id != sample_id:
            raise RuntimeError("screening sample ID differs from source index")
        if sample_id in heldout_ids:
            raise RuntimeError("held-out sample entered screening train")
        train_ids.add(sample_id)

        if str(template_id).startswith("eval_reserved/") or str(
            template_id
        ) in eval_template_ids:
            raise RuntimeError("eval-reserved template entered screening train")
        if _normalized(str(prompt)) in heldout_prompts:
            raise RuntimeError("exact held-out prompt entered screening train")
        encoded_prompt = tokenizer(
            str(prompt), truncation=False, padding=False, add_special_tokens=True
        )
        prompt_tokens = len(encoded_prompt["input_ids"])
        prompt_token_max = max(prompt_token_max, prompt_tokens)
        if prompt_tokens > 512:
            raise RuntimeError("screening prompt exceeds no-truncation ceiling")

        target = codec.project_plan(_decode(target_payload), sample_id=sample_id)
        validate_p11_executor_profile(target)
        plan_tokens = codec.encode(target)["input_ids"]
        if codec.decode(plan_tokens, sample_id=sample_id) != target:
            raise RuntimeError("screening target failed codec round-trip")
        core = execution_state_core(compile_execution_state(target, codec))
        if core.shape != (5, 15) or not np.isfinite(core).all():
            raise RuntimeError("screening core15 target is malformed or non-finite")
        target_hash = _json_sha(target)
        train_plan_hashes.add(target_hash)
        task_counts[task] += 1
        family_counts[f"{task}:{family}"] += 1
        source_counts[len(target["sources"])] += 1
        known_groups = json.loads(str(known_json))
        evidence = json.loads(str(evidence_json))
        if evidence.get("contract") != "synthetic_representation_stress_v1":
            raise RuntimeError("screening evidence contract changed")
        if pair_id is not None or pair_label is not None:
            raise RuntimeError("screening unexpectedly carries pilot pair metadata")

        if task == "generation":
            g_views[str(view_id).split("_screen_v1", 1)[0]] += 1
            if edit_json is not None:
                raise RuntimeError("G screening row carries edit authority")
            if family == "exact_compatibility":
                if target != current or int(target_variant) != 0:
                    raise RuntimeError("exact G row changed canonical P10 target")
            elif family == "generation_population_posterior":
                if not 0 <= int(target_variant) < 4:
                    raise RuntimeError("underspecified G target draw escaped K=4")
                if sketch_codec.encode(compile_scene_sketch(target, codec))[
                    "input_ids"
                ].tolist() != sketch_codec.encode(
                    compile_scene_sketch(current, codec)
                )["input_ids"].tolist():
                    raise RuntimeError("numeric G completion changed SceneSketch")
                score = score_generation_constraints(
                    current,
                    target,
                    known_field_groups=known_groups,
                    source_matching="permutation_invariant",
                )
                if not math.isclose(
                    float(score["task_score"]), 1.0, abs_tol=1.0e-12
                ):
                    raise RuntimeError("G completion violates stated constraints")
            else:
                raise RuntimeError(f"unknown screening G family {family!r}")
        elif task == "understanding":
            if target != current or edit_json is not None:
                raise RuntimeError("U row changed source ScenePlan authority")
            transform_id = str(evidence.get("transform_id"))
            u_transforms[transform_id] += 1
            if transform_id.startswith(
                ("foa_time_mask_", "clap_channel_mask_", "combined_foa25")
            ):
                raise RuntimeError("held-out U transform entered training")
            time_mask = evidence.get("foa_time_mask")
            if time_mask is not None and not (
                0 <= int(time_mask[0]) < int(time_mask[1]) <= valid_frames
            ):
                raise RuntimeError("U FOA stress mask exceeds valid evidence")
            channel_mask = evidence.get("semantic_channel_mask")
            if channel_mask is not None and not (
                0 <= int(channel_mask[0]) < int(channel_mask[1]) <= 512
            ):
                raise RuntimeError("U CLAP stress mask exceeds width 512")
        else:
            if edit_json is None:
                raise RuntimeError("E row lacks atomic patch")
            spec = json.loads(str(edit_json))
            patch_codec.assert_target(current, spec, target)
            operation = str(spec["operation"])
            owner = str(spec.get("source_id") or "global")
            edit_operations[operation] += 1
            edit_owners[f"{operation}:{owner}"] += 1
            direction = _edit_direction(spec)
            if direction is not None:
                edit_directions[f"{operation}:{direction:+d}"] += 1
            prompt_lower = _normalized(str(prompt)).lower()
            if any(
                phrase in prompt_lower
                for phrase in ("resulting target", "resulting bin", "target bin")
            ):
                raise RuntimeError("E prompt leaks a resulting target bin")

        if (ordinal + 1) % 5_000 == 0:
            print(
                json.dumps({"event": "progress", "rows": ordinal + 1}),
                flush=True,
            )

    if len(triplet_identity) != 10_000 or len(train_ids) != 10_000:
        raise RuntimeError("screening lacks 10,000 unique matched scenes")
    if task_counts != {task: 10_000 for task in task_order}:
        raise RuntimeError(f"screening task balance changed: {task_counts}")
    if g_views != {"exact": 3334, "numeric_layout": 3333, "coarse_numeric": 3333}:
        raise RuntimeError(f"screening G view balance changed: {g_views}")
    if train_plan_hashes & heldout_plan_hashes:
        raise RuntimeError("held-out target ScenePlan entered screening train")
    if dict(sorted(edit_operations.items())) != json.loads(
        metadata["e_operation_counts"]
    ):
        raise RuntimeError("E operation metadata/content disagree")
    if dict(sorted(edit_owners.items())) != json.loads(
        metadata["e_operation_owner_counts"]
    ):
        raise RuntimeError("E owner metadata/content disagree")
    if dict(sorted(edit_directions.items())) != json.loads(
        metadata["e_direction_counts"]
    ):
        raise RuntimeError("E direction metadata/content disagree")

    full_batches = 0
    batch_task_patterns: Counter[str] = Counter()
    for start in range(0, len(rows), 8):
        batch = rows[start : start + 8]
        if len(batch) != 8:
            raise RuntimeError("30k screening unexpectedly has a batch-8 tail")
        counts = Counter(str(row[2]) for row in batch)
        if sorted(counts.values()) != [2, 3, 3]:
            raise RuntimeError("interleaved screening batch lost near-even G/U/E mix")
        batch_task_patterns["/".join(str(counts[task]) for task in task_order)] += 1
        full_batches += 1
    if full_batches != 3_750:
        raise RuntimeError("screening full batch count changed")

    runtime: dict[str, Any] | None = None
    if not args.skip_runtime:
        runtime = {}
        # Exercise both ends and a few complete triplets.  Structural checks
        # above already cover all 30,000 rows without decoding every latent.
        runtime_indices = list(range(args.runtime_scenes * 3))
        runtime_indices.extend(range(30_000 - args.runtime_scenes * 3, 30_000))
        lexical_path = Path(v4_config["lexical_cache_path"]).resolve(strict=True)
        lexical_connection = _readonly(lexical_path)
        lexical_metadata = dict(
            lexical_connection.execute("SELECT key,value FROM metadata")
        )
        if lexical_metadata.get("source") != "input_foa_only" or (
            lexical_metadata.get("target_transcript_access") != "forbidden"
        ):
            raise RuntimeError("frozen-ASR cache violates its input-only boundary")
        lexical_target = lexical_connection.execute(
            """
            SELECT ordinal FROM hypotheses
            WHERE has_speech=1 AND confidence>=?
            ORDER BY ordinal LIMIT 1
            """,
            (float(v4_config["lexical_confidence_threshold"]),),
        ).fetchone()
        lexical_connection.close()
        if lexical_target is None:
            raise RuntimeError("screening cache has no reliable lexical row")
        lexical_runtime_row = connection.execute(
            """
            SELECT ordinal FROM rows
            WHERE task='understanding' AND base_target_ordinal=?
            """,
            (int(lexical_target[0]),),
        ).fetchone()
        if lexical_runtime_row is None:
            raise RuntimeError("reliable lexical scene is absent from screening")
        runtime_indices.append(int(lexical_runtime_row[0]))
        runtime_indices = sorted(set(runtime_indices))
        for name, cls, config in (
            ("v4", ScenePlanP11V4Dataset, v4_config),
            ("d0", ScenePlanP11Dataset, d0_config),
        ):
            base = _base_dataset(cls, config, tokenizer)
            overlay = ScenePlanP11V4CurriculumDataset(
                base,
                curriculum_path,
                expected_rows=30_000,
                expected_contract=P11_V4_SCREENING_CONTRACT,
                expected_ordering_contract=ORDERING_CONTRACT,
                expected_ordering_batch_size=8,
            )
            output_kinds: Counter[str] = Counter()
            degraded_rows = 0
            lexical_rows = 0
            for index in runtime_indices:
                carrier, item = overlay[index]
                if tuple(carrier.shape) != (64, 648) or not bool(
                    torch.isfinite(carrier).all()
                ):
                    raise RuntimeError(f"{name} runtime carrier is invalid")
                if item["p11_curriculum_contract"] != P11_V4_SCREENING_CONTRACT:
                    raise RuntimeError(f"{name} runtime contract provenance changed")
                target_tokens = item["p11_target_tokens"]["input_ids"]
                if int(target_tokens.numel()) <= 0:
                    raise RuntimeError(f"{name} runtime target token stream is empty")
                output_kinds[str(item["p11_output_kind"])] += 1
                transform = item["p11_curriculum_evidence_transform"]
                degraded_rows += int(
                    transform["synthetic_representation_stress"]
                    and transform["before"] != transform["after"]
                )
                lexical_rows += int(item.get("p11_input_lexical") is not None)
                if name == "v4":
                    core = item["p11_v4_target_execution_core"]
                    if tuple(core.shape) != (5, 15) or not bool(
                        torch.isfinite(core).all()
                    ):
                        raise RuntimeError("v4 runtime core15 is invalid")
                    for retired_key in (
                        "p11_target_scene_thought_core",
                        "p11_input_scene_thought_core",
                        "p11_scene_thought_delta_core",
                    ):
                        if item.get(retired_key) is not None:
                            raise RuntimeError("retired core40 entered v4 runtime")
            runtime[name] = {
                "rows": len(runtime_indices),
                "output_kinds": dict(sorted(output_kinds.items())),
                "degraded_rows_with_changed_evidence": degraded_rows,
                "reliable_lexical_rows": lexical_rows,
            }
            if name == "v4" and lexical_rows <= 0:
                raise RuntimeError("v4 runtime did not materialize reliable ASR")

    report = {
        "schema": "stable_audio_tools.p11_v4_screening_validation",
        "schema_version": 1,
        "status": "PASS",
        "curriculum": str(curriculum_path),
        "curriculum_sha256": _sha256_file(curriculum_path),
        "contract": P11_V4_SCREENING_CONTRACT,
        "rows": 30_000,
        "base_scenes": 10_000,
        "task_counts": dict(sorted(task_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts_across_rows": dict(sorted(source_counts.items())),
        "prompt_max_tokens": prompt_token_max,
        "truncation": 0,
        "generation_views": dict(sorted(g_views.items())),
        "understanding_transforms": dict(sorted(u_transforms.items())),
        "editing_operations": dict(sorted(edit_operations.items())),
        "editing_owners": dict(sorted(edit_owners.items())),
        "editing_directions": dict(sorted(edit_directions.items())),
        "ordering": {
            "contract": ORDERING_CONTRACT,
            "batch_size": 8,
            "full_batches": full_batches,
            "tail_rows": 0,
            "task_patterns_g_u_e": dict(sorted(batch_task_patterns.items())),
        },
        "leakage": {
            "heldout_sample_overlap": 0,
            "heldout_target_sceneplan_overlap": 0,
            "heldout_prompt_exact_overlap": 0,
            "heldout_reserved_template_overlap": 0,
            "editing_resulting_bin_leakage": 0,
            "target_transcript_cache_access": False,
            "core40_sidecar_used": False,
        },
        "runtime": runtime,
        "p10_alignment": {
            "checkpoint": str(P10_CHECKPOINT),
            "checkpoint_sha256": P10_CHECKPOINT_SHA256,
            "max_duration_sec": 15.0465,
            "max_latent_frames": 648,
            "source_count": "1-4",
            "motion": ["static", "linear"],
            "execution_state": EXECUTION_STATE_CONTRACT,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

    connection.close()
    source_manifest.close()
    source_index.close()
    heldout.close()


if __name__ == "__main__":
    main()
