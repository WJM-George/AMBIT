#!/usr/bin/env python3
"""Strict structural, runtime, and split-leakage gate for P11-v4 challenge-v1."""

from __future__ import annotations
import os

import argparse
import hashlib
import json
import math
import sqlite3
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    score_p11_prediction,
)
from stable_audio_tools.data.sceneplan_p11_single_turn import (  # noqa: E402
    validate_p11_executor_profile,
)
from stable_audio_tools.data.sceneplan_p11_v4_challenge import (  # noqa: E402
    P11_V4_CHALLENGE_CONTRACT,
    P11_V4_CHALLENGE_SCHEMA,
    P11_V4_CHALLENGE_VERSION,
    P11_V4_EVIDENCE_TRANSFORM_CONTRACT,
    P11_V4_REFERENCE_SET_CONTRACT,
    ScenePlanP11V4ChallengeDataset,
)
from stable_audio_tools.data.sceneplan_p11_v4_dataset import (  # noqa: E402
    P11_V4_DATA_CONTRACT,
    P11_V4_SEQUENCE_CONTRACT,
    ScenePlanP11V4Dataset,
)


DEFAULT_CHALLENGE = REPO_ROOT / (
    "artifacts/sceneplan_p11/challenges/"
    "p11_v4_heldout_challenge_v1_20260901.sqlite"
)
DEFAULT_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_heldout900_transfusion_cot_v4_reliable_asr_v1.json"
)
DEFAULT_MODEL = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
)
DEFAULT_TRAIN_INDEX = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/training_index/train.sqlite"
)
DEFAULT_TEST_INDEX = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/training_index/test.sqlite"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _decode(payload: bytes) -> Any:
    return json.loads(zlib.decompress(payload))


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _semantic_signature(plan: Mapping[str, Any]) -> dict[str, Any]:
    sources = []
    for source in plan["sources"]:
        item = {
            "source_id": str(source["source_id"]),
            "kind": str(source["kind"]),
        }
        if item["kind"] == "speech":
            item["speaker_description"] = str(source["speaker_description"])
            item["transcript"] = str(source["transcript"])
        else:
            item["description"] = str(source["description"])
        sources.append(item)
    return {"room": str(plan["room"]["type"]), "sources": sources}


def _stream_split_overlap(
    path: Path,
    *,
    sample_ids: set[str],
    sceneplan_hashes: set[str],
) -> dict[str, Any]:
    connection = _readonly(path)
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    sample_overlap: set[str] = set()
    plan_overlap: set[str] = set()
    rows = 0
    cursor = connection.execute(
        "SELECT sample_id,model_sceneplan_sha256 FROM samples"
    )
    while True:
        batch = cursor.fetchmany(16_384)
        if not batch:
            break
        rows += len(batch)
        for sample_id, digest in batch:
            if str(sample_id) in sample_ids:
                sample_overlap.add(str(sample_id))
            if str(digest) in sceneplan_hashes:
                plan_overlap.add(str(digest))
    connection.close()
    return {
        "path": str(path),
        "split": metadata.get("split"),
        "rows_scanned": rows,
        "sample_id_overlap_count": len(sample_overlap),
        "sceneplan_sha256_overlap_count": len(plan_overlap),
        "sample_id_overlap": sorted(sample_overlap),
        "sceneplan_sha256_overlap": sorted(plan_overlap),
        "pass": not sample_overlap and not plan_overlap,
    }


def _runtime_ordinals(
    connection: sqlite3.Connection, rows_per_view: int
) -> list[int]:
    selected: list[int] = []
    views = [row[0] for row in connection.execute("SELECT DISTINCT view_id FROM rows")]
    for view in sorted(views):
        values = [
            int(row[0])
            for row in connection.execute(
                "SELECT ordinal FROM rows WHERE view_id=? ORDER BY ordinal",
                (str(view),),
            ).fetchall()
        ]
        if not values:
            raise RuntimeError(f"challenge view {view!r} is empty")
        if rows_per_view == 1:
            chosen = values[:1]
        else:
            positions = np.linspace(
                0, len(values) - 1, min(rows_per_view, len(values)), dtype=int
            )
            chosen = [values[int(position)] for position in positions]
        selected.extend(chosen)
    return sorted(set(selected))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runtime-rows-per-view", type=int, default=2)
    parser.add_argument("--skip-runtime", action="store_true")
    parser.add_argument("--skip-split-leakage", action="store_true")
    parser.add_argument("--train-index", type=Path, default=DEFAULT_TRAIN_INDEX)
    parser.add_argument("--test-index", type=Path, default=DEFAULT_TEST_INDEX)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "artifacts/sceneplan_p11/p11_v4_challenge_v1_validation_20260901.json",
    )
    args = parser.parse_args()
    if args.runtime_rows_per_view <= 0:
        raise ValueError("--runtime-rows-per-view must be positive")
    started = time.perf_counter()
    challenge_path = args.challenge.expanduser().resolve(strict=True)
    dataset_config_path = args.dataset_config.expanduser().resolve(strict=True)
    model_config_path = args.model_config.expanduser().resolve(strict=True)
    dataset_config = load_config(dataset_config_path)
    model_config = load_config(model_config_path)
    sources = dataset_config.get("datasets") or []
    if len(sources) != 1:
        raise ValueError("challenge validator requires one base source index")
    manifest_path = Path(dataset_config["manifest_path"]).resolve(strict=True)
    index_path = Path(sources[0]["path"]).resolve(strict=True)
    codec = load_model_sceneplan_codec(dataset_config["codec_path"])
    patch_codec = ScenePlanEditPatchCodec(codec)
    delta_codec = DeltaSceneSketchCodec(codec, patch_codec)

    connection = _readonly(challenge_path)
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    required = {
        "schema": P11_V4_CHALLENGE_SCHEMA,
        "schema_version": str(P11_V4_CHALLENGE_VERSION),
        "contract": P11_V4_CHALLENGE_CONTRACT,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "source_index": str(index_path),
        "source_index_sha256": _sha256_file(index_path),
        "source_index_split": "validation",
        "codec_fingerprint": codec.fingerprint,
        "p11_v4_data_contract": P11_V4_DATA_CONTRACT,
        "p11_v4_sequence_contract": P11_V4_SEQUENCE_CONTRACT,
        "p10_release": "p10-sceneplan-dit-v11-step150000",
        "p10_checkpoint_sha256": (
            "be8c90cd1434bd71f73951531175c3674ff0f3173d5db591e2e1c476152ff59e"
        ),
        "p10_nominal_max_duration_seconds": "15",
        "p10_max_grid_seconds": f"{648 * 1024 / 44_100:.9f}",
        "p10_max_latent_frames": "648",
        "p10_source_count": "1-4",
        "p10_motion_profile": "static,linear",
        "rows_per_base_scene": "10",
        "generation_reference_contract": P11_V4_REFERENCE_SET_CONTRACT,
        "generation_reference_sets_are_exhaustive": "false",
        "underspecified_hidden_exact_target_model_selection": "forbidden",
        "u_degradation_is_real_acoustic_benchmark": "false",
        "challenge_template_partition": "heldout_eval_reserved_v1",
        "exact_compatibility_is_template_generalization_gate": "false",
        "eval_reserved_templates_may_enter_training": "false",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"challenge metadata {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    if not Path(metadata["p10_checkpoint"]).is_file():
        raise RuntimeError("frozen P10-v11 checkpoint is missing")
    if _sha256_file(Path(metadata["builder"])) != metadata.get("builder_sha256"):
        raise RuntimeError("challenge builder changed after the immutable build")

    count = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
    bounds = connection.execute("SELECT MIN(ordinal),MAX(ordinal) FROM rows").fetchone()
    base_scenes = int(metadata["base_scenes"])
    if count != int(metadata["rows"]) or count != base_scenes * 10:
        raise RuntimeError("challenge row count changed")
    if bounds != (0, count - 1):
        raise RuntimeError("challenge ordinals are not contiguous")
    duplicates = int(
        connection.execute(
            "SELECT COUNT(*)-COUNT(DISTINCT challenge_id) FROM rows"
        ).fetchone()[0]
    )
    if duplicates:
        raise RuntimeError("challenge IDs are not unique")

    rows = connection.execute(
        """
        SELECT ordinal,challenge_id,task,family,view_id,template_id,
               base_manifest_ordinal,base_target_ordinal,sample_id,prompt,
               known_field_groups_json,target_sceneplan_zlib,edit_spec_json,
               evidence_transform_json,reference_sceneplans_zlib,
               reference_set_contract,pair_id,pair_label,selection_role
        FROM rows ORDER BY ordinal
        """
    ).fetchall()
    task_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    view_counts: Counter[str] = Counter()
    base_counts: Counter[int] = Counter()
    pair_rows: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
    prompt_texts: list[str] = []
    sample_ids: set[str] = set()
    target_ordinals: set[int] = set()
    original_plan_hashes: set[str] = set()
    reference_count = 0
    reference_unique_numeric_min = None
    max_target_duration_seconds = 0.0
    max_target_latent_frames = 0
    for row in rows:
        (
            ordinal,
            challenge_id,
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
            references_payload,
            reference_contract,
            pair_id,
            pair_label,
            selection_role,
        ) = row
        task = str(task)
        family = str(family)
        task_counts[task] += 1
        family_counts[family] += 1
        view_counts[str(view_id)] += 1
        base_counts[int(target_ordinal)] += 1
        sample_ids.add(str(sample_id))
        target_ordinals.add(int(target_ordinal))
        prompt_texts.append(str(prompt))
        if family in {
            "generation_numeric_posterior",
            "editing_counterfactual_causality",
        } and not str(template_id).startswith("eval_reserved/"):
            raise RuntimeError("model-selection challenge leaked a non-reserved template")
        if family == "exact_compatibility" and selection_role != "compatibility_only":
            raise RuntimeError("exact template compatibility entered model selection")
        if family == "understanding_degraded_evidence" and selection_role != "robustness_diagnostic":
            raise RuntimeError("synthetic U degradation is mislabeled as model selection")
        if family in {
            "generation_numeric_posterior",
            "editing_counterfactual_causality",
        } and selection_role != "model_selection":
            raise RuntimeError("challenge selection role changed")

        target = validate_p11_executor_profile(codec.project_plan(_decode(target_payload)))
        if str(target["sample_id"]) != str(sample_id):
            raise RuntimeError("target/sample ID mismatch")
        target_tokens = codec.encode(target)["input_ids"]
        decoded = codec.decode(target_tokens, sample_id=str(sample_id))
        if not torch.equal(target_tokens, codec.encode(decoded)["input_ids"]):
            raise RuntimeError("challenge target failed ScenePlan round-trip")
        target_frames = int(codec._duration_frame(float(target["duration_sec"])))
        max_target_duration_seconds = max(
            max_target_duration_seconds, float(target["duration_sec"])
        )
        max_target_latent_frames = max(max_target_latent_frames, target_frames)
        if not 1 <= len(target["sources"]) <= 4 or not 1 <= target_frames <= 648:
            raise RuntimeError("challenge target exceeds P10 capability")
        references = _decode(references_payload)
        if not isinstance(references, list) or not references:
            raise RuntimeError("challenge reference set is empty")
        reference_count += len(references)
        evidence = json.loads(str(evidence_json))
        if evidence.get("contract") != P11_V4_EVIDENCE_TRANSFORM_CONTRACT:
            raise RuntimeError("challenge evidence contract changed")

        if task == "generation":
            groups = json.loads(str(known_json))
            numeric_hashes = set()
            target_semantic = _semantic_signature(target)
            for reference in references:
                reference = validate_p11_executor_profile(codec.project_plan(reference))
                if _semantic_signature(reference) != target_semantic:
                    raise RuntimeError("continuous G reference changed semantic authority")
                score = score_generation_constraints(
                    target,
                    reference,
                    known_field_groups=groups,
                    source_matching="permutation_invariant",
                )
                if not math.isclose(float(score["task_score"]), 1.0, abs_tol=1e-12):
                    raise RuntimeError("G reference violates its stated prompt constraints")
                core = torch.from_numpy(
                    execution_state_core(compile_execution_state(reference, codec))
                )
                numeric_hashes.add(_tensor_sha256(core))
            if reference_contract == P11_V4_REFERENCE_SET_CONTRACT:
                if len(references) != 8 or len(numeric_hashes) < 8:
                    raise RuntimeError("underspecified G reference anchors are degenerate")
                reference_unique_numeric_min = (
                    len(numeric_hashes)
                    if reference_unique_numeric_min is None
                    else min(reference_unique_numeric_min, len(numeric_hashes))
                )
            elif len(references) != 1:
                raise RuntimeError("exact G row has multiple targets")
        elif task == "understanding":
            if len(references) != 1 or edit_json is not None:
                raise RuntimeError("U challenge target contract changed")
            if family == "understanding_degraded_evidence" and evidence.get(
                "transform_id"
            ) == "identity_v1":
                raise RuntimeError("degraded U row is actually identity evidence")
        elif task == "editing":
            if edit_json is None:
                raise RuntimeError("E challenge lacks atomic patch")
            base_position = int(base_ordinal) // 3
            base_manifest_row = connection  # placate type narrowing below
            del base_manifest_row
            manifest = _readonly(manifest_path)
            current_payload = manifest.execute(
                "SELECT target_sceneplan_zlib FROM rows WHERE ordinal=?",
                (base_position * 3,),
            ).fetchone()
            manifest.close()
            if current_payload is None:
                raise RuntimeError("E challenge base ScenePlan is missing")
            current = codec.project_plan(_decode(current_payload[0]))
            spec = json.loads(str(edit_json))
            patch_codec.assert_target(current, spec, target)
            score = score_p11_prediction(
                task="editing",
                target_plan=target,
                prediction=patch_codec.apply(current, spec),
                input_plan=current,
                source_matching="persistent_id",
                editing_score_version="patch_applied_v3",
            )
            if not math.isclose(float(score["task_score"]), 1.0, abs_tol=1e-12):
                raise RuntimeError("E target does not score perfectly")
            if pair_id is not None:
                pair_rows[str(pair_id)].append(row)
        else:
            raise RuntimeError(f"unknown challenge task {task!r}")

    if len(base_counts) != base_scenes or set(base_counts.values()) != {10}:
        raise RuntimeError("challenge does not have ten views per held-out scene")
    expected_tasks = {
        "generation": base_scenes * 3,
        "understanding": base_scenes * 4,
        "editing": base_scenes * 3,
    }
    if dict(task_counts) != expected_tasks:
        raise RuntimeError(f"challenge task balance changed: {task_counts}")
    if len(pair_rows) != base_scenes:
        raise RuntimeError("counterfactual E pair coverage changed")
    for pair_id, pair in pair_rows.items():
        if len(pair) != 2:
            raise RuntimeError(f"counterfactual pair {pair_id} is incomplete")
        left_spec = json.loads(str(pair[0][12]))
        right_spec = json.loads(str(pair[1][12]))
        if (
            left_spec["operation"] != right_spec["operation"]
            or left_spec["source_id"] != right_spec["source_id"]
            or pair[0][17] == pair[1][17]
        ):
            raise RuntimeError("E counterfactual pair changed owner or label")
        manifest = _readonly(manifest_path)
        base_position = int(pair[0][6]) // 3
        payload = manifest.execute(
            "SELECT target_sceneplan_zlib FROM rows WHERE ordinal=?",
            (base_position * 3,),
        ).fetchone()[0]
        manifest.close()
        current = codec.project_plan(_decode(payload))
        targets = [codec.project_plan(_decode(value[11])) for value in pair]
        if torch.equal(
            codec.encode(targets[0])["input_ids"],
            codec.encode(targets[1])["input_ids"],
        ):
            raise RuntimeError("E counterfactual targets collapsed")
        delta_tokens = []
        for spec, target in zip((left_spec, right_spec), targets):
            delta = compile_delta_scene_sketch(current, target, spec, codec)
            delta_tokens.append(delta_codec.encode(delta, spec)["input_ids"])
        programs = [delta_codec.decode(tokens) for tokens in delta_tokens]
        if {int(value.get("control_direction", 0)) for value in programs} != {-1, 1}:
            raise RuntimeError("E pair lacks opposite P10 control directions")
        if not (
            all(int(tokens.numel()) == 5 for tokens in delta_tokens)
            and torch.equal(delta_tokens[0][:3], delta_tokens[1][:3])
            and int(delta_tokens[0][3]) != int(delta_tokens[1][3])
            and torch.equal(delta_tokens[0][4:], delta_tokens[1][4:])
        ):
            raise RuntimeError(
                "E pair must keep operation/owner and change only direction token"
            )

    # Tokenization is checked over every prompt, not just runtime samples.
    from transformers import AutoTokenizer

    text_config = model_config["model"]["text"]
    tokenizer = AutoTokenizer.from_pretrained(
        text_config["model_path"], local_files_only=True, use_fast=True
    )
    prompt_lengths: list[int] = []
    for start in range(0, len(prompt_texts), 128):
        encoded = tokenizer(
            prompt_texts[start : start + 128],
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        prompt_lengths.extend(len(value) for value in encoded)
    if max(prompt_lengths) > 512:
        raise RuntimeError("challenge prompt exceeds the no-truncation ceiling")

    runtime = {"status": "SKIPPED", "rows": 0, "ordinals": []}
    if not args.skip_runtime:
        base = ScenePlanP11V4Dataset(
            manifest_path,
            index_path=index_path,
            codec_path=dataset_config["codec_path"],
            tokenizer_spec=(tokenizer, 512, None),
            expected_num_samples=int(dataset_config["expected_num_samples"]),
            index_num_samples=int(dataset_config["index_num_samples"]),
            require_frozen=bool(dataset_config.get("require_complete", True)),
            semantic_cache_path=dataset_config["semantic_cache_path"],
            semantic_dim=int(dataset_config.get("semantic_dim", 512)),
            semantic_encoder_revision=dataset_config["semantic_encoder_revision"],
            lexical_evidence_mode=str(dataset_config.get("lexical_evidence_mode", "none")),
            lexical_max_tokens=int(dataset_config.get("lexical_max_tokens", 128)),
            lexical_cache_path=dataset_config.get("lexical_cache_path"),
            lexical_encoder_revision=dataset_config.get("lexical_encoder_revision"),
            lexical_confidence_threshold=dataset_config.get(
                "lexical_confidence_threshold"
            ),
        )
        overlay = ScenePlanP11V4ChallengeDataset(base, challenge_path)
        selected = _runtime_ordinals(connection, args.runtime_rows_per_view)
        changed_degraded = 0
        for ordinal in selected:
            carrier, row = overlay[ordinal]
            if row.get("p11_challenge_contract") != P11_V4_CHALLENGE_CONTRACT:
                raise RuntimeError("runtime challenge contract missing")
            if not bool(row.get("p11_challenge_eval_only")):
                raise RuntimeError("runtime challenge lost eval-only marker")
            if not bool(torch.isfinite(torch.as_tensor(carrier)).all()):
                raise RuntimeError("runtime challenge carrier is non-finite")
            if not bool(
                torch.isfinite(row["p11_v4_target_execution_core"]).all()
            ):
                raise RuntimeError("runtime target core is non-finite")
            transform = row["p11_challenge_evidence_transform"]
            if transform["synthetic_representation_stress"]:
                before, after = transform["before"], transform["after"]
                if (
                    before["foa_sha256"] == after["foa_sha256"]
                    and before["semantic_sha256"] == after["semantic_sha256"]
                ):
                    raise RuntimeError("degraded U transform changed no evidence")
                changed_degraded += 1
            elif transform["before"] != transform["after"]:
                raise RuntimeError("identity evidence transform changed its input")
            if row["p11_task"] == "editing":
                patch_codec.assert_target(
                    row["p11_input_sceneplan"],
                    row["p11_edit_spec"],
                    row["p11_target_sceneplan"],
                )
        runtime = {
            "status": "PASS",
            "rows": len(selected),
            "ordinals": selected,
            "degraded_rows_with_changed_evidence": changed_degraded,
        }

    leakage: dict[str, Any]
    if args.skip_split_leakage:
        leakage = {"status": "SKIPPED"}
    else:
        validation_index = _readonly(index_path)
        placeholders = ",".join("?" for _ in target_ordinals)
        query = (
            "SELECT sample_id,model_sceneplan_sha256 FROM samples WHERE ordinal IN ("
            + placeholders
            + ")"
        )
        selected_rows = validation_index.execute(
            query, tuple(sorted(target_ordinals))
        ).fetchall()
        validation_index.close()
        if len(selected_rows) != base_scenes:
            raise RuntimeError("selected held-out source coverage changed")
        selected_ids = {str(row[0]) for row in selected_rows}
        original_plan_hashes = {str(row[1]) for row in selected_rows}
        if selected_ids != sample_ids:
            raise RuntimeError("challenge/source held-out identities diverged")
        train = _stream_split_overlap(
            args.train_index.expanduser().resolve(strict=True),
            sample_ids=sample_ids,
            sceneplan_hashes=original_plan_hashes,
        )
        test = _stream_split_overlap(
            args.test_index.expanduser().resolve(strict=True),
            sample_ids=sample_ids,
            sceneplan_hashes=original_plan_hashes,
        )
        leakage = {
            "status": "PASS" if train["pass"] and test["pass"] else "FAIL",
            "base_scene_identity": "sample_id_and_model_sceneplan_sha256",
            "train": train,
            "test": test,
            "template_policy": {
                "challenge_partition": "heldout_eval_reserved_v1",
                "reserved_templates_may_enter_training": False,
                "exact_compatibility_surface_reused": True,
                "exact_rows_are_template_generalization_gate": False,
                "current_train_config_has_challenge_overlay": bool(
                    dataset_config.get("challenge_path")
                ),
            },
        }
        if leakage["status"] != "PASS":
            raise RuntimeError("challenge base scene leaked into train/test")
        if leakage["template_policy"]["current_train_config_has_challenge_overlay"]:
            raise RuntimeError("canonical dataset config unexpectedly trains on challenge rows")

    connection.close()
    report = {
        "schema": "stable_audio_tools.p11_v4_challenge_validation",
        "schema_version": 1,
        "status": "PASS",
        "scope": (
            "P10-v11 structural validity + prompt constraints + E counterfactual "
            "causality + sampled runtime overlay + base-scene split leakage"
        ),
        "challenge": str(challenge_path),
        "challenge_sha256": _sha256_file(challenge_path),
        "dataset_config": str(dataset_config_path),
        "model_config": str(model_config_path),
        "base_scenes": base_scenes,
        "rows": count,
        "task_counts": dict(sorted(task_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "view_counts": dict(sorted(view_counts.items())),
        "reference_sceneplans": reference_count,
        "underspecified_reference_unique_numeric_min": reference_unique_numeric_min,
        "p10_envelope_observed": {
            "nominal_duration_seconds": 15.0,
            "grid_max_seconds": 648 * 1024 / 44_100,
            "max_target_duration_seconds": max_target_duration_seconds,
            "max_target_latent_frames": max_target_latent_frames,
        },
        "prompt_tokens": {
            "max": max(prompt_lengths),
            "p95": float(np.percentile(prompt_lengths, 95)),
            "truncated": 0,
        },
        "counterfactual_pairs": len(pair_rows),
        "runtime": runtime,
        "split_leakage": leakage,
        "important_semantics": {
            "underspecified_g_hidden_exact_target_for_selection": False,
            "reference_completion_sets_are_exhaustive": False,
            "u_degradation_is_real_acoustic_benchmark": False,
            "continuous_noise_may_change_scene_sketch": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    report["report_sha256_without_self"] = _json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
