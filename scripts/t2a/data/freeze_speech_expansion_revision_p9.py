#!/usr/bin/env python3
"""Freeze and smoke-test the revision-6 1.640M ScenePlan training view."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import zlib
from typing import Any

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import (  # noqa: E402
    load_config,
    validate_training_configs,
)
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_44_controls,
    compile_model_renderer_caption,
    validate_model_sceneplan,
)
from stable_audio_tools.data.sceneplan_bucket_sampler import (  # noqa: E402
    sceneplan_bucket_collation,
)
from stable_audio_tools.data.sceneplan_v2_dataset import (  # noqa: E402
    ScenePlanV2Dataset,
)
from stable_audio_tools.models.conditioners import (  # noqa: E402
    ScenePlan44LocalConditioner,
)


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
INDEX_ROOT = REVISION_ROOT / "training_index"
EXPECTED = {"train": 1_600_000, "validation": 32_000, "test": 8_000}
EXPECTED_BUCKETS = {
    "train": (1_200_000, 400_000),
    "validation": (24_000, 8_000),
    "test": (6_000, 2_000),
}
MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s.json"
)
TOKENIZER_ROOT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B")
OVERFIT_ORDINALS = [
    0,
    500_000,
    599_999,
    1_137_500,
    1_168_750,
    1_200_000,
    1_237_500,
    1_268_750,
    1_300_000,
    1_450_000,
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_pass(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if value.get("ok") is not True and value.get("status") not in {
        "PASS",
        "complete",
    } and value.get("state") != "complete":
        raise RuntimeError(f"required audit is not complete/PASS: {path}")
    if rows is not None and int(value.get("rows", -1)) != int(rows):
        raise RuntimeError(f"required audit row count changed: {path}")
    return value


def index_metadata(path: Path, split: str, expected: int) -> dict[str, str]:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError(f"SQLite integrity failed: {path}")
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        rows, unique, minimum, maximum = connection.execute(
            "SELECT COUNT(*),COUNT(DISTINCT sample_id),MIN(ordinal),MAX(ordinal) "
            "FROM samples"
        ).fetchone()
        if (int(rows), int(unique), int(minimum), int(maximum)) != (
            expected,
            expected,
            0,
            expected - 1,
        ):
            raise RuntimeError(f"{split}: frozen ordinal/ID coverage changed")
        short, long = connection.execute(
            "SELECT SUM(latent_frames_valid<=432),"
            "SUM(latent_frames_valid>432) FROM samples"
        ).fetchone()
        if (int(short), int(long)) != EXPECTED_BUCKETS[split]:
            raise RuntimeError(f"{split} length buckets changed: {(short, long)}")
        required = {
            "schema": "stable_audio_tools.sceneplan_v2_training_index",
            "schema_version": "3",
            "contract_revision": "6",
            "model_sceneplan_schema_version": "2",
            "conditioning_contract_revision": "2",
            "caption_compiler_version": "5",
            "split": split,
            "rows": str(expected),
            "latent_channels": "64",
            "max_latent_frames": "648",
            "caption_max_tokens": "512",
            "structured_feature_dim": "9",
            "random_crop": "false",
            "frozen": "true",
            "speech_timing_sidecar": "none",
            "word_level_timestamp_teacher": "false",
        }
        for key, expected_value in required.items():
            if metadata.get(key) != expected_value:
                raise RuntimeError(
                    f"{split}: metadata {key}={metadata.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
        return metadata
    finally:
        connection.close()


def write_dataset_configs() -> list[dict[str, Any]]:
    root = REVISION_ROOT / "p10_configs"
    result = []
    common = {
        "caption_max_tokens": 512,
        "dataset_type": "sceneplan_v2_preencoded",
        "in_order": True,
        "latent_crop_length": 648,
        "latent_downsampling_ratio": 1024,
        "max_item_retries": 0,
        "max_length_sec": 648 * 1024 / 44_100,
        "persistent_workers": True,
        "pin_memory": True,
        "prefetch_factor": 4,
        "random_crop": False,
        "require_complete": True,
        "speech_timing_sidecar": None,
        "word_level_timestamp_teacher": False,
    }
    for split, rows in EXPECTED.items():
        value = {
            "_task": (
                "Frozen revision-6 variable-duration ScenePlan 4+4 view; "
                "valid audio spans 0--15.0465 seconds and padding is never loss."
            ),
            **common,
            "datasets": [
                {
                    "id": f"sceneplan_v2_speech_expansion_noalign_15s_v1_{split}",
                    "path": str(INDEX_ROOT / f"{split}.sqlite"),
                    "weight": 1.0,
                }
            ],
            "drop_last": split == "train",
            "expected_num_samples": rows,
        }
        if split == "train":
            value["length_bucket_batching"] = {
                "enabled": True,
                "long_batch_size": 48,
                "seed": 20260828,
            }
        path = root / f"sceneplan_v2_speech_expansion_noalign_15s_v1_{split}.json"
        atomic_json(path, value)
        result.append(
            {
                "split": split,
                "rows": rows,
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )
    return result


def sceneplan_at(connection: sqlite3.Connection, ordinal: int) -> dict[str, Any]:
    row = connection.execute(
        "SELECT sample_id,model_num_samples,latent_frames_valid,"
        "renderer_caption_zlib,scene_plan_zlib FROM samples WHERE ordinal=?",
        (int(ordinal),),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"missing P9 panel ordinal: {ordinal}")
    sample_id, samples, frames, caption_blob, scene_blob = row
    scene = json.loads(zlib.decompress(scene_blob))
    caption = json.loads(zlib.decompress(caption_blob))
    validate_model_sceneplan(scene)
    if caption != compile_model_renderer_caption(scene):
        raise RuntimeError(f"{sample_id}: caption compiler changed")
    controls = compile_model_44_controls(
        scene,
        model_num_samples=int(samples),
        latent_frames_valid=int(frames),
    )
    return {
        "ordinal": int(ordinal),
        "sample_id": str(sample_id),
        "model_num_samples": int(samples),
        "latent_frames_valid": int(frames),
        "duration_sec": int(samples) / 44_100.0,
        "source_kinds": [source["kind"] for source in scene["sources"]],
        "source_count": len(scene["sources"]),
        "event_shape": list(controls["source_event_frame_ids"].shape),
        "trajectory_shape": list(controls["source_trajectory_features"].shape),
    }


def evaluation_profile(split: str) -> dict[str, Any]:
    expected_composition = {
        "validation": {
            "no_speech": 12_000,
            "speech_only": 5_000,
            "speech_with_overlapping_background": 9_000,
            "speech_with_sequential_background": 6_000,
        },
        "test": {
            "no_speech": 3_000,
            "speech_only": 1_250,
            "speech_with_overlapping_background": 2_250,
            "speech_with_sequential_background": 1_500,
        },
    }
    expected_source_counts = {
        "validation": {1: 9_200, 2: 16_200, 3: 4_400, 4: 2_200},
        "test": {1: 2_300, 2: 4_050, 3: 1_100, 4: 550},
    }
    expected_kind_appearances = {
        "validation": {"music": 21_800, "sound": 21_800, "speech": 20_000},
        "test": {"music": 5_450, "sound": 5_450, "speech": 5_000},
    }
    connection = sqlite3.connect(
        f"file:{INDEX_ROOT / f'{split}.sqlite'}?mode=ro&immutable=1", uri=True
    )
    composition: dict[str, int] = {}
    source_counts: dict[int, int] = {}
    kind_appearances: dict[str, int] = {}
    try:
        for (blob,) in connection.execute("SELECT scene_plan_zlib FROM samples"):
            scene = json.loads(zlib.decompress(blob))
            sources = scene["sources"]
            source_counts[len(sources)] = source_counts.get(len(sources), 0) + 1
            for source in sources:
                kind = str(source["kind"])
                kind_appearances[kind] = kind_appearances.get(kind, 0) + 1
            speech = [source for source in sources if source["kind"] == "speech"]
            background = [source for source in sources if source["kind"] != "speech"]
            if not speech:
                key = "no_speech"
            elif not background:
                key = "speech_only"
            else:
                overlap = any(
                    min(
                        float(speech_source["activity"]["offset_sec"]),
                        float(background_source["activity"]["offset_sec"]),
                    )
                    > max(
                        float(speech_source["activity"]["onset_sec"]),
                        float(background_source["activity"]["onset_sec"]),
                    )
                    for speech_source in speech
                    for background_source in background
                )
                key = (
                    "speech_with_overlapping_background"
                    if overlap
                    else "speech_with_sequential_background"
                )
            composition[key] = composition.get(key, 0) + 1
    finally:
        connection.close()
    if composition != expected_composition[split]:
        raise RuntimeError(f"{split}: final scene composition changed: {composition}")
    if source_counts != expected_source_counts[split]:
        raise RuntimeError(f"{split}: final source-count distribution changed")
    if kind_appearances != expected_kind_appearances[split]:
        raise RuntimeError(f"{split}: final source-kind appearances changed")
    return {
        "scene_composition": composition,
        "source_count_distribution": {str(key): value for key, value in source_counts.items()},
        "source_kind_appearances": kind_appearances,
    }


def loader_gate(tokenizer: Any, *, require_frozen: bool) -> dict[str, Any]:
    index = INDEX_ROOT / "train.sqlite"
    dataset = ScenePlanV2Dataset(
        index,
        tokenizer_spec=(tokenizer, 512),
        expected_num_samples=EXPECTED["train"],
        latent_crop_length=648,
        caption_max_tokens=512,
        random_crop=False,
        require_frozen=require_frozen,
    )
    if dataset.speech_timing_index_path is not None:
        raise RuntimeError("revision-6 loader unexpectedly opened a timing sidecar")
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    try:
        panel = [sceneplan_at(connection, ordinal) for ordinal in OVERFIT_ORDINALS]
    finally:
        connection.close()
    samples = [dataset[ordinal] for ordinal in OVERFIT_ORDINALS]
    by_bucket: dict[int, list[tuple[torch.Tensor, dict[str, Any]]]] = {
        432: [],
        648: [],
    }
    for sample in samples:
        latent, metadata = sample
        if tuple(latent.shape) != (64, 648) or latent.dtype != torch.float16:
            raise RuntimeError("revision-6 loader did not return immutable 648 padding")
        valid = int(metadata["latent_stored_length"])
        if int(metadata["padding_mask"][0].sum()) != valid:
            raise RuntimeError("loader valid-frame mask changed")
        by_bucket[int(metadata["latent_bucket_frames"])].append(sample)
    conditioner = ScenePlan44LocalConditioner(output_dim=256)
    bucket_results = []
    for bucket, bucket_samples in by_bucket.items():
        if not bucket_samples:
            raise RuntimeError(f"P9 panel omitted bucket {bucket}")
        batch = sceneplan_bucket_collation(bucket_samples[:2])
        audio, metadata = batch
        if tuple(audio.shape) != (len(metadata), 64, bucket):
            raise RuntimeError(f"bucket {bucket}: collated latent shape changed")
        encoded, valid = conditioner(
            [row["sceneplan_44"] for row in metadata], device=torch.device("cpu")
        )
        if tuple(encoded.shape) != (len(metadata), 256, bucket):
            raise RuntimeError(f"bucket {bucket}: 4+4 conditioner shape changed")
        if tuple(valid.shape) != (len(metadata), bucket):
            raise RuntimeError(f"bucket {bucket}: local condition mask shape changed")
        if not torch.isfinite(encoded).all():
            raise RuntimeError(f"bucket {bucket}: local conditioning is non-finite")
        bucket_results.append(
            {
                "bucket_frames": bucket,
                "batch": len(metadata),
                "latent_shape": list(audio.shape),
                "conditioning_shape": list(encoded.shape),
                "valid_frames": [int(row["padding_mask"][0].sum()) for row in metadata],
            }
        )
    return {
        "panel": panel,
        "buckets": sorted(bucket_results, key=lambda row: row["bucket_frames"]),
        "require_frozen": require_frozen,
        "timing_sidecar_loaded": False,
        "padding_excluded_from_valid_mask": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-root", type=Path, default=REVISION_ROOT)
    args = parser.parse_args()
    revision = args.revision_root.expanduser().resolve(strict=True)
    if revision != REVISION_ROOT.resolve(strict=True):
        raise ValueError("this freeze script is bound to the canonical revision root")
    started = time.monotonic()

    evidence_paths = {
        "dataset_contract": REPO_ROOT
        / "docs/sceneplan_v2/sceneplan_speech_expansion_noalign_15s_v1.json",
        "pilot_100": revision / "pilot_100/PILOT_GATE.json",
        "donor_registry": revision / "sources/registry/summary.json",
        "speaker_registry": revision
        / "source_annotations/speech_speaker_instruct_v1/registry/summary.json",
        "sceneplan_summary": revision / "sceneplans_model_v2_delta/summary.json",
        "sceneplan_audit": revision / "sceneplans_model_v2_delta/audit.json",
        "p8_summary": revision / "materialized_delta/P8_SUMMARY.json",
        "p8_audit": revision / "materialized_delta/audit.json",
        "eval_sceneplan_summary": revision
        / "sceneplans_model_v2_eval_delta/summary.json",
        "eval_sceneplan_audit": revision
        / "sceneplans_model_v2_eval_delta/audit.json",
        "eval_p8_summary": revision / "materialized_eval_delta/P8_SUMMARY.json",
        "eval_p8_audit": revision / "materialized_eval_delta/audit.json",
        "training_index_summary": revision / "training_index/summary.json",
    }
    pilot = load_pass(evidence_paths["pilot_100"], rows=100)
    if (
        pilot.get("length_bucket_counts") != {"432": 30, "648": 70}
        or pilot.get("speech_timing_sidecar") is not None
        or pilot.get("word_level_timestamp_teacher") is not False
        or pilot.get("all_sources_complete_and_uncropped") is not True
        or pilot.get("single_spatialization_pass") is not True
        or pilot.get("all_latents_finite_float16_with_checksums") is not True
    ):
        raise RuntimeError("revision-6 100-row pilot contract failed")
    load_pass(evidence_paths["donor_registry"], rows=200_000)
    load_pass(evidence_paths["speaker_registry"], rows=200_000)
    sceneplan_audit = load_pass(evidence_paths["sceneplan_audit"], rows=500_000)
    p8_audit = load_pass(evidence_paths["p8_audit"], rows=500_000)
    eval_sceneplan_audit = load_pass(
        evidence_paths["eval_sceneplan_audit"], rows=16_000
    )
    eval_p8_audit = load_pass(evidence_paths["eval_p8_audit"], rows=16_000)
    if (
        eval_sceneplan_audit.get("split_counts")
        != {"validation": 12_000, "test": 4_000}
        or eval_p8_audit.get("split_counts")
        != {"validation": 12_000, "test": 4_000}
    ):
        raise RuntimeError("revision-6 validation/test delta coverage changed")
    training_summary = json.loads(
        evidence_paths["training_index_summary"]
        .resolve(strict=True)
        .read_text(encoding="utf-8")
    )
    if (
        int(training_summary.get("dataset_contract_revision", -1)) != 6
        or training_summary.get("speech_timing_sidecar") is not None
        or training_summary.get("word_level_timestamp_teacher") is not False
        or training_summary.get("train_length_distribution")
        != {"432": 1_200_000, "648": 400_000}
        or training_summary.get("validation_length_distribution")
        != {"432": 24_000, "648": 8_000}
        or training_summary.get("test_length_distribution")
        != {"432": 6_000, "648": 2_000}
    ):
        raise RuntimeError("revision-6 training-index summary contract failed")
    indexes = []
    for split, rows in EXPECTED.items():
        path = INDEX_ROOT / f"{split}.sqlite"
        metadata = index_metadata(path, split, rows)
        indexes.append(
            {
                "split": split,
                "rows": rows,
                "path": str(path),
                "sha256": sha256_file(path),
                "latent_shards": int(metadata["latent_shards"]),
            }
        )
    evaluation_profiles = {
        split: evaluation_profile(split) for split in ("validation", "test")
    }
    configs = write_dataset_configs()
    model_config = load_config(MODEL_CONFIG)
    for config in configs:
        validate_training_configs(model_config, load_config(config["path"]))
    if int(model_config.get("sample_size", -1)) != 648 * 1024:
        raise RuntimeError("P10 model sample_size is not the 648-frame envelope")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ROOT.resolve(strict=True), local_files_only=True
    )
    # Audit the exact production loader before publishing the freeze marker.
    # Formal loading remains fail-closed everywhere else; this is the same
    # two-phase pre-freeze/formal-reopen pattern used by the sound revision.
    loader = loader_gate(tokenizer, require_frozen=False)

    evidence = {
        name: {"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)}
        for name, path in evidence_paths.items()
    }
    p9 = {
        "schema": "stable_audio_tools.sceneplan_speech_expansion_p9_audit",
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_revision": 6,
        "dataset_id": "sceneplan_v2_1p640m_speech_expansion_noalign_15s_v1",
        "rows": 1_640_000,
        "splits": EXPECTED,
        "train_length_distribution": {"432": 1_200_000, "648": 400_000},
        "validation_length_distribution": {"432": 24_000, "648": 8_000},
        "test_length_distribution": {"432": 6_000, "648": 2_000},
        "duration_support_sec": [0.0, 648 * 1024 / 44_100],
        "padding_buckets_are_not_output_durations": True,
        "random_crop": False,
        "speech_timing_sidecar": None,
        "word_level_timestamp_teacher": False,
        "semantic_conditioning": "Qwen cross-attention",
        "structured_conditioning": "4 event plus 4 trajectory frame tracks",
        "independent_cfg_dropout": {
            "caption_unknown_probability": 0.15,
            "structured_unknown_probability": 0.15,
        },
        "sceneplan_audit_excerpt": {
            "temporal_pattern_counts": sceneplan_audit["temporal_pattern_counts"],
            "duration_sec": sceneplan_audit["duration_sec"],
            "caption_qwen_tokens": sceneplan_audit["caption_qwen_tokens"],
        },
        "p8_audit_excerpt": {
            "length_bucket_counts": p8_audit["length_bucket_counts"],
            "latent_frames_valid_range": p8_audit["latent_frames_valid_range"],
            "mixing_mode_counts": p8_audit["mixing_mode_counts"],
        },
        "eval_expansion_excerpt": {
            "sceneplan_split_counts": eval_sceneplan_audit["split_counts"],
            "sceneplan_length_bucket_counts": eval_sceneplan_audit[
                "length_bucket_counts"
            ],
            "materialized_length_bucket_counts": eval_p8_audit[
                "length_bucket_counts"
            ],
        },
        "evaluation_profiles": evaluation_profiles,
        "indexes": indexes,
        "dataset_configs": configs,
        "loader_and_conditioner_gate": loader,
        "evidence": evidence,
        "p10_training_started": False,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    audit_path = revision / "audit/P9_AUDIT.json"
    atomic_json(audit_path, p9)
    freeze = {
        "schema": "stable_audio_tools.sceneplan_v2_p9_freeze_manifest",
        "schema_version": 4,
        "state": "P9_complete_frozen_ready_for_P10_preflight",
        "dataset_contract_revision": 6,
        "dataset_id": p9["dataset_id"],
        "rows": p9["rows"],
        "splits": EXPECTED,
        "max_latent_frames": 648,
        "max_duration_sec": 648 * 1024 / 44_100,
        "runtime_conditioning": (
            "semantic_cross_attention_plus_4_event_plus_4_trajectory"
        ),
        "speech_timing_sidecar": None,
        "p9_audit": str(audit_path),
        "p9_audit_sha256": sha256_file(audit_path),
        "training_index_summary": str(
            evidence_paths["training_index_summary"].resolve(strict=True)
        ),
        "training_index_summary_sha256": sha256_file(
            evidence_paths["training_index_summary"]
        ),
        "model_config": str(MODEL_CONFIG.resolve(strict=True)),
        "model_config_sha256": sha256_file(MODEL_CONFIG),
        "dataset_configs": configs,
        "p10_training_started": False,
        "p11_training_started": False,
    }
    freeze_path = revision / "contracts/freeze_manifest.json"
    atomic_json(freeze_path, freeze)
    marker = {
        "schema": "stable_audio_tools.sceneplan_v2_p9_marker",
        "schema_version": 4,
        "state": freeze["state"],
        "dataset_contract_revision": 6,
        "dataset_id": p9["dataset_id"],
        "freeze_manifest": str(freeze_path),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "p10_training_started": False,
    }
    atomic_json(revision / "FROZEN_P9.json", marker)
    try:
        # Reopen through the marker and all frozen SHA256 links.  This catches
        # a marker/manifest/index mismatch before the stage can be marked PASS.
        formal_loader = loader_gate(tokenizer, require_frozen=True)
    except BaseException:
        # Do not leave a formal training gate behind after a failed reopen.
        (revision / "FROZEN_P9.json").unlink(missing_ok=True)
        raise
    stages = {
        "P0": "base frozen contract plus revision-6 duration/no-align contract",
        "P1": "all selected mono Speech/Music/Sound assets exist and are immutable",
        "P2": str(evidence_paths["donor_registry"]),
        "P3": (
            "frozen legacy validation/test subsets preserved; train and appended "
            "evaluation donors are audio/text/speaker-disjoint"
        ),
        "P4": str(evidence_paths["pilot_100"]),
        "P5": "semantic cross-attention plus frame-aligned 4+4 tests PASS",
        "P6": str(DATASET_ROOT / "pilots/joint_4k"),
        "P7": str(evidence_paths["speaker_registry"]),
        "P7.5": str(evidence_paths["sceneplan_audit"]),
        "P8": (
            f"train={evidence_paths['p8_audit']}; "
            f"eval={evidence_paths['eval_p8_audit']}"
        ),
        "P9": str(audit_path),
    }
    summary = {
        "schema": "stable_audio_tools.sceneplan_speech_expansion_p0_p9_summary",
        "schema_version": 1,
        "status": "PASS",
        "dataset_id": p9["dataset_id"],
        "rows": 1_640_000,
        "splits": EXPECTED,
        "train_length_distribution": {"432": 1_200_000, "648": 400_000},
        "validation_length_distribution": {"432": 24_000, "648": 8_000},
        "test_length_distribution": {"432": 6_000, "648": 2_000},
        "stages": {
            key: {"status": "PASS", "evidence": value}
            for key, value in stages.items()
        },
        "freeze_marker": str(revision / "FROZEN_P9.json"),
        "formal_loader_reopen": formal_loader,
        "p10_readiness": "ready_for_10_sample_overfit_throughput_and_resume_gates",
        "p10_training_started": False,
    }
    atomic_json(revision / "P0_P9_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
