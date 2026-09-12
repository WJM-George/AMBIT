#!/usr/bin/env python3
"""Freeze the complete 8,000-row revision-6 P10 test contract.

Unlike the historical matched panel, this contract preserves every multi-source
ScenePlan.  ``domain`` is therefore a mutually exclusive scene-composition
storage key, while ``source_kinds`` remains multi-label and drives the music,
sound, and speech metric slices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
RUN_ROOT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/dit/"
    "sceneplan_dit_v9_speechexp_noalign_15s_300m_from_scratch_100k"
)
DEFAULT_OUTPUT = REVISION_ROOT / (
    "evaluation/p10_v9_full_test_8000_ckpt20k_100k_v1"
)
CHECKPOINT_STEPS = (20_000, 40_000, 60_000, 80_000, 100_000)
NOISE_NAMESPACE = "sceneplan-p10-revision6-full-test-v1-20260830"
EXPECTED_COMPOSITION = {
    "no_speech": 3_000,
    "speech_only": 1_250,
    "speech_with_overlapping_background": 2_250,
    "speech_with_sequential_background": 1_500,
}
EXPECTED_KIND_APPEARANCES = {"music": 5_450, "sound": 5_450, "speech": 5_000}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    temporary.replace(path)


def noise_seed(sample_id: str) -> int:
    digest = hashlib.sha256(
        f"{NOISE_NAMESPACE}\0{sample_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def read_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number}: expected object")
                yield value


def recipe_paths(root: Path, split: str) -> list[Path]:
    paths = sorted((root / split).glob(f"render-recipes-{split}-*.jsonl"))
    if not paths:
        raise FileNotFoundError(root / split / f"render-recipes-{split}-*.jsonl")
    return paths


def test_recipes() -> dict[str, dict[str, Any]]:
    roots = (
        DATASET_ROOT / "sceneplans_model_v1",
        REVISION_ROOT / "sceneplans_model_v2_eval_delta",
    )
    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(
        path for root in roots for path in recipe_paths(root, "test")
    ):
        sample_id = str(row["sample_id"])
        if sample_id in output:
            raise RuntimeError(f"duplicate test render recipe: {sample_id}")
        output[sample_id] = row
    if len(output) != 8_000:
        raise RuntimeError(f"test recipe count changed: {len(output)} != 8000")
    return output


def reference_map() -> dict[str, dict[str, str]]:
    roots = (
        DATASET_ROOT / "materialized/manifests/test",
        REVISION_ROOT / "materialized_eval_delta/manifests/test",
    )
    output: dict[str, dict[str, str]] = {}
    for root in roots:
        paths = sorted(root.glob("*.parquet"))
        if not paths:
            raise FileNotFoundError(root)
        for path in paths:
            table = pq.read_table(path, columns=["sample_id", "foa_path", "foa_sha256"])
            for item in table.to_pylist():
                sample_id = str(item["sample_id"])
                if sample_id in output:
                    raise RuntimeError(f"duplicate test reference: {sample_id}")
                output[sample_id] = {
                    "reference_foa_path": str(item["foa_path"]),
                    "reference_foa_sha256": str(item["foa_sha256"]),
                }
    if len(output) != 8_000:
        raise RuntimeError(f"test reference count changed: {len(output)} != 8000")
    return output


def source_keys(source: dict[str, Any]) -> dict[str, str | None]:
    asset = source.get("asset_ref") or {}
    speaker_id = source.get("speaker_id")
    dataset_id = asset.get("dataset_id")
    speaker_key = (
        f"{dataset_id}:{speaker_id}"
        if dataset_id is not None and speaker_id is not None
        else None
    )
    return {
        "asset_id": str(asset["asset_id"]) if asset.get("asset_id") else None,
        "identity_hash": (
            str(asset["identity_hash"]) if asset.get("identity_hash") else None
        ),
        "parent_asset_id": (
            str(asset["parent_asset_id"]) if asset.get("parent_asset_id") else None
        ),
        "transcript_hash": (
            str(asset["normalized_transcript_sha256"])
            if asset.get("normalized_transcript_sha256")
            else None
        ),
        "speaker_key": speaker_key,
    }


def leakage_audit(recipes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key_names = (
        "sample_id",
        "asset_id",
        "identity_hash",
        "parent_asset_id",
        "transcript_hash",
        "speaker_key",
    )
    test_sets = {key: set() for key in key_names}
    for row in recipes.values():
        test_sets["sample_id"].add(str(row["sample_id"]))
        for source in row["sources"]:
            for key, value in source_keys(source).items():
                if value is not None:
                    test_sets[key].add(value)

    overlay_root = REVISION_ROOT.parent / "sound_expansion_v1/sceneplans_model_v1"
    overlay_rows = list(read_jsonl(recipe_paths(overlay_root, "train")))
    overlay_ids = {str(row["sample_id"]) for row in overlay_rows}
    if len(overlay_ids) != 70_719:
        raise RuntimeError(f"sound overlay rows changed: {len(overlay_ids)}")
    train_sources = (
        (
            "base",
            read_jsonl(recipe_paths(DATASET_ROOT / "sceneplans_model_v1", "train")),
            overlay_ids,
        ),
        ("sound_overlay", iter(overlay_rows), set()),
        (
            "speech_delta",
            read_jsonl(
                recipe_paths(REVISION_ROOT / "sceneplans_model_v2_delta", "train")
            ),
            set(),
        ),
    )
    overlap = {key: set() for key in key_names}
    effective_rows = 0
    effective_sources = 0
    component_rows: dict[str, int] = {}
    for label, rows, skip_ids in train_sources:
        local_rows = 0
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in skip_ids:
                continue
            local_rows += 1
            effective_rows += 1
            if sample_id in test_sets["sample_id"]:
                overlap["sample_id"].add(sample_id)
            for source in row["sources"]:
                effective_sources += 1
                for key, value in source_keys(source).items():
                    if value is not None and value in test_sets[key]:
                        overlap[key].add(value)
        component_rows[label] = local_rows
    if effective_rows != 1_600_000:
        raise RuntimeError(f"effective train rows changed: {effective_rows}")
    hard_keys = (
        "sample_id",
        "asset_id",
        "identity_hash",
        "parent_asset_id",
        "transcript_hash",
    )
    hard_overlap = {key: len(overlap[key]) for key in hard_keys}
    if any(hard_overlap.values()):
        raise RuntimeError(f"full-test content leakage: {hard_overlap}")
    return {
        "status": "PASS_WITH_SEEN_SPEAKER_STRATUM",
        "effective_train_rows": effective_rows,
        "effective_train_sources": effective_sources,
        "effective_train_component_rows": component_rows,
        "test_unique_counts": {key: len(value) for key, value in test_sets.items()},
        "overlap_counts": {key: len(value) for key, value in overlap.items()},
        "overlap_values": {
            key: sorted(value) for key, value in overlap.items() if value
        },
        "content_disjoint": True,
        "speaker_disjoint": not overlap["speaker_key"],
        "seen_speaker_keys": sorted(overlap["speaker_key"]),
    }


def overlap_seconds(left: dict[str, Any], right: dict[str, Any]) -> float:
    return max(
        0.0,
        min(float(left["offset_sec"]), float(right["offset_sec"]))
        - max(float(left["onset_sec"]), float(right["onset_sec"])),
    )


def composition(plan: dict[str, Any]) -> str:
    speech = [source for source in plan["sources"] if source["kind"] == "speech"]
    if not speech:
        return "no_speech"
    if len(speech) != 1:
        raise RuntimeError(f"{plan['sample_id']}: expected at most one speech source")
    backgrounds = [source for source in plan["sources"] if source["kind"] != "speech"]
    if not backgrounds:
        return "speech_only"
    if any(overlap_seconds(speech[0]["activity"], source["activity"]) > 0 for source in backgrounds):
        return "speech_with_overlapping_background"
    return "speech_with_sequential_background"


def semantic_clause(
    source: dict[str, Any], *, semantic_caption_compiler_version: int
) -> str:
    if source["kind"] == "speech":
        if semantic_caption_compiler_version == 1:
            return f"{source['speaker_description']} says \"{source['transcript']}\""
        if semantic_caption_compiler_version == 2:
            return f"{source['speaker_description']} says: {source['transcript']}"
        raise ValueError(
            "semantic caption compiler version must be exactly 1 or 2"
        )
    return str(source["description"])


def read_index_rows(
    index_path: Path,
    references: dict[str, dict[str, str]],
    recipes: dict[str, dict[str, Any]],
    seen_speakers: set[str],
    *,
    semantic_caption_compiler_version: int,
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(
        f"file:{index_path}?mode=ro&immutable=1", uri=True
    )
    try:
        rows: list[dict[str, Any]] = []
        query = """
            SELECT ordinal, sample_id, model_num_samples, latent_frames_valid,
                   renderer_caption_zlib, scene_plan_zlib
            FROM samples ORDER BY ordinal
        """
        for ordinal, sample_id, samples, frames, caption_blob, plan_blob in connection.execute(query):
            sample_id = str(sample_id)
            caption = json.loads(zlib.decompress(caption_blob))
            plan = json.loads(zlib.decompress(plan_blob))
            if sample_id != str(plan["sample_id"]):
                raise RuntimeError(f"index/ScenePlan mismatch: {sample_id}")
            if sample_id not in references or sample_id not in recipes:
                raise RuntimeError(f"missing full-test provenance: {sample_id}")
            source_kinds = [str(source["kind"]) for source in plan["sources"]]
            kind_counts = Counter(source_kinds)
            scene_composition = composition(plan)
            speech_source = next(
                (source for source in recipes[sample_id]["sources"] if source["kind"] == "speech"),
                None,
            )
            speaker_key = (
                source_keys(speech_source)["speaker_key"]
                if speech_source is not None
                else None
            )
            source_semantic_texts = [
                semantic_clause(
                    source,
                    semantic_caption_compiler_version=(
                        semantic_caption_compiler_version
                    ),
                )
                for source in plan["sources"]
            ]
            row = {
                "ordinal": int(ordinal),
                "panel_id": f"full_{int(ordinal):07d}",
                "sample_id": sample_id,
                "domain": scene_composition,
                "scene_composition": scene_composition,
                "model_num_samples": int(samples),
                "latent_frames_valid": int(frames),
                "length_bucket": 432 if int(frames) <= 432 else 648,
                "duration_sec": float(plan["duration_sec"]),
                "room_type": str(plan["room"]["type"]),
                "source_count": len(plan["sources"]),
                "source_kinds": sorted(set(source_kinds)),
                "source_kind_counts": dict(sorted(kind_counts.items())),
                "source_semantic_texts": source_semantic_texts,
                "semantic_text": "; ".join(source_semantic_texts),
                "renderer_caption": str(caption["text"]),
                "scene_plan": plan,
                "speech_speaker_key": speaker_key,
                "speech_seen_speaker": (
                    speaker_key in seen_speakers if speaker_key is not None else None
                ),
                "noise_seed": noise_seed(sample_id),
                **references[sample_id],
            }
            reference_path = Path(row["reference_foa_path"])
            if not reference_path.is_file():
                raise FileNotFoundError(reference_path)
            rows.append(row)
        return rows
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument(
        "--checkpoint-steps", type=int, nargs="+", default=list(CHECKPOINT_STEPS)
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / (
            "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
            "qwen35_0p8b_300m_model_sceneplan_44.json"
        ),
    )
    parser.add_argument(
        "--test-config",
        type=Path,
        default=REVISION_ROOT
        / "p10_configs/sceneplan_v2_speech_expansion_noalign_15s_v1_test.json",
    )
    parser.add_argument(
        "--semantic-caption-compiler-version",
        type=int,
        choices=(1, 2),
        default=1,
    )
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve(strict=True)
    model_config = args.model_config.expanduser().resolve(strict=True)
    index_path = REVISION_ROOT / "training_index/test.sqlite"
    test_config = args.test_config.expanduser().resolve(strict=True)
    semantic_caption_compiler_version = int(
        args.semantic_caption_compiler_version
    )
    p9_path = REVISION_ROOT / "audit/P9_AUDIT.json"
    p9 = json.loads(p9_path.read_text(encoding="utf-8"))
    if p9.get("status") != "PASS" or p9.get("splits", {}).get("test") != 8_000:
        raise RuntimeError("revision-6 P9 test freeze is not all-pass")
    expected_index_hash = next(
        item["sha256"] for item in p9["indexes"] if item["split"] == "test"
    )
    if sha256_file(index_path) != expected_index_hash:
        raise RuntimeError("frozen revision-6 test index SHA256 changed")

    recipes = test_recipes()
    leakage = leakage_audit(recipes)
    references = reference_map()
    seen_speakers = set(leakage["seen_speaker_keys"])
    rows = read_index_rows(
        index_path,
        references,
        recipes,
        seen_speakers,
        semantic_caption_compiler_version=semantic_caption_compiler_version,
    )
    if len(rows) != 8_000:
        raise RuntimeError(f"full test row count changed: {len(rows)}")
    composition_counts = Counter(row["scene_composition"] for row in rows)
    if dict(composition_counts) != EXPECTED_COMPOSITION:
        raise RuntimeError(
            f"test composition changed: {dict(composition_counts)}"
        )
    kind_appearances = {
        kind: sum(int(row["source_kind_counts"].get(kind, 0)) for row in rows)
        for kind in EXPECTED_KIND_APPEARANCES
    }
    if kind_appearances != EXPECTED_KIND_APPEARANCES:
        raise RuntimeError(f"source-kind appearances changed: {kind_appearances}")
    kind_scene_coverage = {
        kind: sum(kind in row["source_kinds"] for row in rows)
        for kind in EXPECTED_KIND_APPEARANCES
    }
    source_count_counts = Counter(str(row["source_count"]) for row in rows)
    single_source_counts = {
        kind: sum(
            row["source_count"] == 1 and kind in row["source_kinds"]
            for row in rows
        )
        for kind in EXPECTED_KIND_APPEARANCES
    }
    length_counts = Counter(str(row["length_bucket"]) for row in rows)
    if dict(length_counts) != {"432": 6_000, "648": 2_000}:
        raise RuntimeError(f"test length buckets changed: {dict(length_counts)}")

    steps = tuple(int(step) for step in args.checkpoint_steps)
    if not steps or len(steps) != len(set(steps)) or tuple(sorted(steps)) != steps:
        raise ValueError("checkpoint steps must be unique and ascending")
    checkpoints = []
    for step in steps:
        candidates = sorted((run_root / "checkpoints").glob(f"*step={step}.ckpt"))
        if len(candidates) != 1:
            raise RuntimeError(f"expected one step={step} checkpoint: {candidates}")
        path = candidates[0].resolve(strict=True)
        checkpoints.append(
            {
                "step": step,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    panel_path = output_root / "full_test_8000.jsonl"
    atomic_jsonl(panel_path, rows)
    contract = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_full_test_contract",
        "schema_version": 2,
        "status": (
            "FROZEN_FULL_TEST_MULTILABEL_SEMANTIC_V"
            f"{semantic_caption_compiler_version}"
        ),
        "purpose": "complete revision-6 P10 checkpoint and baseline evaluation",
        "test_set": {
            "index": str(index_path.resolve(strict=True)),
            "index_sha256": expected_index_hash,
            "dataset_config": str(test_config.resolve(strict=True)),
            "dataset_config_sha256": sha256_file(test_config),
            "p9_audit": str(p9_path.resolve(strict=True)),
            "p9_audit_sha256": sha256_file(p9_path),
            "all_rows": 8_000,
            "evaluation_rows": 8_000,
            "evaluation_subset": "none; complete frozen test split",
            "panel_filename": panel_path.name,
            "panel_sha256": sha256_file(panel_path),
            "domain_counts": dict(composition_counts),
            "source_kind_appearances": kind_appearances,
            "source_kind_scene_coverage": kind_scene_coverage,
            "source_count_counts": dict(source_count_counts),
            "single_source_counts": single_source_counts,
            "length_bucket_counts": dict(length_counts),
            "max_latent_frames": 648,
            "multi_label_source_kinds": True,
            "content_disjoint_from_train": True,
            "speaker_disjoint_from_train": leakage["speaker_disjoint"],
        },
        "leakage_audit": leakage,
        "checkpoints": checkpoints,
        "sampling": {
            "architecture": "semantic_cross_attention_plus_direct_sceneplan_4+4",
            "model_config": str(model_config),
            "model_config_sha256": sha256_file(model_config),
            "vae_checkpoint": os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt",
            "weights": "EMA DiT plus EMA trainable 4+4 conditioner",
            "sampler": "euler_rectified_flow",
            "steps": 100,
            "cfg_scale": 3.0,
            "rescale_cfg": True,
            "cfg_rescale_phi": 0.4,
            "apg_scale": 0.0,
            "max_latent_frames": 648,
            "negative_condition": "caption and structured controls both unknown",
            "training_cfg_dropout": "caption and structured controls independently 15%",
            "common_noise_seed_namespace": NOISE_NAMESPACE,
            "same_noise_per_sample_across_checkpoints": True,
            "inference_batch_size": 1,
            "semantic_caption_compiler_version": (
                semantic_caption_compiler_version
            ),
            "raw_output": "float32 native WYZX/ACN/SN3D FOA",
        },
        "metric_slices": {
            "complete": 8_000,
            "scene_composition": dict(composition_counts),
            "source_kind_multi_label": kind_appearances,
            "source_kind_scene_coverage": kind_scene_coverage,
            "single_source": single_source_counts,
            "source_count": dict(source_count_counts),
            "speech_seen_speaker": sum(
                row["speech_seen_speaker"] is True for row in rows
            ),
            "speech_unseen_speaker": sum(
                row["speech_seen_speaker"] is False for row in rows
            ),
            "baseline_general_t2a": "no_speech scenes only",
            "baseline_tts": "speech_only scenes only",
            "spatial": (
                "native FOA only; plan DoA on exactly-one-active-source frames, "
                "plus generated/reference mixture-DoA on all coherent active frames"
            ),
        },
    }
    atomic_json(output_root / "EVAL_CONTRACT.json", contract)
    atomic_json(
        output_root / "BUILD_SUMMARY.json",
        {
            "status": "PASS",
            "output_root": str(output_root),
            "rows": len(rows),
            "composition_counts": dict(composition_counts),
            "source_kind_appearances": kind_appearances,
            "source_kind_scene_coverage": kind_scene_coverage,
            "source_count_counts": dict(source_count_counts),
            "single_source_counts": single_source_counts,
            "length_bucket_counts": dict(length_counts),
            "semantic_caption_compiler_version": (
                semantic_caption_compiler_version
            ),
            "leakage_audit": leakage,
            "checkpoints": list(steps),
        },
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_root": str(output_root),
                "rows": len(rows),
                "composition_counts": dict(composition_counts),
                "source_kind_appearances": kind_appearances,
                "source_kind_scene_coverage": kind_scene_coverage,
                "leakage": leakage["overlap_counts"],
                "checkpoints": list(steps),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
