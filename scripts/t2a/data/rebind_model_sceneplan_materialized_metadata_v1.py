#!/usr/bin/env python3
"""Rebind P8 manifests to revised text metadata without touching FOA/latents."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from materialize_model_sceneplan_v1_shard import MATERIALIZED_SCHEMA, canonical_json
from sceneplan_v2_common import DATASET_ROOT, atomic_write_json


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def lines(path: Path) -> list[tuple[dict[str, Any], str]]:
    result: list[tuple[dict[str, Any], str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        value = json.loads(raw)
        if canonical_json(value) != raw:
            raise RuntimeError(f"noncanonical JSONL: {path}:{number}")
        result.append((value, raw))
    return result


def text_neutral_scene(scene: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(canonical_json(scene))
    for source in value["sources"]:
        if source["kind"] == "speech":
            source["speaker_description"] = "<speaker-description>"
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-sceneplan-root", type=Path, required=True)
    parser.add_argument("--new-sceneplan-root", type=Path, default=DATASET_ROOT / "sceneplans_model_v1")
    parser.add_argument("--materialized-root", type=Path, default=DATASET_ROOT / "materialized")
    parser.add_argument("--archive-root", type=Path, required=True)
    args = parser.parse_args()
    old_root = args.old_sceneplan_root.expanduser().resolve(strict=True)
    new_root = args.new_sceneplan_root.expanduser().resolve(strict=True)
    materialized = args.materialized_root.expanduser().resolve(strict=True)
    archive = args.archive_root.expanduser().resolve(strict=False)
    manifests = sorted((materialized / "manifests").glob("*/materialized-*.parquet"))
    if len(manifests) != 1_099:
        raise RuntimeError(f"expected 1099 P8 manifests, found {len(manifests)}")
    staging = materialized / f"manifests.speaker-rebind.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    rows_seen = 0
    changed_speech = 0
    unchanged_nonspeech = 0
    split_counts: Counter[str] = Counter()
    started = time.time()
    for ordinal, manifest in enumerate(manifests, 1):
        table = pq.read_table(manifest)
        rows = table.to_pylist()
        split = str(rows[0]["split"])
        shard = int(rows[0]["work_shard"])
        stem = f"{split}-{shard:05d}.jsonl"
        old_models = lines(old_root / split / f"model-sceneplans-{stem}")
        old_recipes = lines(old_root / split / f"render-recipes-{stem}")
        old_conditions = lines(old_root / split / f"conditioning-{stem}")
        new_models = lines(new_root / split / f"model-sceneplans-{stem}")
        new_recipes = lines(new_root / split / f"render-recipes-{stem}")
        new_conditions = lines(new_root / split / f"conditioning-{stem}")
        if not (
            len(rows) == len(old_models) == len(old_recipes) == len(old_conditions)
            == len(new_models) == len(new_recipes) == len(new_conditions)
        ):
            raise RuntimeError(f"row count mismatch for {split}/{shard}")
        output_rows: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            old_model, old_model_text = old_models[index]
            new_model, new_model_text = new_models[index]
            old_recipe, old_recipe_text = old_recipes[index]
            new_recipe, new_recipe_text = new_recipes[index]
            old_condition, old_condition_text = old_conditions[index]
            new_condition, new_condition_text = new_conditions[index]
            sample_id = str(row["sample_id"])
            if any(
                str(value["sample_id"]) != sample_id
                for value in (old_model, new_model, old_recipe, new_recipe, old_condition, new_condition)
            ):
                raise RuntimeError(f"{sample_id}: three-view sample id/order drift")
            if text_neutral_scene(old_model) != text_neutral_scene(new_model):
                raise RuntimeError(f"{sample_id}: non-speaker ScenePlan state changed")
            old_recipe_link = str(old_recipe.pop("model_sceneplan_sha256"))
            new_recipe_link = str(new_recipe.pop("model_sceneplan_sha256"))
            if old_recipe != new_recipe:
                raise RuntimeError(f"{sample_id}: renderer execution recipe changed")
            old_model_sha = sha256_text(old_model_text)
            new_model_sha = sha256_text(new_model_text)
            old_recipe_sha = sha256_text(old_recipe_text)
            new_recipe_sha = sha256_text(new_recipe_text)
            old_caption_sha = sha256_text(canonical_json(old_condition["renderer_caption"]))
            new_caption_sha = sha256_text(canonical_json(new_condition["renderer_caption"]))
            if (
                old_recipe_link != old_model_sha
                or new_recipe_link != new_model_sha
                or str(row["model_sceneplan_sha256"]) != old_model_sha
                or str(row["render_recipe_sha256"]) != old_recipe_sha
                or str(row["renderer_caption_sha256"]) != old_caption_sha
            ):
                raise RuntimeError(f"{sample_id}: pre-rebind lineage mismatch")
            new_bundle = sha256_text(canonical_json({
                "model_sceneplan_sha256": new_model_sha,
                "render_recipe_sha256": new_recipe_sha,
                "renderer_caption_sha256": new_caption_sha,
            }))
            result = json.loads(str(row["render_result_json"]))
            if (
                result["planned_bundle_sha256"] != row["planned_bundle_sha256"]
                or result["planned_record_sha256"] != row["planned_bundle_sha256"]
                or result["model_sceneplan_sha256"] != old_model_sha
                or result["render_recipe_sha256"] != old_recipe_sha
                or result["renderer_caption_sha256"] != old_caption_sha
            ):
                raise RuntimeError(f"{sample_id}: old render-result lineage mismatch")
            result.update({
                "planned_record_sha256": new_bundle,
                "planned_bundle_sha256": new_bundle,
                "model_sceneplan_sha256": new_model_sha,
                "render_recipe_sha256": new_recipe_sha,
                "renderer_caption_sha256": new_caption_sha,
                "metadata_only_rebind": "speech_speaker_description_v1",
            })
            updated = dict(row)
            updated.update({
                "planned_bundle_sha256": new_bundle,
                "model_sceneplan_sha256": new_model_sha,
                "render_recipe_sha256": new_recipe_sha,
                "renderer_caption_sha256": new_caption_sha,
                "render_result_json": canonical_json(result),
            })
            output_rows.append(updated)
            has_speech = any(source["kind"] == "speech" for source in new_model["sources"])
            if has_speech:
                if old_model_sha == new_model_sha or old_caption_sha == new_caption_sha:
                    raise RuntimeError(f"{sample_id}: speech text did not change")
                changed_speech += 1
            else:
                if old_model_sha != new_model_sha or old_caption_sha != new_caption_sha:
                    raise RuntimeError(f"{sample_id}: no-speech text unexpectedly changed")
                unchanged_nonspeech += 1
        destination = staging / split / manifest.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(output_rows, schema=MATERIALIZED_SCHEMA), destination, compression="zstd")
        if pq.read_metadata(destination).num_rows != len(rows):
            raise RuntimeError(f"reopen row count mismatch: {destination}")
        rows_seen += len(rows)
        split_counts[split] += len(rows)
        if ordinal % 100 == 0:
            print(json.dumps({"rebound_shards": ordinal, "rows": rows_seen}), flush=True)
    if rows_seen != 1_124_000 or changed_speech != 512_000 or unchanged_nonspeech != 612_000:
        raise RuntimeError(
            f"rebind totals mismatch rows={rows_seen} speech={changed_speech} nonspeech={unchanged_nonspeech}"
        )
    archive.mkdir(parents=True, exist_ok=True)
    old_manifest_archive = archive / "materialized_manifests_before_speaker_rebind"
    if old_manifest_archive.exists():
        raise FileExistsError(old_manifest_archive)
    os.replace(materialized / "manifests", old_manifest_archive)
    os.replace(staging, materialized / "manifests")
    summary_path = materialized / "p8_orchestrator_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update({
        "metadata_only_rebind": "speech_speaker_description_v1",
        "metadata_rebound_rows": rows_seen,
        "metadata_rebound_speech_rows": changed_speech,
        "foa_rerendered_rows": 0,
        "latent_reencoded_rows": 0,
        "audio_and_latent_checksums_preserved": True,
        "original_manifests_archive": str(old_manifest_archive),
    })
    atomic_write_json(summary_path, summary)
    report = {
        "schema": "stable_audio_tools.model_sceneplan_p8_metadata_rebind_audit",
        "schema_version": 1,
        "ok": True,
        "rows": rows_seen,
        "changed_speech_rows": changed_speech,
        "unchanged_nonspeech_rows": unchanged_nonspeech,
        "split_counts": dict(split_counts),
        "foa_rerendered_rows": 0,
        "latent_reencoded_rows": 0,
        "audio_and_latent_checksums_preserved": True,
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(materialized / "p8_metadata_rebind_audit.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
