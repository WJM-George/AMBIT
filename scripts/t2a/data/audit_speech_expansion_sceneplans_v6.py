#!/usr/bin/env python3
"""Exhaustively audit the revision-6 500k ScenePlan delta before P8."""

from __future__ import annotations
import os

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.materialize_model_sceneplan_v1_shard import (  # noqa: E402
    load_shard_rows,
)
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_44_controls,
    validate_model_sceneplan,
)


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
DEFAULT_SCENEPLANS = REVISION_ROOT / "sceneplans_model_v2_delta"
DEFAULT_DONORS = REVISION_ROOT / (
    "source_annotations/speech_speaker_instruct_v1/registry/"
    "final_speech_donors_with_speakers.parquet"
)
DEFAULT_OUTPUT = DEFAULT_SCENEPLANS / "audit.json"
EXPECTED_PATTERNS = {
    "direct_speech_only": 75_000,
    "direct_speech_plus_music": 62_500,
    "direct_speech_plus_sound": 62_500,
    "sequential_speech_then_sound": 75_000,
    "sequential_sound_then_speech": 75_000,
    "sequential_speech_then_music": 75_000,
    "sequential_music_then_speech": 75_000,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def clean_background_hashes() -> set[str]:
    paths = (
        DATASET_ROOT
        / "source_annotations/nonspeech_instruct_v2/registry/source_description_registry.parquet",
        DATASET_ROOT
        / "revisions/sound_expansion_v1/source_annotations/registry/source_description_registry.parquet",
    )
    output = set()
    for path in paths:
        table = pq.read_table(
            path,
            filters=[("split", "=", "train"), ("spoken_language_background", "=", False)],
            columns=["source_audio_sha256"],
        )
        output.update(map(str, table["source_audio_sha256"].to_pylist()))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sceneplan-root", type=Path, default=DEFAULT_SCENEPLANS)
    parser.add_argument("--donors", type=Path, default=DEFAULT_DONORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.sceneplan_root.expanduser().resolve(strict=True)
    donors_path = args.donors.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    require(summary.get("state") == "ready_for_p8", "full ScenePlan summary is not ready")
    require(int(summary.get("rows", -1)) == 500_000, "ScenePlan summary is not 500k")
    require(not (REVISION_ROOT / "materialized_delta/P8_SUMMARY.json").exists(), "P8 already started")
    donors = pq.read_table(donors_path).to_pylist()
    donor_assets = {str(row["asset_id"]) for row in donors}
    require(len(donor_assets) == 200_000, "direct donor registry is not 200k unique")
    clean_hashes = clean_background_hashes()
    index = pq.read_table(root / "index.parquet").to_pylist()
    indexed = {str(row["sample_id"]): row for row in index}
    require(len(indexed) == len(index) == 500_000, "ScenePlan index IDs are not 500k unique")
    paths = sorted((root / "train").glob("model-sceneplans-train-*.jsonl"))
    require(len(paths) == int(summary["shards"]), "ScenePlan shard coverage changed")

    started = time.monotonic()
    patterns: Counter[str] = Counter()
    buckets: Counter[int] = Counter()
    rooms: Counter[str] = Counter()
    source_kinds: Counter[str] = Counter()
    motions: Counter[str] = Counter()
    direct_speech_assets: set[str] = set()
    sequential_speech_assets: set[str] = set()
    background_hashes_seen: set[str] = set()
    caption_tokens: list[int] = []
    duration_values: list[float] = []
    seen_ids: set[str] = set()
    for shard_index, path in enumerate(paths, start=1):
        rows = load_shard_rows(path)
        for row in rows:
            sample_id = str(row["sample_id"])
            require(sample_id not in seen_ids, f"duplicate ScenePlan ID: {sample_id}")
            seen_ids.add(sample_id)
            frozen = indexed.get(sample_id)
            require(frozen is not None, f"ScenePlan absent from index: {sample_id}")
            for key in (
                "model_sceneplan_sha256",
                "render_recipe_sha256",
                "renderer_caption_sha256",
            ):
                require(row[key] == frozen[key], f"{sample_id}: {key} mismatch")
            model = json.loads(row["model_sceneplan_json"])
            recipe = json.loads(row["render_recipe_json"])
            validate_model_sceneplan(model)
            require(int(recipe.get("dataset_contract_revision", -1)) == 6, f"{sample_id}: not revision 6")
            require(recipe.get("mixing", {}).get("speech_background_mode") in {"not_applicable", "overlap_calibrated", "sequential_nonoverlap"}, f"{sample_id}: invalid mixing mode")
            pattern = str(recipe["temporal_pattern"])
            require(pattern in EXPECTED_PATTERNS, f"{sample_id}: unknown temporal pattern")
            patterns[pattern] += 1
            audio = recipe["audio_execution"]
            frames = int(audio["latent_frames_valid"])
            samples = int(audio["model_num_samples"])
            require(frames == math.ceil(samples / 1024), f"{sample_id}: frame/sample mismatch")
            bucket = 432 if frames <= 432 else 648
            buckets[bucket] += 1
            duration_values.append(samples / 44_100.0)
            require(samples <= 648 * 1024 and frames <= 648, f"{sample_id}: exceeds 15-second envelope")
            require(abs(float(model["duration_sec"]) - samples / 44_100.0) <= 1.1e-6, f"{sample_id}: duration mismatch")
            caption_tokens.append(int(frozen["caption_qwen_tokens"]))
            require(caption_tokens[-1] <= 512, f"{sample_id}: caption exceeds 512 tokens")
            speech = [source for source in model["sources"] if source["kind"] == "speech"]
            background = [source for source in model["sources"] if source["kind"] != "speech"]
            require(len(speech) == 1, f"{sample_id}: formal speech count is not one")
            require(len(background) == (0 if pattern == "direct_speech_only" else 1), f"{sample_id}: background count mismatch")
            recipe_by_id = {source["source_id"]: source for source in recipe["sources"]}
            speech_asset = str(recipe_by_id[speech[0]["source_id"]]["asset_ref"]["asset_id"])
            if pattern.startswith("direct_"):
                require(speech_asset in donor_assets, f"{sample_id}: direct scene does not use new donor")
                require(speech_asset not in direct_speech_assets, f"{sample_id}: direct donor reused")
                direct_speech_assets.add(speech_asset)
            else:
                require(speech_asset not in donor_assets, f"{sample_id}: sequential scene unexpectedly uses new donor")
                require(speech_asset not in sequential_speech_assets, f"{sample_id}: sequential speech donor reused")
                sequential_speech_assets.add(speech_asset)
            intervals = {
                source["kind"]: (
                    float(source["activity"]["onset_sec"]),
                    float(source["activity"]["offset_sec"]),
                )
                for source in model["sources"]
            }
            if background:
                bg = background[0]
                source_kinds[bg["kind"]] += 1
                asset = recipe_by_id[bg["source_id"]]["asset_ref"]
                bg_hash = str(asset["identity_hash"])
                require(bg_hash in clean_hashes, f"{sample_id}: background is absent from clean registry")
                background_hashes_seen.add(bg_hash)
                speech_interval = intervals["speech"]
                bg_interval = intervals[bg["kind"]]
                overlap = min(speech_interval[1], bg_interval[1]) - max(speech_interval[0], bg_interval[0])
                if pattern.startswith("sequential_"):
                    require(overlap <= 1.1e-6, f"{sample_id}: sequential activities overlap")
                    require(recipe["mixing"]["speech_background_mode"] == "sequential_nonoverlap", f"{sample_id}: sequential mixing mode changed")
                else:
                    require(overlap >= 0.10 - 1.1e-6, f"{sample_id}: direct mixed overlap <100ms")
                    require(recipe["mixing"]["speech_background_mode"] == "overlap_calibrated", f"{sample_id}: direct mixing mode changed")
            else:
                require(recipe["mixing"]["speech_background_mode"] == "not_applicable", f"{sample_id}: speech-only mixing mode changed")
            controls = compile_model_44_controls(
                model,
                model_num_samples=samples,
                latent_frames_valid=frames,
            )
            require(controls["source_event_frame_ids"].shape == (4, frames), f"{sample_id}: 4-event shape changed")
            require(controls["source_trajectory_features"].shape == (4, frames, 5), f"{sample_id}: 4-trajectory shape changed")
            rooms[str(model["room"]["type"])] += 1
            for source in model["sources"]:
                motions[str(source["trajectory"]["type"])] += 1
        if shard_index % 50 == 0 or shard_index == len(paths):
            print(json.dumps({"audited_shards": shard_index, "total_shards": len(paths), "rows": len(seen_ids), "elapsed_sec": round(time.monotonic() - started, 1)}), flush=True)

    require(len(seen_ids) == 500_000, "ScenePlan exhaustive row count changed")
    require(dict(patterns) == EXPECTED_PATTERNS, f"temporal quotas changed: {patterns}")
    require(buckets == Counter({432: 100_000, 648: 400_000}), f"length buckets changed: {buckets}")
    require(len(direct_speech_assets) == 200_000, "direct donor coverage is not exact")
    require(len(sequential_speech_assets) == 300_000, "sequential donor coverage is not exact")
    require(not (direct_speech_assets & sequential_speech_assets), "direct/sequential speech assets overlap")
    require(source_kinds == Counter({"music": 212_500, "sound": 212_500}), f"background balance changed: {source_kinds}")
    token_p99 = float(np.percentile(caption_tokens, 99))
    require(token_p99 <= 384.0 and max(caption_tokens) <= 512, "caption token envelope failed")
    audit = {
        "schema": "stable_audio_tools.sceneplan_speech_expansion_manifest_audit",
        "schema_version": 1,
        "dataset_contract_revision": 6,
        "ok": True,
        "rows": len(seen_ids),
        "shards": len(paths),
        "temporal_pattern_counts": dict(patterns),
        "length_bucket_counts": {str(key): value for key, value in buckets.items()},
        "duration_sec": {"min": min(duration_values), "p50": float(np.percentile(duration_values, 50)), "p99": float(np.percentile(duration_values, 99)), "max": max(duration_values)},
        "room_counts": dict(rooms),
        "motion_counts": dict(motions),
        "source_kind_counts": {"speech": 500_000, **dict(source_kinds)},
        "unique_direct_speech_assets": len(direct_speech_assets),
        "unique_sequential_speech_assets": len(sequential_speech_assets),
        "unique_background_assets_used": len(background_hashes_seen),
        "caption_qwen_tokens": {"p99": token_p99, "max": max(caption_tokens), "truncated": 0},
        "formal_speech_sources_per_scene_max": 1,
        "spoken_language_background_rows": 0,
        "speech_timing_sidecar": None,
        "word_level_timestamp_teacher": False,
        "four_event_plus_four_trajectory_verified": True,
        "p8_started": False,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
