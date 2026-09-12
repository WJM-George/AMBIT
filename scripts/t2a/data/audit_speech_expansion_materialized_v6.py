#!/usr/bin/env python3
"""Exhaustively audit revision-6 FOA rendering and variable-length latents."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
DEFAULT_SCENEPLANS = REVISION_ROOT / "sceneplans_model_v2_delta"
DEFAULT_MATERIALIZED = REVISION_ROOT / "materialized_delta"
DEFAULT_OUTPUT = DEFAULT_MATERIALIZED / "audit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sceneplan-root", type=Path, default=DEFAULT_SCENEPLANS)
    parser.add_argument("--materialized-root", type=Path, default=DEFAULT_MATERIALIZED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sceneplans = args.sceneplan_root.expanduser().resolve(strict=True)
    materialized = args.materialized_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    p8 = json.loads((materialized / "P8_SUMMARY.json").read_text(encoding="utf-8"))
    require(p8.get("status") == "complete" and int(p8.get("rows", -1)) == 500_000, "P8 is not complete")
    manifests = sorted((materialized / "manifests/train").glob("materialized-train-*.parquet"))
    require(len(manifests) == int(p8["shards"]), "materialized shard coverage changed")
    planned_index = {
        str(row["sample_id"]): row
        for row in pq.read_table(sceneplans / "index.parquet").to_pylist()
    }
    require(len(planned_index) == 500_000, "planned index coverage changed")

    started = time.monotonic()
    seen: set[str] = set()
    buckets: Counter[int] = Counter()
    modes: Counter[str] = Counter()
    source_kinds: Counter[str] = Counter()
    latent_bytes = 0
    latent_shard_bytes = 0
    direct_ratios: list[float] = []
    frames_min = 10**9
    frames_max = 0
    for manifest_index, manifest in enumerate(manifests, start=1):
        rows = pq.read_table(manifest).to_pylist()
        require(rows, f"empty materialized manifest: {manifest}")
        rows.sort(key=lambda row: int(row["row_in_shard"]))
        refs = {str(row["latent_ref"]).split("#", 1)[0] for row in rows}
        shard_hashes = {str(row["latent_shard_sha256"]) for row in rows}
        require(len(refs) == len(shard_hashes) == 1, f"{manifest}: latent shard lineage changed")
        latent_path = Path(refs.pop()).resolve(strict=True)
        expected_shard_hash = shard_hashes.pop()
        require(sha256_file(latent_path) == expected_shard_hash, f"{latent_path}: shard hash changed")
        latent_shard_bytes += latent_path.stat().st_size
        with safe_open(str(latent_path), framework="pt", device="cpu") as handle:
            require(set(handle.keys()) == {str(row["sample_id"]) for row in rows}, f"{latent_path}: key coverage changed")
            for row in rows:
                sample_id = str(row["sample_id"])
                require(sample_id not in seen, f"duplicate materialized ID: {sample_id}")
                seen.add(sample_id)
                planned = planned_index.get(sample_id)
                require(planned is not None, f"materialized row was not planned: {sample_id}")
                for key in (
                    "model_sceneplan_sha256",
                    "render_recipe_sha256",
                    "renderer_caption_sha256",
                ):
                    require(str(row[key]) == str(planned[key]), f"{sample_id}: {key} changed")
                frames = int(row["latent_frames_valid"])
                samples = int(row["model_num_samples"])
                require(frames == math.ceil(samples / 1024) and 1 <= frames <= 648, f"{sample_id}: latent geometry changed")
                tensor = handle.get_tensor(sample_id)
                require(tensor.dtype == torch.float16 and tuple(tensor.shape) == (64, frames), f"{sample_id}: latent shape/dtype changed")
                require(torch.isfinite(tensor).all().item(), f"{sample_id}: latent is non-finite")
                require(tensor_sha256(tensor) == str(row["latent_tensor_sha256"]), f"{sample_id}: tensor checksum changed")
                latent_bytes += tensor.numel() * tensor.element_size()
                frames_min = min(frames_min, frames)
                frames_max = max(frames_max, frames)
                buckets[432 if frames <= 432 else 648] += 1
                result = json.loads(str(row["render_result_json"]))
                require(result.get("status") == "ok", f"{sample_id}: render result not ok")
                require(int(result.get("dataset_contract_revision", -1)) == 6, f"{sample_id}: render result not revision 6")
                require(result.get("foa_sha256") == row["foa_sha256"], f"{sample_id}: FOA lineage hash changed")
                require(row["foa_path"] is None, f"{sample_id}: train FOA retained in manifest")
                require(not Path(str(result["foa_path"])).exists(), f"{sample_id}: transient train FOA still exists")
                require(not result.get("stem_refs"), f"{sample_id}: train stems were retained")
                source_qc = result.get("source_qc") or []
                require(len(source_qc) == int(row["source_count"]), f"{sample_id}: source QC count changed")
                for source in source_qc:
                    source_kinds[str(source["kind"])] += 1
                    lineage = source["content_lineage"]
                    require(float(lineage["coverage_fraction"]) == 1.0 and lineage["random_crop"] is False, f"{sample_id}: source was cropped")
                    require(int(source["renderer_qc"]["spatialization_passes"]) == 1, f"{sample_id}: source spatialization count changed")
                loudness = result.get("loudness_qc")
                if int(row["source_count"]) == 1:
                    require(loudness is None, f"{sample_id}: speech-only has mixed loudness QC")
                    modes["not_applicable"] += 1
                else:
                    require(isinstance(loudness, dict), f"{sample_id}: mixed scene lacks loudness QC")
                    mode = str(loudness.get("mode"))
                    require(mode in {"overlap_calibrated", "nonoverlap_independent_rms_normalization"}, f"{sample_id}: invalid loudness mode")
                    modes[mode] += 1
                    if mode == "overlap_calibrated":
                        require(int(loudness["overlap_samples"]) >= round(0.10 * 44_100), f"{sample_id}: overlap QC is too short")
                        target = float(loudness["target_speech_to_aggregate_background_db"])
                        measured = float(loudness["measured_speech_to_aggregate_background_db"])
                        require(abs(target - measured) <= 1.0e-4, f"{sample_id}: overlap loudness calibration drift")
                        direct_ratios.append(target)
                    else:
                        require(int(loudness["overlap_samples"]) == 0, f"{sample_id}: sequential overlap is nonzero")
                        require(loudness["target_speech_to_aggregate_background_db"] is None, f"{sample_id}: sequential scene invented a ratio target")
        if manifest_index % 25 == 0 or manifest_index == len(manifests):
            print(json.dumps({"audited_manifests": manifest_index, "total_manifests": len(manifests), "rows": len(seen), "elapsed_sec": round(time.monotonic() - started, 1)}), flush=True)

    require(len(seen) == len(planned_index) == 500_000, "materialized ID coverage is incomplete")
    require(buckets == Counter({432: 100_000, 648: 400_000}), f"materialized buckets changed: {buckets}")
    require(modes == Counter({"not_applicable": 75_000, "overlap_calibrated": 125_000, "nonoverlap_independent_rms_normalization": 300_000}), f"materialized mixing modes changed: {modes}")
    require(source_kinds == Counter({"speech": 500_000, "music": 212_500, "sound": 212_500}), f"materialized source kinds changed: {source_kinds}")
    required_median = 20.0 * math.log10(0.6 / 0.4)
    measured_median = float(np.median(direct_ratios))
    require(abs(measured_median - required_median) <= 0.02, "overlap loudness median changed")
    storage = os.statvfs("/mnt/sdb")
    free_fraction = storage.f_bavail / storage.f_blocks
    require(free_fraction >= 0.20, "SDB free fraction fell below 20%")
    audit = {
        "schema": "stable_audio_tools.sceneplan_speech_expansion_materialized_audit",
        "schema_version": 1,
        "dataset_contract_revision": 6,
        "ok": True,
        "rows": len(seen),
        "manifests": len(manifests),
        "length_bucket_counts": {str(key): value for key, value in buckets.items()},
        "latent_frames_valid_range": [frames_min, frames_max],
        "latent_tensor_bytes": latent_bytes,
        "latent_shard_bytes": latent_shard_bytes,
        "latent_dtype": "float16",
        "latent_channels": 64,
        "all_latent_shard_and_tensor_checksums_verified": True,
        "source_kind_counts": dict(source_kinds),
        "mixing_mode_counts": dict(modes),
        "overlap_speech_to_background_db": {"median": measured_median, "required_median": required_median},
        "complete_source_lineage_no_crop_verified": True,
        "single_pass_foa_verified": True,
        "train_foa_cleanup_verified": True,
        "sdb_free_fraction": free_fraction,
        "p10_training_started": False,
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
