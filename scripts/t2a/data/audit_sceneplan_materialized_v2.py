#!/usr/bin/env python3
"""Fail-closed P6/P9 audit of rendered FOA, stems, and variable-length latents."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf
import torch
from safetensors import safe_open
from scipy.signal import resample_poly


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from sceneplan_v2_common import (  # noqa: E402
    expected_quotas,
    DATASET_ROOT,
    MAX_LATENT_FRAMES,
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
    atomic_write_json,
    deterministic_digest,
)


CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/construct_dataset/"
    "sceneplan_renderer_v2_1p124m.json"
)
TRUE_PEAK_CEILING = 10.0 ** (-1.0 / 20.0)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)




def true_peak(audio: np.ndarray) -> float:
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim == 1:
        value = value[:, None]
    oversampled = resample_poly(value, 4, 1, axis=0)
    return float(np.max(np.abs(oversampled))) if oversampled.size else 0.0


def audit_audio(task: dict[str, Any]) -> dict[str, Any]:
    path = Path(task["path"])
    blob = path.read_bytes()
    digest = sha256_bytes(blob)
    if digest != task["sha256"]:
        raise RuntimeError(f"audio SHA256 mismatch: {path}")
    audio, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    if rate != MODEL_SAMPLE_RATE or audio.shape != (int(task["num_samples"]), 4):
        raise RuntimeError(f"audio geometry mismatch: {path} {rate} {audio.shape}")
    if not np.isfinite(audio).all():
        raise RuntimeError(f"non-finite stored audio: {path}")
    measured_true_peak = None
    if task["role"] == "foa":
        measured_true_peak = true_peak(audio)
        if measured_true_peak > TRUE_PEAK_CEILING + 3e-5:
            raise RuntimeError(f"stored FOA true peak exceeds -1 dBFS: {path}")
        if abs(measured_true_peak - float(task["expected_true_peak"])) > 5e-5:
            raise RuntimeError(f"stored FOA true peak/result mismatch: {path}")
    return {
        "role": task["role"],
        "num_bytes": len(blob),
        "num_samples": len(audio),
        "true_peak": measured_true_peak,
    }


def directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--sceneplan-root", type=Path)
    parser.add_argument("--materialized-root", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--audio-workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sceneplan_root = (
        args.sceneplan_root
        or (
            DATASET_ROOT / "pilots/joint_4k/sceneplans"
            if args.mode == "pilot"
            else DATASET_ROOT / "sceneplans"
        )
    ).expanduser().resolve(strict=True)
    materialized_root = (
        args.materialized_root
        or (
            DATASET_ROOT / "pilots/joint_4k/materialized"
            if args.mode == "pilot"
            else DATASET_ROOT / "materialized"
        )
    ).expanduser().resolve(strict=True)
    output = (
        args.output
        or (
            DATASET_ROOT / "pilots/joint_4k/qc/materialized_audit.json"
            if args.mode == "pilot"
            else DATASET_ROOT / "qc/p9_materialized_dataset_audit.json"
        )
    ).expanduser().resolve(strict=False)
    try:
        output.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError(f"audit output must be on SDB: {output}") from error
    config = json.loads(args.config.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    expected = expected_quotas(config, args.mode)
    expected_rows = sum(expected.values())
    plan_index = pq.read_table(
        sceneplan_root / "index.parquet",
        columns=[
            "sample_id",
            "split",
            "family",
            "source_count",
            "model_num_samples",
            "latent_frames_valid",
            "record_sha256",
        ],
    )
    require(plan_index.num_rows == expected_rows, "planned index row count mismatch")
    planned = {
        str(row["sample_id"]): row for row in plan_index.to_pylist()
    }
    require(len(planned) == expected_rows, "planned sample ids are not unique")
    manifests = sorted(
        (materialized_root / "manifests").glob("*/materialized-*.parquet")
    )
    require(bool(manifests), "no materialized manifests found")
    require(
        not list((materialized_root / "quarantine").glob("*/*.json")),
        "materialization quarantine is not empty",
    )
    counts: Counter[tuple[str, str, int]] = Counter()
    seen_ids: set[str] = set()
    audio_tasks: list[dict[str, Any]] = []
    target_ratios: list[float] = []
    retained_foa_samples = 0
    latent_bytes = 0
    manifest_bytes = 0
    rows_seen = 0
    min_frames = MAX_LATENT_FRAMES
    max_frames = 0
    started = time.time()
    for manifest_index, path in enumerate(manifests, start=1):
        table = pq.read_table(path)
        rows = table.to_pylist()
        require(bool(rows), f"empty materialized manifest: {path}")
        manifest_bytes += path.stat().st_size
        latent_paths = {str(row["latent_ref"]).split("#", 1)[0] for row in rows}
        require(len(latent_paths) == 1, f"{path}: manifest references multiple latent shards")
        latent_path = Path(latent_paths.pop())
        require(latent_path.is_file(), f"missing latent shard: {latent_path}")
        expected_shard_sha = {str(row["latent_shard_sha256"]) for row in rows}
        require(len(expected_shard_sha) == 1, f"{path}: inconsistent latent shard SHA fields")
        require(
            sha256_file(latent_path) == expected_shard_sha.pop(),
            f"latent shard SHA256 mismatch: {latent_path}",
        )
        latent_bytes += latent_path.stat().st_size
        row_by_id = {str(row["sample_id"]): row for row in rows}
        require(len(row_by_id) == len(rows), f"{path}: duplicate sample ids")
        with safe_open(str(latent_path), framework="pt", device="cpu") as handle:
            require(set(handle.keys()) == set(row_by_id), f"{latent_path}: latent key set mismatch")
            for sample_id, row in row_by_id.items():
                tensor = handle.get_tensor(sample_id)
                frames = int(row["latent_frames_valid"])
                require(
                    tensor.dtype == torch.float16 and tuple(tensor.shape) == (64, frames),
                    f"{sample_id}: variable latent shape/dtype mismatch",
                )
                require(torch.isfinite(tensor).all().item(), f"{sample_id}: non-finite latent")
                require(
                    tensor_sha256(tensor) == row["latent_tensor_sha256"],
                    f"{sample_id}: latent tensor SHA256 mismatch",
                )
                min_frames = min(min_frames, frames)
                max_frames = max(max_frames, frames)
        for row in rows:
            sample_id = str(row["sample_id"])
            require(sample_id in planned, f"{sample_id}: absent from planned index")
            require(sample_id not in seen_ids, f"{sample_id}: materialized more than once")
            seen_ids.add(sample_id)
            plan = planned[sample_id]
            require(
                row["planned_record_sha256"] == plan["record_sha256"],
                f"{sample_id}: planned record SHA changed",
            )
            require(
                row["split"] == plan["split"]
                and row["family"] == plan["family"]
                and int(row["source_count"]) == int(plan["source_count"])
                and int(row["model_num_samples"]) == int(plan["model_num_samples"])
                and int(row["latent_frames_valid"]) == int(plan["latent_frames_valid"]),
                f"{sample_id}: planned/materialized scalar fields mismatch",
            )
            materialized_text = str(row["materialized_record_json"])
            materialized = json.loads(materialized_text)
            require(
                materialized_text == canonical_json(materialized)
                and sha256_bytes(materialized_text.encode("utf-8"))
                == row["materialized_record_sha256"],
                f"{sample_id}: materialized record canonical SHA mismatch",
            )
            target = materialized["target"]
            require(target["materialization_state"] == "encoded", f"{sample_id}: target is not encoded")
            require(
                target["retention"]
                == ("transient_train_foa" if row["split"] == "train" else "retained_eval_foa")
                and target["latent_dtype"] == "float16"
                and int(target["latent_channels"]) == 64,
                f"{sample_id}: encoded target retention/latent contract mismatch",
            )
            require(
                target["latent_ref"] == row["latent_ref"]
                and target["latent_sha256"] == row["latent_tensor_sha256"],
                f"{sample_id}: target latent reference mismatch",
            )
            expected_vae_seed = int(
                deterministic_digest(
                    20260814,
                    "vae-encode",
                    str(row["split"]),
                    int(row["work_shard"]),
                )[:16],
                16,
            )
            require(
                int(row["vae_encode_seed"]) == expected_vae_seed
                and int(target["vae_encode_seed"]) == expected_vae_seed,
                f"{sample_id}: VAE encode seed lineage mismatch",
            )
            result_text = str(row["render_result_json"])
            result = json.loads(result_text)
            require(result_text == canonical_json(result), f"{sample_id}: render result is not canonical")
            require(
                result["status"] == "ok"
                and result["sample_id"] == sample_id
                and result["planned_record_sha256"] == plan["record_sha256"],
                f"{sample_id}: render result status/lineage mismatch",
            )
            require(
                int(result["num_samples"]) == int(plan["model_num_samples"])
                and int(result["latent_frames_valid"]) == int(plan["latent_frames_valid"]),
                f"{sample_id}: render geometry metadata mismatch",
            )
            require(
                float(result["true_peak"]) <= TRUE_PEAK_CEILING + 1e-5
                and float(result["stored_true_peak"]) <= TRUE_PEAK_CEILING + 2e-5,
                f"{sample_id}: render result true peak exceeds -1 dBFS",
            )
            scene = result["materialized_scene_plan"]
            require(
                materialized["scene_plan"] == scene,
                f"{sample_id}: materialized record/rendered ScenePlan mismatch",
            )
            present = [source for source in scene["sources"] if source["present"]]
            require(len(present) == int(row["source_count"]), f"{sample_id}: rendered source count mismatch")
            require(len(result["source_qc"]) == len(present), f"{sample_id}: source QC count mismatch")
            for source, source_qc in zip(present, result["source_qc"]):
                lineage = source_qc["content_lineage"]
                require(
                    lineage["source_audio_sha256"] == source["asset_ref"]["identity_hash"]
                    and float(lineage["coverage_fraction"]) == 1.0
                    and lineage["random_crop"] is False,
                    f"{sample_id}: rendered complete-source lineage mismatch",
                )
                require(
                    int(lineage["model_num_samples"])
                    == int(source["activity"][0]["dry_end_sample"]),
                    f"{sample_id}: rendered source sample count mismatch",
                )
                renderer = source_qc["renderer_qc"]
                require(
                    int(renderer["spatialization_passes"]) == 1
                    and renderer["foa_layout"] == "WYZX_ACN_SN3D"
                    and float(renderer["trajectory_weight_sum_max_abs_error"]) <= 1e-5,
                    f"{sample_id}: renderer pass/layout/trajectory QC mismatch",
                )
                for diagnostic in renderer["keyframe_rir_diagnostics"]:
                    require(
                        39 <= int(diagnostic["residual_direct_peak_sample"]) <= 41
                        and float(diagnostic["max_abs_direction_ratio_error"]) <= 5e-3,
                        f"{sample_id}: spatial alignment/direction diagnostic failed",
                    )
            loudness = result["loudness_qc"]
            if row["family"] == "speech" and int(row["source_count"]) > 1:
                require(loudness is not None, f"{sample_id}: missing speech/background loudness QC")
                target_db = float(loudness["target_speech_to_aggregate_background_db"])
                measured_db = float(loudness["measured_speech_to_aggregate_background_db"])
                require(2.0 <= target_db <= 6.0, f"{sample_id}: loudness target outside 2-6 dB")
                require(abs(target_db - measured_db) <= 1e-4, f"{sample_id}: loudness target not achieved")
                require(int(loudness["overlap_samples"]) >= round(0.1 * MODEL_SAMPLE_RATE), f"{sample_id}: speech/background overlap below 100 ms")
                target_ratios.append(target_db)
            else:
                require(loudness is None, f"{sample_id}: unexpected loudness QC")
            retained = args.mode == "pilot" or str(row["split"]) != "train"
            foa_path = Path(result["foa_path"])
            if retained:
                require(
                    row["foa_path"] == result["foa_path"]
                    and target["foa_path"] == result["foa_path"]
                    and foa_path.is_file(),
                    f"{sample_id}: retained FOA reference/file mismatch",
                )
                require(
                    len(result["stem_refs"]) == len(present),
                    f"{sample_id}: retained source stem count mismatch",
                )
                audio_tasks.append(
                    {
                        "role": "foa",
                        "path": str(foa_path),
                        "sha256": result["foa_sha256"],
                        "num_samples": int(result["num_samples"]),
                        "expected_true_peak": float(result["stored_true_peak"]),
                    }
                )
                retained_foa_samples += int(result["num_samples"])
                for reference in result["stem_refs"]:
                    require(Path(reference["path"]).is_file(), f"{sample_id}: retained stem missing")
                    audio_tasks.append(
                        {
                            "role": "stem",
                            "path": reference["path"],
                            "sha256": reference["sha256"],
                            "num_samples": int(result["num_samples"]),
                            "expected_true_peak": None,
                        }
                    )
            else:
                require(
                    row["foa_path"] is None and target["foa_path"] is None,
                    f"{sample_id}: train FOA was retained in manifest",
                )
                require(not foa_path.exists(), f"{sample_id}: transient train FOA still exists")
                require(not result["stem_refs"], f"{sample_id}: train source stems were rendered")
            require(
                row["foa_sha256"] == result["foa_sha256"]
                and target["foa_sha256"] == result["foa_sha256"],
                f"{sample_id}: FOA checksum lineage mismatch",
            )
            counts[(str(row["split"]), str(row["family"]), int(row["source_count"]))] += 1
            rows_seen += 1
        if manifest_index % 25 == 0 or manifest_index == len(manifests):
            print(
                json.dumps(
                    {
                        "audited_manifests": manifest_index,
                        "total_manifests": len(manifests),
                        "rows": rows_seen,
                        "audio_files_queued": len(audio_tasks),
                        "elapsed_sec": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    require(rows_seen == expected_rows, "materialized row total mismatch")
    require(len(seen_ids) == expected_rows == len(planned), "planned/materialized id set size mismatch")
    require(counts == expected, f"materialized joint quotas mismatch: {counts - expected} / {expected - counts}")
    audio_counts: Counter[str] = Counter()
    audio_bytes: Counter[str] = Counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.audio_workers) as pool:
        for index, result in enumerate(pool.map(audit_audio, audio_tasks, chunksize=4), start=1):
            audio_counts[result["role"]] += 1
            audio_bytes[result["role"]] += int(result["num_bytes"])
            if index % 1000 == 0 or index == len(audio_tasks):
                print(
                    json.dumps(
                        {
                            "audited_audio_files": index,
                            "total_audio_files": len(audio_tasks),
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
    target_median = float(np.median(target_ratios))
    required_median = 20.0 * math.log10(0.6 / 0.4)
    require(
        abs(target_median - required_median) <= (0.1 if args.mode == "pilot" else 0.02),
        "materialized speech/background median does not match 0.6:0.4",
    )
    observed_foa_seconds = retained_foa_samples / MODEL_SAMPLE_RATE
    require(observed_foa_seconds > 0.0, "no retained FOA duration available for storage audit")
    observed_foa_bytes_per_second = audio_bytes["foa"] / observed_foa_seconds
    require(
        math.isfinite(observed_foa_bytes_per_second)
        and observed_foa_bytes_per_second > 0.0,
        "invalid observed FOA bytes/sec",
    )
    storage: dict[str, Any]
    stat = os.statvfs("/mnt/sdb")
    total_bytes = stat.f_blocks * stat.f_frsize
    free_bytes = stat.f_bavail * stat.f_frsize
    if args.mode == "pilot":
        plan_bytes = directory_bytes(sceneplan_root)
        projected_plan = math.ceil(plan_bytes / rows_seen * 1_124_000)
        projected_latents = math.ceil(latent_bytes / rows_seen * 1_124_000)
        projected_manifests = math.ceil(manifest_bytes / rows_seen * 1_124_000)
        projected_eval_audio = math.ceil(sum(audio_bytes.values()) / rows_seen * 24_000)
        projected_train_staging = math.ceil(audio_bytes["foa"] / rows_seen * (8 * 1024))
        projected_total = (
            projected_plan
            + projected_latents
            + projected_manifests
            + projected_eval_audio
            + projected_train_staging
        )
        projected_free_fraction = (free_bytes - projected_total) / total_bytes
        require(projected_free_fraction >= 0.20, "projected SDB free fraction falls below 20%")
        storage = {
            "basis_rows": rows_seen,
            "observed_retained_foa_seconds": observed_foa_seconds,
            "observed_retained_foa_bytes": audio_bytes["foa"],
            "observed_foa_bytes_per_second": observed_foa_bytes_per_second,
            "observed_mean_scene_duration_seconds": observed_foa_seconds / rows_seen,
            "observed_sceneplan_bytes": plan_bytes,
            "observed_latent_bytes": latent_bytes,
            "observed_materialized_manifest_bytes": manifest_bytes,
            "observed_retained_audio_bytes": dict(audio_bytes),
            "projected_sceneplan_bytes_1p124m": projected_plan,
            "projected_latent_bytes_1p124m": projected_latents,
            "projected_materialized_manifest_bytes_1p124m": projected_manifests,
            "projected_retained_eval_audio_bytes_24k": projected_eval_audio,
            "projected_peak_train_foa_staging_bytes_8_shards": projected_train_staging,
            "projected_incremental_bytes": projected_total,
            "sdb_total_bytes": total_bytes,
            "sdb_free_bytes_at_audit": free_bytes,
            "projected_free_fraction": projected_free_fraction,
            "minimum_required_free_fraction": 0.20,
        }
    else:
        storage = {
            "observed_retained_foa_seconds": observed_foa_seconds,
            "observed_retained_foa_bytes": audio_bytes["foa"],
            "observed_foa_bytes_per_second": observed_foa_bytes_per_second,
            "sceneplan_bytes": directory_bytes(sceneplan_root),
            "latent_bytes": latent_bytes,
            "materialized_manifest_bytes": manifest_bytes,
            "retained_audio_bytes": dict(audio_bytes),
            "sdb_total_bytes": total_bytes,
            "sdb_free_bytes_at_audit": free_bytes,
            "sdb_free_fraction": free_bytes / total_bytes,
        }
    summary = {
        "schema": "stable_audio_tools.sceneplan_materialized_dataset_audit",
        "schema_version": 2,
        "ok": True,
        "mode": args.mode,
        "rows": rows_seen,
        "manifests": len(manifests),
        "joint_counts": {"|".join(map(str, key)): value for key, value in sorted(counts.items())},
        "variable_latent_dtype": "float16",
        "latent_channels": 64,
        "latent_frames_valid_range": [min_frames, max_frames],
        "latent_bytes": latent_bytes,
        "all_latent_tensor_checksums_verified": True,
        "retained_audio_file_counts": dict(audio_counts),
        "retained_audio_bytes": dict(audio_bytes),
        "all_retained_audio_checksums_geometry_finite_and_foa_true_peaks_verified": True,
        "train_foa_cleanup_verified": args.mode == "full",
        "complete_source_lineage_no_crop_verified": True,
        "spatialization_passes": 1,
        "foa_layout": "WYZX_ACN_SN3D",
        "speech_to_aggregate_background_db": {
            "minimum": min(target_ratios),
            "median": target_median,
            "maximum": max(target_ratios),
            "required_median": required_median,
        },
        "storage": storage,
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
