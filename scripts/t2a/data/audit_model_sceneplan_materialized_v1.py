#!/usr/bin/env python3
"""Fail-closed exhaustive P9 audit for revision-5 P8 materialization."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors import safe_open


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from audit_sceneplan_materialized_v2 import (  # noqa: E402
    CONFIG,
    TRUE_PEAK_CEILING,
    audit_audio,
    directory_bytes,
    expected_quotas,
)
from build_model_sceneplan_training_index_v1 import read_jsonl_with_offsets  # noqa: E402
from materialize_model_sceneplan_v1_shard import canonical_json, sha256_text  # noqa: E402
from sceneplan_v2_common import (  # noqa: E402
    DATASET_ROOT,
    MAX_LATENT_FRAMES,
    MODEL_SAMPLE_RATE,
    atomic_write_json,
    deterministic_digest,
)
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    compile_model_renderer_caption,
    validate_model_sceneplan,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expected_keyframe_count(source: dict[str, Any]) -> int:
    trajectory = source["trajectory"]
    if trajectory["type"] == "static":
        return 1
    if trajectory["type"] == "linear":
        return 2
    return len(trajectory["keyframes"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sceneplan-root", type=Path, default=DATASET_ROOT / "sceneplans_model_v1"
    )
    parser.add_argument(
        "--materialized-root", type=Path, default=DATASET_ROOT / "materialized"
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--audio-workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument(
        "--output",
        type=Path,
        default=DATASET_ROOT / "qc/p9_model_sceneplan_materialized_audit.json",
    )
    args = parser.parse_args()
    sceneplan_root = args.sceneplan_root.expanduser().resolve(strict=True)
    materialized_root = args.materialized_root.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve(strict=False)
    try:
        sceneplan_root.relative_to("/mnt/sdb")
        materialized_root.relative_to("/mnt/sdb")
        output.relative_to("/mnt/sdb")
    except ValueError as error:
        raise ValueError("P9 inputs and audit output must be on SDB") from error
    p8_summary = json.loads(
        (materialized_root / "p8_orchestrator_summary.json").read_text(encoding="utf-8")
    )
    require(
        p8_summary.get("schema")
        == "stable_audio_tools.model_sceneplan_p8_orchestrator_summary"
        and p8_summary.get("status") == "complete"
        and int(p8_summary.get("dataset_contract_revision", -1)) == 5
        and int(p8_summary.get("rows", -1)) == 1_124_000
        and p8_summary.get("p10_training_started") is False
        and p8_summary.get("p11_training_started") is False,
        "formal revision-5 P8 summary is not complete",
    )
    config = json.loads(
        args.config.expanduser().resolve(strict=True).read_text(encoding="utf-8")
    )
    expected = expected_quotas(config, "full")
    expected_rows = sum(expected.values())
    plan_table = pq.read_table(
        sceneplan_root / "index.parquet",
        columns=[
            "sample_id",
            "split",
            "family",
            "source_count",
            "model_num_samples",
            "latent_frames_valid",
            "work_shard",
            "row_in_shard",
            "model_sceneplan_sha256",
            "render_recipe_sha256",
            "renderer_caption_sha256",
        ],
    )
    require(plan_table.num_rows == expected_rows, "P7.5 index row count mismatch")
    planned = {str(row["sample_id"]): row for row in plan_table.to_pylist()}
    require(len(planned) == expected_rows, "P7.5 sample ids are not unique")
    manifests = sorted(
        (materialized_root / "manifests").glob("*/materialized-*.parquet")
    )
    require(len(manifests) == 1_099, "formal P8 manifest shard count is not 1,099")
    require(
        not list((materialized_root / "quarantine").glob("*/*.json")),
        "P8 quarantine is not empty",
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
    static_sources = 0
    dynamic_sources = 0
    rir_endpoint_diagnostics = 0
    started = time.time()

    for manifest_index, path in enumerate(manifests, start=1):
        rows = pq.read_table(path).to_pylist()
        require(bool(rows), f"empty materialized manifest: {path}")
        manifest_bytes += path.stat().st_size
        split = str(rows[0]["split"])
        shard = int(rows[0]["work_shard"])
        require(
            all(str(row["split"]) == split and int(row["work_shard"]) == shard for row in rows),
            f"{path}: split/work-shard fields differ",
        )
        model_path = (
            sceneplan_root / split / f"model-sceneplans-{split}-{shard:05d}.jsonl"
        )
        recipe_path = (
            sceneplan_root / split / f"render-recipes-{split}-{shard:05d}.jsonl"
        )
        model_lines = read_jsonl_with_offsets(model_path)
        recipe_lines = read_jsonl_with_offsets(recipe_path)
        require(
            len(rows) == len(model_lines) == len(recipe_lines),
            f"{path}: P7.5/P8 row counts differ",
        )
        models = {str(value[0]["sample_id"]): value for value in model_lines}
        recipes = {str(value[0]["sample_id"]): value for value in recipe_lines}
        require(len(models) == len(rows) and len(recipes) == len(rows), f"{path}: duplicate view ids")

        latent_paths = {str(row["latent_ref"]).split("#", 1)[0] for row in rows}
        require(len(latent_paths) == 1, f"{path}: multiple latent shards")
        latent_path = Path(latent_paths.pop())
        require(latent_path.is_file(), f"missing latent shard: {latent_path}")
        expected_shard_sha = {str(row["latent_shard_sha256"]) for row in rows}
        require(len(expected_shard_sha) == 1, f"{path}: inconsistent latent shard hashes")
        require(
            sha256_file(latent_path) == expected_shard_sha.pop(),
            f"latent shard SHA256 mismatch: {latent_path}",
        )
        latent_bytes += latent_path.stat().st_size
        row_by_id = {str(row["sample_id"]): row for row in rows}
        require(len(row_by_id) == len(rows), f"{path}: duplicate materialized ids")
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
            require(sample_id in planned, f"{sample_id}: absent from P7.5 index")
            require(sample_id not in seen_ids, f"{sample_id}: materialized twice")
            seen_ids.add(sample_id)
            plan = planned[sample_id]
            model, model_text, _, _ = models[sample_id]
            recipe, recipe_text, _, _ = recipes[sample_id]
            validate_model_sceneplan(model)
            model_sha = sha256_text(model_text)
            recipe_sha = sha256_text(recipe_text)
            caption = compile_model_renderer_caption(model)
            caption_sha = sha256_text(canonical_json(caption))
            bundle_sha = sha256_text(
                canonical_json(
                    {
                        "model_sceneplan_sha256": model_sha,
                        "render_recipe_sha256": recipe_sha,
                        "renderer_caption_sha256": caption_sha,
                    }
                )
            )
            require(
                row["model_sceneplan_sha256"]
                == plan["model_sceneplan_sha256"]
                == model_sha
                and row["render_recipe_sha256"]
                == plan["render_recipe_sha256"]
                == recipe_sha
                and row["renderer_caption_sha256"]
                == plan["renderer_caption_sha256"]
                == caption_sha
                and row["planned_bundle_sha256"] == bundle_sha,
                f"{sample_id}: P7.5 three-view lineage changed",
            )
            require(recipe["model_sceneplan_sha256"] == model_sha, f"{sample_id}: recipe/model link drift")
            require(
                row["split"] == plan["split"]
                and row["family"] == plan["family"]
                and int(row["source_count"]) == int(plan["source_count"])
                and int(row["model_num_samples"]) == int(plan["model_num_samples"])
                and int(row["latent_frames_valid"]) == int(plan["latent_frames_valid"])
                and int(row["work_shard"]) == int(plan["work_shard"])
                and int(row["row_in_shard"]) == int(plan["row_in_shard"]),
                f"{sample_id}: P7.5/P8 scalar fields mismatch",
            )
            expected_seed = int(
                deterministic_digest(20260814, "vae-encode", split, shard)[:16], 16
            )
            require(int(row["vae_encode_seed"]) == expected_seed, f"{sample_id}: VAE seed drift")
            result_text = str(row["render_result_json"])
            result = json.loads(result_text)
            require(result_text == canonical_json(result), f"{sample_id}: noncanonical render result")
            require(
                result.get("status") == "ok"
                and result.get("sample_id") == sample_id
                and int(result.get("dataset_contract_revision", -1)) == 5
                and result.get("planned_record_sha256") == bundle_sha
                and result.get("planned_bundle_sha256") == bundle_sha
                and result.get("model_sceneplan_sha256") == model_sha
                and result.get("render_recipe_sha256") == recipe_sha
                and result.get("renderer_caption_sha256") == caption_sha
                and result.get("model_sceneplan_immutable_during_materialization") is True
                and len(str(result.get("materialized_execution_sha256") or "")) == 64,
                f"{sample_id}: render result lineage/status mismatch",
            )
            require(
                int(result["num_samples"]) == int(row["model_num_samples"])
                and int(result["latent_frames_valid"]) == int(row["latent_frames_valid"]),
                f"{sample_id}: render geometry mismatch",
            )
            require(
                float(result["true_peak"]) <= TRUE_PEAK_CEILING + 1e-5
                and float(result["stored_true_peak"]) <= TRUE_PEAK_CEILING + 2e-5,
                f"{sample_id}: true peak exceeds -1 dBFS",
            )
            model_sources = list(model["sources"])
            recipe_sources = list(recipe["sources"])
            source_qc = list(result["source_qc"])
            require(
                len(model_sources)
                == len(recipe_sources)
                == len(source_qc)
                == int(row["source_count"]),
                f"{sample_id}: source count/QC mismatch",
            )
            for model_source, recipe_source, qc in zip(
                model_sources, recipe_sources, source_qc
            ):
                require(
                    model_source["source_id"]
                    == recipe_source["source_id"]
                    == qc["source_id"]
                    and model_source["kind"]
                    == recipe_source["kind"]
                    == qc["kind"]
                    and recipe_source["asset_ref"]["asset_id"] == qc["asset_id"],
                    f"{sample_id}: source identity changed during render",
                )
                lineage = qc["content_lineage"]
                window = recipe_source["exact_source_sample_window"]
                require(
                    lineage["source_audio_sha256"]
                    == recipe_source["asset_ref"]["identity_hash"]
                    and float(lineage["coverage_fraction"]) == 1.0
                    and lineage["random_crop"] is False
                    and int(lineage["model_num_samples"])
                    == int(window["dry_end_sample"]) - int(window["dry_start_sample"]),
                    f"{sample_id}: complete-source content lineage mismatch",
                )
                renderer = qc["renderer_qc"]
                diagnostics = renderer["keyframe_rir_diagnostics"]
                require(
                    int(renderer["spatialization_passes"]) == 1
                    and renderer["foa_layout"] == "WYZX_ACN_SN3D"
                    and float(renderer["trajectory_weight_sum_max_abs_error"]) <= 1e-5
                    and len(diagnostics) == expected_keyframe_count(model_source),
                    f"{sample_id}: renderer trajectory/pass/layout mismatch",
                )
                for diagnostic in diagnostics:
                    require(
                        39 <= int(diagnostic["residual_direct_peak_sample"]) <= 41
                        and float(diagnostic["max_abs_direction_ratio_error"]) <= 5e-3,
                        f"{sample_id}: RIR alignment/direction diagnostic failed",
                    )
                rir_endpoint_diagnostics += len(diagnostics)
                if model_source["trajectory"]["type"] == "static":
                    static_sources += 1
                else:
                    dynamic_sources += 1

            loudness = result["loudness_qc"]
            if row["family"] == "speech" and int(row["source_count"]) > 1:
                require(loudness is not None, f"{sample_id}: missing speech/background QC")
                target_db = float(loudness["target_speech_to_aggregate_background_db"])
                measured_db = float(loudness["measured_speech_to_aggregate_background_db"])
                require(2.0 <= target_db <= 6.0, f"{sample_id}: loudness target outside 2-6 dB")
                require(abs(target_db - measured_db) <= 1e-4, f"{sample_id}: loudness target missed")
                require(
                    int(loudness["overlap_samples"]) >= round(0.1 * MODEL_SAMPLE_RATE),
                    f"{sample_id}: speech/background overlap below 100 ms",
                )
                target_ratios.append(target_db)
            else:
                require(loudness is None, f"{sample_id}: unexpected loudness QC")

            retained = split != "train"
            foa_path = Path(result["foa_path"])
            if retained:
                require(
                    row["foa_path"] == result["foa_path"] and foa_path.is_file(),
                    f"{sample_id}: retained FOA missing",
                )
                require(
                    len(result["stem_refs"]) == len(model_sources),
                    f"{sample_id}: retained stem count mismatch",
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
                    require(Path(reference["path"]).is_file(), f"{sample_id}: stem missing")
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
                require(row["foa_path"] is None, f"{sample_id}: train FOA retained in manifest")
                require(not foa_path.exists(), f"{sample_id}: transient train FOA remains")
                require(not result["stem_refs"], f"{sample_id}: train stems were retained")
            require(
                row["foa_sha256"] == result["foa_sha256"],
                f"{sample_id}: FOA checksum lineage mismatch",
            )
            counts[(str(row["split"]), str(row["family"]), int(row["source_count"]))] += 1
            seen_ids.add(sample_id)
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
    require(len(seen_ids) == expected_rows == len(planned), "materialized id set mismatch")
    require(counts == expected, f"joint quota mismatch: {counts - expected} / {expected - counts}")
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
    require(abs(target_median - required_median) <= 0.02, "global 0.6:0.4 median drift")
    stat = os.statvfs("/mnt/sdb")
    total_bytes = stat.f_blocks * stat.f_frsize
    free_bytes = stat.f_bavail * stat.f_frsize
    require(free_bytes / total_bytes >= 0.20, "SDB free fraction fell below 20%")
    observed_foa_seconds = retained_foa_samples / MODEL_SAMPLE_RATE
    observed_foa_bytes_per_second = audio_bytes["foa"] / observed_foa_seconds
    summary = {
        "schema": "stable_audio_tools.model_sceneplan_materialized_dataset_audit",
        "schema_version": 1,
        "dataset_contract_revision": 5,
        "ok": True,
        "rows": rows_seen,
        "manifests": len(manifests),
        "joint_counts": {
            "|".join(map(str, key)): value for key, value in sorted(counts.items())
        },
        "model_sceneplan_immutable_and_three_view_hashes_verified": True,
        "variable_latent_dtype": "float16",
        "latent_channels": 64,
        "latent_frames_valid_range": [min_frames, max_frames],
        "latent_bytes": latent_bytes,
        "all_latent_shard_and_tensor_checksums_verified": True,
        "retained_audio_file_counts": dict(audio_counts),
        "retained_audio_bytes": dict(audio_bytes),
        "all_retained_audio_checksums_geometry_finite_and_true_peaks_verified": True,
        "train_foa_cleanup_verified": True,
        "complete_source_lineage_no_crop_verified": True,
        "source_motion_counts": {"static": static_sources, "dynamic": dynamic_sources},
        "rir_endpoint_diagnostics_verified": rir_endpoint_diagnostics,
        "spatialization_passes": 1,
        "foa_layout": "WYZX_ACN_SN3D",
        "speech_to_aggregate_background_db": {
            "minimum": min(target_ratios),
            "median": target_median,
            "maximum": max(target_ratios),
            "required_median": required_median,
        },
        "storage": {
            "sceneplan_bytes": directory_bytes(sceneplan_root),
            "latent_bytes": latent_bytes,
            "materialized_manifest_bytes": manifest_bytes,
            "retained_audio_bytes": dict(audio_bytes),
            "observed_retained_foa_seconds": observed_foa_seconds,
            "observed_foa_bytes_per_second": observed_foa_bytes_per_second,
            "sdb_total_bytes": total_bytes,
            "sdb_free_bytes_at_audit": free_bytes,
            "sdb_free_fraction": free_bytes / total_bytes,
            "minimum_required_free_fraction": 0.20,
        },
        "p10_training_started": False,
        "p11_training_started": False,
        "elapsed_sec": round(time.time() - started, 3),
    }
    atomic_write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
