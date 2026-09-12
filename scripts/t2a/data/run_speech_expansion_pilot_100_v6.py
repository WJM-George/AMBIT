#!/usr/bin/env python3
"""Build, render, encode, and loader-audit the balanced revision-6 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pyarrow.parquet as pq
from safetensors import safe_open
import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.build_model_sceneplan_training_index_v1 import (  # noqa: E402
    build_split,
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


DATASET_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
PILOT_ROOT = REVISION_ROOT / "pilot_100"
SCENEPLANS = PILOT_ROOT / "sceneplans"
MATERIALIZED = PILOT_ROOT / "materialized"
INDEX_ROOT = PILOT_ROOT / "training_index"
TOKENIZER_ROOT = Path("/mnt/sdc/ckpts/pretrained/Qwen/Qwen3.5-0.8B")
DONOR_REGISTRY = REVISION_ROOT / (
    "source_annotations/speech_speaker_instruct_v1/registry/"
    "final_speech_donors_with_speakers.parquet"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(command: list[str]) -> None:
    print(json.dumps({"event": "run", "command": command}), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    gate_path = PILOT_ROOT / "PILOT_GATE.json"
    donor_registry_sha256 = sha256_file(DONOR_REGISTRY.resolve(strict=True))
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            gate.get("status") == "PASS"
            and gate.get("input_donor_registry_sha256") == donor_registry_sha256
            and all(
            Path(value["path"]).is_file()
            and sha256_file(Path(value["path"])) == value["sha256"]
            for value in gate.get("frozen_outputs", {}).values()
            )
        ):
            print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
            return 0
    started = time.monotonic()
    builder = REPO_ROOT / "scripts/t2a/data/build_speech_expansion_sceneplans_v6.py"
    run(
        [
            sys.executable,
            str(builder),
            "--pilot-balanced",
            "--output-root",
            str(SCENEPLANS),
        ]
    )
    summary_path = SCENEPLANS / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_patterns = {
        "direct_speech_only": 20,
        "direct_speech_plus_music": 20,
        "direct_speech_plus_sound": 20,
        "sequential_speech_then_sound": 10,
        "sequential_sound_then_speech": 10,
        "sequential_speech_then_music": 10,
        "sequential_music_then_speech": 10,
    }
    if not (
        summary.get("state") == "pilot_ready_for_p8"
        and int(summary.get("rows", -1)) == 100
        and summary.get("length_bucket_counts") == {"432": 30, "648": 70}
        and summary.get("temporal_pattern_counts") == expected_patterns
        and summary.get("speech_timing_sidecar") is None
        and summary.get("word_level_timestamp_teacher") is False
    ):
        raise RuntimeError("balanced revision-6 pilot planning contract failed")
    model_shards = sorted(
        (SCENEPLANS / "train").glob("model-sceneplans-train-*.jsonl")
    )
    if len(model_shards) != 1:
        raise RuntimeError("100-row pilot must occupy exactly one shard")
    materializer = (
        REPO_ROOT / "scripts/t2a/data/materialize_model_sceneplan_v1_shard.py"
    )
    run(
        [
            sys.executable,
            str(materializer),
            "--model-sceneplan-shard",
            str(model_shards[0]),
            "--output-root",
            str(MATERIALIZED),
            "--gpu",
            str(args.gpu),
            "--jobs",
            str(args.jobs),
            "--batch-size",
            str(args.batch_size),
            "--retain-stems",
            "--revision-mode",
        ]
    )
    manifests = sorted(
        (MATERIALIZED / "manifests/train").glob("materialized-train-*.parquet")
    )
    if len(manifests) != 1:
        raise RuntimeError("pilot materialization did not produce one manifest")
    rows = pq.read_table(manifests[0]).to_pylist()
    if len(rows) != 100:
        raise RuntimeError("pilot materialized row count changed")
    buckets: Counter[int] = Counter()
    modes: Counter[str] = Counter()
    source_kinds: Counter[str] = Counter()
    listening = []
    latent_paths = {str(row["latent_ref"]).split("#", 1)[0] for row in rows}
    if len(latent_paths) != 1:
        raise RuntimeError("pilot latent shard lineage changed")
    latent_path = Path(latent_paths.pop()).resolve(strict=True)
    expected_shard_hashes = {str(row["latent_shard_sha256"]) for row in rows}
    if expected_shard_hashes != {sha256_file(latent_path)}:
        raise RuntimeError("pilot latent shard checksum failed")
    by_id = {str(row["sample_id"]): row for row in rows}
    with safe_open(str(latent_path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(by_id):
            raise RuntimeError("pilot latent key coverage changed")
        for row_index, row in enumerate(rows):
            sample_id = str(row["sample_id"])
            frames = int(row["latent_frames_valid"])
            buckets[432 if frames <= 432 else 648] += 1
            tensor = handle.get_tensor(sample_id)
            if (
                tensor.dtype != torch.float16
                or tuple(tensor.shape) != (64, frames)
                or not torch.isfinite(tensor).all()
                or tensor_sha256(tensor) != str(row["latent_tensor_sha256"])
            ):
                raise RuntimeError(f"{sample_id}: pilot latent failed")
            result = json.loads(str(row["render_result_json"]))
            if (
                result.get("status") != "ok"
                or int(result.get("dataset_contract_revision", -1)) != 6
                or not Path(str(row["foa_path"])).is_file()
                or sha256_file(Path(str(row["foa_path"]))) != str(row["foa_sha256"])
                or len(result.get("stem_refs") or ()) != int(row["source_count"])
            ):
                raise RuntimeError(f"{sample_id}: pilot render lineage failed")
            for source in result["source_qc"]:
                source_kinds[str(source["kind"])] += 1
                if (
                    float(source["content_lineage"]["coverage_fraction"]) != 1.0
                    or source["content_lineage"]["random_crop"] is not False
                    or int(source["renderer_qc"]["spatialization_passes"]) != 1
                ):
                    raise RuntimeError(f"{sample_id}: pilot complete-source QC failed")
            loudness = result.get("loudness_qc")
            mode = "not_applicable" if loudness is None else str(loudness["mode"])
            modes[mode] += 1
            if mode == "overlap_calibrated" and abs(
                float(loudness["target_speech_to_aggregate_background_db"])
                - float(loudness["measured_speech_to_aggregate_background_db"])
            ) > 1.0e-4:
                raise RuntimeError(f"{sample_id}: pilot overlap ratio drift")
            if mode == "nonoverlap_independent_rms_normalization" and int(
                loudness["overlap_samples"]
            ) != 0:
                raise RuntimeError(f"{sample_id}: pilot sequential overlap drift")
            if row_index in {0, 10, 20, 30, 40, 50, 60, 70, 80, 90}:
                listening.append(
                    {
                        "sample_id": sample_id,
                        "duration_sec": int(row["model_num_samples"]) / 44_100.0,
                        "source_count": int(row["source_count"]),
                        "foa_path": str(row["foa_path"]),
                        "stem_paths": [item["path"] for item in result["stem_refs"]],
                    }
                )
    if buckets != Counter({432: 30, 648: 70}):
        raise RuntimeError(f"pilot latent buckets changed: {buckets}")
    if modes != Counter(
        {
            "not_applicable": 20,
            "overlap_calibrated": 40,
            "nonoverlap_independent_rms_normalization": 40,
        }
    ):
        raise RuntimeError(f"pilot mixing modes changed: {modes}")
    if source_kinds != Counter({"speech": 100, "music": 40, "sound": 40}):
        raise RuntimeError(f"pilot source kinds changed: {source_kinds}")

    index_summary = build_split(
        MATERIALIZED,
        SCENEPLANS,
        INDEX_ROOT,
        "train",
        100,
        contract_revision=6,
        model_sceneplan_schema_version=2,
        max_latent_frames=648,
        mode="speech_expansion_balanced_pilot100",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ROOT.resolve(strict=True), local_files_only=True
    )
    dataset = ScenePlanV2Dataset(
        INDEX_ROOT / "train.sqlite",
        tokenizer_spec=(tokenizer, 512),
        expected_num_samples=100,
        latent_crop_length=648,
        caption_max_tokens=512,
        random_crop=False,
        # This 100-row gate is deliberately executed before the revision-wide
        # P9 freeze exists.  It audits the exact production loader contract,
        # while the formal training indexes remain fail-closed behind P9.
        require_frozen=False,
    )
    selected = [dataset[index] for index in (0, 10, 20, 30, 40, 50, 60, 70, 80, 90)]
    conditioner = ScenePlan44LocalConditioner(output_dim=256)
    loader_receipts = []
    for bucket in (432, 648):
        group = [
            item for item in selected if item[1]["latent_bucket_frames"] == bucket
        ]
        batch = sceneplan_bucket_collation(group)
        latent_batch, metadata = batch
        encoded, valid = conditioner(
            [row["sceneplan_44"] for row in metadata], device=torch.device("cpu")
        )
        if (
            tuple(latent_batch.shape) != (len(group), 64, bucket)
            or tuple(encoded.shape) != (len(group), 256, bucket)
            or tuple(valid.shape) != (len(group), bucket)
            or not torch.isfinite(encoded).all()
        ):
            raise RuntimeError(f"pilot bucket {bucket} loader/conditioner failed")
        loader_receipts.append(
            {
                "bucket_frames": bucket,
                "batch_rows": len(group),
                "latent_shape": list(latent_batch.shape),
                "conditioning_shape": list(encoded.shape),
                "valid_frames": [int(row["padding_mask"][0].sum()) for row in metadata],
            }
        )
    gate = {
        "schema": "stable_audio_tools.sceneplan_speech_expansion_pilot100_gate",
        "schema_version": 1,
        "status": "PASS",
        "dataset_contract_revision": 6,
        "input_donor_registry_sha256": donor_registry_sha256,
        "rows": 100,
        "length_bucket_counts": {"432": 30, "648": 70},
        "temporal_pattern_counts": expected_patterns,
        "source_kind_counts": dict(source_kinds),
        "mixing_mode_counts": dict(modes),
        "all_sources_complete_and_uncropped": True,
        "single_spatialization_pass": True,
        "all_latents_finite_float16_with_checksums": True,
        "loader_and_4plus4": loader_receipts,
        "speech_timing_sidecar": None,
        "word_level_timestamp_teacher": False,
        "listening_samples": listening,
        "index_summary": index_summary,
        "frozen_outputs": {
            "sceneplan_summary": {
                "path": str(summary_path),
                "sha256": sha256_file(summary_path),
            },
            "materialized_manifest": {
                "path": str(manifests[0]),
                "sha256": sha256_file(manifests[0]),
            },
            "latent_shard": {
                "path": str(latent_path),
                "sha256": sha256_file(latent_path),
            },
            "training_index": {
                "path": str(INDEX_ROOT / "train.sqlite"),
                "sha256": sha256_file(INDEX_ROOT / "train.sqlite"),
            },
        },
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    atomic_json(gate_path, gate)
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
