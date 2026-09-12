#!/usr/bin/env python3
"""Render and VAE-encode one Transfusion Editing target work shard.

Unchanged sources retain their exact dry asset, playback window, and per-source
calibration.  The target starts from the source mixture's master gain, which
may only be reduced by the minimum amount required to respect the true-peak
ceiling; it is never boosted.  Optional source parity rerenders the old scene
and requires a byte-identical frozen P10 FOA SHA256 before the target is
accepted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing as mp
import os
import sqlite3
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.materialize_model_sceneplan_v1_shard import (  # noqa: E402
    expand_render_execution,
)
from scripts.t2a.data.materialize_sceneplan_v2_shard import (  # noqa: E402
    load_complete_source,
    load_vae,
    normalize_dry,
    write_pcm24,
)
from scripts.t2a.data.sceneplan_v2_renderer import (  # noqa: E402
    active_rms,
    render_complete_mono_source,
    true_peak,
)
from stable_audio_tools.data.sceneplan_transfusion_editing import (  # noqa: E402
    canonical_json,
    sha256_json,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)


MATERIALIZATION_CONTRACT = "sceneplan_transfusion_editing_materialized_target_v1"
PAIR_GAIN_POLICY = (
    "fixed_member_corrections_nonboosting_peak_safe_target_master_v1"
)
MODEL_SAMPLE_RATE = 44_100
MAX_MODEL_SAMPLES = 648 * 1024
TRUE_PEAK_CEILING = 10.0 ** (-1.0 / 20.0)
ALLOWED_PHYSICAL_GPUS = frozenset(range(16))

MATERIALIZED_SCHEMA = pa.schema(
    [
        ("pair_ordinal", pa.int64()),
        ("pair_id", pa.string()),
        ("split", pa.string()),
        ("work_shard", pa.int32()),
        ("row_in_shard", pa.int16()),
        ("source_sample_id", pa.string()),
        ("target_sample_id", pa.string()),
        ("operation_family", pa.string()),
        ("operation", pa.string()),
        ("model_num_samples", pa.int32()),
        ("latent_frames_valid", pa.int16()),
        ("source_latent_ref", pa.string()),
        ("source_latent_tensor_sha256", pa.string()),
        ("source_foa_sha256", pa.string()),
        ("source_parity_verified", pa.bool_()),
        ("target_foa_path", pa.string()),
        ("target_foa_sha256", pa.string()),
        ("target_latent_ref", pa.string()),
        ("target_latent_tensor_sha256", pa.string()),
        ("target_latent_shard_sha256", pa.string()),
        ("vae_encode_seed", pa.uint64()),
        ("target_render_result_json", pa.string()),
        ("target_render_result_sha256", pa.string()),
        ("pair_record_sha256", pa.string()),
    ]
)


def _unpack(value: bytes) -> Any:
    return json.loads(zlib.decompress(value))


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _interval_mask(source: Mapping[str, Any], num_samples: int) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=np.bool_)
    for interval in source["activity"]:
        mask[
            int(interval["model_onset_sample"]) : int(interval["model_offset_sample"])
        ] = True
    return mask


def _render_fixed_pair_gain(
    model_sceneplan: dict[str, Any],
    render_recipe: dict[str, Any],
    source_render_result: Mapping[str, Any],
    *,
    unchanged_source_ids: set[str],
    allow_nonboosting_peak_clamp: bool,
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    execution = expand_render_execution(model_sceneplan, render_recipe)
    num_samples = int(execution["audio"]["model_num_samples"])
    if not 0 < num_samples <= MAX_MODEL_SAMPLES:
        raise RuntimeError("Editing target is outside the P10 648-frame envelope")
    old_qc = {
        str(item["source_id"]): item
        for item in source_render_result.get("source_qc") or ()
    }
    stems: list[np.ndarray] = []
    source_qc: list[dict[str, Any]] = []
    target_source_rms = 10.0 ** (-24.0 / 20.0)
    for source in execution["sources"]:
        source_id = str(source["source_id"])
        mono, lineage = load_complete_source(
            source, max_model_samples=MAX_MODEL_SAMPLES
        )
        dry, dry_gain = normalize_dry(mono)
        interval = source["activity"][0]
        stem, renderer_qc = render_complete_mono_source(
            dry,
            sample_rate=MODEL_SAMPLE_RATE,
            scene_num_samples=num_samples,
            onset_sample=int(interval["model_onset_sample"]),
            room=execution["room"],
            keyframes=source["motion"]["keyframes"],
        )
        active = _interval_mask(source, num_samples)
        delayed = np.zeros_like(active)
        active_samples = np.flatnonzero(active)
        if not len(active_samples):
            raise RuntimeError(f"{source_id}: target source has no active samples")
        delayed_start = min(num_samples, int(active_samples[0]) + 40)
        delayed_stop = min(num_samples, int(active_samples[-1]) + 1 + 40)
        delayed[delayed_start:delayed_stop] = True
        raw_active_rms = active_rms(stem[0, delayed])
        if raw_active_rms < 1e-9:
            raise RuntimeError(f"{source_id}: rendered W channel is silent")
        normalization_gain = target_source_rms / raw_active_rms
        previous = old_qc.get(source_id)
        if previous is not None:
            if str(previous["asset_id"]) != str(source["asset_ref"]["asset_id"]):
                raise RuntimeError(f"{source_id}: existing source asset changed")
            if not math.isclose(
                float(dry_gain),
                float(previous["dry_normalization_gain"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(f"{source_id}: existing dry normalization drifted")
            correction_db = float(previous["calibrated_gain_correction_db"])
            if source_id in unchanged_source_ids and not math.isclose(
                float(normalization_gain),
                float(previous["source_rms_normalization_gain"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"{source_id}: unchanged rendered-source normalization drifted"
                )
        else:
            correction_db = 0.0
        planned_gain_db = float(source["gain_db"])
        # Preserve the operation order of the frozen P10 renderer so source
        # parity is byte-exact after PCM24 quantization.
        stem = stem * (
            normalization_gain * 10.0 ** (planned_gain_db / 20.0)
        )
        if correction_db:
            stem *= 10.0 ** (correction_db / 20.0)
        stems.append(stem.astype(np.float32, copy=False))
        source_qc.append(
            {
                "source_id": source_id,
                "slot": int(source["slot"]),
                "kind": str(source["kind"]),
                "asset_id": str(source["asset_ref"]["asset_id"]),
                "identity_hash": str(source["asset_ref"]["identity_hash"]),
                "planned_gain_db": planned_gain_db,
                "calibrated_gain_correction_db": correction_db,
                "actual_gain_db": planned_gain_db + correction_db,
                "source_rms_normalization_gain": normalization_gain,
                "dry_normalization_gain": dry_gain,
                "content_lineage": lineage,
                "renderer_qc": renderer_qc,
                "unchanged_member": source_id in unchanged_source_ids,
            }
        )
    mix = np.sum(stems, axis=0, dtype=np.float32)
    source_master_gain = float(source_render_result["master_gain"])
    if not math.isfinite(source_master_gain) or source_master_gain <= 0.0:
        raise RuntimeError("source master gain is invalid")
    raw_true_peak = true_peak(mix)
    if not math.isfinite(raw_true_peak) or raw_true_peak <= 0.0:
        raise RuntimeError("target raw mixture true peak is invalid")
    ceiling_gain = TRUE_PEAK_CEILING / raw_true_peak
    target_master_gain = source_master_gain
    if allow_nonboosting_peak_clamp:
        target_master_gain = min(source_master_gain, ceiling_gain)
    mix = (mix * target_master_gain).astype(np.float32, copy=False)
    stems = [
        (stem * target_master_gain).astype(np.float32, copy=False)
        for stem in stems
    ]
    measured_peak = true_peak(mix)
    gain_qc = {
        "policy": PAIR_GAIN_POLICY,
        "source_master_gain": source_master_gain,
        "target_master_gain": target_master_gain,
        "raw_target_true_peak": raw_true_peak,
        "target_peak_ceiling_gain": ceiling_gain,
        "target_master_clamped": target_master_gain < source_master_gain,
        "target_master_delta_db": 20.0
        * math.log10(target_master_gain / source_master_gain),
        "target_true_peak": measured_peak,
        "true_peak_ceiling": TRUE_PEAK_CEILING,
        "unchanged_member_gain_correction_reused": True,
        "added_member_gain_correction_db": 0.0,
    }
    if measured_peak > TRUE_PEAK_CEILING + 1e-5:
        raise RuntimeError(
            "pair target gain violates target -1 dBFS ceiling: "
            f"{measured_peak} > {TRUE_PEAK_CEILING}"
        )
    return mix, stems, source_qc, gain_qc


def _write_and_hash(path: Path, audio: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_pcm24(path, audio)
    stored, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate != MODEL_SAMPLE_RATE or stored.shape != (audio.shape[-1], 4):
        raise RuntimeError(f"stored FOA geometry changed: {path}")
    if true_peak(stored.T) > TRUE_PEAK_CEILING + 2e-5:
        raise RuntimeError(f"stored FOA exceeds -1 dBFS: {path}")
    return sha256_file(path)


def _render_one(
    row: dict[str, Any],
    output_root: str,
    retain_stems: bool,
    source_parity: bool,
) -> dict[str, Any]:
    started = time.time()
    pair_id = str(row["pair_id"])
    split = str(row["split"])
    work_shard = int(row["work_shard"])
    root = (
        Path(output_root)
        / "materialized/renders"
        / split
        / f"work-{work_shard:05d}"
        / str(row["target_sample_id"])
    )
    result_path = root / "target_render_result.json"
    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        foa_path = Path(existing.get("target_foa_path") or "")
        if (
            existing.get("status") == "ok"
            and existing.get("pair_record_sha256") == row["pair_record_sha256"]
            and foa_path.is_file()
            and sha256_file(foa_path) == existing.get("target_foa_sha256")
            and (
                not (source_parity and retain_stems)
                or bool(existing.get("source_parity_stem_refs"))
            )
        ):
            return existing
    try:
        old_plan = _unpack(row["old_sceneplan_zlib"])
        new_plan = _unpack(row["new_sceneplan_zlib"])
        source_recipe = _unpack(row["source_render_recipe_zlib"])
        target_recipe = _unpack(row["target_render_recipe_zlib"])
        source_result = _unpack(row["source_render_result_zlib"])
        unchanged = set(json.loads(row["unchanged_source_ids_json"]))
        source_parity_sha = None
        source_parity_stem_refs = []
        if source_parity:
            old_ids = {
                str(source["source_id"]) for source in old_plan["sources"]
            }
            old_mix, old_stems, _, _ = _render_fixed_pair_gain(
                old_plan,
                source_recipe,
                source_result,
                unchanged_source_ids=old_ids,
                allow_nonboosting_peak_clamp=False,
            )
            parity_path = root / ".source_parity_WYZX_SN3D.flac"
            source_parity_sha = _write_and_hash(parity_path, old_mix)
            parity_path.unlink(missing_ok=True)
            if source_parity_sha != str(row["source_foa_sha256"]):
                raise RuntimeError(
                    "source rerender is not byte-identical to frozen P10: "
                    f"{source_parity_sha} != {row['source_foa_sha256']}"
                )
            if retain_stems:
                old_source_ids = [
                    str(source["source_id"]) for source in old_plan["sources"]
                ]
                for source_id, stem in zip(old_source_ids, old_stems):
                    stem_path = root / f"source_stem_{source_id}_WYZX_SN3D.flac"
                    write_pcm24(stem_path, stem)
                    source_parity_stem_refs.append(
                        {
                            "source_id": source_id,
                            "path": str(stem_path),
                            "sha256": sha256_file(stem_path),
                        }
                    )
        target_mix, stems, source_qc, gain_qc = _render_fixed_pair_gain(
            new_plan,
            target_recipe,
            source_result,
            unchanged_source_ids=unchanged,
            allow_nonboosting_peak_clamp=True,
        )
        foa_path = root / "foa_WYZX_SN3D.flac"
        target_foa_sha = _write_and_hash(foa_path, target_mix)
        stem_refs = []
        if retain_stems:
            source_ids = [str(source["source_id"]) for source in new_plan["sources"]]
            for source_id, stem in zip(source_ids, stems):
                stem_path = root / f"stem_{source_id}_WYZX_SN3D.flac"
                write_pcm24(stem_path, stem)
                stem_refs.append(
                    {
                        "source_id": source_id,
                        "path": str(stem_path),
                        "sha256": sha256_file(stem_path),
                    }
                )
        result = {
            "schema": MATERIALIZATION_CONTRACT,
            "schema_version": 1,
            "status": "ok",
            "pair_id": pair_id,
            "pair_record_sha256": str(row["pair_record_sha256"]),
            "split": split,
            "work_shard": work_shard,
            "row_in_shard": int(row["row_in_shard"]),
            "source_sample_id": str(row["source_sample_id"]),
            "target_sample_id": str(row["target_sample_id"]),
            "operation_family": str(row["operation_family"]),
            "operation": str(row["operation"]),
            "model_num_samples": int(row["model_num_samples"]),
            "latent_frames_valid": int(row["latent_frames_valid"]),
            "old_sceneplan_sha256": str(row["old_sceneplan_sha256"]),
            "new_sceneplan_sha256": str(row["new_sceneplan_sha256"]),
            "source_render_recipe_sha256": str(
                row["source_render_recipe_sha256"]
            ),
            "target_render_recipe_sha256": str(
                row["target_render_recipe_sha256"]
            ),
            "source_foa_sha256": str(row["source_foa_sha256"]),
            "source_parity_requested": bool(source_parity),
            "source_parity_sha256": source_parity_sha,
            "source_parity_verified": bool(
                source_parity_sha == str(row["source_foa_sha256"])
                if source_parity
                else False
            ),
            "source_parity_stem_refs": source_parity_stem_refs,
            "target_foa_path": str(foa_path),
            "target_foa_sha256": target_foa_sha,
            "pair_gain_qc": gain_qc,
            "source_qc": source_qc,
            "stem_refs": stem_refs,
            "elapsed_sec": round(time.time() - started, 4),
        }
        result_text = canonical_json(result)
        root.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_name(result_path.name + f".tmp.{os.getpid()}")
        temporary.write_text(result_text + "\n", encoding="utf-8")
        os.replace(temporary, result_path)
        return result
    except Exception as error:  # noqa: BLE001
        return {
            "schema": MATERIALIZATION_CONTRACT,
            "schema_version": 1,
            "status": "error",
            "pair_id": pair_id,
            "pair_record_sha256": str(row["pair_record_sha256"]),
            "error": repr(error),
            "elapsed_sec": round(time.time() - started, 4),
        }


def _load_rows(index_path: Path, work_shard: int) -> tuple[dict[str, str], list[dict[str, Any]]]:
    uri = f"file:{index_path}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if (
            metadata.get("schema") != "sceneplan_transfusion_editing_pair_index"
            or metadata.get("state") != "planned_targets_not_materialized"
            or metadata.get("pair_gain_policy") != PAIR_GAIN_POLICY
        ):
            raise RuntimeError("pair index is not compatible planned Editing truth")
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM pairs WHERE work_shard=? ORDER BY row_in_shard",
                (int(work_shard),),
            )
        ]
    finally:
        connection.close()
    if not rows:
        raise RuntimeError(f"pair index has no work shard {work_shard}")
    expected_rows = list(range(len(rows)))
    if [int(row["row_in_shard"]) for row in rows] != expected_rows:
        raise RuntimeError("work shard row indices are not contiguous from zero")
    return metadata, rows


def _atomic_safetensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    save_file(
        dict(tensors),
        str(temporary),
        metadata={"schema": MATERIALIZATION_CONTRACT, "latent_layout": "64xT"},
    )
    with safe_open(str(temporary), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensors):
            raise RuntimeError("target safetensors key set changed after reopen")
        for key, expected in tensors.items():
            observed = handle.get_tensor(key)
            if observed.dtype != torch.float16 or tuple(observed.shape) != tuple(
                expected.shape
            ):
                raise RuntimeError(f"{key}: target latent changed after reopen")
    os.replace(temporary, path)


def _encode(
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    cleanup_foa: bool,
    model: Any | None = None,
) -> Path:
    split = str(rows[0]["split"])
    work_shard = int(rows[0]["work_shard"])
    seed = int.from_bytes(
        hashlib.blake2b(
            f"42:{split}:{work_shard}:editing-target-vae".encode("utf-8"),
            digest_size=8,
            person=b"edit-vae-v1",
        ).digest(),
        "big",
    )
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    owns_model = model is None
    if model is None:
        model = load_vae(device)
    result_by_id = {str(item["pair_id"]): item for item in results}
    tensors: dict[str, torch.Tensor] = {}
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = rows[start : start + int(batch_size)]
        batch_audio = []
        padded_lengths = []
        maximum = 0
        for row in batch_rows:
            result = result_by_id[str(row["pair_id"])]
            audio, rate = sf.read(
                result["target_foa_path"], dtype="float32", always_2d=True
            )
            if rate != MODEL_SAMPLE_RATE or audio.shape != (
                int(row["model_num_samples"]),
                4,
            ):
                raise RuntimeError("target FOA geometry changed before VAE encoding")
            recipe = _unpack(row["target_render_recipe_zlib"])
            padded = int(recipe["audio_execution"]["vae_padded_num_samples"])
            if padded < len(audio) or padded % 1024:
                raise RuntimeError("target VAE padding geometry is invalid")
            batch_audio.append(torch.from_numpy(audio.T.copy()))
            padded_lengths.append(padded)
            maximum = max(maximum, padded)
        value = torch.zeros((len(batch_rows), 4, maximum), dtype=torch.float32)
        for index, audio in enumerate(batch_audio):
            value[index, :, : audio.shape[-1]] = audio
        with torch.inference_mode():
            encoded = model.encode(value.to(device, non_blocking=True))
        if encoded.ndim != 3 or int(encoded.shape[1]) != 64:
            raise RuntimeError(f"unexpected target VAE shape: {tuple(encoded.shape)}")
        for index, row in enumerate(batch_rows):
            frames = int(row["latent_frames_valid"])
            target_id = str(row["target_sample_id"])
            latent = encoded[index, :, :frames].to(torch.float16).cpu().contiguous()
            if tuple(latent.shape) != (64, frames) or not torch.isfinite(latent).all():
                raise RuntimeError(f"{target_id}: invalid target latent")
            tensors[target_id] = latent
    if owns_model:
        del model
        torch.cuda.empty_cache()
    latent_path = Path(rows[0]["target_latent_path"])
    if any(Path(row["target_latent_path"]) != latent_path for row in rows):
        raise RuntimeError("pair work shard names multiple target latent shards")
    _atomic_safetensors(latent_path, tensors)
    latent_shard_sha = sha256_file(latent_path)

    materialized = []
    for row in rows:
        pair_id = str(row["pair_id"])
        target_id = str(row["target_sample_id"])
        result = result_by_id[pair_id]
        result_text = canonical_json(result)
        materialized.append(
            {
                "pair_ordinal": int(row["pair_ordinal"]),
                "pair_id": pair_id,
                "split": str(row["split"]),
                "work_shard": int(row["work_shard"]),
                "row_in_shard": int(row["row_in_shard"]),
                "source_sample_id": str(row["source_sample_id"]),
                "target_sample_id": target_id,
                "operation_family": str(row["operation_family"]),
                "operation": str(row["operation"]),
                "model_num_samples": int(row["model_num_samples"]),
                "latent_frames_valid": int(row["latent_frames_valid"]),
                "source_latent_ref": str(row["source_latent_ref"]),
                "source_latent_tensor_sha256": str(
                    row["source_latent_tensor_sha256"]
                ),
                "source_foa_sha256": str(row["source_foa_sha256"]),
                "source_parity_verified": bool(result["source_parity_verified"]),
                "target_foa_path": (
                    None if cleanup_foa else str(result["target_foa_path"])
                ),
                "target_foa_sha256": str(result["target_foa_sha256"]),
                "target_latent_ref": f"{latent_path}#{target_id}",
                "target_latent_tensor_sha256": _tensor_sha256(tensors[target_id]),
                "target_latent_shard_sha256": latent_shard_sha,
                "vae_encode_seed": seed,
                "target_render_result_json": result_text,
                "target_render_result_sha256": hashlib.sha256(
                    result_text.encode("utf-8")
                ).hexdigest(),
                "pair_record_sha256": str(row["pair_record_sha256"]),
            }
        )
    manifest_path = (
        output_root
        / "materialized/manifests"
        / split
        / f"materialized-{split}-{work_shard:05d}.parquet"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    pq.write_table(
        pa.Table.from_pylist(materialized, schema=MATERIALIZED_SCHEMA),
        temporary,
        compression="zstd",
    )
    if pq.read_metadata(temporary).num_rows != len(rows):
        raise RuntimeError("target materialized manifest row count changed")
    os.replace(temporary, manifest_path)
    if cleanup_foa:
        for result in results:
            Path(result["target_foa_path"]).unlink(missing_ok=True)
            for reference in result.get("stem_refs") or ():
                Path(reference["path"]).unlink(missing_ok=True)
    return manifest_path


def _validate_physical_gpu(index: int) -> torch.device:
    if int(index) not in ALLOWED_PHYSICAL_GPUS:
        raise ValueError("requested CUDA device index is outside the supported range")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {None, "", "0,1,2,3,4,5,6,7"}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES remapping is forbidden here because --gpu is a "
            "physical index"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() <= int(index):
        raise RuntimeError(f"physical CUDA device {index} is unavailable")
    return torch.device(f"cuda:{int(index)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--work-shard", type=int, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--source-parity", action="store_true")
    parser.add_argument("--retain-stems", action="store_true")
    parser.add_argument("--cleanup-foa", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="revalidate and atomically replace an already completed work shard",
    )
    args = parser.parse_args()
    if args.jobs <= 0 or args.batch_size <= 0 or args.work_shard < 0:
        raise ValueError("jobs/batch_size must be positive and work_shard non-negative")
    device = _validate_physical_gpu(args.gpu)
    index_path = args.pair_index.expanduser().resolve(strict=True)
    metadata, rows = _load_rows(index_path, args.work_shard)
    output_root = Path(metadata["target_root"]).resolve()
    done = (
        output_root
        / "materialized/work_done"
        / str(rows[0]["split"])
        / f"work-{args.work_shard:05d}.json"
    )
    if done.is_file() and not args.render_only and not args.replace:
        print(done.read_text(encoding="utf-8"), end="")
        return 0
    started = time.time()
    context = mp.get_context("spawn")
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=context
    ) as pool:
        futures = {
            pool.submit(
                _render_one,
                row,
                str(output_root),
                bool(args.retain_stems),
                bool(args.source_parity),
            ): row["pair_id"]
            for row in rows
        }
        for index, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            results.append(future.result())
            print(
                canonical_json(
                    {
                        "stage": "target_render",
                        "completed": index,
                        "total": len(rows),
                        "ok": sum(item.get("status") == "ok" for item in results),
                        "errors": sum(
                            item.get("status") != "ok" for item in results
                        ),
                        "elapsed_sec": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    by_id = {str(item["pair_id"]): item for item in results}
    results = [by_id[str(row["pair_id"])] for row in rows]
    failures = [item for item in results if item.get("status") != "ok"]
    if failures:
        quarantine = (
            output_root
            / "materialized/quarantine"
            / str(rows[0]["split"])
            / f"work-{args.work_shard:05d}.json"
        )
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        quarantine.write_text(
            json.dumps({"failures": failures}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"target render shard has {len(failures)} failures")
    if args.render_only:
        print(json.dumps({"ok": True, "rendered": len(results)}, indent=2))
        return 0
    manifest = _encode(
        rows,
        results,
        output_root=output_root,
        device=device,
        batch_size=args.batch_size,
        cleanup_foa=bool(args.cleanup_foa),
    )
    summary = {
        "schema": MATERIALIZATION_CONTRACT,
        "schema_version": 1,
        "status": "ok",
        "split": str(rows[0]["split"]),
        "work_shard": int(args.work_shard),
        "rows": len(rows),
        "physical_gpu": int(args.gpu),
        "source_parity_required": bool(args.source_parity),
        "source_parity_verified_rows": sum(
            bool(item["source_parity_verified"]) for item in results
        ),
        "cleanup_foa": bool(args.cleanup_foa),
        "retain_stems": bool(args.retain_stems),
        "materialized_manifest": str(manifest),
        "materialized_manifest_sha256": sha256_file(manifest),
        "elapsed_sec": round(time.time() - started, 2),
    }
    done.parent.mkdir(parents=True, exist_ok=True)
    temporary = done.with_name(done.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, done)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
