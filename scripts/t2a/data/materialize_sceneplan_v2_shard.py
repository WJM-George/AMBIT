#!/usr/bin/env python3
"""Render, QC, VAE-encode, and optionally clean one ScenePlan-v2 shard."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from scipy.signal import resample_poly


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

for value in (SCRIPT_DIR, REPO_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from render_tts_v2_pilot import normalize_dry, write_pcm24  # noqa: E402
from sceneplan_v2_common import (  # noqa: E402
    MAX_MODEL_SAMPLES,
    MODEL_SAMPLE_RATE,
    VAE_CHECKPOINT_SHA256,
    atomic_write_json,
    decode_complete_mono,
    deterministic_digest,
    load_parquet_source,
    normalized_transcript,
    require_dataset_not_frozen,
)
from sceneplan_v2_renderer import (  # noqa: E402
    active_rms,
    peak_safe_master_gain,
    render_complete_mono_source,
    true_peak,
)
from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict  # noqa: E402


VAE_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/autoencoders/"
    "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json"
)
VAE_CHECKPOINT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt")
TRUE_PEAK_CEILING = 10.0 ** (-1.0 / 20.0)
REVISION6_MAX_MODEL_SAMPLES = 648 * 1024


MATERIALIZED_SCHEMA = pa.schema(
    [
        ("sample_id", pa.string()),
        ("split", pa.string()),
        ("family", pa.string()),
        ("source_count", pa.int8()),
        ("model_num_samples", pa.int32()),
        ("latent_frames_valid", pa.int16()),
        ("vae_encode_seed", pa.uint64()),
        ("planned_record_sha256", pa.string()),
        ("materialized_record_json", pa.string()),
        ("materialized_record_sha256", pa.string()),
        ("render_result_json", pa.string()),
        ("foa_path", pa.string()),
        ("foa_sha256", pa.string()),
        ("latent_ref", pa.string()),
        ("latent_tensor_sha256", pa.string()),
        ("latent_shard_sha256", pa.string()),
        ("work_shard", pa.int32()),
        ("row_in_shard", pa.int16()),
    ]
)


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


def max_model_samples_for_row(row: dict[str, Any]) -> int:
    """Resolve the immutable render envelope from the dataset contract."""

    recipe_text = row.get("render_recipe_json")
    if recipe_text is None:
        # Direct revision-4 ScenePlan materialization has no detached recipe.
        return MAX_MODEL_SAMPLES
    revision = int(json.loads(str(recipe_text)).get("dataset_contract_revision", 5))
    if revision in {4, 5}:
        return MAX_MODEL_SAMPLES
    if revision == 6:
        return REVISION6_MAX_MODEL_SAMPLES
    raise RuntimeError(f"unsupported dataset contract revision: {revision}")


def load_complete_source(
    source: dict[str, Any],
    *,
    max_model_samples: int = MAX_MODEL_SAMPLES,
) -> tuple[np.ndarray, dict[str, Any]]:
    asset = source["asset_ref"]
    expected_model_samples = int(source["activity"][0]["dry_end_sample"])
    if source["kind"] == "speech" and all(
        asset.get(key) is not None
        for key in ("parquet_path", "row_group", "row_in_group")
    ):
        locator = {
            "parquet_path": asset["parquet_path"],
            "row_group": int(asset["row_group"]),
            "row_in_group": int(asset["row_in_group"]),
        }
        blob, metadata = load_parquet_source(locator)
        content_hash = sha256_bytes(blob)
        if content_hash != asset["identity_hash"]:
            raise RuntimeError("speech Parquet audio content SHA256 changed")
        mono, rate, native_frames = decode_complete_mono(
            blob, max_model_samples=max_model_samples
        )
        parquet_text = str(
            metadata.get("text_normalized")
            or metadata.get("text_no_preprocessing")
            or metadata.get("text_original")
            or metadata.get("text")
            or ""
        )
        if normalized_transcript(parquet_text) != normalized_transcript(
            source["speech"]["transcript"]
        ):
            raise RuntimeError("speech transcript/Parquet lineage mismatch")
        if rate != int(asset["native_sample_rate_hz"]) or native_frames != int(
            asset["native_num_samples"]
        ):
            raise RuntimeError("speech native audio geometry changed")
    else:
        # Revision 6 adds selectively materialized HiFiTTS-2 utterances.  They
        # are immutable mono files rather than embedded Parquet rows.  Speech
        # still keeps an explicit transcript-lineage checksum in the renderer
        # recipe; the model-facing ScenePlan remains locator-free.
        path = Path(asset["dry_audio_path"]).expanduser().resolve(strict=True)
        blob = path.read_bytes()
        content_hash = sha256_bytes(blob)
        if content_hash != asset["identity_hash"]:
            raise RuntimeError("file-backed source content SHA256 changed")
        audio, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
        if audio.shape[1] != 1:
            raise RuntimeError("file-backed renderer source is no longer mono")
        mono = audio[:, 0]
        native_frames = len(mono)
        if rate != int(asset["native_sample_rate_hz"]) or native_frames != int(
            asset["native_num_samples"]
        ):
            raise RuntimeError("file-backed source native audio geometry changed")
        if source["kind"] == "speech":
            expected_text_hash = str(
                asset.get("normalized_transcript_sha256") or ""
            )
            actual_text_hash = sha256_bytes(
                normalized_transcript(source["speech"]["transcript"]).encode("utf-8")
            )
            if not expected_text_hash or actual_text_hash != expected_text_hash:
                raise RuntimeError("file-backed speech transcript lineage mismatch")
        if rate != MODEL_SAMPLE_RATE:
            divisor = math.gcd(int(rate), MODEL_SAMPLE_RATE)
            mono = resample_poly(
                mono,
                MODEL_SAMPLE_RATE // divisor,
                int(rate) // divisor,
            ).astype(np.float32, copy=False)
        if len(mono) > int(max_model_samples):
            raise ValueError(
                "complete utterance exceeds model limit: "
                f"{len(mono)} > {int(max_model_samples)}"
            )
    if len(mono) != expected_model_samples:
        raise RuntimeError(
            f"complete resampled source length changed: {len(mono)} != {expected_model_samples}"
        )
    if not np.isfinite(mono).all() or not len(mono):
        raise RuntimeError("source is empty or non-finite")
    return mono, {
        "source_audio_sha256": content_hash,
        "native_sample_rate_hz": int(rate),
        "native_num_samples": int(native_frames),
        "model_num_samples": len(mono),
        "coverage_fraction": 1.0,
        "random_crop": False,
    }


def interval_mask(source: dict[str, Any], num_samples: int) -> np.ndarray:
    mask = np.zeros(num_samples, dtype=np.bool_)
    for interval in source["activity"]:
        mask[int(interval["model_onset_sample"]) : int(interval["model_offset_sample"])] = True
    return mask


def calibrate_speech_background(
    stems: list[np.ndarray],
    present: list[dict[str, Any]],
    source_qc: list[dict[str, Any]],
    *,
    num_samples: int,
    mixing_mode: str,
) -> dict[str, Any]:
    """Apply the frozen mixed-scene loudness contract.

    Overlapping speech/background scenes retain the revision-5 aggregate
    background calibration.  Revision 6 additionally permits explicitly
    sequential scenes: their activities must be disjoint and their already
    source-normalized stems are left untouched.  Keeping this decision in one
    pure, directly tested boundary prevents a future renderer change from
    silently treating a sequential scene as an overlap mix.
    """

    speech_indices = [
        index for index, source in enumerate(present) if source["kind"] == "speech"
    ]
    background_indices = [
        index for index, source in enumerate(present) if source["kind"] != "speech"
    ]
    if len(speech_indices) != 1 or not background_indices:
        raise RuntimeError(
            "speech/background calibration requires exactly one speech source "
            "and at least one background source"
        )
    speech_index = speech_indices[0]
    speech_mask = interval_mask(present[speech_index], num_samples)
    background_mask = np.logical_or.reduce(
        [interval_mask(present[index], num_samples) for index in background_indices]
    )
    overlap = speech_mask & background_mask
    overlap_samples = int(overlap.sum())
    if mixing_mode == "sequential_nonoverlap":
        if overlap_samples != 0:
            raise RuntimeError(
                "sequential speech/background activities unexpectedly overlap"
            )
        for index in range(len(source_qc)):
            source_qc[index]["calibrated_gain_correction_db"] = 0.0
            source_qc[index]["actual_gain_db"] = source_qc[index][
                "planned_gain_db"
            ]
        return {
            "mode": "nonoverlap_independent_rms_normalization",
            "target_speech_to_aggregate_background_db": None,
            "measured_speech_to_aggregate_background_db": None,
            "overlap_samples": 0,
            "background_source_count": len(background_indices),
            "aggregate_background_not_per_source": True,
        }
    if mixing_mode != "overlap_calibrated":
        raise RuntimeError(
            f"unsupported speech/background mixing mode: {mixing_mode!r}"
        )
    if overlap_samples < round(0.10 * MODEL_SAMPLE_RATE):
        raise RuntimeError(
            "overlap-calibrated speech/background support is shorter than 100 ms"
        )
    speech_rms = active_rms(stems[speech_index][0, overlap])
    background_sum = np.sum(
        [stems[index] for index in background_indices], axis=0
    )
    background_rms = active_rms(background_sum[0, overlap])
    planned_background_db = float(present[background_indices[0]]["gain_db"])
    planned_target_db = -planned_background_db - 5.0 * math.log10(
        len(background_indices)
    )
    target_ratio = 10.0 ** (planned_target_db / 20.0)
    correction = speech_rms / max(1e-12, target_ratio * background_rms)
    correction_db = 20.0 * math.log10(correction)
    for index in background_indices:
        stems[index] *= correction
        source_qc[index]["calibrated_gain_correction_db"] = correction_db
        source_qc[index]["actual_gain_db"] = (
            source_qc[index]["planned_gain_db"] + correction_db
        )
        present[index]["gain_db"] = source_qc[index]["actual_gain_db"]
    source_qc[speech_index]["calibrated_gain_correction_db"] = 0.0
    source_qc[speech_index]["actual_gain_db"] = source_qc[speech_index][
        "planned_gain_db"
    ]
    background_sum = np.sum(
        [stems[index] for index in background_indices], axis=0
    )
    measured_speech = active_rms(stems[speech_index][0, overlap])
    measured_background = active_rms(background_sum[0, overlap])
    measured_db = 20.0 * math.log10(measured_speech / measured_background)
    return {
        "mode": "overlap_calibrated",
        "target_speech_to_aggregate_background_db": planned_target_db,
        "measured_speech_to_aggregate_background_db": measured_db,
        "overlap_samples": overlap_samples,
        "background_source_count": len(background_indices),
        "aggregate_background_not_per_source": True,
    }


def write_render_result(path: Path, value: dict[str, Any]) -> None:
    atomic_write_json(path, value)


def remove_interrupted_render_temporaries(root: Path) -> int:
    """Remove only atomic-write remnants from an interrupted sample render."""

    if not root.is_dir():
        return 0
    removed = 0
    for pattern in (
        ".foa_WYZX_SN3D.*.flac",
        ".stem_source_*_WYZX_SN3D.*.flac",
        "render_result.json.tmp.*",
    ):
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)
                removed += 1
    return removed


def render_one(
    row: dict[str, Any],
    output_root: str,
    retain_stems: bool,
) -> dict[str, Any]:
    started = time.time()
    record = json.loads(row["record_json"])
    sample_id = str(row["sample_id"])
    split = str(row["split"])
    shard = int(row["work_shard"])
    root = Path(output_root) / split / f"work-{shard:05d}" / sample_id
    remove_interrupted_render_temporaries(root)
    result_path = root / "render_result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        foa = Path(result.get("foa_path") or "")
        if (
            result.get("status") == "ok"
            and result.get("planned_record_sha256") == row["record_sha256"]
            and foa.is_file()
        ):
            return result
    try:
        scene = record["scene_plan"]
        num_samples = int(scene["audio"]["model_num_samples"])
        max_model_samples = max_model_samples_for_row(row)
        if num_samples > max_model_samples:
            raise RuntimeError(
                "scene exceeds dataset contract render envelope: "
                f"{num_samples} > {max_model_samples}"
            )
        present = [source for source in scene["sources"] if source["present"]]
        stems = []
        source_qc = []
        target_source_rms = 10.0 ** (-24.0 / 20.0)
        for source in present:
            mono, lineage = load_complete_source(
                source, max_model_samples=max_model_samples
            )
            dry, dry_gain = normalize_dry(mono)
            interval = source["activity"][0]
            stem, renderer_qc = render_complete_mono_source(
                dry,
                sample_rate=MODEL_SAMPLE_RATE,
                scene_num_samples=num_samples,
                onset_sample=int(interval["model_onset_sample"]),
                room=scene["room"],
                keyframes=source["motion"]["keyframes"],
            )
            mask = interval_mask(source, num_samples)
            delayed_mask = np.zeros_like(mask)
            starts = np.flatnonzero(mask)
            if len(starts):
                delayed_start = min(num_samples, int(starts[0]) + 40)
                delayed_stop = min(num_samples, int(starts[-1]) + 1 + 40)
                delayed_mask[delayed_start:delayed_stop] = True
            raw_active_rms = active_rms(stem[0, delayed_mask])
            if raw_active_rms < 1e-9:
                raise RuntimeError("rendered source W channel is silent on active interval")
            normalization_gain = target_source_rms / raw_active_rms
            planned_gain_db = float(source["gain_db"])
            stem = stem * (normalization_gain * 10.0 ** (planned_gain_db / 20.0))
            stems.append(stem.astype(np.float32, copy=False))
            source_qc.append(
                {
                    "source_id": source["source_id"],
                    "slot": source["slot"],
                    "kind": source["kind"],
                    "asset_id": source["asset_ref"]["asset_id"],
                    "planned_gain_db": planned_gain_db,
                    "source_rms_normalization_gain": normalization_gain,
                    "dry_normalization_gain": dry_gain,
                    "content_lineage": lineage,
                    "renderer_qc": renderer_qc,
                }
            )

        speech_indices = [index for index, source in enumerate(present) if source["kind"] == "speech"]
        background_indices = [index for index, source in enumerate(present) if source["kind"] != "speech"]
        loudness_qc = None
        if speech_indices and background_indices:
            mixing_mode = str(
                (scene.get("mixing") or {}).get(
                    "speech_background_mode", "overlap_calibrated"
                )
            )
            loudness_qc = calibrate_speech_background(
                stems,
                present,
                source_qc,
                num_samples=num_samples,
                mixing_mode=mixing_mode,
            )
        else:
            for index in range(len(source_qc)):
                source_qc[index]["calibrated_gain_correction_db"] = 0.0
                source_qc[index]["actual_gain_db"] = source_qc[index]["planned_gain_db"]

        mix = np.sum(stems, axis=0, dtype=np.float32)
        master_gain, master_qc = peak_safe_master_gain(mix)
        mix = (mix * master_gain).astype(np.float32, copy=False)
        stems = [(stem * master_gain).astype(np.float32, copy=False) for stem in stems]
        measured_peak = true_peak(mix)
        if measured_peak > TRUE_PEAK_CEILING + 1e-5:
            raise RuntimeError("final in-memory true peak exceeds -1 dBFS")
        root.mkdir(parents=True, exist_ok=True)
        foa_path = root / "foa_WYZX_SN3D.flac"
        write_pcm24(foa_path, mix)
        stored, rate = sf.read(str(foa_path), dtype="float32", always_2d=True)
        if rate != MODEL_SAMPLE_RATE or stored.shape != (num_samples, 4):
            raise RuntimeError("stored FOA geometry changed")
        stored_peak = true_peak(stored.T)
        if stored_peak > TRUE_PEAK_CEILING + 2e-5:
            raise RuntimeError("stored true peak exceeds -1 dBFS")
        stem_paths = []
        if retain_stems:
            for source, stem in zip(present, stems):
                stem_path = root / f"stem_{source['source_id']}_WYZX_SN3D.flac"
                write_pcm24(stem_path, stem)
                stem_paths.append(
                    {
                        "source_id": source["source_id"],
                        "path": str(stem_path),
                        "sha256": sha256_file(stem_path),
                    }
                )
        result = {
            "schema": "stable_audio_tools.sceneplan_render_result",
            "schema_version": 2,
            "sample_id": sample_id,
            "status": "ok",
            "split": split,
            "family": row["family"],
            "source_count": int(row["source_count"]),
            "planned_record_sha256": row["record_sha256"],
            "foa_path": str(foa_path),
            "foa_sha256": sha256_file(foa_path),
            "num_samples": num_samples,
            "latent_frames_valid": int(scene["audio"]["latent_frames_valid"]),
            "source_qc": source_qc,
            "loudness_qc": loudness_qc,
            "master_gain": master_gain,
            "master_gain_qc": master_qc,
            "true_peak": measured_peak,
            "stored_true_peak": stored_peak,
            "stem_refs": stem_paths,
            "materialized_scene_plan": scene,
            "elapsed_sec": round(time.time() - started, 4),
        }
        write_render_result(result_path, result)
        return result
    except Exception as exc:  # noqa: BLE001
        return {
            "schema": "stable_audio_tools.sceneplan_render_result",
            "schema_version": 2,
            "sample_id": sample_id,
            "status": "error",
            "planned_record_sha256": row.get("record_sha256"),
            "source_assets": [
                source.get("asset_ref", {}).get("asset_id")
                for source in record.get("scene_plan", {}).get("sources", [])
                if source.get("present")
            ],
            "error": repr(exc),
            "elapsed_sec": round(time.time() - started, 4),
        }


def load_vae(device: torch.device):
    checkpoint_sha256 = sha256_file(VAE_CHECKPOINT)
    if checkpoint_sha256 != VAE_CHECKPOINT_SHA256:
        raise RuntimeError(
            "frozen VAE checkpoint SHA256 changed: "
            f"{checkpoint_sha256} != {VAE_CHECKPOINT_SHA256}"
        )
    config = load_config(VAE_CONFIG)
    model = create_model_from_config(config)
    copy_state_dict(model, load_ckpt_state_dict(str(VAE_CHECKPOINT)))
    return model.eval().requires_grad_(False).to(device)


def atomic_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    save_file(tensors, str(temporary), metadata={"schema": "sceneplan_v2_variable_latents"})
    with safe_open(str(temporary), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(tensors):
            raise RuntimeError("safetensors key set changed after reopen")
        for key, expected in tensors.items():
            actual = handle.get_tensor(key)
            if tuple(actual.shape) != tuple(expected.shape) or actual.dtype != torch.float16:
                raise RuntimeError(f"latent tensor geometry changed after reopen: {key}")
    os.replace(temporary, path)


def tensor_sha256(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def encode_results(
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    output_root: Path,
    device: torch.device,
    batch_size: int,
    cleanup_foa: bool,
    retain_stems: bool,
    model: Any | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    shard_number = int(rows[0]["work_shard"])
    split = str(rows[0]["split"])
    seed = int(deterministic_digest(20260814, "vae-encode", split, shard_number)[:16], 16)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    owns_model = model is None
    if model is None:
        model = load_vae(device)
    tensors: dict[str, torch.Tensor] = {}
    row_by_id = {str(row["sample_id"]): row for row in rows}
    pending = []

    def flush() -> None:
        if not pending:
            return
        audios = []
        valid = []
        max_padded = 0
        for result in pending:
            row = row_by_id[str(result["sample_id"])]
            record = json.loads(row["record_json"])
            audio_spec = record["scene_plan"]["audio"]
            audio, rate = sf.read(result["foa_path"], dtype="float32", always_2d=True)
            if rate != MODEL_SAMPLE_RATE or audio.shape != (
                int(audio_spec["model_num_samples"]),
                4,
            ):
                raise RuntimeError("FOA geometry changed before VAE encoding")
            padded = int(audio_spec["vae_padded_num_samples"])
            max_padded = max(max_padded, padded)
            audios.append(torch.from_numpy(audio.T.copy()))
            valid.append((str(result["sample_id"]), int(audio_spec["latent_frames_valid"]), padded))
        batch = torch.zeros((len(audios), 4, max_padded), dtype=torch.float32)
        for index, (audio, (_, _, padded)) in enumerate(zip(audios, valid)):
            batch[index, :, : audio.shape[-1]] = audio
            if padded < max_padded:
                # Extra batch padding is zero and never included in the stored
                # per-sample latent or loss-valid frame count.
                batch[index, :, padded:] = 0
        with torch.inference_mode():
            latent_batch = model.encode(batch.to(device, non_blocking=True))
        if latent_batch.ndim != 3 or latent_batch.shape[1] != 64:
            raise RuntimeError(f"unexpected VAE latent shape: {tuple(latent_batch.shape)}")
        for index, (sample_id, frames, _) in enumerate(valid):
            latent = latent_batch[index, :, :frames].to(torch.float16).cpu().contiguous()
            if latent.shape != (64, frames) or not torch.isfinite(latent).all():
                raise RuntimeError(f"invalid variable-length latent: {sample_id} {latent.shape}")
            tensors[sample_id] = latent
        pending.clear()

    for result in results:
        pending.append(result)
        if len(pending) >= batch_size:
            flush()
    flush()
    if owns_model:
        del model
    if owns_model and device.type == "cuda":
        torch.cuda.empty_cache()
    latent_path = output_root / "latents" / split / f"latents-{split}-{shard_number:05d}.safetensors"
    atomic_safetensors(latent_path, tensors)
    latent_shard_sha = sha256_file(latent_path)

    materialized = []
    result_by_id = {str(result["sample_id"]): result for result in results}
    for row in rows:
        sample_id = str(row["sample_id"])
        result = result_by_id[sample_id]
        record = json.loads(row["record_json"])
        record["scene_plan"] = result["materialized_scene_plan"]
        record["target"].update(
            {
                "materialization_state": "encoded",
                "foa_path": None if cleanup_foa else result["foa_path"],
                "foa_sha256": result["foa_sha256"],
                "latent_ref": f"{latent_path}#{sample_id}",
                "latent_sha256": tensor_sha256(tensors[sample_id]),
                "vae_encode_seed": seed,
            }
        )
        materialized_json = canonical_json(record)
        materialized.append(
            {
                "sample_id": sample_id,
                "split": row["split"],
                "family": row["family"],
                "source_count": int(row["source_count"]),
                "model_num_samples": int(row["model_num_samples"]),
                "latent_frames_valid": int(row["latent_frames_valid"]),
                "vae_encode_seed": seed,
                "planned_record_sha256": row["record_sha256"],
                "materialized_record_json": materialized_json,
                "materialized_record_sha256": sha256_bytes(materialized_json.encode("utf-8")),
                "render_result_json": canonical_json(result),
                "foa_path": None if cleanup_foa else result["foa_path"],
                "foa_sha256": result["foa_sha256"],
                "latent_ref": f"{latent_path}#{sample_id}",
                "latent_tensor_sha256": tensor_sha256(tensors[sample_id]),
                "latent_shard_sha256": latent_shard_sha,
                "work_shard": int(row["work_shard"]),
                "row_in_shard": int(row["row_in_shard"]),
            }
        )

    manifest_path = output_root / "manifests" / split / f"materialized-{split}-{shard_number:05d}.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(materialized, schema=MATERIALIZED_SCHEMA), temporary, compression="zstd")
    if pq.read_metadata(temporary).num_rows != len(materialized):
        raise RuntimeError("materialized manifest row count changed after reopen")
    os.replace(temporary, manifest_path)

    if cleanup_foa:
        # Each exact per-sample artifact is removed only after both latent and
        # manifest have been atomically reopened and checksummed.
        for result in results:
            sample_root = Path(result["foa_path"]).parent
            Path(result["foa_path"]).unlink(missing_ok=True)
            for reference in result.get("stem_refs") or []:
                Path(reference["path"]).unlink(missing_ok=True)
            (sample_root / "render_result.json").unlink(missing_ok=True)
            sample_root.rmdir()
        render_work_root = Path(results[0]["foa_path"]).parent.parent
        render_work_root.rmdir()
    return materialized, manifest_path


def main() -> int:
    require_dataset_not_frozen()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sceneplan-shard", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--retain-stems", action="store_true")
    parser.add_argument("--cleanup-foa", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    sceneplan_shard = args.sceneplan_shard.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=False)
    try:
        output_root.relative_to(os.environ.get("AMBIT_DATA_ROOT", "data"))
    except ValueError as error:
        raise ValueError(f"materialized outputs must be on SDB: {output_root}") from error
    output_root.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(sceneplan_shard).to_pylist()
    if not rows:
        raise RuntimeError("empty ScenePlan shard")
    shard_number = int(rows[0]["work_shard"])
    split = str(rows[0]["split"])
    done = output_root / "work_done" / split / f"work-{shard_number:05d}.json"
    if done.is_file() and not args.render_only:
        print(done.read_text(encoding="utf-8"), end="")
        return 0
    started = time.time()
    context = mp.get_context("spawn")
    results = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=context
    ) as pool:
        futures = {
            pool.submit(render_one, row, str(output_root / "renders"), args.retain_stems): row[
                "sample_id"
            ]
            for row in rows
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            if index % 50 == 0 or index == len(rows):
                print(
                    json.dumps(
                        {
                            "split": split,
                            "work_shard": shard_number,
                            "rendered": index,
                            "total": len(rows),
                            "ok": sum(result["status"] == "ok" for result in results),
                            "errors": sum(result["status"] != "ok" for result in results),
                            "elapsed_sec": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
    results.sort(key=lambda result: str(result["sample_id"]))
    failures = [result for result in results if result["status"] != "ok"]
    if failures:
        error_root = output_root / "quarantine" / split
        atomic_write_json(
            error_root / f"work-{shard_number:05d}.json",
            {"errors": failures, "error_count": len(failures)},
        )
        raise RuntimeError(f"render shard has {len(failures)} quarantines")
    if args.render_only:
        print(json.dumps({"ok": True, "rendered": len(results), "encoded": 0}, indent=2))
        return 0
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    materialized, manifest = encode_results(
        rows,
        results,
        output_root=output_root,
        device=device,
        batch_size=args.batch_size,
        cleanup_foa=args.cleanup_foa,
        retain_stems=args.retain_stems,
    )
    summary = {
        "schema": "stable_audio_tools.sceneplan_materialized_work_shard",
        "schema_version": 2,
        "split": split,
        "work_shard": shard_number,
        "rows": len(materialized),
        "sceneplan_shard": str(sceneplan_shard),
        "materialized_manifest": str(manifest),
        "cleanup_foa": args.cleanup_foa,
        "retain_stems": args.retain_stems,
        "max_model_num_samples": max(int(row["model_num_samples"]) for row in materialized),
        "max_latent_frames_valid": max(
            int(row["latent_frames_valid"]) for row in materialized
        ),
        "elapsed_sec": round(time.time() - started, 3),
    }
    (output_root / "quarantine" / split / f"work-{shard_number:05d}.json").unlink(
        missing_ok=True
    )
    atomic_write_json(done, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
