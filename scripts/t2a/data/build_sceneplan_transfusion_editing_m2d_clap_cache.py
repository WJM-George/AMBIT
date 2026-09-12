#!/usr/bin/env python3
"""Build one GPU shard of the Editing-AR M2D-CLAP source cache.

This is an offline label/feature job.  It decodes the hash-bound source latent
through the frozen FOA VAE for the M2D audio view.  It may read the stored
source plan only to compile a source semantic caption.  The output contains no
plan or caption text, and the query deliberately selects no target/new-plan
field.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from typing import Any, Iterable
import zlib

import numpy as np
import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    VAE_HOP_SAMPLES,
    compile_model_semantic_caption_v2,
    validate_model_sceneplan,
)
from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.data.sceneplan_transfusion_editing import (  # noqa: E402
    sha256_json,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_m2d_clap import (  # noqa: E402
    EDITING_M2D_AUDIO_PREPROCESS,
    EDITING_M2D_CACHE_SCHEMA,
    EDITING_M2D_CACHE_SCHEMA_VERSION,
    EDITING_M2D_CAPTION_TARGET,
    EDITING_M2D_VAE_CHECKPOINT_SHA256,
    EDITING_M2D_VAE_CONFIG_SHA256,
    editing_m2d_cache_implementation_sha256,
    validate_editing_m2d_temporal_pilot,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (  # noqa: E402
    EDITING_M2D_CLAP_CONTRACT,
    EDITING_M2D_CLAP_EMBED_DIM,
    canonicalize_editing_m2d_embedding,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_runtime import (  # noqa: E402
    decoded_foa_w_to_m2d_waveform,
    editing_m2d_numeric_runtime_fingerprint,
    FrozenEditingM2DCLAP,
    M2D_CLAP_SOURCE_AUDIO_VIEW,
    M2D_CLAP_TEMPORAL_POLICY,
)
from stable_audio_tools.models.utils import (  # noqa: E402
    copy_state_dict,
    load_ckpt_state_dict,
)


ALLOWED_PHYSICAL_GPUS = set(range(16))
DEFAULT_VAE_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/autoencoders/"
    "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json"
)
DEFAULT_VAE_CHECKPOINT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--expected-index-rows", type=int, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--vae-config", type=Path, default=DEFAULT_VAE_CONFIG)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--temporal-pilot", type=Path, required=True)
    parser.add_argument("--temporal-pilot-sha256", required=True)
    parser.add_argument("--vae-batch-size", type=int, default=8)
    parser.add_argument("--audio-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


def _editing_device(physical_gpu: int) -> torch.device:
    physical = int(physical_gpu)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if physical not in ALLOWED_PHYSICAL_GPUS or visible != str(physical):
        raise RuntimeError(
            "Editing M2D cache requires one remapped physical GPU 3--7; "
            f"expected CUDA_VISIBLE_DEVICES={physical}, got {visible!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Editing M2D cache requires exactly one visible GPU")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _source_index_summary(
    path: Path, *, expected_sha: str, expected_rows: int, split: str
) -> dict[str, str]:
    actual_sha = sha256_file(path)
    if actual_sha != str(expected_sha):
        raise RuntimeError("Editing M2D source-index SHA256 changed")
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        count, minimum, maximum = connection.execute(
            "SELECT COUNT(*),MIN(pair_ordinal),MAX(pair_ordinal) FROM pairs"
        ).fetchone()
        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(pairs)")
        }
    finally:
        connection.close()
    required_columns = {
        "pair_ordinal",
        "pair_id",
        "source_sample_id",
        "model_num_samples",
        "latent_frames_valid",
        "old_sceneplan_zlib",
        "old_sceneplan_sha256",
        "source_latent_path",
        "source_latent_key",
        "source_latent_tensor_sha256",
    }
    if not required_columns.issubset(columns):
        raise RuntimeError("Editing M2D source index lacks source-side columns")
    if (
        int(count) != int(expected_rows)
        or int(minimum) != 0
        or int(maximum) != int(expected_rows) - 1
        or metadata.get("split") != str(split)
        or metadata.get("editing_ar_input_contract")
        != "source_foa_latent_plus_raw_edit_request_v2"
        or metadata.get("editing_ar_old_sceneplan_input") != "false"
    ):
        raise RuntimeError("Editing M2D source index contract changed")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "rows": str(int(count)),
        "split": str(split),
        "state": str(metadata.get("state") or ""),
    }


def _selected_records(
    index: Path,
    *,
    max_rows: int,
    shard_index: int,
    num_shards: int,
) -> dict[
    str,
    list[tuple[int, str, str, int, int, str, str, str]],
]:
    # No new-plan, target-latent, target-waveform, or target-caption column is
    # selected here.  The source plan blob is offline caption-label provenance.
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    try:
        rows = connection.execute(
            """
            SELECT pair_ordinal,pair_id,source_sample_id,model_num_samples,
                   latent_frames_valid,source_latent_path,source_latent_key,
                   source_latent_tensor_sha256
            FROM pairs
            WHERE pair_ordinal < ? AND (pair_ordinal % ?) = ?
            ORDER BY source_latent_path,latent_frames_valid,pair_ordinal
            """,
            (int(max_rows), int(num_shards), int(shard_index)),
        )
        by_path: dict[
            str,
            list[tuple[int, str, str, int, int, str, str, str]],
        ] = defaultdict(list)
        for row in rows:
            record = (
                int(row[0]),
                str(row[1]),
                str(row[2]),
                int(row[3]),
                int(row[4]),
                str(row[5]),
                str(row[6]),
                str(row[7]),
            )
            by_path[record[5]].append(record)
    finally:
        connection.close()
    expected = len(range(int(shard_index), int(max_rows), int(num_shards)))
    if sum(len(value) for value in by_path.values()) != expected:
        raise RuntimeError("Editing M2D shard ordinal selection is incomplete")
    return dict(by_path)


def _source_captions(
    connection: sqlite3.Connection,
    records: list[tuple[int, str, str, int, int, str, str, str]],
) -> tuple[list[str], list[str]]:
    ordinals = [int(record[0]) for record in records]
    placeholders = ",".join("?" for _ in ordinals)
    values = {
        int(ordinal): (blob, str(plan_sha))
        for ordinal, blob, plan_sha in connection.execute(
            f"SELECT pair_ordinal,old_sceneplan_zlib,old_sceneplan_sha256 "
            f"FROM pairs WHERE pair_ordinal IN ({placeholders})",
            tuple(ordinals),
        )
    }
    captions = []
    caption_hashes = []
    for ordinal, pair_id, source_sample_id, *_ in records:
        if ordinal not in values:
            raise RuntimeError(f"{pair_id}: source caption provenance is absent")
        blob, expected_plan_sha = values[ordinal]
        try:
            plan = json.loads(zlib.decompress(blob))
        except (TypeError, ValueError, zlib.error) as error:
            raise RuntimeError(f"{pair_id}: source caption plan is invalid") from error
        validate_model_sceneplan(plan)
        if (
            sha256_json(plan) != expected_plan_sha
            or str(plan.get("sample_id")) != source_sample_id
        ):
            raise RuntimeError(f"{pair_id}: source caption provenance changed")
        text = str(compile_model_semantic_caption_v2(plan)["text"])
        if not text.strip():
            raise RuntimeError(f"{pair_id}: source semantic caption is empty")
        captions.append(text)
        caption_hashes.append(hashlib.sha256(text.encode("utf-8")).hexdigest())
    return captions, caption_hashes


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _embedding_blob(value: torch.Tensor) -> tuple[bytes, str, float]:
    if tuple(value.shape) != (EDITING_M2D_CLAP_EMBED_DIM,) or not bool(
        torch.isfinite(value).all()
    ):
        raise RuntimeError("M2D cache embedding shape/value changed")
    quantized = canonicalize_editing_m2d_embedding(value)
    blob = quantized.detach().cpu().contiguous().numpy().tobytes()
    return blob, hashlib.sha256(blob).hexdigest(), float(value.float().norm())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _resume_rows(
    connection: sqlite3.Connection,
    *,
    expected_metadata: dict[str, str],
    expected_records: dict[int, tuple[int, str, str, int, int, str, str, str]],
) -> set[int]:
    """Validate a durable partial cache before trusting any completed row."""

    check = connection.execute("PRAGMA quick_check").fetchone()
    if check is None or str(check[0]).lower() != "ok":
        raise RuntimeError("Editing M2D partial cache failed SQLite quick_check")
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    for key, expected in expected_metadata.items():
        if metadata.get(key) != str(expected):
            raise RuntimeError(
                f"Editing M2D partial cache metadata {key} changed"
            )
    if metadata.get("state") != "building":
        raise RuntimeError("Editing M2D partial cache is not resumable")

    completed: set[int] = set()
    expected_blob_bytes = EDITING_M2D_CLAP_EMBED_DIM * np.dtype(np.float16).itemsize
    cursor = connection.execute(
        """
        SELECT pair_ordinal,pair_id,source_sample_id,
               source_latent_tensor_sha256,source_caption_sha256,
               audio_embedding,text_embedding,audio_embedding_sha256,
               text_embedding_sha256,audio_norm_before_l2,
               text_norm_before_l2,record_sha256
        FROM features ORDER BY pair_ordinal
        """
    )
    for row in cursor:
        (
            ordinal,
            pair_id,
            source_sample_id,
            source_latent_sha,
            caption_sha,
            audio_blob,
            text_blob,
            audio_sha,
            text_sha,
            audio_norm,
            text_norm,
            record_sha,
        ) = row
        ordinal = int(ordinal)
        expected = expected_records.get(ordinal)
        if expected is None or (
            str(pair_id) != expected[1]
            or str(source_sample_id) != expected[2]
            or str(source_latent_sha) != expected[7]
        ):
            raise RuntimeError(
                f"Editing M2D partial row {ordinal} changed source identity"
            )
        if (
            not isinstance(audio_blob, bytes)
            or not isinstance(text_blob, bytes)
            or len(audio_blob) != expected_blob_bytes
            or len(text_blob) != expected_blob_bytes
            or hashlib.sha256(audio_blob).hexdigest() != str(audio_sha)
            or hashlib.sha256(text_blob).hexdigest() != str(text_sha)
            or not math.isfinite(float(audio_norm))
            or not math.isfinite(float(text_norm))
        ):
            raise RuntimeError(f"Editing M2D partial row {ordinal} is corrupt")
        for blob in (audio_blob, text_blob):
            embedding = np.frombuffer(blob, dtype=np.float16)
            norm = float(np.linalg.norm(embedding.astype(np.float32)))
            if not np.isfinite(embedding).all() or not 0.99 <= norm <= 1.01:
                raise RuntimeError(
                    f"Editing M2D partial row {ordinal} lost normalization"
                )
        record_value = {
            "pair_ordinal": ordinal,
            "pair_id": str(pair_id),
            "source_sample_id": str(source_sample_id),
            "source_latent_tensor_sha256": str(source_latent_sha),
            "source_caption_sha256": str(caption_sha),
            "audio_embedding_sha256": str(audio_sha),
            "text_embedding_sha256": str(text_sha),
        }
        if sha256_json(record_value) != str(record_sha):
            raise RuntimeError(
                f"Editing M2D partial row {ordinal} checksum changed"
            )
        completed.add(ordinal)
    if int(metadata.get("rows", -1)) != len(completed):
        raise RuntimeError("Editing M2D partial row counter changed")
    return completed


def _batched_audio_embeddings(
    m2d: FrozenEditingM2DCLAP,
    waveforms: list[torch.Tensor],
    *,
    batch_size: int,
) -> list[torch.Tensor]:
    """Batch only equal-length views; never introduce semantic padding."""

    if int(batch_size) <= 0 or not waveforms:
        raise ValueError("M2D audio batching requires non-empty waveforms")
    outputs: list[torch.Tensor | None] = [None for _ in waveforms]
    by_audio_samples: dict[int, list[int]] = defaultdict(list)
    for position, waveform in enumerate(waveforms):
        if waveform.ndim != 1 or int(waveform.shape[-1]) < 400:
            raise ValueError("M2D cache waveform must be mono [N>=400]")
        by_audio_samples[int(waveform.shape[-1])].append(position)
    for positions in by_audio_samples.values():
        for position_batch in _chunks(positions, int(batch_size)):
            waveform_batch = torch.stack(
                [waveforms[position] for position in position_batch]
            )
            encoded = m2d.encode_audio(waveform_batch).cpu()
            for position, embedding in zip(position_batch, encoded.unbind(0)):
                outputs[position] = embedding
    if any(value is None for value in outputs):
        raise RuntimeError("M2D audio batch lost a source row")
    return [value for value in outputs if value is not None]


def main() -> int:
    args = _parse_args()
    if (
        int(args.expected_index_rows) <= 0
        or int(args.num_shards) != 5
        or not 0 <= int(args.shard_index) < int(args.num_shards)
        or int(args.physical_gpu) != 3 + int(args.shard_index)
        or int(args.vae_batch_size) <= 0
        or int(args.audio_batch_size) <= 0
        or int(args.text_batch_size) <= 0
    ):
        raise ValueError("Editing M2D formal cache requires five GPU3--7 shards")
    max_rows = (
        int(args.expected_index_rows)
        if args.max_rows is None
        else int(args.max_rows)
    )
    if not 2 <= max_rows <= int(args.expected_index_rows):
        raise ValueError("Editing M2D max rows must be within [2,index rows]")
    device = _editing_device(args.physical_gpu)
    torch.set_float32_matmul_precision("high")
    index = args.index.expanduser().resolve(strict=True)
    index_summary = _source_index_summary(
        index,
        expected_sha=args.index_sha256,
        expected_rows=int(args.expected_index_rows),
        split=args.split,
    )
    vae_config = args.vae_config.expanduser().resolve(strict=True)
    vae_checkpoint = args.vae_checkpoint.expanduser().resolve(strict=True)
    if sha256_file(vae_config) != EDITING_M2D_VAE_CONFIG_SHA256:
        raise RuntimeError("frozen FOA VAE configuration SHA256 changed")
    if sha256_file(vae_checkpoint) != EDITING_M2D_VAE_CHECKPOINT_SHA256:
        raise RuntimeError("frozen FOA VAE checkpoint SHA256 changed")
    temporal_pilot = validate_editing_m2d_temporal_pilot(
        args.temporal_pilot,
        expected_sha256=str(args.temporal_pilot_sha256),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    marker_path = output.with_suffix(output.suffix + ".shard.json")
    partial = output.with_suffix(".partial.sqlite")
    if output.exists() or marker_path.exists():
        if not args.overwrite:
            raise FileExistsError(output)
        output.unlink(missing_ok=True)
        marker_path.unlink(missing_ok=True)
    if args.overwrite:
        partial.unlink(missing_ok=True)
        Path(str(partial) + "-wal").unlink(missing_ok=True)
        Path(str(partial) + "-shm").unlink(missing_ok=True)

    records_by_path = _selected_records(
        index,
        max_rows=max_rows,
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
    )
    vae = create_model_from_config(load_config(vae_config))
    copy_state_dict(vae, load_ckpt_state_dict(str(vae_checkpoint)))
    vae.eval().requires_grad_(False).to(device)
    m2d = FrozenEditingM2DCLAP(
        device=device, load_text_encoder=True
    ).eval().requires_grad_(False)

    builder_path = Path(__file__).resolve()
    static_metadata = {
        "schema": EDITING_M2D_CACHE_SCHEMA,
        "schema_version": EDITING_M2D_CACHE_SCHEMA_VERSION,
        "split": str(args.split),
        "full_selection_rows": str(max_rows),
        "source_index": str(index),
        "source_index_sha256": index_summary["sha256"],
        "source_index_rows": index_summary["rows"],
        "source_index_state": index_summary["state"],
        "dimension": str(EDITING_M2D_CLAP_EMBED_DIM),
        "dtype": "float16",
        "semantic_contract": EDITING_M2D_CLAP_CONTRACT,
        "audio_preprocess": EDITING_M2D_AUDIO_PREPROCESS,
        "source_audio_view": M2D_CLAP_SOURCE_AUDIO_VIEW,
        "temporal_policy": M2D_CLAP_TEMPORAL_POLICY,
        "temporal_pilot": temporal_pilot["path"],
        "temporal_pilot_sha256": temporal_pilot["sha256"],
        "caption_target": EDITING_M2D_CAPTION_TARGET,
        "old_sceneplan_model_input": "false",
        "caption_model_input": "false",
        "target_information_used": "false",
        "offline_source_plan_used_for_caption_label": "true",
        "vae_config": str(vae_config),
        "vae_config_sha256": EDITING_M2D_VAE_CONFIG_SHA256,
        "vae_checkpoint": str(vae_checkpoint),
        "vae_checkpoint_sha256": EDITING_M2D_VAE_CHECKPOINT_SHA256,
        "m2d_assets_json": json.dumps(
            m2d.asset_report, ensure_ascii=False, sort_keys=True
        ),
        "builder": str(builder_path),
        "builder_sha256": sha256_file(builder_path),
        "implementation_sha256_json": json.dumps(
            editing_m2d_cache_implementation_sha256(), sort_keys=True
        ),
        "numeric_runtime_fingerprint_json": json.dumps(
            editing_m2d_numeric_runtime_fingerprint(device), sort_keys=True
        ),
        "num_shards": str(int(args.num_shards)),
        "shard_index": str(int(args.shard_index)),
        "physical_gpu": str(int(args.physical_gpu)),
        "selection": "pair_ordinal_lt_N_then_modulo_5_v1",
        "vae_batch_size": str(int(args.vae_batch_size)),
        "audio_m2d_batch_size": str(int(args.audio_batch_size)),
        "text_batch_size": str(int(args.text_batch_size)),
    }
    expected_records = {
        int(record[0]): record
        for records in records_by_path.values()
        for record in records
    }
    resuming = partial.exists()
    destination = sqlite3.connect(partial)
    source = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    started = time.perf_counter()
    written = 0
    resumed_rows = 0
    completed = False
    try:
        if resuming:
            destination.execute("PRAGMA journal_mode=WAL")
            destination.execute("PRAGMA synchronous=NORMAL")
            completed_ordinals = _resume_rows(
                destination,
                expected_metadata=static_metadata,
                expected_records=expected_records,
            )
            written = len(completed_ordinals)
            resumed_rows = written
            records_by_path = {
                path: [
                    record
                    for record in records
                    if int(record[0]) not in completed_ordinals
                ]
                for path, records in records_by_path.items()
            }
            records_by_path = {
                path: records
                for path, records in records_by_path.items()
                if records
            }
            print(
                json.dumps(
                    {
                        "event": "resume",
                        "physical_gpu": int(args.physical_gpu),
                        "shard": int(args.shard_index),
                        "rows": written,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            destination.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE features (
                    pair_ordinal INTEGER PRIMARY KEY,
                    pair_id TEXT NOT NULL UNIQUE,
                    source_sample_id TEXT NOT NULL,
                    source_latent_tensor_sha256 TEXT NOT NULL,
                    source_caption_sha256 TEXT NOT NULL,
                    audio_embedding BLOB NOT NULL,
                    text_embedding BLOB NOT NULL,
                    audio_embedding_sha256 TEXT NOT NULL,
                    text_embedding_sha256 TEXT NOT NULL,
                    audio_norm_before_l2 REAL NOT NULL,
                    text_norm_before_l2 REAL NOT NULL,
                    record_sha256 TEXT NOT NULL
                );
                """
            )
            building_metadata = {
                **static_metadata,
                "state": "building",
                "rows": "0",
            }
            destination.executemany(
                "INSERT INTO metadata(key,value) VALUES (?,?)",
                building_metadata.items(),
            )
            destination.commit()
        for path_index, (latent_path, path_records) in enumerate(
            sorted(records_by_path.items())
        ):
            by_frames: dict[
                int,
                list[tuple[int, str, str, int, int, str, str, str]],
            ] = defaultdict(list)
            for record in path_records:
                by_frames[int(record[4])].append(record)
            with safe_open(latent_path, framework="pt", device="cpu") as tensors:
                for valid_frames, frame_records in sorted(by_frames.items()):
                    ordered_records = sorted(
                        frame_records, key=lambda record: (record[3], record[0])
                    )
                    for records in _chunks(
                        ordered_records, int(args.vae_batch_size)
                    ):
                        latent_rows = []
                        for (
                            _,
                            pair_id,
                            _,
                            model_num_samples,
                            frames,
                            _,
                            latent_key,
                            latent_sha,
                        ) in records:
                            if (
                                frames != valid_frames
                                or math.ceil(model_num_samples / VAE_HOP_SAMPLES)
                                != frames
                            ):
                                raise RuntimeError(f"{pair_id}: source time geometry changed")
                            if latent_key not in tensors.keys():
                                raise RuntimeError(f"{pair_id}: source latent key is absent")
                            latent = tensors.get_tensor(latent_key).clone()
                            if (
                                latent.dtype != torch.float16
                                or tuple(latent.shape) != (64, valid_frames)
                                or not bool(torch.isfinite(latent).all())
                                or _tensor_sha256(latent) != latent_sha
                            ):
                                raise RuntimeError(f"{pair_id}: source latent changed")
                            latent_rows.append(latent)

                        latent_batch = torch.stack(latent_rows).to(
                            device=device, dtype=torch.float32
                        )
                        with torch.inference_mode():
                            decoded = vae.decode(latent_batch).float()
                        expected_decoded_shape = (
                            len(records),
                            4,
                            int(valid_frames) * VAE_HOP_SAMPLES,
                        )
                        if tuple(decoded.shape) != expected_decoded_shape or not bool(
                            torch.isfinite(decoded).all()
                        ):
                            raise RuntimeError(
                                "frozen FOA VAE decoder output geometry/value changed"
                            )

                        captions, caption_hashes = _source_captions(
                            source, records
                        )
                        text_embeddings = []
                        for caption_batch in _chunks(
                            captions, int(args.text_batch_size)
                        ):
                            text_embeddings.extend(
                                m2d.encode_text(caption_batch).cpu().unbind(0)
                            )
                        waveforms = [
                            decoded_foa_w_to_m2d_waveform(
                                audio, valid_samples=int(record[3])
                            )
                            for audio, record in zip(decoded, records)
                        ]
                        audio_embeddings = _batched_audio_embeddings(
                            m2d,
                            waveforms,
                            batch_size=int(args.audio_batch_size),
                        )
                        rows = []
                        for record, audio_embedding, caption_hash, text_embedding in zip(
                            records,
                            audio_embeddings,
                            caption_hashes,
                            text_embeddings,
                        ):
                            (
                                ordinal,
                                pair_id,
                                source_sample_id,
                                model_num_samples,
                                _,
                                _,
                                _,
                                source_latent_sha,
                            ) = record
                            audio_blob, audio_sha, audio_norm = _embedding_blob(
                                audio_embedding
                            )
                            text_blob, text_sha, text_norm = _embedding_blob(
                                text_embedding
                            )
                            record_value = {
                                "pair_ordinal": ordinal,
                                "pair_id": pair_id,
                                "source_sample_id": source_sample_id,
                                "source_latent_tensor_sha256": source_latent_sha,
                                "source_caption_sha256": caption_hash,
                                "audio_embedding_sha256": audio_sha,
                                "text_embedding_sha256": text_sha,
                            }
                            rows.append(
                                (
                                    ordinal,
                                    pair_id,
                                    source_sample_id,
                                    source_latent_sha,
                                    caption_hash,
                                    audio_blob,
                                    text_blob,
                                    audio_sha,
                                    text_sha,
                                    audio_norm,
                                    text_norm,
                                    sha256_json(record_value),
                                )
                            )
                        destination.executemany(
                            "INSERT INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            rows,
                        )
                        written += len(rows)
                        destination.execute(
                            "UPDATE metadata SET value=? WHERE key='rows'",
                            (str(written),),
                        )
                        destination.commit()
            if path_index % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "physical_gpu": int(args.physical_gpu),
                            "shard": int(args.shard_index),
                            "source_latent_shards": path_index + 1,
                            "source_latent_shards_total": len(records_by_path),
                            "rows": written,
                            "elapsed_sec": round(time.perf_counter() - started, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        expected_shard_rows = len(
            range(int(args.shard_index), max_rows, int(args.num_shards))
        )
        if written != expected_shard_rows:
            raise RuntimeError(
                f"Editing M2D shard wrote {written}, expected {expected_shard_rows}"
            )
        elapsed = time.perf_counter() - started
        metadata = {
            **static_metadata,
            "state": "shard_complete",
            "rows": str(written),
            "resumed_rows": str(resumed_rows),
            "elapsed_sec": f"{elapsed:.6f}",
        }
        destination.executemany(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
            metadata.items(),
        )
        destination.commit()
        destination.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        destination.execute("PRAGMA journal_mode=DELETE")
        destination.execute("VACUUM")
        destination.commit()
        completed = True
    finally:
        source.close()
        destination.close()
    if not completed:
        raise RuntimeError("Editing M2D cache shard did not complete")
    os.replace(partial, output)
    shard_record = {
        "schema": "sceneplan_transfusion_editing_m2d_clap_cache_shard",
        "schema_version": int(EDITING_M2D_CACHE_SCHEMA_VERSION),
        "state": "shard_complete",
        "path": str(output),
        "sha256": sha256_file(output),
        "rows": written,
        "full_selection_rows": max_rows,
        "split": str(args.split),
        "source_index": str(index),
        "source_index_sha256": index_summary["sha256"],
        "shard_index": int(args.shard_index),
        "num_shards": int(args.num_shards),
        "physical_gpu": int(args.physical_gpu),
    }
    _atomic_json(marker_path, shard_record)
    print(json.dumps(shard_record, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
