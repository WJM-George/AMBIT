#!/usr/bin/env python3
"""Build the canonical window-level pinned CLAP cache for P11 Understanding."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import (  # noqa: E402
    copy_state_dict,
    load_ckpt_state_dict,
)


DEFAULT_VAE_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/autoencoders/"
    "stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json"
)
DEFAULT_VAE_CHECKPOINT = Path(
    "/mnt/sdc/ckpts/compareVAE_ckpt/unwrapped_wdmix_1350000.ckpt"
)
DEFAULT_CLAP = Path("/mnt/sdc/ckpts/pretrained/laion/clap-htsat-fused")
CLAP_REPO = "laion/clap-htsat-fused"
CLAP_REVISION = "365dea6ef167def6676140ed93bbc43f84dabb28"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_ordinals(
    manifest: Path,
    *,
    ordinal_inventory: Path | None,
    max_rows: int | None,
    shard_index: int,
    num_shards: int,
) -> tuple[list[int], dict[str, str] | None]:
    if ordinal_inventory is not None:
        connection = sqlite3.connect(
            f"file:{ordinal_inventory}?mode=ro&immutable=1", uri=True
        )
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            required = {
                "schema": "stable_audio_tools.p11_cache_ordinal_inventory",
                "schema_version": "2",
                "selection_contract": (
                    "sorted_distinct_source_ordinal_then_strided_shard_v2"
                ),
                "source_manifest": str(manifest),
            }
            for key, expected in required.items():
                if metadata.get(key) != expected:
                    raise RuntimeError(
                        f"ordinal inventory {key}={metadata.get(key)!r}, "
                        f"expected {expected!r}"
                    )
            total = int(metadata.get("rows", -1))
            actual = int(
                connection.execute("SELECT COUNT(*) FROM ordinals").fetchone()[0]
            )
            if total <= 0 or actual != total:
                raise RuntimeError("ordinal inventory row metadata is stale")
            limit = total if max_rows is None else min(int(max_rows), total)
            values = [
                int(row[0])
                for row in connection.execute(
                    "SELECT source_ordinal FROM ordinals "
                    "WHERE position < ? AND (position % ?) = ? "
                    "ORDER BY position",
                    (limit, int(num_shards), int(shard_index)),
                )
            ]
        finally:
            connection.close()
        if not values:
            raise RuntimeError("P11 ordinal inventory selected no base ordinals")
        return values, metadata

    connection = sqlite3.connect(
        f"file:{manifest}?mode=ro&immutable=1", uri=True
    )
    try:
        query = (
            "SELECT DISTINCT source_ordinal FROM rows "
            "ORDER BY source_ordinal"
        )
        values = [int(row[0]) for row in connection.execute(query)]
    finally:
        connection.close()
    if max_rows is not None:
        values = values[: int(max_rows)]
    values = values[int(shard_index) :: int(num_shards)]
    if not values:
        raise RuntimeError("P11 manifest selected no base ordinals")
    return values, None


def _chunks(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), int(size)):
        yield values[start : start + int(size)]


def _mono_for_clap(audio: torch.Tensor, *, valid_samples: int) -> np.ndarray:
    """Use FOA W, peak-normalize, and deterministically resample to 48 kHz."""

    mono = audio[0, : int(valid_samples)].float()
    if not bool(torch.isfinite(mono).all()):
        raise RuntimeError("VAE decoded non-finite FOA")
    peak = mono.abs().amax()
    if float(peak) > 1.0e-8:
        mono = mono / peak * (10.0 ** (-1.0 / 20.0))
    # Preserve P10's complete 15.05-second envelope.  CLAP's per-example
    # input limit is handled by deterministic windows after resampling; an
    # eager ten-second crop would erase valid late-scene evidence.
    mono = F.interpolate(
        mono[None, None],
        size=max(1, round(mono.numel() * 48_000 / 44_100)),
        mode="linear",
        align_corners=False,
    )[0, 0]
    return mono.clamp(-1.0, 1.0).cpu().numpy()


def _semantic_windows(
    waveform: np.ndarray,
    *,
    window_samples: int,
    hop_samples: int,
) -> list[np.ndarray]:
    if waveform.ndim != 1 or waveform.size <= 0:
        raise ValueError("P11 semantic waveform must be one non-empty channel")
    if waveform.size <= int(window_samples):
        starts = [0]
    else:
        last_start = int(waveform.size) - int(window_samples)
        starts = list(range(0, last_start + 1, int(hop_samples)))
        if starts[-1] != last_start:
            # Always include the tail.  Replace a near-duplicate final window
            # but retain the start window so both ends remain observable.
            if len(starts) > 1 and last_start - starts[-1] < int(hop_samples) // 2:
                starts[-1] = last_start
            else:
                starts.append(last_start)
    windows = [waveform[start : start + int(window_samples)] for start in starts]
    windows = [value for value in windows if value.size > 0]
    if not windows:
        raise RuntimeError("P11 semantic windowing produced no window")
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ordinal-inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae-config", type=Path, default=DEFAULT_VAE_CONFIG)
    parser.add_argument("--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT)
    parser.add_argument("--clap-model", type=Path, default=DEFAULT_CLAP)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--representation",
        choices=("global", "windowed"),
        default="windowed",
        help="Windowed is canonical for full 15-second evidence; global is an ablation.",
    )
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--hop-sec", type=float, default=5.0)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("semantic shard must satisfy 0 <= shard-index < num-shards")
    if not 0.25 <= args.window_sec <= 10.0:
        raise ValueError("--window-sec must be within [0.25,10]")
    if not 0.25 <= args.hop_sec <= args.window_sec:
        raise ValueError("--hop-sec must be within [0.25,window-sec]")
    window_samples = int(round(args.window_sec * 48_000))
    hop_samples = int(round(args.hop_sec * 48_000))

    index = args.index.expanduser().resolve(strict=True)
    manifest = args.manifest.expanduser().resolve(strict=True)
    ordinal_inventory = (
        None
        if args.ordinal_inventory is None
        else args.ordinal_inventory.expanduser().resolve(strict=True)
    )
    vae_config_path = args.vae_config.expanduser().resolve(strict=True)
    vae_checkpoint = args.vae_checkpoint.expanduser().resolve(strict=True)
    clap_path = args.clap_model.expanduser().resolve(strict=True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("P11 CLAP cache construction requires CUDA")
    ordinals, inventory_metadata = _selected_ordinals(
        manifest,
        ordinal_inventory=ordinal_inventory,
        max_rows=args.max_rows,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    from transformers import ClapFeatureExtractor, ClapModel

    config = json.loads(vae_config_path.read_text(encoding="utf-8"))
    vae = create_model_from_config(config)
    copy_state_dict(vae, load_ckpt_state_dict(str(vae_checkpoint)))
    vae.eval().requires_grad_(False).to(device)
    clap = ClapModel.from_pretrained(
        clap_path, local_files_only=True
    ).eval().requires_grad_(False).to(device)
    extractor = ClapFeatureExtractor.from_pretrained(
        clap_path, local_files_only=True
    )

    source = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    records: list[tuple[int, str, int, int, str]] = []
    # SQLite defaults to 999 bound variables, so query the requested ordinals
    # in bounded chunks and group them by immutable safetensors shard.
    for ordinal_chunk in _chunks(ordinals, 900):
        placeholders = ",".join("?" for _ in ordinal_chunk)
        records.extend(
            (
                int(ordinal),
                str(key),
                int(valid_frames),
                int(model_num_samples),
                str(path),
            )
            for ordinal, key, valid_frames, model_num_samples, path in source.execute(
                f"""
                SELECT samples.ordinal, samples.latent_key,
                       samples.latent_frames_valid, samples.model_num_samples,
                       latent_shards.path
                FROM samples
                JOIN latent_shards ON latent_shards.id = samples.latent_shard_id
                WHERE samples.ordinal IN ({placeholders})
                """,
                tuple(ordinal_chunk),
            )
        )
    if len(records) != len(ordinals):
        raise RuntimeError(
            f"P11 index resolved {len(records)} of {len(ordinals)} semantic rows"
        )
    by_shard: dict[str, list[tuple[int, str, int, int]]] = defaultdict(list)
    for ordinal, key, frames, samples, path in records:
        by_shard[path].append((ordinal, key, frames, samples))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".sqlite", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    destination = sqlite3.connect(temporary)
    started = time.perf_counter()
    written = 0
    completed = False
    try:
        destination.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE features (
                ordinal INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                windows INTEGER NOT NULL,
                mean_l2_norm REAL NOT NULL
            );
            """
        )
        for shard_index, (shard_path, shard_records) in enumerate(
            sorted(by_shard.items())
        ):
            with safe_open(shard_path, framework="pt", device="cpu") as tensors:
                # Preserve exact VAE decoding for variable-length latents.  A
                # padded batch would make decoder normalization depend on the
                # unrelated examples that happen to share that cache batch.
                records_by_frames: dict[
                    int, list[tuple[int, str, int, int]]
                ] = defaultdict(list)
                for record in shard_records:
                    records_by_frames[int(record[2])].append(record)
                for latent_frames, same_length_records in sorted(
                    records_by_frames.items()
                ):
                    for batch_records in _chunks(
                        sorted(same_length_records), args.batch_size
                    ):
                        latent_rows = []
                        for _, key, recorded_frames, _ in batch_records:
                            latent = tensors.get_tensor(key)
                            expected_shape = (64, int(recorded_frames))
                            if tuple(latent.shape) != expected_shape:
                                raise RuntimeError(
                                    f"latent {key} shape {tuple(latent.shape)} != "
                                    f"index shape {expected_shape}"
                                )
                            if not bool(torch.isfinite(latent).all()):
                                raise RuntimeError(f"latent {key} is non-finite")
                            latent_rows.append(latent)
                        latents = torch.stack(latent_rows).to(
                            device=device, dtype=torch.float32
                        )
                        with torch.inference_mode():
                            decoded = vae.decode(latents).float()
                        waveforms = []
                        window_counts = []
                        for audio, (_, _, _, model_num_samples) in zip(
                            decoded, batch_records
                        ):
                            if int(model_num_samples) > int(audio.shape[-1]):
                                raise RuntimeError(
                                    f"decoded {latent_frames} latent frames to "
                                    f"{audio.shape[-1]} samples, fewer than required "
                                    f"{model_num_samples}"
                                )
                            mono = _mono_for_clap(
                                audio, valid_samples=model_num_samples
                            )
                            windows = (
                                [mono]
                                if args.representation == "global"
                                else _semantic_windows(
                                    mono,
                                    window_samples=window_samples,
                                    hop_samples=hop_samples,
                                )
                            )
                            waveforms.extend(windows)
                            window_counts.append(len(windows))
                        inputs = extractor(
                            waveforms,
                            sampling_rate=48_000,
                            return_tensors="pt",
                        )
                        with torch.inference_mode():
                            clap_output = clap.get_audio_features(
                                **{
                                    key: value.to(device)
                                    for key, value in inputs.items()
                                }
                            )
                            embeddings = clap_output.pooler_output.float().cpu()
                        if embeddings.shape != (sum(window_counts), 512):
                            raise RuntimeError(
                                f"CLAP returned stale shape {tuple(embeddings.shape)}"
                            )
                        if not bool(torch.isfinite(embeddings).all()):
                            raise RuntimeError("CLAP returned non-finite embeddings")
                        rows = []
                        cursor = 0
                        for windows, (ordinal, _, _, _) in zip(
                            window_counts, batch_records
                        ):
                            embedding = embeddings[cursor : cursor + windows]
                            cursor += windows
                            norm = float(embedding.norm(dim=-1).mean())
                            rows.append(
                                (
                                    ordinal,
                                    embedding.numpy().astype(np.float16).tobytes(),
                                    windows,
                                    norm,
                                )
                            )
                        destination.executemany(
                            "INSERT INTO features VALUES (?,?,?,?)", rows
                        )
                        written += len(rows)
            if shard_index % 50 == 0:
                destination.commit()
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "shards": shard_index + 1,
                            "shards_total": len(by_shard),
                            "rows": written,
                            "elapsed_sec": time.perf_counter() - started,
                        }
                    ),
                    flush=True,
                )
        if written != len(ordinals):
            raise RuntimeError(
                f"semantic cache wrote {written} of {len(ordinals)} rows"
            )
        elapsed = time.perf_counter() - started
        metadata = {
            "schema": "stable_audio_tools.p11_semantic_cache",
            "schema_version": "2",
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": _sha256(Path(__file__).resolve()),
            "source_index": str(index),
            "source_manifest": str(manifest),
            "rows": str(written),
            "dimension": "512",
            "dtype": "float16",
            "encoder_repo": CLAP_REPO,
            "encoder_revision": CLAP_REVISION,
            "encoder_model_sha256": _sha256(clap_path / "model.safetensors"),
            "vae_config": str(vae_config_path),
            "vae_checkpoint": str(vae_checkpoint),
            "vae_checkpoint_sha256": _sha256(vae_checkpoint),
            "foa_channel": "W",
            "normalization": "per_clip_peak_minus_1db",
            "sample_rate": "48000",
            "representation": (
                "global_clap_pooler_output"
                if args.representation == "global"
                else "windowed_clap_pooler_output"
            ),
            "window_sec": (
                "10.000000"
                if args.representation == "global"
                else f"{args.window_sec:.6f}"
            ),
            "hop_sec": (
                "10.000000"
                if args.representation == "global"
                else f"{args.hop_sec:.6f}"
            ),
            "shard_index": str(args.shard_index),
            "num_shards": str(args.num_shards),
            "elapsed_sec": f"{elapsed:.6f}",
        }
        if ordinal_inventory is not None:
            assert inventory_metadata is not None
            metadata.update(
                {
                    "ordinal_inventory": str(ordinal_inventory),
                    "ordinal_inventory_sha256": _sha256(ordinal_inventory),
                    "ordinal_selection_contract": inventory_metadata[
                        "selection_contract"
                    ],
                    "source_manifest_sha256": inventory_metadata[
                        "source_manifest_sha256"
                    ],
                }
            )
        destination.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", metadata.items()
        )
        destination.commit()
        destination.execute("VACUUM")
        destination.commit()
        completed = True
    finally:
        destination.close()
        source.close()
        if not completed:
            temporary.unlink(missing_ok=True)
    if output.exists():
        output.unlink()
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "rows": written,
                "encoder_revision": CLAP_REVISION,
                "elapsed_sec": time.perf_counter() - started,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
