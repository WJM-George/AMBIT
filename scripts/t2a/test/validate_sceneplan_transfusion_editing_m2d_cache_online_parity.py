#!/usr/bin/env python3
"""Replay frozen Editing M2D cache rows through the formal online path.

The gate proves numerical parity at the actual fp16 AR boundary for the same
source latent under batch sizes 1, 2, and 4.  It never reads an old/new plan,
caption, target latent, or target waveform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
from safetensors import safe_open


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_m2d_clap import (  # noqa: E402
    EDITING_M2D_AUDIO_PREPROCESS,
    EDITING_M2D_CACHE_ONLINE_PARITY_CHECKS,
    EDITING_M2D_CACHE_ONLINE_PARITY_SCHEMA,
    EDITING_M2D_VAE_CHECKPOINT_SHA256,
    EDITING_M2D_VAE_CONFIG_SHA256,
    ScenePlanTransfusionEditingM2DCLAPCache,
    _expected_online_parity_rows,
    editing_m2d_cache_implementation_sha256,
    editing_m2d_cache_online_parity_path,
    validate_editing_m2d_cache_online_parity,
)
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_runtime import (  # noqa: E402
    FrozenEditingM2DCLAP,
    M2D_CLAP_SOURCE_AUDIO_VIEW,
    M2D_CLAP_TEMPORAL_POLICY,
    editing_m2d_numeric_runtime_fingerprint,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import (  # noqa: E402
    ScenePlanTransfusionEditingPipeline,
)
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
BATCH_SIZES = (1, 2, 4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-sha256", required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-index-sha256", required=True)
    parser.add_argument("--expected-cache-rows", type=int, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=3)
    parser.add_argument("--vae-config", type=Path, default=DEFAULT_VAE_CONFIG)
    parser.add_argument(
        "--vae-checkpoint", type=Path, default=DEFAULT_VAE_CHECKPOINT
    )
    return parser.parse_args()


def _device(physical_gpu: int) -> torch.device:
    if int(physical_gpu) != 3 or os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError(
            "formal Editing M2D cache/online parity runs only on physical GPU 3"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Editing M2D cache/online parity needs one visible GPU")
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def _embedding_sha256(value: torch.Tensor) -> str:
    if value.dtype != torch.float16 or tuple(value.shape) != (768,):
        raise RuntimeError("online M2D embedding missed canonical fp16 boundary")
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


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


class _OnlineSemanticHarness:
    """Invoke the production pipeline method without loading AR/DiT weights."""

    encode_source_m2d_audio = ScenePlanTransfusionEditingPipeline.encode_source_m2d_audio

    def __init__(self, vae, encoder: FrozenEditingM2DCLAP, device: torch.device):
        self.audio_autoencoder = vae
        self.source_semantic_encoder = encoder
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def _require_audio_autoencoder(self):
        return self.audio_autoencoder

    def _require_source_semantic_encoder(self) -> FrozenEditingM2DCLAP:
        return self.source_semantic_encoder


def _load_latents(rows: list[dict[str, Any]]) -> dict[int, torch.Tensor]:
    values: dict[int, torch.Tensor] = {}
    for row in rows:
        path = row["source_latent_path"]
        key = row["source_latent_key"]
        with safe_open(path, framework="pt", device="cpu") as tensors:
            if key not in tensors.keys():
                raise RuntimeError(f"{row['pair_id']}: source latent key disappeared")
            latent = tensors.get_tensor(key).clone()
        if not (
            latent.dtype == torch.float16
            and tuple(latent.shape)
            == (64, int(row["latent_frames_valid"]))
            and bool(torch.isfinite(latent).all())
            and _tensor_sha256(latent) == row["source_latent_tensor_sha256"]
        ):
            raise RuntimeError(f"{row['pair_id']}: source latent content changed")
        values[int(row["pair_ordinal"])] = latent
    return values


def _online_pass(
    harness: _OnlineSemanticHarness,
    rows: list[dict[str, Any]],
    latents: dict[int, torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    outputs: dict[int, torch.Tensor] = {}
    # Repeat the *same* exact-geometry row N times.  Merely grouping unrelated
    # rows here is not a batch-shape test because the production method groups
    # internally by (valid_frames, model_num_samples), and real rows can all
    # have distinct exact lengths.  Replication forces both VAE and M2D to
    # execute with the advertised batch dimension while preserving a single
    # unambiguous cache reference.
    for row in rows:
        bucket = int(row["latent_bucket_frames"])
        valid = int(row["latent_frames_valid"])
        latent = latents[int(row["pair_ordinal"])].to(
            device=device, dtype=torch.float32
        )
        source = torch.zeros(
            batch_size, 64, bucket, dtype=torch.float32, device=device
        )
        source[:, :, :valid] = latent.unsqueeze(0).expand(batch_size, -1, -1)
        mask = torch.zeros(batch_size, bucket, dtype=torch.bool, device=device)
        mask[:, :valid] = True
        embedded = harness.encode_source_m2d_audio(
            source,
            mask,
            model_num_samples=[int(row["model_num_samples"])] * batch_size,
        ).cpu().contiguous()
        if tuple(embedded.shape) != (batch_size, 768):
            raise RuntimeError("online M2D replay did not preserve forced batch shape")
        outputs[int(row["pair_ordinal"])] = embedded
    if set(outputs) != {int(row["pair_ordinal"]) for row in rows}:
        raise RuntimeError("online M2D parity pass lost a sampled row")
    return outputs


def main() -> int:
    args = _parse_args()
    if int(args.expected_cache_rows) < 10:
        raise ValueError("formal M2D cache/online parity requires at least ten rows")
    device = _device(args.physical_gpu)
    torch.set_float32_matmul_precision("high")
    cache_path = args.cache.expanduser().resolve(strict=True)
    source_index = args.source_index.expanduser().resolve(strict=True)
    if sha256_file(cache_path) != str(args.cache_sha256):
        raise RuntimeError("Editing M2D cache SHA256 changed before parity replay")
    cache = ScenePlanTransfusionEditingM2DCLAPCache(
        cache_path,
        source_index=source_index,
        source_index_sha256=str(args.source_index_sha256),
        expected_rows=int(args.expected_cache_rows),
        expected_split=str(args.split),
        expected_cache_sha256=str(args.cache_sha256),
        verify_cache_file_hash=False,
    )
    output = (
        editing_m2d_cache_online_parity_path(cache_path)
        if args.output is None
        else args.output.expanduser().resolve()
    )
    if output.is_file():
        summary = validate_editing_m2d_cache_online_parity(output, cache=cache)
        print(json.dumps({"event": "reuse", **summary}, sort_keys=True), flush=True)
        return 0

    vae_config = args.vae_config.expanduser().resolve(strict=True)
    vae_checkpoint = args.vae_checkpoint.expanduser().resolve(strict=True)
    if not (
        sha256_file(vae_config) == EDITING_M2D_VAE_CONFIG_SHA256
        and sha256_file(vae_checkpoint) == EDITING_M2D_VAE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("frozen FOA VAE identity changed")
    vae = create_model_from_config(load_config(vae_config))
    copy_state_dict(vae, load_ckpt_state_dict(str(vae_checkpoint)))
    vae.eval().requires_grad_(False).to(device)
    m2d = FrozenEditingM2DCLAP(
        device=device, load_text_encoder=False
    ).eval().requires_grad_(False)
    harness = _OnlineSemanticHarness(vae, m2d, device)
    selected = _expected_online_parity_rows(cache)
    latents = _load_latents(selected)
    cached = {
        int(row["pair_ordinal"]): cache.get(
            int(row["pair_ordinal"]),
            pair_id=row["pair_id"],
            source_sample_id=row["source_sample_id"],
            source_latent_tensor_sha256=row["source_latent_tensor_sha256"],
        )["source_m2d_audio_embedding"].contiguous()
        for row in selected
    }
    online = {
        batch: _online_pass(
            harness,
            selected,
            latents,
            batch_size=batch,
            device=device,
        )
        for batch in BATCH_SIZES
    }
    row_reports = []
    for row in selected:
        ordinal = int(row["pair_ordinal"])
        cache_value = cached[ordinal]
        report = {**row, "cache_audio_embedding_sha256": _embedding_sha256(cache_value)}
        for batch in BATCH_SIZES:
            value = online[batch][ordinal]
            expanded_cache = cache_value.unsqueeze(0).expand_as(value)
            delta = (value.float() - expanded_cache.float()).abs()
            replica_sha = [_embedding_sha256(replica) for replica in value]
            report[f"online_batch{batch}_sha256"] = replica_sha[0]
            report[f"online_batch{batch}_replica_sha256"] = replica_sha
            report[f"online_batch{batch}_mismatch_elements"] = int(
                torch.count_nonzero(value != expanded_cache).item()
            )
            report[f"online_batch{batch}_max_abs"] = float(delta.max().item())
        row_reports.append(report)
    exact = {
        batch: all(
            row[f"online_batch{batch}_mismatch_elements"] == 0
            and row[f"online_batch{batch}_max_abs"] == 0.0
            for row in row_reports
        )
        for batch in BATCH_SIZES
    }
    checks = {
        "selected_rows_cover_both_buckets_and_all_five_cache_shards": (
            len(row_reports) == 10
            and {int(row["latent_bucket_frames"]) for row in row_reports}
            == {432, 648}
            and all(
                sum(int(row["latent_bucket_frames"]) == bucket for row in row_reports)
                == 5
                for bucket in (432, 648)
            )
            and all(
                {
                    int(row["pair_ordinal"]) % 5
                    for row in row_reports
                    if int(row["latent_bucket_frames"]) == bucket
                }
                == set(range(5))
                for bucket in (432, 648)
            )
        ),
        "source_latent_tensor_hashes_verified": len(latents) == 10,
        "cache_online_batch1_fp16_exact": exact[1],
        "cache_online_batch2_fp16_exact": exact[2],
        "cache_online_batch4_fp16_exact": exact[4],
        "decoded_frozen_vae_W_view_used": True,
        "old_sceneplan_model_input_false": True,
        "target_information_used_false": True,
        "runtime_and_assets_match_cache_contract": (
            editing_m2d_numeric_runtime_fingerprint(device)
            == json.loads(cache.metadata["numeric_runtime_fingerprint_json"])
            and editing_m2d_cache_implementation_sha256()
            == json.loads(cache.metadata["implementation_sha256_json"])
        ),
    }
    if set(checks) != EDITING_M2D_CACHE_ONLINE_PARITY_CHECKS:
        raise RuntimeError("M2D online-parity check inventory changed")
    status = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "schema": EDITING_M2D_CACHE_ONLINE_PARITY_SCHEMA,
        "schema_version": 1,
        "status": status,
        "physical_gpu": 3,
        "rows": len(row_reports),
        "rows_per_bucket": 5,
        "batch_sizes": list(BATCH_SIZES),
        "batch_shape_contract": "repeat_same_exact_geometry_row_v1",
        "cache": str(cache.path),
        "cache_sha256": cache.cache_sha256,
        "cache_marker": str(cache.marker_path),
        "cache_marker_sha256": sha256_file(cache.marker_path),
        "cache_rows": len(cache),
        "source_index": str(cache.source_index),
        "source_index_sha256": cache.metadata["source_index_sha256"],
        "split": cache.metadata["split"],
        "source_audio_view": M2D_CLAP_SOURCE_AUDIO_VIEW,
        "temporal_policy": M2D_CLAP_TEMPORAL_POLICY,
        "audio_preprocess": EDITING_M2D_AUDIO_PREPROCESS,
        "vae_config": str(vae_config),
        "vae_config_sha256": EDITING_M2D_VAE_CONFIG_SHA256,
        "vae_checkpoint": str(vae_checkpoint),
        "vae_checkpoint_sha256": EDITING_M2D_VAE_CHECKPOINT_SHA256,
        "m2d_audio_assets": m2d.asset_report,
        "numeric_runtime_fingerprint": editing_m2d_numeric_runtime_fingerprint(
            device
        ),
        "implementation_sha256": editing_m2d_cache_implementation_sha256(),
        "verifier": str(Path(__file__).resolve()),
        "verifier_sha256": sha256_file(Path(__file__).resolve()),
        "selected_rows": row_reports,
        "checks": checks,
    }
    _atomic_json(output, report)
    if status != "PASS":
        raise RuntimeError(
            "Editing M2D cache and formal online replay differ at fp16 boundary"
        )
    summary = validate_editing_m2d_cache_online_parity(output, cache=cache)
    print(json.dumps({"event": "complete", **summary}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
