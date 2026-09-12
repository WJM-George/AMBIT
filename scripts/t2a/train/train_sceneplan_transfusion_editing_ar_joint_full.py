#!/usr/bin/env python3
"""Five-GPU full Editing AR training with same-pair Editing-DiT replay.

Latest route, and no other route:

    Editing AR: clean source FOA latent + raw edit instruction
                -> complete new ScenePlan
    Editing DiT: [noisy target latent || clean source FOA latent] + new ScenePlan
                 -> edited target FOA latent

The two objectives execute through the same ``ContinuousTransformer`` block
objects in one DDP forward.  Stored old ScenePlans are offline provenance and
are rejected if they appear in any model-facing row.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import sys
import tempfile
import time
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.sceneplan_bucket_sampler import (  # noqa: E402
    DistributedScenePlanBucketBatchSampler,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (  # noqa: E402
    ScenePlanTransfusionEditingDataset,
    verify_editing_source_latent_shards,
    verify_editing_target_latent_shards,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import (  # noqa: E402
    EDITING_JOINT_DATASET_CONTRACT,
    ScenePlanTransfusionEditingJointDataset,
    collate_editing_joint,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_m2d_clap import (  # noqa: E402
    ScenePlanTransfusionEditingM2DCLAPCache,
    editing_m2d_cache_online_parity_path,
    validate_editing_m2d_cache_online_parity,
)
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import (  # noqa: E402
    EDITING_AR_CONTRACT,
    ScenePlanTransfusionEditingAR,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (  # noqa: E402
    EDITING_M2D_CLAP_CONTRACT,
    EDITING_M2D_CLAP_MODES,
    EDITING_M2D_CLAP_SIDE_INPUT,
    editing_m2d_mode_flags,
    multi_positive_source_caption_infonce,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_runtime import (  # noqa: E402
    M2D_CLAP_SOURCE_AUDIO_VIEW,
    M2D_CLAP_TEMPORAL_POLICY,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_provenance import (  # noqa: E402
    JOINT_TRAINING_SOURCE_PATHS,
    verify_frozen_qwen_runtime,
)
from stable_audio_tools.models.utils import load_ckpt_state_dict  # noqa: E402
from stable_audio_tools.training.factory import (  # noqa: E402
    create_training_wrapper_from_config,
)
from scripts.t2a.eval.select_sceneplan_transfusion_editing_dit_checkpoint import (  # noqa: E402
    AUDITED_SOURCE_PATHS as DIT_SELECTOR_AUDITED_SOURCE_PATHS,
    _gpu_topology,
    _validate_existing_selection as _validate_dit_checkpoint_selection,
)
from scripts.t2a.train.sceneplan_transfusion_editing_dit_run_contract import (  # noqa: E402
    canonical_sha256 as _dit_contract_canonical_sha256,
    validate_contract as validate_dit_training_run_contract,
)
from scripts.t2a.train.sceneplan_transfusion_editing_joint_run_contract import (  # noqa: E402
    CHECKPOINT_SCHEMA as JOINT_CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION as JOINT_CHECKPOINT_SCHEMA_VERSION,
    FINAL_SCHEMA as JOINT_FINAL_SCHEMA,
    FINAL_SCHEMA_VERSION as JOINT_FINAL_SCHEMA_VERSION,
    LATEST_SCHEMA as JOINT_LATEST_SCHEMA,
    LATEST_SCHEMA_VERSION as JOINT_LATEST_SCHEMA_VERSION,
    RNG_SCHEMA as JOINT_RNG_SCHEMA,
    RNG_SCHEMA_VERSION as JOINT_RNG_SCHEMA_VERSION,
    RUN_CONTRACT_NAME as JOINT_RUN_CONTRACT_NAME,
    RUN_CONTRACT_SCHEMA as JOINT_RUN_CONTRACT_SCHEMA,
    RUN_CONTRACT_SCHEMA_VERSION as JOINT_RUN_CONTRACT_SCHEMA_VERSION,
    RUN_IDENTITY_NAME as JOINT_RUN_IDENTITY_NAME,
    ensure_run_identity as ensure_joint_run_identity,
    latest_record as joint_latest_record,
    resolve_resume as resolve_joint_resume,
    validate_rng_inventory as validate_joint_rng_inventory,
)


DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json"
)
DEFAULT_CODEC = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
LATEST_AR_INPUT_CONTRACT = "source_foa_latent_plus_raw_edit_request_v2"
FULL_VISIBLE_GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
FULL_WORLD_SIZE = max(len([item for item in FULL_VISIBLE_GPUS.split(",") if item]), 1)
FORMAL_M2D_AR_SPECIFIC_TRAINABLE_PARAMETERS = 10_034_176
FORMAL_M2D_JOINT_UNIQUE_TRAINABLE_PARAMETERS = 329_348_480
DIT_SELECTION_SCHEMA = "sceneplan_transfusion_editing_dit_checkpoint_selection"
DIT_SELECTION_CONTRACT = (
    "full_20k_10k_select_10k_holdout_ema_rf_paired_source_and_p10_gate_5k_v4"
)
FORBIDDEN_OLD_PLAN_KEYS = {
    "old_sceneplan",
    "old_plan",
    "source_sceneplan",
    "source_plan",
    "previous_sceneplan",
    "previous_plan",
}


def _reject_old_plan_metadata(metadata: Sequence[dict[str, Any]]) -> None:
    for row in metadata:
        normalized = {str(key).lower().replace("-", "_") for key in row}
        leaked = normalized & FORBIDDEN_OLD_PLAN_KEYS
        if leaked:
            raise RuntimeError(f"old ScenePlan leaked into joint Editing: {leaked}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--checkpoint-selection-sha256", required=True)
    parser.add_argument("--dit-gt-audio-gate", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--train-index-sha256", required=True)
    parser.add_argument("--validation-index-sha256", required=True)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument(
        "--source-semantic-mode",
        choices=sorted(EDITING_M2D_CLAP_MODES),
        default="m2d_audio_caption_aux",
    )
    parser.add_argument("--train-m2d-cache", type=Path)
    parser.add_argument("--validation-m2d-cache", type=Path)
    parser.add_argument("--train-m2d-cache-sha256")
    parser.add_argument("--validation-m2d-cache-sha256")
    parser.add_argument("--expected-train-rows", type=int, default=1_000_000)
    parser.add_argument("--expected-validation-rows", type=int, default=20_000)
    parser.add_argument("--short-batch-size", type=int, default=8)
    parser.add_argument("--long-batch-size", type=int, default=5)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=25_000)
    parser.add_argument("--ar-learning-rate", type=float, default=1e-4)
    parser.add_argument("--shared-learning-rate", type=float, default=2e-6)
    parser.add_argument("--dit-learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lambda-ar", type=float, default=0.1)
    parser.add_argument("--lambda-rf", type=float, default=1.0)
    parser.add_argument("--lambda-source-caption", type=float, default=0.05)
    parser.add_argument("--source-caption-temperature", type=float, default=0.07)
    parser.add_argument("--source-semantic-dropout", type=float, default=0.10)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=5_000)
    parser.add_argument("--validation-batches", type=int, default=32)
    parser.add_argument("--validation-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-activation-checkpointing", action="store_true")
    return parser.parse_args()


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


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _distributed() -> tuple[int, int, int, torch.device, dict[str, Any]]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("full Editing requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    if visible != FULL_VISIBLE_GPUS:
        raise RuntimeError(
            f"full Editing may expose only physical GPUs {FULL_VISIBLE_GPUS}; "
            f"got CUDA_VISIBLE_DEVICES={visible!r}"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != FULL_WORLD_SIZE:
        raise RuntimeError("full Editing joint training requires exactly five ranks")
    if not torch.cuda.is_available() or torch.cuda.device_count() != FULL_WORLD_SIZE:
        raise RuntimeError("full Editing requires exactly five visible CUDA devices")
    # This is intentionally checked before NCCL initialization on every rank:
    # if the host's nvidia-smi indices no longer match PCI-ordered CUDA indices,
    # every process fails independently instead of leaving peers in a collective.
    topology = _gpu_topology()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, world_size, device, topology


def _barrier() -> None:
    dist.barrier()


def _seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + 10_003 * int(rank)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _capture_rank_rng_state(
    *, rank: int, device: torch.device | None
) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    cuda_state = None
    if device is not None and device.type == "cuda":
        cuda_state = torch.cuda.get_rng_state(device).cpu()
    return {
        "schema": JOINT_RNG_SCHEMA,
        "schema_version": JOINT_RNG_SCHEMA_VERSION,
        "rank": int(rank),
        "python_random_state": random.getstate(),
        "numpy_random_state": {
            "bit_generator": str(numpy_state[0]),
            "keys": torch.from_numpy(
                np.asarray(numpy_state[1], dtype=np.uint32).astype(np.int64)
            ),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_state": cuda_state,
    }


def _restore_rank_rng_state(
    state: dict[str, Any], *, rank: int, device: torch.device | None
) -> None:
    numpy_state = state.get("numpy_random_state")
    cpu_state = state.get("torch_cpu_rng_state")
    cuda_state = state.get("torch_cuda_rng_state")
    if not (
        state.get("schema") == JOINT_RNG_SCHEMA
        and int(state.get("schema_version", -1)) == JOINT_RNG_SCHEMA_VERSION
        and int(state.get("rank", -1)) == int(rank)
        and isinstance(state.get("python_random_state"), tuple)
        and isinstance(numpy_state, dict)
        and isinstance(numpy_state.get("keys"), torch.Tensor)
        and numpy_state.get("bit_generator") == "MT19937"
        and numpy_state["keys"].dtype == torch.int64
        and int(numpy_state["keys"].numel()) == 624
        and isinstance(cpu_state, torch.Tensor)
        and cpu_state.dtype == torch.uint8
        and (
            (device is None or device.type != "cuda")
            or (
                isinstance(cuda_state, torch.Tensor)
                and cuda_state.dtype == torch.uint8
            )
        )
    ):
        raise RuntimeError(f"joint Editing RNG state for rank {rank} changed")
    try:
        random.setstate(state["python_random_state"])
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                numpy_state["keys"].cpu().numpy().astype(np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(cpu_state.cpu())
        if device is not None and device.type == "cuda":
            assert isinstance(cuda_state, torch.Tensor)
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"joint Editing RNG state for rank {rank} is not restorable"
        ) from exc


def _gather_rank_rng_states(
    *, rank: int, world_size: int, device: torch.device
) -> list[dict[str, Any]]:
    local = _capture_rank_rng_state(rank=rank, device=device)
    gathered: list[dict[str, Any] | None] = [None] * int(world_size)
    dist.all_gather_object(gathered, local)
    if any(value is None for value in gathered):
        raise RuntimeError("joint Editing RNG gather returned an empty rank")
    ordered = sorted(
        (dict(value) for value in gathered if value is not None),
        key=lambda value: int(value.get("rank", -1)),
    )
    validate_joint_rng_inventory(ordered, world_size=world_size)
    return ordered


def _index_summary(path: Path, expected_sha: str, expected_rows: int) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    observed_sha = sha256_file(path)
    if observed_sha != str(expected_sha):
        raise RuntimeError(f"Editing index SHA256 changed: {path}")
    marker_path = path.with_suffix(path.suffix + ".frozen.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        rows, short_rows, long_rows = connection.execute(
            "SELECT COUNT(*),SUM(latent_bucket_frames=432),"
            "SUM(latent_bucket_frames=648) FROM pairs"
        ).fetchone()
    finally:
        connection.close()
    expected = {
        "schema": "sceneplan_transfusion_editing_training_index",
        "schema_version": "1",
        "state": "materialized_complete_frozen",
        "editing_ar_input_contract": LATEST_AR_INPUT_CONTRACT,
        "editing_ar_old_sceneplan_input": "false",
        "target_latents_exhaustively_reopened": "true",
        "target_tensor_hashes_exhaustively_verified": "true",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"Editing index {key} is not latest-route truth")
    rows = int(rows)
    if rows != int(expected_rows) or int(metadata.get("rows", -1)) != rows:
        raise RuntimeError(f"Editing index row count changed: {rows} != {expected_rows}")
    if (
        marker.get("index_sha256") != observed_sha
        or int(marker.get("rows", -1)) != rows
        or marker.get("state") != "materialized_complete_frozen"
    ):
        raise RuntimeError("Editing frozen marker changed")
    source_shard_audit = verify_editing_source_latent_shards(path)
    target_shard_audit = verify_editing_target_latent_shards(path)
    return {
        "path": str(path),
        "sha256": observed_sha,
        "marker_path": str(marker_path.resolve()),
        "marker_sha256": sha256_file(marker_path),
        "split": metadata["split"],
        "rows": rows,
        "short_rows": int(short_rows or 0),
        "long_rows": int(long_rows or 0),
        "editing_ar_input_contract": metadata["editing_ar_input_contract"],
        "editing_ar_old_sceneplan_input": False,
        **source_shard_audit,
        **target_shard_audit,
    }


def _source_semantic_cache_summary(
    path: Path,
    *,
    expected_sha256: str,
    source_index: Path,
    source_index_sha256: str,
    expected_rows: int,
    split: str,
) -> dict[str, Any]:
    """Hash and fully reopen one frozen M2D cache on rank zero."""

    resolved = path.expanduser().resolve(strict=True)
    observed_sha = sha256_file(resolved)
    if observed_sha != str(expected_sha256):
        raise RuntimeError(f"Editing M2D {split} cache SHA256 changed")
    cache = ScenePlanTransfusionEditingM2DCLAPCache(
        resolved,
        source_index=source_index,
        source_index_sha256=source_index_sha256,
        expected_rows=int(expected_rows),
        expected_split=split,
        expected_cache_sha256=observed_sha,
        verify_cache_file_hash=False,
    )
    online_parity = validate_editing_m2d_cache_online_parity(
        editing_m2d_cache_online_parity_path(resolved), cache=cache
    )
    return {
        "path": str(resolved),
        "sha256": observed_sha,
        "marker_path": str(cache.marker_path),
        "marker_sha256": sha256_file(cache.marker_path),
        "source_index": str(cache.source_index),
        "source_index_sha256": cache.metadata["source_index_sha256"],
        "rows": len(cache),
        "split": split,
        "semantic_contract": cache.metadata["semantic_contract"],
        "audio_preprocess": cache.metadata["audio_preprocess"],
        "source_audio_view": cache.metadata["source_audio_view"],
        "temporal_policy": cache.metadata["temporal_policy"],
        "temporal_pilot": dict(cache.temporal_pilot),
        "caption_target": cache.metadata["caption_target"],
        "vae_config": cache.metadata["vae_config"],
        "vae_config_sha256": cache.metadata["vae_config_sha256"],
        "vae_checkpoint": cache.metadata["vae_checkpoint"],
        "vae_checkpoint_sha256": cache.metadata["vae_checkpoint_sha256"],
        "m2d_assets": json.loads(cache.metadata["m2d_assets_json"]),
        "implementation_sha256": json.loads(
            cache.metadata["implementation_sha256_json"]
        ),
        "numeric_runtime_fingerprint": json.loads(
            cache.metadata["numeric_runtime_fingerprint_json"]
        ),
        "cache_online_parity": online_parity,
        "shard_builder": cache.metadata["shard_builder"],
        "shard_builder_sha256": cache.metadata["shard_builder_sha256"],
        "merger": cache.metadata["merger"],
        "merger_sha256": cache.metadata["merger_sha256"],
        "old_sceneplan_model_input": False,
        "caption_model_input": False,
        "target_information_used": False,
    }


def _stratified_validation_batches(
    index: Path, *, batch_count: int, batch_size: int
) -> tuple[list[list[int]], dict[str, Any]]:
    """Choose fixed whole batches spanning every operation and length bucket."""

    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    try:
        strata = [
            (str(operation), int(bucket))
            for operation, bucket in connection.execute(
                "SELECT DISTINCT operation,latent_bucket_frames FROM pairs "
                "ORDER BY operation,latent_bucket_frames"
            )
        ]
        if not strata or int(batch_count) < len(strata):
            raise RuntimeError("validation batch budget cannot cover every stratum")
        base, remainder = divmod(int(batch_count), len(strata))
        batches: list[list[int]] = []
        counts: dict[str, int] = {}
        for stratum_index, (operation, bucket) in enumerate(strata):
            stratum_batches = base + int(stratum_index < remainder)
            needed = stratum_batches * int(batch_size)
            ordinals = [
                int(row[0])
                for row in connection.execute(
                    "SELECT pair_ordinal FROM pairs "
                    "WHERE operation=? AND latent_bucket_frames=? "
                    "ORDER BY pair_id LIMIT ?",
                    (operation, bucket, needed),
                )
            ]
            if len(ordinals) != needed:
                raise RuntimeError("validation stratum lacks fixed audit rows")
            batches.extend(
                ordinals[start : start + int(batch_size)]
                for start in range(0, len(ordinals), int(batch_size))
            )
            counts[f"{bucket}:{operation}"] = len(ordinals)
    finally:
        connection.close()
    flattened = [ordinal for batch in batches for ordinal in batch]
    if (
        len(batches) != int(batch_count)
        or len(flattened) != int(batch_count) * int(batch_size)
        or len(set(flattened)) != len(flattened)
    ):
        raise RuntimeError("fixed validation batches are incomplete or duplicated")
    return batches, {
        "contract": "fixed_bucket_operation_stratified_whole_batches_v1",
        "batches": len(batches),
        "batch_size": int(batch_size),
        "rows": len(flattened),
        "by_bucket_operation": counts,
        "pair_ordinal_sha256": hashlib.sha256(
            "".join(f"{value}\n" for value in flattened).encode("ascii")
        ).hexdigest(),
    }


def _checkpoint_selection_summary(
    path: Path,
    *,
    expected_sha256: str,
    checkpoint: Path,
    validation_summary: dict[str, Any],
    model_config: Path,
) -> dict[str, Any]:
    """Revalidate the independent DiT promotion artifact before CUDA loading."""

    path = path.expanduser().resolve(strict=True)
    observed_selection_sha = sha256_file(path)
    if observed_selection_sha != str(expected_sha256):
        raise RuntimeError("Editing-DiT checkpoint-selection SHA256 changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    selected = Path(value.get("selected_checkpoint", "")).expanduser().resolve(
        strict=True
    )
    candidate_steps = list(range(5_000, 30_001, 5_000))
    candidates = list(value.get("candidates") or [])
    selected_rows = [
        row
        for row in candidates
        if int(row.get("step", -1))
        == int(value.get("selected_checkpoint_step", -2))
    ]
    checks = dict(value.get("selected_promotion_gate", {}).get("checks") or {})
    required_checks = {
        "clean_better_than_zero_source_99pct",
        "clean_better_than_shuffled_source_99pct",
        "clean_better_than_p10_warmstart_99pct",
        "positive_zero_source_prediction_response_99pct",
        "positive_shuffled_source_prediction_response_99pct",
    }
    source_records = dict(value.get("source_sha256") or {})
    expected_source_records = {
        relative: sha256_file((REPO_ROOT / relative).resolve(strict=True))
        for relative in DIT_SELECTOR_AUDITED_SOURCE_PATHS
    }
    source_hashes_valid = source_records == expected_source_records
    training_run = dict(value.get("training_run") or {})
    training_log = Path(training_run.get("log_path", "")).resolve(strict=True)
    dit_run_dir = Path(training_run.get("run_dir", "")).resolve(strict=True)
    contract_record = dict(value.get("training_run_contract") or {})
    dit_contract_path = Path(contract_record.get("path", "")).resolve(
        strict=True
    )
    dit_contract, dit_contract_sha = validate_dit_training_run_contract(
        dit_contract_path, expected_run_dir=dit_run_dir
    )
    if not (
        contract_record.get("sha256") == dit_contract_sha
        and contract_record.get("canonical_sha256")
        == _dit_contract_canonical_sha256(dit_contract)
        and _validate_dit_checkpoint_selection(
            path,
            run_dir=dit_run_dir,
            preflight=Path(value.get("preflight", {}).get("path", "")).resolve(
                strict=True
            ),
            model_config=model_config,
            validation_index=Path(validation_summary["path"]).resolve(strict=True),
            training_run_contract=dit_contract,
            training_run_contract_path=dit_contract_path,
            training_run_contract_sha256=dit_contract_sha,
        )
        == value
    ):
        raise RuntimeError("Editing-DiT selection failed full producer validation")
    if not (
        value.get("schema") == DIT_SELECTION_SCHEMA
        and int(value.get("schema_version", -1)) == 1
        and value.get("status") == "PASS"
        and value.get("selection_contract") == DIT_SELECTION_CONTRACT
        and value.get("physical_gpus") == [3, 4, 5, 6, 7]
        and int(value.get("world_size", -1)) == FULL_WORLD_SIZE
        and value.get("cuda_visible_devices") == FULL_VISIBLE_GPUS
        and value.get("latest_route", {}).get("editing_ar_inputs")
        == ["source_foa_latent", "raw_edit_request"]
        and value.get("latest_route", {}).get("editing_ar_target")
        == "complete_new_sceneplan"
        and value.get("latest_route", {}).get("old_sceneplan_input") is False
        and int(
            value.get("latest_route", {}).get("editing_dit_frame_channels", -1)
        )
        == 384
        and value.get("candidate_steps") == candidate_steps
        and [int(row.get("step", -1)) for row in candidates] == candidate_steps
        and len(selected_rows) == 1
        and selected == checkpoint
        and value.get("selected_checkpoint_sha256") == sha256_file(checkpoint)
        and selected_rows[0].get("checkpoint_sha256")
        == value.get("selected_checkpoint_sha256")
        and int(value.get("validation_index", {}).get("rows", -1)) == 20_000
        and Path(value.get("validation_index", {}).get("path", "")).resolve()
        == Path(validation_summary["path"]).resolve()
        and value.get("validation_index", {}).get("sha256")
        == validation_summary["sha256"]
        and Path(value.get("model_config", {}).get("path", "")).resolve()
        == model_config
        and value.get("model_config", {}).get("sha256")
        == sha256_file(model_config)
        and value.get("p10_warmstart_baseline", {}).get(
            "source_suffix_zero_invariance_pass"
        )
        is True
        and value.get("selected_promotion_gate", {}).get("pass") is True
        and value.get("selected_promotion_gate", {}).get("inference_fold")
        == "sealed_holdout_10k"
        and int(
            value.get("selected_promotion_gate", {}).get(
                "holdout_reuse_count", -1
            )
        )
        == 1
        and int(
            value.get("validation_folds", {})
            .get("selection", {})
            .get("rows", -1)
        )
        == 10_000
        and int(
            value.get("validation_folds", {})
            .get("holdout", {})
            .get("rows", -1)
        )
        == 10_000
        and set(checks) == required_checks
        and all(checks.values())
        and int(training_run.get("training_gate", {}).get("global_step", -1))
        == 30_000
        and training_run.get("training_gate", {}).get("status") == "PASS"
        and training_run.get("log_sha256") == sha256_file(training_log)
        and source_hashes_valid
    ):
        raise RuntimeError("Editing-DiT checkpoint selection is stale or invalid")
    return {
        "path": str(path),
        "sha256": observed_selection_sha,
        "selection_contract": DIT_SELECTION_CONTRACT,
        "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": value["selected_checkpoint_sha256"],
        "selected_checkpoint_step": int(value["selected_checkpoint_step"]),
        "validation_index_sha256": validation_summary["sha256"],
        "training_run_contract": contract_record,
        "selected_clean_source_rf": value["selected_clean_source_rf"],
        "selected_promotion_gate": value["selected_promotion_gate"],
    }


def _unique_trainable_parameters(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    seen: set[int] = set()
    output: list[nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                output.append(parameter)
    if not output:
        raise RuntimeError("joint Editing found no trainable parameters")
    return output


def _copy_ema_to_online(wrapper) -> None:
    if wrapper.diffusion_ema is None:
        raise RuntimeError("Editing-DiT checkpoint must contain EMA weights")
    incompatible = wrapper.diffusion.model.load_state_dict(
        wrapper.diffusion_ema.ema_model.state_dict(), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("could not promote Editing-DiT EMA to online route")
    if wrapper.conditioner_ema is not None:
        wrapper.conditioner_ema.copy_to(wrapper.diffusion.conditioner)


class JointEditingModule(nn.Module):
    """One DDP forward containing both AR CE and aligned RF branches."""

    def __init__(self, *, diffusion, ar: ScenePlanTransfusionEditingAR) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.ar = ar
        if self.ar.shared_transformer is not self.diffusion.model.model.transformer:
            raise RuntimeError("Editing AR and DiT must share one Transformer object")

    def forward(
        self,
        *,
        source_foa_latent: torch.Tensor,
        source_attention_mask: torch.Tensor,
        plan_input_ids: torch.Tensor,
        plan_attention_mask: torch.Tensor,
        raw_edit_requests: Sequence[str],
        metadata: list[dict[str, Any]],
        noised_target: torch.Tensor,
        timesteps: torch.Tensor,
        rf_padding_mask: torch.Tensor,
        source_m2d_audio_embedding: torch.Tensor | None = None,
        source_clap_features: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _reject_old_plan_metadata(metadata)
        context, context_mask = self.ar.encode_edit_instructions(
            raw_edit_requests, device=source_foa_latent.device
        )
        ar_output = self.ar(
            source_foa_latent,
            source_attention_mask,
            plan_input_ids,
            plan_attention_mask,
            context,
            context_mask,
            source_m2d_audio_embedding=source_m2d_audio_embedding,
            **({"source_clap_features": source_clap_features} if source_clap_features is not None else {}),
            return_source_contrastive_query=(
                self.ar.source_semantic_bridge.align_caption
            ),
        )
        if self.ar.source_semantic_bridge.align_caption:
            if not isinstance(ar_output, tuple) or len(ar_output) != 2:
                raise RuntimeError("Editing AR omitted its source-caption query")
            ar_logits, source_caption_query = ar_output
        else:
            if not isinstance(ar_output, torch.Tensor):
                raise RuntimeError("Editing AR returned an unexpected auxiliary")
            ar_logits = ar_output
            source_caption_query = source_foa_latent.new_empty((0, 768))
        conditioning = self.diffusion.conditioner(metadata, source_foa_latent.device)
        conditioned_source = conditioning.get("source_foa_latent", [None])[0]
        if (
            not isinstance(conditioned_source, torch.Tensor)
            or tuple(conditioned_source.shape) != tuple(source_foa_latent.shape)
            or not torch.equal(conditioned_source.float(), source_foa_latent.float())
        ):
            raise RuntimeError("AR and DiT clean source tensors diverged")
        conditioning = dict(conditioning)
        conditioning["source_foa_latent"] = [source_foa_latent, None]
        conditioning_inputs = self.diffusion.get_conditioning_inputs(conditioning)
        rf_prediction = self.diffusion.model(
            noised_target,
            timesteps,
            **conditioning_inputs,
            cfg_dropout_prob=0.0,
            padding_mask=rf_padding_mask,
        )
        return ar_logits, rf_prediction, source_caption_query


def _move_joint_batch(
    batch: dict[str, Any], device: torch.device
) -> tuple[dict[str, Any], torch.Tensor, list[dict[str, Any]], torch.Tensor]:
    ar = batch["ar"]
    for key in (
        "source_foa_latent",
        "source_attention_mask",
        "plan_input_ids",
        "plan_labels",
        "plan_attention_mask",
    ):
        ar[key] = ar[key].to(device, non_blocking=True)
    for key in (
        "source_m2d_audio_embedding",
        "source_caption_m2d_embedding",
        "source_caption_group_ids",
        "source_semantic_group_ids",
    ):
        if key in ar:
            ar[key] = ar[key].to(device, non_blocking=True)
    target = batch["target_foa_latent"].to(
        device=device, dtype=torch.float32, non_blocking=True
    )
    metadata = list(batch["metadata"])
    padding_mask = torch.stack(
        [row["padding_mask"][0] for row in metadata]
    ).to(device, non_blocking=True)
    ar["source_foa_latent"] = ar["source_foa_latent"].to(torch.float32)
    return ar, target, metadata, padding_mask


def _losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prediction: torch.Tensor,
    rf_target: torch.Tensor,
    rf_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ar_sum = F.cross_entropy(
        logits.float().flatten(0, 1),
        labels.flatten(),
        ignore_index=-100,
        reduction="sum",
    )
    ar_tokens = (labels != -100).sum().clamp_min(1)
    squared = (prediction.float() - rf_target.float()) ** 2
    rf_sum = (squared * rf_mask[:, None]).sum()
    rf_values = (rf_mask.sum() * rf_target.shape[1]).clamp_min(1)
    return ar_sum / ar_tokens, rf_sum / rf_values, ar_sum, rf_sum


@torch.no_grad()
def _validation_metrics(
    module: JointEditingModule,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int,
    source_caption_temperature: float,
) -> dict[str, float]:
    module.eval()
    totals = torch.zeros(13, dtype=torch.float64)
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= int(max_batches):
            break
        ar, target, metadata, rf_mask = _move_joint_batch(raw_batch, device)
        context, context_mask = module.ar.encode_edit_instructions(
            ar["raw_edit_requests"], device=device
        )
        source = ar["source_foa_latent"]
        donor = source.roll(1, dims=0)
        donor_mask = ar["source_attention_mask"].roll(1, dims=0)
        semantic = (
            ar.get("source_m2d_audio_embedding")
            if module.ar.source_semantic_bridge.inject_audio
            else None
        )
        clean_semantic_kwargs = (
            {} if semantic is None else {"source_m2d_audio_embedding": semantic}
        )
        zero_semantic_kwargs = (
            {}
            if semantic is None
            else {
                "source_m2d_audio_embedding": semantic,
                "source_m2d_audio_keep_mask": torch.zeros(
                    source.shape[0], device=device, dtype=torch.bool
                ),
            }
        )
        shuffled_semantic_kwargs = (
            {}
            if semantic is None
            else {"source_m2d_audio_embedding": semantic.roll(1, dims=0)}
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            clean_output = module.ar(
                source,
                ar["source_attention_mask"],
                ar["plan_input_ids"],
                ar["plan_attention_mask"],
                context,
                context_mask,
                return_source_contrastive_query=(
                    module.ar.source_semantic_bridge.align_caption
                ),
                **clean_semantic_kwargs,
            )
            zero_logits = module.ar(
                torch.zeros_like(source),
                ar["source_attention_mask"],
                ar["plan_input_ids"],
                ar["plan_attention_mask"],
                context,
                context_mask,
                **zero_semantic_kwargs,
            )
            shuffled_logits = module.ar(
                donor,
                donor_mask,
                ar["plan_input_ids"],
                ar["plan_attention_mask"],
                context,
                context_mask,
                **shuffled_semantic_kwargs,
            )
        if module.ar.source_semantic_bridge.align_caption:
            clean_logits, source_caption_query = clean_output
            contrastive_loss, contrastive_metrics = (
                multi_positive_source_caption_infonce(
                    source_caption_query,
                    ar["source_caption_m2d_embedding"],
                    ar["source_caption_group_ids"],
                    ar["source_semantic_group_ids"],
                    temperature=float(source_caption_temperature),
                    gather_distributed=False,
                )
            )
            contrastive_rows = int(source.shape[0])
        else:
            clean_logits = clean_output
            contrastive_loss = torch.zeros((), device=device)
            contrastive_metrics = {
                "positive_cosine": torch.zeros((), device=device),
                "audio_top1_positive": torch.zeros((), device=device),
            }
            contrastive_rows = 0
        labels = ar["plan_labels"]
        valid = labels != -100
        tokens = int(valid.sum().item())
        clean_sum = F.cross_entropy(
            clean_logits.float().flatten(0, 1),
            labels.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        zero_sum = F.cross_entropy(
            zero_logits.float().flatten(0, 1),
            labels.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        shuffle_sum = F.cross_entropy(
            shuffled_logits.float().flatten(0, 1),
            labels.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        predicted = clean_logits.argmax(dim=-1)
        correct = (predicted.eq(labels) & valid).sum()
        exact = ((predicted.eq(labels) & valid) | ~valid).all(dim=1).sum()

        generator = torch.Generator(device=device).manual_seed(8_100_000 + batch_index)
        noise = torch.randn(
            target.shape, generator=generator, device=device, dtype=target.dtype
        )
        timesteps = torch.full((target.shape[0],), 0.5, device=device)
        noised = 0.5 * target + 0.5 * noise
        conditioning = module.diffusion.conditioner(metadata, device)
        inputs = module.diffusion.get_conditioning_inputs(conditioning)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rf_prediction = module.diffusion.model(
                noised,
                timesteps,
                **inputs,
                cfg_dropout_prob=0.0,
                padding_mask=rf_mask,
            )
        rf_squared = (rf_prediction.float() - (noise - target)) ** 2
        rf_sum = (rf_squared * rf_mask[:, None]).sum()
        rf_values = int(rf_mask.sum().item()) * int(target.shape[1])
        totals += torch.tensor(
            [
                float(clean_sum),
                float(zero_sum),
                float(shuffle_sum),
                tokens,
                float(correct),
                float(exact),
                int(labels.shape[0]),
                float(rf_sum),
                rf_values,
                float(contrastive_loss) * contrastive_rows,
                contrastive_rows,
                float(contrastive_metrics["positive_cosine"])
                * contrastive_rows,
                float(contrastive_metrics["audio_top1_positive"])
                * contrastive_rows,
            ],
            dtype=torch.float64,
        )
    module.train()
    values = totals.tolist()
    token_denominator = max(1.0, values[3])
    return {
        "clean_ar_ce": values[0] / token_denominator,
        "zero_reference_ar_ce": values[1] / token_denominator,
        "shuffled_reference_ar_ce": values[2] / token_denominator,
        "token_accuracy": values[4] / token_denominator,
        "teacher_forced_sequence_exact": values[5] / max(1.0, values[6]),
        "rf_mse": values[7] / max(1.0, values[8]),
        "tokens": values[3],
        "sequences": values[6],
        "source_caption_infonce": values[9] / max(1.0, values[10]),
        "source_caption_positive_cosine": values[11] / max(1.0, values[10]),
        "source_caption_audio_top1_positive": values[12]
        / max(1.0, values[10]),
        "source_caption_rows": values[10],
    }


def _ar_specific_state(ar: ScenePlanTransfusionEditingAR) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in ar.state_dict().items()
        if not key.startswith("editing_dit.")
        and not key.startswith("instruction_conditioner.")
    }


def _save_checkpoint(
    path: Path,
    *,
    module: JointEditingModule,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    run_contract: dict[str, Any],
    rng_states_by_rank: list[dict[str, Any]],
) -> None:
    run_dir = Path(str(run_contract.get("run_dir", ""))).resolve()
    run_id = str(run_contract.get("run_id", ""))
    contract_path = (run_dir / JOINT_RUN_CONTRACT_NAME).resolve(strict=True)
    contract_sha = sha256_file(contract_path)
    if not (
        path.resolve().parent == run_dir / "checkpoints"
        and json.loads(contract_path.read_text(encoding="utf-8")) == run_contract
    ):
        raise RuntimeError("joint Editing checkpoint escaped its run contract")
    validate_joint_rng_inventory(
        rng_states_by_rank, world_size=int(run_contract.get("world_size", -1))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "schema": JOINT_CHECKPOINT_SCHEMA,
            "schema_version": JOINT_CHECKPOINT_SCHEMA_VERSION,
            "contract": EDITING_AR_CONTRACT,
            "joint_dataset_contract": EDITING_JOINT_DATASET_CONTRACT,
            "run_dir": str(run_dir),
            "run_id": run_id,
            "run_contract_path": str(contract_path),
            "run_contract_sha256": contract_sha,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "batch_in_epoch": int(batch_in_epoch),
            "diffusion_state_dict": {
                key: value.detach().cpu()
                for key, value in module.diffusion.state_dict().items()
            },
            "editing_ar_specific_state_dict": _ar_specific_state(module.ar),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "run_contract": run_contract,
            "rng_states_by_rank": rng_states_by_rank,
        },
        temporary,
    )
    os.replace(temporary, path)
    _atomic_json(
        path.parent / "LATEST.json",
        joint_latest_record(
            run_dir=run_dir,
            identity={"run_id": run_id},
            run_contract_path=contract_path,
            run_contract_sha256=contract_sha,
            checkpoint=path.resolve(),
            checkpoint_sha256=sha256_file(path),
            step=global_step,
        ),
    )


def _completed_resume_action(
    run_dir: Path, *, resume_path: Path, global_step: int
) -> str:
    """Decide whether a max-step resume is complete or needs finalization.

    The interval checkpoint is written before the final validation artifact.
    A process interruption in that narrow window must not force a 25K run to
    restart, but recovery is allowed only when the checkpoint and LATEST hash
    are exact.  An existing but inconsistent FINAL remains a hard failure.
    """

    checkpoint, checkpoint_sha = _published_checkpoint_identity(
        run_dir, global_step=global_step
    )
    final_path = run_dir / "FINAL.json"
    if checkpoint != resume_path:
        raise RuntimeError("completed joint Editing LATEST/checkpoint disagree")
    if not final_path.exists():
        return "finalize"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    contract_path = (run_dir / JOINT_RUN_CONTRACT_NAME).resolve(strict=True)
    run_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not (
        final.get("schema") == JOINT_FINAL_SCHEMA
        and int(final.get("schema_version", -1)) == JOINT_FINAL_SCHEMA_VERSION
        and final.get("event") == "complete"
        and int(final.get("step", -1)) == int(global_step)
        and final.get("run_dir") == str(run_dir.resolve())
        and final.get("run_id") == run_contract.get("run_id")
        and final.get("run_contract_path") == str(contract_path)
        and final.get("run_contract_sha256") == sha256_file(contract_path)
        and Path(final.get("checkpoint", "")).resolve(strict=True) == checkpoint
        and final.get("checkpoint_sha256") == checkpoint_sha
        and final.get("shared_transformer_same_object") is True
        and final.get("old_sceneplan_exposed") is False
        and final.get("source_semantic_mode") == "m2d_audio_caption_aux"
        and final.get("source_caption_exposed_as_model_input") is False
    ):
        raise RuntimeError("completed joint Editing FINAL is inconsistent")
    return "reuse"


def _published_checkpoint_identity(
    run_dir: Path, *, global_step: int
) -> tuple[Path, str]:
    """Verify the already-published interval checkpoint and its resume pointer."""

    checkpoint = (
        run_dir / "checkpoints" / f"step-{int(global_step):08d}.pt"
    ).resolve(strict=True)
    latest_path = (run_dir / "checkpoints/LATEST.json").resolve(strict=True)
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    checkpoint_sha = sha256_file(checkpoint)
    contract_path = (run_dir / JOINT_RUN_CONTRACT_NAME).resolve(strict=True)
    run_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected = joint_latest_record(
        run_dir=run_dir.resolve(),
        identity={"run_id": run_contract.get("run_id")},
        run_contract_path=contract_path,
        run_contract_sha256=sha256_file(contract_path),
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        step=global_step,
    )
    if latest != expected:
        raise RuntimeError("completed joint Editing LATEST/checkpoint disagree")
    return checkpoint, checkpoint_sha


def _finalize_joint_run(
    *,
    module: JointEditingModule,
    validation_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    run_dir: Path,
    global_step: int,
) -> dict[str, Any]:
    """Validate the published max-step checkpoint, then atomically mark final."""

    if int(global_step) != int(args.max_steps):
        raise RuntimeError("joint Editing finalization requires the exact max step")
    final_metrics = _validation_metrics(
        module,
        validation_loader,
        device=device,
        max_batches=int(args.validation_batches),
        source_caption_temperature=float(args.source_caption_temperature),
    )
    checkpoint, checkpoint_sha = _published_checkpoint_identity(
        run_dir, global_step=global_step
    )
    contract_path = (run_dir / JOINT_RUN_CONTRACT_NAME).resolve(strict=True)
    run_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    final = {
        "schema": JOINT_FINAL_SCHEMA,
        "schema_version": JOINT_FINAL_SCHEMA_VERSION,
        "event": "complete",
        "step": int(global_step),
        "run_dir": str(run_dir.resolve()),
        "run_id": run_contract["run_id"],
        "run_contract_path": str(contract_path),
        "run_contract_sha256": sha256_file(contract_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "source_semantic_mode": str(args.source_semantic_mode),
        "shared_transformer_same_object": (
            module.ar.shared_transformer
            is module.diffusion.model.model.transformer
        ),
        "old_sceneplan_exposed": False,
        "source_caption_exposed_as_model_input": False,
        **final_metrics,
    }
    _append_jsonl(run_dir / "metrics.jsonl", final)
    _atomic_json(run_dir / "FINAL.json", final)
    print(json.dumps(final, sort_keys=True), flush=True)
    return final


def _load_ar_specific(
    ar: ScenePlanTransfusionEditingAR, state: dict[str, torch.Tensor]
) -> None:
    incompatible = ar.load_state_dict(state, strict=False)
    if incompatible.unexpected_keys or any(
        not (
            key.startswith("editing_dit.")
            or key.startswith("instruction_conditioner.")
        )
        for key in incompatible.missing_keys
    ):
        raise RuntimeError(
            "Editing AR resume adapter mismatch: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )


def _reduce_training_totals(values: Sequence[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(list(values), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def main() -> int:
    args = _parse_args()
    inject_m2d_audio, align_source_caption = editing_m2d_mode_flags(
        args.source_semantic_mode
    )
    positive = (
        args.expected_train_rows,
        args.expected_validation_rows,
        args.short_batch_size,
        args.long_batch_size,
        args.gradient_accumulation,
        args.max_steps,
        args.ar_learning_rate,
        args.shared_learning_rate,
        args.dit_learning_rate,
        args.lambda_ar,
        args.lambda_rf,
        args.gradient_clip,
        args.validate_every,
        args.save_every,
        args.source_caption_temperature,
    )
    finite_values = (
        args.ar_learning_rate,
        args.shared_learning_rate,
        args.dit_learning_rate,
        args.weight_decay,
        args.lambda_ar,
        args.lambda_rf,
        args.lambda_source_caption,
        args.source_caption_temperature,
        args.source_semantic_dropout,
        args.gradient_clip,
    )
    if any(float(value) <= 0 or not math.isfinite(float(value)) for value in positive):
        raise ValueError("full Editing hyperparameters must be positive")
    if any(not math.isfinite(float(value)) for value in finite_values):
        raise ValueError("full Editing floating-point settings must be finite")
    semantic_cache_args = (
        args.train_m2d_cache,
        args.validation_m2d_cache,
        args.train_m2d_cache_sha256,
        args.validation_m2d_cache_sha256,
    )
    if (
        align_source_caption != (float(args.lambda_source_caption) > 0.0)
        or not 0.0 < float(args.source_caption_temperature) <= 1.0
        or not 0.0 <= float(args.source_semantic_dropout) < 1.0
        or (not inject_m2d_audio and float(args.source_semantic_dropout) != 0.0)
        or ((inject_m2d_audio or align_source_caption) and not all(semantic_cache_args))
        or (not (inject_m2d_audio or align_source_caption) and any(semantic_cache_args))
    ):
        raise ValueError(
            "Editing source-semantic mode/cache/loss/dropout settings disagree"
        )
    if (
        int(args.log_every) <= 0
        or int(args.validation_batches) <= 0
        or int(args.validation_batch_size) <= 0
        or int(args.num_workers) < 0
    ):
        raise ValueError("full Editing logging/validation/worker settings are invalid")
    rank, local_rank, world_size, device, gpu_topology = _distributed()
    _seed_everything(int(args.seed), rank)
    torch.set_float32_matmul_precision("high")

    run_dir = args.run_dir.expanduser().resolve()
    train_index = args.train_index.expanduser().resolve(strict=True)
    validation_index = args.validation_index.expanduser().resolve(strict=True)
    train_m2d_cache_path = (
        None
        if args.train_m2d_cache is None
        else args.train_m2d_cache.expanduser().resolve(strict=True)
    )
    validation_m2d_cache_path = (
        None
        if args.validation_m2d_cache is None
        else args.validation_m2d_cache.expanduser().resolve(strict=True)
    )
    model_config_path = args.model_config.expanduser().resolve(strict=True)
    model_config = load_config(model_config_path)
    prompt_configs = [
        value
        for value in model_config["model"]["conditioning"]["configs"]
        if value.get("id") == "prompt"
    ]
    if not (
        len(prompt_configs) == 1
        and prompt_configs[0].get("type") == "qwen_text"
        and prompt_configs[0].get("config", {}).get("enable_grad") is False
    ):
        raise RuntimeError("joint Editing requires the frozen Qwen conditioner")
    base_checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    # Hash the 8.9-GB train index, the selected multi-GB checkpoint, and the
    # unregistered 1.75-GB Qwen runtime exactly once.  Broadcasting immutable
    # summaries avoids five ranks concurrently rereading tens of GB while
    # retaining the same fail-closed checks on every rank.
    startup_audit: list[dict[str, Any] | None] = [None]
    if rank == 0:
        try:
            run_identity_value = ensure_joint_run_identity(run_dir)
            resume_action, resolved_resume = resolve_joint_resume(
                run_dir,
                requested_checkpoint=args.resume,
                max_steps=int(args.max_steps),
                save_every=int(args.save_every),
            )
            if args.resume is None:
                if resume_action != "FRESH":
                    raise RuntimeError(
                        "published joint Editing artifacts require the launcher's "
                        "verified resume"
                    )
            elif resume_action not in {"RESUME", "COMPLETE"}:
                raise RuntimeError("joint Editing requested resume has no checkpoint")
            elif resume_action == "RESUME" and Path(
                str(resolved_resume)
            ).resolve(strict=True) != args.resume.expanduser().resolve(strict=True):
                raise RuntimeError("joint Editing resolver changed the resume target")
            train_summary_value = _index_summary(
                train_index, args.train_index_sha256, args.expected_train_rows
            )
            validation_summary_value = _index_summary(
                validation_index,
                args.validation_index_sha256,
                args.expected_validation_rows,
            )
            if (
                train_summary_value["split"] != "train"
                or validation_summary_value["split"] != "validation"
            ):
                raise RuntimeError("full Editing train/validation split mismatch")
            selection_summary_value = _checkpoint_selection_summary(
                args.checkpoint_selection,
                expected_sha256=args.checkpoint_selection_sha256,
                checkpoint=base_checkpoint,
                validation_summary=validation_summary_value,
                model_config=model_config_path,
            )
            # Import here to keep the standalone scorer's shared metric imports
            # acyclic. This rank-zero audit precedes model/optimizer creation.
            from scripts.t2a.eval.evaluate_sceneplan_transfusion_editing_gt_audio import validate_audio_gate
            gt_audio_path = args.dit_gt_audio_gate.expanduser().resolve(strict=True)
            validate_audio_gate(gt_audio_path, selection_path=args.checkpoint_selection)
            gt_audio_summary = {
                "path": str(gt_audio_path), "sha256": sha256_file(gt_audio_path),
                "plan_origin": "ground_truth", "status": "PASS",
            }
            qwen_runtime_value = verify_frozen_qwen_runtime(
                prompt_configs[0]["config"]["model_path"]
            )
            semantic_cache_value = None
            if inject_m2d_audio or align_source_caption:
                assert train_m2d_cache_path is not None
                assert validation_m2d_cache_path is not None
                semantic_cache_value = {
                    "contract": EDITING_M2D_CLAP_CONTRACT,
                    "mode": str(args.source_semantic_mode),
                    "train": _source_semantic_cache_summary(
                        train_m2d_cache_path,
                        expected_sha256=str(args.train_m2d_cache_sha256),
                        source_index=train_index,
                        source_index_sha256=args.train_index_sha256,
                        expected_rows=int(args.expected_train_rows),
                        split="train",
                    ),
                    "validation": _source_semantic_cache_summary(
                        validation_m2d_cache_path,
                        expected_sha256=str(args.validation_m2d_cache_sha256),
                        source_index=validation_index,
                        source_index_sha256=args.validation_index_sha256,
                        expected_rows=int(args.expected_validation_rows),
                        split="validation",
                    ),
                }
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "checkpoints").mkdir(exist_ok=True)
            if (run_dir / "checkpoints").resolve(strict=True) != (
                run_dir / "checkpoints"
            ):
                raise RuntimeError(
                    "joint Editing checkpoint directory escaped its run"
                )
            startup_audit[0] = {
                "run_identity": run_identity_value,
                "train": train_summary_value,
                "validation": validation_summary_value,
                "selection": selection_summary_value,
                "dit_gt_audio_gate": gt_audio_summary,
                "qwen_runtime": qwen_runtime_value,
                "source_semantic_cache": semantic_cache_value,
            }
        except Exception as exc:  # noqa: BLE001
            startup_audit[0] = {
                "error": f"{type(exc).__name__}: {exc}"
            }
    dist.broadcast_object_list(startup_audit, src=0, device=device)
    if startup_audit[0] is None or startup_audit[0].get("error"):
        detail = None if startup_audit[0] is None else startup_audit[0].get("error")
        raise RuntimeError(f"joint Editing startup audit failed: {detail}")
    train_summary = startup_audit[0]["train"]
    run_identity = startup_audit[0]["run_identity"]
    validation_summary = startup_audit[0]["validation"]
    selection_summary = startup_audit[0]["selection"]
    qwen_runtime = startup_audit[0]["qwen_runtime"]
    source_semantic_cache_summary = startup_audit[0]["source_semantic_cache"]
    _barrier()

    diffusion = create_model_from_config(model_config)
    training_wrapper = create_training_wrapper_from_config(model_config, diffusion)
    base_state, base_metadata = load_ckpt_state_dict(
        str(base_checkpoint), return_metadata=True
    )
    saved_conditioner_ema_names = list(
        base_metadata.get("conditioner_ema_parameter_names") or []
    )
    current_conditioner_ema_names = list(
        training_wrapper.conditioner_ema.parameter_names
        if training_wrapper.conditioner_ema is not None
        else ()
    )
    if saved_conditioner_ema_names != current_conditioner_ema_names:
        raise RuntimeError(
            "conditioner EMA parameter mapping changed between selected DiT "
            "checkpoint and joint model"
        )
    incompatible = training_wrapper.load_state_dict(base_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Editing-DiT base checkpoint mismatch: "
            f"missing={incompatible.missing_keys[:8]}, "
            f"unexpected={incompatible.unexpected_keys[:8]}"
        )
    del base_state
    _copy_ema_to_online(training_wrapper)
    diffusion = training_wrapper.diffusion
    training_wrapper.diffusion_ema = None
    training_wrapper.conditioner_ema = None
    diffusion.pretransform = None
    route_core = diffusion.model.model
    prompt_conditioner = diffusion.conditioner.conditioners["prompt"]
    codec_path = args.codec.expanduser().resolve(strict=True)
    codec = ModelScenePlanCodecV4(codec_path)
    ar = ScenePlanTransfusionEditingAR(
        editing_dit=route_core,
        instruction_conditioner=prompt_conditioner,
        pad_id=codec.pad_id,
        activation_checkpointing=not args.no_activation_checkpointing,
        source_semantic_mode=str(args.source_semantic_mode),
        source_semantic_dropout=float(args.source_semantic_dropout),
    )
    module = JointEditingModule(diffusion=diffusion, ar=ar)
    module.to(device)
    module.train()

    train_base = ScenePlanTransfusionEditingDataset(
        train_index,
        tokenizer_spec=(prompt_conditioner.tokenizer, 512),
        expected_num_samples=int(args.expected_train_rows),
        expected_index_sha256=args.train_index_sha256,
        latent_crop_length=648,
        require_frozen=True,
        verify_tensor_hashes_on_access=False,
    )
    validation_base = ScenePlanTransfusionEditingDataset(
        validation_index,
        tokenizer_spec=(prompt_conditioner.tokenizer, 512),
        expected_num_samples=int(args.expected_validation_rows),
        expected_index_sha256=args.validation_index_sha256,
        latent_crop_length=648,
        require_frozen=True,
        verify_tensor_hashes_on_access=False,
    )
    train_semantic_cache = None
    validation_semantic_cache = None
    if inject_m2d_audio or align_source_caption:
        assert source_semantic_cache_summary is not None
        train_record = source_semantic_cache_summary["train"]
        validation_record = source_semantic_cache_summary["validation"]
        train_semantic_cache = ScenePlanTransfusionEditingM2DCLAPCache(
            train_record["path"],
            source_index=train_index,
            source_index_sha256=args.train_index_sha256,
            expected_rows=int(args.expected_train_rows),
            expected_split="train",
            expected_cache_sha256=train_record["sha256"],
        )
        validation_semantic_cache = ScenePlanTransfusionEditingM2DCLAPCache(
            validation_record["path"],
            source_index=validation_index,
            source_index_sha256=args.validation_index_sha256,
            expected_rows=int(args.expected_validation_rows),
            expected_split="validation",
            expected_cache_sha256=validation_record["sha256"],
        )
    train_dataset = ScenePlanTransfusionEditingJointDataset(
        train_base, codec=codec, source_semantic_cache=train_semantic_cache
    )
    validation_dataset = ScenePlanTransfusionEditingJointDataset(
        validation_base,
        codec=codec,
        source_semantic_cache=validation_semantic_cache,
    )
    train_sampler = DistributedScenePlanBucketBatchSampler(
        train_dataset,
        short_batch_size=int(args.short_batch_size),
        long_batch_size=int(args.long_batch_size),
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(args.seed),
        drop_last=True,
    )
    collate = functools.partial(collate_editing_joint, pad_id=codec.pad_id)
    # Worker base seeds must not consume the model's restored CPU RNG stream
    # when a mid-epoch resume constructs a new DataLoader iterator.
    train_loader_generator = torch.Generator().manual_seed(
        int(args.seed) + 90_001 * int(rank)
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=int(args.num_workers),
        pin_memory=True,
        persistent_workers=int(args.num_workers) > 0,
        collate_fn=collate,
        generator=train_loader_generator,
    )
    validation_batches, validation_probe = _stratified_validation_batches(
        validation_index,
        batch_count=int(args.validation_batches),
        batch_size=int(args.validation_batch_size),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_batches,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )

    ar_specific = _unique_trainable_parameters(
        (
            ar.source_audio_adapter,
            ar.plan_adapter,
            ar.source_semantic_bridge,
        )
    ) + [ar.source_audio_type_embedding, ar.plan_type_embedding]
    ar_ids = {id(parameter) for parameter in ar_specific}
    shared = _unique_trainable_parameters((route_core.transformer.layers,))
    shared_ids = {id(parameter) for parameter in shared}
    remaining = [
        parameter
        for parameter in _unique_trainable_parameters((diffusion,))
        if id(parameter) not in ar_ids and id(parameter) not in shared_ids
    ]
    parameters = [*ar_specific, *shared, *remaining]
    parameter_counts = {
        "editing_ar_adapters": sum(p.numel() for p in ar_specific),
        "shared_transformer_blocks": sum(p.numel() for p in shared),
        "editing_dit_and_conditioners": sum(p.numel() for p in remaining),
    }
    parameter_counts["total_unique"] = sum(parameter_counts.values())
    if (
        str(args.source_semantic_mode) == "m2d_audio_caption_aux"
        and (
            parameter_counts["editing_ar_adapters"]
            != FORMAL_M2D_AR_SPECIFIC_TRAINABLE_PARAMETERS
            or parameter_counts["total_unique"]
            != FORMAL_M2D_JOINT_UNIQUE_TRAINABLE_PARAMETERS
        )
    ):
        raise RuntimeError(
            "formal M2D joint trainable-parameter accounting changed: "
            f"{parameter_counts}"
        )
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise RuntimeError("joint Editing optimizer contains duplicate parameters")
    module_parameter_ids = {
        id(parameter) for parameter in module.parameters() if parameter.requires_grad
    }
    if module_parameter_ids != {id(parameter) for parameter in parameters}:
        raise RuntimeError("joint Editing optimizer does not cover the exact trainable set")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": ar_specific,
                "lr": float(args.ar_learning_rate),
                "base_lr": float(args.ar_learning_rate),
                "group_name": "editing_ar_adapters",
            },
            {
                "params": shared,
                "lr": float(args.shared_learning_rate),
                "base_lr": float(args.shared_learning_rate),
                "group_name": "shared_transformer_blocks",
            },
            {
                "params": remaining,
                "lr": float(args.dit_learning_rate),
                "base_lr": float(args.dit_learning_rate),
                "group_name": "editing_dit_and_conditioners",
            },
        ],
        betas=(0.9, 0.95),
        weight_decay=float(args.weight_decay),
        fused=True,
    )

    def lr_lambda(step: int) -> float:
        if step < int(args.warmup_steps):
            return float(step + 1) / float(max(1, int(args.warmup_steps)))
        progress = (step - int(args.warmup_steps)) / max(
            1, int(args.max_steps) - int(args.warmup_steps)
        )
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0
    start_epoch = 0
    resume_batch = 0

    source_files = [
        (REPO_ROOT / relative).resolve(strict=True)
        for relative in JOINT_TRAINING_SOURCE_PATHS
    ]
    run_contract = {
        "schema": JOINT_RUN_CONTRACT_SCHEMA,
        "schema_version": JOINT_RUN_CONTRACT_SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "run_id": run_identity["run_id"],
        "run_identity": {
            "path": str((run_dir / JOINT_RUN_IDENTITY_NAME).resolve(strict=True)),
            "sha256": sha256_file(
                (run_dir / JOINT_RUN_IDENTITY_NAME).resolve(strict=True)
            ),
        },
        "editing_ar_contract": EDITING_AR_CONTRACT,
        "joint_dataset_contract": EDITING_JOINT_DATASET_CONTRACT,
        "latest_route": {
            "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
            "editing_ar_target": "complete_new_sceneplan",
            "old_sceneplan_input": False,
            "source_caption_model_input": False,
            "target_audio_or_latent_ar_input": False,
            "source_derived_semantic_side_input": (
                EDITING_M2D_CLAP_SIDE_INPUT
                if inject_m2d_audio
                else None
            ),
            "editing_dit_frame_input": [
                "noisy_target_64",
                "new_sceneplan_256",
                "clean_source_foa_latent_64",
            ],
            "ar_and_dit_share_exact_transformer_object": True,
        },
        "seed": int(args.seed),
        "world_size": world_size,
        "physical_gpus": [3, 4, 5, 6, 7],
        "cuda_visible_devices": FULL_VISIBLE_GPUS,
        "cuda_device_order": "PCI_BUS_ID",
        "gpu_topology": gpu_topology,
        "short_batch_size_per_rank": int(args.short_batch_size),
        "long_batch_size_per_rank": int(args.long_batch_size),
        "gradient_accumulation": int(args.gradient_accumulation),
        "max_steps": int(args.max_steps),
        "log_every": int(args.log_every),
        "validate_every": int(args.validate_every),
        "save_every": int(args.save_every),
        "validation_batches": int(args.validation_batches),
        "validation_batch_size": int(args.validation_batch_size),
        "num_workers_per_rank": int(args.num_workers),
        "internal_validation_probe": validation_probe,
        "lambda_ar": float(args.lambda_ar),
        "lambda_rf": float(args.lambda_rf),
        "source_semantic": {
            "contract": EDITING_M2D_CLAP_CONTRACT,
            "mode": str(args.source_semantic_mode),
            "inject_frozen_m2d_audio": bool(inject_m2d_audio),
            "align_source_caption": bool(align_source_caption),
            "audio_embedding_is_source_derived": True,
            "source_audio_view": M2D_CLAP_SOURCE_AUDIO_VIEW,
            "temporal_policy": M2D_CLAP_TEMPORAL_POLICY,
            "caption_is_training_label_only": True,
            "caption_model_input": False,
            "old_sceneplan_model_input": False,
            "new_sceneplan_or_target_information_used": False,
            "lambda_source_caption": float(args.lambda_source_caption),
            "source_caption_temperature": float(args.source_caption_temperature),
            "source_semantic_dropout": float(args.source_semantic_dropout),
            "cache": source_semantic_cache_summary,
        },
        "learning_rates": {
            "editing_ar_adapters": float(args.ar_learning_rate),
            "shared_transformer_blocks": float(args.shared_learning_rate),
            "editing_dit_and_conditioners": float(args.dit_learning_rate),
        },
        "train_index": train_summary,
        "validation_index": validation_summary,
        "base_checkpoint_selection": selection_summary,
        "dit_gt_audio_gate": startup_audit[0]["dit_gt_audio_gate"],
        "base_checkpoint": str(base_checkpoint),
        # The selected checkpoint was already hashed and fail-closed by rank 0
        # in ``startup_audit``.  Reuse that immutable identity instead of
        # making every rank reread the same multi-GB file while constructing
        # an otherwise identical run contract.
        "base_checkpoint_sha256": selection_summary[
            "selected_checkpoint_sha256"
        ],
        "model_config": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
        "frozen_qwen_runtime": qwen_runtime,
        "codec": str(codec_path),
        "codec_fingerprint": codec.fingerprint,
        "activation_checkpointing": not args.no_activation_checkpointing,
        "trainable_parameters": parameter_counts,
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in source_files
        },
    }
    contract_path = run_dir / JOINT_RUN_CONTRACT_NAME
    if rank == 0:
        if contract_path.exists():
            if contract_path.resolve(strict=True) != contract_path:
                raise RuntimeError(
                    "joint Editing run contract escaped its run directory"
                )
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != run_contract:
                raise RuntimeError("resume run contract changed")
        else:
            _atomic_json(contract_path, run_contract)
    _barrier()

    pending_rng_state: dict[str, Any] | None = None
    if args.resume is not None:
        resume_path = args.resume.expanduser().resolve(strict=True)
        checkpoint_dir = (run_dir / "checkpoints").resolve(strict=True)
        try:
            filename_step = int(resume_path.name.removeprefix("step-").removesuffix(".pt"))
        except ValueError as exc:
            raise RuntimeError("joint Editing resume filename is not canonical") from exc
        if not (
            resume_path.parent == checkpoint_dir
            and resume_path.name == f"step-{filename_step:08d}.pt"
            and 0 < filename_step <= int(args.max_steps)
            and filename_step % int(args.save_every) == 0
        ):
            raise RuntimeError("joint Editing resume escaped canonical 5K checkpoints")
        resume_audit: list[dict[str, Any] | None] = [None]
        if rank == 0:
            try:
                latest_path = (checkpoint_dir / "LATEST.json").resolve(strict=True)
                latest = json.loads(latest_path.read_text(encoding="utf-8"))
                checkpoint_sha = sha256_file(resume_path)
                expected_latest = joint_latest_record(
                    run_dir=run_dir,
                    identity=run_identity,
                    run_contract_path=contract_path.resolve(strict=True),
                    run_contract_sha256=sha256_file(contract_path),
                    checkpoint=resume_path,
                    checkpoint_sha256=checkpoint_sha,
                    step=filename_step,
                )
                if latest != expected_latest:
                    raise RuntimeError("joint Editing resume is not the pinned LATEST")
                if (run_dir / "FINAL.json").exists() and filename_step != int(
                    args.max_steps
                ):
                    raise RuntimeError(
                        "completed joint Editing run cannot roll back to an older checkpoint"
                    )
                resume_audit[0] = {
                    "step": filename_step,
                    "checkpoint_sha256": checkpoint_sha,
                }
            except Exception as exc:  # noqa: BLE001
                resume_audit[0] = {"error": f"{type(exc).__name__}: {exc}"}
        dist.broadcast_object_list(resume_audit, src=0, device=device)
        if resume_audit[0] is None or resume_audit[0].get("error"):
            detail = None if resume_audit[0] is None else resume_audit[0].get("error")
            raise RuntimeError(f"joint Editing resume identity failed: {detail}")
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            state.get("schema") != JOINT_CHECKPOINT_SCHEMA
            or int(state.get("schema_version", -1))
            != JOINT_CHECKPOINT_SCHEMA_VERSION
            or state.get("contract") != EDITING_AR_CONTRACT
            or state.get("run_dir") != str(run_dir)
            or state.get("run_id") != run_identity["run_id"]
            or state.get("run_contract_path")
            != str(contract_path.resolve(strict=True))
            or state.get("run_contract_sha256") != sha256_file(contract_path)
            or state.get("run_contract") != run_contract
        ):
            raise RuntimeError("joint Editing resume checkpoint contract mismatch")
        rng_states = state.get("rng_states_by_rank")
        validate_joint_rng_inventory(rng_states, world_size=world_size)
        pending_rng_state = dict(rng_states[rank])
        module.diffusion.load_state_dict(state["diffusion_state_dict"], strict=True)
        _load_ar_specific(ar, state["editing_ar_specific_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        global_step = int(state["global_step"])
        if global_step != filename_step:
            raise RuntimeError("joint Editing resume payload step changed")
        start_epoch = int(state["epoch"])
        resume_batch = int(state["batch_in_epoch"])
        if resume_batch >= len(train_sampler):
            start_epoch += 1
            resume_batch = 0
        sampler_state = train_sampler.resumable_state_dict(at_epoch_boundary=True)
        sampler_state["resume_epoch"] = start_epoch
        train_sampler.load_resumable_state_dict(sampler_state)
        if resume_batch:
            train_sampler.set_resume_batch_offset(resume_batch)
        del state
        if global_step > int(args.max_steps):
            raise RuntimeError("joint Editing resume step exceeds formal max_steps")
        if global_step == int(args.max_steps):
            assert pending_rng_state is not None
            _restore_rank_rng_state(
                pending_rng_state, rank=rank, device=device
            )
            completion_result: list[dict[str, str] | None] = [None]
            if rank == 0:
                try:
                    completion_result[0] = {
                        "action": _completed_resume_action(
                            run_dir,
                            resume_path=resume_path,
                            global_step=global_step,
                        )
                    }
                except Exception as exc:  # noqa: BLE001
                    completion_result[0] = {
                        "error": f"{type(exc).__name__}: {exc}"
                    }
            dist.broadcast_object_list(completion_result, src=0, device=device)
            result = completion_result[0]
            if result is None or result.get("error"):
                detail = None if result is None else result.get("error")
                raise RuntimeError(f"completed joint Editing audit failed: {detail}")
            action = result.get("action")
            if action == "finalize":
                if rank == 0:
                    _finalize_joint_run(
                        module=module,
                        validation_loader=validation_loader,
                        device=device,
                        args=args,
                        run_dir=run_dir,
                        global_step=global_step,
                    )
                    print(
                        json.dumps(
                            {
                                "event": "joint_editing_finalization_recovered",
                                "step": global_step,
                                "checkpoint": str(resume_path),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                _barrier()
                dist.destroy_process_group()
                return 0
            if action != "reuse":
                raise RuntimeError(
                    f"unknown completed joint Editing action: {action!r}"
                )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "joint_editing_complete_resume_reused",
                            "step": global_step,
                            "checkpoint": str(resume_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            _barrier()
            dist.destroy_process_group()
            return 0

    wrapped = DistributedDataParallel(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    if pending_rng_state is not None:
        # Restore only after model/optimizer/DDP construction so no startup
        # operation can perturb the checkpointed stochastic stream.
        _restore_rank_rng_state(pending_rng_state, rank=rank, device=device)
    metrics_path = run_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    training_started = time.perf_counter()
    # AR sum/tokens, RF sum/values, then source-caption loss and diagnostics
    # weighted by their global contrastive row count.
    accumulated = [0.0] * 10
    stop = False

    for epoch in range(start_epoch, 10**9):
        for batch_index, raw_batch in enumerate(train_loader, start=resume_batch):
            ar_batch, target, metadata, rf_mask = _move_joint_batch(raw_batch, device)
            noise = torch.randn_like(target)
            timesteps = torch.rand(target.shape[0], device=device)
            noised = (
                (1.0 - timesteps[:, None, None]) * target
                + timesteps[:, None, None] * noise
            )
            rf_target = noise - target
            microstep = batch_index % int(args.gradient_accumulation)
            should_step = microstep == int(args.gradient_accumulation) - 1
            sync_context = wrapped.no_sync() if not should_step else nullcontext()
            with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                logits, prediction, source_caption_query = wrapped(
                    source_foa_latent=ar_batch["source_foa_latent"],
                    source_attention_mask=ar_batch["source_attention_mask"],
                    plan_input_ids=ar_batch["plan_input_ids"],
                    plan_attention_mask=ar_batch["plan_attention_mask"],
                    raw_edit_requests=ar_batch["raw_edit_requests"],
                    metadata=metadata,
                    noised_target=noised,
                    timesteps=timesteps,
                    rf_padding_mask=rf_mask,
                    source_m2d_audio_embedding=(
                        ar_batch["source_m2d_audio_embedding"]
                        if inject_m2d_audio
                        else None
                    ),
                )
                ar_loss, rf_loss, ar_sum, rf_sum = _losses(
                    logits,
                    ar_batch["plan_labels"],
                    prediction,
                    rf_target,
                    rf_mask,
                )
                if align_source_caption:
                    contrastive_loss, contrastive_metrics = (
                        multi_positive_source_caption_infonce(
                            source_caption_query,
                            ar_batch["source_caption_m2d_embedding"],
                            ar_batch["source_caption_group_ids"],
                            ar_batch["source_semantic_group_ids"],
                            temperature=float(args.source_caption_temperature),
                            gather_distributed=True,
                        )
                    )
                    contrastive_rows = int(
                        contrastive_metrics["global_rows"].item()
                    )
                else:
                    contrastive_loss = logits.new_zeros((), dtype=torch.float32)
                    contrastive_metrics = {
                        "positive_cosine": contrastive_loss.detach(),
                        "audio_top1_positive": contrastive_loss.detach(),
                        "text_top1_positive": contrastive_loss.detach(),
                        "mean_positives_per_row": contrastive_loss.detach(),
                    }
                    contrastive_rows = 0
                loss = (
                    float(args.lambda_ar) * ar_loss
                    + float(args.lambda_rf) * rf_loss
                    + float(args.lambda_source_caption) * contrastive_loss
                ) / int(args.gradient_accumulation)
            loss.backward()
            ar_tokens = int((ar_batch["plan_labels"] != -100).sum().item())
            rf_values = int(rf_mask.sum().item()) * int(target.shape[1])
            accumulated[0] += float(ar_sum.detach())
            accumulated[1] += ar_tokens
            accumulated[2] += float(rf_sum.detach())
            accumulated[3] += rf_values
            accumulated[4] += float(contrastive_loss.detach()) * contrastive_rows
            accumulated[5] += contrastive_rows
            accumulated[6] += (
                float(contrastive_metrics["positive_cosine"]) * contrastive_rows
            )
            accumulated[7] += (
                float(contrastive_metrics["audio_top1_positive"])
                * contrastive_rows
            )
            accumulated[8] += (
                float(contrastive_metrics["text_top1_positive"])
                * contrastive_rows
            )
            accumulated[9] += (
                float(contrastive_metrics["mean_positives_per_row"])
                * contrastive_rows
            )
            if not should_step:
                continue

            ar_grad = torch.nn.utils.clip_grad_norm_(
                ar_specific, float(args.gradient_clip)
            )
            shared_grad = torch.nn.utils.clip_grad_norm_(
                shared, float(args.gradient_clip)
            )
            dit_grad = torch.nn.utils.clip_grad_norm_(
                remaining, float(args.gradient_clip)
            )
            if not all(torch.isfinite(value) for value in (ar_grad, shared_grad, dit_grad)):
                raise RuntimeError(f"non-finite joint Editing gradient at step {global_step}")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % int(args.log_every) == 0:
                reduced = _reduce_training_totals(accumulated, device)
                if rank == 0:
                    event = {
                        "event": "train",
                        "step": global_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_index + 1,
                        "ar_ce": reduced[0] / max(1.0, reduced[1]),
                        "rf_mse": reduced[2] / max(1.0, reduced[3]),
                        "ar_tokens": reduced[1],
                        "rf_values": reduced[3],
                        "source_caption_infonce": reduced[4]
                        / max(1.0, reduced[5]),
                        "source_caption_rows": reduced[5],
                        "source_caption_positive_cosine": reduced[6]
                        / max(1.0, reduced[5]),
                        "source_caption_audio_top1_positive": reduced[7]
                        / max(1.0, reduced[5]),
                        "source_caption_text_top1_positive": reduced[8]
                        / max(1.0, reduced[5]),
                        "source_caption_mean_positives_per_row": reduced[9]
                        / max(1.0, reduced[5]),
                        "gradient_norms": {
                            "editing_ar_adapters": float(ar_grad),
                            "shared_transformer_blocks": float(shared_grad),
                            "editing_dit_and_conditioners": float(dit_grad),
                        },
                        "learning_rates": {
                            str(group["group_name"]): float(group["lr"])
                            for group in optimizer.param_groups
                        },
                        "elapsed_sec": round(time.perf_counter() - training_started, 2),
                    }
                    _append_jsonl(metrics_path, event)
                    print(json.dumps(event, sort_keys=True), flush=True)
                accumulated = [0.0] * 10

            if global_step % int(args.validate_every) == 0:
                _barrier()
                if rank == 0:
                    metrics = _validation_metrics(
                        module,
                        validation_loader,
                        device=device,
                        max_batches=int(args.validation_batches),
                        source_caption_temperature=float(
                            args.source_caption_temperature
                        ),
                    )
                    event = {"event": "validation", "step": global_step, **metrics}
                    _append_jsonl(metrics_path, event)
                    print(json.dumps(event, sort_keys=True), flush=True)
                _barrier()

            if global_step % int(args.save_every) == 0:
                _barrier()
                rng_states_by_rank = _gather_rank_rng_states(
                    rank=rank, world_size=world_size, device=device
                )
                if rank == 0:
                    _save_checkpoint(
                        run_dir / "checkpoints" / f"step-{global_step:08d}.pt",
                        module=module,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        run_contract=run_contract,
                        rng_states_by_rank=rng_states_by_rank,
                    )
                _barrier()
            if global_step >= int(args.max_steps):
                stop = True
                break
        resume_batch = 0
        if stop:
            break
        # Incomplete accumulation tails are deliberately dropped at epoch edges.
        optimizer.zero_grad(set_to_none=True)
        accumulated = [0.0] * 10

    _barrier()
    if rank == 0:
        _finalize_joint_run(
            module=module,
            validation_loader=validation_loader,
            device=device,
            args=args,
            run_dir=run_dir,
            global_step=global_step,
        )
    _barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
