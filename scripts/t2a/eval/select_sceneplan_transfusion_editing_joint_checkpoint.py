#!/usr/bin/env python3
"""Promote one shared Editing AR/RF checkpoint on frozen validation truth.

All five 5K-spaced online joint checkpoints receive a clean pass over the
complete 20K validation set.  Ranking uses a frozen 10K selection fold only.
Exactly one top candidate is then tested once on the disjoint 10K holdout for
AR/RF source dependence and DiT non-inferiority, plus a 500-row stratified
grammar-constrained AR free decode.  There is no fallback to ``LATEST.json``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import functools
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from statistics import NormalDist
import sys
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.select_sceneplan_transfusion_editing_dit_checkpoint import (  # noqa: E402
    CANONICAL_P10,
    DEFAULT_TIMESTEPS,
    EXPECTED_LONG_ROWS,
    EXPECTED_ROWS,
    EXPECTED_SHORT_ROWS,
    PHYSICAL_GPUS,
    VISIBLE_GPUS,
    WORLD_SIZE,
    _all_reduce,
    _broadcast,
    _chunks,
    _evaluate_route,
    _gpu_topology,
    _local_donor_mapping,
    _load_candidate as _load_dit_candidate,
    _make_loader,
    _matrix_summary,
    _paired_report,
    _paired_report_pass,
    _rank0_audit,
    _rank_ordinals,
    _selection_holdout_folds,
    _subset_evaluation,
    _validation_layout,
    _validate_preflight,
)
from scripts.t2a.train.sceneplan_transfusion_editing_dit_run_contract import (  # noqa: E402
    canonical_sha256 as _dit_contract_canonical_sha256,
    validate_contract as validate_dit_training_run_contract,
)
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import (  # noqa: E402
    DIT_SELECTION_CONTRACT,
    DIT_SELECTION_SCHEMA,
    LATEST_AR_INPUT_CONTRACT,
    _checkpoint_selection_summary,
    _index_summary,
    _load_ar_specific,
    _reject_old_plan_metadata,
)
from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (  # noqa: E402
    ScenePlanTransfusionEditingDataset,
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
    EDITING_M2D_VAE_CHECKPOINT_SHA256,
    EDITING_M2D_VAE_CONFIG_SHA256,
    ScenePlanTransfusionEditingM2DCLAPCache,
    editing_m2d_cache_online_parity_path,
    validate_editing_m2d_cache_online_parity,
    validate_editing_m2d_temporal_pilot,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_evaluation import (  # noqa: E402
    score_parsed_generation,
    score_token_sequence,
)
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import (  # noqa: E402
    EDITING_AR_CONTRACT,
    ScenePlanTransfusionEditingAR,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (  # noqa: E402
    EDITING_M2D_CLAP_CONTRACT,
    EDITING_M2D_CLAP_SIDE_INPUT,
    editing_m2d_mode_flags,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_runtime import (  # noqa: E402
    M2D_CLAP_SOURCE_AUDIO_VIEW,
    M2D_CLAP_TEMPORAL_POLICY,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import (  # noqa: E402
    _align_decoded_sceneplan_to_audio_duration,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_provenance import (  # noqa: E402
    JOINT_SELECTION_SOURCE_PATHS,
    JOINT_TRAINING_SOURCE_PATHS,
    verify_frozen_qwen_runtime,
)
from stable_audio_tools.training.factory import (  # noqa: E402
    create_training_wrapper_from_config,
)
from scripts.t2a.train.sceneplan_transfusion_editing_joint_run_contract import (  # noqa: E402
    CHECKPOINT_SCHEMA as JOINT_CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION as JOINT_CHECKPOINT_SCHEMA_VERSION,
    FINAL_SCHEMA as JOINT_FINAL_SCHEMA,
    FINAL_SCHEMA_VERSION as JOINT_FINAL_SCHEMA_VERSION,
    LATEST_SCHEMA as JOINT_LATEST_SCHEMA,
    LATEST_SCHEMA_VERSION as JOINT_LATEST_SCHEMA_VERSION,
    RUN_CONTRACT_SCHEMA as JOINT_RUN_CONTRACT_SCHEMA,
    RUN_CONTRACT_SCHEMA_VERSION as JOINT_RUN_CONTRACT_SCHEMA_VERSION,
    RUN_IDENTITY_NAME as JOINT_RUN_IDENTITY_NAME,
    validate_rng_inventory as validate_joint_rng_inventory,
    validate_run_identity as validate_joint_run_identity,
)


SCHEMA = "sceneplan_transfusion_editing_joint_checkpoint_selection"
SCHEMA_VERSION = 1
SELECTION_CONTRACT = (
    "full_20k_10k_select_10k_holdout_joint_ar_rf_source_"
    "base_dit_noninferiority_m2d_stratified_free_ar_5k_v5"
)
EXPECTED_MAX_STEP = 25_000
CHECKPOINT_EVERY = 5_000
CANDIDATE_STEPS = tuple(range(CHECKPOINT_EVERY, EXPECTED_MAX_STEP + 1, CHECKPOINT_EVERY))
CONFIDENCE = 0.99
FREE_ROWS = 500
FREE_PER_CELL = 50
DEFAULT_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_transfusion_editing_v1")
DEFAULT_RUN = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives//transfusion_editing/mainline/"
    "sceneplan_transfusion_editing_ar_joint_m2d_full_seed42_v3"
)
DEFAULT_DIT_RUN = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives//transfusion_editing/mainline/"
    "sceneplan_transfusion_editing_dit_full_seed42_v1"
)
DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json"
)
DEFAULT_CODEC = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)


def _load_promoted_base_dit_candidate(
    wrapper,
    checkpoint: Path,
    *,
    selection_summary: Mapping[str, Any],
    selection_value: Mapping[str, Any],
    resolved_model_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the selected DiT while preserving its embedded run lineage.

    The rank-0 selection-summary audit already rehashes the complete external
    latent inventory.  Every rank repeats the cheap contract/path/hash checks
    here and supplies the exact embedded contract required by the loader.
    """

    contract_record = dict(selection_summary.get("training_run_contract") or {})
    contract_path = Path(contract_record.get("path", "")).resolve(strict=True)
    dit_run_dir = Path(
        selection_value.get("training_run", {}).get("run_dir", "")
    ).resolve(strict=True)
    contract, contract_sha = validate_dit_training_run_contract(
        contract_path,
        expected_run_dir=dit_run_dir,
        verify_latent_shards=False,
    )
    if not (
        contract_record.get("sha256") == contract_sha
        and contract_record.get("canonical_sha256")
        == _dit_contract_canonical_sha256(contract)
    ):
        raise RuntimeError("selected base DiT training contract changed")
    return _load_dit_candidate(
        wrapper,
        checkpoint,
        step=int(selection_value["selected_checkpoint_step"]),
        resolved_model_config=resolved_model_config,
        training_run_contract=contract,
        training_run_contract_path=contract_path,
        training_run_contract_sha256=contract_sha,
    )
FREE_FLOORS = {
    "token_aligned_symmetric_accuracy": 0.80,
    "ordered_source_count_exact": 0.90,
    "ordered_activity_onset_exact": 0.75,
    "ordered_activity_offset_exact": 0.75,
    "ordered_trajectory_type_exact": 0.75,
    "ordered_start_position_exact": 0.60,
    "ordered_end_position_exact": 0.60,
    "permutation_spatial_soft_score": 0.75,
    "permutation_scene_score": 0.70,
}
AUDITED_SOURCE_PATHS = JOINT_SELECTION_SOURCE_PATHS
EXPECTED_JOINT_TRAINING_SOURCE_PATHS = set(JOINT_TRAINING_SOURCE_PATHS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--validation-index", type=Path)
    parser.add_argument("--validation-index-sha256", required=True)
    parser.add_argument("--base-dit-selection", type=Path)
    parser.add_argument("--short-ar-batch-size", type=int, default=8)
    parser.add_argument("--long-ar-batch-size", type=int, default=5)
    parser.add_argument("--short-rf-batch-size", type=int, default=72)
    parser.add_argument("--long-rf-batch-size", type=int, default=48)
    parser.add_argument("--free-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_sha256() -> dict[str, str]:
    return {
        relative: sha256_file((REPO_ROOT / relative).resolve(strict=True))
        for relative in AUDITED_SOURCE_PATHS
    }


def _distributed() -> tuple[int, int, torch.device]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("joint selection requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if (
        visible != VISIBLE_GPUS
        or world_size != WORLD_SIZE
        or local_rank != rank
        or not 0 <= rank < WORLD_SIZE
        or torch.cuda.device_count() != WORLD_SIZE
    ):
        raise RuntimeError("joint selection requires exact physical GPUs 3-7")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return rank, local_rank, device


def _joint_training_audit(
    run_dir: Path,
    *,
    topology: Mapping[str, Any],
    qwen_runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[int, Path]]]:
    run_dir = run_dir.expanduser().resolve()
    identity_path = (run_dir / JOINT_RUN_IDENTITY_NAME).resolve(strict=True)
    identity = validate_joint_run_identity(run_dir, identity_path=identity_path)
    contract_path = (run_dir / "RUN_CONTRACT.json").resolve(strict=True)
    final_path = (run_dir / "FINAL.json").resolve(strict=True)
    latest_path = (run_dir / "checkpoints/LATEST.json").resolve(strict=True)
    if not (
        contract_path == run_dir / "RUN_CONTRACT.json"
        and final_path == run_dir / "FINAL.json"
        and latest_path == run_dir / "checkpoints/LATEST.json"
    ):
        raise RuntimeError("joint training artifacts escaped their run directory")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    final = json.loads(final_path.read_text(encoding="utf-8"))
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    route = dict(contract.get("latest_route") or {})
    source_semantic = dict(contract.get("source_semantic") or {})
    semantic_cache = dict(source_semantic.get("cache") or {})
    source_records = dict(contract.get("source_sha256") or {})
    if set(source_records) != EXPECTED_JOINT_TRAINING_SOURCE_PATHS or any(
        sha256_file((REPO_ROOT / relative).resolve(strict=True)) != expected
        for relative, expected in source_records.items()
    ):
        raise RuntimeError("joint training source hashes changed")
    cache_records = [
        dict(semantic_cache.get(split) or {}) for split in ("train", "validation")
    ]
    temporal_pilots = [
        dict(record.get("temporal_pilot") or {}) for record in cache_records
    ]
    if any(
        not record.get("path")
        or sha256_file(Path(record["path"]).resolve(strict=True))
        != record.get("sha256")
        or sha256_file(Path(record["marker_path"]).resolve(strict=True))
        != record.get("marker_sha256")
        for record in cache_records
    ):
        raise RuntimeError("joint training M2D cache identity changed")
    validated_pilots = [
        validate_editing_m2d_temporal_pilot(
            record.get("path", ""), expected_sha256=record.get("sha256")
        )
        for record in temporal_pilots
    ]
    if validated_pilots[0] != validated_pilots[1]:
        raise RuntimeError("joint train/validation caches used different M2D pilots")
    if not (
        contract.get("schema") == JOINT_RUN_CONTRACT_SCHEMA
        and int(contract.get("schema_version", -1))
        == JOINT_RUN_CONTRACT_SCHEMA_VERSION
        and contract.get("run_dir") == str(run_dir)
        and contract.get("run_id") == identity["run_id"]
        and contract.get("run_identity")
        == {"path": str(identity_path), "sha256": sha256_file(identity_path)}
        and contract.get("editing_ar_contract") == EDITING_AR_CONTRACT
        and contract.get("joint_dataset_contract") == EDITING_JOINT_DATASET_CONTRACT
        and route.get("editing_ar_inputs")
        == ["source_foa_latent", "raw_edit_request"]
        and route.get("editing_ar_target") == "complete_new_sceneplan"
        and route.get("old_sceneplan_input") is False
        and route.get("source_caption_model_input") is False
        and route.get("target_audio_or_latent_ar_input") is False
        and route.get("source_derived_semantic_side_input")
        == EDITING_M2D_CLAP_SIDE_INPUT
        and route.get("editing_dit_frame_input")
        == ["noisy_target_64", "new_sceneplan_256", "clean_source_foa_latent_64"]
        and route.get("ar_and_dit_share_exact_transformer_object") is True
        and int(contract.get("world_size", -1)) == WORLD_SIZE
        and contract.get("physical_gpus") == PHYSICAL_GPUS
        and contract.get("cuda_visible_devices") == VISIBLE_GPUS
        and contract.get("cuda_device_order") == "PCI_BUS_ID"
        and contract.get("gpu_topology") == dict(topology)
        and contract.get("frozen_qwen_runtime") == dict(qwen_runtime)
        and source_semantic.get("contract") == EDITING_M2D_CLAP_CONTRACT
        and source_semantic.get("mode") == "m2d_audio_caption_aux"
        and source_semantic.get("inject_frozen_m2d_audio") is True
        and source_semantic.get("align_source_caption") is True
        and source_semantic.get("audio_embedding_is_source_derived") is True
        and source_semantic.get("source_audio_view")
        == M2D_CLAP_SOURCE_AUDIO_VIEW
        and source_semantic.get("temporal_policy")
        == M2D_CLAP_TEMPORAL_POLICY
        and source_semantic.get("caption_is_training_label_only") is True
        and source_semantic.get("caption_model_input") is False
        and source_semantic.get("old_sceneplan_model_input") is False
        and source_semantic.get("new_sceneplan_or_target_information_used") is False
        and math.isclose(float(source_semantic.get("lambda_source_caption", -1.0)), 0.05)
        and math.isclose(float(source_semantic.get("source_caption_temperature", -1.0)), 0.07)
        and math.isclose(float(source_semantic.get("source_semantic_dropout", -1.0)), 0.10)
        and semantic_cache.get("contract") == EDITING_M2D_CLAP_CONTRACT
        and semantic_cache.get("mode") == "m2d_audio_caption_aux"
        and cache_records[0].get("split") == "train"
        and int(cache_records[0].get("rows", -1)) == 1_000_000
        and cache_records[0].get("source_audio_view")
        == M2D_CLAP_SOURCE_AUDIO_VIEW
        and cache_records[0].get("temporal_policy")
        == M2D_CLAP_TEMPORAL_POLICY
        and cache_records[0].get("vae_config_sha256")
        == EDITING_M2D_VAE_CONFIG_SHA256
        and cache_records[0].get("vae_checkpoint_sha256")
        == EDITING_M2D_VAE_CHECKPOINT_SHA256
        and cache_records[1].get("split") == "validation"
        and int(cache_records[1].get("rows", -1)) == EXPECTED_ROWS
        and cache_records[1].get("source_audio_view")
        == M2D_CLAP_SOURCE_AUDIO_VIEW
        and cache_records[1].get("temporal_policy")
        == M2D_CLAP_TEMPORAL_POLICY
        and cache_records[1].get("vae_config_sha256")
        == EDITING_M2D_VAE_CONFIG_SHA256
        and cache_records[1].get("vae_checkpoint_sha256")
        == EDITING_M2D_VAE_CHECKPOINT_SHA256
        and int(contract.get("max_steps", -1)) == EXPECTED_MAX_STEP
        and int(contract.get("save_every", -1)) == CHECKPOINT_EVERY
        and int(contract.get("validate_every", -1)) == 500
        and int(contract.get("validation_batches", -1)) == 32
        and int(contract.get("validation_batch_size", -1)) == 4
        and int(contract.get("train_index", {}).get("rows", -1)) == 1_000_000
        and int(contract.get("validation_index", {}).get("rows", -1)) == EXPECTED_ROWS
        and contract.get("base_checkpoint_selection", {}).get("selection_contract")
        == DIT_SELECTION_CONTRACT
        and latest.get("schema") == JOINT_LATEST_SCHEMA
        and int(latest.get("schema_version", -1)) == JOINT_LATEST_SCHEMA_VERSION
        and latest.get("run_dir") == str(run_dir)
        and latest.get("run_id") == identity["run_id"]
        and latest.get("run_contract_path") == str(contract_path)
        and latest.get("run_contract_sha256") == sha256_file(contract_path)
        and final.get("schema") == JOINT_FINAL_SCHEMA
        and int(final.get("schema_version", -1)) == JOINT_FINAL_SCHEMA_VERSION
        and final.get("event") == "complete"
        and int(final.get("step", -1)) == EXPECTED_MAX_STEP
        and final.get("run_dir") == str(run_dir)
        and final.get("run_id") == identity["run_id"]
        and final.get("run_contract_path") == str(contract_path)
        and final.get("run_contract_sha256") == sha256_file(contract_path)
        and Path(final.get("checkpoint", "")).resolve(strict=True)
        == (run_dir / "checkpoints" / f"step-{EXPECTED_MAX_STEP:08d}.pt").resolve()
        and final.get("checkpoint_sha256")
        == latest.get("checkpoint_sha256")
        and final.get("shared_transformer_same_object") is True
        and final.get("old_sceneplan_exposed") is False
        and final.get("source_semantic_mode") == "m2d_audio_caption_aux"
        and final.get("source_caption_exposed_as_model_input") is False
    ):
        raise RuntimeError("joint training completion contract is invalid")
    for split, cache_record, index_record in zip(
        ("train", "validation"),
        cache_records,
        (dict(contract["train_index"]), dict(contract["validation_index"])),
    ):
        cache = ScenePlanTransfusionEditingM2DCLAPCache(
            cache_record["path"],
            source_index=Path(index_record["path"]).resolve(strict=True),
            source_index_sha256=str(index_record["sha256"]),
            expected_rows=int(index_record["rows"]),
            expected_split=split,
            expected_cache_sha256=str(cache_record["sha256"]),
            verify_cache_file_hash=False,
        )
        parity = validate_editing_m2d_cache_online_parity(
            editing_m2d_cache_online_parity_path(cache.path), cache=cache
        )
        if parity != cache_record.get("cache_online_parity"):
            raise RuntimeError(
                f"joint {split} M2D cache/online parity evidence changed"
            )
    checkpoint_dir = (run_dir / "checkpoints").resolve(strict=True)
    if checkpoint_dir != run_dir / "checkpoints":
        raise RuntimeError("joint checkpoint directory escaped its run")
    candidates = [
        (step, (checkpoint_dir / f"step-{step:08d}.pt").resolve(strict=True))
        for step in CANDIDATE_STEPS
    ]
    named_steps = sorted(checkpoint_dir.glob("step-*.pt"))
    if named_steps != [path for _, path in candidates]:
        raise RuntimeError("joint checkpoint candidate set is incomplete or ambiguous")
    final_checkpoint = candidates[-1][1]
    if not (
        Path(latest.get("checkpoint", "")).resolve(strict=True) == final_checkpoint
        and int(latest.get("global_step", -1)) == EXPECTED_MAX_STEP
        and latest.get("checkpoint_sha256") == sha256_file(final_checkpoint)
    ):
        raise RuntimeError("joint LATEST does not identify the completed checkpoint")
    return {
        "run_dir": str(run_dir),
        "run_id": identity["run_id"],
        "run_identity_path": str(identity_path),
        "run_identity_sha256": sha256_file(identity_path),
        "run_contract_path": str(contract_path),
        "run_contract_sha256": sha256_file(contract_path),
        "final_path": str(final_path),
        "final_sha256": sha256_file(final_path),
        "latest_path": str(latest_path),
        "latest_sha256": sha256_file(latest_path),
        "run_contract": contract,
        "final": final,
    }, candidates


def _make_joint_loader(
    dataset: ScenePlanTransfusionEditingJointDataset,
    *,
    short_batch_size: int,
    long_batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, dict[str, Any]]:
    buckets = dataset.length_bucket_indices()
    batches = [
        batch
        for bucket, size in ((432, short_batch_size), (648, long_batch_size))
        for batch in _chunks(buckets[bucket], size)
    ]
    covered = [index for batch in batches for index in batch]
    if sorted(covered) != list(range(len(dataset))):
        raise RuntimeError("joint validation loader dropped or duplicated rows")
    kwargs: dict[str, Any] = {
        "batch_sampler": batches,
        "collate_fn": functools.partial(collate_editing_joint, pad_id=dataset.codec.pad_id),
        "num_workers": int(num_workers),
        "pin_memory": True,
        "persistent_workers": int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs), {
        "batches": len(batches),
        "short_batch_size": int(short_batch_size),
        "long_batch_size": int(long_batch_size),
        "drop_last": False,
    }


class _JointDonorResolver:
    def __init__(
        self,
        dataset: ScenePlanTransfusionEditingJointDataset,
        assigned: Sequence[int],
        donor_by_ordinal: Mapping[int, int],
    ) -> None:
        self.dataset = dataset
        self.local_by_ordinal = {
            int(ordinal): index for index, ordinal in enumerate(assigned)
        }
        self.donor_by_ordinal = dict(donor_by_ordinal)

    def _donor_row(
        self, row: Mapping[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        donor = self.donor_by_ordinal[int(row["pair_ordinal"])]
        return self.dataset[self.local_by_ordinal[donor]]

    def values(
        self, metadata: Sequence[Mapping[str, Any]], frames: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sources = []
        masks = []
        for row in metadata:
            _, donor_metadata, _ = self._donor_row(row)
            sources.append(donor_metadata["source_foa_latent"][:, :frames])
            masks.append(donor_metadata["padding_mask"][0][:frames])
        return (
            torch.stack(sources).to(device=device, dtype=torch.float32, non_blocking=True),
            torch.stack(masks).to(device=device, dtype=torch.bool, non_blocking=True),
        )

    def rf_values(
        self, metadata: Sequence[Mapping[str, Any]], frames: int, device: torch.device
    ) -> torch.Tensor:
        return self.values(metadata, frames, device)[0]

    def ar_values(
        self,
        metadata: Sequence[Mapping[str, Any]],
        frames: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sources = []
        masks = []
        audio_values = []
        caption_values = []
        for row in metadata:
            _, donor_metadata, donor_ar = self._donor_row(row)
            audio = donor_ar.get("source_m2d_audio_embedding")
            caption = donor_ar.get("source_caption_m2d_embedding")
            if not isinstance(audio, torch.Tensor) or not isinstance(
                caption, torch.Tensor
            ):
                raise RuntimeError("formal M2D/caption intervention donor is missing")
            sources.append(donor_metadata["source_foa_latent"][:, :frames])
            masks.append(donor_metadata["padding_mask"][0][:frames])
            audio_values.append(audio)
            caption_values.append(caption)
        return (
            torch.stack(sources).to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            torch.stack(masks).to(
                device=device, dtype=torch.bool, non_blocking=True
            ),
            torch.stack(audio_values).to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
            torch.stack(caption_values).to(
                device=device, dtype=torch.float32, non_blocking=True
            ),
        )


@torch.no_grad()
def _evaluate_ar(
    ar: ScenePlanTransfusionEditingAR,
    loader: DataLoader,
    *,
    device: torch.device,
    variants: Sequence[str],
    donor_resolver: _JointDonorResolver | None = None,
) -> dict[str, Any]:
    allowed_variants = {
        "clean",
        "zero",
        "shuffled",
        "latent_zero",
        "latent_shuffled",
        "m2d_zero",
        "m2d_shuffled",
    }
    if (
        not variants
        or variants[0] != "clean"
        or not set(variants) <= allowed_variants
        or len(set(variants)) != len(variants)
    ):
        raise ValueError("invalid AR source variants")
    inject_m2d = bool(ar.source_semantic_bridge.inject_audio)
    align_caption = bool(ar.source_semantic_bridge.align_caption)
    if not inject_m2d and set(variants) & {"m2d_zero", "m2d_shuffled"}:
        raise ValueError("M2D interventions require an M2D-audio Editing AR")
    needs_donor = bool(
        set(variants) & {"shuffled", "latent_shuffled", "m2d_shuffled"}
    )
    if needs_donor and donor_resolver is None:
        raise RuntimeError("formal shuffled AR requires a fixed donor mapping")
    losses: dict[str, list[torch.Tensor]] = {name: [] for name in variants}
    responses: dict[str, list[torch.Tensor]] = {
        name: [] for name in variants if name != "clean"
    }
    accuracies: list[torch.Tensor] = []
    exact: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    ordinals: list[int] = []
    operations: list[str] = []
    buckets: list[int] = []
    source_counts: list[int] = []
    target_counts: list[int] = []
    caption_cosines: dict[str, list[torch.Tensor]] = (
        {"matched": [], "shuffled": []} if align_caption else {}
    )
    ar.eval().requires_grad_(False)
    for batch in loader:
        ar_batch = batch["ar"]
        metadata = list(batch["metadata"])
        _reject_old_plan_metadata(metadata)
        for key in (
            "source_foa_latent",
            "source_attention_mask",
            "plan_input_ids",
            "plan_labels",
            "plan_attention_mask",
        ):
            ar_batch[key] = ar_batch[key].to(device, non_blocking=True)
        source = ar_batch["source_foa_latent"].to(torch.float32)
        source_mask = ar_batch["source_attention_mask"].to(torch.bool)
        clean_semantic = None
        if inject_m2d:
            clean_semantic = ar_batch.get("source_m2d_audio_embedding")
            if not isinstance(clean_semantic, torch.Tensor):
                raise RuntimeError("formal Editing AR is missing frozen M2D audio")
            clean_semantic = clean_semantic.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        donor_source = donor_mask = donor_semantic = donor_caption = None
        if needs_donor:
            assert donor_resolver is not None
            (
                donor_source,
                donor_mask,
                donor_semantic,
                donor_caption,
            ) = donor_resolver.ar_values(
                metadata, int(source.shape[-1]), device
            )
        values: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
            "clean": (source, source_mask),
            "zero": (torch.zeros_like(source), source_mask),
            "latent_zero": (torch.zeros_like(source), source_mask),
            "m2d_zero": (source, source_mask),
            "m2d_shuffled": (source, source_mask),
        }
        if donor_source is not None and donor_mask is not None:
            values["shuffled"] = (donor_source, donor_mask)
            values["latent_shuffled"] = (donor_source, donor_mask)
        context, context_mask = ar.encode_edit_instructions(
            ar_batch["raw_edit_requests"], device=device
        )
        logits: dict[str, torch.Tensor] = {}
        clean_query = None
        for name in variants:
            semantic_kwargs: dict[str, torch.Tensor] = {}
            if inject_m2d:
                assert clean_semantic is not None
                if name in {"zero", "m2d_zero"}:
                    semantic_kwargs = {
                        "source_m2d_audio_embedding": clean_semantic,
                        "source_m2d_audio_keep_mask": torch.zeros(
                            int(source.shape[0]), device=device, dtype=torch.bool
                        ),
                    }
                elif name in {"shuffled", "m2d_shuffled"}:
                    assert donor_semantic is not None
                    semantic_kwargs = {
                        "source_m2d_audio_embedding": donor_semantic
                    }
                else:
                    semantic_kwargs = {
                        "source_m2d_audio_embedding": clean_semantic
                    }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = ar(
                    values[name][0],
                    values[name][1],
                    ar_batch["plan_input_ids"],
                    ar_batch["plan_attention_mask"],
                    context,
                    context_mask,
                    return_source_contrastive_query=(
                        name == "clean" and align_caption
                    ),
                    **semantic_kwargs,
                )
            if name == "clean" and align_caption:
                if not isinstance(output, tuple) or len(output) != 2:
                    raise RuntimeError("Editing AR omitted source-caption query")
                clean_logits, clean_query = output
                logits[name] = clean_logits.float()
            else:
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError("Editing AR returned an unexpected auxiliary")
                logits[name] = output.float()
        if align_caption:
            if clean_query is None:
                raise RuntimeError("source-caption query was not evaluated")
            caption_target = ar_batch.get("source_caption_m2d_embedding")
            if not isinstance(caption_target, torch.Tensor):
                raise RuntimeError("source-caption evaluation target is missing")
            caption_target = F.normalize(
                caption_target.to(device=device, dtype=torch.float32), dim=-1
            )
            caption_cosines["matched"].append(
                (clean_query.float() * caption_target).sum(dim=-1).cpu()
            )
            if donor_resolver is not None:
                if donor_caption is None:
                    raise RuntimeError("source-caption donor was not resolved")
                donor_caption = F.normalize(
                    donor_caption, dim=-1
                )
                caption_cosines["shuffled"].append(
                    (clean_query.float() * donor_caption).sum(dim=-1).cpu()
                )
        labels = ar_batch["plan_labels"]
        valid = labels.ne(-100)
        row_tokens = valid.sum(dim=1)
        for name in variants:
            token_loss = F.cross_entropy(
                logits[name].flatten(0, 1),
                labels.flatten(),
                ignore_index=-100,
                reduction="none",
            ).reshape_as(labels)
            losses[name].append(
                (token_loss * valid).sum(dim=1) / row_tokens.clamp_min(1)
            )
        prediction = logits["clean"].argmax(dim=-1)
        accuracies.append(
            ((prediction.eq(labels) & valid).sum(dim=1) / row_tokens.clamp_min(1)).cpu()
        )
        exact.append(((prediction.eq(labels) & valid) | ~valid).all(dim=1).float().cpu())
        tokens.append(row_tokens.cpu())
        for name in responses:
            response = (logits["clean"] - logits[name]).abs().mean(dim=-1)
            responses[name].append(
                ((response * valid).sum(dim=1) / row_tokens.clamp_min(1)).cpu()
            )
        for name in variants:
            losses[name][-1] = losses[name][-1].cpu()
        ordinals.extend(int(row["pair_ordinal"]) for row in metadata)
        operations.extend(str(row["operation"]) for row in metadata)
        buckets.extend(int(row["latent_bucket_frames"]) for row in metadata)
        for row in metadata:
            target_count = len(row["model_sceneplan"]["sources"])
            operation = str(row["operation"])
            source_count = (
                target_count - 1
                if operation == "event_addition"
                else target_count + 1
                if operation == "event_removal"
                else target_count
            )
            source_counts.append(source_count)
            target_counts.append(target_count)
    output = {
        "ordinals": ordinals,
        "operations": operations,
        "buckets": buckets,
        "source_counts": source_counts,
        "target_counts": target_counts,
        "tokens": torch.cat(tokens),
        "accuracy": torch.cat(accuracies),
        "exact": torch.cat(exact),
        "losses": {name: torch.cat(rows) for name, rows in losses.items()},
        "response_l1": {name: torch.cat(rows) for name, rows in responses.items()},
        "caption_cosine": {
            name: torch.cat(rows)
            for name, rows in caption_cosines.items()
            if rows
        },
    }
    for tensor in [
        output["tokens"], output["accuracy"], output["exact"],
        *output["losses"].values(), *output["response_l1"].values(),
        *output["caption_cosine"].values(),
    ]:
        if len(tensor) != len(ordinals) or not torch.isfinite(tensor.float()).all():
            raise RuntimeError("AR validation produced incomplete/non-finite rows")
    return output


def _subset_ar(
    value: Mapping[str, Any], fold_by_ordinal: Mapping[int, str], fold: str
) -> dict[str, Any]:
    selected = torch.tensor(
        [fold_by_ordinal[int(ordinal)] == fold for ordinal in value["ordinals"]],
        dtype=torch.bool,
    )
    return {
        **{
            key: [item for item, keep in zip(value[key], selected.tolist()) if keep]
            for key in ("ordinals", "operations", "buckets", "source_counts", "target_counts")
        },
        "tokens": value["tokens"][selected],
        "accuracy": value["accuracy"][selected],
        "exact": value["exact"][selected],
        "losses": {key: tensor[selected] for key, tensor in value["losses"].items()},
        "response_l1": {
            key: tensor[selected] for key, tensor in value["response_l1"].items()
        },
        "caption_cosine": {
            key: tensor[selected] for key, tensor in value["caption_cosine"].items()
        },
    }


def _weighted_stat(values: torch.Tensor, weights: torch.Tensor, device: torch.device) -> dict[str, Any]:
    values = values.to(torch.float64)
    weights = weights.to(torch.float64)
    total, weight, rows = _all_reduce(
        [float((values * weights).sum()), float(weights.sum()), float(len(values))],
        device,
    )
    return {"mean": total / max(1.0, weight), "weight": weight, "rows": int(rows)}


def _ar_summary(value: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    operations = sorted(set(value["operations"]))
    operations = _broadcast(operations, rank=dist.get_rank(), device=device)

    def group(labels: Sequence[Any], expected: Sequence[Any]) -> dict[str, Any]:
        output = {}
        for label in expected:
            selected = torch.tensor([item == label for item in labels], dtype=torch.bool)
            output[str(label)] = {
                "clean_ce": _weighted_stat(
                    value["losses"]["clean"][selected], value["tokens"][selected], device
                ),
                "token_accuracy": _weighted_stat(
                    value["accuracy"][selected], value["tokens"][selected], device
                ),
                "sequence_exact": _weighted_stat(
                    value["exact"][selected], torch.ones(int(selected.sum())), device
                ),
            }
        return output

    summary = {
        "clean_ce": _weighted_stat(value["losses"]["clean"], value["tokens"], device),
        "token_accuracy": _weighted_stat(value["accuracy"], value["tokens"], device),
        "sequence_exact": _weighted_stat(
            value["exact"], torch.ones(len(value["exact"])), device
        ),
        "by_operation": group(value["operations"], operations),
        "by_latent_bucket": group(value["buckets"], [432, 648]),
        "by_source_count": group(value["source_counts"], [1, 2, 3, 4]),
        "by_target_count": group(value["target_counts"], [1, 2, 3, 4]),
    }
    if "matched" in value.get("caption_cosine", {}):
        summary["source_caption_matched_cosine"] = _weighted_stat(
            value["caption_cosine"]["matched"],
            torch.ones(len(value["caption_cosine"]["matched"])),
            device,
        )
    return summary


def _paired_upper_stats(
    difference: torch.Tensor, baseline: torch.Tensor, device: torch.device
) -> dict[str, Any]:
    difference = difference.to(torch.float64).reshape(-1)
    baseline = baseline.to(torch.float64).reshape(-1)
    count, total, square_total, baseline_total = _all_reduce(
        [
            float(len(difference)), float(difference.sum()),
            float((difference**2).sum()), float(baseline.sum()),
        ],
        device,
    )
    mean = total / max(1.0, count)
    variance = max(0.0, (square_total - total * total / count) / (count - 1.0)) if count > 1 else 0.0
    upper = mean + NormalDist().inv_cdf(CONFIDENCE) * math.sqrt(variance / max(1.0, count))
    baseline_mean = baseline_total / max(1.0, count)
    margin = 0.01 * baseline_mean
    return {
        "rows": int(count), "mean_difference": mean,
        "one_sided_upper_confidence_bound": upper,
        "baseline_mean": baseline_mean, "relative_margin": 0.01,
        "absolute_margin": margin, "confidence": CONFIDENCE,
        "pass": upper <= margin,
    }


def _noninferiority_report(
    joint: torch.Tensor,
    baseline: torch.Tensor,
    *,
    operations: Sequence[str],
    buckets: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    joint_rows = joint.to(torch.float64).mean(dim=1)
    baseline_rows = baseline.to(torch.float64).mean(dim=1)
    difference = joint_rows - baseline_rows
    operation_names = sorted(set(operations))
    report = {
        "overall": _paired_upper_stats(difference, baseline_rows, device),
        "by_operation": {},
        "by_latent_bucket": {},
    }
    for name in operation_names:
        selected = torch.tensor([value == name for value in operations])
        report["by_operation"][name] = _paired_upper_stats(
            difference[selected], baseline_rows[selected], device
        )
    for bucket in (432, 648):
        selected = torch.tensor([value == bucket for value in buckets])
        report["by_latent_bucket"][str(bucket)] = _paired_upper_stats(
            difference[selected], baseline_rows[selected], device
        )
    joint_sum, base_sum, count = _all_reduce(
        [float(joint_rows.sum()), float(baseline_rows.sum()), float(len(joint_rows))], device
    )
    report["point_estimate_ratio"] = joint_sum / max(base_sum, 1e-30)
    report["point_estimate_ratio_limit"] = 1.005
    report["point_estimate_pass"] = report["point_estimate_ratio"] <= 1.005
    report["pass"] = (
        report["overall"]["pass"]
        and all(row["pass"] for row in report["by_operation"].values())
        and all(row["pass"] for row in report["by_latent_bucket"].values())
        and report["point_estimate_pass"]
    )
    return report


def _build_joint_model(
    resolved_config: Mapping[str, Any],
    codec: ModelScenePlanCodecV4,
    *,
    source_semantic_mode: str,
    source_semantic_dropout: float,
) -> tuple[Any, ScenePlanTransfusionEditingAR]:
    diffusion = create_model_from_config(dict(resolved_config))
    diffusion.pretransform = None
    prompt = diffusion.conditioner.conditioners["prompt"]
    ar = ScenePlanTransfusionEditingAR(
        editing_dit=diffusion.model.model,
        instruction_conditioner=prompt,
        pad_id=codec.pad_id,
        activation_checkpointing=False,
        source_semantic_mode=str(source_semantic_mode),
        source_semantic_dropout=float(source_semantic_dropout),
    )
    if ar.shared_transformer is not diffusion.model.model.transformer:
        raise RuntimeError("joint selector did not construct one shared Transformer")
    return diffusion, ar


def _load_joint_candidate(
    path: Path,
    *,
    step: int,
    run_contract: Mapping[str, Any],
    diffusion,
    ar: ScenePlanTransfusionEditingAR,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    diffusion_state = payload.get("diffusion_state_dict")
    ar_state = payload.get("editing_ar_specific_state_dict")
    expected_diffusion = diffusion.state_dict()
    expected_ar = {
        key: value for key, value in ar.state_dict().items()
        if not key.startswith("editing_dit.")
        and not key.startswith("instruction_conditioner.")
    }
    run_dir = Path(str(run_contract.get("run_dir", ""))).resolve()
    run_contract_path = (run_dir / "RUN_CONTRACT.json").resolve(strict=True)
    rng_states = payload.get("rng_states_by_rank")
    if not (
        payload.get("schema") == JOINT_CHECKPOINT_SCHEMA
        and int(payload.get("schema_version", -1))
        == JOINT_CHECKPOINT_SCHEMA_VERSION
        and payload.get("contract") == EDITING_AR_CONTRACT
        and payload.get("joint_dataset_contract") == EDITING_JOINT_DATASET_CONTRACT
        and payload.get("run_dir") == str(run_dir)
        and payload.get("run_id") == run_contract.get("run_id")
        and payload.get("run_contract_path") == str(run_contract_path)
        and payload.get("run_contract_sha256") == sha256_file(run_contract_path)
        and int(payload.get("global_step", -1)) == int(step)
        and payload.get("run_contract") == run_contract
        and isinstance(payload.get("optimizer"), dict)
        and isinstance(payload.get("scheduler"), dict)
        and isinstance(diffusion_state, dict)
        and isinstance(ar_state, dict)
        and set(diffusion_state) == set(expected_diffusion)
        and set(ar_state) == set(expected_ar)
        and all(
            tuple(diffusion_state[key].shape) == tuple(value.shape)
            and diffusion_state[key].dtype == value.dtype
            for key, value in expected_diffusion.items()
        )
        and all(
            tuple(ar_state[key].shape) == tuple(value.shape)
            and ar_state[key].dtype == value.dtype
            for key, value in expected_ar.items()
        )
    ):
        raise RuntimeError(f"joint checkpoint payload contract changed: {path}")
    validate_joint_rng_inventory(
        rng_states, world_size=int(run_contract.get("world_size", -1))
    )
    diffusion.load_state_dict(diffusion_state, strict=True)
    _load_ar_specific(ar, ar_state)
    del payload, diffusion_state, ar_state
    ar_specific_parameters = [
        ar.source_audio_type_embedding,
        ar.plan_type_embedding,
        *ar.source_audio_adapter.parameters(),
        *ar.plan_adapter.parameters(),
        *ar.source_semantic_bridge.parameters(),
    ]
    if any(
        not torch.isfinite(parameter.detach()).all()
        for parameter in [*diffusion.parameters(), *ar_specific_parameters]
    ):
        raise RuntimeError(f"joint checkpoint contains non-finite model weights: {path}")
    gc.collect()
    return {
        "global_step": int(step),
        "diffusion_tensors": len(expected_diffusion),
        "editing_ar_specific_tensors": len(expected_ar),
        "editing_ar_specific_parameters": sum(value.numel() for value in expected_ar.values()),
        "shared_transformer_same_object": ar.shared_transformer is diffusion.model.model.transformer,
    }


def _free_ordinals(
    index: Path, fold_by_ordinal: Mapping[int, str]
) -> tuple[list[int], dict[str, Any]]:
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    try:
        rows = [
            (int(ordinal), str(pair_id), str(operation), int(bucket))
            for ordinal, pair_id, operation, bucket in connection.execute(
                "SELECT pair_ordinal,pair_id,operation,latent_bucket_frames "
                "FROM pairs ORDER BY pair_id"
            )
            if fold_by_ordinal[int(ordinal)] == "holdout"
        ]
    finally:
        connection.close()
    cells: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for ordinal, pair_id, operation, bucket in rows:
        cells[(operation, bucket)].append((pair_id, ordinal))
    chosen = []
    counts = {}
    for cell in sorted(cells):
        values = sorted(cells[cell])[:FREE_PER_CELL]
        if len(values) != FREE_PER_CELL:
            raise RuntimeError(f"free AR holdout cell is too small: {cell}")
        chosen.extend(ordinal for _, ordinal in values)
        counts[f"{cell[1]}:{cell[0]}"] = len(values)
    chosen = sorted(chosen)
    if len(cells) != 10 or len(chosen) != FREE_ROWS or len(set(chosen)) != FREE_ROWS:
        raise RuntimeError("free AR rows are not exact 5x2x50 stratification")
    return chosen, {
        "rows": len(chosen),
        "per_bucket_operation_cell": FREE_PER_CELL,
        "by_bucket_operation": counts,
        "pair_ordinal_sha256": hashlib.sha256(
            "".join(f"{value}\n" for value in chosen).encode("ascii")
        ).hexdigest(),
        "source_fold": "sealed_holdout_10k",
    }


def _generate_with_fallback(
    ar: ScenePlanTransfusionEditingAR,
    codec: ModelScenePlanCodecV4,
    source: torch.Tensor,
    mask: torch.Tensor,
    instructions: Sequence[str],
    durations: Sequence[float],
    *,
    max_tokens: int,
    source_m2d_audio_embedding: torch.Tensor | None = None,
) -> list[tuple[list[int] | None, str | None]]:
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = ar.generate_batch(
                source, mask, instructions, codec=codec,
                max_plan_tokens=max_tokens, fixed_duration_sec=durations,
                source_m2d_audio_embedding=source_m2d_audio_embedding,
            )
        return [(value.tolist(), None) for value in outputs]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    torch.cuda.empty_cache()
    if len(instructions) == 1:
        return [(None, error)]
    middle = len(instructions) // 2
    return _generate_with_fallback(
        ar, codec, source[:middle], mask[:middle], instructions[:middle],
        durations[:middle], max_tokens=max_tokens,
        source_m2d_audio_embedding=(
            None
            if source_m2d_audio_embedding is None
            else source_m2d_audio_embedding[:middle]
        ),
    ) + _generate_with_fallback(
        ar, codec, source[middle:], mask[middle:], instructions[middle:],
        durations[middle:], max_tokens=max_tokens,
        source_m2d_audio_embedding=(
            None
            if source_m2d_audio_embedding is None
            else source_m2d_audio_embedding[middle:]
        ),
    )


@torch.no_grad()
def _free_ar_pass(
    ar: ScenePlanTransfusionEditingAR,
    codec: ModelScenePlanCodecV4,
    dataset: ScenePlanTransfusionEditingJointDataset,
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    records = []
    buckets = dataset.length_bucket_indices()
    for bucket in (432, 648):
        for indices in _chunks(buckets.get(bucket, ()), batch_size):
            samples = [dataset[index] for index in indices]
            batch = collate_editing_joint(samples, pad_id=codec.pad_id)
            ar_batch = batch["ar"]
            metadata = list(batch["metadata"])
            _reject_old_plan_metadata(metadata)
            source = ar_batch["source_foa_latent"].to(device=device, dtype=torch.float32)
            mask = ar_batch["source_attention_mask"].to(device=device, dtype=torch.bool)
            source_semantic = None
            if ar.source_semantic_bridge.inject_audio:
                value = ar_batch.get("source_m2d_audio_embedding")
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError("free Editing AR is missing frozen M2D audio")
                source_semantic = value.to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
            instructions = list(ar_batch["raw_edit_requests"])
            durations = [float(row["seconds_total"]) for row in metadata]
            outputs = _generate_with_fallback(
                ar,
                codec,
                source,
                mask,
                instructions,
                durations,
                max_tokens=512,
                source_m2d_audio_embedding=source_semantic,
            )
            for row_index, (tokens, error) in enumerate(outputs):
                row = metadata[row_index]
                target_tokens = samples[row_index][2]["target_token_ids"].tolist()
                record = {
                    "pair_ordinal": int(row["pair_ordinal"]),
                    "pair_id": str(row["pair_id"]),
                    "operation": str(row["operation"]),
                    "latent_bucket_frames": int(bucket),
                    "status": "generation_error" if tokens is None else "ok",
                    "error": error,
                    "metrics": {},
                }
                if tokens is not None:
                    record["predicted_token_ids"] = tokens
                    record["target_token_ids"] = target_tokens
                    record["metrics"].update(score_token_sequence(target_tokens, tokens, codec))
                    try:
                        predicted = _align_decoded_sceneplan_to_audio_duration(
                            codec.decode(tokens, sample_id=str(row["target_sample_id"])),
                            durations[row_index],
                        )
                        target = _align_decoded_sceneplan_to_audio_duration(
                            codec.decode(target_tokens, sample_id=str(row["target_sample_id"])),
                            durations[row_index],
                        )
                        record["metrics"].update(score_parsed_generation(target, predicted))
                        record["predicted_sceneplan"] = predicted
                    except Exception as exc:  # noqa: BLE001
                        record["status"] = "parse_error"
                        record["error"] = f"{type(exc).__name__}: {exc}"
                records.append(record)
    return records


def _lower_bound(values: Sequence[float], floor: float) -> dict[str, Any]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    mean = float(tensor.mean()) if len(tensor) else 0.0
    std = float(tensor.std(unbiased=True)) if len(tensor) > 1 else 0.0
    lower = mean - NormalDist().inv_cdf(CONFIDENCE) * std / math.sqrt(max(1, len(tensor)))
    return {"rows": len(tensor), "mean": mean, "confidence": CONFIDENCE,
            "one_sided_lower_confidence_bound": lower, "floor": floor,
            "pass": lower >= floor}


def _free_gate(
    records: Sequence[Mapping[str, Any]], *, bos_id: int, eos_id: int
) -> dict[str, Any]:
    structural = {
        "rows_exact": len(records) == FREE_ROWS,
        "all_generated_parsed": all(row["status"] == "ok" for row in records),
        "all_grammar_legal": all(float(row["metrics"].get("grammar_legal", 0.0)) == 1.0 for row in records),
        "all_eos_within_512": all(
            row.get("predicted_token_ids")
            and row["predicted_token_ids"][0] == int(bos_id)
            and row["predicted_token_ids"][-1] == int(eos_id)
            and len(row["predicted_token_ids"]) <= 512
            for row in records
        ),
    }
    metric_gates = {}
    cell_gates = {}
    for name, floor in FREE_FLOORS.items():
        values = [float(row["metrics"].get(name, 0.0)) for row in records]
        metric_gates[name] = _lower_bound(values, floor)
        cells = {}
        for operation in sorted({str(row["operation"]) for row in records}):
            for bucket in (432, 648):
                selected = [
                    float(row["metrics"].get(name, 0.0)) for row in records
                    if row["operation"] == operation
                    and int(row["latent_bucket_frames"]) == bucket
                ]
                cells[f"{bucket}:{operation}"] = {
                    "rows": len(selected), "mean": sum(selected) / max(1, len(selected)),
                    "floor": floor - 0.15,
                    "pass": len(selected) == FREE_PER_CELL
                    and sum(selected) / max(1, len(selected)) >= floor - 0.15,
                }
        cell_gates[name] = cells
    checks = {
        **structural,
        "all_metric_lower_bounds": all(row["pass"] for row in metric_gates.values()),
        "all_bucket_operation_floors": all(
            row["pass"] for values in cell_gates.values() for row in values.values()
        ),
    }
    return {"pass": all(checks.values()), "checks": checks,
            "metric_lower_bounds": metric_gates, "bucket_operation": cell_gates,
            "predeclared_floors": FREE_FLOORS}


def _validate_existing_selection(
    path: Path,
    *,
    run_dir: Path,
    preflight: Path,
    model_config: Path,
    codec_path: Path,
    codec_fingerprint: str,
    validation_index: Path,
    validation_sha256: str,
    layout_summary: Mapping[str, Any],
    fold_summary: Mapping[str, Any],
    dit_selection_path: Path,
    dit_selection_sha256: str,
    topology: Mapping[str, Any],
    training_audit: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    selected = Path(value.get("selected_checkpoint", "")).resolve(strict=True)
    expected_training = {
        key: item for key, item in training_audit.items() if key != "run_contract"
    }
    free_records = Path(value.get("free_ar_records_path", "")).resolve(strict=True)
    candidate_rows = list(value.get("candidates") or [])
    expected_candidate_paths = [
        (run_dir / "checkpoints" / f"step-{step:08d}.pt").resolve(strict=True)
        for step in CANDIDATE_STEPS
    ]
    current_candidate_sha256 = {
        step: sha256_file(candidate)
        for step, candidate in zip(CANDIDATE_STEPS, expected_candidate_paths)
    }

    def noninferiority_gate_valid(
        raw: Any, *, rows: int, require_pass: bool
    ) -> bool:
        try:
            overall = raw["overall"]
            by_operation = raw["by_operation"]
            by_bucket = raw["by_latent_bucket"]
            reports = [overall, *by_operation.values(), *by_bucket.values()]
            reports_valid = all(
                int(record["rows"]) > 0
                and math.isfinite(float(record["one_sided_upper_confidence_bound"]))
                and math.isfinite(float(record["absolute_margin"]))
                and bool(record["pass"])
                == (
                    float(record["one_sided_upper_confidence_bound"])
                    <= float(record["absolute_margin"])
                )
                for record in reports
            )
            derived = (
                reports_valid
                and int(overall["rows"]) == rows
                and sum(int(record["rows"]) for record in by_operation.values())
                == rows
                and sum(int(record["rows"]) for record in by_bucket.values()) == rows
                and set(by_bucket) == {"432", "648"}
                and bool(raw["point_estimate_pass"])
                == (
                    float(raw["point_estimate_ratio"])
                    <= float(raw["point_estimate_ratio_limit"])
                )
                and bool(overall["pass"])
                and all(bool(record["pass"]) for record in by_operation.values())
                and all(bool(record["pass"]) for record in by_bucket.values())
                and bool(raw["point_estimate_pass"])
            )
            return bool(raw["pass"]) == derived and (
                bool(raw["pass"]) if require_pass else True
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False

    def positive_paired_report_valid(raw: Any, *, require_all: bool) -> bool:
        try:
            reports = [
                raw["overall"],
                *raw["by_operation"].values(),
                *raw["by_latent_bucket"].values(),
            ]
            if not all(
                int(record["rows"]) > 0
                and math.isfinite(
                    float(record["one_sided_lower_confidence_bound"])
                )
                and bool(record["pass"])
                == (float(record["one_sided_lower_confidence_bound"]) > 0.0)
                for record in reports
            ):
                return False
            return (
                _paired_report_pass(raw)
                if require_all
                else bool(raw["overall"]["pass"])
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            return False

    candidates_valid = len(candidate_rows) == len(CANDIDATE_STEPS)
    for step, expected_path, record in zip(
        CANDIDATE_STEPS, expected_candidate_paths, candidate_rows
    ):
        try:
            metadata = dict(record["checkpoint_metadata"])
            screening = record["selection_10k"]["base_dit_noninferiority"]
            ranking_values = (
                float(record["selection_10k"]["ar"]["clean_ce"]["mean"]),
                float(record["selection_10k"]["ar"]["token_accuracy"]["mean"]),
                float(record["selection_10k"]["rf"]["mean"]),
            )
            candidates_valid = candidates_valid and (
                int(record["step"]) == step
                and Path(record["checkpoint"]).resolve(strict=True) == expected_path
                and record["checkpoint_sha256"] == current_candidate_sha256[step]
                and int(metadata["global_step"]) == step
                and metadata.get("shared_transformer_same_object") is True
                and all(math.isfinite(item) for item in ranking_values)
                and noninferiority_gate_valid(
                    screening, rows=EXPECTED_ROWS // 2, require_pass=False
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            candidates_valid = False
    eligible = [
        record
        for record in candidate_rows
        if bool(
            record.get("selection_10k", {})
            .get("base_dit_noninferiority", {})
            .get("pass")
        )
    ]
    ranked = sorted(
        eligible,
        key=lambda record: (
            float(record["selection_10k"]["ar"]["clean_ce"]["mean"]),
            -float(record["selection_10k"]["ar"]["token_accuracy"]["mean"]),
            float(record["selection_10k"]["rf"]["mean"]),
            -int(record["step"]),
        ),
    ) if candidates_valid else []
    selected_step = int(value.get("selected_checkpoint_step", -1))
    selected_record = ranked[0] if ranked else {}

    _, current_layout = _validation_layout(
        validation_index, expected_sha256=validation_sha256
    )
    current_fold_by_ordinal, current_fold_summary = _selection_holdout_folds(
        current_layout
    )
    expected_assignments = []
    for rank_value in range(WORLD_SIZE):
        assigned, assignment = _rank_ordinals(current_layout, rank_value)
        _, donor_sha = _local_donor_mapping(current_layout, assigned)
        expected_assignments.append(
            {**assignment, "donor_mapping_sha256": donor_sha}
        )
    expected_free_ordinals, expected_free_summary = _free_ordinals(
        validation_index, current_fold_by_ordinal
    )
    free_values = json.loads(free_records.read_text(encoding="utf-8"))
    codec = ModelScenePlanCodecV4(codec_path)
    recomputed_free_gate = _free_gate(
        free_values, bos_id=codec.bos_id, eos_id=codec.eos_id
    )
    placeholders = ",".join("?" for _ in expected_free_ordinals)
    connection = sqlite3.connect(
        f"file:{validation_index}?mode=ro&immutable=1", uri=True
    )
    try:
        free_identity = {
            int(ordinal): (str(pair_id), str(operation), int(bucket))
            for ordinal, pair_id, operation, bucket in connection.execute(
                "SELECT pair_ordinal,pair_id,operation,latent_bucket_frames "
                f"FROM pairs WHERE pair_ordinal IN ({placeholders})",
                expected_free_ordinals,
            )
        }
    finally:
        connection.close()
    free_records_valid = (
        isinstance(free_values, list)
        and [int(record.get("pair_ordinal", -1)) for record in free_values]
        == expected_free_ordinals
        and len(free_identity) == FREE_ROWS
        and all(
            (
                str(record.get("pair_id")),
                str(record.get("operation")),
                int(record.get("latent_bucket_frames", -1)),
            )
            == free_identity[int(record["pair_ordinal"])]
            for record in free_values
        )
    )
    base_gate = dict(value.get("selected_base_dit_noninferiority_gate") or {})
    source_gate = dict(value.get("selected_source_intervention_gate") or {})
    try:
        source_gate_derived = (
            all(
                positive_paired_report_valid(report, require_all=True)
                for group in (
                    source_gate["ar"],
                    source_gate["ar_latent_component"],
                    source_gate["rf"],
                )
                for report in group.values()
            )
            and all(
                positive_paired_report_valid(report, require_all=False)
                for report in source_gate["ar_m2d_component"].values()
            )
            and positive_paired_report_valid(
                source_gate["source_caption_alignment"], require_all=False
            )
            and float(source_gate["confidence"]) == CONFIDENCE
            and source_gate["inference_fold"] == "sealed_holdout_10k"
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        source_gate_derived = False
    if not (
        value.get("schema") == SCHEMA
        and int(value.get("schema_version", -1)) == SCHEMA_VERSION
        and value.get("status") == "PASS"
        and value.get("selection_contract") == SELECTION_CONTRACT
        and value.get("physical_gpus") == PHYSICAL_GPUS
        and int(value.get("world_size", -1)) == WORLD_SIZE
        and value.get("cuda_visible_devices") == VISIBLE_GPUS
        and value.get("gpu_topology") == topology
        and value.get("latest_route", {}).get("editing_ar_inputs")
        == ["source_foa_latent", "raw_edit_request"]
        and value.get("latest_route", {}).get("editing_ar_target")
        == "complete_new_sceneplan"
        and value.get("latest_route", {}).get("old_sceneplan_input") is False
        and value.get("latest_route", {}).get("source_caption_model_input") is False
        and value.get("latest_route", {}).get("source_derived_semantic_side_input")
        == EDITING_M2D_CLAP_SIDE_INPUT
        and int(value.get("latest_route", {}).get("editing_dit_frame_channels", -1))
        == 384
        and value.get("latest_route", {}).get("shared_transformer_same_object")
        is True
        and value.get("training_run") == expected_training
        and value.get("source_semantic")
        == training_audit["run_contract"]["source_semantic"]
        and Path(value.get("preflight", {}).get("path", "")).resolve() == preflight
        and value.get("preflight", {}).get("sha256") == sha256_file(preflight)
        and Path(value.get("model_config", {}).get("path", "")).resolve()
        == model_config
        and value.get("model_config", {}).get("sha256")
        == sha256_file(model_config)
        and Path(value.get("codec", {}).get("path", "")).resolve() == codec_path
        and value.get("codec", {}).get("fingerprint") == codec_fingerprint
        and value.get("validation_index") == dict(layout_summary)
        and Path(value.get("validation_index", {}).get("path", "")).resolve()
        == validation_index
        and value.get("validation_index", {}).get("sha256")
        == validation_sha256
        and value.get("validation_folds") == dict(fold_summary)
        and current_fold_summary == dict(fold_summary)
        and value.get("validation_assignments") == expected_assignments
        and Path(value.get("base_dit_selection", {}).get("path", "")).resolve()
        == dit_selection_path
        and value.get("base_dit_selection", {}).get("sha256")
        == dit_selection_sha256
        and value.get("candidate_steps") == list(CANDIDATE_STEPS)
        and [int(row.get("step", -1)) for row in value.get("candidates", [])]
        == list(CANDIDATE_STEPS)
        and candidates_valid
        and value.get("selection_ranked_steps")
        == [int(record["step"]) for record in ranked]
        and selected_step in CANDIDATE_STEPS
        and int(selected_record.get("step", -1)) == selected_step
        and selected.parent == (run_dir / "checkpoints").resolve()
        and value.get("selected_checkpoint_sha256") == sha256_file(selected)
        and Path(selected_record.get("checkpoint", "")).resolve() == selected
        and selected_record.get("checkpoint_sha256")
        == value.get("selected_checkpoint_sha256")
        and value.get("selected_full_20k_teacher")
        == selected_record.get("clean_full_20k")
        and noninferiority_gate_valid(
            base_gate, rows=EXPECTED_ROWS // 2, require_pass=True
        )
        and source_gate.get("pass") is True
        and source_gate_derived
        and value.get("free_ar_selection") == expected_free_summary
        and free_records_valid
        and value.get("selected_free_ar_gate") == recomputed_free_gate
        and recomputed_free_gate.get("pass") is True
        and value.get("free_ar_records_sha256") == sha256_file(free_records)
        and value.get("source_sha256") == _source_sha256()
    ):
        raise RuntimeError("existing joint Editing selection is stale")
    return value


def validate_published_joint_selection(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Re-run the complete producer-side validation for a formal consumer."""

    path = path.expanduser().resolve(strict=True)
    if expected_sha256 is not None and sha256_file(path) != str(expected_sha256):
        raise RuntimeError("joint Editing selection SHA256 changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    run_dir = Path(value.get("training_run", {}).get("run_dir", "")).resolve(
        strict=True
    )
    preflight = Path(value.get("preflight", {}).get("path", "")).resolve(
        strict=True
    )
    model_config = Path(value.get("model_config", {}).get("path", "")).resolve(
        strict=True
    )
    codec_path = Path(value.get("codec", {}).get("path", "")).resolve(
        strict=True
    )
    validation_index = Path(
        value.get("validation_index", {}).get("path", "")
    ).resolve(strict=True)
    validation_sha = str(value.get("validation_index", {}).get("sha256", ""))
    dit_selection_path = Path(
        value.get("base_dit_selection", {}).get("path", "")
    ).resolve(strict=True)
    dit_selection_sha = str(
        value.get("base_dit_selection", {}).get("sha256", "")
    )
    topology = _gpu_topology()
    qwen_runtime = verify_frozen_qwen_runtime()
    training_audit, _ = _joint_training_audit(
        run_dir, topology=topology, qwen_runtime=qwen_runtime
    )
    layout_summary, layout = _validation_layout(
        validation_index, expected_sha256=validation_sha
    )
    _, fold_summary = _selection_holdout_folds(layout)
    validated = _validate_existing_selection(
        path,
        run_dir=run_dir,
        preflight=preflight,
        model_config=model_config,
        codec_path=codec_path,
        codec_fingerprint=ModelScenePlanCodecV4(codec_path).fingerprint,
        validation_index=validation_index,
        validation_sha256=validation_sha,
        layout_summary=layout_summary,
        fold_summary=fold_summary,
        dit_selection_path=dit_selection_path,
        dit_selection_sha256=dit_selection_sha,
        topology=topology,
        training_audit=training_audit,
    )
    if validated is None or validated != value:
        raise RuntimeError("joint Editing selection did not survive full replay")
    return validated


def main() -> int:
    args = _parse_args()
    if (
        min(args.short_ar_batch_size, args.long_ar_batch_size,
            args.short_rf_batch_size, args.long_rf_batch_size,
            args.free_batch_size) <= 0
        or args.num_workers < 0
        or int(args.seed) != 42
    ):
        raise ValueError("formal joint selection settings changed")
    rank, _, device = _distributed()
    topology = _rank0_audit(_gpu_topology, rank=rank, device=device)
    qwen_runtime = _rank0_audit(
        verify_frozen_qwen_runtime, rank=rank, device=device
    )
    torch.manual_seed(int(args.seed) + rank)
    torch.cuda.manual_seed_all(int(args.seed) + rank)
    torch.set_float32_matmul_precision("high")

    run_dir = args.run_dir.expanduser().resolve(strict=True)
    root = args.data_root.expanduser().resolve(strict=True)
    preflight = (args.preflight or root / "contracts/full_training/PREFLIGHT.json").expanduser().resolve(strict=True)
    model_config = args.model_config.expanduser().resolve(strict=True)
    codec_path = args.codec.expanduser().resolve(strict=True)
    validation_index = (args.validation_index or root / "training_index/validation.sqlite").expanduser().resolve(strict=True)
    dit_selection_path = (args.base_dit_selection or DEFAULT_DIT_RUN / "evaluation/validation_20k_checkpoint_selection/SELECTED.json").expanduser().resolve(strict=True)
    output = (args.output or run_dir / "evaluation/validation_20k_joint_checkpoint_selection/SELECTED.json").expanduser().resolve()

    def validation_identity() -> dict[str, Any]:
        validation_summary = _index_summary(
            validation_index, args.validation_index_sha256, EXPECTED_ROWS
        )
        if validation_summary["split"] != "validation":
            raise RuntimeError("joint selector requires frozen validation split")
        layout_summary, layout = _validation_layout(
            validation_index, expected_sha256=args.validation_index_sha256
        )
        _validate_preflight(
            preflight,
            model_config=model_config,
            validation_index=validation_index,
            validation_sha256=args.validation_index_sha256,
            p10_checkpoint=CANONICAL_P10.resolve(strict=True),
        )
        return {
            "validation_summary": validation_summary,
            "layout_summary": layout_summary,
            "layout": layout,
        }

    validation_identity_value = _rank0_audit(
        validation_identity, rank=rank, device=device
    )
    validation_summary = validation_identity_value["validation_summary"]
    layout_summary = validation_identity_value["layout_summary"]
    layout = validation_identity_value["layout"]
    fold_by_ordinal, fold_summary = _selection_holdout_folds(layout)
    assigned, assignment_summary = _rank_ordinals(layout, rank)
    donor_by_ordinal, donor_sha = _local_donor_mapping(layout, assigned)

    audited = _rank0_audit(
        lambda: _joint_training_audit(
            run_dir, topology=topology, qwen_runtime=qwen_runtime
        ),
        rank=rank,
        device=device,
    )
    training_audit, candidates = audited
    run_contract = training_audit["run_contract"]
    source_semantic_config = dict(run_contract["source_semantic"])
    semantic_cache_config = dict(source_semantic_config["cache"])
    validation_cache_record = dict(semantic_cache_config["validation"])
    if not (
        Path(run_contract["model_config"]).resolve() == model_config
        and run_contract["model_config_sha256"] == sha256_file(model_config)
        and Path(run_contract["codec"]).resolve() == codec_path
        and run_contract["validation_index"]["sha256"] == args.validation_index_sha256
        and Path(run_contract["base_checkpoint_selection"]["path"]).resolve() == dit_selection_path
    ):
        raise RuntimeError("joint run inputs differ from requested selector inputs")
    dit_selection_sha = sha256_file(dit_selection_path)
    base_checkpoint = Path(run_contract["base_checkpoint"]).resolve(strict=True)
    base_selection = _rank0_audit(
        lambda: _checkpoint_selection_summary(
            dit_selection_path,
            expected_sha256=dit_selection_sha,
            checkpoint=base_checkpoint,
            validation_summary=validation_summary,
            model_config=model_config,
        ),
        rank=rank,
        device=device,
    )
    if base_selection != run_contract["base_checkpoint_selection"]:
        raise RuntimeError("joint run embedded a different DiT selection summary")

    existing = _rank0_audit(
        lambda: _validate_existing_selection(
            output,
            run_dir=run_dir,
            preflight=preflight,
            model_config=model_config,
            codec_path=codec_path,
            codec_fingerprint=ModelScenePlanCodecV4(codec_path).fingerprint,
            validation_index=validation_index,
            validation_sha256=args.validation_index_sha256,
            layout_summary=layout_summary,
            fold_summary=fold_summary,
            dit_selection_path=dit_selection_path,
            dit_selection_sha256=dit_selection_sha,
            topology=topology,
            training_audit=training_audit,
        ),
        rank=rank,
        device=device,
    )
    if existing is not None:
        if rank == 0:
            print(json.dumps({"event": "joint_selection_reused", "output": str(output)}, sort_keys=True), flush=True)
        dist.barrier(); dist.destroy_process_group(); return 0

    resolved_config = load_config(model_config)
    codec = ModelScenePlanCodecV4(codec_path)
    validation_semantic_cache = ScenePlanTransfusionEditingM2DCLAPCache(
        validation_cache_record["path"],
        source_index=validation_index,
        source_index_sha256=args.validation_index_sha256,
        expected_rows=EXPECTED_ROWS,
        expected_split="validation",
        expected_cache_sha256=validation_cache_record["sha256"],
        verify_cache_file_hash=True,
    )
    tokenizer_model = create_model_from_config(resolved_config)
    tokenizer = tokenizer_model.conditioner.conditioners["prompt"].tokenizer
    base_dataset = ScenePlanTransfusionEditingDataset(
        validation_index,
        tokenizer_spec=(tokenizer, 512),
        expected_num_samples=len(assigned), index_num_samples=EXPECTED_ROWS,
        expected_index_sha256=args.validation_index_sha256,
        sample_ordinals=assigned, latent_crop_length=648,
        require_frozen=True, verify_tensor_hashes_on_access=False,
    )
    # The temporary model above exists only to supply the immutable tokenizer.
    del tokenizer_model
    gc.collect()
    joint_dataset = ScenePlanTransfusionEditingJointDataset(
        base_dataset, codec=codec, source_semantic_cache=validation_semantic_cache
    )
    ar_loader, ar_loader_summary = _make_joint_loader(
        joint_dataset, short_batch_size=args.short_ar_batch_size,
        long_batch_size=args.long_ar_batch_size, num_workers=args.num_workers,
    )
    rf_loader, rf_loader_summary = _make_loader(
        base_dataset, short_batch_size=args.short_rf_batch_size,
        long_batch_size=args.long_rf_batch_size, num_workers=args.num_workers,
    )
    donor_resolver = _JointDonorResolver(joint_dataset, assigned, donor_by_ordinal)

    base_diffusion = create_model_from_config(resolved_config)
    base_wrapper = create_training_wrapper_from_config(resolved_config, base_diffusion)
    dit_value = json.loads(dit_selection_path.read_text(encoding="utf-8"))
    _load_promoted_base_dit_candidate(
        base_wrapper,
        base_checkpoint,
        selection_summary=base_selection,
        selection_value=dit_value,
        resolved_model_config=resolved_config,
    )
    base_route = base_wrapper.diffusion_ema.ema_model.to(device)
    base_diffusion.conditioner.to(device)
    base_raw = _evaluate_route(
        base_route, base_diffusion, rf_loader, device=device, rank=rank,
        seed=args.seed, timesteps=DEFAULT_TIMESTEPS, source_variants=("clean",),
        conditioner_context=base_wrapper.ema_conditioner_context(),
    )
    if base_raw["ordinals"] != assigned:
        raise RuntimeError("base DiT reproduction order changed")
    base_summary = _matrix_summary(
        base_raw["losses"]["clean"], operations=base_raw["operations"],
        buckets=base_raw["buckets"], timesteps=DEFAULT_TIMESTEPS, device=device,
    )
    expected_base_mean = float(dit_value["selected_clean_source_rf"]["mean"])
    if not math.isclose(base_summary["mean"], expected_base_mean, rel_tol=1e-6, abs_tol=1e-8):
        raise RuntimeError("base DiT full-20K reproduction differs from its selection")
    base_route.to("cpu"); base_diffusion.conditioner.to("cpu")
    del base_route, base_wrapper, base_diffusion
    gc.collect(); torch.cuda.empty_cache()

    diffusion, ar = _build_joint_model(
        resolved_config,
        codec,
        source_semantic_mode=str(source_semantic_config["mode"]),
        source_semantic_dropout=float(
            source_semantic_config["source_semantic_dropout"]
        ),
    )
    diffusion.to(device); ar.to(device)
    candidate_records = []
    local_ar: dict[int, dict[str, Any]] = {}
    local_rf: dict[int, dict[str, Any]] = {}
    checkpoint_shas = {}
    for step, checkpoint in candidates:
        checkpoint_sha = _broadcast(
            sha256_file(checkpoint) if rank == 0 else None, rank=rank, device=device
        )
        metadata = _load_joint_candidate(
            checkpoint, step=step, run_contract=run_contract,
            diffusion=diffusion, ar=ar,
        )
        ar_raw = _evaluate_ar(ar, ar_loader, device=device, variants=("clean",))
        rf_raw = _evaluate_route(
            diffusion.model, diffusion, rf_loader, device=device, rank=rank,
            seed=args.seed, timesteps=DEFAULT_TIMESTEPS, source_variants=("clean",),
            conditioner_context=torch.no_grad(),
        )
        if ar_raw["ordinals"] != assigned or rf_raw["ordinals"] != assigned:
            raise RuntimeError(f"joint candidate {step} validation order changed")
        ar_folds = {fold: _subset_ar(ar_raw, fold_by_ordinal, fold) for fold in ("selection", "holdout")}
        rf_folds = {fold: _subset_evaluation(rf_raw, fold_by_ordinal, fold) for fold in ("selection", "holdout")}
        base_selection_raw = _subset_evaluation(base_raw, fold_by_ordinal, "selection")
        screening = _noninferiority_report(
            rf_folds["selection"]["losses"]["clean"],
            base_selection_raw["losses"]["clean"],
            operations=rf_folds["selection"]["operations"],
            buckets=rf_folds["selection"]["buckets"], device=device,
        )
        record = {
            "step": step, "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha, "checkpoint_metadata": metadata,
            "clean_full_20k": {
                "ar": _ar_summary(ar_raw, device),
                "rf": _matrix_summary(
                    rf_raw["losses"]["clean"], operations=rf_raw["operations"],
                    buckets=rf_raw["buckets"], timesteps=DEFAULT_TIMESTEPS, device=device,
                ),
            },
            "selection_10k": {
                "ar": _ar_summary(ar_folds["selection"], device),
                "rf": _matrix_summary(
                    rf_folds["selection"]["losses"]["clean"],
                    operations=rf_folds["selection"]["operations"],
                    buckets=rf_folds["selection"]["buckets"],
                    timesteps=DEFAULT_TIMESTEPS, device=device,
                ),
                "base_dit_noninferiority": screening,
            },
        }
        candidate_records.append(record)
        local_ar[step] = ar_raw; local_rf[step] = rf_raw
        checkpoint_shas[step] = checkpoint_sha
        if rank == 0:
            print(json.dumps({"event": "joint_candidate_complete", "step": step,
                              "selection_ar_ce": record["selection_10k"]["ar"]["clean_ce"]["mean"],
                              "selection_rf_mse": record["selection_10k"]["rf"]["mean"],
                              "screening_pass": screening["pass"]}, sort_keys=True), flush=True)

    eligible = [row for row in candidate_records if row["selection_10k"]["base_dit_noninferiority"]["pass"]]
    ranked = sorted(
        eligible,
        key=lambda row: (
            float(row["selection_10k"]["ar"]["clean_ce"]["mean"]),
            -float(row["selection_10k"]["ar"]["token_accuracy"]["mean"]),
            float(row["selection_10k"]["rf"]["mean"]), -int(row["step"]),
        ),
    )
    if not ranked:
        raise RuntimeError("no joint candidate passed selection-fold DiT screening")
    selected_record = ranked[0]
    selected_step = int(selected_record["step"])
    selected_checkpoint = Path(selected_record["checkpoint"]).resolve(strict=True)
    _load_joint_candidate(
        selected_checkpoint, step=selected_step, run_contract=run_contract,
        diffusion=diffusion, ar=ar,
    )
    ar_intervention = _evaluate_ar(
        ar,
        ar_loader,
        device=device,
        variants=(
            "clean",
            "zero",
            "shuffled",
            "latent_zero",
            "latent_shuffled",
            "m2d_zero",
            "m2d_shuffled",
        ),
        donor_resolver=donor_resolver,
    )
    rf_intervention = _evaluate_route(
        diffusion.model, diffusion, rf_loader, device=device, rank=rank,
        seed=args.seed, timesteps=DEFAULT_TIMESTEPS,
        source_variants=("clean", "zero", "shuffled"),
        conditioner_context=torch.no_grad(),
        shuffled_source_provider=donor_resolver.rf_values,
    )
    if (
        not torch.equal(ar_intervention["losses"]["clean"], local_ar[selected_step]["losses"]["clean"])
        or not torch.equal(rf_intervention["losses"]["clean"], local_rf[selected_step]["losses"]["clean"])
    ):
        raise RuntimeError("joint selected-candidate replay is not deterministic")
    ar_holdout = _subset_ar(ar_intervention, fold_by_ordinal, "holdout")
    rf_holdout = _subset_evaluation(rf_intervention, fold_by_ordinal, "holdout")
    base_holdout = _subset_evaluation(base_raw, fold_by_ordinal, "holdout")
    base_gate = _noninferiority_report(
        rf_holdout["losses"]["clean"], base_holdout["losses"]["clean"],
        operations=rf_holdout["operations"], buckets=rf_holdout["buckets"], device=device,
    )

    def paired(values: torch.Tensor, labels: Mapping[str, Any]) -> dict[str, Any]:
        return _paired_report(
            values[:, None], operations=labels["operations"],
            buckets=labels["buckets"], device=device,
        )

    ar_gates = {
        "zero_minus_clean_ce": paired(ar_holdout["losses"]["zero"] - ar_holdout["losses"]["clean"], ar_holdout),
        "shuffle_minus_clean_ce": paired(ar_holdout["losses"]["shuffled"] - ar_holdout["losses"]["clean"], ar_holdout),
        "clean_zero_logits_l1": paired(ar_holdout["response_l1"]["zero"], ar_holdout),
        "clean_shuffle_logits_l1": paired(ar_holdout["response_l1"]["shuffled"], ar_holdout),
    }
    latent_component_gates = {
        "zero_latent_minus_clean_ce": paired(
            ar_holdout["losses"]["latent_zero"]
            - ar_holdout["losses"]["clean"],
            ar_holdout,
        ),
        "shuffled_latent_minus_clean_ce": paired(
            ar_holdout["losses"]["latent_shuffled"]
            - ar_holdout["losses"]["clean"],
            ar_holdout,
        ),
        "clean_zero_latent_logits_l1": paired(
            ar_holdout["response_l1"]["latent_zero"], ar_holdout
        ),
        "clean_shuffled_latent_logits_l1": paired(
            ar_holdout["response_l1"]["latent_shuffled"], ar_holdout
        ),
    }
    m2d_component_gates = {
        "zero_m2d_minus_clean_ce": paired(
            ar_holdout["losses"]["m2d_zero"]
            - ar_holdout["losses"]["clean"],
            ar_holdout,
        ),
        "shuffled_m2d_minus_clean_ce": paired(
            ar_holdout["losses"]["m2d_shuffled"]
            - ar_holdout["losses"]["clean"],
            ar_holdout,
        ),
        "clean_zero_m2d_logits_l1": paired(
            ar_holdout["response_l1"]["m2d_zero"], ar_holdout
        ),
        "clean_shuffled_m2d_logits_l1": paired(
            ar_holdout["response_l1"]["m2d_shuffled"], ar_holdout
        ),
    }
    if set(ar_holdout["caption_cosine"]) != {"matched", "shuffled"}:
        raise RuntimeError("selected Editing AR omitted source-caption evidence")
    source_caption_gate = paired(
        ar_holdout["caption_cosine"]["matched"]
        - ar_holdout["caption_cosine"]["shuffled"],
        ar_holdout,
    )
    rf_gates = {
        "zero_minus_clean_mse": _paired_report(
            rf_holdout["losses"]["zero"] - rf_holdout["losses"]["clean"],
            operations=rf_holdout["operations"], buckets=rf_holdout["buckets"], device=device,
        ),
        "shuffle_minus_clean_mse": _paired_report(
            rf_holdout["losses"]["shuffled"] - rf_holdout["losses"]["clean"],
            operations=rf_holdout["operations"], buckets=rf_holdout["buckets"], device=device,
        ),
        "clean_zero_prediction_l1": _paired_report(
            rf_holdout["prediction_l1"]["zero"], operations=rf_holdout["operations"],
            buckets=rf_holdout["buckets"], device=device,
        ),
        "clean_shuffle_prediction_l1": _paired_report(
            rf_holdout["prediction_l1"]["shuffled"], operations=rf_holdout["operations"],
            buckets=rf_holdout["buckets"], device=device,
        ),
    }
    source_gate = {
        "ar": ar_gates,
        "ar_latent_component": latent_component_gates,
        "ar_m2d_component": m2d_component_gates,
        "source_caption_alignment": source_caption_gate,
        "rf": rf_gates,
        "pass": (
            all(
                _paired_report_pass(value)
                for value in [
                    *ar_gates.values(),
                    *latent_component_gates.values(),
                    *rf_gates.values(),
                ]
            )
            # M2D is a global semantic side channel, so its hard dependence
            # claim is made on the full sealed holdout.  Operation/bucket
            # slices remain visible diagnostics rather than post-hoc gates.
            and all(
                bool(value["overall"]["pass"])
                for value in m2d_component_gates.values()
            )
            and bool(source_caption_gate["overall"]["pass"])
        ),
        "confidence": CONFIDENCE, "inference_fold": "sealed_holdout_10k",
    }

    free_ordinals, free_summary = _free_ordinals(validation_index, fold_by_ordinal)
    local_free = free_ordinals[rank::WORLD_SIZE]
    free_base = ScenePlanTransfusionEditingDataset(
        validation_index, tokenizer_spec=(tokenizer, 512),
        expected_num_samples=len(local_free), index_num_samples=EXPECTED_ROWS,
        expected_index_sha256=args.validation_index_sha256,
        sample_ordinals=local_free, latent_crop_length=648,
        require_frozen=True, verify_tensor_hashes_on_access=False,
    )
    free_dataset = ScenePlanTransfusionEditingJointDataset(
        free_base, codec=codec, source_semantic_cache=validation_semantic_cache
    )
    local_records = _free_ar_pass(
        ar, codec, free_dataset, device=device, batch_size=args.free_batch_size
    )
    shard_dir = output.parent / "free_ar_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"rank-{rank}.json"
    _atomic_json(shard_path, {"rank": rank, "records": local_records})
    dist.barrier()

    assignment_dir = output.parent / "rank_assignments"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(assignment_dir / f"rank-{rank}.json", {
        **assignment_summary, "donor_mapping_sha256": donor_sha,
        "ar_loader": ar_loader_summary, "rf_loader": rf_loader_summary,
    })
    dist.barrier()
    if rank == 0:
        all_free = sorted(
            [row for index in range(WORLD_SIZE) for row in json.loads(
                (shard_dir / f"rank-{index}.json").read_text(encoding="utf-8")
            )["records"]],
            key=lambda row: int(row["pair_ordinal"]),
        )
        if [int(row["pair_ordinal"]) for row in all_free] != free_ordinals:
            raise RuntimeError("free AR shards do not exactly cover frozen rows")
        free_gate = _free_gate(all_free, bos_id=codec.bos_id, eos_id=codec.eos_id)
        free_records_path = output.parent / "free_ar_records.json"
        _atomic_json(free_records_path, all_free)
        assignments = [
            json.loads((assignment_dir / f"rank-{index}.json").read_text(encoding="utf-8"))
            for index in range(WORLD_SIZE)
        ]
        final_checks = {
            "base_dit_holdout_noninferiority": bool(base_gate["pass"]),
            "ar_and_rf_source_dependence": bool(source_gate["pass"]),
            "free_ar_quality": bool(free_gate["pass"]),
        }
        if not all(final_checks.values()):
            failure = {
                "schema": SCHEMA, "schema_version": SCHEMA_VERSION,
                "status": "FAIL", "selection_contract": SELECTION_CONTRACT,
                "selected_candidate_step": selected_step,
                "checks": final_checks, "base_dit_gate": base_gate,
                "source_intervention_gate": source_gate, "free_ar_gate": free_gate,
            }
            _atomic_json(output.parent / "FAILED.json", failure)
            raise RuntimeError("top joint candidate failed sealed promotion gates")
        selection = {
            "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "status": "PASS",
            "selection_contract": SELECTION_CONTRACT,
            "latest_route": {
                "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
                "editing_ar_target": "complete_new_sceneplan",
                "old_sceneplan_input": False,
                "source_caption_model_input": False,
                "target_audio_or_latent_ar_input": False,
                "source_derived_semantic_side_input": (
                    EDITING_M2D_CLAP_SIDE_INPUT
                ),
                "editing_dit_frame_input": [
                    "noisy_target_64", "new_sceneplan_256", "clean_source_foa_latent_64"
                ],
                "editing_dit_frame_channels": 384,
                "shared_transformer_same_object": True,
            },
            "physical_gpus": PHYSICAL_GPUS, "world_size": WORLD_SIZE,
            "cuda_visible_devices": VISIBLE_GPUS, "gpu_topology": topology,
            "training_run": {key: value for key, value in training_audit.items() if key != "run_contract"},
            "source_semantic": source_semantic_config,
            "preflight": {"path": str(preflight), "sha256": sha256_file(preflight)},
            "model_config": {"path": str(model_config), "sha256": sha256_file(model_config)},
            "codec": {"path": str(codec_path), "fingerprint": codec.fingerprint},
            "validation_index": layout_summary, "validation_folds": fold_summary,
            "validation_assignments": assignments,
            "base_dit_selection": {**base_selection, "path": str(dit_selection_path), "sha256": dit_selection_sha},
            "base_dit_reproduction": base_summary,
            "evaluation_contract": {
                "all_candidates_clean_rows": EXPECTED_ROWS,
                "ranking_fold": "selection_10k",
                "promotion_fold": "sealed_holdout_10k",
                "holdout_candidates_tested": 1,
                "rf_timesteps": list(DEFAULT_TIMESTEPS),
                "base_dit_relative_noninferiority_margin": 0.01,
                "base_dit_point_ratio_limit": 1.005,
                "source_intervention_confidence": CONFIDENCE,
                "component_interventions": [
                    "latent_zero",
                    "latent_shuffled",
                    "m2d_zero",
                    "m2d_shuffled",
                    "source_caption_matched_vs_shuffled",
                ],
                "free_ar_rows": FREE_ROWS,
                "free_ar_max_tokens": 512,
            },
            "candidate_steps": list(CANDIDATE_STEPS), "candidates": candidate_records,
            "selection_ranked_steps": [int(row["step"]) for row in ranked],
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": checkpoint_shas[selected_step],
            "selected_checkpoint_step": selected_step,
            "selected_full_20k_teacher": selected_record["clean_full_20k"],
            "selected_base_dit_noninferiority_gate": base_gate,
            "selected_source_intervention_gate": source_gate,
            "selected_free_ar_gate": free_gate,
            "free_ar_selection": free_summary,
            "free_ar_records_path": str(free_records_path.resolve()),
            "free_ar_records_sha256": sha256_file(free_records_path),
            "source_sha256": _source_sha256(),
        }
        _atomic_json(output, selection)
        print(json.dumps({"event": "joint_selection_complete", "output": str(output),
                          "selected_step": selected_step,
                          "selected_checkpoint": str(selected_checkpoint)}, sort_keys=True), flush=True)
    dist.barrier(); dist.destroy_process_group(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
