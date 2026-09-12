#!/usr/bin/env python3
"""Independent real-FOA evaluation for the latest Transfusion Editing route.

The model-facing path is exactly::

    source FOA -> frozen VAE -> source latent + raw edit instruction -> new plan
    [noise || clean source latent] + generated new plan -> edited latent
    edited latent -> frozen VAE -> edited FOA

The old/source ScenePlan is never passed to either model.  It is decoded only
inside the offline scorer to identify edited and unchanged time spans.
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
from statistics import NormalDist
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import zlib

import pyarrow.parquet as pq
import soundfile as sf
import torch
from torch import distributed as dist
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.score_sceneplan_dit_p10_core import (  # noqa: E402
    _doa_metrics,
    _paired_doa_metrics,
)
from scripts.t2a.eval.sceneplan_transfusion_editing_content_metrics import (  # noqa: E402
    INDEPENDENT_CONTENT_METRIC_CONTRACT,
    IndependentEditingContentEvaluator,
    verify_independent_content_metric_assets,
)
from scripts.t2a.eval.select_sceneplan_transfusion_editing_dit_checkpoint import (  # noqa: E402
    _gpu_topology,
    _rank0_audit,
)
from scripts.t2a.eval.select_sceneplan_transfusion_editing_joint_checkpoint import (  # noqa: E402
    validate_published_joint_selection,
)
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import (  # noqa: E402
    _index_summary,
)
from stable_audio_tools.data.model_sceneplan import (  # noqa: E402
    MODEL_SAMPLE_RATE,
    VAE_HOP_SAMPLES,
    compile_model_44_controls,
    validate_model_sceneplan,
)
from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (  # noqa: E402
    ScenePlanTransfusionEditingDataset,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
    sha256_json,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import (  # noqa: E402
    ScenePlanTransfusionEditingJointDataset,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_evaluation import (  # noqa: E402
    score_parsed_generation,
    score_token_sequence,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import (  # noqa: E402
    EDITING_M2D_CLAP_SIDE_INPUT,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_runtime import (  # noqa: E402
    verify_editing_m2d_clap_assets,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import (  # noqa: E402
    EDITING_PIPELINE_CONTRACT,
    FROZEN_VAE_CHECKPOINT,
    FROZEN_VAE_CHECKPOINT_SHA256,
    FROZEN_VAE_CONFIG,
    FROZEN_VAE_CONFIG_SHA256,
    JOINT_SELECTION_CONTRACT,
    JOINT_SELECTION_SCHEMA,
    load_sceneplan_transfusion_editing_pipeline,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_provenance import (  # noqa: E402
    verify_frozen_qwen_runtime,
)


SCHEMA = "sceneplan_transfusion_editing_audio_end_to_end"
SCHEMA_VERSION = 1
EVALUATION_CONTRACT = (
    "real_foa_m2d_free_ar_aligned_dit_independent_clap_asr_addition_phase_aware_v6"
)
CALIBRATION_SCHEMA = "sceneplan_transfusion_editing_audio_e2e_calibration"
CALIBRATION_CONTRACT = (
    "validation_1k_stratified_fixed_quality_thresholds_before_test_phase_aware_v6"
)
FINAL_SEAL_SCHEMA = "sceneplan_transfusion_editing_audio_e2e_final_seal"
BATCH_SIZE_SIDECAR_SCHEMA = (
    "sceneplan_transfusion_editing_audio_e2e_batch_size_certification"
)
FORMAL_BATCH_SIZES_PER_RANK = (1, 2, 4)
VISIBLE_GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
PHYSICAL_GPUS = [int(item) for item in VISIBLE_GPUS.split(",") if item]
WORLD_SIZE = max(len(PHYSICAL_GPUS), 1)
CONFIDENCE = 0.99
CALIBRATION_ROWS = 1_000
CALIBRATION_PER_CELL = 100
DEMIX_RIDGE = 0.05
DEMIX_MAX_CONDITION = 20.0
DEMIX_MIN_CALIBRATION_SI_SDR_DB = 0.0
DEFAULT_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_transfusion_editing_v1")
DEFAULT_MODEL_CONFIG = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json"
)
DEFAULT_CODEC = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
OPERATIONS = (
    "event_addition",
    "event_removal",
    "linear_to_static",
    "static_to_linear",
    "stationary_spatial_relocation",
)
SPATIAL_OPERATIONS = (
    "linear_to_static",
    "static_to_linear",
    "stationary_spatial_relocation",
)
REFERENCE_DERIVED_OPERATIONS = ("event_removal", *SPATIAL_OPERATIONS)
FULL_QUALITY_GROUPS = (
    "overall",
    "by_operation",
    "by_latent_bucket",
    "by_operation_bucket",
)
FORBIDDEN_MODEL_KEYS = {
    "old_sceneplan",
    "old_plan",
    "source_sceneplan",
    "source_plan",
    "previous_sceneplan",
    "previous_plan",
}
MODEL_INPUT_CONTRACT = {
    "editing_ar": ["source_foa_latent", "raw_edit_request"],
    "old_sceneplan": False,
    "editing_dit": [
        "noisy_target_64",
        "generated_new_sceneplan_256",
        "clean_source_foa_latent_64",
    ],
    "editing_dit_channels": 384,
    "independent_content_metrics_are_post_inference_only": True,
}
LATEST_ROUTE_CONTRACT = {
    "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
    "editing_ar_target": "complete_new_sceneplan",
    "old_sceneplan_input": False,
    "source_caption_model_input": False,
    "source_derived_semantic_side_input": EDITING_M2D_CLAP_SIDE_INPUT,
    "editing_dit_frame_input": [
        "noisy_target_64",
        "generated_new_sceneplan_256",
        "clean_source_foa_latent_64",
    ],
    "editing_dit_frame_channels": 384,
    "shared_transformer_same_object": True,
}
AGGREGATE_ONLY_METRICS = {"unchanged_demix_source_coverage"}
AUDITED_SOURCE_PATHS = (
    "scripts/t2a/eval/evaluate_sceneplan_transfusion_editing_audio_end_to_end.py",
    "scripts/t2a/eval/run_sceneplan_transfusion_editing_end_to_end_5gpu.sh",
    "scripts/t2a/eval/sceneplan_transfusion_editing_content_metrics.py",
    "scripts/t2a/eval/score_sceneplan_dit_p10_core.py",
    "scripts/t2a/eval/select_sceneplan_transfusion_editing_dit_checkpoint.py",
    "scripts/t2a/eval/frozen_sources/editing_dit_selector_11b57706.py.txt",
    "scripts/t2a/eval/select_sceneplan_transfusion_editing_joint_checkpoint.py",
    "scripts/t2a/test/validate_sceneplan_transfusion_editing_m2d_cache_online_parity.py",
    "scripts/t2a/train/train_sceneplan_transfusion_editing_ar_joint_full.py",
    "stable_audio_tools/configuration.py",
    "stable_audio_tools/data/foa_intensity.py",
    "stable_audio_tools/data/model_sceneplan.py",
    "stable_audio_tools/data/model_sceneplan_codec_v3.py",
    "stable_audio_tools/data/model_sceneplan_codec_v4.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_dataset.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_index.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_joint_dataset.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_plan.py",
    "stable_audio_tools/data/sceneplan_transfusion_editing_m2d_clap.py",
    "stable_audio_tools/data/sceneplan_transfusion_generation_ar_evaluation.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_pipeline.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_m2d_clap.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_m2d_runtime.py",
    "stable_audio_tools/models/sceneplan_transfusion_editing_provenance.py",
    "stable_audio_tools/models/autoencoders.py",
    "stable_audio_tools/models/bottleneck.py",
    "stable_audio_tools/models/blocks.py",
    "stable_audio_tools/models/factory.py",
    "stable_audio_tools/models/utils.py",
)

# Fixed sanity floors are declared in source before any validation or test
# output exists.  Validation freezes a tighter regression threshold, while the
# test split is read only after the calibration artifact has passed.
HIGHER_BETTER_SPECS: dict[str, dict[str, Any]] = {
    "plan_token_aligned_symmetric_accuracy": {
        "floor": 0.65,
        "tolerance": 0.03,
        "groups": FULL_QUALITY_GROUPS,
    },
    "plan_ordered_source_count_exact": {
        "floor": 0.80,
        "tolerance": 0.05,
        "groups": FULL_QUALITY_GROUPS,
    },
    "plan_permutation_scene_score": {
        "floor": 0.55,
        "tolerance": 0.05,
        "groups": FULL_QUALITY_GROUPS,
    },
    "plan_ordered_activity_onset_exact": {
        "floor": 0.50,
        "tolerance": 0.05,
        "groups": FULL_QUALITY_GROUPS,
    },
    "plan_ordered_activity_offset_exact": {
        "floor": 0.50,
        "tolerance": 0.05,
        "groups": FULL_QUALITY_GROUPS,
    },
    "plan_ordered_trajectory_type_exact": {
        "floor": 0.65,
        "tolerance": 0.05,
        "groups": FULL_QUALITY_GROUPS,
    },
    "plan_ordered_start_position_exact": {
        "floor": 0.40,
        "tolerance": 0.05,
        "groups": FULL_QUALITY_GROUPS,
    },
    "plan_ordered_end_position_exact": {
        "floor": 0.40,
        "tolerance": 0.05,
        "groups": FULL_QUALITY_GROUPS,
    },
    "latent_foa_progress": {
        "floor": 0.05,
        "tolerance": 0.02,
        "groups": ("by_operation", "by_operation_bucket"),
        "operations": REFERENCE_DERIVED_OPERATIONS,
    },
    "audio_codec_foa_progress": {
        "floor": 0.05,
        "tolerance": 0.02,
        "groups": ("by_operation", "by_operation_bucket"),
        "operations": REFERENCE_DERIVED_OPERATIONS,
    },
    # Direction-only and trajectory-only edits should preserve the
    # omnidirectional W channel.  Their source->target W error is therefore
    # zero (or numerically negligible), making normalized W progress
    # undefined by construction. New event audio is not phase-identified by a
    # plan: source+new and source-new can describe equally valid additions,
    # while waveform MSE can prefer copying source to a valid generated phase.
    # Use phase-sensitive progress only for source-derived edits; addition
    # remains gated by independent content, timing and source preservation.
    "audio_raw_w_progress": {
        "floor": 0.03,
        "tolerance": 0.02,
        "operations": ("event_removal",),
        "groups": ("by_operation", "by_operation_bucket"),
    },
    "audio_raw_foa_progress": {
        "floor": 0.05,
        "tolerance": 0.02,
        "groups": ("by_operation", "by_operation_bucket"),
        "operations": REFERENCE_DERIVED_OPERATIONS,
    },
    "change_direction_cosine_foa": {
        "floor": 0.05,
        "tolerance": 0.02,
        "groups": ("by_operation", "by_operation_bucket"),
        "operations": REFERENCE_DERIVED_OPERATIONS,
    },
    "change_temporal_iou": {
        "floor": 0.10,
        "tolerance": 0.02,
        "groups": FULL_QUALITY_GROUPS,
    },
    "change_envelope_correlation": {
        "floor": 0.10,
        "tolerance": 0.02,
        "groups": FULL_QUALITY_GROUPS,
        "min_coverage": 0.95,
    },
    "activity_temporal_iou": {
        "floor": 0.50,
        "tolerance": 0.03,
        "groups": FULL_QUALITY_GROUPS,
    },
    "activity_output_target_iou": {
        "floor": 0.50,
        "tolerance": 0.03,
        "groups": FULL_QUALITY_GROUPS,
    },
    "doa_spatial_progress": {
        "floor": 0.05,
        "tolerance": 0.02,
        "operations": SPATIAL_OPERATIONS,
        "groups": ("by_operation", "by_operation_bucket"),
        "min_coverage": 0.95,
    },
    "independent_clap_target_progress": {
        "floor": 0.02,
        "tolerance": 0.02,
        "operations": ("event_addition", "event_removal"),
        "groups": ("by_operation", "by_operation_bucket"),
        "min_coverage": 0.95,
    },
    "independent_clap_edit_text_progress": {
        "floor": 0.02,
        "tolerance": 0.03,
        "operations": ("event_addition", "event_removal"),
        "groups": ("by_operation",),
        "min_coverage": 0.95,
    },
    "removed_speech_recall_progress": {
        "floor": 0.05,
        "tolerance": 0.05,
        "operations": ("event_removal",),
        "groups": ("by_operation",),
        "min_coverage": 0.10,
    },
    "speech_addition_wer_progress": {
        "floor": 0.05,
        "tolerance": 0.05,
        "operations": ("event_addition",),
        "groups": ("by_operation",),
        "min_coverage": 0.10,
    },
    "speech_addition_cer_progress": {
        "floor": 0.05,
        "tolerance": 0.05,
        "operations": ("event_addition",),
        "groups": ("by_operation",),
        "min_coverage": 0.10,
    },
    # Aggregate eligible/total unchanged sources directly.  This is not a
    # per-row mean: every unchanged source contributes to the denominator, so
    # retaining one easy source in an otherwise ineligible row cannot hide
    # source-level abstention.
    "unchanged_demix_source_coverage": {
        "floor": 0.20,
        "tolerance": 0.05,
        "groups": ("overall", "by_operation", "by_latent_bucket"),
        "min_coverage": 0.20,
    },
    "unchanged_demix_si_sdr_delta_vs_copy_db": {
        "floor": -3.0,
        "tolerance": 1.0,
        "groups": ("overall", "by_latent_bucket"),
        "min_coverage": 0.05,
    },
}
LOWER_BETTER_SPECS: dict[str, dict[str, Any]] = {
    "latent_foa_nmse": {
        "ceiling": 1.0,
        "tolerance": 0.05,
        "groups": ("by_operation", "by_operation_bucket"),
        "operations": REFERENCE_DERIVED_OPERATIONS,
    },
    "audio_codec_foa_nmse": {
        "ceiling": 1.0,
        "tolerance": 0.05,
        "groups": ("by_operation", "by_operation_bucket"),
        "operations": REFERENCE_DERIVED_OPERATIONS,
    },
    "doa_target_mean_deg": {
        "ceiling": 45.0,
        "tolerance": 3.0,
        "operations": SPATIAL_OPERATIONS,
        "groups": ("by_operation", "by_operation_bucket"),
        "min_coverage": 0.95,
    },
    "trajectory_start_error_deg": {
        "ceiling": 60.0,
        "tolerance": 3.0,
        "operations": SPATIAL_OPERATIONS,
        "groups": ("by_operation", "by_operation_bucket"),
        "min_coverage": 0.95,
    },
    "trajectory_end_error_deg": {
        "ceiling": 60.0,
        "tolerance": 3.0,
        "operations": SPATIAL_OPERATIONS,
        "groups": ("by_operation", "by_operation_bucket"),
        "min_coverage": 0.95,
    },
    "trajectory_extent_error_deg": {
        "ceiling": 45.0,
        "tolerance": 3.0,
        "operations": ("linear_to_static", "static_to_linear"),
        "groups": ("by_operation", "by_operation_bucket"),
        "min_coverage": 0.95,
    },
    "unchanged_preservation_budget_ratio": {
        "ceiling": 1.25,
        "tolerance": 0.05,
        # Direct non-overlap masks cover at least 100 validation rows in every
        # operation and more than 100 in each length bucket.  Gate those
        # marginals, while avoiding operation x bucket cells whose legitimate
        # abstention coverage can be below 20 rows.
        "groups": ("overall", "by_operation", "by_latent_bucket"),
        "min_coverage": 0.20,
    },
    "speech_preservation_excess_wer": {
        "ceiling": 0.35,
        "tolerance": 0.05,
        "groups": ("overall", "by_operation", "by_latent_bucket"),
        "min_coverage": 0.10,
    },
    "speech_preservation_excess_cer": {
        "ceiling": 0.30,
        "tolerance": 0.05,
        "groups": ("overall", "by_operation", "by_latent_bucket"),
        "min_coverage": 0.10,
    },
}


def _demix_contract() -> dict[str, Any]:
    return {
        "ridge": DEMIX_RIDGE,
        "max_condition_p90": DEMIX_MAX_CONDITION,
        "min_target_source_calibration_si_sdr_db": (
            DEMIX_MIN_CALIBRATION_SI_SDR_DB
        ),
        "source_stems_are_offline_metric_truth_only": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "test"), required=True)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--checkpoint-selection-sha256", required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--expected-index-rows", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-sha256")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--max-plan-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-all-audio", action="store_true")
    parser.add_argument("--listening-rows-per-cell", type=int, default=2)
    return parser.parse_args()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _batch_size_sidecar_value(*, phase: str, batch_size: int) -> dict[str, Any]:
    phase = str(phase)
    batch_size = int(batch_size)
    if phase not in {"calibration", "test"}:
        raise ValueError("audio E2E batch-size sidecar phase is invalid")
    if batch_size not in FORMAL_BATCH_SIZES_PER_RANK:
        raise ValueError(
            "formal audio E2E batch size must be certified in {1,2,4}"
        )
    return {
        "schema": BATCH_SIZE_SIDECAR_SCHEMA,
        "schema_version": 1,
        "status": "CERTIFIED",
        "contract": "fixed_per_rank_batch_size_sidecar_allowlist_v1",
        "phase": phase,
        "certified_batch_sizes_per_rank": list(FORMAL_BATCH_SIZES_PER_RANK),
        "selected_batch_size_per_rank": batch_size,
        "calibration_and_test_must_match": True,
    }


def _require_matching_calibration_batch_size(
    calibration: Mapping[str, Any], *, test_batch_size: int
) -> int:
    """Fail closed unless test inference uses the calibrated batch shape."""

    calibration_batch_size = int(calibration.get("batch_size_per_rank", -1))
    test_batch_size = int(test_batch_size)
    if (
        calibration_batch_size not in FORMAL_BATCH_SIZES_PER_RANK
        or test_batch_size not in FORMAL_BATCH_SIZES_PER_RANK
        or calibration_batch_size != test_batch_size
    ):
        raise RuntimeError(
            "test audio E2E batch size differs from frozen calibration"
        )
    return calibration_batch_size


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _distributed() -> tuple[int, int, torch.device]:
    if os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise RuntimeError("audio Editing E2E requires CUDA_DEVICE_ORDER=PCI_BUS_ID")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not (
        visible == VISIBLE_GPUS
        and world_size == WORLD_SIZE
        and rank == local_rank
        and 0 <= rank < WORLD_SIZE
        and torch.cuda.device_count() == WORLD_SIZE
    ):
        raise RuntimeError("audio Editing E2E requires CUDA_VISIBLE_DEVICES to match the launched world size")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    return rank, local_rank, device


def _broadcast(value: Any, *, rank: int, device: torch.device) -> Any:
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0, device=device)
    return values[0]


def _source_sha256() -> dict[str, str]:
    return {
        relative: sha256_file((REPO_ROOT / relative).resolve(strict=True))
        for relative in AUDITED_SOURCE_PATHS
    }


def _stable_seed(base: int, pair_id: str, namespace: str) -> int:
    digest = hashlib.blake2b(
        f"{int(base)}:{pair_id}:{namespace}".encode("utf-8"),
        digest_size=8,
        person=b"spedit-e2e-v1",
    ).digest()
    return int.from_bytes(digest, "big") % (2**63 - 1)


def _ordinal_sha256(ordinals: Sequence[int]) -> str:
    return hashlib.sha256(
        "".join(f"{int(value)}\n" for value in ordinals).encode("ascii")
    ).hexdigest()


def _layout(index: Path, *, expected_rows: int, expected_split: str) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [
            dict(row)
            for row in connection.execute(
                "SELECT pair_ordinal,pair_id,operation,latent_bucket_frames,"
                "source_domain,target_domain FROM pairs ORDER BY pair_ordinal"
            )
        ]
        invalid = int(
            connection.execute(
                "SELECT COUNT(*) FROM pairs WHERE split!=? OR "
                "materialization_status!='encoded' OR target_foa_path IS NULL OR "
                "target_foa_sha256 IS NULL OR target_latent_tensor_sha256 IS NULL",
                (expected_split,),
            ).fetchone()[0]
        )
    finally:
        connection.close()
    if not (
        len(rows) == int(expected_rows)
        and invalid == 0
        and [int(row["pair_ordinal"]) for row in rows] == list(range(expected_rows))
        and set(str(row["operation"]) for row in rows) == set(OPERATIONS)
        and set(int(row["latent_bucket_frames"]) for row in rows) == {432, 648}
    ):
        raise RuntimeError("audio E2E index layout is incomplete or stale")
    return rows


def _select_ordinals(
    rows: Sequence[Mapping[str, Any]], *, phase: str
) -> tuple[list[int], dict[str, Any]]:
    if phase == "test":
        selected = [int(row["pair_ordinal"]) for row in rows]
        return selected, {
            "contract": "all_frozen_test_rows_v1",
            "rows": len(selected),
            "pair_ordinal_sha256": _ordinal_sha256(selected),
        }
    cells: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for row in rows:
        pair_id = str(row["pair_id"])
        key = (str(row["operation"]), int(row["latent_bucket_frames"]))
        order = hashlib.sha256(
            f"audio-e2e-calibration-v1:{pair_id}".encode("utf-8")
        ).hexdigest()
        cells[key].append((order, int(row["pair_ordinal"])))
    selected = []
    counts = {}
    for cell in sorted(cells):
        values = sorted(cells[cell])[:CALIBRATION_PER_CELL]
        if len(values) != CALIBRATION_PER_CELL:
            raise RuntimeError(f"calibration cell is too small: {cell}")
        selected.extend(ordinal for _, ordinal in values)
        counts[f"{cell[1]}:{cell[0]}"] = len(values)
    selected = sorted(selected)
    if len(cells) != 10 or len(selected) != CALIBRATION_ROWS:
        raise RuntimeError("audio calibration is not exact 5x2x100 stratification")
    return selected, {
        "contract": "sha256_ranked_operation_bucket_5x2x100_v1",
        "rows": len(selected),
        "per_operation_bucket_cell": CALIBRATION_PER_CELL,
        "by_operation_bucket": counts,
        "pair_ordinal_sha256": _ordinal_sha256(selected),
    }


class OfflineTruthResolver:
    """Resolve immutable waveform/scoring truth; never construct model inputs."""

    def __init__(self, index: Path) -> None:
        self.index = index
        self.connection = sqlite3.connect(
            f"file:{index}?mode=ro&immutable=1", uri=True
        )
        self.connection.row_factory = sqlite3.Row
        self.manifests: dict[str, dict[str, dict[str, Any]]] = {}
        self.verified_files: dict[str, str] = {}

    def close(self) -> None:
        self.connection.close()

    @staticmethod
    def _unpack(blob: bytes, expected_sha: str, field: str) -> dict[str, Any]:
        value = json.loads(zlib.decompress(blob))
        validate_model_sceneplan(value)
        if sha256_json(value) != str(expected_sha):
            raise RuntimeError(f"offline {field} checksum changed")
        return value

    def _manifest_row(
        self, manifest_path: str, expected_sha: str, sample_id: str
    ) -> dict[str, Any]:
        path = str(Path(manifest_path).resolve(strict=True))
        if path not in self.manifests:
            if sha256_file(Path(path)) != str(expected_sha):
                raise RuntimeError("source materialization manifest checksum changed")
            rows = pq.read_table(
                path,
                columns=[
                    "sample_id",
                    "foa_path",
                    "foa_sha256",
                    "render_result_json",
                ],
            ).to_pylist()
            mapping = {str(row["sample_id"]): row for row in rows}
            if len(mapping) != len(rows):
                raise RuntimeError("source manifest has duplicate sample ids")
            self.manifests[path] = mapping
        row = self.manifests[path].get(str(sample_id))
        if row is None:
            raise RuntimeError("source sample is absent from its frozen manifest")
        return row

    def _verify_file(self, path: str, expected_sha: str) -> Path:
        resolved = Path(path).resolve(strict=True)
        key = str(resolved)
        if self.verified_files.get(key) != str(expected_sha):
            if sha256_file(resolved) != str(expected_sha):
                raise RuntimeError(f"FOA checksum changed: {resolved}")
            self.verified_files[key] = str(expected_sha)
        return resolved

    @staticmethod
    def _read_foa(path: Path, expected_samples: int) -> torch.Tensor:
        value, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != MODEL_SAMPLE_RATE or value.shape != (int(expected_samples), 4):
            raise RuntimeError(f"FOA geometry changed: {path}")
        tensor = torch.from_numpy(value.T.copy())
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"FOA contains non-finite values: {path}")
        return tensor

    def row(self, ordinal: int) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT pair_ordinal,pair_id,source_sample_id,target_sample_id,"
            "operation,raw_edit_request,source_count,target_count,"
            "latent_bucket_frames,model_num_samples,latent_frames_valid,"
            "old_sceneplan_zlib,new_sceneplan_zlib,old_sceneplan_sha256,"
            "new_sceneplan_sha256,edited_source_ids_json,"
            "unchanged_source_ids_json,source_manifest_path,"
            "source_manifest_sha256,source_foa_sha256,target_foa_path,"
            "target_foa_sha256,source_domain,target_domain "
            "FROM pairs WHERE pair_ordinal=?",
            (int(ordinal),),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"offline truth lacks pair ordinal {ordinal}")
        value = dict(row)
        old_plan = self._unpack(
            value.pop("old_sceneplan_zlib"),
            value.pop("old_sceneplan_sha256"),
            "old ScenePlan",
        )
        new_plan = self._unpack(
            value.pop("new_sceneplan_zlib"),
            value.pop("new_sceneplan_sha256"),
            "new ScenePlan",
        )
        manifest = self._manifest_row(
            str(value["source_manifest_path"]),
            str(value["source_manifest_sha256"]),
            str(value["source_sample_id"]),
        )
        if str(manifest["foa_sha256"]) != str(value["source_foa_sha256"]):
            raise RuntimeError("source FOA hash differs between pair and manifest")
        source_path = self._verify_file(
            str(manifest["foa_path"]), str(value["source_foa_sha256"])
        )
        target_path = self._verify_file(
            str(value["target_foa_path"]), str(value["target_foa_sha256"])
        )
        samples = int(value["model_num_samples"])
        edited_source_ids = tuple(json.loads(value.pop("edited_source_ids_json")))
        unchanged_source_ids = tuple(
            json.loads(value.pop("unchanged_source_ids_json"))
        )
        try:
            source_render = json.loads(str(manifest["render_result_json"]))
        except (TypeError, ValueError) as error:
            raise RuntimeError("source render provenance is invalid") from error
        if not (
            source_render.get("status") == "ok"
            and str(source_render.get("sample_id")) == str(value["source_sample_id"])
            and str(source_render.get("foa_sha256"))
            == str(value["source_foa_sha256"])
        ):
            raise RuntimeError("source render provenance changed")
        stem_refs = {
            str(record.get("source_id")): record
            for record in source_render.get("stem_refs", [])
        }
        if any(source_id not in stem_refs for source_id in unchanged_source_ids):
            raise RuntimeError("source render lacks an unchanged-source stem")
        source_stem_foa = {}
        source_stem_refs = {}
        for source_id in unchanged_source_ids:
            record = stem_refs[source_id]
            stem_path = self._verify_file(
                str(record.get("path") or ""), str(record.get("sha256") or "")
            )
            source_stem_foa[source_id] = self._read_foa(stem_path, samples)
            source_stem_refs[source_id] = {
                "path": str(stem_path),
                "sha256": str(record["sha256"]),
            }
        return {
            **value,
            "edited_source_ids": edited_source_ids,
            "unchanged_source_ids": unchanged_source_ids,
            "offline_old_sceneplan": old_plan,
            "offline_new_sceneplan": new_plan,
            "source_foa_path": str(source_path),
            "target_foa_path": str(target_path),
            "source_foa": self._read_foa(source_path, samples),
            "target_foa": self._read_foa(target_path, samples),
            "source_stem_foa": source_stem_foa,
            "source_stem_refs": source_stem_refs,
        }


def _chunks(values: Sequence[int], size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), int(size)):
        yield [int(value) for value in values[start : start + int(size)]]


def _pad_audio(rows: Sequence[torch.Tensor], bucket: int) -> torch.Tensor:
    output = torch.zeros(
        (len(rows), 4, int(bucket) * VAE_HOP_SAMPLES), dtype=torch.float32
    )
    for index, value in enumerate(rows):
        output[index, :, : value.shape[-1]] = value
    return output


def _initial_noise(
    truths: Sequence[Mapping[str, Any]], bucket: int, base_seed: int
) -> torch.Tensor:
    rows = []
    for truth in truths:
        generator = torch.Generator(device="cpu").manual_seed(
            _stable_seed(base_seed, str(truth["pair_id"]), "dit-initial-noise")
        )
        rows.append(torch.randn((64, int(bucket)), generator=generator))
    return torch.stack(rows)


def _reject_model_key_leak(value: Mapping[str, Any], *, where: str) -> None:
    keys = {str(key).lower().replace("-", "_") for key in value}
    leaked = keys & FORBIDDEN_MODEL_KEYS
    if leaked:
        raise RuntimeError(f"old/source ScenePlan leaked into {where}: {sorted(leaked)}")


def _mse(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    return float((estimate.float() - reference.float()).square().mean())


def _nmse(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = (estimate.float() - reference.float()).square().mean()
    denominator = reference.float().square().mean().clamp_min(1.0e-12)
    return float(numerator / denominator)


def _progress(output_error: float, source_error: float) -> float | None:
    if not math.isfinite(output_error) or not math.isfinite(source_error):
        return None
    if source_error <= 1.0e-12:
        return None
    return 1.0 - output_error / source_error


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1.0e-12:
        return None
    return float(torch.dot(left, right) / denominator)


def _si_sdr(estimate: torch.Tensor, reference: torch.Tensor) -> float | None:
    estimate = estimate.float().reshape(-1)
    reference = reference.float().reshape(-1)
    estimate = estimate - estimate.mean()
    reference = reference - reference.mean()
    reference_energy = reference.square().sum()
    if float(reference_energy) <= 1.0e-12:
        return None
    projection = torch.dot(estimate, reference) * reference / reference_energy
    residual = estimate - projection
    return float(
        10.0
        * torch.log10(
            projection.square().sum().clamp_min(1.0e-12)
            / residual.square().sum().clamp_min(1.0e-12)
        )
    )


def _mrstft_w(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    estimate = estimate.float().reshape(-1)
    reference = reference.float().reshape(-1)
    values = []
    for fft in (2048, 512):
        window = torch.hann_window(fft, device=estimate.device)
        estimated = torch.stft(
            estimate, fft, hop_length=fft // 4, win_length=fft,
            window=window, center=False, return_complex=True,
        ).abs()
        target = torch.stft(
            reference, fft, hop_length=fft // 4, win_length=fft,
            window=window, center=False, return_complex=True,
        ).abs()
        convergence = (estimated - target).norm() / target.norm().clamp_min(1.0e-8)
        log_magnitude = (
            torch.log(estimated.clamp_min(1.0e-7))
            - torch.log(target.clamp_min(1.0e-7))
        ).abs().mean()
        values.append(convergence + log_magnitude)
    return float(torch.stack(values).mean())


def _frame_audio(audio: torch.Tensor, frames: int) -> torch.Tensor:
    target = int(frames) * VAE_HOP_SAMPLES
    value = audio.float().cpu()
    if value.shape[-1] < target:
        value = F.pad(value, (0, target - value.shape[-1]))
    else:
        value = value[:, :target]
    return value.reshape(4, int(frames), VAE_HOP_SAMPLES)


def _plan_activity_mask(
    plan: Mapping[str, Any], source_ids: Sequence[str], frames: int, samples: int
) -> torch.Tensor:
    controls = compile_model_44_controls(
        dict(plan), model_num_samples=int(samples), latent_frames_valid=int(frames)
    )
    active = torch.from_numpy(controls["source_event_frame_ids"]).gt(0)
    wanted = set(str(value) for value in source_ids)
    mask = torch.zeros(int(frames), dtype=torch.bool)
    for source in plan["sources"]:
        if str(source["source_id"]) in wanted:
            source_id = str(source["source_id"])
            slot = int(source_id[7:])
            mask |= active[slot, :frames]
    return mask


def _erode(mask: torch.Tensor, guard: int) -> torch.Tensor:
    if int(guard) <= 0:
        return mask
    width = 2 * int(guard) + 1
    values = F.conv1d(
        mask.float()[None, None], torch.ones(1, 1, width), padding=int(guard)
    )[0, 0]
    return values.eq(float(width))


def _plan_demix_operator(
    plan: Mapping[str, Any], *, frames: int, samples: int, ridge: float = DEMIX_RIDGE
) -> tuple[list[str], torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build one target-plan FOA decoder for offline unchanged-source scoring."""

    if not math.isfinite(float(ridge)) or float(ridge) <= 0.0:
        raise ValueError("FOA demix ridge must be finite and positive")
    controls = compile_model_44_controls(
        dict(plan), model_num_samples=int(samples), latent_frames_valid=int(frames)
    )
    sources = list(plan["sources"])
    source_ids = [str(source["source_id"]) for source in sources]
    slots = [int(source_id[7:]) for source_id in source_ids]
    active = torch.from_numpy(
        controls["source_event_frame_ids"][slots]
    ).gt(0)
    trajectory = torch.from_numpy(
        controls["source_trajectory_features"][slots]
    ).float()
    sin_azimuth = trajectory[:, :, 0]
    cos_azimuth = trajectory[:, :, 1]
    sin_elevation = trajectory[:, :, 2]
    cos_elevation = trajectory[:, :, 3]
    x = cos_elevation * cos_azimuth
    y = cos_elevation * sin_azimuth
    z = sin_elevation
    source_count = len(source_ids)
    steering = torch.zeros(int(frames), 4, source_count)
    steering[:, 0, :] = active.transpose(0, 1).float() / math.sqrt(2.0)
    steering[:, 1, :] = y.transpose(0, 1)
    steering[:, 2, :] = z.transpose(0, 1)
    steering[:, 3, :] = x.transpose(0, 1)
    identity = torch.eye(source_count).unsqueeze(0)
    decoder = torch.linalg.solve(
        steering.transpose(1, 2) @ steering + float(ridge) * identity,
        steering.transpose(1, 2),
    )
    condition_by_source: dict[str, list[float]] = {
        source_id: [] for source_id in source_ids
    }
    for frame_index in range(int(frames)):
        columns = torch.nonzero(active[:, frame_index], as_tuple=False).flatten()
        if len(columns) <= 1:
            for column in columns.tolist():
                condition_by_source[source_ids[int(column)]].append(1.0)
            continue
        condition = float(torch.linalg.cond(steering[frame_index, :, columns]))
        for column in columns.tolist():
            condition_by_source[source_ids[int(column)]].append(condition)
    condition_p90 = {
        source_id: (
            float(torch.quantile(torch.tensor(values), 0.9)) if values else math.inf
        )
        for source_id, values in condition_by_source.items()
    }
    return source_ids, decoder, active, condition_p90


def _apply_plan_demix(
    audio: torch.Tensor,
    *,
    decoder: torch.Tensor,
    active: torch.Tensor,
    frames: int,
    samples: int,
) -> torch.Tensor:
    framed = _frame_audio(audio, int(frames)).permute(1, 0, 2)
    stems = (decoder @ framed).permute(1, 0, 2)
    stems = stems * active[:, :, None]
    return stems.reshape(len(active), -1)[:, : int(samples)]


def _activity_and_change_metrics(
    *,
    output: torch.Tensor,
    source_codec: torch.Tensor,
    target_codec: torch.Tensor,
    target_raw: torch.Tensor,
    old_plan: Mapping[str, Any],
    new_plan: Mapping[str, Any],
    edited_ids: Sequence[str],
    unchanged_ids: Sequence[str],
    source_stems: Mapping[str, torch.Tensor],
    frames: int,
    samples: int,
) -> dict[str, float | int | None]:
    output_frames = _frame_audio(output, frames)
    source_frames = _frame_audio(source_codec, frames)
    target_frames = _frame_audio(target_codec, frames)
    raw_frames = _frame_audio(target_raw, frames)
    scene_active = _plan_activity_mask(
        new_plan,
        [str(source["source_id"]) for source in new_plan["sources"]],
        frames,
        samples,
    )
    target_rms = raw_frames.square().mean(dim=(0, 2)).sqrt()
    output_rms = output_frames.square().mean(dim=(0, 2)).sqrt()
    active_median = float(target_rms[scene_active].median()) if scene_active.any() else 0.0
    # Calibrate against the rendered target itself; a global absolute threshold
    # would incorrectly mark quiet but valid scenes as silent.
    activity_threshold = max(1.0e-8, 0.1 * active_median)
    detected = output_rms >= activity_threshold
    target_detected = target_rms >= activity_threshold
    union = detected | scene_active
    activity_iou = (
        float((detected & scene_active).sum() / union.sum()) if union.any() else None
    )
    target_union = target_detected | scene_active
    target_plan_iou = (
        float((target_detected & scene_active).sum() / target_union.sum())
        if target_union.any()
        else None
    )
    paired_union = detected | target_detected
    output_target_iou = (
        float((detected & target_detected).sum() / paired_union.sum())
        if paired_union.any()
        else None
    )

    edited_mask = _plan_activity_mask(old_plan, edited_ids, frames, samples)
    edited_mask |= _plan_activity_mask(new_plan, edited_ids, frames, samples)
    target_change = (target_frames - source_frames).square().mean(dim=(0, 2)).sqrt()
    output_change = (output_frames - source_frames).square().mean(dim=(0, 2)).sqrt()
    change_median = (
        float(target_change[edited_mask].median()) if edited_mask.any() else 0.0
    )
    change_threshold = max(1.0e-7, 0.1 * change_median)
    target_changed = target_change.ge(change_threshold) & edited_mask
    output_changed = output_change.ge(change_threshold)
    change_union = target_changed | output_changed
    change_iou = (
        float((target_changed & output_changed).sum() / change_union.sum())
        if change_union.any()
        else None
    )
    change_correlation = None
    if int(edited_mask.sum()) >= 2:
        left = output_change[edited_mask]
        right = target_change[edited_mask]
        left = left - left.mean()
        right = right - right.mean()
        denominator = left.norm() * right.norm()
        if float(denominator) > 1.0e-12:
            change_correlation = float(torch.dot(left, right) / denominator)

    unchanged = _plan_activity_mask(new_plan, unchanged_ids, frames, samples)
    unchanged &= ~edited_mask
    unchanged = _erode(unchanged, 2)
    preservation_ratio = None
    preservation_output_mse = None
    preservation_copy_mse = None
    preservation_codec_mse = None
    if unchanged.any():
        sample_mask = unchanged.repeat_interleave(VAE_HOP_SAMPLES)[:samples]
        output_region = output[:, :samples][:, sample_mask]
        source_region = source_codec[:, :samples][:, sample_mask]
        target_region = target_codec[:, :samples][:, sample_mask]
        raw_region = target_raw[:, :samples][:, sample_mask]
        preservation_output_mse = _mse(output_region, target_region)
        preservation_copy_mse = _mse(source_region, target_region)
        preservation_codec_mse = _mse(target_region, raw_region)
        budget = preservation_copy_mse + preservation_codec_mse
        if budget > 1.0e-12:
            preservation_ratio = preservation_output_mse / budget

    demix_total = len(unchanged_ids)
    demix_eligible = 0
    demix_conditions = []
    demix_calibration_sisdr = []
    demix_output_sisdr = []
    demix_copy_sisdr = []
    demix_sisdr_deltas = []
    demix_output_mrstft = []
    demix_copy_mrstft = []
    if demix_total:
        if set(source_stems) != set(str(value) for value in unchanged_ids):
            raise RuntimeError("unchanged-source stem evidence is incomplete")
        source_ids, decoder, source_active, condition_p90 = _plan_demix_operator(
            new_plan, frames=frames, samples=samples
        )
        output_demix = _apply_plan_demix(
            output,
            decoder=decoder,
            active=source_active,
            frames=frames,
            samples=samples,
        )
        target_demix = _apply_plan_demix(
            target_codec,
            decoder=decoder,
            active=source_active,
            frames=frames,
            samples=samples,
        )
        copy_demix = _apply_plan_demix(
            source_codec,
            decoder=decoder,
            active=source_active,
            frames=frames,
            samples=samples,
        )
        source_index = {source_id: index for index, source_id in enumerate(source_ids)}
        for source_id in unchanged_ids:
            source_id = str(source_id)
            slot = source_index[source_id]
            frame_mask = _erode(source_active[slot], 2)
            sample_mask = frame_mask.repeat_interleave(VAE_HOP_SAMPLES)[:samples]
            condition = float(condition_p90[source_id])
            if (
                int(sample_mask.sum()) < 2048
                or not math.isfinite(condition)
                or condition > DEMIX_MAX_CONDITION
            ):
                continue
            stem_w = _frame_audio(source_stems[source_id], frames)[0].reshape(-1)[
                :samples
            ][sample_mask]
            target_w = target_demix[slot, sample_mask]
            output_w = output_demix[slot, sample_mask]
            copy_w = copy_demix[slot, sample_mask]
            calibration_sisdr = _si_sdr(target_w, stem_w)
            output_sisdr = _si_sdr(output_w, target_w)
            copy_sisdr = _si_sdr(copy_w, target_w)
            if (
                calibration_sisdr is None
                or calibration_sisdr < DEMIX_MIN_CALIBRATION_SI_SDR_DB
                or output_sisdr is None
                or copy_sisdr is None
            ):
                continue
            demix_eligible += 1
            demix_conditions.append(condition)
            demix_calibration_sisdr.append(calibration_sisdr)
            demix_output_sisdr.append(output_sisdr)
            demix_copy_sisdr.append(copy_sisdr)
            demix_sisdr_deltas.append(output_sisdr - copy_sisdr)
            demix_output_mrstft.append(_mrstft_w(output_w, target_w))
            demix_copy_mrstft.append(_mrstft_w(copy_w, target_w))

    def mean(values: Sequence[float]) -> float | None:
        return None if not values else sum(values) / len(values)

    return {
        "activity_temporal_iou": activity_iou,
        "activity_target_plan_iou": target_plan_iou,
        "activity_output_target_iou": output_target_iou,
        "activity_threshold": activity_threshold,
        "edited_interval_frames": int(edited_mask.sum()),
        "change_temporal_iou": change_iou,
        "change_envelope_correlation": change_correlation,
        "change_threshold": change_threshold,
        "unchanged_direct_frames": int(unchanged.sum()),
        "unchanged_output_target_mse": preservation_output_mse,
        "unchanged_copy_target_mse": preservation_copy_mse,
        "unchanged_target_codec_mse": preservation_codec_mse,
        "unchanged_preservation_budget_ratio": preservation_ratio,
        "unchanged_demix_total_sources": demix_total,
        "unchanged_demix_eligible_sources": demix_eligible,
        "unchanged_demix_eligibility_fraction": (
            None if demix_total == 0 else demix_eligible / demix_total
        ),
        "unchanged_demix_condition_p90": mean(demix_conditions),
        "unchanged_demix_target_source_calibration_si_sdr_db": mean(
            demix_calibration_sisdr
        ),
        "unchanged_demix_output_target_si_sdr_db": mean(demix_output_sisdr),
        "unchanged_demix_copy_target_si_sdr_db": mean(demix_copy_sisdr),
        "unchanged_demix_si_sdr_delta_vs_copy_db": mean(demix_sisdr_deltas),
        "unchanged_demix_output_target_mrstft": mean(demix_output_mrstft),
        "unchanged_demix_copy_target_mrstft": mean(demix_copy_mrstft),
    }


def _audio_metrics(
    *,
    output: torch.Tensor,
    source_raw: torch.Tensor,
    target_raw: torch.Tensor,
    source_codec: torch.Tensor,
    target_codec: torch.Tensor,
) -> dict[str, float | None]:
    codec_foa_output = _nmse(output, target_codec)
    codec_foa_source = _nmse(source_codec, target_codec)
    raw_foa_output = _nmse(output, target_raw)
    raw_foa_source = _nmse(source_raw, target_raw)
    raw_w_output = _nmse(output[:1], target_raw[:1])
    raw_w_source = _nmse(source_raw[:1], target_raw[:1])
    output_sisdr = _si_sdr(output[0], target_raw[0])
    source_sisdr = _si_sdr(source_raw[0], target_raw[0])
    return {
        "audio_codec_foa_nmse": codec_foa_output,
        "audio_codec_source_foa_nmse": codec_foa_source,
        "audio_codec_foa_progress": _progress(codec_foa_output, codec_foa_source),
        "audio_raw_foa_nmse": raw_foa_output,
        "audio_raw_source_foa_nmse": raw_foa_source,
        "audio_raw_foa_progress": _progress(raw_foa_output, raw_foa_source),
        "audio_raw_w_nmse": raw_w_output,
        "audio_raw_source_w_nmse": raw_w_source,
        "audio_raw_w_progress": _progress(raw_w_output, raw_w_source),
        "audio_w_si_sdr_db": output_sisdr,
        "audio_source_w_si_sdr_db": source_sisdr,
        "audio_w_si_sdr_improvement_db": (
            None
            if output_sisdr is None or source_sisdr is None
            else output_sisdr - source_sisdr
        ),
        "audio_w_mrstft": _mrstft_w(output[0], target_raw[0]),
        "audio_source_w_mrstft": _mrstft_w(source_raw[0], target_raw[0]),
        "source_codec_reconstruction_foa_nmse": _nmse(source_codec, source_raw),
        "target_codec_ceiling_foa_nmse": _nmse(target_codec, target_raw),
        "change_direction_cosine_foa": _cosine(
            output - source_codec, target_codec - source_codec
        ),
        "change_direction_cosine_w": _cosine(
            output[:1] - source_codec[:1], target_codec[:1] - source_codec[:1]
        ),
    }


def _plan_metrics(
    *,
    codec: ModelScenePlanCodecV4,
    target_plan: Mapping[str, Any],
    predicted_plan: Mapping[str, Any],
    target_tokens: Sequence[int],
    predicted_tokens: Sequence[int],
) -> dict[str, float]:
    values = score_token_sequence(target_tokens, predicted_tokens, codec)
    # Match checkpoint selection: score AR fields against what its vocabulary
    # can express. Raw continuous plans remain the independent audio truth.
    codec_target = codec.decode(target_tokens, sample_id=str(target_plan["sample_id"]))
    values.update(score_parsed_generation(codec_target, dict(predicted_plan)))
    return {f"plan_{key}": float(value) for key, value in values.items()}


def _atomic_wav(path: Path, audio: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + f".tmp.{os.getpid()}.wav")
    sf.write(
        temporary,
        audio.detach().float().cpu().T.contiguous().numpy(),
        MODEL_SAMPLE_RATE,
        subtype="FLOAT",
        format="WAV",
    )
    os.replace(temporary, path)
    return sha256_file(path)


def _finite_metrics(metrics: Mapping[str, Any]) -> None:
    for key, value in metrics.items():
        if value is not None and isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise RuntimeError(f"non-finite E2E metric: {key}={value}")


@torch.no_grad()
def _process_batch(
    *,
    pipeline,
    content_evaluator: IndependentEditingContentEvaluator,
    codec: ModelScenePlanCodecV4,
    samples: Sequence[tuple[torch.Tensor, dict[str, Any], dict[str, Any]]],
    truths: Sequence[Mapping[str, Any]],
    bucket: int,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    listening_ordinals: set[int],
    sampled_gt_result: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    metadata = [sample[1] for sample in samples]
    model_inputs = {
        "source_foa": _pad_audio([truth["source_foa"] for truth in truths], bucket),
        "edit_instructions": [str(row["raw_edit_request"]) for row in metadata],
        "model_num_samples": [int(row["model_num_samples"]) for row in metadata],
        "vae_seeds": [
            _stable_seed(args.seed, str(row["pair_id"]), "source-vae")
            for row in metadata
        ],
        "max_plan_tokens": int(args.max_plan_tokens),
        "steps": int(args.ode_steps),
        "cfg_scale": float(args.cfg_scale),
        "initial_noise": _initial_noise(truths, bucket, args.seed),
    }
    _reject_model_key_leak(model_inputs, where="real-audio Editing model call")
    # The standalone DiT gate supplies an already sampled/decoded GT-plan
    # result. It shares only the post-inference scorer with the AR evaluator.
    # Its records must never count as freely generated AR plans.
    result = (
        pipeline.edit_audio(**model_inputs)
        if sampled_gt_result is None else sampled_gt_result
    )
    target_latents = torch.stack(
        [sample[0][:, :bucket] for sample in samples]
    ).to(device=device, dtype=torch.float32)
    target_codec, _ = pipeline.decode_foa_latents(
        target_latents,
        model_num_samples=[int(row["model_num_samples"]) for row in metadata],
    )
    content_scores = content_evaluator.score_batch(
        outputs=[
            result["edited_foa"][index, :, : int(row["model_num_samples"])]
            for index, row in enumerate(metadata)
        ],
        sources=[
            result["source_codec_foa"][index, :, : int(row["model_num_samples"])]
            for index, row in enumerate(metadata)
        ],
        targets=[
            target_codec[index, :, : int(row["model_num_samples"])]
            for index, row in enumerate(metadata)
        ],
        old_plans=[truth["offline_old_sceneplan"] for truth in truths],
        new_plans=[truth["offline_new_sceneplan"] for truth in truths],
        edited_source_ids=[truth["edited_source_ids"] for truth in truths],
        unchanged_source_ids=[truth["unchanged_source_ids"] for truth in truths],
        operations=[str(truth["operation"]) for truth in truths],
    )
    if len(content_scores) != len(samples):
        raise RuntimeError("independent content scorer lost an Editing row")
    records = []
    for index, (sample, row, truth, content_score) in enumerate(
        zip(samples, metadata, truths, content_scores)
    ):
        ordinal = int(row["pair_ordinal"])
        if not (
            str(row["pair_id"]) == str(truth["pair_id"])
            and str(row["raw_edit_request"]) == str(truth["raw_edit_request"])
            and row.get("editing_persistent_target_sceneplan", row["model_sceneplan"])
            == truth["offline_new_sceneplan"]
            and int(row["latent_bucket_frames"]) == int(bucket)
        ):
            raise RuntimeError("dataset/offline audio truth alignment changed")
        count = int(row["model_num_samples"])
        frames = int(row["latent_frames_valid"])
        output = result["edited_foa"][index, :, :count]
        source_codec = result["source_codec_foa"][index, :, :count]
        target_codec_row = target_codec[index, :, :count]
        source_raw = truth["source_foa"].to(device)
        target_raw = truth["target_foa"].to(device)
        metrics: dict[str, Any] = {}
        predicted_tokens = []
        predicted_plan = result["new_sceneplans"][index]
        if sampled_gt_result is None:
            target_tokens = sample[2]["target_token_ids"].tolist()
            predicted_tokens = result["new_sceneplan_token_ids"][index].tolist()
            metrics.update(
                _plan_metrics(
                    codec=codec,
                    target_plan=row["model_sceneplan"],
                    predicted_plan=predicted_plan,
                    target_tokens=target_tokens,
                    predicted_tokens=predicted_tokens,
                )
            )
        source_latent = result["source_foa_latent"][index, :, :frames]
        output_latent = result["edited_foa_latent"][index, :, :frames]
        target_latent = target_latents[index, :, :frames]
        latent_output_error = _nmse(output_latent, target_latent)
        latent_source_error = _nmse(source_latent, target_latent)
        metrics.update(
            {
                "latent_foa_nmse": latent_output_error,
                "latent_source_foa_nmse": latent_source_error,
                "latent_foa_progress": _progress(
                    latent_output_error, latent_source_error
                ),
                "fresh_vs_stored_source_latent_mse": _mse(
                    source_latent,
                    row["source_foa_latent"][:, :frames].to(device).float(),
                ),
            }
        )
        metrics.update(
            _audio_metrics(
                output=output,
                source_raw=source_raw,
                target_raw=target_raw,
                source_codec=source_codec,
                target_codec=target_codec_row,
            )
        )
        output_cpu = output.float().cpu()
        source_codec_cpu = source_codec.float().cpu()
        target_codec_cpu = target_codec_row.float().cpu()
        source_raw_cpu = truth["source_foa"].float()
        target_raw_cpu = truth["target_foa"].float()
        metrics.update(
            _activity_and_change_metrics(
                output=output_cpu,
                source_codec=source_codec_cpu,
                target_codec=target_codec_cpu,
                target_raw=target_raw_cpu,
                old_plan=truth["offline_old_sceneplan"],
                new_plan=truth["offline_new_sceneplan"],
                edited_ids=truth["edited_source_ids"],
                unchanged_ids=truth["unchanged_source_ids"],
                source_stems=truth["source_stem_foa"],
                frames=frames,
                samples=count,
            )
        )
        plan_doa = _doa_metrics(
            output_cpu,
            truth["offline_new_sceneplan"],
            model_num_samples=count,
            latent_frames=frames,
        )
        target_doa = _paired_doa_metrics(
            output_cpu,
            target_raw_cpu,
            truth["offline_new_sceneplan"],
            model_num_samples=count,
            latent_frames=frames,
        )
        source_target_doa = _paired_doa_metrics(
            source_raw_cpu,
            target_raw_cpu,
            truth["offline_new_sceneplan"],
            model_num_samples=count,
            latent_frames=frames,
        )
        output_angle = target_doa["spherical_error_mean_deg"]
        source_angle = source_target_doa["spherical_error_mean_deg"]
        spatial_progress = None
        if output_angle is not None and source_angle is not None and source_angle >= 1.0:
            spatial_progress = 1.0 - float(output_angle) / float(source_angle)
        metrics.update(
            {
                "doa_plan_mean_deg": plan_doa["spherical_error_mean_deg"],
                "doa_plan_p95_deg": plan_doa["spherical_error_p95_deg"],
                "doa_target_mean_deg": output_angle,
                "doa_target_p95_deg": target_doa["spherical_error_p95_deg"],
                "doa_source_target_mean_deg": source_angle,
                "doa_spatial_progress": spatial_progress,
                "trajectory_start_error_deg": plan_doa[
                    "start_spherical_error_mean_deg"
                ],
                "trajectory_end_error_deg": plan_doa[
                    "end_spherical_error_mean_deg"
                ],
                "trajectory_extent_error_deg": plan_doa[
                    "trajectory_extent_error_deg"
                ],
            }
        )
        if content_score.get("contract") != INDEPENDENT_CONTENT_METRIC_CONTRACT:
            raise RuntimeError("independent content metric contract changed")
        metrics.update(content_score["metrics"])
        _finite_metrics(metrics)
        audio_path = None
        audio_sha = None
        if args.save_all_audio or ordinal in listening_ordinals:
            audio_path = (
                output_dir / "audio" / f"rank-{dist.get_rank()}" /
                f"{ordinal:05d}_{truth['pair_id']}.wav"
            )
            audio_sha = _atomic_wav(audio_path, output_cpu)
        records.append(
            {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "status": "ok",
                "pair_ordinal": ordinal,
                "pair_id": str(truth["pair_id"]),
                "operation": str(truth["operation"]),
                "source_domain": str(truth["source_domain"]),
                "target_domain": str(truth["target_domain"]),
                "latent_bucket_frames": int(bucket),
                "latent_frames_valid": frames,
                "model_num_samples": count,
                "source_count": int(truth["source_count"]),
                "target_count": int(truth["target_count"]),
                "edited_source_ids": list(truth["edited_source_ids"]),
                "unchanged_source_ids": list(truth["unchanged_source_ids"]),
                "source_stem_refs": truth["source_stem_refs"],
                "source_foa_path": str(truth["source_foa_path"]),
                "target_foa_path": str(truth["target_foa_path"]),
                "edited_foa_path": None if audio_path is None else str(audio_path.resolve()),
                "edited_foa_sha256": audio_sha,
                "generated_sceneplan": predicted_plan,
                "generated_plan_tokens": predicted_tokens,
                "independent_content_metrics": content_score["diagnostics"],
                "independent_content_metric_contract": content_score["contract"],
                "model_input_contract": (
                    dict(MODEL_INPUT_CONTRACT) if sampled_gt_result is None else {
                        "editing_ar": None,
                        "old_sceneplan": False,
                        "editing_dit": [
                            "noisy_target_64", "gt_new_sceneplan_256",
                            "source_foa_latent_64",
                        ],
                        "editing_dit_channels": 384,
                        "independent_content_metrics_are_post_inference_only": True,
                    }
                ),
                "plan_origin": "free_ar" if sampled_gt_result is None else "ground_truth",
                "offline_metric_truth": [
                    "old_sceneplan",
                    "new_sceneplan",
                    "edited_source_ids",
                    "unchanged_source_ids",
                    "source_stem_foa",
                ],
                "metrics": metrics,
            }
        )
    return records


def _error_records(
    samples: Sequence[tuple[torch.Tensor, dict[str, Any], dict[str, Any]]],
    error: Exception,
) -> list[dict[str, Any]]:
    return [
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "pair_ordinal": int(sample[1]["pair_ordinal"]),
            "pair_id": str(sample[1]["pair_id"]),
            "operation": str(sample[1]["operation"]),
            "latent_bucket_frames": int(sample[1]["latent_bucket_frames"]),
            "error": f"{type(error).__name__}: {error}",
            "model_input_contract": {
                "editing_ar": ["source_foa_latent", "raw_edit_request"],
                "old_sceneplan": False,
                "editing_dit_channels": 384,
            },
        }
        for sample in samples
    ]


def _validate_batch_records(
    records: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    *,
    bucket: int,
) -> None:
    """Bind a resumable shard's records to exact frozen-index identities."""

    if len(records) != len(expected_rows):
        raise RuntimeError("audio E2E shard record count changed")
    for record, expected in zip(records, expected_rows):
        model_contract = dict(record.get("model_input_contract") or {})
        if not (
            record.get("schema") == SCHEMA
            and int(record.get("schema_version", -1)) == SCHEMA_VERSION
            and record.get("status") in {"ok", "error"}
            and int(record.get("pair_ordinal", -1))
            == int(expected["pair_ordinal"])
            and str(record.get("pair_id")) == str(expected["pair_id"])
            and str(record.get("operation")) == str(expected["operation"])
            and int(record.get("latent_bucket_frames", -1)) == int(bucket)
            and int(expected["latent_bucket_frames"]) == int(bucket)
            and model_contract.get("old_sceneplan") is False
            and int(model_contract.get("editing_dit_channels", -1)) == 384
        ):
            raise RuntimeError("audio E2E shard row identity/contract changed")
        if record.get("status") == "ok":
            if (
                record.get("independent_content_metric_contract")
                != INDEPENDENT_CONTENT_METRIC_CONTRACT
                or not isinstance(record.get("metrics"), dict)
            ):
                raise RuntimeError("audio E2E successful shard row is incomplete")
            _finite_metrics(record["metrics"])


def _stat(values: Sequence[float], *, group_rows: int) -> dict[str, Any]:
    tensor = torch.tensor(list(values), dtype=torch.float64)
    denominator = int(group_rows)
    if denominator < len(tensor) or denominator < 0:
        raise RuntimeError("metric coverage denominator is invalid")
    if not len(tensor):
        return {
            "rows": 0,
            "group_rows": denominator,
            "coverage_fraction": 0.0,
            "mean": None,
            "std": None,
            "one_sided_lower_confidence_bound": None,
            "one_sided_upper_confidence_bound": None,
            "confidence": CONFIDENCE,
        }
    mean = float(tensor.mean())
    std = float(tensor.std(unbiased=True)) if len(tensor) > 1 else 0.0
    radius = NormalDist().inv_cdf(CONFIDENCE) * std / math.sqrt(len(tensor))
    return {
        "rows": len(tensor),
        "group_rows": denominator,
        "coverage_fraction": len(tensor) / max(1, denominator),
        "mean": mean,
        "std": std,
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "one_sided_lower_confidence_bound": mean - radius,
        "one_sided_upper_confidence_bound": mean + radius,
        "confidence": CONFIDENCE,
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    def summarize(selected: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        selected_rows = list(selected)
        output = []
        for row in selected_rows:
            value = row.get("metrics", {}).get(metric)
            if value is not None:
                output.append(float(value))
        return _stat(output, group_rows=len(selected_rows))

    return {
        "overall": summarize(rows),
        "by_operation": {
            operation: summarize(
                row for row in rows if row.get("operation") == operation
            )
            for operation in OPERATIONS
        },
        "by_latent_bucket": {
            str(bucket): summarize(
                row
                for row in rows
                if int(row.get("latent_bucket_frames", -1)) == bucket
            )
            for bucket in (432, 648)
        },
        "by_operation_bucket": {
            f"{bucket}:{operation}": summarize(
                row
                for row in rows
                if row.get("operation") == operation
                and int(row.get("latent_bucket_frames", -1)) == bucket
            )
            for bucket in (432, 648)
            for operation in OPERATIONS
        },
    }


def _unchanged_source_coverage_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def summarize(selected: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        selected_rows = list(selected)
        eligible_sources = 0
        total_sources = 0
        rows_with_unchanged_sources = 0
        rows_with_eligible_sources = 0
        for row in selected_rows:
            metrics = row.get("metrics", {})
            eligible = metrics.get("unchanged_demix_eligible_sources")
            total = metrics.get("unchanged_demix_total_sources")
            if (
                not isinstance(eligible, int)
                or isinstance(eligible, bool)
                or not isinstance(total, int)
                or isinstance(total, bool)
                or not 0 <= eligible <= total
            ):
                raise RuntimeError(
                    "unchanged demix source coverage counts are missing or invalid"
                )
            eligible_sources += eligible
            total_sources += total
            rows_with_unchanged_sources += int(total > 0)
            rows_with_eligible_sources += int(eligible > 0)
        if total_sources == 0:
            return {
                "rows": 0,
                "group_rows": 0,
                "coverage_fraction": 0.0,
                "mean": None,
                "std": None,
                "one_sided_lower_confidence_bound": None,
                "one_sided_upper_confidence_bound": None,
                "confidence": CONFIDENCE,
                "eligible_sources": 0,
                "total_sources": 0,
                "selected_pair_rows": len(selected_rows),
                "rows_with_unchanged_sources": 0,
                "rows_with_eligible_sources": 0,
            }
        fraction = eligible_sources / total_sources
        # Wilson bounds remain in [0,1] at sparse coverage and avoid the
        # zero-variance false certainty of a plain normal interval.
        z = NormalDist().inv_cdf(CONFIDENCE)
        denominator = 1.0 + z * z / total_sources
        center = (fraction + z * z / (2.0 * total_sources)) / denominator
        radius = (
            z
            * math.sqrt(
                fraction * (1.0 - fraction) / total_sources
                + z * z / (4.0 * total_sources * total_sources)
            )
            / denominator
        )
        return {
            "rows": eligible_sources,
            "group_rows": total_sources,
            "coverage_fraction": fraction,
            "mean": fraction,
            "std": math.sqrt(fraction * (1.0 - fraction)),
            "min": 0.0,
            "max": 1.0,
            "one_sided_lower_confidence_bound": max(0.0, center - radius),
            "one_sided_upper_confidence_bound": min(1.0, center + radius),
            "confidence": CONFIDENCE,
            "eligible_sources": eligible_sources,
            "total_sources": total_sources,
            "selected_pair_rows": len(selected_rows),
            "rows_with_unchanged_sources": rows_with_unchanged_sources,
            "rows_with_eligible_sources": rows_with_eligible_sources,
        }

    return {
        "overall": summarize(rows),
        "by_operation": {
            operation: summarize(
                row for row in rows if row.get("operation") == operation
            )
            for operation in OPERATIONS
        },
        "by_latent_bucket": {
            str(bucket): summarize(
                row
                for row in rows
                if int(row.get("latent_bucket_frames", -1)) == bucket
            )
            for bucket in (432, 648)
        },
        "by_operation_bucket": {
            f"{bucket}:{operation}": summarize(
                row
                for row in rows
                if row.get("operation") == operation
                and int(row.get("latent_bucket_frames", -1)) == bucket
            )
            for bucket in (432, 648)
            for operation in OPERATIONS
        },
    }


def _all_metric_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            str(key)
            for row in rows
            if row.get("status") == "ok"
            for key in row.get("metrics", {})
        }
    )
    summaries = {name: _metric_summary(rows, name) for name in names}
    summaries["unchanged_demix_source_coverage"] = (
        _unchanged_source_coverage_summary(rows)
    )
    return summaries


def _calibration_thresholds(
    summaries: Mapping[str, Any], *,
    higher_specs: Mapping[str, Any] | None = None,
    lower_specs: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    thresholds: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for metric, spec in (HIGHER_BETTER_SPECS if higher_specs is None else higher_specs).items():
        groups = {}
        group_names = tuple(
            str(value)
            for value in spec.get(
                "groups", ("overall", "by_operation", "by_latent_bucket")
            )
        )
        unknown_groups = set(group_names) - {
            "overall",
            "by_operation",
            "by_latent_bucket",
            "by_operation_bucket",
        }
        if unknown_groups:
            raise RuntimeError(
                f"{metric} threshold references unknown groups: "
                f"{sorted(unknown_groups)}"
            )
        for group_name in group_names:
            source = summaries[metric][group_name]
            items = {"overall": source} if group_name == "overall" else source
            if group_name in {"by_operation", "by_operation_bucket"} and "operations" in spec:
                applicable = tuple(str(value) for value in spec["operations"])
                label_operation = {
                    str(label): (
                        str(label).split(":", 1)[1]
                        if group_name == "by_operation_bucket"
                        else str(label)
                    )
                    for label in items
                }
                present = set(label_operation.values())
                missing = set(applicable) - present
                if missing:
                    raise RuntimeError(
                        f"{metric} applicability references missing operations: "
                        f"{sorted(missing)}"
                    )
                items = {
                    label: record
                    for label, record in items.items()
                    if label_operation[str(label)] in applicable
                }
            groups[group_name] = {}
            for label, record in items.items():
                lower = record["one_sided_lower_confidence_bound"]
                minimum_rows = 100 if label == "overall" else 20
                minimum_coverage = float(spec.get("min_coverage", 1.0))
                validation_coverage = float(record["coverage_fraction"])
                enough = int(record["rows"]) >= minimum_rows
                coverage_pass = validation_coverage >= minimum_coverage
                passed = (
                    enough
                    and coverage_pass
                    and lower is not None
                    and float(lower) >= spec["floor"]
                )
                checks[f"{metric}:{group_name}:{label}"] = bool(passed)
                groups[group_name][label] = {
                    "direction": "higher",
                    "validation_rows": int(record["rows"]),
                    "validation_group_rows": int(record["group_rows"]),
                    "validation_coverage_fraction": validation_coverage,
                    "minimum_rows": minimum_rows,
                    "fixed_minimum_coverage_fraction": minimum_coverage,
                    "test_minimum_coverage_fraction": max(
                        minimum_coverage,
                        validation_coverage
                        - float(spec.get("coverage_tolerance", 0.02)),
                    ),
                    "validation_lower_bound": lower,
                    "fixed_floor": spec["floor"],
                    "regression_tolerance": spec["tolerance"],
                    "test_minimum_lower_bound": (
                        None
                        if lower is None
                        else max(spec["floor"], float(lower) - spec["tolerance"])
                    ),
                }
        thresholds[metric] = groups
    for metric, spec in (LOWER_BETTER_SPECS if lower_specs is None else lower_specs).items():
        groups = {}
        for group_name in tuple(
            str(value) for value in spec.get("groups", ("overall",))
        ):
            if group_name not in {
                "overall",
                "by_operation",
                "by_latent_bucket",
                "by_operation_bucket",
            }:
                raise RuntimeError(
                    f"{metric} threshold references unknown group: {group_name}"
                )
            source = summaries[metric][group_name]
            items = {"overall": source} if group_name == "overall" else source
            if group_name in {"by_operation", "by_operation_bucket"} and "operations" in spec:
                applicable = tuple(str(value) for value in spec["operations"])
                label_operation = {
                    str(label): (
                        str(label).split(":", 1)[1]
                        if group_name == "by_operation_bucket"
                        else str(label)
                    )
                    for label in items
                }
                present = set(label_operation.values())
                missing = set(applicable) - present
                if missing:
                    raise RuntimeError(
                        f"{metric} applicability references missing operations: "
                        f"{sorted(missing)}"
                    )
                items = {
                    label: record
                    for label, record in items.items()
                    if label_operation[str(label)] in applicable
                }
            groups[group_name] = {}
            for label, record in items.items():
                upper = record["one_sided_upper_confidence_bound"]
                minimum_rows = 100 if label == "overall" else 20
                minimum_coverage = float(spec.get("min_coverage", 1.0))
                validation_coverage = float(record["coverage_fraction"])
                passed = (
                    int(record["rows"]) >= minimum_rows
                    and validation_coverage >= minimum_coverage
                    and upper is not None
                    and float(upper) <= spec["ceiling"]
                )
                checks[f"{metric}:{group_name}:{label}"] = bool(passed)
                groups[group_name][label] = {
                    "direction": "lower",
                    "validation_rows": int(record["rows"]),
                    "validation_group_rows": int(record["group_rows"]),
                    "validation_coverage_fraction": validation_coverage,
                    "minimum_rows": minimum_rows,
                    "fixed_minimum_coverage_fraction": minimum_coverage,
                    "test_minimum_coverage_fraction": max(
                        minimum_coverage,
                        validation_coverage
                        - float(spec.get("coverage_tolerance", 0.02)),
                    ),
                    "validation_upper_bound": upper,
                    "fixed_ceiling": spec["ceiling"],
                    "regression_tolerance": spec["tolerance"],
                    "test_maximum_upper_bound": (
                        None
                        if upper is None
                        else min(
                            spec["ceiling"],
                            float(upper) + spec["tolerance"],
                        )
                    ),
                }
        thresholds[metric] = groups
    return thresholds, checks


def _test_threshold_checks(
    summaries: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> dict[str, bool]:
    checks = {}
    for metric, groups in thresholds.items():
        for group_name, records in groups.items():
            for label, threshold in records.items():
                observed = (
                    summaries[metric][group_name]
                    if group_name == "overall"
                    else summaries[metric][group_name][label]
                )
                coverage_pass = (
                    int(observed["rows"]) >= int(threshold["minimum_rows"])
                    and int(observed["group_rows"]) > 0
                    and float(observed["coverage_fraction"])
                    >= float(threshold["test_minimum_coverage_fraction"])
                )
                if threshold["direction"] == "higher":
                    bound = observed["one_sided_lower_confidence_bound"]
                    limit = threshold["test_minimum_lower_bound"]
                    passed = (
                        coverage_pass
                        and bound is not None
                        and limit is not None
                        and float(bound) >= float(limit)
                    )
                else:
                    bound = observed["one_sided_upper_confidence_bound"]
                    limit = threshold["test_maximum_upper_bound"]
                    passed = (
                        coverage_pass
                        and bound is not None
                        and limit is not None
                        and float(bound) <= float(limit)
                    )
                checks[f"{metric}:{group_name}:{label}"] = bool(passed)
    return checks


def _validated_metric_row_records(
    path: Path,
    *,
    index: Path,
    row_selection: Mapping[str, Any],
    expected_rows: int,
) -> list[dict[str, Any]]:
    """Reopen metric rows and bind them to the frozen index before promotion."""

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"blank audio E2E row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid audio E2E row JSON at line {line_number}"
                ) from error
            if not isinstance(row, dict):
                raise RuntimeError("audio E2E row is not an object")
            records.append(row)
    if len(records) != int(expected_rows):
        raise RuntimeError("audio E2E row file has the wrong row count")
    ordinals = [int(row.get("pair_ordinal", -1)) for row in records]
    if ordinals != sorted(ordinals) or len(set(ordinals)) != len(ordinals):
        raise RuntimeError("audio E2E row ordinals are not unique and sorted")
    required_metrics = (
        set(HIGHER_BETTER_SPECS) | set(LOWER_BETTER_SPECS)
    ) - AGGREGATE_ONLY_METRICS
    for row in records:
        metrics = row.get("metrics")
        model_contract = dict(row.get("model_input_contract") or {})
        if not (
            row.get("schema") == SCHEMA
            and int(row.get("schema_version", -1)) == SCHEMA_VERSION
            and row.get("status") == "ok"
            and isinstance(row.get("pair_id"), str)
            and bool(row.get("pair_id"))
            and row.get("operation") in OPERATIONS
            and int(row.get("latent_bucket_frames", -1)) in (432, 648)
            and isinstance(metrics, dict)
            and required_metrics <= set(metrics)
            and model_contract == MODEL_INPUT_CONTRACT
            and metrics.get("plan_grammar_legal") == 1.0
            and row.get("independent_content_metric_contract")
            == INDEPENDENT_CONTENT_METRIC_CONTRACT
        ):
            raise RuntimeError("audio E2E row contract is stale or incomplete")
        _finite_metrics(metrics)

    index_rows: dict[int, tuple[str, str, int]] = {}
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    try:
        for start in range(0, len(ordinals), 500):
            selected = ordinals[start : start + 500]
            placeholders = ",".join("?" for _ in selected)
            for ordinal, pair_id, operation, bucket in connection.execute(
                "SELECT pair_ordinal,pair_id,operation,latent_bucket_frames "
                f"FROM pairs WHERE pair_ordinal IN ({placeholders})",
                tuple(selected),
            ):
                index_rows[int(ordinal)] = (
                    str(pair_id), str(operation), int(bucket)
                )
    finally:
        connection.close()
    for row, ordinal in zip(records, ordinals):
        if index_rows.get(ordinal) != (
            str(row["pair_id"]),
            str(row["operation"]),
            int(row["latent_bucket_frames"]),
        ):
            raise RuntimeError("audio E2E row/index identity changed")
    counts = {
        f"{bucket}:{operation}": sum(
            int(row["latent_bucket_frames"]) == bucket
            and row["operation"] == operation
            for row in records
        )
        for bucket in (432, 648)
        for operation in OPERATIONS
    }
    selection_contract = row_selection.get("contract")
    selection_valid = (
        int(row_selection.get("rows", -1)) == len(records)
        and _ordinal_sha256(ordinals) == row_selection.get("pair_ordinal_sha256")
    )
    if selection_contract == "sha256_ranked_operation_bucket_5x2x100_v1":
        selection_valid = (
            selection_valid
            and int(row_selection.get("per_operation_bucket_cell", -1))
            == CALIBRATION_PER_CELL
            and counts == dict(row_selection.get("by_operation_bucket") or {})
        )
    elif selection_contract == "all_frozen_test_rows_v1":
        selection_valid = selection_valid and ordinals == list(range(len(records)))
    else:
        selection_valid = False
    if not selection_valid:
        raise RuntimeError("audio E2E row selection evidence changed")
    return records


def _calibration_checks_are_closed(
    stored_checks: Any,
    derived_quality_checks: Mapping[str, bool],
) -> bool:
    """Require every replay-derived quality gate to be present and passing."""

    if (
        not isinstance(stored_checks, Mapping)
        or not stored_checks
        or not derived_quality_checks
    ):
        return False
    return (
        all(bool(item) for item in stored_checks.values())
        and all(bool(item) for item in derived_quality_checks.values())
        and all(
            key in stored_checks
            and bool(stored_checks[key]) == bool(passed)
            for key, passed in derived_quality_checks.items()
        )
    )


def _validate_calibration(
    path: Path,
    *,
    expected_sha: str,
    selection: Mapping[str, Any],
    selection_path: Path,
    selection_sha: str,
    model_config: Path,
    codec: Path,
    source_hashes: Mapping[str, str],
    independent_content_assets: Mapping[str, Any],
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256_file(resolved) != str(expected_sha):
        raise RuntimeError("audio E2E calibration SHA256 changed")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    validation_index = dict(value.get("validation_index") or {})
    selected_validation_index = dict(selection.get("validation_index") or {})
    row_selection = dict(value.get("row_selection") or {})
    run_contract_path = Path(value.get("run_contract", "")).resolve(strict=True)
    row_records_path = Path(value.get("row_records", "")).resolve(strict=True)
    run_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
    batch_size = int(value.get("batch_size_per_rank", -1))
    batch_size_record = dict(value.get("batch_size_certification") or {})
    batch_size_path = Path(batch_size_record.get("path", "")).resolve(strict=True)
    batch_size_value = json.loads(batch_size_path.read_text(encoding="utf-8"))
    expected_batch_size_value = _batch_size_sidecar_value(
        phase="calibration", batch_size=batch_size
    )
    stored_summaries = dict(value.get("metric_summaries") or {})
    validated_rows = _validated_metric_row_records(
        row_records_path,
        index=Path(validation_index.get("path", "")).resolve(strict=True),
        row_selection=row_selection,
        expected_rows=CALIBRATION_ROWS,
    )
    recomputed_summaries = _all_metric_summaries(validated_rows)
    derived_thresholds, derived_quality_checks = _calibration_thresholds(
        recomputed_summaries
    )
    if not (
        value.get("schema") == CALIBRATION_SCHEMA
        and int(value.get("schema_version", -1)) == 1
        and value.get("status") == "PASS"
        and value.get("calibration_contract") == CALIBRATION_CONTRACT
        and value.get("evaluation_contract") == EVALUATION_CONTRACT
        and int(value.get("rows", -1)) == CALIBRATION_ROWS
        and int(value.get("successful_rows", -1)) == CALIBRATION_ROWS
        and value.get("joint_checkpoint_selection_sha256") == selection_sha
        and Path(value.get("joint_checkpoint_selection", "")).resolve()
        == selection_path
        and Path(value.get("selected_checkpoint", "")).resolve(strict=True)
        == Path(selection.get("selected_checkpoint", "")).resolve(strict=True)
        and value.get("selected_checkpoint_sha256")
        == selection.get("selected_checkpoint_sha256")
        and value.get("model_config_sha256") == sha256_file(model_config)
        and Path(value.get("model_config", "")).resolve() == model_config
        and Path(value.get("codec", "")).resolve() == codec
        and value.get("codec_fingerprint")
        == selection.get("codec", {}).get("fingerprint")
        and int(value.get("ode_steps", -1)) == 20
        and math.isclose(float(value.get("cfg_scale", math.nan)), 1.0)
        and int(value.get("max_plan_tokens", -1)) == 512
        and int(value.get("seed", -1)) == 42
        and batch_size in FORMAL_BATCH_SIZES_PER_RANK
        and batch_size_path.parent == resolved.parent
        and batch_size_value == expected_batch_size_value
        and batch_size_record
        == {
            "path": str(batch_size_path),
            "sha256": sha256_file(batch_size_path),
            "value": expected_batch_size_value,
        }
        and value.get("per_row_seed_contract")
        == "blake2b_pair_id_namespace_rank_batch_invariant_v1"
        and Path(validation_index.get("path", "")).resolve()
        == Path(selected_validation_index.get("path", "")).resolve(strict=True)
        and validation_index.get("sha256")
        == selected_validation_index.get("sha256")
        and int(validation_index.get("rows", -1)) == 20_000
        and validation_index.get("split") == "validation"
        and row_selection.get("contract")
        == "sha256_ranked_operation_bucket_5x2x100_v1"
        and int(row_selection.get("rows", -1)) == CALIBRATION_ROWS
        and int(row_selection.get("per_operation_bucket_cell", -1))
        == CALIBRATION_PER_CELL
        and set(row_selection.get("by_operation_bucket") or {})
        == {f"{bucket}:{operation}" for bucket in (432, 648) for operation in OPERATIONS}
        and all(
            int(count) == CALIBRATION_PER_CELL
            for count in (row_selection.get("by_operation_bucket") or {}).values()
        )
        and len(str(row_selection.get("pair_ordinal_sha256") or "")) == 64
        and value.get("thresholds") == derived_thresholds
        and stored_summaries == recomputed_summaries
        and value.get("quality_checks") == derived_quality_checks
        and value.get("thresholds_sha256")
        == _canonical_sha256(derived_thresholds)
        and value.get("metric_summaries_sha256")
        == _canonical_sha256(stored_summaries)
        and value.get("run_contract_sha256") == sha256_file(run_contract_path)
        and value.get("row_records_sha256") == sha256_file(row_records_path)
        and run_contract.get("phase") == "calibration"
        and run_contract.get("evaluation_contract") == EVALUATION_CONTRACT
        and run_contract.get("checkpoint_selection_sha256") == selection_sha
        and run_contract.get("index", {}).get("sha256")
        == validation_index.get("sha256")
        and run_contract.get("row_selection") == row_selection
        and run_contract.get("independent_content_metric_assets")
        == dict(independent_content_assets)
        and run_contract.get("independent_content_metric_contract")
        == INDEPENDENT_CONTENT_METRIC_CONTRACT
        and run_contract.get("unchanged_source_demix") == _demix_contract()
        and int(run_contract.get("ode_steps", -1)) == 20
        and math.isclose(float(run_contract.get("cfg_scale", math.nan)), 1.0)
        and int(run_contract.get("max_plan_tokens", -1)) == 512
        and int(run_contract.get("seed", -1)) == 42
        and int(run_contract.get("batch_size_per_rank", -1)) == batch_size
        and run_contract.get("batch_size_certification") == batch_size_record
        and value.get("source_sha256") == dict(source_hashes)
        and value.get("independent_content_metric_assets")
        == dict(independent_content_assets)
        and value.get("independent_content_metric_contract")
        == INDEPENDENT_CONTENT_METRIC_CONTRACT
        and value.get("unchanged_source_demix") == _demix_contract()
        and value.get("frozen_vae_config_sha256") == FROZEN_VAE_CONFIG_SHA256
        and value.get("frozen_vae_checkpoint_sha256")
        == FROZEN_VAE_CHECKPOINT_SHA256
        and value.get("frozen_qwen_runtime") == verify_frozen_qwen_runtime()
        and _calibration_checks_are_closed(
            value.get("checks"), derived_quality_checks
        )
    ):
        raise RuntimeError("audio E2E calibration is stale or did not pass")
    return value


def _listening_ordinals(
    layout: Sequence[Mapping[str, Any]], count: int
) -> set[int]:
    cells: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
    for row in layout:
        key = (str(row["operation"]), int(row["latent_bucket_frames"]))
        digest = hashlib.sha256(
            f"audio-e2e-listening-v1:{row['pair_id']}".encode("utf-8")
        ).hexdigest()
        cells[key].append((digest, int(row["pair_ordinal"])))
    return {
        ordinal
        for values in cells.values()
        for _, ordinal in sorted(values)[: int(count)]
    }


def _audio_artifact_is_bound(row: Mapping[str, Any]) -> bool:
    path_value = row.get("edited_foa_path")
    expected_sha = row.get("edited_foa_sha256")
    if not isinstance(path_value, str) or not path_value or not isinstance(
        expected_sha, str
    ):
        return False
    try:
        path = Path(path_value).resolve(strict=True)
    except OSError:
        return False
    return expected_sha == sha256_file(path)


def _structural_report(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_rows: int,
    listening_ordinals: set[int],
    save_all_audio: bool,
    shared_transformer_same_object: bool,
    old_sceneplan_model_input: bool,
) -> dict[str, Any]:
    successful = [row for row in records if row.get("status") == "ok"]
    listening_rows = [
        row
        for row in successful
        if int(row.get("pair_ordinal", -1)) in listening_ordinals
    ]
    return {
        "rows_exact": len(records) == int(expected_rows),
        "all_rows_successful": len(successful) == len(records),
        "all_model_contracts_latest": all(
            row.get("model_input_contract") == MODEL_INPUT_CONTRACT
            for row in records
        ),
        "all_generated_plans_grammar_legal": all(
            row.get("metrics", {}).get("plan_grammar_legal") == 1.0
            for row in successful
        ),
        "all_independent_content_metrics_post_inference": all(
            row.get("independent_content_metric_contract")
            == INDEPENDENT_CONTENT_METRIC_CONTRACT
            and row.get("model_input_contract", {}).get(
                "independent_content_metrics_are_post_inference_only"
            )
            is True
            for row in successful
        ),
        "all_saved_audio_bound": (
            not bool(save_all_audio)
            or all(_audio_artifact_is_bound(row) for row in successful)
        ),
        "all_listening_audio_bound": (
            len(listening_rows) == len(listening_ordinals)
            and all(_audio_artifact_is_bound(row) for row in listening_rows)
        ),
        "shared_transformer_same_object": bool(shared_transformer_same_object),
        "old_sceneplan_model_input": bool(old_sceneplan_model_input),
        "source_audio_encoded_rows": len(successful),
        "ar_free_rows": len(successful),
        "dit_sampled_rows": len(successful),
        "decoded_audio_rows": len(successful),
    }


def _evaluation_checks(
    structural: Mapping[str, Any],
    quality_checks: Mapping[str, bool],
    *,
    expected_rows: int,
) -> dict[str, bool]:
    return {
        **{
            key: bool(value is False if key == "old_sceneplan_model_input" else value)
            for key, value in structural.items()
            if not key.endswith("_rows")
        },
        "all_stage_row_counts_exact": all(
            int(structural[key]) == int(expected_rows)
            for key in (
                "source_audio_encoded_rows",
                "ar_free_rows",
                "dit_sampled_rows",
                "decoded_audio_rows",
            )
        ),
        **{str(key): bool(value) for key, value in quality_checks.items()},
    }


def _validated_rank_records(
    output_dir: Path,
    *,
    run_contract: Mapping[str, Any],
    layout: Sequence[Mapping[str, Any]],
    selected: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected_values = [int(value) for value in selected]
    selected_set = set(selected_values)
    layout_by_ordinal = {
        int(row["pair_ordinal"]): row
        for row in layout
        if int(row["pair_ordinal"]) in selected_set
    }
    contract_sha = _canonical_sha256(run_contract)
    records: list[dict[str, Any]] = []
    rank_final_inventory = []
    for rank_value in range(WORLD_SIZE):
        expected_rank_ordinals = [
            ordinal
            for bucket in (432, 648)
            for ordinal in [
                value
                for index_in_bucket, value in enumerate(
                    row["pair_ordinal"]
                    for row in layout
                    if int(row["latent_bucket_frames"]) == bucket
                    and int(row["pair_ordinal"]) in selected_set
                )
                if index_in_bucket % WORLD_SIZE == rank_value
            ]
        ]
        rank_final_path = (
            output_dir / "shards" / f"rank-{rank_value}.FINAL.json"
        ).resolve(strict=True)
        rank_final = json.loads(rank_final_path.read_text(encoding="utf-8"))
        shard_paths = sorted(
            (output_dir / "shards" / f"rank-{rank_value}").glob("batch-*.json")
        )
        current_inventory = [
            {"name": path.name, "sha256": sha256_file(path)}
            for path in shard_paths
        ]
        if not (
            rank_final.get("contract_sha256") == contract_sha
            and int(rank_final.get("rank", -1)) == rank_value
            and int(rank_final.get("physical_gpu", -1)) == rank_value + 3
            and int(rank_final.get("rows", -1)) == len(expected_rank_ordinals)
            and int(rank_final.get("batches", -1)) == len(shard_paths)
            and rank_final.get("pair_ordinal_sha256")
            == _ordinal_sha256(expected_rank_ordinals)
            and rank_final.get("batch_shards") == current_inventory
            and rank_final.get("batch_shard_inventory_sha256")
            == _canonical_sha256(current_inventory)
        ):
            raise RuntimeError("audio E2E rank final contract changed")
        observed_rank_ordinals = []
        for expected_batch, shard_path in enumerate(shard_paths):
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            shard_ordinals = [int(value) for value in shard.get("pair_ordinals", [])]
            if not (
                shard.get("contract_sha256") == contract_sha
                and int(shard.get("rank", -1)) == rank_value
                and int(shard.get("physical_gpu", -1)) == rank_value + 3
                and int(shard.get("batch", -1)) == expected_batch
                and int(shard.get("latent_bucket_frames", -1)) in (432, 648)
                and len(shard_ordinals) == len(shard.get("records", []))
                and all(value in layout_by_ordinal for value in shard_ordinals)
            ):
                raise RuntimeError("audio E2E batch contract changed")
            _validate_batch_records(
                shard["records"],
                [layout_by_ordinal[value] for value in shard_ordinals],
                bucket=int(shard["latent_bucket_frames"]),
            )
            observed_rank_ordinals.extend(shard_ordinals)
            records.extend(shard["records"])
        if observed_rank_ordinals != expected_rank_ordinals:
            raise RuntimeError("audio E2E rank shards changed ordinal coverage")
        rank_final_inventory.append(
            {"path": str(rank_final_path), "sha256": sha256_file(rank_final_path)}
        )
    records.sort(key=lambda row: int(row["pair_ordinal"]))
    if [int(row["pair_ordinal"]) for row in records] != selected_values:
        raise RuntimeError("audio E2E rank shards do not exactly cover selected rows")
    return records, rank_final_inventory


def _validated_batch_size_record(
    record: Mapping[str, Any],
    *,
    phase: str,
    batch_size: int,
    expected_parent: Path,
) -> dict[str, Any]:
    expected_value = _batch_size_sidecar_value(
        phase=phase, batch_size=int(batch_size)
    )
    path = Path(str(record.get("path") or "")).resolve(strict=True)
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected_record = {
        "path": str(path),
        "sha256": sha256_file(path),
        "value": expected_value,
    }
    if (
        path.parent != expected_parent.resolve(strict=True)
        or observed != expected_value
        or dict(record) != expected_record
    ):
        raise RuntimeError("audio E2E batch-size certification changed")
    return expected_record


def _require_replayed_final_derivations(
    final: Mapping[str, Any],
    *,
    structural: Mapping[str, Any],
    checks: Mapping[str, bool],
    summaries: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> None:
    """Reject stored PASS fields that disagree with artifact-derived values."""

    if not (
        final.get("structural") == dict(structural)
        and final.get("checks") == dict(checks)
        and final.get("metric_summaries") == dict(summaries)
        and final.get("thresholds") == dict(thresholds)
        and checks
        and all(bool(value) for value in checks.values())
    ):
        raise RuntimeError("audio E2E FINAL derived checks changed or did not pass")


def validate_audio_e2e_final(
    path: str | Path,
    *,
    expected_sha256: str,
    write_seal: bool = False,
) -> dict[str, Any]:
    """Fully replay one successful formal E2E report and verify/write its seal.

    This verifier is deliberately CPU-only.  It reopens the selected checkpoint
    promotion, frozen index and latent inventories, metric rows, rank shards,
    saved listening/audio files, calibration, metric assets, and every derived
    quality/structural check before accepting ``FINAL.json``.
    """

    final_path = Path(path).expanduser().resolve(strict=True)
    final_sha = sha256_file(final_path)
    if final_sha != str(expected_sha256):
        raise RuntimeError("audio E2E FINAL SHA256 changed")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if not isinstance(final, dict):
        raise RuntimeError("audio E2E FINAL is not a JSON object")
    phase = str(final.get("phase") or "")
    if phase not in {"calibration", "test"}:
        raise RuntimeError("audio E2E FINAL phase is invalid")
    expected_index_rows = 20_000 if phase == "calibration" else 5_000
    expected_selected_rows = CALIBRATION_ROWS if phase == "calibration" else 5_000
    expected_split = "validation" if phase == "calibration" else "test"

    contract_path = Path(str(final.get("run_contract") or "")).resolve(strict=True)
    row_path = Path(str(final.get("row_records") or "")).resolve(strict=True)
    if contract_path.parent != final_path.parent or row_path.parent != final_path.parent:
        raise RuntimeError("audio E2E FINAL references artifacts outside its run")
    run_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if final.get("run_contract_sha256") != sha256_file(contract_path):
        raise RuntimeError("audio E2E run contract SHA256 changed")
    if final.get("row_records_sha256") != sha256_file(row_path):
        raise RuntimeError("audio E2E row-record SHA256 changed")

    selection_path = Path(
        str(final.get("joint_checkpoint_selection") or "")
    ).resolve(strict=True)
    selection_sha = sha256_file(selection_path)
    if selection_sha != final.get("joint_checkpoint_selection_sha256"):
        raise RuntimeError("audio E2E checkpoint-selection SHA256 changed")
    selection = validate_published_joint_selection(
        selection_path, expected_sha256=selection_sha
    )
    checkpoint = Path(str(final.get("selected_checkpoint") or "")).resolve(
        strict=True
    )
    model_config = Path(str(final.get("model_config") or "")).resolve(strict=True)
    codec_path = Path(str(final.get("codec") or "")).resolve(strict=True)
    codec = ModelScenePlanCodecV4(codec_path)

    index_record = dict(run_contract.get("index") or {})
    index = Path(str(index_record.get("path") or "")).resolve(strict=True)
    replayed_index = _index_summary(
        index, str(index_record.get("sha256") or ""), expected_index_rows
    )
    if replayed_index != index_record or replayed_index.get("split") != expected_split:
        raise RuntimeError("audio E2E frozen index replay changed")
    layout = _layout(
        index, expected_rows=expected_index_rows, expected_split=expected_split
    )
    selected, row_selection = _select_ordinals(layout, phase=phase)
    if run_contract.get("row_selection") != row_selection:
        raise RuntimeError("audio E2E row selection changed")

    batch_size = int(final.get("batch_size_per_rank", -1))
    batch_size_record = _validated_batch_size_record(
        dict(final.get("batch_size_certification") or {}),
        phase=phase,
        batch_size=batch_size,
        expected_parent=final_path.parent,
    )
    source_hashes = _source_sha256()
    independent_content_assets = verify_independent_content_metric_assets()
    frozen_m2d_clap_assets = verify_editing_m2d_clap_assets(require_text=False)
    qwen_runtime = verify_frozen_qwen_runtime()
    if not (
        run_contract.get("schema") == SCHEMA
        and int(run_contract.get("schema_version", -1)) == SCHEMA_VERSION
        and run_contract.get("evaluation_contract") == EVALUATION_CONTRACT
        and run_contract.get("phase") == phase
        and run_contract.get("physical_gpus") == PHYSICAL_GPUS
        and int(run_contract.get("world_size", -1)) == WORLD_SIZE
        and run_contract.get("cuda_visible_devices") == VISIBLE_GPUS
        and run_contract.get("cuda_device_order") == "PCI_BUS_ID"
        and run_contract.get("gpu_topology") == selection.get("gpu_topology")
        and run_contract.get("latest_route") == LATEST_ROUTE_CONTRACT
        and Path(str(run_contract.get("checkpoint_selection") or "")).resolve()
        == selection_path
        and run_contract.get("checkpoint_selection_sha256") == selection_sha
        and Path(str(run_contract.get("checkpoint") or "")).resolve() == checkpoint
        and run_contract.get("checkpoint_sha256") == sha256_file(checkpoint)
        and int(run_contract.get("checkpoint_step", -1))
        == int(selection.get("selected_checkpoint_step", -2))
        and Path(str(run_contract.get("model_config") or "")).resolve()
        == model_config
        and run_contract.get("model_config_sha256") == sha256_file(model_config)
        and Path(str(run_contract.get("codec") or "")).resolve() == codec_path
        and run_contract.get("codec_fingerprint") == codec.fingerprint
        and int(run_contract.get("batch_size_per_rank", -1)) == batch_size
        and run_contract.get("batch_size_certification") == batch_size_record
        and int(run_contract.get("ode_steps", -1)) == 20
        and math.isclose(float(run_contract.get("cfg_scale", math.nan)), 1.0)
        and int(run_contract.get("max_plan_tokens", -1)) == 512
        and int(run_contract.get("seed", -1)) == 42
        and run_contract.get("per_row_seed_contract")
        == "blake2b_pair_id_namespace_rank_batch_invariant_v1"
        and run_contract.get("source_sha256") == source_hashes
        and run_contract.get("independent_content_metric_contract")
        == INDEPENDENT_CONTENT_METRIC_CONTRACT
        and run_contract.get("independent_content_metric_assets")
        == independent_content_assets
        and run_contract.get("unchanged_source_demix") == _demix_contract()
        and Path(str(run_contract.get("frozen_vae_config") or "")).resolve()
        == FROZEN_VAE_CONFIG.resolve(strict=True)
        and run_contract.get("frozen_vae_config_sha256")
        == FROZEN_VAE_CONFIG_SHA256
        and Path(str(run_contract.get("frozen_vae_checkpoint") or "")).resolve()
        == FROZEN_VAE_CHECKPOINT.resolve(strict=True)
        and run_contract.get("frozen_vae_checkpoint_sha256")
        == FROZEN_VAE_CHECKPOINT_SHA256
        and run_contract.get("frozen_qwen_runtime") == qwen_runtime
        and run_contract.get("source_semantic_mode") == "m2d_audio_caption_aux"
        and run_contract.get("frozen_m2d_clap_assets")
        == frozen_m2d_clap_assets
        and selection.get("source_semantic", {}).get("mode")
        == "m2d_audio_caption_aux"
        and selection.get("source_semantic", {}).get("caption_model_input")
        is False
    ):
        raise RuntimeError("audio E2E run contract replay changed")
    if (
        sha256_file(FROZEN_VAE_CONFIG.resolve(strict=True))
        != FROZEN_VAE_CONFIG_SHA256
        or sha256_file(FROZEN_VAE_CHECKPOINT.resolve(strict=True))
        != FROZEN_VAE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("audio E2E frozen VAE assets changed")

    rows = _validated_metric_row_records(
        row_path,
        index=index,
        row_selection=row_selection,
        expected_rows=expected_selected_rows,
    )
    shard_rows, rank_final_inventory = _validated_rank_records(
        final_path.parent,
        run_contract=run_contract,
        layout=layout,
        selected=selected,
    )
    if shard_rows != rows:
        raise RuntimeError("audio E2E global rows differ from sealed rank shards")

    calibration_binding: dict[str, Any] | None
    if phase == "calibration":
        if run_contract.get("calibration") is not None or final.get(
            "calibration"
        ) is not None:
            raise RuntimeError("calibration audio E2E unexpectedly references calibration")
        calibration_record = dict(final.get("calibration_artifact") or {})
        calibration_path = Path(
            str(calibration_record.get("path") or "")
        ).resolve(strict=True)
        calibration_sha = sha256_file(calibration_path)
        if (
            calibration_path.parent != final_path.parent
            or calibration_record
            != {"path": str(calibration_path), "sha256": calibration_sha}
        ):
            raise RuntimeError("audio E2E calibration artifact binding changed")
        calibration = _validate_calibration(
            calibration_path,
            expected_sha=calibration_sha,
            selection=selection,
            selection_path=selection_path,
            selection_sha=selection_sha,
            model_config=model_config,
            codec=codec_path,
            source_hashes=source_hashes,
            independent_content_assets=independent_content_assets,
        )
        if Path(str(calibration.get("full_report") or "")).resolve() != final_path:
            raise RuntimeError("audio E2E calibration/full-report binding changed")
        thresholds, quality_checks = _calibration_thresholds(
            _all_metric_summaries(rows)
        )
        calibration_binding = calibration_record
    else:
        calibration_record = dict(run_contract.get("calibration") or {})
        calibration_path = Path(
            str(calibration_record.get("path") or "")
        ).resolve(strict=True)
        calibration_sha = sha256_file(calibration_path)
        if (
            calibration_record.get("sha256") != calibration_sha
            or calibration_record.get("contract") != CALIBRATION_CONTRACT
            or final.get("calibration") != calibration_record
        ):
            raise RuntimeError("audio E2E test calibration binding changed")
        calibration = _validate_calibration(
            calibration_path,
            expected_sha=calibration_sha,
            selection=selection,
            selection_path=selection_path,
            selection_sha=selection_sha,
            model_config=model_config,
            codec=codec_path,
            source_hashes=source_hashes,
            independent_content_assets=independent_content_assets,
        )
        _require_matching_calibration_batch_size(
            calibration, test_batch_size=batch_size
        )
        thresholds = calibration["thresholds"]
        quality_checks = _test_threshold_checks(
            _all_metric_summaries(rows), thresholds
        )
        calibration_binding = calibration_record

    summaries = _all_metric_summaries(rows)
    listening = _listening_ordinals(
        [
            row
            for row in layout
            if int(row["pair_ordinal"]) in set(int(value) for value in selected)
        ],
        int(run_contract.get("listening_rows_per_operation_bucket", -1)),
    )
    expected_listening = (
        int(run_contract.get("listening_rows_per_operation_bucket", -1))
        * len(OPERATIONS)
        * 2
    )
    if len(listening) != expected_listening:
        raise RuntimeError("audio E2E listening selection replay changed")
    structural = _structural_report(
        rows,
        expected_rows=expected_selected_rows,
        listening_ordinals=listening,
        save_all_audio=bool(run_contract.get("save_all_audio")),
        shared_transformer_same_object=bool(
            final.get("shared_transformer_same_object")
        ),
        old_sceneplan_model_input=bool(final.get("old_sceneplan_model_input")),
    )
    checks = _evaluation_checks(
        structural, quality_checks, expected_rows=expected_selected_rows
    )
    if not (
        final.get("schema") == SCHEMA
        and int(final.get("schema_version", -1)) == SCHEMA_VERSION
        and final.get("status") == "PASS"
        and final.get("execution_complete") is True
        and final.get("evaluation_passed") is True
        and final.get("evaluation_contract") == EVALUATION_CONTRACT
        and int(final.get("rows", -1)) == expected_selected_rows
        and int(final.get("successful_rows", -1)) == expected_selected_rows
        and int(final.get("error_rows", -1)) == 0
        and int(final.get("source_audio_encoded_rows", -1))
        == expected_selected_rows
        and int(final.get("ar_free_rows", -1)) == expected_selected_rows
        and int(final.get("dit_sampled_rows", -1)) == expected_selected_rows
        and int(final.get("decoded_audio_rows", -1)) == expected_selected_rows
        and final.get("physical_gpus") == PHYSICAL_GPUS
        and int(final.get("world_size", -1)) == WORLD_SIZE
        and final.get("joint_checkpoint_selection_sha256") == selection_sha
        and Path(str(final.get("selected_checkpoint") or "")).resolve()
        == checkpoint
        and final.get("selected_checkpoint_sha256") == sha256_file(checkpoint)
        and int(final.get("selected_checkpoint_step", -1))
        == int(selection.get("selected_checkpoint_step", -2))
        and final.get("pipeline_contract") == EDITING_PIPELINE_CONTRACT
        and final.get("shared_transformer_same_object") is True
        and final.get("old_sceneplan_model_input") is False
        and final.get("source_caption_model_input") is False
        and final.get("source_sha256") == source_hashes
        and final.get("model_config_sha256") == sha256_file(model_config)
        and final.get("codec_fingerprint") == codec.fingerprint
        and final.get("frozen_vae_config_sha256") == FROZEN_VAE_CONFIG_SHA256
        and final.get("frozen_vae_checkpoint_sha256")
        == FROZEN_VAE_CHECKPOINT_SHA256
        and final.get("frozen_qwen_runtime") == qwen_runtime
        and final.get("source_semantic_mode") == "m2d_audio_caption_aux"
        and final.get("frozen_m2d_clap_assets") == frozen_m2d_clap_assets
        and final.get("independent_content_metric_contract")
        == INDEPENDENT_CONTENT_METRIC_CONTRACT
        and final.get("independent_content_metric_assets")
        == independent_content_assets
        and final.get("unchanged_source_demix") == _demix_contract()
        and final.get("batch_size_certification") == batch_size_record
    ):
        raise RuntimeError("audio E2E FINAL replay changed or did not pass")
    _require_replayed_final_derivations(
        final,
        structural=structural,
        checks=checks,
        summaries=summaries,
        thresholds=thresholds,
    )

    seal = {
        "schema": FINAL_SEAL_SCHEMA,
        "schema_version": 1,
        "status": "SEALED",
        "seal_contract": "full_cpu_artifact_and_gate_replay_v1",
        "phase": phase,
        "final": {"path": str(final_path), "sha256": final_sha},
        "run_contract": {
            "path": str(contract_path),
            "sha256": sha256_file(contract_path),
        },
        "row_records": {"path": str(row_path), "sha256": sha256_file(row_path)},
        "rank_finals": rank_final_inventory,
        "rank_final_inventory_sha256": _canonical_sha256(rank_final_inventory),
        "checkpoint_selection": {
            "path": str(selection_path),
            "sha256": selection_sha,
        },
        "selected_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "index": index_record,
        "calibration": calibration_binding,
        "batch_size_certification": batch_size_record,
        "checks_sha256": _canonical_sha256(checks),
        "metric_summaries_sha256": _canonical_sha256(summaries),
        "source_sha256": source_hashes,
        "frozen_m2d_clap_assets": frozen_m2d_clap_assets,
    }
    seal_path = final_path.parent / "SEALED.json"
    if write_seal:
        _atomic_json(seal_path, seal)
    else:
        existing_seal_path = seal_path.resolve(strict=True)
        existing_seal = json.loads(existing_seal_path.read_text(encoding="utf-8"))
        if existing_seal != seal:
            raise RuntimeError("audio E2E SEALED artifact changed")
    if json.loads(seal_path.read_text(encoding="utf-8")) != seal:
        raise RuntimeError("audio E2E SEALED publication is not atomic")
    return seal


def main() -> int:
    args = _parse_args()
    if not (
        int(args.batch_size) in FORMAL_BATCH_SIZES_PER_RANK
        and int(args.ode_steps) == 20
        and math.isclose(float(args.cfg_scale), 1.0)
        and int(args.max_plan_tokens) == 512
        and int(args.seed) == 42
        and int(args.listening_rows_per_cell) >= 0
        and (args.phase == "test") == (int(args.expected_index_rows) == 5_000)
        and (args.phase == "calibration") == (int(args.expected_index_rows) == 20_000)
    ):
        raise ValueError("formal audio E2E settings changed")
    rank, _, device = _distributed()
    topology = _rank0_audit(_gpu_topology, rank=rank, device=device)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(int(args.seed) + rank)
    torch.cuda.manual_seed_all(int(args.seed) + rank)

    selection_path = args.checkpoint_selection.expanduser().resolve(strict=True)
    selection_sha = sha256_file(selection_path)
    if selection_sha != str(args.checkpoint_selection_sha256):
        raise RuntimeError("joint checkpoint selection SHA256 changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not (
        selection.get("schema") == JOINT_SELECTION_SCHEMA
        and selection.get("status") == "PASS"
        and selection.get("selection_contract") == JOINT_SELECTION_CONTRACT
    ):
        raise RuntimeError("joint checkpoint was not independently promoted")
    checkpoint = Path(selection["selected_checkpoint"]).resolve(strict=True)
    model_config = args.model_config.expanduser().resolve(strict=True)
    codec_path = args.codec.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = _source_sha256()
    independent_content_assets = _broadcast(
        verify_independent_content_metric_assets() if rank == 0 else None,
        rank=rank,
        device=device,
    )
    calibration = None
    calibration_sha = None
    if args.phase == "test":
        if args.calibration is None or args.calibration_sha256 is None:
            raise RuntimeError("test audio E2E requires a pinned calibration")
        calibration_sha = str(args.calibration_sha256)
        calibration = _validate_calibration(
            args.calibration,
            expected_sha=calibration_sha,
            selection=selection,
            selection_path=selection_path,
            selection_sha=selection_sha,
            model_config=model_config,
            codec=codec_path,
            source_hashes=source_hashes,
            independent_content_assets=independent_content_assets,
        )
        _require_matching_calibration_batch_size(
            calibration, test_batch_size=int(args.batch_size)
        )

    batch_size_sidecar_value = _batch_size_sidecar_value(
        phase=args.phase, batch_size=int(args.batch_size)
    )
    batch_size_sidecar_path = output_dir / "BATCH_SIZE.json"
    if rank == 0:
        if batch_size_sidecar_path.exists():
            existing_batch_size = json.loads(
                batch_size_sidecar_path.read_text(encoding="utf-8")
            )
            if existing_batch_size != batch_size_sidecar_value:
                raise RuntimeError("existing audio E2E batch-size sidecar changed")
        else:
            _atomic_json(batch_size_sidecar_path, batch_size_sidecar_value)
    dist.barrier()
    batch_size_record = {
        "path": str(batch_size_sidecar_path.resolve(strict=True)),
        "sha256": sha256_file(batch_size_sidecar_path),
        "value": batch_size_sidecar_value,
    }

    # Test isolation is structural, not merely procedural: the test SQLite is
    # neither resolved, hashed, nor opened until the frozen validation
    # calibration has been fully replay-validated above.  Calibration runs do
    # not have this dependency and open only their validation index here.
    index = args.index.expanduser().resolve(strict=True)
    index_summary = _index_summary(
        index, args.index_sha256, int(args.expected_index_rows)
    )
    expected_split = "validation" if args.phase == "calibration" else "test"
    if index_summary["split"] != expected_split:
        raise RuntimeError("audio E2E phase/index split mismatch")
    layout = _layout(
        index,
        expected_rows=int(args.expected_index_rows),
        expected_split=expected_split,
    )
    selected, selection_summary = _select_ordinals(layout, phase=args.phase)
    selected_set = set(selected)

    pipeline, load_report = load_sceneplan_transfusion_editing_pipeline(
        checkpoint=checkpoint,
        checkpoint_selection=selection_path,
        checkpoint_selection_sha256=selection_sha,
        model_config=model_config,
        codec=codec_path,
        device=device,
        load_audio_autoencoder=True,
    )
    if not (
        load_report.shared_transformer_same_object
        and load_report.old_sceneplan_input is False
        and load_report.source_semantic_mode == "m2d_audio_caption_aux"
        and load_report.source_caption_model_input is False
        and isinstance(load_report.frozen_m2d_clap_assets, dict)
        and load_report.frozen_m2d_clap_assets.get("license_scope")
        == "internal_noncommercial_evaluation_only"
        and load_report.frozen_vae_config_sha256 == FROZEN_VAE_CONFIG_SHA256
        and load_report.frozen_vae_checkpoint_sha256
        == FROZEN_VAE_CHECKPOINT_SHA256
    ):
        raise RuntimeError("real-audio Editing pipeline identity changed")
    content_evaluator = IndependentEditingContentEvaluator(
        device=device, device_index=int(device.index)
    )
    codec = pipeline.codec
    run_contract = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "evaluation_contract": EVALUATION_CONTRACT,
        "phase": args.phase,
        "physical_gpus": PHYSICAL_GPUS,
        "world_size": WORLD_SIZE,
        "cuda_visible_devices": VISIBLE_GPUS,
        "cuda_device_order": "PCI_BUS_ID",
        "gpu_topology": topology,
        "latest_route": dict(LATEST_ROUTE_CONTRACT),
        "checkpoint_selection": str(selection_path),
        "checkpoint_selection_sha256": selection_sha,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(selection["selected_checkpoint_step"]),
        "model_config": str(model_config),
        "model_config_sha256": sha256_file(model_config),
        "codec": str(codec_path),
        "codec_fingerprint": codec.fingerprint,
        "index": index_summary,
        "row_selection": selection_summary,
        "batch_size_per_rank": int(args.batch_size),
        "batch_size_certification": batch_size_record,
        "ode_steps": int(args.ode_steps),
        "cfg_scale": float(args.cfg_scale),
        "max_plan_tokens": int(args.max_plan_tokens),
        "seed": int(args.seed),
        "per_row_seed_contract": "blake2b_pair_id_namespace_rank_batch_invariant_v1",
        "save_all_audio": bool(args.save_all_audio),
        "listening_rows_per_operation_bucket": int(args.listening_rows_per_cell),
        "frozen_vae_config": load_report.frozen_vae_config,
        "frozen_vae_config_sha256": load_report.frozen_vae_config_sha256,
        "frozen_vae_checkpoint": load_report.frozen_vae_checkpoint,
        "frozen_vae_checkpoint_sha256": load_report.frozen_vae_checkpoint_sha256,
        "frozen_qwen_runtime": load_report.frozen_qwen_runtime,
        "source_semantic_mode": load_report.source_semantic_mode,
        "frozen_m2d_clap_assets": load_report.frozen_m2d_clap_assets,
        "unchanged_source_demix": _demix_contract(),
        "independent_content_metric_contract": INDEPENDENT_CONTENT_METRIC_CONTRACT,
        "independent_content_metric_assets": independent_content_assets,
        "calibration": (
            None
            if calibration is None
            else {
                "path": str(Path(args.calibration).resolve(strict=True)),
                "sha256": calibration_sha,
                "contract": calibration["calibration_contract"],
            }
        ),
        "source_sha256": source_hashes,
    }
    contract_sha = _canonical_sha256(run_contract)
    contract_path = output_dir / "CONTRACT.json"
    if rank == 0:
        if contract_path.exists():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != run_contract:
                raise RuntimeError("existing audio E2E run contract changed")
        else:
            _atomic_json(contract_path, run_contract)
        history = output_dir / "attempt_history"
        terminal_names = ["FINAL.json", "FAILED.json", "SEALED.json"]
        if args.phase == "calibration":
            terminal_names.append("CALIBRATION.json")
        for terminal_name in terminal_names:
            terminal = output_dir / terminal_name
            if terminal.is_file():
                history.mkdir(exist_ok=True)
                digest = sha256_file(terminal)[:16]
                archived = history / (
                    f"{terminal.stem}.{time.time_ns()}.{digest}{terminal.suffix}"
                )
                os.replace(terminal, archived)
    dist.barrier()

    local_ordinals = [
        ordinal
        for bucket in (432, 648)
        for ordinal in [
            value
            for index_in_bucket, value in enumerate(
                row["pair_ordinal"]
                for row in layout
                if int(row["latent_bucket_frames"]) == bucket
                and int(row["pair_ordinal"]) in selected_set
            )
            if index_in_bucket % WORLD_SIZE == rank
        ]
    ]
    base_dataset = ScenePlanTransfusionEditingDataset(
        index,
        tokenizer_spec=(
            pipeline.diffusion.conditioner.conditioners["prompt"].tokenizer,
            512,
        ),
        expected_num_samples=len(local_ordinals),
        index_num_samples=int(args.expected_index_rows),
        expected_index_sha256=args.index_sha256,
        sample_ordinals=local_ordinals,
        latent_crop_length=648,
        require_frozen=True,
        verify_tensor_hashes_on_access=True,
    )
    joint_dataset = ScenePlanTransfusionEditingJointDataset(
        base_dataset, codec=codec, max_plan_tokens=1024
    )
    truth_resolver = OfflineTruthResolver(index)
    listening = _listening_ordinals(
        [
            row
            for row in layout
            if int(row["pair_ordinal"]) in selected_set
        ],
        int(args.listening_rows_per_cell),
    )
    expected_listening = int(args.listening_rows_per_cell) * len(OPERATIONS) * 2
    if len(listening) != expected_listening:
        raise RuntimeError("audio E2E listening selection does not cover every cell")
    shard_root = output_dir / "shards" / f"rank-{rank}"
    shard_root.mkdir(parents=True, exist_ok=True)
    batch_number = 0
    for bucket in (432, 648):
        local_indices = list(joint_dataset.length_bucket_indices().get(bucket, ()))
        for indices in _chunks(local_indices, int(args.batch_size)):
            samples = [joint_dataset[index_value] for index_value in indices]
            expected_ordinals = [int(sample[1]["pair_ordinal"]) for sample in samples]
            shard_path = shard_root / f"batch-{batch_number:06d}.json"
            batch_number += 1
            if shard_path.is_file():
                existing = json.loads(shard_path.read_text(encoding="utf-8"))
                if not (
                    existing.get("contract_sha256") == contract_sha
                    and int(existing.get("rank", -1)) == rank
                    and int(existing.get("physical_gpu", -1)) == rank + 3
                    and int(existing.get("batch", -1)) == batch_number - 1
                    and int(existing.get("latent_bucket_frames", -1)) == bucket
                    and existing.get("pair_ordinals") == expected_ordinals
                    and len(existing.get("records", [])) == len(expected_ordinals)
                ):
                    raise RuntimeError(f"stale audio E2E shard: {shard_path}")
                _validate_batch_records(
                    existing["records"],
                    [sample[1] for sample in samples],
                    bucket=bucket,
                )
                continue
            truths = [truth_resolver.row(ordinal) for ordinal in expected_ordinals]
            try:
                records = _process_batch(
                    pipeline=pipeline,
                    content_evaluator=content_evaluator,
                    codec=codec,
                    samples=samples,
                    truths=truths,
                    bucket=bucket,
                    device=device,
                    args=args,
                    output_dir=output_dir,
                    listening_ordinals=listening,
                )
            except Exception as batch_error:  # noqa: BLE001
                torch.cuda.empty_cache()
                if len(samples) == 1:
                    records = _error_records(samples, batch_error)
                else:
                    records = []
                    for sample, truth in zip(samples, truths):
                        try:
                            records.extend(
                                _process_batch(
                                    pipeline=pipeline,
                                    content_evaluator=content_evaluator,
                                    codec=codec,
                                    samples=[sample],
                                    truths=[truth],
                                    bucket=bucket,
                                    device=device,
                                    args=args,
                                    output_dir=output_dir,
                                    listening_ordinals=listening,
                                )
                            )
                        except Exception as row_error:  # noqa: BLE001
                            torch.cuda.empty_cache()
                            records.extend(_error_records([sample], row_error))
            _validate_batch_records(
                records,
                [sample[1] for sample in samples],
                bucket=bucket,
            )
            _atomic_json(
                shard_path,
                {
                    "contract_sha256": contract_sha,
                    "rank": rank,
                    "physical_gpu": rank + 3,
                    "batch": batch_number - 1,
                    "latent_bucket_frames": bucket,
                    "pair_ordinals": expected_ordinals,
                    "records": records,
                },
            )
            print(
                json.dumps(
                    {
                        "event": "audio_e2e_batch",
                        "phase": args.phase,
                        "rank": rank,
                        "batch": batch_number - 1,
                        "rows": len(records),
                        "errors": sum(row["status"] != "ok" for row in records),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    truth_resolver.close()
    batch_shards = [
        {
            "name": path.name,
            "sha256": sha256_file(path),
        }
        for path in sorted(shard_root.glob("batch-*.json"))
    ]
    if len(batch_shards) != batch_number:
        raise RuntimeError("audio E2E rank has an unexpected batch-shard inventory")
    _atomic_json(
        output_dir / "shards" / f"rank-{rank}.FINAL.json",
        {
            "contract_sha256": contract_sha,
            "rank": rank,
            "physical_gpu": rank + 3,
            "rows": len(local_ordinals),
            "batches": batch_number,
            "pair_ordinal_sha256": _ordinal_sha256(local_ordinals),
            "batch_shards": batch_shards,
            "batch_shard_inventory_sha256": _canonical_sha256(batch_shards),
        },
    )
    dist.barrier()

    if rank == 0:
        records = []
        layout_by_ordinal = {
            int(row["pair_ordinal"]): row
            for row in layout
            if int(row["pair_ordinal"]) in selected_set
        }
        for rank_value in range(WORLD_SIZE):
            expected_rank_ordinals = [
                ordinal
                for bucket in (432, 648)
                for ordinal in [
                    value
                    for index_in_bucket, value in enumerate(
                        row["pair_ordinal"]
                        for row in layout
                        if int(row["latent_bucket_frames"]) == bucket
                        and int(row["pair_ordinal"]) in selected_set
                    )
                    if index_in_bucket % WORLD_SIZE == rank_value
                ]
            ]
            rank_final = json.loads(
                (output_dir / "shards" / f"rank-{rank_value}.FINAL.json").read_text(
                    encoding="utf-8"
                )
            )
            shard_paths = sorted(
                (output_dir / "shards" / f"rank-{rank_value}").glob(
                    "batch-*.json"
                )
            )
            current_inventory = [
                {"name": path.name, "sha256": sha256_file(path)}
                for path in shard_paths
            ]
            if not (
                rank_final.get("contract_sha256") == contract_sha
                and int(rank_final.get("rank", -1)) == rank_value
                and int(rank_final.get("physical_gpu", -1)) == rank_value + 3
                and int(rank_final.get("rows", -1)) == len(expected_rank_ordinals)
                and int(rank_final.get("batches", -1)) == len(shard_paths)
                and rank_final.get("pair_ordinal_sha256")
                == _ordinal_sha256(expected_rank_ordinals)
                and rank_final.get("batch_shards") == current_inventory
                and rank_final.get("batch_shard_inventory_sha256")
                == _canonical_sha256(current_inventory)
            ):
                raise RuntimeError("audio E2E rank final contract changed")
            observed_rank_ordinals = []
            for expected_batch, shard_path in enumerate(shard_paths):
                shard = json.loads(shard_path.read_text(encoding="utf-8"))
                shard_ordinals = [int(value) for value in shard.get("pair_ordinals", [])]
                if not (
                    shard.get("contract_sha256") == contract_sha
                    and int(shard.get("rank", -1)) == rank_value
                    and int(shard.get("physical_gpu", -1)) == rank_value + 3
                    and int(shard.get("batch", -1)) == expected_batch
                    and int(shard.get("latent_bucket_frames", -1)) in (432, 648)
                    and len(shard_ordinals) == len(shard.get("records", []))
                ):
                    raise RuntimeError("audio E2E batch contract changed")
                expected_identity = [layout_by_ordinal[value] for value in shard_ordinals]
                _validate_batch_records(
                    shard["records"],
                    expected_identity,
                    bucket=int(shard["latent_bucket_frames"]),
                )
                observed_rank_ordinals.extend(shard_ordinals)
                records.extend(shard["records"])
            if observed_rank_ordinals != expected_rank_ordinals:
                raise RuntimeError("audio E2E rank shards changed ordinal coverage")
        records.sort(key=lambda row: int(row["pair_ordinal"]))
        if [int(row["pair_ordinal"]) for row in records] != selected:
            raise RuntimeError("audio E2E shards do not exactly cover selected rows")
        row_path = output_dir / "ROWS.jsonl"
        _atomic_jsonl(row_path, records)
        successful = [row for row in records if row["status"] == "ok"]
        summaries = _all_metric_summaries(successful)
        structural = _structural_report(
            records,
            expected_rows=len(selected),
            listening_ordinals=listening,
            save_all_audio=bool(args.save_all_audio),
            shared_transformer_same_object=(
                load_report.shared_transformer_same_object
            ),
            old_sceneplan_model_input=load_report.old_sceneplan_input,
        )
        if args.phase == "calibration":
            thresholds, quality_checks = _calibration_thresholds(summaries)
        else:
            thresholds = calibration["thresholds"]
            quality_checks = _test_threshold_checks(summaries, thresholds)
        checks = _evaluation_checks(
            structural, quality_checks, expected_rows=len(selected)
        )
        status = "PASS" if all(checks.values()) else "FAIL"
        final = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "execution_complete": True,
            "evaluation_passed": status == "PASS",
            "evaluation_contract": EVALUATION_CONTRACT,
            "phase": args.phase,
            "rows": len(records),
            "successful_rows": len(successful),
            "error_rows": len(records) - len(successful),
            "physical_gpus": PHYSICAL_GPUS,
            "world_size": WORLD_SIZE,
            "joint_checkpoint_selection": str(selection_path),
            "joint_checkpoint_selection_sha256": selection_sha,
            "selected_checkpoint": str(checkpoint),
            "selected_checkpoint_sha256": sha256_file(checkpoint),
            "selected_checkpoint_step": int(selection["selected_checkpoint_step"]),
            "pipeline_contract": EDITING_PIPELINE_CONTRACT,
            "shared_transformer_same_object": True,
            "old_sceneplan_model_input": False,
            "source_caption_model_input": False,
            "source_semantic_mode": load_report.source_semantic_mode,
            "frozen_m2d_clap_assets": load_report.frozen_m2d_clap_assets,
            "unchanged_source_demix": _demix_contract(),
            "independent_content_metric_contract": (
                INDEPENDENT_CONTENT_METRIC_CONTRACT
            ),
            "independent_content_metric_assets": independent_content_assets,
            "source_audio_encoded_rows": len(successful),
            "ar_free_rows": len(successful),
            "dit_sampled_rows": len(successful),
            "decoded_audio_rows": len(successful),
            "structural": structural,
            "checks": checks,
            "metric_summaries": summaries,
            "thresholds": thresholds,
            "row_records": str(row_path.resolve()),
            "row_records_sha256": sha256_file(row_path),
            "run_contract": str(contract_path.resolve()),
            "run_contract_sha256": sha256_file(contract_path),
            "source_sha256": source_hashes,
            "model_config": str(model_config),
            "model_config_sha256": sha256_file(model_config),
            "codec": str(codec_path),
            "codec_fingerprint": codec.fingerprint,
            "frozen_vae_config_sha256": FROZEN_VAE_CONFIG_SHA256,
            "frozen_vae_checkpoint_sha256": FROZEN_VAE_CHECKPOINT_SHA256,
            "frozen_qwen_runtime": load_report.frozen_qwen_runtime,
            "calibration": run_contract["calibration"],
            "batch_size_per_rank": int(args.batch_size),
            "batch_size_certification": batch_size_record,
        }
        if args.phase == "calibration":
            calibration_artifact = {
                "schema": CALIBRATION_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "status": status,
                "calibration_contract": CALIBRATION_CONTRACT,
                "evaluation_contract": EVALUATION_CONTRACT,
                "rows": len(records),
                "successful_rows": len(successful),
                "checks": checks,
                "quality_checks": quality_checks,
                "thresholds": thresholds,
                "thresholds_sha256": _canonical_sha256(thresholds),
                "metric_summaries": summaries,
                "metric_summaries_sha256": _canonical_sha256(summaries),
                "joint_checkpoint_selection": str(selection_path),
                "joint_checkpoint_selection_sha256": selection_sha,
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_sha256": sha256_file(checkpoint),
                "model_config": str(model_config),
                "model_config_sha256": sha256_file(model_config),
                "codec": str(codec_path),
                "codec_fingerprint": codec.fingerprint,
                "validation_index": index_summary,
                "row_selection": selection_summary,
                "batch_size_per_rank": int(args.batch_size),
                "batch_size_certification": batch_size_record,
                "ode_steps": int(args.ode_steps),
                "cfg_scale": float(args.cfg_scale),
                "max_plan_tokens": int(args.max_plan_tokens),
                "seed": int(args.seed),
                "per_row_seed_contract": (
                    "blake2b_pair_id_namespace_rank_batch_invariant_v1"
                ),
                "frozen_vae_config_sha256": FROZEN_VAE_CONFIG_SHA256,
                "frozen_vae_checkpoint_sha256": FROZEN_VAE_CHECKPOINT_SHA256,
                "frozen_qwen_runtime": load_report.frozen_qwen_runtime,
                "source_semantic_mode": load_report.source_semantic_mode,
                "frozen_m2d_clap_assets": load_report.frozen_m2d_clap_assets,
                "unchanged_source_demix": _demix_contract(),
                "independent_content_metric_contract": (
                    INDEPENDENT_CONTENT_METRIC_CONTRACT
                ),
                "independent_content_metric_assets": independent_content_assets,
                "source_sha256": source_hashes,
                "run_contract": str(contract_path.resolve()),
                "run_contract_sha256": sha256_file(contract_path),
                "row_records": str(row_path.resolve()),
                "row_records_sha256": sha256_file(row_path),
                "full_report": str((output_dir / "FINAL.json").resolve()),
            }
            calibration_path = output_dir / "CALIBRATION.json"
            _atomic_json(calibration_path, calibration_artifact)
            final["calibration_artifact"] = {
                "path": str(calibration_path.resolve(strict=True)),
                "sha256": sha256_file(calibration_path),
            }
        final_path = output_dir / "FINAL.json"
        _atomic_json(final_path, final)
        print(
            json.dumps(
                {
                    "event": "audio_e2e_complete",
                    "phase": args.phase,
                    "status": status,
                    "rows": len(records),
                    "output": str((output_dir / "FINAL.json").resolve()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if status != "PASS":
            failure = [key for key, value in checks.items() if not value]
            _atomic_json(output_dir / "FAILED.json", {"checks": checks, "failed": failure})
    status_value = [None]
    if rank == 0:
        status_value[0] = json.loads(
            (output_dir / "FINAL.json").read_text(encoding="utf-8")
        )["status"]
    dist.broadcast_object_list(status_value, src=0, device=device)
    dist.barrier()
    dist.destroy_process_group()
    # Full replay rehashes every rank shard, saved waveform, latent inventory,
    # candidate checkpoint and calibration artifact.  Run it only after DDP is
    # torn down so the other four ranks are not held in a collective long
    # enough to risk the process-group timeout.
    if rank == 0 and status_value[0] == "PASS":
        final_path = output_dir / "FINAL.json"
        validate_audio_e2e_final(
            final_path,
            expected_sha256=sha256_file(final_path),
            write_seal=True,
        )
    return 0 if status_value[0] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
