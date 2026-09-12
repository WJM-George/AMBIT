#!/usr/bin/env python3
"""Distributed, resumable 8K evaluation for P10-v11 shared Generation AR."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
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
from torch import distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.model_sceneplan_codec_v3 import (  # noqa: E402
    SOURCE_COUNT_TOKENS,
)
from stable_audio_tools.data.scene_plan import (  # noqa: E402
    LOSS_GRAMMAR,
    LOSS_MOTION,
    LOSS_ROOM,
    LOSS_SEMANTIC,
    LOSS_SPATIAL_METRIC,
    LOSS_SPEECH_CONTENT,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_dataset import (  # noqa: E402
    GenerationARSQLiteDataset,
    collate_generation_ar,
    manifest_summary,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_contract import (  # noqa: E402
    canonical_json_bytes as _canonical_json_bytes,
    canonical_sha256 as _canonical_sha256,
    expected_checkpoint_steps as _expected_full_checkpoint_steps,
    expected_selection_candidate_steps as _expected_selection_candidate_steps,
    sha256_file as _sha256_file,
    validate_parent_lineage_artifacts as _validate_parent_lineage_artifacts,
)
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_evaluation import (  # noqa: E402
    GENERATION_AR_EVALUATION_CONTRACT,
    score_parsed_generation,
    score_token_sequence,
    summarize_prediction_records,
)
from stable_audio_tools.models.sceneplan_transfusion_generation_ar import (  # noqa: E402
    GENERATION_AR_CONTRACT,
    load_p10v11_generation_ar,
)


DEFAULT_EVALUATION_MANIFEST = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/transfusion_shared_v1/"
    "generation_ar/test.sqlite"
)
CODEC_PATH = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
EXPECTED_MANIFEST_SHA256 = {
    "train": "e1921b167a09135ae3c3bdf28ecac509f7175a387cb9c0f54b5d7a3608a1799c",
    "validation": "697113f9c38f190cb7e54bf8863c3e3b78dfa77de76daeb9b1ea7c035fa6dd4c",
    "test": "f67d3b809f01981d4aff99f9990efd8e85d21ff61ee7d5728f26e620b2af2b6e",
}
EXPECTED_MANIFEST_ROWS = {"train": 1_600_000, "validation": 32_000, "test": 8_000}
LOSS_GROUP_NAMES = {
    LOSS_GRAMMAR: "grammar",
    LOSS_SEMANTIC: "semantic",
    LOSS_ROOM: "room",
    LOSS_SPATIAL_METRIC: "spatial_metric",
    LOSS_MOTION: "motion",
    LOSS_SPEECH_CONTENT: "speech_content",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help=(
            "Required when evaluating a validation-selected intermediate "
            "checkpoint; final-step checkpoints may omit it."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-manifest",
        "--test-manifest",
        dest="evaluation_manifest",
        type=Path,
        default=DEFAULT_EVALUATION_MANIFEST,
        help="ScenePlan Generation AR manifest to evaluate.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Expected split recorded by the evaluation manifest.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--teacher-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-plan-tokens", type=int, default=512)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--log-every-batches", type=int, default=10)
    return parser.parse_args()


def _validate_training_checkpoint(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    state: dict[str, Any],
    codec,
    p10_report,
    selection_manifest: Path | None,
) -> dict[str, Any]:
    """Prove that evaluation follows a completed, immutable full-data run."""

    if state.get("contract") != GENERATION_AR_CONTRACT:
        raise RuntimeError("Generation AR evaluation checkpoint contract mismatch")
    training_contract = state.get("run_contract")
    if not isinstance(training_contract, dict):
        raise RuntimeError("Generation AR checkpoint is missing its training contract")
    if training_contract.get("mode") != "full":
        raise RuntimeError("8K evaluation requires a full-training checkpoint")
    requested_steps = int(training_contract.get("requested_steps", -1))
    expected_checkpoint_steps = _expected_full_checkpoint_steps(training_contract)
    expected_candidate_steps = _expected_selection_candidate_steps(training_contract)
    coverage = dict(training_contract.get("training_row_coverage") or {})
    if (
        int(training_contract.get("schema_version", -1)) not in (2, 3)
        or list(training_contract.get("checkpoint_steps") or ())
        != expected_checkpoint_steps
        or int(
            training_contract.get("train_manifest", {})
            .get("metadata", {})
            .get("rows", -1)
        )
        != EXPECTED_MANIFEST_ROWS["train"]
        or int(
            training_contract.get("validation_manifest", {})
            .get("metadata", {})
            .get("rows", -1)
        )
        != EXPECTED_MANIFEST_ROWS["validation"]
        or training_contract.get("train_manifest_sha256")
        != EXPECTED_MANIFEST_SHA256["train"]
        or training_contract.get("validation_manifest_sha256")
        != EXPECTED_MANIFEST_SHA256["validation"]
        or int(coverage.get("unique_rows_per_epoch", -1))
        != EXPECTED_MANIFEST_ROWS["train"]
        or int(coverage.get("dropped_rows_per_epoch", -1)) != 0
        or int(coverage.get("duplicated_rows_per_epoch", -1)) != 0
        or bool(coverage.get("drop_last", True))
    ):
        raise RuntimeError("Generation AR exact full-run contract mismatch")
    if int(training_contract.get("seed", -1)) != 42:
        raise RuntimeError("Generation AR training seed contract mismatch")
    if int(training_contract.get("world_size", -1)) != 3:
        raise RuntimeError("Generation AR training world-size contract mismatch")
    if training_contract.get("cuda_visible_devices", "").replace(" ", "") != "0,1,2":
        raise RuntimeError("Generation AR training GPU contract mismatch")
    if training_contract.get("codec_fingerprint") != codec.fingerprint:
        raise RuntimeError("Generation AR training/evaluation codec mismatch")
    if training_contract.get("p10_load") != p10_report.as_dict():
        raise RuntimeError("Generation AR training/evaluation P10-v11 mismatch")
    parent_checkpoint = _validate_parent_lineage_artifacts(training_contract)
    if int(training_contract.get("schema_version", -1)) >= 3:
        schedule = dict(training_contract.get("learning_rate_schedule") or {})
        stage_start_step = int(training_contract.get("stage_start_step", -1))
        if (
            schedule.get("type") != "stage_local_cosine_floor_v1"
            or int(schedule.get("stage_start_step", -1)) != stage_start_step
            or int(schedule.get("stage_end_step", -1)) != requested_steps
            or int(schedule.get("schedule_steps", -1))
            != requested_steps - stage_start_step
            or list(training_contract.get("selection_candidate_steps") or ())
            != expected_candidate_steps
        ):
            raise RuntimeError("Generation AR staged LR/selection contract mismatch")

    for relative, expected_sha256 in dict(
        training_contract.get("source_sha256") or {}
    ).items():
        source = (REPO_ROOT / str(relative)).resolve(strict=True)
        if _sha256_file(source) != str(expected_sha256):
            raise RuntimeError(
                f"training source changed since checkpoint creation: {relative}"
            )

    run_dir = checkpoint.parent.parent.resolve(strict=True)
    run_contract_path = run_dir / "RUN_CONTRACT.json"
    final_path = run_dir / "FINAL.json"
    if not run_contract_path.is_file() or not final_path.is_file():
        raise RuntimeError("8K evaluation requires a completed training run directory")
    disk_contract = json.loads(run_contract_path.read_text(encoding="utf-8"))
    if disk_contract != training_contract:
        raise RuntimeError("checkpoint and on-disk training contracts differ")
    final_event = json.loads(final_path.read_text(encoding="utf-8"))
    if (
        requested_steps <= 0
        or final_event.get("event") != "complete"
        or final_event.get("mode") != "full"
        or int(final_event.get("step", -1)) != requested_steps
    ):
        raise RuntimeError("full Generation AR training completion proof is invalid")

    checkpoint_step = int(state.get("global_step", -1))
    selection: dict[str, Any] | None = None
    selection_sha256: str | None = None
    if checkpoint_step != requested_steps and selection_manifest is None:
        raise RuntimeError(
            "an intermediate checkpoint requires --selection-manifest after "
            "full training completion"
        )
    if selection_manifest is not None:
        selected_path = selection_manifest.expanduser().resolve(strict=True)
        selection = json.loads(selected_path.read_text(encoding="utf-8"))
        if (
            selection.get("schema")
            != "stable_audio_tools.sceneplan_transfusion_generation_ar_checkpoint_selection"
            or selection.get("status") != "COMPLETE"
        ):
            raise RuntimeError("checkpoint selection manifest contract mismatch")
        if Path(str(selection.get("training_run_dir", ""))).resolve() != run_dir:
            raise RuntimeError("checkpoint selection belongs to another training run")
        if Path(str(selection.get("selected_checkpoint", ""))).resolve() != checkpoint:
            raise RuntimeError("evaluation checkpoint is not the selected checkpoint")
        if str(selection.get("selected_checkpoint_sha256")) != checkpoint_sha256:
            raise RuntimeError("selected checkpoint SHA256 mismatch")
        if int(selection.get("selected_checkpoint_step", -1)) != checkpoint_step:
            raise RuntimeError("selected checkpoint step mismatch")
        if int(selection.get("training_final_step", -1)) != requested_steps:
            raise RuntimeError("selection manifest lacks full-training completion")
        if str(selection.get("training_final_sha256")) != _sha256_file(final_path):
            raise RuntimeError("selection manifest FINAL.json SHA256 mismatch")
        if str(selection.get("training_run_contract_sha256")) != _sha256_file(
            run_contract_path
        ):
            raise RuntimeError("selection manifest training contract SHA256 mismatch")
        expected_candidate_policy = (
            "selected_parent_plus_every_stage_half_epoch_full_32k_validation"
            if parent_checkpoint is not None
            else (
                "every_half_epoch_full_32k_validation"
                if training_contract.get("checkpoint_policy") == "interval"
                else "half_epoch_and_epoch_end_only"
            )
        )
        if (
            selection.get("selection_contract")
            != "full_32k_validation_minimum_token_ce_v1"
            or selection.get("candidate_policy") != expected_candidate_policy
        ):
            raise RuntimeError("checkpoint selection policy mismatch")
        candidates = list(selection.get("candidates") or ())
        candidate_steps = sorted(int(row.get("step", -1)) for row in candidates)
        if candidate_steps != expected_candidate_steps:
            raise RuntimeError("checkpoint selection candidates are incomplete")
        expected_validation_rows = int(
            training_contract["validation_manifest"]["metadata"]["rows"]
        )
        selected_candidates = []
        for candidate in candidates:
            candidate_path = Path(str(candidate.get("checkpoint", ""))).resolve(
                strict=True
            )
            if candidate_path.parent != checkpoint.parent and (
                parent_checkpoint is None or candidate_path != parent_checkpoint
            ):
                raise RuntimeError(
                    "selection candidate is outside the training lineage"
                )
            if str(candidate.get("checkpoint_sha256")) != _sha256_file(candidate_path):
                raise RuntimeError("selection candidate SHA256 mismatch")
            if int(candidate.get("validation", {}).get("sequences", -1)) != (
                expected_validation_rows
            ):
                raise RuntimeError("selection candidate lacks full validation coverage")
            if candidate_path == checkpoint:
                selected_candidates.append(candidate)
        if len(selected_candidates) != 1:
            raise RuntimeError(
                "selected checkpoint is absent or duplicated in candidates"
            )
        selected_metrics = selected_candidates[0].get("validation")
        if selected_metrics != selection.get("selected_validation"):
            raise RuntimeError("selected validation metrics mismatch")
        expected_selected = min(
            candidates,
            key=lambda row: (
                float(row["validation"]["loss"]),
                -float(row["validation"]["token_accuracy"]),
                -int(row["step"]),
            ),
        )
        if Path(str(expected_selected["checkpoint"])).resolve() != checkpoint:
            raise RuntimeError("selected checkpoint is not the validation optimum")
        validation_path = Path(
            str(selection.get("validation_manifest", {}).get("path", ""))
        ).resolve(strict=True)
        if str(selection.get("validation_manifest_sha256")) != _sha256_file(
            validation_path
        ):
            raise RuntimeError("selection validation manifest SHA256 mismatch")
        for relative, expected_sha256 in dict(
            selection.get("source_sha256") or {}
        ).items():
            source = (REPO_ROOT / str(relative)).resolve(strict=True)
            if _sha256_file(source) != str(expected_sha256):
                raise RuntimeError(f"checkpoint selector source changed: {relative}")
        selection_sha256 = _sha256_file(selected_path)

    return {
        "training_run_dir": str(run_dir),
        "training_run_contract": str(run_contract_path),
        "training_run_contract_sha256": _sha256_file(run_contract_path),
        "training_final": str(final_path),
        "training_final_sha256": _sha256_file(final_path),
        "training_final_step": requested_steps,
        "selection_manifest": (
            str(selection_manifest.expanduser().resolve())
            if selection_manifest is not None
            else None
        ),
        "selection_manifest_sha256": selection_sha256,
        "selection": selection,
    }


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


def _distributed() -> tuple[int, int, int, torch.device]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").replace(" ", "")
    if visible != "0,1,2":
        raise RuntimeError(
            "Generation AR 8K evaluation requires CUDA_VISIBLE_DEVICES=0,1,2"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 3 or not torch.cuda.is_available():
        raise RuntimeError("Generation AR 8K evaluation requires exactly three GPUs")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    return rank, local_rank, world_size, device


def _move_teacher_batch(batch: dict[str, Any], device: torch.device) -> None:
    for key in (
        "plan_input_ids",
        "plan_labels",
        "plan_loss_group_ids",
        "plan_attention_mask",
    ):
        batch[key] = batch[key].to(device, non_blocking=True)


@torch.no_grad()
def _teacher_forced_metrics(
    model,
    dataset: GenerationARSQLiteDataset,
    codec,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        persistent_workers=int(num_workers) > 0,
        collate_fn=functools.partial(collate_generation_ar, pad_id=codec.pad_id),
    )
    group_ids = sorted(LOSS_GROUP_NAMES)
    # loss, correct, tokens, sequence-correct, sequences; then triples per group.
    source_count_offset = 5 + 3 * len(group_ids)
    values = torch.zeros(source_count_offset + 16, device=device, dtype=torch.float64)
    source_count_ids = torch.tensor(
        [codec.token_to_id[token] for token in SOURCE_COUNT_TOKENS],
        device=device,
        dtype=torch.long,
    )
    model.eval()
    for batch in loader:
        _move_teacher_batch(batch, device)
        context, context_mask = model.encode_requests(
            batch["raw_user_requests"], device=device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                batch["plan_input_ids"],
                batch["plan_attention_mask"],
                context,
                context_mask,
            )
        labels = batch["plan_labels"]
        valid = labels != -100
        losses = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(labels)
        predictions = logits.argmax(dim=-1)
        correct = predictions.eq(labels) & valid
        values[0] += losses[valid].double().sum()
        values[1] += correct.double().sum()
        values[2] += valid.double().sum()
        values[3] += (correct | ~valid).all(dim=1).double().sum()
        values[4] += labels.shape[0]
        groups = batch["plan_loss_group_ids"]
        for index, group_id in enumerate(group_ids):
            mask = valid & groups.eq(int(group_id))
            offset = 5 + 3 * index
            values[offset] += losses[mask].double().sum()
            values[offset + 1] += correct[mask].double().sum()
            values[offset + 2] += mask.double().sum()
        count_matches = labels.unsqueeze(-1).eq(source_count_ids.view(1, 1, -1))
        count_mask = count_matches.any(dim=-1)
        count_truth = count_matches[count_mask].long().argmax(dim=-1)
        count_prediction = (
            logits[count_mask].index_select(-1, source_count_ids).argmax(dim=-1)
        )
        confusion = torch.bincount(count_truth * 4 + count_prediction, minlength=16)
        values[source_count_offset:] += confusion.double()
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    result = {
        "loss": float(values[0].item() / max(1.0, values[2].item())),
        "token_accuracy": float(values[1].item() / max(1.0, values[2].item())),
        "sequence_exact": float(values[3].item() / max(1.0, values[4].item())),
        "tokens": int(values[2].item()),
        "sequences": int(values[4].item()),
        "loss_groups": {},
    }
    for index, group_id in enumerate(group_ids):
        offset = 5 + 3 * index
        count = values[offset + 2].item()
        result["loss_groups"][LOSS_GROUP_NAMES[group_id]] = {
            "loss": float(values[offset].item() / max(1.0, count)),
            "accuracy": float(values[offset + 1].item() / max(1.0, count)),
            "tokens": int(count),
        }
    confusion = values[source_count_offset:].reshape(4, 4)
    if int(confusion.sum().item()) != int(values[4].item()):
        raise RuntimeError("teacher-forced source-count coverage is incomplete")
    result["source_count"] = {
        "accuracy": float(
            confusion.diagonal().sum().item() / max(1.0, confusion.sum().item())
        ),
        "confusion_target_to_prediction": {
            str(target + 1): {
                str(prediction + 1): int(confusion[target, prediction].item())
                for prediction in range(4)
            }
            for target in range(4)
        },
        "recall_by_target": {
            str(target + 1): float(
                confusion[target, target].item()
                / max(1.0, confusion[target].sum().item())
            )
            for target in range(4)
        },
    }
    return result


def _open_prediction_shard(
    path: Path, *, rank: int, world_size: int, run_contract_sha256: str
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            ordinal INTEGER PRIMARY KEY,
            sample_id TEXT NOT NULL,
            template_id TEXT NOT NULL,
            source_count INTEGER NOT NULL,
            target_token_count INTEGER NOT NULL,
            predicted_token_count INTEGER,
            status TEXT NOT NULL,
            error TEXT,
            target_sceneplan_sha256 TEXT NOT NULL,
            prediction_sceneplan_sha256 TEXT,
            prediction_sceneplan_zlib BLOB,
            predicted_token_ids_u16le BLOB,
            metrics_json TEXT NOT NULL,
            generation_sec REAL NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_attempt_history (
            attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ordinal INTEGER NOT NULL,
            sample_id TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            metrics_json TEXT NOT NULL,
            generation_sec REAL NOT NULL,
            archived_unix REAL NOT NULL
        )
        """
    )
    expected = {
        "contract": GENERATION_AR_EVALUATION_CONTRACT,
        "rank": str(rank),
        "world_size": str(world_size),
        "run_contract_sha256": run_contract_sha256,
    }
    existing = dict(connection.execute("SELECT key,value FROM metadata"))
    if existing:
        for key, value in expected.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"prediction shard contract mismatch for {path}: {key}"
                )
    else:
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", expected.items()
        )
        connection.execute(
            "INSERT INTO metadata(key,value) VALUES ('status','IN_PROGRESS')"
        )
    # Successful rows are immutable resume points.  Failed model-output rows
    # are archived and retried so a transient OOM/worker fault cannot become a
    # permanent result merely because a shard exists.
    connection.execute(
        """
        INSERT INTO prediction_attempt_history(
            ordinal,sample_id,status,error,metrics_json,generation_sec,archived_unix
        )
        SELECT ordinal,sample_id,status,error,metrics_json,generation_sec,?
        FROM predictions WHERE status != 'ok'
        """,
        (time.time(),),
    )
    connection.execute("DELETE FROM predictions WHERE status != 'ok'")
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES ('status','IN_PROGRESS')"
    )
    connection.commit()
    return connection


def _generate_with_fallback(
    model,
    rows: list[dict[str, Any]],
    codec,
    *,
    device: torch.device,
    max_plan_tokens: int,
) -> list[tuple[list[int] | None, str | None, float]]:
    """Bisect a failed batch so one malformed decode cannot discard its peers."""

    started = time.perf_counter()
    failure: str | None = None
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.generate_constrained(
                [row["raw_user_request"] for row in rows],
                codec,
                device=device,
                max_plan_tokens=int(max_plan_tokens),
            )
        elapsed = (time.perf_counter() - started) / len(rows)
        return [(list(tokens), None, elapsed) for tokens in outputs]
    except Exception as exc:
        # Preserve only text, then leave the exception scope so its traceback no
        # longer owns the failed Qwen context / autoregressive KV cache.
        failure = f"{type(exc).__name__}: {exc}"
    torch.cuda.empty_cache()
    if len(rows) == 1:
        return [(None, failure, time.perf_counter() - started)]
    middle = len(rows) // 2
    left = _generate_with_fallback(
        model,
        rows[:middle],
        codec,
        device=device,
        max_plan_tokens=max_plan_tokens,
    )
    right = _generate_with_fallback(
        model,
        rows[middle:],
        codec,
        device=device,
        max_plan_tokens=max_plan_tokens,
    )
    return left + right


def _source_rows(
    manifest: Path, *, rank: int, world_size: int, row_limit: int
) -> Iterable[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{manifest}?mode=ro&immutable=1", uri=True)
    try:
        cursor = connection.execute(
            """
            SELECT ordinal,sample_id,template_id,raw_user_request,
                   target_sceneplan_zlib,target_sceneplan_sha256,
                   target_token_ids_u16le,target_token_count,source_count
            FROM rows
            WHERE ordinal < ? AND (ordinal % ?) = ?
            ORDER BY ordinal
            """,
            (int(row_limit), int(world_size), int(rank)),
        )
        for values in cursor:
            yield dict(
                zip(
                    (
                        "ordinal",
                        "sample_id",
                        "template_id",
                        "raw_user_request",
                        "target_sceneplan_zlib",
                        "target_sceneplan_sha256",
                        "target_token_ids_u16le",
                        "target_token_count",
                        "source_count",
                    ),
                    values,
                )
            )
    finally:
        connection.close()


def _write_prediction(
    connection: sqlite3.Connection,
    row: dict[str, Any],
    generated: list[int] | None,
    error: str | None,
    generation_sec: float,
    codec,
) -> None:
    metrics: dict[str, float] = {}
    prediction_blob = None
    prediction_sha256 = None
    prediction_tokens = None
    predicted_count = None
    status = "generation_error"
    stored_error = error
    if generated is not None:
        target_bytes = zlib.decompress(row["target_sceneplan_zlib"])
        target_sha256 = hashlib.sha256(target_bytes).hexdigest()
        if target_sha256 != str(row["target_sceneplan_sha256"]):
            raise RuntimeError(
                f"target ScenePlan SHA256 mismatch at ordinal {row['ordinal']}"
            )
        target = json.loads(target_bytes)
        target_tokens = (
            np.frombuffer(row["target_token_ids_u16le"], dtype="<u2")
            .astype(np.int64)
            .tolist()
        )
        if len(target_tokens) != int(row["target_token_count"]):
            raise RuntimeError(
                f"target token count mismatch at ordinal {row['ordinal']}"
            )
        metrics.update(score_token_sequence(target_tokens, generated, codec))
        prediction_tokens = np.asarray(generated, dtype="<u2").tobytes()
        predicted_count = len(generated)
        try:
            prediction = codec.decode(generated, sample_id=str(row["sample_id"]))
        except Exception as exc:
            metrics["parse_rate"] = 0.0
            status = "parse_error"
            stored_error = f"{type(exc).__name__}: {exc}"
        else:
            # Scoring and serialization are evaluator infrastructure: never
            # mislabel their failures as a model parse outcome.
            metrics["parse_rate"] = 1.0
            metrics.update(score_parsed_generation(target, prediction))
            canonical = _canonical_json_bytes(prediction)
            prediction_blob = zlib.compress(canonical, level=9)
            prediction_sha256 = hashlib.sha256(canonical).hexdigest()
            status = "ok"
            stored_error = None
    connection.execute(
        """
        INSERT INTO predictions(
            ordinal,sample_id,template_id,source_count,target_token_count,
            predicted_token_count,status,error,target_sceneplan_sha256,
            prediction_sceneplan_sha256,prediction_sceneplan_zlib,
            predicted_token_ids_u16le,metrics_json,generation_sec
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            int(row["ordinal"]),
            str(row["sample_id"]),
            str(row["template_id"]),
            int(row["source_count"]),
            int(row["target_token_count"]),
            predicted_count,
            status,
            stored_error,
            str(row["target_sceneplan_sha256"]),
            prediction_sha256,
            prediction_blob,
            prediction_tokens,
            json.dumps(metrics, sort_keys=True, separators=(",", ":")),
            float(generation_sec),
        ),
    )


def _load_aggregate_records(
    prediction_dir: Path, *, world_size: int, expected_rows: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank in range(int(world_size)):
        path = prediction_dir / f"rank_{rank:03d}.sqlite"
        connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        if metadata.get("status") != "COMPLETE":
            raise RuntimeError(f"prediction shard is incomplete: {path}")
        for (
            ordinal,
            status,
            source_count,
            template_id,
            metrics_json,
        ) in connection.execute(
            """
            SELECT ordinal,status,source_count,template_id,metrics_json
            FROM predictions ORDER BY ordinal
            """
        ):
            output.append(
                {
                    "ordinal": int(ordinal),
                    "status": str(status),
                    "source_count": int(source_count),
                    "template_id": str(template_id),
                    "metrics": json.loads(metrics_json),
                }
            )
        connection.close()
    if sorted(int(row["ordinal"]) for row in output) != list(range(expected_rows)):
        raise RuntimeError("prediction shards do not exactly cover the requested rows")
    return output


def main() -> int:
    args = _parse_args()
    if min(args.batch_size, args.teacher_batch_size, args.max_plan_tokens) <= 0:
        raise ValueError("evaluation batch sizes and token limit must be positive")
    rank, local_rank, world_size, device = _distributed()
    torch.manual_seed(42 + rank)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    evaluation_manifest = args.evaluation_manifest.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve(strict=False)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "predictions").mkdir(exist_ok=True)
    dist.barrier()

    codec = ModelScenePlanCodecV4(CODEC_PATH)
    model, p10_report = load_p10v11_generation_ar(
        pad_id=codec.pad_id,
        verify_sha256=rank == 0,
        activation_checkpointing=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_sha = [_sha256_file(checkpoint) if rank == 0 else None]
    dist.broadcast_object_list(checkpoint_sha, src=0, device=device)
    training_proof = _validate_training_checkpoint(
        checkpoint=checkpoint,
        checkpoint_sha256=str(checkpoint_sha[0]),
        state=state,
        codec=codec,
        p10_report=p10_report,
        selection_manifest=args.selection_manifest,
    )
    model.load_trainable_state_dict(state["ar_adapter"])
    model.p10_dit.to(device=device, dtype=torch.bfloat16)
    model.prompt_conditioner.to(device=device, dtype=torch.bfloat16)
    model.ar_adapter.to(device=device, dtype=torch.float32)
    model.eval()

    evaluation_manifest_summary = manifest_summary(evaluation_manifest)
    manifest_split = str(evaluation_manifest_summary["metadata"].get("split", ""))
    if manifest_split != str(args.split):
        raise RuntimeError(
            "evaluation manifest split mismatch: "
            f"expected {args.split}, found {manifest_split}"
        )
    evaluation_manifest_sha = [_sha256_file(evaluation_manifest) if rank == 0 else None]
    dist.broadcast_object_list(evaluation_manifest_sha, src=0, device=device)
    total_rows = int(evaluation_manifest_summary["metadata"]["rows"])
    if (
        str(evaluation_manifest_sha[0]) != EXPECTED_MANIFEST_SHA256[str(args.split)]
        or total_rows != EXPECTED_MANIFEST_ROWS[str(args.split)]
    ):
        raise RuntimeError("evaluation manifest identity/row-count mismatch")
    row_limit = (
        total_rows if args.max_rows is None else min(total_rows, int(args.max_rows))
    )
    if row_limit <= 0:
        raise ValueError("--max-rows must be positive")
    source_files = [
        Path(__file__).resolve(),
        REPO_ROOT
        / "stable_audio_tools/data/sceneplan_transfusion_generation_ar_evaluation.py",
        REPO_ROOT
        / "stable_audio_tools/data/sceneplan_transfusion_generation_ar_dataset.py",
        REPO_ROOT
        / "stable_audio_tools/data/sceneplan_transfusion_generation_ar_contract.py",
        REPO_ROOT / "stable_audio_tools/data/sceneplan_p11_metrics.py",
        REPO_ROOT / "stable_audio_tools/data/model_sceneplan_codec_v4.py",
        REPO_ROOT / "stable_audio_tools/data/scene_plan.py",
        REPO_ROOT / "stable_audio_tools/models/sceneplan_transfusion_generation_ar.py",
        REPO_ROOT / "stable_audio_tools/models/transformer.py",
    ]
    run_contract = {
        "schema": "stable_audio_tools.sceneplan_transfusion_generation_ar_evaluation_run",
        "schema_version": 2,
        "contract": GENERATION_AR_EVALUATION_CONTRACT,
        "generation_ar_contract": GENERATION_AR_CONTRACT,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha[0],
        "checkpoint_step": int(state["global_step"]),
        "training_completion_proof": training_proof,
        "evaluation_split": str(args.split),
        "evaluation_manifest": evaluation_manifest_summary,
        "evaluation_manifest_sha256": evaluation_manifest_sha[0],
        "row_limit": row_limit,
        "world_size": world_size,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "batch_size_per_rank": int(args.batch_size),
        "teacher_batch_size_per_rank": int(args.teacher_batch_size),
        "max_plan_tokens": int(args.max_plan_tokens),
        "codec_path": str(CODEC_PATH),
        "codec_fingerprint": codec.fingerprint,
        "p10_load": p10_report.as_dict(),
        "source_sha256": {
            str(path.relative_to(REPO_ROOT)): _sha256_file(path)
            for path in source_files
        },
    }
    contract_sha = _canonical_sha256(run_contract)
    contract_path = output_dir / "RUN_CONTRACT.json"
    if rank == 0:
        if contract_path.exists():
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
            if existing != run_contract:
                raise RuntimeError("evaluation output directory contract mismatch")
        else:
            _atomic_json(contract_path, run_contract)
    dist.barrier()

    assigned_ordinals = tuple(range(rank, row_limit, world_size))
    teacher_dataset = GenerationARSQLiteDataset(
        evaluation_manifest,
        split=str(args.split),
        row_ordinals=assigned_ordinals,
    )
    teacher = _teacher_forced_metrics(
        model,
        teacher_dataset,
        codec,
        device=device,
        batch_size=int(args.teacher_batch_size),
        num_workers=int(args.num_workers),
    )
    teacher_dataset.close()
    if rank == 0:
        _atomic_json(output_dir / "TEACHER_FORCED.json", teacher)
        print(json.dumps({"event": "teacher_forced_complete", **teacher}), flush=True)
    dist.barrier()

    shard_path = output_dir / "predictions" / f"rank_{rank:03d}.sqlite"
    shard = _open_prediction_shard(
        shard_path,
        rank=rank,
        world_size=world_size,
        run_contract_sha256=contract_sha,
    )
    completed = {
        int(ordinal): (str(sample_id), str(target_sha256), int(target_count))
        for ordinal, sample_id, target_sha256, target_count in shard.execute(
            """
            SELECT ordinal,sample_id,target_sceneplan_sha256,target_token_count
            FROM predictions
            """
        )
    }
    pending_batch: list[dict[str, Any]] = []
    processed_batches = 0
    started = time.perf_counter()

    def flush() -> None:
        nonlocal processed_batches
        if not pending_batch:
            return
        outputs = _generate_with_fallback(
            model,
            pending_batch,
            codec,
            device=device,
            max_plan_tokens=int(args.max_plan_tokens),
        )
        for row, (generated, error, generation_sec) in zip(pending_batch, outputs):
            _write_prediction(shard, row, generated, error, generation_sec, codec)
        shard.commit()
        processed_batches += 1
        if processed_batches % int(args.log_every_batches) == 0:
            count = int(shard.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
            print(
                json.dumps(
                    {
                        "event": "free_decode_progress",
                        "rank": rank,
                        "rows": count,
                        "assigned_rows": len(assigned_ordinals),
                        "elapsed_sec": round(time.perf_counter() - started, 3),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        pending_batch.clear()

    for row in _source_rows(
        evaluation_manifest,
        rank=rank,
        world_size=world_size,
        row_limit=row_limit,
    ):
        ordinal = int(row["ordinal"])
        if ordinal in completed:
            expected_identity = (
                str(row["sample_id"]),
                str(row["target_sceneplan_sha256"]),
                int(row["target_token_count"]),
            )
            if completed[ordinal] != expected_identity:
                raise RuntimeError(
                    f"resumed prediction identity mismatch at ordinal {ordinal}"
                )
            continue
        pending_batch.append(row)
        if len(pending_batch) == int(args.batch_size):
            flush()
    flush()
    stored_rows = int(shard.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])
    if stored_rows != len(assigned_ordinals):
        raise RuntimeError(
            f"rank {rank} prediction coverage mismatch: {stored_rows} "
            f"!= {len(assigned_ordinals)}"
        )
    shard.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES ('rows',?)",
        (str(stored_rows),),
    )
    shard.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES ('status','COMPLETE')"
    )
    shard.commit()
    shard.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    shard.close()
    dist.barrier()

    unchanged = [
        (
            _sha256_file(evaluation_manifest) == str(evaluation_manifest_sha[0])
            if rank == 0
            else None
        )
    ]
    dist.broadcast_object_list(unchanged, src=0, device=device)
    if not bool(unchanged[0]):
        raise RuntimeError(
            "evaluation manifest changed during Generation AR evaluation"
        )
    dist.barrier()

    evaluation_passed = [False]
    if rank == 0:
        records = _load_aggregate_records(
            output_dir / "predictions",
            world_size=world_size,
            expected_rows=row_limit,
        )
        free_decode = summarize_prediction_records(records, expected_rows=row_limit)
        status_counts = dict(free_decode.get("status_counts") or {})
        no_model_output_errors = (
            int(status_counts.get("ok", 0)) == row_limit
            and sum(int(value) for value in status_counts.values()) == row_limit
        )
        exact_coverage = bool(
            dict(free_decode.get("coverage") or {}).get("ordinal_coverage_exact", False)
        )
        evaluation_passed[0] = bool(no_model_output_errors and exact_coverage)
        summary = {
            "status": (
                "PASS" if evaluation_passed[0] else "COMPLETE_WITH_MODEL_OUTPUT_ERRORS"
            ),
            "contract": GENERATION_AR_EVALUATION_CONTRACT,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha[0],
            "checkpoint_step": int(state["global_step"]),
            "evaluation_split": str(args.split),
            "evaluation_manifest": str(evaluation_manifest),
            "evaluation_manifest_sha256": evaluation_manifest_sha[0],
            "rows": row_limit,
            "teacher_forced": teacher,
            "free_decode": free_decode,
        }
        _atomic_json(output_dir / "SUMMARY.json", summary)
        print(json.dumps({"event": "evaluation_complete", **summary}), flush=True)
    dist.barrier()
    dist.broadcast_object_list(evaluation_passed, src=0, device=device)
    dist.destroy_process_group()
    return 0 if bool(evaluation_passed[0]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
