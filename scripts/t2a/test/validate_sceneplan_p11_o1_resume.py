#!/usr/bin/env python3
"""Validate a real eight-GPU P11 full-state O(1) resume smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any

import torch


BENCHMARK_PREFIX = "SAT_BENCHMARK_RESULT="
HEALTH_PREFIX = "SAT_TRAINING_GATE_RESULT="
LOADER_SCHEMA = "stable_audio_tools.resumable_dataloader"
SAMPLER_SCHEMA = "stable_audio_tools.p11_ordered_batch_sampler"
SAMPLER_CONTRACT = "strided_shuffle_false_drop_last_false_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _last_prefixed_json(path: Path, prefix: str) -> dict[str, Any]:
    found = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = line.find(prefix)
        if marker >= 0:
            found = json.loads(line[marker + len(prefix) :])
    if not isinstance(found, dict):
        raise RuntimeError(f"{path} contains no final {prefix.rstrip('=')}")
    return found


def _checkpoint_record(
    path: Path,
    *,
    expected_step: int,
    expected_batches: int,
    expected_epoch_batches: int,
    expected_items: int,
    expected_curriculum_sha256: str,
) -> dict[str, Any]:
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    global_step = int(checkpoint.get("global_step", -1))
    if global_step != expected_step:
        raise RuntimeError(
            f"{path} global_step={global_step}, expected {expected_step}"
        )
    fit_state = ((checkpoint.get("loops") or {}).get("fit_loop") or {}).get(
        "state_dict"
    ) or {}
    combined = fit_state.get("combined_loader")
    if not isinstance(combined, list) or len(combined) != 1:
        raise RuntimeError(f"{path} lacks one checkpointed training loader")
    loader = combined[0]
    expected_loader = {
        "schema": LOADER_SCHEMA,
        "version": 2,
        "batches_yielded": expected_batches,
        "epoch_batches": expected_epoch_batches,
        "dataset_items": expected_items,
        "at_epoch_boundary": False,
    }
    for key, expected in expected_loader.items():
        if loader.get(key) != expected:
            raise RuntimeError(
                f"{path} loader {key}={loader.get(key)!r}, expected {expected!r}"
            )
    sampler = loader.get("batch_sampler_state")
    if not isinstance(sampler, dict):
        raise RuntimeError(f"{path} lacks the O(1) ordered sampler state")
    expected_sampler = {
        "schema": SAMPLER_SCHEMA,
        "version": 1,
        "resume_epoch": 0,
        "dataset_items": expected_items,
        "dataset_fingerprint": f"sha256:{expected_curriculum_sha256}",
        "batch_size": 8,
        "num_replicas": 8,
        "checkpoint_writer_rank": 0,
        "ordering_contract": SAMPLER_CONTRACT,
    }
    for key, expected in expected_sampler.items():
        if sampler.get(key) != expected:
            raise RuntimeError(
                f"{path} sampler {key}={sampler.get(key)!r}, expected {expected!r}"
            )
    optimizer_states = checkpoint.get("optimizer_states") or []
    if len(optimizer_states) != 1:
        raise RuntimeError(f"{path} does not contain one optimizer state")
    del checkpoint
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "global_step": global_step,
        "loader_batches_yielded": loader["batches_yielded"],
        "sampler_resume_epoch": sampler["resume_epoch"],
        "dataset_fingerprint": sampler["dataset_fingerprint"],
    }


def _expected_measured_ids(
    curriculum: Path,
    *,
    first_batch: int,
    num_batches: int,
) -> list[list[str]]:
    expected: list[list[str]] = [[] for _ in range(8)]
    connection = sqlite3.connect(
        f"file:{curriculum}?mode=ro&immutable=1", uri=True
    )
    try:
        for batch_index in range(first_batch, first_batch + num_batches):
            for rank in range(8):
                ordinals = [
                    (batch_index * 8 + local_position) * 8 + rank
                    for local_position in range(8)
                ]
                placeholders = ",".join("?" for _ in ordinals)
                rows = connection.execute(
                    f"SELECT ordinal,curriculum_id FROM rows "
                    f"WHERE ordinal IN ({placeholders}) ORDER BY ordinal",
                    ordinals,
                ).fetchall()
                if [int(row[0]) for row in rows] != ordinals:
                    raise RuntimeError("curriculum ordinal lookup is incomplete")
                expected[rank].extend(str(row[1]) for row in rows)
    finally:
        connection.close()
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    curriculum = args.curriculum.expanduser().resolve(strict=True)
    stage1 = args.stage1_checkpoint.expanduser().resolve(strict=True)
    stage2 = args.stage2_checkpoint.expanduser().resolve(strict=True)
    log = args.log.expanduser().resolve(strict=True)
    curriculum_sha256 = _sha256_file(curriculum)
    connection = sqlite3.connect(
        f"file:{curriculum}?mode=ro&immutable=1", uri=True
    )
    try:
        items = int(connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0])
    finally:
        connection.close()
    if items <= 0 or items % 64:
        raise RuntimeError("resume curriculum is not exactly divisible by global batch 64")
    epoch_batches = items // 64

    stage1_record = _checkpoint_record(
        stage1,
        expected_step=8,
        expected_batches=8,
        expected_epoch_batches=epoch_batches,
        expected_items=items,
        expected_curriculum_sha256=curriculum_sha256,
    )
    stage2_record = _checkpoint_record(
        stage2,
        expected_step=16,
        expected_batches=16,
        expected_epoch_batches=epoch_batches,
        expected_items=items,
        expected_curriculum_sha256=curriculum_sha256,
    )

    log_text = log.read_text(encoding="utf-8", errors="replace")
    ordered_loader_announcements = log_text.count("resume=O(1)")
    if ordered_loader_announcements < 16:
        raise RuntimeError(
            "both eight-rank stages did not announce the ordered O(1) loader"
        )
    if "Restored all states from the checkpoint" not in log_text:
        raise RuntimeError("Lightning did not confirm full-state restoration")
    step_rng_announcements = log_text.count("contract=rank_global_step_v1")
    if step_rng_announcements < 16:
        raise RuntimeError(
            "both eight-rank stages did not enable global-step-derived RNG"
        )

    benchmark = _last_prefixed_json(log, BENCHMARK_PREFIX)
    health = _last_prefixed_json(log, HEALTH_PREFIX)
    expected_benchmark = {
        "initial_global_step": 8,
        "measurement_start_global_step": 10,
        "target_final_global_step": 16,
        "warmup_batches": 2,
        "measured_optimizer_steps": 6,
        "world_size": 8,
        "batch_size_per_gpu": 8,
        "curriculum_rows_disjoint_across_ranks": True,
        "sampled_curriculum_batches_all_gue": True,
    }
    for key, expected in expected_benchmark.items():
        if benchmark.get(key) != expected:
            raise RuntimeError(
                f"restored benchmark {key}={benchmark.get(key)!r}, "
                f"expected {expected!r}"
            )
    observed_ids = benchmark.get("rank_curriculum_identities")
    expected_ids = _expected_measured_ids(
        curriculum, first_batch=10, num_batches=6
    )
    if observed_ids != expected_ids:
        raise RuntimeError(
            "restored run did not begin its measured stream at curriculum batch 10"
        )
    expected_health = {
        "status": "PASS",
        "initial_global_step": 8,
        "global_step": 16,
        "optimizer_events": 8,
    }
    for key, expected in expected_health.items():
        if health.get(key) != expected:
            raise RuntimeError(
                f"restored health {key}={health.get(key)!r}, expected {expected!r}"
            )
    for key, expected in (
        ("ema_initial", {"p11_ema": 8}),
        ("ema_final", {"p11_ema": 16}),
        ("ema_advances", {"p11_ema": 8}),
        ("optimizer_state_step_max", 16),
    ):
        if health.get(key) != expected:
            raise RuntimeError(
                f"restored health {key}={health.get(key)!r}, expected {expected!r}"
            )
    distributed = health.get("distributed_health") or {}
    for key, expected in (
        ("world_size", 8),
        ("optimizer_events_min", 8),
        ("optimizer_events_max", 8),
        ("optimizer_state_step_min", 16),
        ("optimizer_state_step_max", 16),
        ("ema_advance_min", 8),
        ("ema_advance_max", 8),
    ):
        if distributed.get(key) != expected:
            raise RuntimeError(
                f"restored distributed health {key}={distributed.get(key)!r}, "
                f"expected {expected!r}"
            )

    report = {
        "schema": "stable_audio_tools.p11_v4_o1_resume_smoke",
        "schema_version": 1,
        "status": "PASS",
        "contract": "full_state_ddp8_ordered_cursor_o1_resume_v1",
        "curriculum": {
            "path": str(curriculum),
            "bytes": curriculum.stat().st_size,
            "sha256": curriculum_sha256,
            "rows": items,
            "epoch_batches": epoch_batches,
        },
        "stage1": stage1_record,
        "stage2": stage2_record,
        "restored_cursor": {
            "checkpoint_batch_offset": 8,
            "warmup_batches": 2,
            "first_measured_batch": 10,
            "measured_batches": 6,
            "exact_curriculum_ids_match": True,
            "prefix_rows_materialized_by_ordered_sampler": 0,
        },
        "full_state": {
            "lightning_restore_confirmation": True,
            "optimizer_steps_advanced": 8,
            "ema_advanced": int((health.get("ema_advances") or {}).get("p11_ema", 0)),
            "ordered_loader_rank_announcements": ordered_loader_announcements,
            "step_rng_rank_announcements": step_rng_announcements,
            "rng_contract": "seed_rank_global_step_v1",
        },
        "log": {"path": str(log), "sha256": _sha256_file(log)},
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"P11_O1_RESUME_REPORT={output}")


if __name__ == "__main__":
    main()
