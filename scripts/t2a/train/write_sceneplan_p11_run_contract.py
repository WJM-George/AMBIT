#!/usr/bin/env python3
"""Write an immutable, source-hashed P11 training launch contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from stable_audio_tools.configuration import load_config


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_P10_RELEASE = REPO_ROOT / (
    "artifacts/releases/P10_SCENEPLAN_DIT_V11_150K_RELEASE.json"
)
RUNTIME_SOURCES = (
    "train.py",
    "scripts/t2a/train/run_sceneplan_p11.sh",
    "scripts/t2a/train/run_sceneplan_p11_v4_o1_resume_smoke_8gpu.sh",
    "scripts/t2a/train/write_sceneplan_p11_run_contract.py",
    "scripts/t2a/data/build_sceneplan_p11_v4_curriculum.py",
    "scripts/t2a/data/build_sceneplan_p11_v4_pair_aware_curriculum.py",
    "scripts/t2a/test/audit_sceneplan_p11_lexical_cache.py",
    "scripts/t2a/test/audit_sceneplan_p11_v4_ddp_sampler.py",
    "scripts/t2a/test/smoke_sceneplan_p11_v4_graph.py",
    "scripts/t2a/test/validate_sceneplan_p11_v4_contract.py",
    "scripts/t2a/test/validate_sceneplan_p11_v4_control_direction.py",
    "scripts/t2a/test/validate_sceneplan_p11_v4_curriculum.py",
    "scripts/t2a/test/validate_sceneplan_p11_v4_delta_owner.py",
    "scripts/t2a/test/validate_sceneplan_p11_v4_matched_arms.py",
    "scripts/t2a/test/validate_sceneplan_p11_v4_screening.py",
    "scripts/t2a/test/validate_sceneplan_p11_v4_sequence_budget.py",
    "scripts/t2a/test/validate_sceneplan_p11_gpu_preflight.py",
    "scripts/t2a/test/validate_sceneplan_p11_o1_resume.py",
    "stable_audio_tools/configuration.py",
    "stable_audio_tools/data/dataset.py",
    "stable_audio_tools/data/resumable_dataloader.py",
    "stable_audio_tools/data/scene_sketch_v1.py",
    "stable_audio_tools/data/sceneplan_edit_patch.py",
    "stable_audio_tools/data/sceneplan_p11_metrics.py",
    "stable_audio_tools/data/sceneplan_p11_ordered_sampler.py",
    "stable_audio_tools/data/sceneplan_p11_dataset.py",
    "stable_audio_tools/data/sceneplan_p11_single_turn.py",
    "stable_audio_tools/data/sceneplan_p11_v4_curriculum.py",
    "stable_audio_tools/data/sceneplan_p11_v4_dataset.py",
    "stable_audio_tools/models/scene_thought_p11_v4.py",
    "stable_audio_tools/models/sceneplan_p11.py",
    "stable_audio_tools/models/sceneplan_p11_v4.py",
    "stable_audio_tools/training/sceneplan_p11_v4.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--p10-release", type=Path, default=DEFAULT_P10_RELEASE)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--benchmark-warmup-batches", type=int, required=True)
    parser.add_argument("--ddp-bucket-cap-mb", type=int, required=True)
    parser.add_argument("--ddp-comm-hook", required=True)
    parser.add_argument("--bind-to-gpu-numa", type=int, choices=(0, 1), required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, required=True)
    parser.add_argument("--save-top-k", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ckpt-path", type=Path)
    args = parser.parse_args()

    model_path = args.model_config.expanduser().resolve(strict=True)
    dataset_path = args.dataset_config.expanduser().resolve(strict=True)
    p10_release_path = args.p10_release.expanduser().resolve(strict=True)
    model = load_config(model_path)
    dataset = load_config(dataset_path)
    p10_release = json.loads(p10_release_path.read_text(encoding="utf-8"))
    executor = model.get("model", {}).get("executor", {})
    transfusion = model.get("model", {}).get("transfusion_cot", {})
    released_checkpoint = p10_release.get("checkpoint", {})
    p10_binding = {
        "release_status_frozen": p10_release.get("status") == "frozen_canonical",
        "release_immutable": p10_release.get("immutable") is True,
        "checkpoint_path_matches": (
            executor.get("canonical_checkpoint") == released_checkpoint.get("path")
        ),
        "checkpoint_sha256_matches": (
            executor.get("canonical_checkpoint_sha256")
            == released_checkpoint.get("sha256")
        ),
        "checkpoint_step_matches": (
            executor.get("canonical_checkpoint_step")
            == released_checkpoint.get("step")
        ),
        "max_frames_match": (
            executor.get("p10_max_latent_frames")
            == p10_release.get("runtime_envelope", {}).get("max_latent_frames")
            == 648
        ),
    }
    if not all(p10_binding.values()):
        failed = [key for key, value in p10_binding.items() if not value]
        raise RuntimeError(f"P10 release binding failed: {failed}")

    gpu_ids = [int(value) for value in args.gpu_ids.split(",")]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("GPU ids must be unique")
    scale_profiles = {"scale_preflight", "scale_trial", "resume_smoke"}
    if args.profile in scale_profiles and gpu_ids != list(
        range(8)
    ):
        raise ValueError("canonical scale profiles require physical GPUs 0..7")
    if args.batch_size != 8 or args.seed != 42:
        raise ValueError("canonical P11 runs require batch/GPU=8 and seed=42")

    resume_checkpoint = None
    resume_loader_state = None
    initial_global_step = 0
    if args.ckpt_path is not None:
        checkpoint_path = args.ckpt_path.expanduser().resolve(strict=True)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        initial_global_step = int(checkpoint.get("global_step", -1))
        if initial_global_step < 0:
            raise RuntimeError("resume checkpoint lacks a valid global_step")
        if len(checkpoint.get("optimizer_states") or []) != 1:
            raise RuntimeError("resume checkpoint lacks the canonical optimizer state")
        fit_state = (
            ((checkpoint.get("loops") or {}).get("fit_loop") or {}).get(
                "state_dict"
            )
            or {}
        )
        combined_loader = fit_state.get("combined_loader")
        if isinstance(combined_loader, list) and len(combined_loader) == 1:
            resume_loader_state = combined_loader[0]
        resume_checkpoint = {
            **_file_record(checkpoint_path),
            "global_step": initial_global_step,
            "pytorch_lightning_version": checkpoint.get(
                "pytorch-lightning_version"
            ),
        }
        del checkpoint
    if args.max_steps <= initial_global_step:
        raise RuntimeError(
            "target max_steps must exceed the resume checkpoint global_step: "
            f"{args.max_steps} <= {initial_global_step}"
        )

    curriculum_record = None
    curriculum_rows = None
    curriculum_value = dataset.get("p11_v4_curriculum_path")
    if curriculum_value:
        curriculum_path = Path(str(curriculum_value)).expanduser().resolve(strict=True)
        curriculum_record = _file_record(curriculum_path)
        configured_curriculum_sha256 = str(
            dataset.get("p11_v4_curriculum_sha256", "")
        ).removeprefix("sha256:").strip().lower()
        if configured_curriculum_sha256 != curriculum_record["sha256"]:
            raise RuntimeError(
                "curriculum SQLite SHA256 does not match the resolved dataset "
                "config"
            )
        with sqlite3.connect(
            f"file:{curriculum_path}?mode=ro&immutable=1", uri=True
        ) as connection:
            curriculum_metadata = dict(
                connection.execute("SELECT key,value FROM metadata")
            )
            curriculum_rows = int(
                connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
            )
        expected_curriculum_rows = int(
            dataset.get("p11_v4_curriculum_expected_rows", -1)
        )
        if expected_curriculum_rows <= 0:
            raise RuntimeError(
                "curriculum dataset config lacks a positive "
                "p11_v4_curriculum_expected_rows"
            )
        if curriculum_rows != expected_curriculum_rows:
            raise RuntimeError(
                "curriculum row count does not match the resolved dataset config: "
                f"{curriculum_rows} != {expected_curriculum_rows}"
            )
        configured_ordering = dataset.get(
            "p11_v4_curriculum_ordering_contract"
        )
        configured_ordering_batch = dataset.get(
            "p11_v4_curriculum_ordering_batch_size"
        )
        if curriculum_metadata.get("contract") != dataset.get(
            "p11_v4_curriculum_contract"
        ):
            raise RuntimeError("curriculum SQLite/data-config contract mismatch")
        if curriculum_metadata.get("ordering_contract") != configured_ordering:
            raise RuntimeError("curriculum SQLite/data-config ordering mismatch")
        if (
            configured_ordering is not None
            and int(curriculum_metadata.get("ordering_batch_size", -1))
            != int(configured_ordering_batch)
        ):
            raise RuntimeError("curriculum SQLite/data-config batch ordering mismatch")
        if args.profile in scale_profiles:
            required_ordering = "p11_v4_ddp8_rank_balanced_pair_aware_batch8_v7"
            if configured_ordering != required_ordering:
                raise RuntimeError(
                    "canonical scale runs require the DDP8 rank-balanced v7 "
                    f"curriculum, got {configured_ordering!r}"
                )
            required_scale_metadata = {
                "ordering_world_size": "8",
                "ordering_batch_size": "8",
                "ordering_global_batch_size": "64",
                "ddp_rank_task_counts_identical": "true",
                "distributed_sampler_contract": (
                    "strided_shuffle_false_drop_last_false_v1"
                ),
            }
            for key, expected in required_scale_metadata.items():
                if curriculum_metadata.get(key) != expected:
                    raise RuntimeError(
                        f"canonical scale curriculum {key}="
                        f"{curriculum_metadata.get(key)!r}, expected {expected!r}"
                    )
            if curriculum_rows % (len(gpu_ids) * args.batch_size):
                raise RuntimeError(
                    "canonical scale curriculum has a partial global batch"
                )
        curriculum_record.update(
            {
                "actual_rows": curriculum_rows,
                "expected_rows": expected_curriculum_rows,
                "configured_sha256": configured_curriculum_sha256,
                "contract": dataset.get("p11_v4_curriculum_contract"),
                "ordering_contract": configured_ordering,
                "ordering_batch_size": configured_ordering_batch,
                "ordering_world_size": curriculum_metadata.get(
                    "ordering_world_size"
                ),
                "ordering_global_batch_size": curriculum_metadata.get(
                    "ordering_global_batch_size"
                ),
                "ddp_rank_task_counts_json": curriculum_metadata.get(
                    "ddp_rank_task_counts_json"
                ),
            }
        )
        if args.ckpt_path is not None and configured_ordering is not None:
            if not isinstance(resume_loader_state, dict):
                raise RuntimeError(
                    "ordered-curriculum resume checkpoint lacks one DataLoader state"
                )
            expected_epoch_batches = curriculum_rows // (
                len(gpu_ids) * args.batch_size
            )
            expected_batches_yielded = initial_global_step % expected_epoch_batches
            expected_loader_state = {
                "schema": "stable_audio_tools.resumable_dataloader",
                "version": 2,
                "batches_yielded": expected_batches_yielded,
                "epoch_batches": expected_epoch_batches,
                "dataset_items": curriculum_rows,
                "at_epoch_boundary": expected_batches_yielded == 0,
            }
            for key, expected in expected_loader_state.items():
                if resume_loader_state.get(key) != expected:
                    raise RuntimeError(
                        f"resume DataLoader {key}="
                        f"{resume_loader_state.get(key)!r}, expected {expected!r}"
                    )
            sampler_state = resume_loader_state.get("batch_sampler_state")
            if not isinstance(sampler_state, dict):
                raise RuntimeError(
                    "ordered-curriculum checkpoint predates the O(1) sampler state"
                )
            expected_sampler_state = {
                "schema": "stable_audio_tools.p11_ordered_batch_sampler",
                "version": 1,
                "resume_epoch": initial_global_step // expected_epoch_batches,
                "dataset_items": curriculum_rows,
                "dataset_fingerprint": (
                    f"sha256:{configured_curriculum_sha256}"
                ),
                "batch_size": args.batch_size,
                "num_replicas": len(gpu_ids),
                "ordering_contract": (
                    "strided_shuffle_false_drop_last_false_v1"
                ),
            }
            for key, expected in expected_sampler_state.items():
                if sampler_state.get(key) != expected:
                    raise RuntimeError(
                        f"resume ordered sampler {key}="
                        f"{sampler_state.get(key)!r}, expected {expected!r}"
                    )
            writer_rank = int(sampler_state.get("checkpoint_writer_rank", -1))
            if not 0 <= writer_rank < len(gpu_ids):
                raise RuntimeError("resume ordered sampler writer rank is invalid")
            resume_checkpoint["dataloader_state"] = resume_loader_state
        build_report = curriculum_path.with_suffix(".build.json")
        if build_report.is_file():
            curriculum_record["build_report"] = _file_record(build_report)
            build_payload = json.loads(build_report.read_text(encoding="utf-8"))
            if (
                build_payload.get("status") != "BUILT"
                or int(build_payload.get("rows", -1)) != curriculum_rows
                or build_payload.get("output_sha256")
                != curriculum_record["sha256"]
                or build_payload.get("ordering_contract") != configured_ordering
            ):
                raise RuntimeError(
                    "curriculum build report does not certify the selected SQLite"
                )
            if args.profile in scale_profiles:
                ddp_audit = (build_payload.get("audit") or {}).get("ddp") or {}
                if (
                    ddp_audit.get("world_size") != 8
                    or ddp_audit.get("local_batch_size") != 8
                    or ddp_audit.get("global_batch_size") != 64
                    or ddp_audit.get("local_batches_with_all_tasks")
                    != ddp_audit.get("local_batches")
                    or ddp_audit.get("local_batches_with_complete_pair")
                    != ddp_audit.get("local_batches")
                    or ddp_audit.get("rank_task_counts_identical") is not True
                ):
                    raise RuntimeError(
                        "curriculum build report lacks a passing DDP8 layout audit"
                    )
                curriculum_record["ddp_layout_audit"] = ddp_audit
        elif args.profile in scale_profiles:
            raise RuntimeError("canonical scale curriculum lacks its build report")
        if args.profile in scale_profiles:
            curriculum_builder = Path(curriculum_metadata["builder"]).resolve(
                strict=True
            )
            if _sha256_file(curriculum_builder) != curriculum_metadata.get(
                "builder_sha256"
            ):
                raise RuntimeError(
                    "canonical scale curriculum builder source changed after build"
                )

    source_records = {
        relative: _file_record(REPO_ROOT / relative) for relative in RUNTIME_SOURCES
    }
    sample_exposures = args.max_steps * args.batch_size * len(gpu_ids)
    report: dict[str, Any] = {
        "schema": "stable_audio_tools.p11_v4_training_launch_contract",
        "schema_version": 2,
        "status": "FROZEN_FOR_LAUNCH",
        "created_at": datetime.now().astimezone().isoformat(),
        "run_name": args.run_name,
        "profile": args.profile,
        "arm": args.arm,
        "architecture": model.get("model_type"),
        "route_id": model.get("route_id"),
        "transfusion_cot_contract": transfusion.get("contract"),
        "model_arm": transfusion.get("thought", {}).get("arm"),
        "git_revision": _git_revision(),
        "invocation": (
            f"RUN_NAME={args.run_name} P11_ARM={args.arm} "
            f"scripts/t2a/train/run_sceneplan_p11.sh {args.profile}"
        ),
        "training": {
            "gpu_ids": gpu_ids,
            "world_size": len(gpu_ids),
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": args.batch_size * len(gpu_ids),
            "num_workers_per_rank": args.num_workers,
            "benchmark_warmup_batches": args.benchmark_warmup_batches,
            "initial_global_step": initial_global_step,
            "max_optimizer_steps": args.max_steps,
            "sample_exposures": sample_exposures,
            "new_sample_exposures": (
                (args.max_steps - initial_global_step)
                * args.batch_size
                * len(gpu_ids)
            ),
            "effective_curriculum_rows": curriculum_rows,
            "nominal_curriculum_passes": (
                sample_exposures / curriculum_rows if curriculum_rows else None
            ),
            "checkpoint_every": args.checkpoint_every,
            "save_top_k": args.save_top_k,
            "seed": args.seed,
            "model_rng_contract": (
                "seed_rank_global_step_v1"
                if dataset.get("p11_v4_curriculum_ordering_contract")
                else "seed_rank_segment_start_v1"
            ),
            "precision": "bf16-mixed",
            "strategy": "ddp_static" if len(gpu_ids) > 1 else "auto",
            "ddp_bucket_cap_mb": args.ddp_bucket_cap_mb,
            "ddp_comm_hook": args.ddp_comm_hook,
            "bind_to_gpu_numa": bool(args.bind_to_gpu_numa),
            "scale_performance_gate": (
                {
                    "min_global_loader_samples_per_second": 40.0,
                    "min_rank_mean_gpu_utilization_percent": 45.0,
                    "min_peak_allocated_gib": 30.0,
                    "cross_rank_curriculum_identity_overlap_count": 0,
                }
                if args.profile in {"scale_preflight", "scale_trial"}
                else None
            ),
        },
        "model_config": {
            **_file_record(model_path),
            "resolved_sha256": _json_sha256(model),
        },
        "dataset_config": {
            **_file_record(dataset_path),
            "resolved_sha256": _json_sha256(dataset),
            "base_manifest_expected_num_samples": dataset.get(
                "expected_num_samples"
            ),
            "curriculum_expected_rows": dataset.get(
                "p11_v4_curriculum_expected_rows"
            ),
        },
        "curriculum": curriculum_record,
        "resume_checkpoint": resume_checkpoint,
        "p10_release": {
            **_file_record(p10_release_path),
            "release_id": p10_release.get("release_id"),
            "checkpoint": released_checkpoint,
            "binding_gates": p10_binding,
        },
        "runtime_sources": source_records,
    }
    report["report_sha256_without_self"] = _json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"P11_RUN_CONTRACT={args.output.resolve()}")


if __name__ == "__main__":
    main()
