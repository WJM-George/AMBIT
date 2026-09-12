#!/usr/bin/env python3
"""Freeze and verify the P11-v4 medium-scale post-training protocol.

The protocol is written before the candidate step-10k checkpoint exists.  It
pins the planner-only evaluation surface, all eight GPU assignments, immutable
inputs, evaluator entry points, historical-baseline scope, and the frozen P10
oracle executor.  The post-training runner verifies every pinned byte before it
starts any scientific GPU job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SCHEMA = "stable_audio_tools.p11_v4_scale10k_eval_protocol"
SCHEMA_VERSION = 1
VERIFICATION_SCHEMA = "stable_audio_tools.p11_v4_scale10k_eval_protocol_verification"
EXPECTED_CHECKPOINT_STEP = 10_000
EXPECTED_SEED = 42
EXPECTED_WORLD_SIZE = 8
EXPECTED_BATCH_PER_GPU = 8

EVALUATION_SOURCE_PATHS = (
    "scripts/t2a/eval/manage_sceneplan_p11_v4_scale_eval_protocol.py",
    "scripts/t2a/eval/run_sceneplan_p11_v4_scale_eval_8gpu.sh",
    "scripts/t2a/eval/watch_sceneplan_p11_v4_scale_eval.sh",
    "scripts/t2a/eval/monitor_sceneplan_p11_v4_gpu_usage.py",
    "scripts/t2a/eval/evaluate_sceneplan_p11_v4_challenge.py",
    "scripts/t2a/eval/evaluate_sceneplan_p11_v4.py",
    "scripts/t2a/eval/evaluate_sceneplan_p11_v4_sketch_exposure.py",
    "scripts/t2a/eval/project_sceneplan_p11_v4_k1_quality.py",
    "scripts/t2a/eval/summarize_sceneplan_p11_v4_u_inventory.py",
    "scripts/t2a/eval/summarize_sceneplan_p11_v4_scale_decision.py",
    "scripts/t2a/eval/run_sceneplan_p11_v4_lexical_pair_same_gpu.sh",
    "scripts/t2a/eval/summarize_sceneplan_p11_v4_lexical_ab.py",
    "scripts/t2a/eval/summarize_sceneplan_p11_v4_cross_process_repro.py",
    "scripts/t2a/eval/merge_sceneplan_p11_v4_challenge_shards.py",
    "scripts/t2a/eval/evaluate_sceneplan_p11_v4_p10_closure.py",
    "scripts/t2a/eval/summarize_sceneplan_p11_v4_p10_closure.py",
    "artifacts/sceneplan_p11/protocol_snapshots/"
    "summarize_sceneplan_p11_v4_scale_decision_v1_sha5b8f3dfb.py",
    "stable_audio_tools/configuration.py",
    "stable_audio_tools/data/scene_sketch_v1.py",
    "stable_audio_tools/data/sceneplan_p11_v4_dataset.py",
    "stable_audio_tools/models/scene_thought_p11_v4.py",
    "stable_audio_tools/models/sceneplan_p11_v4.py",
    "stable_audio_tools/inference/sceneplan_cot.py",
)

GPU_ASSIGNMENTS = {
    "gpu_0": "generation_k148",
    "gpu_1": "understanding_degraded_k148",
    "gpu_2": "exact_compatibility_k148",
    "gpu_3": "editing_counterfactual_k148",
    "gpu_4": "thought_causal_interventions",
    "gpu_5": "same_gpu_reliable_asr_on_drop_pair",
    "gpu_6": "cross_process_reference_replay",
    "gpu_7": "understanding_scene_sketch_exposure_u40",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, *, require_self_hash: bool = False) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    claimed = value.get("report_sha256_without_self")
    if require_self_hash and not isinstance(claimed, str):
        raise RuntimeError(f"missing report self-hash: {path}")
    if claimed is not None:
        unhashed = dict(value)
        unhashed.pop("report_sha256_without_self", None)
        if claimed != _json_sha256(unhashed):
            raise RuntimeError(f"report self-hash mismatch: {path}")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"immutable input is not a regular file: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(resolved),
    }


def _resolved_config_record(path: Path) -> dict[str, Any]:
    from stable_audio_tools.configuration import load_config

    resolved = path.expanduser().resolve(strict=True)
    config = load_config(resolved)
    return {
        "path": str(resolved),
        "leaf_sha256": _sha256_file(resolved),
        "resolved_sha256": _json_sha256(config),
    }


def _dataset_dependency_paths(dataset_config: Path) -> dict[str, Path]:
    from stable_audio_tools.configuration import load_config

    config = load_config(dataset_config)
    dependencies: dict[str, Path] = {}
    for key in ("manifest_path", "semantic_cache_path", "lexical_cache_path"):
        value = config.get(key)
        if value:
            dependencies[f"heldout_dataset:{key}"] = Path(str(value))
    for index, dataset in enumerate(config.get("datasets") or []):
        value = dataset.get("path") if isinstance(dataset, Mapping) else None
        if value:
            dependencies[f"heldout_dataset:index_{index}"] = Path(str(value))
    codec_value = config.get("codec_path")
    if codec_value:
        codec_root = Path(str(codec_value)).expanduser().resolve(strict=True)
        if not codec_root.is_dir():
            raise RuntimeError(f"codec_path is not a directory: {codec_root}")
        codec_files = sorted(path for path in codec_root.rglob("*") if path.is_file())
        if not codec_files:
            raise RuntimeError(f"codec_path has no files: {codec_root}")
        for path in codec_files:
            dependencies[f"heldout_dataset:codec:{path.relative_to(codec_root)}"] = path
    return dependencies


def _baseline_summary(path: Path, *, scope: str) -> dict[str, Any]:
    report = _load_json(path)
    checkpoints = report.get("checkpoints") or []
    checkpoint = checkpoints[0] if len(checkpoints) == 1 else {}
    return {
        "path": str(path.resolve(strict=True)),
        "schema": report.get("schema"),
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "arm": report.get("arm"),
        "checkpoint": checkpoint.get("checkpoint"),
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "model_config_resolved_sha256": checkpoint.get(
            "model_config_resolved_sha256"
        ),
        "comparison_scope": scope,
    }


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    value["report_sha256_without_self"] = _json_sha256(value)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _freeze(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.expanduser().resolve(strict=True)
    run_root = args.run_root.expanduser().resolve(strict=True)
    checkpoint_dir = run_root / "checkpoints"
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"candidate checkpoint directory is missing: {checkpoint_dir}")
    checkpoint_glob = f"*step={EXPECTED_CHECKPOINT_STEP}.ckpt"
    existing_checkpoints = sorted(checkpoint_dir.glob(checkpoint_glob))
    if existing_checkpoints:
        raise RuntimeError(
            "protocol must be frozen before the candidate checkpoint exists: "
            + ", ".join(str(path) for path in existing_checkpoints)
        )

    launch_contract_path = run_root / "training_launch_contract.json"
    launch = _load_json(launch_contract_path, require_self_hash=True)
    training = launch.get("training") or {}
    if launch.get("status") != "FROZEN_FOR_LAUNCH":
        raise RuntimeError("candidate launch contract is not frozen")
    launch_expectations = {
        "arm_is_canonical": launch.get("arm") == "canonical",
        "profile_is_scale_trial": launch.get("profile") == "scale_trial",
        "seed_is_42": int(training.get("seed", -1)) == EXPECTED_SEED,
        "world_size_is_8": int(training.get("world_size", -1))
        == EXPECTED_WORLD_SIZE,
        "batch_per_gpu_is_8": int(training.get("batch_size_per_gpu", -1))
        == EXPECTED_BATCH_PER_GPU,
        "global_batch_is_64": int(training.get("global_batch_size", -1)) == 64,
        "max_optimizer_steps_is_10000": int(
            training.get("max_optimizer_steps", -1)
        )
        == EXPECTED_CHECKPOINT_STEP,
        "checkpoint_every_is_10000": int(training.get("checkpoint_every", -1))
        == EXPECTED_CHECKPOINT_STEP,
    }
    if not all(launch_expectations.values()):
        raise RuntimeError(f"candidate launch contract mismatch: {launch_expectations}")

    model_config = args.model_config.expanduser().resolve(strict=True)
    dataset_config = args.dataset_config.expanduser().resolve(strict=True)
    p10_release_path = args.p10_release.expanduser().resolve(strict=True)
    p10_capability_path = args.p10_capability.expanduser().resolve(strict=True)
    p10_release = _load_json(p10_release_path)
    p10_capability = _load_json(p10_capability_path)
    if p10_release.get("status") != "frozen_canonical" or not p10_release.get(
        "immutable"
    ):
        raise RuntimeError("P10 release is not immutable frozen_canonical")
    if p10_release.get("release_id") != "p10-sceneplan-dit-v11-step150000":
        raise RuntimeError("unexpected frozen P10 release id")
    if p10_capability.get("contract") != "p10_sceneplan_44_capability_v1":
        raise RuntimeError("unexpected P10 capability contract")

    immutable_paths: dict[str, Path] = {
        "candidate_training_launch_contract": launch_contract_path,
        "candidate_model_config_leaf": model_config,
        "heldout_dataset_config_leaf": dataset_config,
        "heldout_challenge": args.challenge,
        "baseline_d0_historical": args.d0_baseline,
        "baseline_direct_historical_pre_inventory_repair": args.direct_baseline,
        "baseline_flow_pre_inventory_repair": args.screen_flow_baseline,
        "p10_release": p10_release_path,
        "p10_capability_contract": p10_capability_path,
        "p10_checkpoint": Path(str(p10_release["checkpoint"]["path"])),
        "p10_model_config": Path(str(p10_release["model_config"]["path"])),
        "p10_vae_checkpoint": Path(
            str(p10_release["pretransform"]["checkpoint"]["path"])
        ),
        "p10_vae_model_config": Path(
            str(p10_release["pretransform"]["model_config"]["path"])
        ),
    }
    immutable_paths.update(_dataset_dependency_paths(dataset_config))
    for relative in EVALUATION_SOURCE_PATHS:
        immutable_paths[f"source:{relative}"] = repo_root / relative

    immutable_inputs = {
        label: _file_record(path)
        for label, path in sorted(immutable_paths.items())
    }
    p10_checkpoint_record = immutable_inputs["p10_checkpoint"]
    declared_p10_sha = str(p10_release["checkpoint"]["sha256"])
    if p10_checkpoint_record["sha256"] != declared_p10_sha:
        raise RuntimeError("frozen P10 checkpoint hash differs from release")
    if (
        p10_capability["executor"]["canonical_checkpoint_sha256"]
        != declared_p10_sha
    ):
        raise RuntimeError("P10 release and capability checkpoint hashes disagree")

    resolved_model = _resolved_config_record(model_config)
    resolved_dataset = _resolved_config_record(dataset_config)
    launch_model = launch.get("model_config") or {}
    if launch_model.get("resolved_sha256") != resolved_model["resolved_sha256"]:
        raise RuntimeError("runtime model recipe differs from frozen training launch")

    frozen_at_unix_ns = time.time_ns()
    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_CANDIDATE_CHECKPOINT",
        "created_at": datetime.now().astimezone().isoformat(),
        "frozen_at_unix_ns": frozen_at_unix_ns,
        "candidate_run": {
            "run_name": launch.get("run_name"),
            "run_root": str(run_root),
            "launch_contract_sha256": immutable_inputs[
                "candidate_training_launch_contract"
            ]["sha256"],
            "launch_contract_self_sha256": launch.get(
                "report_sha256_without_self"
            ),
            "checkpoint_step": EXPECTED_CHECKPOINT_STEP,
            "expected_checkpoint_glob": str(checkpoint_dir / checkpoint_glob),
            "candidate_checkpoint_absent_at_freeze": True,
            "candidate_checkpoint_matches_at_freeze": [],
            "launch_expectations": launch_expectations,
        },
        "scientific_scope": {
            "selection_target": "P11_planner_only",
            "primary_evidence": (
                "pre_render SceneSketch/DeltaSketch, ExecutionState/DeltaThought, "
                "AtomicPatch, assembled ScenePlan, posterior, and intervention metrics"
            ),
            "p10_role": "fixed_oracle_executor_ceiling",
            "p10_release_id": p10_release.get("release_id"),
            "p10_checkpoint_sha256": declared_p10_sha,
            "p10_improvement_or_model_selection_in_scope": False,
            "p10_oracle_rendering_residual_is_p11_error": False,
            "p10_closure_role": (
                "secondary integration and audible causal-consequence evidence only"
            ),
        },
        "evaluation": {
            "weights": "ema",
            "seed": EXPECTED_SEED,
            "qwen_scientific_kernel": "torch_reference",
            "discrete_decode_mode": "prefix_recompute",
            "draws": 8,
            "k_values": [1, 4, 8],
            "rows_per_primary_view": 30,
            "challenge_families": [
                "generation_numeric_posterior",
                "understanding_degraded_evidence",
                "exact_compatibility",
                "editing_counterfactual_causality",
            ],
            "gpu_count": EXPECTED_WORLD_SIZE,
            "gpu_assignment_contract": (
                "p11_posttrain_eight_independent_diagnostics_v1"
            ),
            "gpu_assignments": GPU_ASSIGNMENTS,
            "telemetry": {
                "interval_sec": 5,
                "all_eight_gpus_must_observe_cuda_process": True,
                "all_eight_gpus_must_observe_nonzero_utilization": True,
                "missing_assignment_invalidates_operational_evidence": True,
            },
            "frozen_threshold_authority": (
                "artifacts/sceneplan_p11/protocol_snapshots/"
                "summarize_sceneplan_p11_v4_scale_decision_v1_sha5b8f3dfb.py"
            ),
            "thresholds_may_change_after_candidate_results": False,
        },
        "baseline_scope": {
            "comparison_contract": (
                "p11_u_inventory_repair_isolation_then_matched_method_ab_v1"
            ),
            "d0": _baseline_summary(
                args.d0_baseline,
                scope="historical_accuracy_reference_pre_inventory_repair",
            ),
            "flow_pre_repair": _baseline_summary(
                args.screen_flow_baseline,
                scope="same_arm_inventory_repair_isolation_baseline",
            ),
            "direct_pre_repair": _baseline_summary(
                args.direct_baseline,
                scope="historical_reference_not_final_matched_comparator",
            ),
            "matched_direct_10k_required_before_canonical_promotion": True,
            "matched_direct_launch_condition": (
                "only if repaired Flow-R1 preserves unique posterior/causal "
                "capability and shows a positive inventory-repair signal"
            ),
        },
        "resolved_configs": {
            "candidate_model": resolved_model,
            "heldout_dataset": resolved_dataset,
        },
        "immutable_inputs": immutable_inputs,
    }
    _write_new_json(args.output, protocol)
    print(json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _verify(args: argparse.Namespace) -> int:
    verification_started_ns = time.time_ns()
    protocol_path = args.protocol.expanduser().resolve(strict=True)
    protocol = _load_json(protocol_path, require_self_hash=True)
    if (
        protocol.get("schema") != SCHEMA
        or int(protocol.get("schema_version", -1)) != SCHEMA_VERSION
        or protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_CHECKPOINT"
    ):
        raise RuntimeError("unexpected evaluation protocol schema or status")

    mismatches: list[dict[str, Any]] = []
    immutable = protocol.get("immutable_inputs") or {}
    for label, expected in immutable.items():
        path = Path(str(expected["path"])).resolve(strict=True)
        actual_bytes = path.stat().st_size
        actual_sha = _sha256_file(path)
        if actual_bytes != int(expected["bytes"]) or actual_sha != expected["sha256"]:
            mismatches.append(
                {
                    "label": label,
                    "path": str(path),
                    "expected_bytes": int(expected["bytes"]),
                    "actual_bytes": int(actual_bytes),
                    "expected_sha256": expected["sha256"],
                    "actual_sha256": actual_sha,
                }
            )
    for label, expected in (protocol.get("resolved_configs") or {}).items():
        actual = _resolved_config_record(Path(str(expected["path"])))
        if actual["resolved_sha256"] != expected["resolved_sha256"]:
            mismatches.append(
                {
                    "label": f"resolved_config:{label}",
                    "path": actual["path"],
                    "expected_sha256": expected["resolved_sha256"],
                    "actual_sha256": actual["resolved_sha256"],
                }
            )
    if mismatches:
        raise RuntimeError(
            "frozen evaluation protocol mismatch: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    match = re.search(r"step=([0-9]+)\.ckpt$", checkpoint.name)
    checkpoint_step = int(match.group(1)) if match else -1
    candidate = protocol["candidate_run"]
    expected_root = Path(str(candidate["run_root"])).resolve(strict=True)
    checkpoint_belongs_to_run = checkpoint.parent == expected_root / "checkpoints"
    checkpoint_predates_protocol = (
        checkpoint.stat().st_mtime_ns <= int(protocol["frozen_at_unix_ns"])
    )
    checkpoint_gates = {
        "checkpoint_step_is_10000": checkpoint_step == EXPECTED_CHECKPOINT_STEP,
        "checkpoint_belongs_to_frozen_run": checkpoint_belongs_to_run,
        "protocol_predates_candidate_checkpoint": not checkpoint_predates_protocol,
        "candidate_was_absent_at_protocol_freeze": candidate.get(
            "candidate_checkpoint_absent_at_freeze"
        )
        is True,
    }
    if not all(checkpoint_gates.values()):
        raise RuntimeError(f"candidate checkpoint violates protocol: {checkpoint_gates}")

    report: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "verified_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "path": str(protocol_path),
            "bytes": protocol_path.stat().st_size,
            "sha256": _sha256_file(protocol_path),
            "self_sha256": protocol.get("report_sha256_without_self"),
            "status": protocol.get("status"),
        },
        "candidate_checkpoint": _file_record(checkpoint),
        "checkpoint_gates": checkpoint_gates,
        "immutable_inputs_verified": len(immutable),
        "immutable_input_mismatches": [],
        "resolved_configs_verified": len(protocol.get("resolved_configs") or {}),
        "gpu_assignment_contract": protocol["evaluation"][
            "gpu_assignment_contract"
        ],
        "gpu_assignments": protocol["evaluation"]["gpu_assignments"],
        "selection_target": protocol["scientific_scope"]["selection_target"],
        "p10_role": protocol["scientific_scope"]["p10_role"],
        "elapsed_verification_sec": None,
    }
    report["elapsed_verification_sec"] = float(
        (time.time_ns() - verification_started_ns) / 1e9
    )
    if args.output is not None:
        _write_new_json(args.output, report)
    else:
        report["report_sha256_without_self"] = _json_sha256(report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--run-root", type=Path, required=True)
    freeze.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    freeze.add_argument("--model-config", type=Path, required=True)
    freeze.add_argument("--dataset-config", type=Path, required=True)
    freeze.add_argument("--challenge", type=Path, required=True)
    freeze.add_argument("--d0-baseline", type=Path, required=True)
    freeze.add_argument("--direct-baseline", type=Path, required=True)
    freeze.add_argument("--screen-flow-baseline", type=Path, required=True)
    freeze.add_argument(
        "--p10-release",
        type=Path,
        default=REPO_ROOT / "artifacts/releases/P10_SCENEPLAN_DIT_V11_150K_RELEASE.json",
    )
    freeze.add_argument(
        "--p10-capability",
        type=Path,
        default=REPO_ROOT
        / "docs/sceneplan_v2/p10_sceneplan_44_capability_v1_20260830.json",
    )
    freeze.set_defaults(func=_freeze)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--protocol", type=Path, required=True)
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument("--output", type=Path)
    verify.set_defaults(func=_verify)
    return parser


def main() -> int:
    args = _parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
