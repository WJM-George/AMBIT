#!/usr/bin/env python3
"""Render a matched P11-v4 panel through the frozen canonical P10 executor.

This is deliberately a downstream closure test, not another language metric.
It consumes the exact canonical ScenePlan artifacts saved by one passing
evaluator-v10 report, verifies their hashes and scores without another model
decode, then renders predicted and target plans with the same P10 noise seed.
Editing additionally renders the current plan with that same seed so the
predicted and target waveform deltas can be compared without stochastic noise
confounding.

No waveform is retained by default.  Tensor hashes, QC, spectral agreement,
FOA active-intensity agreement, and edit-delta metrics are stored in the
machine-readable report.  The three arms can therefore run on separate GPUs
without sharing mutable audio caches; target-render hashes must agree in the
cross-arm summary.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_P10_RELEASE = (
    REPO_ROOT / "artifacts/releases/P10_SCENEPLAN_DIT_V11_150K_RELEASE.json"
)
EXPECTED_EVAL_SCHEMA = "stable_audio_tools.p11_v4_unified_challenge_eval"
EXPECTED_EVAL_VERSION = 10
TASKS = ("generation", "understanding", "editing")
CANONICAL_QWEN_KERNEL_MODE = "torch_reference"
P10_MAX_LATENT_FRAMES = 648
P10_VAE_HOP_SAMPLES = 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return float(sum(finite) / len(finite)) if finite else None


def _quality_report(path: Path, *, arm: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    report = json.loads(resolved.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema") != EXPECTED_EVAL_SCHEMA
        or int(report.get("schema_version", -1)) != EXPECTED_EVAL_VERSION
        or report.get("status") != "PASS"
        or report.get("arm") != arm
    ):
        raise RuntimeError("closure requires a passing matched evaluator-v10 report")
    if report.get("qwen_kernel_mode") != CANONICAL_QWEN_KERNEL_MODE:
        raise RuntimeError(
            "closure requires the deterministic torch_reference evaluator"
        )
    if int(report.get("draws", -1)) != 1 or report.get("k_values") != [1]:
        raise RuntimeError("closure requires the deterministic K=1 quality report")
    claimed = report.get("report_sha256_without_self")
    unhashed = dict(report)
    unhashed.pop("report_sha256_without_self", None)
    if claimed != _json_sha256(unhashed):
        raise RuntimeError("quality report self-hash mismatch")
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1:
        raise RuntimeError("closure accepts exactly one P11 checkpoint")
    report["_path"] = str(resolved)
    report["_file_sha256"] = _sha256_file(resolved)
    return report


def _valid_quality_row(row: Mapping[str, Any]) -> bool:
    scored = row.get("scored") or []
    return bool(
        len(scored) == 1
        and scored[0].get("valid") is True
        and scored[0].get("p10_conditioning_valid") is True
    )


def _select_balanced_panel(
    report: Mapping[str, Any], *, rows_per_task: int
) -> list[int]:
    """Round-robin across family/view strata, then fill deterministically."""

    selected: list[int] = []
    for task in TASKS:
        candidates = [
            row
            for row in report["aggregate_rows"]
            if row.get("task") == task and _valid_quality_row(row)
        ]
        strata: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in candidates:
            strata[(str(row["family"]), str(row["view_id"]))].append(row)
        for values in strata.values():
            values.sort(key=lambda value: int(value["ordinal"]))
        task_selected: list[int] = []
        while len(task_selected) < rows_per_task:
            advanced = False
            for key in sorted(strata):
                if not strata[key]:
                    continue
                task_selected.append(int(strata[key].pop(0)["ordinal"]))
                advanced = True
                if len(task_selected) == rows_per_task:
                    break
            if not advanced:
                break
        if len(task_selected) != rows_per_task:
            raise RuntimeError(
                f"quality report lacks {rows_per_task} valid {task} rows"
            )
        selected.extend(task_selected)
    return sorted(selected)


def _stable_seed(root_seed: int, challenge_id: str) -> int:
    payload = f"p11-v4-p10-closure-v1\0{root_seed}\0{challenge_id}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _fit_length(value: torch.Tensor, samples: int) -> torch.Tensor:
    if int(value.shape[-1]) > samples:
        return value[..., :samples]
    if int(value.shape[-1]) < samples:
        return F.pad(value, (0, samples - int(value.shape[-1])))
    return value


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.float().flatten()
    right = right.float().flatten()
    denominator = float(left.norm() * right.norm())
    if denominator <= 1.0e-12:
        return None
    return float(torch.dot(left, right) / denominator)


def _spectral_pair(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    """Compare W-channel magnitude without treating waveform phase as semantics."""

    samples = min(int(left.shape[-1]), int(right.shape[-1]))
    left_w = left[0, :samples].float()
    right_w = right[0, :samples].float()
    n_fft = 1024
    hop = 512
    if samples < n_fft:
        left_w = F.pad(left_w, (0, n_fft - samples))
        right_w = F.pad(right_w, (0, n_fft - samples))
    window = torch.hann_window(n_fft)
    left_mag = torch.stft(
        left_w,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        center=False,
        return_complex=True,
    ).abs()
    right_mag = torch.stft(
        right_w,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        center=False,
        return_complex=True,
    ).abs()
    left_log = torch.log1p(left_mag)
    right_log = torch.log1p(right_mag)
    denominator = float(right_mag.square().sum().sqrt())
    spectral_convergence = (
        None
        if denominator <= 1.0e-12
        else float((left_mag - right_mag).square().sum().sqrt() / denominator)
    )
    return {
        "w_log_magnitude_l1": float((left_log - right_log).abs().mean()),
        "w_magnitude_cosine": _cosine(left_mag, right_mag),
        "w_spectral_convergence": spectral_convergence,
        "n_fft": n_fft,
        "hop_length": hop,
    }


def _pair_metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    target_samples = int(right.shape[-1])
    fitted = _fit_length(left, target_samples)
    delta = fitted.float() - right.float()
    return {
        "length_match": int(left.shape[-1]) == target_samples,
        "waveform_exact": bool(torch.equal(fitted, right)),
        "waveform_mae": float(delta.abs().mean()),
        "waveform_rmse": float(delta.square().mean().sqrt()),
        "waveform_cosine": _cosine(fitted, right),
        "spectral": _spectral_pair(fitted, right),
    }


def _edit_delta_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    current: torch.Tensor,
) -> dict[str, Any]:
    samples = int(target.shape[-1])
    predicted = _fit_length(predicted, samples).float()
    current = _fit_length(current, samples).float()
    target = target.float()
    predicted_delta = predicted - current
    target_delta = target - current
    error = predicted_delta - target_delta
    target_rms = float(target_delta.square().mean().sqrt())
    predicted_rms = float(predicted_delta.square().mean().sqrt())
    return {
        "predicted_delta_rms": predicted_rms,
        "target_delta_rms": target_rms,
        "delta_rmse": float(error.square().mean().sqrt()),
        "delta_cosine": _cosine(predicted_delta, target_delta),
        "target_is_noop_audio": bool(torch.equal(target, current)),
        "predicted_is_noop_audio": bool(torch.equal(predicted, current)),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    groups = {"all": list(rows)}
    groups.update(
        {
            f"task:{task}": [row for row in rows if row["task"] == task]
            for task in TASKS
        }
    )
    for group, values in groups.items():
        output[group] = {
            "rows": len(values),
            "quality_task_score_mean": _mean(
                [row["quality_task_score"] for row in values]
            ),
            "predicted_target_audio_exact_rate": _mean(
                [float(row["predicted_target"]["waveform_exact"]) for row in values]
            ),
            # A predicted ScenePlan may legally choose a duration different
            # from the hidden target.  Keep that disagreement as a quality
            # metric; it is not an executor-integrity failure when each render
            # has exactly the length requested by its own ScenePlan.
            "predicted_target_length_match_rate": _mean(
                [float(row["predicted_target"]["length_match"]) for row in values]
            ),
            "predicted_target_waveform_rmse": _mean(
                [row["predicted_target"]["waveform_rmse"] for row in values]
            ),
            "predicted_target_w_magnitude_cosine": _mean(
                [
                    row["predicted_target"]["spectral"]["w_magnitude_cosine"]
                    for row in values
                ]
            ),
            "predicted_target_w_spectral_convergence": _mean(
                [
                    row["predicted_target"]["spectral"][
                        "w_spectral_convergence"
                    ]
                    for row in values
                ]
            ),
            "predicted_target_foa_spherical_error_mean_deg": _mean(
                [
                    row["predicted_target_foa"]["spherical_error_mean_deg"]
                    for row in values
                ]
            ),
            "predicted_plan_foa_spherical_error_mean_deg": _mean(
                [
                    row["predicted_plan_foa"]["spherical_error_mean_deg"]
                    for row in values
                ]
            ),
            "predicted_plan_activity_temporal_iou": _mean(
                [row["predicted_plan_activity"]["temporal_iou"] for row in values]
            ),
            "edit_delta_cosine": _mean(
                [
                    None
                    if row.get("editing_same_seed") is None
                    else row["editing_same_seed"]["delta_cosine"]
                    for row in values
                ]
            ),
        }
    return output


def _render_length_integrity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    """Check P10 against each plan, never against a different target plan."""

    maximum_samples = P10_MAX_LATENT_FRAMES * P10_VAE_HOP_SAMPLES
    return {
        "all_predicted_render_lengths_match_own_plan": all(
            int(row["predicted_shape"][-1])
            == int(row["predicted_model_num_samples"])
            for row in rows
        ),
        "all_target_render_lengths_match_own_plan": all(
            int(row["target_shape"][-1]) == int(row["target_model_num_samples"])
            for row in rows
        ),
        "all_plan_lengths_follow_vae_frame_contract": all(
            int(row["predicted_model_num_samples"])
            == int(row["predicted_latent_frames_valid"])
            * P10_VAE_HOP_SAMPLES
            and int(row["target_model_num_samples"])
            == int(row["target_latent_frames_valid"])
            * P10_VAE_HOP_SAMPLES
            for row in rows
        ),
        "all_plan_lengths_within_p10_capability": all(
            0 < int(row["predicted_model_num_samples"]) <= maximum_samples
            and 0 < int(row["target_model_num_samples"]) <= maximum_samples
            and 0 < int(row["predicted_latent_frames_valid"])
            <= P10_MAX_LATENT_FRAMES
            and 0 < int(row["target_latent_frames_valid"])
            <= P10_MAX_LATENT_FRAMES
            for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("d0", "direct", "flow"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("online", "ema"), default="ema")
    parser.add_argument("--rows-per-task", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--p10-release", type=Path, default=DEFAULT_P10_RELEASE)
    parser.add_argument("--p10-steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_per_task <= 0:
        raise ValueError("--rows-per-task must be positive")
    if args.p10_steps != 100:
        raise ValueError("claimed closure requires the canonical P10 100-step sampler")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("frozen P10 closure requires CUDA")

    quality = _quality_report(args.quality_report, arm=args.arm)
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    checkpoint_provenance = quality["checkpoints"][0]
    if str(checkpoint) != checkpoint_provenance["checkpoint"]:
        raise RuntimeError("closure checkpoint path differs from quality report")
    if _sha256_file(checkpoint) != checkpoint_provenance["checkpoint_sha256"]:
        raise RuntimeError("closure P11 checkpoint SHA256 changed")
    if args.weights != quality["weights"]:
        raise RuntimeError("closure and quality report must use the same weights")
    model_config = Path(quality["model_config"]).resolve(strict=True)
    dataset_config = Path(quality["dataset_config"]).resolve(strict=True)
    challenge_path = Path(quality["challenge"]).resolve(strict=True)
    if _sha256_file(challenge_path) != quality["challenge_sha256"]:
        raise RuntimeError("closure challenge SHA256 changed")
    selected = _select_balanced_panel(quality, rows_per_task=args.rows_per_task)
    quality_rows = {
        int(row["ordinal"]): row for row in quality["aggregate_rows"]
    }

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(args.seed)

    from scripts.t2a.eval.evaluate_sceneplan_p11_v4_challenge import (
        _dataset,
        _load_model,
        _score_draw,
    )
    from stable_audio_tools.data.scene_sketch_v1 import (
        compile_delta_scene_sketch,
        compile_execution_state,
        compile_p10_from_contract,
        compile_scene_sketch,
    )
    from stable_audio_tools.data.sceneplan_p11_single_turn import (
        P10_CANONICAL_CHECKPOINT,
        P10_CANONICAL_CHECKPOINT_SHA256,
        P10_CANONICAL_CHECKPOINT_STEP,
        P10_CANONICAL_EXECUTOR_FAMILY,
        P10_CANONICAL_MODEL_CONFIG,
        P10_CANONICAL_MODEL_CONFIG_SHA256,
        P11Task,
        finalize_sceneplan_for_p10,
    )
    from stable_audio_tools.inference.sceneplan_cot import P10ScenePlanDiTExecutor

    planner_started = time.perf_counter()
    wrapper, planner, planner_provenance = _load_model(
        model_config_path=model_config,
        checkpoint_path=checkpoint,
        device=torch.device("cpu"),
        weights=args.weights,
    )
    planner_provenance["qwen_runtime_kernels"] = planner.configure_qwen_runtime_kernels(
        quality["qwen_kernel_mode"]
    )
    challenge = _dataset(
        planner=planner,
        dataset_config_path=dataset_config,
        challenge_path=challenge_path,
    )

    def bundle(plan: Mapping[str, Any]):
        sample_id = str(plan["sample_id"])
        encoded = planner.plan_codec.encode(
            plan, max_tokens=planner.plan_max_tokens
        )
        return finalize_sceneplan_for_p10(
            planner.plan_codec,
            encoded,
            tokenizer=planner.tokenizer,
            task=P11Task.GENERATION,
            sample_id=sample_id,
        )

    bundles: list[dict[str, Any]] = []
    for ordinal in selected:
        _, metadata = challenge[ordinal]
        expected_score = quality_rows[ordinal]["scored"][0]
        plan = expected_score.get("sceneplan")
        if not isinstance(plan, Mapping):
            raise RuntimeError(
                f"quality report lacks frozen ScenePlan artifact at ordinal {ordinal}"
            )
        plan = dict(plan)
        sketch = compile_scene_sketch(plan, planner.plan_codec)
        execution = compile_execution_state(plan, planner.plan_codec)
        p10_conditions = compile_p10_from_contract(
            sketch, execution, planner.plan_codec
        )
        patch = expected_score.get("patch")
        if str(metadata["p11_task"]) == "editing":
            if not isinstance(patch, Mapping):
                raise RuntimeError(
                    f"quality report lacks editing patch at ordinal {ordinal}"
                )
            delta_sketch = compile_delta_scene_sketch(
                metadata["p11_input_sceneplan"],
                plan,
                patch,
                planner.plan_codec,
            )
            discrete_tokens = planner.delta_sketch_codec.encode(
                delta_sketch, patch
            )["input_ids"]
        else:
            discrete_tokens = planner.scene_sketch_codec.encode(sketch)[
                "input_ids"
            ]
        output = {
            "sceneplan": plan,
            "plan_tokens": planner.plan_codec.encode(
                plan, max_tokens=planner.plan_max_tokens
            )["input_ids"],
            "p10_conditions": p10_conditions,
            "patch": patch,
            "discrete_tokens": discrete_tokens,
            "thought_core": None,
            "diagnostics": {},
        }
        scored = _score_draw(planner, metadata, output)
        quality_artifact = {
            "source": "frozen_scored_sceneplan_artifact_v1",
            "plan_hash_exact": (
                scored["plan_sha256"] == expected_score["plan_sha256"]
            ),
            "semantic_hash_exact": (
                scored["semantic_sha256"] == expected_score["semantic_sha256"]
            ),
            "p10_semantic_hash_exact": (
                scored["p10_semantic_sha256"]
                == expected_score["p10_semantic_sha256"]
            ),
            "numeric_hash_exact": (
                scored["numeric_sha256"] == expected_score["numeric_sha256"]
            ),
            "discrete_tokens_hash_exact": (
                scored["discrete_tokens_sha256"]
                == expected_score["discrete_tokens_sha256"]
            ),
            "editing_patch_exact": (
                scored.get("patch") == expected_score.get("patch")
            ),
            "task_score_drift": abs(
                float(scored["task_score"])
                - float(expected_score["task_score"])
            ),
            "reference_anchor_rmse_drift": abs(
                float(scored["reference_anchor_nearest_core_rmse"])
                - float(expected_score["reference_anchor_nearest_core_rmse"])
            ),
            "task_metrics_exact": (
                scored["task_metrics"] == expected_score["task_metrics"]
            ),
            "roundtrip": bool(scored["roundtrip"]),
            "finite": bool(scored["finite"]),
        }
        quality_artifact["verified_exact"] = bool(
            all(
                quality_artifact[key]
                for key in (
                    "plan_hash_exact",
                    "semantic_hash_exact",
                    "p10_semantic_hash_exact",
                    "numeric_hash_exact",
                    "discrete_tokens_hash_exact",
                    "editing_patch_exact",
                    "task_metrics_exact",
                    "roundtrip",
                    "finite",
                )
            )
            and quality_artifact["task_score_drift"] == 0.0
            and quality_artifact["reference_anchor_rmse_drift"] == 0.0
        )

        row_bundles = {
            "ordinal": int(ordinal),
            "challenge_id": str(metadata["p11_challenge_id"]),
            "task": str(metadata["p11_task"]),
            "family": str(metadata["p11_challenge_family"]),
            "view_id": str(metadata["p11_prompt_view_id"]),
            "edit_operation": metadata.get("p11_edit_kind"),
            "quality_task_score": float(scored["task_score"]),
            "quality_plan_sha256": scored["plan_sha256"],
            "quality_semantic_sha256": scored["semantic_sha256"],
            "quality_numeric_sha256": scored["numeric_sha256"],
            "quality_artifact": quality_artifact,
            "predicted_plan": plan,
            "target_plan": metadata["p11_target_sceneplan"],
            "predicted": bundle(plan),
            "target": bundle(metadata["p11_target_sceneplan"]),
            "current": None,
        }
        if row_bundles["task"] == "editing":
            row_bundles["current"] = bundle(metadata["p11_input_sceneplan"])
        bundles.append(row_bundles)

    if not all(item["quality_artifact"]["verified_exact"] for item in bundles):
        failures = [
            {
                "ordinal": item["ordinal"],
                **item["quality_artifact"],
            }
            for item in bundles
            if not item["quality_artifact"]["verified_exact"]
        ]
        raise RuntimeError(
            "P11 frozen quality artifact failed exact verification before P10 "
            f"render: {failures}"
        )

    planner_seconds = time.perf_counter() - planner_started
    del challenge, planner, wrapper
    gc.collect()
    torch.cuda.empty_cache()

    release_path = args.p10_release.expanduser().resolve(strict=True)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("status") != "frozen_canonical" or not release.get("immutable"):
        raise RuntimeError("P10 release is not frozen canonical")
    p10_config = Path(release["model_config"]["path"]).resolve(strict=True)
    p10_checkpoint = Path(release["checkpoint"]["path"]).resolve(strict=True)
    vae_checkpoint = Path(release["pretransform"]["checkpoint"]["path"]).resolve(
        strict=True
    )
    p10_identity = {
        "release": str(release_path),
        "release_sha256": _sha256_file(release_path),
        "release_id": release["release_id"],
        "executor_family": release["executor_family"],
        "model_config": str(p10_config),
        "model_config_sha256": _sha256_file(p10_config),
        "checkpoint": str(p10_checkpoint),
        "checkpoint_step": int(release["checkpoint"]["step"]),
        "checkpoint_sha256": _sha256_file(p10_checkpoint),
        "vae_checkpoint": str(vae_checkpoint),
        "vae_checkpoint_sha256": _sha256_file(vae_checkpoint),
    }
    expected_p10 = {
        "executor_family": P10_CANONICAL_EXECUTOR_FAMILY,
        "model_config": P10_CANONICAL_MODEL_CONFIG,
        "model_config_sha256": P10_CANONICAL_MODEL_CONFIG_SHA256,
        "checkpoint": P10_CANONICAL_CHECKPOINT,
        "checkpoint_step": P10_CANONICAL_CHECKPOINT_STEP,
        "checkpoint_sha256": P10_CANONICAL_CHECKPOINT_SHA256,
        "vae_checkpoint_sha256": release["pretransform"]["checkpoint"]["sha256"],
    }
    for key, expected in expected_p10.items():
        if p10_identity[key] != expected:
            raise RuntimeError(
                f"canonical P10 identity mismatch for {key}: "
                f"{p10_identity[key]!r} != {expected!r}"
            )

    p10_load_started = time.perf_counter()
    executor = P10ScenePlanDiTExecutor.from_checkpoints(
        model_config_path=p10_config,
        checkpoint_path=p10_checkpoint,
        vae_checkpoint_path=vae_checkpoint,
        device=device,
        steps=args.p10_steps,
        cfg_scale=float(release["canonical_inference"]["cfg_scale"]),
        rescale_cfg=bool(release["canonical_inference"]["rescale_cfg"]),
        cfg_rescale_phi=float(
            release["canonical_inference"]["cfg_rescale_phi"]
        ),
        apg_scale=float(release["canonical_inference"]["apg_scale"]),
    )
    p10_load_seconds = time.perf_counter() - p10_load_started

    from scripts.t2a.eval.sceneplan_44_eval_common import audio_qc
    from scripts.t2a.eval.score_sceneplan_dit_p10_core import (
        _activity_metrics,
        _doa_metrics,
        _paired_doa_metrics,
    )

    render_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    repeat_exact: bool | None = None
    for index, item in enumerate(bundles):
        seed = _stable_seed(args.seed, item["challenge_id"])
        target_audio = executor.render(item["target"], seed=seed)
        predicted_audio = executor.render(item["predicted"], seed=seed)
        if index == 0:
            repeated = executor.render(item["target"], seed=seed)
            repeat_exact = bool(torch.equal(target_audio, repeated))
            del repeated
        target_samples = int(item["target"].model_num_samples)
        target_frames = int(item["target"].latent_frames_valid)
        predicted_samples = int(item["predicted"].model_num_samples)
        predicted_frames = int(item["predicted"].latent_frames_valid)
        predicted_for_target = _fit_length(predicted_audio, target_samples)
        row = {
            "ordinal": item["ordinal"],
            "challenge_id": item["challenge_id"],
            "task": item["task"],
            "family": item["family"],
            "view_id": item["view_id"],
            "edit_operation": item["edit_operation"],
            "render_seed": seed,
            "quality_task_score": item["quality_task_score"],
            "quality_plan_sha256": item["quality_plan_sha256"],
            "quality_semantic_sha256": item["quality_semantic_sha256"],
            "quality_numeric_sha256": item["quality_numeric_sha256"],
            "quality_artifact": item["quality_artifact"],
            "target_plan_sha256": _json_sha256(item["target_plan"]),
            "predicted_plan_equals_target": (
                item["predicted_plan"] == item["target_plan"]
            ),
            "predicted_shape": list(predicted_audio.shape),
            "target_shape": list(target_audio.shape),
            "predicted_model_num_samples": predicted_samples,
            "target_model_num_samples": target_samples,
            "predicted_latent_frames_valid": predicted_frames,
            "target_latent_frames_valid": target_frames,
            "predicted_foa_sha256": _tensor_sha256(predicted_audio),
            "target_foa_sha256": _tensor_sha256(target_audio),
            "predicted_qc": audio_qc(predicted_audio),
            "target_qc": audio_qc(target_audio),
            "predicted_target": _pair_metrics(predicted_audio, target_audio),
            "predicted_target_foa": _paired_doa_metrics(
                predicted_for_target,
                target_audio,
                item["target_plan"],
                model_num_samples=target_samples,
                latent_frames=target_frames,
            ),
            "predicted_plan_foa": _doa_metrics(
                predicted_audio,
                item["predicted_plan"],
                model_num_samples=predicted_samples,
                latent_frames=predicted_frames,
            ),
            "target_plan_foa": _doa_metrics(
                target_audio,
                item["target_plan"],
                model_num_samples=target_samples,
                latent_frames=target_frames,
            ),
            "predicted_plan_activity": _activity_metrics(
                predicted_audio,
                item["predicted_plan"],
                model_num_samples=predicted_samples,
                latent_frames=predicted_frames,
            ),
            "target_plan_activity": _activity_metrics(
                target_audio,
                item["target_plan"],
                model_num_samples=target_samples,
                latent_frames=target_frames,
            ),
            "editing_same_seed": None,
        }
        if item["current"] is not None:
            current_audio = executor.render(item["current"], seed=seed)
            row["current_shape"] = list(current_audio.shape)
            row["current_foa_sha256"] = _tensor_sha256(current_audio)
            row["current_qc"] = audio_qc(current_audio)
            row["editing_same_seed"] = _edit_delta_metrics(
                predicted_audio, target_audio, current_audio
            )
            del current_audio
        rows.append(row)
        del target_audio, predicted_audio, predicted_for_target

    render_seconds = time.perf_counter() - render_started
    exact_rows = [row for row in rows if row["predicted_plan_equals_target"]]
    integrity_gates = {
        "quality_report_pass": True,
        "quality_prediction_artifact_verified_exact": all(
            row["quality_artifact"]["verified_exact"] for row in rows
        ),
        "canonical_p10_release_identity": True,
        "canonical_100_step_sampler": args.p10_steps == 100,
        "same_seed_repeat_exact": repeat_exact is True,
        "all_predicted_renders_finite_foa": all(
            row["predicted_qc"].get("finite") is True
            and row["predicted_qc"].get("channels") == 4
            for row in rows
        ),
        "all_target_renders_finite_foa": all(
            row["target_qc"].get("finite") is True
            and row["target_qc"].get("channels") == 4
            for row in rows
        ),
        **_render_length_integrity(rows),
        "at_least_one_exact_plan_anchor": bool(exact_rows),
        "exact_plan_implies_exact_same_seed_audio": bool(exact_rows)
        and all(row["predicted_target"]["waveform_exact"] for row in exact_rows),
        "editing_uses_same_seed_triplet": all(
            row["editing_same_seed"] is not None
            for row in rows
            if row["task"] == "editing"
        ),
    }
    status = "PASS" if all(integrity_gates.values()) else "FAIL"
    report: dict[str, Any] = {
        "schema": "stable_audio_tools.p11_v4_frozen_p10_foa_closure",
        "schema_version": 3,
        "status": status,
        "scope": (
            "P11 planning consequences measured through one frozen canonical "
            "P10 oracle executor ceiling; this report does not re-evaluate or "
            "attribute the frozen executor's residual quality to P11"
        ),
        "attribution_contract": {
            "contract": "p11_plan_error_with_frozen_p10_oracle_ceiling_v1",
            "primary_p11_objects": [
                "SceneSketch_or_DeltaSketch",
                "ExecutionState_or_DeltaThought",
                "assembled_ScenePlan_or_AtomicPatch",
            ],
            "oracle_ceiling_path": (
                "target_ScenePlan -> frozen_P10_v11_150k(seed42) -> oracle_FOA"
            ),
            "p11_system_path": (
                "input -> P11_predicted_ScenePlan -> same_frozen_P10(seed42) "
                "-> predicted_FOA"
            ),
            "p10_checkpoint_and_sampler_fixed": True,
            "same_p10_noise_seed_per_paired_plan": True,
            "p10_oracle_plan_residual_is_p11_error": False,
            "p10_improvement_is_out_of_scope": True,
            "foa_difference_interpretation": (
                "downstream consequence of a P11 plan difference under the "
                "fixed executor, not an independent P11 waveform-quality score"
            ),
        },
        "arm": args.arm,
        "weights": args.weights,
        "root_seed": args.seed,
        "selection_contract": "round_robin_family_view_balanced_by_task_v1",
        "quality_prediction_artifact_contract": {
            "contract": "frozen_scored_sceneplan_artifact_v1",
            "inference_rerun": False,
            "sceneplan_hash": "exact",
            "semantic_and_p10_caption_hash": "exact",
            "numeric_hash": "exact",
            "discrete_tokens_hash": "exact",
            "editing_patch": "exact",
            "task_metrics": "exact_recompute",
        },
        "rows_per_task": args.rows_per_task,
        "selected_ordinals": selected,
        "quality_report": {
            "path": quality["_path"],
            "file_sha256": quality["_file_sha256"],
            "report_sha256_without_self": quality["report_sha256_without_self"],
            "challenge_sha256": quality["challenge_sha256"],
            "dataset_config": quality["dataset_config"],
            "evaluator_contract": quality["evaluator_contract"],
            "discrete_decode_mode": quality["discrete_decode_mode"],
            "qwen_kernel_mode": quality["qwen_kernel_mode"],
        },
        "p11": planner_provenance,
        "p10": p10_identity,
        "sampling": release["canonical_inference"],
        "integrity_gates": integrity_gates,
        "performance": {
            "planner_decode_seconds": 0.0,
            "p11_artifact_validation_seconds": planner_seconds,
            "p10_load_seconds": p10_load_seconds,
            "p10_render_seconds": render_seconds,
            "p10_renders": len(rows) * 2
            + sum(row["task"] == "editing" for row in rows)
            + 1,
            "p10_renders_per_second": (
                (
                    len(rows) * 2
                    + sum(row["task"] == "editing" for row in rows)
                    + 1
                )
                / render_seconds
            ),
            "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (2**30),
        },
        "aggregate": _aggregate(rows),
        "rows": rows,
        "quality_decision": "REQUIRES_MATCHED_CROSS_ARM_CLOSURE_SUMMARY",
    }
    report["report_sha256_without_self"] = _json_sha256(report)
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
