#!/usr/bin/env python3
"""Nested K-sample posterior diagnostic for P11-v4 Flow-R1.

The discrete SceneSketch/DeltaSketch is decoded once per example.  K explicit,
sample-isolated noise seeds vary only the continuous ExecutionState.  This
script measures validity, semantic immutability, unquantized and executable
diversity, and best-of-K target coverage.  It is not a full P10 FOA render gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_MODEL = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
)
DEFAULT_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_pilot90_transfusion_cot_v4_reliable_asr_v1.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/p11_v4_flow_r1_posterior_pilot_20260901.json"
)
_GENERATION_DURATION = re.compile(
    r"^Create a ([0-9]+(?:\.[0-9]+)?)-second FOA(?: spatial-audio)? scene(?:\.| )"
)


def _ids(value: Any) -> torch.Tensor:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    return torch.as_tensor(value, dtype=torch.long).flatten().cpu()


def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _duration_from_prompt(text: str) -> float | None:
    match = _GENERATION_DURATION.match(" ".join(str(text).split()))
    return None if match is None else float(match.group(1))


def _semantic_signature(sketch: Mapping[str, Any]) -> dict[str, Any]:
    source_fields = (
        "source_id",
        "kind",
        "description",
        "speaker_description",
        "transcript",
    )
    return {
        "room_intent": sketch.get("room_intent"),
        "sources": [
            {key: source[key] for key in source_fields if key in source}
            for source in sketch.get("sources", [])
        ],
    }


def _p10_numeric_sha256(p10: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in (
        "source_event_frame_ids",
        "source_trajectory_features",
        "source_present_mask",
    ):
        value = np.ascontiguousarray(np.asarray(p10["sceneplan_44"][key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _sample_seed(root_seed: int, row_key: str, draw_index: int) -> int:
    payload = f"p11-v4-flow-posterior-v1\0{root_seed}\0{row_key}\0{draw_index}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _select_rows(dataset: Any, rows_per_task: int) -> list[int]:
    selected = {task: [] for task in ("generation", "understanding", "editing")}
    for ordinal in range(len(dataset)):
        _, metadata = dataset[ordinal]
        task = str(metadata["p11_task"])
        if len(selected[task]) >= rows_per_task:
            continue
        if task == "editing" and not selected[task] and metadata.get(
            "p11_edit_kind"
        ) not in {"rotate_source", "distance_source", "retime_source"}:
            continue
        selected[task].append(ordinal)
        if all(len(values) == rows_per_task for values in selected.values()):
            break
    if not all(len(values) == rows_per_task for values in selected.values()):
        raise RuntimeError(f"could not select balanced posterior panel: {selected}")
    return sorted(value for values in selected.values() for value in values)


def _active_mask(metadata: Mapping[str, Any], *, editing: bool) -> torch.Tensor:
    target = torch.as_tensor(
        metadata["p11_v4_target_source_mask"], dtype=torch.bool
    ).flatten()
    if editing:
        target |= torch.as_tensor(
            metadata["p11_v4_input_source_mask"], dtype=torch.bool
        ).flatten()
    slots = torch.cat([torch.ones(1, dtype=torch.bool), target])
    return slots[:, None].expand(5, 15)


def _pairwise_rmse(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.shape[0] <= 1:
        return 0.0
    distances = []
    for left, right in itertools.combinations(range(values.shape[0]), 2):
        distances.append(float((values[left][mask] - values[right][mask]).square().mean().sqrt()))
    return float(sum(distances) / len(distances))


def _core_target_metrics(
    values: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    values = values.detach().float().cpu()
    target = target.detach().float().cpu()
    errors = (values[:, mask] - target[mask]).square().mean(dim=1).sqrt()
    lower = values[:, mask].amin(dim=0)
    upper = values[:, mask].amax(dim=0)
    covered = (target[mask] >= lower) & (target[mask] <= upper)
    return {
        "mean_draw_rmse": float(errors.mean()),
        "best_draw_rmse": float(errors.min()),
        "ensemble_mean_rmse": float(
            (values[:, mask].mean(dim=0) - target[mask]).square().mean().sqrt()
        ),
        "target_coordinate_interval_coverage": float(covered.float().mean()),
    }


def _summarize_prefix(
    draws: list[dict[str, Any]],
    *,
    metadata: Mapping[str, Any],
    k: int,
) -> dict[str, Any]:
    one = draws[:k]
    task = str(metadata["p11_task"])
    editing = task == "editing"
    mask = _active_mask(metadata, editing=editing)
    thought = torch.stack([draw["thought_core"] for draw in one]).float().cpu()
    assembled = torch.stack([draw["assembled_core"] for draw in one]).float().cpu()
    thought_target = torch.as_tensor(
        metadata[
            "p11_v4_delta_execution_core"
            if editing
            else "p11_v4_target_execution_core"
        ],
        dtype=torch.float32,
    )
    assembled_target = torch.as_tensor(
        metadata["p11_v4_target_execution_core"], dtype=torch.float32
    )
    semantic_hashes = {draw["semantic_sha256"] for draw in one}
    p10_caption_hashes = {draw["p10_semantic_sha256"] for draw in one}
    noise_hashes = {draw["noise_sha256"] for draw in one}
    plan_hashes = {draw["plan_sha256"] for draw in one}
    numeric_hashes = {draw["p10_numeric_sha256"] for draw in one}
    patch_hashes = {
        _json_sha256(draw["patch"]) for draw in one if draw["patch"] is not None
    }
    task_scores = [float(draw["task_metrics"]["task_score"]) for draw in one]
    return {
        "k": k,
        "valid_rate": sum(bool(draw["valid"]) for draw in one) / k,
        "parse_rate": sum(bool(draw["parsed"]) for draw in one) / k,
        "roundtrip_rate": sum(bool(draw["roundtrip_exact"]) for draw in one) / k,
        "all_finite": all(bool(draw["finite"]) for draw in one),
        "discrete_authority_shared": all(
            draw["discrete_sha256"] == one[0]["discrete_sha256"] for draw in one
        ),
        "semantic_immutability": len(semantic_hashes) == 1
        and len(p10_caption_hashes) == 1,
        "unique_noise_count": len(noise_hashes),
        "unique_unquantized_thought_count": len(
            {_tensor_sha256(draw["thought_core"]) for draw in one}
        ),
        "unique_sceneplan_count": len(plan_hashes),
        "unique_executable_numeric_count": len(numeric_hashes),
        "unique_atomic_patch_count": len(patch_hashes) if editing else None,
        "thought_pairwise_rmse": _pairwise_rmse(thought, mask),
        "assembled_pairwise_rmse": _pairwise_rmse(assembled, mask),
        "task_score_mean": float(sum(task_scores) / len(task_scores)),
        "task_score_best": float(max(task_scores)),
        "final_plan_exact_any": any(bool(draw["final_plan_exact"]) for draw in one),
        "atomic_patch_exact_any": (
            any(draw["atomic_patch_exact"] is True for draw in one)
            if editing
            else None
        ),
        "thought_target": _core_target_metrics(thought, thought_target, mask),
        "assembled_target": _core_target_metrics(assembled, assembled_target, mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows-per-task", type=int, default=3)
    parser.add_argument("--k-values", default="1,4,8")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    k_values = sorted({int(value) for value in args.k_values.split(",")})
    if not k_values or k_values[0] <= 0 or k_values[-1] > 32:
        raise ValueError("K values must be unique integers within [1,32]")
    if args.rows_per_task <= 0:
        raise ValueError("rows-per-task must be positive")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    from stable_audio_tools.configuration import load_config
    from stable_audio_tools.data.scene_sketch_v1 import execution_state_core
    from stable_audio_tools.data.sceneplan_p11_metrics import score_p11_prediction
    from stable_audio_tools.data.sceneplan_p11_single_turn import (
        validate_p11_executor_profile,
    )
    from stable_audio_tools.data.sceneplan_p11_v4_dataset import ScenePlanP11V4Dataset
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.training.factory import create_training_wrapper_from_config

    model_config_path = args.model_config.resolve(strict=True)
    dataset_config_path = args.dataset_config.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    model_config = load_config(model_config_path)
    dataset_config = load_config(dataset_config_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model = create_model_from_config(model_config)
    wrapper = create_training_wrapper_from_config(model_config, model)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    checkpoint_step = int(checkpoint.get("global_step", -1))
    del checkpoint
    gc.collect()
    wrapper.eval().requires_grad_(False).to(device)
    weights = "online"
    ema_step = None
    if not args.online:
        if wrapper.p11_ema is None:
            raise RuntimeError("P11-v4 checkpoint does not expose EMA")
        ema_step = int(wrapper.p11_ema.step.detach().cpu())
        wrapper.p11_ema.copy_to(wrapper.p11)
        weights = "ema"
    planner = wrapper.p11
    if planner.execution_reasoner.continuous_objective != "rectified_flow":
        raise ValueError("posterior evaluator requires the Flow-R1 arm")

    sources = dataset_config.get("datasets") or []
    if len(sources) != 1:
        raise ValueError("posterior evaluator requires one frozen source index")
    dataset = ScenePlanP11V4Dataset(
        dataset_config["manifest_path"],
        index_path=sources[0]["path"],
        codec_path=dataset_config["codec_path"],
        tokenizer_spec=(planner.tokenizer, 512, None),
        expected_num_samples=int(dataset_config["expected_num_samples"]),
        index_num_samples=int(dataset_config["index_num_samples"]),
        require_frozen=bool(dataset_config.get("require_complete", True)),
        semantic_cache_path=dataset_config.get("semantic_cache_path"),
        semantic_dim=int(dataset_config.get("semantic_dim", 512)),
        semantic_encoder_revision=dataset_config.get("semantic_encoder_revision"),
        lexical_evidence_mode=str(dataset_config.get("lexical_evidence_mode", "none")),
        lexical_max_tokens=int(dataset_config.get("lexical_max_tokens", 128)),
        lexical_cache_path=dataset_config.get("lexical_cache_path"),
        lexical_encoder_revision=dataset_config.get("lexical_encoder_revision"),
        lexical_confidence_threshold=dataset_config.get(
            "lexical_confidence_threshold"
        ),
    )
    selected = _select_rows(dataset, args.rows_per_task)
    max_k = max(k_values)
    rows = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for ordinal in selected:
        _, metadata = dataset[ordinal]
        task = str(metadata["p11_task"])
        sample_id = str(metadata["p11_target_sample_id"])
        prompt_fingerprint = _json_sha256(metadata["p11_prompt_text"])
        row_key = f"{task}\0{sample_id}\0{prompt_fingerprint}"
        input_foa = metadata.get("p11_input_foa")
        if input_foa is not None:
            input_foa = torch.as_tensor(input_foa)
            valid_frames = int(torch.as_tensor(metadata["p11_input_valid_mask"]).sum())
            input_foa = input_foa[:, :valid_frames]
        if task == "generation":
            duration_sec = _duration_from_prompt(metadata["p11_prompt_text"])
        elif task == "understanding":
            duration_sec = float(input_foa.shape[-1]) * 1024.0 / 44_100.0
        else:
            duration_sec = float(metadata["p11_input_sceneplan"]["duration_sec"])
        seeds = [_sample_seed(args.seed, row_key, index) for index in range(max_k)]
        row = {
            "ordinal": ordinal,
            "sample_id": sample_id,
            "task": task,
            "edit_kind": metadata.get("p11_edit_kind"),
            "prompt_sha256": prompt_fingerprint,
            "noise_seeds": seeds,
        }
        try:
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with torch.inference_mode(), autocast:
                outputs = planner.decode_transfusion_cot_samples(
                    metadata["p11_prompt"],
                    task=task,
                    noise_seeds=seeds,
                    input_foa=input_foa,
                    input_semantic=metadata.get("p11_input_semantic"),
                    input_lexical=metadata.get("p11_input_lexical"),
                    input_sceneplan=metadata.get("p11_input_sceneplan"),
                    duration_sec=duration_sec,
                    sample_id=sample_id,
                    temperature=0.0,
                )
            draws = []
            for output in outputs:
                plan_ids = _ids(output["plan_tokens"])
                decoded = planner.plan_codec.decode(plan_ids, sample_id=sample_id)
                validate_p11_executor_profile(decoded)
                recoded = _ids(planner.plan_codec.encode(decoded))
                assembled_core = torch.from_numpy(
                    execution_state_core(output["execution_state"])
                ).float()
                task_metrics = score_p11_prediction(
                    task=task,
                    target_plan=metadata["p11_target_sceneplan"],
                    prediction=output["sceneplan"],
                    input_plan=metadata.get("p11_input_sceneplan"),
                    source_matching=metadata["p11_source_matching"],
                    editing_score_version=metadata["p11_editing_score_version"],
                    known_field_groups=(
                        metadata.get("p11_prompt_known_field_groups")
                        if task == "generation"
                        else None
                    ),
                )
                draws.append(
                    {
                        "seed": int(output["diagnostics"]["thought_noise_seeds"][0]),
                        "noise_sha256": output["diagnostics"]["thought_noise_sha256"],
                        "parsed": True,
                        "roundtrip_exact": bool(torch.equal(plan_ids, recoded)),
                        "valid": True,
                        "finite": bool(
                            torch.isfinite(output["thought_core"]).all()
                            and torch.isfinite(assembled_core).all()
                        ),
                        "discrete_sha256": hashlib.sha256(
                            _ids(output["discrete_tokens"]).numpy().tobytes()
                        ).hexdigest(),
                        "semantic_sha256": _json_sha256(
                            _semantic_signature(output["scene_sketch"])
                        ),
                        "p10_semantic_sha256": _json_sha256(
                            output["p10_conditions"]["semantic_caption"]
                        ),
                        "plan_sha256": _json_sha256(output["sceneplan"]),
                        "p10_numeric_sha256": _p10_numeric_sha256(
                            output["p10_conditions"]
                        ),
                        "thought_core": output["thought_core"].detach().float().cpu(),
                        "assembled_core": assembled_core,
                        "final_plan_exact": bool(
                            torch.equal(
                                plan_ids,
                                _ids(metadata["p11_v4_final_sceneplan_tokens"]),
                            )
                        ),
                        "atomic_patch_exact": (
                            None
                            if task != "editing"
                            else bool(
                                torch.equal(
                                    _ids(output["patch_tokens"]),
                                    _ids(metadata["p11_v4_atomic_patch_tokens"]),
                                )
                            )
                        ),
                        "patch": output["patch"],
                        "task_metrics": task_metrics,
                    }
                )
            row["prefixes"] = {
                str(k): _summarize_prefix(draws, metadata=metadata, k=k)
                for k in k_values
            }
            row["draws"] = [
                {
                    key: value
                    for key, value in draw.items()
                    if key not in {"thought_core", "assembled_core"}
                }
                for draw in draws
            ]
        except Exception as error:  # noqa: BLE001 - fail-closed report retains row.
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)

    elapsed = time.perf_counter() - started
    by_task = {}
    for task in ("generation", "understanding", "editing"):
        task_rows = [row for row in rows if row["task"] == task and "prefixes" in row]
        by_task[task] = {}
        for k in k_values:
            prefix = [row["prefixes"][str(k)] for row in task_rows]
            by_task[task][str(k)] = {
                "rows": len(prefix),
                "valid_rate": _mean([value["valid_rate"] for value in prefix]),
                "semantic_immutability_rate": _mean(
                    [float(value["semantic_immutability"]) for value in prefix]
                ),
                "nondegenerate_unquantized_rate": _mean(
                    [
                        float(value["unique_unquantized_thought_count"] > 1)
                        for value in prefix
                    ]
                ),
                "nondegenerate_executable_rate": _mean(
                    [
                        float(value["unique_executable_numeric_count"] > 1)
                        for value in prefix
                    ]
                ),
                "unique_sceneplans_mean": _mean(
                    [float(value["unique_sceneplan_count"]) for value in prefix]
                ),
                "thought_pairwise_rmse": _mean(
                    [value["thought_pairwise_rmse"] for value in prefix]
                ),
                "assembled_pairwise_rmse": _mean(
                    [value["assembled_pairwise_rmse"] for value in prefix]
                ),
                "task_score_mean": _mean(
                    [value["task_score_mean"] for value in prefix]
                ),
                "task_score_best": _mean(
                    [value["task_score_best"] for value in prefix]
                ),
                "best_draw_rmse": _mean(
                    [value["assembled_target"]["best_draw_rmse"] for value in prefix]
                ),
                "target_coordinate_interval_coverage": _mean(
                    [
                        value["assembled_target"][
                            "target_coordinate_interval_coverage"
                        ]
                        for value in prefix
                    ]
                ),
            }

    structural_pass = all(
        "prefixes" in row
        and all(
            row["prefixes"][str(k)]["valid_rate"] == 1.0
            and row["prefixes"][str(k)]["roundtrip_rate"] == 1.0
            and row["prefixes"][str(k)]["all_finite"]
            and row["prefixes"][str(k)]["discrete_authority_shared"]
            and row["prefixes"][str(k)]["semantic_immutability"]
            and row["prefixes"][str(k)]["unique_noise_count"] == k
            for k in k_values
        )
        for row in rows
    )
    report = {
        "schema": "stable_audio_tools.p11_v4_flow_posterior_eval",
        "schema_version": 1,
        "status": "PASS" if structural_pass else "FAIL",
        "scope": (
            "nested K-sample learned-output diagnostic; P10 conditioning only, "
            "not rendered FOA and not a full-training promotion"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "weights": weights,
        "ema_step": ema_step,
        "model_config": str(model_config_path),
        "dataset_config": str(dataset_config_path),
        "reasoning_arm": planner.transfusion_cot_arm,
        "continuous_objective": planner.execution_reasoner.continuous_objective,
        "seed_schedule": (
            "sha256(root_seed,task,sample_id,prompt_sha256,draw_index)_mod_2^63"
        ),
        "root_seed": args.seed,
        "k_values": k_values,
        "selected_ordinals": selected,
        "rows": len(rows),
        "elapsed_seconds": elapsed,
        "rows_per_second": len(rows) / elapsed,
        "peak_memory_gib": (
            torch.cuda.max_memory_allocated(device) / (2**30)
            if device.type == "cuda"
            else None
        ),
        "structural_posterior_gate": structural_pass,
        "quality_claim": "not_established_by_this_gate",
        "by_task": by_task,
        "rows_detail": rows,
    }
    report["report_sha256_without_self"] = _json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
