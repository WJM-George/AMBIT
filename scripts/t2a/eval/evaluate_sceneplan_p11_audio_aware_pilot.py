#!/usr/bin/env python3
"""Learned pilot gate for the active audio-aware P11 contract.

This evaluator deliberately keeps P10 out of the scoring loop.  It checks the
planner boundary itself: constrained ScenePlan/Patch decoding, the shared U/E
FOA observation stage, fallible-prior isolation, and executable continuous
thought interventions.  P10 closure is a later, separately attributed gate.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_audio_aware_v1.json"
)
DEFAULT_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_audio_aware_v1_pilot90.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/audio_aware_v1/"
    "pilot90_learned_gate_seed42.json"
)
CANONICAL_DECODE_MODE = "prefix_recompute"


def _ids(value: Any) -> torch.Tensor:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    return torch.as_tensor(value, dtype=torch.long).flatten().cpu()


def _json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_signature(plan: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "source_id",
        "kind",
        "description",
        "speaker_description",
        "transcript",
    )
    return {
        "room": plan.get("room"),
        "sources": [
            {key: source[key] for key in fields if key in source}
            for source in plan.get("sources", [])
        ],
    }


def _numeric_signature(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "duration_sec": plan.get("duration_sec"),
        "sources": [
            {
                "source_id": source.get("source_id"),
                "activity": source.get("activity"),
                "trajectory": source.get("trajectory"),
                "gain_db": source.get("gain_db"),
            }
            for source in plan.get("sources", [])
        ],
    }


def _tensor_rmse(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    left_tensor = torch.as_tensor(left).detach().float().cpu()
    right_tensor = torch.as_tensor(right).detach().float().cpu()
    if left_tensor.shape != right_tensor.shape:
        return None
    return float((left_tensor - right_tensor).square().mean().sqrt())


def _masked_rmse(prediction: Any, target: Any, mask: Any) -> float | None:
    prediction = torch.as_tensor(prediction).detach().float().cpu()
    target = torch.as_tensor(target).detach().float().cpu()
    mask = torch.as_tensor(mask).detach().bool().cpu()
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("continuous diagnostic tensors do not align")
    if not bool(mask.any()):
        return None
    return float((prediction[mask] - target[mask]).square().mean().sqrt())


def _conditional_edit_target(
    planner: Any,
    metadata: Mapping[str, Any],
    observed_plan: Mapping[str, Any],
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any]]:
    """Apply the requested edit to the model's own audio observation.

    The frozen manifest patch is still the end-to-end reference.  This second
    target isolates Editing from upstream U error.  ``replace_source`` is the
    only operation whose concrete Atomic Patch repeats observed numeric fields;
    its instruction says to preserve execution, so those fields must come from
    the model-observed plan rather than the hidden oracle observation.
    """

    patch = copy.deepcopy(dict(metadata["p11_edit_spec"]))
    if str(patch.get("operation")) == "replace_source":
        source_id = str(patch["source_id"])
        current = next(
            source
            for source in observed_plan["sources"]
            if str(source["source_id"]) == source_id
        )
        replacement = patch["new_source"]
        replacement["activity"] = copy.deepcopy(current["activity"])
        replacement["trajectory"] = copy.deepcopy(current["trajectory"])
        replacement["gain_db"] = float(current.get("gain_db", 0.0))
    patch_tokens = planner.patch_codec.encode(patch)["input_ids"]
    canonical_patch = planner.patch_codec.decode(patch_tokens)
    target = planner.patch_codec.apply(observed_plan, canonical_patch)
    return canonical_patch, patch_tokens, target


def _token_label(codec: Any, token_id: int) -> str:
    grammar = {value: key for key, value in codec.token_to_id.items()}
    if int(token_id) in grammar:
        return grammar[int(token_id)]
    piece = int(token_id) - int(codec.text_offset)
    if 0 <= piece < int(codec.text_vocab_size):
        return f"text:{codec.text_processor.id_to_piece(piece)}"
    return f"unknown:{int(token_id)}"


def _first_token_error(codec: Any, predicted: Any, target: Any) -> dict[str, Any] | None:
    left = _ids(predicted).tolist()
    right = _ids(target).tolist()
    limit = min(len(left), len(right))
    index = next((value for value in range(limit) if left[value] != right[value]), limit)
    if index == len(left) == len(right):
        return None
    left_id = None if index >= len(left) else int(left[index])
    right_id = None if index >= len(right) else int(right[index])
    return {
        "index": index,
        "predicted_id": left_id,
        "predicted_token": None if left_id is None else _token_label(codec, left_id),
        "target_id": right_id,
        "target_token": None if right_id is None else _token_label(codec, right_id),
        "predicted_length": len(left),
        "target_length": len(right),
    }


def _select_rows(dataset: Any, rows_per_task: int) -> tuple[list[int], int, int]:
    selected: dict[str, list[int]] = defaultdict(list)
    retime_ordinal = -1
    second_edit_ordinal = -1
    for ordinal in range(len(dataset)):
        _, metadata = dataset[ordinal]
        task = str(metadata["p11_task"])
        if len(selected[task]) < rows_per_task:
            selected[task].append(ordinal)
        if task == "editing":
            if metadata.get("p11_edit_kind") == "retime_source" and retime_ordinal < 0:
                retime_ordinal = ordinal
            if metadata.get("p11_edit_kind") != "retime_source" and second_edit_ordinal < 0:
                second_edit_ordinal = ordinal
        if (
            all(len(selected[name]) >= rows_per_task for name in ("generation", "understanding", "editing"))
            and retime_ordinal >= 0
            and second_edit_ordinal >= 0
        ):
            break
    if not all(
        len(selected[name]) == rows_per_task
        for name in ("generation", "understanding", "editing")
    ):
        raise RuntimeError(f"could not select a balanced pilot panel: {dict(selected)}")
    if retime_ordinal < 0 or second_edit_ordinal < 0:
        raise RuntimeError("pilot lacks the E intervention rows")
    ordinals = sorted(
        ordinal
        for task in ("generation", "understanding", "editing")
        for ordinal in selected[task]
    )
    return ordinals, retime_ordinal, second_edit_ordinal


def _model_inputs(metadata: Mapping[str, Any]) -> dict[str, Any]:
    task = str(metadata["p11_task"])
    kwargs: dict[str, Any] = {
        "task": task,
        "sample_id": str(metadata["p11_target_sample_id"]),
        "temperature": 0.0,
        "discrete_seed": 42,
        "noise_seed": 42,
    }
    if task in {"understanding", "editing"}:
        kwargs.update(
            {
                "input_foa": metadata["p11_input_foa"],
                "input_valid_mask": metadata["p11_input_valid_mask"],
                "input_semantic": metadata["p11_input_semantic"],
                "input_lexical": metadata.get("p11_input_lexical"),
            }
        )
    if task == "editing":
        kwargs["input_sceneplan"] = metadata.get("p11_prior_sceneplan")
    else:
        kwargs["duration_sec"] = float(
            metadata["p11_target_sceneplan"]["duration_sec"]
        )
    return kwargs


def _decode(planner: Any, metadata: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    kwargs = _model_inputs(metadata)
    kwargs.update(overrides)
    return planner.decode_transfusion_cot(metadata["p11_prompt"], **kwargs)


def _summarize_output(
    planner: Any,
    metadata: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    from stable_audio_tools.data.sceneplan_p11_metrics import score_p11_prediction
    from stable_audio_tools.data.sceneplan_p11_single_turn import (
        validate_p11_executor_profile,
    )

    task = str(metadata["p11_task"])
    plan = output["sceneplan"]
    validate_p11_executor_profile(plan)
    plan_ids = _ids(output["plan_tokens"])
    decoded = planner.plan_codec.decode(
        plan_ids, sample_id=str(metadata["p11_target_sample_id"])
    )
    validate_p11_executor_profile(decoded)
    recoded = _ids(planner.plan_codec.encode(decoded))
    target_plan_ids = _ids(planner.plan_codec.encode(metadata["p11_target_sceneplan"]))
    patch_roundtrip = None
    patch_exact = None
    predicted_patch = None
    target_patch = None
    if task == "editing":
        patch_ids = _ids(output["patch_tokens"])
        atomic_target = metadata["p11_atomic_patch_tokens"]
        patch_roundtrip = bool(
            torch.equal(
                patch_ids,
                _ids(planner.patch_codec.canonicalize(patch_ids)),
            )
        )
        patch_exact = bool(torch.equal(patch_ids, _ids(atomic_target)))
        predicted_patch = output["patch"]
        target_patch = metadata["p11_edit_spec"]
        predicted_discrete = output["delta_tokens"]
        target_discrete = metadata["p11_delta_scene_sketch_tokens"]
    else:
        predicted_discrete = output["discrete_tokens"]
        target_discrete = metadata["p11_observed_scene_sketch_tokens"]
    observed_plan = output.get("observed_sceneplan")
    end_to_end_metrics = score_p11_prediction(
        task=task,
        target_plan=metadata["p11_target_sceneplan"],
        prediction=plan,
        input_plan=metadata.get("p11_prior_sceneplan"),
        observed_plan=metadata.get("p11_observed_sceneplan_target"),
        source_matching=str(metadata["p11_source_matching"]),
        editing_score_version=str(metadata["p11_editing_score_version"]),
        known_field_groups=(
            metadata.get("p11_prompt_known_field_groups")
            if task == "generation"
            else None
        ),
    )
    metrics = end_to_end_metrics
    conditional_patch = None
    conditional_patch_exact = None
    conditional_edit_applicable = None
    conditional_edit_error = None
    reference_patch_exact = patch_exact
    if task == "editing":
        from stable_audio_tools.data.model_sceneplan_codec import (
            ModelScenePlanCodecError,
        )

        if observed_plan is None:
            raise RuntimeError("audio-aware Editing output lacks observed ScenePlan")
        try:
            conditional_patch, conditional_patch_tokens, conditional_target = (
                _conditional_edit_target(planner, metadata, observed_plan)
            )
            metrics = score_p11_prediction(
                task=task,
                target_plan=conditional_target,
                prediction=plan,
                input_plan=metadata.get("p11_prior_sceneplan"),
                observed_plan=observed_plan,
                source_matching=str(metadata["p11_source_matching"]),
                editing_score_version=str(metadata["p11_editing_score_version"]),
            )
            conditional_edit_applicable = True
            conditional_patch_exact = bool(
                torch.equal(
                    _ids(output["patch_tokens"]),
                    _ids(conditional_patch_tokens),
                )
            )
        except (ModelScenePlanCodecError, StopIteration, KeyError) as error:
            # An audio observation can omit the instructed source or assign an
            # incompatible kind (for example speech -> sound).  That is an
            # upstream observation/compatibility failure, not a parser crash.
            # Keep end-to-end and observation scores auditable, and assign the
            # conditional pipeline a zero rather than silently dropping it.
            conditional_edit_applicable = False
            conditional_edit_error = f"{type(error).__name__}: {error}"
            conditional_patch_exact = False
            metrics = {
                "scene_score": 0.0,
                "task_score": 0.0,
                "conditional_applicable": 0.0,
            }
        if conditional_edit_applicable:
            metrics["conditional_applicable"] = 1.0
        metrics.update(
            {
                f"end_to_end_{key}": float(value)
                for key, value in end_to_end_metrics.items()
            }
        )
        observation_metrics = score_p11_prediction(
            task="understanding",
            target_plan=metadata["p11_observed_sceneplan_target"],
            prediction=observed_plan,
            source_matching=str(metadata["p11_source_matching"]),
        )
        metrics.update(
            {
                f"observation_{key}": float(value)
                for key, value in observation_metrics.items()
            }
        )
        patch_exact = conditional_patch_exact
    thought = (
        output.get("delta_thought_core")
        if task == "editing"
        else output.get("thought_core")
    )
    continuous_metrics: dict[str, float] = {}
    if task == "editing":
        from stable_audio_tools.data.scene_sketch_v1 import (
            audio_aware_delta_control_mask,
        )

        target_delta = torch.as_tensor(metadata["p11_delta_execution_core"]).float()
        predicted_delta = torch.as_tensor(
            output["delta_raw_thought_core"]
        ).detach().float().cpu()
        target_program = planner.delta_sketch_codec.decode(
            metadata["p11_delta_scene_sketch_tokens"]["input_ids"]
        )
        control_mask = torch.from_numpy(
            audio_aware_delta_control_mask(target_program)
        )
        control_rmse = _masked_rmse(predicted_delta, target_delta, control_mask)
        zero_control_rmse = _masked_rmse(
            torch.zeros_like(target_delta), target_delta, control_mask
        )
        if control_rmse is not None:
            continuous_metrics["delta_control_rmse"] = control_rmse
            continuous_metrics["delta_zero_control_rmse"] = float(
                zero_control_rmse
            )
            if zero_control_rmse and zero_control_rmse > 0.0:
                continuous_metrics["delta_control_skill_over_zero"] = float(
                    1.0 - (control_rmse / zero_control_rmse) ** 2
                )
        continuous_metrics["delta_control_coordinates"] = float(
            control_mask.sum()
        )

        observed_target = torch.as_tensor(
            metadata["p11_observed_execution_core"]
        ).float()
        observed_prediction = torch.as_tensor(
            output["observed_raw_thought_core"]
        ).detach().float().cpu()
        observed_mask = torch.zeros_like(observed_target, dtype=torch.bool)
        observed_mask[0, 0] = True
        for source_index, present in enumerate(
            torch.as_tensor(metadata["p11_observed_source_mask"]).bool().tolist()
        ):
            if present:
                observed_mask[source_index + 1, 1:] = True
        continuous_metrics["observation_core_rmse"] = float(
            _masked_rmse(observed_prediction, observed_target, observed_mask)
        )
    else:
        target_core = torch.as_tensor(metadata["p11_observed_execution_core"]).float()
        continuous_metrics["thought_core_rmse"] = float(
            _tensor_rmse(thought, target_core)
        )
    diagnostics = output.get("diagnostics") or {}
    if task == "editing":
        observation_diagnostics = diagnostics.get("observed") or {}
        delta_diagnostics = diagnostics.get("delta") or {}
        budget_forced_tokens = int(
            observation_diagnostics.get("budget_forced_tokens", 0)
        ) + int(delta_diagnostics.get("budget_forced_tokens", 0))
        grammar_interventions = int(
            observation_diagnostics.get("grammar_interventions", 0)
        ) + int(delta_diagnostics.get("grammar_interventions", 0))
    else:
        budget_forced_tokens = int(diagnostics.get("budget_forced_tokens", 0))
        grammar_interventions = int(diagnostics.get("grammar_interventions", 0))
    finite = bool(torch.isfinite(torch.as_tensor(thought)).all())
    if output.get("observed_thought_core") is not None:
        finite = finite and bool(
            torch.isfinite(torch.as_tensor(output["observed_thought_core"])).all()
        )
    return {
        "valid": True,
        "parsed": True,
        "roundtrip_exact": bool(torch.equal(plan_ids, recoded)),
        "plan_exact": bool(torch.equal(plan_ids, target_plan_ids)),
        "patch_roundtrip_exact": patch_roundtrip,
        "patch_exact": patch_exact,
        "reference_patch_exact": reference_patch_exact,
        "conditional_patch_exact": conditional_patch_exact,
        "conditional_edit_applicable": conditional_edit_applicable,
        "conditional_edit_error": conditional_edit_error,
        "predicted_patch": predicted_patch,
        "target_patch": target_patch,
        "conditional_target_patch": conditional_patch,
        "discrete_exact": bool(
            torch.equal(_ids(predicted_discrete), _ids(target_discrete))
        ),
        "first_discrete_token_error": _first_token_error(
            planner.plan_codec, predicted_discrete, target_discrete
        ),
        "finite": finite,
        "budget_forced_tokens": budget_forced_tokens,
        "grammar_interventions": grammar_interventions,
        "task_score": float(metrics["task_score"]),
        "scene_score": float(metrics["scene_score"]),
        "metrics": {
            str(key): float(value)
            for key, value in {**metrics, **continuous_metrics}.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        },
        "plan_sha256": _json_hash(plan),
        "semantic_sha256": _json_hash(_semantic_signature(plan)),
        "numeric_sha256": _json_hash(_numeric_signature(plan)),
        "patch_sha256": (
            None if output.get("patch") is None else _json_hash(output["patch"])
        ),
        "thought_core": torch.as_tensor(thought).detach().float().cpu(),
        "observed_plan_sha256": (
            None if observed_plan is None else _json_hash(observed_plan)
        ),
        "observed_semantic_sha256": (
            None
            if observed_plan is None
            else _json_hash(_semantic_signature(observed_plan))
        ),
        "observed_numeric_sha256": (
            None
            if observed_plan is None
            else _json_hash(_numeric_signature(observed_plan))
        ),
        "observed_thought_core": (
            None
            if output.get("observed_thought_core") is None
            else torch.as_tensor(output["observed_thought_core"])
            .detach()
            .float()
            .cpu()
        ),
    }


def _comparison(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "plan_changed": reference["plan_sha256"] != candidate["plan_sha256"],
        "semantic_changed": reference["semantic_sha256"] != candidate["semantic_sha256"],
        "numeric_changed": reference["numeric_sha256"] != candidate["numeric_sha256"],
        "patch_changed": reference["patch_sha256"] != candidate["patch_sha256"],
        "thought_rmse": _tensor_rmse(reference["thought_core"], candidate["thought_core"]),
        "observed_plan_changed": (
            reference["observed_plan_sha256"] != candidate["observed_plan_sha256"]
        ),
        "observed_semantic_changed": (
            reference["observed_semantic_sha256"]
            != candidate["observed_semantic_sha256"]
        ),
        "observed_numeric_changed": (
            reference["observed_numeric_sha256"]
            != candidate["observed_numeric_sha256"]
        ),
        "observed_thought_rmse": _tensor_rmse(
            reference["observed_thought_core"], candidate["observed_thought_core"]
        ),
    }


def _public_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if not isinstance(item, torch.Tensor)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        help="pre-audited hash, used to avoid re-reading one checkpoint per worker",
    )
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--decode-mode",
        choices=("cached", "prefix_recompute"),
        default=CANONICAL_DECODE_MODE,
    )
    parser.add_argument(
        "--allow-noncanonical-decode-diagnostic",
        action="store_true",
        help=(
            "allow cached decoding for an explicitly non-canonical diagnostic; "
            "such outputs are rejected by the full-pilot merger"
        ),
    )
    parser.add_argument("--rows-per-task", type=int, default=3)
    parser.add_argument(
        "--ordinals",
        help="comma-separated manifest rows for a disjoint baseline-only worker",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--skip-interventions", action="store_true")
    args = parser.parse_args()
    if args.seed != 42:
        raise ValueError("active P11 evaluation seed is frozen to 42")
    if (
        args.decode_mode != CANONICAL_DECODE_MODE
        and not args.allow_noncanonical_decode_diagnostic
    ):
        raise ValueError(
            "active P11 learned evaluation requires prefix_recompute; cached "
            "decoding is diagnostic-only because it fails Qwen3.5 decision parity"
        )
    if args.rows_per_task <= 0:
        raise ValueError("rows-per-task must be positive")
    if args.ordinals and not args.skip_interventions:
        raise ValueError("explicit ordinal workers must use --skip-interventions")

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)

    from stable_audio_tools.configuration import load_config
    from stable_audio_tools.data.sceneplan_p11_v4_dataset import (
        ScenePlanP11AudioAwareDataset,
    )
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.training.factory import create_training_wrapper_from_config

    model_config_path = args.model_config.resolve(strict=True)
    dataset_config_path = args.dataset_config.resolve(strict=True)
    checkpoint_path = args.checkpoint.resolve(strict=True)
    if args.checkpoint_sha256 is not None and (
        len(args.checkpoint_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.checkpoint_sha256)
    ):
        raise ValueError("checkpoint SHA256 must be 64 lowercase hexadecimal digits")
    model_config = load_config(model_config_path)
    dataset_config = load_config(dataset_config_path)
    if model_config.get("model_type") != "sceneplan_p11_audio_aware_v1":
        raise ValueError("learned gate requires the active audio-aware P11 model")
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
    weight_kind = "online"
    ema_step = None
    if not args.online:
        if wrapper.p11_ema is None:
            raise RuntimeError("active P11 checkpoint lacks EMA")
        ema_step = int(wrapper.p11_ema.step.detach().cpu())
        wrapper.p11_ema.copy_to(wrapper.p11)
        weight_kind = "ema"
    planner = wrapper.p11
    planner.discrete_decode_mode = args.decode_mode

    sources = dataset_config.get("datasets") or []
    if len(sources) != 1:
        raise ValueError("learned pilot gate requires one frozen P10 source index")
    dataset = ScenePlanP11AudioAwareDataset(
        dataset_config["manifest_path"],
        index_path=sources[0]["path"],
        codec_path=dataset_config["codec_path"],
        tokenizer_spec=(planner.tokenizer, 512, None),
        expected_num_samples=int(dataset_config["expected_num_samples"]),
        index_num_samples=int(dataset_config["index_num_samples"]),
        require_frozen=bool(dataset_config.get("require_complete", True)),
        semantic_cache_path=dataset_config["semantic_cache_path"],
        semantic_dim=int(dataset_config["semantic_dim"]),
        semantic_encoder_revision=str(dataset_config["semantic_encoder_revision"]),
        lexical_evidence_mode=str(dataset_config.get("lexical_evidence_mode", "none")),
        lexical_max_tokens=int(dataset_config.get("lexical_max_tokens", 128)),
        lexical_cache_path=dataset_config.get("lexical_cache_path"),
        lexical_encoder_revision=dataset_config.get("lexical_encoder_revision"),
        lexical_confidence_threshold=dataset_config.get(
            "lexical_confidence_threshold"
        ),
    )
    if args.ordinals:
        selected = sorted(
            {
                int(value)
                for value in str(args.ordinals).split(",")
                if value.strip()
            }
        )
        if not selected or selected[0] < 0 or selected[-1] >= len(dataset):
            raise ValueError("explicit pilot ordinals lie outside the manifest")
        retime_ordinal = -1
        second_edit_ordinal = -1
    else:
        selected, retime_ordinal, second_edit_ordinal = _select_rows(
            dataset, args.rows_per_task
        )
    selected_task_counts: dict[str, int] = defaultdict(int)
    for ordinal in selected:
        selected_task_counts[str(dataset[ordinal][1]["p11_task"])] += 1
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    decoded: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else torch.autocast("cpu", enabled=False)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), autocast:
        for ordinal in selected:
            _, metadata = dataset[ordinal]
            row = {
                "ordinal": ordinal,
                "task": metadata["p11_task"],
                "sample_id": metadata["p11_target_sample_id"],
                "edit_kind": metadata.get("p11_edit_kind"),
                "editing_evidence_mode": metadata.get("p11_editing_evidence_mode"),
            }
            try:
                output = _decode(planner, metadata)
                summary = _summarize_output(planner, metadata, output)
                decoded[ordinal] = summary
                row.update(_public_summary(summary))
            except Exception as error:  # keep the complete panel auditable
                failure = {
                    "ordinal": ordinal,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }
                partial = getattr(error, "partial_tokens", None)
                if partial is not None:
                    target_discrete = (
                        metadata["p11_delta_scene_sketch_tokens"]
                        if metadata["p11_task"] == "editing"
                        else metadata["p11_observed_scene_sketch_tokens"]
                    )
                    partial_ids = _ids(partial)
                    failure.update(
                        {
                            "partial_token_count": int(partial_ids.numel()),
                            "first_discrete_token_error": _first_token_error(
                                planner.plan_codec,
                                partial_ids,
                                target_discrete,
                            ),
                            "partial_tail": [
                                _token_label(planner.plan_codec, int(value))
                                for value in partial_ids[-16:].tolist()
                            ],
                        }
                    )
                row.update({"valid": False, "error": repr(error)})
                failures.append(failure)
            rows.append(row)

        interventions: dict[str, Any] = {}
        if not args.skip_interventions and not failures:
            u_ordinals = [ordinal for ordinal in selected if dataset[ordinal][1]["p11_task"] == "understanding"]
            u_ordinal = u_ordinals[0]
            u_swap_ordinal = u_ordinals[1] if len(u_ordinals) > 1 else next(
                ordinal
                for ordinal in range(len(dataset))
                if ordinal != u_ordinal and dataset[ordinal][1]["p11_task"] == "understanding"
            )
            _, u_meta = dataset[u_ordinal]
            _, u_swap = dataset[u_swap_ordinal]
            u_base = decoded[u_ordinal]
            u_variants = {
                "zero_foa": _decode(
                    planner,
                    u_meta,
                    input_foa=torch.zeros_like(u_meta["p11_input_foa"]),
                ),
                "swap_foa": _decode(
                    planner,
                    u_meta,
                    input_foa=u_swap["p11_input_foa"],
                    input_valid_mask=u_swap["p11_input_valid_mask"],
                ),
                "zero_semantic": _decode(
                    planner,
                    u_meta,
                    input_semantic=torch.zeros_like(u_meta["p11_input_semantic"]),
                ),
                "swap_semantic": _decode(
                    planner,
                    u_meta,
                    input_semantic=u_swap["p11_input_semantic"],
                ),
                "noise_seed_43": _decode(planner, u_meta, noise_seed=43),
                "zero_thought": _decode(
                    planner, u_meta, scene_thought_intervention="zero_thought"
                ),
                "shuffle_source_slots": _decode(
                    planner,
                    u_meta,
                    scene_thought_intervention="shuffle_source_slots",
                ),
            }
            interventions["understanding"] = {
                name: _comparison(
                    u_base, _summarize_output(planner, u_meta, output)
                )
                for name, output in u_variants.items()
            }

            _, e_meta = dataset[retime_ordinal]
            _, e_instruction_swap = dataset[second_edit_ordinal]
            e_base_output = _decode(planner, e_meta)
            e_base = _summarize_output(planner, e_meta, e_base_output)
            corrupt_prior = e_instruction_swap["p11_observed_sceneplan_target"]
            e_variants = {
                "prior_none": _decode(planner, e_meta, input_sceneplan=None),
                "prior_correct": _decode(
                    planner,
                    e_meta,
                    input_sceneplan=e_meta["p11_observed_sceneplan_target"],
                ),
                "prior_corrupt_cross_scene": _decode(
                    planner, e_meta, input_sceneplan=corrupt_prior
                ),
                "instruction_swap": planner.decode_transfusion_cot(
                    e_instruction_swap["p11_prompt"],
                    **_model_inputs(e_meta),
                ),
                "zero_foa": _decode(
                    planner,
                    e_meta,
                    input_foa=torch.zeros_like(e_meta["p11_input_foa"]),
                ),
                "zero_semantic": _decode(
                    planner,
                    e_meta,
                    input_semantic=torch.zeros_like(e_meta["p11_input_semantic"]),
                ),
                "zero_observation_thought": _decode(
                    planner,
                    e_meta,
                    observation_thought_intervention="zero_thought",
                ),
                "zero_delta_thought": _decode(
                    planner,
                    e_meta,
                    delta_thought_intervention="zero_thought",
                ),
                "flip_retime_delta": _decode(
                    planner,
                    e_meta,
                    delta_thought_intervention="flip_retime_delta",
                ),
            }
            interventions["editing"] = {
                name: _comparison(
                    e_base, _summarize_output(planner, e_meta, output)
                )
                for name, output in e_variants.items()
            }
            interventions["editing"]["row"] = {
                "ordinal": retime_ordinal,
                "edit_kind": e_meta["p11_edit_kind"],
                "instruction_swap_ordinal": second_edit_ordinal,
            }

    valid_rows = [row for row in rows if row.get("valid")]
    by_task: dict[str, dict[str, Any]] = {}
    for task in ("generation", "understanding", "editing"):
        task_rows = [row for row in valid_rows if row["task"] == task]
        expected_rows = int(selected_task_counts.get(task, 0))
        by_task[task] = {
            "rows": len(task_rows),
            "expected_rows": expected_rows,
            "parse_rate": (
                len(task_rows) / expected_rows if expected_rows else None
            ),
            "roundtrip_rate": (
                sum(bool(row["roundtrip_exact"]) for row in task_rows)
                / expected_rows
                if expected_rows
                else None
            ),
            "exact_rate": (
                sum(bool(row["plan_exact"]) for row in task_rows)
                / expected_rows
                if expected_rows
                else None
            ),
            "discrete_exact_rate": (
                sum(bool(row["discrete_exact"]) for row in task_rows)
                / expected_rows
                if expected_rows
                else None
            ),
            "mean_task_score": (
                sum(float(row["task_score"]) for row in task_rows) / len(task_rows)
                if task_rows
                else None
            ),
            "patch_exact_rate": (
                sum(bool(row["patch_exact"]) for row in task_rows)
                / expected_rows
                if task == "editing" and expected_rows
                else None
            ),
        }

    hard_checks = {
        "all_selected_rows_valid": len(valid_rows) == len(selected),
        "all_selected_rows_finite": all(bool(row.get("finite")) for row in valid_rows),
        "all_selected_rows_roundtrip": all(
            bool(row.get("roundtrip_exact")) for row in valid_rows
        ),
    }
    if not args.skip_interventions and not failures:
        u_checks = interventions["understanding"]
        e_checks = interventions["editing"]
        hard_checks.update(
            {
                "u_flow_noise_changes_thought": (
                    float(u_checks["noise_seed_43"]["thought_rmse"] or 0.0) > 0.0
                ),
                "u_flow_noise_preserves_semantics": not bool(
                    u_checks["noise_seed_43"]["semantic_changed"]
                ),
                "u_zero_thought_preserves_semantics": not bool(
                    u_checks["zero_thought"]["semantic_changed"]
                ),
                "e_prior_cannot_change_observation": all(
                    not bool(e_checks[name]["observed_plan_changed"])
                    for name in (
                        "prior_none",
                        "prior_correct",
                        "prior_corrupt_cross_scene",
                    )
                ),
                "e_instruction_cannot_change_observation": not bool(
                    e_checks["instruction_swap"]["observed_plan_changed"]
                ),
                "e_observation_thought_preserves_semantics": not bool(
                    e_checks["zero_observation_thought"]["observed_semantic_changed"]
                ),
                "e_delta_thought_cannot_change_observation": not bool(
                    e_checks["zero_delta_thought"]["observed_plan_changed"]
                ),
            }
        )
    diagnostics = {}
    if not args.skip_interventions and not failures:
        diagnostics = {
            "u_foa_evidence_sensitive": any(
                bool(interventions["understanding"][name]["plan_changed"])
                or float(interventions["understanding"][name]["thought_rmse"] or 0.0) > 0.0
                for name in ("zero_foa", "swap_foa")
            ),
            "u_clap_evidence_sensitive": any(
                bool(interventions["understanding"][name]["plan_changed"])
                or float(interventions["understanding"][name]["thought_rmse"] or 0.0) > 0.0
                for name in ("zero_semantic", "swap_semantic")
            ),
            "u_zero_thought_changes_numeric_execution": bool(
                interventions["understanding"]["zero_thought"]["numeric_changed"]
            ),
            "u_shuffle_changes_numeric_execution": bool(
                interventions["understanding"]["shuffle_source_slots"]["numeric_changed"]
            ),
            "e_instruction_changes_patch": bool(
                interventions["editing"]["instruction_swap"]["patch_changed"]
            ),
            "e_foa_evidence_sensitive": bool(
                interventions["editing"]["zero_foa"]["observed_plan_changed"]
            )
            or float(
                interventions["editing"]["zero_foa"]["observed_thought_rmse"] or 0.0
            )
            > 0.0,
            "e_clap_evidence_sensitive": bool(
                interventions["editing"]["zero_semantic"]["observed_plan_changed"]
            )
            or float(
                interventions["editing"]["zero_semantic"]["observed_thought_rmse"] or 0.0
            )
            > 0.0,
            "e_zero_observation_thought_changes_observed_numeric": bool(
                interventions["editing"]["zero_observation_thought"]["observed_numeric_changed"]
            ),
            "e_zero_delta_thought_changes_revised_numeric": bool(
                interventions["editing"]["zero_delta_thought"]["numeric_changed"]
            ),
            "e_flip_retime_changes_revised_numeric": bool(
                interventions["editing"]["flip_retime_delta"]["numeric_changed"]
            ),
        }

    report = {
        "schema": "stable_audio_tools.p11_audio_aware_learned_pilot_gate",
        "version": 3,
        "metric_contract": "observation_conditional_edit_end_to_end_v2",
        "status": "PASS" if all(hard_checks.values()) and not failures else "FAIL",
        "scope": "P11_only_no_P10_render",
        "seed": 42,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": (
            args.checkpoint_sha256
            if args.checkpoint_sha256 is not None
            else _file_hash(checkpoint_path)
        ),
        "checkpoint_step": checkpoint_step,
        "weights": weight_kind,
        "decode_mode": args.decode_mode,
        "canonical_decode_mode": CANONICAL_DECODE_MODE,
        "canonical_decode": args.decode_mode == CANONICAL_DECODE_MODE,
        "ema_step": ema_step,
        "model_config": str(model_config_path),
        "dataset_config": str(dataset_config_path),
        "selected_ordinals": selected,
        "rows_per_task": None if args.ordinals else args.rows_per_task,
        "explicit_ordinal_worker": bool(args.ordinals),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30
            if device.type == "cuda"
            else 0.0
        ),
        "hard_checks": hard_checks,
        "diagnostics_not_hard_gates": diagnostics,
        "summary_by_task": by_task,
        "failures": failures,
        "rows": rows,
        "interventions": interventions,
        "semantic_encoder_decision": (
            "Run a held-out current-CLAP vs M2D-CLAP A/B only if semantic errors "
            "or weak semantic intervention sensitivity persist after this pilot; "
            "do not attribute FOA timing/spatial errors to CLAP."
        ),
    }
    if not all(math.isfinite(float(row["task_score"])) for row in valid_rows):
        report["status"] = "FAIL"
        report["failures"].append({"error": "non-finite task score"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "summary_by_task": by_task,
        "hard_checks": hard_checks,
        "diagnostics": diagnostics,
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
