#!/usr/bin/env python3
"""Unified D0/Direct/Flow evaluator for the P11-v4 held-out challenge."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.p11_challenge_metrics import _mean, _aggregate
from functools import partial
from stable_audio_tools.data.artifact_io import digest, sha as _sha256_file

# Existing report hashes serialize nonfinite diagnostics using JSON's legacy policy.
_json_sha256 = partial(digest, allow_nan=True)


DEFAULT_CHALLENGE = REPO_ROOT / (
    "artifacts/sceneplan_p11/challenges/"
    "p11_v4_heldout_challenge_v1_20260901.sqlite"
)
DEFAULT_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_heldout900_transfusion_cot_v4_reliable_asr_v1.json"
)
MODEL_CONFIGS = {
    "d0": REPO_ROOT
    / (
        "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
        "qwen35_0p8b_sceneplan_p11_baseline_discrete_d0.json"
    ),
    "direct": REPO_ROOT
    / (
        "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
        "qwen35_0p8b_sceneplan_p11_baseline_direct_mse.json"
    ),
    "flow": REPO_ROOT
    / (
        "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
        "qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
    ),
}
LEXICAL_AUTHORITY_INTERVENTIONS = (
    "normal",
    "drop_reliable_asr",
)
EVALUATOR_SCHEMA_VERSION = 10
CANONICAL_QWEN_KERNEL_MODE = "torch_reference"
_GENERATION_DURATION = re.compile(
    r"^Create a ([0-9]+(?:\.[0-9]+)?)-second FOA(?: spatial-audio)? scene"
)






def _integer_tensor_sha256(value: Any) -> str:
    payload = (
        torch.as_tensor(value, dtype=torch.int64)
        .detach()
        .cpu()
        .contiguous()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _resolved_config_provenance(path: Path) -> dict[str, Any]:
    """Fingerprint both the leaf JSON and its fully inherited contract."""

    from stable_audio_tools.configuration import load_config

    resolved = load_config(path)
    return {
        "path": str(path),
        "source_sha256": _sha256_file(path),
        "resolved_sha256": _json_sha256(resolved),
    }


def _lexical_evidence_provenance(dataset_config_path: Path) -> dict[str, Any]:
    """Fingerprint the optional frozen-ASR artifact, not only its JSON pointer."""

    from stable_audio_tools.configuration import load_config

    config = load_config(dataset_config_path)
    mode = str(config.get("lexical_evidence_mode", "none"))
    cache_value = config.get("lexical_cache_path")
    provenance: dict[str, Any] = {
        "mode": mode,
        "enabled": mode != "none",
        "cache_path": None,
        "cache_sha256": None,
        "cache_bytes": None,
        "encoder_revision": config.get("lexical_encoder_revision"),
        "confidence_threshold": config.get("lexical_confidence_threshold"),
        "max_tokens": int(config.get("lexical_max_tokens", 128)),
        "cache_metadata_sha256": None,
    }
    if mode == "none":
        if cache_value is not None:
            raise ValueError("lexical mode 'none' must not retain a cache path")
        return provenance
    if cache_value is None:
        raise ValueError("enabled lexical evidence requires a frozen cache path")

    cache_path = Path(str(cache_value)).expanduser().resolve(strict=True)
    connection = sqlite3.connect(
        f"file:{cache_path}?mode=ro&immutable=1",
        uri=True,
        check_same_thread=False,
    )
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
    finally:
        connection.close()
    provenance.update(
        {
            "cache_path": str(cache_path),
            "cache_sha256": _sha256_file(cache_path),
            "cache_bytes": cache_path.stat().st_size,
            "cache_metadata_sha256": _json_sha256(metadata),
            "cache_schema": metadata.get("schema"),
            "cache_schema_version": metadata.get("schema_version"),
            "cache_contract": metadata.get("contract"),
            "cache_confidence_contract": metadata.get("confidence_contract"),
            "cache_source": metadata.get("source"),
            "target_transcript_access": metadata.get("target_transcript_access"),
        }
    )
    return provenance


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_source_provenance(device: torch.device) -> dict[str, Any]:
    """Freeze evaluator/model/kernel identity alongside every scientific report.

    A checkpoint and config hash are insufficient for Qwen3.5: its hybrid
    GatedDeltaNet path is supplied partly by Transformers and partly by optional
    FLA/causal-conv kernels.  This record makes reports from different source or
    kernel states explicitly incomparable instead of silently mixing them.
    """

    from stable_audio_tools.data import (
        scene_sketch_v1,
        sceneplan_p11_lexical_cache,
        sceneplan_p11_metrics,
        sceneplan_p11_v4_challenge,
        sceneplan_p11_v4_dataset,
    )
    from stable_audio_tools.models import (
        scene_thought_p11_v4,
        sceneplan_p11,
        sceneplan_p11_v4,
    )
    from transformers.models.qwen3_5 import modeling_qwen3_5

    source_modules = {
        "evaluator": sys.modules[__name__],
        "sceneplan_p11": sceneplan_p11,
        "sceneplan_p11_v4": sceneplan_p11_v4,
        "scene_thought_p11_v4": scene_thought_p11_v4,
        "scene_sketch_v1": scene_sketch_v1,
        "sceneplan_p11_lexical_cache": sceneplan_p11_lexical_cache,
        "sceneplan_p11_metrics": sceneplan_p11_metrics,
        "sceneplan_p11_v4_challenge": sceneplan_p11_v4_challenge,
        "sceneplan_p11_v4_dataset": sceneplan_p11_v4_dataset,
        "transformers_qwen3_5": modeling_qwen3_5,
    }
    try:
        from fla.ops.gated_delta_rule import chunk as fla_gated_delta_chunk
        from fla.ops.common import chunk_delta_h as fla_chunk_delta_h

        source_modules["fla_gated_delta_chunk"] = fla_gated_delta_chunk
        source_modules["fla_chunk_delta_h"] = fla_chunk_delta_h
    except Exception:  # noqa: BLE001 - absence is part of frozen provenance.
        pass
    source_files: dict[str, dict[str, Any]] = {}
    for label, module in source_modules.items():
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            source_files[label] = {"path": None, "sha256": None}
            continue
        path = Path(raw_path).resolve(strict=True)
        source_files[label] = {"path": str(path), "sha256": _sha256_file(path)}

    def callable_id(value: Any) -> str | None:
        if value is None:
            return None
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", getattr(value, "__name__", None))
        return f"{module}.{name}" if name is not None else str(type(value))

    cuda: dict[str, Any] | None = None
    if device.type == "cuda":
        logical_index = torch.cuda.current_device() if device.index is None else device.index
        properties = torch.cuda.get_device_properties(logical_index)
        cuda = {
            "logical_device_index": int(logical_index),
            "name": properties.name,
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
            "torch_cuda_version": torch.version.cuda,
        }
    return {
        "contract": "p11_v4_evaluator_runtime_source_provenance_v2",
        "python": platform.python_version(),
        "packages": {
            name: _distribution_version(name)
            for name in (
                "torch",
                "transformers",
                "flash-linear-attention",
                "causal-conv1d",
                "triton",
            )
        },
        "determinism": {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        },
        "qwen35_gated_delta_runtime": {
            "fast_path_available": bool(
                modeling_qwen3_5.is_fast_path_available
            ),
            "causal_conv1d": callable_id(modeling_qwen3_5.causal_conv1d_fn),
            "chunk_gated_delta_rule": callable_id(
                modeling_qwen3_5.chunk_gated_delta_rule
            ),
            "recurrent_gated_delta_rule": callable_id(
                modeling_qwen3_5.fused_recurrent_gated_delta_rule
            ),
        },
        "cuda": cuda,
        "source_files": source_files,
    }


def _ids(value: Any) -> torch.Tensor:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    return torch.as_tensor(value, dtype=torch.long).flatten().cpu()




def _duration(metadata: Mapping[str, Any]) -> tuple[float, str]:
    task = str(metadata["p11_task"])
    if task == "generation":
        prompt = " ".join(str(metadata["p11_prompt_text"]).split())
        match = _GENERATION_DURATION.match(prompt)
        if match is None:
            raise ValueError("challenge G prompt lacks an executable duration")
        return float(match.group(1)), "prompt"
    if task == "understanding":
        valid = int(torch.as_tensor(metadata["p11_input_valid_mask"]).sum())
        return valid * 1024.0 / 44_100.0, "input_foa_valid_frames"
    return float(metadata["p11_input_sceneplan"]["duration_sec"]), "current_sceneplan"


def _seed(root: int, row_key: str, draw: int, stream: str) -> int:
    payload = (
        f"p11-v4-challenge-eval-v1\0{root}\0{row_key}\0{draw}\0{stream}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def _row_seed_key(metadata: Mapping[str, Any]) -> str:
    pair = metadata.get("p11_challenge_pair_id")
    return str(pair or metadata["p11_challenge_id"])


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
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
        "room": str(plan["room"]["type"]),
        "sources": [
            {key: source[key] for key in fields if key in source}
            for source in plan["sources"]
        ],
    }


def _active_mask(metadata: Mapping[str, Any]) -> torch.Tensor:
    target = torch.as_tensor(
        metadata["p11_v4_target_source_mask"], dtype=torch.bool
    ).flatten()
    if metadata["p11_task"] == "editing":
        target |= torch.as_tensor(
            metadata["p11_v4_input_source_mask"], dtype=torch.bool
        ).flatten()
    slots = torch.cat([torch.ones(1, dtype=torch.bool), target])
    return slots[:, None].expand(5, 15)


def _pairwise_rmse(values: torch.Tensor, mask: torch.Tensor) -> float:
    if values.shape[0] <= 1:
        return 0.0
    distances = [
        float((values[left][mask] - values[right][mask]).square().mean().sqrt())
        for left, right in itertools.combinations(range(values.shape[0]), 2)
    ]
    return float(sum(distances) / len(distances))


def _calibration(
    values: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float | None]:
    values = values.detach().float().cpu()[:, mask]
    target = target.detach().float().cpu()[mask]
    mean = values.mean(dim=0)
    error = mean - target
    rmse = float(error.square().mean().sqrt())
    if values.shape[0] < 2:
        return {
            "available": 0.0,
            "ensemble_mean_rmse": rmse,
            "spread_mean": 0.0,
            "coverage_68": None,
            "coverage_95": None,
            "absolute_calibration_error_68": None,
            "absolute_calibration_error_95": None,
        }
    spread = values.std(dim=0, unbiased=False)
    absolute = error.abs()
    coverage68 = float((absolute <= spread).float().mean())
    coverage95 = float((absolute <= 1.96 * spread).float().mean())
    return {
        "available": 1.0,
        "ensemble_mean_rmse": rmse,
        "spread_mean": float(spread.mean()),
        "coverage_68": coverage68,
        "coverage_95": coverage95,
        "absolute_calibration_error_68": abs(coverage68 - 0.6827),
        "absolute_calibration_error_95": abs(coverage95 - 0.95),
    }


def _rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu = torch.random.get_rng_state().clone()
    cuda = (
        torch.cuda.get_rng_state(device).clone()
        if device.type == "cuda"
        else None
    )
    return cpu, cuda


def _rng_equal(
    before: tuple[torch.Tensor, torch.Tensor | None],
    after: tuple[torch.Tensor, torch.Tensor | None],
) -> bool:
    return bool(
        torch.equal(before[0], after[0])
        and (
            before[1] is None
            or (after[1] is not None and torch.equal(before[1], after[1]))
        )
    )


def _select_rows(
    challenge_path: Path,
    *,
    rows_per_view: int,
    families: set[str] | None,
) -> list[int]:
    connection = sqlite3.connect(
        f"file:{challenge_path}?mode=ro&immutable=1", uri=True
    )
    where = ""
    parameters: list[Any] = []
    if families:
        where = " WHERE family IN (" + ",".join("?" for _ in families) + ")"
        parameters.extend(sorted(families))
    rows = connection.execute(
        "SELECT ordinal,family,view_id,pair_id FROM rows"
        + where
        + " ORDER BY ordinal",
        parameters,
    ).fetchall()
    connection.close()
    if rows_per_view == 0:
        return [int(row[0]) for row in rows]
    by_view: dict[tuple[str, str], list[int]] = defaultdict(list)
    counterfactual_pairs: dict[str, list[int]] = defaultdict(list)
    for ordinal, family, view, pair_id in rows:
        if str(family) == "editing_counterfactual_causality":
            counterfactual_pairs[str(pair_id)].append(int(ordinal))
        else:
            by_view[(str(family), str(view))].append(int(ordinal))
    selected = []
    for values in by_view.values():
        selected.extend(values[:rows_per_view])
    pair_ids = sorted(counterfactual_pairs)[:rows_per_view]
    for pair_id in pair_ids:
        values = counterfactual_pairs[pair_id]
        if len(values) != 2:
            raise RuntimeError(f"selected E pair {pair_id} is incomplete")
        selected.extend(values)
    if not selected:
        raise RuntimeError("challenge selector produced no rows")
    return sorted(set(selected))


def _load_model(
    *,
    model_config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    weights: str,
) -> tuple[Any, Any, dict[str, Any]]:
    from stable_audio_tools.configuration import load_config
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.training.factory import create_training_wrapper_from_config

    model_config = load_config(model_config_path)
    runtime_config_sha256 = _json_sha256(model_config)
    model = create_model_from_config(model_config)
    wrapper = create_training_wrapper_from_config(model_config, model)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    wrapper.load_state_dict(checkpoint["state_dict"], strict=True)
    global_step = int(checkpoint.get("global_step", -1))
    checkpoint_model_config = checkpoint.get("model_config")
    if not isinstance(checkpoint_model_config, Mapping):
        raise RuntimeError(
            "P11-v4 checkpoint lacks its resolved model_config provenance"
        )
    checkpoint_config_sha256 = _json_sha256(checkpoint_model_config)
    if checkpoint_config_sha256 != runtime_config_sha256:
        raise RuntimeError(
            "runtime model config does not match the resolved config embedded "
            "in the checkpoint: "
            f"{runtime_config_sha256} != {checkpoint_config_sha256}"
        )
    del checkpoint
    gc.collect()
    wrapper.eval().requires_grad_(False).to(device)
    ema_step = None
    if weights == "ema":
        if wrapper.p11_ema is None:
            raise RuntimeError("requested EMA but checkpoint has no P11 EMA")
        ema_step = int(wrapper.p11_ema.step.detach().cpu())
        wrapper.p11_ema.copy_to(wrapper.p11)
    provenance = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "global_step": global_step,
        "weights": weights,
        "ema_step": ema_step,
        "model_config": str(model_config_path),
        "model_config_sha256": _sha256_file(model_config_path),
        "model_config_source_sha256": _sha256_file(model_config_path),
        "model_config_resolved_sha256": runtime_config_sha256,
        "checkpoint_model_config_resolved_sha256": checkpoint_config_sha256,
        "checkpoint_model_config_matches_runtime": True,
    }
    return wrapper, wrapper.p11, provenance


def _dataset(
    *,
    planner: Any,
    dataset_config_path: Path,
    challenge_path: Path,
) -> Any:
    from stable_audio_tools.configuration import load_config
    from stable_audio_tools.data.sceneplan_p11_v4_challenge import (
        ScenePlanP11V4ChallengeDataset,
    )
    from stable_audio_tools.data.sceneplan_p11_v4_dataset import ScenePlanP11V4Dataset

    config = load_config(dataset_config_path)
    sources = config.get("datasets") or []
    if len(sources) != 1:
        raise ValueError("challenge evaluator requires one frozen source index")
    base = ScenePlanP11V4Dataset(
        config["manifest_path"],
        index_path=sources[0]["path"],
        codec_path=config["codec_path"],
        tokenizer_spec=(planner.tokenizer, 512, None),
        expected_num_samples=int(config["expected_num_samples"]),
        index_num_samples=int(config["index_num_samples"]),
        require_frozen=bool(config.get("require_complete", True)),
        semantic_cache_path=config["semantic_cache_path"],
        semantic_dim=int(config.get("semantic_dim", 512)),
        semantic_encoder_revision=config["semantic_encoder_revision"],
        lexical_evidence_mode=str(config.get("lexical_evidence_mode", "none")),
        lexical_max_tokens=int(config.get("lexical_max_tokens", 128)),
        lexical_cache_path=config.get("lexical_cache_path"),
        lexical_encoder_revision=config.get("lexical_encoder_revision"),
        lexical_confidence_threshold=config.get(
            "lexical_confidence_threshold"
        ),
    )
    return ScenePlanP11V4ChallengeDataset(base, challenge_path)


def _prepare_inputs(metadata: Mapping[str, Any]) -> dict[str, Any]:
    input_foa = metadata.get("p11_input_foa")
    if input_foa is not None:
        input_foa = torch.as_tensor(input_foa)
    duration, duration_source = _duration(metadata)
    return {
        "prompt": metadata["p11_prompt"],
        "task": str(metadata["p11_task"]),
        "input_foa": input_foa,
        "input_valid_mask": metadata.get("p11_input_valid_mask"),
        "input_semantic": metadata.get("p11_input_semantic"),
        "input_lexical": metadata.get("p11_input_lexical"),
        "input_sceneplan": metadata.get("p11_input_sceneplan"),
        "duration_sec": duration,
        "duration_source": duration_source,
        "sample_id": str(metadata["p11_target_sample_id"]),
    }


def _apply_lexical_authority_intervention(
    inputs: Mapping[str, Any], intervention: str
) -> dict[str, Any]:
    """Apply an evaluator-only lexical counterfactual before model access.

    ``drop_reliable_asr`` removes the complete frozen-ASR object.  It does not
    blank only the transcript or substitute target text, so the intervention
    cannot leak hidden ScenePlan supervision into the model.
    """

    if intervention not in LEXICAL_AUTHORITY_INTERVENTIONS:
        raise ValueError(f"unsupported lexical authority intervention: {intervention}")
    output = dict(inputs)
    if intervention == "drop_reliable_asr":
        output["input_lexical"] = None
    return output


def _normalize_d0_output(
    planner: Any,
    token_ids: torch.Tensor,
    diagnostics: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
    input_lexical: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from stable_audio_tools.data.scene_sketch_v1 import (
        apply_reliable_lexical_authority_to_sketch,
        compile_execution_state,
        compile_p10_from_contract,
        compile_scene_sketch,
        execution_state_core,
        select_reliable_lexical_source_owner,
    )
    from stable_audio_tools.data.sceneplan_p11_lexical_cache import (
        P11_LEXICAL_ASSEMBLER_POLICY,
        parse_reliable_lexical_authority,
    )

    task = str(metadata["p11_task"])
    sample_id = str(metadata["p11_target_sample_id"])
    if task == "editing":
        patch = planner.patch_codec.decode(token_ids)
        sceneplan = planner.patch_codec.apply(
            metadata["p11_input_sceneplan"], patch
        )
        patch_tokens = token_ids
    else:
        patch = None
        sceneplan = planner.plan_codec.decode(token_ids, sample_id=sample_id)
        patch_tokens = None
    raw_sceneplan = sceneplan
    sketch = compile_scene_sketch(raw_sceneplan, planner.plan_codec)
    execution = compile_execution_state(sceneplan, planner.plan_codec)
    lexical_diagnostics = {
        "lexical_injection_policy": P11_LEXICAL_ASSEMBLER_POLICY,
        "lexical_authority_stage": "d0_post_autoregressive_pre_p10_assembly",
        "lexical_authority_applied": False,
        "lexical_authority_action": None,
        "lexical_authority_source_id": None,
        "lexical_authority_confidence": None,
        "lexical_authority_transcript_sha256": None,
        "lexical_authority_numeric_state_immutable": True,
    }
    if input_lexical is not None:
        if task != "understanding":
            raise ValueError("reliable ASR authority is legal only for Understanding")
        authority = parse_reliable_lexical_authority(input_lexical)
        source_id, action = select_reliable_lexical_source_owner(
            sketch,
            source_kind_scores=diagnostics.get("source_kind_scores", []),
            codec=planner.plan_codec,
        )
        sketch = apply_reliable_lexical_authority_to_sketch(
            sketch,
            transcript=authority["transcript"],
            source_id=source_id,
            codec=planner.plan_codec,
        )
        assembled = compile_p10_from_contract(
            sketch, execution, planner.plan_codec
        )
        sceneplan = assembled["sceneplan"]
        reconstructed_execution = compile_execution_state(
            sceneplan, planner.plan_codec
        )
        if not np.array_equal(
            execution_state_core(execution),
            execution_state_core(reconstructed_execution),
        ):
            raise RuntimeError("D0 lexical assembly changed numeric ExecutionState")
        lexical_diagnostics.update(
            {
                "lexical_authority_applied": True,
                "lexical_authority_action": action,
                "lexical_authority_source_id": source_id,
                "lexical_authority_confidence": float(authority["confidence"]),
                "lexical_authority_transcript_sha256": hashlib.sha256(
                    authority["transcript"].encode("utf-8")
                ).hexdigest(),
            }
        )
    else:
        assembled = compile_p10_from_contract(
            sketch, execution, planner.plan_codec
        )
        sceneplan = assembled["sceneplan"]
    final_plan_tokens = planner.plan_codec.encode(
        sceneplan, max_tokens=planner.plan_max_tokens
    )["input_ids"]
    return {
        "sceneplan": sceneplan,
        "scene_sketch": sketch,
        "execution_state": execution,
        "plan_tokens": final_plan_tokens,
        "patch": patch,
        "patch_tokens": patch_tokens,
        "output_tokens": token_ids if task == "editing" else final_plan_tokens,
        "discrete_tokens": token_ids if task == "editing" else final_plan_tokens,
        "d0_autoregressive_tokens": token_ids,
        "thought_core": None,
        "p10_conditions": assembled,
        "diagnostics": {**dict(diagnostics), **lexical_diagnostics},
    }


def _decode_d0(
    planner: Any,
    metadata: Mapping[str, Any],
    *,
    draws: int,
    root_seed: int,
    temperature: float,
    discrete_decode_mode: str,
    lexical_authority_intervention: str = "normal",
) -> tuple[list[dict[str, Any]], list[bool]]:
    inputs = _apply_lexical_authority_intervention(
        _prepare_inputs(metadata), lexical_authority_intervention
    )
    row_key = _row_seed_key(metadata)
    outputs = []
    rng_checks = []
    device = planner.plan_embedding.weight.device
    for draw in range(draws):
        sampling_seed = _seed(root_seed, row_key, draw, "discrete")
        before = _rng_state(device)
        tokens, diagnostics = planner.decode_output_tokens(
            inputs["prompt"],
            task=inputs["task"],
            input_foa=inputs["input_foa"],
            input_valid_mask=inputs["input_valid_mask"],
            input_semantic=inputs["input_semantic"],
            input_sceneplan=inputs["input_sceneplan"],
            duration_sec=inputs["duration_sec"],
            temperature=temperature,
            constrained=True,
            sampling_seed=sampling_seed,
            discrete_decode_mode=discrete_decode_mode,
        )
        after = _rng_state(device)
        rng_checks.append(_rng_equal(before, after))
        output = _normalize_d0_output(
            planner,
            tokens,
            diagnostics,
            metadata=metadata,
            input_lexical=inputs["input_lexical"],
        )
        output["diagnostics"]["posterior_draw_index"] = draw
        outputs.append(output)
    return outputs, rng_checks


def _decode_flow(
    planner: Any,
    metadata: Mapping[str, Any],
    *,
    draws: int,
    root_seed: int,
    discrete_decode_mode: str,
    lexical_authority_intervention: str = "normal",
) -> tuple[list[dict[str, Any]], list[bool]]:
    inputs = _apply_lexical_authority_intervention(
        _prepare_inputs(metadata), lexical_authority_intervention
    )
    row_key = _row_seed_key(metadata)
    discrete_seed = _seed(root_seed, row_key, 0, "discrete")
    device = planner.plan_embedding.weight.device
    before = _rng_state(device)
    direct_delta_editing = bool(
        inputs["task"] == "editing"
        and planner.execution_reasoner.editing_uses_direct
    )
    if direct_delta_editing:
        # Canonical P11-v4 is deliberately hybrid: G/U use Flow-R1, while E
        # uses the dedicated deterministic Direct-MSE DeltaThought head.  A
        # Flow-arm evaluator must therefore not inject posterior noise into E.
        # Decode the shared DeltaSketch/DeltaThought once and expose repeated
        # K-prefixes only so the fixed challenge has the same draw budget.
        deterministic = planner.decode_transfusion_cot(
            inputs["prompt"],
            task=inputs["task"],
            input_foa=inputs["input_foa"],
            input_valid_mask=inputs["input_valid_mask"],
            input_semantic=inputs["input_semantic"],
            input_lexical=inputs["input_lexical"],
            input_sceneplan=inputs["input_sceneplan"],
            duration_sec=inputs["duration_sec"],
            sample_id=inputs["sample_id"],
            temperature=0.0,
            discrete_seed=discrete_seed,
            discrete_decode_mode=discrete_decode_mode,
        )
        outputs = [
            {
                **deterministic,
                "diagnostics": {
                    **dict(deterministic.get("diagnostics") or {}),
                    "posterior_draw_index": draw,
                    "posterior_draw_count": draws,
                    "posterior_draw_is_stochastic": False,
                    "deterministic_delta_repeated_across_draws": True,
                },
            }
            for draw in range(draws)
        ]
    else:
        noise_seeds = [
            _seed(root_seed, row_key, draw, "continuous")
            for draw in range(draws)
        ]
        outputs = planner.decode_transfusion_cot_samples(
            inputs["prompt"],
            task=inputs["task"],
            noise_seeds=noise_seeds,
            input_foa=inputs["input_foa"],
            input_valid_mask=inputs["input_valid_mask"],
            input_semantic=inputs["input_semantic"],
            input_lexical=inputs["input_lexical"],
            input_sceneplan=inputs["input_sceneplan"],
            duration_sec=inputs["duration_sec"],
            sample_id=inputs["sample_id"],
            temperature=0.0,
            discrete_seed=discrete_seed,
            discrete_decode_mode=discrete_decode_mode,
        )
        outputs = [
            {
                **output,
                "diagnostics": {
                    **dict(output.get("diagnostics") or {}),
                    "posterior_draw_is_stochastic": True,
                },
            }
            for output in outputs
        ]
    after = _rng_state(device)
    return outputs, [_rng_equal(before, after)] * len(outputs)


def _decode_direct(
    planner: Any,
    metadata: Mapping[str, Any],
    *,
    root_seed: int,
    discrete_decode_mode: str,
    lexical_authority_intervention: str = "normal",
) -> tuple[dict[str, Any], bool]:
    inputs = _apply_lexical_authority_intervention(
        _prepare_inputs(metadata), lexical_authority_intervention
    )
    row_key = _row_seed_key(metadata)
    discrete_seed = _seed(root_seed, row_key, 0, "discrete")
    device = planner.plan_embedding.weight.device
    before = _rng_state(device)
    output = planner.decode_transfusion_cot(
        inputs["prompt"],
        task=inputs["task"],
        input_foa=inputs["input_foa"],
        input_valid_mask=inputs["input_valid_mask"],
        input_semantic=inputs["input_semantic"],
        input_lexical=inputs["input_lexical"],
        input_sceneplan=inputs["input_sceneplan"],
        duration_sec=inputs["duration_sec"],
        sample_id=inputs["sample_id"],
        temperature=0.0,
        discrete_seed=discrete_seed,
        discrete_decode_mode=discrete_decode_mode,
    )
    after = _rng_state(device)
    return output, _rng_equal(before, after)


def _score_draw(
    planner: Any,
    metadata: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    from stable_audio_tools.data.scene_sketch_v1 import (
        compile_execution_state,
        execution_state_core,
    )
    from stable_audio_tools.data.sceneplan_p11_metrics import score_p11_prediction
    from stable_audio_tools.data.sceneplan_p11_single_turn import (
        validate_p11_executor_profile,
    )

    task = str(metadata["p11_task"])
    plan = validate_p11_executor_profile(output["sceneplan"])
    plan_ids = _ids(output["plan_tokens"])
    decoded = planner.plan_codec.decode(
        plan_ids, sample_id=str(metadata["p11_target_sample_id"])
    )
    recoded = _ids(planner.plan_codec.encode(decoded))
    metrics = score_p11_prediction(
        task=task,
        target_plan=metadata["p11_target_sceneplan"],
        prediction=plan,
        input_plan=metadata.get("p11_input_sceneplan"),
        source_matching=metadata["p11_source_matching"],
        editing_score_version=metadata["p11_editing_score_version"],
        known_field_groups=(
            metadata.get("p11_prompt_known_field_groups")
            if task == "generation"
            else None
        ),
    )
    core = torch.from_numpy(
        execution_state_core(compile_execution_state(plan, planner.plan_codec))
    ).float()
    reference_cores = [
        torch.from_numpy(
            execution_state_core(
                compile_execution_state(reference, planner.plan_codec)
            )
        ).float()
        for reference in metadata["p11_challenge_reference_sceneplans"]
    ]
    mask = _active_mask(metadata)
    nearest = min(
        float((core[mask] - reference[mask]).square().mean().sqrt())
        for reference in reference_cores
    )
    patch_exact = None
    if task == "editing":
        predicted_patch = output.get("patch")
        expected_patch = metadata.get("p11_edit_spec")
        patch_exact = bool(
            predicted_patch is not None
            and expected_patch is not None
            and planner.patch_codec.decode(
                _ids(planner.patch_codec.encode(predicted_patch))
            )
            == planner.patch_codec.decode(
                _ids(planner.patch_codec.encode(expected_patch))
            )
        )
    return {
        "valid": True,
        "parse": True,
        "roundtrip": bool(torch.equal(plan_ids, recoded)),
        "finite": bool(torch.isfinite(core).all()),
        "plan_sha256": _json_sha256(plan),
        "sceneplan": plan,
        "semantic_sha256": _json_sha256(_semantic_signature(plan)),
        "numeric_sha256": _tensor_sha256(core),
        "discrete_tokens_sha256": _integer_tensor_sha256(
            output["discrete_tokens"]
        ),
        "thought_core_sha256": (
            None
            if output.get("thought_core") is None
            else _tensor_sha256(torch.as_tensor(output["thought_core"]))
        ),
        "p10_semantic_sha256": _json_sha256(
            output["p10_conditions"]["semantic_caption"]
        ),
        "p10_conditioning_valid": True,
        "task_metrics": metrics,
        "task_score": float(metrics["task_score"]),
        "constraint_pass": (
            bool(float(metrics["task_score"]) >= 1.0 - 1e-9)
            if task == "generation"
            else None
        ),
        "patch_exact": patch_exact,
        "target_plan_exact": bool(
            torch.equal(
                plan_ids, _ids(metadata["p11_v4_final_sceneplan_tokens"])
            )
        ),
        "reference_anchor_nearest_core_rmse": nearest,
        "core": core,
        "patch": output.get("patch"),
        "diagnostics": dict(output.get("diagnostics") or {}),
    }


def _summarize_prefix(
    scored: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    k: int,
) -> dict[str, Any]:
    one = list(scored[:k])
    valid = [value for value in one if value.get("valid")]
    if not valid:
        generation = str(metadata["p11_task"]) == "generation"
        return {
            "k": k,
            "valid_rate": 0.0,
            "roundtrip_rate": 0.0,
            "finite_rate": 0.0,
            "p10_conditioning_valid_rate": 0.0,
            "task_score_mean": 0.0,
            "oracle_best_of_k_task_score": 0.0,
            "constraint_pass_rate": 0.0 if generation else None,
            "constraint_pass_any": False if generation else None,
            "semantic_immutability": False,
            "unique_semantic_count": 0,
            "unique_numeric_count": 0,
            "unique_sceneplan_count": 0,
            "numeric_pairwise_rmse": 0.0,
            "reference_anchor_nearest_core_rmse_mean": None,
            "reference_anchor_nearest_core_rmse_best": None,
            "target_plan_exact_any": False,
            "patch_exact_any": (
                False if str(metadata["p11_task"]) == "editing" else None
            ),
            "calibration": None,
            "hidden_exact_target_used_for_model_selection": False,
            "invalid_draws": k,
        }
    cores = torch.stack([value["core"] for value in valid]).float()
    mask = _active_mask(metadata)
    semantic_hashes = {str(value["semantic_sha256"]) for value in valid}
    numeric_hashes = {str(value["numeric_sha256"]) for value in valid}
    plan_hashes = {str(value["plan_sha256"]) for value in valid}
    target = torch.as_tensor(
        metadata["p11_v4_target_execution_core"], dtype=torch.float32
    )
    hidden_exact_diagnostic_only = bool(
        metadata.get("p11_challenge_hidden_exact_target_is_diagnostic_only")
    )
    calibration = (
        None
        if hidden_exact_diagnostic_only
        else _calibration(cores, target, mask)
    )
    task_scores = [float(value["task_score"]) for value in valid]
    constraints = [
        bool(value["constraint_pass"])
        for value in valid
        if value["constraint_pass"] is not None
    ]
    return {
        "k": k,
        "valid_rate": len(valid) / k,
        "roundtrip_rate": sum(bool(value["roundtrip"]) for value in valid) / k,
        "finite_rate": sum(bool(value["finite"]) for value in valid) / k,
        "p10_conditioning_valid_rate": sum(
            bool(value["p10_conditioning_valid"]) for value in valid
        )
        / k,
        "task_score_mean": float(sum(task_scores) / len(task_scores)),
        "oracle_best_of_k_task_score": float(max(task_scores)),
        "constraint_pass_rate": (
            None if not constraints else sum(constraints) / len(constraints)
        ),
        "constraint_pass_any": None if not constraints else any(constraints),
        "semantic_immutability": len(semantic_hashes) == 1,
        "unique_semantic_count": len(semantic_hashes),
        "unique_numeric_count": len(numeric_hashes),
        "unique_sceneplan_count": len(plan_hashes),
        "numeric_pairwise_rmse": _pairwise_rmse(cores, mask),
        "reference_anchor_nearest_core_rmse_mean": _mean(
            [float(value["reference_anchor_nearest_core_rmse"]) for value in valid]
        ),
        "reference_anchor_nearest_core_rmse_best": min(
            float(value["reference_anchor_nearest_core_rmse"]) for value in valid
        ),
        "target_plan_exact_any": any(bool(value["target_plan_exact"]) for value in valid),
        "patch_exact_any": (
            any(value["patch_exact"] is True for value in valid)
            if metadata["p11_task"] == "editing"
            else None
        ),
        "calibration": calibration,
        "hidden_exact_target_used_for_model_selection": False,
    }




def _counterfactual_summary(
    rows: Sequence[Mapping[str, Any]], k_values: Sequence[int]
) -> dict[str, Any]:
    pairs: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("pair_id"):
            pairs[str(row["pair_id"])].append(row)
    output = {}
    for k in k_values:
        successes = []
        contrasts = []
        for values in pairs.values():
            if len(values) != 2 or any(len(value.get("scored", [])) < k for value in values):
                continue
            for draw in range(k):
                left = values[0]["scored"][draw]
                right = values[1]["scored"][draw]
                if not left.get("valid") or not right.get("valid"):
                    successes.append(False)
                    contrasts.append(False)
                    continue
                successes.append(
                    left.get("patch_exact") is True
                    and right.get("patch_exact") is True
                )
                contrasts.append(left.get("patch") != right.get("patch"))
        output[str(k)] = {
            "pairs": len(pairs),
            "matched_draws": len(successes),
            "both_counterfactual_targets_correct_rate": _mean(
                [float(value) for value in successes]
            ),
            "prompt_counterfactual_changes_patch_rate": _mean(
                [float(value) for value in contrasts]
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("d0", "direct", "flow"), required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Repeat only for a true Direct-MSE checkpoint ensemble.",
    )
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--challenge", type=Path, default=DEFAULT_CHALLENGE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--weights", choices=("online", "ema"), default="online")
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--k-values", default="1,4")
    parser.add_argument("--d0-temperature", type=float, default=0.8)
    parser.add_argument(
        "--discrete-decode-mode",
        choices=("cached", "prefix_recompute"),
        default="prefix_recompute",
    )
    parser.add_argument("--rows-per-view", type=int, default=1)
    parser.add_argument("--families", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lexical-authority-intervention",
        choices=LEXICAL_AUTHORITY_INTERVENTIONS,
        default="normal",
        help=(
            "Evaluator-only causal intervention. drop_reliable_asr removes "
            "the complete frozen-ASR input while retaining the same model, "
            "checkpoint, challenge rows, and RNG seeds."
        ),
    )
    parser.add_argument(
        "--qwen-kernel-mode",
        choices=(
            "fast",
            "fast_pinned_warps2",
            "fast_fixed_bv32_w2_s2",
            "torch_reference",
        ),
        default=CANONICAL_QWEN_KERNEL_MODE,
        help=(
            "Canonical scientific evaluation uses Transformers' deterministic "
            "torch_reference route. FLA modes are throughput diagnostics only: "
            "even a pinned forward config can drift across fresh CUDA processes."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.draws <= 32:
        raise ValueError("--draws must be within [1,32]")
    if args.rows_per_view < 0:
        raise ValueError("--rows-per-view must be non-negative")
    if args.arm != "direct" and len(args.checkpoint) != 1:
        raise ValueError("D0/Flow evaluator accepts exactly one checkpoint")
    if args.arm == "direct" and args.draws != len(args.checkpoint):
        raise ValueError(
            "Direct draws must equal independently trained checkpoint count; "
            "one checkpoint cannot be repeated as a fake ensemble"
        )
    if args.d0_temperature <= 0.0 and args.arm == "d0" and args.draws > 1:
        raise ValueError("multi-draw D0 requires positive constrained-sampling temperature")
    k_values = sorted({int(value) for value in args.k_values.split(",")})
    if not k_values or k_values[0] <= 0 or k_values[-1] > args.draws:
        raise ValueError("K values must lie within the available draw budget")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model_config_path = (
        MODEL_CONFIGS[args.arm]
        if args.model_config is None
        else args.model_config
    ).expanduser().resolve(strict=True)
    dataset_config_path = args.dataset_config.expanduser().resolve(strict=True)
    challenge_path = args.challenge.expanduser().resolve(strict=True)
    checkpoints = [value.expanduser().resolve(strict=True) for value in args.checkpoint]
    families = {
        value.strip() for value in args.families.split(",") if value.strip()
    } or None
    selected = _select_rows(
        challenge_path,
        rows_per_view=args.rows_per_view,
        families=families,
    )

    # The first model supplies the shared tokenizer used to materialize the
    # immutable challenge rows.  Direct ensembles reload subsequent members
    # sequentially so GPU memory does not scale with ensemble size.
    wrapper, planner, provenance = _load_model(
        model_config_path=model_config_path,
        checkpoint_path=checkpoints[0],
        device=device,
        weights=args.weights,
    )
    provenance["qwen_runtime_kernels"] = planner.configure_qwen_runtime_kernels(
        args.qwen_kernel_mode
    )
    challenge = _dataset(
        planner=planner,
        dataset_config_path=dataset_config_path,
        challenge_path=challenge_path,
    )
    materialized = []
    for ordinal in selected:
        _, metadata = challenge[ordinal]
        materialized.append((ordinal, metadata))
    del challenge

    row_outputs: dict[int, list[dict[str, Any]]] = {
        ordinal: [] for ordinal, _ in materialized
    }
    row_rng: dict[int, list[bool]] = {ordinal: [] for ordinal, _ in materialized}
    model_provenance = []
    performance = []

    def evaluate_loaded(one_planner: Any, member_index: int) -> None:
        if args.lexical_authority_intervention == "drop_reliable_asr":
            from stable_audio_tools.data.sceneplan_p11_lexical_cache import (
                P11_LEXICAL_ASSEMBLER_POLICY,
            )

            if args.arm != "d0":
                if (
                    getattr(one_planner, "lexical_injection_policy", None)
                    != P11_LEXICAL_ASSEMBLER_POLICY
                ):
                    raise RuntimeError(
                        "drop_reliable_asr requires the deterministic lexical assembler"
                    )
                one_planner.lexical_evidence_required = False
            # The dataset still materializes and fingerprints the immutable ASR
            # cache, but no lexical object crosses either the model (Flow/Direct)
            # or the evaluator-owned D0 post-assembly boundary.
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        decoded = 0
        for ordinal, metadata in materialized:
            autocast = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            try:
                with torch.inference_mode(), autocast:
                    if args.arm == "d0":
                        outputs, checks = _decode_d0(
                            one_planner,
                            metadata,
                            draws=args.draws,
                            root_seed=args.seed,
                            temperature=args.d0_temperature,
                            discrete_decode_mode=args.discrete_decode_mode,
                            lexical_authority_intervention=(
                                args.lexical_authority_intervention
                            ),
                        )
                    elif args.arm == "flow":
                        outputs, checks = _decode_flow(
                            one_planner,
                            metadata,
                            draws=args.draws,
                            root_seed=args.seed,
                            discrete_decode_mode=args.discrete_decode_mode,
                            lexical_authority_intervention=(
                                args.lexical_authority_intervention
                            ),
                        )
                    else:
                        output, check = _decode_direct(
                            one_planner,
                            metadata,
                            root_seed=args.seed + member_index,
                            discrete_decode_mode=args.discrete_decode_mode,
                            lexical_authority_intervention=(
                                args.lexical_authority_intervention
                            ),
                        )
                        outputs, checks = [output], [check]
                row_outputs[ordinal].extend(outputs)
                row_rng[ordinal].extend(checks)
                decoded += len(outputs)
            except Exception as error:  # noqa: BLE001 - retain fail-closed row.
                missing = 1 if args.arm == "direct" else args.draws
                for _ in range(missing):
                    row_outputs[ordinal].append(
                        {
                            "error": f"{type(error).__name__}: {error}",
                            "diagnostics": {
                                "posterior_draw_index": len(row_outputs[ordinal])
                            },
                        }
                    )
                    row_rng[ordinal].append(True)
                    decoded += 1
        elapsed = time.perf_counter() - started
        performance.append(
            {
                "member_index": member_index,
                "decoded_draws": decoded,
                "elapsed_seconds": elapsed,
                "draws_per_second": decoded / elapsed,
                "peak_memory_gib": (
                    torch.cuda.max_memory_allocated(device) / (2**30)
                    if device.type == "cuda"
                    else None
                ),
            }
        )

    model_provenance.append(provenance)
    evaluate_loaded(planner, 0)
    del planner, wrapper
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for member_index, checkpoint in enumerate(checkpoints[1:], start=1):
        wrapper, planner, provenance = _load_model(
            model_config_path=model_config_path,
            checkpoint_path=checkpoint,
            device=device,
            weights=args.weights,
        )
        provenance["qwen_runtime_kernels"] = planner.configure_qwen_runtime_kernels(
            args.qwen_kernel_mode
        )
        model_provenance.append(provenance)
        evaluate_loaded(planner, member_index)
        del planner, wrapper
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Reload only the codec/token embeddings needed for strict output scoring.
    wrapper, planner, scoring_provenance = _load_model(
        model_config_path=model_config_path,
        checkpoint_path=checkpoints[0],
        device=torch.device("cpu"),
        weights=args.weights,
    )
    rows = []
    for ordinal, metadata in materialized:
        scored = []
        for output in row_outputs[ordinal]:
            if "error" in output:
                scored.append(
                    {
                        "valid": False,
                        "parse": False,
                        "roundtrip": False,
                        "finite": False,
                        "task_score": 0.0,
                        "error": output["error"],
                        "diagnostics": output.get("diagnostics", {}),
                    }
                )
                continue
            try:
                scored.append(_score_draw(planner, metadata, output))
            except Exception as error:  # noqa: BLE001
                scored.append(
                    {
                        "valid": False,
                        "parse": False,
                        "roundtrip": False,
                        "finite": False,
                        "task_score": 0.0,
                        "error": f"{type(error).__name__}: {error}",
                        "diagnostics": output.get("diagnostics", {}),
                    }
                )
        row = {
            "ordinal": ordinal,
            "challenge_id": metadata["p11_challenge_id"],
            "task": metadata["p11_task"],
            "family": metadata["p11_challenge_family"],
            "edit_operation": (
                None
                if metadata.get("p11_edit_kind") is None
                else str(metadata["p11_edit_kind"])
            ),
            "view_id": metadata["p11_prompt_view_id"],
            "selection_role": metadata["p11_challenge_selection_role"],
            "pair_id": metadata.get("p11_challenge_pair_id"),
            "pair_label": metadata.get("p11_challenge_pair_label"),
            "duration_source": _duration(metadata)[1],
            "rng_isolation_pass": all(row_rng[ordinal]),
            "scored": scored,
            "prefixes": {
                str(k): _summarize_prefix(scored, metadata, k=k)
                for k in k_values
                if len(scored) >= k
            },
        }
        rows.append(row)
    del planner, wrapper

    aggregate = _aggregate(rows, k_values)
    counterfactual = _counterfactual_summary(rows, k_values)
    evaluator_integrity = all(
        row["rng_isolation_pass"] and len(row["scored"]) == args.draws
        for row in rows
    )
    public_rows = []
    for row in rows:
        public_rows.append(
            {
                **{key: value for key, value in row.items() if key != "scored"},
                "scored": [
                    {key: value for key, value in value.items() if key != "core"}
                    for value in row["scored"]
                ],
            }
        )
    scientific_kernel = args.qwen_kernel_mode == CANONICAL_QWEN_KERNEL_MODE
    evaluator_contract = (
        "hybrid_flow_gu_direct_e_torch_reference_fair_lexical_boundary_v10"
        if scientific_kernel
        else (
            "hybrid_flow_gu_direct_e_fast_kernel_diagnostic_"
            f"{args.qwen_kernel_mode}_v10"
        )
    )
    report = {
        "schema": "stable_audio_tools.p11_v4_unified_challenge_eval",
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "evaluator_contract": evaluator_contract,
        "status": "PASS" if evaluator_integrity else "FAIL",
        "scope": (
            "matched P10-conditioning challenge evaluation; no rendered FOA, "
            "and no promotion claim from pilot checkpoints"
        ),
        "arm": args.arm,
        "architecture": {
            "d0": (
                "full ScenePlan autoregressive constrained sampling + "
                "input-only deterministic lexical post-assembly"
            ),
            "direct": "SceneSketch + deterministic Direct-MSE ExecutionState",
            "flow": (
                "SceneSketch + Flow-R1 G/U posterior ExecutionState + "
                "deterministic Direct DeltaThought E"
            ),
        }[args.arm],
        "model_config": str(model_config_path),
        "dataset_config": str(dataset_config_path),
        "config_provenance": {
            "model": _resolved_config_provenance(model_config_path),
            "dataset": _resolved_config_provenance(dataset_config_path),
        },
        "lexical_evidence_provenance": _lexical_evidence_provenance(
            dataset_config_path
        ),
        "lexical_authority_intervention": {
            "mode": args.lexical_authority_intervention,
            "input_lexical_removed_before_model": (
                args.lexical_authority_intervention == "drop_reliable_asr"
            ),
            "model_weights_changed": False,
            "target_transcript_access": "forbidden",
        },
        "qwen_kernel_mode": args.qwen_kernel_mode,
        "scientific_kernel_contract": {
            "canonical_mode": CANONICAL_QWEN_KERNEL_MODE,
            "scientific_report": scientific_kernel,
            "fast_fla_modes_are_diagnostic_only": True,
            "same_device_fresh_process_exact_replay_required": scientific_kernel,
        },
        "challenge": str(challenge_path),
        "challenge_sha256": _sha256_file(challenge_path),
        "runtime_source_provenance": _runtime_source_provenance(device),
        "checkpoints": model_provenance,
        "scoring_checkpoint_reload": scoring_provenance,
        "weights": args.weights,
        "root_seed": args.seed,
        "draws": args.draws,
        "k_values": k_values,
        "d0_temperature": args.d0_temperature if args.arm == "d0" else None,
        "discrete_decode_mode": args.discrete_decode_mode,
        "selected_ordinals": selected,
        "rows": len(rows),
        "rows_per_view": args.rows_per_view,
        "families_filter": None if families is None else sorted(families),
        "rng_contract": "sample_local_explicit_generator_v1",
        "rng_isolation_rate": _mean(
            [float(bool(row["rng_isolation_pass"])) for row in rows]
        ),
        "performance": performance,
        "aggregate": aggregate,
        "editing_counterfactual": counterfactual,
        "posterior_fairness": {
            "flow_draws_are_explicit_noise_samples": False,
            "flow_gu_draws_are_explicit_noise_samples": args.arm == "flow",
            "flow_e_draws_are_deterministic_direct_delta": args.arm == "flow",
            "flow_e_repeated_k_draws_claim_posterior_diversity": False,
            "d0_draws_are_constrained_token_samples": args.arm == "d0",
            "direct_draws_are_independent_checkpoints": args.arm == "direct",
            "direct_ensemble_size": len(checkpoints) if args.arm == "direct" else None,
            "direct_posterior_comparison_ready": (
                args.arm != "direct" or len(checkpoints) >= 2
            ),
            "oracle_best_of_k_is_deployable_selection_metric": False,
        },
        "calibration_contract": (
            "empirical_ensemble_gaussian_interval_diagnostic_v1; unavailable at K=1"
        ),
        "p10_closure": {
            "conditioning_compiled_for_every_valid_draw": True,
            "rendered_foa_cycle": "NOT_RUN_IN_THIS_STAGE",
            "reason": "challenge/evaluator falsification precedes expensive frozen-P10 render",
        },
        "quality_decision": "NOT_ESTABLISHED_BY_EVALUATOR_SMOKE",
        "aggregate_rows": public_rows,
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
