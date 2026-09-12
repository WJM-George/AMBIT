#!/usr/bin/env python3
"""One real-GPU graph gate for active audio-aware P11.

This is one forward/backward/optimizer/EMA step, not a training run. It also
proves that E carries source FOA and CLAP evidence into the model graph and
that the dedicated Direct-MSE DeltaExecutionState head receives gradients.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config, validate_t2a_config  # noqa: E402
from stable_audio_tools.data.dataset import create_dataloader_from_config  # noqa: E402
from stable_audio_tools.data.sceneplan_p11_single_turn import P11Task  # noqa: E402
from stable_audio_tools.data.text_conditioning import collect_conditioner_tokenizers  # noqa: E402
from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.training.factory import create_training_wrapper_from_config  # noqa: E402


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
    "pilot90_real_graph_smoke_seed42.json"
)
NUMERIC_DELTA_OPERATIONS = {"add_source", "move_source", "retime_source"}


def _gradient_norm(module: torch.nn.Module | list[torch.nn.Module]) -> float:
    modules = [module] if isinstance(module, torch.nn.Module) else list(module)
    gradients = [
        parameter.grad.detach().float().norm()
        for one_module in modules
        for parameter in one_module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return float(torch.stack(gradients).norm()) if gradients else 0.0


def _optimizer(wrapper) -> torch.optim.Optimizer:
    configured = wrapper.configure_optimizers()
    candidates = configured[0] if isinstance(configured, tuple) else configured
    optimizer = candidates[0] if isinstance(candidates, (list, tuple)) else candidates
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise RuntimeError("audio-aware P11 wrapper did not create one optimizer")
    return optimizer


def _finite_metrics(metrics: dict[str, Any]) -> bool:
    return all(
        not isinstance(value, torch.Tensor)
        or bool(torch.isfinite(value.detach()).all())
        for value in metrics.values()
    )


def _token_count(value: Any) -> int:
    if isinstance(value, dict):
        if value.get("attention_mask") is not None:
            return int(torch.as_tensor(value["attention_mask"]).bool().sum())
        value = value["input_ids"]
    return int(torch.as_tensor(value).numel())


def _row_sequence_score(row: dict[str, Any]) -> int:
    """Exact longest active stage for pilot batch selection."""

    task = str(row["p11_task"])
    prompt = _token_count(row["p11_prompt"])
    observed = _token_count(row["p11_observed_scene_sketch_tokens"])
    if task == "generation":
        return 1 + prompt + observed + 9
    valid = int(torch.as_tensor(row["p11_input_valid_mask"]).bool().sum())
    audio = 2 + max(1, valid // 8) + 8
    if task == "understanding":
        return 1 + prompt + audio + observed + 9
    prior = (
        0
        if row["p11_prior_sceneplan_tokens"] is None
        else 2 + _token_count(row["p11_prior_sceneplan_tokens"])
    )
    delta = _token_count(row["p11_delta_scene_sketch_tokens"])
    return 1 + prompt + prior + audio + observed + delta + 18


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-scan-batches", type=int, default=32)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 3:
        raise ValueError("audio-aware graph gate needs a G/U/E-containing batch")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real audio-aware P11 graph gate requires CUDA")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model_config_path = args.model_config.expanduser().resolve(strict=True)
    dataset_config_path = args.dataset_config.expanduser().resolve(strict=True)
    model_config = load_config(model_config_path)
    dataset_config = load_config(dataset_config_path)
    summary = validate_t2a_config(model_config, dataset_config)
    if summary["model_type"] != "sceneplan_p11_audio_aware_v1":
        raise ValueError("graph smoke accepts only the active P11 model contract")
    model = create_model_from_config(model_config)
    wrapper = create_training_wrapper_from_config(model_config, model).to(device)
    tokenizers = collect_conditioner_tokenizers(model, model_config)
    loader = create_dataloader_from_config(
        dataset_config,
        batch_size=int(args.batch_size),
        sample_size=int(model_config["sample_size"]),
        sample_rate=int(model_config["sample_rate"]),
        audio_channels=int(model_config["audio_channels"]),
        num_workers=int(args.num_workers),
        shuffle=False,
        tokenizers=tokenizers,
    )

    batch = None
    selected_batch_index = None
    selected_sequence_score = -1
    for batch_index, candidate in enumerate(loader):
        if batch_index >= int(args.max_scan_batches):
            break
        metadata = list(candidate[1])
        tasks = {str(row["p11_task"]) for row in metadata}
        if tasks != {"generation", "understanding", "editing"}:
            continue
        editing = [row for row in metadata if row["p11_task"] == "editing"]
        if not editing or any(
            row["p11_input_foa"] is None
            or row["p11_input_semantic"] is None
            or row["p11_observed_sceneplan_target"] is None
            or row["p11_delta_scene_sketch_tokens"] is None
            or row["p11_revised_sceneplan_target"] is None
            for row in editing
        ):
            continue
        if not any(
            str(row.get("p11_edit_kind")) in NUMERIC_DELTA_OPERATIONS
            for row in editing
        ):
            continue
        score = max(_row_sequence_score(row) for row in metadata)
        if score > selected_sequence_score:
            batch = candidate
            selected_batch_index = batch_index
            selected_sequence_score = score
    if batch is None:
        raise RuntimeError("no complete audio-aware G/U/E batch was found")
    metadata = list(batch[1])
    tasks = [str(row["p11_task"]) for row in metadata]

    torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic()
    loss = wrapper.training_step(batch, 0)
    torch.cuda.synchronize(device)
    forward_seconds = time.monotonic() - start
    metrics = wrapper._last_step_metrics
    if not bool(torch.isfinite(loss)) or not _finite_metrics(metrics):
        raise RuntimeError("audio-aware P11 graph produced non-finite loss/metrics")
    for name in (
        "observation_ce",
        "delta_ce",
        "flow",
        "observation_solve",
        "delta_solve",
        "delta_control",
        "locality",
        "text_end_ce",
        "scene_eos_ce",
        "u_inventory_ce",
    ):
        if not bool(torch.isfinite(metrics[f"train/{name}"])):
            raise RuntimeError(f"active objective {name} is non-finite")
    if float(metrics["train/task_editing"]) <= 0.0:
        raise RuntimeError("graph batch contains no active Editing row")
    if float(metrics["train/delta_control_rows"]) <= 0.0:
        raise RuntimeError(
            "graph batch contains no operation-owned numeric Editing row"
        )
    if float(metrics["train/input_audio_tokens"]) <= 0.0:
        raise RuntimeError("U/E source FOA tokens did not enter the graph")
    if float(metrics["train/input_semantic_tokens"]) <= 0.0:
        raise RuntimeError("U/E CLAP tokens did not enter the graph")

    start = time.monotonic()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.monotonic() - start
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients or not all(bool(torch.isfinite(value).all()) for value in gradients):
        raise RuntimeError("audio-aware graph found missing/non-finite gradients")
    delta_head = model.execution_reasoner.delta_head
    if delta_head is None:
        raise RuntimeError("active Editing lost its dedicated delta head")
    gradient_norms = {
        "lora_understanding": _gradient_norm(
            [a for a in model.lora_adapters if a.sct_stage == "understanding"]
        ),
        "lora_generation": _gradient_norm(
            [a for a in model.lora_adapters if a.sct_stage == "generation"]
        ),
        "temporal_bridge": _gradient_norm(model.temporal_tokenizer),
        "semantic_bridge": _gradient_norm(model.semantic_resampler),
        "plan_embedding": _gradient_norm(model.plan_embedding),
        "execution_reasoner": _gradient_norm(model.execution_reasoner),
        "delta_execution_head": _gradient_norm(delta_head),
    }
    disconnected = [name for name, value in gradient_norms.items() if value <= 0.0]
    if disconnected:
        raise RuntimeError(f"active trainable subgraphs are disconnected: {disconnected}")

    # Mechanistic input check independent of learned accuracy: with every
    # other E input fixed, replacing only source FOA or CLAP must alter the
    # exact context tensor consumed by the planner.
    edit_row = next(row for row in metadata if row["p11_task"] == "editing")
    model.eval()
    with torch.no_grad():
        qwen_dtype = model._ensure_qwen_device(device)
        kwargs = {
            "task": P11Task.EDITING,
            "prompt": edit_row["p11_prompt"],
            "input_valid_mask": edit_row["p11_input_valid_mask"],
            "input_plan": edit_row["p11_prior_sceneplan_tokens"],
            "input_lexical": edit_row["p11_input_lexical"],
            "device": device,
            "qwen_dtype": qwen_dtype,
        }
        source_context, _ = model._v4_context_embeddings(
            input_foa=edit_row["p11_input_foa"],
            input_semantic=edit_row["p11_input_semantic"],
            **kwargs,
        )
        zero_foa_context, _ = model._v4_context_embeddings(
            input_foa=torch.zeros_like(edit_row["p11_input_foa"]),
            input_semantic=edit_row["p11_input_semantic"],
            **kwargs,
        )
        zero_clap_context, _ = model._v4_context_embeddings(
            input_foa=edit_row["p11_input_foa"],
            input_semantic=torch.zeros_like(edit_row["p11_input_semantic"]),
            **kwargs,
        )
    if source_context.shape != zero_foa_context.shape or source_context.shape != zero_clap_context.shape:
        raise RuntimeError("E evidence intervention unexpectedly changed context shape")
    foa_context_effect = float((source_context - zero_foa_context).abs().max())
    clap_context_effect = float((source_context - zero_clap_context).abs().max())
    if foa_context_effect <= 1.0e-7 or clap_context_effect <= 1.0e-7:
        raise RuntimeError("E source FOA or CLAP evidence is disconnected from context")

    optimizer = _optimizer(wrapper)
    parameter = next(
        value
        for value in model.parameters()
        if value.requires_grad and value.grad is not None and bool(value.grad.ne(0).any())
    )
    before = parameter.detach().clone()
    optimizer.step()
    if torch.equal(before, parameter.detach()):
        raise RuntimeError("audio-aware optimizer step changed no inspected parameter")
    ema_before = int(wrapper.p11_ema.step.detach().cpu())
    wrapper.on_before_zero_grad()
    ema_after = int(wrapper.p11_ema.step.detach().cpu())
    if ema_after != ema_before + 1:
        raise RuntimeError("audio-aware EMA did not advance exactly once")
    optimizer.zero_grad(set_to_none=True)

    report = {
        "schema": "stable_audio_tools.p11_audio_aware_graph_smoke",
        "schema_version": 1,
        "status": "PASS",
        "scope": "one_real_backbone_forward_backward_optimizer_ema_step_not_training",
        "model_config": str(model_config_path),
        "dataset_config": str(dataset_config_path),
        "architecture": summary["architecture"],
        "model_type": summary["model_type"],
        "device": str(device),
        "batch_size": len(tasks),
        "selected_batch_index": selected_batch_index,
        "selected_max_sequence_tokens": selected_sequence_score,
        "tasks": tasks,
        "loss": float(loss.detach()),
        "losses": {
            key.removeprefix("train/"): float(value)
            for key, value in metrics.items()
            if key.startswith("train/") and isinstance(value, torch.Tensor)
        },
        "gradient_tensors": len(gradients),
        "nonzero_gradient_tensors": sum(int(bool(value.ne(0).any())) for value in gradients),
        "gradient_norms": gradient_norms,
        "editing_evidence_context_effect": {
            "zero_source_foa_max_abs": foa_context_effect,
            "zero_clap_max_abs": clap_context_effect,
        },
        "contracts": {
            "model": model.active_model_contract,
            "thought_arm": model.transfusion_cot_arm,
            "editing_input": "source_foa_plus_clap_plus_instruction_plus_optional_plan",
            "editing_delta": "semantics_only_delta_sketch_plus_direct_mse_execution_delta",
            "target_foa_supervision": False,
        },
        "all_gradients_finite": True,
        "optimizer_step": "PASS",
        "ema_step": {"before": ema_before, "after": ema_after},
        "timing_seconds": {"forward": forward_seconds, "backward": backward_seconds},
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / (2**30),
        "eight_gpu_training_authorized": False,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
