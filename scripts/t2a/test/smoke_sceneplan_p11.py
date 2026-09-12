#!/usr/bin/env python3
"""CPU graph smoke for the required discrete D0 baseline.

The production Qwen checkpoint uses CUDA-only recurrent kernels and is covered
by the GPU preflight.  This smoke substitutes a tiny causal backbone while
exercising the real P11 codec, manifest, task routing, LoRA injection, hybrid
audio bridge, loss, backward graph, and optimizer update on CPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.sceneplan_p11_v4_curriculum import (  # noqa: E402
    ScenePlanP11V4CurriculumDataset,
)
from stable_audio_tools.data.sceneplan_p11_dataset import ScenePlanP11Dataset  # noqa: E402
from stable_audio_tools.models.sceneplan_p11 import ScenePlanP11Planner  # noqa: E402
from stable_audio_tools.training.factory import (  # noqa: E402
    create_training_wrapper_from_config,
)


DEFAULT_MODEL = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_baseline_discrete_d0.json"
)
DEFAULT_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_pilot90_curriculum_pair_aware_discrete_d0.json"
)


class _ModuloEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return nn.functional.embedding(token_ids.remainder(self.weight.shape[0]), self.weight)


class _TinyDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        source = self.norm(value)
        attention = torch.tanh(
            (self.q_proj(source) + self.k_proj(source) + self.v_proj(source)) / 3.0
        )
        value = value + self.o_proj(attention)
        source = self.norm(value)
        return value + self.down_proj(
            nn.functional.silu(self.gate_proj(source)) * self.up_proj(source)
        )


class _TinyBackbone(nn.Module):
    def __init__(self, hidden_size: int = 64, depth: int = 2) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed_tokens = _ModuloEmbedding(4096, hidden_size)
        self.layers = nn.ModuleList(
            [_TinyDecoderLayer(hidden_size) for _ in range(depth)]
        )

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool,
        **_: object,
    ) -> SimpleNamespace:
        if attention_mask is not None:
            raise RuntimeError("canonical P11 smoke unexpectedly entered a ragged path")
        value = inputs_embeds
        positions = torch.arange(
            1, value.shape[1] + 1, device=value.device, dtype=value.dtype
        ).view(1, -1, 1)
        value = torch.cumsum(value, dim=1) / positions
        for layer in self.layers:
            value = layer(value)
        return SimpleNamespace(last_hidden_state=value, past_key_values=None)


class _TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyBackbone()


def _gradient_norm(module: nn.Module) -> float:
    values = [
        parameter.grad.detach().float().norm()
        for parameter in module.parameters()
        if parameter.grad is not None
    ]
    return float(torch.stack(values).norm()) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 3:
        raise ValueError("P11 CPU graph smoke requires batch-size >= 3")
    torch.manual_seed(20260831)

    model_config = copy.deepcopy(load_config(args.model_config))
    dataset_config = load_config(args.dataset_config)
    text = model_config["model"]["text"]
    text["hidden_size"] = 64
    text["initialize_plan_embeddings_from_qwen"] = False
    text["lora"]["top_layers"] = 2
    text["lora"]["target_modules"] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    model_config["model"]["audio_bridge"]["temporal"]["connector_depth"] = 1
    model_config["model"]["audio_bridge"]["semantic"]["depth"] = 1
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        text["model_path"], local_files_only=True, use_fast=True
    )
    with patch.object(
        AutoModelForCausalLM,
        "from_pretrained",
        return_value=_TinyCausalLM(),
    ):
        model = ScenePlanP11Planner(model_config).cpu().train()
    # The tiny test double contains ordinary decoder layers rather than the
    # production Qwen3.5 GatedDeltaNet kernels. It is already CPU-native, so
    # mark the production-only kernel replacement as satisfied for this smoke.
    model._qwen_cpu_fallback_enabled = True
    wrapper = create_training_wrapper_from_config(model_config, model)
    if wrapper.p11 is not model:
        raise RuntimeError("P11 training factory replaced the planner instance")
    dataset = ScenePlanP11Dataset(
        dataset_config["manifest_path"],
        index_path=dataset_config["datasets"][0]["path"],
        codec_path=dataset_config["codec_path"],
        tokenizer_spec=(tokenizer, 512, None),
        expected_num_samples=int(dataset_config["expected_num_samples"]),
        index_num_samples=int(dataset_config["index_num_samples"]),
        require_frozen=True,
        semantic_cache_path=dataset_config["semantic_cache_path"],
        semantic_dim=int(dataset_config["semantic_dim"]),
        semantic_encoder_revision=dataset_config["semantic_encoder_revision"],
    )
    if dataset_config.get("p11_v4_curriculum_path") is not None:
        dataset = ScenePlanP11V4CurriculumDataset(
            dataset,
            dataset_config["p11_v4_curriculum_path"],
            expected_rows=int(
                dataset_config["p11_v4_curriculum_expected_rows"]
            ),
            expected_contract=dataset_config.get(
                "p11_v4_curriculum_contract"
            ),
            expected_ordering_contract=dataset_config.get(
                "p11_v4_curriculum_ordering_contract"
            ),
            expected_ordering_batch_size=dataset_config.get(
                "p11_v4_curriculum_ordering_batch_size"
            ),
        )
    rows = [dataset[index][1] for index in range(args.batch_size)]

    def forward(selected):
        return model.forward_planner(
            [row["p11_prompt"] for row in selected],
            [row["p11_target_tokens"] for row in selected],
            tasks=[row["p11_task"] for row in selected],
            input_foa=[row["p11_input_foa"] for row in selected],
            input_valid_masks=[row["p11_input_valid_mask"] for row in selected],
            input_semantic=[row["p11_input_semantic"] for row in selected],
            input_plans=[row["p11_input_sceneplan_tokens"] for row in selected],
            loss_group_weights=model_config["training"]["planner_loss_group_weights"],
        )

    loss, metrics = forward(rows)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("P11 CPU smoke produced non-finite loss")
    loss.backward()
    gradient_norms = {
        "lora": _gradient_norm(model.lora_adapters),
        "temporal": _gradient_norm(model.temporal_tokenizer),
        "semantic": _gradient_norm(model.semantic_resampler),
        "plan_embedding": _gradient_norm(model.plan_embedding),
    }
    if any(not value > 0.0 for value in gradient_norms.values()):
        raise RuntimeError(f"P11 CPU smoke found a disconnected graph: {gradient_norms}")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.0e-4,
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    sparse_rows = [rows[0], rows[2]]
    sparse_loss, _ = forward(sparse_rows)
    sparse_loss.backward()
    missing_gradients = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        raise RuntimeError(
            "P11 static DDP graph drops parameters when U is absent: "
            f"{missing_gradients[:5]}"
        )
    model.eval()
    edit_tokens, decode = model.decode_output_tokens(
        rows[2]["p11_prompt"],
        task=rows[2]["p11_task"],
        input_sceneplan=rows[2]["p11_input_sceneplan"],
        max_tokens=model.patch_max_tokens,
        constrained=True,
    )
    if not decode["terminated"]:
        raise RuntimeError(
            "P11 CPU smoke constrained decode did not terminate: "
            f"tokens={edit_tokens.tolist()} diagnostics={decode}"
        )
    model.patch_codec.canonicalize(edit_tokens)
    report = {
        "status": "PASS",
        "smoke": "sceneplan_p11_cpu_graph",
        "tasks": [row["p11_task"] for row in rows],
        "output_kinds": [row["p11_output_kind"] for row in rows],
        "loss": float(loss.detach()),
        "arm": "discrete_d0",
        "batch_size": len(rows),
        "planner_tokens": int(metrics["planner_tokens"]),
        "per_row_ce": metrics["planner_ce_per_row"].tolist(),
        "gradient_norms": gradient_norms,
        "static_graph_missing_gradients": len(missing_gradients),
        "training_wrapper": type(wrapper).__name__,
        "decode": decode,
        "non_finite": 0,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
