#!/usr/bin/env python3
"""Validate exact no-truncation budgets for active audio-aware P11 rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.sceneplan_p11_v4_dataset import (  # noqa: E402
    ScenePlanP11AudioAwareDataset,
)


DEFAULT_MODEL = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_audio_aware_v1.json"
)
DEFAULT_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_audio_aware_v1_pilot90.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/audio_aware_v1/pilot90_sequence_budget_seed42.json"
)


def _token_count(value: Mapping[str, torch.Tensor] | torch.Tensor) -> int:
    if isinstance(value, Mapping):
        mask = value.get("attention_mask")
        if mask is not None:
            return int(torch.as_tensor(mask, dtype=torch.bool).sum())
        value = value["input_ids"]
    return int(torch.as_tensor(value).numel())


class _SequenceBudgetView(torch.utils.data.Dataset):
    """Evaluate each exact runtime stage and return only tiny counters."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        temporal_stride: int,
        semantic_queries: int,
        lexical_policy: str,
        thought_slots: int,
    ) -> None:
        self.dataset = dataset
        self.temporal_stride = int(temporal_stride)
        self.semantic_queries = int(semantic_queries)
        self.lexical_policy = str(lexical_policy)
        self.thought_slots = int(thought_slots)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        _, row = self.dataset[index]
        task = str(row["p11_task"])
        prompt_tokens = _token_count(row["p11_prompt"])
        observation_prompt_tokens = _token_count(row["p11_observation_prompt"])
        input_plan_tokens = (
            0
            if row["p11_prior_sceneplan_tokens"] is None
            else _token_count(row["p11_prior_sceneplan_tokens"])
        )
        temporal_tokens = 0
        semantic_tokens = 0
        lexical_tokens = 0
        lexical_context_tokens = 0
        if task in {"understanding", "editing"}:
            valid_frames = int(
                torch.as_tensor(
                    row["p11_input_valid_mask"], dtype=torch.bool
                ).sum()
            )
            temporal_tokens = max(1, valid_frames // self.temporal_stride)
            semantic_tokens = self.semantic_queries
            if row["p11_input_lexical"] is not None:
                lexical_tokens = _token_count(row["p11_input_lexical"])
                if self.lexical_policy == "optional_reliable_speech_only_v1":
                    lexical_context_tokens = lexical_tokens
                elif (
                    self.lexical_policy
                    != "deterministic_reliable_speech_assembler_v1"
                ):
                    raise RuntimeError(
                        "unsupported lexical injection policy "
                        f"{self.lexical_policy!r}"
                    )

        def context_length(prompt: int, *, include_prior: bool) -> int:
            length = 1 + int(prompt)
            if include_prior and input_plan_tokens:
                length += 2 + input_plan_tokens
            if task in {"understanding", "editing"}:
                length += 2 + temporal_tokens + semantic_tokens
                if lexical_context_tokens:
                    length += 2 + lexical_context_tokens
            return length

        observed_tokens = _token_count(row["p11_observed_scene_sketch_tokens"])
        observation_context = context_length(
            observation_prompt_tokens, include_prior=False
        )
        observation_discrete_train = observation_context + observed_tokens + 1
        # context + <SCENE_SKETCH> + tokens + </SCENE_SKETCH>
        # + <SCENE_THOUGHT> + K slots + </SCENE_THOUGHT>
        observation_thought_train = (
            observation_context + observed_tokens + self.thought_slots + 4
        )

        delta_tokens = 0
        delta_context = 0
        delta_discrete_train = 0
        delta_thought_train = 0
        if task == "editing":
            delta_tokens = _token_count(row["p11_delta_scene_sketch_tokens"])
            delta_base = context_length(prompt_tokens, include_prior=True)
            # The delta stage appends the model-visible observed sketch and
            # observed thought to its independent E evidence context.
            delta_context = (
                delta_base + observed_tokens + self.thought_slots + 4
            )
            delta_discrete_train = delta_context + delta_tokens + 1
            delta_thought_train = (
                delta_context + delta_tokens + self.thought_slots + 4
            )
        required = max(
            observation_discrete_train,
            observation_thought_train,
            delta_discrete_train,
            delta_thought_train,
        )
        return {
            "row": int(index),
            "task": task,
            "required": required,
            "observation_context": observation_context,
            "delta_context": delta_context,
            "prompt": prompt_tokens,
            "observation_prompt": observation_prompt_tokens,
            "input_plan": input_plan_tokens,
            "temporal": temporal_tokens,
            "semantic": semantic_tokens,
            "lexical": lexical_tokens,
            "lexical_context": lexical_context_tokens,
            "observed_discrete": observed_tokens,
            "delta_discrete": delta_tokens,
            "observation_discrete_train": observation_discrete_train,
            "observation_thought_train": observation_thought_train,
            "delta_discrete_train": delta_discrete_train,
            "delta_thought_train": delta_thought_train,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.num_workers <= 32:
        raise ValueError("--num-workers must be within [0,32]")
    model_path = args.model_config.expanduser().resolve(strict=True)
    dataset_path = args.dataset_config.expanduser().resolve(strict=True)
    model_config = load_config(model_path)
    dataset_config = load_config(dataset_path)
    text = model_config["model"]["text"]
    transfusion = model_config["model"]["transfusion_cot"]
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        text["model_path"], local_files_only=True, use_fast=True
    )
    dataset = ScenePlanP11AudioAwareDataset(
        dataset_config["manifest_path"],
        index_path=dataset_config["datasets"][0]["path"],
        codec_path=dataset_config["codec_path"],
        tokenizer_spec=(tokenizer, int(text["max_length"]), None),
        expected_num_samples=int(dataset_config["expected_num_samples"]),
        index_num_samples=int(dataset_config["index_num_samples"]),
        require_frozen=bool(dataset_config["require_complete"]),
        semantic_cache_path=dataset_config["semantic_cache_path"],
        semantic_dim=int(dataset_config["semantic_dim"]),
        semantic_encoder_revision=dataset_config["semantic_encoder_revision"],
        lexical_evidence_mode=dataset_config.get("lexical_evidence_mode", "none"),
        lexical_max_tokens=int(dataset_config.get("lexical_max_tokens", 128)),
        lexical_cache_path=dataset_config.get("lexical_cache_path"),
        lexical_encoder_revision=dataset_config.get("lexical_encoder_revision"),
        lexical_confidence_threshold=dataset_config.get(
            "lexical_confidence_threshold"
        ),
    )
    if dataset_config.get("p11_v4_curriculum_path") is not None:
        raise ValueError(
            "active audio-aware P11 cannot consume the retired v4 curriculum"
        )
    sequence_limit = int(text["sequence_length"])
    temporal_stride = int(model_config["model"]["audio_bridge"]["temporal"]["stride"])
    semantic_queries = int(
        model_config["model"]["audio_bridge"]["semantic"]["num_queries"]
    )
    raw_lexical_policy = transfusion.get("lexical_evidence", {}).get(
        "injection_policy"
    )
    lexical_policy = (
        "disabled_v1" if raw_lexical_policy is None else str(raw_lexical_policy)
    )
    thought_slots = int(transfusion["thought"]["slot_count"])
    maxima: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    violations: list[dict[str, Any]] = []
    budget_view = _SequenceBudgetView(
        dataset,
        temporal_stride=temporal_stride,
        semantic_queries=semantic_queries,
        lexical_policy=lexical_policy,
        thought_slots=thought_slots,
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": budget_view,
        "batch_size": None,
        "shuffle": False,
        "num_workers": args.num_workers,
    }
    if args.num_workers:
        loader_kwargs["prefetch_factor"] = 2
    for details in torch.utils.data.DataLoader(**loader_kwargs):
        task = str(details["task"])
        counts[task] += 1
        required = int(details["required"])
        if task not in maxima or required > int(maxima[task]["required"]):
            maxima[task] = details
        if required > sequence_limit:
            violations.append(details)
    if violations:
        raise RuntimeError(
            f"audio-aware P11 has {len(violations)} sequence-budget violations; "
            f"first={violations[0]}"
        )
    report = {
        "schema": "stable_audio_tools.p11_audio_aware_sequence_budget",
        "schema_version": 1,
        "status": "PASS",
        "model_config": str(model_path),
        "dataset_config": str(dataset_path),
        "rows": len(dataset),
        "task_counts": dict(sorted(counts.items())),
        "sequence_limit": sequence_limit,
        "maxima": maxima,
        "violations": 0,
        "truncation": 0,
        "scene_sketch_max_tokens": int(transfusion["scene_sketch_max_tokens"]),
        "delta_sketch_max_tokens": int(transfusion["delta_sketch_max_tokens"]),
        "thought_slots": thought_slots,
        "lexical_injection_policy": lexical_policy,
        "validator_workers": args.num_workers,
        "full_row_scan": True,
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
