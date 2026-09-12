#!/usr/bin/env python3
"""Fail-closed gate for the inference-aligned P11-v4 DeltaSketch owner."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec_v3 import (  # noqa: E402
    SOURCE_SLOT_TOKENS,
)
from stable_audio_tools.data.sceneplan_p11_v4_curriculum import (  # noqa: E402
    ScenePlanP11V4CurriculumDataset,
)
from stable_audio_tools.data.sceneplan_p11_v4_dataset import (  # noqa: E402
    ScenePlanP11V4Dataset,
)
from stable_audio_tools.models.sceneplan_p11_v4 import (  # noqa: E402
    P11_V4_DELTA_OWNER_CONTRACT,
    P11_V4_DELTA_OWNER_OBJECTIVE,
    _delta_owner_objective,
)


DEFAULT_MODEL = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/"
    "qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
)
DEFAULT_DATASET = REPO_ROOT / (
    "stable_audio_tools/configs/dataset_configs/"
    "sceneplan_p11_pilot90_curriculum_pair_aware_transfusion_cot_v4_reliable_asr_v1.json"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/p11_v4_delta_owner_gate_20260902.json"
)


def _token_count(value: Mapping[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(value, Mapping):
        value = value["input_ids"]
    return torch.as_tensor(value, dtype=torch.long).flatten()


def _gradient_probe() -> dict[str, Any]:
    logits = torch.tensor([3.0, 1.0, 100.0, -100.0], requires_grad=True)
    loss, correct = _delta_owner_objective(
        logits,
        0,
        torch.tensor([True, True, False, False]),
        margin=2.0,
        margin_weight=1.0,
    )
    loss.backward()
    if logits.grad is None:
        raise RuntimeError("DeltaSketch owner objective produced no gradient")
    if not (
        float(logits.grad[0]) < 0.0
        and float(logits.grad[1]) > 0.0
        and bool(logits.grad[2:].eq(0.0).all())
    ):
        raise RuntimeError("DeltaSketch owner gradient escaped legal inference logits")
    return {
        "loss": float(loss.detach()),
        "correct": float(correct.detach()),
        "gradient": [float(value) for value in logits.grad],
        "absent_source_gradient_max_abs": float(logits.grad[2:].abs().max()),
    }


class _DeltaOwnerView(torch.utils.data.Dataset):
    """Scan every ordinal while materializing only owner-relevant E rows."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        *,
        source_token_ids: tuple[int, ...],
    ) -> None:
        self.dataset = dataset
        self.source_token_ids = tuple(int(value) for value in source_token_ids)
        self.token_to_slot = {
            token_id: slot for slot, token_id in enumerate(self.source_token_ids)
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Curriculum task identity is immutable SQLite metadata. Looking at it
        # first avoids decoding FOA/CLAP/ScenePlan payloads for G/U rows that
        # the owner gate has always ignored, while still visiting every ordinal.
        if isinstance(self.dataset, ScenePlanP11V4CurriculumDataset):
            task_row = self.dataset._db().execute(
                "SELECT task FROM rows WHERE ordinal=?", (int(index),)
            ).fetchone()
            if task_row is None:
                raise RuntimeError(f"P11 curriculum lacks ordinal {index}")
            if str(task_row[0]) != "editing":
                return {"editing": False}

        _, row = self.dataset[index]
        if str(row["p11_task"]) != "editing":
            return {"editing": False}
        ids = _token_count(row["p11_target_tokens"])
        target_token = int(ids[2]) if int(ids.numel()) >= 4 else -1
        target_slot = self.token_to_slot.get(target_token)
        operation = str((row.get("p11_edit_spec") or {}).get("operation") or "")
        if target_slot is None:
            if operation in {
                "rotate_source",
                "distance_source",
                "retime_source",
                "remove_source",
            }:
                raise RuntimeError(f"row {index} lost its DeltaSketch owner token")
            return {
                "editing": True,
                "has_owner": False,
                "operation": operation,
                "ambiguous": False,
            }

        legal_mask = torch.as_tensor(
            row["p11_v4_input_source_mask"], dtype=torch.bool
        ).flatten()
        if tuple(legal_mask.shape) != (4,) or not bool(legal_mask[target_slot]):
            raise RuntimeError(f"row {index} owner is absent from current ScenePlan")
        allowed = self.dataset.base.delta_sketch_codec.allowed_next_ids(
            ids[:2], input_sceneplan=row["p11_input_sceneplan"]
        ) if isinstance(
            self.dataset, ScenePlanP11V4CurriculumDataset
        ) else self.dataset.delta_sketch_codec.allowed_next_ids(
            ids[:2], input_sceneplan=row["p11_input_sceneplan"]
        )
        expected_allowed = {
            self.source_token_ids[slot]
            for slot, present in enumerate(legal_mask.tolist())
            if present
        }
        if allowed != expected_allowed:
            raise RuntimeError(
                f"row {index} train legal-owner mask differs from inference grammar"
            )
        return {
            "editing": True,
            "has_owner": True,
            "operation": operation,
            "ambiguous": int(legal_mask.sum()) > 1,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if not 0 <= args.num_workers <= 32:
        raise ValueError("--num-workers must be within [0,32]")

    model_path = args.model_config.expanduser().resolve(strict=True)
    dataset_path = args.dataset_config.expanduser().resolve(strict=True)
    model_config = load_config(model_path)
    dataset_config = load_config(dataset_path)
    owner = model_config["model"]["transfusion_cot"]["delta_owner"]
    expected = {
        "contract": P11_V4_DELTA_OWNER_CONTRACT,
        "objective": P11_V4_DELTA_OWNER_OBJECTIVE,
        "authority": "delta_sketch_owner_token_v1",
        "context": "edit_instruction_plus_current_sceneplan_v1",
        "inference": "grammar_constrained_argmax_v1",
        "legal_source_inventory": "input_sceneplan_source_mask_v1",
        "loss_normalization": "active_owner_rows_v1",
    }
    for key, value in expected.items():
        if owner.get(key) != value:
            raise RuntimeError(f"DeltaSketch owner config {key} changed")
    if float(owner["margin"]) <= 0.0 or float(owner["margin_weight"]) <= 0.0:
        raise RuntimeError("DeltaSketch owner margin objective is disabled")
    owner_weight = float(
        model_config["training"]["transfusion_cot_loss_weights"]["owner"]
    )
    if owner_weight <= 0.0:
        raise RuntimeError("DeltaSketch owner training weight is disabled")

    from transformers import AutoTokenizer

    text_config = model_config["model"]["text"]
    tokenizer = AutoTokenizer.from_pretrained(
        text_config["model_path"], local_files_only=True, use_fast=True
    )
    base = ScenePlanP11V4Dataset(
        dataset_config["manifest_path"],
        index_path=dataset_config["datasets"][0]["path"],
        codec_path=dataset_config["codec_path"],
        tokenizer_spec=(tokenizer, int(text_config["max_length"]), None),
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
    dataset = base
    curriculum_path = dataset_config.get("p11_v4_curriculum_path")
    if curriculum_path is not None:
        dataset = ScenePlanP11V4CurriculumDataset(
            base,
            curriculum_path,
            expected_rows=int(dataset_config["p11_v4_curriculum_expected_rows"]),
            expected_contract=str(
                dataset_config.get(
                    "p11_v4_curriculum_contract",
                    "p10_v11_train_only_gue_multitarget_v1",
                )
            ),
            expected_ordering_contract=dataset_config.get(
                "p11_v4_curriculum_ordering_contract"
            ),
            expected_ordering_batch_size=dataset_config.get(
                "p11_v4_curriculum_ordering_batch_size"
            ),
        )

    source_token_ids = tuple(base.codec._tid(token) for token in SOURCE_SLOT_TOKENS)
    owner_rows = 0
    ambiguous_rows = 0
    editing_rows = 0
    operation_counts: Counter[str] = Counter()
    owner_view = _DeltaOwnerView(
        dataset, source_token_ids=source_token_ids
    )
    loader_kwargs: dict[str, Any] = {
        "dataset": owner_view,
        "batch_size": None,
        "shuffle": False,
        "num_workers": args.num_workers,
    }
    if args.num_workers:
        loader_kwargs["prefetch_factor"] = 2
    for result in torch.utils.data.DataLoader(**loader_kwargs):
        if not bool(result["editing"]):
            continue
        editing_rows += 1
        if not bool(result.get("has_owner", False)):
            continue
        owner_rows += 1
        operation_counts[str(result["operation"])] += 1
        ambiguous_rows += int(bool(result["ambiguous"]))
    if owner_rows <= 0 or ambiguous_rows <= 0:
        raise RuntimeError("DeltaSketch owner gate lacks an ambiguous source decision")

    report = {
        "schema": "stable_audio_tools.p11_v4_delta_owner_gate",
        "schema_version": 1,
        "status": "PASS",
        "model_config": str(model_path),
        "dataset_config": str(dataset_path),
        "contract": owner,
        "training_weight": owner_weight,
        "rows": len(dataset),
        "editing_rows": editing_rows,
        "owner_rows": owner_rows,
        "ambiguous_owner_rows": ambiguous_rows,
        "operation_counts": dict(sorted(operation_counts.items())),
        "train_inference_legal_set_mismatches": 0,
        "validator_workers": args.num_workers,
        "full_ordinal_scan": True,
        "non_edit_rows_materialized": False,
        "gradient_probe": _gradient_probe(),
        "second_owner_authority": False,
        "eight_gpu_training_authorized": False,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
