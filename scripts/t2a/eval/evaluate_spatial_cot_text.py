#!/usr/bin/env python3
"""Evaluate source-disjoint Spatial-CoT planning and understanding.

Teacher-forced mode measures aligned CE and FSM-constrained token decisions,
then deliberately mismatches the instruction, persistent state, or target FOA.
Free-decode mode runs a small immutable subset without rendering audio; the
separate closed-loop gate remains responsible for generated-state roll-in.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.train.gpu_preflight import assert_gpu_driver_healthy  # noqa: E402

assert_gpu_driver_healthy()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.t2a.eval.evaluate_spatial_cot_checkpoint import (  # noqa: E402
    DEFAULT_CODEC_ROOT,
    DEFAULT_MODEL_CONFIG,
    PLAN_GROUP_NAMES,
    _edit_field_metrics,
    _plan_field_metrics,
    _provider,
    _teacher_forced_plan_diagnostic,
)
from scripts.t2a.eval.spatial_cot_text_gate import (  # noqa: E402
    EVALUATOR_VERSION,
    RESULT_SCHEMA,
    atomic_json,
    file_sha256,
    load_text_panel,
    verify_panel_families,
)
from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.spatial_family_dataset import (  # noqa: E402
    SpatialFamilyDataset,
)
from stable_audio_tools.models import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import load_ckpt_state_dict  # noqa: E402


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _canonical_device(device: torch.device | str) -> torch.device:
    """Resolve aliases such as ``cuda`` to the process-local CUDA index."""

    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def _install_qwen_prefix_cache(
    model,
    captions: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> int:
    """Encode every frozen-Qwen prompt once, then reuse its exact soft prefix."""

    if batch_size < 1 or any(not isinstance(caption, str) for caption in captions):
        raise ValueError("Qwen prefix cache requires string captions and batch_size>=1")
    cache_device = _canonical_device(device)
    original = model._encode_qwen_prefix
    unique = list(dict.fromkeys(captions))
    cache: dict[
        str,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ],
    ] = {}
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            ids, soft, source_summaries, source_summary_masks = original(
                batch, device
            )
            if (source_summaries is None) != (source_summary_masks is None):
                raise RuntimeError(
                    "Qwen source summaries and masks must either both be present "
                    "or both be absent"
                )
            summaries = (
                source_summaries
                if source_summaries is not None
                else [None] * len(batch)
            )
            summary_masks = (
                source_summary_masks
                if source_summary_masks is not None
                else [None] * len(batch)
            )
            cache.update(
                zip(batch, zip(ids, soft, summaries, summary_masks))
            )

    def cached_encode(
        requested: Sequence[str], requested_device
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        if _canonical_device(requested_device) != cache_device:
            raise RuntimeError("Qwen prefix cache cannot move between devices")
        missing = [caption for caption in requested if caption not in cache]
        if missing:
            raise KeyError(f"Qwen prefix cache missed {len(missing)} captions")
        cached = [cache[caption] for caption in requested]
        summaries_present = any(item[2] is not None for item in cached)
        if summaries_present and not all(
            item[2] is not None and item[3] is not None for item in cached
        ):
            raise RuntimeError("Qwen prefix cache mixed source-summary contracts")
        return (
            [item[0] for item in cached],
            [item[1] for item in cached],
            [item[2] for item in cached] if summaries_present else None,
            [item[3] for item in cached] if summaries_present else None,
        )

    # This evaluator owns the model instance and runs under inference_mode.
    # Keeping the cache local avoids adding evaluation state to checkpoints.
    model.__dict__["_encode_qwen_prefix"] = cached_encode
    return len(cache)


def _objective_configs(model_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    objectives = {
        str(item.get("id")): item
        for item in model_config.get("training", {}).get("objectives", [])
        if isinstance(item, dict)
    }
    required = ("state_planner", "understanding")
    missing = [name for name in required if not isinstance(objectives.get(name), dict)]
    if missing:
        raise ValueError(f"model config is missing objectives: {missing}")
    return {name: objectives[name] for name in required}


def _load_families(
    dataset: SpatialFamilyDataset,
    panel: dict[str, Any],
    ranks: Sequence[int],
) -> list[dict[str, Any]]:
    turns_required = int(panel["turns"])
    families: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for rank in ranks:
        latents, info = dataset[rank]
        turns = list(info.get("family_turn_metadata") or [])
        if len(turns) < turns_required or int(latents.shape[0]) < turns_required:
            raise RuntimeError(f"family rank {rank} has fewer than {turns_required} turns")
        family_id = str(info["family_id"])
        identities.append({"family_rank": rank, "family_id": family_id})
        families.append(
            {
                "family_rank": rank,
                "family_id": family_id,
                "latents": latents,
                "turns": turns[:turns_required],
            }
        )
    verify_panel_families(panel, identities)
    return families


def _plan_weight(
    model,
    plan: Any,
    loss_group_weights: Mapping[str | int, float] | None,
) -> float:
    target = model._plan_input_ids(plan)
    normalized = model._normalize_plan_group_weights(loss_group_weights)
    if normalized is None:
        return float(target.numel())
    groups = model._plan_group_ids(plan, expected_length=int(target.numel()))
    if groups is None:
        raise ValueError("loss-group weights require target loss_group_ids")
    return sum(
        float(weight) * int(groups.eq(int(group_id)).sum())
        for group_id, weight in normalized.items()
    )


def _accumulate_loss(
    totals: dict[str, list[float]], key: str, loss: torch.Tensor, weight: float
) -> None:
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"non-positive planner loss weight for {key}: {weight}")
    value = float(loss.detach().float().cpu())
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite planner loss for {key}")
    aggregate = totals.setdefault(key, [0.0, 0.0])
    aggregate[0] += value * weight
    aggregate[1] += weight


def _mean_loss(totals: dict[str, list[float]], key: str) -> float:
    numerator, denominator = totals[key]
    return numerator / denominator


def _compact_diagnostic(diagnostic: dict[str, Any], plan: Any) -> dict[str, Any]:
    groups = torch.as_tensor(plan.get("loss_group_ids"), dtype=torch.long).flatten()
    if groups.numel() != int(diagnostic["token_count"]):
        raise RuntimeError("teacher diagnostic group count does not match tokens")
    group_tokens = Counter(int(value) for value in groups.tolist())
    group_errors = Counter(
        int(error["group_id"])
        for error in diagnostic["errors"]
        if error.get("group_id") is not None
    )
    return {
        "token_count": int(diagnostic["token_count"]),
        "ambiguous_token_count": int(diagnostic["ambiguous_token_count"]),
        "error_count": int(diagnostic["error_count"]),
        "token_accuracy": float(diagnostic["token_accuracy"]),
        "exact": bool(diagnostic["exact"]),
        "qwen_prefix_tokens": int(diagnostic["qwen_prefix_tokens"]),
        "group_token_counts": {
            PLAN_GROUP_NAMES.get(group, str(group)): count
            for group, count in sorted(group_tokens.items())
        },
        "group_error_counts": {
            PLAN_GROUP_NAMES.get(group, str(group)): count
            for group, count in sorted(group_errors.items())
        },
        "first_errors": [
            {
                "index": int(error["index"]),
                "group": error.get("group"),
                "target": error.get("target"),
                "predicted": error.get("predicted"),
                "target_rank": int(error["target_rank"]),
                "target_minus_best_logit": float(
                    error["target_minus_best_logit"]
                ),
            }
            for error in diagnostic["errors"][:8]
        ],
    }


def _aggregate_diagnostics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tokens = sum(int(item["token_count"]) for item in records)
    ambiguous = sum(int(item["ambiguous_token_count"]) for item in records)
    errors = sum(int(item["error_count"]) for item in records)
    exact = sum(bool(item["exact"]) for item in records)
    group_tokens: Counter[str] = Counter()
    group_errors: Counter[str] = Counter()
    for item in records:
        group_tokens.update(item["group_token_counts"])
        group_errors.update(item["group_error_counts"])
    return {
        "example_count": len(records),
        "token_count": tokens,
        "ambiguous_token_count": ambiguous,
        "error_count": errors,
        "token_accuracy": (tokens - errors) / tokens,
        "ambiguous_token_accuracy": (
            (ambiguous - errors) / ambiguous if ambiguous else 1.0
        ),
        "exact_count": exact,
        "exact_fraction": exact / len(records),
        "group_metrics": {
            group: {
                "token_count": count,
                "error_count": group_errors[group],
                "token_accuracy": (count - group_errors[group]) / count,
            }
            for group, count in sorted(group_tokens.items())
        },
    }


def _planner_loss(
    model,
    *,
    captions: Sequence[Any],
    targets: Sequence[Any],
    contexts: Sequence[Mapping[str, torch.Tensor]],
    prefixes: Sequence[Any] | None,
    objective: dict[str, Any],
) -> tuple[torch.Tensor, float]:
    loss = model.forward_planner(
        captions,
        targets,
        loss_group_weights=objective.get("loss_group_weights"),
        context_modalities=contexts,
        prefix_plan_tokens=prefixes,
    )
    weight = sum(
        _plan_weight(model, target, objective.get("loss_group_weights"))
        for target in targets
    )
    return loss, weight


def _teacher_evaluation(
    *,
    model,
    model_config: dict[str, Any],
    families: Sequence[dict[str, Any]],
    turns: int,
    shift: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    objectives = _objective_configs(model_config)
    totals: dict[str, list[float]] = {}
    diagnostics: dict[str, list[dict[str, Any]]] = {
        "state_planner": [],
        "understanding": [],
    }
    records: list[dict[str, Any]] = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        for turn_index in range(turns):
            for start in range(0, len(families), batch_size):
                indices = list(range(start, min(start + batch_size, len(families))))
                targets = [families[index]["turns"][turn_index] for index in indices]
                donors = [
                    families[(index + shift) % len(families)]["turns"][turn_index]
                    for index in indices
                ]
                plans = [turn["spatial_plan_tokens"] for turn in targets]
                planner_contexts = [
                    {
                        "previous_foa": torch.as_tensor(
                            turn["previous_foa"], device=device, dtype=torch.float32
                        )
                    }
                    for turn in targets
                ]
                donor_contexts = [
                    {
                        "previous_foa": torch.as_tensor(
                            turn["previous_foa"], device=device, dtype=torch.float32
                        )
                    }
                    for turn in donors
                ]
                prefixes = [turn.get("previous_plan_tokens") for turn in targets]
                donor_prefixes = [turn.get("previous_plan_tokens") for turn in donors]

                aligned, weight = _planner_loss(
                    model,
                    captions=[turn["planner_prompt"] for turn in targets],
                    targets=plans,
                    contexts=planner_contexts,
                    prefixes=prefixes,
                    objective=objectives["state_planner"],
                )
                _accumulate_loss(totals, "planner.aligned", aligned, weight)
                if turn_index > 0:
                    _accumulate_loss(
                        totals, "planner.state_reference", aligned, weight
                    )
                instruction, _ = _planner_loss(
                    model,
                    captions=[turn["planner_prompt"] for turn in donors],
                    targets=plans,
                    contexts=planner_contexts,
                    prefixes=prefixes,
                    objective=objectives["state_planner"],
                )
                _accumulate_loss(totals, "planner.instruction", instruction, weight)
                full, _ = _planner_loss(
                    model,
                    captions=[turn["planner_prompt"] for turn in donors],
                    targets=plans,
                    contexts=donor_contexts,
                    prefixes=donor_prefixes,
                    objective=objectives["state_planner"],
                )
                _accumulate_loss(totals, "planner.full", full, weight)
                state_value = None
                if turn_index > 0:
                    state, _ = _planner_loss(
                        model,
                        captions=[turn["planner_prompt"] for turn in targets],
                        targets=plans,
                        contexts=donor_contexts,
                        prefixes=donor_prefixes,
                        objective=objectives["state_planner"],
                    )
                    _accumulate_loss(totals, "planner.state", state, weight)
                    state_value = float(state.detach().float().cpu())

                understanding_contexts = [
                    {
                        "foa_latent": torch.as_tensor(
                            families[index]["latents"][turn_index],
                            device=device,
                            dtype=torch.float32,
                        )
                    }
                    for index in indices
                ]
                donor_understanding_contexts = [
                    {
                        "foa_latent": torch.as_tensor(
                            families[(index + shift) % len(families)]["latents"][
                                turn_index
                            ],
                            device=device,
                            dtype=torch.float32,
                        )
                    }
                    for index in indices
                ]
                understanding, understanding_weight = _planner_loss(
                    model,
                    captions=[turn["understanding_prompt"] for turn in targets],
                    targets=plans,
                    contexts=understanding_contexts,
                    prefixes=None,
                    objective=objectives["understanding"],
                )
                _accumulate_loss(
                    totals,
                    "understanding.aligned",
                    understanding,
                    understanding_weight,
                )
                audio_cf, _ = _planner_loss(
                    model,
                    captions=[turn["understanding_prompt"] for turn in targets],
                    targets=plans,
                    contexts=donor_understanding_contexts,
                    prefixes=None,
                    objective=objectives["understanding"],
                )
                _accumulate_loss(
                    totals, "understanding.audio", audio_cf, understanding_weight
                )

                records.append(
                    {
                        "turn": turn_index,
                        "family_ranks": [families[index]["family_rank"] for index in indices],
                        "state_planner_ce": {
                            "aligned": float(aligned.detach().float().cpu()),
                            "instruction_counterfactual": float(
                                instruction.detach().float().cpu()
                            ),
                            "state_counterfactual": state_value,
                            "full_counterfactual": float(full.detach().float().cpu()),
                        },
                        "understanding_ce": {
                            "aligned": float(understanding.detach().float().cpu()),
                            "audio_counterfactual": float(
                                audio_cf.detach().float().cpu()
                            ),
                        },
                    }
                )

                for local_index, (index, target) in enumerate(zip(indices, targets)):
                    planner_diag = _compact_diagnostic(
                        _teacher_forced_plan_diagnostic(
                            model,
                            caption=target["planner_prompt"],
                            plan_tokens=target["spatial_plan_tokens"],
                            context_modalities=planner_contexts[local_index],
                            prefix_plan_tokens=prefixes[local_index],
                        ),
                        target["spatial_plan_tokens"],
                    )
                    understanding_diag = _compact_diagnostic(
                        _teacher_forced_plan_diagnostic(
                            model,
                            caption=target["understanding_prompt"],
                            plan_tokens=target["spatial_plan_tokens"],
                            context_modalities=understanding_contexts[local_index],
                        ),
                        target["spatial_plan_tokens"],
                    )
                    identity = {
                        "family_rank": families[index]["family_rank"],
                        "family_id": families[index]["family_id"],
                        "turn": turn_index,
                    }
                    planner_diag.update(identity)
                    understanding_diag.update(identity)
                    diagnostics["state_planner"].append(planner_diag)
                    diagnostics["understanding"].append(understanding_diag)

    planner = _aggregate_diagnostics(diagnostics["state_planner"])
    understanding = _aggregate_diagnostics(diagnostics["understanding"])
    planner_aligned = _mean_loss(totals, "planner.aligned")
    planner_state_reference = _mean_loss(totals, "planner.state_reference")
    understanding_aligned = _mean_loss(totals, "understanding.aligned")
    planner.update(
        {
            "aligned_ce": planner_aligned,
            "counterfactual": {
                "instruction": {
                    "ce": _mean_loss(totals, "planner.instruction"),
                    "aligned_reference_ce": planner_aligned,
                    "ce_gap": _mean_loss(totals, "planner.instruction")
                    - planner_aligned,
                },
                "state": {
                    "ce": _mean_loss(totals, "planner.state"),
                    "aligned_reference_ce": planner_state_reference,
                    "ce_gap": _mean_loss(totals, "planner.state")
                    - planner_state_reference,
                },
                "full": {
                    "ce": _mean_loss(totals, "planner.full"),
                    "aligned_reference_ce": planner_aligned,
                    "ce_gap": _mean_loss(totals, "planner.full") - planner_aligned,
                },
            },
        }
    )
    understanding.update(
        {
            "aligned_ce": understanding_aligned,
            "counterfactual": {
                "audio": {
                    "ce": _mean_loss(totals, "understanding.audio"),
                    "aligned_reference_ce": understanding_aligned,
                    "ce_gap": _mean_loss(totals, "understanding.audio")
                    - understanding_aligned,
                }
            },
        }
    )
    return {
        "state_planner": planner,
        "understanding": understanding,
    }, records


def _sequence_metrics(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    predicted = predicted.detach().long().cpu().flatten()
    target = target.detach().long().cpu().flatten()
    denominator = max(int(predicted.numel()), int(target.numel()))
    overlap = min(int(predicted.numel()), int(target.numel()))
    matches = int(predicted[:overlap].eq(target[:overlap]).sum())
    return {
        "predicted_tokens": int(predicted.numel()),
        "target_tokens": int(target.numel()),
        "token_matches": matches,
        "token_denominator": denominator,
        "token_accuracy": matches / denominator,
        "token_exact": bool(torch.equal(predicted, target)),
    }


def _free_record(
    *,
    model,
    family: dict[str, Any],
    turn_index: int,
    objective: str,
    max_plan_tokens: int,
    device: torch.device,
) -> dict[str, Any]:
    turn = family["turns"][turn_index]
    target_tokens = torch.as_tensor(
        turn["spatial_plan_tokens"]["input_ids"], dtype=torch.long
    )
    codec = model.get_plan_codec()
    target_plan = codec.decode(target_tokens)
    base = {
        "family_rank": family["family_rank"],
        "family_id": family["family_id"],
        "turn": turn_index,
        "objective": objective,
    }
    try:
        if objective == "state_planner":
            predicted = model.generate_plan(
                turn["planner_prompt"],
                temperature=0.0,
                max_plan_tokens=max_plan_tokens,
                context_modalities={
                    "previous_foa": torch.as_tensor(
                        turn["previous_foa"], device=device, dtype=torch.float32
                    )
                },
                prefix_plan_tokens=turn.get("previous_plan_tokens"),
            )
        elif objective == "understanding":
            predicted = model.generate_plan(
                turn["understanding_prompt"],
                temperature=0.0,
                max_plan_tokens=max_plan_tokens,
                context_modalities={
                    "foa_latent": torch.as_tensor(
                        family["latents"][turn_index],
                        device=device,
                        dtype=torch.float32,
                    )
                },
            )
        else:
            raise ValueError(f"unknown free-decode objective: {objective}")
        predicted_plan = codec.decode(predicted)
        edit_metrics = _edit_field_metrics(
            predicted_plan,
            target_plan,
            turn.get("previous_scene_plan"),
            turn.get("diff"),
        )
        return {
            **base,
            "status": "PASS",
            **_sequence_metrics(predicted, target_tokens),
            "field_metrics": _plan_field_metrics(predicted_plan, target_plan),
            "edit_metrics": edit_metrics,
        }
    except Exception as error:  # Preserve all cases; comparison remains fail-closed.
        return {
            **base,
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def _aggregate_free(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successes = [item for item in records if item["status"] == "PASS"]
    token_matches = sum(int(item["token_matches"]) for item in successes)
    token_denominator = sum(int(item["token_denominator"]) for item in successes)
    component_matches = sum(
        int(item["field_metrics"]["source_component_matches"]) for item in successes
    )
    component_count = sum(
        int(item["field_metrics"]["source_component_count"]) for item in successes
    )
    total = len(records)
    edit_records = [
        item
        for item in successes
        if bool((item.get("edit_metrics") or {}).get("applicable"))
    ]
    edit_field_matches = sum(
        int(item["edit_metrics"]["changed_field_matches"])
        for item in edit_records
    )
    edit_field_count = sum(
        int(item["edit_metrics"]["changed_field_count"])
        for item in edit_records
    )
    return {
        "example_count": total,
        "success_count": len(successes),
        "failure_count": total - len(successes),
        "token_matches": token_matches,
        "token_denominator": token_denominator,
        "token_accuracy": token_matches / token_denominator if token_denominator else 0.0,
        "token_exact_fraction": sum(item["token_exact"] for item in successes) / total,
        "decoded_exact_fraction": sum(
            item["field_metrics"]["decoded_plan_exact"] for item in successes
        )
        / total,
        "source_component_matches": component_matches,
        "source_component_count": component_count,
        "source_component_accuracy": (
            component_matches / component_count if component_count else 0.0
        ),
        "edit_example_count": len(edit_records),
        "edit_field_matches": edit_field_matches,
        "edit_field_count": edit_field_count,
        "edit_field_accuracy": (
            edit_field_matches / edit_field_count if edit_field_count else None
        ),
        "edit_exact_fraction": (
            sum(
                bool(item["edit_metrics"]["changed_fields_exact"])
                for item in edit_records
            )
            / len(edit_records)
            if edit_records
            else None
        ),
    }


def _free_evaluation(
    *,
    model,
    families: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, int]],
    max_plan_tokens: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_rank = {item["family_rank"]: item for item in families}
    records = []
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        for case in cases:
            family = by_rank[int(case["family_rank"])]
            for objective in ("state_planner", "understanding"):
                records.append(
                    _free_record(
                        model=model,
                        family=family,
                        turn_index=int(case["turn"]),
                        objective=objective,
                        max_plan_tokens=max_plan_tokens,
                        device=device,
                    )
                )
    return {
        objective: _aggregate_free(
            [item for item in records if item["objective"] == objective]
        )
        for objective in ("state_planner", "understanding")
    }, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "online"), default="online")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--latent-root", type=Path)
    parser.add_argument("--codec-root", type=Path, default=DEFAULT_CODEC_ROOT)
    parser.add_argument(
        "--mode",
        choices=("teacher_forced_counterfactual", "free_decode"),
        default="teacher_forced_counterfactual",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--qwen-cache-batch-size", type=int, default=16)
    parser.add_argument("--max-plan-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    panel_preview, _ = load_text_panel(args.panel)
    latent_root = (
        args.latent_root
        if args.latent_root is not None
        else Path(panel_preview["latent_root"])
    ).expanduser().resolve()
    panel, ranks = load_text_panel(args.panel, latent_root=latent_root)
    for path in (
        args.checkpoint,
        args.model_config,
        args.panel,
        latent_root / "READY",
        args.codec_root / "READY",
    ):
        if not path.expanduser().resolve().is_file():
            raise FileNotFoundError(path)
    if (
        args.batch_size < 1
        or args.qwen_cache_batch_size < 1
        or args.max_plan_tokens < 1
    ):
        raise ValueError("batch sizes and max-plan-tokens must be positive")

    device = _canonical_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    _seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    dataset = SpatialFamilyDataset(
        [
            {
                "path": str(latent_root),
                "custom_metadata_fn": _provider(args.codec_root),
            }
        ],
        require_ready=True,
        max_open_shards=1,
    )
    families = _load_families(dataset, panel, ranks)
    model_config = load_config(args.model_config)
    model = create_model_from_config(model_config)
    checkpoint_state, checkpoint_metadata = load_ckpt_state_dict(
        str(args.checkpoint), return_metadata=True
    )
    warmstart = model.load_pretrained_route_state_dict(
        checkpoint_state,
        prefer_ema=args.weights == "ema",
        source_model_config=checkpoint_metadata.get("model_config"),
        source_text_conditioner_ema_names=checkpoint_metadata.get(
            "text_conditioner_ema_parameter_names"
        ),
    )
    del checkpoint_state, checkpoint_metadata
    model = model.eval().requires_grad_(False).to(device)

    if args.mode == "teacher_forced_counterfactual":
        cached_prompts = [
            str(turn[key])
            for family in families
            for turn in family["turns"]
            for key in ("planner_prompt", "understanding_prompt")
        ]
    else:
        families_by_rank = {family["family_rank"]: family for family in families}
        cached_prompts = [
            str(
                families_by_rank[int(case["family_rank"])]["turns"][
                    int(case["turn"])
                ][key]
            )
            for case in panel["free_decode_cases"]
            for key in ("planner_prompt", "understanding_prompt")
        ]
    qwen_cache_entries = _install_qwen_prefix_cache(
        model,
        cached_prompts,
        device=device,
        batch_size=args.qwen_cache_batch_size,
    )

    if args.mode == "teacher_forced_counterfactual":
        metrics, records = _teacher_evaluation(
            model=model,
            model_config=model_config,
            families=families,
            turns=int(panel["turns"]),
            shift=int(panel["counterfactual_shift"]),
            batch_size=args.batch_size,
            device=device,
        )
        case_count = len(families) * int(panel["turns"])
    else:
        metrics, records = _free_evaluation(
            model=model,
            families=families,
            cases=panel["free_decode_cases"],
            max_plan_tokens=args.max_plan_tokens,
            device=device,
        )
        case_count = len(panel["free_decode_cases"]) * 2

    report = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "DIAGNOSTIC",
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "weights": args.weights,
        "model_config": str(args.model_config.expanduser().resolve()),
        "latent_root": str(latent_root),
        "panel": str(args.panel.expanduser().resolve()),
        "panel_name": panel["name"],
        "panel_sha256": file_sha256(args.panel.expanduser().resolve()),
        "family_ranks": list(ranks),
        "case_count": case_count,
        "settings": {
            "turns": int(panel["turns"]),
            "counterfactual_shift": int(panel["counterfactual_shift"]),
            "batch_size": args.batch_size,
            "qwen_cache_batch_size": args.qwen_cache_batch_size,
            "qwen_cache_entries": qwen_cache_entries,
            "seed": args.seed,
        },
        "warmstart": warmstart,
        "metrics": metrics,
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "RESULT.json"
    atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
