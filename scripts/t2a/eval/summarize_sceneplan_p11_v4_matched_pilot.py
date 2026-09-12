#!/usr/bin/env python3
"""Summarize a matched P11-v4 unified-challenge pilot.

This is the single comparison surface for the active D0, Direct-MSE, and
Flow-R1 arms. It accepts only evaluator schema-v8 reports produced on the same
immutable challenge rows, one exact pinned FLA kernel and lexical-cache provenance, plus
training logs with the same batch/step budget. A small held-out smoke can
reject an arm, but never authorizes a full-corpus or multi-GPU run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


TASKS = ("generation", "understanding", "editing")
EXPECTED_ARMS = {
    "flow": "flow",
    "direct": "direct",
    "d0": "d0",
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def _last_log_object(path: Path, prefix: str) -> dict[str, Any]:
    values: list[dict[str, Any]] = []
    payload = path.resolve(strict=True).read_text(
        encoding="utf-8", errors="replace"
    ).replace("\r", "\n")
    for line in payload.splitlines():
        offset = line.find(prefix)
        if offset < 0:
            continue
        try:
            value = json.loads(line[offset + len(prefix) :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            values.append(value)
    if not values:
        raise RuntimeError(f"{path} contains no complete {prefix}")
    return values[-1]


def _training(path: Path) -> dict[str, Any]:
    benchmark = _last_log_object(path, "SAT_BENCHMARK_RESULT=")
    gate = _last_log_object(path, "SAT_TRAINING_GATE_RESULT=")
    step = int(benchmark["target_final_global_step"])
    if gate.get("status") != "PASS" or int(gate.get("global_step", -1)) != step:
        raise RuntimeError(f"{path} did not pass its final training gate")
    return {
        "gate": "PASS",
        "global_step": step,
        "batch_size_per_gpu": int(benchmark["batch_size_per_gpu"]),
        "world_size": int(benchmark["world_size"]),
        "samples_per_second": float(benchmark["global_samples_per_second"]),
        "seconds_per_step": float(benchmark["seconds_per_step"]),
        "peak_allocated_gib": float(benchmark["peak_allocated_gib"]),
        "optimizer_events": int(gate["optimizer_events"]),
        "ema_advances": dict(gate["ema_advances"]),
        "loss_window_ratios": {
            key: float(value["last_over_first"])
            for key, value in gate["metric_windows"].items()
        },
    }


def _prefix(report: Mapping[str, Any], group: str, k: int) -> Mapping[str, Any]:
    try:
        value = report["aggregate"][group][str(k)]
    except KeyError as exc:
        raise RuntimeError(f"report lacks aggregate {group!r} at K={k}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"aggregate {group!r} at K={k} is not an object")
    return value


def _evaluation(report: Mapping[str, Any], *, k: int) -> dict[str, Any]:
    tasks = {
        task: {
            "rows": int(_prefix(report, f"task:{task}", k)["rows"]),
            "task_score": float(
                _prefix(report, f"task:{task}", k)["task_score_mean"]
            ),
            "valid_rate": float(_prefix(report, f"task:{task}", k)["valid_rate"]),
        }
        for task in TASKS
    }
    all_metrics = _prefix(report, "all", k)
    counterfactual = report.get("editing_counterfactual", {}).get(str(k), {})
    task_macro = sum(value["task_score"] for value in tasks.values()) / len(tasks)
    return {
        "k": k,
        "rows": int(all_metrics["rows"]),
        "row_weighted_task_score": float(all_metrics["task_score_mean"]),
        "gue_macro_score": float(task_macro),
        "valid_rate": float(all_metrics["valid_rate"]),
        "semantic_immutability_rate": float(
            all_metrics["semantic_immutability_rate"]
        ),
        "editing_counterfactual_both_correct_rate": counterfactual.get(
            "both_counterfactual_targets_correct_rate"
        ),
        "tasks": tasks,
        "performance": list(report["performance"]),
    }


def _arm(
    name: str,
    report_path: Path,
    log_path: Path,
    *,
    k: int,
) -> dict[str, Any]:
    report = _load(report_path)
    if (
        report.get("schema") != "stable_audio_tools.p11_v4_unified_challenge_eval"
        or int(report.get("schema_version", -1)) != 9
        or report.get("status") != "PASS"
    ):
        raise RuntimeError(f"{name} is not a passing unified challenge v9 report")
    if report.get("arm") != EXPECTED_ARMS[name]:
        raise RuntimeError(
            f"{name} report declares arm={report.get('arm')!r}, "
            f"expected {EXPECTED_ARMS[name]!r}"
        )
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 1:
        raise RuntimeError("matched K=1 pilot requires one checkpoint per arm")
    checkpoint = checkpoints[0]
    if checkpoint.get("checkpoint_model_config_matches_runtime") is not True:
        raise RuntimeError(
            f"{name} checkpoint lacks resolved-config identity proof"
        )
    training = _training(log_path)
    if int(checkpoint["global_step"]) != training["global_step"]:
        raise RuntimeError(f"{name} report/log checkpoint steps disagree")
    return {
        "architecture": report["architecture"],
        "checkpoint": checkpoint,
        "evaluation": _evaluation(report, k=k),
        "training": training,
        "report": str(report_path.resolve()),
        "train_log": str(log_path.resolve()),
    }


def _winner(arms: Mapping[str, Mapping[str, Any]], path: tuple[str, ...]) -> list[str]:
    values: dict[str, float] = {}
    for name, arm in arms.items():
        value: Any = arm
        for key in path:
            value = value[key]
        values[name] = float(value)
    best = max(values.values())
    return [name for name, value in values.items() if abs(value - best) <= 1.0e-12]


def _causal_evidence(
    path: Path,
    *,
    flow_report: Mapping[str, Any],
) -> dict[str, Any]:
    report = _load(path)
    if (
        report.get("schema")
        != "stable_audio_tools.p11_v4_causal_intervention_eval"
        or int(report.get("schema_version", -1)) != 2
        or report.get("status") != "PASS"
        or report.get("arm") != "flow"
    ):
        raise RuntimeError("Flow causal evidence is not a passing v2 report")
    flow_checkpoint = flow_report["checkpoints"][0]
    if report.get("checkpoint", {}).get("checkpoint_sha256") != flow_checkpoint.get(
        "checkpoint_sha256"
    ):
        raise RuntimeError("Flow causal report used a different checkpoint")
    if report.get("challenge_sha256") != flow_report.get("challenge_sha256"):
        raise RuntimeError("Flow causal report used a different immutable challenge")
    if report.get("weights") != flow_report.get("weights"):
        raise RuntimeError("Flow causal report used different checkpoint weights")
    if report.get("root_seed") != flow_report.get("root_seed"):
        raise RuntimeError("Flow causal report used a different evaluator root seed")
    if report.get("discrete_decode_mode") != flow_report.get(
        "discrete_decode_mode"
    ):
        raise RuntimeError("Flow causal report used a different decode mode")
    if (
        report.get("config_provenance", {}).get("dataset", {}).get(
            "resolved_sha256"
        )
        != flow_report.get("config_provenance", {}).get("dataset", {}).get(
            "resolved_sha256"
        )
    ):
        raise RuntimeError("Flow causal report used a different dataset contract")
    if (
        report.get("config_provenance", {}).get("model", {}).get(
            "resolved_sha256"
        )
        != flow_report.get("config_provenance", {}).get("model", {}).get(
            "resolved_sha256"
        )
    ):
        raise RuntimeError("Flow causal report used a different model contract")
    causal_runtime = json.loads(json.dumps(report["runtime_source_provenance"]))
    causal_evaluator = causal_runtime["source_files"].pop(
        "causal_intervention_evaluator", None
    )
    if causal_evaluator is None:
        raise RuntimeError("Flow causal report lacks its evaluator source hash")
    if causal_runtime != flow_report.get("runtime_source_provenance"):
        raise RuntimeError(
            "Flow causal and quality reports used different model/kernel source"
        )
    selected = {int(value) for value in report.get("selected_ordinals", [])}
    quality_selected = {
        int(value) for value in flow_report.get("selected_ordinals", [])
    }
    if not selected or not selected.issubset(quality_selected):
        raise RuntimeError(
            "Flow causal rows must be a non-empty subset of quality rows"
        )
    gates = report.get("gates") or {}
    required_gates = (
        "evaluator_integrity",
        "discrete_and_semantic_authority_isolation",
        "editing_operation_matches_expected",
        "required_numeric_causal_response",
    )
    if not all(gates.get(key) is True for key in required_gates):
        raise RuntimeError("Flow causal report did not pass every required gate")
    return {
        "report": str(path.resolve()),
        "report_sha256_without_self": report.get("report_sha256_without_self"),
        "rows": int(report["rows"]),
        "selected_ordinals": sorted(selected),
        "intervention_decodes": int(report["intervention_decodes"]),
        "gates": {key: bool(gates[key]) for key in required_gates},
        "aggregate": report["aggregate"],
        "mechanistic_dependency_established_on_panel": bool(
            report.get("claim_boundary", {}).get(
                "mechanistic_dependency_established_on_panel"
            )
        ),
        "quality_superiority_established": False,
        "frozen_p10_audio_closure_established": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for arm in EXPECTED_ARMS:
        parser.add_argument(f"--{arm}-report", type=Path, required=True)
        parser.add_argument(f"--{arm}-log", type=Path, required=True)
    parser.add_argument("--flow-intervention-report", type=Path)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--training-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.k <= 0:
        raise ValueError("--k must be positive")

    reports = {name: _load(getattr(args, f"{name}_report")) for name in EXPECTED_ARMS}
    invariants = (
        "challenge_sha256",
        "dataset_config",
        "selected_ordinals",
        "rows",
        "rows_per_view",
        "root_seed",
        "weights",
        "evaluator_contract",
        "discrete_decode_mode",
        "rng_contract",
        "qwen_kernel_mode",
        "lexical_evidence_provenance",
        "lexical_authority_intervention",
    )
    reference = reports["flow"]
    for name, value in reports.items():
        for key in invariants:
            if value.get(key) != reference.get(key):
                raise RuntimeError(f"matched report invariant {key!r} differs for {name}")
        if (
            value.get("config_provenance", {}).get("dataset", {}).get(
                "resolved_sha256"
            )
            != reference.get("config_provenance", {}).get("dataset", {}).get(
                "resolved_sha256"
            )
        ):
            raise RuntimeError(
                f"matched resolved dataset config differs for {name}"
            )
        if value.get("runtime_source_provenance") != reference.get(
            "runtime_source_provenance"
        ):
            raise RuntimeError(
                f"matched evaluator/model/runtime source provenance differs for {name}"
            )

    arms = {
        name: _arm(
            name,
            getattr(args, f"{name}_report"),
            getattr(args, f"{name}_log"),
            k=args.k,
        )
        for name in EXPECTED_ARMS
    }
    recipes = {
        (
            arm["training"]["global_step"],
            arm["training"]["batch_size_per_gpu"],
            arm["training"]["world_size"],
        )
        for arm in arms.values()
    }
    if len(recipes) != 1:
        raise RuntimeError(f"training recipes are not matched: {sorted(recipes)}")

    flow = arms["flow"]["evaluation"]
    causal_evidence = (
        None
        if args.flow_intervention_report is None
        else _causal_evidence(
            args.flow_intervention_report,
            flow_report=reports["flow"],
        )
    )
    baseline_names = ("direct", "d0")
    best_baseline_macro = max(
        arms[name]["evaluation"]["gue_macro_score"] for name in baseline_names
    )
    flow_margin = float(flow["gue_macro_score"] - best_baseline_macro)
    flow_macro_margin_vs_direct = float(
        flow["gue_macro_score"]
        - arms["direct"]["evaluation"]["gue_macro_score"]
    )
    flow_macro_margin_vs_d0 = float(
        flow["gue_macro_score"] - arms["d0"]["evaluation"]["gue_macro_score"]
    )
    flow_weighted_margin_vs_direct = float(
        flow["row_weighted_task_score"]
        - arms["direct"]["evaluation"]["row_weighted_task_score"]
    )
    flow_weighted_margin_vs_d0 = float(
        flow["row_weighted_task_score"]
        - arms["d0"]["evaluation"]["row_weighted_task_score"]
    )
    task_winners = {
        task: _winner(arms, ("evaluation", "tasks", task, "task_score"))
        for task in TASKS
    }
    matched_step = next(iter(recipes))[0]
    screening = matched_step == 10_000
    report: dict[str, Any] = {
        "schema": "stable_audio_tools.p11_v4_matched_challenge_pilot",
        "schema_version": 5,
        "status": "PASS",
        "scope": (
            "matched 10k-step one-seed idea screen; no frozen-P10 FOA render "
            "and no full-corpus or multi-GPU authorization"
            if screening
            else "matched held-out pilot; no frozen-P10 FOA render and no "
            "full-corpus or multi-GPU authorization"
        ),
        "matched_evaluation": {
            key: reference[key] for key in invariants
        },
        "matched_training_recipe": {
            "global_step": matched_step,
            "batch_size_per_gpu": next(iter(recipes))[1],
            "world_size": next(iter(recipes))[2],
            "seed": int(args.training_seed),
        },
        "arms": arms,
        "causal_intervention_evidence": causal_evidence,
        "winners": {
            "gue_macro_score": _winner(arms, ("evaluation", "gue_macro_score")),
            "row_weighted_task_score": _winner(
                arms, ("evaluation", "row_weighted_task_score")
            ),
            "tasks": task_winners,
        },
        "flow_evidence": {
            "gue_macro_margin_vs_best_baseline": flow_margin,
            "gue_macro_margin_vs_direct": flow_macro_margin_vs_direct,
            "gue_macro_margin_vs_d0": flow_macro_margin_vs_d0,
            "row_weighted_margin_vs_direct": flow_weighted_margin_vs_direct,
            "row_weighted_margin_vs_d0": flow_weighted_margin_vs_d0,
            "understanding_margin_vs_best_baseline": float(
                flow["tasks"]["understanding"]["task_score"]
                - max(
                    arms[name]["evaluation"]["tasks"]["understanding"][
                        "task_score"
                    ]
                    for name in baseline_names
                )
            ),
            "point_estimate_superiority_observed": flow_margin > 0.0,
            "point_estimate_improves_over_direct": (
                flow_macro_margin_vs_direct > 0.0
            ),
            "mechanistic_dependency_established_on_panel": bool(
                causal_evidence
                and causal_evidence[
                    "mechanistic_dependency_established_on_panel"
                ]
            ),
            "overall_superiority_established": False,
            "establishment_blocker": (
                f"one training seed and {int(reference['rows'])} selected rows "
                "do not establish statistical superiority"
            ),
        },
        "decision": {
            "full_or_eight_gpu_training": "HOLD",
            "idea_status": (
                "mechanistic dependency and best matched quality are both positive; proceed to unique-capability and frozen-P10 closure gates"
                if flow_margin > 0.0 and causal_evidence is not None
                else (
                    "mechanistic Transfusion-CoT dependency is established and Flow-R1 improves over Direct-MSE, but it does not beat D0 at this pilot budget"
                    if causal_evidence is not None
                    and flow_macro_margin_vs_direct > 0.0
                    else "continuous posterior signal may exist, but overall superiority is not established"
                )
            ),
            "next_gate": (
                "keep the canonical seed fixed at 42, run unique-capability and frozen-P10 closure gates, "
                "and keep full-corpus/eight-GPU scale-up locked until they pass"
            ),
        },
    }
    canonical = json.dumps(
        report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    report["report_sha256_without_self"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
