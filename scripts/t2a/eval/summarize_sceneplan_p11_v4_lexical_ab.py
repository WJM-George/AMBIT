#!/usr/bin/env python3
"""Fail-closed summary for a same-GPU P11-v4 reliable-ASR causal A/B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


MIN_RELIABLE_ROWS = 20
MIN_RELIABLE_MEAN_DELTA = 0.01
MIN_RELIABLE_POSITIVE_RATE = 0.90


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _score(row: Mapping[str, Any]) -> Mapping[str, Any]:
    scored = row.get("scored") or []
    if len(scored) != 1:
        raise ValueError("lexical causal A/B requires exactly one draw per row")
    return scored[0]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _exact_output_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(key) == right.get(key)
        for key in (
            "plan_sha256",
            "semantic_sha256",
            "numeric_sha256",
            "task_score",
        )
    )


def _load_pairing_contract(
    path: Path,
    *,
    asr_on: Path,
    asr_drop: Path,
) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("lexical pairing contract is not a JSON object")
    claimed = value.get("report_sha256_without_self")
    unhashed = dict(value)
    unhashed.pop("report_sha256_without_self", None)
    if claimed != _json_sha256(unhashed):
        raise RuntimeError("lexical pairing contract self-hash mismatch")
    expected = {
        "asr_on": asr_on,
        "asr_drop": asr_drop,
    }
    outputs = value.get("outputs") or {}
    for name, report_path in expected.items():
        record = outputs.get(name) or {}
        if (
            Path(str(record.get("path"))).expanduser().resolve(strict=True)
            != report_path
            or record.get("sha256") != _sha256_file(report_path)
        ):
            raise RuntimeError(f"lexical pairing contract does not bind {name}")
    return resolved, value


def _causal_gate_values(
    *,
    on_valid_rate: float,
    drop_valid_rate: float,
    applied_deltas: Sequence[float],
    unapplied_exact: Sequence[bool],
    non_u_exact: Sequence[bool],
    drop_authority_applied: Sequence[bool],
    understanding_on_mean: float,
    understanding_drop_mean: float,
    arm: str,
    numeric_states_equal: Sequence[bool],
) -> dict[str, bool]:
    mean_delta = _mean(applied_deltas)
    positive_rate = (
        sum(value > 0.0 for value in applied_deltas) / len(applied_deltas)
        if applied_deltas
        else 0.0
    )
    return {
        "valid_rate_one": on_valid_rate == 1.0 and drop_valid_rate == 1.0,
        "enough_reliable_asr_rows": len(applied_deltas) >= MIN_RELIABLE_ROWS,
        "reliable_asr_mean_improves_materially": mean_delta is not None
        and mean_delta >= MIN_RELIABLE_MEAN_DELTA,
        "reliable_asr_positive_rate": positive_rate
        >= MIN_RELIABLE_POSITIVE_RATE,
        "drop_authority_never_applied": not any(drop_authority_applied),
        "unapplied_u_rows_exactly_unchanged": bool(unapplied_exact)
        and all(unapplied_exact),
        "generation_and_editing_exactly_unchanged": all(non_u_exact),
        "understanding_mean_improves": understanding_on_mean
        > understanding_drop_mean,
        "d0_post_assembly_numeric_state_unchanged": arm != "d0"
        or all(numeric_states_equal),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-on", type=Path, required=True)
    parser.add_argument("--asr-drop", type=Path, required=True)
    parser.add_argument("--pairing-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "asr_on": args.asr_on.expanduser().resolve(strict=True),
        "asr_drop": args.asr_drop.expanduser().resolve(strict=True),
    }
    reports = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    on = reports["asr_on"]
    drop = reports["asr_drop"]
    pairing_path, pairing = _load_pairing_contract(
        args.pairing_contract,
        asr_on=paths["asr_on"],
        asr_drop=paths["asr_drop"],
    )
    comparability = {
        "reports_pass": on.get("status") == drop.get("status") == "PASS",
        "schema": on.get("schema") == drop.get("schema"),
        "schema_version": on.get("schema_version") == drop.get("schema_version"),
        "arm": on.get("arm") == drop.get("arm"),
        "evaluator_contract": (
            on.get("evaluator_contract") == drop.get("evaluator_contract")
        ),
        "challenge_sha256": (
            on.get("challenge_sha256") == drop.get("challenge_sha256")
        ),
        "selected_ordinals": (
            on.get("selected_ordinals") == drop.get("selected_ordinals")
        ),
        "root_seed": on.get("root_seed") == drop.get("root_seed"),
        "decode_protocol": all(
            on.get(key) == drop.get(key)
            for key in (
                "draws",
                "k_values",
                "discrete_decode_mode",
                "qwen_kernel_mode",
            )
        ),
        "checkpoint_sha256": (
            on["checkpoints"][0]["checkpoint_sha256"]
            == drop["checkpoints"][0]["checkpoint_sha256"]
        ),
        "model_config": (
            on["config_provenance"]["model"]
            == drop["config_provenance"]["model"]
        ),
        "dataset_config": (
            on["config_provenance"]["dataset"]
            == drop["config_provenance"]["dataset"]
        ),
        "runtime_sources": (
            on["runtime_source_provenance"]["source_files"]
            == drop["runtime_source_provenance"]["source_files"]
        ),
        "determinism": (
            on["runtime_source_provenance"]["determinism"]
            == drop["runtime_source_provenance"]["determinism"]
        ),
        "kernel_pin": (
            on["checkpoints"][0]["qwen_runtime_kernels"]
            == drop["checkpoints"][0]["qwen_runtime_kernels"]
        ),
        "lexical_cache": (
            on["lexical_evidence_provenance"]
            == drop["lexical_evidence_provenance"]
        ),
        "intervention_modes": (
            on["lexical_authority_intervention"]["mode"] == "normal"
            and drop["lexical_authority_intervention"]["mode"]
            == "drop_reliable_asr"
            and not on["lexical_authority_intervention"][
                "input_lexical_removed_before_model"
            ]
            and drop["lexical_authority_intervention"][
                "input_lexical_removed_before_model"
            ]
        ),
        "target_access_forbidden": (
            on["lexical_evidence_provenance"].get("target_transcript_access")
            == "forbidden"
            and on["lexical_authority_intervention"].get(
                "target_transcript_access"
            )
            == "forbidden"
            and drop["lexical_authority_intervention"].get(
                "target_transcript_access"
            )
            == "forbidden"
        ),
        "same_physical_gpu_sequential": (
            pairing.get("schema")
            == "stable_audio_tools.p11_v4_lexical_pair_execution"
            and int(pairing.get("schema_version", -1)) == 1
            and pairing.get("status") == "PASS"
            and pairing.get("sequential") is True
            and pairing.get("same_physical_gpu") is True
            and bool((pairing.get("physical_gpu") or {}).get("uuid"))
        ),
    }
    if not all(comparability.values()):
        failed = [key for key, value in comparability.items() if not value]
        raise RuntimeError(f"lexical A/B reports are incomparable: {failed}")

    rows = {
        name: {int(row["ordinal"]): row for row in report["aggregate_rows"]}
        for name, report in reports.items()
    }
    row_results = []
    task_deltas: dict[str, list[float]] = {}
    for ordinal in on["selected_ordinals"]:
        on_row = rows["asr_on"][int(ordinal)]
        drop_row = rows["asr_drop"][int(ordinal)]
        if any(
            on_row.get(key) != drop_row.get(key)
            for key in ("challenge_id", "task", "family", "view_id")
        ):
            raise RuntimeError(f"row identity changed at ordinal {ordinal}")
        on_score = _score(on_row)
        drop_score = _score(drop_row)
        task = str(on_row["task"])
        delta = float(on_score["task_score"]) - float(drop_score["task_score"])
        task_deltas.setdefault(task, []).append(delta)
        diagnostics = dict(on_score.get("diagnostics") or {})
        drop_diagnostics = dict(drop_score.get("diagnostics") or {})
        applied = bool(diagnostics.get("lexical_authority_applied", False))
        drop_applied = bool(
            drop_diagnostics.get("lexical_authority_applied", False)
        )
        on_metrics = dict(on_score.get("task_metrics") or {})
        drop_metrics = dict(drop_score.get("task_metrics") or {})
        row_results.append(
            {
                "ordinal": int(ordinal),
                "challenge_id": on_row["challenge_id"],
                "task": task,
                "family": on_row["family"],
                "lexical_authority_applied": applied,
                "drop_lexical_authority_applied": drop_applied,
                "lexical_authority_action": diagnostics.get(
                    "lexical_authority_action"
                ),
                "task_score_asr_drop": float(drop_score["task_score"]),
                "task_score_asr_on": float(on_score["task_score"]),
                "task_score_delta": delta,
                "kind_accuracy_asr_drop": drop_metrics.get("kind_accuracy"),
                "kind_accuracy_asr_on": on_metrics.get("kind_accuracy"),
                "transcript_word_accuracy_asr_drop": drop_metrics.get(
                    "transcript_word_accuracy"
                ),
                "transcript_word_accuracy_asr_on": on_metrics.get(
                    "transcript_word_accuracy"
                ),
                "exact_output_equal": _exact_output_equal(on_score, drop_score),
                "numeric_state_equal": (
                    on_score.get("numeric_sha256")
                    == drop_score.get("numeric_sha256")
                ),
            }
        )

    applied_u = [
        row
        for row in row_results
        if row["task"] == "understanding"
        and row["lexical_authority_applied"]
    ]
    unapplied_u = [
        row
        for row in row_results
        if row["task"] == "understanding"
        and not row["lexical_authority_applied"]
    ]
    non_u = [row for row in row_results if row["task"] != "understanding"]
    reliable_positive_rate = (
        sum(float(row["task_score_delta"]) > 0.0 for row in applied_u)
        / len(applied_u)
        if applied_u
        else 0.0
    )
    reliable_mean_delta = _mean(
        [float(row["task_score_delta"]) for row in applied_u]
    )
    causal_gates = _causal_gate_values(
        on_valid_rate=float(on["aggregate"]["all"]["1"]["valid_rate"]),
        drop_valid_rate=float(drop["aggregate"]["all"]["1"]["valid_rate"]),
        applied_deltas=[float(row["task_score_delta"]) for row in applied_u],
        unapplied_exact=[bool(row["exact_output_equal"]) for row in unapplied_u],
        non_u_exact=[bool(row["exact_output_equal"]) for row in non_u],
        drop_authority_applied=[
            bool(row["drop_lexical_authority_applied"]) for row in row_results
        ],
        understanding_on_mean=float(
            on["aggregate"]["task:understanding"]["1"]["task_score_mean"]
        ),
        understanding_drop_mean=float(
            drop["aggregate"]["task:understanding"]["1"]["task_score_mean"]
        ),
        arm=str(on.get("arm")),
        numeric_states_equal=[
            bool(row["numeric_state_equal"]) for row in row_results
        ],
    )
    status = "PASS" if all(causal_gates.values()) else "FAIL"
    aggregate = {
        "overall": {
            "asr_drop": drop["aggregate"]["all"]["1"]["task_score_mean"],
            "asr_on": on["aggregate"]["all"]["1"]["task_score_mean"],
        },
        "tasks": {
            task: {
                "rows": len(values),
                "mean_delta": _mean(values),
                "positive_rows": sum(value > 0.0 for value in values),
                "unchanged_rows": sum(value == 0.0 for value in values),
                "negative_rows": sum(value < 0.0 for value in values),
            }
            for task, values in sorted(task_deltas.items())
        },
        "reliable_asr_understanding": {
            "rows": len(applied_u),
            "mean_delta": reliable_mean_delta,
            "positive_rows": sum(
                float(row["task_score_delta"]) > 0.0 for row in applied_u
            ),
            "positive_rate": reliable_positive_rate,
            "unchanged_rows": sum(
                float(row["task_score_delta"]) == 0.0 for row in applied_u
            ),
            "negative_rows": sum(
                float(row["task_score_delta"]) < 0.0 for row in applied_u
            ),
        },
        "no_reliable_asr_understanding": {
            "rows": len(unapplied_u),
            "mean_delta": _mean(
                [float(row["task_score_delta"]) for row in unapplied_u]
            ),
            "exactly_unchanged_rows": sum(
                bool(row["exact_output_equal"]) for row in unapplied_u
            ),
        },
    }
    aggregate["overall"]["delta"] = (
        aggregate["overall"]["asr_on"] - aggregate["overall"]["asr_drop"]
    )
    report = {
        "schema": "stable_audio_tools.p11_v4_reliable_asr_causal_ab",
        "schema_version": 2,
        "status": status,
        "decision": (
            "RELIABLE_ASR_LEXICAL_AUTHORITY_SUPPORTED_ON_MATCHED_CAUSAL_SMOKE"
            if status == "PASS"
            else "RELIABLE_ASR_LEXICAL_AUTHORITY_NOT_ESTABLISHED"
        ),
        "scope": (
            "same checkpoint and challenge, reliable-ASR on/drop intervention; "
            "pilot evidence only, not a full P11 promotion claim"
        ),
        "arm": on.get("arm"),
        "inputs": {
            name: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for name, path in paths.items()
        },
        "pairing_contract": {
            "path": str(pairing_path),
            "sha256": _sha256_file(pairing_path),
            "physical_gpu": pairing["physical_gpu"],
        },
        "thresholds_frozen_before_corrected_ab": {
            "minimum_reliable_rows": MIN_RELIABLE_ROWS,
            "minimum_reliable_mean_delta": MIN_RELIABLE_MEAN_DELTA,
            "minimum_reliable_positive_rate": MIN_RELIABLE_POSITIVE_RATE,
            "unapplied_rows_require_exact_output_equality": True,
        },
        "comparability_gates": comparability,
        "causal_gates": causal_gates,
        "aggregate": aggregate,
        "rows": row_results,
    }
    report["report_sha256_without_self"] = _json_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
