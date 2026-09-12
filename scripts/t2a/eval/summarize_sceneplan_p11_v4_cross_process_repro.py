#!/usr/bin/env python3
"""Fail-closed cross-process reproducibility summary for P11 challenge evals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_SCHEMA = "stable_audio_tools.p11_v4_unified_challenge_eval"
EXPECTED_SCHEMA_VERSION = 10
CANONICAL_QWEN_KERNEL_MODE = "torch_reference"


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


def _row_map(report: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(row["ordinal"]): row for row in report["aggregate_rows"]}


def _scientific_draw(draw: Mapping[str, Any]) -> dict[str, Any]:
    """Retain every prediction/scoring field; omit no stochastic evidence."""

    return dict(draw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [
        args.first.expanduser().resolve(strict=True),
        args.second.expanduser().resolve(strict=True),
    ]
    first, second = [
        json.loads(path.read_text(encoding="utf-8")) for path in paths
    ]
    if any(
        report.get("schema") != EXPECTED_SCHEMA
        or int(report.get("schema_version", -1)) != EXPECTED_SCHEMA_VERSION
        for report in (first, second)
    ):
        raise RuntimeError("cross-process replay requires evaluator-v10 reports")
    comparable_fields = (
        "schema",
        "schema_version",
        "evaluator_contract",
        "arm",
        "challenge_sha256",
        "selected_ordinals",
        "root_seed",
        "draws",
        "k_values",
        "d0_temperature",
        "discrete_decode_mode",
        "qwen_kernel_mode",
        "scientific_kernel_contract",
        "weights",
        "families_filter",
        "rows_per_view",
        "lexical_authority_intervention",
        "config_provenance",
        "lexical_evidence_provenance",
    )
    comparability = {
        "reports_pass": first.get("status") == second.get("status") == "PASS",
        **{
            field: first.get(field) == second.get(field)
            for field in comparable_fields
        },
        "checkpoint_sha256": (
            first["checkpoints"][0]["checkpoint_sha256"]
            == second["checkpoints"][0]["checkpoint_sha256"]
        ),
        "runtime_source_files": (
            first["runtime_source_provenance"]["source_files"]
            == second["runtime_source_provenance"]["source_files"]
        ),
        "runtime_determinism": (
            first["runtime_source_provenance"]["determinism"]
            == second["runtime_source_provenance"]["determinism"]
        ),
        "kernel_config": (
            first["checkpoints"][0]["qwen_runtime_kernels"]
            == second["checkpoints"][0]["qwen_runtime_kernels"]
        ),
    }
    if not all(comparability.values()):
        failed = [key for key, value in comparability.items() if not value]
        raise RuntimeError(f"cross-process reports are incomparable: {failed}")

    left_rows = _row_map(first)
    right_rows = _row_map(second)
    row_results = []
    for ordinal in first["selected_ordinals"]:
        left = left_rows[int(ordinal)]
        right = right_rows[int(ordinal)]
        identity_equal = all(
            left.get(key) == right.get(key)
            for key in (
                "challenge_id",
                "task",
                "family",
                "view_id",
                "pair_id",
                "pair_label",
            )
        )
        scored_equal = [
            _scientific_draw(value) for value in left["scored"]
        ] == [_scientific_draw(value) for value in right["scored"]]
        prefixes_equal = left["prefixes"] == right["prefixes"]
        rng_equal = left["rng_isolation_pass"] == right["rng_isolation_pass"]
        row_results.append(
            {
                "ordinal": int(ordinal),
                "challenge_id": left["challenge_id"],
                "identity_exact": identity_equal,
                "scored_draws_exact": scored_equal,
                "prefixes_exact": prefixes_equal,
                "rng_audit_exact": rng_equal,
                "pass": identity_equal
                and scored_equal
                and prefixes_equal
                and rng_equal,
            }
        )

    gates = {
        "all_rows_exact": all(row["pass"] for row in row_results),
        "aggregate_exact": first["aggregate"] == second["aggregate"],
        "editing_counterfactual_exact": (
            first["editing_counterfactual"]
            == second["editing_counterfactual"]
        ),
        "rng_isolation_rate_exact": (
            first["rng_isolation_rate"] == second["rng_isolation_rate"]
        ),
        "deterministic_torch_reference_kernel": (
            first["qwen_kernel_mode"] == CANONICAL_QWEN_KERNEL_MODE
            and first["scientific_kernel_contract"]["scientific_report"]
            is True
            and first["checkpoints"][0]["qwen_runtime_kernels"][
                "fast_kernel_pin"
            ]
            is None
        ),
    }
    status = "PASS" if all(gates.values()) else "FAIL"
    report = {
        "schema": "stable_audio_tools.p11_v4_cross_process_repro",
        "schema_version": 2,
        "status": status,
        "decision": (
            "TORCH_REFERENCE_CROSS_PROCESS_EXACT_REPRO_ESTABLISHED"
            if status == "PASS"
            else "CROSS_PROCESS_REPRO_NOT_ESTABLISHED"
        ),
        "scope": (
            "same checkpoint/config/challenge/seeds in two fresh evaluator "
            "processes; exact equality required for every scored draw"
        ),
        "inputs": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in paths
        ],
        "comparability_gates": comparability,
        "reproducibility_gates": gates,
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
