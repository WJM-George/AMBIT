#!/usr/bin/env python3
"""Compare a P10-v12 candidate with Dense-150k on identical rows and noise.

The P10-v12 evaluation builder records the immutable Dense baseline but the
generic P10 summarizer only compares checkpoints inside one evaluation root.
This tool closes that gap by joining per-output metrics on panel/sample IDs and
emitting deterministic bootstrap confidence intervals for candidate - Dense.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (  # noqa: E402
    atomic_json,
    read_jsonl,
    sha256_file,
    summarize,
)


Getter = Callable[[dict[str, Any]], float | int | None]


def _indexed_values(
    rows: list[dict[str, Any]],
    *,
    step: int,
    domain: str | None,
    getter: Getter,
) -> dict[str, tuple[str, float | None]]:
    output: dict[str, tuple[str, float | None]] = {}
    for row in rows:
        if int(row["checkpoint_step"]) != int(step):
            continue
        if domain is not None and str(row.get("domain")) != domain:
            continue
        panel_id = str(row["panel_id"])
        if panel_id in output:
            raise RuntimeError(f"duplicate metric row for {panel_id} at step {step}")
        raw_value = getter(row)
        value = None if raw_value is None else float(raw_value)
        if value is not None and not math.isfinite(value):
            raise RuntimeError(
                f"non-finite metric for {panel_id} at step {step}: {value}"
            )
        output[panel_id] = (str(row["sample_id"]), value)
    return output


def paired_delta(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    baseline_step: int,
    candidate_step: int,
    domain: str | None,
    getter: Getter,
) -> dict[str, Any]:
    baseline = _indexed_values(
        baseline_rows,
        step=baseline_step,
        domain=domain,
        getter=getter,
    )
    candidate = _indexed_values(
        candidate_rows,
        step=candidate_step,
        domain=domain,
        getter=getter,
    )
    if not baseline or set(baseline) != set(candidate):
        raise RuntimeError(
            "candidate and Dense metric panels differ: "
            f"baseline={len(baseline)} candidate={len(candidate)} domain={domain}"
        )
    sample_mismatch = [
        panel_id
        for panel_id in sorted(baseline)
        if baseline[panel_id][0] != candidate[panel_id][0]
    ]
    if sample_mismatch:
        raise RuntimeError(
            f"candidate and Dense sample IDs differ for {sample_mismatch[:5]}"
        )
    deltas = [
        None
        if baseline[panel_id][1] is None or candidate[panel_id][1] is None
        else candidate[panel_id][1] - baseline[panel_id][1]
        for panel_id in sorted(baseline)
    ]
    return summarize(deltas)


def _load_metrics(metrics_root: Path) -> dict[str, list[dict[str, Any]]]:
    files = {
        "core": metrics_root / "core_per_output.jsonl",
        "clap": metrics_root / "clap_per_output.jsonl",
        "speech": metrics_root / "speech_per_output.jsonl",
        "distributional": metrics_root / "distributional_per_output.jsonl",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing per-output metrics: {missing}")
    return {name: read_jsonl(path) for name, path in files.items()}


def _summary_row(path: Path, step: int) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        row for row in report["rows"] if int(row["checkpoint_step"]) == int(step)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one summary row for step {step} in {path}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-eval-root", type=Path, required=True)
    parser.add_argument("--candidate-step", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.candidate_step <= 0:
        parser.error("--candidate-step must be positive")

    root = args.candidate_eval_root.expanduser().resolve(strict=True)
    contract_path = root / "EVAL_CONTRACT.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not (
        contract.get("schema")
        == "stable_audio_tools.p10_v12_upgrade_eval_contract"
        and contract.get("status") == "FROZEN_BALANCED_1200"
        and contract.get("comparison_contract", {}).get("same_panel") is True
        and contract.get("comparison_contract", {}).get("same_noise_per_sample")
        is True
    ):
        raise RuntimeError("candidate is not a frozen matched P10-v12 evaluation")

    checkpoint_rows = [
        row
        for row in contract["checkpoints"]
        if int(row["step"]) == args.candidate_step
    ]
    if len(checkpoint_rows) != 1:
        raise RuntimeError("candidate step is absent or duplicated in the contract")

    baseline = contract["dense_150k_baseline"]
    baseline_step = int(baseline["step"])
    if baseline_step != 150_000:
        raise RuntimeError("contract baseline is not Dense-150k")
    baseline_metrics_root = Path(baseline["metric_root"]).resolve(strict=True)
    candidate_metrics_root = (root / "metrics").resolve(strict=True)
    baseline_metrics = _load_metrics(baseline_metrics_root)
    candidate_metrics = _load_metrics(candidate_metrics_root)

    specifications: list[tuple[str, str, str, str | None, Getter]] = [
        (
            "music_clap_text_cosine",
            "higher",
            "clap",
            "music",
            lambda row: row["generated_text_cosine"],
        ),
        (
            "sound_clap_text_cosine",
            "higher",
            "clap",
            "sound",
            lambda row: row["generated_text_cosine"],
        ),
        (
            "music_paired_kl_pann",
            "lower",
            "distributional",
            "music",
            lambda row: row["paired_kl_pann_softmax"],
        ),
        (
            "sound_paired_kl_pann",
            "lower",
            "distributional",
            "sound",
            lambda row: row["paired_kl_pann_softmax"],
        ),
        (
            "speech_wer",
            "lower",
            "speech",
            None,
            lambda row: row["generated_errors"]["wer"],
        ),
        (
            "speech_cer",
            "lower",
            "speech",
            None,
            lambda row: row["generated_errors"]["cer"],
        ),
        (
            "speech_utmos",
            "higher",
            "speech",
            None,
            lambda row: row["generated_utmos"],
        ),
    ]
    for domain in ("music", "sound", "speech"):
        specifications.extend(
            [
                (
                    f"{domain}_plan_doa_error_deg",
                    "lower",
                    "core",
                    domain,
                    lambda row: row["generated_doa"][
                        "spherical_error_mean_deg"
                    ],
                ),
                (
                    f"{domain}_generated_reference_doa_error_deg",
                    "lower",
                    "core",
                    domain,
                    lambda row: row["generated_reference_doa"][
                        "spherical_error_mean_deg"
                    ],
                ),
                (
                    f"{domain}_activity_iou",
                    "higher",
                    "core",
                    domain,
                    lambda row: row["generated_activity"]["temporal_iou"],
                ),
            ]
        )

    comparisons = {}
    for name, direction, report_name, domain, getter in specifications:
        comparisons[name] = {
            "better_direction": direction,
            "delta": paired_delta(
                baseline_metrics[report_name],
                candidate_metrics[report_name],
                baseline_step=baseline_step,
                candidate_step=args.candidate_step,
                domain=domain,
                getter=getter,
            ),
        }

    baseline_summary_path = baseline_metrics_root / "EVALUATION_SUMMARY.json"
    candidate_summary_path = candidate_metrics_root / "EVALUATION_SUMMARY.json"
    baseline_summary = _summary_row(baseline_summary_path, baseline_step)
    candidate_summary = _summary_row(candidate_summary_path, args.candidate_step)
    aggregate_deltas = {
        name: float(candidate_summary[name]) - float(baseline_summary[name])
        for name in (
            "music_fad_vggish_diagnostic",
            "sound_fad_vggish_diagnostic",
            "music_fd_pann",
            "sound_fd_pann",
            "speech_corpus_wer",
            "speech_corpus_cer",
        )
    }

    output = {
        "schema": "stable_audio_tools.p10_v12_dense150k_paired_comparison",
        "schema_version": 1,
        "status": "PASS",
        "delta_definition": "candidate minus Dense-150k on identical panel IDs and noise seeds",
        "upgrade_arm": contract["upgrade_arm"],
        "panel_sha256": contract["test_set"]["panel_sha256"],
        "candidate_step": args.candidate_step,
        "candidate_checkpoint": checkpoint_rows[0],
        "candidate_eval_root": str(root),
        "candidate_eval_contract_sha256": sha256_file(contract_path),
        "baseline_step": baseline_step,
        "baseline_checkpoint_sha256": baseline["sha256"],
        "baseline_eval_root": baseline["eval_root"],
        "paired_metrics": comparisons,
        "aggregate_candidate_minus_baseline": aggregate_deltas,
        "aggregate_deltas_have_no_paired_ci": True,
    }
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else candidate_metrics_root / f"DENSE150K_PAIRED_STEP_{args.candidate_step}.json"
    )
    atomic_json(output_path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
