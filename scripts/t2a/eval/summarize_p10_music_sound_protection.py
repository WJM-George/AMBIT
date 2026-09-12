#!/usr/bin/env python3
"""Summarize and gate a fixed50 Music/Sound checkpoint-pair comparison."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v10_music_sound_protection_fixed50"
)
DOMAINS = ("music", "sound")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(value: dict[str, Any]) -> float:
    result = value.get("mean")
    if result is None:
        raise RuntimeError("required aggregate has no finite mean")
    return float(result)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _upper_limit(baseline: float, *, relative: float, absolute: float) -> float:
    return max(baseline * relative, baseline + absolute)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    contract = _read(root / "EVAL_CONTRACT.json")
    core = _read(root / "metrics/CORE_SUMMARY.json")["aggregates"]
    clap = _read(root / "metrics/CLAP_SUMMARY.json")["aggregates"]
    distributional = _read(root / "metrics/DISTRIBUTIONAL_SUMMARY.json")[
        "aggregates"
    ]
    steps = sorted(int(row["step"]) for row in contract["checkpoints"])
    if len(steps) != 2 or len(set(steps)) != 2:
        raise RuntimeError("Music/Sound protection requires exactly two checkpoints")
    baseline_step, candidate_step = steps

    rows: dict[str, Any] = {}
    checks: list[dict[str, Any]] = []
    for domain in DOMAINS:
        rows[domain] = {}
        for step in steps:
            key = str(step)
            rows[domain][key] = {
                "text_clap": _mean(clap[key][domain]["clap_text_audio_cosine"]),
                "paired_clap": _mean(
                    clap[key][domain]["paired_generated_reference_clap_cosine"]
                ),
                "fd_clap": float(clap[key][domain]["fd_clap"]),
                "fad_vggish": float(
                    distributional[key][domain]["fad_vggish_diagnostic"]
                ),
                "fd_pann": float(distributional[key][domain]["fd_pann"]),
                "kl_pann": _mean(
                    distributional[key][domain]["paired_kl_pann_softmax"]
                ),
                "plan_doa_deg": _mean(
                    core[key][domain]["plan_spherical_error_mean_deg"]
                ),
                "trajectory_extent_error_deg": _mean(
                    core[key][domain]["plan_trajectory_extent_error_deg"]
                ),
                "activity_iou": _mean(
                    core[key][domain]["activity_temporal_iou"]
                ),
                "raw_clipped_fraction_max": float(
                    core[key][domain]["raw_fraction_abs_ge_1"]["max"] or 0.0
                ),
            }
        baseline = rows[domain][str(baseline_step)]
        candidate = rows[domain][str(candidate_step)]
        deltas = {
            metric: candidate[metric] - baseline[metric]
            for metric in baseline
        }
        rows[domain][f"delta_{candidate_step}_minus_{baseline_step}"] = deltas

        domain_checks = {
            "text_clap_delta_ge_-0.015": deltas["text_clap"] >= -0.015,
            "paired_clap_delta_ge_-0.02": deltas["paired_clap"] >= -0.02,
            "activity_iou_delta_ge_-0.03": deltas["activity_iou"] >= -0.03,
            "plan_doa_regression_le_2deg": deltas["plan_doa_deg"] <= 2.0,
            "trajectory_extent_regression_le_2deg": (
                deltas["trajectory_extent_error_deg"] <= 2.0
            ),
            "fd_clap_within_n50_guard": candidate["fd_clap"]
            <= _upper_limit(baseline["fd_clap"], relative=1.25, absolute=0.10),
            "fad_vggish_within_n50_guard": candidate["fad_vggish"]
            <= _upper_limit(baseline["fad_vggish"], relative=1.25, absolute=0.25),
            "fd_pann_within_n50_guard": candidate["fd_pann"]
            <= _upper_limit(baseline["fd_pann"], relative=1.25, absolute=0.50),
            "kl_pann_within_n50_guard": candidate["kl_pann"]
            <= _upper_limit(baseline["kl_pann"], relative=1.25, absolute=0.05),
            "no_material_clipping": candidate["raw_clipped_fraction_max"] <= 1.0e-3,
        }
        checks.extend(
            {
                "domain": domain,
                "check": name,
                "pass": bool(passed),
            }
            for name, passed in domain_checks.items()
        )

    failed = [row for row in checks if not row["pass"]]
    status = "PASS" if not failed else "HOLD"
    summary = {
        "schema": "stable_audio_tools.p10_music_sound_protection_summary",
        "schema_version": 1,
        "status": status,
        "decision": (
            "Music/Sound protection passed; continuation may proceed after user approval."
            if status == "PASS"
            else "Do not resume training; inspect the failed Music/Sound protection checks."
        ),
        "evaluation_contract": str(root / "EVAL_CONTRACT.json"),
        "evaluation_rows_per_domain": 50,
        "checkpoints": steps,
        "semantic_caption_compiler_version": int(
            contract["sampling"]["semantic_caption_compiler_version"]
        ),
        "rows": rows,
        "checks": checks,
        "failed_checks": failed,
        "small_sample_warning": (
            "FAD/FD are matched N=50 checkpoint-selection diagnostics, not paper-scale estimates."
        ),
    }
    _atomic_text(
        root / "PROTECTION_SUMMARY.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    lines = [
        f"# P10 Music/Sound protection: {baseline_step // 1000}k vs {candidate_step // 1000}k",
        "",
        f"Gate: **{status}**",
        "",
        "| domain | step | text CLAP ↑ | paired CLAP ↑ | FAD ↓ | KL-PANN ↓ | DoA ° ↓ | extent ° ↓ | activity IoU ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAINS:
        for step in steps:
            value = rows[domain][str(step)]
            lines.append(
                f"| {domain} | {step // 1000}k | {value['text_clap']:.4f} | "
                f"{value['paired_clap']:.4f} | {value['fad_vggish']:.4f} | "
                f"{value['kl_pann']:.4f} | {value['plan_doa_deg']:.2f} | "
                f"{value['trajectory_extent_error_deg']:.2f} | {value['activity_iou']:.4f} |"
            )
    lines.extend(
        [
            "",
            summary["decision"],
            "",
            summary["small_sample_warning"],
        ]
    )
    if failed:
        lines.extend(
            [
                "",
                "Failed checks:",
                *[
                    f"- {row['domain']}: {row['check']}"
                    for row in failed
                ],
            ]
        )
    _atomic_text(root / "PROTECTION_SUMMARY.md", "\n".join(lines) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
