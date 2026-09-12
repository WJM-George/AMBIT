#!/usr/bin/env python3
"""Summarize a frozen matched P10 checkpoint benchmark and listening paths."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (
    DEFAULT_EVAL_ROOT,
    DOMAINS,
    atomic_json,
    checkpoint_steps,
    load_output_rows,
    load_panel,
    read_jsonl,
    summarize,
)


def _mean(summary: dict[str, Any]) -> float | None:
    value = summary.get("mean")
    return None if value is None else float(value)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _paired_checkpoint_deltas(
    metrics_root: Path, steps: tuple[int, ...]
) -> list[dict[str, Any]]:
    core_rows = read_jsonl(metrics_root / "core_per_output.jsonl")
    clap_rows = read_jsonl(metrics_root / "clap_per_output.jsonl")
    speech_rows = read_jsonl(metrics_root / "speech_per_output.jsonl")
    distributional_rows = read_jsonl(
        metrics_root / "distributional_per_output.jsonl"
    )
    specifications = [
        ("music_clap_text_cosine", "higher", clap_rows, "music", lambda row: row["generated_text_cosine"]),
        ("sound_clap_text_cosine", "higher", clap_rows, "sound", lambda row: row["generated_text_cosine"]),
        ("music_paired_kl_pann", "lower", distributional_rows, "music", lambda row: row["paired_kl_pann_softmax"]),
        ("sound_paired_kl_pann", "lower", distributional_rows, "sound", lambda row: row["paired_kl_pann_softmax"]),
        ("speech_wer", "lower", speech_rows, None, lambda row: row["generated_errors"]["wer"]),
        ("speech_cer", "lower", speech_rows, None, lambda row: row["generated_errors"]["cer"]),
        ("speech_utmos", "higher", speech_rows, None, lambda row: row["generated_utmos"]),
    ]
    for domain in DOMAINS:
        specifications.extend(
            [
                (
                    f"{domain}_plan_doa_error_deg",
                    "lower",
                    core_rows,
                    domain,
                    lambda row: row["generated_doa"]["spherical_error_mean_deg"],
                ),
                (
                    f"{domain}_generated_reference_doa_error_deg",
                    "lower",
                    core_rows,
                    domain,
                    lambda row: row["generated_reference_doa"]["spherical_error_mean_deg"],
                ),
                (
                    f"{domain}_activity_iou",
                    "higher",
                    core_rows,
                    domain,
                    lambda row: row["generated_activity"]["temporal_iou"],
                ),
            ]
        )

    output = []
    for from_step, to_step in zip(steps, steps[1:]):
        comparison: dict[str, Any] = {
            "from_step": from_step,
            "to_step": to_step,
            "delta_definition": "to_step minus from_step on identical panel IDs and seeds",
            "metrics": {},
        }
        for name, direction, source_rows, domain, getter in specifications:
            filtered = [
                row
                for row in source_rows
                if domain is None or row.get("domain") == domain
            ]
            left = {
                row["panel_id"]: getter(row)
                for row in filtered
                if int(row["checkpoint_step"]) == from_step
            }
            right = {
                row["panel_id"]: getter(row)
                for row in filtered
                if int(row["checkpoint_step"]) == to_step
            }
            if set(left) != set(right) or not left:
                raise RuntimeError(
                    f"paired checkpoint panel mismatch for {name}: {from_step}->{to_step}"
                )
            deltas = [
                None
                if left[panel_id] is None or right[panel_id] is None
                else float(right[panel_id]) - float(left[panel_id])
                for panel_id in sorted(left)
            ]
            comparison["metrics"][name] = {
                "better_direction": direction,
                "delta": summarize(deltas),
            }
        output.append(comparison)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    steps = checkpoint_steps(root)
    metrics = root / "metrics"
    required = {
        "core": metrics / "CORE_SUMMARY.json",
        "clap": metrics / "CLAP_SUMMARY.json",
        "speech": metrics / "SPEECH_SUMMARY.json",
        "distributional": metrics / "DISTRIBUTIONAL_SUMMARY.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    reports = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in required.items()
    }
    if any(report.get("status") != "PASS" for report in reports.values()):
        raise RuntimeError("one or more P10 metric phases did not pass")

    panel = load_panel(root)
    domain_counts = {
        domain: sum(row["domain"] == domain for row in panel) for domain in DOMAINS
    }
    evaluation_rows = len(panel)
    vae_summary_path = root / "vae_reconstruction/SUMMARY.json"
    vae_summary = (
        json.loads(vae_summary_path.read_text(encoding="utf-8"))
        if vae_summary_path.is_file()
        else None
    )

    rows = []
    for step in steps:
        key = str(step)
        core = reports["core"]["aggregates"][key]
        clap = reports["clap"]["aggregates"][key]
        speech = reports["speech"]["aggregates"][key]
        distributional = reports["distributional"]["aggregates"][key]
        doa_values = [
            _mean(core[domain]["plan_spherical_error_mean_deg"]) for domain in DOMAINS
        ]
        doa_values = [value for value in doa_values if value is not None]
        reference_doa_values = [
            _mean(core[domain]["reference_plan_spherical_error_mean_deg"])
            for domain in DOMAINS
            if "reference_plan_spherical_error_mean_deg" in core[domain]
        ]
        reference_doa_values = [
            value for value in reference_doa_values if value is not None
        ]
        paired_doa_values = [
            _mean(core[domain]["generated_reference_spherical_error_mean_deg"])
            for domain in DOMAINS
            if "generated_reference_spherical_error_mean_deg" in core[domain]
        ]
        paired_doa_values = [value for value in paired_doa_values if value is not None]
        summary_row = {
                "checkpoint_step": step,
                "music_clap_text_cosine": _mean(clap["music"]["clap_text_audio_cosine"]),
                "sound_clap_text_cosine": _mean(clap["sound"]["clap_text_audio_cosine"]),
                "music_fd_clap_diagnostic": float(
                    clap["music"].get("fd_clap", clap["music"].get("fd_clap_diagnostic_n5"))
                ),
                "sound_fd_clap_diagnostic": float(
                    clap["sound"].get("fd_clap", clap["sound"].get("fd_clap_diagnostic_n5"))
                ),
                "music_fad_vggish_diagnostic": float(distributional["music"]["fad_vggish_diagnostic"]),
                "sound_fad_vggish_diagnostic": float(distributional["sound"]["fad_vggish_diagnostic"]),
                "music_kl_pann": _mean(distributional["music"]["paired_kl_pann_softmax"]),
                "sound_kl_pann": _mean(distributional["sound"]["paired_kl_pann_softmax"]),
                "speech_wer": _mean(speech["wer"]),
                "speech_cer": _mean(speech["cer"]),
                "speech_corpus_wer": float(
                    speech.get("corpus_wer", _mean(speech["wer"]))
                ),
                "speech_corpus_cer": float(
                    speech.get("corpus_cer", _mean(speech["cer"]))
                ),
                "speech_pesq_wb_diagnostic": _mean(speech["pesq_wb_diagnostic"]),
                "speech_stoi_diagnostic": _mean(speech["stoi_diagnostic"]),
                "speech_si_sdr_db_diagnostic": _mean(speech["si_sdr_db_diagnostic"]),
                "speech_utmos": _mean(speech["utmos"]),
                "mean_plan_doa_error_deg_all_domains": sum(doa_values) / len(doa_values),
                "mean_reference_plan_doa_error_deg_all_domains": (
                    None
                    if not reference_doa_values
                    else sum(reference_doa_values) / len(reference_doa_values)
                ),
                "mean_generated_reference_doa_error_deg_all_domains": (
                    None
                    if not paired_doa_values
                    else sum(paired_doa_values) / len(paired_doa_values)
                ),
                "speech_activity_iou": _mean(core["speech"]["activity_temporal_iou"]),
                "max_raw_clipped_fraction": max(
                    float(core[domain]["raw_fraction_abs_ge_1"]["max"]) for domain in DOMAINS
                ),
            }
        for domain in DOMAINS:
            summary_row[f"{domain}_plan_doa_error_deg"] = _mean(
                core[domain]["plan_spherical_error_mean_deg"]
            )
            summary_row[f"{domain}_reference_plan_doa_error_deg"] = _mean(
                core[domain].get(
                    "reference_plan_spherical_error_mean_deg",
                    {"mean": None},
                )
            )
            summary_row[f"{domain}_generated_reference_doa_error_deg"] = _mean(
                core[domain].get(
                    "generated_reference_spherical_error_mean_deg",
                    {"mean": None},
                )
            )
            summary_row[f"{domain}_trajectory_extent_error_deg"] = _mean(
                core[domain].get("plan_trajectory_extent_error_deg", {"mean": None})
            )
            summary_row[f"{domain}_activity_iou"] = _mean(
                core[domain]["activity_temporal_iou"]
            )
        for domain in ("music", "sound"):
            summary_row[f"{domain}_fd_pann"] = float(
                distributional[domain].get(
                    "fd_pann",
                    distributional[domain].get("fd_pann_diagnostic_n5"),
                )
            )
        rows.append(summary_row)

    outputs = load_output_rows(root)
    reference_rows = {
        row["panel_id"]: row for row in read_jsonl(metrics / "reference_panel.jsonl")
    }
    listening: dict[str, Any] = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_listening_index",
        "schema_version": 1,
        "evaluation_rows": evaluation_rows,
        "checkpoint_outputs": evaluation_rows * len(steps),
        "panels": [],
    }
    for panel_row in panel:
        panel_outputs = [row for row in outputs if row["panel_id"] == panel_row["panel_id"]]
        panel_outputs.sort(key=lambda row: int(row["checkpoint_step"]))
        listening["panels"].append(
            {
                "panel_id": panel_row["panel_id"],
                "domain": panel_row["domain"],
                "sample_id": panel_row["sample_id"],
                "semantic_text": panel_row["semantic_text"],
                "renderer_caption": panel_row["renderer_caption"],
                "reference_stereo_path": reference_rows[panel_row["panel_id"]]["reference_stereo_path"],
                "checkpoints": [
                    {
                        "step": int(row["checkpoint_step"]),
                        "generated_stereo_path": row["generated_stereo_path"],
                        "generated_foa_path": row["generated_foa_path"],
                    }
                    for row in panel_outputs
                ],
            }
        )

    summary = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_matched_panel_evaluation",
        "schema_version": 1,
        "status": "PASS",
        "evaluation_rows": evaluation_rows,
        "domain_counts": domain_counts,
        "checkpoint_steps": list(steps),
        "checkpoint_outputs": evaluation_rows * len(steps),
        "same_rows_and_noise_across_checkpoints": True,
        "utmos_available": bool(reports["speech"]["utmos_available"]),
        "utmos_error": reports["speech"].get("utmos_error"),
        "vae_codec_ceiling": vae_summary,
        "paired_adjacent_checkpoint_deltas": _paired_checkpoint_deltas(metrics, steps),
        "small_sample_warning": (
            f"All metrics use {min(domain_counts.values())} matched clips per domain. "
            "This is a checkpoint-selection benchmark; FAD/FD values do not replace "
            "the 1,000-row/domain paper protocol."
        ),
        "rows": rows,
        "reports": {name: str(path.resolve()) for name, path in required.items()},
        "listening_index": str((metrics / "LISTENING_INDEX.json").resolve()),
    }
    atomic_json(metrics / "LISTENING_INDEX.json", listening)
    atomic_json(metrics / "EVALUATION_SUMMARY.json", summary)

    lines = [
        f"# ScenePlan DiT P10 — fixed {evaluation_rows}-row checkpoint evaluation",
        "",
        (
            f"The same {domain_counts['music']} music, {domain_counts['sound']} sound, "
            f"and {domain_counts['speech']} speech test rows and the same noise seed "
            f"per row are used at every checkpoint ({evaluation_rows * len(steps)} "
            "generated outputs total)."
        ),
        "",
        "| step | music CLAP ↑ | sound CLAP ↑ | music FAD-VGGish ↓* | sound FAD-VGGish ↓* | speech WER ↓ | speech CER ↓ | plan DoA ° ↓ | speech activity IoU ↑ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {step:,} | {music_clap} | {sound_clap} | {music_fad} | {sound_fad} | {wer} | {cer} | {doa} | {iou} |".format(
                step=row["checkpoint_step"],
                music_clap=_fmt(row["music_clap_text_cosine"]),
                sound_clap=_fmt(row["sound_clap_text_cosine"]),
                music_fad=_fmt(row["music_fad_vggish_diagnostic"], 2),
                sound_fad=_fmt(row["sound_fad_vggish_diagnostic"], 2),
                wer=_fmt(row["speech_corpus_wer"]),
                cer=_fmt(row["speech_corpus_cer"]),
                doa=_fmt(row["mean_plan_doa_error_deg_all_domains"], 2),
                iou=_fmt(row["speech_activity_iou"]),
            )
        )
    for domain in ("music", "sound"):
        lines.extend(
            [
                "",
                f"## {domain.title()}",
                "",
                "| step | CLAP ↑ | FAD-VGGish ↓* | FD-PANN ↓* | KL-PANN ↓ | plan DoA ° ↓ | generated↔GT DoA ° ↓ | trajectory extent error ° ↓ | activity IoU ↑ |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| {step:,} | {clap} | {fad} | {fd} | {kl} | {plan_doa} | {paired_doa} | {extent} | {iou} |".format(
                    step=row["checkpoint_step"],
                    clap=_fmt(row[f"{domain}_clap_text_cosine"]),
                    fad=_fmt(row[f"{domain}_fad_vggish_diagnostic"], 2),
                    fd=_fmt(row[f"{domain}_fd_pann"], 2),
                    kl=_fmt(row[f"{domain}_kl_pann"]),
                    plan_doa=_fmt(row[f"{domain}_plan_doa_error_deg"], 2),
                    paired_doa=_fmt(
                        row[f"{domain}_generated_reference_doa_error_deg"], 2
                    ),
                    extent=_fmt(row[f"{domain}_trajectory_extent_error_deg"], 2),
                    iou=_fmt(row[f"{domain}_activity_iou"]),
                )
            )
    lines.extend(
        [
            "",
            "## Speech",
            "",
            "| step | corpus WER ↓ | corpus CER ↓ | UTMOS ↑ | PESQ-WB † ↑ | STOI † ↑ | SI-SDR † dB ↑ | plan DoA ° ↓ | generated↔GT DoA ° ↓ | activity IoU ↑ |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {step:,} | {wer} | {cer} | {utmos} | {pesq} | {stoi} | {sisdr} | {plan_doa} | {paired_doa} | {iou} |".format(
                step=row["checkpoint_step"],
                wer=_fmt(row["speech_corpus_wer"]),
                cer=_fmt(row["speech_corpus_cer"]),
                utmos=_fmt(row["speech_utmos"]),
                pesq=_fmt(row["speech_pesq_wb_diagnostic"]),
                stoi=_fmt(row["speech_stoi_diagnostic"]),
                sisdr=_fmt(row["speech_si_sdr_db_diagnostic"], 2),
                plan_doa=_fmt(row["speech_plan_doa_error_deg"], 2),
                paired_doa=_fmt(row["speech_generated_reference_doa_error_deg"], 2),
                iou=_fmt(row["speech_activity_iou"]),
            )
        )
    lines.extend(
        [
            "",
            (
                f"\\* FAD/FD use {min(domain_counts.values())} clips per domain and are "
                "matched checkpoint-comparison values, not the publication-scale estimates."
            ),
            "† Paired speech waveform metrics are diagnostics for stochastic generation, not hard gates.",
            "",
            (
                "All raw FOA files are finite four-channel WYZX/ACN/SN3D float32 audio. "
                "The maximum fraction of samples at or above |1| is "
                f"{max(row['max_raw_clipped_fraction'] for row in rows):.3e}; "
                "the raw float files are not clipped. UTMOS availability is recorded "
                "in EVALUATION_SUMMARY.json."
            ),
            "",
        ]
    )
    if vae_summary is not None:
        lines.extend(
            [
                (
                    "Frozen VAE ceiling over this panel: W-channel SI-SDR "
                    f"{float(vae_summary['mean_w_channel_si_sdr_db']):.2f} dB; "
                    "four-channel RMSE "
                    f"{float(vae_summary['mean_all_channel_rmse']):.4f}."
                ),
                "",
            ]
        )
    _atomic_text(metrics / "EVALUATION_SUMMARY.md", "\n".join(lines))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
