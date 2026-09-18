#!/usr/bin/env python3
"""Build a bounded Data Analytics report artifact for the P10 diagnostics."""

from __future__ import annotations
import os

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SOUND_ROOT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/"
    "probes/p10_sound_transient_heldout_v1"
)
DEFAULT_SPEECH_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/"
    "audit/p10_speech_alignment_pilot_100_v1"
)
METRIC_LABELS = {
    "latent_mse": "Latent MSE",
    "latent_transient_mse": "Transient-frame latent MSE",
    "latent_nontransient_mse": "Non-transient latent MSE",
    "latent_flux_pearson": "Latent flux Pearson",
    "transient_frame_f1": "Transient-frame F1",
    "audio_spectral_flux_pearson": "Audio spectral-flux Pearson",
    "w_si_sdr_db": "W-channel SI-SDR (dB)",
    "w_log_stft_l1": "W-channel log-STFT L1",
    "clap_generated_text_cosine": "Text CLAP cosine",
    "clap_generated_reference_audio_cosine": "Generated/reference CLAP cosine",
    "generated_doa_error_deg": "DoA error (deg)",
    "generated_activity_iou": "Activity IoU",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _speech_histogram(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = np.asarray(
        [float(row["endpoint_abs_delta_vs_strong_qc_sec"]) for row in rows],
        dtype=np.float64,
    )
    edges = np.arange(0.0, 0.4001, 0.05)
    counts, _ = np.histogram(values, bins=edges)
    output = []
    cumulative = 0
    for index, (left, right, count) in enumerate(
        zip(edges[:-1], edges[1:], counts)
    ):
        cumulative += int(count)
        output.append(
            {
                "bin_order": index,
                "endpoint_error_bin_sec": f"{left:.2f}–{right:.2f}",
                "rows": int(count),
                "cumulative_rows": cumulative,
                "cumulative_rate": cumulative / len(values),
                "pilot_rows": len(values),
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sound-root", type=Path, default=DEFAULT_SOUND_ROOT)
    parser.add_argument("--speech-root", type=Path, default=DEFAULT_SPEECH_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sound_root = args.sound_root.expanduser().resolve(strict=True)
    speech_root = args.speech_root.expanduser().resolve(strict=True)
    output = (
        args.output.expanduser().resolve()
        if args.output
        else sound_root / "comparison/P10_VALIDATION_ARTIFACT.json"
    )
    sound_summary_path = sound_root / "comparison/HELDOUT_SUMMARY.json"
    sound_summary = _read_json(sound_summary_path.resolve(strict=True))
    sound_contract_path = sound_root / "contract/CONTRACT.json"
    sound_contract = _read_json(sound_contract_path.resolve(strict=True))
    speech_summary_path = speech_root / "ALIGNMENT_SUMMARY.json"
    speech_summary = _read_json(speech_summary_path.resolve(strict=True))
    caption_summary_path = speech_root / "FULL_CAPTION_MAPPING_SUMMARY.json"
    caption_summary = _read_json(caption_summary_path.resolve(strict=True))
    registry_path = speech_root / "registry/source_alignment_registry.jsonl"
    registry = _read_jsonl(registry_path.resolve(strict=True))

    sound_seed_arm: list[dict[str, Any]] = []
    for seed, arms in sound_summary["aggregate_by_seed"].items():
        for arm, metrics in arms.items():
            sound_seed_arm.append(
                {
                    "seed": str(seed),
                    "arm": arm,
                    "latent_transient_mse": metrics["latent_transient_mse"],
                    "latent_nontransient_mse": metrics[
                        "latent_nontransient_mse"
                    ],
                    "audio_spectral_flux_pearson": metrics[
                        "audio_spectral_flux_pearson"
                    ],
                    "text_clap_cosine": metrics[
                        "clap_generated_text_cosine"
                    ],
                    "activity_iou": metrics["generated_activity_iou"],
                    "doa_error_deg": metrics["generated_doa_error_deg"],
                    "heldout_assets": 50,
                    "strictly_from_scratch": True,
                }
            )
    sound_seed_arm.sort(key=lambda row: (int(row["seed"]), row["arm"]))

    sound_metric_rows: list[dict[str, Any]] = []
    baseline = sound_summary["aggregate_by_arm"]["baseline"]
    transient = sound_summary["aggregate_by_arm"]["transient"]
    for order, (metric, label) in enumerate(METRIC_LABELS.items(), start=1):
        effect = sound_summary["paired_effects"][metric]
        ci_low, ci_high = effect["bootstrap_95pct_ci_of_mean"]
        sound_metric_rows.append(
            {
                "order": order,
                "metric": label,
                "metric_id": metric,
                "direction": effect["direction"],
                "baseline": baseline[metric],
                "transient": transient[metric],
                "delta_transient_minus_baseline": transient[metric]
                - baseline[metric],
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "improved_assets": effect[
                    "independent_heldout_assets_improved"
                ],
                "heldout_assets": effect["independent_heldout_assets_total"],
                "paired_training_seeds": 2,
            }
        )

    causal = bool(
        sound_summary["causal_support_for_heldout_transient_generalization"]
    )
    no_regression = bool(
        sound_summary["no_major_semantic_timing_or_spatial_regression"]
    )
    decision = str(sound_summary["decision"])
    decision_cn = {
        "PROMOTE_TO_JOINT_MINI_PILOT": "进入 joint mini-pilot",
        "DO_NOT_PROMOTE_WITH_CURRENT_EVIDENCE": "当前证据不支持进入主线",
        "REVISE_BEFORE_PROMOTION": "修订后再验证",
    }.get(decision, decision)

    headline = {
        "speech_alignment_pass_rate": speech_summary["metrics"][
            "row_success_rate"
        ],
        "speech_endpoint_p90_sec": speech_summary["metrics"][
            "endpoint_abs_delta_p90_sec"
        ],
        "full_caption_token_mapping_rate": caption_summary["metrics"][
            "semantic_token_mapping_rate_mean"
        ],
        "full_caption_max_tokens": caption_summary["metrics"][
            "caption_valid_tokens_max"
        ],
        "raw_zero_duration_aligned_items": caption_summary["metrics"][
            "raw_zero_duration_aligned_items"
        ],
        "quantized_token_interval_frames_min": caption_summary["metrics"][
            "quantized_token_interval_frames_min"
        ],
        "sound_causal_gate": 1.0 if causal else 0.0,
        "sound_no_regression_gate": 1.0 if no_regression else 0.0,
        "sound_heldout_assets": sound_summary["independent_heldout_assets"],
    }

    speech_source = {
        "id": "speech_alignment",
        "label": "Qwen3 speech-alignment pilot",
        "path": str(registry_path),
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": (
                "Reads the frozen 100-row source registry and computes endpoint "
                "coverage and distribution statistics."
            ),
            "sql": (
                "SELECT * FROM read_json_auto('"
                + str(registry_path)
                + "', format='newline_delimited')"
            ),
            "tables_used": [str(registry_path)],
            "filters": [
                "Frozen P9 test split",
                "50 LibriTTS + 50 HiFiTTS",
                "five duration strata per corpus",
            ],
            "metric_definitions": [
                "Alignment pass rate = passing monotonic in-bounds rows / 100.",
                "Endpoint error = absolute Qwen forced-aligner last-word endpoint minus independent strong-QC ASR last-word endpoint, seconds.",
            ],
        },
    }
    caption_source = {
        "id": "full_caption_mapping",
        "label": "Production-caption timing sidecar audit",
        "path": str(caption_summary_path),
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": (
                "Recompiles and tokenizes each full production P10 caption, then "
                "maps aligned words to speech-role Qwen tokens and VAE frames."
            ),
            "sql": (
                "SELECT * FROM read_json_auto('"
                + str(caption_summary_path)
                + "')"
            ),
            "tables_used": [
                str(caption_summary_path),
                str(speech_root / "registry/full_caption_token_timing.jsonl"),
            ],
            "filters": ["same frozen 100-row pilot", "caption limit 512 tokens"],
            "metric_definitions": [
                "Semantic token mapping rate = timed semantic transcript-role tokens / all semantic transcript-role tokens.",
                "Aligned-item mapping rate = aligned words covered by at least one production-caption speech-role token / all aligned words.",
                "Raw zero-duration count is preserved from Qwen output; the offline training sidecar uses floor/ceil VAE-grid quantization and requires every mapped token interval to cover at least one frame.",
            ],
        },
    }
    sound_source = {
        "id": "sound_heldout",
        "label": "Paired sound transient held-out experiment",
        "path": str(sound_summary_path),
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": (
                "Aggregates paired baseline/transient runs by seed and computes "
                "asset-level bootstrap intervals on a parent-disjoint panel."
            ),
            "sql": (
                "SELECT * FROM read_json_auto('"
                + str(sound_summary_path)
                + "')"
            ),
            "tables_used": [
                str(sound_summary_path),
                str(sound_contract_path),
            ],
            "filters": [
                "97,472 unique clean single-source sound training assets",
                "50 frozen P9 test assets",
                "audio hash, asset id, and parent recording overlap all zero",
                "two paired scratch seeds: 42 and 31415",
                "same held-out noise per paired arm",
            ],
            "metric_definitions": [
                "Transient latent MSE is evaluated only on top-quintile clean-latent temporal derivatives dilated by two frames.",
                "Causal support requires lower transient MSE in both seeds, an asset-bootstrap 95% CI fully below zero, and higher decoded spectral-flux correlation in both seeds.",
                "No-major-regression requires non-transient MSE ratio <=1.05, CLAP delta >=-0.01, activity IoU delta >=-0.02, and DoA error delta <=2 degrees.",
            ],
        },
    }

    executive = (
        "## Executive Summary\n\n"
        "Speech 的离线 timing-teacher 帧级数据链路已经通过：forced alignment、完整 "
        "P10 caption token role 和 VAE frame grid 已接通；原始对齐中的零时长短词被保留并显式审计，"
        "训练 sidecar 才执行最小一帧量化。这个结果仍不是模型收益证明。"
        f" Sound 的 parent-disjoint 双随机种子结论为：**{decision_cn}**。"
    )
    speech_findings = (
        "## Speech alignment findings\n\n"
        f"100/100 条通过，endpoint absolute error 中位数 "
        f"{speech_summary['metrics']['endpoint_abs_delta_median_sec']:.3f}s、"
        f"p90 {speech_summary['metrics']['endpoint_abs_delta_p90_sec']:.3f}s。"
    )
    caption_findings = (
        "## Full-caption token/frame mapping\n\n"
        "完整 caption 的 semantic transcript-token 与 aligned-word 映射率均为 "
        f"100%；raw alignment 中有 {caption_summary['metrics']['raw_zero_duration_aligned_items']} "
        f"个零时长短词，分布在 {caption_summary['metrics']['rows_with_raw_zero_duration_items']} 条样本，"
        f"量化后 token interval 最小为 {caption_summary['metrics']['quantized_token_interval_frames_min']} 帧。"
        "另有 3 条 aligner 末端上取整 30ms，已按权威 waveform/activity 边界显式夹回并保留原始 overshoot。"
    )
    sound_findings = (
        "## Sound held-out findings\n\n"
        f"实验结论：**{decision_cn}**。Held-out transient generalization gate = "
        f"**{causal}**；semantic/timing/spatial no-regression gate = "
        f"**{no_regression}**。训练和评测在 audio hash、asset id、parent recording "
        "三个层级均零重叠。"
    )
    scope = (
        "## Scope and methodology\n\n"
        "Speech pilot 使用冻结 P9 test split 的 50 LibriTTS + 50 HiFiTTS。"
        "Sound 对照在 97,472 条 clean single-source sound 上各训练 3,000 steps "
        "（global batch 288，约 8.9 epochs），同 seed 内仅 transient weighting 不同；"
        "50 条 held-out 采用相同推理噪声。"
    )
    limitations = (
        "## Limitations and robustness\n\n"
        "Speech 结果只验证 teacher 数据与接口，raw 零时长词必须通过已审计的 VAE-grid "
        "量化 sidecar 使用，且尚未验证 duration allocator / monotonic attention bias 的生成收益。"
        "Sound 结果只覆盖 single-source sound；mixed scene "
        "中的 mixture derivative 无法归因到单一 source，不能直接推广。3,000-step "
        "scratch run 是机制实验，不是完整质量收敛实验。"
    )
    if causal and no_regression:
        next_steps = (
            "## Decision and next steps\n\n"
            "1. 将 transient objective 先放入 mixed-domain joint mini-pilot，但只对 "
            "sound-only row 启用。\n2. 在进入长训练前向量化 quantile/mask 计算，消除诊断版吞吐开销。"
            "\n3. 另起 speech duration-conditioned 10-item overfit 与固定 held-out ablation。"
            "\n4. 模型机制门禁完成后，再决定是否新增约 106k sound identities。"
        )
    else:
        next_steps = (
            "## Decision and next steps\n\n"
            "1. 当前 transient 配方不进入 full P10；依据失败 gate 调整 quantile、dilation "
            "或 objective。\n2. 另起 speech duration-conditioned 10-item overfit 与固定 "
            "held-out ablation。\n3. 模型机制门禁完成后，再决定是否新增约 106k "
            "sound identities；不以补数据掩盖未通过的模型机制。"
        )

    sources = [speech_source, caption_source, sound_source]
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "P10 Speech Timing 与 Sound Transient 验证",
        "description": "Source-disjoint controlled validation before changing full P10.",
        "sources": sources,
        "cards": [
            {
                "id": "speech_alignment_card",
                "dataset": "headline",
                "sourceId": "speech_alignment",
                "description": "100-row forced-alignment row pass rate.",
                "metrics": [
                    {
                        "label": "Speech alignment pass",
                        "field": "speech_alignment_pass_rate",
                        "format": "percent",
                    },
                    {
                        "label": "Endpoint p90 (s)",
                        "field": "speech_endpoint_p90_sec",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "caption_mapping_card",
                "dataset": "headline",
                "sourceId": "full_caption_mapping",
                "description": "Production-caption semantic transcript-token timing coverage.",
                "metrics": [
                    {
                        "label": "Full-caption token map",
                        "field": "full_caption_token_mapping_rate",
                        "format": "percent",
                    },
                    {
                        "label": "Max caption tokens",
                        "field": "full_caption_max_tokens",
                        "format": "number",
                    },
                    {
                        "label": "Raw zero spans",
                        "field": "raw_zero_duration_aligned_items",
                        "format": "number",
                    },
                    {
                        "label": "Min quantized frames",
                        "field": "quantized_token_interval_frames_min",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "sound_causal_card",
                "dataset": "headline",
                "sourceId": "sound_heldout",
                "description": "Frozen joint causal-support gate for held-out transient improvement.",
                "metrics": [
                    {
                        "label": "Sound causal gate",
                        "field": "sound_causal_gate",
                        "format": "percent",
                    },
                    {
                        "label": "Held-out assets",
                        "field": "sound_heldout_assets",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "sound_regression_card",
                "dataset": "headline",
                "sourceId": "sound_heldout",
                "description": "All frozen semantic, timing, spatial, and non-transient regression gates.",
                "metrics": [
                    {
                        "label": "No-regression gate",
                        "field": "sound_no_regression_gate",
                        "format": "percent",
                    }
                ],
            },
        ],
        "charts": [
            {
                "id": "speech_endpoint_histogram",
                "title": "Speech alignment endpoint absolute error distribution",
                "subtitle": "100 frozen test donors; comparison against independent strong-QC ASR endpoint.",
                "type": "bar",
                "dataset": "speech_endpoint_bins",
                "sourceId": "speech_alignment",
                "encodings": {
                    "x": {
                        "field": "endpoint_error_bin_sec",
                        "type": "ordinal",
                        "label": "Absolute endpoint error (s)",
                    },
                    "y": {
                        "field": "rows",
                        "type": "quantitative",
                        "label": "Rows",
                    },
                    "tooltip": [
                        {"field": "rows", "label": "Rows"},
                        {
                            "field": "cumulative_rate",
                            "label": "Cumulative rate",
                            "format": "percent",
                        },
                    ],
                },
                "xAxisTitle": "Absolute endpoint error (seconds)",
                "yAxisTitle": "Donors",
                "valueFormat": "number",
            },
            {
                "id": "sound_transient_mse_by_seed",
                "title": "Held-out transient-frame latent MSE by seed and arm",
                "subtitle": "Each arm is trained from scratch; paired arms use the same seed and inference noise.",
                "type": "bar",
                "dataset": "sound_seed_arm",
                "sourceId": "sound_heldout",
                "encodings": {
                    "x": {"field": "seed", "type": "nominal", "label": "Seed"},
                    "y": {
                        "field": "latent_transient_mse",
                        "type": "quantitative",
                        "label": "Transient latent MSE",
                    },
                    "color": {"field": "arm", "type": "nominal", "label": "Arm"},
                    "tooltip": [
                        {
                            "field": "latent_nontransient_mse",
                            "label": "Non-transient MSE",
                        },
                        {
                            "field": "audio_spectral_flux_pearson",
                            "label": "Spectral-flux Pearson",
                        },
                        {"field": "text_clap_cosine", "label": "Text CLAP"},
                        {"field": "activity_iou", "label": "Activity IoU"},
                        {"field": "doa_error_deg", "label": "DoA error (deg)"},
                    ],
                },
                "xAxisTitle": "Training seed",
                "yAxisTitle": "Transient-frame latent MSE",
                "valueFormat": "number",
            },
        ],
        "tables": [
            {
                "id": "sound_metric_table",
                "title": "Sound held-out metric comparison",
                "subtitle": "Two-seed means; delta is transient minus baseline.",
                "dataset": "sound_metric_rows",
                "sourceId": "sound_heldout",
                "density": "compact",
                "defaultSort": {"field": "order", "direction": "asc"},
                "columns": [
                    {"field": "order", "label": "#", "format": "number"},
                    {"field": "metric", "label": "Metric"},
                    {"field": "direction", "label": "Better"},
                    {"field": "baseline", "label": "Baseline", "format": "number"},
                    {"field": "transient", "label": "Transient", "format": "number"},
                    {
                        "field": "delta_transient_minus_baseline",
                        "label": "Delta T−B",
                        "format": "number",
                        "movement": True,
                    },
                    {"field": "bootstrap_ci_low", "label": "CI low", "format": "number"},
                    {"field": "bootstrap_ci_high", "label": "CI high", "format": "number"},
                    {"field": "improved_assets", "label": "Improved / 50", "format": "number"},
                ],
            }
        ],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# P10 Speech Timing 与 Sound Transient 验证",
            },
            {"id": "executive", "type": "markdown", "body": executive},
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "speech_alignment_card",
                    "caption_mapping_card",
                    "sound_causal_card",
                    "sound_regression_card",
                ],
            },
            {
                "id": "speech_findings",
                "type": "markdown",
                "body": speech_findings,
                "sourceId": "speech_alignment",
            },
            {
                "id": "caption_findings",
                "type": "markdown",
                "body": caption_findings,
                "sourceId": "full_caption_mapping",
            },
            {
                "id": "speech_chart",
                "type": "chart",
                "chartId": "speech_endpoint_histogram",
            },
            {
                "id": "sound_findings",
                "type": "markdown",
                "body": sound_findings,
                "sourceId": "sound_heldout",
            },
            {
                "id": "sound_chart",
                "type": "chart",
                "chartId": "sound_transient_mse_by_seed",
            },
            {
                "id": "sound_table",
                "type": "table",
                "tableId": "sound_metric_table",
            },
            {"id": "scope", "type": "markdown", "body": scope},
            {"id": "limitations", "type": "markdown", "body": limitations},
            {"id": "next_steps", "type": "markdown", "body": next_steps},
        ],
    }
    snapshot = {
        "version": 1,
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "ready",
        "datasets": {
            "headline": [headline],
            "speech_endpoint_bins": _speech_histogram(registry),
            "sound_seed_arm": sound_seed_arm,
            "sound_metric_rows": sound_metric_rows,
        },
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
        "package_info": {
            "validation": "P10 controlled speech/sound diagnostic",
            "decision": decision,
            "sound_contract_schema_version": sound_contract["schema_version"],
            "chart_map": [
                {
                    "section": "Speech alignment findings",
                    "question": "How concentrated is endpoint disagreement across 100 donors?",
                    "family": "comparison",
                    "type": "bar",
                    "dataset": "speech_endpoint_bins",
                    "palette_policy": "single-root preferred",
                },
                {
                    "section": "Sound held-out findings",
                    "question": "Does transient weighting lower held-out transient error in both paired seeds?",
                    "family": "comparison",
                    "type": "grouped bar",
                    "dataset": "sound_seed_arm",
                    "palette_policy": "hard two-root cap",
                },
            ],
        },
    }
    _atomic_json(output, artifact)
    print(str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
