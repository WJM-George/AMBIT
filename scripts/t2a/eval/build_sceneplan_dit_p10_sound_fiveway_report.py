#!/usr/bin/env python3
"""Build the technical listening report for the simple-sound five-way audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/p10_eval/"
    "p10_sound_simple_5way_v1"
)
SYSTEMS = (
    "reference",
    "vae_1p35m_reconstruction",
    "r5_50k",
    "r6_10k",
    "r6_20k",
)
DISPLAY = {
    "reference": "Reference",
    "vae_1p35m_reconstruction": "1.35M WDMix VAE reconstruction",
    "r5_50k": "r5 50k",
    "r6_10k": "r6 10k",
    "r6_20k": "r6 20k",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _mean(aggregate: dict[str, Any], key: str) -> float:
    value = aggregate[key]
    if isinstance(value, dict):
        value = value["mean"]
    return float(value)


def _f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _pct(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    panel = _read_jsonl(root / "SOUND_SIMPLE_PANEL_5.jsonl")
    selection = _read_json(root / "SELECTION_AUDIT.json")
    metrics = _read_json(root / "metrics/FIVE_WAY_METRICS.json")
    per_sample = _read_jsonl(root / "metrics/FIVE_WAY_PER_SAMPLE.jsonl")
    montage = _read_json(root / "listening_montages/MONTAGE_INDEX.json")
    aggregates = metrics["aggregates"]
    if not (
        selection.get("status") == "PASS"
        and metrics.get("status") == "PASS"
        and montage.get("status") == "PASS"
        and len(panel) == 5
        and len(per_sample) == 25
    ):
        raise RuntimeError("five-way evaluation is incomplete")

    model_systems = ("r5_50k", "r6_10k", "r6_20k")
    best_text = max(
        model_systems,
        key=lambda system: _mean(aggregates[system], "clap_text_audio_cosine"),
    )
    best_fad = min(
        model_systems,
        key=lambda system: float(aggregates[system]["fad_vggish_diagnostic_n5"]),
    )
    best_pann = min(
        model_systems,
        key=lambda system: float(aggregates[system]["fd_pann_diagnostic_n5"]),
    )

    lines = [
        "# P10 simple-sound five-way evaluation",
        "",
        "## Technical summary",
        "",
        (
            f"在这组按全量 source 频率筛选、并通过原标签/A2T 对齐门禁的 5 条简单 sound 上，"
            f"**{DISPLAY[best_text]}** 的文本 CLAP 最好，**{DISPLAY[best_fad]}** 的 "
            f"FAD-VGGish 最好，**{DISPLAY[best_pann]}** 的 FD-PANN 最好。整体证据支持 "
            "r6 到 20k 已改善 sound 语义，但不支持“所有维度都已解决”。"
        ),
        "",
        (
            "1.35M WDMix VAE reconstruction 在 paired CLAP、FAD、FD-PANN、KL-PANN "
            "与空间误差上都明显领先三个 DiT，因此当前 sound 的主要差距在 DiT 生成，"
            "不是 VAE codec ceiling。"
        ),
        "",
        (
            "逐条看，engine、cat、keyboard、siren 已出现可用的语义信号；toilet flushing "
            "仍是共同失败项。训练保持暂停，先听完五路匹配样本，再决定继续 warm-start、"
            "做采样/结构修复，或另开 r6-from-scratch A/B。"
        ),
        "",
        "## Five-way aggregate results",
        "",
        "| system | text CLAP ↑ | paired ref CLAP ↑ | retrieval ↑ | FD-CLAP ↓ | FAD-VGGish ↓ | FD-PANN ↓ | KL-PANN ↓ | DoA error ° ↓ | activity IoU ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        aggregate = aggregates[system]
        lines.append(
            "| "
            + " | ".join(
                [
                    DISPLAY[system],
                    _f(_mean(aggregate, "clap_text_audio_cosine")),
                    _f(_mean(aggregate, "paired_reference_clap_cosine")),
                    _pct(float(aggregate["within_panel_text_retrieval_top1"])),
                    _f(float(aggregate["fd_clap_diagnostic_n5"])),
                    _f(float(aggregate["fad_vggish_diagnostic_n5"])),
                    _f(float(aggregate["fd_pann_diagnostic_n5"])),
                    _f(_mean(aggregate, "paired_kl_pann_softmax")),
                    _f(_mean(aggregate, "plan_spherical_error_mean_deg"), 1),
                    _f(_mean(aggregate, "activity_temporal_iou")),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "Reference 是目标锚点；VAE reconstruction 是 codec ceiling，不是一个 text-to-audio 系统。"
            "Text CLAP 偶尔高于 Reference 只表示更贴近 CLAP 的文本方向，不表示音质或真实性超过原音。",
            "",
            "## What is common in the source universe",
            "",
            (
                f"频率统计覆盖 {selection['frequency_basis']['sound_unique_sources']:,} 个 unique sound source，"
                f"以及 {selection['frequency_basis']['sound_sceneplan_references']:,} 次 ScenePlan source 引用。"
                "下面的族允许重叠，不能相加成 100%。"
            ),
            "",
            "| event family | unique sources | ScenePlan references | reference share |",
            "|---|---:|---:|---:|",
        ]
    )
    frequency_rows = sorted(
        selection["frequency_basis"]["families"].items(),
        key=lambda item: int(item[1]["sceneplan_references_all_splits"]),
        reverse=True,
    )
    for family, values in frequency_rows:
        lines.append(
            f"| {family} | {int(values['unique_sources_all_splits']):,} | "
            f"{int(values['sceneplan_references_all_splits']):,} | "
            f"{float(values['reference_share_all_splits']) * 100.0:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Frozen easy-case panel and prompts",
            "",
            "每条都是正式 testset 的单 sound source；无 music、无 TTS、无 spoken-language background。",
            "",
        ]
    )
    for row in panel:
        lines.extend(
            [
                f"### {row['panel_id']} — {row['demo_name']}",
                "",
                f"- Original label: `{row['raw_label']}`",
                f"- Source description: {row['semantic_text']}",
                f"- Full renderer caption: {row['renderer_caption']}",
                (
                    f"- Scene: room={row['room_type']}, motion={row['motion_type']}, "
                    f"activity coverage={float(row['activity_fraction']) * 100.0:.1f}%"
                ),
                "",
                (
                    "Five-way montage order: **Reference → VAE reconstruction → r5 50k → "
                    "r6 10k → r6 20k**; clips are separated by 0.75 seconds."
                ),
                "",
                f"[Listen to {row['panel_id']} five-way montage]({montage['by_case'][row['panel_id']]['path']})",
                "",
                "| system | text CLAP ↑ | paired ref CLAP ↑ | KL-PANN ↓ | DoA ° ↓ | activity IoU ↑ |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for system in SYSTEMS:
            item = next(
                value
                for value in per_sample
                if value["system"] == system and value["panel_id"] == row["panel_id"]
            )
            value = item["metrics"]
            doa = value["plan_spherical_error_mean_deg"]
            lines.append(
                f"| {DISPLAY[system]} | {_f(float(value['clap_text_audio_cosine']))} | "
                f"{_f(float(value['paired_reference_clap_cosine']))} | "
                f"{_f(float(value['paired_kl_pann_softmax']))} | "
                f"{'n/a' if doa is None else _f(float(doa), 1)} | "
                f"{_f(float(value['activity_temporal_iou']))} |"
            )
        lines.append("")

    lines.extend(
        [
            "## System-wise montages",
            "",
            "每个 montage 的 case 顺序都是 engine → cat → keyboard → toilet → siren。",
            "",
        ]
    )
    for system in SYSTEMS:
        lines.append(
            f"- [{DISPLAY[system]} — five sound cases]({montage['by_system'][system]['path']})"
        )

    lines.extend(
        [
            "",
            "## Metric definitions and method",
            "",
            "- Text CLAP: W channel 与 source semantic description 的 CLAP cosine；越高越好。",
            "- Paired reference CLAP: 当前音频与同一 case Reference 的 CLAP audio embedding cosine；越高越好。",
            "- FAD-VGGish / FD-CLAP / FD-PANN: 5 条匹配集合对 Reference 的分布距离；越低越好。",
            "- KL-PANN: 每条音频相对 Reference 的 PANN class posterior KL 后取均值；越低越好。",
            "- DoA error: 原始四通道 WYZX/ACN/SN3D 上的 active-intensity 球面角误差；越低越好。",
            "- Activity IoU: 从音频能量检测出的有效帧与 ScenePlan activity mask 的 IoU；越高越好。",
            "- 语义/分布指标只读取 W channel；空间指标使用未经归一化的四通道 FOA。",
            "- 试听 preview 使用固定 FOA-to-stereo 解码并逐条响度归一化；preview 不参与任何客观指标。",
            "- 三个 DiT 对每条 case 使用完全相同的噪声 seed、30-step Euler RF、CFG=3.0、APG=1.0。",
            "",
            "## Limitations and decision boundary",
            "",
            "- N=5 是 smoke/demo benchmark。FAD、FD 和 top-1 只能用于同一面板的诊断对比，不能作为论文级总体结论。",
            "- Reference 的 renderer、room 与 FOA intensity estimator 本身构成空间测量底噪；例如 toilet Reference 的 DoA error 已较高。",
            "- 这组 easy cases 有意减少语义复杂度，不替代原来的随机 15 条面板；它用于回答“模型是否连常见简单 sound 都能生成”。",
            "- 没有画趋势图：只有 5 个离散系统且 N=5，精确表格和 matched audio 比图表更不容易误导。",
            "- 在人工听感确认前，不恢复训练，也不删除 warm-start run。",
            "",
            "## Recommended next step",
            "",
            "先逐条听五个 case，重点判断：keyboard 的 r6-20k 语义提升是否真实、cat 是否只生成了短促片段、"
            "toilet 是否仍完全跑偏、siren 是否音色正确但分类分布异常。若 r6-20k 的主观结果与指标一致，"
            "可继续 warm-start；若常见简单事件仍普遍错误，再做 r6-from-scratch 与采样/损失结构 A/B，而不是归咎于 VAE。",
            "",
            "## Further questions",
            "",
            "- toilet failure 是训练覆盖、caption binding、motion control，还是采样 CFG 的问题？",
            "- r6-20k 的 text CLAP 提升是否伴随真实感/瞬态质量提升？",
            "- 简单事件的改善能否扩展到多 source 场景，而不牺牲空间与 activity？",
            "",
        ]
    )
    destination = root / "SOUND_FIVE_WAY_EVALUATION.md"
    _atomic_text(destination, "\n".join(lines))
    receipt = {
        "status": "PASS",
        "report": str(destination.resolve()),
        "panel_rows": len(panel),
        "systems": list(SYSTEMS),
        "per_sample_rows": len(per_sample),
        "selection_audit_status": selection["status"],
        "metrics_status": metrics["status"],
        "montage_status": montage["status"],
    }
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
