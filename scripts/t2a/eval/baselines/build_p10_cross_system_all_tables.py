#!/usr/bin/env python3
"""Build complete Markdown, CSV, LaTeX, and MCP-report tables for P10 eval."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_BENCHMARK_ROOT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_balanced_1200_ckpt20k_100k_v1/cross_system_baselines"
)

REFERENCE_ID = "ground_truth"
VAE_ID = "vae_codec_ceiling"
OURS_STEPS = (20_000, 40_000, 60_000, 80_000, 100_000)

DOMAIN_ORDERS = {
    "music": [
        REFERENCE_ID,
        VAE_ID,
        *(f"ours_sceneplan_foa_{step}" for step in OURS_STEPS),
        "stable_audio_open_1_0",
        "tangoflux",
        "audiox_turbo",
    ],
    "sound": [
        REFERENCE_ID,
        VAE_ID,
        *(f"ours_sceneplan_foa_{step}" for step in OURS_STEPS),
        "stable_audio_open_1_0",
        "tangoflux",
        "audiox_turbo",
        "mmaudio_large_44k_v2_text_only",
        "woosh_flow",
    ],
    "speech": [
        REFERENCE_ID,
        VAE_ID,
        *(f"ours_sceneplan_foa_{step}" for step in OURS_STEPS),
        "qwen3_tts_1p7b_voice_design",
    ],
}


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "↑": r"$\uparrow$",
        "↓": r"$\downarrow$",
    }
    return "".join(replacements.get(char, char) for char in value)


def spatial_for(metrics: dict[str, Any], system_id: str, domain: str) -> dict[str, Any]:
    return metrics["native_foa_spatial"].get(system_id, {}).get(domain, {})


def audio_rows(metrics: dict[str, Any], domain: str) -> list[dict[str, Any]]:
    values = metrics["audio_domains"][domain]
    rows = []
    for order, system_id in enumerate(DOMAIN_ORDERS[domain], start=1):
        value = values[system_id]
        spatial = spatial_for(metrics, system_id, domain)
        rows.append(
            {
                "order": order,
                "domain": domain,
                "system_id": system_id,
                "system": value["display_name"],
                "checkpoint_step": (
                    int(system_id.rsplit("_", 1)[-1])
                    if system_id.startswith("ours_sceneplan_foa_")
                    else None
                ),
                "clap": value["clap_text_audio_cosine"]["mean"],
                "paired_clap": value["paired_generated_reference_clap_cosine"]["mean"],
                "fd_clap": value["fd_clap_diagnostic"],
                "fad_vggish": value["fad_vggish_diagnostic"],
                "fd_pann": value["fd_pann_diagnostic"],
                "kl_pann": value["paired_kl_pann_softmax"]["mean"],
                "doa_error_deg": spatial.get("doa_spherical_error_deg"),
                "azimuth_mae_deg": spatial.get("azimuth_mae_deg"),
                "elevation_mae_deg": spatial.get("elevation_mae_deg"),
                "trajectory_extent_error_deg": spatial.get(
                    "trajectory_extent_error_deg"
                ),
                "activity_iou": spatial.get("activity_iou"),
                "wer": None,
                "seen_speaker_wer": None,
                "unseen_speaker_wer": None,
                "cer": None,
                "utmos": None,
            }
        )
    return rows


def speech_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    values = metrics["speech"]
    rows = []
    for order, system_id in enumerate(DOMAIN_ORDERS["speech"], start=1):
        value = values[system_id]
        spatial = spatial_for(metrics, system_id, "speech")
        rows.append(
            {
                "order": order,
                "domain": "speech",
                "system_id": system_id,
                "system": value["display_name"],
                "checkpoint_step": (
                    int(system_id.rsplit("_", 1)[-1])
                    if system_id.startswith("ours_sceneplan_foa_")
                    else None
                ),
                "clap": None,
                "paired_clap": None,
                "fd_clap": None,
                "fad_vggish": None,
                "fd_pann": None,
                "kl_pann": None,
                "doa_error_deg": spatial.get("doa_spherical_error_deg"),
                "azimuth_mae_deg": spatial.get("azimuth_mae_deg"),
                "elevation_mae_deg": spatial.get("elevation_mae_deg"),
                "trajectory_extent_error_deg": spatial.get(
                    "trajectory_extent_error_deg"
                ),
                "activity_iou": spatial.get("activity_iou"),
                "wer": value["corpus_wer"],
                "seen_speaker_wer": value["subgroups"]["seen_speaker"][
                    "corpus_wer"
                ],
                "unseen_speaker_wer": value["subgroups"]["unseen_speaker"][
                    "corpus_wer"
                ],
                "cer": value["corpus_cer"],
                "utmos": value["utmos"]["mean"],
            }
        )
    return rows


def checkpoint_rows(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_domain_step = {
        (row["domain"], row["checkpoint_step"]): row
        for row in all_rows
        if row["checkpoint_step"] is not None
    }
    rows = []
    for step in OURS_STEPS:
        music = by_domain_step[("music", step)]
        sound = by_domain_step[("sound", step)]
        speech = by_domain_step[("speech", step)]
        rows.append(
            {
                "checkpoint_step": step,
                "checkpoint": f"{step // 1000}k",
                "music_clap": music["clap"],
                "music_paired_clap": music["paired_clap"],
                "music_fad": music["fad_vggish"],
                "music_doa_deg": music["doa_error_deg"],
                "sound_clap": sound["clap"],
                "sound_paired_clap": sound["paired_clap"],
                "sound_fad": sound["fad_vggish"],
                "sound_doa_deg": sound["doa_error_deg"],
                "speech_wer": speech["wer"],
                "speech_seen_wer": speech["seen_speaker_wer"],
                "speech_unseen_wer": speech["unseen_speaker_wer"],
                "speech_utmos": speech["utmos"],
                "speech_doa_deg": speech["doa_error_deg"],
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---:" if i else "---" for i in range(len(headers))) + "|",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def build_markdown(
    checkpoint: list[dict[str, Any]],
    music: list[dict[str, Any]],
    sound: list[dict[str, Any]],
    speech: list[dict[str, Any]],
) -> str:
    overview = markdown_table(
        [
            "Step",
            "Music CLAP ↑",
            "Music FAD ↓",
            "Music DoA ↓",
            "Sound CLAP ↑",
            "Sound FAD ↓",
            "Sound DoA ↓",
            "Speech WER ↓",
            "Speech UTMOS ↑",
            "Speech DoA ↓",
        ],
        [
            [
                row["checkpoint"],
                fmt(row["music_clap"]),
                fmt(row["music_fad"]),
                fmt(row["music_doa_deg"], 2),
                fmt(row["sound_clap"]),
                fmt(row["sound_fad"]),
                fmt(row["sound_doa_deg"], 2),
                fmt(row["speech_wer"]),
                fmt(row["speech_utmos"]),
                fmt(row["speech_doa_deg"], 2),
            ]
            for row in checkpoint
        ],
    )

    def audio_table(rows: list[dict[str, Any]]) -> str:
        return markdown_table(
            [
                "System",
                "CLAP ↑",
                "Paired CLAP ↑",
                "FD-CLAP ↓",
                "FAD-VGGish ↓",
                "FD-PANN ↓",
                "KL-PANN ↓",
                "DoA ↓",
                "Azimuth ↓",
                "Elevation ↓",
                "Trajectory ↓",
                "Activity IoU ↑",
            ],
            [
                [
                    row["system"],
                    fmt(row["clap"]),
                    fmt(row["paired_clap"]),
                    fmt(row["fd_clap"]),
                    fmt(row["fad_vggish"]),
                    fmt(row["fd_pann"]),
                    fmt(row["kl_pann"]),
                    fmt(row["doa_error_deg"], 2),
                    fmt(row["azimuth_mae_deg"], 2),
                    fmt(row["elevation_mae_deg"], 2),
                    fmt(row["trajectory_extent_error_deg"], 2),
                    fmt(row["activity_iou"]),
                ]
                for row in rows
            ],
        )

    speech_table = markdown_table(
        [
            "System",
            "WER ↓",
            "Seen WER ↓",
            "Unseen WER ↓",
            "CER ↓",
            "UTMOS ↑",
            "DoA ↓",
            "Azimuth ↓",
            "Elevation ↓",
            "Activity IoU ↑",
        ],
        [
            [
                row["system"],
                fmt(row["wer"]),
                fmt(row["seen_speaker_wer"]),
                fmt(row["unseen_speaker_wer"]),
                fmt(row["cer"]),
                fmt(row["utmos"]),
                fmt(row["doa_error_deg"], 2),
                fmt(row["azimuth_mae_deg"], 2),
                fmt(row["elevation_mae_deg"], 2),
                fmt(row["activity_iou"]),
            ]
            for row in speech
        ],
    )

    return f"""# P10 Balanced Cross-System Evaluation — Complete Tables

The frozen benchmark contains 400 single-source Music, 400 Sound, and 400 Speech scenes. Quality metrics use the native FOA W channel for our model and the frozen mono view for public baselines. Spatial metrics are reported only for native FOA systems; `N/A` means the metric is not applicable, not zero.

## Checkpoint overview

{overview}

## Music — all systems

{audio_table(music)}

## Sound — all systems

{audio_table(sound)}

## Speech — all systems

{speech_table}

## Interpretation

- Use **100k** as the primary unified checkpoint: it is strongest for Speech and Music distributional quality, while remaining close to 80k on Sound.
- Retain **80k** as the Sound-specialist checkpoint: it has the best Sound paired CLAP, FAD, FD-PANN, and activity IoU among our checkpoints.
- Public mono/stereo systems have `N/A` spatial fields. Their channels were not duplicated or treated as FOA.
- Absolute text CLAP and paired/distributional metrics answer different questions; no single metric should determine the checkpoint alone.

## Scope and caveats

These are matched diagnostic estimates over 400 clips per domain. They are suitable for checkpoint selection and controlled baseline comparison, but do not replace a larger publication-scale benchmark. Speech includes 171 seen-speaker and 229 unseen-speaker scenes and reports both subgroups.
"""


def latex_table(
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
) -> str:
    columns = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(latex_escape(value) for value in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(latex_escape(value) for value in row) + r" \\" for row in rows
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            rf"\caption{{{latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def build_latex(
    checkpoint: list[dict[str, Any]],
    music: list[dict[str, Any]],
    sound: list[dict[str, Any]],
    speech: list[dict[str, Any]],
) -> str:
    overview_headers = [
        "Step",
        "M-CLAP ↑",
        "M-FAD ↓",
        "M-DoA ↓",
        "S-CLAP ↑",
        "S-FAD ↓",
        "S-DoA ↓",
        "WER ↓",
        "UTMOS ↑",
        "Sp-DoA ↓",
    ]
    overview_rows = [
        [
            row["checkpoint"],
            fmt(row["music_clap"]),
            fmt(row["music_fad"]),
            fmt(row["music_doa_deg"], 2),
            fmt(row["sound_clap"]),
            fmt(row["sound_fad"]),
            fmt(row["sound_doa_deg"], 2),
            fmt(row["speech_wer"]),
            fmt(row["speech_utmos"]),
            fmt(row["speech_doa_deg"], 2),
        ]
        for row in checkpoint
    ]

    audio_headers = [
        "System",
        "CLAP ↑",
        "P-CLAP ↑",
        "FD-C ↓",
        "FAD ↓",
        "FD-P ↓",
        "KL-P ↓",
        "DoA ↓",
        "Traj. ↓",
        "IoU ↑",
    ]

    def audio_latex_rows(rows: list[dict[str, Any]]) -> list[list[str]]:
        return [
            [
                row["system"],
                fmt(row["clap"]),
                fmt(row["paired_clap"]),
                fmt(row["fd_clap"]),
                fmt(row["fad_vggish"]),
                fmt(row["fd_pann"]),
                fmt(row["kl_pann"]),
                fmt(row["doa_error_deg"], 2),
                fmt(row["trajectory_extent_error_deg"], 2),
                fmt(row["activity_iou"]),
            ]
            for row in rows
        ]

    speech_headers = [
        "System",
        "WER ↓",
        "Seen ↓",
        "Unseen ↓",
        "CER ↓",
        "UTMOS ↑",
        "DoA ↓",
        "IoU ↑",
    ]
    speech_latex_rows = [
        [
            row["system"],
            fmt(row["wer"]),
            fmt(row["seen_speaker_wer"]),
            fmt(row["unseen_speaker_wer"]),
            fmt(row["cer"]),
            fmt(row["utmos"]),
            fmt(row["doa_error_deg"], 2),
            fmt(row["activity_iou"]),
        ]
        for row in speech
    ]
    return "\n\n".join(
        [
            "% Requires \\usepackage{booktabs,graphicx}",
            latex_table(
                "Checkpoint comparison on the balanced P10 test panel.",
                "tab:p10_checkpoint_overview",
                overview_headers,
                overview_rows,
            ),
            latex_table(
                "Music cross-system comparison on 400 matched prompts.",
                "tab:p10_music_all",
                audio_headers,
                audio_latex_rows(music),
            ),
            latex_table(
                "Sound cross-system comparison on 400 matched prompts.",
                "tab:p10_sound_all",
                audio_headers,
                audio_latex_rows(sound),
            ),
            latex_table(
                "Speech cross-system comparison on 400 matched prompts.",
                "tab:p10_speech_all",
                speech_headers,
                speech_latex_rows,
            ),
        ]
    ) + "\n"


def artifact_payload(
    source_path: Path,
    checkpoint: list[dict[str, Any]],
    music: list[dict[str, Any]],
    sound: list[dict[str, Any]],
    speech: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat()
    source_id = "p10_final_metrics"
    source = {
        "id": source_id,
        "label": "P10 balanced cross-system final metrics",
        "path": str(source_path),
        "query": {
            "engine": "DuckDB",
            "language": "SQL",
            "sql": (
                "SELECT * FROM read_json_auto('"
                + str(source_path).replace("'", "''")
                + "', format = 'auto');"
            ),
            "description": "Deterministic extraction from the completed P10 cross-system metric artifact.",
            "tables_used": [source_path.name],
            "filters": [
                "400 frozen single-source Music scenes",
                "400 frozen single-source Sound scenes",
                "400 frozen single-source Speech scenes",
                "same semantic prompt and target duration for each compatible system",
            ],
            "metric_definitions": [
                "CLAP: mean cosine similarity between generated audio and prompt text; higher is better.",
                "Paired CLAP: mean cosine similarity between generated audio and its paired reference; higher is better.",
                "FAD-VGGish, FD-CLAP, FD-PANN: feature-distribution distances to references; lower is better.",
                "WER and CER: corpus edit error rates against the exact transcript; lower is better.",
                "DoA error: mean spherical direction error in degrees against the ScenePlan trajectory; lower is better.",
            ],
        },
    }

    overview_columns = [
        ("checkpoint_step", "Step"),
        ("music_clap", "Music CLAP"),
        ("music_fad", "Music FAD"),
        ("music_doa_deg", "Music DoA°"),
        ("sound_clap", "Sound CLAP"),
        ("sound_fad", "Sound FAD"),
        ("sound_doa_deg", "Sound DoA°"),
        ("speech_wer", "Speech WER"),
        ("speech_utmos", "Speech UTMOS"),
        ("speech_doa_deg", "Speech DoA°"),
    ]
    audio_columns = [
        ("order", "#"),
        ("system", "System"),
        ("clap", "CLAP ↑"),
        ("paired_clap", "Paired CLAP ↑"),
        ("fd_clap", "FD-CLAP ↓"),
        ("fad_vggish", "FAD-VGGish ↓"),
        ("fd_pann", "FD-PANN ↓"),
        ("kl_pann", "KL-PANN ↓"),
        ("doa_error_deg", "DoA° ↓"),
        ("trajectory_extent_error_deg", "Trajectory° ↓"),
        ("activity_iou", "Activity IoU ↑"),
    ]
    speech_columns = [
        ("order", "#"),
        ("system", "System"),
        ("wer", "WER ↓"),
        ("seen_speaker_wer", "Seen WER ↓"),
        ("unseen_speaker_wer", "Unseen WER ↓"),
        ("cer", "CER ↓"),
        ("utmos", "UTMOS ↑"),
        ("doa_error_deg", "DoA° ↓"),
        ("activity_iou", "Activity IoU ↑"),
    ]

    def columns(values: list[tuple[str, str]]) -> list[dict[str, Any]]:
        return [
            {
                "field": field,
                "label": label,
                "format": "number" if field != "system" else None,
                "type": "number" if field != "system" else "text",
            }
            for field, label in values
        ]

    tables = [
        {
            "id": "checkpoint_overview",
            "title": "Checkpoint overview",
            "subtitle": "Five checkpoints on the same 400/400/400 frozen panel",
            "dataset": "checkpoint_overview",
            "defaultSort": {"field": "checkpoint_step", "direction": "asc"},
            "density": "dense",
            "sourceId": source_id,
            "layout": "full",
            "columns": columns(overview_columns),
        },
        *[
            {
                "id": f"{domain}_all_systems",
                "title": f"{domain.title()} — all systems",
                "subtitle": f"400 matched {domain} scenes; native FOA spatial metrics only",
                "dataset": f"{domain}_rows",
                "defaultSort": {"field": "order", "direction": "asc"},
                "density": "dense",
                "sourceId": source_id,
                "layout": "full",
                "columns": columns(audio_columns if domain != "speech" else speech_columns),
            }
            for domain in ("music", "sound", "speech")
        ],
    ]
    for table in tables:
        for column in table["columns"]:
            if column.get("format") is None:
                column.pop("format", None)

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "P10 Balanced Cross-System Evaluation — Complete Results",
        "description": "Complete checkpoint, public-baseline, VAE-ceiling, and native-FOA results on the frozen balanced panel.",
        "generatedAt": generated_at,
        "sources": [source],
        "charts": [
            {
                "id": "speech_wer_by_checkpoint",
                "title": "Speech WER across checkpoints",
                "subtitle": "Corpus WER on 400 frozen Speech scenes; lower is better",
                "type": "bar",
                "dataset": "checkpoint_overview",
                "sourceId": source_id,
                "layout": "full",
                "encodings": {
                    "x": {
                        "field": "checkpoint",
                        "type": "ordinal",
                        "label": "Checkpoint",
                    },
                    "y": {
                        "field": "speech_wer",
                        "type": "quantitative",
                        "label": "Corpus WER",
                        "format": "number",
                    },
                    "tooltip": [
                        {"field": "speech_seen_wer", "label": "Seen-speaker WER"},
                        {"field": "speech_unseen_wer", "label": "Unseen-speaker WER"},
                        {"field": "speech_utmos", "label": "UTMOS"},
                        {"field": "speech_doa_deg", "label": "DoA error", "unit": "deg"},
                    ],
                },
                "valueFormat": "number",
                "surface": {
                    "orientation": "vertical",
                    "showLegend": False,
                    "showValueLabels": True,
                },
            }
        ],
        "tables": tables,
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# P10 Balanced Cross-System Evaluation — Complete Results",
                "layout": "full",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": source_id,
                "layout": "full",
                "body": (
                    "## Technical summary\n\n"
                    "**100k is the recommended unified checkpoint.** It gives the strongest "
                    "Speech result (WER 0.289, UTMOS 3.073, DoA error 9.16°) and the best "
                    "Music FAD among our checkpoints (0.841). **80k remains the Sound-specialist "
                    "checkpoint**, with Sound paired CLAP 0.470 and FAD 1.122. Public baselines "
                    "retain higher absolute text CLAP, while our late checkpoints are closer to "
                    "the paired reference and reference feature distribution."
                ),
            },
            {
                "id": "checkpoint_finding",
                "type": "markdown",
                "sourceId": source_id,
                "layout": "full",
                "body": (
                    "## Speech improves consistently through 100k\n\n"
                    "Speech corpus WER falls from 0.831 at 20k to 0.289 at 100k. The final "
                    "checkpoint still trails Qwen3-TTS (0.091 WER), so continued Speech "
                    "optimization remains justified even though spatial control is stable."
                ),
            },
            {
                "id": "speech_wer_chart_block",
                "type": "chart",
                "chartId": "speech_wer_by_checkpoint",
                "layout": "full",
            },
            {
                "id": "overview_section",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## One table aligns all five checkpoints\n\n"
                    "The overview keeps one representative semantic/distribution metric and one "
                    "native-FOA direction metric per domain. Use the domain tables below for the "
                    "complete metric set."
                ),
            },
            {"id": "overview_table_block", "type": "table", "tableId": "checkpoint_overview", "layout": "full"},
            {
                "id": "music_section",
                "type": "markdown",
                "sourceId": source_id,
                "layout": "full",
                "body": (
                    "## Music quality converges late, while text CLAP remains a gap\n\n"
                    "100k has our best Music FAD (0.841) and FD-CLAP (0.156). Stable Audio Open "
                    "and AudioX-Turbo score higher on absolute text CLAP, but are farther from the "
                    "paired references and feature distribution on this panel."
                ),
            },
            {"id": "music_table_block", "type": "table", "tableId": "music_all_systems", "layout": "full"},
            {
                "id": "sound_section",
                "type": "markdown",
                "sourceId": source_id,
                "layout": "full",
                "body": (
                    "## Sound peaks at 80k on most reference-alignment metrics\n\n"
                    "80k leads our Sound paired CLAP, FAD, FD-PANN, and activity IoU. The 100k "
                    "checkpoint is close and has slightly better absolute CLAP and KL-PANN, which "
                    "is why 100k remains defensible as the single deployment checkpoint."
                ),
            },
            {"id": "sound_table_block", "type": "table", "tableId": "sound_all_systems", "layout": "full"},
            {
                "id": "speech_section",
                "type": "markdown",
                "sourceId": source_id,
                "layout": "full",
                "body": (
                    "## Speech improves monotonically but retains an unseen-speaker gap\n\n"
                    "At 100k, seen-speaker WER is 0.240 and unseen-speaker WER is 0.316. Qwen3-TTS "
                    "is the stronger pure-TTS reference, while our model uniquely supplies native "
                    "FOA direction and activity control."
                ),
            },
            {"id": "speech_table_block", "type": "table", "tableId": "speech_all_systems", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Scope, definitions, and comparison basis\n\n"
                    "The panel contains 400 single-source scenes per domain. Music/Sound quality "
                    "uses our native FOA W channel; mono baselines are unchanged and stereo "
                    "baselines use an arithmetic-mean mono view. Spatial scores apply only to "
                    "native WYZX/ACN/SN3D FOA. Speech includes 171 seen-speaker and 229 "
                    "unseen-speaker scenes."
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Matched generation and deterministic scoring\n\n"
                    "Every compatible system receives the same frozen semantic prompt and target "
                    "duration. Five ScenePlan-FOA checkpoints, the frozen VAE reconstruction, "
                    "ground truth, and six runnable public baselines are scored with one metric "
                    "implementation. Baseline spatial fields remain N/A rather than duplicating "
                    "mono channels into pseudo-FOA."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Limitations and robustness\n\n"
                    "These are matched diagnostic estimates over 400 clips per domain, not the "
                    "final publication-scale population benchmark. Absolute text CLAP, paired "
                    "CLAP, and feature-distribution distances measure different properties and "
                    "should not be collapsed into one unvalidated scalar score. The generation "
                    "audit passed for all 3,600 baseline outputs and all numeric metric leaves are finite."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. Use 100k for the unified P10 checkpoint and retain 80k for Sound-focused listening checks.\n"
                    "2. Diagnose the absolute text-CLAP gap without sacrificing reference-distribution quality.\n"
                    "3. Target Speech transcript fidelity and unseen-speaker generalization before claiming parity with specialist TTS."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "layout": "full",
                "body": (
                    "## Further questions\n\n"
                    "Would the 80k-versus-100k Sound difference persist on the larger external test "
                    "set, and does listener preference agree with paired CLAP/FAD rather than "
                    "absolute text CLAP?"
                ),
            },
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "checkpoint_overview": checkpoint,
                "music_rows": music,
                "sound_rows": sound,
                "speech_rows": speech,
            },
        },
        "sources": [source],
        "package_info": {
            "report_id": "p10_balanced_cross_system_complete_results",
            "snapshot_type": "frozen evaluation",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    args = parser.parse_args()
    root = args.benchmark_root.expanduser().resolve(strict=True)
    metrics_path = root / "metrics" / "CROSS_SYSTEM_METRICS.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        metrics.get("status") != "PASS"
        or metrics.get("panel_rows") != 1200
        or metrics.get("domain_counts")
        != {"music": 400, "sound": 400, "speech": 400}
    ):
        raise RuntimeError("the frozen cross-system metric artifact is incomplete")

    music = audio_rows(metrics, "music")
    sound = audio_rows(metrics, "sound")
    speech = speech_rows(metrics)
    all_rows = music + sound + speech
    checkpoint = checkpoint_rows(all_rows)
    output_root = root / "tables"

    atomic_text(
        output_root / "ALL_RESULTS_TABLES.md",
        build_markdown(checkpoint, music, sound, speech),
    )
    atomic_text(
        output_root / "ALL_RESULTS_TABLES.tex",
        build_latex(checkpoint, music, sound, speech),
    )

    csv_path = output_root / "ALL_RESULTS_LONG.csv"
    csv_temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    csv_temporary.replace(csv_path)

    checkpoint_path = output_root / "CHECKPOINT_OVERVIEW.csv"
    checkpoint_temporary = checkpoint_path.with_name(
        f".{checkpoint_path.name}.tmp-{os.getpid()}"
    )
    with checkpoint_temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checkpoint[0]))
        writer.writeheader()
        writer.writerows(checkpoint)
    checkpoint_temporary.replace(checkpoint_path)

    artifact = artifact_payload(metrics_path, checkpoint, music, sound, speech)
    atomic_json(output_root / "REPORT_ARTIFACT.json", artifact)
    summary = {
        "status": "PASS",
        "checkpoint_rows": len(checkpoint),
        "music_rows": len(music),
        "sound_rows": len(sound),
        "speech_rows": len(speech),
        "outputs": {
            "markdown": str(output_root / "ALL_RESULTS_TABLES.md"),
            "csv": str(csv_path),
            "checkpoint_csv": str(checkpoint_path),
            "latex": str(output_root / "ALL_RESULTS_TABLES.tex"),
            "artifact": str(output_root / "REPORT_ARTIFACT.json"),
        },
    }
    atomic_json(output_root / "ALL_RESULTS_BUILD_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
