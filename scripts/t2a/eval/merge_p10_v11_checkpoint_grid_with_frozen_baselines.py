#!/usr/bin/env python3
"""Merge new native P10 metrics with panel-identical frozen baselines.

Public systems are not regenerated: their audio, prompt, duration, panel and
metric protocol are unchanged.  Their frozen metric leaves are copied with
explicit provenance, while every 110k--150k ``ours`` leaf is rebuilt from the
new inference outputs.  The script also emits compact comparison tables and a
transparent domain-balanced checkpoint-selection rank.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_dit_p10_panel_common import (  # noqa: E402
    checkpoint_steps,
    load_panel,
    read_jsonl,
    sha256_file,
    summarize,
)


DEFAULT_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_balanced_1200_ckpt110k_150k_semantic_v2"
)
DEFAULT_FROZEN = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_balanced_1200_ckpt20k_100k_v1/cross_system_baselines"
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _mean(value: dict[str, Any]) -> float | None:
    item = value.get("mean")
    return None if item is None else float(item)


def _aggregate_speech(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row["generated_errors"] for row in rows]
    return {
        "rows": len(rows),
        "wer": summarize(item["wer"] for item in errors),
        "cer": summarize(item["cer"] for item in errors),
        "corpus_wer": sum(int(item["word_edits"]) for item in errors)
        / max(sum(int(item["reference_words"]) for item in errors), 1),
        "corpus_cer": sum(int(item["char_edits"]) for item in errors)
        / max(sum(int(item["reference_chars"]) for item in errors), 1),
        "utmos": summarize(row.get("generated_utmos") for row in rows),
    }


def _speech_leaf(
    step: int,
    rows: list[dict[str, Any]],
    panel_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    chosen = [row for row in rows if int(row["checkpoint_step"]) == step]
    if len(chosen) != 400:
        raise RuntimeError(f"step {step}: expected 400 speech rows, found {len(chosen)}")
    decorated = []
    for row in chosen:
        panel = panel_by_id[row["panel_id"]]
        item = copy.deepcopy(row)
        item["speech_seen_speaker"] = bool(panel["speech_seen_speaker"])
        item["speech_speaker_key"] = str(panel["speech_speaker_key"])
        item["length_bucket"] = int(panel["length_bucket"])
        decorated.append(item)
    overall = _aggregate_speech(decorated)
    overall.update(
        {
            "display_name": f"Ours ScenePlan-FOA ({step // 1000}k, semantic-v2)",
            "subgroups": {
                "seen_speaker": _aggregate_speech(
                    [row for row in decorated if row["speech_seen_speaker"]]
                ),
                "unseen_speaker": _aggregate_speech(
                    [row for row in decorated if not row["speech_seen_speaker"]]
                ),
                "latent_le_432": _aggregate_speech(
                    [row for row in decorated if row["length_bucket"] <= 432]
                ),
                "latent_433_648": _aggregate_speech(
                    [row for row in decorated if row["length_bucket"] > 432]
                ),
            },
            "per_output": [
                {
                    "panel_id": row["panel_id"],
                    "asr": row["generated_asr"]["text"],
                    "reference": row["exact_transcript"],
                    "wer": row["generated_errors"]["wer"],
                    "cer": row["generated_errors"]["cer"],
                    "word_errors": row["generated_errors"]["word_edits"],
                    "reference_words": row["generated_errors"]["reference_words"],
                    "char_errors": row["generated_errors"]["char_edits"],
                    "reference_chars": row["generated_errors"]["reference_chars"],
                    "utmos": row.get("generated_utmos"),
                    "speech_seen_speaker": row["speech_seen_speaker"],
                    "speech_speaker_key": row["speech_speaker_key"],
                    "length_bucket": row["length_bucket"],
                    "generated_foa_path": row["generated_foa_path"],
                }
                for row in decorated
            ],
        }
    )
    return overall


def _ours_id(step: int) -> str:
    return f"ours_sceneplan_foa_{step}"


def _audio_leaf(
    step: int,
    domain: str,
    clap: dict[str, Any],
    distributional: dict[str, Any],
) -> dict[str, Any]:
    c = clap["aggregates"][str(step)][domain]
    d = distributional["aggregates"][str(step)][domain]
    return {
        "display_name": f"Ours ScenePlan-FOA ({step // 1000}k, semantic-v2)",
        "rows": int(c["rows"]),
        "clap_text_audio_cosine": c["clap_text_audio_cosine"],
        "paired_generated_reference_clap_cosine": c[
            "paired_generated_reference_clap_cosine"
        ],
        "fd_clap_diagnostic": float(c["fd_clap"]),
        "fad_vggish_diagnostic": float(d["fad_vggish_diagnostic"]),
        "fd_pann_diagnostic": float(d["fd_pann"]),
        "paired_kl_pann_softmax": d["paired_kl_pann_softmax"],
    }


def _spatial_leaf(core: dict[str, Any], step: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for domain in ("music", "sound", "speech"):
        value = core["aggregates"][str(step)][domain]
        output[domain] = {
            "doa_spherical_error_deg": _mean(value["plan_spherical_error_mean_deg"]),
            "azimuth_mae_deg": _mean(value["plan_azimuth_circular_mae_deg"]),
            "elevation_mae_deg": _mean(value["plan_elevation_mae_deg"]),
            "trajectory_extent_error_deg": _mean(
                value.get("plan_trajectory_extent_error_deg", {"mean": None})
            ),
            "activity_iou": _mean(value["activity_temporal_iou"]),
        }
    return output


def _rank(values: dict[int, float], *, higher: bool) -> dict[int, float]:
    ordered = sorted(set(values.values()), reverse=higher)
    return {step: float(ordered.index(value) + 1) for step, value in values.items()}


def _mean_values(values: Iterable[float]) -> float:
    kept = list(values)
    return sum(kept) / len(kept)


def _selection(metrics: dict[str, Any], steps: tuple[int, ...]) -> dict[str, Any]:
    blocks: dict[str, list[tuple[str, bool, dict[int, float]]]] = {
        "music": [],
        "sound": [],
        "speech": [],
        "spatial": [],
    }
    for domain in ("music", "sound"):
        values = metrics["audio_domains"][domain]
        blocks[domain] = [
            (
                "text_clap",
                True,
                {step: _mean(values[_ours_id(step)]["clap_text_audio_cosine"]) for step in steps},
            ),
            (
                "paired_clap",
                True,
                {
                    step: _mean(
                        values[_ours_id(step)][
                            "paired_generated_reference_clap_cosine"
                        ]
                    )
                    for step in steps
                },
            ),
            (
                "fad_vggish",
                False,
                {step: values[_ours_id(step)]["fad_vggish_diagnostic"] for step in steps},
            ),
            (
                "fd_pann",
                False,
                {step: values[_ours_id(step)]["fd_pann_diagnostic"] for step in steps},
            ),
            (
                "kl_pann",
                False,
                {
                    step: _mean(values[_ours_id(step)]["paired_kl_pann_softmax"])
                    for step in steps
                },
            ),
        ]
    speech = metrics["speech"]
    blocks["speech"] = [
        ("corpus_wer", False, {step: speech[_ours_id(step)]["corpus_wer"] for step in steps}),
        ("corpus_cer", False, {step: speech[_ours_id(step)]["corpus_cer"] for step in steps}),
        ("utmos", True, {step: _mean(speech[_ours_id(step)]["utmos"]) for step in steps}),
    ]
    spatial = metrics["native_foa_spatial"]
    blocks["spatial"] = [
        (
            "mean_plan_doa",
            False,
            {
                step: _mean_values(
                    spatial[_ours_id(step)][domain]["doa_spherical_error_deg"]
                    for domain in ("music", "sound", "speech")
                )
                for step in steps
            },
        ),
        (
            "mean_activity_iou",
            True,
            {
                step: _mean_values(
                    spatial[_ours_id(step)][domain]["activity_iou"]
                    for domain in ("music", "sound", "speech")
                )
                for step in steps
            },
        ),
    ]
    block_ranks: dict[str, dict[int, float]] = {}
    leaves: dict[str, Any] = {}
    for block, specifications in blocks.items():
        ranks_by_metric = {
            name: _rank(values, higher=higher)
            for name, higher, values in specifications
        }
        block_ranks[block] = {
            step: _mean_values(rank[step] for rank in ranks_by_metric.values())
            for step in steps
        }
        leaves[block] = {
            name: {
                "higher_is_better": higher,
                "values": {str(step): values[step] for step in steps},
                "ranks": {str(step): ranks_by_metric[name][step] for step in steps},
            }
            for name, higher, values in specifications
        }
    unified = {
        step: _mean_values(block_ranks[block][step] for block in blocks)
        for step in steps
    }
    best = min(steps, key=lambda step: (unified[step], -step))
    return {
        "status": "PASS",
        "selection_is_for_listening_not_a_publication_metric": True,
        "method": (
            "equal mean rank within Music, Sound, Speech, and spatial-control blocks; "
            "then equal mean rank across the four blocks"
        ),
        "checkpoint_steps": list(steps),
        "block_metric_details": leaves,
        "block_mean_ranks": {
            block: {str(step): value for step, value in ranks.items()}
            for block, ranks in block_ranks.items()
        },
        "unified_mean_rank": {str(step): unified[step] for step in steps},
        "recommended_checkpoint_step": best,
        "warning": "Inspect per-domain metrics and listening samples; do not treat this rank as a paper metric.",
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def _markdown(metrics: dict[str, Any], selection: dict[str, Any]) -> str:
    steps = tuple(int(step) for step in metrics["checkpoint_steps"])
    spatial = metrics["native_foa_spatial"]
    overview = []
    for step in steps:
        system = _ours_id(step)
        music = metrics["audio_domains"]["music"][system]
        sound = metrics["audio_domains"]["sound"][system]
        speech = metrics["speech"][system]
        mean_doa = _mean_values(
            spatial[system][domain]["doa_spherical_error_deg"]
            for domain in ("music", "sound", "speech")
        )
        overview.append(
            [
                f"{step // 1000}k",
                _fmt(_mean(music["clap_text_audio_cosine"])),
                _fmt(music["fad_vggish_diagnostic"]),
                _fmt(_mean(sound["clap_text_audio_cosine"])),
                _fmt(sound["fad_vggish_diagnostic"]),
                _fmt(speech["corpus_wer"]),
                _fmt(speech["corpus_cer"]),
                _fmt(_mean(speech["utmos"])),
                _fmt(mean_doa, 2),
                _fmt(selection["unified_mean_rank"][str(step)], 2),
            ]
        )

    def audio_table(domain: str) -> str:
        values = metrics["audio_domains"][domain]
        order = ["ground_truth", "vae_codec_ceiling", *(_ours_id(step) for step in steps)]
        order += [key for key in values if key not in order]
        rows = []
        for system in order:
            value = values[system]
            position = spatial.get(system, {}).get(domain, {})
            rows.append(
                [
                    value["display_name"],
                    _fmt(_mean(value["clap_text_audio_cosine"])),
                    _fmt(_mean(value["paired_generated_reference_clap_cosine"])),
                    _fmt(value["fd_clap_diagnostic"]),
                    _fmt(value["fad_vggish_diagnostic"]),
                    _fmt(value["fd_pann_diagnostic"]),
                    _fmt(_mean(value["paired_kl_pann_softmax"])),
                    _fmt(position.get("doa_spherical_error_deg"), 2),
                    _fmt(position.get("trajectory_extent_error_deg"), 2),
                    _fmt(position.get("activity_iou")),
                ]
            )
        return _table(
            ["System", "CLAP ↑", "Paired CLAP ↑", "FD-CLAP ↓", "FAD ↓", "FD-PANN ↓", "KL-PANN ↓", "DoA° ↓", "Traj° ↓", "IoU ↑"],
            rows,
        )

    speech_values = metrics["speech"]
    speech_order = ["ground_truth", "vae_codec_ceiling", *(_ours_id(step) for step in steps)]
    speech_order += [key for key in speech_values if key not in speech_order]
    speech_rows = []
    for system in speech_order:
        value = speech_values[system]
        position = spatial.get(system, {}).get("speech", {})
        speech_rows.append(
            [
                value["display_name"],
                _fmt(value["corpus_wer"]),
                _fmt(value["subgroups"]["seen_speaker"]["corpus_wer"]),
                _fmt(value["subgroups"]["unseen_speaker"]["corpus_wer"]),
                _fmt(value["corpus_cer"]),
                _fmt(_mean(value["utmos"])),
                _fmt(position.get("doa_spherical_error_deg"), 2),
                _fmt(position.get("activity_iou")),
            ]
        )

    best = int(selection["recommended_checkpoint_step"])
    return "\n\n".join(
        [
            "# P10 semantic-v2 checkpoint grid and frozen-baseline comparison",
            (
                "The exact frozen 400 Music + 400 Sound + 400 Speech panel and per-sample "
                "noise are shared by 110k--150k. Public baseline generations/metrics are "
                "reused from the panel-identical frozen benchmark; mono/stereo systems remain "
                "N/A for spatial metrics."
            ),
            "## Checkpoint overview",
            _table(
                ["Step", "Music CLAP ↑", "Music FAD ↓", "Sound CLAP ↑", "Sound FAD ↓", "Speech WER ↓", "Speech CER ↓", "UTMOS ↑", "Mean DoA° ↓", "Balanced rank ↓"],
                overview,
            ),
            f"Transparent equal-block listening recommendation: **{best // 1000}k**.",
            "## Music",
            audio_table("music"),
            "## Sound",
            audio_table("sound"),
            "## Speech",
            _table(
                ["System", "WER ↓", "Seen WER ↓", "Unseen WER ↓", "CER ↓", "UTMOS ↑", "DoA° ↓", "IoU ↑"],
                speech_rows,
            ),
            "## Protocol notes",
            "- Ours is native WYZX/ACN/SN3D FOA; Music/Sound quality uses W.\n- Public mono/stereo baselines are compared only in their applicable quality lane.\n- Speech v2 uses `speaker says: exact transcript`; event/speech masks, ScenePlans, trajectories and noise seeds are unchanged.\n- FAD/FD use 400 clips per domain and are checkpoint-selection diagnostics.",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--frozen-baseline-root", type=Path, default=DEFAULT_FROZEN)
    args = parser.parse_args()
    root = args.eval_root.expanduser().resolve(strict=True)
    frozen = args.frozen_baseline_root.expanduser().resolve(strict=True)
    steps = checkpoint_steps(root)
    if steps != (110_000, 120_000, 130_000, 140_000, 150_000):
        raise RuntimeError(f"unexpected checkpoint grid: {steps}")
    contract = json.loads((root / "EVAL_CONTRACT.json").read_text(encoding="utf-8"))
    frozen_contract = json.loads(
        (frozen / "BENCHMARK_CONTRACT.json").read_text(encoding="utf-8")
    )
    if contract["test_set"]["panel_sha256"] != frozen_contract["source_panel_sha256"]:
        raise RuntimeError("new and frozen baseline panels differ")
    if int(contract["sampling"]["semantic_caption_compiler_version"]) != 2:
        raise RuntimeError("new checkpoint grid is not semantic-caption v2")

    metric_root = root / "metrics"
    required = {
        "core": metric_root / "CORE_SUMMARY.json",
        "clap": metric_root / "CLAP_SUMMARY.json",
        "distributional": metric_root / "DISTRIBUTIONAL_SUMMARY.json",
        "speech": metric_root / "SPEECH_SUMMARY.json",
        "speech_rows": metric_root / "speech_per_output.jsonl",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    core = json.loads(required["core"].read_text(encoding="utf-8"))
    clap = json.loads(required["clap"].read_text(encoding="utf-8"))
    distributional = json.loads(
        required["distributional"].read_text(encoding="utf-8")
    )
    speech_summary = json.loads(required["speech"].read_text(encoding="utf-8"))
    if any(value.get("status") != "PASS" for value in (core, clap, distributional, speech_summary)):
        raise RuntimeError("one or more native metric stages did not pass")
    speech_rows = read_jsonl(required["speech_rows"])
    panel = load_panel(root)
    panel_by_id = {row["panel_id"]: row for row in panel}

    frozen_metrics_path = frozen / "metrics/CROSS_SYSTEM_METRICS.json"
    merged = copy.deepcopy(json.loads(frozen_metrics_path.read_text(encoding="utf-8")))
    for domain in ("music", "sound"):
        merged["audio_domains"][domain] = {
            key: value
            for key, value in merged["audio_domains"][domain].items()
            if not key.startswith("ours_sceneplan_foa_")
        }
    merged["speech"] = {
        key: value
        for key, value in merged["speech"].items()
        if not key.startswith("ours_sceneplan_foa_")
    }
    merged["native_foa_spatial"] = {
        key: value
        for key, value in merged["native_foa_spatial"].items()
        if not key.startswith("ours_sceneplan_foa_")
    }
    for step in steps:
        system = _ours_id(step)
        for domain in ("music", "sound"):
            merged["audio_domains"][domain][system] = _audio_leaf(
                step, domain, clap, distributional
            )
        merged["speech"][system] = _speech_leaf(
            step, speech_rows, panel_by_id
        )
        merged["native_foa_spatial"][system] = _spatial_leaf(core, step)
    merged.update(
        {
            "schema_version": 3,
            "status": "PASS",
            "checkpoint_steps": list(steps),
            "ours_system_ids": [_ours_id(step) for step in steps],
            "panel_sha256": contract["test_set"]["panel_sha256"],
            "semantic_caption_compiler_version": 2,
            "baseline_reuse_provenance": {
                "frozen_metrics": str(frozen_metrics_path),
                "frozen_metrics_sha256": sha256_file(frozen_metrics_path),
                "panel_identical": True,
                "public_outputs_and_metric_leaves_reused": True,
                "new_ours_metric_leaves_recomputed": True,
            },
        }
    )
    selection = _selection(merged, steps)
    merged["checkpoint_selection"] = selection

    output_root = root / "cross_system_baselines"
    metrics_path = output_root / "metrics/CROSS_SYSTEM_METRICS.json"
    _atomic_json(metrics_path, merged)
    _atomic_json(output_root / "metrics/BEST_CHECKPOINT.json", selection)
    markdown = _markdown(merged, selection)
    _atomic_text(output_root / "tables/ALL_RESULTS_TABLES.md", markdown)

    csv_path = output_root / "tables/CHECKPOINT_OVERVIEW.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    fields = [
        "step",
        "music_clap",
        "music_fad",
        "sound_clap",
        "sound_fad",
        "speech_wer",
        "speech_cer",
        "speech_utmos",
        "mean_doa_deg",
        "balanced_rank",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            system = _ours_id(step)
            writer.writerow(
                {
                    "step": step,
                    "music_clap": _mean(merged["audio_domains"]["music"][system]["clap_text_audio_cosine"]),
                    "music_fad": merged["audio_domains"]["music"][system]["fad_vggish_diagnostic"],
                    "sound_clap": _mean(merged["audio_domains"]["sound"][system]["clap_text_audio_cosine"]),
                    "sound_fad": merged["audio_domains"]["sound"][system]["fad_vggish_diagnostic"],
                    "speech_wer": merged["speech"][system]["corpus_wer"],
                    "speech_cer": merged["speech"][system]["corpus_cer"],
                    "speech_utmos": _mean(merged["speech"][system]["utmos"]),
                    "mean_doa_deg": _mean_values(
                        merged["native_foa_spatial"][system][domain]["doa_spherical_error_deg"]
                        for domain in ("music", "sound", "speech")
                    ),
                    "balanced_rank": selection["unified_mean_rank"][str(step)],
                }
            )
    temporary.replace(csv_path)

    summary = {
        "status": "PASS",
        "recommended_checkpoint_step": selection["recommended_checkpoint_step"],
        "checkpoint_steps": list(steps),
        "panel_rows": len(panel),
        "panel_sha256": contract["test_set"]["panel_sha256"],
        "metrics": str(metrics_path),
        "table": str(output_root / "tables/ALL_RESULTS_TABLES.md"),
        "checkpoint_csv": str(csv_path),
    }
    _atomic_json(output_root / "MERGE_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
