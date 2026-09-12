#!/usr/bin/env python3
"""Select and audit final P10 listening examples after quantitative scoring."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import soundfile as sf


DEFAULT_EVAL = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_balanced_1200_ckpt110k_150k_semantic_v2"
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _percentiles(values: dict[str, float], *, higher: bool) -> dict[str, float]:
    ordered = sorted(values.values(), reverse=higher)
    denominator = max(len(ordered) - 1, 1)
    return {
        key: 1.0 - ordered.index(value) / denominator
        for key, value in values.items()
    }


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z]+", value.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def _diverse_top(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: item["selection_score"], reverse=True):
        words = _tokens(row["model_prompt_text"])
        if all(_jaccard(words, _tokens(item["model_prompt_text"])) < 0.72 for item in selected):
            selected.append(row)
        if len(selected) == count:
            return selected
    for row in sorted(candidates, key=lambda item: item["selection_score"], reverse=True):
        if row not in selected:
            selected.append(row)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"could not select {count} listening rows")
    return selected


def _audio(path: Path) -> dict[str, Any]:
    info = sf.info(path)
    if info.frames <= 0 or info.samplerate != 44_100:
        raise RuntimeError(f"invalid listening audio: {path}")
    return {
        "path": str(path),
        "channels": int(info.channels),
        "sample_rate": int(info.samplerate),
        "duration_sec": float(info.duration),
    }


def _spatial_facts(scene_plan: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for source in scene_plan["sources"]:
        trajectory = source["trajectory"]
        value = {
            "source_id": source["source_id"],
            "kind": source["kind"],
            "activity": source["activity"],
            "trajectory_type": trajectory["type"],
        }
        if trajectory["type"] == "static":
            value["position"] = trajectory["position"]
        else:
            value["start"] = trajectory["start"]
            value["end"] = trajectory["end"]
        output.append(value)
    return output


def _metadata(root: Path, step: int, domain: str, panel_id: str) -> dict[str, Any]:
    path = root / "outputs" / f"step_{step:06d}" / domain / panel_id / "metadata.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS" or int(value["checkpoint_step"]) != step:
        raise RuntimeError(f"invalid output metadata: {path}")
    value["metadata_path"] = str(path)
    return value


def _single_source(eval_root: Path, step: int) -> list[dict[str, Any]]:
    metrics = eval_root / "metrics"
    core_rows = [
        row for row in _read_jsonl(metrics / "core_per_output.jsonl")
        if int(row["checkpoint_step"]) == step
    ]
    clap_rows = [
        row for row in _read_jsonl(metrics / "clap_per_output.jsonl")
        if int(row["checkpoint_step"]) == step
    ]
    speech_rows = [
        row for row in _read_jsonl(metrics / "speech_per_output.jsonl")
        if int(row["checkpoint_step"]) == step
    ]
    core = {row["panel_id"]: row for row in core_rows}
    clap = {row["panel_id"]: row for row in clap_rows}
    speech = {row["panel_id"]: row for row in speech_rows}
    output: list[dict[str, Any]] = []

    for domain in ("music", "sound"):
        ids = [row["panel_id"] for row in clap_rows if row["domain"] == domain]
        text_pct = _percentiles({key: float(clap[key]["generated_text_cosine"]) for key in ids}, higher=True)
        pair_pct = _percentiles({key: float(clap[key]["generated_reference_audio_cosine"]) for key in ids}, higher=True)
        doa_pct = _percentiles({key: float(core[key]["generated_doa"]["spherical_error_mean_deg"]) for key in ids}, higher=False)
        activity_pct = _percentiles({key: float(core[key]["generated_activity"]["temporal_iou"]) for key in ids}, higher=True)
        candidates = []
        for panel_id in ids:
            metadata = _metadata(eval_root, step, domain, panel_id)
            candidates.append(
                {
                    "group": f"best_{domain}",
                    "panel_id": panel_id,
                    "sample_id": metadata["sample_id"],
                    "checkpoint_step": step,
                    "model_prompt_text": metadata["model_prompt_text"],
                    "renderer_caption": metadata["renderer_caption"],
                    "scene_plan": metadata["scene_plan"],
                    "spatial_facts": _spatial_facts(metadata["scene_plan"]),
                    "selection_score": 0.40 * text_pct[panel_id] + 0.30 * pair_pct[panel_id] + 0.15 * doa_pct[panel_id] + 0.15 * activity_pct[panel_id],
                    "metrics": {
                        "text_clap": clap[panel_id]["generated_text_cosine"],
                        "paired_clap": clap[panel_id]["generated_reference_audio_cosine"],
                        "plan_doa_error_deg": core[panel_id]["generated_doa"]["spherical_error_mean_deg"],
                        "activity_iou": core[panel_id]["generated_activity"]["temporal_iou"],
                    },
                    "audio": {
                        "stereo": _audio(Path(metadata["generated_stereo_path"])),
                        "foa": _audio(Path(metadata["generated_foa_path"])),
                    },
                    "metadata_path": metadata["metadata_path"],
                }
            )
        output.extend(_diverse_top(candidates, 3))

    ids = list(speech)
    wer_pct = _percentiles({key: float(speech[key]["generated_errors"]["wer"]) for key in ids}, higher=False)
    cer_pct = _percentiles({key: float(speech[key]["generated_errors"]["cer"]) for key in ids}, higher=False)
    utmos_pct = _percentiles({key: float(speech[key]["generated_utmos"]) for key in ids}, higher=True)
    doa_pct = _percentiles({key: float(core[key]["generated_doa"]["spherical_error_mean_deg"]) for key in ids}, higher=False)
    activity_pct = _percentiles({key: float(core[key]["generated_activity"]["temporal_iou"]) for key in ids}, higher=True)
    candidates = []
    for panel_id in ids:
        metadata = _metadata(eval_root, step, "speech", panel_id)
        candidates.append(
            {
                "group": "best_speech",
                "panel_id": panel_id,
                "sample_id": metadata["sample_id"],
                "checkpoint_step": step,
                "model_prompt_text": metadata["model_prompt_text"],
                "renderer_caption": metadata["renderer_caption"],
                "scene_plan": metadata["scene_plan"],
                "spatial_facts": _spatial_facts(metadata["scene_plan"]),
                "speech_seen_speaker": bool(metadata["speech_seen_speaker"]),
                "exact_transcript": speech[panel_id]["exact_transcript"],
                "generated_asr": speech[panel_id]["generated_asr"]["text"],
                "selection_score": 0.35 * wer_pct[panel_id] + 0.15 * cer_pct[panel_id] + 0.30 * utmos_pct[panel_id] + 0.10 * doa_pct[panel_id] + 0.10 * activity_pct[panel_id],
                "metrics": {
                    "wer": speech[panel_id]["generated_errors"]["wer"],
                    "cer": speech[panel_id]["generated_errors"]["cer"],
                    "utmos": speech[panel_id]["generated_utmos"],
                    "plan_doa_error_deg": core[panel_id]["generated_doa"]["spherical_error_mean_deg"],
                    "activity_iou": core[panel_id]["generated_activity"]["temporal_iou"],
                },
                "audio": {
                    "stereo": _audio(Path(metadata["generated_stereo_path"])),
                    "foa": _audio(Path(metadata["generated_foa_path"])),
                },
                "metadata_path": metadata["metadata_path"],
            }
        )
    # Preserve speaker-generalization diversity while still taking metric-best rows.
    seen = _diverse_top([row for row in candidates if row["speech_seen_speaker"]], 1)
    unseen = _diverse_top([row for row in candidates if not row["speech_seen_speaker"]], 2)
    output.extend(seen + unseen)
    return output


def _spatial_entries(showcase_root: Path, step: int) -> list[dict[str, Any]]:
    contract = json.loads((showcase_root / "EVAL_CONTRACT.json").read_text(encoding="utf-8"))
    panel = _read_jsonl(showcase_root / contract["test_set"]["panel_filename"])
    entries = []
    for row in panel:
        metadata = _metadata(showcase_root, step, row["domain"], row["panel_id"])
        entries.append(
            {
                "group": row["domain"],
                "panel_id": row["panel_id"],
                "sample_id": metadata["sample_id"],
                "checkpoint_step": step,
                "model_prompt_text": metadata["model_prompt_text"],
                "renderer_caption": metadata["renderer_caption"],
                "scene_plan": metadata["scene_plan"],
                "spatial_facts": _spatial_facts(metadata["scene_plan"]),
                "audio": {
                    "stereo": _audio(Path(metadata["generated_stereo_path"])),
                    "foa": _audio(Path(metadata["generated_foa_path"])),
                },
                "metadata_path": metadata["metadata_path"],
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--spatial-showcase-root", type=Path, required=True)
    args = parser.parse_args()
    eval_root = args.eval_root.expanduser().resolve(strict=True)
    showcase_root = args.spatial_showcase_root.expanduser().resolve(strict=True)
    best = json.loads(
        (eval_root / "cross_system_baselines/metrics/BEST_CHECKPOINT.json").read_text(encoding="utf-8")
    )
    step = int(best["recommended_checkpoint_step"])
    entries = _single_source(eval_root, step) + _spatial_entries(showcase_root, step)
    if len(entries) != 15:
        raise RuntimeError(f"expected 15 final listening entries, got {len(entries)}")
    report = {
        "schema": "stable_audio_tools.p10_v11_final_listening_report",
        "schema_version": 1,
        "status": "PASS",
        "recommended_checkpoint_step": step,
        "selection_method": {
            "single_source": "three metric-strong, semantically diverse rows per domain; Speech includes one seen and two unseen speakers",
            "spatial": "six precommitted disjoint-test ScenePlans selected before hearing outputs",
        },
        "entries": entries,
    }
    output = eval_root / "listening"
    _atomic_text(output / "FINAL_LISTENING_REPORT.json", json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Final P10 listening report",
        "",
        f"Quantitatively recommended checkpoint: **{step // 1000}k**.",
        "",
    ]
    for row in entries:
        lines.extend(
            [
                f"## {row['group']} — {row['panel_id']}",
                "",
                f"- Model prompt: {row['model_prompt_text']}",
                f"- Stereo preview: `{row['audio']['stereo']['path']}`",
                f"- Native FOA: `{row['audio']['foa']['path']}`",
                f"- ScenePlan: `{row['metadata_path']}`",
                "",
            ]
        )
    _atomic_text(output / "FINAL_LISTENING_REPORT.md", "\n".join(lines))
    print(json.dumps({"status": "PASS", "step": step, "entries": len(entries), "report": str(output / "FINAL_LISTENING_REPORT.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
