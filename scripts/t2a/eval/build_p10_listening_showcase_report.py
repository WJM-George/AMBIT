#!/usr/bin/env python3
"""Build and audit the final P10 qualitative listening showcase.

The report combines nine high-performing, semantically de-duplicated
single-source examples from the balanced 1,200-row benchmark with six
explicit spatial-composition examples rendered by the frozen 100k DiT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import soundfile as sf


DEFAULT_BALANCED_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_balanced_1200_ckpt20k_100k_v1"
)
DEFAULT_SHOWCASE_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_100k_listening_showcase_v1"
)

BEST_SINGLE_SOURCE = {
    "best_music": {
        "step": 100_000,
        "panel_ids": ("music_098", "music_309", "music_386"),
        "selection": "high joint CLAP/spatial/activity quality with instrument/style diversity",
    },
    "best_sound": {
        "step": 80_000,
        "panel_ids": ("sound_245", "sound_043", "sound_124"),
        "selection": "80k aggregate Sound specialist; high joint CLAP/spatial/activity quality with event diversity",
    },
    "best_speech": {
        "step": 100_000,
        "panel_ids": ("speech_108", "speech_078", "speech_357"),
        "selection": "high WER/UTMOS/spatial/activity quality with seen/unseen and voice diversity",
    },
}

SPATIAL_SELECTION = {
    "speech_left_music_right": (
        "full_0001119",
        "full_0005497",
        "full_0005618",
    ),
    "dynamic_speech": (
        "full_0000690",
        "full_0001420",
        "full_0005555",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _jsonl_index(path: Path, keys: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                result[tuple(row[key] for key in keys)] = row
    return result


def _audio_audit(path: Path, expected_channels: int) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 44:
        raise RuntimeError(f"missing or empty audio: {path}")
    info = sf.info(path)
    if info.channels != expected_channels:
        raise RuntimeError(f"{path}: expected {expected_channels} channels, got {info.channels}")
    if info.samplerate != 44_100:
        raise RuntimeError(f"{path}: expected 44100 Hz, got {info.samplerate}")
    if info.frames <= 0:
        raise RuntimeError(f"{path}: no audio frames")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "channels": info.channels,
        "sample_rate": info.samplerate,
        "frames": info.frames,
        "duration_sec": info.duration,
        "format": info.format,
        "subtype": info.subtype,
    }


def _speech_source(scene_plan: dict[str, Any]) -> dict[str, Any]:
    sources = [source for source in scene_plan["sources"] if source["kind"] == "speech"]
    if len(sources) != 1:
        raise RuntimeError("showcase scene must contain exactly one formal Speech source")
    return sources[0]


def _spatial_facts(scene_plan: dict[str, Any]) -> list[dict[str, Any]]:
    facts = []
    for source in scene_plan["sources"]:
        trajectory = source["trajectory"]
        fact: dict[str, Any] = {
            "source_id": source["source_id"],
            "kind": source["kind"],
            "activity": source["activity"],
            "trajectory_type": trajectory["type"],
        }
        if trajectory["type"] == "static":
            fact["position"] = trajectory["position"]
        else:
            fact["start"] = trajectory["start"]
            fact["end"] = trajectory["end"]
        facts.append(fact)
    return facts


def _audit_metadata(metadata_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS":
        raise RuntimeError(f"non-PASS metadata: {metadata_path}")
    qc = metadata["qc"]
    if not qc["finite"] or float(qc["fraction_abs_ge_1"]) != 0.0:
        raise RuntimeError(f"invalid generated FOA QC: {metadata_path}")
    foa = _audio_audit(Path(metadata["generated_foa_path"]), 4)
    stereo = _audio_audit(Path(metadata["generated_stereo_path"]), 2)
    if foa["sha256"] != metadata["generated_foa_sha256"]:
        raise RuntimeError(f"FOA checksum mismatch: {metadata_path}")
    if stereo["sha256"] != metadata["generated_stereo_sha256"]:
        raise RuntimeError(f"stereo checksum mismatch: {metadata_path}")
    return metadata, {"foa": foa, "stereo": stereo}


def _single_source_entries(balanced_root: Path) -> list[dict[str, Any]]:
    clap = _jsonl_index(
        balanced_root / "metrics" / "clap_per_output.jsonl",
        ("checkpoint_step", "panel_id"),
    )
    core = _jsonl_index(
        balanced_root / "metrics" / "core_per_output.jsonl",
        ("checkpoint_step", "panel_id"),
    )
    cross_system = json.loads(
        (
            balanced_root
            / "cross_system_baselines"
            / "metrics"
            / "CROSS_SYSTEM_METRICS.json"
        ).read_text(encoding="utf-8")
    )
    speech_metrics = {
        row["panel_id"]: row
        for row in cross_system["speech"]["ours_sceneplan_foa_100000"]["per_output"]
    }

    entries: list[dict[str, Any]] = []
    for group, spec in BEST_SINGLE_SOURCE.items():
        step = int(spec["step"])
        for panel_id in spec["panel_ids"]:
            domain = panel_id.split("_", 1)[0]
            metadata_path = (
                balanced_root
                / "outputs"
                / f"step_{step:06d}"
                / domain
                / panel_id
                / "metadata.json"
            )
            metadata, audio = _audit_metadata(metadata_path)
            core_row = core[(step, panel_id)]
            metrics: dict[str, Any] = {
                "doa_error_deg": core_row["generated_doa"]["spherical_error_mean_deg"],
                "activity_iou": core_row["generated_activity"]["temporal_iou"],
            }
            if domain in {"music", "sound"}:
                clap_row = clap[(step, panel_id)]
                metrics.update(
                    {
                        "text_clap": clap_row["generated_text_cosine"],
                        "paired_audio_clap": clap_row["generated_reference_audio_cosine"],
                    }
                )
            else:
                speech_row = speech_metrics[panel_id]
                metrics.update(
                    {
                        "wer": speech_row["wer"],
                        "cer": speech_row["cer"],
                        "utmos": speech_row["utmos"],
                    }
                )
            entries.append(
                {
                    "group": group,
                    "selection": spec["selection"],
                    "panel_id": panel_id,
                    "sample_id": metadata["sample_id"],
                    "checkpoint_step": step,
                    "semantic_text": metadata["semantic_text"],
                    "renderer_caption": metadata["renderer_caption"],
                    "scene_plan": metadata["scene_plan"],
                    "spatial_facts": _spatial_facts(metadata["scene_plan"]),
                    "metrics": metrics,
                    "qc": metadata["qc"],
                    "audio": audio,
                    "metadata_path": str(metadata_path),
                    "status": "PASS",
                }
            )
    return entries


def _spatial_entries(showcase_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for group, panel_ids in SPATIAL_SELECTION.items():
        for panel_id in panel_ids:
            metadata_path = (
                showcase_root
                / "outputs"
                / "step_100000"
                / group
                / panel_id
                / "metadata.json"
            )
            metadata, audio = _audit_metadata(metadata_path)
            speech = _speech_source(metadata["scene_plan"])
            requirement: dict[str, Any]
            if group == "speech_left_music_right":
                music = [
                    source for source in metadata["scene_plan"]["sources"]
                    if source["kind"] == "music"
                ]
                if len(metadata["scene_plan"]["sources"]) != 2 or len(music) != 1:
                    raise RuntimeError(f"{panel_id}: expected exactly one Speech and one Music")
                speech_azimuth = float(speech["trajectory"]["position"]["azimuth_deg"])
                music_azimuth = float(music[0]["trajectory"]["position"]["azimuth_deg"])
                if speech_azimuth < 45.0 or music_azimuth > -45.0:
                    raise RuntimeError(f"{panel_id}: left/right sign contract failed")
                requirement = {
                    "speech_azimuth_deg": speech_azimuth,
                    "music_azimuth_deg": music_azimuth,
                    "positive_azimuth_is_listener_left": True,
                    "pass": True,
                }
            else:
                trajectory = speech["trajectory"]
                if trajectory["type"] != "linear":
                    raise RuntimeError(f"{panel_id}: Speech is not dynamic")
                start = float(trajectory["start"]["azimuth_deg"])
                end = float(trajectory["end"]["azimuth_deg"])
                extent = abs((end - start + 180.0) % 360.0 - 180.0)
                if extent < 90.0:
                    raise RuntimeError(f"{panel_id}: Speech trajectory is too small")
                requirement = {
                    "speech_start_azimuth_deg": start,
                    "speech_end_azimuth_deg": end,
                    "shortest_azimuth_extent_deg": extent,
                    "pass": True,
                }
            entries.append(
                {
                    "group": group,
                    "selection": "explicit frozen full-test spatial contract",
                    "panel_id": panel_id,
                    "sample_id": metadata["sample_id"],
                    "checkpoint_step": 100_000,
                    "semantic_text": metadata["semantic_text"],
                    "renderer_caption": metadata["renderer_caption"],
                    "scene_plan": metadata["scene_plan"],
                    "spatial_facts": _spatial_facts(metadata["scene_plan"]),
                    "spatial_requirement": requirement,
                    "qc": metadata["qc"],
                    "audio": audio,
                    "metadata_path": str(metadata_path),
                    "status": "PASS",
                }
            )
    return entries


def _markdown(entries: list[dict[str, Any]], manifest_path: Path) -> str:
    labels = {
        "best_music": "Music：高分且去重的 3 条（100k）",
        "best_sound": "Sound：高分且去重的 3 条（80k）",
        "best_speech": "Speech：高分且音色多样的 3 条（100k）",
        "speech_left_music_right": "Speech 在左、Music 在右（100k 补跑）",
        "dynamic_speech": "动态 Speech（100k 补跑）",
    }
    lines = [
        "# P10 Listening Showcase",
        "",
        "所有条目均来自冻结且与训练内容隔离的 test panel；FOA 为原生四通道输出，页面播放器使用其双耳预览。",
        "本项目方位角约定为正值在听者左侧、负值在听者右侧。",
        "",
        f"Machine-readable manifest: `{manifest_path}`",
        "",
    ]
    for group in labels:
        group_entries = [entry for entry in entries if entry["group"] == group]
        lines.extend([f"## {labels[group]}", ""])
        for index, entry in enumerate(group_entries, 1):
            metrics = entry.get("metrics") or entry.get("spatial_requirement") or {}
            compact_metrics = ", ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in metrics.items()
                if key != "pass"
            )
            lines.extend(
                [
                    f"### {index}. {entry['panel_id']} — step {entry['checkpoint_step']:,}",
                    "",
                    f"Prompt: {entry['semantic_text']}",
                    "",
                    f"Renderer caption: {entry['renderer_caption']}",
                    "",
                    f"Metrics/contract: {compact_metrics}",
                    "",
                    f"![audio]({entry['audio']['stereo']['path']})",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--balanced-root", type=Path, default=DEFAULT_BALANCED_ROOT)
    parser.add_argument("--showcase-root", type=Path, default=DEFAULT_SHOWCASE_ROOT)
    args = parser.parse_args()

    balanced_root = args.balanced_root.expanduser().resolve(strict=True)
    showcase_root = args.showcase_root.expanduser().resolve(strict=True)
    entries = _single_source_entries(balanced_root) + _spatial_entries(showcase_root)
    counts = {
        group: sum(entry["group"] == group for entry in entries)
        for group in (*BEST_SINGLE_SOURCE, *SPATIAL_SELECTION)
    }
    expected = {group: 3 for group in counts}
    if counts != expected or len(entries) != 15:
        raise RuntimeError(f"unexpected showcase counts: {counts}")

    manifest_path = showcase_root / "SHOWCASE_MANIFEST.json"
    manifest = {
        "schema": "stable_audio_tools.sceneplan_dit_p10_listening_showcase",
        "schema_version": 1,
        "status": "PASS",
        "primary_unified_checkpoint_step": 100_000,
        "sound_specialist_checkpoint_step": 80_000,
        "azimuth_convention": "positive=listener-left, negative=listener-right",
        "balanced_evaluation_root": str(balanced_root),
        "spatial_showcase_root": str(showcase_root),
        "counts": counts,
        "entries": entries,
    }
    _atomic_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    report_path = showcase_root / "LISTENING_SHOWCASE.md"
    _atomic_text(report_path, _markdown(entries, manifest_path) + "\n")
    summary = {
        "status": "PASS",
        "entries": len(entries),
        "counts": counts,
        "all_foa_qc_finite": all(entry["qc"]["finite"] for entry in entries),
        "all_foa_unclipped": all(entry["qc"]["fraction_abs_ge_1"] == 0.0 for entry in entries),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "report": str(report_path),
        "report_sha256": _sha256(report_path),
    }
    summary_path = showcase_root / "SHOWCASE_AUDIT.json"
    _atomic_text(summary_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
