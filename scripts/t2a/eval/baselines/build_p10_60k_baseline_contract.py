#!/usr/bin/env python3
"""Freeze the cross-system 15-row P10 baseline benchmark contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_EVAL = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/"
    "sceneplan_dit_v7_sao_300m_from_scratch_160k/evaluation/"
    "p10_gt_vae_20k_40k_60k_instrumental_music_v1"
)
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/baselines/p10_60k_15row_v1")
REPO_LOCK = Path(
    "." + "/evaluation_benchmark/contracts/REPOS.lock.json"
)


BASELINES: tuple[dict[str, Any], ...] = (
    {
        "id": "stable_audio_open_1_0",
        "display_name": "Stable Audio Open 1.0",
        "domains": ("music", "sound"),
        "repo": "stable-audio-tools",
        "model_repo": "stabilityai/stable-audio-open-1.0",
        "model_revision": "f21265c1e2710b3bd2386596943f0007f55f802e",
        "native_channels": 2,
        "sample_rate_hz": 44_100,
        "settings": {
            "steps": 100,
            "cfg_scale": 7.0,
            "sigma_min": 0.3,
            "sigma_max": 500.0,
            "sampler": "dpmpp-3m-sde",
        },
    },
    {
        "id": "tangoflux",
        "display_name": "TangoFlux",
        "domains": ("music", "sound"),
        "repo": "TangoFlux",
        "model_repo": "declare-lab/TangoFlux",
        "model_revision": "367005e963cb3a9fb2e03a46104d7de23e34ceea",
        "native_channels": 2,
        "sample_rate_hz": 44_100,
        "settings": {"steps": 50, "cfg_scale": 4.5},
    },
    {
        "id": "audiox_turbo",
        "display_name": "AudioX-Turbo",
        "domains": ("music", "sound"),
        "repo": "AudioX-Turbo",
        "model_repo": "HKUSTAudio/AudioX-Turbo",
        "model_revision": "67af549c42aabdb666e559cb4993eddf48b62f08",
        "native_channels": 2,
        "sample_rate_hz": 44_100,
        "settings": {
            "steps": 4,
            "cfg": "distilled-no-cfg",
            "conditioning_seconds": 10,
        },
    },
    {
        "id": "mmaudio_large_44k_v2_text_only",
        "display_name": "MMAudio-L v2 (text-only)",
        "domains": ("sound",),
        "repo": "MMAudio",
        "model_repo": "hkchengrex/MMAudio",
        "model_revision": "eb13a1a98fdbec91753775c57b074ccdfc60587c",
        "native_channels": 1,
        "sample_rate_hz": 44_100,
        "settings": {"steps": 25, "cfg_scale": 4.5, "video_condition": None},
    },
    {
        "id": "woosh_flow",
        "display_name": "Woosh-Flow",
        "domains": ("sound",),
        "repo": "Woosh",
        "model_repo": "github-release:v1.0.0/Woosh-Flow.zip",
        "model_revision": "v1.0.0",
        "model_dependencies": (
            "github-release:v1.0.0/TextConditionerA.zip",
            "github-release:v1.0.0/Woosh-AE.zip",
        ),
        "native_channels": 1,
        "sample_rate_hz": 48_000,
        "settings": {"cfg_scale": 4.5, "solver": "adaptive-flow-matching"},
    },
    {
        "id": "qwen3_tts_1p7b_voice_design",
        "display_name": "Qwen3-TTS 1.7B VoiceDesign",
        "domains": ("speech",),
        "repo": "Qwen3-TTS",
        "model_repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "model_revision": "5ecdb67327fd37bb2e042aab12ff7391903235d3",
        "native_channels": 1,
        "sample_rate_hz": 24_000,
        "settings": {"language": "English", "voice_condition": "speaker_description"},
    },
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-eval", type=Path, default=DEFAULT_SOURCE_EVAL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    source = args.source_eval.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    panel_path = source / "listening_panel_15.jsonl"
    panel = _read_jsonl(panel_path)
    counts = {domain: sum(row["domain"] == domain for row in panel) for domain in ("music", "sound", "speech")}
    if len(panel) != 15 or counts != {"music": 5, "sound": 5, "speech": 5}:
        raise RuntimeError(f"invalid fixed panel: rows={len(panel)} counts={counts}")
    if any(row["domain"] == "music" and row.get("spoken_language_background") for row in panel):
        raise RuntimeError("the frozen music panel must be instrumental")

    repo_lock = json.loads(REPO_LOCK.read_text(encoding="utf-8"))
    repo_commits = {row["name"]: row["commit"] for row in repo_lock["repositories"]}
    requests: list[dict[str, Any]] = []
    for baseline in BASELINES:
        if baseline["repo"] not in repo_commits:
            raise RuntimeError(f"unlocked upstream repo: {baseline['repo']}")
        for row in panel:
            domain = row["domain"]
            if domain not in baseline["domains"]:
                continue
            request: dict[str, Any] = {
                "baseline_id": baseline["id"],
                "baseline_display_name": baseline["display_name"],
                "domain": domain,
                "panel_id": row["panel_id"],
                "sample_id": row["sample_id"],
                "seed": int(row["noise_seed"] % (2**31 - 1)),
                "duration_sec": float(row["duration_sec"]),
                "semantic_prompt": row["semantic_text"],
                "reference_foa_path": row["reference_foa_path"],
                "ours_60k_foa_path": str(
                    source
                    / "outputs"
                    / "step_060000"
                    / domain
                    / row["panel_id"]
                    / "generated_foa_float32.wav"
                ),
                "native_output_path": str(
                    output_root
                    / "outputs"
                    / baseline["id"]
                    / domain
                    / row["panel_id"]
                    / "native.wav"
                ),
                "quality_w_path": str(
                    output_root
                    / "outputs"
                    / baseline["id"]
                    / domain
                    / row["panel_id"]
                    / "quality_w.wav"
                ),
            }
            if domain == "speech":
                source_spec = row["scene_plan"]["sources"][0]
                request["transcript"] = source_spec["transcript"]
                request["speaker_description"] = source_spec["speaker_description"]
            requests.append(request)

    expected = sum(len(row["domains"]) * 5 for row in BASELINES)
    if len(requests) != expected:
        raise RuntimeError(f"generation request count changed: {len(requests)} != {expected}")

    contract = {
        "schema": "sceneplan_foa.p10_cross_system_baseline_contract",
        "schema_version": 1,
        "status": "FROZEN",
        "source_eval_root": str(source),
        "source_panel_path": str(panel_path),
        "source_panel_sha256": _sha256(panel_path),
        "ours_checkpoint_step": 60_000,
        "domain_counts": counts,
        "generation_request_count": len(requests),
        "quality_view": {
            "ours": "native FOA W channel only",
            "mono_baseline": "unchanged",
            "stereo_baseline": "fixed arithmetic mean downmix",
            "level_handling": "per-clip peak normalization to -1 dBFS inside each frozen metric backend",
            "forbidden": "treating W,W,W,W as spatial audio or duplicating channels in metric weights",
        },
        "spatial_view": {
            "ours": "native WYZX/ACN/SN3D FOA",
            "non_spatial_baselines": "N/A",
            "metrics": [
                "plan_spherical_error_mean_deg",
                "plan_azimuth_circular_mae_deg",
                "plan_elevation_mae_deg",
                "activity_temporal_iou",
            ],
        },
        "metrics": {
            "music": ["CLAP", "FD-CLAP", "FAD-VGGish", "KL-PANN", "spatial_view"],
            "sound": ["CLAP", "FD-CLAP", "FAD-VGGish", "KL-PANN", "spatial_view"],
            "speech": ["WER", "CER", "UTMOS", "spatial_view"],
            "small_sample_warning": "five clips per domain; FAD/FD are diagnostic only",
        },
        "baselines": [
            {**row, "domains": list(row["domains"]), "repo_commit": repo_commits[row["repo"]]}
            for row in BASELINES
        ],
        "withheld": [
            {
                "id": "stable_audio_3",
                "reason": "official weights are gated and this machine has no authenticated accepted token",
            },
            {
                "id": "native_foa_literature_baselines",
                "systems": ["Diff-SAGe", "ImmerseDiffusion", "SonicMotion", "SwanSphere"],
                "reason": "no runnable official checkpoint for the fixed local panel as of 2026-08-23",
            },
        ],
    }
    _atomic_json(output_root / "BENCHMARK_CONTRACT.json", contract)
    _atomic_jsonl(output_root / "generation_requests.jsonl", requests)
    _atomic_json(
        Path("." + "/evaluation_benchmark/contracts/P10_60K_15ROW_V1.json"),
        contract,
    )
    print(json.dumps({"status": "PASS", "requests": len(requests), "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
