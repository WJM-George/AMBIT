#!/usr/bin/env python3
"""Freeze runnable public-baseline requests for the P10 v11 3k test slice."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_EVAL = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_stratified_test_3000_semantic_v2"
)
OUTPUT_ROOT = SOURCE_EVAL / "cross_system_baselines"
REPO_LOCK = Path(
    "/home/tanhe/dataset_storage/evaluation_benchmark/contracts/REPOS.lock.json"
)


BASELINES: tuple[dict[str, Any], ...] = (
    {
        "id": "stable_audio_open_1_0",
        "display_name": "Stable Audio Open 1.0",
        "repo": "stable-audio-tools",
        "model_repo": "stabilityai/stable-audio-open-1.0",
        "model_revision": "f21265c1e2710b3bd2386596943f0007f55f802e",
        "native_channels": 2,
        "sample_rate_hz": 44_100,
        "lane": "general_t2a_no_speech",
        "settings": {"steps": 100, "cfg_scale": 7.0, "sampler": "dpmpp-3m-sde"},
    },
    {
        "id": "tangoflux",
        "display_name": "TangoFlux",
        "repo": "TangoFlux",
        "model_repo": "declare-lab/TangoFlux",
        "model_revision": "367005e963cb3a9fb2e03a46104d7de23e34ceea",
        "native_channels": 2,
        "sample_rate_hz": 44_100,
        "lane": "general_t2a_no_speech",
        "settings": {"steps": 50, "cfg_scale": 4.5},
    },
    {
        "id": "audiox_turbo",
        "display_name": "AudioX-Turbo",
        "repo": "AudioX-Turbo",
        "model_repo": "HKUSTAudio/AudioX-Turbo",
        "model_revision": "67af549c42aabdb666e559cb4993eddf48b62f08",
        "native_channels": 2,
        "sample_rate_hz": 44_100,
        "lane": "general_t2a_no_speech",
        "settings": {"steps": 4, "cfg": "distilled-no-cfg"},
    },
    {
        "id": "audiox_maf",
        "display_name": "AudioX-MAF",
        "repo": "AudioX",
        "model_repo": "HKUSTAudio/AudioX-MAF",
        "model_revision": "0a6575a6fd58039281584ad1c6f9e895233e8ca7",
        "native_channels": 2,
        "sample_rate_hz": 44_100,
        "lane": "general_t2a_no_speech",
        "settings": {"steps": 250, "cfg_scale": 7.0, "sampler": "dpmpp-3m-sde"},
    },
    {
        "id": "audiox_maf_mmdit",
        "display_name": "AudioX-MAF-MMDiT",
        "repo": "AudioX",
        "model_repo": "HKUSTAudio/AudioX-MAF-MMDiT",
        "model_revision": "2db085273efb141facde8d15a22a4f7cd9734df4",
        "native_channels": 2,
        "sample_rate_hz": 44_100,
        "lane": "general_t2a_no_speech",
        "settings": {"steps": 250, "cfg_scale": 7.0, "sampler": "dpmpp-3m-sde"},
    },
    {
        "id": "mmaudio_large_44k_v2_text_only",
        "display_name": "MMAudio-L v2 (text-only)",
        "repo": "MMAudio",
        "model_repo": "hkchengrex/MMAudio",
        "model_revision": "eb13a1a98fdbec91753775c57b074ccdfc60587c",
        "native_channels": 1,
        "sample_rate_hz": 44_100,
        "lane": "sound_specialist_no_speech",
        "settings": {"steps": 25, "cfg_scale": 4.5, "video_condition": None},
    },
    {
        "id": "woosh_flow",
        "display_name": "Woosh-Flow",
        "repo": "Woosh",
        "model_repo": "github-release:v1.0.0/Woosh-Flow.zip",
        "model_revision": "v1.0.0",
        "native_channels": 1,
        "sample_rate_hz": 48_000,
        "lane": "sound_specialist_no_speech",
        "settings": {"cfg_scale": 4.5, "solver": "adaptive-flow-matching"},
    },
    {
        "id": "qwen3_tts_1p7b_voice_design",
        "display_name": "Qwen3-TTS 1.7B VoiceDesign",
        "repo": "Qwen3-TTS",
        "model_repo": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "model_revision": "5ecdb67327fd37bb2e042aab12ff7391903235d3",
        "native_channels": 1,
        "sample_rate_hz": 24_000,
        "lane": "tts_speech_only",
        "settings": {"language": "English", "voice_condition": "speaker_description"},
    },
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def eligible(row: dict[str, Any], lane: str) -> bool:
    composition = row["scene_composition"]
    kinds = set(row["source_kinds"])
    if lane == "general_t2a_no_speech":
        return composition == "no_speech"
    if lane == "sound_specialist_no_speech":
        return composition == "no_speech" and "sound" in kinds
    if lane == "tts_speech_only":
        return (
            composition == "speech_only"
            and int(row["source_count"]) == 1
            and kinds == {"speech"}
        )
    raise ValueError(f"unknown baseline lane: {lane}")


def request_domain(lane: str) -> str:
    return {
        "general_t2a_no_speech": "music_sound",
        "sound_specialist_no_speech": "sound",
        "tts_speech_only": "speech",
    }[lane]


def main() -> int:
    source = SOURCE_EVAL.resolve(strict=True)
    output_root = OUTPUT_ROOT.resolve()
    eval_contract_path = source / "EVAL_CONTRACT.json"
    eval_contract = read_json(eval_contract_path)
    panel_path = source / eval_contract["test_set"]["panel_filename"]
    panel = read_jsonl(panel_path)
    if len(panel) != 3000 or sha256(panel_path) != eval_contract["test_set"]["panel_sha256"]:
        raise RuntimeError("the frozen P10 v11 3k panel changed")
    if len({row["panel_id"] for row in panel}) != len(panel):
        raise RuntimeError("panel_id is not unique")

    composition_counts = Counter(row["scene_composition"] for row in panel)
    expected_compositions = {
        "no_speech": 1124,
        "speech_only": 469,
        "speech_with_overlapping_background": 846,
        "speech_with_sequential_background": 561,
    }
    if dict(composition_counts) != expected_compositions:
        raise RuntimeError(f"test composition changed: {dict(composition_counts)}")

    repo_lock = read_json(REPO_LOCK)
    repo_commits = {row["name"]: row["commit"] for row in repo_lock["repositories"]}
    requests: list[dict[str, Any]] = []
    counts_by_baseline: Counter[str] = Counter()
    for baseline in BASELINES:
        if baseline["repo"] not in repo_commits:
            raise RuntimeError(f"unlocked repository: {baseline['repo']}")
        lane = str(baseline["lane"])
        for row in panel:
            if not eligible(row, lane):
                continue
            score_domains = [
                kind for kind in ("music", "sound", "speech") if kind in row["source_kinds"]
            ]
            request: dict[str, Any] = {
                "baseline_id": baseline["id"],
                "baseline_display_name": baseline["display_name"],
                "evaluation_lane": lane,
                "domain": request_domain(lane),
                "score_domains": score_domains,
                "panel_id": row["panel_id"],
                "sample_id": row["sample_id"],
                "seed": int(row["noise_seed"] % (2**31 - 1)),
                "duration_sec": float(row["duration_sec"]),
                "semantic_prompt": row["semantic_text"],
                "reference_foa_path": row["reference_foa_path"],
                "scene_composition": row["scene_composition"],
                "source_count": int(row["source_count"]),
                "source_kinds": row["source_kinds"],
                "source_kind_counts": row["source_kind_counts"],
                "length_bucket": int(row["length_bucket"]),
                "room_type": row["room_type"],
                "native_output_path": str(
                    output_root
                    / "outputs"
                    / baseline["id"]
                    / row["panel_id"]
                    / "native.wav"
                ),
                "quality_w_path": str(
                    output_root
                    / "outputs"
                    / baseline["id"]
                    / row["panel_id"]
                    / "quality_w.wav"
                ),
            }
            if lane == "tts_speech_only":
                speech = row["scene_plan"]["sources"][0]
                request.update(
                    {
                        "transcript": speech["transcript"],
                        "speaker_description": speech["speaker_description"],
                        "speech_seen_speaker": bool(row["speech_seen_speaker"]),
                        "speech_speaker_key": row["speech_speaker_key"],
                    }
                )
            requests.append(request)
            counts_by_baseline[str(baseline["id"])] += 1

    expected_by_baseline = {
        "stable_audio_open_1_0": 1124,
        "tangoflux": 1124,
        "audiox_turbo": 1124,
        "audiox_maf": 1124,
        "audiox_maf_mmdit": 1124,
        "mmaudio_large_44k_v2_text_only": 745,
        "woosh_flow": 745,
        "qwen3_tts_1p7b_voice_design": 469,
    }
    if dict(counts_by_baseline) != expected_by_baseline:
        raise RuntimeError(f"baseline request counts changed: {dict(counts_by_baseline)}")
    pairs = [(row["baseline_id"], row["panel_id"]) for row in requests]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("duplicate baseline/panel request")

    contract = {
        "schema": "sceneplan_foa.p10_v11_stratified_3000_baseline_contract",
        "schema_version": 1,
        "status": "FROZEN",
        "source_eval_root": str(source),
        "source_eval_contract": str(eval_contract_path),
        "source_eval_contract_sha256": sha256(eval_contract_path),
        "source_panel_path": str(panel_path),
        "source_panel_sha256": sha256(panel_path),
        "source_panel_rows": len(panel),
        "generation_request_count": len(requests),
        "request_counts_by_baseline": dict(counts_by_baseline),
        "evaluation_lanes": {
            "general_t2a_no_speech": {
                "rows": 1124,
                "description": "all no-formal-speech rows; score Music and Sound presence slices",
            },
            "sound_specialist_no_speech": {
                "rows": 745,
                "description": "no-formal-speech rows containing at least one Sound source",
            },
            "tts_speech_only": {
                "rows": 469,
                "description": "single-source speech-only rows with exact transcript and speaker description",
            },
        },
        "quality_view": {
            "ours": "native FOA W channel",
            "mono_baseline": "unchanged",
            "stereo_baseline": "fixed arithmetic mean downmix",
            "spatial_metrics_for_non_foa_baselines": "N/A",
        },
        "core_metrics": {
            "music_sound": ["CLAP", "FAD/FD", "KL"],
            "speech": ["WER", "CER", "UTMOS"],
            "native_foa_only": [
                "spherical DoA error",
                "azimuth/elevation error",
                "activity IoU",
            ],
        },
        "baselines": [
            {**baseline, "repo_commit": repo_commits[baseline["repo"]]}
            for baseline in BASELINES
        ],
    }
    atomic_json(output_root / "BENCHMARK_CONTRACT.json", contract)
    atomic_jsonl(output_root / "generation_requests.jsonl", requests)
    smoke_requests = []
    for baseline in BASELINES:
        smoke_requests.append(
            next(row for row in requests if row["baseline_id"] == baseline["id"])
        )
    atomic_jsonl(output_root / "smoke_requests.jsonl", smoke_requests)
    atomic_json(
        Path(
            "/home/tanhe/dataset_storage/evaluation_benchmark/contracts/"
            "P10_V11_STRATIFIED_3000_BASELINES.json"
        ),
        contract,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "panel_rows": len(panel),
                "requests": len(requests),
                "by_baseline": dict(counts_by_baseline),
                "output_root": str(output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
