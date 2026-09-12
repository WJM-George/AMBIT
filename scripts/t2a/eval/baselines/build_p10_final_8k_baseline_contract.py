#!/usr/bin/env python3
"""Freeze public-baseline requests for the final 8,000-row P10 benchmark.

Ordinary public models receive only ``semantic_text``: the deterministic join
of per-source descriptions, with speaker description plus exact transcript for
Speech.  They never receive ScenePlan JSON, source IDs, activity, room, gain,
or trajectories.  P10 keeps its full ScenePlan input as its native interface;
the conditioning modality is disclosed in the result table.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.baselines.build_p10_v11_stratified_3000_baseline_contract import (  # noqa: E501
    BASELINES,
    REPO_LOCK,
)


SOURCE_EVAL = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_full_test_8000_semantic_v2"
)
OUTPUT_ROOT = SOURCE_EVAL / "cross_system_baselines_final_8k"
REUSE_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_stratified_test_3000_semantic_v2/cross_system_baselines"
)

GENERAL_IDS = {
    "stable_audio_open_1_0",
    "tangoflux",
    "audiox_turbo",
    "audiox_maf",
    "audiox_maf_mmdit",
}
SOUND_IDS = {"mmaudio_large_44k_v2_text_only", "woosh_flow"}
SPEECH_IDS = {"qwen3_tts_1p7b_voice_design"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


def baseline_lane(baseline_id: str) -> str:
    if baseline_id in GENERAL_IDS:
        return "general_raw_text_all_scenes"
    if baseline_id in SOUND_IDS:
        return "sound_raw_text_sound_presence"
    if baseline_id in SPEECH_IDS:
        return "tts_single_source_speech_only"
    raise ValueError(f"uncontracted baseline: {baseline_id}")


def eligible(row: dict[str, Any], baseline_id: str) -> bool:
    kinds = set(row["source_kinds"])
    if baseline_id in GENERAL_IDS:
        return True
    if baseline_id in SOUND_IDS:
        return "sound" in kinds
    if baseline_id in SPEECH_IDS:
        return int(row["source_count"]) == 1 and kinds == {"speech"}
    raise ValueError(f"uncontracted baseline: {baseline_id}")


def request_domain(baseline_id: str) -> str:
    if baseline_id in GENERAL_IDS:
        # Retain the legacy value so already-frozen 3k outputs can be reused
        # byte-for-byte when prompt, seed, and duration are identical.
        return "music_sound"
    if baseline_id in SOUND_IDS:
        return "sound"
    if baseline_id in SPEECH_IDS:
        return "speech"
    raise ValueError(f"uncontracted baseline: {baseline_id}")


def composition_signature(row: dict[str, Any]) -> str:
    counts = row["source_kind_counts"]
    return (
        "M" * int(counts.get("music", 0))
        + "S" * int(counts.get("sound", 0))
        + ("Sp" if int(counts.get("speech", 0)) else "")
    )


def speech_source(row: dict[str, Any]) -> dict[str, Any] | None:
    matches = [source for source in row["scene_plan"]["sources"] if source["kind"] == "speech"]
    if len(matches) > 1:
        raise RuntimeError(f"more than one formal Speech source: {row['panel_id']}")
    return matches[0] if matches else None


def load_reusable_requests() -> dict[tuple[str, str], dict[str, Any]]:
    manifest = REUSE_ROOT / "generation_requests.jsonl"
    if not manifest.is_file():
        return {}
    return {
        (row["baseline_id"], row["panel_id"]): row
        for row in read_jsonl(manifest)
    }


def reusable_output(
    baseline_id: str,
    row: dict[str, Any],
    old: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    previous = old.get((baseline_id, row["panel_id"]))
    if previous is None:
        return None
    checks = {
        "sample_id": row["sample_id"],
        "semantic_prompt": row["semantic_text"],
        "seed": int(row["noise_seed"] % (2**31 - 1)),
        "duration_sec": float(row["duration_sec"]),
        "reference_foa_path": row["reference_foa_path"],
    }
    for key, value in checks.items():
        if previous.get(key) != value:
            return None
    native = Path(previous["native_output_path"])
    quality = Path(previous["quality_w_path"])
    metadata = native.with_name("generation.json")
    if not (native.is_file() and quality.is_file() and metadata.is_file()):
        return None
    generation = read_json(metadata)
    if not (
        generation.get("status") == "PASS"
        and generation.get("baseline_id") == baseline_id
        and generation.get("panel_id") == row["panel_id"]
        and generation.get("domain") == request_domain(baseline_id)
        and int(generation.get("seed", -1)) == checks["seed"]
    ):
        return None
    return {
        "native_output_path": str(native.resolve(strict=True)),
        "quality_w_path": str(quality.resolve(strict=True)),
        "reused_from_contract": str((REUSE_ROOT / "BENCHMARK_CONTRACT.json").resolve(strict=True)),
    }


def main() -> int:
    source = SOURCE_EVAL.resolve(strict=True)
    output = OUTPUT_ROOT.resolve()
    eval_contract_path = source / "EVAL_CONTRACT.json"
    eval_contract = read_json(eval_contract_path)
    panel_path = source / eval_contract["test_set"]["panel_filename"]
    panel = read_jsonl(panel_path)
    if len(panel) != 8000:
        raise RuntimeError(f"full panel changed: {len(panel)} != 8000")
    if sha256(panel_path) != eval_contract["test_set"]["panel_sha256"]:
        raise RuntimeError("full 8k panel SHA256 changed")
    if len({row["panel_id"] for row in panel}) != len(panel):
        raise RuntimeError("panel_id is not unique")
    for row in panel:
        expected = "; ".join(row["source_semantic_texts"])
        if row["semantic_text"] != expected:
            raise RuntimeError(f"raw semantic join changed: {row['panel_id']}")
        if any(
            token in row["semantic_text"]
            for token in ("azimuth_deg", "elevation_deg", "distance_m", "gain_db")
        ):
            raise RuntimeError(f"structured control leaked into raw text: {row['panel_id']}")

    repo_lock = read_json(REPO_LOCK)
    repo_commits = {item["name"]: item["commit"] for item in repo_lock["repositories"]}
    reusable = load_reusable_requests()
    requests: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    reused_counts: Counter[str] = Counter()
    for baseline in BASELINES:
        baseline_id = str(baseline["id"])
        if baseline["repo"] not in repo_commits:
            raise RuntimeError(f"unlocked repository: {baseline['repo']}")
        lane = baseline_lane(baseline_id)
        for row in panel:
            if not eligible(row, baseline_id):
                continue
            score_domains = [
                domain
                for domain in ("music", "sound", "speech")
                if domain in row["source_kinds"]
            ]
            # Sound-specialist systems receive the complete raw mixed-scene
            # prompt when a Sound source is present, but they are admitted
            # only to the Sound table.  Counting their Music+Sound rows as a
            # partial Music lane would compare 1,988 rows against the 4,013-row
            # full Music lane and is therefore invalid.
            if baseline_id in SOUND_IDS:
                score_domains = ["sound"]
            target_root = output / "outputs" / baseline_id / row["panel_id"]
            paths = {
                "native_output_path": str(target_root / "native.wav"),
                "quality_w_path": str(target_root / "quality_w.wav"),
            }
            old_paths = reusable_output(baseline_id, row, reusable)
            if old_paths is not None:
                paths = old_paths
                reused_counts[baseline_id] += 1
            request: dict[str, Any] = {
                "baseline_id": baseline_id,
                "baseline_display_name": baseline["display_name"],
                "evaluation_lane": lane,
                "domain": request_domain(baseline_id),
                "score_domains": score_domains,
                "panel_id": row["panel_id"],
                "sample_id": row["sample_id"],
                "seed": int(row["noise_seed"] % (2**31 - 1)),
                "duration_sec": float(row["duration_sec"]),
                "semantic_prompt": row["semantic_text"],
                "source_semantic_texts": row["source_semantic_texts"],
                "prompt_contract": "raw source descriptions joined by semicolon-space only",
                "reference_foa_path": row["reference_foa_path"],
                "reference_foa_sha256": row["reference_foa_sha256"],
                "scene_composition": row["scene_composition"],
                "source_count": int(row["source_count"]),
                "source_kinds": row["source_kinds"],
                "source_kind_counts": row["source_kind_counts"],
                "composition_signature": composition_signature(row),
                "length_bucket": int(row["length_bucket"]),
                "room_type": row["room_type"],
                **paths,
            }
            speech = speech_source(row)
            if speech is not None:
                request.update(
                    {
                        "transcript": speech["transcript"],
                        "speaker_description": speech["speaker_description"],
                        "speech_seen_speaker": bool(row["speech_seen_speaker"]),
                        "speech_speaker_key": row["speech_speaker_key"],
                    }
                )
            requests.append(request)
            counts[baseline_id] += 1

    expected_counts = {
        **{baseline_id: 8000 for baseline_id in GENERAL_IDS},
        **{baseline_id: 4013 for baseline_id in SOUND_IDS},
        **{baseline_id: 1250 for baseline_id in SPEECH_IDS},
    }
    if dict(counts) != expected_counts:
        raise RuntimeError(f"request counts changed: {dict(counts)}")
    pairs = [(row["baseline_id"], row["panel_id"]) for row in requests]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("duplicate baseline/panel request")

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "generation_requests.jsonl"
    atomic_jsonl(manifest_path, requests)
    panel_presence = {
        domain: {
            "rows": sum(domain in row["source_kinds"] for row in panel),
            "by_source_count": dict(
                sorted(
                    Counter(
                        str(row["source_count"])
                        for row in panel
                        if domain in row["source_kinds"]
                    ).items()
                )
            ),
        }
        for domain in ("music", "sound", "speech")
    }
    frozen_baselines = []
    for baseline in BASELINES:
        baseline_id = str(baseline["id"])
        frozen_baselines.append(
            {
                **baseline,
                "lane": baseline_lane(baseline_id),
                "repo_commit": repo_commits[baseline["repo"]],
            }
        )
    contract = {
        "schema": "sceneplan_foa.p10_final_8000_public_baseline_contract",
        "schema_version": 1,
        "status": "FROZEN",
        "source_eval_root": str(source),
        "source_eval_contract": str(eval_contract_path),
        "source_eval_contract_sha256": sha256(eval_contract_path),
        "source_panel_path": str(panel_path),
        "source_panel_sha256": sha256(panel_path),
        "source_panel_rows": len(panel),
        "generation_request_count": len(requests),
        "request_counts_by_baseline": dict(counts),
        "reused_valid_output_counts_by_baseline": dict(reused_counts),
        "new_generation_request_count": len(requests) - sum(reused_counts.values()),
        "domain_presence": panel_presence,
        "prompt_contract": {
            "baseline_input": "raw source descriptions joined in ScenePlan order",
            "speech_clause": "speaker description says: exact transcript",
            "forbidden": [
                "ScenePlan JSON",
                "source IDs",
                "room",
                "gain",
                "activity onset/offset",
                "coordinates",
                "trajectory",
            ],
            "ours_input": "complete native ScenePlan",
            "disclosure": "native-interface comparison, not equal conditioning bandwidth",
        },
        "lanes": {
            "general_raw_text_all_scenes": {
                "rows": 8000,
                "models": sorted(GENERAL_IDS),
                "table_domains": ["music", "sound", "speech"],
            },
            "sound_raw_text_sound_presence": {
                "rows": 4013,
                "models": sorted(SOUND_IDS),
                "table_domains": ["sound"],
            },
            "tts_single_source_speech_only": {
                "rows": 1250,
                "models": sorted(SPEECH_IDS),
                "table_domains": ["speech"],
                "source_count_2_3_4": "N/A",
            },
        },
        "quality_view": {
            "reference": "GT native FOA W channel",
            "ours": "generated native FOA W channel",
            "mono_baseline": "unchanged",
            "stereo_baseline": "fixed arithmetic-mean downmix",
            "W_W_W_W": "forbidden as FOA; no spatial score for public mono/stereo models",
        },
        "baselines": frozen_baselines,
        "generation_manifest": str(manifest_path),
        "generation_manifest_sha256": sha256(manifest_path),
    }
    atomic_json(output / "BENCHMARK_CONTRACT.json", contract)
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
