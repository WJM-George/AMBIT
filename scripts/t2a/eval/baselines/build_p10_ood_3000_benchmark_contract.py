#!/usr/bin/env python3
"""Freeze P10 and public-baseline requests for the deduplicated OOD 3k panel.

The public systems receive exactly the natural semantic text stored in the
panel.  P10 receives a deterministic, neutral, single-source ScenePlan built
from the same semantic content.  Natural OOD references are mono, so this
contract evaluates content quality only and explicitly forbids spatial scores.
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


OOD_ROOT = Path("/mnt/sdb/audio_dataset/evaluation_benchmark/p10_ood_3000_v1")
OUTPUT_ROOT = OOD_ROOT / "cross_system_benchmark"
P10_INTERNAL_EVAL = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v11_150k_full_test_8000_semantic_v2"
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    os.replace(temporary, path)


def _eligible(domain: str, baseline_id: str) -> bool:
    if baseline_id in GENERAL_IDS:
        return True
    if baseline_id in SOUND_IDS:
        return domain == "sound"
    if baseline_id in SPEECH_IDS:
        return domain == "speech"
    raise ValueError(f"uncontracted baseline: {baseline_id}")


def _lane(baseline_id: str) -> str:
    if baseline_id in GENERAL_IDS:
        return "general_raw_text_all_domains"
    if baseline_id in SOUND_IDS:
        return "sound_raw_text"
    if baseline_id in SPEECH_IDS:
        return "tts_speech"
    raise ValueError(f"uncontracted baseline: {baseline_id}")


def main() -> int:
    ood_root = OOD_ROOT.resolve(strict=True)
    output_root = OUTPUT_ROOT.resolve()
    summary_path = (ood_root / "OOD_3000_SUMMARY.json").resolve(strict=True)
    summary = _read_json(summary_path)
    if summary.get("status") != "PASS_FROZEN" or int(summary["rows"]) != 3000:
        raise RuntimeError("OOD 3k freeze marker is not PASS")
    panel_path = Path(summary["artifacts"]["panel_jsonl"]).resolve(strict=True)
    if _sha256(panel_path) != summary["artifacts"]["panel_jsonl_sha256"]:
        raise RuntimeError("OOD panel SHA256 changed")
    panel = _read_jsonl(panel_path)
    if len(panel) != 3000:
        raise RuntimeError(f"OOD panel changed: {len(panel)} != 3000")
    domains = Counter(str(row["domain"]) for row in panel)
    if domains != Counter({"music": 1000, "sound": 1000, "speech": 1000}):
        raise RuntimeError(f"OOD domains changed: {dict(domains)}")
    if len({row["panel_id"] for row in panel}) != len(panel):
        raise RuntimeError("OOD panel IDs are not unique")
    if len({row["reference_audio_sha256"] for row in panel}) != len(panel):
        raise RuntimeError("OOD reference audio hashes are not unique")
    if len({row["chromaprint_sha256"] for row in panel}) != len(panel):
        raise RuntimeError("OOD Chromaprints are not unique")
    for row in panel:
        if row["semantic_text"] != "; ".join(row["source_semantic_texts"]):
            raise RuntimeError(f"semantic prompt changed: {row['panel_id']}")
        if int(row["source_count"]) != 1:
            raise RuntimeError("OOD v1 must remain single-source")

    internal_root = P10_INTERNAL_EVAL.resolve(strict=True)
    internal_contract_path = (internal_root / "EVAL_CONTRACT.json").resolve(strict=True)
    internal_contract = _read_json(internal_contract_path)
    matches = [
        item for item in internal_contract["checkpoints"] if int(item["step"]) == 150000
    ]
    if len(matches) != 1:
        raise RuntimeError("internal P10 contract has no unique 150k checkpoint")
    checkpoint = Path(matches[0]["path"]).resolve(strict=True)
    if _sha256(checkpoint) != matches[0]["sha256"]:
        raise RuntimeError("P10 150k checkpoint SHA256 changed")

    repo_lock = _read_json(REPO_LOCK)
    repo_commits = {item["name"]: item["commit"] for item in repo_lock["repositories"]}
    requests: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for baseline in BASELINES:
        baseline_id = str(baseline["id"])
        if baseline["repo"] not in repo_commits:
            raise RuntimeError(f"unlocked repository: {baseline['repo']}")
        for row in panel:
            domain = str(row["domain"])
            if not _eligible(domain, baseline_id):
                continue
            target = output_root / "outputs" / baseline_id / row["panel_id"]
            request: dict[str, Any] = {
                "baseline_id": baseline_id,
                "baseline_display_name": baseline["display_name"],
                "evaluation_lane": _lane(baseline_id),
                "domain": domain,
                "score_domains": [domain],
                "panel_id": row["panel_id"],
                "sample_id": row["sample_id"],
                "seed": int(row["noise_seed"] % (2**31 - 1)),
                "duration_sec": float(row["requested_duration_sec"]),
                "semantic_prompt": row["semantic_text"],
                "source_semantic_texts": row["source_semantic_texts"],
                "prompt_contract": "natural OOD semantic text only",
                "reference_audio_path": row["reference_audio_path"],
                "reference_audio_sha256": row["reference_audio_sha256"],
                "reference_duration_sec": float(row["reference_duration_sec"]),
                "source_dataset": row["source_dataset"],
                "source_count": 1,
                "source_kinds": [domain],
                "native_output_path": str(target / "native.wav"),
                "quality_w_path": str(target / "quality_w.wav"),
            }
            if domain == "speech":
                request.update(
                    {
                        "transcript": row["exact_transcript"],
                        "speaker_description": row["speaker_description"],
                        "speech_gender": row["gender"],
                    }
                )
            requests.append(request)
            counts[baseline_id] += 1

    expected_counts = {
        **{baseline_id: 3000 for baseline_id in GENERAL_IDS},
        **{baseline_id: 1000 for baseline_id in SOUND_IDS},
        **{baseline_id: 1000 for baseline_id in SPEECH_IDS},
    }
    if dict(counts) != expected_counts:
        raise RuntimeError(f"OOD request counts changed: {dict(counts)}")
    pairs = [(row["baseline_id"], row["panel_id"]) for row in requests]
    if len(pairs) != len(set(pairs)):
        raise RuntimeError("duplicate OOD baseline/panel request")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "generation_requests.jsonl"
    _atomic_jsonl(manifest_path, requests)
    frozen_baselines = [
        {
            **baseline,
            "lane": _lane(str(baseline["id"])),
            "repo_commit": repo_commits[baseline["repo"]],
        }
        for baseline in BASELINES
    ]
    contract = {
        "schema": "sceneplan_foa.p10_ood_3000_benchmark_contract",
        "schema_version": 1,
        "status": "FROZEN",
        "ood_summary": str(summary_path),
        "ood_summary_sha256": _sha256(summary_path),
        "source_panel_path": str(panel_path),
        "source_panel_sha256": _sha256(panel_path),
        "source_panel_rows": len(panel),
        "domain_counts": dict(domains),
        "generation_manifest": str(manifest_path),
        "generation_manifest_sha256": _sha256(manifest_path),
        "generation_request_count": len(requests),
        "request_counts_by_baseline": dict(counts),
        "baselines": frozen_baselines,
        "p10": {
            "checkpoint_step": 150000,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "source_eval_contract": str(internal_contract_path),
            "source_eval_contract_sha256": _sha256(internal_contract_path),
            "output_root": str(output_root / "outputs" / "ours_p10_150k"),
            "input": "deterministic neutral single-source ScenePlan",
        },
        "prompt_contract": {
            "public_baselines": "natural OOD semantic text only",
            "ours": "same semantics in a neutral dry static-front ScenePlan",
            "forbidden_for_public_baselines": [
                "ScenePlan JSON",
                "source IDs",
                "room",
                "gain",
                "activity",
                "coordinates",
                "trajectory",
            ],
        },
        "quality_view": {
            "reference": "natural mono OOD waveform",
            "ours": "native generated FOA W channel",
            "mono_baseline": "unchanged",
            "stereo_baseline": "arithmetic-mean downmix",
            "spatial_metrics": "N/A for every system because references are mono",
        },
        "dedup": summary["dedup"],
    }
    _atomic_json(output_root / "BENCHMARK_CONTRACT.json", contract)
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
