#!/usr/bin/env python3
"""Freeze a matched 50-Music/50-Sound 100k-vs-105k protection panel."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_dit_p10_panel_common import sha256_file
from stable_audio_tools.data.model_sceneplan import (
    compile_model_semantic_caption,
    compile_model_semantic_caption_v2,
)


REVISION_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/"
    "revisions/speech_expansion_noalign_15s_v1"
)
DEFAULT_SOURCE = REVISION_ROOT / "evaluation/p10_v9_balanced_1200_ckpt20k_100k_v1"
DEFAULT_OUTPUT = REVISION_ROOT / "evaluation/p10_v10_music_sound_protection_fixed50"
DEFAULT_105K = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/"
    "sceneplan_dit_v10_semantic_v2_protected_resume_110k/"
    "checkpoints/epoch=33-step=105000.ckpt"
)
DEFAULT_MODEL_CONFIG = Path(
    "." + "/stable-audio-tools/stable_audio_tools/configs/"
    "model_configs/txt2audio/t2a/dit/"
    "qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s_resume_cosine_10k.json"
)
NAMESPACE = "p10-v10-music-sound-protection-fixed50-20260830"
DOMAINS = ("music", "sound")


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
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _stratum(row: dict[str, Any]) -> tuple[str, str, int]:
    source = row["scene_plan"]["sources"][0]
    return (
        str(row["room_type"]),
        str(source["trajectory"]["type"]),
        int(row["length_bucket"]),
    )


def _rank(domain: str, sample_id: str) -> str:
    return hashlib.sha256(
        f"{NAMESPACE}\0{domain}\0{sample_id}".encode("utf-8")
    ).hexdigest()


def _sample(
    candidates: list[dict[str, Any]], domain: str, target: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[_stratum(row)].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: _rank(domain, str(row["sample_id"])))

    quotas: dict[tuple[str, str, int], int] = {}
    remainders: list[tuple[float, str, tuple[str, str, int]]] = []
    for key, rows in groups.items():
        exact = target * len(rows) / len(candidates)
        quotas[key] = int(math.floor(exact))
        remainders.append((exact - quotas[key], repr(key), key))
    for _fraction, _label, key in sorted(remainders, reverse=True)[
        : target - sum(quotas.values())
    ]:
        quotas[key] += 1
    if sum(quotas.values()) != target:
        raise RuntimeError(f"{domain}: stratified quotas do not sum to {target}")
    chosen = [
        row
        for key in sorted(groups, key=repr)
        for row in groups[key][: quotas[key]]
    ]
    chosen.sort(key=lambda row: _rank(domain, str(row["sample_id"])))
    if len(chosen) != target or len({row["sample_id"] for row in chosen}) != target:
        raise RuntimeError(f"{domain}: invalid protection sample")
    return chosen, {
        "candidates": len(candidates),
        "selected": len(chosen),
        "candidate_strata": {
            repr(key): len(groups[key]) for key in sorted(groups, key=repr)
        },
        "selected_strata": {
            repr(key): quotas[key] for key in sorted(groups, key=repr)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-105k", type=Path, default=DEFAULT_105K)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--rows-per-domain", type=int, default=50)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    checkpoint_105k = args.checkpoint_105k.expanduser().resolve(strict=True)
    model_config = args.model_config.expanduser().resolve(strict=True)
    target = int(args.rows_per_domain)
    if target != 50:
        raise ValueError("the protection gate is frozen at 50 rows per domain")

    source_contract_path = source_root / "EVAL_CONTRACT.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_panel_path = source_root / source_contract["test_set"]["panel_filename"]
    if sha256_file(source_panel_path) != source_contract["test_set"]["panel_sha256"]:
        raise RuntimeError("source balanced panel SHA256 changed")
    source_rows = [
        json.loads(line)
        for line in source_panel_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    panel: list[dict[str, Any]] = []
    audits: dict[str, Any] = {}
    for domain in DOMAINS:
        candidates = [
            row
            for row in source_rows
            if row["domain"] == domain
            and int(row["source_count"]) == 1
            and row["source_kind_counts"] == {domain: 1}
        ]
        chosen, audit = _sample(candidates, domain, target)
        audits[domain] = audit
        for row in chosen:
            item = copy.deepcopy(row)
            item["source_balanced_panel_id"] = item["panel_id"]
            legacy = compile_model_semantic_caption(item["scene_plan"])
            canonical = compile_model_semantic_caption_v2(item["scene_plan"])
            if legacy["text"] != canonical["text"]:
                raise RuntimeError(
                    f"{item['sample_id']}: non-Speech semantic text changed across v1/v2"
                )
            item["model_prompt_text"] = canonical["text"]
            item["model_event_regions"] = canonical["event_regions"]
            item["model_speech_regions"] = canonical["speech_regions"]
            panel.append(item)

    counts = Counter(str(row["domain"]) for row in panel)
    if counts != Counter({"music": 50, "sound": 50}):
        raise RuntimeError(f"protection panel domain counts changed: {dict(counts)}")
    if len({row["sample_id"] for row in panel}) != 100:
        raise RuntimeError("protection panel sample IDs are not unique")

    checkpoint_100k = [
        copy.deepcopy(row)
        for row in source_contract["checkpoints"]
        if int(row["step"]) == 100_000
    ]
    if len(checkpoint_100k) != 1:
        raise RuntimeError("source contract does not contain exactly one 100k checkpoint")
    checkpoints = checkpoint_100k + [
        {
            "step": 105_000,
            "path": str(checkpoint_105k),
            "bytes": checkpoint_105k.stat().st_size,
            "sha256": sha256_file(checkpoint_105k),
        }
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    panel_path = output_root / "music_sound_protection_fixed50.jsonl"
    _atomic_jsonl(panel_path, panel)
    sampling = copy.deepcopy(source_contract["sampling"])
    sampling.update(
        {
            "model_config": str(model_config),
            "model_config_sha256": sha256_file(model_config),
            "semantic_caption_compiler_version": 2,
            "same_noise_per_sample_across_checkpoints": True,
        }
    )
    contract = {
        "schema": "stable_audio_tools.p10_music_sound_protection_contract",
        "schema_version": 1,
        "status": "FROZEN",
        "purpose": "100k-vs-105k non-regression gate after Speech semantic-caption adaptation",
        "source_balanced_contract": str(source_contract_path),
        "source_balanced_contract_sha256": sha256_file(source_contract_path),
        "selection_namespace": NAMESPACE,
        "selection": {
            "content_disjoint_from_train": True,
            "single_source_domain_pure": True,
            "representative_strata": ["room", "motion", "length_bucket"],
            "audit": audits,
        },
        "test_set": {
            **source_contract["test_set"],
            "evaluation_rows": 100,
            "evaluation_subset": "representative single-source 50 Music + 50 Sound",
            "panel_filename": panel_path.name,
            "panel_sha256": sha256_file(panel_path),
            "domain_counts": {"music": 50, "sound": 50},
            "single_source_domain_pure": True,
        },
        "leakage_audit": source_contract["leakage_audit"],
        "checkpoints": checkpoints,
        "sampling": sampling,
        "metric_protocol": {
            "quality_channel": "native FOA W",
            "semantic": "CLAP text/audio and paired reference cosine",
            "distributional": "FAD-VGGish, FD-CLAP, FD-PANN, KL-PANN diagnostic N=50",
            "temporal": "activity IoU/onset/offset from native FOA",
            "spatial": "native FOA plan DoA and trajectory extent",
            "checkpoint_noise": "identical ScenePlan and seed at 100k and 105k",
        },
    }
    _atomic_json(output_root / "EVAL_CONTRACT.json", contract)
    summary = {
        "status": "PASS",
        "output_root": str(output_root),
        "rows": len(panel),
        "domain_counts": dict(counts),
        "checkpoints": [row["step"] for row in checkpoints],
        "semantic_caption_compiler_version": 2,
        "non_speech_v1_v2_text_identical": True,
    }
    _atomic_json(output_root / "BUILD_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
