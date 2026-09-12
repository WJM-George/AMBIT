#!/usr/bin/env python3
"""Freeze a stratified, clean-direct 50-row Speech panel for P10 A/B tests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.model_sceneplan import (
    compile_model_semantic_caption,
    compile_model_semantic_caption_v2,
)


DEFAULT_SOURCE = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_full_test_8000_ckpt20k_100k_v1"
)

ATTRIBUTION = re.compile(
    r"\b(?:said|says|replied|asked|answered|cried|shouted|whispered|"
    r"murmured|exclaimed|remarked|continued|added|declared)\b",
    re.IGNORECASE,
)
QUOTES = {'"', "\u201c", "\u201d", "\u2018", "\u2019"}

# Exactly 25 seen and 25 unseen speakers; mixed rows cover overlap and sequence.
QUOTAS = {
    ("speech_only", False): 10,
    ("speech_only", True): 10,
    ("speech_plus_music_overlapping", False): 4,
    ("speech_plus_music_overlapping", True): 4,
    ("speech_plus_music_sequential", False): 4,
    ("speech_plus_music_sequential", True): 3,
    ("speech_plus_sound_overlapping", False): 4,
    ("speech_plus_sound_overlapping", True): 4,
    ("speech_plus_sound_sequential", False): 3,
    ("speech_plus_sound_sequential", True): 4,
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


def _speech(row: dict[str, Any]) -> dict[str, Any] | None:
    values = [source for source in row["scene_plan"]["sources"] if source["kind"] == "speech"]
    return values[0] if len(values) == 1 else None


def _stratum(row: dict[str, Any]) -> str | None:
    kinds = [source["kind"] for source in row["scene_plan"]["sources"]]
    if kinds == ["speech"]:
        return "speech_only"
    if len(kinds) != 2 or kinds.count("speech") != 1:
        return None
    background = "music" if "music" in kinds else "sound" if "sound" in kinds else None
    if background is None:
        return None
    composition = str(row["scene_composition"])
    if composition == "speech_with_overlapping_background":
        relation = "overlapping"
    elif composition == "speech_with_sequential_background":
        relation = "sequential"
    else:
        return None
    return f"speech_plus_{background}_{relation}"


def _eligible(row: dict[str, Any]) -> bool:
    source = _speech(row)
    if source is None or int(row["latent_frames_valid"]) > 648:
        return False
    transcript = str(source["transcript"]).strip()
    words = re.findall(r"[a-z0-9']+", transcript.lower())
    return bool(
        5 <= len(words) <= 25
        and not any(mark in transcript for mark in QUOTES)
        and not ATTRIBUTION.search(transcript)
        and _stratum(row) is not None
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-eval-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--semantic-caption-version", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()

    source_root = args.source_eval_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    checkpoint_path = args.checkpoint_path.expanduser().resolve(strict=True)
    source_contract_path = source_root / "EVAL_CONTRACT.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_panel = source_root / source_contract["test_set"]["panel_filename"]
    rows = [json.loads(line) for line in source_panel.read_text(encoding="utf-8").splitlines() if line]

    pools: dict[tuple[str, bool], list[dict[str, Any]]] = {key: [] for key in QUOTAS}
    for row in rows:
        if not _eligible(row):
            continue
        key = (_stratum(row), bool(row["speech_seen_speaker"]))
        if key in pools:
            pools[key].append(row)

    selected: list[dict[str, Any]] = []
    compiler = compile_model_semantic_caption if args.semantic_caption_version == 1 else compile_model_semantic_caption_v2
    for key, count in QUOTAS.items():
        pool = pools[key]
        if len(pool) < count:
            raise RuntimeError(f"insufficient held-out rows for {key}: {len(pool)} < {count}")
        for original in pool[:count]:
            row = copy.deepcopy(original)
            row["source_eval_domain"] = row["domain"]
            row["domain"] = key[0]
            compiled = compiler(row["scene_plan"])
            row["model_prompt_text"] = compiled["text"]
            row["model_event_regions"] = compiled["event_regions"]
            row["model_speech_regions"] = compiled["speech_regions"]
            selected.append(row)

    panel_path = output_root / "speech_fixed50.jsonl"
    _atomic_text(
        panel_path,
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
    )

    source_100k = [item for item in source_contract["checkpoints"] if int(item["step"]) == 100_000]
    if len(source_100k) != 1:
        raise RuntimeError("source contract must expose exactly one 100k checkpoint")
    checkpoint = copy.deepcopy(source_100k[0])
    checkpoint.update(
        {
            "step": int(args.checkpoint_step),
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": _sha256(checkpoint_path),
        }
    )
    domain_counts = dict(Counter(str(row["domain"]) for row in selected))
    contract = copy.deepcopy(source_contract)
    contract.update(
        {
            "schema": "stable_audio_tools.p10_speech_fixed50_semantic_grid_contract",
            "schema_version": 1,
            "status": "FROZEN",
            "purpose": "controlled 100k/105k x v1/v2 held-out Speech comparison",
            "checkpoints": [checkpoint],
            "source_full_test_contract": str(source_contract_path),
            "source_full_test_contract_sha256": _sha256(source_contract_path),
        }
    )
    contract["sampling"]["semantic_caption_compiler_version"] = int(args.semantic_caption_version)
    contract["test_set"].update(
        {
            "evaluation_rows": len(selected),
            "evaluation_subset": "50 clean direct held-out Speech rows; 20 only, 15 +music, 15 +sound",
            "domain_counts": domain_counts,
            "panel_filename": panel_path.name,
            "panel_sha256": _sha256(panel_path),
            "single_source_domain_pure": False,
            "max_latent_frames": 648,
        }
    )
    contract["selection"] = {
        "quotas": {f"{key[0]}|seen={key[1]}": value for key, value in QUOTAS.items()},
        "total": len(selected),
        "seen_speakers": sum(bool(row["speech_seen_speaker"]) for row in selected),
        "unseen_speakers": sum(not bool(row["speech_seen_speaker"]) for row in selected),
        "direct_transcript_no_quotes_or_attribution": True,
        "transcript_words": [5, 25],
        "max_latent_frames": 648,
        "content_disjoint_frozen_test_inherited": True,
    }
    contract_path = output_root / "EVAL_CONTRACT.json"
    _atomic_text(contract_path, json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    summary = {
        "status": "PASS",
        "rows": len(selected),
        "domain_counts": domain_counts,
        "seen_speakers": contract["selection"]["seen_speakers"],
        "unseen_speakers": contract["selection"]["unseen_speakers"],
        "checkpoint_step": int(args.checkpoint_step),
        "semantic_caption_version": int(args.semantic_caption_version),
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "panel": str(panel_path),
        "panel_sha256": _sha256(panel_path),
    }
    _atomic_text(output_root / "BUILD_SUMMARY.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
