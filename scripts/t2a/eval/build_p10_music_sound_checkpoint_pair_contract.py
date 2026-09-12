#!/usr/bin/env python3
"""Reuse the frozen fixed50 Music/Sound panel for a checkpoint pair."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.sceneplan_dit_p10_panel_common import sha256_file


REVISION_ROOT = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/"
    "revisions/speech_expansion_noalign_15s_v1"
)
DEFAULT_SOURCE = REVISION_ROOT / "evaluation/p10_v10_music_sound_protection_fixed50"
DEFAULT_OUTPUT = REVISION_ROOT / "evaluation/p10_v10_music_sound_protection_105k_110k_fixed50"
DEFAULT_105K = Path(
    "/mnt/sdb/model_archives/p10_pre_v11_20260831/"
    "sceneplan_dit_v10_semantic_v2_protected_resume_110k/"
    "checkpoints/epoch=33-step=105000.ckpt"
)
DEFAULT_110K = Path(
    "/mnt/sdb/model_archives/p10_pre_v11_20260831/"
    "sceneplan_dit_v10_semantic_v2_protected_resume_110k/"
    "checkpoints/epoch=35-step=110000.ckpt"
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _checkpoint(step: int, path: Path) -> dict[str, object]:
    path = path.expanduser().resolve(strict=True)
    return {
        "step": int(step),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-step", type=int, default=105_000)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_105K)
    parser.add_argument("--candidate-step", type=int, default=110_000)
    parser.add_argument("--candidate-checkpoint", type=Path, default=DEFAULT_110K)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    source_contract_path = source_root / "EVAL_CONTRACT.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_panel = source_root / source_contract["test_set"]["panel_filename"]
    if sha256_file(source_panel) != source_contract["test_set"]["panel_sha256"]:
        raise RuntimeError("frozen Music/Sound protection panel changed")

    output_root.mkdir(parents=True, exist_ok=True)
    panel_path = output_root / "music_sound_protection_fixed50.jsonl"
    _atomic_text(panel_path, source_panel.read_text(encoding="utf-8"))
    contract = copy.deepcopy(source_contract)
    contract.update(
        {
            "schema": "stable_audio_tools.p10_music_sound_checkpoint_pair_contract",
            "schema_version": 1,
            "status": "FROZEN",
            "purpose": (
                f"{args.baseline_step}-vs-{args.candidate_step} non-regression gate "
                "on the unchanged fixed50 Music/Sound panel"
            ),
            "source_protection_contract": str(source_contract_path),
            "source_protection_contract_sha256": sha256_file(source_contract_path),
            "checkpoints": [
                _checkpoint(args.baseline_step, args.baseline_checkpoint),
                _checkpoint(args.candidate_step, args.candidate_checkpoint),
            ],
        }
    )
    contract["test_set"]["panel_filename"] = panel_path.name
    contract["test_set"]["panel_sha256"] = sha256_file(panel_path)
    contract_path = output_root / "EVAL_CONTRACT.json"
    _atomic_text(
        contract_path,
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    summary = {
        "status": "PASS",
        "rows": int(contract["test_set"]["evaluation_rows"]),
        "domain_counts": contract["test_set"]["domain_counts"],
        "checkpoints": [args.baseline_step, args.candidate_step],
        "semantic_caption_compiler_version": int(
            contract["sampling"]["semantic_caption_compiler_version"]
        ),
        "panel_sha256": contract["test_set"]["panel_sha256"],
    }
    _atomic_text(
        output_root / "BUILD_SUMMARY.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
