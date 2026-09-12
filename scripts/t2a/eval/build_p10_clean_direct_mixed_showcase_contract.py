#!/usr/bin/env python3
"""Freeze six clean direct-speech mixed scenes for qualitative P10 review.

The derived panel contains three Speech+Music and three Speech+Sound rows from
the already-frozen, content-disjoint full test set.  Every formal transcript is
a direct utterance: it contains no embedded quotation marks and no conservative
literary attribution suffix such as ``said`` or ``replied``.  The exact semantic
caption compiled for P10 cross-attention is persisted beside each ScenePlan so
the listening report cannot accidentally show a renderer caption or a shortened
transcript in place of the model input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
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
DEFAULT_OUTPUT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_100k_clean_direct_mixed_showcase_v1"
)
DEFAULT_OUTPUT_V2 = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_100k_clean_direct_mixed_showcase_noquote_v2"
)

SELECTION = {
    "speech_plus_music": (
        "full_0005497",
        "full_0005618",
        "full_0005555",
    ),
    "speech_plus_sound": (
        "full_0000912",
        "full_0000818",
        "full_0000882",
    ),
}

QUOTE_MARKS = {'"', "\u201c", "\u201d", "\u2018", "\u2019"}
ATTRIBUTION = re.compile(
    r"\b(?:said|says|replied|asked|answered|cried|shouted|whispered|"
    r"murmured|exclaimed|remarked|continued|added|declared)\b",
    re.IGNORECASE,
)


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


def _speech_source(row: dict[str, Any]) -> dict[str, Any]:
    values = [
        source
        for source in row["scene_plan"]["sources"]
        if source["kind"] == "speech"
    ]
    if len(values) != 1:
        raise RuntimeError(f"{row['panel_id']}: expected one formal Speech source")
    return values[0]


def _validate(row: dict[str, Any], expected_background: str) -> None:
    sources = row["scene_plan"]["sources"]
    speech = _speech_source(row)
    backgrounds = [source for source in sources if source["kind"] != "speech"]
    if len(sources) != 2 or len(backgrounds) != 1:
        raise RuntimeError(
            f"{row['panel_id']}: expected exactly one Speech and one background"
        )
    if backgrounds[0]["kind"] != expected_background:
        raise RuntimeError(
            f"{row['panel_id']}: expected {expected_background}, "
            f"got {backgrounds[0]['kind']}"
        )
    transcript = str(speech["transcript"]).strip()
    if not transcript or any(mark in transcript for mark in QUOTE_MARKS):
        raise RuntimeError(f"{row['panel_id']}: transcript contains quotation marks")
    if ATTRIBUTION.search(transcript):
        raise RuntimeError(f"{row['panel_id']}: transcript contains attribution language")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-eval-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help=(
            "Optional architecture-compatible checkpoint override.  This keeps "
            "the frozen ScenePlans and noise seeds while evaluating a later run."
        ),
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        help="Required with --checkpoint-path; recorded in the frozen contract.",
    )
    parser.add_argument(
        "--semantic-caption-version", type=int, choices=(1, 2), default=2
    )
    args = parser.parse_args()

    source_root = args.source_eval_root.expanduser().resolve(strict=True)
    output_root = (
        args.output_root
        or (DEFAULT_OUTPUT if args.semantic_caption_version == 1 else DEFAULT_OUTPUT_V2)
    ).expanduser().resolve()
    source_contract_path = source_root / "EVAL_CONTRACT.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_panel_path = source_root / source_contract["test_set"]["panel_filename"]
    rows = [
        json.loads(line)
        for line in source_panel_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["panel_id"]): row for row in rows}

    selected: list[dict[str, Any]] = []
    for domain, panel_ids in SELECTION.items():
        expected_background = domain.removeprefix("speech_plus_")
        for panel_id in panel_ids:
            row = copy.deepcopy(by_id[panel_id])
            _validate(row, expected_background)
            row["source_eval_domain"] = row["domain"]
            row["domain"] = domain
            compiler = (
                compile_model_semantic_caption
                if args.semantic_caption_version == 1
                else compile_model_semantic_caption_v2
            )
            compiled = compiler(row["scene_plan"])
            row["model_prompt_text"] = compiled["text"]
            row["model_event_regions"] = compiled["event_regions"]
            row["model_speech_regions"] = compiled["speech_regions"]
            selected.append(row)

    panel_path = output_root / "clean_direct_mixed_6.jsonl"
    _atomic_text(
        panel_path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
    )

    source_checkpoints = [
        item for item in source_contract["checkpoints"] if int(item["step"]) == 100_000
    ]
    if len(source_checkpoints) != 1:
        raise RuntimeError("source contract must expose exactly one 100k checkpoint")
    if (args.checkpoint_path is None) != (args.checkpoint_step is None):
        raise ValueError("--checkpoint-path and --checkpoint-step must be provided together")
    if args.checkpoint_path is None:
        checkpoints = source_checkpoints
        checkpoint_step = 100_000
    else:
        checkpoint_path = args.checkpoint_path.expanduser().resolve(strict=True)
        checkpoint_step = int(args.checkpoint_step)
        if checkpoint_step <= 0:
            raise ValueError("--checkpoint-step must be positive")
        checkpoint_item = copy.deepcopy(source_checkpoints[0])
        checkpoint_item.update(
            {
                "step": checkpoint_step,
                "path": str(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
                "sha256": _sha256(checkpoint_path),
            }
        )
        checkpoints = [checkpoint_item]

    contract = copy.deepcopy(source_contract)
    contract.update(
        {
            "schema": "stable_audio_tools.sceneplan_dit_p10_clean_direct_mixed_contract",
            "schema_version": 1,
            "status": f"FROZEN_{checkpoint_step // 1000}K_CLEAN_DIRECT_MIXED_"
            + ("V1" if args.semantic_caption_version == 1 else "NOQUOTE_V2"),
            "purpose": "review exact P10 prompts for clean direct Speech mixed scenes",
            "checkpoints": checkpoints,
            "source_full_test_contract": str(source_contract_path),
            "source_full_test_contract_sha256": _sha256(source_contract_path),
            "selection": {
                "speech_plus_music": list(SELECTION["speech_plus_music"]),
                "speech_plus_sound": list(SELECTION["speech_plus_sound"]),
                "no_embedded_quotes": True,
                "no_conservative_attribution_verbs": True,
                "all_rows_from_content_disjoint_frozen_test": True,
            },
        }
    )
    contract["sampling"]["common_noise_seed_namespace"] = (
        "sceneplan-p10-v9-100k-clean-direct-mixed-v1-20260830"
    )
    contract["sampling"]["semantic_caption_compiler_version"] = int(
        args.semantic_caption_version
    )
    contract["test_set"].update(
        {
            "evaluation_rows": len(selected),
            "evaluation_subset": "three clean Speech+Music and three clean Speech+Sound",
            "domain_counts": {domain: len(ids) for domain, ids in SELECTION.items()},
            "panel_filename": panel_path.name,
            "panel_sha256": _sha256(panel_path),
            "single_source_domain_pure": False,
        }
    )
    contract_path = output_root / "EVAL_CONTRACT.json"
    _atomic_text(
        contract_path,
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    summary = {
        "status": "PASS",
        "rows": len(selected),
        "domain_counts": contract["test_set"]["domain_counts"],
        "checkpoint_step": checkpoint_step,
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "panel": str(panel_path),
        "panel_sha256": _sha256(panel_path),
        "model_prompt_compiler": (
            f"sceneplan_semantic_caption_v{args.semantic_caption_version}"
        ),
    }
    _atomic_text(
        output_root / "BUILD_SUMMARY.json",
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
