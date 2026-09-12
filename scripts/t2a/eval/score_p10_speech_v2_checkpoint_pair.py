#!/usr/bin/env python3
"""Score the same fixed50 Speech panel at two v2-only checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from faster_whisper import WhisperModel


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval.score_p10_semantic_caption_checkpoint_grid import (
    DEFAULT_WHISPER,
    _aggregate,
    _errors_and_tail,
    _load_utmos,
    _metadata,
    _read_panel,
    _single_checkpoint,
    _speech,
    _speech_window,
    _transcribe,
    _utmos_score,
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--guard-sec", type=float, default=0.25)
    args = parser.parse_args()

    roots = {
        "baseline": args.baseline_root.expanduser().resolve(strict=True),
        "candidate": args.candidate_root.expanduser().resolve(strict=True),
    }
    panels = {label: _read_panel(root) for label, root in roots.items()}
    identity = [
        (row["panel_id"], row["scene_plan"], int(row["noise_seed"]))
        for row in panels["baseline"]
    ]
    for label, panel in panels.items():
        candidate_identity = [
            (row["panel_id"], row["scene_plan"], int(row["noise_seed"]))
            for row in panel
        ]
        if candidate_identity != identity:
            raise RuntimeError(f"{label}: fixed50 ScenePlans/order/noise changed")
        if any("who says: " not in str(row["model_prompt_text"]) for row in panel):
            raise RuntimeError(f"{label}: panel is not entirely semantic-caption v2")

    whisper = WhisperModel(
        str(args.whisper_model.expanduser().resolve(strict=True)),
        device="cuda",
        device_index=args.device_index,
        compute_type="float16",
    )
    device = torch.device(f"cuda:{args.device_index}")
    utmos, utmos_error = _load_utmos(device)
    scored = []
    checkpoints = {}
    for label, root in roots.items():
        step, checkpoint_sha256 = _single_checkpoint(root)
        checkpoints[label] = {
            "step": step,
            "sha256": checkpoint_sha256,
            "root": str(root),
        }
        for row in panels[label]:
            source = _speech(row["scene_plan"])
            metadata = _metadata(root, step, row)
            if metadata["checkpoint_sha256"] != checkpoint_sha256:
                raise RuntimeError(f"{label}/{row['panel_id']}: checkpoint SHA changed")
            waveform = _speech_window(
                Path(metadata["generated_foa_path"]), source, args.guard_sec
            )
            hypothesis = _transcribe(whisper, waveform)
            value = {
                "cell": label,
                "checkpoint_step": step,
                "panel_id": row["panel_id"],
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "noise_seed": int(row["noise_seed"]),
                "prompt": row["model_prompt_text"],
                "transcript": source["transcript"],
                "asr_hypothesis": hypothesis,
                "errors": _errors_and_tail(hypothesis, source["transcript"]),
                "utmos": None if utmos is None else _utmos_score(utmos, waveform, device),
                "generated_foa_path": metadata["generated_foa_path"],
                "generated_stereo_path": metadata["generated_stereo_path"],
            }
            scored.append(value)
            print(
                json.dumps(
                    {
                        "event": "speech_v2_pair_scored",
                        "cell": label,
                        "panel_id": row["panel_id"],
                        "wer": value["errors"]["wer"],
                    }
                ),
                flush=True,
            )

    aggregates = {
        label: _aggregate([row for row in scored if row["cell"] == label])
        for label in roots
    }
    before, after = aggregates["baseline"], aggregates["candidate"]
    tolerance = 1.0 / len(identity)
    checks = {
        "wer_non_degradation": after["corpus_wer"] <= before["corpus_wer"] + 0.005,
        "cer_non_degradation": after["corpus_cer"] <= before["corpus_cer"] + 0.005,
        "completion_non_degradation": (
            after["complete_sentence_proxy_rate"]
            >= before["complete_sentence_proxy_rate"] - tolerance
        ),
        "tail_truncation_non_degradation": (
            after["tail_truncation_proxy_rate"]
            <= before["tail_truncation_proxy_rate"] + tolerance
        ),
        "utmos_non_degradation": (
            before["utmos_mean"] is not None
            and after["utmos_mean"] is not None
            and after["utmos_mean"] >= before["utmos_mean"] - 0.05
        ),
    }
    status = "PASS" if all(checks.values()) else "HOLD"
    summary = {
        "schema": "stable_audio_tools.p10_speech_v2_checkpoint_pair",
        "schema_version": 1,
        "status": status,
        "rows_per_checkpoint": len(identity),
        "checkpoints": checkpoints,
        "aggregates": aggregates,
        "delta_candidate_minus_baseline": {
            key: (
                None
                if after[key] is None or before[key] is None
                else after[key] - before[key]
            )
            for key in (
                "corpus_wer",
                "corpus_cer",
                "complete_sentence_proxy_rate",
                "tail_truncation_proxy_rate",
                "utmos_mean",
            )
        },
        "checks": checks,
        "utmos_error": utmos_error,
    }
    output = args.output_root.expanduser().resolve()
    _atomic_text(output / "SUMMARY.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    baseline_step = checkpoints["baseline"]["step"] // 1000
    candidate_step = checkpoints["candidate"]["step"] // 1000
    lines = [
        f"# P10 Speech v2 fixed50: {baseline_step}k vs {candidate_step}k",
        "",
        f"Gate: **{status}**",
        "",
        "| step | WER ↓ | CER ↓ | complete ↑ | tail truncation ↓ | UTMOS ↑ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("baseline", "candidate"):
        value = aggregates[label]
        step = checkpoints[label]["step"] // 1000
        utmos_text = (
            "N/A" if value["utmos_mean"] is None else f"{value['utmos_mean']:.4f}"
        )
        lines.append(
            f"| {step}k | {value['corpus_wer']:.4f} | {value['corpus_cer']:.4f} | "
            f"{value['complete_sentence_proxy_rate']:.1%} | "
            f"{value['tail_truncation_proxy_rate']:.1%} | "
            f"{utmos_text} |"
        )
    _atomic_text(output / "SUMMARY.md", "\n".join(lines) + "\n")
    _atomic_text(
        output / "PROMPTS_AND_OUTPUTS.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in scored),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
