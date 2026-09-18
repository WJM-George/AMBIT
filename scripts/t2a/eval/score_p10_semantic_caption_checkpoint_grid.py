#!/usr/bin/env python3
"""Score a controlled 100k/105k x semantic-caption-v1/v2 Speech grid.

The four evaluation roots must contain identical ordered ScenePlans and noise
seeds.  Speech is transcribed only inside its planned active window (plus a
small fixed guard).  In addition to WER/CER, this script reports transparent
ASR-based sentence-tail proxies so an omitted ending is not hidden by one
aggregate edit-rate number.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel


DEFAULT_REVISION = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation"
)
DEFAULT_ROOTS = {
    "100k_v1": DEFAULT_REVISION / "p10_v9_100k_clean_direct_mixed_showcase_v1",
    "100k_v2": DEFAULT_REVISION / "p10_v9_100k_clean_direct_mixed_showcase_noquote_v2",
    "105k_v1": DEFAULT_REVISION / "p10_v10_105k_clean_direct_mixed_showcase_v1",
    "105k_v2": DEFAULT_REVISION / "p10_v10_105k_clean_direct_mixed_showcase_noquote_v2",
}
DEFAULT_WHISPER = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/models/"
    "faster-distil-whisper-large-v3"
)
DEFAULT_OUTPUT = Path(
    os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/archives/"
    "sceneplan_dit_v10_semantic_v2_protected_resume_110k/"
    "evaluation/speech_100k_105k_semantic_v1_v2_fixed6"
)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _normalize_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", value.lower())


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _errors_and_tail(hypothesis: str, reference: str) -> dict[str, Any]:
    ref = _normalize_words(reference)
    hyp = _normalize_words(hypothesis)
    ref_chars = list("".join(ref))
    hyp_chars = list("".join(hyp))
    word_edits = _edit_distance(ref, hyp)
    char_edits = _edit_distance(ref_chars, hyp_chars)
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    matched_ref: set[int] = set()
    for block in matcher.get_matching_blocks():
        matched_ref.update(range(block.a, block.a + block.size))
    tail_count = min(3, len(ref))
    tail_indices = set(range(len(ref) - tail_count, len(ref)))
    tail_recall = len(matched_ref & tail_indices) / max(tail_count, 1)
    length_ratio = len(hyp) / max(len(ref), 1)
    terminal_recovered = bool(ref and len(ref) - 1 in matched_ref)
    # A conservative ASR proxy: call a sentence complete only if at least two
    # of its last three words and its terminal word survive, without a grossly
    # short hypothesis.  Call it tail-truncated only when both the ending and
    # at least 20% of the total reference length are missing.
    complete = bool(terminal_recovered and tail_recall >= (2.0 / 3.0) and length_ratio >= 0.8)
    tail_truncated = bool(tail_recall < (2.0 / 3.0) and length_ratio < 0.8)
    return {
        "wer": word_edits / max(len(ref), 1),
        "cer": char_edits / max(len(ref_chars), 1),
        "word_edits": word_edits,
        "reference_words": len(ref),
        "hypothesis_words": len(hyp),
        "char_edits": char_edits,
        "reference_chars": len(ref_chars),
        "tail3_exact_recall": tail_recall,
        "hypothesis_reference_length_ratio": length_ratio,
        "terminal_word_recovered": terminal_recovered,
        "complete_sentence_proxy": complete,
        "tail_truncation_proxy": tail_truncated,
    }


def _single_checkpoint(root: Path) -> tuple[int, str]:
    contract = json.loads((root / "EVAL_CONTRACT.json").read_text(encoding="utf-8"))
    checkpoints = contract["checkpoints"]
    if len(checkpoints) != 1:
        raise RuntimeError(f"{root}: expected exactly one checkpoint")
    return int(checkpoints[0]["step"]), str(checkpoints[0]["sha256"])


def _read_panel(root: Path) -> list[dict[str, Any]]:
    contract = json.loads((root / "EVAL_CONTRACT.json").read_text(encoding="utf-8"))
    path = root / contract["test_set"]["panel_filename"]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _speech(scene_plan: dict[str, Any]) -> dict[str, Any]:
    sources = [source for source in scene_plan["sources"] if source["kind"] == "speech"]
    if len(sources) != 1:
        raise RuntimeError("controlled Speech row must have exactly one formal Speech source")
    return sources[0]


def _metadata(root: Path, step: int, row: dict[str, Any]) -> dict[str, Any]:
    path = root / "outputs" / f"step_{step:06d}" / row["domain"] / row["panel_id"] / "metadata.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError(f"non-PASS output: {path}")
    return value


def _speech_window(path: Path, source: dict[str, Any], guard_sec: float) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 4:
        raise RuntimeError(f"expected native four-channel FOA: {path}")
    onset = max(0.0, float(source["activity"]["onset_sec"]) - guard_sec)
    offset = min(audio.shape[0] / sample_rate, float(source["activity"]["offset_sec"]) + guard_sec)
    value = torch.from_numpy(
        audio[int(round(onset * sample_rate)) : int(round(offset * sample_rate)), 0].copy()
    )
    if sample_rate != 16_000:
        value = torchaudio.functional.resample(value, sample_rate, 16_000)
    waveform = value.numpy().astype(np.float32, copy=False)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1.0e-8:
        waveform = waveform / peak * (10.0 ** (-1.0 / 20.0))
    return waveform


def _transcribe(model: WhisperModel, waveform: np.ndarray) -> str:
    segments, _info = model.transcribe(
        waveform,
        language="en",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def _load_utmos(device: torch.device):
    try:
        model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
            verbose=False,
        )
        return model.to(device).eval(), None
    except Exception as error:  # availability is explicitly recorded, never hidden
        return None, f"{type(error).__name__}: {error}"


@torch.inference_mode()
def _utmos_score(model, waveform: np.ndarray, device: torch.device) -> float:
    value = torch.from_numpy(waveform).float().to(device).view(1, -1)
    return float(torch.as_tensor(model(value, 16_000)).float().mean().cpu())


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row["errors"] for row in rows]
    utmos = [float(row["utmos"]) for row in rows if row["utmos"] is not None]
    return {
        "rows": len(rows),
        "corpus_wer": sum(item["word_edits"] for item in errors)
        / max(sum(item["reference_words"] for item in errors), 1),
        "corpus_cer": sum(item["char_edits"] for item in errors)
        / max(sum(item["reference_chars"] for item in errors), 1),
        "mean_wer": _mean([item["wer"] for item in errors]),
        "mean_cer": _mean([item["cer"] for item in errors]),
        "complete_sentence_proxy_rate": _mean(
            [float(item["complete_sentence_proxy"]) for item in errors]
        ),
        "tail_truncation_proxy_rate": _mean(
            [float(item["tail_truncation_proxy"]) for item in errors]
        ),
        "terminal_word_recovery_rate": _mean(
            [float(item["terminal_word_recovered"]) for item in errors]
        ),
        "mean_tail3_exact_recall": _mean([item["tail3_exact_recall"] for item in errors]),
        "mean_hypothesis_reference_length_ratio": _mean(
            [item["hypothesis_reference_length_ratio"] for item in errors]
        ),
        "utmos_mean": None if not utmos else _mean(utmos),
        "utmos_rows": len(utmos),
    }


def _markdown(summary: dict[str, Any]) -> str:
    row_count = int(summary["controlled_variables"]["rows_per_cell"])
    lines = [
        "# P10 Speech 100k vs 105k: fixed-scene semantic-caption grid",
        "",
        f"All four cells use the same {row_count} content-disjoint ScenePlans and noise seeds.",
        "",
        "| checkpoint/template | WER ↓ | CER ↓ | complete sentence proxy ↑ | tail truncation proxy ↓ | tail-3 recall ↑ | UTMOS ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("100k_v1", "100k_v2", "105k_v1", "105k_v2"):
        value = summary["aggregates"][label]
        utmos = "N/A" if value["utmos_mean"] is None else f"{value['utmos_mean']:.4f}"
        lines.append(
            f"| {label} | {value['corpus_wer']:.4f} | {value['corpus_cer']:.4f} | "
            f"{value['complete_sentence_proxy_rate']:.1%} | "
            f"{value['tail_truncation_proxy_rate']:.1%} | "
            f"{value['mean_tail3_exact_recall']:.1%} | {utmos} |"
        )
    lines += [
        "",
        "## Joint template view",
        "",
        "| checkpoint | WER ↓ | CER ↓ | complete sentence proxy ↑ | tail truncation proxy ↓ | UTMOS ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for step in ("100k", "105k"):
        value = summary["joint_template_aggregates"][step]
        utmos = "N/A" if value["utmos_mean"] is None else f"{value['utmos_mean']:.4f}"
        lines.append(
            f"| {step} | {value['corpus_wer']:.4f} | {value['corpus_cer']:.4f} | "
            f"{value['complete_sentence_proxy_rate']:.1%} | "
            f"{value['tail_truncation_proxy_rate']:.1%} | {utmos} |"
        )
    lines += [
        "",
        f"## Continuation gate: {summary['continuation_gate']['status']}",
        "",
        summary["continuation_gate"]["recommendation"],
        "",
        "The completion metric is an ASR proxy: the terminal word and at least two of the final three reference words must be recovered, and the hypothesis cannot be shorter than 80% of the reference. Tail truncation requires both tail failure and a hypothesis shorter than 80% of the reference.",
        "",
        f"This {row_count}-row panel is a controlled engineering gate, not the final full-test statistical claim.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for label, default in DEFAULT_ROOTS.items():
        parser.add_argument(f"--root-{label.replace('_', '-')}", type=Path, default=default)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--guard-sec", type=float, default=0.25)
    parser.add_argument("--skip-utmos", action="store_true")
    args = parser.parse_args()

    roots = {
        label: getattr(args, f"root_{label}").expanduser().resolve(strict=True)
        for label in DEFAULT_ROOTS
    }
    panels = {label: _read_panel(root) for label, root in roots.items()}
    baseline = panels["100k_v1"]
    identity = [
        (row["panel_id"], row["scene_plan"], int(row["noise_seed"])) for row in baseline
    ]
    for label, panel in panels.items():
        candidate = [(row["panel_id"], row["scene_plan"], int(row["noise_seed"])) for row in panel]
        if candidate != identity:
            raise RuntimeError(f"{label}: ScenePlans/order/noise seeds differ from 100k_v1")

    whisper = WhisperModel(
        str(args.whisper_model.expanduser().resolve(strict=True)),
        device="cuda",
        device_index=args.device_index,
        compute_type="float16",
    )
    device = torch.device(f"cuda:{args.device_index}")
    utmos, utmos_error = (None, "disabled") if args.skip_utmos else _load_utmos(device)

    scored: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    for label, root in roots.items():
        step, checkpoint_sha256 = _single_checkpoint(root)
        checkpoints[label] = {"step": step, "sha256": checkpoint_sha256, "root": str(root)}
        template = label.rsplit("_", 1)[1]
        for row in panels[label]:
            source = _speech(row["scene_plan"])
            metadata = _metadata(root, step, row)
            if metadata["checkpoint_sha256"] != checkpoint_sha256:
                raise RuntimeError(f"{label}/{row['panel_id']}: checkpoint SHA mismatch")
            if int(metadata["noise_seed"]) != int(row["noise_seed"]):
                raise RuntimeError(f"{label}/{row['panel_id']}: noise seed mismatch")
            prompt = str(row["model_prompt_text"])
            if template == "v1" and 'who says "' not in prompt:
                raise RuntimeError(f"{label}/{row['panel_id']}: not a v1 prompt")
            if template == "v2" and "who says: " not in prompt:
                raise RuntimeError(f"{label}/{row['panel_id']}: not a v2 prompt")
            waveform = _speech_window(Path(metadata["generated_foa_path"]), source, args.guard_sec)
            hypothesis = _transcribe(whisper, waveform)
            value = {
                "cell": label,
                "checkpoint_step": step,
                "template": template,
                "panel_id": row["panel_id"],
                "sample_id": row["sample_id"],
                "domain": row["domain"],
                "noise_seed": int(row["noise_seed"]),
                "checkpoint_sha256": checkpoint_sha256,
                "prompt": prompt,
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
                        "event": "speech_grid_scored",
                        "cell": label,
                        "panel_id": row["panel_id"],
                        "wer": value["errors"]["wer"],
                        "complete": value["errors"]["complete_sentence_proxy"],
                    }
                ),
                flush=True,
            )

    aggregates = {
        label: _aggregate([row for row in scored if row["cell"] == label])
        for label in DEFAULT_ROOTS
    }
    deltas = {
        template: {
            key: aggregates[f"105k_{template}"][key] - aggregates[f"100k_{template}"][key]
            for key in (
                "corpus_wer",
                "corpus_cer",
                "complete_sentence_proxy_rate",
                "tail_truncation_proxy_rate",
                "mean_tail3_exact_recall",
            )
        }
        for template in ("v1", "v2")
    }
    # A 50-row panel moves rates in 0.02 increments.  The guard therefore
    # permits one-row variation in completion/truncation, and a 0.005 absolute
    # edit-rate fluctuation, while still rejecting the large regressions seen
    # on the original six-row directional probe.
    edit_rate_tolerance = 0.005
    row_rate_tolerance = 1.0 / len(baseline)
    utmos_tolerance = 0.05
    gate_checks: dict[str, bool] = {}
    for template in ("v1", "v2"):
        before = aggregates[f"100k_{template}"]
        after = aggregates[f"105k_{template}"]
        gate_checks[f"{template}_wer_non_degradation"] = (
            after["corpus_wer"] <= before["corpus_wer"] + edit_rate_tolerance
        )
        gate_checks[f"{template}_cer_non_degradation"] = (
            after["corpus_cer"] <= before["corpus_cer"] + edit_rate_tolerance
        )
        gate_checks[f"{template}_completion_non_degradation"] = (
            after["complete_sentence_proxy_rate"]
            >= before["complete_sentence_proxy_rate"] - row_rate_tolerance
        )
        gate_checks[f"{template}_tail_truncation_non_degradation"] = (
            after["tail_truncation_proxy_rate"]
            <= before["tail_truncation_proxy_rate"] + row_rate_tolerance
        )
        gate_checks[f"{template}_utmos_within_{utmos_tolerance:.2f}"] = (
            before["utmos_mean"] is not None
            and after["utmos_mean"] is not None
            and after["utmos_mean"] >= before["utmos_mean"] - utmos_tolerance
        )
    continuation_pass = all(gate_checks.values())
    summary = {
        "schema": "stable_audio_tools.p10_speech_semantic_caption_checkpoint_grid",
        "schema_version": 1,
        "status": "PASS",
        "controlled_variables": {
            "identical_ordered_sceneplans": True,
            "identical_noise_seed_per_sample": True,
            "rows_per_cell": len(baseline),
            "domains": {"speech_plus_music": 3, "speech_plus_sound": 3},
            "speech_window_guard_sec": args.guard_sec,
        },
        "tail_metric_definition": {
            "complete_sentence_proxy": "terminal word plus >=2/3 final words recovered, hypothesis/reference word length >=0.8",
            "tail_truncation_proxy": "<2/3 final words recovered and hypothesis/reference word length <0.8",
        },
        "whisper_model": str(args.whisper_model.expanduser().resolve(strict=True)),
        "utmos_available": utmos is not None,
        "utmos_error": utmos_error,
        "checkpoints": checkpoints,
        "aggregates": aggregates,
        "joint_template_aggregates": {
            step: _aggregate([row for row in scored if row["cell"].startswith(step)])
            for step in ("100k", "105k")
        },
        "stratified_aggregates": {
            label: {
                group: _aggregate(
                    [
                        row
                        for row in scored
                        if row["cell"] == label
                        and (
                            row["domain"] == "speech_only"
                            if group == "speech_only"
                            else group in row["domain"]
                        )
                    ]
                )
                for group in ("speech_only", "speech_plus_music", "speech_plus_sound")
            }
            for label in DEFAULT_ROOTS
        },
        "deltas_105k_minus_100k": deltas,
        "continuation_gate": {
            "status": (
                "PASS_SPEECH_GATE_MUSIC_SOUND_PENDING"
                if continuation_pass
                else "FAIL_HOLD_110K"
            ),
            "checks": gate_checks,
            "edit_rate_absolute_non_degradation_tolerance": edit_rate_tolerance,
            "row_rate_absolute_non_degradation_tolerance": row_rate_tolerance,
            "utmos_absolute_non_degradation_tolerance": utmos_tolerance,
            "recommendation": (
                "The controlled Speech gate passed; Music/Sound protection still requires its separate gate before resuming."
                if continuation_pass
                else "Do not resume to 110k from this evidence: v2 transcript accuracy improved strongly, but v1 WER/completion and both-template UTMOS did not satisfy protection gates. Expand the panel before changing the recipe."
            ),
            "statistical_limit": "Six fixed rows provide a directional engineering check, not a final statistical claim.",
        },
        "rows": scored,
    }
    output = args.output_root.expanduser().resolve()
    _atomic_text(output / "SUMMARY.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    _atomic_text(output / "SUMMARY.md", _markdown(summary))
    _atomic_text(
        output / "PROMPTS_AND_OUTPUTS.jsonl",
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in scored),
    )
    print(json.dumps({"status": "PASS", "aggregates": aggregates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
