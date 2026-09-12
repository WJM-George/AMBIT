#!/usr/bin/env python3
"""Audit and compare legacy-quote versus explicit-role P10 prompts.

Both roots must contain the same six frozen ScenePlans rendered from the same
checkpoint and noise seeds.  The script transcribes the planned Speech-active
window, reports WER/CER, and creates a deterministic listening montage in the
order ``v1 legacy quotes -> silence -> v2 explicit roles``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel


DEFAULT_V1_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_100k_clean_direct_mixed_showcase_v1"
)
DEFAULT_V2_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/evaluation/"
    "p10_v9_100k_clean_direct_mixed_showcase_noquote_v2"
)
DEFAULT_WHISPER = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/models/"
    "faster-distil-whisper-large-v3"
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


def _atomic_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}{path.suffix}")
    sf.write(temporary, audio, sample_rate, subtype="PCM_16")
    temporary.replace(path)


def _read_panel(root: Path) -> list[dict[str, Any]]:
    contract = json.loads((root / "EVAL_CONTRACT.json").read_text(encoding="utf-8"))
    panel = root / contract["test_set"]["panel_filename"]
    return [
        json.loads(line)
        for line in panel.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _speech(scene_plan: dict[str, Any]) -> dict[str, Any]:
    values = [source for source in scene_plan["sources"] if source["kind"] == "speech"]
    if len(values) != 1:
        raise RuntimeError("A/B scene must contain exactly one formal Speech source")
    return values[0]


def _metadata(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = (
        root
        / "outputs"
        / "step_100000"
        / row["domain"]
        / row["panel_id"]
        / "metadata.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS":
        raise RuntimeError(f"non-PASS output: {path}")
    return value


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


def _errors(hypothesis: str, reference: str) -> dict[str, Any]:
    ref_words = _normalize_words(reference)
    hyp_words = _normalize_words(hypothesis)
    ref_chars = list("".join(ref_words))
    hyp_chars = list("".join(hyp_words))
    word_edits = _edit_distance(ref_words, hyp_words)
    char_edits = _edit_distance(ref_chars, hyp_chars)
    return {
        "wer": word_edits / max(len(ref_words), 1),
        "cer": char_edits / max(len(ref_chars), 1),
        "word_edits": word_edits,
        "reference_words": len(ref_words),
        "char_edits": char_edits,
        "reference_chars": len(ref_chars),
    }


def _speech_window(path: Path, source: dict[str, Any], guard_sec: float) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] != 4:
        raise RuntimeError(f"expected native four-channel FOA: {path}")
    onset = max(0.0, float(source["activity"]["onset_sec"]) - guard_sec)
    offset = min(audio.shape[0] / sample_rate, float(source["activity"]["offset_sec"]) + guard_sec)
    start = int(round(onset * sample_rate))
    stop = int(round(offset * sample_rate))
    waveform = torch.from_numpy(audio[start:stop, 0].copy())
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    value = waveform.numpy().astype(np.float32, copy=False)
    peak = float(np.max(np.abs(value))) if value.size else 0.0
    if peak > 1.0e-8:
        value = value / peak * (10.0 ** (-1.0 / 20.0))
    return value


def _transcribe(model: WhisperModel, waveform: np.ndarray) -> str:
    segments, _info = model.transcribe(
        waveform,
        language="en",
        beam_size=5,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return " ".join(segment.text.strip() for segment in segments).strip()


def _aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    errors = [row[key]["errors"] for row in rows]
    return {
        "rows": len(errors),
        "mean_wer": float(np.mean([value["wer"] for value in errors])),
        "corpus_wer": sum(value["word_edits"] for value in errors)
        / max(sum(value["reference_words"] for value in errors), 1),
        "mean_cer": float(np.mean([value["cer"] for value in errors])),
        "corpus_cer": sum(value["char_edits"] for value in errors)
        / max(sum(value["reference_chars"] for value in errors), 1),
    }


def _montage(v1_path: Path, v2_path: Path, output: Path) -> None:
    v1, sr1 = sf.read(v1_path, dtype="float32", always_2d=True)
    v2, sr2 = sf.read(v2_path, dtype="float32", always_2d=True)
    if sr1 != sr2 or v1.shape[1] != 2 or v2.shape[1] != 2:
        raise RuntimeError("A/B previews must be matching stereo files")
    silence = np.zeros((int(round(0.75 * sr1)), 2), dtype=np.float32)
    _atomic_wav(output, np.concatenate((v1, silence, v2), axis=0), sr1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1-root", type=Path, default=DEFAULT_V1_ROOT)
    parser.add_argument("--v2-root", type=Path, default=DEFAULT_V2_ROOT)
    parser.add_argument("--whisper-model", type=Path, default=DEFAULT_WHISPER)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--guard-sec", type=float, default=0.25)
    args = parser.parse_args()

    v1_root = args.v1_root.expanduser().resolve(strict=True)
    v2_root = args.v2_root.expanduser().resolve(strict=True)
    v1_rows = _read_panel(v1_root)
    v2_rows = _read_panel(v2_root)
    if [row["panel_id"] for row in v1_rows] != [row["panel_id"] for row in v2_rows]:
        raise RuntimeError("v1/v2 panels do not contain the same ordered samples")

    whisper = WhisperModel(
        str(args.whisper_model.expanduser().resolve(strict=True)),
        device="cuda",
        device_index=args.device_index,
        compute_type="float16",
    )
    results: list[dict[str, Any]] = []
    montage_root = v2_root / "ab_listening"
    for v1_row, v2_row in zip(v1_rows, v2_rows):
        if v1_row["scene_plan"] != v2_row["scene_plan"]:
            raise RuntimeError(f"{v1_row['panel_id']}: ScenePlan changed across A/B")
        if int(v1_row["noise_seed"]) != int(v2_row["noise_seed"]):
            raise RuntimeError(f"{v1_row['panel_id']}: noise seed changed across A/B")
        if 'who says "' not in v1_row["model_prompt_text"]:
            raise RuntimeError(f"{v1_row['panel_id']}: v1 prompt has no legacy delimiter")
        if "who says: " not in v2_row["model_prompt_text"]:
            raise RuntimeError(f"{v1_row['panel_id']}: v2 prompt has no explicit separator")

        v1_meta = _metadata(v1_root, v1_row)
        v2_meta = _metadata(v2_root, v2_row)
        if (
            v1_meta["checkpoint_sha256"] != v2_meta["checkpoint_sha256"]
            or int(v1_meta["noise_seed"]) != int(v2_meta["noise_seed"])
        ):
            raise RuntimeError(f"{v1_row['panel_id']}: checkpoint/noise mismatch")
        source = _speech(v1_row["scene_plan"])
        transcript = str(source["transcript"])
        variants = {}
        for name, metadata in (("v1", v1_meta), ("v2", v2_meta)):
            waveform = _speech_window(
                Path(metadata["generated_foa_path"]), source, args.guard_sec
            )
            hypothesis = _transcribe(whisper, waveform)
            variants[name] = {
                "asr": hypothesis,
                "errors": _errors(hypothesis, transcript),
                "generated_foa_path": metadata["generated_foa_path"],
                "generated_stereo_path": metadata["generated_stereo_path"],
            }
        montage_path = montage_root / f"{v1_row['panel_id']}_v1_then_v2.wav"
        _montage(
            Path(v1_meta["generated_stereo_path"]),
            Path(v2_meta["generated_stereo_path"]),
            montage_path,
        )
        result = {
            "panel_id": v1_row["panel_id"],
            "domain": v1_row["domain"],
            "sample_id": v1_row["sample_id"],
            "transcript": transcript,
            "v1_prompt": v1_row["model_prompt_text"],
            "v2_prompt": v2_row["model_prompt_text"],
            "noise_seed": int(v1_row["noise_seed"]),
            "checkpoint_sha256": v1_meta["checkpoint_sha256"],
            "v1": variants["v1"],
            "v2": variants["v2"],
            "montage_path": str(montage_path),
            "montage_sha256": _sha256(montage_path),
        }
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "ab_scored",
                    "panel_id": result["panel_id"],
                    "v1_wer": result["v1"]["errors"]["wer"],
                    "v2_wer": result["v2"]["errors"]["wer"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    summary = {
        "schema": "stable_audio_tools.p10_semantic_caption_v1_v2_ab",
        "schema_version": 1,
        "status": "PASS",
        "controlled_variables": {
            "same_sceneplans": True,
            "same_checkpoint": True,
            "same_noise_seeds": True,
            "same_sceneplan_44": True,
            "only_prompt_separator_changed": True,
        },
        "speech_window_guard_sec": args.guard_sec,
        "v1": _aggregate(results, "v1"),
        "v2": _aggregate(results, "v2"),
        "rows": results,
    }
    report_path = v2_root / "SEMANTIC_CAPTION_V1_V2_AB.json"
    _atomic_text(report_path, json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "v1", "v2")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
