#!/usr/bin/env python3
"""Prepare or finalize speaker descriptions for revision-6 speech donors.

Known LibriTTS/HiFiTTS speakers reuse the frozen speaker identity registry.
Only speakers absent from that registry (principally HiFiTTS-2) are annotated:
three deterministic clean utterances are concatenated once and sent to
Qwen3-Omni-Instruct.  Exact utterance transcripts are never passed through or
rewritten by this stage.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.data.sceneplan_v2_common import load_parquet_source  # noqa: E402


DATASET_ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m")
REVISION_ROOT = DATASET_ROOT / "revisions/speech_expansion_noalign_15s_v1"
DEFAULT_DONORS = REVISION_ROOT / "sources/registry/final_speech_donors.parquet"
DEFAULT_EXISTING = DATASET_ROOT / (
    "source_annotations/speech_speaker_instruct_v1/registry/"
    "speech_speaker_description_registry.parquet"
)
DEFAULT_OUTPUT_ROOT = REVISION_ROOT / "source_annotations/speech_speaker_instruct_v1"
TARGET_SAMPLE_RATE = 24_000
MAX_SPEAKER_EXCERPT_SEC = 6.0
MAX_SPEAKER_EXCERPT_SAMPLES = round(
    TARGET_SAMPLE_RATE * MAX_SPEAKER_EXCERPT_SEC
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def known_speakers(path: Path) -> dict[str, str]:
    table = pq.read_table(
        path, columns=["speaker_key", "speaker_identity_description"]
    )
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for speaker, description in zip(
        table["speaker_key"].to_pylist(),
        table["speaker_identity_description"].to_pylist(),
    ):
        counts[str(speaker)][str(description)] += 1
    return {
        speaker: sorted(values.items(), key=lambda item: (-item[1], item[0]))[0][0]
        for speaker, values in counts.items()
    }


def speaker_excerpt(mono: np.ndarray, sample_rate: int) -> np.ndarray:
    """Create a bounded voice-identity excerpt without touching donor audio.

    Revision-6 donors may legitimately exceed the legacy 10-second envelope.
    Qwen only needs a representative timbre sample for speaker-description
    annotation, so this auxiliary view uses a deterministic center excerpt.
    The complete utterance, exact transcript, ScenePlan, and P8 render remain
    untouched and retain their full 0--15 second geometry.
    """

    mono = np.asarray(mono, dtype=np.float32)
    if mono.ndim != 1:
        raise RuntimeError(f"speaker excerpt must be mono, got {mono.shape}")
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        mono = resample_poly(
            mono,
            TARGET_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32, copy=False)
    if len(mono) > MAX_SPEAKER_EXCERPT_SAMPLES:
        start = (len(mono) - MAX_SPEAKER_EXCERPT_SAMPLES) // 2
        mono = mono[start : start + MAX_SPEAKER_EXCERPT_SAMPLES]
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak <= 1.0e-5 or not np.isfinite(mono).all():
        raise RuntimeError("invalid speaker excerpt")
    if peak > 0.98:
        mono = mono * (0.98 / peak)
    return np.asarray(mono, dtype=np.float32)


def read_donor_audio(row: dict[str, Any]) -> np.ndarray:
    source_path = str(row.get("source_audio_path") or "")
    if source_path:
        path = Path(source_path).resolve(strict=True)
        if sha256_file(path) != str(row["source_audio_sha256"]):
            raise RuntimeError(f"{row['asset_id']}: speaker source hash changed")
        samples, sample_rate = sf.read(
            str(path),
            dtype="float32",
            always_2d=True,
        )
        if samples.shape[1] != 1:
            raise RuntimeError(f"{row['asset_id']}: speaker excerpt is not mono")
        mono = samples[:, 0]
    else:
        locator = json.loads(str(row["locator_json"]))
        blob, _ = load_parquet_source(locator)
        if hashlib.sha256(blob).hexdigest() != str(row["source_audio_sha256"]):
            raise RuntimeError(f"{row['asset_id']}: speaker source hash changed")
        samples, sample_rate = sf.read(
            io.BytesIO(blob), dtype="float32", always_2d=True
        )
        if samples.shape[1] != 1:
            raise RuntimeError(f"{row['asset_id']}: speaker excerpt is not mono")
        mono = samples[:, 0]
    try:
        return speaker_excerpt(mono, int(sample_rate))
    except RuntimeError as error:
        raise RuntimeError(f"{row['asset_id']}: {error}") from error


def prepare(donors: Path, existing: Path, output_root: Path) -> dict[str, Any]:
    rows = pq.read_table(donors).to_pylist()
    known = known_speakers(existing)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["speaker_key"]) not in known:
            grouped[str(row["speaker_key"])].append(row)
    summary_path = output_root / "qwen_inputs/summary.json"
    manifest_path = output_root / "qwen_inputs/speaker_profile_inputs.jsonl"
    if summary_path.is_file() and manifest_path.is_file():
        frozen = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            frozen.get("state") == "ready_for_qwen_instruct"
            and int(frozen.get("donor_rows", -1)) == len(rows)
            and int(frozen.get("qwen_profile_groups", -1)) == len(grouped)
            and float(frozen.get("speaker_excerpt_max_sec", -1.0))
            == MAX_SPEAKER_EXCERPT_SEC
            and frozen.get("training_donor_audio_mutated") is False
            and frozen.get("exact_transcript_exposed_to_qwen") is False
            and frozen.get("input_jsonl_sha256") == sha256_file(manifest_path)
        ):
            manifest_rows = [
                json.loads(line)
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if (
                {str(row["id"]) for row in manifest_rows} == set(grouped)
                and len(manifest_rows) == len(grouped)
                and all(Path(str(row["audio_path"])).is_file() for row in manifest_rows)
            ):
                return frozen
    audio_root = output_root / "qwen_inputs/audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    silence = np.zeros(round(0.35 * TARGET_SAMPLE_RATE), dtype=np.float32)
    manifest: list[dict[str, Any]] = []
    for speaker in sorted(grouped):
        candidates = sorted(
            grouped[speaker],
            key=lambda row: (
                not 2.5 <= float(row["duration_sec"]) <= 8.0,
                abs(float(row["duration_sec"]) - 4.5),
                str(row["selection_rank"]),
            ),
        )
        chosen = candidates[: min(3, len(candidates))]
        if not chosen:
            raise RuntimeError(f"{speaker}: no representative donor")
        chunks: list[np.ndarray] = []
        for index, row in enumerate(chosen):
            if index:
                chunks.append(silence)
            chunks.append(read_donor_audio(row))
        composite = np.concatenate(chunks)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "__", speaker)
        audio_path = audio_root / f"{safe}.wav"
        sf.write(audio_path, composite, TARGET_SAMPLE_RATE, subtype="PCM_16")
        manifest.append(
            {
                "id": speaker,
                "audio_path": str(audio_path),
                "source_dataset": speaker.split(":", 1)[0],
                "representative_asset_ids": [row["asset_id"] for row in chosen],
                "representative_durations_sec": [
                    row["duration_sec"] for row in chosen
                ],
            }
        )
    atomic_jsonl(manifest_path, manifest)
    summary = {
        "schema": "stable_audio_tools.speech_expansion_speaker_profile_inputs",
        "schema_version": 1,
        "state": "ready_for_qwen_instruct",
        "donor_rows": len(rows),
        "donor_speakers": len({row["speaker_key"] for row in rows}),
        "known_speakers_reused": len(
            {row["speaker_key"] for row in rows if row["speaker_key"] in known}
        ),
        "qwen_profile_groups": len(manifest),
        "input_jsonl": str(manifest_path),
        "input_jsonl_sha256": sha256_file(manifest_path),
        "audio_root": str(audio_root),
        "speaker_excerpt_max_sec": MAX_SPEAKER_EXCERPT_SEC,
        "training_donor_audio_mutated": False,
        "exact_transcript_exposed_to_qwen": False,
    }
    atomic_json(summary_path, summary)
    return summary


def clean_profile(value: str) -> str:
    text = " ".join(str(value).split()).strip(" \t\r\n.?!;:,\"'")
    if any(character.isalpha() and not character.isascii() for character in text):
        raise RuntimeError(f"speaker profile is not English: {value!r}")
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text)
    # The prompt's 15--25 words are a soft stylistic target.  Shorter direct
    # noun phrases such as "a warm, clear, low-pitched adult male voice" are
    # already complete speaker identities and must not be rejected merely for
    # having seven lexical words.
    if not 5 <= len(words) <= 40:
        raise RuntimeError(f"speaker profile word count is implausible: {value!r}")
    if not re.match(r"^(?:an?|the)\b", text, re.I):
        text = "a speaker with " + text[0].lower() + text[1:]
    return text


def read_qwen_outputs(root: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for path in sorted(root.glob("speaker_profiles*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            speaker = str(row["id"])
            if speaker in outputs:
                raise RuntimeError(f"duplicate Qwen speaker profile: {speaker}")
            if row.get("finish_reason") != "stop" or row.get("generation_capped"):
                raise RuntimeError(f"incomplete Qwen speaker profile: {speaker}")
            outputs[speaker] = clean_profile(str(row["source_description"]))
    return outputs


def finalize(donors: Path, existing: Path, output_root: Path) -> dict[str, Any]:
    donor_table = pq.read_table(donors)
    rows = donor_table.to_pylist()
    known = known_speakers(existing)
    qwen = read_qwen_outputs(output_root / "qwen_outputs")
    needed = {str(row["speaker_key"]) for row in rows} - set(known)
    if set(qwen) != needed:
        raise RuntimeError(
            f"Qwen speaker coverage mismatch: outputs={len(qwen)} needed={len(needed)}"
        )
    output_rows = []
    for row in rows:
        speaker = str(row["speaker_key"])
        value = dict(row)
        if speaker in known:
            description = known[speaker]
            provenance = "frozen_speech_speaker_instruct_v1_identity"
        else:
            description = qwen[speaker]
            provenance = "qwen3_omni_30b_a3b_instruct_three_excerpt_v1"
        value["speaker_description"] = description
        value["speaker_description_provenance"] = provenance
        output_rows.append(value)
    output = output_root / "registry/final_speech_donors_with_speakers.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    pq.write_table(
        pa.Table.from_pylist(output_rows),
        temporary,
        compression="zstd",
        row_group_size=8192,
    )
    if pq.read_metadata(temporary).num_rows != len(rows):
        raise RuntimeError("speaker-enriched donor registry reopen mismatch")
    os.replace(temporary, output)
    descriptions = Counter(row["speaker_description"] for row in output_rows)
    summary = {
        "schema": "stable_audio_tools.speech_expansion_speaker_registry_summary",
        "schema_version": 1,
        "state": "complete",
        "rows": len(output_rows),
        "unique_speakers": len({row["speaker_key"] for row in output_rows}),
        "unique_descriptions": len(descriptions),
        "generic_english_audiobook_narrator_rows": descriptions[
            "an English audiobook narrator"
        ],
        "provenance_counts": dict(
            Counter(row["speaker_description_provenance"] for row in output_rows)
        ),
        "exact_transcript_mutated": False,
        "registry": str(output),
        "registry_sha256": sha256_file(output),
    }
    atomic_json(output.parent / "summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "finalize"))
    parser.add_argument("--donors", type=Path, default=DEFAULT_DONORS)
    parser.add_argument("--existing-registry", type=Path, default=DEFAULT_EXISTING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    donors = args.donors.expanduser().resolve(strict=True)
    existing = args.existing_registry.expanduser().resolve(strict=True)
    output = args.output_root.expanduser().resolve(strict=False)
    if not str(output).startswith(os.environ.get("AMBIT_DATA_ROOT", "data")):
        raise ValueError("speaker annotation outputs must remain on SDB")
    result = (
        prepare(donors, existing, output)
        if args.command == "prepare"
        else finalize(donors, existing, output)
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
