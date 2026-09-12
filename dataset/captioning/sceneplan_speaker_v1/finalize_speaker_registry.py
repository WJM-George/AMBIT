#!/usr/bin/env python3
"""Finalize and audit one diverse speaker description per formal speech asset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_speaker_profile_inputs import (
    DATASET_ROOT, DEFAULT_LEDGER, DEFAULT_OUTPUT, EXCLUDED, FORMAL_POOLS,
    HIFI_GENDER, LIBRITTS_P, load_prompt_speakers, load_style, profile_key,
)


SCHEMA = pa.schema([
    ("schema", pa.string()), ("schema_version", pa.int16()),
    ("asset_id", pa.string()), ("source_audio_sha256", pa.string()),
    ("source_dataset", pa.string()), ("speaker_id", pa.string()),
    ("speaker_key", pa.string()), ("split", pa.string()),
    ("speaker_profile_key", pa.string()),
    ("speaker_identity_description", pa.string()),
    ("delivery_description", pa.string()),
    ("speaker_description", pa.string()),
    ("identity_provenance", pa.string()), ("delivery_provenance", pa.string()),
    ("libritts_p_style_valid", pa.bool_()),
])

ACOUSTIC = ["thick", "thin", "tensed", "relaxed", "powerful", "weak", "bright", "dark", "soft", "hard", "clear", "muffled", "raspy", "sharp", "light"]
DELIVERY = ["fluent", "halting", "calm", "lively", "intense", "friendly", "reassuring", "sincere", "elegant", "kind", "strict", "refreshing", "cool"]
REWRITE = {"tensed": "tense", "halting": "hesitant", "adult-like": "adult"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_phrase(value: str) -> str:
    text = " ".join(str(value).split()).strip(" \t\r\n.?!;:,\"'")
    text = re.sub(r"^(?:the )?audio (?:features|contains|presents)\s+", "", text, flags=re.I)
    text = re.sub(r"^(?:the )?(?:speaker|voice) (?:is|sounds)\s+", "a speaker with ", text, flags=re.I)
    if not re.match(r"^(?:an?|the)\b", text, flags=re.I):
        text = "a speaker with " + text[0].lower() + text[1:]
    if len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", text)) < 6:
        raise RuntimeError(f"Qwen speaker profile is too short: {value!r}")
    if any(character.isalpha() and not character.isascii() for character in text):
        raise RuntimeError(f"Qwen speaker profile is not English: {value!r}")
    return text


def enforce_official_gender(text: str, profile_key: str) -> str:
    """Resolve Qwen role/character guesses in favor of corpus metadata."""
    gender = profile_key.rsplit(":", 1)[-1]
    if gender == "F":
        replacements = (
            (r"\bman's\b", "woman's"), (r"\bmen's\b", "women's"),
            (r"\bmale\b", "female"), (r"\bman\b", "woman"),
        )
    elif gender == "M":
        replacements = (
            (r"\bwoman's\b", "man's"), (r"\bwomen's\b", "men's"),
            (r"\bfemale\b", "male"), (r"\bwoman\b", "man"),
        )
    else:
        return text
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def read_qwen_outputs(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.glob("speaker_profiles.shard*-of-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["id"])
            if key in result:
                raise RuntimeError(f"duplicate Qwen profile: {key}")
            result[key] = enforce_official_gender(
                clean_phrase(str(row["source_description"])), key
            )
    return result


def load_human_prompts(root: Path) -> dict[str, list[list[str]]]:
    result: dict[str, list[list[str]]] = defaultdict(list)
    for name in ("df1_en.csv", "df2_en.csv", "df3_en.csv"):
        with (root / "data" / name).open(encoding="utf-8") as source:
            for line in source:
                speaker, labels = line.rstrip("\n").split("|", 1)
                result[speaker].append([value.strip() for value in labels.split(",") if value.strip()])
    return result


def base_label(value: str) -> str:
    return re.sub(r"^(?:very|slightly)\s+", "", value.strip())


def human_identity(speaker: str, annotations: list[list[str]], gender: str | None) -> str:
    counts = Counter(base_label(value) for values in annotations for value in values)
    forms: dict[str, Counter[str]] = defaultdict(Counter)
    for values in annotations:
        for value in values:
            forms[base_label(value)][value] += 1
    if gender == "F" or counts["feminine"] > counts["masculine"]:
        gender_text = "female"
    elif gender == "M" or counts["masculine"] > counts["feminine"]:
        gender_text = "male"
    else:
        gender_text = "gender-neutral"
    if counts["young"] > max(counts["middle-aged"], counts["mature"]):
        age = "young adult"
    elif counts["middle-aged"] or counts["mature"]:
        age = "mature adult"
    else:
        age = "adult"
    subject = f"a {age} {gender_text} narrator"
    if subject.startswith("a adult"):
        subject = "an" + subject[1:]

    def preferred(label: str) -> str:
        value = sorted(
            forms[label].items(), key=lambda item: (-item[1], item[0])
        )[0][0]
        modifier = ""
        base = value
        match = re.match(r"^(very|slightly)\s+(.+)$", value)
        if match:
            modifier, base = match.groups()
            modifier += " "
        return modifier + REWRITE.get(base, base)

    acoustic_groups = (
        ("thick", "thin"), ("tensed", "relaxed"), ("powerful", "weak"),
        ("bright", "dark"), ("soft", "hard"), ("clear", "muffled"),
        ("raspy",), ("sharp", "light"),
    )
    delivery_groups = (
        ("fluent", "halting"), ("calm", "lively", "intense"),
        ("friendly", "strict"), ("reassuring",), ("sincere",),
        ("elegant",), ("kind",), ("refreshing",), ("cool",),
    )

    def group_winners(groups: tuple[tuple[str, ...], ...], limit: int) -> list[str]:
        chosen: list[str] = []
        for group in groups:
            candidates = [label for label in group if counts[label]]
            if not candidates:
                continue
            winner = sorted(candidates, key=lambda label: (-counts[label], label))[0]
            chosen.append(preferred(winner))
            if len(chosen) == limit:
                break
        return chosen

    acoustic = group_winners(acoustic_groups, 5)
    expressive = group_winners(delivery_groups, 3)
    if not acoustic:
        acoustic = ["clear", "natural"]
    phrase = f"{subject} with a {', '.join(acoustic[:-1]) + (' and ' if len(acoustic) > 1 else '') + acoustic[-1]} vocal quality"
    if expressive:
        phrase += f" and a {', '.join(expressive[:-1]) + (' and ' if len(expressive) > 1 else '') + expressive[-1]} delivery"
    return phrase


def delivery(style_row: dict[str, str] | None) -> tuple[str, bool]:
    if not style_row or str(style_row.get("invalid", "1")) != "0":
        return "", False
    pitch = str(style_row["pitch"]).replace("very ", "very ")
    speed = str(style_row["speaking_speed"])
    energy = str(style_row["energy"])
    pace = {"very slow": "very slowly", "slow": "slowly", "normal": "at a measured pace", "fast": "quickly", "very fast": "very quickly"}.get(speed, f"at a {speed} pace")
    energy_text = {"low": "soft energy", "normal": "balanced energy", "high": "strong energy"}.get(energy, f"{energy} energy")
    return f"speaking in a {pitch}-pitched register {pace} with {energy_text}", True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--libritts-p", type=Path, default=LIBRITTS_P)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    ledger = args.ledger.expanduser().resolve(strict=True)
    libri = args.libritts_p.expanduser().resolve(strict=True)
    root = args.annotation_root.expanduser().resolve(strict=True)
    qwen = read_qwen_outputs(root / "qwen_outputs")
    prompt_speakers = load_prompt_speakers(libri)
    human = load_human_prompts(libri)
    style = load_style(libri)
    rows = [row for row in pq.read_table(ledger).to_pylist() if str(row["pool"]) in FORMAL_POOLS]
    output_rows: list[dict[str, Any]] = []
    qwen_used: set[str] = set()
    descriptions: Counter[str] = Counter()
    for row in rows:
        source_dataset = str(row["source_dataset"])
        speaker = str(row["speaker_id"])
        style_row = style.get(str(row["source_id"])) if source_dataset == "libritts" else None
        fallback_key = profile_key(row, prompt_speakers, style)
        if fallback_key is not None:
            if fallback_key not in qwen:
                raise RuntimeError(f"missing Qwen fallback profile: {fallback_key}")
            identity = qwen[fallback_key]
            provenance = "qwen3_omni_30b_a3b_instruct_three_excerpt_v1"
            qwen_used.add(fallback_key)
        else:
            gender = style_row.get("gender") if style_row else None
            identity = human_identity(speaker, human[speaker], gender)
            provenance = "libritts_p_three_human_annotations_e79ba689"
        delivery_text, style_valid = delivery(style_row)
        combined = identity + (", " + delivery_text if delivery_text else "")
        descriptions[combined] += 1
        output_rows.append({
            "schema": "stable_audio_tools.sceneplan_speech_speaker_registry_entry",
            "schema_version": 1,
            "asset_id": str(row["asset_id"]),
            "source_audio_sha256": str(row["source_audio_sha256"]),
            "source_dataset": source_dataset,
            "speaker_id": speaker,
            "speaker_key": str(row["speaker_key"]),
            "split": str(row["pool"]),
            "speaker_profile_key": fallback_key or f"libritts:{speaker}:official",
            "speaker_identity_description": identity,
            "delivery_description": delivery_text,
            "speaker_description": combined,
            "identity_provenance": provenance,
            "delivery_provenance": "libritts_p_utterance_style_v230922" if style_valid else "not_available",
            "libritts_p_style_valid": style_valid,
        })
    if len(output_rows) != 512_000 or len({row["asset_id"] for row in output_rows}) != len(output_rows):
        raise RuntimeError("formal speech registry is not exactly 512,000 unique assets")
    expected_qwen = set(qwen)
    if qwen_used != expected_qwen:
        raise RuntimeError(f"unused/missing Qwen profiles: used={len(qwen_used)} output={len(expected_qwen)}")
    registry_dir = root / "registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    output = registry_dir / "speech_speaker_description_registry.parquet"
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(output_rows, schema=SCHEMA), temporary, compression="zstd", row_group_size=8192)
    if pq.read_metadata(temporary).num_rows != 512_000:
        raise RuntimeError("registry reopen count mismatch")
    os.replace(temporary, output)
    word_counts = [len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", row["speaker_description"])) for row in output_rows]
    audit = {
        "schema": "stable_audio_tools.sceneplan_speech_speaker_registry_audit",
        "schema_version": 1,
        "ok": True,
        "rows": len(output_rows),
        "unique_assets": len({row["asset_id"] for row in output_rows}),
        "unique_audio_hashes": len({row["source_audio_sha256"] for row in output_rows}),
        "unique_speakers": len({row["speaker_key"] for row in output_rows}),
        "unique_identity_descriptions": len({row["speaker_identity_description"] for row in output_rows}),
        "unique_compiled_descriptions": len(descriptions),
        "constant_generic_description_rows": descriptions["an English audiobook narrator"],
        "qwen_profile_groups": len(qwen_used),
        "identity_provenance_counts": dict(Counter(row["identity_provenance"] for row in output_rows)),
        "delivery_provenance_counts": dict(Counter(row["delivery_provenance"] for row in output_rows)),
        "split_counts": dict(Counter(row["split"] for row in output_rows)),
        "source_dataset_counts": dict(Counter(row["source_dataset"] for row in output_rows)),
        "description_words": {"min": min(word_counts), "p50": float(np.percentile(word_counts, 50)), "p99": float(np.percentile(word_counts, 99)), "max": max(word_counts)},
        "registry": str(output),
        "registry_sha256": sha256_file(output),
        "exact_transcript_mutated": False,
        "foa_or_latent_mutated": False,
    }
    audit_path = registry_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
