#!/usr/bin/env python3
"""Merge and minimally audit the 50-music/50-sound Instruct pilot."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from description_contract import (
    PROMPT_TEMPLATE_VERSION,
    TARGET_MAX_WORDS,
    TARGET_MIN_WORDS,
    compact_whitespace,
    prompt_contract_sha256,
    validate_source_description,
)


ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/audit/a2t_pilot_100")
PROMPT = Path(__file__).with_name("source_description_prompt_v4.txt")
MODEL = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen3-Omni-30B-A3B-Instruct")
MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"non-object JSON at {path}:{line_number}")
            rows.append(row)
    return rows


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def representative_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for kind in ("music", "sound"):
        for group in ("generic_legacy_label", "descriptive_legacy_label"):
            candidates = [
                row
                for row in rows
                if row["kind"] == kind and row["pilot_group"] == group
            ]
            samples.extend(candidates[:5])
    return samples


def main() -> int:
    root = ROOT.resolve(strict=True)
    prompt = PROMPT.resolve(strict=True)
    model = MODEL.resolve(strict=True)
    selection_path = root / "selection.jsonl"
    selection = load_jsonl(selection_path)
    selection_by_id = {str(row["annotation_id"]): row for row in selection}
    if len(selection_by_id) != 100:
        raise RuntimeError("selection must contain 100 unique rows")
    if Counter(row["kind"] for row in selection) != Counter(
        {"music": 50, "sound": 50}
    ):
        raise RuntimeError("selection must contain 50 music and 50 sound")

    shard_paths = sorted(root.glob("source_descriptions_instruct.shard*-of-*.jsonl"))
    if len(shard_paths) != 4:
        raise RuntimeError(f"expected four shards, found {len(shard_paths)}")
    raw_rows = [row for path in shard_paths for row in load_jsonl(path)]
    raw_by_id = {str(row["id"]): row for row in raw_rows}
    if len(raw_rows) != 100 or len(raw_by_id) != 100:
        raise RuntimeError("outputs must contain 100 unique rows")
    if set(raw_by_id) != set(selection_by_id):
        raise RuntimeError("selection and output ids differ")

    prompt_file_sha256 = file_sha256(prompt)
    prompt_sha256 = prompt_contract_sha256(prompt.read_text(encoding="utf-8"))
    merged: list[dict[str, Any]] = []
    flags: list[dict[str, Any]] = []
    word_counts: list[int] = []
    token_counts: list[int] = []
    hard_failure_rows = 0

    for selected in selection:
        annotation_id = str(selected["annotation_id"])
        raw = raw_by_id[annotation_id]
        audio_hash = str(selected["source_audio_sha256"])
        audio_path = Path(str(selected["audio_path"])).resolve(strict=True)
        if annotation_id != f"sha256:{audio_hash}":
            raise RuntimeError(f"annotation/hash mismatch: {annotation_id}")
        if str(raw.get("source_audio_sha256")) != audio_hash:
            raise RuntimeError(f"output lineage mismatch: {annotation_id}")
        if file_sha256(audio_path) != audio_hash:
            raise RuntimeError(f"audio checksum mismatch: {annotation_id}")
        if str(raw.get("kind")) != str(selected["kind"]):
            raise RuntimeError(f"kind mismatch: {annotation_id}")
        if Path(str(raw["model_path"])).resolve() != model:
            raise RuntimeError(f"model path mismatch: {annotation_id}")
        if str(raw.get("model_revision")) != MODEL_REVISION:
            raise RuntimeError(f"model revision mismatch: {annotation_id}")
        if str(raw.get("engine")) != "transformers":
            raise RuntimeError(f"unapproved engine: {annotation_id}")
        if str(raw.get("prompt_file_sha256")) != prompt_file_sha256:
            raise RuntimeError(f"prompt file mismatch: {annotation_id}")
        if str(raw.get("prompt_sha256")) != prompt_sha256:
            raise RuntimeError(f"prompt contract mismatch: {annotation_id}")
        if str(raw.get("prompt_template_version")) != PROMPT_TEMPLATE_VERSION:
            raise RuntimeError(f"prompt template mismatch: {annotation_id}")
        if str(raw.get("decoder_constraint")) != "none":
            raise RuntimeError(f"unexpected decoder constraint: {annotation_id}")

        description = compact_whitespace(str(raw.get("source_description") or ""))
        decoder_text = compact_whitespace(str(raw.get("decoder_text") or ""))
        if description != decoder_text:
            raise RuntimeError(f"description was rewritten: {annotation_id}")
        if hashlib.sha256(description.encode()).hexdigest() != str(
            raw.get("source_description_sha256")
        ):
            raise RuntimeError(f"description checksum mismatch: {annotation_id}")

        qc = validate_source_description(description)
        hard_flags = list(qc.hard_flags)
        if str(raw.get("finish_reason")) == "length" or bool(
            raw.get("generation_capped")
        ):
            hard_flags.append("generation_truncated")
        hard_flags = list(dict.fromkeys(hard_flags))
        hard_failure_rows += int(bool(hard_flags))
        if hard_flags or qc.soft_flags:
            flags.append(
                {
                    "annotation_id": annotation_id,
                    "kind": selected["kind"],
                    "hard_flags": hard_flags,
                    "soft_flags": list(qc.soft_flags),
                    "source_description": description,
                    "audio_path": str(audio_path),
                }
            )

        merged.append(
            {
                "schema": "stable_audio_tools.sceneplan_source_description",
                "schema_version": 4,
                "annotation_id": annotation_id,
                "source_audio_sha256": audio_hash,
                "asset_id": str(selected["asset_id"]),
                "source_dataset": str(selected["source_dataset"]),
                "kind": str(selected["kind"]),
                "pilot_group": str(selected["pilot_group"]),
                "raw_label_for_audit_only": str(selected["raw_label"]),
                "audio_path": str(audio_path),
                "source_description": description,
                "description_word_count": qc.word_count,
                "hard_qc_flags": hard_flags,
                "soft_qc_flags": list(qc.soft_flags),
                "provenance": {
                    "model_revision": MODEL_REVISION,
                    "engine": "transformers",
                    "engine_version": str(raw["engine_version"]),
                    "prompt_sha256": prompt_sha256,
                    "generated_tokens_observability_only": int(
                        raw["generated_tokens"]
                    ),
                    "finish_reason": str(raw["finish_reason"]),
                    "decoder_text_sha256": str(raw["decoder_text_sha256"]),
                },
            }
        )
        word_counts.append(qc.word_count)
        token_counts.append(int(raw["generated_tokens"]))

    merged_path = root / "source_descriptions_instruct_100.jsonl"
    flags_path = root / "source_description_instruct_qc_flags.jsonl"
    samples_path = root / "source_description_instruct_samples_20.jsonl"
    samples = representative_samples(merged)
    atomic_text(merged_path, "".join(stable_json(row) + "\n" for row in merged))
    atomic_text(flags_path, "".join(stable_json(row) + "\n" for row in flags))
    atomic_text(samples_path, "".join(stable_json(row) + "\n" for row in samples))

    automatic_ok = hard_failure_rows == 0
    summary = {
        "schema": "stable_audio_tools.sceneplan_source_description_pilot_qc",
        "schema_version": 4,
        "rows": 100,
        "kind_counts": {"music": 50, "sound": 50},
        "contract": "english_semantic_audio_description",
        "target_words_soft_only": [TARGET_MIN_WORDS, TARGET_MAX_WORDS],
        "hard_gates": ["nonempty", "predominantly_english", "not_truncated"],
        "model_revision": MODEL_REVISION,
        "engine": "transformers",
        "prompt_path": str(prompt),
        "prompt_sha256": prompt_sha256,
        "word_count": {
            "min": min(word_counts),
            "p50": percentile(word_counts, 0.50),
            "p90": percentile(word_counts, 0.90),
            "p99": percentile(word_counts, 0.99),
            "max": max(word_counts),
            "within_soft_target": sum(
                TARGET_MIN_WORDS <= value <= TARGET_MAX_WORDS
                for value in word_counts
            ),
        },
        "generated_tokens_observability_only": {
            "min": min(token_counts),
            "p50": percentile(token_counts, 0.50),
            "p90": percentile(token_counts, 0.90),
            "p99": percentile(token_counts, 0.99),
            "max": max(token_counts),
        },
        "hard_failure_rows": hard_failure_rows,
        "automatic_hard_gates_ok": automatic_ok,
        "pilot_acceptance_status": (
            "awaiting_user_sample_review" if automatic_ok else "automatic_qc_failed"
        ),
        "ready_for_revised_sceneplans": False,
        "selection_sha256": file_sha256(selection_path),
        "raw_shards": [str(path) for path in shard_paths],
        "merged_jsonl": str(merged_path),
        "merged_sha256": file_sha256(merged_path),
        "samples_jsonl": str(samples_path),
        "samples_sha256": file_sha256(samples_path),
        "full_scale_started": False,
        "p8_started": False,
        "p9_started": False,
    }
    summary_path = root / "source_description_instruct_qc_summary.json"
    atomic_json(summary_path, summary)
    marker = root / (
        "SOURCE_DESCRIPTION_INSTRUCT_PILOT_AWAITING_USER_REVIEW"
        if automatic_ok
        else "SOURCE_DESCRIPTION_INSTRUCT_PILOT_NEEDS_REVIEW"
    )
    atomic_text(marker, stable_json(summary) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if automatic_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
