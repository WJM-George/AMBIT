#!/usr/bin/env python3
"""Audit and calibrate a frozen-ASR cache on a train-only P11 manifest.

This script may inspect target speech presence/transcripts for calibration, but
that information is never written back to the input-evidence cache and is
forbidden at planner runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import statistics
import zlib
from pathlib import Path
from typing import Any


def _words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(value).lower())


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


def _targets(manifest: Path) -> dict[int, dict[str, Any]]:
    connection = sqlite3.connect(
        f"file:{manifest}?mode=ro&immutable=1", uri=True
    )
    result: dict[int, dict[str, Any]] = {}
    try:
        # The cache transcribes the indexed input FOA.  Editing rows reuse that
        # ordinal but intentionally carry an edited target ScenePlan, so they
        # are not a valid speech-presence/transcript reference for the audio.
        # Every frozen manifest has one Understanding row per target ordinal;
        # use that unedited inverse-ScenePlan target exclusively.
        for ordinal, payload in connection.execute(
            "SELECT target_ordinal,target_sceneplan_zlib FROM rows "
            "WHERE task='understanding' ORDER BY ordinal"
        ):
            plan = json.loads(zlib.decompress(payload))
            prior = result.setdefault(int(ordinal), plan)
            if prior != plan:
                raise RuntimeError(
                    f"target ordinal {ordinal} maps to multiple ScenePlans"
                )
    finally:
        connection.close()
    return result


def _rates(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = bool(row["has_speech"]) and float(row["confidence"]) >= threshold
        target = bool(row["target_has_speech"])
        if predicted and target:
            tp += 1
        elif predicted:
            fp += 1
        elif target:
            fn += 1
        else:
            tn += 1
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2.0 * precision * recall / max(precision + recall, 1.0e-12),
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--default-threshold", type=float, default=0.65)
    args = parser.parse_args()
    if not 0.0 <= args.default_threshold <= 1.0:
        raise ValueError("--default-threshold must be within [0,1]")
    manifest = args.manifest.expanduser().resolve(strict=True)
    cache = args.cache.expanduser().resolve(strict=True)
    targets = _targets(manifest)

    connection = sqlite3.connect(f"file:{cache}?mode=ro&immutable=1", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        hypotheses = {
            int(row[0]): {
                "text": str(row[1]),
                "has_speech": bool(int(row[2])),
                "confidence": float(row[3]),
            }
            for row in connection.execute(
                "SELECT ordinal,text,has_speech,confidence "
                "FROM hypotheses ORDER BY ordinal"
            )
        }
    finally:
        connection.close()
    if set(hypotheses) != set(targets):
        raise RuntimeError("ASR cache and manifest target ordinals do not align")

    rows = []
    word_edits = reference_words = 0
    for ordinal in sorted(targets):
        speech_sources = [
            source
            for source in targets[ordinal]["sources"]
            if source["kind"] == "speech"
        ]
        reference = " ".join(
            str(source.get("transcript", "")).strip()
            for source in speech_sources
            if str(source.get("transcript", "")).strip()
        )
        hypothesis = hypotheses[ordinal]
        reference_tokens = _words(reference)
        hypothesis_tokens = _words(hypothesis["text"])
        if speech_sources:
            word_edits += _edit_distance(reference_tokens, hypothesis_tokens)
            reference_words += len(reference_tokens)
        rows.append(
            {
                **hypothesis,
                "target_has_speech": bool(speech_sources),
            }
        )
    if any(not math.isfinite(row["confidence"]) for row in rows):
        raise RuntimeError("ASR cache confidence is non-finite")

    thresholds = sorted(
        {
            0.0,
            1.0,
            float(args.default_threshold),
            *(round(index / 100.0, 2) for index in range(5, 100, 5)),
        }
    )
    sweep = [_rates(rows, threshold) for threshold in thresholds]
    selected = max(
        sweep,
        key=lambda value: (
            value["balanced_accuracy"],
            value["f1"],
            value["threshold"],
        ),
    )
    default = next(
        value
        for value in sweep
        if value["threshold"] == float(args.default_threshold)
    )
    report = {
        "schema": "stable_audio_tools.p11_lexical_cache_train_calibration",
        "schema_version": 1,
        "status": "PASS",
        "scope": (
            "train-only target-aware calibration; target fields are forbidden "
            "from runtime lexical evidence"
        ),
        "manifest": str(manifest),
        "cache": str(cache),
        "cache_contract": metadata.get("contract"),
        "confidence_contract": metadata.get("confidence_contract"),
        "rows": len(rows),
        "target_speech_present": sum(row["target_has_speech"] for row in rows),
        "target_speech_absent": sum(not row["target_has_speech"] for row in rows),
        "asr_hypothesis_present": sum(row["has_speech"] for row in rows),
        "confidence": {
            "target_speech_present": _summary(
                [row["confidence"] for row in rows if row["target_has_speech"]]
            ),
            "target_speech_absent": _summary(
                [row["confidence"] for row in rows if not row["target_has_speech"]]
            ),
        },
        "speech_aggregate_wer": word_edits / max(reference_words, 1),
        "speech_reference_words": reference_words,
        "default_gate": default,
        "selected_train_only_gate": selected,
        "threshold_sweep": sweep,
        "runtime_policy": (
            "CLAP always; add ASR tokens only when has_speech=true and "
            "confidence>=frozen threshold"
        ),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
