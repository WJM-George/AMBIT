#!/usr/bin/env python3
"""Audit frozen-ASR reliability without exposing target text to P11 inputs."""

from __future__ import annotations
import os

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys
import zlib
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.sceneplan_p11_lexical_cache import (  # noqa: E402
    P11_LEXICAL_CACHE_SCHEMA,
    P11_LEXICAL_CACHE_VERSION,
    P11_LEXICAL_CONFIDENCE_CONTRACT,
    P11_LEXICAL_EVIDENCE_CONTRACT,
)


ROOT = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2")
DEFAULT_CACHE = ROOT / (
    "lexical_cache/p11_train_trial10k_owner_balanced_distilwhisper_v1.sqlite"
)
DEFAULT_MANIFEST = ROOT / "manifests/p11_train_trial30k_owner_balanced_v1.sqlite"
DEFAULT_INDEX = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1/training_index/train.sqlite"
)
DEFAULT_OUTPUT = REPO_ROOT / (
    "artifacts/sceneplan_p11/p11_train10k_reliable_asr_audit_20260902.json"
)
_WORD = re.compile(r"[a-z0-9']+")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _words(text: str) -> list[str]:
    return _WORD.findall(str(text).lower())


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_word in enumerate(left, 1):
        current = [left_index]
        for right_index, right_word in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_word != right_word),
                )
            )
        previous = current
    return previous[-1]


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _classification(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        predicted = bool(row["has_speech"]) and float(row["confidence"]) >= threshold
        target = bool(row["target_has_speech"])
        counts[
            "tp" if predicted and target else
            "fp" if predicted else
            "fn" if target else
            "tn"
        ] += 1
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
        "false_positive_rate": _safe_ratio(fp, fp + tn),
        "reliable_rows": tp + fp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must lie within [0,1]")

    cache_path = args.cache.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    index_path = args.index.expanduser().resolve(strict=True)
    cache = _readonly(cache_path)
    manifest = _readonly(manifest_path)
    index = _readonly(index_path)
    metadata = dict(cache.execute("SELECT key,value FROM metadata"))
    required = {
        "schema": P11_LEXICAL_CACHE_SCHEMA,
        "schema_version": str(P11_LEXICAL_CACHE_VERSION),
        "contract": P11_LEXICAL_EVIDENCE_CONTRACT,
        "confidence_contract": P11_LEXICAL_CONFIDENCE_CONTRACT,
        "source_manifest": str(manifest_path),
        "source_index": str(index_path),
        "source": "input_foa_only",
        "target_transcript_access": "forbidden",
        "rows": "10000",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"lexical cache {key}={metadata.get(key)!r}, expected {expected!r}"
            )

    ordinals = [
        int(row[0])
        for row in manifest.execute(
            "SELECT DISTINCT target_ordinal FROM rows ORDER BY target_ordinal"
        )
    ]
    if len(ordinals) != 10_000:
        raise RuntimeError("lexical audit requires the exact 10k screening scenes")
    hypotheses = {
        int(row[0]): {
            "text": str(row[1]),
            "has_speech": bool(int(row[2])),
            "confidence": float(row[3]),
        }
        for row in cache.execute(
            "SELECT ordinal,text,has_speech,confidence FROM hypotheses ORDER BY ordinal"
        )
    }
    if sorted(hypotheses) != ordinals:
        raise RuntimeError("lexical cache coverage differs from screening manifest")

    rows: list[dict[str, Any]] = []
    transcript_reference_words = 0
    transcript_errors = 0
    for offset in range(0, len(ordinals), 900):
        chunk = ordinals[offset : offset + 900]
        placeholders = ",".join("?" for _ in chunk)
        plans = {
            int(ordinal): json.loads(zlib.decompress(payload))
            for ordinal, payload in index.execute(
                f"SELECT ordinal,scene_plan_zlib FROM samples "
                f"WHERE ordinal IN ({placeholders})",
                tuple(chunk),
            )
        }
        if len(plans) != len(chunk):
            raise RuntimeError("lexical audit could not resolve source ScenePlans")
        for ordinal in chunk:
            plan = plans[ordinal]
            speech = [
                source for source in plan["sources"] if source["kind"] == "speech"
            ]
            if len(speech) > 1:
                raise RuntimeError("P10 profile unexpectedly has multiple speech sources")
            hypothesis = hypotheses[ordinal]
            row = {
                **hypothesis,
                "target_has_speech": bool(speech),
            }
            rows.append(row)
            if (
                speech
                and hypothesis["has_speech"]
                and hypothesis["confidence"] >= args.threshold
            ):
                reference = _words(speech[0]["transcript"])
                predicted = _words(hypothesis["text"])
                transcript_reference_words += len(reference)
                transcript_errors += _edit_distance(reference, predicted)

    if any(
        not math.isfinite(row["confidence"])
        or not 0.0 <= row["confidence"] <= 1.0
        for row in rows
    ):
        raise RuntimeError("lexical cache contains invalid confidence")
    frozen = _classification(rows, args.threshold)
    threshold_grid = [round(value / 100.0, 2) for value in range(70, 97, 2)]
    grid = [_classification(rows, value) for value in threshold_grid]
    report = {
        "schema": "stable_audio_tools.p11_lexical_cache_audit",
        "schema_version": 1,
        "status": "PASS",
        "scope": (
            "offline train-side reliability audit only; target labels and transcript "
            "are never written into or returned by the input cache"
        ),
        "cache": str(cache_path),
        "cache_sha256": _sha256_file(cache_path),
        "rows": len(rows),
        "target_speech_rows": sum(row["target_has_speech"] for row in rows),
        "raw_asr_speech_rows": sum(row["has_speech"] for row in rows),
        "frozen_threshold": frozen,
        "threshold_grid_diagnostic_only": grid,
        "reliable_true_speech_word_error_rate": _safe_ratio(
            transcript_errors, transcript_reference_words
        ),
        "reliable_true_speech_reference_words": transcript_reference_words,
        "input_boundary": {
            "source": metadata["source"],
            "target_transcript_access": metadata["target_transcript_access"],
            "target_side_fields_written_to_cache": False,
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

    cache.close()
    manifest.close()
    index.close()


if __name__ == "__main__":
    main()
