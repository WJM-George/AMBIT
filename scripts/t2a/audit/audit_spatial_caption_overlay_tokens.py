#!/usr/bin/env python3
"""Audit every caption in a finalized Spatial-CoT overlay with Qwen tokenizer."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.spatial_caption_templates import (  # noqa: E402
    SEMANTIC_CAPTION_TEMPLATE_COUNT,
    SEMANTIC_CAPTION_TEMPLATE_VERSION,
)


_TOKENIZER = None


def _init_worker(tokenizer_path: str) -> None:
    global _TOKENIZER
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path)


def _caption_batches(
    paths: Iterable[Path], batch_size: int
) -> Iterable[tuple[list[str], list[dict[str, Any]]]]:
    texts: list[str] = []
    identities: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                family = json.loads(line)
                if family.get("template_version") != SEMANTIC_CAPTION_TEMPLATE_VERSION:
                    raise ValueError(
                        f"caption template version mismatch at {path}:{line_number}"
                    )
                turns = family.get("turns")
                if not isinstance(turns, list) or not turns:
                    raise ValueError(f"caption family has no turns at {path}:{line_number}")
                for turn_index, turn in enumerate(turns):
                    text = turn.get("semantic_caption")
                    if not isinstance(text, str) or not text:
                        raise ValueError(
                            f"empty caption at {path}:{line_number} turn {turn_index}"
                        )
                    texts.append(text)
                    identities.append(
                        {
                            "family_id": family.get("family_id"),
                            "turn_id": turn.get("turn_id"),
                            "characters": len(text),
                        }
                    )
                    if len(texts) >= batch_size:
                        yield texts, identities
                        texts, identities = [], []
    if texts:
        yield texts, identities


def _audit_partition(arguments: tuple[list[str], int, int]) -> dict[str, Any]:
    paths_raw, batch_size, max_length = arguments
    if _TOKENIZER is None:
        raise RuntimeError("tokenizer worker was not initialized")
    histogram: Counter[int] = Counter()
    families = 0
    turns = 0
    overflow: list[dict[str, Any]] = []
    maximum: dict[str, Any] = {"tokens": 0}
    paths = [Path(value) for value in paths_raw]
    for path in paths:
        with path.open("rb") as handle:
            families += sum(1 for line in handle if line.strip())
    for texts, identities in _caption_batches(paths, batch_size):
        encoded = _TOKENIZER(
            texts,
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_length=True,
        )
        lengths = [int(value) for value in encoded["length"]]
        if len(lengths) != len(texts):
            raise RuntimeError("tokenizer length batch mismatch")
        turns += len(lengths)
        histogram.update(lengths)
        for text, identity, length in zip(texts, identities, lengths):
            if length > int(maximum["tokens"]):
                maximum = {
                    **identity,
                    "tokens": length,
                    "caption": text,
                }
            if length > max_length and len(overflow) < 32:
                overflow.append(
                    {**identity, "tokens": length, "caption": text}
                )
    return {
        "families": families,
        "turns": turns,
        "histogram": dict(histogram),
        "maximum": maximum,
        "overflow": overflow,
    }


def _percentile(histogram: Counter[int], fraction: float) -> int:
    total = sum(histogram.values())
    if not total:
        return 0
    target = max(1, int((total - 1) * fraction) + 1)
    cumulative = 0
    for length in sorted(histogram):
        cumulative += histogram[length]
        if cumulative >= target:
            return int(length)
    return int(max(histogram))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.overlay_root.expanduser().resolve()
    ready = json.loads((root / "READY").read_text(encoding="utf-8"))
    paths = sorted((root / "shards").glob("captions-*.jsonl"))
    if not paths:
        raise SystemExit(f"caption overlay has no shards: {root}")
    workers = max(1, min(int(args.workers), len(paths)))
    partitions = [paths[index::workers] for index in range(workers)]
    work = [
        ([str(path) for path in partition], int(args.batch_size), int(args.max_length))
        for partition in partitions
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(args.tokenizer.expanduser().resolve()),),
    ) as executor:
        rows = list(executor.map(_audit_partition, work))

    histogram: Counter[int] = Counter()
    overflow: list[dict[str, Any]] = []
    maximum: dict[str, Any] = {"tokens": 0}
    for row in rows:
        histogram.update({int(key): int(value) for key, value in row["histogram"].items()})
        overflow.extend(row["overflow"])
        if int(row["maximum"]["tokens"]) > int(maximum["tokens"]):
            maximum = row["maximum"]
    families = sum(int(row["families"]) for row in rows)
    turns = sum(int(row["turns"]) for row in rows)
    overflow_count = sum(
        count for length, count in histogram.items() if length > args.max_length
    )
    failures = []
    if families != int(ready.get("families", -1)):
        failures.append("family_count")
    if turns != int(ready.get("turns", -1)):
        failures.append("turn_count")
    if ready.get("template_version") != SEMANTIC_CAPTION_TEMPLATE_VERSION:
        failures.append("template_version")
    if int(ready.get("template_count", -1)) != SEMANTIC_CAPTION_TEMPLATE_COUNT:
        failures.append("template_count")
    if overflow_count:
        failures.append("token_overflow")
    report = {
        "status": "PASS" if not failures else "FAIL",
        "schema": "stable_audio_tools.spatial_caption_token_audit",
        "schema_version": 1,
        "overlay_root": str(root),
        "tokenizer": str(args.tokenizer.expanduser().resolve()),
        "template_version": SEMANTIC_CAPTION_TEMPLATE_VERSION,
        "template_count": SEMANTIC_CAPTION_TEMPLATE_COUNT,
        "families": families,
        "turns": turns,
        "max_length": int(args.max_length),
        "token_lengths": {
            "mean": (
                sum(length * count for length, count in histogram.items()) / turns
                if turns
                else 0.0
            ),
            "p50": _percentile(histogram, 0.50),
            "p90": _percentile(histogram, 0.90),
            "p99": _percentile(histogram, 0.99),
            "max": max(histogram, default=0),
            "overflow": overflow_count,
        },
        "maximum_example": maximum,
        "overflow_examples": overflow[:100],
        "failures": failures,
    }
    _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
