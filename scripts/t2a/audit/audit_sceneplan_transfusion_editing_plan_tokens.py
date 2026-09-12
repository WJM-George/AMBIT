#!/usr/bin/env python3
"""Exhaustively audit codec-v4 new-ScenePlan lengths in an Editing index."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any
import zlib

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.sceneplan_transfusion_editing import (  # noqa: E402
    sha256_json,
)
from stable_audio_tools.data.sceneplan_transfusion_editing_index import (  # noqa: E402
    sha256_file,
)


DEFAULT_CODEC = Path(
    "/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probabilities = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
    observed = np.quantile(values, probabilities)
    return {
        label: float(value)
        for label, value in zip(
            ("min", "p50", "p90", "p95", "p99", "p999", "max"),
            observed,
        )
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--codec", type=Path, default=DEFAULT_CODEC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--required-max-tokens", type=int, default=1024)
    parser.add_argument("--evaluation-max-tokens", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if min(
        args.expected_rows,
        args.required_max_tokens,
        args.evaluation_max_tokens,
        args.progress_every,
    ) <= 0:
        raise ValueError("row/token/progress limits must be positive")
    index = args.index.expanduser().resolve(strict=True)
    codec_path = args.codec.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    index_sha = sha256_file(index)
    codec = ModelScenePlanCodecV4(codec_path)
    connection = sqlite3.connect(
        f"file:{index}?mode=ro&immutable=1", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    stored_rows = int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    if stored_rows != int(args.expected_rows):
        raise RuntimeError(
            f"Editing token audit rows changed: {stored_rows} != {args.expected_rows}"
        )
    if not (
        metadata.get("editing_ar_input_contract")
        == "source_foa_latent_plus_raw_edit_request_v2"
        and metadata.get("editing_ar_old_sceneplan_input") == "false"
    ):
        raise RuntimeError("Editing token audit received a stale AR route")

    lengths = np.empty(stored_rows, dtype=np.int32)
    histogram: Counter[int] = Counter()
    by_operation: dict[str, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "max_tokens": 0, "over_evaluation_budget": 0}
    )
    by_bucket: dict[int, dict[str, int]] = defaultdict(
        lambda: {"rows": 0, "max_tokens": 0, "over_evaluation_budget": 0}
    )
    longest: list[tuple[int, int, str, str, int]] = []
    row_count = 0
    started = time.perf_counter()
    cursor = connection.execute(
        "SELECT pair_ordinal,pair_id,operation,latent_bucket_frames,"
        "new_sceneplan_zlib,new_sceneplan_sha256 "
        "FROM pairs ORDER BY pair_ordinal"
    )
    for (
        pair_ordinal,
        pair_id,
        operation,
        bucket,
        packed_plan,
        expected_plan_sha,
    ) in cursor:
        ordinal = int(pair_ordinal)
        if ordinal != row_count:
            raise RuntimeError(
                f"Editing pair ordinals are not contiguous at row {row_count}: {ordinal}"
            )
        try:
            plan = json.loads(zlib.decompress(packed_plan))
        except (TypeError, ValueError, zlib.error) as error:
            raise RuntimeError(f"{pair_id}: invalid new ScenePlan blob") from error
        if sha256_json(plan) != str(expected_plan_sha):
            raise RuntimeError(f"{pair_id}: new ScenePlan SHA256 changed")
        encoded = codec.encode(plan, max_tokens=int(args.required_max_tokens))
        token_count = int(encoded["input_ids"].numel())
        lengths[row_count] = token_count
        histogram[token_count] += 1
        over_evaluation = int(token_count > int(args.evaluation_max_tokens))
        operation_stats = by_operation[str(operation)]
        operation_stats["rows"] += 1
        operation_stats["max_tokens"] = max(
            operation_stats["max_tokens"], token_count
        )
        operation_stats["over_evaluation_budget"] += over_evaluation
        bucket_stats = by_bucket[int(bucket)]
        bucket_stats["rows"] += 1
        bucket_stats["max_tokens"] = max(bucket_stats["max_tokens"], token_count)
        bucket_stats["over_evaluation_budget"] += over_evaluation
        candidate = (
            token_count,
            ordinal,
            str(pair_id),
            str(operation),
            int(bucket),
        )
        if len(longest) < 20:
            heapq.heappush(longest, candidate)
        elif candidate > longest[0]:
            heapq.heapreplace(longest, candidate)
        row_count += 1
        if row_count % int(args.progress_every) == 0:
            print(
                json.dumps(
                    {
                        "stage": "codec_v4_token_audit",
                        "rows": row_count,
                        "total": stored_rows,
                        "max_tokens": int(lengths[:row_count].max()),
                        "elapsed_sec": round(time.perf_counter() - started, 2),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    connection.close()
    if row_count != stored_rows:
        raise RuntimeError(f"token audit read {row_count} rows, expected {stored_rows}")

    maximum = int(lengths.max())
    required_covered = maximum <= int(args.required_max_tokens)
    evaluation_covered = maximum <= int(args.evaluation_max_tokens)
    result = {
        "schema": "sceneplan_transfusion_editing_plan_token_audit",
        "schema_version": 1,
        "status": "PASS" if required_covered else "FAIL",
        "index_path": str(index),
        "index_sha256": index_sha,
        "index_schema": metadata.get("schema"),
        "index_state": metadata.get("state"),
        "rows": row_count,
        "codec_path": str(codec_path),
        "codec_fingerprint": codec.fingerprint,
        "editing_ar_inputs": ["source_foa_latent", "raw_edit_request"],
        "old_sceneplan_input": False,
        "required_max_tokens": int(args.required_max_tokens),
        "required_budget_covers_all_rows": required_covered,
        "evaluation_max_tokens": int(args.evaluation_max_tokens),
        "evaluation_budget_covers_all_rows": evaluation_covered,
        "rows_over_evaluation_budget": int(
            (lengths > int(args.evaluation_max_tokens)).sum()
        ),
        "token_length": _quantiles(lengths),
        "token_length_histogram": {
            str(key): int(value) for key, value in sorted(histogram.items())
        },
        "by_operation": dict(sorted(by_operation.items())),
        "by_latent_bucket": {
            str(key): value for key, value in sorted(by_bucket.items())
        },
        "longest_rows": [
            {
                "tokens": tokens,
                "pair_ordinal": ordinal,
                "pair_id": pair_id,
                "operation": operation,
                "latent_bucket_frames": bucket,
            }
            for tokens, ordinal, pair_id, operation, bucket in sorted(
                longest, reverse=True
            )
        ],
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "auditor_path": str(Path(__file__).resolve()),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
    }
    _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if required_covered else 1


if __name__ == "__main__":
    raise SystemExit(main())
