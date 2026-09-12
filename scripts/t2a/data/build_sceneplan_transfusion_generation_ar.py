#!/usr/bin/env python3
"""Build the split-disjoint raw-request manifests for Generation AR.

This is a Generation-only artifact.  It reads the immutable P10 revision-6
ScenePlan indexes and writes raw natural requests plus exact codec-v4 targets;
it never reads or writes Editing pairs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
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
from stable_audio_tools.data.sceneplan_transfusion_generation import (  # noqa: E402
    GENERATION_RAW_REQUEST_CONTRACT,
    GENERATION_REQUEST_MAX_QWEN_TOKENS,
    GENERATION_REQUEST_NUMERIC_CONTRACT,
    GENERATION_REQUEST_SEED,
    GENERATION_TARGET_CONTRACT,
    TEMPLATE_IDS_BY_SPLIT,
    build_generation_request,
)


REVISION_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/revisions/"
    "speech_expansion_noalign_15s_v1"
)
OUTPUT_ROOT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/"
    "transfusion_shared_v1/generation_ar"
)
CODEC_PATH = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/"
    "model_sceneplan_codec_v4"
)
TOKENIZER_PATH = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B")
MAX_PLAN_TOKENS = 1024

SPLIT_AUTHORITY = {
    "train": {
        "rows": 1_600_000,
        "sha256": "ca65d8bd80f7060ea21a3c12135c0bfc2f3a21afacf35ebda87e09d9da4c033d",
    },
    "validation": {
        "rows": 32_000,
        "sha256": "8cfc9a21f11e3d271b87319c0258923a91c8219febb1dcd47c2b051612dd2ba1",
    },
    "test": {
        "rows": 8_000,
        "sha256": "d90b00c4f28fd3395d1b04dbe46ea7e87a08ca72e8e2491aaf8c69227b7ae90c",
    },
}

MANIFEST_SCHEMA = "stable_audio_tools.sceneplan_transfusion_generation_ar"
MANIFEST_SCHEMA_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _readonly_sqlite(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def _tensor_blob(values: Any, dtype: str) -> bytes:
    return np.asarray(values.detach().cpu().numpy(), dtype=dtype).tobytes()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=tuple(SPLIT_AUTHORITY))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=GENERATION_REQUEST_SEED)
    parser.add_argument("--codec", type=Path, default=CODEC_PATH)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    split = str(args.split)
    if int(args.seed) != GENERATION_REQUEST_SEED:
        raise ValueError("Generation Transfusion v1 uses frozen seed 42")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive")

    authority = SPLIT_AUTHORITY[split]
    source_path = (REVISION_ROOT / "training_index" / f"{split}.sqlite").resolve(
        strict=True
    )
    codec_path = args.codec.expanduser().resolve(strict=True)
    tokenizer_path = args.tokenizer.expanduser().resolve(strict=True)
    output = (
        args.output.expanduser().resolve(strict=False)
        if args.output is not None
        else (OUTPUT_ROOT / f"{split}.sqlite").resolve(strict=False)
    )
    start = int(args.start)
    total_rows = int(authority["rows"])
    stop = total_rows if args.limit is None else start + int(args.limit)
    if not 0 <= start < stop <= total_rows:
        raise ValueError(
            f"requested ordinal interval [{start},{stop}) is outside {split} "
            f"rows [0,{total_rows})"
        )
    is_full_split = start == 0 and stop == total_rows
    if not is_full_split and args.output is None:
        raise ValueError("a partial pilot requires an explicit --output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = output.with_suffix(".report.json")
    if output.exists() or report_path.exists():
        raise FileExistsError(
            f"refusing to overwrite Generation artifact: {output} or {report_path}"
        )

    actual_source_sha256 = _sha256_file(source_path)
    if actual_source_sha256 != str(authority["sha256"]):
        raise RuntimeError(
            f"canonical {split} source index SHA256 changed: "
            f"{actual_source_sha256}"
        )

    source = _readonly_sqlite(source_path)
    source_metadata = dict(source.execute("SELECT key,value FROM metadata"))
    observed_rows = int(source.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    if (
        observed_rows != total_rows
        or int(source_metadata.get("rows", -1)) != total_rows
        or source_metadata.get("schema")
        != "stable_audio_tools.sceneplan_v2_training_index"
        or source_metadata.get("schema_version") != "3"
        or source_metadata.get("contract_revision") != "6"
        or source_metadata.get("frozen") != "true"
    ):
        raise RuntimeError(f"canonical {split} source-index contract changed")

    codec = ModelScenePlanCodecV4(codec_path)
    if codec.fingerprint != "b512c31b96c6775af6e46e9a5cdf3d513658dcab61b6d357853a019ac16e3874":
        raise RuntimeError("frozen codec-v4 fingerprint changed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=False
    )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".building", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    destination = sqlite3.connect(temporary)
    destination.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE rows (
            ordinal INTEGER PRIMARY KEY,
            source_ordinal INTEGER NOT NULL UNIQUE,
            sample_id TEXT NOT NULL UNIQUE,
            template_id TEXT NOT NULL,
            raw_user_request TEXT NOT NULL,
            raw_user_request_sha256 TEXT NOT NULL,
            raw_request_qwen_tokens INTEGER NOT NULL,
            target_sceneplan_zlib BLOB NOT NULL,
            target_sceneplan_sha256 TEXT NOT NULL,
            target_token_ids_u16le BLOB NOT NULL,
            target_loss_group_ids_i16le BLOB NOT NULL,
            target_token_count INTEGER NOT NULL,
            target_tokens_sha256 TEXT NOT NULL,
            source_sceneplan_sha256 TEXT NOT NULL,
            latent_frames_valid INTEGER NOT NULL,
            source_count INTEGER NOT NULL,
            room_type TEXT NOT NULL,
            motion_signature TEXT NOT NULL
        );
        """
    )

    template_counts: Counter[str] = Counter()
    source_count_counts: Counter[int] = Counter()
    room_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()
    max_raw_tokens = 0
    max_plan_tokens = 0
    completed_rows = 0
    next_progress_rows = 10_000
    started = time.perf_counter()
    complete = False

    def flush(pending: list[dict[str, Any]]) -> None:
        nonlocal completed_rows, max_raw_tokens, max_plan_tokens
        if not pending:
            return
        tokenized = tokenizer(
            [record["raw_user_request"] for record in pending],
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        raw_lengths = [len(value) for value in tokenized]
        if max(raw_lengths) > GENERATION_REQUEST_MAX_QWEN_TOKENS:
            offenders = [
                (pending[index]["sample_id"], length)
                for index, length in enumerate(raw_lengths)
                if length > GENERATION_REQUEST_MAX_QWEN_TOKENS
            ]
            raise RuntimeError(
                "Generation raw request exceeds the frozen 512-token Qwen "
                f"limit: {offenders[:8]}"
            )
        sql_rows = []
        for record, raw_length in zip(pending, raw_lengths):
            plan_text = _canonical_json(record["target_sceneplan"])
            plan_bytes = plan_text.encode("utf-8")
            encoded = codec.encode(
                record["target_sceneplan"], max_tokens=MAX_PLAN_TOKENS
            )
            token_blob = _tensor_blob(encoded["input_ids"], "<u2")
            group_blob = _tensor_blob(encoded["loss_group_ids"], "<i2")
            token_count = int(encoded["input_ids"].numel())
            sql_rows.append(
                (
                    completed_rows + len(sql_rows),
                    record["source_ordinal"],
                    record["sample_id"],
                    record["template_id"],
                    record["raw_user_request"],
                    _sha256_bytes(record["raw_user_request"].encode("utf-8")),
                    raw_length,
                    zlib.compress(plan_bytes, level=1),
                    _sha256_bytes(plan_bytes),
                    token_blob,
                    group_blob,
                    token_count,
                    _sha256_bytes(token_blob),
                    record["source_sceneplan_sha256"],
                    record["latent_frames_valid"],
                    len(record["target_sceneplan"]["sources"]),
                    str(record["target_sceneplan"]["room"]["type"]),
                    ",".join(
                        str(source["trajectory"]["type"])
                        for source in record["target_sceneplan"]["sources"]
                    ),
                )
            )
            template_counts[record["template_id"]] += 1
            source_count_counts[len(record["target_sceneplan"]["sources"])] += 1
            room_counts[str(record["target_sceneplan"]["room"]["type"])] += 1
            motion_counts.update(
                str(source["trajectory"]["type"])
                for source in record["target_sceneplan"]["sources"]
            )
            max_raw_tokens = max(max_raw_tokens, int(raw_length))
            max_plan_tokens = max(max_plan_tokens, token_count)
        destination.executemany(
            "INSERT INTO rows VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            sql_rows,
        )
        completed_rows += len(sql_rows)
        pending.clear()

    try:
        pending: list[dict[str, Any]] = []
        cursor = source.execute(
            """
            SELECT ordinal, sample_id, latent_frames_valid, scene_plan_zlib,
                   model_sceneplan_sha256
            FROM samples
            WHERE ordinal >= ? AND ordinal < ?
            ORDER BY ordinal
            """,
            (start, stop),
        )
        for source_ordinal, sample_id, valid_frames, plan_zlib, source_plan_sha in cursor:
            plan_bytes = zlib.decompress(plan_zlib)
            source_plan = json.loads(plan_bytes)
            canonical_source_bytes = _canonical_json(source_plan).encode("utf-8")
            if (
                str(source_plan["sample_id"]) != str(sample_id)
                or _sha256_bytes(canonical_source_bytes) != str(source_plan_sha)
            ):
                raise RuntimeError(
                    f"source ScenePlan provenance mismatch at ordinal {source_ordinal}"
                )
            request = build_generation_request(
                source_plan, codec, split=split, seed=int(args.seed)
            )
            pending.append(
                {
                    "source_ordinal": int(source_ordinal),
                    "sample_id": str(sample_id),
                    "latent_frames_valid": int(valid_frames),
                    "source_sceneplan_sha256": str(source_plan_sha),
                    "raw_user_request": request.text,
                    "template_id": request.template_id,
                    "target_sceneplan": request.target_sceneplan,
                }
            )
            if len(pending) >= int(args.batch_size):
                flush(pending)
                if completed_rows >= next_progress_rows:
                    destination.commit()
                    print(
                        json.dumps(
                            {
                                "event": "progress",
                                "split": split,
                                "rows": completed_rows,
                                "target_rows": stop - start,
                                "max_raw_qwen_tokens": max_raw_tokens,
                                "max_plan_tokens": max_plan_tokens,
                                "elapsed_sec": round(time.perf_counter() - started, 3),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    while next_progress_rows <= completed_rows:
                        next_progress_rows += 10_000
        flush(pending)
        expected_output_rows = stop - start
        if completed_rows != expected_output_rows:
            raise RuntimeError(
                f"Generation build produced {completed_rows} rows, expected "
                f"{expected_output_rows}"
            )
        metadata = {
            "schema": MANIFEST_SCHEMA,
            "schema_version": str(MANIFEST_SCHEMA_VERSION),
            "raw_request_contract": GENERATION_RAW_REQUEST_CONTRACT,
            "raw_request_numeric_contract": GENERATION_REQUEST_NUMERIC_CONTRACT,
            "target_contract": GENERATION_TARGET_CONTRACT,
            "split": split,
            "seed": str(args.seed),
            "rows": str(completed_rows),
            "source_ordinal_start": str(start),
            "source_ordinal_stop": str(stop),
            "is_full_split": str(is_full_split).lower(),
            "source_index": str(source_path),
            "source_index_rows": str(total_rows),
            "source_index_sha256": actual_source_sha256,
            "codec_path": str(codec_path),
            "codec_fingerprint": codec.fingerprint,
            "tokenizer_path": str(tokenizer_path),
            "template_numbers_json": json.dumps(TEMPLATE_IDS_BY_SPLIT[split]),
            "template_counts_json": json.dumps(template_counts, sort_keys=True),
            "source_count_counts_json": json.dumps(source_count_counts, sort_keys=True),
            "room_counts_json": json.dumps(room_counts, sort_keys=True),
            "motion_counts_json": json.dumps(motion_counts, sort_keys=True),
            "raw_request_qwen_max_tokens": str(max_raw_tokens),
            "raw_request_qwen_token_limit": str(GENERATION_REQUEST_MAX_QWEN_TOKENS),
            "raw_request_truncation_count": "0",
            "plan_max_tokens_observed": str(max_plan_tokens),
            "plan_token_limit": str(MAX_PLAN_TOKENS),
            "sample_id_is_ar_target": "false",
            "renderer_or_asset_lineage_in_request": "false",
            "builder_implementation_sha256": _sha256_file(
                Path(__file__).resolve(strict=True)
            ),
            "request_compiler_implementation_sha256": _sha256_file(
                REPO_ROOT
                / "stable_audio_tools/data/sceneplan_transfusion_generation.py"
            ),
        }
        destination.executemany(
            "INSERT INTO metadata(key,value) VALUES (?,?)", metadata.items()
        )
        destination.executescript(
            """
            CREATE INDEX rows_template ON rows(template_id);
            CREATE INDEX rows_source_count ON rows(source_count);
            CREATE INDEX rows_length ON rows(latent_frames_valid);
            ANALYZE;
            """
        )
        destination.commit()
        complete = True
    finally:
        destination.close()
        source.close()
        if not complete:
            temporary.unlink(missing_ok=True)

    os.replace(temporary, output)
    output_sha256 = _sha256_file(output)
    report = {
        "schema": "stable_audio_tools.sceneplan_transfusion_generation_ar_build",
        "schema_version": 1,
        "status": "PASS",
        "split": split,
        "rows": completed_rows,
        "is_full_split": is_full_split,
        "source_ordinal_interval": [start, stop],
        "source_index": str(source_path),
        "source_index_sha256": actual_source_sha256,
        "output": str(output),
        "output_sha256": output_sha256,
        "codec_fingerprint": codec.fingerprint,
        "template_counts": dict(sorted(template_counts.items())),
        "source_count_counts": {
            str(key): value for key, value in sorted(source_count_counts.items())
        },
        "room_counts": dict(sorted(room_counts.items())),
        "motion_counts": dict(sorted(motion_counts.items())),
        "max_raw_qwen_tokens": max_raw_tokens,
        "max_plan_tokens": max_plan_tokens,
        "truncated_rows": 0,
        "sample_id_leakage_rows": 0,
        "elapsed_sec": round(time.perf_counter() - started, 3),
    }
    _atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
