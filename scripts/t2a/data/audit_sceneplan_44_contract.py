#!/usr/bin/env python3
"""Audit semantic token roles and direct 4+4 controls on frozen ScenePlans."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from stable_audio_tools.data.model_sceneplan import (
    compile_model_44_controls,
    compile_model_semantic_caption_v2,
    tokenize_model_semantic_caption,
)


DEFAULT_INDEX = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/training_index/train.sqlite"
)
DEFAULT_OUTPUT = Path(
    os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/audit/conditioning_v3/"
    "sceneplan_44_train_audit.json"
)
DEFAULT_QWEN = Path(os.environ.get("AMBIT_CKPT_ROOT", "checkpoints") + "/pretrained/Qwen/Qwen3.5-0.8B")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordinals(rows: int, requested: int) -> list[int]:
    if requested <= 0 or requested >= rows:
        return list(range(rows))
    return sorted(
        {
            int(round(value))
            for value in np.linspace(0, rows - 1, requested, dtype=np.float64)
        }
    )


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    index = args.index.expanduser().resolve(strict=True)
    qwen = args.qwen.expanduser().resolve(strict=True)
    tokenizer = AutoTokenizer.from_pretrained(str(qwen))
    connection = sqlite3.connect(
        f"file:{index}?mode=ro&immutable=1", uri=True, check_same_thread=False
    )
    rows = int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    selected = _ordinals(rows, int(args.samples))
    counts = Counter()
    max_tokens = 0
    max_token_sample = None
    started = time.monotonic()

    for checked, ordinal in enumerate(selected, start=1):
        row = connection.execute(
            """
            SELECT sample_id, model_num_samples, latent_frames_valid,
                   scene_plan_zlib
            FROM samples WHERE ordinal=?
            """,
            (ordinal,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing ordinal {ordinal}")
        sample_id, model_samples, valid_frames, compressed = row
        sceneplan = json.loads(zlib.decompress(compressed))
        caption = compile_model_semantic_caption_v2(sceneplan)
        if int(caption.get("compiler_version", -1)) != 2:
            raise RuntimeError(f"{sample_id}: canonical semantic compiler is not v2")
        if ' who says "' in caption["text"]:
            raise RuntimeError(f"{sample_id}: protocol quote leaked into canonical caption")
        tokenized = tokenize_model_semantic_caption(
            caption, tokenizer, max_length=512
        )
        attention = tokenized["attention_mask"].astype(bool)
        event_tokens = tokenized["event_source_ids"]
        speech_tokens = tokenized["speech_source_ids"]
        token_count = int(attention.sum())
        if token_count > max_tokens:
            max_tokens = token_count
            max_token_sample = sample_id
        if np.any((event_tokens > 0) & (speech_tokens > 0)):
            raise RuntimeError(f"{sample_id}: event/speech token roles overlap")
        if np.any(event_tokens[~attention] != 0) or np.any(
            speech_tokens[~attention] != 0
        ):
            raise RuntimeError(f"{sample_id}: padding carries a token role")

        expected_sources = {
            int(source["source_id"].split("_")[-1]) + 1
            for source in sceneplan["sources"]
        }
        observed_events = set(int(value) for value in event_tokens if value > 0)
        if observed_events != expected_sources:
            raise RuntimeError(
                f"{sample_id}: event labels {observed_events} != {expected_sources}"
            )
        speech_sources = [
            source for source in sceneplan["sources"] if source["kind"] == "speech"
        ]
        expected_speech = (
            {
                int(speech_sources[0]["source_id"].split("_")[-1]) + 1
            }
            if speech_sources
            else set()
        )
        observed_speech = set(int(value) for value in speech_tokens if value > 0)
        if observed_speech != expected_speech:
            raise RuntimeError(
                f"{sample_id}: speech labels {observed_speech} != {expected_speech}"
            )

        controls = compile_model_44_controls(
            sceneplan,
            model_num_samples=int(model_samples),
            latent_frames_valid=int(valid_frames),
        )
        event_frames = controls["source_event_frame_ids"]
        trajectory = controls["source_trajectory_features"]
        speech_active = controls["speech_active_frame_mask"]
        if event_frames.shape != (4, int(valid_frames)):
            raise RuntimeError(f"{sample_id}: event-frame geometry changed")
        if trajectory.shape != (4, int(valid_frames), 5):
            raise RuntimeError(f"{sample_id}: trajectory geometry changed")
        if speech_active.shape != (int(valid_frames),):
            raise RuntimeError(f"{sample_id}: speech supervision geometry changed")
        if not np.isfinite(trajectory).all():
            raise RuntimeError(f"{sample_id}: non-finite trajectory")
        for slot in range(4):
            allowed = {0, slot + 1}
            observed = set(int(value) for value in np.unique(event_frames[slot]))
            if not observed.issubset(allowed):
                raise RuntimeError(
                    f"{sample_id}: slot {slot} carries event ids {observed}"
                )
        active = event_frames > 0
        if np.any(trajectory[~active] != 0):
            raise RuntimeError(f"{sample_id}: inactive trajectory is not exact zero")
        if active.any():
            az_norm = np.square(trajectory[..., 0]) + np.square(
                trajectory[..., 1]
            )
            el_norm = np.square(trajectory[..., 2]) + np.square(
                trajectory[..., 3]
            )
            if not np.allclose(az_norm[active], 1.0, atol=2e-5):
                raise RuntimeError(f"{sample_id}: azimuth unit circle drift")
            if not np.allclose(el_norm[active], 1.0, atol=2e-5):
                raise RuntimeError(f"{sample_id}: elevation unit circle drift")
            if np.any(trajectory[..., 4][active] <= 0):
                raise RuntimeError(f"{sample_id}: active distance must be positive")
        if bool(speech_sources) != bool(speech_active.any()):
            raise RuntimeError(f"{sample_id}: speech supervision presence drift")

        counts[f"sources_{len(sceneplan['sources'])}"] += 1
        counts[f"room_{sceneplan['room']['type']}"] += 1
        for source in sceneplan["sources"]:
            counts[f"kind_{source['kind']}"] += 1
            counts[f"motion_{source['trajectory']['type']}"] += 1
        if checked % 2_000 == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "checked": checked,
                        "selected": len(selected),
                        "elapsed_sec": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )

    report = {
        "schema": "stable_audio_tools.sceneplan_44_conditioning_audit",
        "schema_version": 1,
        "ok": True,
        "index": str(index),
        "index_sha256": _sha256(index),
        "index_rows": rows,
        "rows_checked": len(selected),
        "selection": "all" if len(selected) == rows else "evenly_spaced",
        "caption_max_tokens": 512,
        "semantic_caption_compiler_version": 2,
        "semantic_caption_surface": "who says: <exact transcript>",
        "observed_max_tokens": max_tokens,
        "observed_max_token_sample": max_token_sample,
        "caption_roles": {
            "event_source_ids": [-1, 0, 1, 2, 3, 4],
            "speech_source_ids": [-1, 0, 1, 2, 3, 4],
            "positive_data_ids": [0, 1, 2, 3, 4],
            "cfg_unknown_id": -1,
        },
        "local_condition": {
            "event_tracks": 4,
            "trajectory_tracks": 4,
            "trajectory_features": [
                "sin_azimuth",
                "cos_azimuth",
                "sin_elevation",
                "cos_elevation",
                "log1p_distance_m",
            ],
            "gain_db_used": False,
            "derived_effective_gain_used": False,
        },
        "counts": dict(sorted(counts.items())),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    _atomic_json(args.output.expanduser().resolve(strict=False), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
