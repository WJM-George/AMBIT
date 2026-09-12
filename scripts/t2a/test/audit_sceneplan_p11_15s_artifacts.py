#!/usr/bin/env python3
"""Read-only audit of canonical P11-v6 648-frame codec and manifests."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402
from stable_audio_tools.data.model_sceneplan_codec import (  # noqa: E402
    load_model_sceneplan_codec,
)
from stable_audio_tools.data.model_sceneplan_codec_v4 import (  # noqa: E402
    MAX_FRAMES,
    ModelScenePlanCodecV4,
)
from stable_audio_tools.data.sceneplan_p11_dataset import (  # noqa: E402
    P11_MANIFEST_SCHEMA,
    P11_MANIFEST_VERSION,
)


CONFIG_ROOT = REPO_ROOT / "stable_audio_tools/configs/dataset_configs"
SPECS = {
    "pilot90": ("sceneplan_p11_pilot90.json", 30, 90, None),
    "heldout900": ("sceneplan_p11_heldout900.json", 300, 900, None),
    "validation": ("sceneplan_p11_validation.json", 32_000, 96_000, (24_000, 8_000)),
    "test": ("sceneplan_p11_test.json", 8_000, 24_000, (6_000, 2_000)),
    "train": ("sceneplan_p11_train.json", 1_600_000, 4_800_000, (1_200_000, 400_000)),
}


def _readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        f"file:{path.resolve(strict=True)}?mode=ro&immutable=1", uri=True
    )
    connection.execute("PRAGMA query_only=ON")
    return connection


def main() -> None:
    reports = {}
    fingerprints = set()
    for name, (config_name, expected_base, expected_rows, expected_buckets) in SPECS.items():
        config = load_config(CONFIG_ROOT / config_name)
        if int(config["latent_crop_length"]) != MAX_FRAMES:
            raise RuntimeError(f"{name}: latent_crop_length is not {MAX_FRAMES}")
        if (
            config.get("semantic_representation")
            != "windowed_clap_pooler_output"
            or float(config.get("semantic_window_sec", -1.0)) != 5.0
            or float(config.get("semantic_hop_sec", -1.0)) != 5.0
        ):
            raise RuntimeError(f"{name}: canonical 15-second CLAP windowing changed")
        codec = load_model_sceneplan_codec(config["codec_path"])
        if not isinstance(codec, ModelScenePlanCodecV4) or codec.max_frames != MAX_FRAMES:
            raise RuntimeError(f"{name}: canonical codec-v4/648 mismatch")
        fingerprints.add(codec.fingerprint)
        manifest_path = Path(config["manifest_path"])
        index_path = Path(config["datasets"][0]["path"]).resolve(strict=True)
        connection = _readonly(manifest_path)
        metadata = dict(connection.execute("SELECT key,value FROM metadata"))
        required = {
            "schema": P11_MANIFEST_SCHEMA,
            "schema_version": str(P11_MANIFEST_VERSION),
            "p10_dataset_contract_revision": "6",
            "planner_max_latent_frames": str(MAX_FRAMES),
            "codec_fingerprint": codec.fingerprint,
            "source_index": str(index_path),
            "base_samples": str(expected_base),
            "rows": str(expected_rows),
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise RuntimeError(
                    f"{name}: metadata {key}={metadata.get(key)!r}, expected {expected!r}"
                )
        counts = {
            str(task): int(count)
            for task, count in connection.execute(
                "SELECT task,COUNT(*) FROM rows GROUP BY task"
            )
        }
        if counts != {
            "generation": expected_base,
            "understanding": expected_base,
            "editing": expected_base,
        }:
            raise RuntimeError(f"{name}: G/U/E counts changed: {counts}")
        connection.execute("ATTACH DATABASE ? AS source", (str(index_path),))
        short, long, minimum, maximum, missing = connection.execute(
            """
            SELECT SUM(s.latent_frames_valid<=432),
                   SUM(s.latent_frames_valid>432),
                   MIN(s.latent_frames_valid),MAX(s.latent_frames_valid),
                   SUM(s.ordinal IS NULL)
            FROM rows r LEFT JOIN source.samples s ON s.ordinal=r.target_ordinal
            WHERE r.task='generation'
            """
        ).fetchone()
        connection.close()
        buckets = (int(short), int(long))
        if int(missing or 0) != 0 or not 1 <= int(minimum) <= int(maximum) <= MAX_FRAMES:
            raise RuntimeError(f"{name}: source-index coverage or frame envelope changed")
        if expected_buckets is not None and buckets != expected_buckets:
            raise RuntimeError(f"{name}: 432/648 bucket counts changed: {buckets}")
        if expected_buckets is None and int(long) <= 0:
            raise RuntimeError(f"{name}: sampled gate contains no >432-frame scene")
        reports[name] = {
            "base_scenes": expected_base,
            "rows": expected_rows,
            "short_le_432": int(short),
            "long_gt_432": int(long),
            "min_frames": int(minimum),
            "max_frames": int(maximum),
        }
    if len(fingerprints) != 1:
        raise RuntimeError("canonical P11 configs do not share one codec fingerprint")
    print(
        json.dumps(
            {
                "status": "PASS",
                "schema": "stable_audio_tools.p11_15s_artifact_audit",
                "schema_version": 1,
                "codec_fingerprint": next(iter(fingerprints)),
                "max_latent_frames": MAX_FRAMES,
                "semantic_evidence": {
                    "representation": "windowed_clap_pooler_output",
                    "window_sec": 5.0,
                    "hop_sec": 5.0,
                },
                "manifests": reports,
                "semantic_cache_gate": "PENDING_NOT_BYPASSED",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
