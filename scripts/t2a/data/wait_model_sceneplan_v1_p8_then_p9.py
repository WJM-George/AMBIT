#!/usr/bin/env python3
"""Wait for the active formal P8 run, then start P9 only on an all-pass summary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sceneplan_v2_common import DATASET_ROOT, atomic_write_json  # noqa: E402


def pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def main() -> int:
    root = DATASET_ROOT.resolve(strict=True)
    materialized = root / "materialized"
    state_path = materialized / "p8_p9_chain_state.json"
    p8_state_path = materialized / "p8_orchestrator_state.json"
    p8_summary_path = materialized / "p8_orchestrator_summary.json"
    started = time.time()
    last_done = -1
    while True:
        if p8_summary_path.is_file():
            summary = json.loads(p8_summary_path.read_text(encoding="utf-8"))
            if not (
                summary.get("schema")
                == "stable_audio_tools.model_sceneplan_p8_orchestrator_summary"
                and summary.get("status") == "complete"
                and int(summary.get("dataset_contract_revision", -1)) == 5
                and int(summary.get("rows", -1)) == 1_124_000
                and summary.get("p10_training_started") is False
                and summary.get("p11_training_started") is False
            ):
                raise RuntimeError("P8 summary exists but is not a valid P9 gate")
            break
        state = json.loads(p8_state_path.read_text(encoding="utf-8"))
        if state.get("status") == "failed":
            raise RuntimeError("formal P8 failed; refusing to start P9")
        pids = [int(value) for value in state.get("worker_pids", [])]
        if not pids or not any(pid_alive(pid) for pid in pids):
            raise RuntimeError("no live P8 workers and no complete P8 summary")
        done = sum(
            1
            for split in ("train", "validation", "test")
            for _ in (materialized / "work_done" / split).glob("work-*.json")
        )
        if done != last_done:
            atomic_write_json(
                state_path,
                {
                    "schema": "stable_audio_tools.model_sceneplan_p8_p9_chain_state",
                    "schema_version": 1,
                    "dataset_contract_revision": 5,
                    "status": "waiting_for_p8",
                    "p8_done_shards": done,
                    "p8_total_shards": 1_099,
                    "elapsed_sec": round(time.time() - started, 3),
                    "p10_training_started": False,
                    "p11_training_started": False,
                },
            )
            print(
                json.dumps(
                    {
                        "chain": "waiting_for_p8",
                        "done_shards": done,
                        "total_shards": 1_099,
                    }
                ),
                flush=True,
            )
            last_done = done
        time.sleep(30)
    atomic_write_json(
        state_path,
        {
            "schema": "stable_audio_tools.model_sceneplan_p8_p9_chain_state",
            "schema_version": 1,
            "dataset_contract_revision": 5,
            "status": "running_p9",
            "p8_done_shards": 1_099,
            "p8_total_shards": 1_099,
            "elapsed_sec": round(time.time() - started, 3),
            "p10_training_started": False,
            "p11_training_started": False,
        },
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "run_model_sceneplan_v1_p9.py")],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        atomic_write_json(
            state_path,
            {
                "schema": "stable_audio_tools.model_sceneplan_p8_p9_chain_state",
                "schema_version": 1,
                "dataset_contract_revision": 5,
                "status": "p9_failed",
                "returncode": result.returncode,
                "elapsed_sec": round(time.time() - started, 3),
                "p10_training_started": False,
                "p11_training_started": False,
            },
        )
        return result.returncode
    atomic_write_json(
        state_path,
        {
            "schema": "stable_audio_tools.model_sceneplan_p8_p9_chain_state",
            "schema_version": 1,
            "dataset_contract_revision": 5,
            "status": "p8_p9_complete_waiting_before_p10",
            "elapsed_sec": round(time.time() - started, 3),
            "p10_training_started": False,
            "p11_training_started": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
