"""CPU checks for isolation, sampling, reporting and the GPU probe's data replay."""
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts.t2a.test.profile_sceneplan_transfusion_editing_clap44_gpu import (
    CELLS, ROOT, batch_digest, compare_state, make_loader, run, select_probe_rows,
    summarize_ranks, verify_launcher_lock,
)


def fixture_index(tmp_path, split="train"):
    path = tmp_path / f"{split}.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
    connection.executemany("INSERT INTO metadata VALUES (?,?)",
        (("split", split), ("rows", "2000"), ("state", "materialized_complete_frozen")))
    if split == "train":
        # Like the real index, put the long examples in a later contiguous block.
        connection.execute("CREATE TABLE pairs(pair_ordinal INTEGER PRIMARY KEY,pair_id TEXT,operation TEXT,latent_bucket_frames INTEGER)")
        cells = sorted(CELLS, key=lambda cell: (cell[1], cell[0]))
        connection.executemany("INSERT INTO pairs VALUES (?,?,?,?)", [
            (i, f"pair-{i}", cells[i // 200][0], cells[i // 200][1]) for i in range(2000)])
    connection.commit(); connection.close()
    return path


def test_train_sampling_covers_later_length_block_and_is_reproducible(tmp_path):
    index = fixture_index(tmp_path)
    selected = select_probe_rows(index, expected_rows=2000, total_pairs=160, seed=42)
    assert selected == select_probe_rows(index, expected_rows=2000, total_pairs=160, seed=42)
    assert len({row["ordinal"] for row in selected["rows"]}) == 160
    for operation, bucket in CELLS:
        assert sum(row["operation"] == operation and row["bucket_frames"] == bucket for row in selected["rows"]) == 16
    # Randomize before rank partitioning, so stride-five rank assignment does
    # not accidentally confine a rank to one or two operation/length cells.
    for rank in range(5):
        assert len({(row["operation"], row["bucket_frames"]) for row in selected["rows"][rank::5]}) >= 8


@pytest.mark.parametrize("split", ["validation", "test"])
def test_held_out_split_is_rejected_before_any_pair_query(tmp_path, split):
    # The forbidden index deliberately has no pairs table.
    with pytest.raises(ValueError, match="frozen training split"):
        select_probe_rows(fixture_index(tmp_path, split), expected_rows=2000, total_pairs=160, seed=42)


def test_lock_proof_requires_actual_flock_on_exact_file(tmp_path):
    path, other = tmp_path / "lock", tmp_path / "other-lock"
    other.touch()
    with path.open("w") as handle:
        with pytest.raises(RuntimeError, match="does not hold"):
            verify_launcher_lock(path, pid=os.getpid(), fd=handle.fileno())
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert verify_launcher_lock(path, pid=os.getpid(), fd=handle.fileno())["inode"] == path.stat().st_ino
        with pytest.raises(RuntimeError, match="does not hold"):
            verify_launcher_lock(other, pid=os.getpid(), fd=handle.fileno())


def test_missing_launcher_lock_fails_before_cuda_initialization(monkeypatch, tmp_path):
    from scripts.t2a.train import editing_gpu_runtime as runtime
    monkeypatch.delenv("EDITING_GPU_LEASES", raising=False)
    monkeypatch.setattr(runtime, "gpu_topology", lambda: {"physical_indices": [0, 1, 2]})
    monkeypatch.setattr(runtime, "resource_paths", lambda topology: [tmp_path / "required-lock"])
    def forbidden(*args, **kwargs):
        pytest.fail("GPU initialization happened before resource ownership was established")
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(RuntimeError, match="GPU launcher"):
        run(SimpleNamespace(output=tmp_path, max_wall_seconds=900))


def test_encoder_validation_rejects_busy_gpu_before_loading(monkeypatch, tmp_path):
    from contextlib import contextmanager
    from scripts.t2a.train import editing_gpu_runtime as runtime
    from scripts.t2a.eval.evaluate_sceneplan_transfusion_editing_clap44 import main
    index = tmp_path / "validation.sqlite"
    index.touch()
    monkeypatch.setattr(runtime, "gpu_topology", lambda: {"devices": [{"uuid": "GPU-test"}]})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    @contextmanager
    def busy(topology):
        raise RuntimeError("a requested GPU already has a compute process; leave it running")
        yield
    monkeypatch.setattr(runtime, "gpu_lease", busy)
    monkeypatch.setattr(sys, "argv", ["clap44-validation", "--checkpoint", str(tmp_path / "absent.pt"),
        "--validation-index", str(index), "--preflight", str(tmp_path / "absent.json"),
        "--output", str(tmp_path / "output"), "--device", "cuda"])
    with pytest.raises(RuntimeError, match="already has a compute process"):
        main()
    assert not (tmp_path / "output").exists()


def test_launcher_refuses_held_lock_without_creating_an_attempt(tmp_path):
    original = ROOT / "scripts/t2a/test/run_sceneplan_transfusion_editing_clap44_gpu_preflight_5gpu.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    smi = fake_bin / "nvidia-smi"
    smi.write_text("#!/bin/sh\nprintf '0, 0000:01:00.0, GPU-probe-test, Test GPU\\n'\n")
    smi.chmod(0o755)
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    lock = lock_dir / "GPU-probe-test.lock"
    launcher = original
    output = tmp_path / "attempts"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", EDITING_GPUS="0", EDITING_GPU_LOCK_DIR=str(lock_dir),
        EDITING_GPU_LEASES="", EDITING_DIT_RUN_CONTRACT=str(tmp_path / "no-dit-contract"),
        PATH=str(fake_bin) + os.pathsep + os.environ["PATH"], CLAP44_PROBE_ROOT=str(output), P10_REPO=str(ROOT),
        CLAP44_PROBE_CONFIG=str(ROOT / "stable_audio_tools/configs/model_configs/txt2audio/t2a/editing_clap44_v1.json"))
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 1 and "resource is reserved" in result.stderr
    assert not output.exists()
    result = subprocess.run(["bash", str(launcher)], env=dict(env, CLAP44_PROBE_MAX_WALL_SECONDS="901"),
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and not output.exists()
    result = subprocess.run(["bash", str(launcher)], env=dict(env, CLAP44_PROBE_CONFIG=str(tmp_path / "missing.json")),
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and "configuration is not readable" in result.stderr and not output.exists()


def test_recovery_comparison_reports_small_optimizer_differences_and_structure():
    expected = {"state": {3: {"step": torch.tensor(10.), "exp_avg": torch.tensor([0., 1.])}}, "lr": .003}
    actual = {"state": {3: {"step": torch.tensor(10.), "exp_avg": torch.tensor([1e-9, 1.])}}, "lr": .003}
    comparison = compare_state(expected, actual)
    assert not comparison["exact"] and comparison["different_fields"] == 1
    assert comparison["max_absolute_error"] == pytest.approx(1e-9)
    assert comparison["max_relative_error"] == 1.
    assert "exp_avg" in comparison["max_absolute_error_path"]
    assert compare_state(expected, expected)["exact"]
    assert not compare_state({"rng": (1, 2)}, {"rng": [1, 2]})["exact"]
    assert not compare_state(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["exact"]
    assert not compare_state(torch.ones(1), torch.ones(2))["exact"]
    assert not compare_state(float("inf"), float("inf"))["exact"]
    json.dumps(comparison, allow_nan=False)


def rank_results():
    return [{"rank": rank, "timed_steps": 8, "timed_wall_seconds": float(rank + 1),
             "resume_comparison": {"model": {"exact": True}, "optimizer": {"exact": True}, "rng": {"exact": True}}}
            for rank in range(5)]


def test_throughput_uses_slowest_rank_and_never_asserts_model_quality():
    records = rank_results()
    result = summarize_ranks(records, timed_steps=8, pairs_per_gpu=32, world_size=5)
    assert result["global_pairs_per_second"] == 8 * 32
    assert result["global_audio_views_per_second"] == 8 * 32 * 2
    assert result["resume_exact_all_ranks"] is True
    assert result["quality_gate_passed"] is False and result["automatic_training_promotion"] is False
    records[3]["resume_comparison"]["rng"]["exact"] = False
    result = summarize_ranks(records, timed_steps=8, pairs_per_gpu=32, world_size=5)
    assert result["status"] == "PROBE_COMPLETE_RESUME_DIFFERENCES_REQUIRE_REVIEW"
    assert result["quality_gate_passed"] is False
    with pytest.raises(ValueError, match="rank coverage"):
        summarize_ranks(records[:-1], timed_steps=8, pairs_per_gpu=32, world_size=5)
    records[0]["timed_steps"] = 7
    with pytest.raises(ValueError, match="timing window"):
        summarize_ranks(records, timed_steps=8, pairs_per_gpu=32, world_size=5)


def test_three_rank_probe_retains_all_pairs_and_balanced_strata(tmp_path):
    selected = select_probe_rows(fixture_index(tmp_path), expected_rows=2000,
                                total_pairs=12 * 32 * 3, seed=42)
    assert len(selected["rows"]) == len({row["ordinal"] for row in selected["rows"]}) == 1152
    assert sorted(row["pairs"] for row in selected["stratum_quotas"]) == [115] * 8 + [116] * 2
    assert all(len(selected["rows"][rank::3]) == 384 for rank in range(3))
    result = summarize_ranks(rank_results()[:3], timed_steps=8, pairs_per_gpu=32, world_size=3)
    assert result["global_pairs_per_second"] == 256


class TinyPairs(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return [{"latent": torch.full((64, 12 + index), float(index + offset), dtype=torch.float16),
                 "label": {"pair_id": f"pair-{index}", "role": role}, "counterfactuals": []}
                for offset, role in enumerate(("source", "target"))]


def test_spawn_loader_replays_batches_without_consuming_model_rng():
    training = {"seed": 42, "pairs_per_gpu": 2, "workers_per_gpu": 2}
    torch.manual_seed(123)
    state = torch.get_rng_state().clone()
    first = [batch_digest(batch) for batch in make_loader(TinyPairs(), training, rank=2)]
    assert torch.equal(state, torch.get_rng_state())
    recovered = iter(make_loader(TinyPairs(), training, rank=2))
    next(recovered); next(recovered)
    assert [batch_digest(batch) for batch in recovered] == first[2:]
    assert torch.equal(state, torch.get_rng_state())
