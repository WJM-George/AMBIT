#!/usr/bin/env python3
"""Bounded CLAP44 throughput and recovery probe on explicitly allocated GPUs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from stable_audio_tools.data.sceneplan_transfusion_editing_clap44 import EditingCLAP44Dataset, collate_clap44
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import CLAP44Config, EditingCLAP44
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_text import FrozenCLAP44TextFeatures
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_pretrain import (
    load_training_checkpoint, optimizer_and_scheduler, save_checkpoint, sha256, training_step,
)
from scripts.t2a.train.train_sceneplan_transfusion_editing_clap44 import rng_state, restore_rng

SCHEMA = "editing_clap44_gpu_preflight_v1"
OPERATIONS = ("event_addition", "event_removal", "linear_to_static", "static_to_linear", "stationary_spatial_relocation")
CELLS = tuple((op, bucket) for op in OPERATIONS for bucket in (432, 648))
DATA_ROOT = Path("/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1")
RESOURCE_LOCK = DATA_ROOT / "materialized/locks/training-chain.lock"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_launcher_lock(path=RESOURCE_LOCK, *, pid=None, fd=7):
    """Torchrun closes child FDs: prove the living launcher retains its flock."""
    pid = int(os.environ.get("EDITING_CLAP44_PREFLIGHT_LOCK_PID", "0")) if pid is None else pid
    if pid <= 0:
        raise RuntimeError("use the resource-locked CLAP44 GPU probe launcher")
    expected = Path(path).resolve(strict=True).stat()
    actual = Path(f"/proc/{pid}/fd/{fd}").stat()
    information = Path(f"/proc/{pid}/fdinfo/{fd}").read_text()
    if ((actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino) or
            not any(line.startswith("lock:") and "FLOCK" in line and "WRITE" in line
                    for line in information.splitlines())):
        raise RuntimeError("the CLAP44 launcher does not hold the Editing resource lock")
    return {"path": str(Path(path).resolve()), "pid": pid, "fd": fd, "inode": actual.st_ino}


def select_probe_rows(index, *, expected_rows, total_pairs, seed):
    """Read only train metadata at fixed sampled ordinals, never scan held-out rows."""
    if total_pairs < len(CELLS):
        raise ValueError("probe pair count must cover all ten strata")
    connection = sqlite3.connect(Path(index).resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
    try:
        metadata = dict(connection.execute("SELECT key,value FROM metadata WHERE key IN ('split','rows','state')"))
        if (metadata.get("split") != "train" or int(metadata.get("rows", -1)) != expected_rows or
                metadata.get("state") != "materialized_complete_frozen"):
            raise ValueError("GPU diagnostics require the frozen training split")
        quota, remainder = divmod(total_pairs, len(CELLS))
        quota_order = list(CELLS)
        random.Random(seed).shuffle(quota_order)
        quotas = {cell: quota + (i < remainder) for i, cell in enumerate(quota_order)}
        selected = {cell: [] for cell in CELLS}
        # Uniform ordinal sampling covers both length blocks even when the
        # training index is physically sorted by length. This uses no blobs.
        candidates = random.Random(seed).sample(range(expected_rows), min(expected_rows, max(4096, total_pairs * 24)))
        inspected = 0
        for start in range(0, len(candidates), 256):
            chunk = candidates[start:start + 256]
            placeholders = ",".join("?" for _ in chunk)
            rows = {row[0]: row for row in connection.execute(
                f"SELECT pair_ordinal,pair_id,operation,latent_bucket_frames FROM pairs WHERE pair_ordinal IN ({placeholders})", chunk)}
            if len(rows) != len(chunk):
                raise RuntimeError("sampled train ordinals are missing")
            inspected += len(chunk)
            for ordinal in chunk:
                row = rows[ordinal]
                cell = (row[2], row[3])
                if cell not in selected:
                    raise ValueError("unexpected Editing operation or length bucket")
                if len(selected[cell]) < quotas[cell]:
                    selected[cell].append({"ordinal": row[0], "pair_id": row[1], "operation": row[2], "bucket_frames": row[3]})
            if all(len(values) == quotas[cell] for cell, values in selected.items()):
                break
        if any(len(values) != quotas[cell] for cell, values in selected.items()):
            raise RuntimeError("bounded metadata sample did not fill all train strata")
        rows = [row for cell in CELLS for row in selected[cell]]
        random.Random(seed + 1).shuffle(rows)
        return {"rows": rows, "sampled_metadata_rows": inspected,
                "pairs_per_stratum": quota if not remainder else None,
                "stratum_quotas": [{"operation": cell[0], "bucket_frames": cell[1], "pairs": quotas[cell]}
                                  for cell in CELLS]}
    finally:
        connection.close()


def compare_state(expected, actual):
    """Report exactness and numerical differences without silently choosing tolerances."""
    result = {"exact": True, "values_compared": 0, "different_fields": 0,
              "max_absolute_error": 0., "max_relative_error": 0., "examples": []}

    def different(path, reason):
        result["exact"] = False
        result["different_fields"] += 1
        if len(result["examples"]) < 20:
            result["examples"].append({"path": path, "reason": reason})

    def visit(left, right, path):
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor) or left.shape != right.shape or left.dtype != right.dtype:
                different(path, "tensor type/shape/dtype changed")
                return
            left, right = left.detach().cpu(), right.detach().cpu()
            result["values_compared"] += left.numel()
            if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
                different(path, "non-finite tensor")
            elif not torch.equal(left, right):
                delta = (left.double() - right.double()).abs()
                scale = torch.maximum(left.double().abs(), right.double().abs()).clamp_min(1e-12)
                absolute, relative = float(delta.max()), float((delta / scale).max())
                if absolute > result["max_absolute_error"]:
                    result["max_absolute_error"] = absolute
                    result["max_absolute_error_path"] = path
                result["max_relative_error"] = max(result["max_relative_error"], relative)
                different(path, "tensor values changed")
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                different(path, "dictionary keys changed")
                return
            for key in left:
                visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                different(path, "sequence structure changed")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, f"{path}[{index}]")
        else:
            result["values_compared"] += 1
            if type(left) is not type(right) or left != right or (isinstance(left, float) and not math.isfinite(left)):
                different(path, "scalar value/type changed")
                if isinstance(left, float) and isinstance(right, float) and math.isfinite(left) and math.isfinite(right):
                    absolute = abs(left - right)
                    if absolute > result["max_absolute_error"]:
                        result["max_absolute_error"] = absolute
                        result["max_absolute_error_path"] = path
                    result["max_relative_error"] = max(result["max_relative_error"], absolute / max(abs(left), abs(right), 1e-12))
    visit(expected, actual, "state")
    return result


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_copy(item) for item in value)
    return deepcopy(value)


def batch_digest(batch):
    digest = hashlib.sha256()
    for key in sorted(batch):
        value = batch[key]
        digest.update(key.encode())
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.cpu().contiguous().numpy().tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())
    return digest.hexdigest()


class CudaPhases:
    def __init__(self):
        self.events = {}

    @contextmanager
    def __call__(self, name):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events[name] = (start, end)

    def milliseconds(self):
        # The caller synchronizes before reading. These are device-timeline
        # elapsed times, which can include host/communication gaps, not pure kernels.
        return {name: start.elapsed_time(end) for name, (start, end) in self.events.items()}


def summarize_ranks(records, *, timed_steps, pairs_per_gpu, world_size):
    if world_size < 1 or len(records) != world_size or sorted(row["rank"] for row in records) != list(range(world_size)):
        raise ValueError("incomplete GPU probe rank coverage")
    wall = max(row["timed_wall_seconds"] for row in records)
    if not math.isfinite(wall) or wall <= 0 or any(row["timed_steps"] != timed_steps for row in records):
        raise ValueError("incomplete or invalid timing window")
    exact = all(all(value["exact"] for value in row["resume_comparison"].values()) for row in records)
    return {"status": "PROBE_COMPLETE_EXACT_RESUME_NOT_QUALITY_PASS" if exact else "PROBE_COMPLETE_RESUME_DIFFERENCES_REQUIRE_REVIEW",
            "resume_exact_all_ranks": exact, "global_pairs_per_second": timed_steps * pairs_per_gpu * world_size / wall,
            "global_audio_views_per_second": 2 * timed_steps * pairs_per_gpu * world_size / wall,
            "slowest_rank_timed_wall_seconds": wall, "quality_gate_passed": False,
            "automatic_training_promotion": False, "rank_results": records}


def make_loader(dataset, training, rank):
    generator = torch.Generator().manual_seed(int(training["seed"]) + 900001 + rank)
    workers = int(training["workers_per_gpu"])
    return DataLoader(dataset, batch_size=int(training["pairs_per_gpu"]), shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True, collate_fn=collate_clap44,
        persistent_workers=False, generator=generator, multiprocessing_context="spawn" if workers else None)


def audit_contract(args, topology, lock):
    world = len(topology["physical_indices"])
    cfg = json.loads(args.config.read_text())
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("status") != "PASS" or any(preflight["indices"][split]["rows"] != size
            for split, size in (("train", 1000000), ("validation", 20000), ("test", 5000))):
        raise ValueError("the full approved 1M/20k/5k preflight is required")
    index = args.index.resolve(strict=True)
    if str(index) != preflight["indices"]["train"]["path"] or sha256(index) != preflight["indices"]["train"]["sha256"]:
        raise ValueError("probe index differs from the frozen training index")
    dataset = EditingCLAP44Dataset(index, expected_rows=1000000)
    if dataset.index_sha256 != preflight["indices"]["train"]["sha256"] or dataset.split != "train":
        raise ValueError("probe dataset marker differs from the training preflight")
    if int(cfg["training"]["pairs_per_gpu"]) != 32 or int(cfg["training"]["workers_per_gpu"]) != 4:
        raise ValueError("this baseline is frozen at 32 pairs and 4 loader workers per GPU")
    diagnostic_cfg = deepcopy(cfg)
    total_steps = args.warmup_steps + args.timed_steps + args.continuation_steps
    diagnostic_cfg["training"].update(max_steps=total_steps, warmup_steps=args.warmup_steps)
    sampling = select_probe_rows(index, expected_rows=1000000,
        total_pairs=total_steps * 32 * world, seed=int(cfg["training"]["seed"]))
    text_path = Path(cfg["text"]["model_path"]).resolve(strict=True)
    assets = sorted({*text_path.glob("*.json"), *text_path.glob("*.safetensors"), *text_path.glob("*.model"),
                     *text_path.glob("*.txt"), *text_path.glob("*.jinja")})
    if not any(path.suffix == ".safetensors" for path in assets):
        raise ValueError("frozen Qwen weights are absent")
    frontend = cfg["frontend"]
    vae_config = json.loads(Path(frontend["config_path"]).read_text())
    if (vae_config["sample_rate"], vae_config["audio_channels"], vae_config["model"]["latent_dim"],
            vae_config["model"]["downsampling_ratio"]) != (44100, 4, 64, 1024):
        raise ValueError("probe must use the native 44.1kHz FOA frontend")
    if sha256(frontend["checkpoint_path"]) != frontend["checkpoint_sha256"]:
        raise ValueError("native frontend checkpoint identity changed")
    dependencies = [Path(__file__).resolve(), args.config.resolve(),
        ROOT / "scripts/t2a/test/run_sceneplan_transfusion_editing_clap44_gpu_preflight_5gpu.sh",
        ROOT / "scripts/t2a/train/train_sceneplan_transfusion_editing_clap44.py",
        ROOT / "scripts/t2a/train/editing_gpu_runtime.py",
        ROOT / "scripts/t2a/eval/select_sceneplan_transfusion_editing_dit_checkpoint.py",
        *sorted((ROOT / "stable_audio_tools/models").glob("sceneplan_transfusion_editing_clap44*.py")),
        ROOT / "stable_audio_tools/training/sceneplan_transfusion_editing_clap44_pretrain.py",
        ROOT / "stable_audio_tools/data/sceneplan_transfusion_editing_clap44.py",
        ROOT / "stable_audio_tools/data/sceneplan_transfusion_editing.py",
        ROOT / "stable_audio_tools/data/model_sceneplan.py", ROOT / "stable_audio_tools/models/conditioners.py"]
    return {"schema": SCHEMA, "config": diagnostic_cfg, "formal_config": cfg,
        "schedule_difference": "short diagnostic warmup/cosine schedule; same forward, backward and optimizer implementation",
        "world_size": world, "gpu_topology": topology, "launcher_lock": lock,
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "autocast_dtype": "bfloat16",
            "optimizer_dtype": "float32", "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "train_index_path": str(index), "train_index_sha256": dataset.index_sha256,
        "preflight_sha256": sha256(args.preflight), "sampling": sampling,
        "source_sha256": {str(path): sha256(path) for path in dependencies},
        "text_files": {str(path): sha256(path) for path in assets},
        "frontend_files": {frontend["config_path"]: sha256(frontend["config_path"]), frontend["checkpoint_path"]: frontend["checkpoint_sha256"]},
        "warmup_steps": args.warmup_steps, "timed_steps": args.timed_steps,
        "continuation_steps": args.continuation_steps, "max_wall_seconds": args.max_wall_seconds,
        "m2d_used": False, "independent_test_opened": False, "quality_gate_passed": False,
        "timing_note": "synchronized per-step device events and wall time including data wait; phase events may include host/communication gaps"}


def run(args):
    # Resource ownership and explicit UUID mapping are verified before CUDA.
    from scripts.t2a.train import editing_gpu_runtime as runtime
    from scripts.t2a.eval.select_sceneplan_transfusion_editing_dit_checkpoint import _rank0_audit
    started = time.monotonic()
    def budget():
        if time.monotonic() - started >= args.max_wall_seconds:
            raise TimeoutError("CLAP44 probe exhausted its wall-time budget")
    rank, local_rank, world, device, topology = runtime.distributed(
        timeout_seconds=min(300, args.max_wall_seconds))
    lock = runtime.verify_launcher_leases(topology)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the formal BF16 CLAP44 path is unsupported on this GPU")
    monitor = monitor_handle = None
    try:
        def prepare():
            if (args.output / "TRAIN_CONTRACT.json").exists() or (args.output / "RESULT.json").exists():
                raise RuntimeError("GPU probes require a fresh attempt directory")
            contract = audit_contract(args, topology, lock)
            write_json(args.output / "TRAIN_CONTRACT.json", contract)
            return contract
        contract = _rank0_audit(prepare, rank=rank, device=device)
        budget()
        cfg = contract["config"]
        training = cfg["training"]
        seed = int(training["seed"]) + rank
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        rows = contract["sampling"]["rows"][rank::world]
        dataset = EditingCLAP44Dataset(args.index, expected_rows=1000000, row_ordinals=[row["ordinal"] for row in rows])
        model_cfg = CLAP44Config(**cfg["model"])
        model = DistributedDataParallel(EditingCLAP44(model_cfg).to(device), device_ids=[local_rank])
        optimizer, scheduler = optimizer_and_scheduler(model, training)
        text = FrozenCLAP44TextFeatures(cfg["text"]["model_path"], hidden_dim=model_cfg.text_dim,
            max_tokens=int(cfg["text"]["max_tokens"]), batch_size=int(cfg["text"]["batch_size"])).eval()
        loader = make_loader(dataset, training, rank)
        iterator = iter(loader)
        if rank == 0:
            monitor_handle = (args.output / "gpu-activity.csv").open("w")
            monitor = subprocess.Popen(["nvidia-smi", "-i", ",".join(map(str, topology["physical_indices"])),
                "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits", "--loop-ms=1000"], stdout=monitor_handle, stderr=subprocess.STDOUT)
        checkpoint_step = args.warmup_steps + args.timed_steps
        timed_records = []
        def advance(batch, step, timed=False):
            timer = CudaPhases() if timed else None
            losses = training_step(model, text, optimizer, scheduler, batch, step=step,
                training=training, device=device, phase_context=timer)
            torch.cuda.synchronize(device)
            return {key: float(value) for key, value in losses.items()}, {} if timer is None else timer.milliseconds()
        for step in range(args.warmup_steps):
            budget()
            advance(next(iterator), step)
        dist.barrier()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        timed_started_unix = time.time()
        timed_started = time.monotonic()
        for step in range(args.warmup_steps, checkpoint_step):
            budget()
            before = time.monotonic()
            batch = next(iterator)
            data_wait = time.monotonic() - before
            losses, phases = advance(batch, step, timed=True)
            record = {"step": step + 1, "data_wait_seconds": data_wait,
                "wall_seconds": time.monotonic() - before, "device_phase_milliseconds": phases, "losses": losses}
            timed_records.append(record)
            with (args.output / f"rank-{rank}-steps.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
        timed_wall = time.monotonic() - timed_started
        peak = {"allocated_bytes": torch.cuda.max_memory_allocated(device), "reserved_bytes": torch.cuda.max_memory_reserved(device)}
        states = [None] * world
        dist.all_gather_object(states, rng_state())
        before = time.monotonic()
        manifest = _rank0_audit(lambda: save_checkpoint(args.output / f"step-{checkpoint_step:06d}.pt", model.module,
            optimizer, scheduler, checkpoint_step, 0, checkpoint_step, contract, states), rank=rank, device=device)
        write_seconds = time.monotonic() - before
        expected_losses, expected_batches = [], []
        for step in range(checkpoint_step, checkpoint_step + args.continuation_steps):
            budget()
            batch = next(iterator)
            expected_batches.append(batch_digest(batch))
            expected_losses.append(advance(batch, step)[0])
        expected = {"model": cpu_copy(model.module.state_dict()), "optimizer": cpu_copy(optimizer.state_dict()),
                    "scheduler": cpu_copy(scheduler.state_dict()), "rng": rng_state(), "losses": expected_losses}
        del iterator, loader, model, optimizer, scheduler, batch
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
        before = time.monotonic()
        payload, restored_manifest = load_training_checkpoint(manifest["checkpoint"], contract=contract)
        if manifest != restored_manifest:
            raise RuntimeError("saved probe checkpoint changed")
        read_seconds = time.monotonic() - before
        before = time.monotonic()
        model = DistributedDataParallel(EditingCLAP44(model_cfg).to(device), device_ids=[local_rank])
        optimizer, scheduler = optimizer_and_scheduler(model, training)
        model.module.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        saved_rng = payload["rng_states"][rank]
        del payload
        torch.cuda.synchronize(device)
        restore_seconds = time.monotonic() - before
        loader = make_loader(dataset, training, rank)
        iterator = iter(loader)
        for _ in range(checkpoint_step):
            budget()
            next(iterator)
        actual_losses = []
        for offset, step in enumerate(range(checkpoint_step, checkpoint_step + args.continuation_steps)):
            budget()
            batch = next(iterator)
            if batch_digest(batch) != expected_batches[offset]:
                raise RuntimeError("resumed loader did not reproduce identical per-rank batches")
            if offset == 0:
                restore_rng(saved_rng)
            actual_losses.append(advance(batch, step)[0])
        actual = {"model": model.module.state_dict(), "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict(), "rng": rng_state(), "losses": actual_losses}
        comparison = {key: compare_state(expected[key], actual[key]) for key in expected}
        result = {"rank": rank, "physical_gpu": topology["physical_indices"][local_rank], "timed_steps": len(timed_records),
            "timed_wall_seconds": timed_wall, "timed_started_unix": timed_started_unix, "timed_step_records": timed_records,
            "timed_peak_memory": peak, "checkpoint_write_and_broadcast_seconds": write_seconds,
            "checkpoint_read_seconds": read_seconds, "model_optimizer_restore_seconds": restore_seconds,
            "resume_batches_sha256": expected_batches, "resume_comparison": comparison,
            "elapsed_seconds": time.monotonic() - started, "quality_gate_passed": False}
        write_json(args.output / f"rank-{rank}-result.json", result)
        records = [None] * world
        dist.all_gather_object(records, result)
        budget()
        def publish():
            report = summarize_ranks(records, timed_steps=args.timed_steps, pairs_per_gpu=32, world_size=world)
            report.update(schema=SCHEMA, contract_sha256=sha256(args.output / "TRAIN_CONTRACT.json"),
                checkpoint=manifest, gpu_activity_sampler_exit_code=monitor.poll(),
                independent_test_opened=False, next="Review throughput, memory and recovery differences before a separate training run.")
            write_json(args.output / "RESULT.json", report)
            print("CLAP44_GPU_PROBE=" + json.dumps({key: report[key] for key in
                ("status", "global_pairs_per_second", "resume_exact_all_ranks", "quality_gate_passed")}), flush=True)
            return True
        _rank0_audit(publish, rank=rank, device=device)
        return 0 if all(all(value["exact"] for value in row["resume_comparison"].values()) for row in records) else 3
    except Exception as exc:
        write_json(args.output / f"rank-{rank}-failure.json", {"schema": SCHEMA, "status": "PROBE_FAILED",
            "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started, "quality_gate_passed": False})
        raise
    finally:
        if monitor is not None:
            if monitor.poll() is None:
                monitor.terminate()
                try:
                    monitor.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    monitor.kill(); monitor.wait(timeout=5)
            monitor_handle.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DATA_ROOT / "training_index/train.sqlite")
    parser.add_argument("--preflight", type=Path, default=DATA_ROOT / "contracts/full_training/PREFLIGHT.json")
    parser.add_argument("--config", type=Path, default=ROOT / "stable_audio_tools/configs/model_configs/txt2audio/t2a/editing_clap44_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--timed-steps", type=int, default=8)
    parser.add_argument("--continuation-steps", type=int, default=2)
    parser.add_argument("--max-wall-seconds", type=int, default=900)
    args = parser.parse_args()
    if (not 60 <= args.max_wall_seconds <= 900 or not 2 <= args.warmup_steps <= 4 or
            not 2 <= args.timed_steps <= 8 or args.continuation_steps != 2):
        parser.error("probe is bounded to 60-900s, 2-4 warmup, 2-8 timed and exactly 2 continuation steps")
    if not args.output.is_dir():
        parser.error("the launcher must create a fresh attempt directory first")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
