#!/usr/bin/env python3
"""Standalone CLAP44 pretraining. It never launches/restarts DiT or AR jobs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from scripts.t2a.train.editing_gpu_runtime import distributed

from stable_audio_tools.data.sceneplan_transfusion_editing_clap44 import CLAP44_DATA_CONTRACT, EditingCLAP44Dataset, collate_clap44
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import CLAP44_CONTRACT, CLAP44Config, EditingCLAP44, binding_negative_loss, clap44_objective
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_text import FrozenCLAP44TextFeatures
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_pretrain import (
    load_training_checkpoint, optimizer_and_scheduler, resolve_training_resume, save_checkpoint, training_step,
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path); tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def rng_state():
    name, keys, position, has_gauss, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": {"name": name, "keys": torch.from_numpy(keys.astype(np.int64)), "position": position, "has_gauss": has_gauss, "cached": cached}, "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value["name"], value["keys"].numpy().astype(np.uint32), value["position"], value["has_gauss"], value["cached"]))
    torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state(state["cuda"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    rank, local_rank, world, device, topology = distributed()
    from scripts.t2a.eval.select_sceneplan_transfusion_editing_dit_checkpoint import _rank0_audit
    cfg = json.loads(args.config.read_text()); model_cfg = CLAP44Config(**cfg["model"])
    training = cfg["training"]
    for key in ("pairs_per_gpu", "max_steps", "warmup_steps", "checkpoint_every", "log_every"):
        if int(training[key]) < 1: raise ValueError(f"CLAP44 {key} must be positive")
    if int(training["workers_per_gpu"]) < 0 or not 0 < int(training["warmup_steps"]) < int(training["max_steps"]):
        raise ValueError("invalid CLAP44 worker count/warmup schedule")
    if not all(math.isfinite(float(training[k])) for k in ("learning_rate", "scene_weight", "binding_weight")) or float(training["learning_rate"]) <= 0 or float(training["scene_weight"]) <= 0 or float(training["binding_weight"]) < 0:
        raise ValueError("invalid CLAP44 optimizer/loss settings")
    seed = int(training["seed"])
    random.seed(seed + rank); np.random.seed(seed + rank); torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("status") != "PASS" or preflight["indices"]["train"]["rows"] != 1000000 or preflight["indices"]["validation"]["rows"] != 20000 or preflight["indices"]["test"]["rows"] != 5000:
        raise ValueError("CLAP44 must reuse the complete approved 1M/20k/5k data partition")
    index = args.index.resolve(strict=True)
    if str(index) != preflight["indices"]["train"]["path"]:
        raise ValueError("CLAP44 training cannot use a held-out or foreign index")
    # Hash once on rank 0; broadcast the outcome so a failed check cannot
    # leave the other ranks entering model training.
    validation = _rank0_audit(lambda:sha256(index)==preflight["indices"]["train"]["sha256"],rank=rank,device=device)
    if validation is not True:
        raise RuntimeError("CLAP44 train index SHA256 changed")
    dataset = EditingCLAP44Dataset(index, expected_rows=1000000)
    if dataset.index_sha256 != preflight["indices"]["train"]["sha256"]:
        raise RuntimeError("CLAP44 train marker and preflight SHA256 disagree")
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=seed, drop_last=True)
    loader_generator = torch.Generator()
    loader = DataLoader(dataset, batch_size=int(training["pairs_per_gpu"]), sampler=sampler, num_workers=int(training["workers_per_gpu"]), pin_memory=True, drop_last=True, collate_fn=collate_clap44, persistent_workers=int(training["workers_per_gpu"]) > 0,generator=loader_generator,
        multiprocessing_context="spawn" if int(training["workers_per_gpu"])>0 else None)
    text_path = Path(cfg["text"]["model_path"]).resolve(strict=True)
    source_paths = [Path(__file__).resolve()] + [ROOT / x for x in (
        "stable_audio_tools/models/sceneplan_transfusion_editing_clap44.py",
        "stable_audio_tools/models/sceneplan_transfusion_editing_clap44_text.py",
        "stable_audio_tools/models/sceneplan_transfusion_editing_clap44_io.py",
        "stable_audio_tools/training/sceneplan_transfusion_editing_clap44_pretrain.py",
        "scripts/t2a/eval/select_sceneplan_transfusion_editing_dit_checkpoint.py",
        "scripts/t2a/train/editing_gpu_runtime.py",
        "stable_audio_tools/data/sceneplan_transfusion_editing_clap44.py",
        "stable_audio_tools/models/conditioners.py", "stable_audio_tools/data/model_sceneplan.py",
        "stable_audio_tools/data/sceneplan_transfusion_editing.py",
    )]
    contract = {"schema": CLAP44_CONTRACT, "data_contract": CLAP44_DATA_CONTRACT, "config": cfg, "world_size": world, "train_index_sha256": dataset.index_sha256, "preflight_sha256": sha256(args.preflight), "text_files": {}, "source_sha256": {str(x): sha256(x) for x in source_paths}, "input": "clean_foa_latent_44100hz_hop1024", "labels_only": ["source_and_target_descriptions", "source_and_target_bound_scene_descriptions"], "m2d_used": False, "test_split_used_for_training_or_selection": False}
    def audit_frozen_assets():
        assets = sorted({*text_path.glob("*.json"), *text_path.glob("*.safetensors"), *text_path.glob("*.model"), *text_path.glob("*.txt"), *text_path.glob("*.jinja")})
        if not any(x.suffix == ".safetensors" for x in assets):
            raise RuntimeError("local frozen Qwen weights are missing")
        contract["text_files"] = {str(x): sha256(x) for x in assets}
        frontend = cfg["frontend"]
        frontend_config = json.loads(Path(frontend["config_path"]).read_text())
        if (frontend_config["sample_rate"], frontend_config["audio_channels"], frontend_config["model"]["latent_dim"], frontend_config["model"]["downsampling_ratio"]) != (44100, 4, 64, 1024):
            raise RuntimeError("CLAP44 frontend is not the native FOA VAE")
        if sha256(frontend["checkpoint_path"]) != frontend["checkpoint_sha256"]:
            raise RuntimeError("CLAP44 frontend checkpoint identity changed")
        contract["frontend_files"] = {frontend["config_path"]: sha256(frontend["config_path"]), frontend["checkpoint_path"]: frontend["checkpoint_sha256"]}
        contract["gpu_topology"] = topology
        contract["physical_gpus"] = topology["physical_indices"]
        contract["dataloader_rng"] = "dedicated_epoch_generator_seed_plus_900001_plus_epoch_world_plus_rank"
        contract["dataloader_multiprocessing_context"] = "spawn" if int(training["workers_per_gpu"])>0 else None
        return contract
    contract = _rank0_audit(audit_frozen_assets,rank=rank,device=device)
    def audit_output():
        cp = args.output / "TRAIN_CONTRACT.json"
        if cp.exists() and json.loads(cp.read_text()) != contract:
            raise RuntimeError("CLAP44 output belongs to another training contract")
        if not cp.exists(): atomic_json(cp, contract)
        if args.resume is None and any(args.output.glob("step-*.pt")):
            raise RuntimeError("existing CLAP44 candidates require explicit --resume")
        if args.resume is not None:
            latest = resolve_training_resume(args.output)
            if latest is None or Path(latest["checkpoint"]) != args.resume.resolve(strict=True):
                raise RuntimeError("CLAP44 training must resume its verified latest candidate")
        return True
    _rank0_audit(audit_output,rank=rank,device=device)
    model = EditingCLAP44(model_cfg).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])
    text_encoder = FrozenCLAP44TextFeatures(str(text_path), hidden_dim=model_cfg.text_dim, max_tokens=int(cfg["text"]["max_tokens"]), batch_size=int(cfg["text"]["batch_size"])).eval()
    optimizer,scheduler = optimizer_and_scheduler(model,training)
    max_steps = int(training["max_steps"]); warmup = int(training["warmup_steps"])
    step = epoch = next_batch = 0
    resume_rng = None
    if args.resume is not None:
        payload,_ = load_training_checkpoint(args.resume,contract=contract)
        model.module.load_state_dict(payload["model"], strict=True); optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        step, epoch, next_batch = int(payload["step"]), int(payload["epoch"]), int(payload["next_batch"])
        if not 0 <= step <= max_steps or epoch < 0 or not 0 <= next_batch <= len(loader) or len(payload["rng_states"]) != world:
            raise RuntimeError("CLAP44 resume progress/RNG state is invalid")
        resume_rng = payload["rng_states"][rank]
    started = time.monotonic()
    while step < max_steps:
        sampler.set_epoch(epoch)
        loader_generator.manual_seed(seed+900001+epoch*world+rank)
        for batch_index, batch in enumerate(loader):
            if batch_index < next_batch: continue
            if resume_rng is not None: restore_rng(resume_rng); resume_rng = None
            losses = training_step(model,text_encoder,optimizer,scheduler,batch,step=step,training=training,device=device)
            step += 1
            if rank == 0 and step % int(training["log_every"]) == 0:
                metrics = {"step": step, "epoch": epoch, **{key:float(value) for key,value in losses.items()}, "elapsed_sec": time.monotonic() - started}
                print("CLAP44_TRAIN=" + json.dumps(metrics), flush=True)
                with (args.output / "metrics.jsonl").open("a") as f: f.write(json.dumps(metrics) + "\n")
            if step % int(training["checkpoint_every"]) == 0 or step == max_steps:
                states = [None] * world; dist.all_gather_object(states, rng_state())
                _rank0_audit(lambda:save_checkpoint(args.output / f"step-{step:06d}.pt", model.module, optimizer, scheduler, step, epoch, batch_index + 1, contract, states),rank=rank,device=device)
            if step >= max_steps: break
        epoch += 1; next_batch = 0
    if rank == 0:
        atomic_json(args.output / "FIT_COMPLETE.json", {"status": "FIT_COMPLETE_NOT_QUALITY_PASS", "steps": step, "contract_sha256": sha256(args.output / "TRAIN_CONTRACT.json"), "next": "Frozen validation retrieval, binding discrimination, then matched AR ablations; no automatic quality promotion."})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
