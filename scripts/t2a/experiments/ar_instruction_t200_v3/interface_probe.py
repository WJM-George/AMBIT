"""Bounded three-rank AR interface/backward check on real T200 train data.

No optimizer update, checkpoint selection or quality claim is made here.
The same frozen factual encoder and retained AR parent are used on all ranks.
Both native source-length buckets exercise DDP, with native Adam states resident
for a realistic baseline memory measurement. Production resume and larger-batch
throughput remain separate checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scripts.t2a.experiments.ar_instruction_t200_v1 import runtime as original_runtime
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint
from scripts.t2a.experiments.ar_instruction_t200_v1.instruction_data import DatasetWithInstructions, InstructionOverlay
from scripts.t2a.experiments.ar_instruction_t200_v2 import ar_only
from scripts.t2a.experiments.ar_instruction_t200_v3.factual_encoder import load_factual_encoder
from scripts.t2a.experiments.ar_source_grounding_v1 import allocated_runtime
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, state_hash, write
from scripts.t2a.experiments.ar_source_grounding_v1.numerics import install_autotune_observer
from scripts.t2a.train import train_sceneplan_transfusion_editing_ar_clap44 as native


def now():
    return datetime.now().astimezone().isoformat()


def compact_fingerprint(value):
    result = fingerprint(value)
    return {k: v for k, v in result.items() if k != "tree"}


def gather(value, world):
    values = [None] * world
    dist.all_gather_object(values, value)
    return values


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    if plan["physical_gpus"] != [5, 6, 7] or os.environ.get("EDITING_GPUS") != "5,6,7":
        raise RuntimeError("this probe owns only GPU5–7")
    if plan["optimizer_updates"] != 0 or plan["independent_test_used"] is not False:
        raise ValueError("interface probe cannot be used as a training/test launcher")
    for filename, expected in plan["source_sha256"].items():
        if sha(filename) != expected:
            raise RuntimeError(f"probe-bound input changed: {filename}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rank, local_rank, world, device, topology = allocated_runtime.distributed(timeout_seconds=600)
    if world != 3:
        raise RuntimeError("three DDP ranks required")
    out = plan_path.parent / f"rank{rank}"
    out.mkdir(exist_ok=False)
    finish_numerics = install_autotune_observer({"name": f"rank{rank}", "mode": "capture"}, plan_path.parent)
    cfg = json.loads(Path(plan["configuration"]).read_text())
    flash_calls, restore_flash = original_runtime.install_flash(cfg["numerical_execution"])
    write(out / "STARTED.json", {"at": now(), "pid": os.getpid(), "rank": rank,
          "physical_gpu": topology["physical_indices"][rank], "plan_sha256": sha(plan_path)})
    try:
        native._seed_everything(plan["seed"], rank)
        encoder, identity = load_factual_encoder(plan["clap"]["path"],
            expected_sha256=plan["clap"]["sha256"], preflight_path=plan["preflight"])
        parent_payload, parent_identity = native.load_joint_checkpoint(
            plan["parent"]["checkpoint"], verify_sources=True, require_latest=True)
        if parent_identity != plan["parent"]:
            raise RuntimeError("retained AR parent changed")
        codec = native.ModelScenePlanCodecV4(Path(plan["codec"]))
        module = native.build_model(native.load_config(plan["model_config"]), plan["p10_checkpoint"],
            codec, encoder, "global_and_sequence", 0., training_mode="ar_pretrain")
        module.diffusion.load_state_dict(parent_payload["diffusion_state_dict"], strict=True)
        native.load_ar_specific(module.ar, parent_payload["editing_ar_specific_state_dict"])
        groups, scope = ar_only.configure(module, cfg)
        if scope != json.loads(Path(plan["optimizer_scope"]).read_text()):
            raise RuntimeError("native AR-only reachable parameter scope changed")
        module.to(device).train()
        if (module.ar.source_clap_model.training or any(p.requires_grad for p in encoder.parameters())
                or any(p.requires_grad for p in module.ar.instruction_conditioner.model.parameters())):
            raise RuntimeError("CLAP/Qwen must remain frozen")
        optimizer = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=cfg["weight_decay"], fused=True)
        optimizer.load_state_dict(parent_payload["optimizer"])
        optimizer_before = compact_fingerprint(optimizer.state_dict())
        if optimizer_before != compact_fingerprint(parent_payload["optimizer"]):
            raise RuntimeError("native Adam state did not load exactly")
        del parent_payload
        gc.collect()
        qwen_before = state_hash(module.ar.instruction_conditioner.model)
        if qwen_before != plan["qwen_state_sha256"]:
            raise RuntimeError("frozen Qwen differs from its pinned native state")
        parameters_before = compact_fingerprint(dict(module.named_parameters()))
        clap_before = state_hash(encoder)
        write(out / "DEPENDENCIES_LOADED.json", {"at": now(), "clap": identity,
              "parent": parent_identity, "optimizer_loaded_exactly": True,
              "optimizer": optimizer_before, "qwen_state_sha256": qwen_before,
              "trainable_parameter_counts": scope["trainable_parameters"]})
        raw_index = cfg["instruction_data"]["overlays"]["train"]
        tokenizer = module.ar.instruction_conditioner.tokenizer
        base = native.ScenePlanTransfusionEditingDataset(raw_index["native_index_path"],
            tokenizer_spec=(tokenizer, 512), expected_num_samples=1000000,
            expected_index_sha256=raw_index["native_index_sha256"], latent_crop_length=648,
            require_frozen=True, verify_tensor_hashes_on_access=True)
        overlay = InstructionOverlay(raw_index["path"], expected_sha256=raw_index["sha256"],
            native_index_path=raw_index["native_index_path"],
            native_index_sha256=raw_index["native_index_sha256"], expected_rows=1000000, split="train")
        dataset = DatasetWithInstructions(native.ScenePlanTransfusionEditingJointDataset(base, codec=codec), overlay, joint=True)
        sampler = native.DistributedScenePlanBucketBatchSampler(dataset, short_batch_size=8, long_batch_size=5,
            num_replicas=world, rank=rank, shuffle=True, seed=42, drop_last=True)
        batches = {}
        for position, indices in enumerate(sampler):
            bucket = 432 if len(indices) == 8 else 648
            if bucket not in batches:
                start = time.perf_counter()
                raw = native.collate_editing_joint([dataset[i] for i in indices], pad_id=codec.pad_id)
                batches[bucket] = (raw, {"sampler_epoch": 0, "sampler_batch_position": position,
                    "indices": indices, "CPU_read_seconds": time.perf_counter() - start})
            if len(batches) == 2:
                break
        if set(batches) != {432, 648}:
            raise RuntimeError("probe did not cover both native source-length buckets")
        wrapped = DistributedDataParallel(module, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False, gradient_as_bucket_view=True)
        records = []
        for bucket in (432, 648):
            batch, batch_identity = batches[bucket]
            ar = native._move_joint_batch(batch, device)[0]
            if ar["source_foa_latent"].shape[-1] != bucket:
                raise RuntimeError("native source-prefix geometry changed")
            all_rows = gather(batch_identity["indices"], world)
            if len(set(sum(all_rows, []))) != sum(map(len, all_rows)):
                raise RuntimeError("DDP rank slices overlap")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                features = encoder.source_features(ar["source_foa_latent"], ar["source_attention_mask"])
            feature_shapes = {k: list(v.shape) for k, v in features.items() if isinstance(v, torch.Tensor)}
            if any(v.requires_grad for v in features.values() if isinstance(v, torch.Tensor)):
                raise RuntimeError("frozen source feature leaked gradients")
            for repeat in range(2):
                optimizer.zero_grad(set_to_none=True)
                count = (ar["plan_labels"] != -100).sum().to(torch.float64)
                dist.all_reduce(count)
                torch.cuda.synchronize(device); dist.barrier()
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = wrapped(source_foa_latent=ar["source_foa_latent"],
                        source_attention_mask=ar["source_attention_mask"],
                        plan_input_ids=ar["plan_input_ids"], plan_attention_mask=ar["plan_attention_mask"],
                        raw_edit_requests=ar["raw_edit_requests"])
                    numerator = ar_only.ce_sum(logits, ar["plan_labels"])
                    loss = numerator * world / count
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("non-finite native AR cross entropy")
                loss.backward()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                peak = torch.cuda.max_memory_allocated(device)
                trainable = {n: p for n, p in module.named_parameters() if p.requires_grad}
                missing = [n for n, p in trainable.items() if p.grad is None]
                leaked = [n for n, p in module.named_parameters() if not p.requires_grad and p.grad is not None]
                if missing or leaked or any(p.grad is not None for p in module.ar.instruction_conditioner.model.parameters()):
                    raise RuntimeError(f"invalid native AR gradient scope: {missing=} {leaked=}")
                if any(not bool(torch.isfinite(p.grad).all()) for p in trainable.values()):
                    raise RuntimeError("non-finite native AR gradient")
                gradient = compact_fingerprint({n: p.grad for n, p in trainable.items()})
                ranks = gather({"rank": rank, "gradient_sha256": gradient["sha256"],
                    "seconds": elapsed, "peak_allocated_bytes": peak, "CE_sum": float(numerator.detach())}, world)
                if len({r["gradient_sha256"] for r in ranks}) != 1:
                    raise RuntimeError("DDP synchronized gradients differ across ranks")
                record = {"bucket": bucket, "repeat": repeat, "warmup": repeat == 0,
                    "batch": batch_identity, "global_pairs": sum(map(len, all_rows)),
                    "global_valid_plan_tokens": int(count), "source_feature_shapes": feature_shapes,
                    "rank_measurements": ranks, "all_rank_gradients_exactly_equal": True,
                    "every_trainable_gradient_present_and_finite": True, "frozen_gradients_absent": True}
                record["global_AR_CE"] = sum(r["CE_sum"] for r in ranks) / int(count)
                record["forward_backward_pairs_per_second"] = record["global_pairs"] / max(r["seconds"] for r in ranks)
                records.append(record)
                write(out / f"BUCKET{bucket}_REPEAT{repeat}.json", record)
                del logits, numerator, loss
        optimizer.zero_grad(set_to_none=True)
        if (parameters_before != compact_fingerprint(dict(module.named_parameters()))
                or optimizer_before != compact_fingerprint(optimizer.state_dict())
                or qwen_before != state_hash(module.ar.instruction_conditioner.model)
                or clap_before != state_hash(encoder)):
            raise RuntimeError("interface-only probe modified parameters or optimizer")
        finish_numerics()
        complete = {"at": now(), "status": "THREE_RANK_INTERFACE_BACKWARD_PASS",
            "physical_gpus": [5, 6, 7], "rank": rank, "clap_checkpoint_sha256": identity["sha256"],
            "source_FOA_and_T200_instruction_are_only_model_conditions": True,
            "GT_target_plan_used_only_for_teacher_forcing_and_loss": True,
            "source_old_plan_not_a_model_input": True, "optimizer_updates": 0,
            "all_parameters_optimizer_and_frozen_Qwen_unchanged": True,
            "native_flash_backward_deterministic": True, "flash_calls": flash_calls,
            "DDP_backward_verified": True, "native_resume_verified": False,
            "short_adaptation_quality_verified": False, "full_AR_expansion_allowed": False,
            "independent_test_used": False, "records": records,
            "timing_scope": "Real three-rank forward/backward baseline, Adam states resident; excludes optimizer.step and timed data loading. Repeated same-batch probe is not training throughput or quality."}
        write(out / "COMPLETE.json", complete)
        gather({"rank": rank, "complete_sha256": sha(out / "COMPLETE.json")}, world)
        if rank == 0:
            write(plan_path.parent / "COMPLETE.json", {**complete, "rank": "all",
                "rank_completions": {str(i): sha(plan_path.parent / f"rank{i}" / "COMPLETE.json") for i in range(world)}})
            print(json.dumps({"status": complete["status"], "optimizer_updates": 0}), flush=True)
    except BaseException:
        write(out / "FAILURE.json", {"at": now(), "error": traceback.format_exc(), "quality_gate_passed": False})
        raise
    finally:
        restore_flash()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    run(parser.parse_args().plan.resolve(strict=True))
