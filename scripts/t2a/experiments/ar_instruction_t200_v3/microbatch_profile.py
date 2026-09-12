"""Compare real three-rank AR microbatches at the same effective batch.

This is a bounded forward/backward profile with resident native Adam state.
It does not update weights or establish AR quality or checkpoint recovery.
The existing factual-encoder loader remains the shared dependency entry.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scripts.t2a.experiments.ar_factual_clap_v1 import integration
from scripts.t2a.experiments.ar_instruction_t200_v1 import runtime as original_runtime
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint
from scripts.t2a.experiments.ar_instruction_t200_v1.instruction_data import DatasetWithInstructions, InstructionOverlay
from scripts.t2a.experiments.ar_instruction_t200_v2 import ar_only
from scripts.t2a.experiments.ar_source_grounding_v1 import allocated_runtime
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, state_hash, write
from scripts.t2a.experiments.ar_source_grounding_v1.numerics import install_autotune_observer
from scripts.t2a.train import train_sceneplan_transfusion_editing_ar_clap44 as native


def now():
    return datetime.now().astimezone().isoformat()


def gather(value, world):
    values = [None] * world
    dist.all_gather_object(values, value)
    return values


def gradient_comparison(current, reference):
    dot = reference_square = difference_square = current_square = 0.0
    for name, value in current.items():
        left, right = value.float(), reference[name].float()
        dot += float((left * right).sum(dtype=torch.float64))
        reference_square += float(right.square().sum(dtype=torch.float64))
        current_square += float(left.square().sum(dtype=torch.float64))
        difference_square += float((left - right).square().sum(dtype=torch.float64))
    return {
        "cosine": dot / max(math.sqrt(reference_square * current_square), 1e-30),
        "relative_L2": math.sqrt(difference_square / max(reference_square, 1e-30)),
    }


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    if plan["physical_gpus"] != [5, 6, 7] or os.environ.get("EDITING_GPUS") != "5,6,7":
        raise RuntimeError("profile owns GPU5–7 only")
    if plan["optimizer_updates"] != 0 or plan["independent_test_used"] is not False:
        raise RuntimeError("profile is not a training or test-set launcher")
    if plan["profiles"] != [[8, 5, 4], [16, 10, 2], [32, 20, 1]]:
        raise RuntimeError("profile must preserve effective per-rank batches32/20")
    for path, expected in plan["source_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError(f"bound input changed: {path}")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rank, local_rank, world, device, topology = allocated_runtime.distributed(timeout_seconds=900)
    if world != 3:
        raise RuntimeError("three ranks required")
    out = plan_path.parent / f"rank{rank}"
    out.mkdir(exist_ok=False)
    cfg = json.loads(Path(plan["configuration"]).read_text())
    finish_numerics = install_autotune_observer({"name": f"rank{rank}", "mode": "capture"}, plan_path.parent)
    flash_calls, restore_flash = original_runtime.install_flash(cfg["numerical_execution"])
    write(out / "STARTED.json", {"at": now(), "pid": os.getpid(), "physical_gpu": topology["physical_indices"][rank]})
    try:
        native._seed_everything(plan["seed"], rank)
        module, codec, parent, encoder_identity, transfer = integration.build_from_ar_parent(
            plan["parent"], plan["clap"], preflight_path=plan["preflight"])
        groups, scope = ar_only.configure(module, cfg)
        if scope != json.loads(Path(plan["optimizer_scope"]).read_text()):
            raise RuntimeError("AR optimizer parameter scope changed")
        module.to(device).train()
        optimizer = torch.optim.AdamW(groups, betas=(.9, .95), weight_decay=cfg["weight_decay"], fused=True)
        optimizer.load_state_dict(parent["optimizer"])
        optimizer_before = fingerprint(optimizer.state_dict())["sha256"]
        if optimizer_before != fingerprint(parent["optimizer"])["sha256"]:
            raise RuntimeError("native Adam state differs")
        del parent
        gc.collect()
        parameters_before = fingerprint(dict(module.named_parameters()))["sha256"]
        qwen_before = state_hash(module.ar.instruction_conditioner.model)
        clap_before = state_hash(module.ar.source_clap_model)
        if qwen_before != plan["qwen_state_sha256"]:
            raise RuntimeError("Qwen state changed")
        if (module.ar.source_clap_model.training
                or any(p.requires_grad for p in module.ar.source_clap_model.parameters())
                or any(p.requires_grad for p in module.ar.instruction_conditioner.model.parameters())):
            raise RuntimeError("CLAP and Qwen must remain frozen")
        write(out / "DEPENDENCIES_LOADED.json", {"at": now(), "encoder": encoder_identity["sha256"],
            "existing_loader": integration.__file__, "optimizer_state_sha256": optimizer_before,
            "source_features_from_audio_only": True})
        raw_index = cfg["instruction_data"]["overlays"]["train"]
        tokenizer = module.ar.instruction_conditioner.tokenizer
        base = native.ScenePlanTransfusionEditingDataset(raw_index["native_index_path"],
            tokenizer_spec=(tokenizer, 512), expected_num_samples=1000000,
            expected_index_sha256=raw_index["native_index_sha256"], latent_crop_length=648,
            require_frozen=True, verify_tensor_hashes_on_access=True)
        overlay = InstructionOverlay(raw_index["path"], expected_sha256=raw_index["sha256"],
            native_index_path=raw_index["native_index_path"], native_index_sha256=raw_index["native_index_sha256"],
            expected_rows=1000000, split="train")
        dataset = DatasetWithInstructions(native.ScenePlanTransfusionEditingJointDataset(base, codec=codec), overlay, joint=True)
        sampler = native.DistributedScenePlanBucketBatchSampler(dataset, short_batch_size=32, long_batch_size=20,
            num_replicas=world, rank=rank, shuffle=True, seed=plan["seed"], drop_last=True)
        samples = {}
        for position, indices in enumerate(sampler):
            bucket = 432 if len(indices) == 32 else 648
            if bucket not in samples:
                start = time.perf_counter()
                rows = [dataset[i] for i in indices]
                samples[bucket] = (rows, {"indices": indices, "sampler_epoch": 0,
                    "sampler_batch_position": position, "CPU_read_seconds": time.perf_counter() - start})
            if len(samples) == 2:
                break
        if set(samples) != {432, 648}:
            raise RuntimeError("both length buckets are required")
        wrapped = DistributedDataParallel(module, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)
        trainable = {n: p for n, p in module.named_parameters() if p.requires_grad}
        results = []
        for bucket in (432, 648):
            rows, population = samples[bucket]
            all_indices = gather(population["indices"], world)
            global_pairs = sum(map(len, all_indices))
            if len(set(sum(all_indices, []))) != global_pairs:
                raise RuntimeError("rank sample slices overlap")
            reference_gradient = None
            reference_CE = None
            reference_tokens = None
            for short_batch, long_batch, accumulation in plan["profiles"]:
                batch_size = short_batch if bucket == 432 else long_batch
                batches = [native.collate_editing_joint(rows[i:i + batch_size], pad_id=codec.pad_id)
                           for i in range(0, len(rows), batch_size)]
                if len(batches) != accumulation:
                    raise RuntimeError("effective batch changed")
                count = torch.tensor(sum(int((b["ar"]["plan_labels"] != -100).sum()) for b in batches),
                                     dtype=torch.float64, device=device)
                dist.all_reduce(count)
                if reference_tokens is None:
                    reference_tokens = int(count)
                if int(count) != reference_tokens:
                    raise RuntimeError("profile target-token population changed")
                repeats = []
                for repeat in range(plan["warmup_repeats"] + plan["measured_repeats"]):
                    optimizer.zero_grad(set_to_none=True)
                    loss_sums = torch.zeros((), dtype=torch.float64, device=device)
                    dist.barrier(); torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    for i, batch in enumerate(batches):
                        ar = native._move_joint_batch(batch, device)[0]
                        if ar["source_foa_latent"].shape[-1] != bucket:
                            raise RuntimeError("source prefix length changed")
                        sync = wrapped.no_sync() if i + 1 < len(batches) else nullcontext()
                        with sync:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                logits = wrapped(source_foa_latent=ar["source_foa_latent"],
                                    source_attention_mask=ar["source_attention_mask"],
                                    plan_input_ids=ar["plan_input_ids"], plan_attention_mask=ar["plan_attention_mask"],
                                    raw_edit_requests=ar["raw_edit_requests"])
                                numerator = ar_only.ce_sum(logits, ar["plan_labels"])
                                loss = numerator * world / count
                            if not bool(torch.isfinite(loss)):
                                raise RuntimeError("non-finite CE")
                            loss.backward()
                        loss_sums += numerator.detach().double()
                        del ar, logits, numerator, loss
                    torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - start
                    peak = torch.cuda.max_memory_allocated(device)
                    missing = [n for n, p in trainable.items() if p.grad is None]
                    leaked = [n for n, p in module.named_parameters() if not p.requires_grad and p.grad is not None]
                    leaked += ["Qwen." + n for n, p in module.ar.instruction_conditioner.model.named_parameters() if p.grad is not None]
                    if missing or leaked or any(not bool(torch.isfinite(p.grad).all()) for p in trainable.values()):
                        raise RuntimeError(f"invalid gradients: {missing=} {leaked=}")
                    grad = {n: p.grad.detach().cpu().clone() for n, p in trainable.items()}
                    measurements = gather({"rank": rank, "seconds": elapsed, "peak_allocated_bytes": peak,
                        "CE_sum": float(loss_sums), "gradient_sha256": fingerprint(grad)["sha256"]}, world)
                    if len({m["gradient_sha256"] for m in measurements}) != 1:
                        raise RuntimeError("DDP gradients differ between ranks")
                    CE = sum(m["CE_sum"] for m in measurements) / int(count)
                    if reference_gradient is None:
                        reference_gradient = grad
                        reference_CE = CE
                    comparison = gradient_comparison(grad, reference_gradient)
                    numeric_pass = (abs(CE - reference_CE) <= plan["numerical_screen"]["max_absolute_CE_difference"]
                        and comparison["relative_L2"] <= plan["numerical_screen"]["max_gradient_relative_L2"]
                        and comparison["cosine"] >= plan["numerical_screen"]["min_gradient_cosine"])
                    item = {"repeat": repeat, "warmup": repeat < plan["warmup_repeats"],
                        "global_AR_CE": CE, "CE_difference_from_reference": CE - reference_CE,
                        "gradient_comparison": comparison, "numerical_screen_passed": numeric_pass,
                        "rank_measurements": measurements,
                        "global_pairs_per_second": global_pairs / max(m["seconds"] for m in measurements),
                        "global_valid_plan_tokens_per_second": int(count) / max(m["seconds"] for m in measurements)}
                    repeats.append(item)
                    del grad
                result = {"bucket": bucket, "microbatch_per_rank": batch_size, "accumulation": accumulation,
                    "global_pairs": global_pairs, "global_valid_plan_tokens": int(count), "population": population,
                    "all_rank_indices": all_indices, "repeats": repeats,
                    "median_pairs_per_second": statistics.median(x["global_pairs_per_second"] for x in repeats if not x["warmup"]),
                    "numerical_screen_passed": all(x["numerical_screen_passed"] for x in repeats)}
                results.append(result)
                write(out / f"BUCKET{bucket}_BATCH{batch_size}.json", result)
            del reference_gradient
        optimizer.zero_grad(set_to_none=True)
        if (parameters_before != fingerprint(dict(module.named_parameters()))["sha256"]
                or optimizer_before != fingerprint(optimizer.state_dict())["sha256"]
                or qwen_before != state_hash(module.ar.instruction_conditioner.model)
                or clap_before != state_hash(module.ar.source_clap_model)):
            raise RuntimeError("profile changed model/optimizer/Qwen")
        finish_numerics()
        complete = {"at": now(), "status": "PROFILE_COMPLETE", "physical_gpus": [5, 6, 7],
            "rank": rank, "profiles": results, "optimizer_updates": 0,
            "all_parameters_optimizer_and_Qwen_unchanged": True, "flash_calls": flash_calls,
            "timing_scope": "Source/instruction encoding, CPU-to-GPU batch transfer, AR forward/backward and DDP sync. Excludes disk loading, optimizer.step, gradient auditing and checkpoint IO.",
            "same_actual_train_rows_and_effective_batch_across_profiles": True,
            "numerical_screen_is_not_AR_quality_or_exact_resume": True,
            "independent_test_used": False, "quality_gate_passed": False}
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
