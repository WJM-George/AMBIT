#!/usr/bin/env python3
"""P10 AR planning pretraining or quality-gated native CLAP44 joint training.

The same loop supports bounded matched pilots and the subsequent full run.
Formal promotion/selection remains separate from FIT_COMPLETE.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import functools
import itertools
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from stable_audio_tools.configuration import load_config
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.data.sceneplan_bucket_sampler import DistributedScenePlanBucketBatchSampler
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import ScenePlanTransfusionEditingDataset
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import ScenePlanTransfusionEditingJointDataset, collate_editing_joint
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import ScenePlanTransfusionEditingAR
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_io import file_sha256, load_clap44_checkpoint
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import JOINT44_RUN_SCHEMA, AR_PRETRAIN_RUN_SCHEMA, atomic_json, ensure_run_identity, load_ar_specific, load_joint_checkpoint, save_joint_checkpoint, codec_artifact_sha256
from stable_audio_tools.models.sceneplan_transfusion_editing_provenance import JOINT_TRAINING_SOURCE_PATHS, verify_frozen_qwen_runtime
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.factory import create_training_wrapper_from_config
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_joint import CLAP44_AR_VARIANTS, CLAP44JointEditingModule, CLAP44ARPretrainModule, normalized_joint_loss, optimizer_groups, source_feature_distillation, validation_metrics
from scripts.t2a.train.editing_gpu_runtime import distributed as _distributed
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import DEFAULT_CODEC, DEFAULT_MODEL_CONFIG, _checkpoint_selection_summary, _copy_ema_to_online, _gather_rank_rng_states, _index_summary, _move_joint_batch, _restore_rank_rng_state, _seed_everything, _stratified_validation_batches, _losses


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--variant",choices=sorted(CLAP44_AR_VARIANTS),default="global_and_sequence")
    parser.add_argument("--training-mode", choices=("joint", "ar_pretrain"), default="joint")
    parser.add_argument("--p10-checkpoint", type=Path)
    parser.add_argument("--preflight",type=Path,required=True)
    parser.add_argument("--dit-selection",type=Path)
    parser.add_argument("--dit-gt-audio-gate",type=Path)
    parser.add_argument("--clap-checkpoint",type=Path)
    parser.add_argument("--clap-validation-report",type=Path)
    parser.add_argument("--model-config",type=Path,default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--codec",type=Path,default=DEFAULT_CODEC)
    parser.add_argument("--resume",type=Path)
    args = parser.parse_args(argv)
    if args.training_mode == "joint" and (args.dit_selection is None or args.dit_gt_audio_gate is None):
        parser.error("joint training requires --dit-selection and --dit-gt-audio-gate")
    if args.training_mode == "ar_pretrain" and args.p10_checkpoint is None:
        parser.error("ar_pretrain requires --p10-checkpoint")
    if args.training_mode == "ar_pretrain" and (args.dit_selection is not None or args.dit_gt_audio_gate is not None):
        parser.error("AR pretraining uses a P10 base, not a qualified Editing DiT selection")
    if args.variant != "latent_only" and (args.clap_checkpoint is None or args.clap_validation_report is None):
        parser.error("CLAP feature variants require an encoder checkpoint and its validation report")
    if (args.clap_checkpoint is None) != (args.clap_validation_report is None):
        parser.error("provide both CLAP checkpoint and validation report, or neither for latent_only")
    return args


def validate_config(value, training_mode="joint"):
    schedule = value["schedule"]
    for key in ("max_steps","warmup_steps","save_every","validate_every","log_every","gradient_accumulation","short_batch_size","long_batch_size","validation_batches","validation_batch_size"):
        if int(schedule[key]) < 1: raise ValueError(f"CLAP44 joint {key} must be positive")
    if schedule["warmup_steps"] >= schedule["max_steps"] or schedule["validation_batches"] < 10 or schedule["validation_batch_size"] < 2 or schedule["num_workers"] < 0:
        raise ValueError("CLAP44 joint schedule does not support warmup/all validation strata")
    for key in ("ar_lr","shared_lr","dit_lr","lambda_ar","lambda_rf","gradient_clip"):
        if not math.isfinite(float(value[key])) or value[key] <= 0: raise ValueError(f"invalid CLAP44 joint {key}")
    for key in ("weight_decay","lambda_source_distillation","scene_distillation_weight"):
        if not math.isfinite(float(value[key])) or value[key] < 0: raise ValueError(f"invalid CLAP44 joint {key}")
    if not 0 <= value["source_feature_dropout"] < 1:
        raise ValueError("invalid CLAP44 joint source feature dropout")
    if training_mode == "ar_pretrain":
        if value["purpose"] != "AR_planning_pretraining":
            raise ValueError("AR pretraining requires its own planning configuration")
        return
    if value["purpose"] not in {"matched_AR_pilot","full_AR_training"}:
        raise ValueError("CLAP44 joint purpose must be explicit")
    if value["purpose"] == "matched_AR_pilot" and schedule["max_steps"] > 1000:
        raise ValueError("CLAP44 pilot budget exceeds 1000 updates")
    if value["purpose"] == "full_AR_training" and (schedule["max_steps"] != 25000 or schedule["save_every"] != 5000):
        raise ValueError("CLAP44 full run must retain all five 5k candidates through 25k")


def audit_clap_validation(path,checkpoint,preflight,*,require_full=True):
    path = Path(path).resolve(strict=True); report = json.loads(path.read_text())
    if report.get("status") != "DIAGNOSTICS_COMPLETE_NOT_QUALITY_PASS" or report.get("checkpoint",{}).get("sha256") != checkpoint["sha256"] or report.get("independent_test_used") is not False or report.get("validation_index_sha256") != preflight["indices"]["validation"]["sha256"]:
        raise RuntimeError("CLAP44 validation report has different checkpoint/data provenance")
    manifest = path.parent/"VALIDATION_MANIFEST.json"
    if file_sha256(manifest) != report["validation_manifest_sha256"]:
        raise RuntimeError("CLAP44 validation population changed")
    population = json.loads(manifest.read_text())
    ordinals = population["pair_ordinals"]
    if len(set(ordinals)) != len(ordinals) or len(ordinals) != report["pairs"] or report["audio_views"] != 2*len(ordinals) or population["validation_index_sha256"] != report["validation_index_sha256"]:
        raise RuntimeError("CLAP44 validation population is incomplete")
    if require_full and (report.get("full_validation") is not True or sorted(ordinals) != list(range(20000))):
        raise RuntimeError("CLAP44 AR training requires full 20k encoder diagnostics")
    for head in ("semantic","scene"):
        for direction in ("audio_to_text","text_to_audio"):
            values = report["retrieval"][head][direction]
            if values["queries"] != 2*len(ordinals) or values["candidates"] != 2*len(ordinals) or not all(math.isfinite(float(values[key])) and 0 <= values[key] <= 1 for key in ("r_at_1","r_at_5","r_at_10","mrr")):
                raise RuntimeError("CLAP44 retrieval diagnostics are invalid")
    return {"path":str(path),"sha256":file_sha256(path),"validation_manifest_sha256":report["validation_manifest_sha256"],"pairs":report["pairs"],"quality_gate_passed":False}


def build_model(model_config,base_checkpoint,codec,clap,variant,feature_dropout,training_mode="joint"):
    diffusion = create_model_from_config(model_config)
    state,metadata = load_ckpt_state_dict(str(base_checkpoint),return_metadata=True)
    if training_mode == "ar_pretrain":
        source_config = metadata.get("model_config")
        if not isinstance(source_config, dict):
            raise RuntimeError("P10 checkpoint must contain its source model configuration")
        inheritance = diffusion.load_pretrained_route_state_dict(state, prefer_ema=True,
            source_model_config=source_config,
            source_conditioner_ema_names=metadata.get("conditioner_ema_parameter_names"))
        if inheritance["missing"] or inheritance["modality_mapping"] != "exact_name_and_shape_plus_trained_prefix_input_expansion":
            raise RuntimeError("P10 AR initialization did not preserve the trained backbone")
    else:
        wrapper = create_training_wrapper_from_config(model_config,diffusion)
        expected = list(wrapper.conditioner_ema.parameter_names if wrapper.conditioner_ema is not None else ())
        if list(metadata.get("conditioner_ema_parameter_names") or []) != expected:
            raise RuntimeError("selected DiT conditioner EMA mapping changed")
        wrapper.load_state_dict(state,strict=True)
        _copy_ema_to_online(wrapper)
        diffusion = wrapper.diffusion
        wrapper.diffusion_ema = None; wrapper.conditioner_ema = None
    del state
    diffusion.pretransform = None
    ar = ScenePlanTransfusionEditingAR(editing_dit=diffusion.model.model,instruction_conditioner=diffusion.conditioner.conditioners["prompt"],pad_id=codec.pad_id,source_semantic_mode="latent_only" if variant=="latent_only" else "clap44_audio_caption_aux",source_semantic_dropout=0. if variant=="latent_only" else feature_dropout,source_clap_model=None if variant=="latent_only" else clap,clap44_global_features=variant in {"global_only","global_and_sequence"},clap44_sequence_features=variant in {"sequence_only","global_and_sequence"})
    cls = CLAP44ARPretrainModule if training_mode == "ar_pretrain" else CLAP44JointEditingModule
    return cls(diffusion=diffusion,ar=ar)


def run_source_inventory():
    paths = {ROOT/p for p in JOINT_TRAINING_SOURCE_PATHS}
    paths.update(ROOT.glob("stable_audio_tools/models/sceneplan_transfusion_editing_clap44*.py"))
    paths.update(ROOT.glob("stable_audio_tools/training/sceneplan_transfusion_editing_clap44*.py"))
    paths.update([Path(__file__).resolve(),ROOT/"stable_audio_tools/data/sceneplan_transfusion_editing_clap44.py",
                  ROOT/"scripts/t2a/train/editing_gpu_runtime.py"])
    return {str(p.relative_to(ROOT)):file_sha256(p) for p in sorted(paths)}


def main():
    args = parse_args(); cfg = json.loads(args.config.read_text()); validate_config(cfg, args.training_mode)
    schedule = cfg["schedule"]
    rank,local_rank,world,device,topology = _distributed()
    _seed_everything(int(cfg["seed"]),rank)
    run_dir = args.run_dir.resolve(); model_path = args.model_config.resolve(strict=True)
    model_config = load_config(model_path)
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("status") != "PASS" or [preflight["indices"][x]["rows"] for x in ("train","validation","test")] != [1000000,20000,5000]:
        raise RuntimeError("CLAP44 joint training requires the complete 1M/20k/5k preflight")
    # All variants load the same frozen encoder before adapter initialization.
    # Its loader preserves CPU RNG, so the latent-only control starts from
    # exactly the same AR initialization and data/noise seed policy.
    clap,clap_identity = (None, None) if args.clap_checkpoint is None else load_clap44_checkpoint(args.clap_checkpoint)
    if clap_identity is not None and clap_identity["contract"]["preflight_sha256"] != file_sha256(args.preflight):
        raise RuntimeError("CLAP44 encoder and joint training data differ")
    startup = [None]
    if rank == 0:
        try:
            identity = ensure_run_identity(run_dir)
            if args.resume is None and any(run_dir.glob("checkpoints/step-*.pt")):
                raise RuntimeError("existing CLAP44 joint candidates require explicit resume")
            summaries = {split:_index_summary(Path(preflight["indices"][split]["path"]),preflight["indices"][split]["sha256"],preflight["indices"][split]["rows"]) for split in ("train","validation")}
            if args.training_mode == "ar_pretrain":
                base = args.p10_checkpoint.resolve(strict=True)
                digest = file_sha256(base)
                if digest != model_config.get("_source_checkpoint_sha256"):
                    raise RuntimeError("P10 checkpoint does not match the model's pinned initialization")
                base_record = {"checkpoint": str(base), "sha256": digest, "initialization": "P10_EMA"}
                gate = {"status": "NOT_APPLICABLE_TO_AR_PRETRAINING", "quality_gate_passed": False}
            else:
                selection = json.loads(args.dit_selection.read_text())
                base = Path(selection["selected_checkpoint"]).resolve(strict=True)
                base_record = _checkpoint_selection_summary(args.dit_selection,expected_sha256=file_sha256(args.dit_selection),checkpoint=base,validation_summary=summaries["validation"],model_config=model_path)
                from scripts.t2a.eval.evaluate_sceneplan_transfusion_editing_gt_audio import validate_audio_gate
                validate_audio_gate(args.dit_gt_audio_gate,selection_path=args.dit_selection)
                gate = {"path":str(args.dit_gt_audio_gate.resolve()),"sha256":file_sha256(args.dit_gt_audio_gate),"status":"PASS"}
            clap_report = None if clap_identity is None else audit_clap_validation(args.clap_validation_report,clap_identity,preflight)
            prompt = next(x for x in model_config["model"]["conditioning"]["configs"] if x["id"]=="prompt")
            if prompt["config"]["enable_grad"] is not False: raise RuntimeError("joint Qwen must be frozen")
            qwen = verify_frozen_qwen_runtime(prompt["config"]["model_path"])
            startup[0] = {"identity":identity,"indices":summaries,"base":str(base),"base_selection":base_record,"dit_gt_audio_gate":gate,"clap_validation":clap_report,"qwen":qwen,"sources":run_source_inventory()}
        except Exception as error:
            startup[0] = {"error":f"{type(error).__name__}: {error}"}
    dist.broadcast_object_list(startup,src=0)
    audit = startup[0]
    if audit is None or "error" in audit: raise RuntimeError(f"CLAP44 joint startup failed: {audit}")
    codec_path = args.codec.resolve(strict=True); codec = ModelScenePlanCodecV4(codec_path)
    module = build_model(model_config,audit["base"],codec,clap,args.variant,float(cfg["source_feature_dropout"]),args.training_mode).to(device).train()
    tokenizer = module.diffusion.conditioner.conditioners["prompt"].tokenizer
    datasets = {}
    for split in ("train","validation"):
        record = audit["indices"][split]
        base = ScenePlanTransfusionEditingDataset(record["path"],tokenizer_spec=(tokenizer,512),expected_num_samples=record["rows"],expected_index_sha256=record["sha256"],latent_crop_length=648,require_frozen=True,verify_tensor_hashes_on_access=False)
        datasets[split] = ScenePlanTransfusionEditingJointDataset(base,codec=codec)
    sampler = DistributedScenePlanBucketBatchSampler(datasets["train"],short_batch_size=schedule["short_batch_size"],long_batch_size=schedule["long_batch_size"],num_replicas=world,rank=rank,shuffle=True,seed=cfg["seed"],drop_last=True)
    collate = functools.partial(collate_editing_joint,pad_id=codec.pad_id)
    loader_generator = torch.Generator().manual_seed(cfg["seed"]+90001*rank)
    loader = DataLoader(datasets["train"],batch_sampler=sampler,num_workers=schedule["num_workers"],pin_memory=True,persistent_workers=schedule["num_workers"]>0,collate_fn=collate,generator=loader_generator,
        multiprocessing_context="spawn" if schedule["num_workers"]>0 else None)
    batches,probe = _stratified_validation_batches(Path(audit["indices"]["validation"]["path"]),batch_count=schedule["validation_batches"],batch_size=schedule["validation_batch_size"])
    val_loader = DataLoader(datasets["validation"],batch_sampler=batches,num_workers=0,collate_fn=collate,
        generator=torch.Generator().manual_seed(cfg["seed"]+909001))
    groups,counts = optimizer_groups(module,ar_lr=cfg["ar_lr"],shared_lr=cfg["shared_lr"],dit_lr=cfg["dit_lr"])
    optimizer = torch.optim.AdamW(groups,betas=(.9,.95),weight_decay=cfg["weight_decay"],fused=True)
    def lr_multiplier(step):
        if step < schedule["warmup_steps"]: return (step+1)/schedule["warmup_steps"]
        progress = min(1.,(step-schedule["warmup_steps"])/(schedule["max_steps"]-schedule["warmup_steps"]))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_multiplier)
    contract = {"schema":AR_PRETRAIN_RUN_SCHEMA if args.training_mode=="ar_pretrain" else JOINT44_RUN_SCHEMA,"training_mode":args.training_mode,"rf_mode":"p10_generation_zero_reference" if args.training_mode=="ar_pretrain" else "editing_reference","run_dir":str(run_dir),"run_id":audit["identity"]["run_id"],"repo_root":str(ROOT),"ar_contract":module.ar.ar_contract,"variant":args.variant,"config":cfg,"schedule":schedule,"world_size":world,"physical_gpus":topology["physical_indices"],"gpu_topology":topology,"m2d_used":False,"independent_test_used":False,"model_config":str(model_path),"model_config_sha256":file_sha256(model_path),"codec":str(codec_path),"codec_sha256":codec_artifact_sha256(codec_path),"codec_fingerprint":codec.fingerprint,"indices":audit["indices"],"base_selection":audit["base_selection"],"dit_gt_audio_gate":audit["dit_gt_audio_gate"],"clap_checkpoint":None if clap_identity is None else {key:clap_identity[key] for key in ("path","sha256","step")},"clap_validation":audit["clap_validation"],"frozen_qwen_runtime":audit["qwen"],"internal_validation_probe":probe,"trainable_parameters":counts,"source_sha256":audit["sources"],"runtime_inputs":["source_foa_latent","raw_edit_request"],"old_plan_input":False,"source_caption_input":False,"target_audio_ar_input":False,"auxiliary":"frozen_source_audio_dual_head_distillation" if args.variant!="latent_only" else None,"loss_normalization":"global_valid_plan_tokens_RF_values_source_rows_per_optimizer_step"}
    contract.update(dataloader_multiprocessing_context="spawn" if schedule["num_workers"]>0 else None,
        dataloader_rng="dedicated_epoch_generator_seed_plus_900001_plus_epoch_world_plus_rank",
        validation_loader_rng="dedicated_generator_seed_plus_909001")
    if rank == 0:
        cp = run_dir/"RUN_CONTRACT.json"
        if cp.exists() and json.loads(cp.read_text()) != contract: raise RuntimeError("CLAP44 joint output contract changed")
        if not cp.exists(): atomic_json(cp,contract)
    dist.barrier()
    step = epoch = next_batch = 0; pending_rng = None
    if args.resume is not None:
        payload,_ = load_joint_checkpoint(args.resume,expected_contract=contract,require_latest=True)
        module.diffusion.load_state_dict(payload["diffusion_state_dict"],strict=True)
        load_ar_specific(module.ar,payload["editing_ar_specific_state_dict"])
        optimizer.load_state_dict(payload["optimizer"]); scheduler.load_state_dict(payload["scheduler"])
        step,epoch,next_batch = int(payload["global_step"]),int(payload["epoch"]),int(payload["next_batch"])
        pending_rng = payload["rng_states_by_rank"][rank]
        if next_batch > len(sampler): raise RuntimeError("CLAP44 resume batch exceeds its sampler epoch")
        if next_batch == len(sampler): epoch+=1; next_batch=0
        state = sampler.resumable_state_dict(at_epoch_boundary=True); state["resume_epoch"]=epoch; sampler.load_resumable_state_dict(state)
        if next_batch: sampler.set_resume_batch_offset(next_batch)
        del payload
    wrapped = DistributedDataParallel(module,device_ids=[local_rank],broadcast_buffers=False,find_unused_parameters=False,gradient_as_bucket_view=True)
    started = time.monotonic()
    while step < schedule["max_steps"]:
        loader_generator.manual_seed(cfg["seed"]+900001+epoch*world+rank)
        iterator = iter(loader)
        if pending_rng is not None: _restore_rank_rng_state(pending_rng,rank=rank,device=device); pending_rng=None
        while True:
            window = list(itertools.islice(iterator,schedule["gradient_accumulation"]))
            if not window: break
            # Normalize the entire optimizer window, including a short final
            # window, over all ranks. This prevents duration/token-count bias.
            denominators = torch.tensor([sum(int((b["ar"]["plan_labels"]!=-100).sum()) for b in window),sum(sum(int(row["padding_mask"][0].sum())*64 for row in b["metadata"]) for b in window),sum(len(b["ar"]["plan_labels"]) for b in window)],device=device,dtype=torch.float64)
            dist.all_reduce(denominators)
            if not bool((denominators>0).all()): raise RuntimeError("empty CLAP44 joint optimizer window")
            sums = torch.zeros(3,device=device,dtype=torch.float64)
            optimizer.zero_grad(set_to_none=True)
            for micro,batch in enumerate(window):
                ar,target,metadata,rf_mask = _move_joint_batch(batch,device)
                # RF noise/timesteps remain identical across feature variants,
                # irrespective of feature dropout or model RNG consumption.
                rf_seed = int(cfg["seed"])+31_000_001+step*world*schedule["gradient_accumulation"]+rank*schedule["gradient_accumulation"]+micro
                generator = torch.Generator(device=device).manual_seed(rf_seed)
                noise = torch.randn(target.shape,device=device,dtype=target.dtype,generator=generator)
                times = torch.rand(len(target),device=device,generator=generator)
                noised = (1-times[:,None,None])*target+times[:,None,None]*noise
                sync = wrapped.no_sync() if micro+1<len(window) else nullcontext()
                with sync:
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        logits,prediction,query,teacher = wrapped(source_foa_latent=ar["source_foa_latent"],source_attention_mask=ar["source_attention_mask"],plan_input_ids=ar["plan_input_ids"],plan_attention_mask=ar["plan_attention_mask"],raw_edit_requests=ar["raw_edit_requests"],metadata=metadata,noised_target=noised,timesteps=times,rf_padding_mask=rf_mask)
                        _,_,ar_sum,rf_sum = _losses(logits,ar["plan_labels"],prediction,noise-target,rf_mask)
                        aux_sum = logits.sum()*0 if args.variant=="latent_only" else source_feature_distillation(query,teacher,semantic_dim=clap.config.semantic_dim,scene_weight=cfg["scene_distillation_weight"])[0].sum()
                        loss = normalized_joint_loss(ar_sum,rf_sum,aux_sum,denominators,world_size=world,lambda_ar=cfg["lambda_ar"],lambda_rf=cfg["lambda_rf"],lambda_auxiliary=cfg["lambda_source_distillation"])
                    if not bool(torch.isfinite(loss)): raise RuntimeError("non-finite CLAP44 joint loss")
                    loss.backward()
                sums += torch.stack((ar_sum.detach(),rf_sum.detach(),aux_sum.detach())).double()
            grad_norms = [float(torch.nn.utils.clip_grad_norm_(group["params"],cfg["gradient_clip"],error_if_nonfinite=True)) for group in optimizer.param_groups]
            optimizer.step(); scheduler.step(); step+=1; next_batch+=len(window)
            if step % schedule["log_every"] == 0:
                dist.all_reduce(sums)
                if rank==0:
                    values = (sums/denominators).tolist(); event={"step":step,"epoch":epoch,"ar_ce":values[0],"rf_mse":values[1],"source_distillation":values[2],"grad_norms":grad_norms,"elapsed_sec":time.monotonic()-started}
                    print("CLAP44_AR_TRAIN="+json.dumps(event),flush=True)
                    with (run_dir/"metrics.jsonl").open("a") as stream: stream.write(json.dumps(event)+"\n")
            if step % schedule["validate_every"] == 0 or step==schedule["max_steps"]:
                if rank==0:
                    result=validation_metrics(module,val_loader,device=device,max_batches=schedule["validation_batches"],scene_weight=cfg["scene_distillation_weight"])
                    atomic_json(run_dir/f"validation-step-{step:08d}.json",{"step":step,**result})
                dist.barrier()
            if step % schedule["save_every"] == 0 or step==schedule["max_steps"]:
                states=_gather_rank_rng_states(rank=rank,world_size=world,device=device)
                if rank==0: save_joint_checkpoint(run_dir/f"checkpoints/step-{step:08d}.pt",module=module,optimizer=optimizer,scheduler=scheduler,step=step,epoch=epoch,next_batch=next_batch,contract=contract,rng_states=states)
                dist.barrier()
            if step==schedule["max_steps"]: break
        epoch+=1; next_batch=0
    if rank==0:
        atomic_json(run_dir/"FIT_COMPLETE.json",{"status":"FIT_COMPLETE_NOT_QUALITY_PASS","step":step,"run_contract_sha256":file_sha256(run_dir/"RUN_CONTRACT.json"),"variant":args.variant,"training_mode":args.training_mode,"next":"Validate free AR plans, then transfer adapters/distill onto a qualified Editing DiT and verify joint alignment." if args.training_mode=="ar_pretrain" else "Full validation candidate selection, free AR complete plans, post-joint GT audio regression and actual edited FOA evaluation; independent test stays sealed."})
    dist.destroy_process_group()


if __name__=="__main__": main()
