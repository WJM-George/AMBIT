#!/usr/bin/env python3
"""Full 20k native CLAP44 AR/RF selection, with one sealed holdout candidate."""
from __future__ import annotations
import os

import argparse
import gc
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))

import torch
from torch import distributed as dist
from stable_audio_tools.configuration import load_config
from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import ScenePlanTransfusionEditingDataset
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import ScenePlanTransfusionEditingJointDataset
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_io import file_sha256
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import atomic_json, load_ar_specific, load_joint_checkpoint
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_pipeline import load_clap44_joint_candidate
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_selection_io import (
    EVAL_SCHEMA, SELECTION_STATUS, audit_full_run, candidate_derivations, derive_selection,
    load_work, save_work, source_inventory, validate_evaluation_contract, validate_selected,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_provenance import verify_frozen_qwen_runtime
from stable_audio_tools.training.factory import create_training_wrapper_from_config
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_selection import (
    POLICY, SOURCE_VARIANTS, evaluate_ar, free_ar_pass,
)
from scripts.t2a.eval.select_sceneplan_transfusion_editing_dit_checkpoint import (
    DEFAULT_TIMESTEPS, _evaluate_route, _gpu_topology, _local_donor_mapping, _make_loader,
    _rank0_audit, _rank_ordinals, _selection_holdout_folds, _validation_layout,
)
from scripts.t2a.eval.select_sceneplan_transfusion_editing_joint_checkpoint import (
    _distributed, _free_ordinals, _JointDonorResolver, _load_promoted_base_dit_candidate, _make_joint_loader,
)
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import _checkpoint_selection_summary, _index_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    parser.add_argument("--preflight",type=Path,default=Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_transfusion_editing_v1/contracts/full_training/PREFLIGHT.json"))
    parser.add_argument("--short-ar-batch-size",type=int,default=8)
    parser.add_argument("--long-ar-batch-size",type=int,default=5)
    parser.add_argument("--short-rf-batch-size",type=int,default=72)
    parser.add_argument("--long-rf-batch-size",type=int,default=48)
    parser.add_argument("--free-batch-size",type=int,default=2)
    parser.add_argument("--num-workers",type=int,default=4)
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--selection-sha256")
    return parser.parse_args()


def prepare(args, directory):
    contract, run = audit_full_run(args.run_dir)
    index = Path(contract["indices"]["validation"]["path"]).resolve(strict=True)
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("status") != "PASS" or any(
        preflight["indices"][split]["rows"] != count for split,count in (("train",1000000),("validation",20000),("test",5000))
    ) or any(preflight["indices"][split]["sha256"] != contract["indices"][split]["sha256"] for split in ("train","validation")):
        raise RuntimeError("native CLAP44 selection requires the original complete preflight")
    index_summary = _index_summary(index,contract["indices"]["validation"]["sha256"],20000)
    layout_summary, layout = _validation_layout(index,expected_sha256=index_summary["sha256"])
    folds, fold_summary = _selection_holdout_folds(layout)
    free, free_summary = _free_ordinals(index,folds)
    base_record = contract["base_selection"]
    base_selection = _checkpoint_selection_summary(
        Path(base_record["path"]),expected_sha256=base_record["sha256"],
        checkpoint=Path(base_record["selected_checkpoint"]),validation_summary=index_summary,
        model_config=Path(contract["model_config"]))
    if base_selection != base_record:
        raise RuntimeError("native joint run changed its promoted base DiT")
    from scripts.t2a.eval.evaluate_sceneplan_transfusion_editing_gt_audio import validate_audio_gate
    gate = contract["dit_gt_audio_gate"]
    if file_sha256(gate["path"]) != gate["sha256"]:
        raise RuntimeError("base DiT GT audio gate changed")
    validate_audio_gate(Path(gate["path"]),selection_path=Path(base_record["path"]))
    cfg = load_config(contract["model_config"])
    prompt = next(x for x in cfg["model"]["conditioning"]["configs"] if x["id"]=="prompt")
    if verify_frozen_qwen_runtime(prompt["config"]["model_path"]) != contract["frozen_qwen_runtime"]:
        raise RuntimeError("native joint selection Qwen runtime changed")
    value = {
        "schema":EVAL_SCHEMA, "output_dir":str(directory), "training_run":run, "policy":POLICY,
        "variant":contract["variant"], "validation_index":layout_summary, "layout":[list(row) for row in layout],
        "folds":fold_summary, "free_ordinals":free, "free_population":free_summary,
        "preflight":{"path":str(args.preflight.resolve()),"sha256":file_sha256(args.preflight)},
        "base_dit_selection":base_record, "dit_gt_audio_gate":gate, "rf_timesteps":list(DEFAULT_TIMESTEPS),
        "settings":{key:getattr(args,key) for key in ("short_ar_batch_size","long_ar_batch_size","short_rf_batch_size","long_rf_batch_size","free_batch_size","num_workers")},
        "seed":42, "physical_gpus":[3,4,5,6,7], "world_size":5, "gpu_topology":_gpu_topology(),
        "source_sha256":source_inventory(), "quality_gate_passed":False,
    }
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/"EVALUATION_CONTRACT.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError("existing evaluation contract differs; keep its evidence intact")
    else:
        if any(directory.iterdir()):
            raise RuntimeError("evaluation artifacts exist without their contract")
        atomic_json(path,value)
    return value, contract


def _restore_candidate(pipeline, path, contract):
    payload, identity = load_joint_checkpoint(path,expected_contract=contract,verify_sources=False)
    pipeline.diffusion.load_state_dict(payload["diffusion_state_dict"],strict=True)
    load_ar_specific(pipeline.editing_ar,payload["editing_ar_specific_state_dict"])
    pipeline.eval()
    return identity


def main():
    args = parse_args()
    if args.verify_only:
        from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_selection_io import validate_selected
        if not args.selection_sha256:
            raise RuntimeError("selection verification needs its pinned SHA256")
        directory = (args.output_dir or args.run_dir/"evaluation/validation_20k_clap44_joint_selection").resolve()
        validate_selected(directory/"SELECTED.json",expected_sha256=args.selection_sha256)
        print(json.dumps({"event":"clap44_joint_selection_verified","status":"PLAN_RF_GATE_PASS_NOT_AUDIO_QUALITY_PASS"}))
        return 0
    if min(args.short_ar_batch_size,args.long_ar_batch_size,args.short_rf_batch_size,args.long_rf_batch_size,args.free_batch_size) < 1 or args.num_workers < 0:
        raise ValueError("invalid selection batch settings")
    rank, _, device = _distributed()
    torch.manual_seed(42+rank)
    torch.cuda.manual_seed_all(42+rank)
    torch.set_float32_matmul_precision("high")
    directory = (args.output_dir or args.run_dir/"evaluation/validation_20k_clap44_joint_selection").resolve()
    value, contract = _rank0_audit(lambda:prepare(args,directory),rank=rank,device=device)
    selected_path = directory/"SELECTED.json"
    def existing_result():
        if selected_path.exists():
            validate_selected(selected_path,expected_sha256=file_sha256(selected_path))
            return "passed"
        if (directory/"FAILED.json").exists():
            saved = json.loads((directory/"FAILED.json").read_text())
            derived = derive_selection(directory,value,contract)
            if saved != derived:
                raise RuntimeError("saved failed selection cannot be replayed")
            return "failed"
        return None
    existing = _rank0_audit(existing_result,rank=rank,device=device)
    if existing is not None:
        dist.destroy_process_group()
        return 0 if existing=="passed" else 2
    assigned, _ = _rank_ordinals(value["layout"],rank)
    donors, _ = _local_donor_mapping(value["layout"],assigned)
    index = Path(value["validation_index"]["path"])
    config = load_config(contract["model_config"])
    base_diffusion = create_model_from_config(config)
    tokenizer = base_diffusion.conditioner.conditioners["prompt"].tokenizer
    codec = ModelScenePlanCodecV4(contract["codec"])
    def dataset(ordinals):
        return ScenePlanTransfusionEditingDataset(index,tokenizer_spec=(tokenizer,512),
            expected_num_samples=len(ordinals),index_num_samples=20000,
            expected_index_sha256=value["validation_index"]["sha256"],sample_ordinals=ordinals,
            latent_crop_length=648,require_frozen=True,verify_tensor_hashes_on_access=False)
    base_dataset = dataset(assigned)
    joint_dataset = ScenePlanTransfusionEditingJointDataset(base_dataset,codec=codec)
    ar_loader,_ = _make_joint_loader(joint_dataset,short_batch_size=args.short_ar_batch_size,long_batch_size=args.long_ar_batch_size,num_workers=args.num_workers)
    rf_loader,_ = _make_loader(base_dataset,short_batch_size=args.short_rf_batch_size,long_batch_size=args.long_rf_batch_size,num_workers=args.num_workers)
    donor_resolver = _JointDonorResolver(joint_dataset,assigned,donors)
    if load_work(directory,"base",rank,required=False) is None:
        wrapper = create_training_wrapper_from_config(config,base_diffusion)
        base = value["base_dit_selection"]
        _load_promoted_base_dit_candidate(wrapper,Path(base["selected_checkpoint"]),
            selection_summary=base,selection_value=json.loads(Path(base["path"]).read_text()),resolved_model_config=config)
        route = wrapper.diffusion_ema.ema_model.to(device)
        base_diffusion.conditioner.to(device)
        raw = _evaluate_route(route,base_diffusion,rf_loader,device=device,rank=rank,seed=42,
            timesteps=DEFAULT_TIMESTEPS,source_variants=("clean",),conditioner_context=wrapper.ema_conditioner_context())
        save_work(directory,"base",rank,raw)
        del route,wrapper,raw
    del base_diffusion
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    candidates = value["training_run"]["candidates"]
    pipeline,_ = load_clap44_joint_candidate(candidates[0]["checkpoint"],device=device,load_audio_autoencoder=False)
    current = candidates[0]["step"]
    for candidate in candidates:
        step = candidate["step"]
        phase = f"clean-{step}"
        if load_work(directory,phase,rank,required=False) is None:
            if current != step:
                _restore_candidate(pipeline,candidate["checkpoint"],contract)
                current = step
            ar_raw = evaluate_ar(pipeline.editing_ar,ar_loader,device=device)
            rf_raw = _evaluate_route(pipeline.diffusion.model,pipeline.diffusion,rf_loader,device=device,rank=rank,
                seed=42,timesteps=DEFAULT_TIMESTEPS,source_variants=("clean",),conditioner_context=torch.no_grad())
            save_work(directory,phase,rank,{"ar":ar_raw,"rf":rf_raw})
        dist.barrier()
        if rank == 0:
            print("CLAP44_JOINT_SELECTION="+json.dumps({"event":"candidate_full_20k_complete","step":step}),flush=True)
    def pin_ranking():
        records, ranked, _, _ = candidate_derivations(directory,value)
        if not ranked:
            atomic_json(directory/"FAILED.json",{"status":"FAIL","failure":"no_selection_fold_candidate","candidates":records})
            return None
        pin = {"evaluation_contract_sha256":file_sha256(directory/"EVALUATION_CONTRACT.json"),
               "ranked_steps":[row["step"] for row in ranked],"selected_step":ranked[0]["step"]}
        path = directory/"RANKING.json"
        if path.exists() and json.loads(path.read_text()) != pin:
            raise RuntimeError("native joint candidate ranking changed")
        if not path.exists():
            atomic_json(path,pin)
        return pin["selected_step"]
    selected = _rank0_audit(pin_ranking,rank=rank,device=device)
    if selected is None:
        dist.destroy_process_group()
        return 2
    candidate = next(row for row in candidates if row["step"]==selected)
    if current != selected:
        _restore_candidate(pipeline,candidate["checkpoint"],contract)
    phase = f"interventions-{selected}"
    if load_work(directory,phase,rank,required=False) is None:
        variants = ("clean","zero","shuffled") if contract["variant"]=="latent_only" else SOURCE_VARIANTS
        ar_raw = evaluate_ar(pipeline.editing_ar,ar_loader,device=device,variants=variants,donor_resolver=donor_resolver)
        rf_raw = _evaluate_route(pipeline.diffusion.model,pipeline.diffusion,rf_loader,device=device,rank=rank,
            seed=42,timesteps=DEFAULT_TIMESTEPS,source_variants=("clean","zero","shuffled"),
            conditioner_context=torch.no_grad(),shuffled_source_provider=donor_resolver.rf_values)
        save_work(directory,phase,rank,{"ar":ar_raw,"rf":rf_raw})
    phase = f"free-{selected}"
    if load_work(directory,phase,rank,required=False) is None:
        free_dataset = ScenePlanTransfusionEditingJointDataset(dataset(value["free_ordinals"][rank::5]),codec=codec)
        records = free_ar_pass(pipeline.editing_ar,codec,free_dataset,device=device,batch_size=args.free_batch_size)
        save_work(directory,phase,rank,records)
    dist.barrier()
    def publish():
        validate_evaluation_contract(directory)
        result = derive_selection(directory,value,contract)
        path = selected_path if result["status"]==SELECTION_STATUS else directory/"FAILED.json"
        atomic_json(path,result)
        print("CLAP44_JOINT_SELECTION="+json.dumps({"event":"plan_rf_gates_complete","status":result["status"],"path":str(path)}),flush=True)
        return result["status"]==SELECTION_STATUS
    passed = _rank0_audit(publish,rank=rank,device=device)
    dist.destroy_process_group()
    return 0 if passed else 2


if __name__=="__main__":
    raise SystemExit(main())
