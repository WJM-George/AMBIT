#!/usr/bin/env python3
"""Native CLAP44 free-AR audio calibration, followed by sealed independent test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))

import torch
from torch import distributed as dist
from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_audio_end_to_end as audio
from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_gt_audio as gt
from scripts.t2a.eval import select_sceneplan_transfusion_editing_dit_checkpoint as rf
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import ScenePlanTransfusionEditingDataset
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import ScenePlanTransfusionEditingJointDataset
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_pipeline import (
    CLAP44_PIPELINE_CONTRACT, load_clap44_joint_candidate,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_audio_io import (
    SCHEMA, audio_assets, check_record, code_hashes, derive_result, freeze_value,
    phase_index, policy, rank_ordinals, upstream_identity, validate_result,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("calibration","test"),required=True)
    parser.add_argument("--joint-selection",type=Path,required=True)
    parser.add_argument("--joint-selection-sha256",required=True)
    parser.add_argument("--post-joint-gt-gate",type=Path,required=True)
    parser.add_argument("--preflight",type=Path,default=Path("/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1/contracts/full_training/PREFLIGHT.json"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--calibration",type=Path)
    parser.add_argument("--calibration-sha256")
    parser.add_argument("--batch-size",type=int,choices=(1,2,4),default=2)
    parser.add_argument("--verify-only",action="store_true")
    return parser.parse_args()


def prepare(args, directory):
    identity, run = upstream_identity(args.joint_selection,args.joint_selection_sha256,args.post_joint_gt_gate)
    preflight_record = gt._artifact(args.preflight)
    preflight = json.loads(args.preflight.read_text())
    if preflight.get("status") != "PASS" or any(
        preflight["indices"][split]["rows"] != rows for split,rows in (("train",1000000),("validation",20000),("test",5000))
    ) or any(preflight["indices"][split]["sha256"] != run["indices"][split]["sha256"] for split in ("train","validation")):
        raise RuntimeError("native audio evaluation requires the original full-data preflight")
    if args.phase=="test" and (args.calibration is None or args.calibration_sha256 is None):
        raise RuntimeError("native independent test needs a pinned passed calibration")
    if args.phase=="calibration" and (args.calibration is not None or args.calibration_sha256 is not None):
        raise RuntimeError("calibration must not consume test/calibration overrides")
    calibration_record = freeze_record = None
    advertised_test = None
    def calibration_before_test_access():
        nonlocal calibration_record,freeze_record,advertised_test
        result = validate_result(args.calibration,expected_sha256=args.calibration_sha256,
            selection_path=args.joint_selection,selection_sha256=args.joint_selection_sha256,phase="calibration")
        cal_contract = json.loads(gt._verify_artifact(result["contract"]).read_text())
        if cal_contract["batch_size_per_rank"] != args.batch_size or cal_contract["identity"] != identity:
            raise RuntimeError("native independent test changed its calibrated model/batch settings")
        calibration_record = {"path":str(args.calibration.resolve(strict=True)),"sha256":args.calibration_sha256}
        advertised_test = preflight["indices"]["test"]
        frozen = freeze_value(identity,calibration_record,args.batch_size,preflight_record,advertised_test)
        directory.mkdir(parents=True,exist_ok=True)
        path = directory/"PRETEST_FREEZE.json"
        if path.exists() and json.loads(path.read_text()) != frozen:
            raise RuntimeError("independent-test model/evaluation freeze changed")
        if not path.exists(): audio._atomic_json(path,frozen)
        freeze_record = gt._artifact(path)
        return result
    # This helper calls the calibration validator before touching test paths.
    index, record, calibration = phase_index(preflight,phase=args.phase,
        validate_calibration=calibration_before_test_access)
    split, count = ("test",5000) if args.phase=="test" else ("validation",20000)
    index_summary = audio._index_summary(index,record["sha256"],count)
    if index_summary["split"] != split:
        raise RuntimeError("native audio phase/index split mismatch")
    layout = audio._layout(index,expected_rows=count,expected_split=split)
    selected, population = audio._select_ordinals(layout,phase=args.phase)
    listening = sorted(audio._listening_ordinals([row for row in layout if row["pair_ordinal"] in set(selected)],5))
    if len(listening) != 50:
        raise RuntimeError("native audio listening population must cover every operation/length cell")
    contract = {
        "schema":SCHEMA,"phase":args.phase,"identity":identity,"policy":policy(),"source_sha256":code_hashes(),
        "physical_gpus":[3,4,5,6,7],"world_size":5,"gpu_topology":rf._gpu_topology(),
        "preflight":preflight_record,"index":index_summary,"selected_ordinals":selected,"row_selection":population,
        "listening_ordinals":listening,"batch_size_per_rank":args.batch_size,"frozen_assets":audio_assets(run),
        "calibration":calibration_record,"pretest_freeze":freeze_record,"advertised_test_index":advertised_test,
    }
    directory.mkdir(parents=True,exist_ok=True)
    path = directory/"CONTRACT.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise RuntimeError("existing native audio contract changed; preserve its artifacts")
    if not path.exists():
        if any(p.name != "PRETEST_FREEZE.json" for p in directory.iterdir()):
            raise RuntimeError("native audio output has artifacts without its contract")
        audio._atomic_json(path,contract)
    return contract, layout, calibration


def main():
    args = parse_args()
    directory = args.output_dir.resolve()
    result_path = directory/"RESULT.json"
    if args.verify_only:
        validate_result(result_path,expected_sha256=gt.sha256_file(result_path),
            selection_path=args.joint_selection,selection_sha256=args.joint_selection_sha256,phase=args.phase)
        print(json.dumps({"event":"clap44_audio_result_verified","phase":args.phase,"status":"PASS"}))
        return 0
    rank, _, device = audio._distributed()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42+rank)
    torch.cuda.manual_seed_all(42+rank)
    contract, layout, calibration = rf._rank0_audit(lambda:prepare(args,directory),rank=rank,device=device)
    digest = gt._digest(contract)
    def existing_result():
        if not result_path.exists():
            return None
        saved = json.loads(result_path.read_text())
        derived = derive_result(directory,contract,layout,calibration=calibration)
        if saved != derived:
            raise RuntimeError("completed native audio result changed on replay")
        return saved["status"]
    existing = rf._rank0_audit(existing_result,rank=rank,device=device)
    if existing is not None:
        dist.destroy_process_group()
        if rank==0 and existing=="PASS":
            validate_result(result_path,expected_sha256=gt.sha256_file(result_path),
                selection_path=args.joint_selection,selection_sha256=args.joint_selection_sha256,phase=args.phase)
        return 0 if existing=="PASS" else 2
    ordinals = rank_ordinals(layout,contract["selected_ordinals"],rank)
    pipeline, report = load_clap44_joint_candidate(contract["identity"]["joint_checkpoint"]["path"],device=device)
    if report["checkpoint_sha256"] != contract["identity"]["joint_checkpoint"]["sha256"] or report["m2d_used"] is not False:
        raise RuntimeError("native audio runtime loaded a different checkpoint/route")
    scorer = audio.IndependentEditingContentEvaluator(device=device,device_index=int(device.index))
    index = Path(contract["index"]["path"])
    base_dataset = ScenePlanTransfusionEditingDataset(index,
        tokenizer_spec=(pipeline.diffusion.conditioner.conditioners["prompt"].tokenizer,512),
        expected_num_samples=len(ordinals),index_num_samples=contract["index"]["rows"],
        expected_index_sha256=contract["index"]["sha256"],sample_ordinals=ordinals,
        latent_crop_length=648,require_frozen=True,verify_tensor_hashes_on_access=True)
    dataset = ScenePlanTransfusionEditingJointDataset(base_dataset,codec=pipeline.codec,max_plan_tokens=1024)
    truth = audio.OfflineTruthResolver(index)
    runtime_args = argparse.Namespace(seed=42,max_plan_tokens=512,ode_steps=20,cfg_scale=1.,save_all_audio=True)
    folder = directory/"shards"/f"rank-{rank}"
    folder.mkdir(parents=True,exist_ok=True)
    number = 0
    try:
        for bucket in (432,648):
            for indices in audio._chunks(dataset.length_bucket_indices().get(bucket,()),args.batch_size):
                samples = [dataset[i] for i in indices]
                rows = [sample[1] for sample in samples]
                expected = [row["pair_ordinal"] for row in rows]
                path = folder/f"batch-{number:06d}.json"
                if path.exists():
                    saved = json.loads(path.read_text())
                    if (saved.get("contract_sha256") != digest or saved.get("rank") != rank or
                            saved.get("batch") != number or saved.get("pair_ordinals") != expected or
                            len(saved.get("records",[])) != len(rows)):
                        raise RuntimeError("native audio resume shard identity changed")
                    for record,row in zip(saved["records"],rows): check_record(record,row,contract)
                    number += 1
                    continue
                truths = [truth.row(ordinal) for ordinal in expected]
                def process(selected_samples,selected_truths):
                    return audio._process_batch(pipeline=pipeline,content_evaluator=scorer,codec=pipeline.codec,
                        samples=selected_samples,truths=selected_truths,bucket=bucket,device=device,
                        args=runtime_args,output_dir=directory,listening_ordinals=set(contract["listening_ordinals"]))
                try:
                    records = process(samples,truths)
                except Exception as batch_error:
                    torch.cuda.empty_cache()
                    if len(samples)==1:
                        records = audio._error_records(samples,batch_error)
                    else:
                        records = []
                        for sample,row_truth in zip(samples,truths):
                            try:
                                records.extend(process([sample],[row_truth]))
                            except Exception as row_error:
                                torch.cuda.empty_cache()
                                records.extend(audio._error_records([sample],row_error))
                for record,row in zip(records,rows):
                    record.update(contract_sha256=digest,runtime_route=CLAP44_PIPELINE_CONTRACT)
                    check_record(record,row,contract)
                if len(records) != len(rows):
                    raise RuntimeError("native audio inference omitted rows")
                audio._atomic_json(path,{"contract_sha256":digest,"rank":rank,"batch":number,
                    "pair_ordinals":expected,"records":records})
                number += 1
                print(json.dumps({"event":"clap44_audio_batch","phase":args.phase,"rank":rank,"batch":number,
                    "rows":len(records),"errors":sum(row["status"]!="ok" for row in records)}),flush=True)
    finally:
        truth.close()
    dist.barrier()
    def publish():
        result = derive_result(directory,contract,layout,calibration=calibration)
        audio._atomic_json(result_path,result)
        audio._atomic_json(directory/"LISTENING.json",{"phase":args.phase,
            "representative_ordinals":contract["listening_ordinals"],"all_edited_audio_saved":True,
            "audio_root":str(directory/"audio"),"result":gt._artifact(result_path)})
        return result["status"]
    status = rf._rank0_audit(publish,rank=rank,device=device)
    dist.destroy_process_group()
    if rank==0:
        if status=="PASS":
            validate_result(result_path,expected_sha256=gt.sha256_file(result_path),
                selection_path=args.joint_selection,selection_sha256=args.joint_selection_sha256,phase=args.phase)
        print(json.dumps({"event":"clap44_audio_complete","phase":args.phase,"status":status,"result":str(result_path)}),flush=True)
    return 0 if status=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
