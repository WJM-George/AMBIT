#!/usr/bin/env python3
"""Native CLAP44 post-joint GT-plan decoded-audio regression gate.

The AR does not generate this gate's plans. The selected joint DiT must retain
the original GT-plan quality, including actual zero/shuffled-reference audio.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))

import torch
import soundfile as sf
from torch import distributed as dist
from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_gt_audio as gt
from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_audio_end_to_end as audio
from scripts.t2a.eval import select_sceneplan_transfusion_editing_dit_checkpoint as rf
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import ScenePlanTransfusionEditingDataset
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_pipeline import load_clap44_joint_candidate
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_selection_io import (
    source_inventory as selection_sources, validate_selected,
)

SCHEMA = "editing_clap44_post_joint_gt_audio_v1"


def code_hashes():
    paths = set(gt._code_hashes()) | set(selection_sources()) | {
        str(Path(__file__).resolve().relative_to(REPO)),
        "scripts/t2a/eval/run_sceneplan_transfusion_editing_clap44_gt_audio_5gpu.sh",
    }
    return {name:gt.sha256_file(REPO/name) for name in sorted(paths)}


def policy():
    return {"inherited_gt_audio_policy":gt._policy(),
            "phase":"post_joint_native_clap44",
            "checkpoint_weights":"selected_joint_online_shared_backbone",
            "gt_plan_representation":"canonical_gt", "ar_generates_plans":False}


def selected_identity(selection_path, selection_sha256):
    selection_path = Path(selection_path).resolve(strict=True)
    selected = validate_selected(selection_path,expected_sha256=selection_sha256)
    evaluation = json.loads(Path(selected["evaluation_contract"]).read_text())
    run_dir = Path(evaluation["training_run"]["run_dir"])
    run_path = run_dir/"RUN_CONTRACT.json"
    run = json.loads(run_path.read_text())
    baseline_path = Path(run["dit_gt_audio_gate"]["path"])
    base_selection = Path(run["base_selection"]["path"])
    if gt._artifact(baseline_path) != {key:run["dit_gt_audio_gate"][key] for key in ("path","sha256")}:
        raise RuntimeError("native joint training changed its pre-joint decoded-audio gate")
    gt.validate_audio_gate(baseline_path,selection_path=base_selection)
    index = Path(run["indices"]["validation"]["path"])
    if gt.sha256_file(index) != run["indices"]["validation"]["sha256"]:
        raise RuntimeError("native post-joint audio validation index changed")
    return {
        "joint_selection":gt._artifact(selection_path),
        "joint_checkpoint":{"path":selected["selected_checkpoint"],"sha256":selected["selected_checkpoint_sha256"]},
        "joint_run_contract":gt._artifact(run_path), "base_selection":gt._artifact(base_selection),
        "pre_joint_gate":gt._artifact(baseline_path), "model_config":gt._artifact(Path(run["model_config"])),
        "codec":run["codec"], "codec_sha256":run["codec_sha256"], "variant":run["variant"],
        "validation_index":gt._artifact(index),
        "validation_marker":gt._artifact(index.with_suffix(".sqlite.frozen.json")),
    }


def check_record(record, row, contract, digest):
    gt._check_record(record,row,digest)
    if (record.get("plan_representation") != "canonical_gt" or
            record.get("donor_ordinal") != contract["donors"][str(row["pair_ordinal"])] or
            record.get("native_joint_checkpoint_sha256") != contract["identity"]["joint_checkpoint"]["sha256"]):
        raise RuntimeError("native post-joint GT audio row lost canonical plan/reference/checkpoint binding")
    samples = int(row["model_num_samples"])
    if record.get("model_num_samples") != samples or record.get("latent_frames_valid") != (samples+1023)//1024:
        raise RuntimeError("native GT audio source length changed")
    for artifact in record["variant_audio"].values():
        info = sf.info(artifact["path"])
        if info.samplerate != 44100 or info.channels != 4 or info.frames != samples:
            raise RuntimeError("native GT waveform must be exact-length 44.1kHz four-channel FOA")


def _records(directory, contract, layout):
    records, artifacts = [], []
    digest = gt._digest(contract)
    for ordinal in contract["selected_ordinals"]:
        path = directory/"records"/f"{ordinal:05d}.json"
        record = json.loads(path.read_text())
        check_record(record,layout[ordinal],contract,digest)
        records.append(record)
        artifacts.append(gt._artifact(path))
    return records, artifacts


def validate_native_gt_audio(path, *, selection_path, selection_sha256):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    contract_path = gt._verify_artifact(value["contract"])
    if contract_path != path.parent/"CONTRACT.json":
        raise RuntimeError("native GT gate contract escaped its output")
    contract = json.loads(contract_path.read_text())
    identity = selected_identity(selection_path,selection_sha256)
    if (value.get("schema") != SCHEMA or contract.get("schema") != SCHEMA or
            contract.get("policy") != policy() or contract.get("identity") != identity or
            contract.get("source_sha256") != code_hashes()):
        raise RuntimeError("native GT audio policy/selection/code changed")
    assets = {"config":gt._artifact(audio.FROZEN_VAE_CONFIG),"checkpoint":gt._artifact(audio.FROZEN_VAE_CHECKPOINT)}
    if contract["vae_assets"] != assets or contract["content_assets"] != audio.verify_independent_content_metric_assets():
        raise RuntimeError("native GT audio frozen decoder/scorer assets changed")
    layout, selected, donors = gt._validation_rows(Path(identity["validation_index"]["path"]))
    if contract["selected_ordinals"] != selected or contract["donors"] != {str(k):v for k,v in donors.items()}:
        raise RuntimeError("native GT audio population or reference mapping changed")
    records, artifacts = _records(path.parent,contract,layout)
    baseline = gt.validate_audio_gate(Path(identity["pre_joint_gate"]["path"]),selection_path=Path(identity["base_selection"]["path"]))
    result = gt._summarize(records,baseline=baseline)
    if value.get("record_artifacts") != artifacts or value.get("result") != result or result["status"] != "PASS":
        raise RuntimeError("native post-joint GT decoded-audio regression gate did not pass")
    if value.get("end_to_end_quality_gate_passed") is not False:
        raise RuntimeError("GT-plan audio must not claim free-AR end-to-end quality")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-selection",type=Path,required=True)
    parser.add_argument("--joint-selection-sha256",required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--verify-only",action="store_true")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if args.verify_only:
        validate_native_gt_audio(directory/"GATE.json",selection_path=args.joint_selection,
                                 selection_sha256=args.joint_selection_sha256)
        print(json.dumps({"event":"clap44_post_joint_gt_audio_verified","status":"PASS"}))
        return 0
    rank, _, device = rf._distributed()
    torch.set_float32_matmul_precision("high")
    identity = rf._rank0_audit(lambda:selected_identity(args.joint_selection,args.joint_selection_sha256),rank=rank,device=device)
    def prepare():
        layout, selected, donors = gt._validation_rows(Path(identity["validation_index"]["path"]))
        contract = {"schema":SCHEMA,"policy":policy(),"identity":identity,"selected_ordinals":selected,
                    "donors":{str(k):v for k,v in donors.items()},"source_sha256":code_hashes(),
                    "gpu_topology":rf._gpu_topology(),"physical_gpus":[3,4,5,6,7],"world_size":5,
                    "content_assets":audio.verify_independent_content_metric_assets(),
                    "vae_assets":{"config":gt._artifact(audio.FROZEN_VAE_CONFIG),"checkpoint":gt._artifact(audio.FROZEN_VAE_CHECKPOINT)}}
        directory.mkdir(parents=True,exist_ok=True)
        path = directory/"CONTRACT.json"
        if path.exists() and json.loads(path.read_text()) != contract:
            raise RuntimeError("existing native GT audio contract changed; preserve its evidence")
        if not path.exists():
            if any(directory.iterdir()): raise RuntimeError("native GT audio artifacts have no contract")
            audio._atomic_json(path,contract)
        return contract, layout
    contract, layout = rf._rank0_audit(prepare,rank=rank,device=device)
    digest = gt._digest(contract)
    pending = []
    for ordinal in contract["selected_ordinals"][rank::5]:
        path = directory/"records"/f"{ordinal:05d}.json"
        if path.exists():
            check_record(json.loads(path.read_text()),layout[ordinal],contract,digest)
        else:
            pending.append(ordinal)
    if pending:
        # The rank-0 audit already replayed selection gates. Each worker still
        # loads the exact checksum-bound candidate and frozen native encoder.
        pipeline, report = load_clap44_joint_candidate(identity["joint_checkpoint"]["path"],device=device)
        if report["checkpoint_sha256"] != identity["joint_checkpoint"]["sha256"]:
            raise RuntimeError("native GT audio candidate changed while loading")
        scorer = audio.IndependentEditingContentEvaluator(device=device,device_index=int(device.index))
        index = Path(identity["validation_index"]["path"])
        dataset = ScenePlanTransfusionEditingDataset(index,
            tokenizer_spec=(pipeline.diffusion.conditioner.conditioners["prompt"].tokenizer,512),
            expected_num_samples=len(pending),index_num_samples=20000,
            expected_index_sha256=identity["validation_index"]["sha256"],sample_ordinals=pending,
            latent_crop_length=648,require_frozen=True,verify_tensor_hashes_on_access=True)
        truth = audio.OfflineTruthResolver(index)
        try:
            for offset,ordinal in enumerate(pending):
                try:
                    record = gt._evaluate_row(pipeline=pipeline,scorer=scorer,sample=dataset[offset],
                        truth=truth.row(ordinal),donor_truth=truth.row(contract["donors"][str(ordinal)]),
                        device=device,output_dir=directory,contract_sha=digest,canonical_codec=pipeline.codec)
                    record["native_joint_checkpoint_sha256"] = identity["joint_checkpoint"]["sha256"]
                    check_record(record,layout[ordinal],contract,digest)
                    audio._atomic_json(directory/"records"/f"{ordinal:05d}.json",record)
                except Exception as error:
                    audio._atomic_json(directory/"errors"/f"{ordinal:05d}.json",
                        {"pair_ordinal":ordinal,"contract_sha256":digest,"error":f"{type(error).__name__}: {error}"})
                    raise
                if (offset+1)%10 == 0:
                    print(json.dumps({"event":"clap44_gt_audio_progress","rank":rank,"completed":offset+1,"pending_at_start":len(pending)}),flush=True)
        finally:
            truth.close()
    dist.barrier()
    def publish():
        records, artifacts = _records(directory,contract,layout)
        baseline = gt.validate_audio_gate(Path(identity["pre_joint_gate"]["path"]),selection_path=Path(identity["base_selection"]["path"]))
        result = gt._summarize(records,baseline=baseline)
        value = {"schema":SCHEMA,"contract":gt._artifact(directory/"CONTRACT.json"),
                 "record_artifacts":artifacts,"result":result,"end_to_end_quality_gate_passed":False}
        path = directory/"GATE.json"
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError("native GT gate changed on replay; preserve the prior result")
        if not path.exists(): audio._atomic_json(path,value)
        audio._atomic_json(directory/"LISTENING.json",{
            "representative_examples":[next(row["pair_ordinal"] for row in records
                if row["operation"]==operation and row["latent_bucket_frames"]==bucket)
                for operation in audio.OPERATIONS for bucket in (432,648)],
            "worst_examples":result["worst_examples"],"all_three_reference_variants_saved":True})
        return result["status"]
    status = rf._rank0_audit(publish,rank=rank,device=device)
    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({"event":"clap44_post_joint_gt_audio_complete","status":status}),flush=True)
    return 0 if status=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
