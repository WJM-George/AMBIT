"""Bound, replayable evidence for native CLAP44 plan/RF selection.

Passing this selection authorizes subsequent audio evaluation, not a claim
that edited audio or the sealed independent test has passed.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import sqlite3
import zlib
import uuid

import torch

from .sceneplan_transfusion_editing_clap44_io import file_sha256
from .sceneplan_transfusion_editing_clap44_joint_io import (
    atomic_json, codec_artifact_sha256, load_joint_checkpoint, validate_run_contract,
)
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_selection import (
    CANDIDATE_STEPS, OPERATIONS, POLICY, _free_gate, noninferiority,
    rank_candidates, score_free_record, source_gates, subset, summary,
)
from scripts.t2a.eval.select_sceneplan_transfusion_editing_joint_checkpoint import _free_ordinals
from scripts.t2a.eval.select_sceneplan_transfusion_editing_dit_checkpoint import DEFAULT_TIMESTEPS, _rank_ordinals, _selection_holdout_folds

REPO = Path(__file__).resolve().parents[2]
EVAL_SCHEMA = "editing_clap44_joint_evaluation_v1"
SELECTION_SCHEMA = "editing_clap44_joint_selection_v1"
SELECTION_STATUS = "PLAN_RF_GATE_PASS_NOT_AUDIO_QUALITY_PASS"
WORK_SCHEMA = "editing_clap44_joint_evaluation_work_v1"


def source_inventory():
    from .sceneplan_transfusion_editing_provenance import JOINT_SELECTION_SOURCE_PATHS
    paths = {REPO/item for item in JOINT_SELECTION_SOURCE_PATHS}
    paths.update(REPO.glob("stable_audio_tools/models/sceneplan_transfusion_editing_clap44*.py"))
    paths.update(REPO.glob("stable_audio_tools/training/sceneplan_transfusion_editing_clap44*.py"))
    paths.update([
        REPO/"scripts/t2a/eval/select_sceneplan_transfusion_editing_clap44_joint.py",
        REPO/"scripts/t2a/eval/run_sceneplan_transfusion_editing_clap44_joint_selection_5gpu.sh",
    ])
    return {str(path.relative_to(REPO)):file_sha256(path) for path in sorted(paths)}


def audit_full_run(run_dir):
    run_dir = Path(run_dir).resolve(strict=True)
    contract = json.loads((run_dir/"RUN_CONTRACT.json").read_text())
    from .sceneplan_transfusion_editing_clap44_joint_io import JOINT44_RUN_SCHEMA
    if contract.get("schema") != JOINT44_RUN_SCHEMA:
        raise RuntimeError("AR pretraining requires adapter transfer and joint alignment before formal editing selection")
    digest = validate_run_contract(contract, run_dir)
    if (contract["config"]["purpose"] != "full_AR_training" or
            contract["schedule"]["max_steps"] != 25000 or contract["schedule"]["save_every"] != 5000 or
            contract["world_size"] != 5 or contract["physical_gpus"] != [3,4,5,6,7]):
        raise RuntimeError("formal CLAP44 selection requires the full 25k five-GPU run")
    final_path = run_dir/"FIT_COMPLETE.json"
    final = json.loads(final_path.read_text())
    if final.get("status") != "FIT_COMPLETE_NOT_QUALITY_PASS" or final.get("step") != 25000 or final.get("run_contract_sha256") != digest or final.get("variant") != contract["variant"]:
        raise RuntimeError("CLAP44 full run has not completed its prescribed fit")
    paths = [run_dir/"checkpoints"/f"step-{step:08d}.pt" for step in CANDIDATE_STEPS]
    if sorted((run_dir/"checkpoints").glob("step-*.pt")) != paths:
        raise RuntimeError("CLAP44 formal selection requires exactly five retained candidates")
    candidates = []
    for path in paths:
        payload, manifest = load_joint_checkpoint(path, expected_contract=contract,
                                                  verify_sources=False, require_latest=path==paths[-1])
        del payload
        candidates.append(manifest)
    return contract, {"run_dir":str(run_dir), "run_contract_sha256":digest,
                      "fit_complete_sha256":file_sha256(final_path),
                      "latest_sha256":file_sha256(run_dir/"checkpoints/LATEST.json"),
                      "candidates":candidates}


def validate_evaluation_contract(directory, *, audit_run=True):
    directory = Path(directory).resolve(strict=True)
    path = directory/"EVALUATION_CONTRACT.json"
    value = json.loads(path.read_text())
    if value.get("schema") != EVAL_SCHEMA or value.get("output_dir") != str(directory) or value.get("policy") != POLICY or value.get("source_sha256") != source_inventory():
        raise RuntimeError("native joint evaluation contract/source/policy changed")
    if value.get("seed") != 42 or value.get("world_size") != 5 or value.get("physical_gpus") != [3,4,5,6,7] or value.get("quality_gate_passed") is not False:
        raise RuntimeError("native joint evaluation execution scope changed")
    if audit_run:
        contract, audit = audit_full_run(value["training_run"]["run_dir"])
        if audit != value["training_run"]:
            raise RuntimeError("native joint candidates changed after evaluation began")
    else:
        contract = json.loads((Path(value["training_run"]["run_dir"])/"RUN_CONTRACT.json").read_text())
        if file_sha256(Path(contract["run_dir"])/"RUN_CONTRACT.json") != value["training_run"]["run_contract_sha256"]:
            raise RuntimeError("native joint run contract changed")
    if file_sha256(contract["model_config"]) != contract["model_config_sha256"] or codec_artifact_sha256(contract["codec"]) != contract["codec_sha256"]:
        raise RuntimeError("native joint model/codec files changed")
    index = Path(value["validation_index"]["path"]).resolve(strict=True)
    if (value["validation_index"]["rows"] != 20000 or file_sha256(index) != value["validation_index"]["sha256"] or
            value["validation_index"]["sha256"] != contract["indices"]["validation"]["sha256"]):
        raise RuntimeError("native joint selection changed its frozen validation population")
    if file_sha256(value["validation_index"]["marker_path"]) != value["validation_index"]["marker_sha256"]:
        raise RuntimeError("native joint frozen-validation marker changed")
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1",uri=True)
    try:
        layout = [list(row) for row in connection.execute(
            "SELECT pair_ordinal,latent_bucket_frames,operation FROM pairs WHERE split='validation' ORDER BY pair_ordinal")]
    finally:
        connection.close()
    if layout != value["layout"] or [x[0] for x in layout] != list(range(20000)) or {x[2] for x in layout} != set(OPERATIONS):
        raise RuntimeError("native joint evaluation layout changed")
    folds, fold_summary = _selection_holdout_folds(layout)
    free, free_summary = _free_ordinals(index,folds)
    if fold_summary != value["folds"] or free != value["free_ordinals"] or free_summary != value["free_population"]:
        raise RuntimeError("native joint holdout/free-plan populations changed")
    for key in ("preflight","base_dit_selection","dit_gt_audio_gate"):
        record = value[key]
        if file_sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"native joint evaluation prerequisite changed: {key}")
    if (value["base_dit_selection"] != contract["base_selection"] or
            value["dit_gt_audio_gate"] != contract["dit_gt_audio_gate"] or
            value["rf_timesteps"] != list(DEFAULT_TIMESTEPS)):
        raise RuntimeError("native joint evaluation changed its base DiT/RF protocol")
    return value, contract, file_sha256(path)


def _work_path(directory, phase, rank):
    if re.fullmatch(r"[a-z][a-z0-9_-]*", phase) is None or rank not in range(5):
        raise ValueError("invalid evaluation work identity")
    return Path(directory)/"work"/f"{phase}-rank-{rank}.pt"


def save_work(directory, phase, rank, value):
    path = _work_path(directory,phase,rank)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.with_suffix(".json").exists():
        raise FileExistsError("published evaluation work is immutable")
    if path.exists():
        os.replace(path,path.with_name(path.name+f".unpublished.{uuid.uuid4().hex}"))
    payload = {"schema":WORK_SCHEMA, "evaluation_contract_sha256":file_sha256(Path(directory)/"EVALUATION_CONTRACT.json"),
               "phase":phase, "rank":rank, "value":value}
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    torch.save(payload,temporary)
    os.replace(temporary,path)
    atomic_json(path.with_suffix(".json"), {"schema":WORK_SCHEMA, "sha256":file_sha256(path),
                "evaluation_contract_sha256":payload["evaluation_contract_sha256"], "phase":phase, "rank":rank})


def load_work(directory, phase, rank, *, required=True):
    path = _work_path(directory,phase,rank)
    manifest_path = path.with_suffix(".json")
    if not manifest_path.exists() and not required:
        return None
    manifest = json.loads(manifest_path.read_text())
    expected = {"schema":WORK_SCHEMA, "sha256":file_sha256(path),
                "evaluation_contract_sha256":file_sha256(Path(directory)/"EVALUATION_CONTRACT.json"),
                "phase":phase, "rank":rank}
    if manifest != expected:
        raise RuntimeError("evaluation work hash/contract/rank changed")
    payload = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    if {key:payload.get(key) for key in ("schema","evaluation_contract_sha256","phase","rank")} != {key:expected[key] for key in ("schema","evaluation_contract_sha256","phase","rank")}:
        raise RuntimeError("evaluation work payload identity changed")
    return payload["value"]


def merge_raw(values, layout, *, ar=False):
    if len(values) != 5:
        raise RuntimeError("evaluation evidence requires all five ranks")
    expected = {ordinal:(bucket,operation) for ordinal,bucket,operation in layout}
    for rank, raw in enumerate(values):
        assigned, _ = _rank_ordinals(layout,rank)
        if raw["ordinals"] != assigned or len(set(assigned)) != len(assigned):
            raise RuntimeError("evaluation rank dropped, duplicated or reordered rows")
        if any((bucket,operation) != expected[ordinal] for ordinal,bucket,operation in zip(assigned,raw["buckets"],raw["operations"])):
            raise RuntimeError("evaluation labels differ from frozen validation")
        def validate(value):
            if isinstance(value, torch.Tensor):
                if value.ndim not in (1,2) or len(value) != len(assigned) or not torch.isfinite(value).all():
                    raise RuntimeError("evaluation work contains incomplete/non-finite tensors")
            elif isinstance(value, dict):
                for item in value.values(): validate(item)
            elif isinstance(value, list):
                if len(value) != len(assigned): raise RuntimeError("evaluation metadata length changed")
            else:
                raise RuntimeError("unexpected evaluation raw field")
        validate(raw)
        matrices = list(raw["losses"].values())
        if any(x.shape != ((len(assigned),) if ar else (len(assigned),len(DEFAULT_TIMESTEPS))) or (x < 0).any() for x in matrices):
            raise RuntimeError("evaluation objective dimensions/values changed")
        responses = raw["response_l1" if ar else "prediction_l1"]
        if any(x.shape != ((len(assigned),) if ar else (len(assigned),len(DEFAULT_TIMESTEPS))) or (x < 0).any() for x in responses.values()):
            raise RuntimeError("evaluation response dimensions/values changed")
        if ar and any(x.shape != (len(assigned),) or (x.abs() > 1.00001).any() for x in raw["teacher_cosine"].values()):
            raise RuntimeError("invalid source teacher cosine evidence")
        if ar and (not (raw["tokens"] > 0).all() or not ((raw["accuracy"]>=0)&(raw["accuracy"]<=1)).all() or not ((raw["exact"]==0)|(raw["exact"]==1)).all()):
            raise RuntimeError("invalid AR token/accuracy evidence")
    order = torch.argsort(torch.tensor([x for raw in values for x in raw["ordinals"]]))
    def merge(items):
        if isinstance(items[0],dict):
            if any(set(item) != set(items[0]) for item in items): raise RuntimeError("rank metric sets differ")
            return {key:merge([item[key] for item in items]) for key in items[0]}
        if isinstance(items[0],torch.Tensor):
            return torch.cat(items)[order]
        flat = [x for item in items for x in item]
        return [flat[i] for i in order.tolist()]
    result = merge(values)
    if result["ordinals"] != list(range(20000)):
        raise RuntimeError("native joint evidence does not cover all 20k pairs")
    return result


def candidate_derivations(directory, value):
    layout = value["layout"]
    folds, _ = _selection_holdout_folds(layout)
    base = merge_raw([load_work(directory,"base",rank) for rank in range(5)],layout)
    mean = float(base["losses"]["clean"].double().mean())
    expected_mean = value["base_dit_selection"]["selected_clean_source_rf"]["mean"]
    if not math.isclose(mean,expected_mean,rel_tol=1e-6,abs_tol=1e-8):
        raise RuntimeError("base DiT no longer reproduces its selected full-20k RF result")
    records, raw_candidates = [], {}
    for candidate in value["training_run"]["candidates"]:
        step = candidate["step"]
        parts = [load_work(directory,f"clean-{step}",rank) for rank in range(5)]
        ar = merge_raw([part["ar"] for part in parts],layout,ar=True)
        rf = merge_raw([part["rf"] for part in parts],layout)
        ar_select, rf_select = subset(ar,folds,"selection"), subset(rf,folds,"selection")
        metrics = summary(ar_select,rf_select)
        metrics["base_dit_noninferiority"] = noninferiority(rf_select,subset(base,folds,"selection"))
        records.append({**candidate, "clean_full_20k":summary(ar,rf), "selection_10k":metrics})
        raw_candidates[step] = {"ar":ar,"rf":rf}
    ranked = rank_candidates(records)
    return records, ranked, raw_candidates, base


def verify_free_truth(records, value, codec):
    from stable_audio_tools.data.sceneplan_transfusion_editing import sha256_json
    from stable_audio_tools.data.sceneplan_transfusion_editing_plan import canonicalize_editing_plan
    records = sorted(records,key=lambda row:row["pair_ordinal"])
    if [row["pair_ordinal"] for row in records] != value["free_ordinals"]:
        raise RuntimeError("free AR evidence does not exactly cover the fixed 500 holdout pairs")
    index = value["validation_index"]["path"]
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1",uri=True)
    replayed = []
    try:
        for record in records:
            result = connection.execute("SELECT pair_id,operation,latent_bucket_frames,target_sample_id,model_num_samples,new_sceneplan_zlib,new_sceneplan_sha256 FROM pairs WHERE pair_ordinal=? AND split='validation'",(record["pair_ordinal"],)).fetchone()
            pair, operation, bucket, target_id, samples, blob, digest = result
            plan = json.loads(zlib.decompress(blob))
            if sha256_json(plan) != digest: raise RuntimeError("free AR target plan hash changed")
            plan,_ = canonicalize_editing_plan(plan,codec=codec)
            truth = codec.encode(plan,max_tokens=1024)["input_ids"].tolist()
            if (record["pair_id"] != pair or record["operation"] != operation or record["latent_bucket_frames"] != bucket or
                    record["target_sample_id"] != target_id or record["duration_sec"] != samples/44100 or
                    record["target_token_ids"] != truth):
                raise RuntimeError("free AR evidence labels differ from frozen validation truth")
            replayed.append(score_free_record(record,codec))
    finally:
        connection.close()
    return replayed


def derive_selection(directory, value, contract):
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    records, ranked, raw, base = candidate_derivations(directory,value)
    if not ranked:
        return {"status":"FAIL", "failure":"no_selection_fold_candidate", "candidates":records}
    winner = ranked[0]
    step = winner["step"]
    pin = json.loads((Path(directory)/"RANKING.json").read_text())
    expected_pin = {"evaluation_contract_sha256":file_sha256(Path(directory)/"EVALUATION_CONTRACT.json"),
                    "ranked_steps":[row["step"] for row in ranked], "selected_step":step}
    if pin != expected_pin:
        raise RuntimeError("selected candidate was not pinned before holdout interventions")
    pieces = [load_work(directory,f"interventions-{step}",rank) for rank in range(5)]
    ar = merge_raw([part["ar"] for part in pieces],value["layout"],ar=True)
    rf = merge_raw([part["rf"] for part in pieces],value["layout"])
    if not torch.equal(ar["losses"]["clean"],raw[step]["ar"]["losses"]["clean"]) or not torch.equal(rf["losses"]["clean"],raw[step]["rf"]["losses"]["clean"]):
        raise RuntimeError("selected-candidate clean replay is not deterministic")
    folds,_ = _selection_holdout_folds(value["layout"])
    ar_holdout, rf_holdout = subset(ar,folds,"holdout"), subset(rf,folds,"holdout")
    base_gate = noninferiority(rf_holdout,subset(base,folds,"holdout"))
    reference_gate = source_gates(ar_holdout,rf_holdout,variant=contract["variant"],
                                 require_teacher_alignment=contract["config"]["lambda_source_distillation"]>0)
    free = [row for rank in range(5) for row in load_work(directory,f"free-{step}",rank)]
    codec = ModelScenePlanCodecV4(contract["codec"])
    free = verify_free_truth(free,value,codec)
    free_gate = _free_gate(free,bos_id=codec.bos_id,eos_id=codec.eos_id)
    checks = {"base_dit_holdout_noninferiority":base_gate["pass"],
              "ar_rf_source_dependence":reference_gate["pass"], "free_complete_plan_quality":free_gate["pass"]}
    return {"schema":SELECTION_SCHEMA, "status":SELECTION_STATUS if all(checks.values()) else "FAIL",
            "evaluation_contract":str(Path(directory)/"EVALUATION_CONTRACT.json"),
            "evaluation_contract_sha256":file_sha256(Path(directory)/"EVALUATION_CONTRACT.json"),
            "candidate_steps":list(CANDIDATE_STEPS), "candidates":records,
            "ranked_steps":[row["step"] for row in ranked], "selected_checkpoint":winner["checkpoint"],
            "selected_checkpoint_sha256":winner["checkpoint_sha256"], "selected_step":step,
            "checks":checks, "base_dit_gate":base_gate, "source_gate":reference_gate, "free_plan_gate":free_gate,
            "free_records":free, "variant":contract["variant"], "independent_test_used":False,
            "quality_gate_passed":False, "next":"Post-joint GT decoded-audio regression and free AR edited-FOA evaluation"}


def validate_selected(path, *, expected_sha256):
    path = Path(path).resolve(strict=True)
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("native CLAP44 selection SHA256 changed")
    saved = json.loads(path.read_text())
    value, contract, _ = validate_evaluation_contract(path.parent)
    derived = derive_selection(path.parent,value,contract)
    if saved != derived or derived["status"] != SELECTION_STATUS:
        raise RuntimeError("native CLAP44 selection gates cannot be replayed from complete evidence")
    return derived
