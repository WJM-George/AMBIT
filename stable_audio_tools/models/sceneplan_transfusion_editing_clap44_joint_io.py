"""Distinct, recoverable CLAP44 AR/RF bundles; no M2D checkpoint masquerading."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import re
import uuid

import torch

from .sceneplan_transfusion_editing_ar import EDITING_AR_CLAP44_CONTRACT, EDITING_AR_CONTRACT
from .sceneplan_transfusion_editing_clap44_io import file_sha256

JOINT44_RUN_SCHEMA = "editing_clap44_joint_run_v1"
JOINT44_CHECKPOINT_SCHEMA = "editing_clap44_joint_checkpoint_v1"
JOINT44_RUN_ID_SCHEMA = "editing_clap44_joint_identity_v1"
AR_PRETRAIN_RUN_SCHEMA = "editing_clap44_ar_pretrain_run_v1"
AR_PRETRAIN_CHECKPOINT_SCHEMA = "editing_clap44_ar_pretrain_checkpoint_v1"


def checkpoint_schema(contract):
    return AR_PRETRAIN_CHECKPOINT_SCHEMA if contract["schema"] == AR_PRETRAIN_RUN_SCHEMA else JOINT44_CHECKPOINT_SCHEMA


def codec_artifact_sha256(directory):
    """The codec is a ready directory, not one file."""
    directory = Path(directory).resolve(strict=True)
    files = {name:file_sha256(directory/name) for name in ("READY", "codec.json", "sentencepiece.model")}
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    os.replace(temporary,path)


def ensure_run_identity(run_dir):
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True,exist_ok=True)
    path = run_dir/"RUN_IDENTITY.json"
    if not path.exists():
        if any(run_dir.glob("checkpoints/step-*.pt")) or (run_dir/"RUN_CONTRACT.json").exists():
            raise RuntimeError("CLAP44 run artifacts exist without their identity")
        value = {"schema":JOINT44_RUN_ID_SCHEMA,"run_dir":str(run_dir),"run_id":str(uuid.uuid4())}
        with path.open("x") as stream: json.dump(value,stream,indent=2)
    value = json.loads(path.read_text())
    if value.get("schema") != JOINT44_RUN_ID_SCHEMA or value.get("run_dir") != str(run_dir) or str(uuid.UUID(value["run_id"])) != value["run_id"]:
        raise RuntimeError("CLAP44 run identity is invalid")
    return value


def ar_specific_state(ar):
    excluded = ("editing_dit.","instruction_conditioner.","source_clap_model.")
    return {key:value.detach().cpu() for key,value in ar.state_dict().items() if not key.startswith(excluded)}


def load_ar_specific(ar,state):
    expected = set(ar_specific_state(ar))
    if set(state) != expected:
        raise RuntimeError("CLAP44 AR adapter checkpoint keys changed")
    incompatible = ar.load_state_dict(state,strict=False)
    if incompatible.unexpected_keys or any(not key.startswith(("editing_dit.","instruction_conditioner.","source_clap_model.")) for key in incompatible.missing_keys):
        raise RuntimeError("CLAP44 AR adapter checkpoint is incomplete")


def validate_run_contract(contract,run_dir,*,verify_sources=True):
    run_dir = Path(run_dir).resolve(strict=True)
    identity = ensure_run_identity(run_dir)
    path = run_dir/"RUN_CONTRACT.json"
    if contract.get("schema") not in {JOINT44_RUN_SCHEMA, AR_PRETRAIN_RUN_SCHEMA} or contract.get("run_dir") != str(run_dir) or contract.get("run_id") != identity["run_id"] or json.loads(path.read_text()) != contract:
        raise RuntimeError("CLAP44 joint run contract identity changed")
    if contract["schema"] == AR_PRETRAIN_RUN_SCHEMA:
        if (contract.get("training_mode") != "ar_pretrain" or
                contract.get("dit_gt_audio_gate", {}).get("status") != "NOT_APPLICABLE_TO_AR_PRETRAINING" or
                contract.get("rf_mode") != "p10_generation_zero_reference"):
            raise ValueError("AR pretraining must remain distinct from qualified joint editing")
    elif contract.get("training_mode", "joint") != "joint":
        raise ValueError("AR pretraining cannot use a joint-training schema")
    if contract.get("ar_contract") not in {EDITING_AR_CLAP44_CONTRACT,EDITING_AR_CONTRACT} or contract.get("m2d_used") is not False or contract.get("independent_test_used") is not False:
        raise ValueError("CLAP44 joint run contains a foreign route")
    if contract.get("variant") not in {"latent_only","global_only","sequence_only","global_and_sequence"}:
        raise ValueError("CLAP44 joint variant is invalid")
    if contract["variant"] == "latent_only":
        if contract["ar_contract"] != EDITING_AR_CONTRACT: raise ValueError("latent baseline contract mismatch")
    elif contract["ar_contract"] != EDITING_AR_CLAP44_CONTRACT:
        raise ValueError("CLAP44 variant contract mismatch")
    if verify_sources:
        if not contract.get("source_sha256"): raise RuntimeError("CLAP44 joint source inventory is missing")
        repo = Path(contract["repo_root"])
        for relative,expected in contract["source_sha256"].items():
            path = (repo/relative).resolve(strict=True)
            if not path.is_relative_to(repo.resolve()) or file_sha256(path) != expected:
                raise RuntimeError(f"CLAP44 joint source changed: {relative}")
    return file_sha256(run_dir/"RUN_CONTRACT.json")


def save_joint_checkpoint(path,*,module,optimizer,scheduler,step,epoch,next_batch,contract,rng_states):
    run_dir = Path(contract["run_dir"]).resolve(strict=True)
    digest = validate_run_contract(contract,run_dir,verify_sources=False)
    path = Path(path)
    if path.resolve().parent != run_dir/"checkpoints" or path.name != f"step-{step:08d}.pt" or not 0 < step <= contract["schedule"]["max_steps"]:
        raise ValueError("CLAP44 checkpoint escaped its run/step")
    from scripts.t2a.train.sceneplan_transfusion_editing_joint_run_contract import validate_rng_inventory
    validate_rng_inventory(rng_states,world_size=contract["world_size"])
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() or path.with_suffix(".manifest.json").exists():
        raise FileExistsError("CLAP44 candidates are immutable; recover publication before resuming")
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    schema = checkpoint_schema(contract)
    torch.save({"schema":schema,"run_contract":contract,"run_contract_sha256":digest,"global_step":step,"epoch":epoch,"next_batch":next_batch,"diffusion_state_dict":{key:value.detach().cpu() for key,value in module.diffusion.state_dict().items()},"editing_ar_specific_state_dict":ar_specific_state(module.ar),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"rng_states_by_rank":rng_states,"quality_gate_passed":False},temporary)
    os.replace(temporary,path)
    record = {"schema":schema,"checkpoint":str(path.resolve()),"checkpoint_sha256":file_sha256(path),"step":step,"run_contract_sha256":digest,"run_id":contract["run_id"]}
    atomic_json(path.with_suffix(".manifest.json"),record)
    atomic_json(path.parent/"LATEST.json",record)
    return record


def _checkpoint_payload(path, *, expected_contract=None, verify_sources=True):
    """Validate a finalized binary even if publication was interrupted."""
    payload = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    contract = payload["run_contract"]
    run_dir = Path(contract["run_dir"]).resolve(strict=True)
    digest = validate_run_contract(contract,run_dir,verify_sources=verify_sources)
    step = int(payload["global_step"])
    schema = checkpoint_schema(contract)
    if payload.get("schema") != schema or payload.get("run_contract_sha256") != digest or path.parent != run_dir/"checkpoints" or path.name != f"step-{step:08d}.pt" or not 0 < step <= contract["schedule"]["max_steps"] or payload.get("quality_gate_passed") is not False:
        raise RuntimeError("CLAP44 checkpoint lineage/step mismatch")
    if expected_contract is not None and contract != expected_contract:
        raise RuntimeError("CLAP44 joint resume contract changed")
    from scripts.t2a.train.sceneplan_transfusion_editing_joint_run_contract import validate_rng_inventory
    validate_rng_inventory(payload["rng_states_by_rank"],world_size=contract["world_size"])
    if int(payload["epoch"]) < 0 or int(payload["next_batch"]) < 0:
        raise ValueError("CLAP44 checkpoint sampler progress is invalid")
    for name in ("diffusion_state_dict", "editing_ar_specific_state_dict", "optimizer", "scheduler"):
        if not isinstance(payload.get(name), dict) or not payload[name]:
            raise RuntimeError(f"CLAP44 checkpoint omitted {name}")
    if not isinstance(payload["optimizer"].get("param_groups"), list) or not payload["optimizer"]["param_groups"] or not isinstance(payload["optimizer"].get("state"),dict) or not payload["optimizer"]["state"]:
        raise RuntimeError("CLAP44 checkpoint omitted optimizer state/groups")
    if payload["scheduler"].get("last_epoch") != step:
        raise RuntimeError("CLAP44 checkpoint scheduler and optimizer-step progress differ")
    return payload, {"schema":schema,"checkpoint":str(path),"checkpoint_sha256":file_sha256(path),"step":step,"run_contract_sha256":digest,"run_id":contract["run_id"]}


def _require_unchanged_file(path, before):
    after = path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns) != (after.st_ino,after.st_size,after.st_mtime_ns):
        raise RuntimeError("CLAP44 checkpoint changed while loading")


def load_joint_checkpoint(path,*,expected_contract=None,verify_sources=True,require_latest=False):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    payload, expected = _checkpoint_payload(path, expected_contract=expected_contract, verify_sources=verify_sources)
    if manifest != expected:
        raise RuntimeError("CLAP44 joint checkpoint manifest/hash mismatch")
    if require_latest and json.loads((path.parent/"LATEST.json").read_text()) != manifest:
        raise RuntimeError("CLAP44 resume is not the pinned latest candidate")
    _require_unchanged_file(path, before)
    return payload,manifest


def resolve_joint_resume(run_dir, *, verify_sources=True):
    """Recover only a complete, own-run finalized checkpoint publication.

    Caller must hold the Editing training-chain lock. Temporary binaries are
    preserved and never adopted. A present but mismatched manifest is an
    integrity failure, not permission to rehash and accept changed weights.
    The returned pointer is safe to pass to the normal strict loader.
    """
    run_dir = Path(run_dir).resolve(strict=True)
    contract = json.loads((run_dir/"RUN_CONTRACT.json").read_text())
    validate_run_contract(contract, run_dir, verify_sources=verify_sources)
    directory = run_dir/"checkpoints"
    if directory.is_symlink():
        raise RuntimeError("CLAP44 checkpoint directory escaped its run")
    candidates = sorted(directory.glob("step-*.pt"))
    if not candidates:
        if (directory/"LATEST.json").exists() or any(directory.glob("step-*.manifest.json")):
            raise RuntimeError("CLAP44 checkpoint pointers exist without any binary")
        return None
    for candidate in candidates:
        if candidate.is_symlink() or re.fullmatch(r"step-[0-9]{8}\.pt", candidate.name) is None:
            raise RuntimeError("CLAP44 candidate path is invalid")
    candidate = candidates[-1].resolve(strict=True)
    before = candidate.stat()
    manifest_path = candidate.with_suffix(".manifest.json")
    if manifest_path.exists():
        payload, record = load_joint_checkpoint(candidate, expected_contract=contract, verify_sources=False)
    else:
        payload, record = _checkpoint_payload(candidate, expected_contract=contract, verify_sources=False)
        _require_unchanged_file(candidate, before)
        atomic_json(manifest_path, record)
    del payload
    latest_path = directory/"LATEST.json"
    try:
        latest = json.loads(latest_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        latest = None
    if latest != record:
        if latest_path.exists():
            backup = latest_path.with_name(f"LATEST.previous.{uuid.uuid4().hex}.json")
            os.replace(latest_path, backup)
        atomic_json(latest_path, record)
        atomic_json(directory/"PUBLICATION_RECOVERY.json", {
            "schema":"editing_clap44_checkpoint_publication_recovery_v1",
            "checkpoint":record,
            "action":"published_validated_own_run_checkpoint",
            "quality_gate_passed":False,
        })
    return record
