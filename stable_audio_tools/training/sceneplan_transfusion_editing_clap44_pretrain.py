"""Shared CLAP44 training step and immutable, resumable optimizer snapshots."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re

import torch

from stable_audio_tools.models.sceneplan_transfusion_editing_clap44 import binding_negative_loss, clap44_objective

CHECKPOINT_SCHEMA = "editing_clap44_optimizer_checkpoint_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1048576),b""): digest.update(block)
    return digest.hexdigest()


def _atomic_json(path,value):
    path = Path(path)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)


def optimizer_and_scheduler(model,training):
    optimizer = torch.optim.AdamW(model.parameters(),lr=float(training["learning_rate"]),
        betas=(.9,.95),weight_decay=.01)
    maximum,warmup = int(training["max_steps"]),int(training["warmup_steps"])
    def schedule(step):
        if step < warmup: return max(1,step+1)/max(1,warmup)
        return .05+.95*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,maximum-warmup))))/2
    return optimizer,torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)


def training_step(model,text_encoder,optimizer,scheduler,batch,*,step,training,device,phase_context=None):
    """Exactly the production forward/backward path, optionally timed by phase."""
    core = getattr(model,"module",model)
    context = (lambda name:nullcontext()) if phase_context is None else phase_context
    autocast = torch.autocast("cuda",dtype=torch.bfloat16) if torch.device(device).type=="cuda" else nullcontext()
    labels = batch["labels"]
    descriptions = [row["semantic_text"] for row in labels]+[row["scene_text"] for row in labels]
    warmup = int(training["warmup_steps"])
    with autocast:
        with context("text_features"):
            features = text_encoder(descriptions+batch["negative_scene_texts"],device)
        with context("encoder_and_loss"):
            n = len(labels)
            audio,text = model(batch["latent"].to(device,non_blocking=True),
                batch["mask"].to(device,non_blocking=True),features[:n],features[n:2*n])
            losses = clap44_objective(core,audio,text,labels,scene_weight=float(training["scene_weight"]),
                include_edit_negatives=step>=warmup)
            extra = features[2*n:]
            if extra.numel():
                negative_scene = core.encode_text_features(extra,extra)["scene"]
                binding = binding_negative_loss(audio["scene"],text["scene"],negative_scene,batch["negative_owners"])
            else:
                binding = audio["scene"].sum()*0
            ramp = min(1.,max(0.,(step-warmup)/max(1,warmup)))
            loss = losses["loss"]+ramp*float(training["binding_weight"])*binding
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("non-finite CLAP44 objective")
    with context("backward_update"):
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
    return {"loss":loss.detach(),"semantic_loss":losses["semantic"].detach(),
            "scene_loss":losses["scene"].detach(),"binding_loss":binding.detach()}


def _validate_payload(payload,contract):
    if payload.get("contract") != contract:
        raise RuntimeError("CLAP44 resume lineage mismatch")
    step,epoch,next_batch = (payload.get(key) for key in ("step","epoch","next_batch"))
    if (type(step) is not int or not 0<step<=int(contract["config"]["training"]["max_steps"]) or
            type(epoch) is not int or epoch<0 or type(next_batch) is not int or next_batch<1):
        raise RuntimeError("CLAP44 checkpoint progress is invalid")
    if payload.get("quality_gate_passed") is not False:
        raise RuntimeError("CLAP44 optimizer checkpoints cannot assert quality promotion")
    if payload.get("scheduler",{}).get("last_epoch") != step:
        raise RuntimeError("CLAP44 scheduler progress disagrees with optimizer step")
    state = payload.get("optimizer",{}).get("state",{})
    if not state or any(float(value.get("step",-1))!=step for value in state.values()):
        raise RuntimeError("CLAP44 optimizer state is absent or has stale progress")
    if not payload.get("model") or not all(isinstance(value,torch.Tensor) and
            bool(torch.isfinite(value).all()) for value in payload["model"].values()):
        raise RuntimeError("CLAP44 checkpoint contains invalid model tensors")
    for value in state.values():
        if any(isinstance(item,torch.Tensor) and not bool(torch.isfinite(item).all()) for item in value.values()):
            raise RuntimeError("CLAP44 checkpoint contains invalid optimizer tensors")
    rng = payload.get("rng_states")
    if not isinstance(rng,list) or len(rng)!=int(contract["world_size"]):
        raise RuntimeError("CLAP44 checkpoint has incomplete per-rank RNG")
    for value in rng:
        if not isinstance(value,dict) or set(value)!={"python","numpy","torch","cuda"}:
            raise RuntimeError("CLAP44 checkpoint RNG fields are incomplete")
        if any(not isinstance(value[key],torch.Tensor) or value[key].dtype!=torch.uint8 or
               value[key].ndim!=1 or not value[key].numel() for key in ("torch","cuda")):
            raise RuntimeError("CLAP44 checkpoint RNG tensors are invalid")


def _manifest(path,payload):
    return {"schema":CHECKPOINT_SCHEMA,"checkpoint":str(path.resolve()),"sha256":sha256(path),
            "step":payload["step"],"epoch":payload["epoch"],"next_batch":payload["next_batch"],
            "contract_sha256":sha256(path.parent/"TRAIN_CONTRACT.json"),"quality_gate_passed":False}


def load_training_checkpoint(path,*,contract,require_manifest=True):
    path = Path(path).resolve(strict=True)
    if not re.fullmatch(r"step-[0-9]{6}\.pt",path.name):
        raise RuntimeError("CLAP44 checkpoint filename is not a finalized candidate")
    if json.loads((path.parent/"TRAIN_CONTRACT.json").read_text()) != contract:
        raise RuntimeError("CLAP44 checkpoint directory belongs to another contract")
    before = path.stat()
    payload = torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    _validate_payload(payload,contract)
    if path.name != f"step-{payload['step']:06d}.pt":
        raise RuntimeError("CLAP44 checkpoint filename and saved step disagree")
    manifest = _manifest(path,payload)
    sidecar = path.with_suffix(".manifest.json")
    if sidecar.exists():
        if json.loads(sidecar.read_text()) != manifest:
            raise RuntimeError("published CLAP44 checkpoint manifest/hash changed")
    elif require_manifest:
        raise RuntimeError("CLAP44 checkpoint manifest publication is incomplete")
    after = path.stat()
    if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) != (
            after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise RuntimeError("CLAP44 checkpoint changed while loading")
    return payload,manifest


def save_checkpoint(path,model,optimizer,scheduler,step,epoch,next_batch,contract,rng_states):
    path = Path(path)
    if path.exists() or path.with_suffix(".manifest.json").exists():
        raise RuntimeError("never overwrite a published CLAP44 candidate")
    if any(int(candidate.stem.removeprefix("step-"))>=step for candidate in path.parent.glob("step-*.pt")):
        raise RuntimeError("CLAP44 save would roll back an existing candidate sequence")
    if json.loads((path.parent/"TRAIN_CONTRACT.json").read_text()) != contract:
        raise RuntimeError("CLAP44 save directory contract changed")
    payload = {"contract":contract,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),"step":step,"epoch":epoch,"next_batch":next_batch,
        "rng_states":rng_states,"quality_gate_passed":False}
    _validate_payload(payload,contract)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        torch.save(payload,handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)
    _,manifest = load_training_checkpoint(path,contract=contract,require_manifest=False)
    _atomic_json(path.with_suffix(".manifest.json"),manifest)
    _atomic_json(path.parent/"LATEST.json",manifest)
    return manifest


def resolve_training_resume(directory):
    """Recover publication of finalized binaries and return the verified latest."""
    directory = Path(directory).resolve(strict=True)
    contract = json.loads((directory/"TRAIN_CONTRACT.json").read_text())
    for filename,digest in contract["source_sha256"].items():
        if sha256(filename)!=digest: raise RuntimeError("CLAP44 training source changed")
    paths = sorted(directory.glob("step-*.pt"))
    if not paths:
        if (directory/"LATEST.json").exists(): raise RuntimeError("CLAP44 latest points to no candidates")
        return None
    recovered = []
    latest = None
    for path in paths:
        _,manifest = load_training_checkpoint(path,contract=contract,require_manifest=False)
        sidecar = path.with_suffix(".manifest.json")
        if not sidecar.exists():
            _atomic_json(sidecar,manifest)
            recovered.append({"checkpoint":manifest["checkpoint"],"publication":"missing_manifest"})
        latest = manifest
    pointer = directory/"LATEST.json"
    try:
        previous = json.loads(pointer.read_text()) if pointer.exists() else None
    except json.JSONDecodeError:
        previous = None
    if previous!=latest:
        if pointer.exists():
            backup = directory/f"LATEST.unpublished.{sha256(pointer)}.json"
            if not backup.exists(): os.replace(pointer,backup)
        _atomic_json(pointer,latest)
        recovered.append({"checkpoint":latest["checkpoint"],"publication":"latest_pointer"})
    if recovered:
        _atomic_json(directory/"PUBLICATION_RECOVERY.json",{"schema":CHECKPOINT_SCHEMA,"recovered":recovered,
            "latest":latest,"quality_gate_passed":False})
    return latest
