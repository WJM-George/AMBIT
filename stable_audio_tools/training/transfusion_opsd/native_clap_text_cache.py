"""Checkpoint-bound native CLAP text projections.

Raw frozen Qwen features and learned CLAP projections are different objects.
Changing the CLAP checkpoint requires recomputing the latter, even when the
Qwen encoder and requested strings stay unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity(checkpoint):
    keys = ("path", "sha256", "step", "new_updates", "initialization_step")
    value = {key: checkpoint[key] for key in keys}
    if (not Path(value["path"]).is_absolute() or len(value["sha256"]) != 64
            or value["step"] != value["new_updates"] or value["new_updates"] <= 0):
        raise ValueError("Bind an absolute native checkpoint and its additional-update count")
    return value


def save_native_text_cache(directory, *, labels, raw_features, semantic_features,
                           checkpoint, text_encoder_provenance):
    from safetensors.torch import save_file
    labels = list(labels)
    if (not labels or len(set(labels)) != len(labels)
            or any(not isinstance(x, str) or not x.strip() for x in labels)):
        raise ValueError("Native text cache labels must be unique nonempty strings")
    if (raw_features.shape != (len(labels), 1024)
            or semantic_features.shape != (len(labels), 512)
            or not torch.isfinite(raw_features).all() or not torch.isfinite(semantic_features).all()
            or not torch.allclose(semantic_features.float().norm(dim=-1),
                                  torch.ones(len(labels), device=semantic_features.device), atol=1e-4, rtol=1e-4)):
        raise ValueError("Expected finite native Qwen1024 and normalized CLAP semantic512 features")
    identity = _identity(checkpoint)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "features.safetensors"
    save_file({"qwen.raw": raw_features.detach().float().cpu().contiguous(),
               "text.semantic": semantic_features.detach().float().cpu().contiguous()}, str(path))
    manifest = dict(contract="checkpoint_bound_native_clap_text_cache_v1", checkpoint=identity,
                    labels=labels, text_encoder_provenance=text_encoder_provenance,
                    features=dict(path=str(path), sha256=_sha(path)),
                    scope="The projected text.semantic bank belongs only to this exact CLAP checkpoint. qwen.raw is a separate frozen text representation.")
    target = directory / "MANIFEST.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return target


def load_native_semantic_cache(manifest_path, *, checkpoint, expected_labels, device):
    from safetensors.torch import load_file
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("contract") != "checkpoint_bound_native_clap_text_cache_v1":
        raise ValueError("Expected a checkpoint-bound native CLAP text cache")
    if manifest["checkpoint"] != _identity(checkpoint):
        raise ValueError("Native text projection belongs to a different CLAP checkpoint")
    labels = list(expected_labels)
    if manifest["labels"] != labels or len(set(labels)) != len(labels):
        raise ValueError("Native text cache labels or order differ")
    ref = manifest["features"]
    if _sha(ref["path"]) != ref["sha256"]:
        raise ValueError("Native text cache feature file changed")
    features = load_file(ref["path"], device=str(device))
    value = features["text.semantic"]
    if value.shape != (len(labels), 512) or not torch.isfinite(value).all():
        raise ValueError("Native text cache has invalid semantic feature geometry")
    return {label: value[i:i+1].detach() for i, label in enumerate(labels)}
