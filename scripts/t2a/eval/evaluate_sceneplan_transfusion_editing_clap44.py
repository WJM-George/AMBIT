#!/usr/bin/env python3
"""Frozen validation diagnostics for CLAP44. This does not approve an AR model."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext, ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader

from stable_audio_tools.data.sceneplan_transfusion_editing_clap44 import EditingCLAP44Dataset, collate_clap44
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_io import file_sha256, load_clap44_checkpoint
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_text import FrozenCLAP44TextFeatures
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_metrics import margin_summary, retrieval_metrics


def validation_ordinals(index, pairs, seed):
    """Freeze a balanced operation/duration diagnostic subset, or all 20k."""
    if not 1 <= pairs <= 20000: raise ValueError("validation pairs must be in [1,20000]")
    if pairs == 20000: return list(range(20000))
    groups = defaultdict(list)
    with sqlite3.connect(Path(index).as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        for ordinal, pair_id, operation, bucket in db.execute("SELECT pair_ordinal,pair_id,operation,latent_bucket_frames FROM pairs"):
            key = hashlib.sha256(f"clap44-val-{seed}-{pair_id}".encode()).hexdigest()
            groups[(operation, bucket)].append((key, ordinal))
    if pairs < len(groups): raise ValueError("diagnostic subset must cover every operation/duration stratum")
    if sum(len(group) for group in groups.values()) < pairs:
        raise ValueError("validation index has insufficient rows")
    ordered = [iter(sorted(groups[key])) for key in sorted(groups)]
    selected = []
    while len(selected) < pairs:
        for group in ordered:
            value = next(group, None)
            if value is not None: selected.append(value[1])
            if len(selected) == pairs: break
    return sorted(selected)


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-pairs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retrieval-chunk", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.batch_pairs < 1 or args.workers < 0 or args.retrieval_chunk < 1:
        raise ValueError("invalid validation batching")
    index = args.validation_index.resolve(strict=True)
    from scripts.t2a.train.editing_gpu_runtime import gpu_topology, gpu_lease, configure_visibility
    with ExitStack() as resources:
        if args.device == "cuda":
            topology = gpu_topology()
            if len(topology["devices"]) != 1:
                raise ValueError("CLAP44 extraction currently needs one explicitly selected GPU")
            if torch.cuda.is_initialized():
                raise RuntimeError("select the evaluation GPU before initializing CUDA")
            resources.enter_context(gpu_lease(topology))
            configure_visibility(topology)
        evaluate(args, index)


def evaluate(args, index):
    device = torch.device(args.device)
    preflight = json.loads(args.preflight.read_text())
    val = preflight["indices"]["validation"]
    if preflight.get("status") != "PASS" or val["rows"] != 20000 or str(index) != val["path"] or file_sha256(index) != val["sha256"]:
        raise RuntimeError("CLAP44 validation must use the approved frozen 20k index")
    model, checkpoint = load_clap44_checkpoint(args.checkpoint, device=device)
    contract = checkpoint["contract"]
    if contract["preflight_sha256"] != file_sha256(args.preflight):
        raise RuntimeError("CLAP44 train and validation preflight identities differ")
    for filename, expected in contract["text_files"].items():
        if file_sha256(filename) != expected: raise RuntimeError(f"frozen CLAP44 text asset changed: {filename}")
    cfg = contract["config"]["text"]
    text_encoder = FrozenCLAP44TextFeatures(cfg["model_path"], hidden_dim=model.config.text_dim, max_tokens=cfg["max_tokens"], batch_size=cfg["batch_size"])
    ordinals = validation_ordinals(index, args.pairs, args.seed)
    dataset = EditingCLAP44Dataset(index, expected_rows=20000, row_ordinals=ordinals)
    if dataset.index_sha256 != val["sha256"]:
        raise RuntimeError("CLAP44 validation marker and preflight SHA256 disagree")
    loader = DataLoader(dataset, batch_size=args.batch_pairs, num_workers=args.workers, collate_fn=collate_clap44, shuffle=False,
        generator=torch.Generator().manual_seed(args.seed+909001),
        multiprocessing_context="spawn" if args.workers else None)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "editing_clap44_validation_manifest_v1", "validation_index_sha256": val["sha256"], "seed": args.seed, "pair_ordinals": ordinals}
    manifest_path = args.output / "VALIDATION_MANIFEST.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError("CLAP44 output contains another validation population")
    report_path = args.output / "REPORT.json"
    if report_path.exists(): raise RuntimeError("preserve the existing report; choose a separate candidate output")
    atomic_json(manifest_path, manifest)
    banks = {side: {head: [] for head in ("semantic", "scene")} for side in ("audio", "text")}
    labels = []; binding_by_kind = defaultdict(dict); binding_all = {}; edit_by_operation = defaultdict(list)
    for batch_number, batch in enumerate(loader):
        current = batch["labels"]; n = len(current)
        captions = [x["semantic_text"] for x in current] + [x["scene_text"] for x in current] + batch["negative_scene_texts"]
        context = torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else nullcontext()
        with context:
            features = text_encoder(captions, device)
            audio, text = model(batch["latent"].to(device), batch["mask"].to(device), features[:n], features[n:2*n])
            if batch["negative_scene_texts"]:
                negative = model.encode_text_features(features[2*n:], features[2*n:])["scene"]
                owners = batch["negative_owners"].to(device)
                margin = (audio["scene"] * text["scene"]).sum(-1)[owners] - (audio["scene"][owners] * negative).sum(-1)
                for owner, kind, value in zip(owners.tolist(), batch["negative_kinds"], margin.tolist()):
                    pair = current[owner]["pair_id"]
                    binding_by_kind[kind][pair] = min(binding_by_kind[kind].get(pair, float("inf")), value)
                    binding_all[pair] = min(binding_all.get(pair, float("inf")), value)
        for side, output in (("audio", audio), ("text", text)):
            for head in banks[side]: banks[side][head].append(output[head].float().cpu())
        for i in range(0, n, 2):
            if current[i]["scene_key"] == current[i+1]["scene_key"]: continue
            similarity = audio["scene"][i:i+2] @ text["scene"][i:i+2].T
            # Both directions must recognize which bound scene belongs to
            # which audio. One independent editing pair contributes one vote.
            margin = min(float(similarity[0,0]-similarity[0,1]), float(similarity[1,1]-similarity[1,0]), float(similarity[0,0]-similarity[1,0]), float(similarity[1,1]-similarity[0,1]))
            edit_by_operation[current[i]["operation"]].append(margin)
        labels.extend(current)
        if batch_number % 100 == 0: print(f"CLAP44_VALIDATION views={len(labels)}/{2*len(dataset)}", flush=True)
    retrieval = {}
    for head in ("semantic", "scene"):
        audio = torch.cat(banks["audio"][head]).to(device); text = torch.cat(banks["text"][head]).to(device)
        keys = [x[f"{head}_key"] for x in labels]
        retrieval[head] = {"audio_to_text": retrieval_metrics(audio, text, keys, keys, chunk_size=args.retrieval_chunk), "text_to_audio": retrieval_metrics(text, audio, keys, keys, chunk_size=args.retrieval_chunk)}
    report = {
        "status": "DIAGNOSTICS_COMPLETE_NOT_QUALITY_PASS", "observed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {k: checkpoint[k] for k in ("path", "sha256", "step")},
        "validation_manifest_sha256": file_sha256(manifest_path), "validation_index_sha256": val["sha256"],
        "pairs": len(dataset), "audio_views": len(labels), "full_validation": len(dataset) == 20000,
        "retrieval": retrieval, "paired_edit_discrimination": {k: margin_summary(v) for k,v in edit_by_operation.items()},
        "binding_discrimination": {k: margin_summary(list(v.values())) for k,v in binding_by_kind.items()},
        "all_binding_negatives_correct_per_pair": margin_summary(list(binding_all.values())),
        "binding_pair_coverage": len(binding_all)/len(dataset), "binding_aggregation": "worst margin over all source/target counterfactuals per editing pair",
        "evaluation_source_sha256": {str(p): file_sha256(p) for p in (Path(__file__).resolve(), ROOT / "stable_audio_tools/training/sceneplan_transfusion_editing_clap44_metrics.py")},
        "independent_test_used": False, "quality_gate_passed": False,
        "next": "Matched AR full-plan generation and decoded FOA editing ablations; this report is not an AR or DiT gate.",
    }
    atomic_json(report_path, report)
    print(json.dumps({"report": str(report_path), "status": report["status"], "pairs": len(dataset)}), flush=True)


if __name__ == "__main__": main()
