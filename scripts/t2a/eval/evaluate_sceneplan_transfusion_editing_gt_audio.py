#!/usr/bin/env python3
"""Require sampled, decoded GT-plan Editing evidence before starting AR.

Validation only: this entry point has no test-index or threshold override.
RF checkpoint selection remains necessary. This additional gate evaluates the
selected EMA with the exact shared DiT sampler and frozen FOA codec.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Mapping, Sequence

import torch
from torch import distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_audio_end_to_end as audio
from scripts.t2a.eval import select_sceneplan_transfusion_editing_dit_checkpoint as rf
from stable_audio_tools.configuration import load_config
from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import ScenePlanTransfusionEditingDataset
from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import (
    ScenePlanTransfusionEditingDiTPipeline,
    _load_frozen_foa_vae,
    load_sceneplan_transfusion_editing_pipeline,
)
from stable_audio_tools.training.factory import create_training_wrapper_from_config

SCHEMA = "sceneplan_transfusion_editing_gt_audio_gate_v1"
POLICY = "validation_5x2x100_gt_plan_real_source_ema_euler20_cfg1_reference_audio_v1"
VARIANTS = ("clean", "zero", "shuffled")
HIGHER = {key: deepcopy(value) for key, value in audio.HIGHER_BETTER_SPECS.items()
          if not key.startswith("plan_")}
LOWER = deepcopy(audio.LOWER_BETTER_SPECS)
for variant in VARIANTS[1:]:
    HIGHER[f"reference_{variant}_audio_gain"] = {
        "floor": 0.02, "tolerance": 0.02, "groups": audio.FULL_QUALITY_GROUPS,
    }
    HIGHER[f"reference_{variant}_latent_gain"] = {
        "floor": 0.02, "tolerance": 0.02, "groups": audio.FULL_QUALITY_GROUPS,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def _artifact(path: Path) -> dict[str, str]:
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def _verify_artifact(value: Mapping[str, Any]) -> Path:
    path = Path(value["path"]).resolve(strict=True)
    if sha256_file(path) != value["sha256"]:
        raise RuntimeError(f"GT audio artifact changed: {path}")
    return path


def _code_hashes() -> dict[str, str]:
    paths = set(audio.AUDITED_SOURCE_PATHS) | {
        str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "scripts/t2a/eval/run_sceneplan_transfusion_editing_gt_audio_5gpu.sh",
        "scripts/t2a/train/run_sceneplan_transfusion_editing_dit_full_5gpu.sh",
        "scripts/t2a/train/run_sceneplan_transfusion_editing_ar_joint_full_5gpu.sh",
    }
    return {path: sha256_file(REPO_ROOT / path) for path in sorted(paths)}


def _policy() -> dict[str, Any]:
    value = {"name": POLICY, "rows": 1000, "split": "validation",
            "higher": HIGHER, "lower": LOWER, "confidence": audio.CONFIDENCE,
            "source_variants": list(VARIANTS), "ode_steps": 20, "cfg_scale": 1.0,
            "batch_size_per_rank": 1, "seed": 42,
            "plan_origin": "ground_truth", "old_plan_model_input": False,
            "reference_donor": "different_source_same_bucket_nearest_longer_audio_crop_sha256_tiebreak_v1",
            "vae_reference": "fresh_raw_foa_per_pair_seed_fp16_boundary_v1"}
    # Metric groups are tuples in the shared Python scorer and arrays in JSON.
    # Keep the in-memory identity equal to its durable representation on resume.
    return json.loads(json.dumps(value))


def _validation_rows(index: Path) -> tuple[list[dict[str, Any]], list[int], dict[int, int]]:
    layout = audio._layout(index, expected_rows=20_000, expected_split="validation")
    selected, _ = audio._select_ordinals(layout, phase="calibration")
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    try:
        extra = {int(ordinal): (str(source), int(samples)) for ordinal, source, samples
                 in connection.execute("SELECT pair_ordinal,source_sample_id,model_num_samples FROM pairs")}
    finally:
        connection.close()
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in layout:
        row["source_sample_id"], row["model_num_samples"] = extra[row["pair_ordinal"]]
        groups[row["latent_bucket_frames"]].append(row)
    donors = {}
    for ordinal in selected:
        row = layout[ordinal]
        candidates = [other for other in groups[row["latent_bucket_frames"]]
                      if other["source_sample_id"] != row["source_sample_id"]
                      and other["model_num_samples"] >= row["model_num_samples"]]
        if not candidates:
            raise RuntimeError(f"no independent reference with exact duration for {ordinal}")
        donor = min(candidates, key=lambda other: (
            other["model_num_samples"] - row["model_num_samples"],
            hashlib.sha256(f"gt-audio-donor-v1:{row['pair_id']}:{other['pair_id']}".encode()).hexdigest()))
        donors[ordinal] = donor["pair_ordinal"]
    return layout, selected, donors


def _selected_identity(selection_path: Path) -> dict[str, Any]:
    selection_path = selection_path.resolve(strict=True)
    advertised = json.loads(selection_path.read_text())
    run_dir = Path(advertised["training_run"]["run_dir"]).resolve(strict=True)
    train_path = run_dir / rf.TRAIN_CONTRACT_NAME
    train_contract, train_sha = rf.validate_training_run_contract(train_path, expected_run_dir=run_dir)
    model = Path(advertised["model_config"]["path"]).resolve(strict=True)
    index = Path(advertised["validation_index"]["path"]).resolve(strict=True)
    selected = rf._validate_existing_selection(
        selection_path, run_dir=run_dir,
        preflight=Path(advertised["preflight"]["path"]).resolve(strict=True),
        model_config=model, validation_index=index,
        training_run_contract=train_contract, training_run_contract_path=train_path,
        training_run_contract_sha256=train_sha,
    )
    if selected is None or selected.get("status") != "PASS":
        raise RuntimeError("a replay-validated RF promotion is required before GT audio")
    return {"selection": _artifact(selection_path), "checkpoint": _artifact(Path(selected["selected_checkpoint"])),
            "step": int(selected["selected_checkpoint_step"]), "model_config": _artifact(model),
            "validation_index": _artifact(index),
            "validation_marker": _artifact(index.with_suffix(".sqlite.frozen.json")),
            "training_contract": train_contract, "training_contract_file": _artifact(train_path)}


def _check_record(record: Mapping[str, Any], row: Mapping[str, Any], contract_sha: str) -> None:
    if not (record.get("schema") == SCHEMA and record.get("status") == "ok"
            and record.get("contract_sha256") == contract_sha
            and record.get("plan_origin") == "ground_truth"
            and record.get("pair_ordinal") == row["pair_ordinal"]
            and record.get("pair_id") == row["pair_id"]
            and record.get("operation") == row["operation"]
            and record.get("latent_bucket_frames") == row["latent_bucket_frames"]
            and record.get("model_input_contract", {}).get("editing_ar") is None
            and record.get("model_input_contract", {}).get("old_sceneplan") is False
            and set(record.get("variant_audio", {})) == set(VARIANTS)):
        raise RuntimeError("GT audio row identity/route changed")
    audio._finite_metrics(record["metrics"])
    if any(key.startswith("plan_") for key in record["metrics"]):
        raise RuntimeError("GT plan scores cannot count as free AR evidence")
    for artifact in record["variant_audio"].values():
        _verify_artifact(artifact)
    if record["variant_audio"]["clean"] != {
        "path": record["edited_foa_path"], "sha256": record["edited_foa_sha256"]
    }:
        raise RuntimeError("clean reference scores/audio binding changed")


def _summarize(records: Sequence[Mapping[str, Any]], *, baseline=None) -> dict[str, Any]:
    summaries = audio._all_metric_summaries(records)
    # Missing/non-finite metrics, insufficient coverage and small groups fail
    # at the same fixed bounds as the existing final audio evaluator.
    for name in set(HIGHER) | set(LOWER):
        summaries.setdefault(name, audio._metric_summary(records, name))
    thresholds, checks = audio._calibration_thresholds(summaries, higher_specs=HIGHER, lower_specs=LOWER)
    if baseline is not None:
        checks.update({f"post_joint_nonregression:{key}": passed for key, passed in
                       audio._test_threshold_checks(summaries, baseline["result"]["thresholds"]).items()})
    difficulty = {}
    for field in ("source_count", "unchanged_source_count", "domain_pair"):
        difficulty[field] = {}
        for value in sorted({str(row["difficulty"][field]) for row in records}):
            rows = [row for row in records if str(row["difficulty"][field]) == value]
            difficulty[field][value] = {
                "rows": len(rows), "fraction": len(rows) / len(records),
                "metrics": {metric: audio._metric_summary(rows, metric)["overall"]
                            for metric in ("audio_codec_foa_progress", "doa_target_mean_deg",
                                           "unchanged_preservation_budget_ratio",
                                           "reference_shuffled_audio_gain")},
            }
    return {"status": "PASS" if checks and all(checks.values()) else "FAIL",
            "rows": len(records), "metrics": summaries, "difficulty": difficulty,
            "thresholds": thresholds, "checks": checks,
            "failed_checks": [key for key, passed in checks.items() if not passed],
            "worst_examples": [row["pair_ordinal"] for row in sorted(
                records, key=lambda row: row["metrics"].get("audio_codec_foa_progress", -1e30))[:50]]}


def validate_audio_gate(path: Path, *, selection_path: Path, joint_selection_path: Path | None = None) -> dict[str, Any]:
    """Replay seals and fixed checks; a hand-written PASS cannot promote AR."""
    path = path.resolve(strict=True)
    value = json.loads(path.read_text())
    contract_path = _verify_artifact(value["contract"])
    contract = json.loads(contract_path.read_text())
    phase = "pre_joint" if joint_selection_path is None else "post_joint"
    if not (value.get("schema") == SCHEMA and contract.get("schema") == SCHEMA
            and contract.get("phase") == phase
            and contract.get("policy") == _policy() and contract.get("source_sha256") == _code_hashes()
            and contract["identity"]["selection"] == _artifact(selection_path)):
        raise RuntimeError("GT audio policy/code/selection binding changed")
    identity = contract["identity"]
    baseline = None
    if joint_selection_path is not None:
        if identity.get("joint_selection") != _artifact(joint_selection_path):
            raise RuntimeError("post-joint audio checkpoint selection changed")
        _verify_artifact(identity["joint_checkpoint"])
        _verify_artifact(identity["joint_codec_config"])
        baseline = validate_audio_gate(_verify_artifact(identity["pre_joint_gate"]), selection_path=selection_path)
    for key in ("checkpoint", "model_config", "validation_index", "validation_marker", "training_contract_file"):
        _verify_artifact(identity[key])
    if contract["content_assets"] != audio.verify_independent_content_metric_assets():
        raise RuntimeError("independent GT audio scorer assets changed")
    if contract["vae_assets"] != {
        "config": _artifact(audio.FROZEN_VAE_CONFIG),
        "checkpoint": _artifact(audio.FROZEN_VAE_CHECKPOINT),
    }:
        raise RuntimeError("GT audio VAE assets changed")
    layout, selected, donors = _validation_rows(Path(identity["validation_index"]["path"]))
    if contract["selected_ordinals"] != selected or contract["donors"] != {str(k): v for k, v in donors.items()}:
        raise RuntimeError("GT audio fixed sample/reference selection changed")
    records = []
    if len(value["record_artifacts"]) != len(selected):
        raise RuntimeError("GT audio gate lacks all 1000 records")
    for artifact, ordinal in zip(value["record_artifacts"], selected):
        record = json.loads(_verify_artifact(artifact).read_text())
        _check_record(record, layout[ordinal], _digest(contract))
        if record.get("donor_ordinal") != donors[ordinal]:
            raise RuntimeError("GT audio reference donor changed")
        records.append(record)
    recomputed = _summarize(records, baseline=baseline)
    if value.get("result") != recomputed or recomputed["status"] != "PASS":
        raise RuntimeError("GT audio fixed quality/reference gate did not pass")
    return value


@torch.no_grad()
def _evaluate_row(*, pipeline, scorer, sample, truth, donor_truth, device, output_dir, contract_sha, canonical_codec=None):
    target, metadata = sample
    ordinal = int(metadata["pair_ordinal"])
    bucket = int(metadata["latent_bucket_frames"])
    count = int(metadata["model_num_samples"])
    frames = int(metadata["latent_frames_valid"])
    if not (truth["pair_id"] == metadata["pair_id"]
            and truth["offline_new_sceneplan"] == metadata["model_sceneplan"]
            and donor_truth["source_sample_id"] != truth["source_sample_id"]
            and donor_truth["model_num_samples"] >= count):
        raise RuntimeError("GT audio source/plan/donor alignment changed")
    seed = audio._stable_seed(42, metadata["pair_id"], "source-vae")
    source, mask = pipeline.encode_source_foa(
        audio._pad_audio([truth["source_foa"]], bucket), model_num_samples=[count], vae_seeds=[seed])
    donor, donor_mask = pipeline.encode_source_foa(
        audio._pad_audio([donor_truth["source_foa"]], bucket), model_num_samples=[count],
        vae_seeds=[audio._stable_seed(42, donor_truth["pair_id"], "source-vae")])
    if not torch.equal(mask, donor_mask):
        raise RuntimeError("reference ablation must preserve the exact source mask")
    source_codec, _ = pipeline.decode_foa_latents(source, model_num_samples=[count])
    target_latent = target[None, :, :bucket].to(device).float()
    target_codec, _ = pipeline.decode_foa_latents(target_latent, model_num_samples=[count])
    noise = audio._initial_noise([truth], bucket, 42)
    model_plan = metadata["model_sceneplan"]
    if canonical_codec is not None:
        from stable_audio_tools.data.sceneplan_transfusion_editing_plan import canonicalize_editing_plan
        model_plan, _ = canonicalize_editing_plan(model_plan, codec=canonical_codec)
    variant_audio, errors = {}, {}
    clean_result = None
    for name, reference in zip(VARIANTS, (source, torch.zeros_like(source), donor)):
        edited = pipeline.sample_edited_latents(reference, mask, [model_plan],
            model_num_samples=[count], steps=20, cfg_scale=1.0, initial_noise=noise)
        waveform, waveform_mask = pipeline.decode_foa_latents(edited, model_num_samples=[count])
        audio_path = output_dir / "audio" / name / f"{ordinal:05d}.wav"
        variant_audio[name] = {"path": str(audio_path.resolve()),
                               "sha256": audio._atomic_wav(audio_path, waveform[0, :, :count].cpu())}
        errors[name] = {
            "audio": audio._nmse(waveform[0, :, :count], target_codec[0, :, :count]),
            "latent": audio._nmse(edited[0, :, :frames], target_latent[0, :, :frames]),
        }
        if name == "clean":
            clean_result = {"edited_foa": waveform, "source_codec_foa": source_codec,
                "source_foa_latent": source, "source_attention_mask": mask,
                "sample_attention_mask": waveform_mask, "edited_foa_latent": edited,
                "new_sceneplans": [model_plan]}
    args = argparse.Namespace(seed=42, max_plan_tokens=512, ode_steps=20, cfg_scale=1.0, save_all_audio=False)
    record = audio._process_batch(pipeline=pipeline, content_evaluator=scorer, codec=None,
        samples=[(target, metadata, {})], truths=[truth], bucket=bucket, device=device,
        args=args, output_dir=output_dir, listening_ordinals=set(), sampled_gt_result=clean_result)[0]
    for variant in VARIANTS[1:]:
        for kind in ("audio", "latent"):
            record["metrics"][f"reference_{variant}_{kind}_gain"] = (
                audio._progress(errors["clean"][kind], errors[variant][kind]))
    record.update(schema=SCHEMA, contract_sha256=contract_sha, variant_audio=variant_audio,
        plan_representation="persistent_gt" if canonical_codec is None else "canonical_gt",
        donor_ordinal=int(donor_truth["pair_ordinal"]), reference_errors=errors,
        edited_foa_path=variant_audio["clean"]["path"], edited_foa_sha256=variant_audio["clean"]["sha256"],
        difficulty={"source_count": len(truth["offline_old_sceneplan"]["sources"]),
                    "unchanged_source_count": len(truth["unchanged_source_ids"]),
                    "domain_pair": f"{truth['source_domain']}->{truth['target_domain']}"})
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--joint-selection", type=Path)
    parser.add_argument("--pre-joint-gate", type=Path)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if bool(args.joint_selection) != bool(args.pre_joint_gate):
        raise ValueError("post-joint audio requires both a joint selection and the pre-joint gate")
    if args.verify_only:
        validate_audio_gate(output / "GATE.json", selection_path=args.selection, joint_selection_path=args.joint_selection)
        print(json.dumps({"event": "editing_gt_audio_gate_verified", "status": "PASS"}))
        return 0
    rank, _, device = rf._distributed()
    torch.set_float32_matmul_precision("high")
    def identity_audit():
        identity = _selected_identity(args.selection)
        if args.joint_selection is not None:
            validate_audio_gate(args.pre_joint_gate, selection_path=args.selection)
            selection = audio.validate_published_joint_selection(args.joint_selection)
            if (selection["model_config"]["sha256"] != identity["model_config"]["sha256"]
                    or selection["validation_index"]["sha256"] != identity["validation_index"]["sha256"]):
                raise RuntimeError("joint and base DiT model/data identities differ")
            joint_contract = json.loads(Path(selection["training_run"]["run_contract_path"]).read_text())
            if joint_contract.get("dit_gt_audio_gate", {}).get("sha256") != sha256_file(args.pre_joint_gate):
                raise RuntimeError("joint training did not bind the pre-joint audio gate")
            identity.update(joint_selection=_artifact(args.joint_selection),
                joint_checkpoint=_artifact(Path(selection["selected_checkpoint"])),
                joint_codec_config=_artifact(Path(selection["codec"]["path"]) / "codec.json"),
                pre_joint_gate=_artifact(args.pre_joint_gate))
        return identity
    identity = rf._rank0_audit(identity_audit, rank=rank, device=device)
    def contract_audit():
        layout, selected, donors = _validation_rows(Path(identity["validation_index"]["path"]))
        value = {"schema": SCHEMA, "policy": _policy(), "identity": identity,
            "phase": "pre_joint" if args.joint_selection is None else "post_joint",
            "selected_ordinals": selected, "donors": {str(k): v for k, v in donors.items()},
            "source_sha256": _code_hashes(), "gpu_topology": rf._gpu_topology(),
            "content_assets": audio.verify_independent_content_metric_assets(),
            "vae_assets": {"config": _artifact(audio.FROZEN_VAE_CONFIG),
                           "checkpoint": _artifact(audio.FROZEN_VAE_CHECKPOINT)}}
        path = output / "CONTRACT.json"
        if path.exists() and json.loads(path.read_text()) != value:
            raise RuntimeError("existing GT audio run contract changed; preserve it and diagnose")
        if not path.exists(): audio._atomic_json(path, value)
        return {"contract": value, "layout": layout}
    audit = rf._rank0_audit(contract_audit, rank=rank, device=device)
    contract, layout = audit["contract"], audit["layout"]
    contract_sha = _digest(contract)
    selected = contract["selected_ordinals"]
    assigned = selected[rank::5]
    pending = []
    for ordinal in assigned:
        path = output / "records" / f"{ordinal:05d}.json"
        if path.is_file():
            _check_record(json.loads(path.read_text()), layout[ordinal], contract_sha)
        else:
            pending.append(ordinal)
    if pending:
        config = load_config(Path(identity["model_config"]["path"]))
        canonical_codec = None
        if args.joint_selection is None:
            diffusion = create_model_from_config(config)
            wrapper = create_training_wrapper_from_config(config, diffusion)
            if wrapper.diffusion_ema is None or wrapper.conditioner_ema is None:
                raise RuntimeError("GT audio requires both selected EMA branches")
            rf._load_candidate(wrapper, Path(identity["checkpoint"]["path"]), step=identity["step"],
                resolved_model_config=config, training_run_contract=identity["training_contract"],
                training_run_contract_path=Path(identity["training_contract_file"]["path"]),
                training_run_contract_sha256=identity["training_contract_file"]["sha256"])
            diffusion.model = wrapper.diffusion_ema.ema_model
            diffusion.pretransform = None
            diffusion.eval().requires_grad_(False).to(device)
            vae, _ = _load_frozen_foa_vae(device)
            pipeline = ScenePlanTransfusionEditingDiTPipeline(diffusion=diffusion, audio_autoencoder=vae)
            conditioner_context = wrapper.ema_conditioner_context()
        else:
            pipeline, _ = load_sceneplan_transfusion_editing_pipeline(
                checkpoint=identity["joint_checkpoint"]["path"],
                checkpoint_selection=identity["joint_selection"]["path"],
                checkpoint_selection_sha256=identity["joint_selection"]["sha256"],
                model_config=identity["model_config"]["path"],
                codec=Path(identity["joint_codec_config"]["path"]).parent, device=device)
            diffusion = pipeline.diffusion
            canonical_codec = pipeline.codec
            conditioner_context = nullcontext()
        scorer = audio.IndependentEditingContentEvaluator(device=device, device_index=int(device.index))
        dataset = ScenePlanTransfusionEditingDataset(Path(identity["validation_index"]["path"]),
            tokenizer_spec=(diffusion.conditioner.conditioners["prompt"].tokenizer, 512),
            expected_num_samples=len(pending), index_num_samples=20000,
            expected_index_sha256=identity["validation_index"]["sha256"], sample_ordinals=pending,
            latent_crop_length=648, require_frozen=True, verify_tensor_hashes_on_access=True)
        truth_resolver = audio.OfflineTruthResolver(Path(identity["validation_index"]["path"]))
        try:
            with conditioner_context:
                for offset, ordinal in enumerate(pending):
                    try:
                        record = _evaluate_row(pipeline=pipeline, scorer=scorer, sample=dataset[offset],
                            truth=truth_resolver.row(ordinal),
                            donor_truth=truth_resolver.row(contract["donors"][str(ordinal)]),
                            device=device, output_dir=output, contract_sha=contract_sha, canonical_codec=canonical_codec)
                        _check_record(record, layout[ordinal], contract_sha)
                        audio._atomic_json(output / "records" / f"{ordinal:05d}.json", record)
                    except Exception as error:
                        audio._atomic_json(output / "errors" / f"{ordinal:05d}.json", {
                            "pair_ordinal": ordinal, "contract_sha256": contract_sha,
                            "error": f"{type(error).__name__}: {error}"})
                        raise
                    if (offset + 1) % 10 == 0:
                        print(json.dumps({"event": "editing_gt_audio_progress", "rank": rank,
                            "completed_this_attempt": offset + 1, "pending_at_start": len(pending)}), flush=True)
        finally:
            truth_resolver.close()
    dist.barrier()
    def publish():
        records, artifacts = [], []
        for ordinal in selected:
            path = output / "records" / f"{ordinal:05d}.json"
            record = json.loads(path.read_text())
            _check_record(record, layout[ordinal], contract_sha)
            records.append(record)
            artifacts.append(_artifact(path))
        baseline = None if args.pre_joint_gate is None else json.loads(args.pre_joint_gate.read_text())
        result = _summarize(records, baseline=baseline)
        value = {"schema": SCHEMA, "contract": _artifact(output / "CONTRACT.json"),
                 "record_artifacts": artifacts, "result": result}
        audio._atomic_json(output / "GATE.json", value)
        audio._atomic_json(output / "LISTENING.json", {
            "worst_examples": result["worst_examples"],
            "representative_examples": [next(row["pair_ordinal"] for row in records
                if row["operation"] == operation and row["latent_bucket_frames"] == bucket)
                for operation in audio.OPERATIONS for bucket in (432, 648)],
            "all_variant_audio_saved": True})
        return {"status": result["status"], "failed_checks": result["failed_checks"]}
    result = rf._rank0_audit(publish, rank=rank, device=device)
    if rank == 0: print(json.dumps({"event": "editing_gt_audio_complete", **result}), flush=True)
    dist.destroy_process_group()
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
