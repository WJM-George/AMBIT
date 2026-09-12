"""Native CLAP44 selection measurements and replayable CPU gate derivations.

Keep the established Editing RF/free-plan floors. Source component tests use
our frozen encoder rather than any M2D cache or caption input.
"""
from __future__ import annotations

from contextlib import nullcontext
import math
from statistics import NormalDist

import torch
from torch.nn import functional as F

from scripts.t2a.eval.select_sceneplan_transfusion_editing_joint_checkpoint import (
    FREE_FLOORS, _free_gate, _JointDonorResolver, _chunks,
)
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import _reject_old_plan_metadata
from stable_audio_tools.data.sceneplan_transfusion_editing_joint_dataset import collate_editing_joint
from stable_audio_tools.data.sceneplan_transfusion_editing import EDIT_OPERATIONS
from stable_audio_tools.data.sceneplan_transfusion_generation_ar_evaluation import score_parsed_generation, score_token_sequence
from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import _align_decoded_sceneplan_to_audio_duration

CONFIDENCE = 0.99
CANDIDATE_STEPS = (5000, 10000, 15000, 20000, 25000)
OPERATIONS = tuple(EDIT_OPERATIONS)
SOURCE_VARIANTS = ("clean", "zero", "shuffled", "latent_zero", "latent_shuffled", "clap_zero", "clap_shuffled")
POLICY = {
    "validation_pairs":20000, "selection_pairs":10000, "holdout_pairs":10000,
    "holdout_candidates_tested":1, "candidate_steps":list(CANDIDATE_STEPS),
    "ranking":["AR_CE_ascending","AR_token_accuracy_descending","RF_MSE_ascending","step_descending"],
    "rf_relative_noninferiority_margin":0.01, "rf_point_ratio_limit":1.005,
    "confidence":CONFIDENCE, "free_ar_pairs":500, "free_ar_max_tokens":512,
    "free_ar_per_operation_bucket":50, "free_ar_floors":FREE_FLOORS,
    "independent_test_used":False, "audio_quality_claim":False,
}


def _moments(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if len(values) < 2 or not torch.isfinite(values).all():
        raise ValueError("selection statistics require at least two finite independent pairs")
    mean = float(values.mean())
    stderr = float(values.std(unbiased=True)) / math.sqrt(len(values))
    return len(values), mean, stderr


def _groups(raw):
    yield "overall", "all", torch.ones(len(raw["ordinals"]), dtype=torch.bool)
    for name in sorted(set(raw["operations"])):
        yield "by_operation", name, torch.tensor([x == name for x in raw["operations"]])
    for bucket in (432, 648):
        yield "by_latent_bucket", str(bucket), torch.tensor([x == bucket for x in raw["buckets"]])


def paired_gate(values, raw):
    values = torch.as_tensor(values, dtype=torch.float64)
    if values.ndim == 2:
        values = values.mean(1)
    output = {"by_operation":{}, "by_latent_bucket":{}}
    for group, name, mask in _groups(raw):
        count, mean, stderr = _moments(values[mask])
        lower = mean - NormalDist().inv_cdf(CONFIDENCE)*stderr
        record = {"rows":count, "mean_difference":mean, "standard_error":stderr,
                  "confidence":CONFIDENCE, "one_sided_lower_confidence_bound":lower, "pass":lower > 0}
        if group == "overall":
            output[group] = record
        else:
            output[group][name] = record
    output["pass"] = all(record["pass"] for record in
                         [output["overall"], *output["by_operation"].values(), *output["by_latent_bucket"].values()])
    return output


def noninferiority(joint, base):
    if joint["ordinals"] != base["ordinals"]:
        raise ValueError("RF comparison is not paired on the same population")
    current = joint["losses"]["clean"].double().mean(1)
    reference = base["losses"]["clean"].double().mean(1)
    output = {"by_operation":{}, "by_latent_bucket":{}}
    for group, name, mask in _groups(joint):
        count, mean, stderr = _moments((current-reference)[mask])
        baseline = float(reference[mask].mean())
        if baseline <= 0:
            raise ValueError("RF baseline must be positive")
        upper = mean+NormalDist().inv_cdf(CONFIDENCE)*stderr
        record = {"rows":count, "mean_difference":mean, "baseline_mean":baseline,
                  "confidence":CONFIDENCE, "one_sided_upper_confidence_bound":upper,
                  "relative_margin":0.01, "absolute_margin":0.01*baseline, "pass":upper <= 0.01*baseline}
        if group == "overall":
            output[group] = record
        else:
            output[group][name] = record
    ratio = float(current.sum()/reference.sum())
    output.update(point_estimate_ratio=ratio, point_estimate_ratio_limit=1.005, point_estimate_pass=ratio <= 1.005)
    output["pass"] = output["point_estimate_pass"] and all(record["pass"] for record in
                        [output["overall"], *output["by_operation"].values(), *output["by_latent_bucket"].values()])
    return output


def subset(raw, folds, fold):
    keep = [i for i, ordinal in enumerate(raw["ordinals"]) if folds[ordinal] == fold]
    def select(value):
        if isinstance(value, torch.Tensor):
            return value[keep]
        if isinstance(value, dict):
            return {key:select(item) for key, item in value.items()}
        if isinstance(value, list):
            return [value[i] for i in keep]
        raise TypeError("unexpected raw evaluation field")
    return {key:select(value) for key, value in raw.items()}


def summary(ar, rf):
    def statistic(raw, values, weights=None):
        result = {"by_operation":{}, "by_latent_bucket":{}}
        values = values.double()
        if values.ndim == 2:
            values = values.mean(1)
        weights = torch.ones_like(values) if weights is None else weights.double()
        for group, name, mask in _groups(raw):
            weight = float(weights[mask].sum())
            if weight <= 0:
                raise ValueError("empty selection stratum")
            record = {"mean":float((values[mask]*weights[mask]).sum()/weight), "rows":int(mask.sum()), "weight":weight}
            if group == "overall":
                result.update(record)
            else:
                result[group][name] = record
        for field in ("source_counts","target_counts"):
            if field not in raw:
                continue
            key = "by_"+field.removesuffix("s")
            result[key] = {}
            for count in sorted(set(raw[field])):
                mask = torch.tensor([x==count for x in raw[field]])
                weight = float(weights[mask].sum())
                result[key][str(count)] = {"mean":float((values[mask]*weights[mask]).sum()/weight),
                                           "rows":int(mask.sum()),"weight":weight}
        return result
    if ar["ordinals"] != rf["ordinals"]:
        raise ValueError("AR and RF summaries must cover identical ordered pairs")
    rf_difficulty = {**rf,**{key:ar[key] for key in ("source_counts","target_counts") if key in ar}}
    return {
        "ar":{"clean_ce":statistic(ar, ar["losses"]["clean"], ar["tokens"]),
              "token_accuracy":statistic(ar, ar["accuracy"], ar["tokens"]),
              "teacher_forced_sequence_exact":statistic(ar, ar["exact"])},
        "rf":statistic(rf_difficulty, rf["losses"]["clean"]),
    }


def rank_candidates(candidates):
    eligible = [row for row in candidates if row["selection_10k"]["base_dit_noninferiority"]["pass"]]
    return sorted(eligible, key=lambda row: (
        row["selection_10k"]["ar"]["clean_ce"]["mean"],
        -row["selection_10k"]["ar"]["token_accuracy"]["mean"],
        row["selection_10k"]["rf"]["mean"], -row["step"],
    ))


def source_gates(ar, rf, *, variant, require_teacher_alignment):
    def ar_component(zero, shuffled):
        return {
            "zero_minus_clean_ce":paired_gate(ar["losses"][zero]-ar["losses"]["clean"], ar),
            "shuffled_minus_clean_ce":paired_gate(ar["losses"][shuffled]-ar["losses"]["clean"], ar),
            "zero_response_l1":paired_gate(ar["response_l1"][zero], ar),
            "shuffled_response_l1":paired_gate(ar["response_l1"][shuffled], ar),
        }
    output = {"ar":ar_component("zero","shuffled"),
              "rf":{**{f"{name}_minus_clean_mse":paired_gate(rf["losses"][name]-rf["losses"]["clean"], rf) for name in ("zero","shuffled")},
                    **{f"{name}_prediction_l1":paired_gate(rf["prediction_l1"][name], rf) for name in ("zero","shuffled")}}}
    full = [*output["ar"].values(), *output["rf"].values()]
    overall = []
    if variant != "latent_only":
        output["ar_latent_component"] = ar_component("latent_zero","latent_shuffled")
        output["ar_clap_component"] = ar_component("clap_zero","clap_shuffled")
        full.extend(output["ar_latent_component"].values())
        # Global semantic feature dependence is required overall; retain every
        # operation/bucket slice as a diagnostic, as in the previous route.
        overall.extend(output["ar_clap_component"].values())
        if require_teacher_alignment:
            output["source_teacher_alignment"] = {
                head:paired_gate(ar["teacher_cosine"][head+"_matched"]-ar["teacher_cosine"][head+"_shuffled"], ar)
                for head in ("semantic","scene")}
            overall.extend(output["source_teacher_alignment"].values())
    output["pass"] = all(x["pass"] for x in full) and all(x["overall"]["pass"] for x in overall)
    output["confidence"] = CONFIDENCE
    return output


def _autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def evaluate_ar(ar, loader, *, device, variants=("clean",), donor_resolver=None):
    if not variants or variants[0] != "clean" or len(set(variants)) != len(variants) or not set(variants) <= set(SOURCE_VARIANTS):
        raise ValueError("invalid native CLAP44 source intervention list")
    encoder = ar.source_clap_model
    if encoder is None and set(variants)-{"clean","zero","shuffled"}:
        raise ValueError("latent baseline has no CLAP component")
    need_donor = any("shuffled" in name for name in variants)
    if need_donor and donor_resolver is None:
        raise ValueError("source interventions require frozen donor mapping")
    output = {"ordinals":[], "operations":[], "buckets":[], "source_counts":[], "target_counts":[],
              "tokens":[], "accuracy":[], "exact":[],
              "losses":{name:[] for name in variants},
              "response_l1":{name:[] for name in variants if name != "clean"}, "teacher_cosine":{}}
    ar.eval().requires_grad_(False)
    for batch in loader:
        rows = batch["metadata"]
        _reject_old_plan_metadata(rows)
        values = {key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in batch["ar"].items()}
        source, mask = values["source_foa_latent"].float(), values["source_attention_mask"].bool()
        context, context_mask = ar.encode_edit_instructions(values["raw_edit_requests"], device=device)
        donor_source = donor_mask = None
        if need_donor:
            donor_source, donor_mask = donor_resolver.values(rows, source.shape[-1], device)
        with _autocast(device):
            features = None if encoder is None else encoder.source_features(source,mask)
            donor_features = None if encoder is None or not need_donor else encoder.source_features(donor_source,donor_mask)
        labels = values["plan_labels"]
        valid = labels.ne(-100)
        counts = valid.sum(1)
        if not bool((counts > 0).all()):
            raise RuntimeError("AR selection has an empty plan label")
        clean_logits = None
        for name in variants:
            x, keep = source, mask
            if name in ("zero","latent_zero"):
                x = torch.zeros_like(source)
            elif name in ("shuffled","latent_shuffled"):
                x, keep = donor_source, donor_mask
            kwargs = {}
            if features is not None:
                kwargs["source_clap_features"] = donor_features if name in ("shuffled","clap_shuffled") else features
                if name in ("zero","clap_zero"):
                    kwargs["source_clap_keep_mask"] = torch.zeros(len(source), device=device, dtype=torch.bool)
            with _autocast(device):
                result = ar(x,keep,values["plan_input_ids"],values["plan_attention_mask"],context,context_mask,
                            return_source_contrastive_query=name=="clean" and features is not None, **kwargs)
            if isinstance(result, tuple):
                logits, query = result
                if donor_features is not None:
                    split = encoder.config.semantic_dim
                    for head, section in (("semantic",slice(None,split)),("scene",slice(split,None))):
                        for pairing, teacher in (("matched",features),("shuffled",donor_features)):
                            cosine = F.cosine_similarity(query[:,section].float(), teacher["global"][:,section].to(query).float(),dim=-1)
                            output["teacher_cosine"].setdefault(head+"_"+pairing,[]).append(cosine.cpu())
            else:
                logits = result
            logits = logits.float()
            loss = F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction="none").reshape_as(labels)
            output["losses"][name].append((loss.sum(1)/counts).cpu())
            if name == "clean":
                clean_logits = logits
                correct = logits.argmax(-1).eq(labels)
                output["accuracy"].append(((correct & valid).sum(1)/counts).cpu())
                output["exact"].append((correct | ~valid).all(1).float().cpu())
            else:
                response = (clean_logits-logits).abs().mean(-1)
                output["response_l1"][name].append(((response*valid).sum(1)/counts).cpu())
        output["tokens"].append(counts.cpu())
        for row in rows:
            count = len(row["model_sceneplan"]["sources"])
            operation = str(row["operation"])
            output["ordinals"].append(int(row["pair_ordinal"]))
            output["operations"].append(operation)
            output["buckets"].append(int(row["latent_bucket_frames"]))
            output["target_counts"].append(count)
            output["source_counts"].append(count-1 if operation=="event_addition" else count+1 if operation=="event_removal" else count)
    for key in ("tokens","accuracy","exact"):
        output[key] = torch.cat(output[key])
    for key in ("losses","response_l1","teacher_cosine"):
        output[key] = {name:torch.cat(value) for name,value in output[key].items()}
    return output


def generate_with_fallback(ar, codec, source, mask, instructions, durations):
    try:
        with _autocast(source.device), torch.no_grad():
            values = ar.generate_batch(source,mask,instructions,codec=codec,max_plan_tokens=512,fixed_duration_sec=durations)
        if len(values) != len(instructions):
            raise RuntimeError("AR omitted output rows")
        return [(value.tolist(),None) for value in values]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        if source.device.type == "cuda":
            torch.cuda.empty_cache()
        if len(instructions) == 1:
            return [(None,error)]
        middle = len(instructions)//2
        return generate_with_fallback(ar,codec,source[:middle],mask[:middle],instructions[:middle],durations[:middle]) + generate_with_fallback(ar,codec,source[middle:],mask[middle:],instructions[middle:],durations[middle:])


def score_free_record(record, codec):
    """Derive scores from stored complete token sequences, never saved scores."""
    result = {key:value for key,value in record.items() if key not in ("metrics","predicted_sceneplan")}
    result["metrics"] = {}
    tokens = record.get("predicted_token_ids")
    if tokens is None:
        result["status"] = "generation_error"
        return result
    target_tokens = record["target_token_ids"]
    result["metrics"].update(score_token_sequence(target_tokens,tokens,codec))
    try:
        predicted = _align_decoded_sceneplan_to_audio_duration(codec.decode(tokens,sample_id=record["target_sample_id"]),record["duration_sec"])
        target = _align_decoded_sceneplan_to_audio_duration(codec.decode(target_tokens,sample_id=record["target_sample_id"]),record["duration_sec"])
        result["metrics"].update(score_parsed_generation(target,predicted))
        result.update(predicted_sceneplan=predicted,status="ok",error=None)
    except Exception as exc:
        result.update(status="parse_error",error=f"{type(exc).__name__}: {exc}")
    return result


@torch.no_grad()
def free_ar_pass(ar, codec, dataset, *, device, batch_size):
    records = []
    ar.eval().requires_grad_(False)
    for bucket in (432,648):
        for indices in _chunks(dataset.length_bucket_indices().get(bucket,()),batch_size):
            samples = [dataset[index] for index in indices]
            batch = collate_editing_joint(samples,pad_id=codec.pad_id)
            rows, values = batch["metadata"], batch["ar"]
            _reject_old_plan_metadata(rows)
            outputs = generate_with_fallback(ar,codec,values["source_foa_latent"].to(device).float(),
                        values["source_attention_mask"].to(device).bool(),values["raw_edit_requests"],
                        [float(row["seconds_total"]) for row in rows])
            for index,(tokens,error) in enumerate(outputs):
                row = rows[index]
                record = {"pair_ordinal":int(row["pair_ordinal"]), "pair_id":str(row["pair_id"]),
                          "operation":str(row["operation"]), "latent_bucket_frames":bucket,
                          "target_sample_id":str(row["target_sample_id"]), "duration_sec":float(row["seconds_total"]),
                          "predicted_token_ids":tokens, "target_token_ids":samples[index][2]["target_token_ids"].tolist(),
                          "error":error}
                records.append(score_free_record(record,codec))
    return records
