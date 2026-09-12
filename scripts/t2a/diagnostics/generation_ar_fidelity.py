"""Request-field diagnostics with explicit missing-source denominators.

These are diagnostic tolerance sweeps, not user-approved acceptance thresholds.
Description token F1 is reported by the original evaluator as a lexical proxy;
this module deliberately does not label it semantic accuracy.
"""
from __future__ import annotations

from collections import Counter
import math

TOLERANCES = {
    "strict": {"time_s": .10, "azimuth_deg": 5., "elevation_deg": 3., "distance_m": .10},
    "medium": {"time_s": .25, "azimuth_deg": 10., "elevation_deg": 5., "distance_m": .25},
    "loose": {"time_s": .50, "azimuth_deg": 20., "elevation_deg": 10., "distance_m": .50},
}


def endpoints(source):
    trajectory = source["trajectory"]
    if trajectory["type"] == "static":
        return trajectory["position"], trajectory["position"]
    if trajectory["type"] == "linear":
        return trajectory["start"], trajectory["end"]
    return trajectory["keyframes"][0]["position"], trajectory["keyframes"][-1]["position"]


def compare_fields(target, prediction):
    refs = target["sources"]
    preds = prediction["sources"] if prediction is not None else []
    by_id = {s["source_id"]: s for s in preds}
    if len(by_id) != len(preds):
        raise ValueError("duplicate predicted source_id")
    target_ids = {s["source_id"] for s in refs}
    sources = []
    for ref in refs:
        hyp = by_id.get(ref["source_id"])
        item = {"source_id": ref["source_id"], "matched": hyp is not None,
                "motion_correct": False, "kind_correct": False, "errors": {},
                "passes": {tier: False for tier in TOLERANCES}}
        if hyp is not None:
            item["motion_correct"] = ref["trajectory"]["type"] == hyp["trajectory"]["type"]
            item["kind_correct"] = ref["kind"] == hyp["kind"]
            errors = item["errors"]
            for boundary in ("onset", "offset"):
                errors[f"{boundary}_s"] = abs(ref["activity"][f"{boundary}_sec"] - hyp["activity"][f"{boundary}_sec"])
            for label, left, right in zip(("start", "end"), endpoints(ref), endpoints(hyp)):
                errors[f"{label}_azimuth_deg"] = abs((left["azimuth_deg"] - right["azimuth_deg"] + 180) % 360 - 180)
                errors[f"{label}_elevation_deg"] = abs(left["elevation_deg"] - right["elevation_deg"])
                errors[f"{label}_distance_m"] = abs(left["distance_m"] - right["distance_m"])
            if not all(math.isfinite(x) for x in errors.values()):
                raise ValueError("nonfinite field error")
            for tier, limits in TOLERANCES.items():
                item["passes"][tier] = item["motion_correct"] and item["kind_correct"] and all(
                    error <= limits["time_s" if field in ("onset_s", "offset_s") else field.split("_", 1)[1]]
                    for field, error in errors.items()
                )
        sources.append(item)
    count_correct = prediction is not None and len(refs) == len(preds)
    return {"target_count": len(refs), "predicted_count": len(preds) if prediction is not None else None,
            "count_correct": count_correct, "missing_sources": sum(not s["matched"] for s in sources),
            "extra_sources": len(set(by_id) - target_ids), "sources": sources,
            "scene_spatiotemporal_pass": {tier: count_correct and all(s["passes"][tier] for s in sources)
                                          for tier in TOLERANCES}}


def _percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def summarize_fields(records):
    if not records:
        raise ValueError("empty fidelity panel")
    sources = [s for r in records for s in r["sources"]]
    matched = [s for s in sources if s["matched"]]
    errors = {}
    for key in sorted({k for s in matched for k in s["errors"]}):
        values = [s["errors"][key] for s in matched]
        limit_key = "time_s" if key in ("onset_s", "offset_s") else key.split("_", 1)[1]
        errors[key] = {"matched_source_mae": sum(values) / len(values),
                       "matched_source_p90": _percentile(values, .90),
                       "matched_sources": len(values), "requested_sources": len(sources),
                       "within_tolerance_per_requested_source": {
                           tier: sum(v <= limits[limit_key] for v in values) / len(sources)
                           for tier, limits in TOLERANCES.items()}}
    confusion = {str(k): dict(Counter(str(r["predicted_count"]) for r in records if r["target_count"] == k))
                 for k in range(1, 5)}
    return {"scenes": len(records), "requested_sources": len(sources), "matched_sources": len(matched),
            "source_count_accuracy": sum(r["count_correct"] for r in records) / len(records),
            "count_confusion": confusion, "missing_sources": sum(r["missing_sources"] for r in records),
            "extra_sources": sum(r["extra_sources"] for r in records),
            "motion_accuracy_per_requested_source": sum(s["motion_correct"] for s in sources) / len(sources),
            "kind_accuracy_per_requested_source": sum(s["kind_correct"] for s in sources) / len(sources),
            "field_errors": errors,
            "scene_spatiotemporal_pass_rate": {tier: sum(r["scene_spatiotemporal_pass"][tier] for r in records) / len(records)
                                                 for tier in TOLERANCES},
            "tolerances": TOLERANCES,
            "limits": "Persistent IDs; absent sources fail all tolerance rates. MAE/P90 explicitly condition on matched sources. Description semantics and intermediate curved-path geometry are not included in the spatiotemporal conjunction."}
