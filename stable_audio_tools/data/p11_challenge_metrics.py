"""Common prefix aggregation for full and sharded P11 challenge reports."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

def _mean(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _aggregate(
    rows: Sequence[Mapping[str, Any]], k_values: Sequence[int]
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["family"])].append(row)
        groups[f"task:{row['task']}"].append(row)
        if row.get("edit_operation") is not None:
            groups[f"edit_operation:{row['edit_operation']}"].append(row)
        groups["all"].append(row)
    output: dict[str, Any] = {}
    for group, values in sorted(groups.items()):
        output[group] = {}
        for k in k_values:
            prefixes = [
                row["prefixes"][str(k)]
                for row in values
                if str(k) in row.get("prefixes", {})
            ]
            if not prefixes:
                continue
            output[group][str(k)] = {
                "rows": len(prefixes),
                "valid_rate": _mean([float(value["valid_rate"]) for value in prefixes]),
                "task_score_mean": _mean(
                    [float(value["task_score_mean"]) for value in prefixes]
                ),
                "oracle_best_of_k_task_score": _mean(
                    [float(value["oracle_best_of_k_task_score"]) for value in prefixes]
                ),
                "semantic_immutability_rate": _mean(
                    [float(bool(value["semantic_immutability"])) for value in prefixes]
                ),
                "nondegenerate_numeric_rate": _mean(
                    [float(int(value["unique_numeric_count"]) > 1) for value in prefixes]
                ),
                "numeric_pairwise_rmse": _mean(
                    [float(value["numeric_pairwise_rmse"]) for value in prefixes]
                ),
                "constraint_pass_rate": _mean(
                    [
                        float(value["constraint_pass_rate"])
                        for value in prefixes
                        if value["constraint_pass_rate"] is not None
                    ]
                ),
                "constraint_pass_any_rate": _mean(
                    [
                        float(bool(value["constraint_pass_any"]))
                        for value in prefixes
                        if value["constraint_pass_any"] is not None
                    ]
                ),
                "reference_anchor_nearest_core_rmse": _mean(
                    [
                        float(value["reference_anchor_nearest_core_rmse_mean"])
                        for value in prefixes
                        if value["reference_anchor_nearest_core_rmse_mean"]
                        is not None
                    ]
                ),
                "calibration": {
                    key: _mean(
                        [
                            float(value["calibration"][key])
                            for value in prefixes
                            if value.get("calibration") is not None
                            and value["calibration"].get(key) is not None
                        ]
                    )
                    for key in (
                        "ensemble_mean_rmse",
                        "spread_mean",
                        "coverage_68",
                        "coverage_95",
                        "absolute_calibration_error_68",
                        "absolute_calibration_error_95",
                    )
                },
            }
    return output


