#!/usr/bin/env python3
"""Prove that canonical Flow and Direct differ only in continuous objective."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.configuration import load_config  # noqa: E402


CONFIG_ROOT = REPO_ROOT / (
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11"
)
DEFAULT_FLOW = CONFIG_ROOT / "qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json"
DEFAULT_DIRECT = CONFIG_ROOT / "qwen35_0p8b_sceneplan_p11_baseline_direct_mse.json"
ALLOWED_DIFFERENCES = {
    "_task",
    "model.transfusion_cot.thought.arm",
    "model.transfusion_cot.thought.continuous_objective",
    "training.transfusion_cot_loss_weights.flow",
}
INVENTORY_CONTRACT = "u_same_decoder_finite_inventory_aux_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": path, "flow": left, "direct": right}]
    if isinstance(left, dict):
        output = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                output.append(
                    {
                        "path": child,
                        "flow": left.get(key, "<MISSING>"),
                        "direct": right.get(key, "<MISSING>"),
                    }
                )
            else:
                output.extend(_differences(left[key], right[key], child))
        return output
    if isinstance(left, list):
        return [] if left == right else [{"path": path, "flow": left, "direct": right}]
    return [] if left == right else [{"path": path, "flow": left, "direct": right}]


def _inventory(config: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(
        config["model"]["transfusion_cot"]["discrete_supervision"][
            "understanding_inventory"
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-config", type=Path, default=DEFAULT_FLOW)
    parser.add_argument("--direct-config", type=Path, default=DEFAULT_DIRECT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    flow_path = args.flow_config.expanduser().resolve(strict=True)
    direct_path = args.direct_config.expanduser().resolve(strict=True)
    flow = load_config(flow_path)
    direct = load_config(direct_path)
    differences = _differences(flow, direct)
    observed_paths = {item["path"] for item in differences}
    flow_thought = flow["model"]["transfusion_cot"]["thought"]
    direct_thought = direct["model"]["transfusion_cot"]["thought"]
    flow_inventory = _inventory(flow)
    direct_inventory = _inventory(direct)
    gates = {
        "exact_allowed_difference_set": observed_paths == ALLOWED_DIFFERENCES,
        "flow_arm": flow_thought.get("arm") == "sketch_first_transfusion_cot_v4",
        "flow_objective": flow_thought.get("continuous_objective")
        == "rectified_flow",
        "direct_arm": direct_thought.get("arm") == "sketch_first_direct_mse_v4",
        "direct_objective": direct_thought.get("continuous_objective")
        == "direct_mse",
        "flow_loss_positive": float(
            flow["training"]["transfusion_cot_loss_weights"]["flow"]
        )
        > 0.0,
        "direct_flow_loss_zero": float(
            direct["training"]["transfusion_cot_loss_weights"]["flow"]
        )
        == 0.0,
        "same_u_inventory_supervision": flow_inventory == direct_inventory,
        "same_decoder_inventory_authority": flow_inventory.get("contract")
        == INVENTORY_CONTRACT
        and flow_inventory.get("authority")
        == "scene_sketch_autoregressive_logits_v1",
        "semantic_execution_ownership_identical": all(
            flow["model"]["transfusion_cot"].get(key)
            == direct["model"]["transfusion_cot"].get(key)
            for key in ("semantic_from_execution_state", "numeric_from_scene_sketch")
        ),
        "p10_executor_binding_identical": flow["model"].get("executor")
        == direct["model"].get("executor"),
    }
    report: dict[str, Any] = {
        "schema": "stable_audio_tools.p11_v4_matched_arm_config_gate",
        "schema_version": 1,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "comparison_contract": "same_graph_supervision_data_except_continuous_objective_v1",
        "allowed_difference_paths": sorted(ALLOWED_DIFFERENCES),
        "observed_differences": differences,
        "gates": gates,
        "u_inventory_supervision": flow_inventory,
        "flow": {
            "path": str(flow_path),
            "source_sha256": _sha256_file(flow_path),
            "resolved_sha256": _json_sha256(flow),
        },
        "direct": {
            "path": str(direct_path),
            "source_sha256": _sha256_file(direct_path),
            "resolved_sha256": _json_sha256(direct),
        },
        "invocation": shlex.join([sys.executable, *sys.argv]),
    }
    report["report_sha256_without_self"] = _json_sha256(report)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "observed_difference_paths": sorted(observed_paths),
                "gates": gates,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
