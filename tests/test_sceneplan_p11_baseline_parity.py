from __future__ import annotations

from pathlib import Path

from stable_audio_tools.configuration import load_config


CONFIG_ROOT = Path(
    "stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11"
)


def _different_leaves(left, right, path=()):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = set()
        for key in set(left) | set(right):
            if key not in left or key not in right:
                differences.add(path + (key,))
            else:
                differences.update(
                    _different_leaves(left[key], right[key], path + (key,))
                )
        return differences
    return {path} if left != right else set()


def test_direct_mse_is_a_matched_flow_objective_ablation():
    flow = load_config(CONFIG_ROOT / "qwen35_0p8b_sceneplan_p11.json")
    direct = load_config(
        CONFIG_ROOT / "qwen35_0p8b_sceneplan_p11_baseline_direct_mse.json"
    )

    assert _different_leaves(flow, direct) == {
        ("_task",),
        ("model", "transfusion_cot", "thought", "arm"),
        (
            "model",
            "transfusion_cot",
            "thought",
            "continuous_objective",
        ),
        ("training", "transfusion_cot_loss_weights", "flow"),
    }
    assert direct["training"]["transfusion_cot_loss_weights"]["owner"] == 1.0
    assert (
        direct["model"]["transfusion_cot"]["delta_owner"]
        == flow["model"]["transfusion_cot"]["delta_owner"]
    )
    assert (
        direct["model"]["transfusion_cot"]["thought"]["executable_axis"]
        == flow["model"]["transfusion_cot"]["thought"]["executable_axis"]
    )

