from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from train import TrainingHealthGateCallback


PREFIX = "SAT_TRAINING_GATE_RESULT="


def _fixture(ratio_metrics: list[str]):
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    optimizer.state[parameter]["step"] = torch.tensor(4)
    trainer = SimpleNamespace(
        world_size=1,
        global_step=4,
        is_global_zero=True,
        strategy=SimpleNamespace(root_device=torch.device("cpu")),
        optimizers=[optimizer],
    )
    module = SimpleNamespace(p11_ema=SimpleNamespace(step=torch.tensor(0)))
    callback = TrainingHealthGateCallback(
        ["train/loss", "train/executable_axis_rotate_source"],
        ratio_metrics=ratio_metrics,
        window=2,
        max_loss_ratio=0.95,
        expected_world_size=1,
    )
    callback.on_train_start(trainer, module)
    callback.optimizer_events = 4
    callback.gradient_norms = [1.0] * 4
    return callback, trainer, module


def _observe(callback, trainer, module, rows):
    for batch_index, (loss, rotate_loss, rotate_rows) in enumerate(rows):
        module._last_step_metrics = {
            "train/loss": torch.tensor(loss),
            "train/executable_axis_rotate_source": torch.tensor(rotate_loss),
            "train/executable_axis_rotate_source_rows": torch.tensor(rotate_rows),
        }
        callback.on_train_batch_end(
            trainer, module, None, None, batch_index
        )
    module.p11_ema.step = torch.tensor(4)


def _result(capsys):
    lines = capsys.readouterr().out.splitlines()
    payloads = [
        json.loads(line[len(PREFIX) :])
        for line in lines
        if line.startswith(PREFIX)
    ]
    assert len(payloads) == 1
    return payloads[0]


def test_sparse_placeholders_are_excluded_and_active_rows_are_weighted(capsys):
    callback, trainer, module = _fixture(ratio_metrics=["train/loss"])
    _observe(
        callback,
        trainer,
        module,
        [
            (4.0, 0.0, 0),
            (3.0, 2.0, 2),
            (2.0, 0.0, 0),
            (1.0, 3.0, 4),
        ],
    )
    callback.on_fit_end(trainer, module)
    result = _result(capsys)

    sparse = result["metric_windows"][
        "train/executable_axis_rotate_source"
    ]
    assert sparse["activity_metric"] == (
        "train/executable_axis_rotate_source_rows"
    )
    assert sparse["active_observations"] == 2
    assert sparse["first_count"] == 2
    assert sparse["first_active_weight"] == pytest.approx(6.0)
    assert sparse["first_mean"] == pytest.approx((2.0 * 2 + 3.0 * 4) / 6)
    assert sparse["ratio_gated"] is False
    assert result["metric_window_aggregation"] == (
        "active_row_weighted_sum_across_ddp_ranks_v2"
    )


def test_sparse_monitor_can_rise_without_replacing_decode_capability_gate(capsys):
    callback, trainer, module = _fixture(ratio_metrics=["train/loss"])
    _observe(
        callback,
        trainer,
        module,
        [
            (4.0, 1.0, 2),
            (3.0, 1.0, 2),
            (2.0, 2.0, 2),
            (1.0, 2.0, 2),
        ],
    )
    callback.on_fit_end(trainer, module)
    result = _result(capsys)
    assert result["status"] == "PASS"
    assert result["metric_windows"][
        "train/executable_axis_rotate_source"
    ]["last_over_first"] == pytest.approx(2.0)


def test_ratio_gated_metric_still_fails_when_it_rises():
    callback, trainer, module = _fixture(
        ratio_metrics=["train/loss", "train/executable_axis_rotate_source"]
    )
    _observe(
        callback,
        trainer,
        module,
        [
            (4.0, 1.0, 2),
            (3.0, 1.0, 2),
            (2.0, 2.0, 2),
            (1.0, 2.0, 2),
        ],
    )
    with pytest.raises(RuntimeError, match="loss ratios exceed"):
        callback.on_fit_end(trainer, module)


def test_required_sparse_metric_without_active_rows_fails_closed():
    callback, trainer, module = _fixture(ratio_metrics=["train/loss"])
    _observe(
        callback,
        trainer,
        module,
        [
            (4.0, 0.0, 0),
            (3.0, 0.0, 0),
            (2.0, 0.0, 0),
            (1.0, 0.0, 0),
        ],
    )
    with pytest.raises(RuntimeError, match="without active rows"):
        callback.on_fit_end(trainer, module)


def test_route_specific_ce_uses_task_count_as_activity_weight():
    callback = TrainingHealthGateCallback(
        ["train/generation_discrete_ce"], window=2
    )
    trainer = SimpleNamespace()
    module = SimpleNamespace()
    for batch_index, (value, rows) in enumerate(((0.0, 0), (2.0, 2), (3.0, 4))):
        module._last_step_metrics = {
            "train/generation_discrete_ce": torch.tensor(value),
            "train/task_generation": torch.tensor(rows),
        }
        callback.on_train_batch_end(
            trainer, module, None, None, batch_index
        )

    assert callback.metric_activity_names[
        "train/generation_discrete_ce"
    ] == "train/task_generation"
    assert callback.first_metrics["train/generation_discrete_ce"] == [
        (2.0, 2.0),
        (3.0, 4.0),
    ]
    assert list(callback.last_metrics["train/generation_discrete_ce"]) == [
        (2.0, 2.0),
        (3.0, 4.0),
    ]
