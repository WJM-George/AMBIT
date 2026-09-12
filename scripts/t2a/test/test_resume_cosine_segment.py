from types import SimpleNamespace

import pytest
import torch

from train import ResumeCosineSegmentCallback


def _trainer(*, step, scheduler):
    return SimpleNamespace(
        ckpt_path="source.ckpt",
        global_step=step,
        is_global_zero=False,
        lr_scheduler_configs=[SimpleNamespace(scheduler=scheduler)],
    )


def test_new_cosine_segment_replaces_source_cosine_contract_at_boundary():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=5e-5)
    source_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10_000,
        eta_min=5e-5,
    )
    optimizer.param_groups[0]["lr"] = 5e-5

    callback = ResumeCosineSegmentCallback(
        start_step=110_000,
        end_step=150_000,
        eta_min=1e-5,
    )
    callback.on_train_start(
        _trainer(step=110_000, scheduler=source_scheduler),
        SimpleNamespace(),
    )

    assert source_scheduler.T_max == 40_000
    assert source_scheduler.eta_min == pytest.approx(1e-5)
    assert source_scheduler.last_epoch == 0
    assert source_scheduler.base_lrs == pytest.approx([5e-5])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-5)


def test_mid_segment_resume_validates_without_resetting_phase():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=40_000,
        eta_min=1e-5,
    )
    scheduler.last_epoch = 10_000
    scheduler._step_count = 10_001
    scheduler._last_lr = [4e-5]

    callback = ResumeCosineSegmentCallback(
        start_step=110_000,
        end_step=150_000,
        eta_min=1e-5,
    )
    callback.on_train_start(
        _trainer(step=120_000, scheduler=scheduler),
        SimpleNamespace(),
    )

    assert scheduler.last_epoch == 10_000
    assert optimizer.param_groups[0]["lr"] == pytest.approx(4e-5)


def test_mid_segment_resume_rejects_wrong_scheduler_contract():
    parameter = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.SGD([parameter], lr=4e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=10_000,
        eta_min=1e-5,
    )
    scheduler.last_epoch = 10_000

    callback = ResumeCosineSegmentCallback(
        start_step=110_000,
        end_step=150_000,
        eta_min=1e-5,
    )
    with pytest.raises(RuntimeError, match="T_max"):
        callback.on_train_start(
            _trainer(step=120_000, scheduler=scheduler),
            SimpleNamespace(),
        )
