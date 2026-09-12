import json
import os
from pathlib import Path
import subprocess

import pytest
import torch
from torch import nn

from scripts.t2a.train import editing_gpu_runtime as runtime
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_clap44 import parse_args, validate_config
from stable_audio_tools.models.sceneplan_transfusion_editing_ar import EDITING_AR_CONTRACT
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import (
    AR_PRETRAIN_RUN_SCHEMA, AR_PRETRAIN_CHECKPOINT_SCHEMA, JOINT44_CHECKPOINT_SCHEMA,
    atomic_json, ensure_run_identity, load_joint_checkpoint, save_joint_checkpoint,
)
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_joint import CLAP44ARPretrainModule

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value, expected", [("0,1,2", [0, 1, 2]), ("7", [7]), (" 6, 2 ", [6, 2]), ("11,9", [11, 9])])
def test_gpu_allocation_is_explicit_and_ordered(value, expected):
    assert runtime.selected_gpus(value) == expected
    assert runtime.launch_command(["torchrun", "--nproc_per_node={gpu_count}"], len(expected))[-1] == f"--nproc_per_node={len(expected)}"


@pytest.mark.parametrize("value", ["", "0,0", "0,-1", "0,", "all", "0;echo bad"])
def test_invalid_gpu_allocations_are_rejected(value):
    with pytest.raises(ValueError):
        runtime.selected_gpus(value)


@pytest.fixture
def gpu_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("EDITING_GPUS", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("EDITING_GPU_LOCK_DIR", str(tmp_path / "locks"))
    monkeypatch.setenv("EDITING_DATA_ROOT", str(tmp_path))
    contract = tmp_path / "dit.json"
    contract.write_text(json.dumps({"training": {"physical_gpus": [6, 7]}}))
    monkeypatch.setenv("EDITING_DIT_RUN_CONTRACT", str(contract))
    def query(command, **kwargs):
        if "--query-compute-apps=gpu_uuid,pid" in command:
            return ""
        return "\n".join(f"{i}, 0000:{20-i:02x}:00.0, GPU-example-{i}, test GPU" for i in range(8))
    monkeypatch.setattr(runtime.subprocess, "check_output", query)
    return tmp_path


def test_uuid_mapping_does_not_assume_pci_or_index_sorting(gpu_environment, monkeypatch):
    topology = runtime.gpu_topology([7, 2])
    runtime.configure_visibility(topology)
    assert os.environ["EDITING_GPUS"] == "7,2"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-example-7,GPU-example-2"
    assert topology["mapping"][1]["physical_index"] == 2
    with pytest.raises(ValueError, match="does not exist"):
        runtime.gpu_topology([8])


def test_disjoint_jobs_coexist_and_overlaps_are_rejected(gpu_environment, monkeypatch):
    first = runtime.gpu_topology([0, 1])
    second = runtime.gpu_topology([2])
    with runtime.gpu_lease(first) as leases:
        monkeypatch.setenv("EDITING_GPU_LEASES", json.dumps(leases))
        assert runtime.verify_launcher_leases(first) == leases
        with runtime.gpu_lease(second):
            with pytest.raises(RuntimeError, match="reserved"):
                with runtime.gpu_lease(runtime.gpu_topology([1, 2])):
                    pytest.fail("overlapping GPU lease was accepted")
    with runtime.gpu_lease(first):
        pass


def test_legacy_dit_lock_is_chosen_from_its_recorded_allocation(gpu_environment):
    assert all(path.name != "training-chain.lock" for path in runtime.resource_paths(runtime.gpu_topology([0, 2])))
    assert any(path.name == "training-chain.lock" for path in runtime.resource_paths(runtime.gpu_topology([6])))


def test_busy_gpu_is_not_launched(gpu_environment, monkeypatch):
    topology = runtime.gpu_topology([1])
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **kw: "GPU-example-1, 12345\n")
    with pytest.raises(RuntimeError, match="already has a compute process"):
        with runtime.gpu_lease(topology):
            pytest.fail("busy GPU was leased for training")


def test_distributed_checks_world_and_uses_local_rank_without_cuda_work(gpu_environment, monkeypatch):
    monkeypatch.setenv("EDITING_GPUS", "7,2,0")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    calls = []
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: calls.append(rank))
    monkeypatch.setattr(torch.distributed, "init_process_group", lambda *a, **kw: calls.append(kw["device_id"]))
    with runtime.gpu_lease(runtime.gpu_topology()) as leases:
        monkeypatch.setenv("EDITING_GPU_LEASES", json.dumps(leases))
        rank, local, world, device, topology = runtime.distributed()
        assert (rank, local, world) == (1, 1, 3)
        assert calls == [1, torch.device("cuda", 1)]
        assert topology["mapping"][1]["physical_index"] == 2
        monkeypatch.setenv("WORLD_SIZE", "5")
        with pytest.raises(RuntimeError, match="ranks must match"):
            runtime.distributed()


@pytest.mark.parametrize("gpus", ["0,1,2", "7", "6,2"])
def test_shell_uses_selected_world_and_pretrain_arguments(tmp_path, gpus):
    binary = tmp_path / ".venv/bin/torchrun"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/usr/bin/env python3\nimport json,os,sys\nprint(json.dumps({'args':sys.argv[1:],'gpus':os.environ['EDITING_GPUS']}))\n")
    binary.chmod(0o755)
    env = {**os.environ, "P10_REPO": str(tmp_path), "EDITING_GPUS": gpus,
           "EDITING_GPU_LEASES": "test-launcher-already-owns-leases", "EDITING_AR_MODE": "ar_pretrain",
           "CLAP44_AR_VARIANT": "latent_only", "EDITING_P10_CHECKPOINT": str(tmp_path / "p10.ckpt"),
           "CLAP44_AR_RUN_DIR": str(tmp_path / "run")}
    env.pop("CLAP44_CHECKPOINT", None)
    env.pop("CLAP44_VALIDATION_REPORT", None)
    result = subprocess.run(["bash", str(ROOT / "scripts/t2a/train/run_sceneplan_transfusion_editing_ar_clap44.sh")],
                            env=env, text=True, capture_output=True, check=True)
    record = json.loads(result.stdout)
    assert f"--nproc_per_node={len(gpus.split(','))}" in record["args"]
    assert record["gpus"] == gpus
    assert "--p10-checkpoint" in record["args"] and "--dit-gt-audio-gate" not in record["args"]
    assert "--clap-checkpoint" not in record["args"]


def test_pretraining_and_joint_startup_require_different_evidence():
    basic = ["--run-dir", "run", "--config", "cfg", "--preflight", "data", "--variant", "latent_only"]
    args = parse_args(basic + ["--training-mode", "ar_pretrain", "--p10-checkpoint", "p10"])
    assert args.dit_gt_audio_gate is None and args.clap_checkpoint is None
    with pytest.raises(SystemExit):
        parse_args(basic)
    with pytest.raises(SystemExit):
        parse_args(basic + ["--training-mode", "ar_pretrain"])
    with pytest.raises(SystemExit):
        parse_args(basic + ["--training-mode", "ar_pretrain", "--p10-checkpoint", "p10", "--dit-gt-audio-gate", "fake-pass"])
    config = json.loads((ROOT / "stable_audio_tools/configs/model_configs/txt2audio/t2a/editing_ar_clap44_pretrain_v1.json").read_text())
    validate_config(config, "ar_pretrain")
    with pytest.raises(ValueError, match="purpose"):
        validate_config(config, "joint")


def pretrain_module():
    class Stack(nn.Module):
        def __init__(self):
            super().__init__(); self.layers = nn.Linear(2, 2)
        def forward(self, x):
            return self.layers(x)
    stack = Stack()
    class AR(nn.Module):
        def __init__(self):
            super().__init__(); self.shared_transformer = stack; self.source_clap_model = None
            self.head = nn.Linear(2, 4)
        def encode_edit_instructions(self, text, device):
            return torch.zeros(len(text), 1, 2), torch.ones(len(text), 1, dtype=torch.bool)
        def forward(self, source, mask, ids, plan_mask, context, context_mask, **kwargs):
            return self.head(stack(source[:, :2].mean(-1)))[:, None].expand(-1, ids.shape[1], -1)
    class RF(nn.Module):
        def __init__(self):
            super().__init__(); self.model = nn.Module(); self.model.transformer = stack
            self.output = nn.Linear(2, 64)
        def forward(self, noised, times, source, **kwargs):
            self.last_source = source.detach().clone()
            return self.output(stack((noised + source)[:, :2].transpose(1, 2))).transpose(1, 2)
    class Conditioner(nn.Module):
        def forward(self, metadata, device):
            return {"source_foa_latent": [torch.stack([row["source"] for row in metadata]), None]}
    class Diffusion(nn.Module):
        def __init__(self):
            super().__init__(); self.model = RF(); self.conditioner = Conditioner()
        def get_conditioning_inputs(self, value):
            return {"source": value["source_foa_latent"][0]}
    return CLAP44ARPretrainModule(diffusion=Diffusion(), ar=AR())


def test_pretraining_keeps_real_ar_reference_and_zeroes_only_rf_reference():
    torch.manual_seed(42)
    module = pretrain_module()
    source = torch.randn(2, 64, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    kwargs = dict(source_foa_latent=source, source_attention_mask=mask, plan_input_ids=torch.ones(2, 2, dtype=torch.long),
                  plan_attention_mask=torch.ones(2, 2, dtype=torch.bool), raw_edit_requests=["move", "remove"],
                  metadata=[{"source": row} for row in source], noised_target=torch.randn_like(source),
                  timesteps=torch.ones(2) * .5, rf_padding_mask=mask)
    ar, rf, _, _ = module(**kwargs)
    assert torch.count_nonzero(module.diffusion.model.last_source) == 0
    other_ar, other_rf, _, _ = module(**{**kwargs, "source_foa_latent": source + 10,
        "metadata": [{"source": row + 10} for row in source]})
    torch.testing.assert_close(rf, other_rf, rtol=0, atol=0)
    assert not torch.equal(ar, other_ar)
    shared = module.ar.shared_transformer.layers.weight
    ga = torch.autograd.grad(ar.square().mean(), shared, retain_graph=True)[0]
    gr = torch.autograd.grad(rf.square().mean(), shared, retain_graph=True)[0]
    (ar.square().mean() + rf.square().mean()).backward()
    assert ga.abs().sum() > 0 and gr.abs().sum() > 0
    torch.testing.assert_close(shared.grad, ga + gr)


def test_pretrain_checkpoint_restores_and_cannot_enter_formal_selection(tmp_path):
    from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import _capture_rank_rng_state
    from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_selection_io import audit_full_run
    identity = ensure_run_identity(tmp_path)
    contract = {"schema": AR_PRETRAIN_RUN_SCHEMA, "training_mode": "ar_pretrain", "run_dir": str(tmp_path),
                "run_id": identity["run_id"], "repo_root": str(tmp_path), "ar_contract": EDITING_AR_CONTRACT,
                "variant": "latent_only", "m2d_used": False, "independent_test_used": False, "world_size": 1,
                "schedule": {"max_steps": 2}, "source_sha256": {},
                "dit_gt_audio_gate": {"status": "NOT_APPLICABLE_TO_AR_PRETRAINING"},
                "rf_mode": "p10_generation_zero_reference"}
    atomic_json(tmp_path / "RUN_CONTRACT.json", contract)
    module = pretrain_module()
    optimizer = torch.optim.AdamW(module.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.)
    sum(p.square().sum() for p in module.parameters()).backward()
    optimizer.step(); scheduler.step()
    rng = _capture_rank_rng_state(rank=0, device=None)
    rng["torch_cuda_rng_state"] = torch.zeros(8, dtype=torch.uint8)
    path = tmp_path / "checkpoints/step-00000001.pt"
    save_joint_checkpoint(path, module=module, optimizer=optimizer, scheduler=scheduler, step=1,
                          epoch=0, next_batch=4, contract=contract, rng_states=[rng])
    payload, record = load_joint_checkpoint(path, expected_contract=contract, verify_sources=False, require_latest=True)
    assert record["schema"] == AR_PRETRAIN_CHECKPOINT_SCHEMA != JOINT44_CHECKPOINT_SCHEMA
    assert payload["quality_gate_passed"] is False
    with pytest.raises(RuntimeError, match="adapter transfer"):
        audit_full_run(tmp_path)
