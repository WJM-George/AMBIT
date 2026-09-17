"""Selected-CLAP T200 AR on the existing native AR-only training loop."""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime
import functools
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel

from scripts.t2a.experiments.ar_instruction_t200_v1 import runtime as base
from scripts.t2a.experiments.ar_instruction_t200_v1.fingerprints import fingerprint, training_state
from scripts.t2a.experiments.ar_instruction_t200_v2 import runtime as single, distributed as three
from scripts.t2a.experiments.ar_instruction_t200_v3 import policy
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, state_hash, write
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import _capture_rank_rng_state
from scripts.t2a.train.sceneplan_transfusion_editing_joint_run_contract import validate_rng_inventory
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import validate_run_contract

native = single.native


def source_inventory():
    paths = [Path(__file__), Path(policy.__file__), Path(three.__file__),
        ROOT / "scripts/t2a/experiments/ar_factual_clap_v1/integration.py",
        ROOT / "scripts/t2a/experiments/clap_factual50k_v1/evaluation.py",
        ROOT / "scripts/t2a/experiments/clap_factual50k_v1/checkpoint.py",
        ROOT / "scripts/t2a/experiments/clap_scene_supervision_v1/state.py"]
    return {**single.source_inventory(), **{str(p.relative_to(ROOT)): sha(p) for p in paths}}


def trainer_tree():
    original, counts = single.trainer_tree()
    counts = {**counts, "current_clap_preflight": 0, "continuous_LR": 0,
              "deferred_validation": 0, "new_update_checkpoint_clock": 0}
    class Edit(ast.NodeTransformer):
        def visit_Compare(self, node):
            if ast.unparse(node) == "clap_identity['contract']['preflight_sha256'] != file_sha256(args.preflight)":
                counts["current_clap_preflight"] += 1
                replacement = ast.parse("not _clap_preflight_matches(clap_identity, args.preflight)", mode="eval").body
                replacement._v3_original = copy.deepcopy(node)
                return replacement
            return self.generic_visit(node)
        def visit_FunctionDef(self, node):
            if node.name == "lr_multiplier":
                counts["continuous_LR"] += 1
                node._v3_original = copy.deepcopy(node)
                node.body = ast.parse("return _current_lr(step, cfg)").body
                return node
            return self.generic_visit(node)
        def visit_If(self, node):
            expression = ast.unparse(node.test)
            if expression == "step % schedule['validate_every'] == 0 or step == schedule['max_steps']":
                counts["deferred_validation"] += 1
                replacement = ast.Pass(); replacement._v3_original = copy.deepcopy(node)
                return replacement
            if expression == "step % schedule['save_every'] == 0 or step == schedule['max_steps']":
                counts["new_update_checkpoint_clock"] += 1
                node._v3_original = copy.deepcopy(node)
                node.test = ast.parse("(step - cfg['current_AR_policy']['parent_step']) % schedule['save_every'] == 0 or step == schedule['max_steps']", mode="eval").body
                return node
            return self.generic_visit(node)
    changed = Edit().visit(copy.deepcopy(original))
    for key in ("current_clap_preflight", "continuous_LR", "deferred_validation", "new_update_checkpoint_clock"):
        assert counts[key] == 1, counts
    class Undo(ast.NodeTransformer):
        def generic_visit(self, node):
            return node._v3_original if hasattr(node, "_v3_original") else super().generic_visit(node)
    assert ast.dump(Undo().visit(copy.deepcopy(changed)), include_attributes=False) == ast.dump(original, include_attributes=False)
    return ast.fix_missing_locations(changed), counts


class Hooks(three.Hooks):
    """Keep native AR-only Adam state; declare the single-to-three-rank fork."""
    def load(self, path, *, expected_contract=None, verify_sources=True, require_latest=False):
        parent = self.cfg["initial_state_transfer"]["parent_checkpoint"]
        if Path(path).resolve() != Path(parent["checkpoint"]).resolve():
            payload, identity = native.load_joint_checkpoint(path, expected_contract=expected_contract,
                verify_sources=verify_sources, require_latest=require_latest)
            self.loaded_step = int(payload["global_step"])
            write(self.out / "NATIVE_RESUME_LOADED.json", {"checkpoint": identity,
                "complete_state_sha256": fingerprint(training_state(payload))["sha256"],
                "native_loader_all_checks_enabled": True})
            return payload, identity
        if self.initial_loaded or not verify_sources or not require_latest:
            raise RuntimeError("parent transfer requires the verified native loader")
        run = Path(self.phase["run_dir"])
        if list(run.glob("checkpoints/step-*.pt")):
            raise RuntimeError("resume the existing run checkpoint instead of restarting its parent")
        validate_run_contract(expected_contract, run)
        payload, identity = native.load_joint_checkpoint(path,
            expected_contract=policy.read(self.spec["parent_contract"]), verify_sources=True, require_latest=True)
        if identity != parent or payload["run_contract"]["training_objective"] != "AR_CE_ONLY":
            raise RuntimeError("parent is not the retained AR-only state")
        original = payload["run_contract"]
        for key in ("schema", "model_config", "model_config_sha256", "base_selection", "codec", "codec_sha256",
                    "codec_fingerprint", "indices", "training_mode", "variant", "ar_contract", "frozen_qwen_runtime",
                    "runtime_inputs", "old_plan_input", "source_caption_input", "target_audio_ar_input", "independent_test_used"):
            if original[key] != expected_contract[key]:
                raise RuntimeError(f"parent architecture/data boundary changed: {key}")
        cfg_policy = policy.validate_config(self.cfg)
        if original["world_size"] != 1 or original["physical_gpus"] != [7]:
            raise RuntimeError("initialization requires the declared GPU7 parent")
        initial = fingerprint(training_state(payload))
        if initial != policy.read(self.spec["initial_state_fingerprint"]):
            raise RuntimeError("parent complete state changed")
        if payload["global_step"] != payload["scheduler"]["last_epoch"] or payload["global_step"] != cfg_policy["parent_step"]:
            raise RuntimeError("parent update clock changed")
        for group, base_lr, last_lr in zip(payload["optimizer"]["param_groups"],
                payload["scheduler"]["base_lrs"], payload["scheduler"]["_last_lr"], strict=True):
            if group["lr"] != last_lr or group["initial_lr"] != base_lr or not math.isclose(last_lr / base_lr, cfg_policy["start_lr_factor"], abs_tol=1e-15):
                raise RuntimeError("learning rate transfer is discontinuous")
        if self.rank == 2:
            rank_rng = {**copy.deepcopy(payload["rng_states_by_rank"][0]), "rank": 2}
        else:
            native._seed_everything(cfg_policy["new_rank_seed"], self.rank)
            rank_rng = _capture_rank_rng_state(rank=self.rank, device=torch.device("cuda", self.rank))
        states = [None] * 3
        dist.all_gather_object(states, rank_rng)
        validate_rng_inventory(states, world_size=3)
        if fingerprint({**states[2], "rank": 0}) != fingerprint(payload["rng_states_by_rank"][0]):
            raise RuntimeError("GPU7 RNG was not preserved")
        result = {**payload, "rng_states_by_rank": states, "epoch": 0, "next_batch": 0}
        write(self.out / "INITIAL_STATE_LOADED.json", {"at": datetime.now().astimezone().isoformat(),
            "parent_checkpoint": identity, "parent_complete_state_sha256": initial["sha256"],
            "models_optimizer_scheduler_preserved_without_reprojection": True,
            "GPU7_RNG_preserved_at_rank2": True, "all_rank_rng_sha256": fingerprint(states)["sha256"],
            "parent_data_position": {"epoch": payload["epoch"], "next_batch": payload["next_batch"]},
            "destination_data_position": {"epoch": 0, "next_batch": 0},
            "declared_topology_fork_not_exact_single_to_three_resume": True,
            "destination_contract_sha256": sha(run / "RUN_CONTRACT.json")})
        self.initial_loaded = True
        self.loaded_step = int(result["global_step"])
        return result, identity


def check_phase(spec, phase, cfg):
    current = policy.validate_config(cfg)
    policy.check_selection(cfg)
    if os.environ.get("EDITING_GPUS") != "5,6,7" or int(os.environ.get("WORLD_SIZE", "0")) != 3:
        raise RuntimeError("use the three-rank GPU5–7 launcher")
    if base.allocated_runtime.gpu_topology([5, 6, 7]) != spec["destination_topology"]:
        raise RuntimeError("GPU topology changed")
    updates = phase["stop_at_step"] - current["parent_step"]
    if not phase["stop_after_prefix"] or updates <= 0:
        raise ValueError("every phase needs a positive explicit stop")
    if phase["stage"] == "resume_proof":
        if updates > 4:
            raise ValueError("resume proof cannot expand into training")
    elif phase["stage"] in ("short_adaptation", "production"):
        gate_ref = phase["prerequisite"]
        if sha(gate_ref["path"]) != gate_ref["sha256"]:
            raise RuntimeError("phase prerequisite changed")
        gate = policy.read(gate_ref["path"])
        if gate.get("three_rank_native_resume_exact") is not True:
            raise RuntimeError("AR native restart proof is required")
        if phase["stage"] == "short_adaptation" and updates > 250:
            raise ValueError("short adaptation is limited to250 new updates")
        if phase["stage"] == "production" and (updates != 50000 or gate.get("full_AR_expansion_allowed") is not True):
            raise RuntimeError("production requires reviewed short adaptation and50k budget")
    else:
        raise ValueError("unknown phase")
    for record in cfg["current_AR_policy"]["FLA_catalogs"].values():
        if sha(Path(record["directory"]) / "AUTOTUNE_CATALOG.json") != record["sha256"]:
            raise RuntimeError("the fixed FLA configuration catalog changed")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment-plan", type=Path, required=True)
    parser.add_argument("--phase", required=True)
    args, native_args = parser.parse_known_args()
    plan = policy.read(args.experiment_plan)
    spec = policy.read(plan["experiment_spec"])
    phase = spec["phases"][args.phase]
    cfg = policy.read(phase["config"])
    for path, expected in plan["source_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError(f"bound phase input changed: {path}")
    check_phase(spec, phase, cfg)
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or os.environ.get("FLA_CACHE_MODE") != "disabled":
        raise RuntimeError("the selected numerical runtime is not configured")
    if torch.cuda.is_initialized():
        raise RuntimeError("GPU allocation must precede CUDA initialization")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hooks = Hooks(spec, phase, cfg)
    hooks.out.mkdir(parents=True, exist_ok=False)
    tree, counts = trainer_tree()
    write(hooks.out / "NATIVE_AST_QA.json", {"counts": counts,
        "reverse_transform_recovers_existing_AR_only_loop_exactly": True,
        "AR_forward_CE_sampler_accumulation_clipping_Adam_and_native_checkpoint_kept": True,
        "RF_validation_deferred_to_separate_external_DiT_evaluation": True})
    reference = cfg["current_AR_policy"]["FLA_catalogs"][str(hooks.rank)]
    finish_numerics = base.numerics.install_autotune_observer({"name": "autotune", "mode": "pin",
        "reference_case": reference["directory"]}, hooks.out)
    flash_calls, restore_flash = base.install_flash(cfg["numerical_execution"])
    binding = policy.EncoderBinding(cfg)
    namespace = dict(native.__dict__,
        _distributed=functools.partial(base.allocated_runtime.distributed, timeout_seconds=900),
        DistributedDataParallel=functools.partial(DistributedDataParallel, static_graph=True, bucket_cap_mb=2048),
        run_source_inventory=source_inventory, validate_config=policy.validate_config,
        load_clap44_checkpoint=binding.load, audit_clap_validation=binding.audit_validation,
        _clap_preflight_matches=binding.preflight_matches, _current_lr=policy.lr_multiplier,
        optimizer_groups=hooks.optimizer_groups, load_joint_checkpoint=hooks.load,
        _text_module=hooks.wrap_module, _text_dataset=hooks.dataset, _text_contract=binding.amend_contract,
        _text_window=hooks.window, _text_update=hooks.update, _ar_ce_sum=single.ar_only.ce_sum,
        _ar_ready=hooks.ready, _ar_gradient_audit=hooks.gradient_audit)
    exec(compile(tree, __file__ + "::native_AR_only_current_selection", "exec"), namespace)
    sys.argv = [native.__file__, *native_args]
    try:
        try:
            namespace["main"]()
        except base.PrefixComplete:
            dist.barrier()
        else:
            raise RuntimeError("native loop ended without its declared bounded stop")
        finish_numerics()
        before = policy.read(hooks.out / "FROZEN_STATE_BEFORE.json")
        after = {"qwen": state_hash(hooks.module.ar.instruction_conditioner.model),
                 "clap": state_hash(hooks.module.ar.source_clap_model)}
        if before != after or hooks.frozen_before != hooks.frozen_state() or hooks.rf_training_calls:
            raise RuntimeError("frozen state or AR-only objective changed")
        write(hooks.out / "RUNTIME_COMPLETE.json", {"at": datetime.now().astimezone().isoformat(),
            "stage": phase["stage"], "step": phase["stop_at_step"],
            "new_updates_this_phase": phase["stop_at_step"] - hooks.loaded_step,
            "all_frozen_parameters_exact": True, "frozen_states": after,
            "RF_training_calls": hooks.rf_training_calls, "flash_calls": flash_calls,
            "fixed_FLA_catalog_sha256": reference["sha256"],
            "quality_gate_passed": False, "independent_test_used": False})
    finally:
        restore_flash()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
