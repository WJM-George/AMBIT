"""Current CLAP selection and explicit AR training-clock policy.

The September 10 selection permits frozen-encoder AR adaptation. It does not
assert the historical all-strata CLAP gate or final AR/audio quality passed.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from scripts.t2a.experiments.ar_factual_clap_v1 import integration
from scripts.t2a.experiments.ar_instruction_t200_v2 import runtime as ar_only_runtime

SCHEMA = "editing_ar_T200_selected_CLAP_three_rank_v3"
SELECTION_SCHEMA = "CLAP_effect_selection_acceptance_for_AR_integration_v1"
NATIVE20K_SHA = "844490fb1a2de0c98091a2dec2aef72a049f0984a4f97909b9a81bbf7bf55300"
sha = integration.file_sha256


def read(path):
    return json.loads(Path(path).read_text())


def validate_config(cfg, training_mode="ar_pretrain"):
    ar_only_runtime.validate_config(cfg, training_mode)
    policy = cfg["current_AR_policy"]
    if policy["schema"] != SCHEMA or policy["physical_gpus"] != [5, 6, 7]:
        raise ValueError("current AR policy requires GPU5–7")
    schedule = cfg["schedule"]
    if policy["new_updates"] != 50000 or schedule["max_steps"] != policy["parent_step"] + 50000:
        raise ValueError("AR budget must be 50000 new optimizer updates")
    if schedule["save_every"] != 10000:
        raise ValueError("AR must save each10000 new updates")
    sizes = (schedule["short_batch_size"], schedule["long_batch_size"], schedule["gradient_accumulation"])
    if sizes not in ((8, 5, 4), (16, 10, 2), (32, 20, 1)):
        raise ValueError("AR effective global batch must remain96/60")
    if not 0 < policy["start_lr_factor"] <= 1 or not 0 < policy["rewarm_steps"] < 50000:
        raise ValueError("invalid continuous learning-rate transfer")
    if cfg["instruction_data"]["mode"] != "t200":
        raise ValueError("the new AR uses T200 requests")
    return policy


def lr_multiplier(step, cfg):
    policy = cfg["current_AR_policy"]
    offset = max(0, step - policy["parent_step"])
    if offset <= policy["rewarm_steps"]:
        return policy["start_lr_factor"] + (1 - policy["start_lr_factor"]) * offset / policy["rewarm_steps"]
    progress = min(1., (offset - policy["rewarm_steps"]) / (policy["new_updates"] - policy["rewarm_steps"]))
    return .1 + .9 * .5 * (1 + math.cos(math.pi * progress))


def check_selection(cfg):
    binding = cfg["clap_dependency"]
    if binding["schema"] != SCHEMA:
        raise ValueError("wrong current CLAP binding schema")
    selected = binding["effect_selection"]
    if sha(selected["path"]) != selected["sha256"]:
        raise RuntimeError("CLAP selection review changed")
    review = read(selected["path"])
    if (review["schema"] != SELECTION_SCHEMA or review["accepted_for_AR_integration"] is not True
            or review["CLAP_effect_selection_complete"] is not True
            or review["independent_test_used"] is not False):
        raise RuntimeError("CLAP integration selection has not passed")
    reference = binding["checkpoint"]
    fields = {k: reference[k] for k in ("path", "sha256", "step")}
    role = binding["role"]
    if role == "selected":
        if reference["format"] != "factual50k" or fields != review["selected_checkpoint"]:
            raise RuntimeError("AR encoder is not the selected CLAP")
    elif role == "native_control":
        if reference["format"] != "native_clap44" or reference["sha256"] != NATIVE20K_SHA or reference["step"] != 20000:
            raise RuntimeError("AR control must use the actual protected native20k")
    else:
        raise ValueError("only selected encoder and declared native control are supported")
    for path, expected in review["source_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError(f"CLAP selection evidence changed: {path}")
    return review


class EncoderBinding:
    """Reuse the existing loader while binding the actual current review."""
    def __init__(self, cfg):
        check_selection(cfg)
        self.binding = copy.deepcopy(cfg["clap_dependency"])
        self.identity = self.dependency = None

    def load(self, path, *, device="cpu"):
        reference = self.binding["checkpoint"]
        if Path(path).resolve() != Path(reference["path"]).resolve():
            raise RuntimeError("encoder argument differs from the selected dependency")
        model, self.identity, self.dependency = integration.load_source_encoder(reference,
            preflight_path=self.binding["preflight"], device=device)
        return model, self.identity

    def preflight_matches(self, identity, path):
        return (identity == self.identity and self.dependency is not None
            and Path(path).resolve() == Path(self.binding["preflight"]).resolve()
            and sha(path) == self.dependency["preflight_sha256"])

    def audit_validation(self, path, checkpoint, preflight, *, require_full=True):
        if Path(path).resolve() != Path(self.binding["validation_report"]).resolve() or not require_full:
            raise RuntimeError("AR requires its bound full20k encoder report")
        if checkpoint != self.identity:
            raise RuntimeError("diagnostics must match the loaded encoder")
        return integration.native.audit_clap_validation(path, checkpoint, preflight, require_full=True)

    def amend_contract(self, contract, cfg):
        validate_config(cfg, contract["training_mode"])
        if contract["physical_gpus"] != [5, 6, 7] or contract["world_size"] != 3:
            raise ValueError("AR topology changed")
        if contract["runtime_inputs"] != ["source_foa_latent", "raw_edit_request"] or contract["variant"] != "global_and_sequence":
            raise ValueError("AR model input roles changed")
        if self.dependency is None or contract["clap_checkpoint"] != self.dependency["checkpoint"]:
            raise RuntimeError("AR contract does not describe the loaded encoder")
        contract.update(instruction_data=copy.deepcopy(cfg["instruction_data"]),
            initial_state_transfer=copy.deepcopy(cfg["initial_state_transfer"]),
            training_objective="AR_CE_ONLY", rf_training=False,
            rf_mode_scope="RF is not run during AR-only training; frozen external Editing DiT is evaluated separately.",
            auxiliary=None, loss_normalization="global_valid_plan_tokens_per_optimizer_step",
            optimizer_scope=copy.deepcopy(cfg["optimizer_scope"]),
            current_AR_policy=copy.deepcopy(cfg["current_AR_policy"]),
            clap_dependency=copy.deepcopy(self.dependency),
            clap_effect_selection=copy.deepcopy(self.binding["effect_selection"]),
            clap_dependency_scope="Frozen source audio encoder only; no readout/CLAP optimizer transfer.",
            in_training_RF_validation=False)
