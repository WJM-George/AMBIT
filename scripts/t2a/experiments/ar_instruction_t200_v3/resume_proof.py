"""Run continuous four updates and native two-plus-two restart, then compare."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from scripts.t2a.experiments.ar_source_grounding_v1.evidence import sha, write


def read(path):
    return json.loads(Path(path).read_text())


def command(plan_path, name):
    plan = read(plan_path)
    spec = read(plan["experiment_spec"])
    phase = spec["phases"][name]
    cfg = read(phase["config"])
    parent = read(spec["parent_contract"])
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=3",
        str(ROOT / "scripts/t2a/experiments/ar_instruction_t200_v3/runtime.py"),
        "--experiment-plan", str(plan_path), "--phase", name,
        "--run-dir", phase["run_dir"], "--config", phase["config"],
        "--variant", "global_and_sequence", "--training-mode", "ar_pretrain",
        "--p10-checkpoint", parent["base_selection"]["checkpoint"],
        "--preflight", cfg["clap_dependency"]["preflight"],
        "--clap-checkpoint", cfg["clap_dependency"]["checkpoint"]["path"],
        "--clap-validation-report", cfg["clap_dependency"]["validation_report"],
        "--model-config", parent["model_config"], "--codec", parent["codec"],
        "--resume", phase["resume"]]


def run_phase(plan_path, name):
    plan = read(plan_path)
    spec = read(plan["experiment_spec"])
    phase = spec["phases"][name]
    for path, expected in plan["source_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError(f"bound input changed: {path}")
    log = Path(phase["artifacts"]).parent / f"{name}.log"
    print(json.dumps({"at": datetime.now().astimezone().isoformat(), "phase": name,
        "status": "starting", "log": str(log)}), flush=True)
    with log.open("x") as stream:
        subprocess.run(command(plan_path, name), cwd=ROOT, check=True, stdout=stream, stderr=subprocess.STDOUT)
    print(json.dumps({"at": datetime.now().astimezone().isoformat(), "phase": name,
        "status": "complete"}), flush=True)


def review(plan_path):
    spec = read(read(plan_path)["experiment_spec"])
    refs = {}
    results = []
    def bound(path):
        refs[str(path)] = sha(path)
        return read(path)
    def windows(path):
        refs[str(path)] = sha(path)
        return [json.loads(line) for line in path.read_text().splitlines()]
    for rank in range(3):
        dirs = {name: Path(spec["phases"][name]["artifacts"]) / f"rank{rank}"
                for name in ("continuous4", "split2", "resumed2")}
        for name, directory in dirs.items():
            done = bound(directory / "RUNTIME_COMPLETE.json")
            assert done["all_frozen_parameters_exact"] and done["rf_training_calls"] == 0
        for step, comparison in ((5252, "split2"), (5254, "resumed2")):
            first = bound(dirs["continuous4"] / f"PREFIX_FINGERPRINT_step{step}.json")
            second = bound(dirs[comparison] / f"PREFIX_FINGERPRINT_step{step}.json")
            if first != second:
                raise RuntimeError(f"continuous versus {comparison} complete state differs at step{step}, rank{rank}")
            results.append({"rank": rank, "step": step, "complete_state_sha256": first["sha256"]})
        assert (windows(dirs["continuous4"] / "training_windows.jsonl") ==
                windows(dirs["split2"] / "training_windows.jsonl") + windows(dirs["resumed2"] / "training_windows.jsonl"))
        loaded = bound(dirs["resumed2"] / "NATIVE_RESUME_LOADED.json")
        assert loaded["native_loader_all_checks_enabled"] and loaded["checkpoint"]["step"] == 5252
    return {"schema": "AR_three_rank_native_restart_review_v3", "at": datetime.now().astimezone().isoformat(),
        "three_rank_native_resume_exact": True, "continuous_and_restarted_windows_exact": True,
        "compared_state": "Every model and Adam tensor, scheduler, all-rank RNG, epoch and next batch.",
        "results": results, "source_sha256": refs, "full_AR_expansion_allowed": False,
        "AR_quality_gate_passed": False, "independent_test_used": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ["EDITING_GPUS"] == "5,6,7"
    out = args.plan.parent
    try:
        for name in ("continuous4", "split2", "resumed2"):
            run_phase(args.plan, name)
        result = review(args.plan)
        write(out / "REVIEW.json", result)
        print(json.dumps({"status": "resume_proof_passed", "review": str(out / "REVIEW.json")}), flush=True)
    except Exception as error:
        write(out / "FAILURE.json", {"at": datetime.now().astimezone().isoformat(),
            "error": repr(error), "full_AR_expansion_allowed": False})
        raise


if __name__ == "__main__":
    main()
