"""Publish a local, reproducible native Editing release after audio test proof."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shlex
import sqlite3

from . import sceneplan_transfusion_editing_clap44_audio_io as proof

gt, audio = proof.gt, proof.audio
FILES = {
    "evaluation_report":"EVALUATION_REPORT.json",
    "listening_package":"LISTENING.json",
    "reproduction":"REPRODUCTION.json",
    "recovery":"RECOVERY.json",
    "known_limitations":"KNOWN_LIMITATIONS.json",
}
SCOPE = "checksum_bound_local_archive_with_frozen_source_snapshot_v1"


def _read(artifact):
    return json.loads(gt._verify_artifact(artifact).read_text())


def _json_bytes(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def _immutable(path, content):
    """Resume only matching unpublished work; never rewrite published evidence."""
    path = Path(path)
    data = content if isinstance(content,bytes) else _json_bytes(content)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"release artifact changed; preserve it: {path}")
        return gt._artifact(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)
    return gt._artifact(path)


def _context(test_path, test_sha, selection_path, selection_sha):
    # This replays calibration and its freeze before opening the test index.
    result = proof.validate_result(test_path,expected_sha256=test_sha,
        selection_path=selection_path,selection_sha256=selection_sha,phase="test")
    if (result.get("status") != "PASS" or result.get("final_quality_passed") is not True or
            result.get("rows") != 5000 or result.get("successful_rows") != 5000 or
            result.get("independent_test_used") is not True):
        raise RuntimeError("release requires all 5000 independent free-AR audio rows to pass")
    contract = _read(result["contract"])
    identity = contract["identity"]
    selected = {"path":str(Path(selection_path).resolve(strict=True)),"sha256":selection_sha}
    if (identity["joint_selection"] != selected or result["joint_selection"] != selected or
            result["selected_checkpoint"] != identity["joint_checkpoint"] or
            result["post_joint_gt_audio_gate"] != identity["post_joint_gt_audio_gate"]):
        raise RuntimeError("release selected checkpoint differs from its independent audio evidence")
    run = _read(identity["joint_run_contract"])
    from .sceneplan_transfusion_editing_clap44_io import load_clap44_checkpoint
    encoder, encoder_identity = load_clap44_checkpoint(
        run["clap_checkpoint"]["path"],expected_sha256=run["clap_checkpoint"]["sha256"],device="cpu")
    del encoder
    clap_contract = encoder_identity["contract"]
    clap_contract_artifact = gt._artifact(Path(run["clap_checkpoint"]["path"]).parent/"TRAIN_CONTRACT.json")
    if _read(clap_contract_artifact) != clap_contract:
        raise RuntimeError("release CLAP pretraining configuration changed")
    calibration = _read(contract["calibration"])
    gt_gate = _read(identity["post_joint_gt_audio_gate"])
    gt_contract = _read(gt_gate["contract"])
    records = []
    for artifact in result["record_artifacts"]:
        records.extend(_read(artifact)["records"])
    if len(records) != 5000 or sorted(row["pair_ordinal"] for row in records) != contract["selected_ordinals"]:
        raise RuntimeError("release audio records do not cover the complete passed test")
    return {
        "test":result,"test_artifact":{"path":str(Path(test_path).resolve(strict=True)),"sha256":test_sha},
        "contract":contract,"run":run,"calibration":calibration,"gt_gate":gt_gate,
        "gt_contract":gt_contract,"records":records,
        "clap_contract":clap_contract,"clap_contract_artifact":clap_contract_artifact,
    }


def _weak_examples(records):
    """Five weakest applicable rows per metric, without inventing a new score."""
    result = {}
    for higher,specs in ((True,audio.HIGHER_BETTER_SPECS),(False,audio.LOWER_BETTER_SPECS)):
        for metric,spec in sorted(specs.items()):
            applicable = set(spec.get("operations",audio.OPERATIONS))
            rows = [row for row in records if row["operation"] in applicable and
                    isinstance(row["metrics"].get(metric),(int,float)) and
                    math.isfinite(row["metrics"][metric])]
            ordered = sorted(rows,key=lambda row:(
                row["metrics"][metric] if higher else -row["metrics"][metric],row["pair_ordinal"]))
            result[metric] = [{"pair_ordinal":row["pair_ordinal"],"value":row["metrics"][metric]}
                              for row in ordered[:5]]
    return result


def _listening(context):
    contract, records = context["contract"], context["records"]
    by_ordinal = {row["pair_ordinal"]:row for row in records}
    representative = contract["listening_ordinals"]
    if len(representative) != 50:
        raise RuntimeError("release needs all 50 frozen representative listening rows")
    counts = {f"{operation}:{bucket}":0 for operation in audio.OPERATIONS for bucket in (432,648)}
    for ordinal in representative:
        row = by_ordinal[ordinal]
        counts[f"{row['operation']}:{row['latent_bucket_frames']}"] += 1
    if set(counts.values()) != {5}:
        raise RuntimeError("release listening set lost operation/length coverage")
    weakest = _weak_examples(records)
    # Include one weakest row per metric and at most 50 distinct extras.
    weak_ordinals = sorted({rows[0]["pair_ordinal"] for rows in weakest.values() if rows})[:50]
    chosen = sorted(set(representative)|set(weak_ordinals))
    examples = []
    index = gt._verify_artifact({key:contract["index"][key] for key in ("path","sha256")})
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1",uri=True)
    try:
        for ordinal in chosen:
            row = by_ordinal[ordinal]
            truth = connection.execute(
                "SELECT pair_id,raw_edit_request,model_num_samples FROM pairs WHERE pair_ordinal=?",(ordinal,)).fetchone()
            if truth is None or truth[0] != row["pair_id"] or truth[2] != row["model_num_samples"]:
                raise RuntimeError("release listening instruction/audio identity changed")
            source = gt._artifact(Path(row["source_foa_path"]))
            target = gt._artifact(Path(row["target_foa_path"]))
            edited = {"path":row["edited_foa_path"],"sha256":row["edited_foa_sha256"]}
            for artifact in (source,target,edited):
                proof.check_audio(artifact["path"],artifact["sha256"],samples=row["model_num_samples"])
            examples.append({
                "pair_ordinal":ordinal,"pair_id":row["pair_id"],"edit_instruction":truth[1],
                "operation":row["operation"],"latent_bucket_frames":row["latent_bucket_frames"],
                "model_num_samples":row["model_num_samples"],"source_foa":source,"target_foa":target,"edited_foa":edited,
                "generated_sceneplan":row["generated_sceneplan"],"metrics":row["metrics"],
                "representative":ordinal in representative,"weak_example":ordinal in weak_ordinals,
            })
    finally:
        connection.close()
    return {
        "schema":"editing_clap44_release_listening_v1","sample_rate":44100,"channels":4,
        "representative_ordinals":representative,"representative_coverage":counts,
        "weak_ordinals":weak_ordinals,"weakest_by_metric":weakest,"examples":examples,
        "all_5000_edited_audio_remain_bound_in":context["test_artifact"],
        "human_listening_review":"not_recorded_by_automated_publication",
    }


def _command(argv, env=None):
    env = {} if env is None else env
    words = (["env"]+[f"{key}={value}" for key,value in sorted(env.items())]) if env else []
    return {"argv":[str(item) for item in argv],"env":env,
            "shell":shlex.join(words+[str(item) for item in argv])}


def _commands(context, directory):
    run, contract = context["run"], context["contract"]
    repo = proof.REPO
    python = repo/".venv/bin/python"
    identity = contract["identity"]
    run_dir = Path(run["run_dir"])
    base_selection_path = Path(context["gt_contract"]["identity"]["base_selection"]["path"])
    base_run = base_selection_path.parents[2]
    common = {"CLAP44_AR_RUN_DIR":str(run_dir),
              "CLAP44_JOINT_SELECTION_SHA256":identity["joint_selection"]["sha256"],
              "CLAP44_AUDIO_BATCH_SIZE":str(contract["batch_size_per_rank"])}
    entries = {
        "verify_joint_selection":_command([python,repo/"scripts/t2a/eval/select_sceneplan_transfusion_editing_clap44_joint.py",
            "--run-dir",run_dir,"--output-dir",Path(identity["joint_selection"]["path"]).parent,
            "--verify-only","--selection-sha256",identity["joint_selection"]["sha256"]]),
        "verify_gt_audio":_command([python,repo/"scripts/t2a/eval/evaluate_sceneplan_transfusion_editing_clap44_gt_audio.py",
            "--joint-selection",identity["joint_selection"]["path"],"--joint-selection-sha256",identity["joint_selection"]["sha256"],
            "--output-dir",Path(identity["post_joint_gt_audio_gate"]["path"]).parent,"--verify-only"]),
    }
    for phase,artifact in (("calibration",contract["calibration"]),("test",context["test_artifact"])):
        argv = [python,repo/"scripts/t2a/eval/evaluate_sceneplan_transfusion_editing_clap44_audio.py",
            "--phase",phase,"--joint-selection",identity["joint_selection"]["path"],
            "--joint-selection-sha256",identity["joint_selection"]["sha256"],
            "--post-joint-gt-gate",identity["post_joint_gt_audio_gate"]["path"],
            "--output-dir",Path(artifact["path"]).parent,"--verify-only"]
        entries[f"verify_audio_{phase}"] = _command(argv)
        env = {**common,"CLAP44_AUDIO_PHASE":phase}
        if phase=="test": env["CLAP44_AUDIO_CALIBRATION_SHA256"] = contract["calibration"]["sha256"]
        resume_args = ["--joint-selection",identity["joint_selection"]["path"],
            "--post-joint-gt-gate",identity["post_joint_gt_audio_gate"]["path"],
            "--preflight",contract["preflight"]["path"],"--output-dir",Path(artifact["path"]).parent]
        if phase=="test": resume_args.extend(["--calibration",contract["calibration"]["path"]])
        entries[f"resume_audio_{phase}"] = _command(
            ["bash",repo/"scripts/t2a/eval/run_sceneplan_transfusion_editing_clap44_audio_5gpu.sh",*resume_args],env)
    training_env = {**common,"CLAP44_AR_PHASE":"full","CLAP44_AR_VARIANT":run["variant"],
        "CLAP44_CHECKPOINT":run["clap_checkpoint"]["path"],"CLAP44_VALIDATION_REPORT":run["clap_validation"]["path"],
        "BASE_DIT_RUN":str(base_run)}
    entries["resume_joint_training"] = _command(
        ["bash",repo/"scripts/t2a/train/run_sceneplan_transfusion_editing_ar_clap44_5gpu.sh",
         "--config",directory/"AR_TRAIN_CONFIG.json","--model-config",run["model_config"],"--codec",run["codec"],
         "--preflight",contract["preflight"]["path"],"--dit-selection",run["base_selection"]["path"],
         "--dit-gt-audio-gate",run["dit_gt_audio_gate"]["path"]],training_env)
    entries["resume_joint_selection"] = _command(
        ["bash",repo/"scripts/t2a/eval/run_sceneplan_transfusion_editing_clap44_joint_selection_5gpu.sh",
         "--preflight",contract["preflight"]["path"],"--output-dir",Path(identity["joint_selection"]["path"]).parent],common)
    entries["resume_gt_audio"] = _command(
        ["bash",repo/"scripts/t2a/eval/run_sceneplan_transfusion_editing_clap44_gt_audio_5gpu.sh",
         "--joint-selection",identity["joint_selection"]["path"],
         "--output-dir",Path(identity["post_joint_gt_audio_gate"]["path"]).parent],common)
    entries["reproduce_clap_pretraining_in_new_run"] = _command(
        ["bash",repo/"scripts/t2a/train/run_sceneplan_transfusion_editing_clap44_5gpu.sh",
         "--config",directory/"CLAP_TRAIN_CONFIG.json","--preflight",contract["preflight"]["path"],
         "--index",run["indices"]["train"]["path"]],
        {"CLAP44_RUN_ROOT":str(directory/"reproduction_runs/clap44")})
    entries["reproduce_joint_training_in_new_run"] = _command(
        entries["resume_joint_training"]["argv"],
        {**training_env,"CLAP44_AR_RUN_DIR":str(directory/"reproduction_runs/ar_joint")})
    return entries


def _payloads(context, directory):
    test, contract, run = context["test"],context["contract"],context["run"]
    identity = contract["identity"]
    listening = _listening(context)
    report = {
        "schema":"editing_clap44_release_report_v1","status":"PASS","checkpoint":identity["joint_checkpoint"],
        "feature_variant":run["variant"],"custom_clap_used":run["variant"]!="latent_only",
        "runtime_inputs":["source_foa_audio","raw_edit_instruction"],"pipeline_contract":proof.CLAP44_PIPELINE_CONTRACT,
        "dataset_pairs":{"train":1000000,"validation":20000,"test":5000},
        "joint_selection":identity["joint_selection"],"post_joint_gt_audio":identity["post_joint_gt_audio_gate"],
        "post_joint_gt_audio_result":context["gt_gate"]["result"],
        "calibration":contract["calibration"],"calibration_checks":context["calibration"]["checks"],
        "independent_test":context["test_artifact"],"test_checks":test["checks"],
        "test_metric_summaries":test["metric_summaries"],"test_difficulty":test["difficulty"],
        "sampling_and_metrics":contract["policy"],"frozen_assets":contract["frozen_assets"],
        "m2d_used":False,"human_listening_review":listening["human_listening_review"],
    }
    commands = _commands(context,directory)
    reproduction = {
        "schema":"editing_clap44_reproduction_v1","artifact_scope":SCOPE,
        "repo":str(proof.REPO),"training_run_contract":identity["joint_run_contract"],
        "model_config":identity["model_config"],"codec":{"path":run["codec"],"sha256":run["codec_sha256"]},
        "frozen_assets":contract["frozen_assets"],"preflight":contract["preflight"],
        "clap_pretraining_contract":context["clap_contract_artifact"],
        "commands":commands,
        "source_restore":"Restore sources/ into an isolated copy of the matching repository and locked environment, exposed at the recorded absolute repository path (for example in an isolated container). External assets retain their bound paths. Do not replace an active training checkout.",
        "metric_reproduction":"The evaluation commands retain the frozen rank/batch population and per-pair seeds. A new run needs a separate output directory; completed artifacts are verified, never silently overwritten.",
        "inference_entry":str(proof.REPO/"scripts/t2a/inference/edit_foa_with_clap44.py"),
        "inference_arguments":["--release",str(directory/"RELEASE.json"),"--release-sha256","<published SHA256>",
                               "--source","<44.1kHz four-channel WAV>","--instruction","<raw edit instruction>",
                               "--output-dir","<new output directory>"],
    }
    recovery = {
        "schema":"editing_clap44_recovery_v1","selected_checkpoint":identity["joint_checkpoint"],
        "training_run_contract":identity["joint_run_contract"],"commands":{key:value for key,value in commands.items() if key.startswith("resume_")},
        "checkpoint_rule":"Resume the verified latest training state, including optimizer/scheduler/sampler/per-rank RNG. The quality-selected checkpoint may be earlier and must not replace LATEST.",
        "evaluation_rule":"Reuse only rows and phases matching the frozen contracts. Preserve failures and diagnose changed evidence; never relabel training completion as quality PASS.",
        "resource_rule":"Resume launchers take an exclusive lock for the active editing job and use the GPUs listed in CUDA_VISIBLE_DEVICES.",
        "release_rule":"Rerun the publisher with the same pinned selection/test and output directory to finish interrupted publication. Different or changed files are refused.",
    }
    limitations = {
        "schema":"editing_clap44_known_limitations_v1",
        "validated_scope":{"test_pairs":5000,"operations":list(audio.OPERATIONS),"sample_rate":44100,
                           "channels":4,"latent_buckets":[432,648],"batch_size_per_rank":contract["batch_size_per_rank"]},
        "items":[
            "Passing aggregate and stratified gates does not make every individual edit correct; inspect the weak cases and metric confidence/coverage in the linked report.",
            "Quality evidence applies to this frozen synthetic Editing distribution and these five edit operations; arbitrary instructions, recordings and source counts are not independently certified.",
            "A new event has no unique sample waveform or phase. Independent content, activity and preservation metrics govern addition quality under the frozen policy.",
            "Runtime remains native 44.1kHz four-channel FOA. The independent post-inference scorer may use its own resampling without changing the model input.",
            "Device, batch size or numerical backend changes can alter generated tokens and audio. Use the frozen evaluation commands to reproduce reported metrics.",
            "The publication script does not claim that a human has listened to the outputs.",
        ],
        "human_listening_review":listening["human_listening_review"],
        "weakest_by_metric":listening["weakest_by_metric"],
    }
    return {"evaluation_report":report,"listening_package":listening,"reproduction":reproduction,
            "recovery":recovery,"known_limitations":limitations}


def _build_contract(context,directory):
    return {"schema":"editing_clap44_release_build_v1","directory":str(directory),
        "independent_test":context["test_artifact"],"joint_selection":context["contract"]["identity"]["joint_selection"],
        "source_sha256":proof.code_hashes(),"artifact_scope":SCOPE}


def _manifest(context,directory,build,payloads):
    contract = context["contract"]
    return {
        "schema":proof.RELEASE_SCHEMA,"status":"PASS","directory":str(directory),"artifact_scope":SCOPE,
        "pipeline_contract":proof.CLAP44_PIPELINE_CONTRACT,"model_input_contract":audio.MODEL_INPUT_CONTRACT,
        "checkpoint":contract["identity"]["joint_checkpoint"],"joint_selection":contract["identity"]["joint_selection"],
        "feature_variant":context["run"]["variant"],"custom_clap_used":context["run"]["variant"]!="latent_only",
        "independent_test":context["test_artifact"],"calibration":contract["calibration"],
        "training_run_contract":contract["identity"]["joint_run_contract"],
        "post_joint_gt_audio_gate":contract["identity"]["post_joint_gt_audio_gate"],
        "pretest_freeze":contract["pretest_freeze"],"policy":contract["policy"],"source_sha256":build["source_sha256"],
        "build_contract":gt._artifact(directory/"BUILD_CONTRACT.json"),
        "ar_training_config":gt._artifact(directory/"AR_TRAIN_CONFIG.json"),
        "clap_training_config":gt._artifact(directory/"CLAP_TRAIN_CONFIG.json"),
        "clap_pretraining_contract":context["clap_contract_artifact"],
        "source_snapshot":{name:gt._artifact(directory/"sources"/name) for name in build["source_sha256"]},
        "deliverables":{key:gt._artifact(directory/FILES[key]) for key in payloads},
        "independent_test_pairs":5000,"final_quality_passed":True,
    }


def publish_release(*, joint_selection, selection_sha256, independent_test, test_sha256, output_dir):
    context = _context(independent_test,test_sha256,joint_selection,selection_sha256)
    directory = Path(output_dir).resolve()
    build = _build_contract(context,directory)
    payloads = _payloads(context,directory)
    directory.mkdir(parents=True,exist_ok=True)
    if not (directory/"BUILD_CONTRACT.json").exists() and any(directory.iterdir()):
        raise RuntimeError("release output has unidentified artifacts; preserve it")
    _immutable(directory/"BUILD_CONTRACT.json",build)
    _immutable(directory/"AR_TRAIN_CONFIG.json",context["run"]["config"])
    _immutable(directory/"CLAP_TRAIN_CONFIG.json",context["clap_contract"]["config"])
    for name,digest in build["source_sha256"].items():
        path = proof.REPO/name
        data = path.read_bytes()
        if gt.sha256_file(path) != digest:
            raise RuntimeError("native release source changed during publication")
        artifact = _immutable(directory/"sources"/name,data)
        if artifact["sha256"] != digest:
            raise RuntimeError("native release source snapshot differs from its frozen code")
    for key,payload in payloads.items(): _immutable(directory/FILES[key],payload)
    manifest = _manifest(context,directory,build,payloads)
    artifact = _immutable(directory/"RELEASE.json",manifest)
    validate_release(artifact["path"],expected_sha256=artifact["sha256"])
    return artifact


def validate_release(path, *, expected_sha256):
    path = Path(path).resolve(strict=True)
    if not expected_sha256 or gt.sha256_file(path) != expected_sha256:
        raise RuntimeError("native release requires its pinned SHA256")
    release = json.loads(path.read_text())
    if release.get("schema") != proof.RELEASE_SCHEMA or release.get("status") != "PASS":
        raise RuntimeError("native CLAP44 release was not validated")
    if (set(release.get("deliverables",{})) != set(FILES) or
            release.get("pipeline_contract") != proof.CLAP44_PIPELINE_CONTRACT or
            release.get("model_input_contract") != audio.MODEL_INPUT_CONTRACT or
            release.get("final_quality_passed") is not True):
        raise RuntimeError("native CLAP44 release lacks its required delivery artifacts/contracts")
    directory = path.parent
    if release.get("directory") != str(directory) or path.name != "RELEASE.json":
        raise RuntimeError("native release escaped its fixed archive directory")
    context = _context(release["independent_test"]["path"],release["independent_test"]["sha256"],
                       release["joint_selection"]["path"],release["joint_selection"]["sha256"])
    build = _build_contract(context,directory)
    if _read(release["build_contract"]) != build:
        raise RuntimeError("native release build/model/source identity changed")
    if _read(release["ar_training_config"]) != context["run"]["config"]:
        raise RuntimeError("native release recovery configuration changed")
    if _read(release["clap_training_config"]) != context["clap_contract"]["config"]:
        raise RuntimeError("native release CLAP recovery configuration changed")
    if set(release["source_snapshot"]) != set(build["source_sha256"]):
        raise RuntimeError("native release source snapshot is incomplete")
    for name,digest in build["source_sha256"].items():
        record = release["source_snapshot"][name]
        if record["sha256"] != digest or gt._verify_artifact(record) != directory/"sources"/name:
            raise RuntimeError("native release frozen source copy changed")
    payloads = _payloads(context,directory)
    for key,payload in payloads.items():
        if _read(release["deliverables"][key]) != payload:
            raise RuntimeError(f"native release {key} cannot be replayed from its passed evidence")
    expected = _manifest(context,directory,build,payloads)
    if release != expected:
        raise RuntimeError("native release manifest does not match its proven artifacts")
    return release
