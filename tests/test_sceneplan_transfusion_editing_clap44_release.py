"""Synthetic CPU publication/runtime tests; no trained quality evidence."""
from argparse import Namespace
from copy import deepcopy
import fcntl
import json
from pathlib import Path
import shlex
import sqlite3

import numpy as np
import pytest
import soundfile as sf
import torch

from stable_audio_tools.models import sceneplan_transfusion_editing_clap44_audio_io as proof
from stable_audio_tools.models import sceneplan_transfusion_editing_clap44_pipeline as runtime
from stable_audio_tools.models import sceneplan_transfusion_editing_clap44_release as release
from stable_audio_tools.models import sceneplan_transfusion_editing_clap44_io as encoder_io
from scripts.t2a.inference import edit_foa_with_clap44 as inference

gt, audio = proof.gt, proof.audio


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True)+"\n")
    return gt._artifact(path)


@pytest.fixture
def synthetic_release(tmp_path,monkeypatch):
    repo = tmp_path/"repository with spaces"
    repo.mkdir()
    (repo/"bound.py").write_text("frozen_fixture = True\n")
    monkeypatch.setattr(proof,"REPO",repo)
    monkeypatch.setattr(proof,"code_hashes",lambda:{"bound.py":gt.sha256_file(repo/"bound.py")})
    assets = tmp_path/"synthetic-assets"
    assets.mkdir()
    waveform = assets/"source.wav"
    sf.write(waveform,np.zeros((64,4),np.float32),44100,subtype="FLOAT")
    wave = gt._artifact(waveform)
    selection = write(assets/"joint/evaluation/custom-selection/SELECTED.json",{"fixture":"selection"})
    checkpoint = write(assets/"joint/checkpoints/chosen.pt",{"fixture":"weights"})
    base = write(assets/"base/evaluation/custom-dit/SELECTED.json",{"fixture":"base"})
    pre_gt = write(assets/"base/evaluation/custom-gt/GATE.json",{"fixture":"pre-gt"})
    model = write(assets/"model.json",{"fixture":"model"})
    preflight = write(assets/"PREFLIGHT.json",{"fixture":"preflight"})
    run = {
        "run_dir":str(assets/"joint"),"variant":"global_and_sequence","config":{"seed":42,"purpose":"full_AR_training"},
        "model_config":model["path"],"codec":str(assets/"codec"),"codec_sha256":"fixture",
        "clap_checkpoint":{"path":str(assets/"clap/step-002000.pt"),"sha256":"fixture"},
        "clap_validation":{"path":str(assets/"clap/validation/REPORT.json"),"sha256":"fixture"},
        "base_selection":base,"dit_gt_audio_gate":pre_gt,
        "indices":{"train":{"path":str(assets/"train.sqlite")}},
    }
    clap_contract = {"config":{"seed":42,"model":{"fixture":"encoder"}}}
    write(assets/"clap/TRAIN_CONTRACT.json",clap_contract)
    monkeypatch.setattr(encoder_io,"load_clap44_checkpoint",lambda *a,**kw:(object(),{"contract":clap_contract}))
    run_artifact = write(assets/"joint/RUN_CONTRACT.json",run)
    gt_contract = write(assets/"joint/evaluation/custom-gt/CONTRACT.json",{"identity":{"base_selection":base}})
    gt_gate = write(assets/"joint/evaluation/custom-gt/GATE.json",{"contract":gt_contract,"result":{"status":"PASS"}})
    calibration = write(assets/"joint/evaluation/custom-calibration/RESULT.json",
                        {"phase":"calibration","status":"PASS","checks":{"quality":True},"thresholds":{}})
    freeze = write(assets/"joint/evaluation/custom-test/PRETEST_FREEZE.json",{"fixture":"frozen before test"})
    records = []
    for operation in audio.OPERATIONS:
        for bucket in (432,648):
            for _ in range(500):
                ordinal = len(records)
                records.append({
                    "pair_ordinal":ordinal,"pair_id":f"pair-{ordinal}","operation":operation,
                    "latent_bucket_frames":bucket,"model_num_samples":64,
                    "source_foa_path":wave["path"],"target_foa_path":wave["path"],
                    "edited_foa_path":wave["path"],"edited_foa_sha256":wave["sha256"],
                    "generated_sceneplan":{"sample_id":f"edited-{ordinal}","sources":[]},
                    "metrics":{"plan_grammar_legal":1.,"audio_codec_foa_progress":ordinal/5000,
                               "doa_target_mean_deg":1.+ordinal/5000},
                })
    index = assets/"synthetic-test.sqlite"
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE pairs(pair_ordinal INTEGER,pair_id TEXT,raw_edit_request TEXT,model_num_samples INTEGER)")
        connection.executemany("INSERT INTO pairs VALUES(?,?,?,64)",[
            (row["pair_ordinal"],row["pair_id"],"move the bell left") for row in records])
    record_artifact = write(assets/"joint/evaluation/custom-test/shards/rank-0/batch-000000.json",{"records":records})
    identity = {"joint_selection":selection,"joint_checkpoint":checkpoint,"joint_run_contract":run_artifact,
                "post_joint_gt_audio_gate":gt_gate,"model_config":model}
    contract = {
        "identity":identity,"calibration":calibration,"pretest_freeze":freeze,"preflight":preflight,
        "index":gt._artifact(index),"policy":proof.policy(),"frozen_assets":{"fixture":"frozen"},
        "selected_ordinals":list(range(5000)),
        "listening_ordinals":[base+i for base in range(0,5000,500) for i in range(5)],"batch_size_per_rank":2,
    }
    contract_artifact = write(assets/"joint/evaluation/custom-test/CONTRACT.json",contract)
    result = {"status":"PASS","phase":"test","final_quality_passed":True,"rows":5000,"successful_rows":5000,
        "independent_test_used":True,"contract":contract_artifact,"joint_selection":selection,"selected_checkpoint":checkpoint,
        "post_joint_gt_audio_gate":gt_gate,"record_artifacts":[record_artifact],
        "checks":{"quality":True},"metric_summaries":{"synthetic":{}},"difficulty":{"source_count":{}}}
    test = write(assets/"joint/evaluation/custom-test/RESULT.json",result)
    validations = []
    def verified(path,**kwargs):
        assert Path(path).resolve()==Path(test["path"]) and kwargs["expected_sha256"]==test["sha256"]
        assert kwargs["selection_sha256"]==selection["sha256"] and kwargs["phase"]=="test"
        validations.append(kwargs["phase"])
        return deepcopy(result)
    monkeypatch.setattr(proof,"validate_result",verified)
    args = dict(joint_selection=selection["path"],selection_sha256=selection["sha256"],
                independent_test=test["path"],test_sha256=test["sha256"],output_dir=tmp_path/"release")
    return args,result,validations,contract


def test_publication_binds_full_test_listening_sources_and_actual_recovery_paths(synthetic_release):
    args,_,validations,contract = synthetic_release
    artifact = release.publish_release(**args)
    seal = release.validate_release(artifact["path"],expected_sha256=artifact["sha256"])
    assert len(validations)==3  # initial proof, post-publication check, explicit reload
    assert seal["independent_test_pairs"]==5000 and seal["checkpoint"]==contract["identity"]["joint_checkpoint"]
    assert set(seal["deliverables"])==set(release.FILES)
    assert seal["source_snapshot"]["bound.py"]["sha256"]==seal["source_sha256"]["bound.py"]
    listening = json.loads(Path(seal["deliverables"]["listening_package"]["path"]).read_text())
    assert len(listening["representative_ordinals"])==50
    assert set(listening["representative_coverage"].values())=={5}
    assert listening["human_listening_review"]=="not_recorded_by_automated_publication"
    assert all(row["source_foa"] and row["target_foa"] and row["edited_foa"] and row["edit_instruction"] for row in listening["examples"])
    # Phase-unidentified additions are excluded from phase-sensitive weak ranking.
    assert all(row["pair_ordinal"]>=1000 for row in listening["weakest_by_metric"]["audio_codec_foa_progress"])
    reproduction = json.loads(Path(seal["deliverables"]["reproduction"]["path"]).read_text())
    commands = reproduction["commands"]
    assert commands["resume_audio_test"]["argv"][-1]==contract["calibration"]["path"]
    assert str(Path(args["independent_test"]).parent) in commands["resume_audio_test"]["argv"]
    assert str(Path(args["joint_selection"]).parent) in commands["resume_joint_selection"]["argv"]
    assert "--selection-sha256" in commands["verify_joint_selection"]["argv"]
    assert commands["reproduce_clap_pretraining_in_new_run"]["env"]["CLAP44_RUN_ROOT"].endswith("/reproduction_runs/clap44")
    assert commands["reproduce_joint_training_in_new_run"]["env"]["CLAP44_AR_RUN_DIR"]!=str(Path(args["joint_selection"]).parents[2])
    for command in commands.values():
        words = shlex.split(command["shell"])
        assert words[-len(command["argv"]):]==command["argv"]


def test_interrupted_publication_reuses_identical_artifacts(synthetic_release,monkeypatch):
    args,_,_,_ = synthetic_release
    original = release._immutable
    def interrupted(path,value):
        if Path(path).name=="RELEASE.json": raise InterruptedError("publication interrupted")
        return original(path,value)
    monkeypatch.setattr(release,"_immutable",interrupted)
    with pytest.raises(InterruptedError): release.publish_release(**args)
    root = args["output_dir"]
    files = {str(path.relative_to(root)):path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}
    assert not (root/"RELEASE.json").exists()
    monkeypatch.setattr(release,"_immutable",original)
    artifact = release.publish_release(**args)
    assert Path(artifact["path"]).exists()
    assert all((root/name).stat().st_mtime_ns==mtime for name,mtime in files.items())
    assert release.publish_release(**args)==artifact


@pytest.mark.parametrize("change",["report","listening","missing_source","checkpoint","missing_deliverable"])
def test_release_replays_deliverables_even_if_manifest_hash_is_updated(synthetic_release,change):
    args,_,_,_ = synthetic_release
    artifact = release.publish_release(**args)
    path = Path(artifact["path"])
    seal = json.loads(path.read_text())
    if change in ("report","listening"):
        key = "evaluation_report" if change=="report" else "listening_package"
        target = Path(seal["deliverables"][key]["path"])
        value = json.loads(target.read_text())
        if change=="report": value["dataset_pairs"]["test"]=1000
        else: value["representative_ordinals"].pop()
        seal["deliverables"][key] = write(target,value)
    elif change=="missing_source": seal["source_snapshot"].clear()
    elif change=="checkpoint": seal["checkpoint"]["sha256"]="other checkpoint"
    elif change=="missing_deliverable": seal["deliverables"].pop("known_limitations")
    write(path,seal)
    with pytest.raises(RuntimeError):
        release.validate_release(path,expected_sha256=gt.sha256_file(path))


def test_failed_independent_proof_creates_no_release(synthetic_release,monkeypatch):
    args,_,_,_ = synthetic_release
    def failed(*a,**kw): raise RuntimeError("independent quality failed")
    monkeypatch.setattr(proof,"validate_result",failed)
    with pytest.raises(RuntimeError,match="independent quality failed"):
        release.publish_release(**args)
    assert not args["output_dir"].exists()


def test_formal_loader_requires_release_before_loading_candidate(synthetic_release,monkeypatch):
    args,_,_,contract = synthetic_release
    artifact = release.publish_release(**args)
    calls = []
    def candidate(path,**kwargs):
        assert kwargs["load_audio_autoencoder"] is True
        calls.append(path)
        return object(),{"checkpoint_sha256":contract["identity"]["joint_checkpoint"]["sha256"],
            "m2d_used":False,"shared_transformer_same_object":True}
    monkeypatch.setattr(runtime,"load_clap44_joint_candidate",candidate)
    _,report = runtime.load_clap44_validated_release(artifact["path"],expected_sha256=artifact["sha256"],device="cpu")
    assert report["quality_gate_passed"] and report["independent_test_pairs"]==5000
    assert len(calls)==1
    with pytest.raises(RuntimeError,match="pinned SHA256"):
        runtime.load_clap44_validated_release(artifact["path"],expected_sha256="wrong",device="cpu")
    assert len(calls)==1
    monkeypatch.setattr(runtime,"load_clap44_joint_candidate",lambda *a,**kw:(object(),
        {"checkpoint_sha256":"other","m2d_used":False,"shared_transformer_same_object":True}))
    with pytest.raises(RuntimeError,match="different checkpoint"):
        runtime.load_clap44_validated_release(artifact["path"],expected_sha256=artifact["sha256"],device="cpu")


def test_formal_audio_request_outputs_complete_plan_and_exact_foa_then_reuses_result(tmp_path,monkeypatch):
    waveform = tmp_path/"input.wav"
    sf.write(waveform,np.ones((64,4),np.float32)*.1,44100,subtype="FLOAT")
    seal = write(tmp_path/"RELEASE.json",{"fixture":"release oracle"})
    args = Namespace(source=waveform,release=Path(seal["path"]),release_sha256=seal["sha256"],
        instruction="move the bell left",output_dir=tmp_path/"output",seed=42,device="cpu",physical_gpu=3)
    calls = []
    class Pipeline:
        def edit_audio(self,source,instructions,**kwargs):
            assert source.shape==(1,4,64) and instructions==[args.instruction]
            assert set(kwargs)=={"model_num_samples","sample_rate","vae_seeds","noise_seed","max_plan_tokens","steps","cfg_scale"}
            assert kwargs["model_num_samples"]==[64] and kwargs["sample_rate"]==44100
            calls.append("model")
            return {"edited_foa":torch.cat((source+0.1,torch.zeros(1,4,64)),dim=-1),
                "sample_attention_mask":torch.cat((torch.ones(1,64,dtype=torch.bool),torch.zeros(1,64,dtype=torch.bool)),dim=-1),
                "new_sceneplans":[{"sample_id":"generated","duration_sec":64/44100,"sources":[{"source_id":"source_0"}]}],
                "new_sceneplan_token_ids":[torch.tensor([1,7,2])]}
    monkeypatch.setattr(inference,"load_clap44_validated_release",lambda *a,**kw:(Pipeline(),
        {"quality_gate_passed":True,"release_sha256":seal["sha256"]}))
    monkeypatch.setattr(inference,"validate_release",lambda *a,**kw:{"fixture":"proof replayed"})
    result = inference.run(args)
    info = sf.info(result["edited_foa"]["path"])
    assert (info.samplerate,info.channels,info.frames)==(44100,4,64)
    plan = json.loads(Path(result["new_sceneplan"]["path"]).read_text())
    assert plan["origin"]=="free_ar" and plan["token_ids"]==[1,7,2] and len(plan["sceneplan"]["sources"])==1
    assert result["individual_edit_quality"]=="not_inferred_from_model_release_status"
    assert inference.run(args)==result and calls==["model"]
    args.instruction="remove everything"
    with pytest.raises(RuntimeError,match="another source/instruction"):
        inference.run(args)


def test_locked_editing_chain_prevents_any_cuda_probe(tmp_path,monkeypatch):
    lock = tmp_path/"training-chain.lock"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:False)
    def forbidden(*a,**kw): raise AssertionError("CUDA was accessed while another Editing task held the lock")
    monkeypatch.setattr(torch.cuda,"is_available",forbidden)
    with lock.open("a+b") as owner:
        fcntl.flock(owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(RuntimeError,match="already running"):
            with inference.editing_device("cuda",3,lock_path=lock): pass


def test_authorized_physical_gpu_is_selected_by_uuid_without_using_cuda(tmp_path,monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","3,4,5,6,7")
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:False)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"device_count",lambda:1)
    uuid = "GPU-00000000-0000-0000-0000-000000000003"
    def topology(argv,**kwargs):
        assert argv[:3]==["nvidia-smi","-i","3"]
        return uuid+"\n"
    monkeypatch.setattr(inference.subprocess,"check_output",topology)
    with inference.editing_device("cuda",3,lock_path=tmp_path/"lock") as device:
        assert device==torch.device("cuda",0)
        assert inference.os.environ["CUDA_VISIBLE_DEVICES"]==uuid
    with pytest.raises(ValueError,match="only physical"):
        with inference.editing_device("cuda",0,lock_path=tmp_path/"lock"): pass
