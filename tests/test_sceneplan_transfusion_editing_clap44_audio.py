"""CPU gate/recovery tests with synthetic evidence, never model quality claims."""
import os
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np
import pytest
import soundfile as sf
import torch

from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_clap44_audio as cli
from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_clap44_gt_audio as native_gt
from stable_audio_tools.models import sceneplan_transfusion_editing_clap44_audio_io as native
from test_sceneplan_transfusion_editing_audio_e2e import _quality_rows

audio, gt = native.audio, native.gt


def _write(path, value):
    audio._atomic_json(path,value)
    return gt._artifact(path)


def _wave(tmp_path, *, rate=44100, channels=4, samples=64):
    path = tmp_path/"edited.wav"
    sf.write(path,np.zeros((samples,channels),dtype=np.float32),rate,subtype="FLOAT")
    return gt._artifact(path)


def _record(artifact):
    row = {"pair_ordinal":0,"pair_id":"pair-0","operation":"event_addition",
           "latent_bucket_frames":432,"source_domain":"speech","target_domain":"sound","model_num_samples":64}
    contract = {"fixture":"synthetic audio proof"}
    record = {**row,"schema":audio.SCHEMA,"schema_version":audio.SCHEMA_VERSION,"status":"ok",
        "model_input_contract":deepcopy(audio.MODEL_INPUT_CONTRACT),
        "independent_content_metric_contract":audio.INDEPENDENT_CONTENT_METRIC_CONTRACT,
        "plan_origin":"free_ar","metrics":{"plan_grammar_legal":1.},
        "generated_sceneplan":{"sample_id":"fixture"},"generated_plan_tokens":[1,2],
        "latent_frames_valid":1,"edited_foa_path":artifact["path"],"edited_foa_sha256":artifact["sha256"],
        "runtime_route":native.CLAP44_PIPELINE_CONTRACT,"contract_sha256":gt._digest(contract)}
    return row,contract,record


def test_independent_path_is_unread_until_calibration_passes(tmp_path):
    class UntouchedIndices(dict):
        def __getitem__(self,key):
            raise AssertionError("independent index metadata accessed before calibration")
    preflight = {"indices":UntouchedIndices()}
    def failed(): raise RuntimeError("calibration failed")
    with pytest.raises(RuntimeError,match="calibration failed"):
        native.phase_index(preflight,phase="test",validate_calibration=failed)
    with pytest.raises(RuntimeError,match="passed native audio calibration"):
        native.phase_index(preflight,phase="test",validate_calibration=lambda:{"status":"FAIL","phase":"calibration"})
    path = tmp_path/"validation.sqlite"
    path.touch()
    index,_,calibration = native.phase_index({"indices":{"validation":{"path":str(path)}}},
        phase="calibration",validate_calibration=failed)
    assert index == path and calibration is None


def test_cli_writes_model_freeze_before_resolving_independent_index(tmp_path,monkeypatch):
    identity = {"joint_checkpoint":{"path":"/synthetic/checkpoint","sha256":"fixture"}}
    preflight = {"status":"PASS","indices":{split:{"rows":rows,"sha256":split,
        "path":str(tmp_path/f"{split}.sqlite")} for split,rows in (("train",1000000),("validation",20000),("test",5000))}}
    preflight_path = tmp_path/"PREFLIGHT.json"
    _write(preflight_path,preflight)
    calibration_dir = tmp_path/"calibration"
    cal_contract = _write(calibration_dir/"CONTRACT.json",{"identity":identity,"batch_size_per_rank":2})
    cal_path = calibration_dir/"RESULT.json"
    cal = {"contract":cal_contract,"status":"PASS","phase":"calibration"}
    _write(cal_path,cal)
    args = Namespace(phase="test",preflight=preflight_path,joint_selection="fixture",
        joint_selection_sha256="fixture",post_joint_gt_gate="fixture",calibration=cal_path,
        calibration_sha256=gt.sha256_file(cal_path),batch_size=2)
    monkeypatch.setattr(cli,"upstream_identity",lambda *a:(identity,{"indices":preflight["indices"]}))
    monkeypatch.setattr(cli,"validate_result",lambda *a,**kw:cal)
    directory = tmp_path/"independent-output"
    original_resolve = Path.resolve
    def resolve(path,*a,**kw):
        if path == tmp_path/"test.sqlite":
            saved = json.loads((directory/"PRETEST_FREEZE.json").read_text())
            assert saved["identity"] == identity and saved["calibration"]["sha256"] == args.calibration_sha256
            assert saved["advertised_test_index"] == preflight["indices"]["test"]
            raise RuntimeError("observed freeze before independent path access")
        return original_resolve(path,*a,**kw)
    monkeypatch.setattr(Path,"resolve",resolve)
    with pytest.raises(RuntimeError,match="observed freeze"):
        cli.prepare(args,directory)
    assert not (directory/"CONTRACT.json").exists()
    # A different calibrated batch must fail before creating a second freeze.
    args.batch_size = 4
    with pytest.raises(RuntimeError,match="calibrated model/batch"):
        cli.prepare(args,tmp_path/"different-batch")
    assert not (tmp_path/"different-batch").exists()


@pytest.mark.parametrize("rate,channels,samples",[(48000,4,64),(44100,2,64),(44100,4,65)])
def test_waveform_header_cannot_be_bypassed_with_a_matching_hash(tmp_path,rate,channels,samples):
    artifact = _wave(tmp_path,rate=rate,channels=channels,samples=samples)
    with pytest.raises(RuntimeError,match="exact-length"):
        native.check_audio(artifact["path"],artifact["sha256"],samples=64)


@pytest.mark.parametrize("change",["gt_plan","runtime","length","frames","domain","missing_tokens","audio_sha"])
def test_resume_rejects_wrong_native_audio_evidence(tmp_path,change):
    row,contract,record = _record(_wave(tmp_path))
    native.check_record(record,row,contract)
    if change == "gt_plan": record["plan_origin"] = "ground_truth"
    elif change == "runtime": record["runtime_route"] = "legacy_m2d"
    elif change == "length": record["model_num_samples"] = 65
    elif change == "frames": record["latent_frames_valid"] = 2
    elif change == "domain": record["source_domain"] = "sound"
    elif change == "missing_tokens": record["generated_plan_tokens"] = []
    elif change == "audio_sha": record["edited_foa_sha256"] = "changed"
    with pytest.raises(RuntimeError): native.check_record(record,row,contract)


@pytest.mark.parametrize("change",["canonical","donor","checkpoint","frames","gt_as_ar"])
def test_post_joint_gt_evidence_binds_three_reference_variants(tmp_path,change):
    artifact = _wave(tmp_path)
    row,_,_ = _record(artifact)
    contract = {"donors":{"0":5},"identity":{"joint_checkpoint":{"sha256":"selected"}}}
    record = {**row,"schema":gt.SCHEMA,"status":"ok","contract_sha256":"fixture",
        "plan_origin":"ground_truth","metrics":{},"variant_audio":{k:artifact for k in gt.VARIANTS},
        "model_input_contract":{"editing_ar":None,"old_sceneplan":False},
        "edited_foa_path":artifact["path"],"edited_foa_sha256":artifact["sha256"],
        "plan_representation":"canonical_gt","donor_ordinal":5,
        "native_joint_checkpoint_sha256":"selected","latent_frames_valid":1}
    native_gt.check_record(record,row,contract,"fixture")
    if change == "canonical": record["plan_representation"] = "persistent"
    elif change == "donor": record["donor_ordinal"] = 4
    elif change == "checkpoint": record["native_joint_checkpoint_sha256"] = "other"
    elif change == "frames": record["latent_frames_valid"] = 2
    elif change == "gt_as_ar": record["metrics"]["plan_token_accuracy"] = 1.
    with pytest.raises(RuntimeError): native_gt.check_record(record,row,contract,"fixture")


def test_only_audio_and_instruction_reach_the_native_runtime(tmp_path):
    class Runtime:
        def edit_audio(self,**kwargs):
            assert set(kwargs) == {"source_foa","edit_instructions","model_num_samples","vae_seeds",
                "max_plan_tokens","steps","cfg_scale","initial_noise"}
            assert kwargs["edit_instructions"] == ["move the bell left"]
            assert torch.equal(kwargs["source_foa"][0,:,:64],torch.ones(4,64))
            raise RuntimeError("runtime observed only authorized inputs")
    row = {"pair_id":"fixture","raw_edit_request":"move the bell left","model_num_samples":64}
    truth = {**row,"source_foa":torch.ones(4,64),"offline_old_sceneplan":{"secret":"old"},
             "offline_new_sceneplan":{"secret":"target"},"target_foa":torch.zeros(4,64)}
    with pytest.raises(RuntimeError,match="only authorized inputs"):
        audio._process_batch(pipeline=Runtime(),content_evaluator=None,codec=None,
            samples=[(torch.zeros(64,432),row,{})],truths=[truth],bucket=432,device=torch.device("cpu"),
            args=Namespace(seed=42,max_plan_tokens=512,ode_steps=20,cfg_scale=1.,save_all_audio=True),
            output_dir=tmp_path,listening_ordinals=set())


def _synthetic_evidence(directory):
    directory.mkdir()
    artifact = _wave(directory)
    records = _quality_rows()
    layout = [{key:row[key] for key in ("pair_ordinal","pair_id","operation","latent_bucket_frames")} for row in records]
    selected = list(range(1000))
    contract = {"phase":"calibration","selected_ordinals":selected,"batch_size_per_rank":2,
        "listening_ordinals":sorted(audio._listening_ordinals(layout,5)),
        "identity":{"joint_checkpoint":{"path":"/synthetic/checkpoint","sha256":"fixture"},
                    "joint_selection":{"path":"/synthetic/selection","sha256":"fixture"},
                    "post_joint_gt_audio_gate":{"path":"/synthetic/gt","sha256":"fixture"}}}
    _write(directory/"CONTRACT.json",contract)
    for record in records:
        record.update(source_count=2,target_count=2,source_domain="speech",target_domain="sound",
            plan_origin="free_ar",generated_sceneplan={"sample_id":"synthetic"},generated_plan_tokens=[1,2],
            model_num_samples=64,latent_frames_valid=1,edited_foa_path=artifact["path"],edited_foa_sha256=artifact["sha256"],
            runtime_route=native.CLAP44_PIPELINE_CONTRACT,contract_sha256=gt._digest(contract))
    for rank in range(5):
        for number,ordinals in enumerate(native.rank_batches(layout,selected,rank,2)):
            _write(directory/"shards"/f"rank-{rank}"/f"batch-{number:06d}.json",
                {"rank":rank,"batch":number,"pair_ordinals":ordinals,"contract_sha256":gt._digest(contract),
                 "records":[records[i] for i in ordinals]})
    return contract,layout


def test_synthetic_calibration_replays_original_quality_and_never_claims_independent_pass(tmp_path):
    directory = tmp_path/"synthetic-calibration"
    contract,layout = _synthetic_evidence(directory)
    first = native.derive_result(directory,contract,layout)
    assert first["status"] == "PASS" and first["rows"] == 1000
    assert not first["independent_test_used"] and not first["final_quality_passed"]
    assert first == native.derive_result(directory,contract,layout)
    reference,_ = audio._calibration_thresholds(first["metric_summaries"])
    assert first["thresholds"] == reference
    # One failed inference must stay in the denominator and prevent promotion.
    path = directory/"shards/rank-0/batch-000000.json"
    shard = json.loads(path.read_text())
    shard["records"][0]["status"] = "error"
    _write(path,shard)
    failed = native.derive_result(directory,contract,layout)
    assert failed["status"] == "FAIL" and failed["rows"] == 1000 and failed["successful_rows"] == 999
    assert not failed["checks"]["all_rows_successful"]


def test_resume_requires_exact_five_rank_batch_coverage(tmp_path):
    directory = tmp_path/"synthetic-calibration"
    contract,layout = _synthetic_evidence(directory)
    path = directory/"shards/rank-0/batch-000000.json"
    shard = json.loads(path.read_text())
    shard["pair_ordinals"].reverse()
    shard["records"].reverse()
    _write(path,shard)
    with pytest.raises(RuntimeError,match="shard identity"):
        native.collect_records(directory,contract,layout)
    path.unlink()
    with pytest.raises(RuntimeError,match="exactly cover"):
        native.collect_records(directory,contract,layout)


def test_generated_plan_scores_are_rebound_to_frozen_truth_and_actual_tokens(tmp_path):
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.data.sceneplan_transfusion_editing import sha256_json
    from stable_audio_tools.data.sceneplan_transfusion_editing_plan import canonicalize_editing_plan
    from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import _align_decoded_sceneplan_to_audio_duration
    codec_path = Path(os.environ.get("AMBIT_DATA_ROOT", "data") + "/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    if not codec_path.is_dir(): pytest.skip("local codec unavailable")
    codec = ModelScenePlanCodecV4(codec_path)
    truth = {"sample_id":"target_0","duration_sec":1.,"room":{"type":"moderate"},
        "sources":[{"source_id":"source_0","kind":"sound","description":"a bell",
            "activity":{"onset_sec":.1,"offset_sec":.8},"gain_db":0.,
            "trajectory":{"type":"static","position":{"azimuth_deg":60.,"elevation_deg":0.,"distance_m":2.}}}]}
    target,_ = canonicalize_editing_plan(truth,codec=codec)
    tokens = codec.encode(target,max_tokens=1024)["input_ids"].tolist()
    generated = _align_decoded_sceneplan_to_audio_duration(codec.decode(tokens,sample_id="edited_000000"),1.)
    scores = audio._plan_metrics(codec=codec,target_plan=target,predicted_plan=generated,target_tokens=tokens,predicted_tokens=tokens)
    index = tmp_path/"synthetic-validation.sqlite"
    with sqlite3.connect(index) as connection:
        connection.execute("CREATE TABLE pairs(pair_ordinal INTEGER,pair_id TEXT,model_num_samples INTEGER,new_sceneplan_zlib BLOB,new_sceneplan_sha256 TEXT)")
        connection.execute("INSERT INTO pairs VALUES(0,'pair-0',44100,?,?)",(zlib.compress(json.dumps(truth).encode()),sha256_json(truth)))
    record = {"status":"ok","pair_ordinal":0,"pair_id":"pair-0","model_num_samples":44100,
        "generated_sceneplan":generated,"generated_plan_tokens":tokens,"metrics":scores}
    native.rebind_plan_metrics([record],index,codec)
    changed = deepcopy(record)
    changed["metrics"]["plan_grammar_legal"] = .5
    with pytest.raises(RuntimeError,match="cannot be replayed"):
        native.rebind_plan_metrics([changed],index,codec)
    changed = deepcopy(record)
    changed["generated_sceneplan"]["sources"][0]["gain_db"] += 1
    with pytest.raises(RuntimeError,match="differs from its generated tokens"):
        native.rebind_plan_metrics([changed],index,codec)
    with sqlite3.connect(index) as connection:
        connection.execute("UPDATE pairs SET new_sceneplan_sha256='changed'")
    with pytest.raises(RuntimeError,match="frozen plan/sample identity"):
        native.rebind_plan_metrics([record],index,codec)


def test_empty_release_cannot_pass_even_if_it_declares_pass(tmp_path):
    path = tmp_path/"RELEASE.json"
    _write(path,{"schema":native.RELEASE_SCHEMA,"status":"PASS","deliverables":{}})
    with pytest.raises(RuntimeError,match="required delivery"):
        native.validate_release(path,expected_sha256=gt.sha256_file(path))
