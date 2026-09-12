"""CPU evidence tests; these fixtures are not trained-model quality results."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3
import zlib

import pytest
import torch
from torch import nn

from stable_audio_tools.models.sceneplan_transfusion_editing_ar import EDITING_AR_CLAP44_CONTRACT
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_io import file_sha256
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_joint_io import (
    JOINT44_RUN_SCHEMA, atomic_json, codec_artifact_sha256, ensure_run_identity,
    load_joint_checkpoint, resolve_joint_resume, save_joint_checkpoint,
)
from stable_audio_tools.models.sceneplan_transfusion_editing_clap44_selection_io import (
    SELECTION_STATUS, candidate_derivations, derive_selection, load_work, merge_raw, save_work, verify_free_truth,
)
from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_selection import (
    OPERATIONS, SOURCE_VARIANTS, evaluate_ar, noninferiority, paired_gate, rank_candidates, subset,
)
from scripts.t2a.train.train_sceneplan_transfusion_editing_ar_joint_full import _capture_rank_rng_state
from scripts.t2a.eval.select_sceneplan_transfusion_editing_dit_checkpoint import (
    DEFAULT_TIMESTEPS, _rank_ordinals, _selection_holdout_folds,
)


@pytest.fixture
def own_run(tmp_path):
    identity = ensure_run_identity(tmp_path)
    source = tmp_path/"bound.py"
    source.write_text("model = 1\n")
    contract = {"schema":JOINT44_RUN_SCHEMA,"run_dir":str(tmp_path),"run_id":identity["run_id"],
                "repo_root":str(tmp_path),"ar_contract":EDITING_AR_CLAP44_CONTRACT,
                "variant":"global_and_sequence","m2d_used":False,"independent_test_used":False,
                "world_size":1,"schedule":{"max_steps":2},"source_sha256":{"bound.py":file_sha256(source)}}
    atomic_json(tmp_path/"RUN_CONTRACT.json",contract)
    module = nn.Module()
    module.diffusion = nn.Linear(3,3)
    module.ar = nn.Module()
    module.ar.adapter = nn.Linear(3,3)
    optimizer = torch.optim.AdamW(module.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    def save(step):
        optimizer.zero_grad()
        sum(p.square().sum() for p in module.parameters()).backward()
        optimizer.step()
        scheduler.step()
        rng = _capture_rank_rng_state(rank=0,device=None)
        rng["torch_cuda_rng_state"] = torch.zeros(8,dtype=torch.uint8)
        return save_joint_checkpoint(tmp_path/f"checkpoints/step-{step:08d}.pt",
            module=module,optimizer=optimizer,scheduler=scheduler,step=step,epoch=0,
            next_batch=step*4,contract=contract,rng_states=[rng])
    return tmp_path, contract, module, save


@pytest.mark.parametrize("failure", ["missing_manifest","stale_latest","missing_latest","invalid_latest_json"])
def test_checkpoint_publication_recovery_retains_exact_saved_state(own_run, failure):
    root, contract, module, save = own_run
    first = save(1)
    second = save(2)
    path = Path(second["checkpoint"])
    latest = root/"checkpoints/LATEST.json"
    if failure == "missing_manifest":
        path.with_suffix(".manifest.json").unlink()
        atomic_json(latest,first)
    elif failure == "stale_latest":
        atomic_json(latest,first)
    elif failure == "missing_latest":
        latest.unlink()
    else:
        latest.write_text("{partial")
    assert resolve_joint_resume(root) == second
    payload,_ = load_joint_checkpoint(path,expected_contract=contract,require_latest=True)
    assert payload["global_step"] == 2 and payload["next_batch"] == 8
    assert payload["scheduler"]["last_epoch"] == 2
    for key,tensor in module.diffusion.state_dict().items():
        torch.testing.assert_close(payload["diffusion_state_dict"][key],tensor)
    assert file_sha256(first["checkpoint"]) == first["checkpoint_sha256"]
    assert (root/"checkpoints/PUBLICATION_RECOVERY.json").is_file()


def test_recovery_refuses_changed_published_weights_and_foreign_sources(own_run):
    root, _, _, save = own_run
    record = save(1)
    path = Path(record["checkpoint"])
    payload = torch.load(path,weights_only=True)
    payload["diffusion_state_dict"]["weight"] += 1
    torch.save(payload,path)
    original_manifest = path.with_suffix(".manifest.json").read_bytes()
    with pytest.raises(RuntimeError,match="manifest/hash"):
        resolve_joint_resume(root)
    assert path.with_suffix(".manifest.json").read_bytes() == original_manifest
    (root/"bound.py").write_text("model = 2\n")
    with pytest.raises(RuntimeError,match="source changed"):
        resolve_joint_resume(root)


def test_partial_temporary_binary_is_never_adopted(own_run):
    root, _, _, _ = own_run
    directory = root/"checkpoints"
    directory.mkdir()
    temp = directory/"step-00000001.pt.tmp.123"
    temp.write_bytes(b"partial")
    assert resolve_joint_resume(root) is None
    assert temp.read_bytes() == b"partial"


def test_unpublished_checkpoint_with_stale_optimizer_progress_is_not_recovered(own_run):
    root, _, _, save = own_run
    record = save(1)
    path = Path(record["checkpoint"])
    payload = torch.load(path,weights_only=True)
    payload["scheduler"]["last_epoch"] = 0
    torch.save(payload,path)
    path.with_suffix(".manifest.json").unlink()
    with pytest.raises(RuntimeError,match="optimizer-step progress differ"):
        resolve_joint_resume(root)


def test_real_codec_directory_identity_covers_config_and_vocabulary(tmp_path):
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    real = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    if not real.is_dir(): pytest.skip("local frozen codec unavailable")
    for name in ("READY","codec.json","sentencepiece.model"):
        shutil.copyfile(real/name,tmp_path/name)
    assert ModelScenePlanCodecV4(tmp_path).fingerprint == ModelScenePlanCodecV4(real).fingerprint
    before = codec_artifact_sha256(tmp_path)
    assert before == codec_artifact_sha256(real)
    with (tmp_path/"sentencepiece.model").open("ab") as stream:
        stream.write(b"changed")
    assert codec_artifact_sha256(tmp_path) != before


def small_rf():
    operations = [name for name in OPERATIONS for _ in range(40)]
    buckets = [432,648]*100
    return {"ordinals":list(range(200)),"operations":operations,"buckets":buckets,
            "losses":{"clean":torch.ones(200,len(DEFAULT_TIMESTEPS),dtype=torch.float64)}}


def test_rf_noninferiority_preserves_old_limits_and_independent_pair_count(tmp_path):
    import torch.distributed as dist
    from scripts.t2a.eval.select_sceneplan_transfusion_editing_joint_checkpoint import _noninferiority_report
    base = small_rf()
    joint = copy.deepcopy(base)
    joint["losses"]["clean"] += .002
    observed = noninferiority(joint,base)
    assert observed["pass"] and observed["overall"]["rows"] == 200
    dist.init_process_group("gloo",init_method="file://"+str(tmp_path/"gloo"),rank=0,world_size=1)
    try:
        legacy = _noninferiority_report(joint["losses"]["clean"],base["losses"]["clean"],
            operations=base["operations"],buckets=base["buckets"],device=torch.device("cpu"))
    finally:
        dist.destroy_process_group()
    assert observed["pass"] == legacy["pass"]
    # The legacy sum-of-squares formula leaves a ~1e-11 cancellation
    # residual on a constant vector; the native derivation uses stable std.
    assert observed["overall"]["one_sided_upper_confidence_bound"] == pytest.approx(legacy["overall"]["one_sided_upper_confidence_bound"],abs=1e-10)
    joint["losses"]["clean"][:40] = 1.011
    assert not noninferiority(joint,base)["by_operation"][OPERATIONS[0]]["pass"]
    joint["losses"]["clean"].fill_(1.006)
    assert not noninferiority(joint,base)["point_estimate_pass"]


def test_source_gate_uses_pairs_and_strictly_positive_confidence_bound():
    raw = small_rf()
    assert paired_gate(torch.ones(200,6),raw)["pass"]
    assert not paired_gate(torch.zeros(200,6),raw)["pass"]
    values = torch.ones(200,6)
    values[:40] = -1
    assert not paired_gate(values,raw)["pass"]
    with pytest.raises(ValueError,match="finite"):
        paired_gate(torch.full((200,6),float("nan")),raw)


def test_checkpoint_ranking_cannot_use_holdout_or_failed_rf_candidate():
    def candidate(step,ce,rf_pass=True):
        return {"step":step,"selection_10k":{"ar":{"clean_ce":{"mean":ce},"token_accuracy":{"mean":.8}},
                "rf":{"mean":1.},"base_dit_noninferiority":{"pass":rf_pass}},
                "clean_full_20k":{"ar_ce":1000.}}
    rows = [candidate(5000,.5),candidate(10000,.1,False),candidate(15000,.7)]
    assert [row["step"] for row in rank_candidates(rows)] == [5000,15000]
    rows[0]["clean_full_20k"] = {"ar_ce":1e10,"holdout_gate":False}
    rows[2]["clean_full_20k"] = {"ar_ce":0.,"holdout_gate":True}
    assert rank_candidates(rows)[0]["step"] == 5000


def test_work_resume_detects_stale_identity_and_changed_bytes(tmp_path):
    atomic_json(tmp_path/"EVALUATION_CONTRACT.json",{"bound":"run_1"})
    assert load_work(tmp_path,"base",0,required=False) is None
    save_work(tmp_path,"base",0,{"scores":torch.ones(3)})
    assert torch.equal(load_work(tmp_path,"base",0)["scores"],torch.ones(3))
    with pytest.raises(FileExistsError):
        save_work(tmp_path,"base",0,{"scores":torch.zeros(3)})
    path = tmp_path/"work/base-rank-0.pt"
    with path.open("ab") as stream: stream.write(b"changed")
    with pytest.raises(RuntimeError,match="hash/contract"):
        load_work(tmp_path,"base",0)


def test_full_population_merge_rejects_rank_duplicates_and_uses_same_folds():
    layout = [(ordinal,432 if ordinal < 15000 else 648,OPERATIONS[ordinal%5]) for ordinal in range(20000)]
    values = []
    for rank in range(5):
        assigned,_ = _rank_ordinals(layout,rank)
        values.append({"ordinals":assigned,"buckets":[layout[i][1] for i in assigned],
                       "operations":[layout[i][2] for i in assigned],
                       "losses":{"clean":torch.ones(4000,len(DEFAULT_TIMESTEPS))},"prediction_l1":{}})
    merged = merge_raw(values,layout)
    assert merged["ordinals"] == list(range(20000))
    folds,_ = _selection_holdout_folds(layout)
    assert len(subset(merged,folds,"selection")["ordinals"]) == 10000
    assert len(subset(merged,folds,"holdout")["ordinals"]) == 10000
    values[1]["ordinals"] = values[0]["ordinals"]
    with pytest.raises(RuntimeError,match="rank dropped"):
        merge_raw(values,layout)


def test_native_ar_evaluator_derives_source_features_and_runs_every_intervention():
    path = Path(__file__).with_name("test_sceneplan_transfusion_editing_clap44_integration.py")
    spec = importlib.util.spec_from_file_location("clap44_selection_harness",path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    torch.manual_seed(19)
    ar = fixture.ar_harness().eval()
    ar.encode_edit_instructions = lambda texts,device: (torch.zeros(len(texts),1,1024),torch.ones(len(texts),1,dtype=torch.bool))
    with torch.no_grad():
        ar.source_semantic_bridge.global_projection.weight.normal_(std=.03)
    x = torch.randn(2,64,12)
    mask = torch.ones(2,12,dtype=torch.bool)
    tokens = torch.tensor([[1,3,4],[1,4,5]])
    rows = [{"pair_ordinal":i,"operation":OPERATIONS[0],"latent_bucket_frames":432,
             "model_sceneplan":{"sources":[{},{}]}} for i in range(2)]
    batch = {"metadata":rows,"ar":{"source_foa_latent":x,"source_attention_mask":mask,
             "plan_input_ids":tokens,"plan_labels":tokens,"plan_attention_mask":torch.ones_like(tokens,dtype=torch.bool),
             "raw_edit_requests":["add a bell","add a bell"]}}
    class Donor:
        def values(self,*args): return x.flip(0),mask.flip(0)
    result = evaluate_ar(ar,[batch],device=torch.device("cpu"),variants=SOURCE_VARIANTS,donor_resolver=Donor())
    assert set(result["losses"]) == set(SOURCE_VARIANTS)
    assert set(result["teacher_cosine"]) == {"semantic_matched","semantic_shuffled","scene_matched","scene_shuffled"}
    for name in SOURCE_VARIANTS[1:]:
        assert result["response_l1"][name].min() > 0
    assert all(p.grad is None and not p.requires_grad for p in ar.source_clap_model.parameters())


def test_free_plan_scores_rebind_to_actual_index_truth(tmp_path):
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.data.sceneplan_transfusion_editing import sha256_json
    from stable_audio_tools.data.sceneplan_transfusion_editing_plan import canonicalize_editing_plan
    codec_dir = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    if not codec_dir.is_dir(): pytest.skip("local codec unavailable")
    codec = ModelScenePlanCodecV4(codec_dir)
    plan = {"sample_id":"target_0","duration_sec":1.,"room":{"type":"moderate"},
            "sources":[{"source_id":"source_0","kind":"sound","description":"a bell","activity":{"onset_sec":.1,"offset_sec":.8},
                        "gain_db":0.,"trajectory":{"type":"static","position":{"azimuth_deg":60.,"elevation_deg":0.,"distance_m":2.}}}]}
    canonical,_ = canonicalize_editing_plan(plan,codec=codec)
    tokens = codec.encode(canonical,max_tokens=1024)["input_ids"].tolist()
    path = tmp_path/"validation.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE pairs(pair_ordinal INTEGER,split TEXT,pair_id TEXT,operation TEXT,latent_bucket_frames INTEGER,target_sample_id TEXT,model_num_samples INTEGER,new_sceneplan_zlib BLOB,new_sceneplan_sha256 TEXT)")
    connection.execute("INSERT INTO pairs VALUES(0,'validation','pair_0',?,432,'target_0',44100,?,?)",
                       (OPERATIONS[0],zlib.compress(json.dumps(plan).encode()),sha256_json(plan)))
    connection.commit()
    connection.close()
    record = {"pair_ordinal":0,"pair_id":"pair_0","operation":OPERATIONS[0],"latent_bucket_frames":432,
              "target_sample_id":"target_0","duration_sec":1.,"target_token_ids":tokens,"predicted_token_ids":tokens,
              "metrics":{"grammar_legal":0.},"error":None}
    value = {"validation_index":{"path":str(path)},"free_ordinals":[0]}
    replay = verify_free_truth([record],value,codec)
    assert replay[0]["metrics"]["grammar_legal"] == 1.
    record["target_token_ids"] = tokens[:-1]
    with pytest.raises(RuntimeError,match="frozen validation truth"):
        verify_free_truth([record],value,codec)


@pytest.mark.parametrize("bad_holdout", [False,True])
def test_complete_synthetic_selection_replays_all_evidence_without_holdout_fallback(tmp_path,bad_holdout):
    """Exercise orchestration on synthetic scores, never real model evidence."""
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    from stable_audio_tools.data.sceneplan_transfusion_editing import sha256_json
    from stable_audio_tools.data.sceneplan_transfusion_editing_plan import canonicalize_editing_plan
    from stable_audio_tools.training.sceneplan_transfusion_editing_clap44_selection import CANDIDATE_STEPS
    codec_dir = Path("/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4")
    if not codec_dir.is_dir(): pytest.skip("local codec unavailable")
    codec = ModelScenePlanCodecV4(codec_dir)
    layout = [(i,432 if i<15000 else 648,OPERATIONS[i%5]) for i in range(20000)]
    folds,_ = _selection_holdout_folds(layout)
    free = sorted(i for operation in OPERATIONS for bucket in (432,648)
                  for i in [j for j,b,op in layout if op==operation and b==bucket and folds[j]=="holdout"][:50])
    value = {"layout":layout,"training_run":{"candidates":[{"step":step,"checkpoint":f"/synthetic/step-{step}.pt",
             "checkpoint_sha256":str(step)} for step in CANDIDATE_STEPS]},
             "base_dit_selection":{"selected_clean_source_rf":{"mean":1.}},
             "validation_index":{"path":str(tmp_path/"validation.sqlite")},"free_ordinals":free}
    contract = {"variant":"global_and_sequence","codec":str(codec_dir),
                "config":{"lambda_source_distillation":.05}}
    atomic_json(tmp_path/"EVALUATION_CONTRACT.json",{"purpose":"synthetic_CPU_test_only"})
    for rank in range(5):
        assigned,_ = _rank_ordinals(layout,rank)
        base = {"ordinals":assigned,"buckets":[layout[i][1] for i in assigned],"operations":[layout[i][2] for i in assigned],
                "losses":{"clean":torch.ones(4000,6,dtype=torch.float64)},"prediction_l1":{}}
        save_work(tmp_path,"base",rank,base)
        for step in CANDIDATE_STEPS:
            rf = copy.deepcopy(base)
            rf["losses"]["clean"] += .001
            if bad_holdout and step==25000:
                rf["losses"]["clean"][[i for i,ordinal in enumerate(assigned) if folds[ordinal]=="holdout"]] = 1.1
            ar = {key:base[key] for key in ("ordinals","operations","buckets")}
            ar.update(tokens=torch.ones(4000)*30,accuracy=torch.ones(4000)*.9,exact=torch.ones(4000),
                      source_counts=[1]*4000,target_counts=[1]*4000,
                      losses={"clean":torch.full((4000,),1-step/100000.)},response_l1={},teacher_cosine={})
            save_work(tmp_path,f"clean-{step}",rank,{"ar":ar,"rf":rf})
        for name in SOURCE_VARIANTS[1:]:
            ar["losses"][name] = ar["losses"]["clean"]+.1
            ar["response_l1"][name] = torch.ones(4000)*.1
        ar["teacher_cosine"] = {head+"_"+pairing:torch.full((4000,),score)
                               for head in ("semantic","scene") for pairing,score in (("matched",.9),("shuffled",.1))}
        for name in ("zero","shuffled"):
            rf["losses"][name] = rf["losses"]["clean"]+.1
            rf["prediction_l1"][name] = torch.ones(4000,6)*.1
        save_work(tmp_path,"interventions-25000",rank,{"ar":ar,"rf":rf})
    _,ranked,_,_ = candidate_derivations(tmp_path,value)
    assert ranked[0]["step"] == 25000
    atomic_json(tmp_path/"RANKING.json",{"evaluation_contract_sha256":file_sha256(tmp_path/"EVALUATION_CONTRACT.json"),
                "ranked_steps":[row["step"] for row in ranked],"selected_step":25000})
    db = sqlite3.connect(value["validation_index"]["path"])
    db.execute("CREATE TABLE pairs(pair_ordinal INTEGER,split TEXT,pair_id TEXT,operation TEXT,latent_bucket_frames INTEGER,target_sample_id TEXT,model_num_samples INTEGER,new_sceneplan_zlib BLOB,new_sceneplan_sha256 TEXT)")
    free_records = []
    for ordinal in free:
        plan = {"sample_id":f"target_{ordinal}","duration_sec":1.,"room":{"type":"moderate"},
                "sources":[{"source_id":"source_0","kind":"sound","description":"a bell","activity":{"onset_sec":.1,"offset_sec":.8},
                            "gain_db":0.,"trajectory":{"type":"static","position":{"azimuth_deg":60.,"elevation_deg":0.,"distance_m":2.}}}]}
        canonical,_ = canonicalize_editing_plan(plan,codec=codec)
        tokens = codec.encode(canonical,max_tokens=1024)["input_ids"].tolist()
        _,bucket,operation = layout[ordinal]
        db.execute("INSERT INTO pairs VALUES(?,'validation',?,?,?,?,44100,?,?)",
                   (ordinal,f"pair_{ordinal}",operation,bucket,plan["sample_id"],zlib.compress(json.dumps(plan).encode()),sha256_json(plan)))
        free_records.append({"pair_ordinal":ordinal,"pair_id":f"pair_{ordinal}","operation":operation,
                             "latent_bucket_frames":bucket,"target_sample_id":plan["sample_id"],"duration_sec":1.,
                             "target_token_ids":tokens,"predicted_token_ids":tokens,"error":None})
    db.commit()
    db.close()
    for rank in range(5): save_work(tmp_path,"free-25000",rank,free_records[rank::5])
    result = derive_selection(tmp_path,value,contract)
    assert result["selected_step"] == 25000
    assert result["status"] == ("FAIL" if bad_holdout else SELECTION_STATUS)
    assert result["quality_gate_passed"] is False
    assert result["free_plan_gate"]["pass"]
    assert result["source_gate"]["pass"]
    assert result["base_dit_gate"]["pass"] is (not bad_holdout)
