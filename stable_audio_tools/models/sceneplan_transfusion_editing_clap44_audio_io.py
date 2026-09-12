"""Native CLAP44 decoded-audio evidence, calibration and independent-test seal."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import zlib

import soundfile as sf

from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_audio_end_to_end as audio
from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_gt_audio as gt
from scripts.t2a.eval import evaluate_sceneplan_transfusion_editing_clap44_gt_audio as native_gt
from .sceneplan_transfusion_editing_clap44_joint_io import codec_artifact_sha256
from .sceneplan_transfusion_editing_clap44_pipeline import CLAP44_PIPELINE_CONTRACT
from .sceneplan_transfusion_editing_provenance import verify_frozen_qwen_runtime

REPO = Path(__file__).resolve().parents[2]
SCHEMA = "editing_clap44_audio_end_to_end_v1"
FREEZE_SCHEMA = "editing_clap44_pretest_model_evaluation_freeze_v1"
RELEASE_SCHEMA = "editing_clap44_validated_release_v1"


def policy():
    return json.loads(json.dumps({
        "schema":SCHEMA, "pipeline_contract":CLAP44_PIPELINE_CONTRACT,
        "higher":audio.HIGHER_BETTER_SPECS, "lower":audio.LOWER_BETTER_SPECS,
        "confidence":audio.CONFIDENCE, "demix":audio._demix_contract(),
        "validation_pairs":20000, "calibration_pairs":1000,
        "calibration_per_operation_bucket":100, "independent_test_pairs":5000,
        "allowed_batch_sizes_per_rank":[1,2,4], "calibration_test_batch_size_must_match":True,
        "ode_steps":20, "cfg_scale":1., "max_plan_tokens":512, "seed":42,
        "sample_rate":44100, "channels":4, "save_all_audio":True,
        "listening_rows_per_operation_bucket":5,
        "model_input_contract":audio.MODEL_INPUT_CONTRACT,
        "m2d_used":False, "gt_plans_count_as_free_ar":False,
    }))


def code_hashes():
    paths = set(native_gt.code_hashes()) | {
        str(Path(__file__).resolve().relative_to(REPO)),
        "scripts/t2a/eval/evaluate_sceneplan_transfusion_editing_clap44_audio.py",
        "scripts/t2a/eval/run_sceneplan_transfusion_editing_clap44_audio_5gpu.sh",
        "scripts/t2a/eval/publish_sceneplan_transfusion_editing_clap44_release.py",
        "scripts/t2a/inference/edit_foa_with_clap44.py",
    }
    return {name:gt.sha256_file(REPO/name) for name in sorted(paths)}


def check_audio(path, digest, *, samples):
    path = Path(path).resolve(strict=True)
    if gt.sha256_file(path) != digest:
        raise RuntimeError("native edited audio checksum changed")
    info = sf.info(str(path))
    if info.samplerate != 44100 or info.channels != 4 or info.frames != samples:
        raise RuntimeError("native edited audio is not exact-length 44.1kHz four-channel FOA")


def audio_assets(run):
    from stable_audio_tools.configuration import load_config
    cfg = load_config(run["model_config"])
    prompt = next(row for row in cfg["model"]["conditioning"]["configs"] if row["id"]=="prompt")
    qwen = verify_frozen_qwen_runtime(prompt["config"]["model_path"])
    if qwen != run["frozen_qwen_runtime"]:
        raise RuntimeError("native audio Qwen assets changed")
    clap = run["clap_checkpoint"]
    if gt.sha256_file(clap["path"]) != clap["sha256"]:
        raise RuntimeError("native audio CLAP checkpoint changed")
    if codec_artifact_sha256(run["codec"]) != run["codec_sha256"]:
        raise RuntimeError("native audio ScenePlan codec changed")
    return {"vae_config":gt._artifact(audio.FROZEN_VAE_CONFIG),
            "vae_checkpoint":gt._artifact(audio.FROZEN_VAE_CHECKPOINT),
            "qwen":qwen, "clap_checkpoint":clap,
            "independent_content":audio.verify_independent_content_metric_assets()}


def upstream_identity(selection_path, selection_sha256, gt_gate_path):
    value = native_gt.validate_native_gt_audio(gt_gate_path,selection_path=selection_path,
                                               selection_sha256=selection_sha256)
    gt_contract = json.loads(gt._verify_artifact(value["contract"]).read_text())
    identity = gt_contract["identity"]
    run = json.loads(gt._verify_artifact(identity["joint_run_contract"]).read_text())
    return {"joint_selection":identity["joint_selection"],
            "joint_checkpoint":identity["joint_checkpoint"],
            "joint_run_contract":identity["joint_run_contract"],
            "post_joint_gt_audio_gate":gt._artifact(Path(gt_gate_path)),
            "model_config":identity["model_config"], "codec":identity["codec"],
            "codec_sha256":identity["codec_sha256"], "variant":identity["variant"]}, run


def phase_index(preflight, *, phase, validate_calibration):
    """Never resolve/hash/open the test path before validated calibration."""
    if phase == "test":
        calibration = validate_calibration()
        if calibration.get("status") != "PASS" or calibration.get("phase") != "calibration":
            raise RuntimeError("independent test requires passed native audio calibration")
        split = "test"
    elif phase == "calibration":
        calibration = None
        split = "validation"
    else:
        raise ValueError("unknown native audio phase")
    record = preflight["indices"][split]
    return Path(record["path"]).resolve(strict=True), record, calibration


def rank_ordinals(layout, selected, rank):
    if rank not in range(5):
        raise ValueError("native audio rank must belong to its five-worker evaluation")
    chosen = set(selected)
    return [ordinal for bucket in (432,648)
            for i,ordinal in enumerate(row["pair_ordinal"] for row in layout
                if row["latent_bucket_frames"]==bucket and row["pair_ordinal"] in chosen)
            if i%5==rank]


def rank_batches(layout, selected, rank, batch_size):
    if batch_size not in (1,2,4):
        raise ValueError("native audio batch size must be fixed before calibration")
    by_ordinal = {row["pair_ordinal"]:row for row in layout}
    ordinals = rank_ordinals(layout,selected,rank)
    return [group for bucket in (432,648)
            for group in audio._chunks([i for i in ordinals if by_ordinal[i]["latent_bucket_frames"]==bucket],batch_size)]


def check_record(record, row, contract):
    audio._validate_batch_records([record],[row],bucket=row["latent_bucket_frames"])
    if record.get("contract_sha256") != gt._digest(contract) or record.get("runtime_route") != CLAP44_PIPELINE_CONTRACT:
        raise RuntimeError("native audio row has a different runtime/contract")
    if record["status"] == "ok":
        samples = record.get("model_num_samples")
        if (type(samples) is not int or not 0 < samples <= row["latent_bucket_frames"]*1024 or
                record.get("latent_frames_valid") != (samples+1023)//1024 or
                ("model_num_samples" in row and samples != row["model_num_samples"])):
            raise RuntimeError("native audio source length/padding changed")
        for field in ("source_domain","target_domain"):
            if field in row and record.get(field) != row[field]:
                raise RuntimeError("native audio difficulty labels changed")
        if record.get("plan_origin") != "free_ar" or record.get("model_input_contract") != audio.MODEL_INPUT_CONTRACT:
            raise RuntimeError("GT plans or offline truth cannot count as free AR inference")
        if not record.get("generated_plan_tokens") or not record.get("generated_sceneplan"):
            raise RuntimeError("native audio row omitted its freely generated complete plan")
        check_audio(record["edited_foa_path"],record["edited_foa_sha256"],samples=record["model_num_samples"])


def collect_records(directory, contract, layout):
    selected = contract["selected_ordinals"]
    layout_by_ordinal = {row["pair_ordinal"]:row for row in layout}
    records, artifacts = [], []
    for rank in range(5):
        expected = rank_ordinals(layout,selected,rank)
        batches = rank_batches(layout,selected,rank,contract["batch_size_per_rank"])
        folder = Path(directory)/"shards"/f"rank-{rank}"
        paths = sorted(folder.glob("batch-*.json"))
        if len(paths) != len(batches):
            raise RuntimeError("native audio rank does not exactly cover its frozen batches")
        observed = []
        for number,path in enumerate(paths):
            shard = json.loads(path.read_text())
            ordinals = shard["pair_ordinals"]
            if (shard.get("contract_sha256") != gt._digest(contract) or shard.get("rank") != rank or
                    shard.get("batch") != number or len(ordinals) != len(shard["records"]) or
                    ordinals != list(batches[number]) or not ordinals):
                raise RuntimeError("native audio shard identity changed")
            for record,ordinal in zip(shard["records"],ordinals):
                check_record(record,layout_by_ordinal[ordinal],contract)
                if record["pair_ordinal"] != ordinal:
                    raise RuntimeError("native audio shard record ordering changed")
            observed.extend(ordinals)
            records.extend(shard["records"])
            artifacts.append(gt._artifact(path))
        if observed != expected:
            raise RuntimeError("native audio rank does not exactly cover its frozen rows")
    records.sort(key=lambda row:row["pair_ordinal"])
    if [row["pair_ordinal"] for row in records] != selected:
        raise RuntimeError("native audio evidence omitted or duplicated selected rows")
    return records, artifacts


def rebind_plan_metrics(records, index, codec):
    """Recompute plan scores from frozen labels and saved generated tokens."""
    from stable_audio_tools.data.sceneplan_transfusion_editing_plan import canonicalize_editing_plan
    from stable_audio_tools.models.sceneplan_transfusion_editing_pipeline import _align_decoded_sceneplan_to_audio_duration
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1",uri=True)
    try:
        for record in records:
            if record["status"] != "ok":
                continue
            row = connection.execute(
                "SELECT pair_id,model_num_samples,new_sceneplan_zlib,new_sceneplan_sha256 FROM pairs WHERE pair_ordinal=?",
                (record["pair_ordinal"],)).fetchone()
            if row is None:
                raise RuntimeError("native audio row is absent from frozen truth")
            pair, samples, blob, digest = row
            truth = json.loads(zlib.decompress(blob))
            if audio.sha256_json(truth) != digest or pair != record["pair_id"] or samples != record["model_num_samples"]:
                raise RuntimeError("native audio frozen plan/sample identity changed")
            target,_ = canonicalize_editing_plan(truth,codec=codec)
            target_tokens = codec.encode(target,max_tokens=1024)["input_ids"].tolist()
            generated_tokens = record["generated_plan_tokens"]
            generated = _align_decoded_sceneplan_to_audio_duration(
                codec.decode(generated_tokens,sample_id=record["generated_sceneplan"]["sample_id"]),samples/44100)
            if generated != record["generated_sceneplan"]:
                raise RuntimeError("saved generated plan differs from its generated tokens")
            scores = audio._plan_metrics(codec=codec,target_plan=target,predicted_plan=generated,
                        target_tokens=target_tokens,predicted_tokens=generated_tokens)
            if {key:record["metrics"].get(key) for key in scores} != scores:
                raise RuntimeError("native free-plan metrics cannot be replayed against frozen truth")
    finally:
        connection.close()


def derive_result(directory, contract, layout, *, calibration=None):
    records, artifacts = collect_records(directory,contract,layout)
    successful = [row for row in records if row["status"]=="ok"]
    summaries = audio._all_metric_summaries(successful)
    for metric in set(audio.HIGHER_BETTER_SPECS)|set(audio.LOWER_BETTER_SPECS):
        summaries.setdefault(metric,audio._metric_summary(successful,metric))
    listening = set(contract["listening_ordinals"])
    structural = audio._structural_report(records,expected_rows=len(contract["selected_ordinals"]),
        listening_ordinals=listening,save_all_audio=True,
        shared_transformer_same_object=True,old_sceneplan_model_input=False)
    if contract["phase"] == "calibration":
        thresholds, quality = audio._calibration_thresholds(summaries)
    else:
        if calibration is None or calibration.get("phase") != "calibration" or calibration.get("status") != "PASS":
            raise RuntimeError("test results cannot choose or relax their own thresholds")
        thresholds = calibration["thresholds"]
        quality = audio._test_threshold_checks(summaries,thresholds)
    checks = audio._evaluation_checks(structural,quality,expected_rows=len(records))
    difficulty = {}
    for field in ("source_count","target_count","source_domain","target_domain"):
        difficulty[field] = {}
        for label in sorted({str(row[field]) for row in successful}):
            rows = [row for row in successful if str(row[field])==label]
            difficulty[field][label] = {
                "rows":len(rows),"fraction":len(rows)/max(len(records),1),
                "metrics":{metric:audio._metric_summary(rows,metric)["overall"] for metric in
                    ("plan_permutation_scene_score","audio_codec_foa_progress","doa_target_mean_deg","unchanged_preservation_budget_ratio")}}
    return {
        "schema":SCHEMA,"phase":contract["phase"],"status":"PASS" if checks and all(checks.values()) else "FAIL",
        "contract":gt._artifact(Path(directory)/"CONTRACT.json"),
        "rows":len(records),"successful_rows":len(successful),"checks":checks,"structural":structural,
        "quality_checks":quality,"thresholds":thresholds,"metric_summaries":summaries,"difficulty":difficulty,
        "record_artifacts":artifacts,"selected_checkpoint":contract["identity"]["joint_checkpoint"],
        "joint_selection":contract["identity"]["joint_selection"],
        "post_joint_gt_audio_gate":contract["identity"]["post_joint_gt_audio_gate"],
        "independent_test_used":contract["phase"]=="test",
        "final_quality_passed":contract["phase"]=="test" and bool(checks) and all(checks.values()),
    }


def validate_result(path, *, expected_sha256, selection_path, selection_sha256, phase):
    from stable_audio_tools.data.model_sceneplan_codec_v4 import ModelScenePlanCodecV4
    path = Path(path).resolve(strict=True)
    if gt.sha256_file(path) != expected_sha256:
        raise RuntimeError("native audio result SHA256 changed")
    saved = json.loads(path.read_text())
    contract_path = gt._verify_artifact(saved["contract"])
    if contract_path != path.parent/"CONTRACT.json":
        raise RuntimeError("native audio result contract escaped its run")
    contract = json.loads(contract_path.read_text())
    if (saved.get("schema") != SCHEMA or saved.get("phase") != phase or contract.get("schema") != SCHEMA or
            contract.get("phase") != phase or contract.get("policy") != policy() or
            contract.get("source_sha256") != code_hashes() or contract.get("physical_gpus") != [3,4,5,6,7] or
            contract.get("world_size") != 5 or contract.get("batch_size_per_rank") not in (1,2,4)):
        raise RuntimeError("native audio policy/code/execution contract changed")
    identity, run = upstream_identity(selection_path,selection_sha256,contract["identity"]["post_joint_gt_audio_gate"]["path"])
    if identity != contract["identity"] or audio_assets(run) != contract["frozen_assets"]:
        raise RuntimeError("native audio model or scoring assets changed")
    # The independent index is untouched until calibration has been replayed.
    calibration = None
    if phase == "test":
        record = contract["calibration"]
        calibration = validate_result(record["path"],expected_sha256=record["sha256"],
            selection_path=selection_path,selection_sha256=selection_sha256,phase="calibration")
        cal_contract = json.loads(gt._verify_artifact(calibration["contract"]).read_text())
        if cal_contract["batch_size_per_rank"] != contract["batch_size_per_rank"]:
            raise RuntimeError("native calibration/test batch sizes differ")
        freeze = json.loads(gt._verify_artifact(contract["pretest_freeze"]).read_text())
        if freeze != freeze_value(contract["identity"],contract["calibration"],contract["batch_size_per_rank"],
                                  contract["preflight"],contract["advertised_test_index"]):
            raise RuntimeError("native model/evaluation freeze changed after calibration")
    elif phase != "calibration" or any(contract.get(key) is not None for key in
                                     ("calibration","pretest_freeze","advertised_test_index")):
        raise RuntimeError("invalid native audio phase")
    preflight = json.loads(gt._verify_artifact(contract["preflight"]).read_text())
    if preflight.get("status") != "PASS" or any(
            preflight["indices"][split]["rows"] != rows
            for split,rows in (("train",1000000),("validation",20000),("test",5000))) or any(
            preflight["indices"][split]["sha256"] != run["indices"][split]["sha256"]
            for split in ("train","validation")):
        raise RuntimeError("native audio preflight no longer binds the trained population")
    if phase == "test" and (contract["advertised_test_index"] != preflight["indices"]["test"] or
            gt._verify_artifact(contract["pretest_freeze"]) != path.parent/"PRETEST_FREEZE.json"):
        raise RuntimeError("native test index differs from its pre-access freeze")
    split, expected_rows = ("test",5000) if phase=="test" else ("validation",20000)
    record = preflight["indices"][split]
    index = Path(contract["index"]["path"]).resolve(strict=True)
    if (str(index) != str(Path(record["path"]).resolve(strict=True)) or
            gt.sha256_file(index) != record["sha256"] or contract["index"]["sha256"] != record["sha256"] or
            contract["index"]["rows"] != expected_rows or contract["index"]["split"] != split):
        raise RuntimeError("native audio phase/index identity changed")
    layout = audio._layout(index,expected_rows=expected_rows,expected_split=split)
    selected, population = audio._select_ordinals(layout,phase=phase)
    listening = sorted(audio._listening_ordinals([row for row in layout if row["pair_ordinal"] in set(selected)],5))
    if contract["selected_ordinals"] != selected or contract["row_selection"] != population or contract["listening_ordinals"] != listening:
        raise RuntimeError("native audio selected/listening populations changed")
    records,_ = collect_records(path.parent,contract,layout)
    rebind_plan_metrics(records,index,ModelScenePlanCodecV4(run["codec"]))
    derived = derive_result(path.parent,contract,layout,calibration=calibration)
    if saved != derived or derived["status"] != "PASS":
        raise RuntimeError("native audio quality result cannot be replayed as passing")
    return derived


def freeze_value(identity, calibration, batch_size, preflight, advertised_test_index):
    return {"schema":FREEZE_SCHEMA,"identity":identity,"calibration":calibration,"policy":policy(),
            "source_sha256":code_hashes(),"batch_size_per_rank":batch_size,
            "preflight":preflight,"advertised_test_index":advertised_test_index,
            "test_access_authorized_only_after_validated_calibration":True}


def validate_release(path, *, expected_sha256):
    from .sceneplan_transfusion_editing_clap44_release import validate_release as replay
    return replay(path,expected_sha256=expected_sha256)
