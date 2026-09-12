#!/usr/bin/env bash
set -euo pipefail

REPO="${P10_REPO:-${AMBIT_CKPT_ROOT}/stable-audio-tools-workspace}"
ROOT="${EDITING_DATA_ROOT:-${AMBIT_DATA_ROOT}/sceneplan_transfusion_editing_v1}"
PREFLIGHT="$ROOT/contracts/full_training/PREFLIGHT.json"
VALIDATION_INDEX="$ROOT/training_index/validation.sqlite"
TEST_INDEX="$ROOT/training_index/test.sqlite"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_sceneplan_transfusion_editing_dit_full_v1.json}"
CODEC="${CODEC:-${AMBIT_DATA_ROOT}/sceneplan_v2_1p124m/p11_single_turn_15s_v2/model_sceneplan_codec_v4}"
JOINT_RUN="${JOINT_RUN:-${AMBIT_CKPT_ROOT}/transfusion_editing/mainline/sceneplan_transfusion_editing_ar_joint_m2d_full_seed42_v3}"
JOINT_SELECTION="${JOINT_CHECKPOINT_SELECTION:-$JOINT_RUN/evaluation/validation_20k_joint_checkpoint_selection/SELECTED.json}"
EVAL_ROOT="${OUTPUT_DIR:-$JOINT_RUN/evaluation/audio_end_to_end_phase_aware_v6}"
CALIBRATION_DIR="$EVAL_ROOT/validation_calibration_1k"
TEST_DIR="$EVAL_ROOT/test_5k"
FORMAL_BATCH_SIZE="${AUDIO_E2E_BATCH_SIZE:-2}"

if [[ "${M2D_NONCOMMERCIAL_EVALUATION_ACK:-}" != "1" ]]; then
    echo "[editing-audio-e2e] M2D is evaluation-only; authorization acknowledgement is required" >&2
    exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "[ambit] set CUDA_VISIBLE_DEVICES to the GPUs for this job" >&2
    exit 2
fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
case "$FORMAL_BATCH_SIZE" in
    1|2|4) ;;
    *)
        echo "[editing-audio-e2e] formal per-rank batch size must be one of 1, 2, or 4" >&2
        exit 2
        ;;
esac
for required in "$PREFLIGHT" "$VALIDATION_INDEX" "$MODEL_CONFIG" "$CODEC/codec.json" "$JOINT_SELECTION"; do
    if [[ ! -r "$required" ]]; then
        echo "[editing-audio-e2e] required artifact is missing: $required" >&2
        exit 1
    fi
done

mapfile -t pinned < <("$REPO/.venv/bin/python" - "$REPO" "$PREFLIGHT" "$VALIDATION_INDEX" "$JOINT_SELECTION" <<'PY'
import json,sys,torch
from pathlib import Path

repo=Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0,str(repo))

from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
from stable_audio_tools.models.sceneplan_transfusion_editing_m2d_clap import EDITING_M2D_CLAP_SIDE_INPUT
from scripts.t2a.eval.select_sceneplan_transfusion_editing_joint_checkpoint import (
    validate_published_joint_selection,
)

torch.set_float32_matmul_precision("high")
preflight=Path(sys.argv[2]).resolve(strict=True)
validation=Path(sys.argv[3]).resolve(strict=True)
selection_path=Path(sys.argv[4]).resolve(strict=True)
value=json.loads(preflight.read_text(encoding="utf-8"))
selection=json.loads(selection_path.read_text(encoding="utf-8"))
latest=value.get("latest_route",{})
indices=value.get("indices",{})
validation_record=indices.get("validation",{})
if not (
    value.get("schema")=="sceneplan_transfusion_editing_full_training_preflight"
    and int(value.get("schema_version",-1))==2
    and value.get("status")=="PASS"
    and latest.get("editing_ar_inputs")==["source_foa_latent","raw_edit_request"]
    and latest.get("editing_ar_target")=="complete_new_sceneplan"
    and latest.get("old_sceneplan_input") is False
    and latest.get("editing_dit_frame_channels")==384
    and latest.get("editing_dit_clean_source_always_present") is True
    and latest.get("editing_evaluation_max_plan_tokens")==512
    and value.get("split_disjointness",{}).get("status")=="PASS"
    and value.get("split_disjointness",{}).get("checks")=={
        "train_vs_test":True,
        "train_vs_validation":True,
        "validation_vs_test":True,
    }
    and Path(validation_record.get("path","")).resolve()==validation
    and validation_record.get("sha256")==sha256_file(validation)
    and int(validation_record.get("rows",-1))==20000
    and all(
        validation_record.get(f"{role}_latent_shards_exhaustively_verified")
        is True
        and validation_record.get(f"{role}_pair_rows")==20000
        and validation_record.get(f"{role}_latent_shards",0)>0
        and len(validation_record.get(
            f"{role}_latent_shard_inventory_sha256", ""
        ))==64
        for role in ("source","target")
    )
    and selection.get("schema")=="sceneplan_transfusion_editing_joint_checkpoint_selection"
    and selection.get("status")=="PASS"
    and selection.get("selection_contract")=="full_20k_10k_select_10k_holdout_joint_ar_rf_source_base_dit_noninferiority_m2d_stratified_free_ar_5k_v5"
    and selection.get("physical_gpus")==[3,4,5,6,7]
    and int(selection.get("world_size",-1))==5
    and selection.get("latest_route",{}).get("editing_ar_inputs")==["source_foa_latent","raw_edit_request"]
    and selection.get("latest_route",{}).get("old_sceneplan_input") is False
    and selection.get("latest_route",{}).get("source_caption_model_input") is False
    and selection.get("latest_route",{}).get("source_derived_semantic_side_input")==EDITING_M2D_CLAP_SIDE_INPUT
    and selection.get("latest_route",{}).get("editing_dit_frame_channels")==384
    and selection.get("selected_base_dit_noninferiority_gate",{}).get("pass") is True
    and selection.get("selected_source_intervention_gate",{}).get("pass") is True
    and selection.get("selected_free_ar_gate",{}).get("pass") is True
    and selection.get("source_semantic",{}).get("mode")=="m2d_audio_caption_aux"
    and selection.get("source_semantic",{}).get("caption_model_input") is False
):
    raise SystemExit("preflight or joint checkpoint promotion is stale")
checkpoint=Path(selection.get("selected_checkpoint","")).resolve(strict=True)
if selection.get("selected_checkpoint_sha256")!=sha256_file(checkpoint):
    raise SystemExit("selected joint checkpoint SHA256 changed")
selection_sha=sha256_file(selection_path)
if validate_published_joint_selection(
    selection_path, expected_sha256=selection_sha
) != selection:
    raise SystemExit("joint checkpoint promotion full replay changed")
print(validation_record["sha256"])
print(selection_sha)
joint_contract=json.loads(Path(selection["training_run"]["run_contract_path"]).read_text())
print(joint_contract["base_checkpoint_selection"]["path"])
print(joint_contract["dit_gt_audio_gate"]["path"])
PY
)
if [[ "${#pinned[@]}" -ne 4 ]]; then
    echo "[editing-audio-e2e] could not resolve frozen identities" >&2
    exit 1
fi
VALIDATION_SHA256="${pinned[0]}"
JOINT_SELECTION_SHA256="${pinned[1]}"

# Isolate DiT regression after shared AR/RF training using GT plans, the same
# fixed validation rows/noise/reference interventions and the pre-joint bounds.
# The following AR end-to-end evaluation then measures planning + execution.
env DIT_CHECKPOINT_SELECTION="${pinned[2]}" \
    GT_AUDIO_PRE_JOINT_GATE="${pinned[3]}" \
    GT_AUDIO_JOINT_SELECTION="$JOINT_SELECTION" \
    GT_AUDIO_OUTPUT="$JOINT_RUN/evaluation/validation_1k_gt_audio_post_joint" \
    "$REPO/scripts/t2a/eval/run_sceneplan_transfusion_editing_gt_audio_5gpu.sh"

mkdir -p "$CALIBRATION_DIR/logs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

set +e
"$REPO/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node=5 \
    "$REPO/scripts/t2a/eval/evaluate_sceneplan_transfusion_editing_audio_end_to_end.py" \
    --phase calibration \
    --checkpoint-selection "$JOINT_SELECTION" \
    --checkpoint-selection-sha256 "$JOINT_SELECTION_SHA256" \
    --model-config "$MODEL_CONFIG" \
    --codec "$CODEC" \
    --index "$VALIDATION_INDEX" \
    --index-sha256 "$VALIDATION_SHA256" \
    --expected-index-rows 20000 \
    --output-dir "$CALIBRATION_DIR" \
    --batch-size "$FORMAL_BATCH_SIZE" \
    --ode-steps 20 \
    --cfg-scale 1.0 \
    --max-plan-tokens 512 \
    --seed 42 \
    --listening-rows-per-cell 2 \
    2>&1 | tee -a "$CALIBRATION_DIR/logs/evaluate.log"
calibration_status="${PIPESTATUS[0]}"
set -e
if [[ "$calibration_status" != "0" ]]; then
    exit "$calibration_status"
fi

CALIBRATION="$CALIBRATION_DIR/CALIBRATION.json"
CALIBRATION_FINAL="$CALIBRATION_DIR/FINAL.json"
if [[ ! -r "$TEST_INDEX" ]]; then
    echo "[editing-audio-e2e] test index is missing after calibration: $TEST_INDEX" >&2
    exit 1
fi
mapfile -t test_pinned < <("$REPO/.venv/bin/python" - \
    "$REPO" "$CALIBRATION" "$CALIBRATION_FINAL" "$PREFLIGHT" "$TEST_INDEX" "$JOINT_SELECTION" \
    "$MODEL_CONFIG" "$CODEC" <<'PY'
import json,sys,torch
from pathlib import Path

repo=Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0,str(repo))

from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
from scripts.t2a.eval.evaluate_sceneplan_transfusion_editing_audio_end_to_end import (
    _source_sha256,
    _validate_calibration,
    validate_audio_e2e_final,
)
from scripts.t2a.eval.sceneplan_transfusion_editing_content_metrics import (
    verify_independent_content_metric_assets,
)

torch.set_float32_matmul_precision("high")
calibration=Path(sys.argv[2]).resolve(strict=True)
calibration_final=Path(sys.argv[3]).resolve(strict=True)
selection_path=Path(sys.argv[6]).resolve(strict=True)
model_config=Path(sys.argv[7]).resolve(strict=True)
codec=Path(sys.argv[8]).resolve(strict=True)
selection=json.loads(selection_path.read_text(encoding="utf-8"))
calibration_sha=sha256_file(calibration)
validate_audio_e2e_final(
    calibration_final,
    expected_sha256=sha256_file(calibration_final),
    write_seal=False,
)
_validate_calibration(
    calibration,
    expected_sha=calibration_sha,
    selection=selection,
    selection_path=selection_path,
    selection_sha=sha256_file(selection_path),
    model_config=model_config,
    codec=codec,
    source_hashes=_source_sha256(),
    independent_content_assets=verify_independent_content_metric_assets(),
)

# The test SQLite is intentionally first resolved and hashed only after the
# complete calibration replay above succeeds.
preflight=Path(sys.argv[4]).resolve(strict=True)
test=Path(sys.argv[5]).resolve(strict=True)
value=json.loads(preflight.read_text(encoding="utf-8"))
test_record=(value.get("indices") or {}).get("test") or {}
test_sha=sha256_file(test)
if not (
    value.get("schema")=="sceneplan_transfusion_editing_full_training_preflight"
    and int(value.get("schema_version",-1))==2
    and value.get("status")=="PASS"
    and value.get("split_disjointness",{}).get("status")=="PASS"
    and all(value.get("split_disjointness",{}).get("checks",{}).values())
    and Path(test_record.get("path","")).resolve()==test
    and test_record.get("sha256")==test_sha
    and int(test_record.get("rows",-1))==5000
    and all(
        test_record.get(f"{role}_latent_shards_exhaustively_verified") is True
        and test_record.get(f"{role}_pair_rows")==5000
        and test_record.get(f"{role}_latent_shards",0)>0
        and len(test_record.get(
            f"{role}_latent_shard_inventory_sha256", ""
        ))==64
        for role in ("source","target")
    )
):
    raise SystemExit("test index or preflight binding changed")
print(calibration_sha)
print(test_sha)
PY
)
if [[ "${#test_pinned[@]}" -ne 2 ]]; then
    echo "[editing-audio-e2e] could not validate calibration and test seal" >&2
    exit 1
fi
CALIBRATION_SHA256="${test_pinned[0]}"
TEST_SHA256="${test_pinned[1]}"
mkdir -p "$TEST_DIR/logs"

save_audio_args=()
if [[ "${SAVE_ALL_AUDIO:-1}" == "1" ]]; then
    save_audio_args=(--save-all-audio)
fi
set +e
"$REPO/.venv/bin/torchrun" \
    --standalone \
    --nproc_per_node=5 \
    "$REPO/scripts/t2a/eval/evaluate_sceneplan_transfusion_editing_audio_end_to_end.py" \
    --phase test \
    --checkpoint-selection "$JOINT_SELECTION" \
    --checkpoint-selection-sha256 "$JOINT_SELECTION_SHA256" \
    --model-config "$MODEL_CONFIG" \
    --codec "$CODEC" \
    --index "$TEST_INDEX" \
    --index-sha256 "$TEST_SHA256" \
    --expected-index-rows 5000 \
    --output-dir "$TEST_DIR" \
    --calibration "$CALIBRATION" \
    --calibration-sha256 "$CALIBRATION_SHA256" \
    --batch-size "$FORMAL_BATCH_SIZE" \
    --ode-steps 20 \
    --cfg-scale 1.0 \
    --max-plan-tokens 512 \
    --seed 42 \
    --listening-rows-per-cell 10 \
    "${save_audio_args[@]}" \
    2>&1 | tee -a "$TEST_DIR/logs/evaluate.log"
test_status="${PIPESTATUS[0]}"
set -e
if [[ "$test_status" != "0" ]]; then
    exit "$test_status"
fi

"$REPO/.venv/bin/python" - "$REPO" "$TEST_DIR/FINAL.json" <<'PY'
import json,sys
from pathlib import Path

repo=Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0,str(repo))

from stable_audio_tools.data.sceneplan_transfusion_editing_index import sha256_file
from scripts.t2a.eval.evaluate_sceneplan_transfusion_editing_audio_end_to_end import (
    validate_audio_e2e_final,
)

final_path=Path(sys.argv[2]).resolve(strict=True)
seal=validate_audio_e2e_final(
    final_path,
    expected_sha256=sha256_file(final_path),
    write_seal=False,
)
print(json.dumps({
    "event":"editing_audio_e2e_final_replay_passed",
    "final_sha256":sha256_file(final_path),
    "seal_sha256":sha256_file(final_path.parent / "SEALED.json"),
    "phase":seal["phase"],
},sort_keys=True))
PY
