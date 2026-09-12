#!/usr/bin/env bash
set -euo pipefail

REPO="${P10_REPO:-/home/tanhe/dataset_storage/stable-audio-tools}"
REVISION_ROOT="${REVISION_ROOT:-/mnt/sdb/audio_dataset/sceneplan_v2_1p124m/revisions/speech_expansion_noalign_15s_v1}"
FREEZE="$REVISION_ROOT/contracts/freeze_manifest.json"
PREFLIGHT="$REVISION_ROOT/P10_PREFLIGHT_GATE.json"
SOURCE_GATE="/mnt/sdb/model_archives/p10_pre_v11_20260831/sceneplan_dit_v10_semantic_v2_protected_resume_110k/evaluation/P10_110K_POSTTRAIN_GATE.json"
SOURCE_CKPT="${SOURCE_CKPT:-/mnt/sdb/model_archives/p10_pre_v11_20260831/sceneplan_dit_v10_semantic_v2_protected_resume_110k/checkpoints/epoch=35-step=110000.ckpt}"
SOURCE_CKPT_SHA256="${SOURCE_CKPT_SHA256:-8becc5533a204f1be2cc5813d25e7572cea9327dc179b5be3659dec9ac270f43}"

BASE_MODEL_CONFIG="$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s.json"
MODEL_CONFIG="${MODEL_CONFIG:-$REPO/stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/qwen35_0p8b_300m_model_sceneplan_44_soundexp_noalign_15s_resume_cosine_40k.json}"
DATASET_CONFIG="${DATASET_CONFIG:-$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_v2_speech_expansion_noalign_15s_v1_train_semantic_v2.json}"
VAL_DATASET_CONFIG="${VAL_DATASET_CONFIG:-$REPO/stable_audio_tools/configs/dataset_configs/sceneplan_v2_speech_expansion_noalign_15s_v1_validation_semantic_v2.json}"

"$REPO/.venv/bin/python" - \
    "$FREEZE" "$PREFLIGHT" "$SOURCE_GATE" "$SOURCE_CKPT" \
    "$SOURCE_CKPT_SHA256" "$BASE_MODEL_CONFIG" "$MODEL_CONFIG" \
    "$DATASET_CONFIG" "$VAL_DATASET_CONFIG" <<'PY'
import copy
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from stable_audio_tools.configuration import load_config, validate_training_configs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


(
    freeze_path,
    preflight_path,
    source_gate_path,
    source_ckpt_path,
    expected_source_sha,
    base_model_path,
    model_path,
    train_path,
    validation_path,
) = sys.argv[1:]
(
    freeze_path,
    preflight_path,
    source_gate_path,
    source_ckpt_path,
    base_model_path,
    model_path,
    train_path,
    validation_path,
) = (
    Path(value).resolve(strict=True)
    for value in (
        freeze_path,
        preflight_path,
        source_gate_path,
        source_ckpt_path,
        base_model_path,
        model_path,
        train_path,
        validation_path,
    )
)

if source_ckpt_path.name != "epoch=35-step=110000.ckpt":
    raise SystemExit(f"unexpected 110k source checkpoint: {source_ckpt_path}")
observed_source_sha = sha256(source_ckpt_path)
if observed_source_sha != expected_source_sha:
    raise SystemExit(
        "protected 110k checkpoint SHA256 changed: "
        f"{observed_source_sha} != {expected_source_sha}"
    )

freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
if not (
    freeze.get("state") == "P9_complete_frozen_ready_for_P10_preflight"
    and freeze.get("dataset_contract_revision") == 6
    and freeze.get("max_latent_frames") == 648
    and freeze.get("speech_timing_sidecar") is None
    and preflight.get("status") == "PASS"
    and preflight.get("dataset_contract_revision") == 6
    and preflight.get("overfit_10_sample") == "PASS"
    and preflight.get("throughput_8gpu") == "PASS"
    and preflight.get("checkpoint_resume") == "PASS"
):
    raise SystemExit("revision-6 P9/P10 source gate is not all-PASS")

source_gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
if not (
    source_gate.get("schema") == "stable_audio_tools.p10_110k_posttrain_gate"
    and source_gate.get("status") == "PASS"
    and source_gate.get("speech_exit_status") == 0
    and source_gate.get("music_sound_exit_status") == 0
):
    raise SystemExit("110k Speech/Music/Sound post-training gate is not PASS")
speech_summary = json.loads(
    Path(source_gate["speech_summary"]).resolve(strict=True).read_text(encoding="utf-8")
)
music_sound_summary = json.loads(
    Path(source_gate["music_sound_summary"])
    .resolve(strict=True)
    .read_text(encoding="utf-8")
)
if not (
    speech_summary.get("status") == "PASS"
    and speech_summary.get("checkpoints", {})
    .get("candidate", {})
    .get("step")
    == 110_000
    and speech_summary.get("checkpoints", {})
    .get("candidate", {})
    .get("sha256")
    == observed_source_sha
    and music_sound_summary.get("status") == "PASS"
    and music_sound_summary.get("failed_checks") == []
):
    raise SystemExit("110k source evaluation summaries are inconsistent")

base_model = load_config(base_model_path)
model = load_config(model_path)
train = load_config(train_path)
validation = load_config(validation_path)
validate_training_configs(model, train)
validate_training_configs(model, validation)


def without_metadata_and_scheduler(value):
    value = copy.deepcopy(value)
    for key in list(value):
        if key.startswith("_"):
            value.pop(key)
    value["training"]["optimizer_configs"]["diffusion"].pop("scheduler")
    return value


if without_metadata_and_scheduler(model) != without_metadata_and_scheduler(base_model):
    raise SystemExit("150k continuation changed architecture, optimizer, or loss")
scheduler = model["training"]["optimizer_configs"]["diffusion"]["scheduler"]
if scheduler != {
    "type": "CosineAnnealingLR",
    "config": {"T_max": 40_000, "eta_min": 0.00001},
}:
    raise SystemExit(f"unexpected 110k-to-150k scheduler: {scheduler}")
cfg = model["training"].get("sceneplan_cfg_dropout")
if cfg != {
    "mode": "independent",
    "caption_unknown_prob": 0.15,
    "structured_unknown_prob": 0.15,
}:
    raise SystemExit(f"independent 15% CFG contract changed: {cfg}")

for config, split, expected_rows, expected_buckets in (
    (train, "train", 1_600_000, (1_200_000, 400_000)),
    (validation, "validation", 32_000, (24_000, 8_000)),
):
    if not (
        config.get("semantic_caption_mode") == "v2"
        and config.get("semantic_caption_v2_probability") == 1.0
        and config.get("semantic_caption_seed") == 20260830
        and config.get("expected_num_samples") == expected_rows
        and config.get("word_level_timestamp_teacher") is False
    ):
        raise SystemExit(f"canonical semantic-caption-v2 {split} contract changed")
    forbidden = {
        "require_speech_timing",
        "speech_timing_index_path",
        "speech_timing_index_sha256",
        "expected_speech_timing_rows",
    }
    if forbidden & set(config):
        raise SystemExit(f"{split} unexpectedly uses a timing teacher")
    index = Path(config["datasets"][0]["path"]).resolve(strict=True)
    connection = sqlite3.connect(f"file:{index}?mode=ro&immutable=1", uri=True)
    metadata = dict(connection.execute("SELECT key,value FROM metadata"))
    rows, short, long = connection.execute(
        "SELECT COUNT(*),SUM(latent_frames_valid<=432),"
        "SUM(latent_frames_valid>432) FROM samples"
    ).fetchone()
    connection.close()
    if not (
        metadata.get("contract_revision") == "6"
        and metadata.get("speech_timing_sidecar") == "none"
        and int(rows) == expected_rows
        and (int(short), int(long)) == expected_buckets
    ):
        raise SystemExit(f"frozen revision-6 {split} index changed: {index}")

if train.get("semantic_epoch_resume_migration") != "paired_to_fixed":
    raise SystemExit("canonical-v2 sampler resume migration contract changed")
if train.get("length_bucket_batching") != {
    "enabled": True,
    "long_batch_size": 48,
    "seed": 20260828,
}:
    raise SystemExit("protected 432/648 train batching changed")

print(
    "SAT_PROTECTED_150K_CONTINUATION_CONTRACT="
    + json.dumps(
        {
            "source_checkpoint": str(source_ckpt_path),
            "source_checkpoint_sha256": observed_source_sha,
            "source_step": 110_000,
            "source_posttrain_gate": "PASS",
            "target_step": 150_000,
            "checkpoint_steps": [120_000, 130_000, 140_000, 150_000],
            "semantic_caption": "canonical_v2_only",
            "cfg": cfg,
            "cosine": {
                "start_step": 110_000,
                "end_step": 150_000,
                "T_max": 40_000,
                "start_lr": "inherit_110k_without_jump",
                "eta_min": 0.00001,
            },
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

"$REPO/.venv/bin/python" "$REPO/scripts/t2a/train/verify_qwen35_fast_path.py"

export RUN_NAME="${RUN_NAME:-sceneplan_dit_v11_semantic_v2_protected_resume_150k}"
export RUN_LABEL="${RUN_LABEL:-sceneplan-44-semantic-v2-protected-resume-150k}"
export RUN_ROOT="${RUN_ROOT:-/mnt/sdc/ckpts/dit/$RUN_NAME}"
export RUN_CATEGORY="mainline"
export MODEL_CONFIG DATASET_CONFIG VAL_DATASET_CONFIG

export LOAD_PRETRANSFORM=0
export PRETRAINED_CKPT=""
export SAT_PRETRAINED_ROUTE_EXPECTATION_JSON=""
export PRETRAINED_MODALITY_CKPT=""
export REQUIRE_NO_PRETRAINED=1

# First launch restores the immutable, evaluated 110k state. Later launches
# let the common runner select this run's own last/step checkpoint.
if [[ -z "${RESUME_CKPT+x}" ]]; then
    if [[ -r "$RUN_ROOT/checkpoints/last.ckpt" ]] \
        || find "$RUN_ROOT/checkpoints" -maxdepth 1 -type f -name '*step=*.ckpt' \
            -print -quit 2>/dev/null | grep -q .; then
        export RESUME_CKPT=""
    else
        export RESUME_CKPT="$SOURCE_CKPT"
    fi
fi

export SAT_RESUME_COSINE_START_STEP=110000
export SAT_RESUME_COSINE_END_STEP=150000
export SAT_RESUME_COSINE_ETA_MIN=0.00001

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS=8
export BATCH_SIZE="${BATCH_SIZE:-72}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
export ACCUM_BATCHES=1
export TRAINING_STRATEGY=ddp_static
export DDP_BUCKET_CAP_MB="${DDP_BUCKET_CAP_MB:-50}"
export DDP_COMM_HOOK="${DDP_COMM_HOOK:-none}"
export BIND_TO_GPU_NUMA=1
export RANK_AWARE_TRAINING_SEED=1
export GRADIENT_CLIP_VAL=1.0
export ENABLE_TORCH_COMPILE="${ENABLE_TORCH_COMPILE:-0}"

export MAX_STEPS="${MAX_STEPS:-150000}"
export CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
export SAVE_TOP_K="${SAVE_TOP_K:--1}"
export VAL_EVERY="${VAL_EVERY:-4000}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-64}"
export DISABLE_VALIDATION="${DISABLE_VALIDATION:-0}"
export LOGGER="${LOGGER:-wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export TRAINING_GATE=1
export TRAINING_GATE_WINDOW="${TRAINING_GATE_WINDOW:-100}"
export TRAINING_GATE_MAX_LOSS_RATIO="${TRAINING_GATE_MAX_LOSS_RATIO:--1}"
export TRAINING_GATE_GRADIENT_EVERY="${TRAINING_GATE_GRADIENT_EVERY:-50}"
export MIN_FREE_DISK_GIB="${MIN_FREE_DISK_GIB:-100}"
export MIN_ROOT_FREE_GIB="${MIN_ROOT_FREE_GIB:-20}"
export TEMP_DIR="${TEMP_DIR:-/dev/shm/sceneplan_v11_semantic_v2}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/sdc/ckpts/dit/triton-cache/sceneplan_qwen35_fla052_cc170}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[sceneplan-44-semantic-v2-150k] continuation preflight PASS"
    exit 0
fi

exec "$REPO/scripts/t2a/train/run_t2a_common_8gpu.sh"
