#!/usr/bin/env bash
# VAE checkpoint sweep - one script, two eval modes.
#
# MODE=heldout   (default) unwrap -> compare_vae_heldout.py
# MODE=preencode           unwrap -> pre-encode -> decode -> eval_vae_recon_full.py
#
# Usage:
#   bash scripts/vae/eval/run_vae_sweep.sh
#   MODE=preencode GPU=1 bash scripts/vae/eval/run_vae_sweep.sh
#   STEPS="200000 400000" bash scripts/vae/eval/run_vae_sweep.sh
set -euo pipefail

SAT=.
cd "$SAT"

MODE="${MODE:-heldout}"
CKPT_ROOT="${CKPT_ROOT:-${AMBIT_CKPT_ROOT}/vae_ds1024_z64_construct}"
MODEL_CFG="${MODEL_CFG:-stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024_z64.json}"
HELDOUT_CFG="stable_audio_tools/configs/dataset_configs/local_4ch_vae_heldout_eval_100.json"
DIT_CFG="stable_audio_tools/configs/dataset_configs/construct_dataset/preencode_4ch_construct.json"

GPU="${GPU:-1}"
NUM="${NUM:-100}"
WITH_SELD="${WITH_SELD:-1}"
FORCE="${FORCE:-0}"
BUILD_SUBSET="${BUILD_SUBSET:-0}"
PREENCODE_DIT="${PREENCODE_DIT:-0}"

if [ "$MODE" = "preencode" ]; then
  EVAL_ROOT="${EVAL_ROOT:-${AMBIT_CKPT_ROOT}/eval_metric/ckpt_sweep}"
  LATENT_ROOT="${LATENT_ROOT:-${AMBIT_CKPT_ROOT}/audio_latents/vae_eval/ckpt_sweep}"
else
  EVAL_ROOT="${EVAL_ROOT:-${AMBIT_CKPT_ROOT}/eval_metric}"
fi

if [ -n "${STEPS:-}" ]; then
  # shellcheck disable=SC2206
  STEP_LIST=($STEPS)
else
  mapfile -t STEP_LIST < <(ls "$CKPT_ROOT"/epoch=*-step=*.ckpt 2>/dev/null \
    | sed 's/.*step=//;s/.ckpt//' | sort -n | uniq)
fi

mkdir -p "$EVAL_ROOT/logs" "$CKPT_ROOT"
if [ "$MODE" = "preencode" ]; then
  mkdir -p "$LATENT_ROOT"
fi

need_subset=0
if [ "$BUILD_SUBSET" = "1" ]; then need_subset=1; fi
if [ "$MODE" = "preencode" ] && [ ! -f "$HELDOUT_CFG" ]; then need_subset=1; fi
if [ "$need_subset" = "1" ]; then
  uv run python scripts/vae/eval/build_vae_eval_subset.py --mode heldout --num "$NUM"
fi

find_raw_ckpt() {
  local step="$1"
  local matches=("$CKPT_ROOT"/epoch=*-step="${step}".ckpt)
  [ -e "${matches[0]}" ] || return 1
  printf '%s\n' "${matches[0]}"
}

unwrap_step() {
  local step="$1" raw="$2"
  local out="$CKPT_ROOT/unwrapped_ds1024_z64_step=${step}.ckpt"
  [ -f "$out" ] && [ "$FORCE" != "1" ] && return 0
  CUDA_VISIBLE_DEVICES="" uv run python unwrap_model.py \
    --model-config "$MODEL_CFG" --ckpt-path "$raw" \
    --name "$CKPT_ROOT/unwrapped_ds1024_z64_step=${step}"
}

eval_heldout() {
  local step="$1"
  local unwrapped="$CKPT_ROOT/unwrapped_ds1024_z64_step=${step}.ckpt"
  local out_dir="$EVAL_ROOT/step_${step}"
  [ -f "$out_dir/heldout_summary.json" ] && [ "$FORCE" != "1" ] && return 0
  mkdir -p "$out_dir"
  local seld=()
  [ "$WITH_SELD" = "1" ] && seld=(--with-seld)
  CUDA_VISIBLE_DEVICES="$GPU" uv run python dataset/evaluation/compare_vae_heldout.py \
    --num "$NUM" "${seld[@]}" --vae4-config "$MODEL_CFG" --vae4-ckpt "$unwrapped" --out "$out_dir" \
    2>&1 | tee "$EVAL_ROOT/logs/step_${step}.log"
}

preencode_eval_subset() {
  local step="$1"
  local unwrapped="$CKPT_ROOT/unwrapped_ds1024_z64_step=${step}.ckpt"
  local lat_dir="$LATENT_ROOT/step_${step}"
  [ -f "$lat_dir/details.json" ] && [ "$FORCE" != "1" ] && return 0
  CUDA_VISIBLE_DEVICES="$GPU" uv run python pre_encode_4ch.py \
    --model-config "$MODEL_CFG" --ckpt-path "$unwrapped" \
    --dataset-config "$HELDOUT_CFG" --output-path "$lat_dir" \
    --no-pad --batch-size 1 --num-workers 4
}

decode_step() {
  local step="$1"
  local lat_dir="$LATENT_ROOT/step_${step}"
  local recon_dir="$EVAL_ROOT/step_${step}/recon"
  [ -f "$recon_dir/manifest.json" ] && [ "$FORCE" != "1" ] && return 0
  mkdir -p "$recon_dir"
  CUDA_VISIBLE_DEVICES="$GPU" uv run python scripts/vae/eval/decode_latents_4ch.py \
    --latent-root "$lat_dir" --output-dir "$recon_dir" --decode-all --no-listen-stereo
}

eval_preencode() {
  local step="$1"
  local recon_dir="$EVAL_ROOT/step_${step}/recon"
  local out_dir="$EVAL_ROOT/step_${step}"
  [ -f "$out_dir/preencode_summary.json" ] && [ "$FORCE" != "1" ] && return 0
  local seld=()
  [ "$WITH_SELD" = "1" ] && seld=(--with-seld)
  CUDA_VISIBLE_DEVICES="$GPU" uv run python dataset/evaluation/eval_vae_recon_full.py \
    --recon-dir "$recon_dir" --out "$out_dir" "${seld[@]}" \
    2>&1 | tee "$EVAL_ROOT/logs/step_${step}.log"
}

preencode_dit() {
  local step="$1"
  local unwrapped="$CKPT_ROOT/unwrapped_ds1024_z64_step=${step}.ckpt"
  local lat_dir="${AMBIT_CKPT_ROOT}/audio_latents/construct_4ch_step=${step}"
  [ -f "$lat_dir/details.json" ] && [ "$FORCE" != "1" ] && return 0
  CUDA_VISIBLE_DEVICES="${PREENCODE_GPU:-0,1,2,3}" uv run python pre_encode_4ch.py \
    --model-config "$MODEL_CFG" --ckpt-path "$unwrapped" \
    --dataset-config "$DIT_CFG" --output-path "$lat_dir" \
    --no-pad --batch-size 1 --num-workers 8
}

MISSING=()
for step in "${STEP_LIST[@]}"; do
  if ! raw="$(find_raw_ckpt "$step")"; then
    echo "missing checkpoint: step=$step" >&2
    MISSING+=("$step")
    continue
  fi
  unwrap_step "$step" "$raw"
  if [ "$MODE" = "heldout" ]; then
    eval_heldout "$step"
  else
    preencode_eval_subset "$step"
    decode_step "$step"
    eval_preencode "$step"
  fi
  if [ "$PREENCODE_DIT" = "1" ]; then
    preencode_dit "$step"
  fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
  echo "missing steps: ${MISSING[*]}" >&2
fi

uv run python scripts/vae/eval/summarize_vae_sweep.py --eval-root "$EVAL_ROOT" --mode "$MODE" || true
