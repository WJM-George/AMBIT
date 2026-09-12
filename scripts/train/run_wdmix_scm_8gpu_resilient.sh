#!/usr/bin/env bash
# Resilient launcher for the 方案2 (W-downmix + grouped-KL) 8-GPU run.
#
# Infra hardening (why each piece exists):
#  - FIXED checkpoint dir (--checkpoint-dir): all ckpts land in ONE stable place,
#    NOT a per-restart wandb-run-id subdir. This is what previously caused the run
#    to silently "go backwards" to the 1M base after every NCCL restart.
#  - Resume = HIGHEST-step ckpt (parse step=NNN, not mtime): a failed restart can
#    write a lower-step ckpt with a newer timestamp; we must never pick that.
#  - Pre-launch zombie sweep: an NCCL hang can leave some ranks alive holding GPU
#    memory; kill any leftover train_4ch.py for THIS save-dir before relaunching.
#  - Short DDP timeout (--ddp-timeout-min): a hung collective aborts in ~10min
#    instead of the 30min default, so auto-restart is fast.
#  - Checkpoints every 10k steps: a crash costs <=~2h of steps, not 9h.
set -u

cd /home/tanhe/dataset_storage/stable-audio-tools

PY=/home/tanhe/dataset_storage/stable-audio-tools/.venv/bin/python
export PATH="/home/tanhe/dataset_storage/stable-audio-tools/.venv/bin:$PATH"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline
# Surface NCCL async errors promptly instead of hanging the full watchdog window.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

# Pin the GPUs. An empty/stale CUDA_VISIBLE_DEVICES inherited from the launching
# shell makes torch.cuda.is_available() false; Lightning used to answer that by
# training the whole run on CPU, which presents as a silent hang. ${:-} also
# substitutes when the variable is set but empty, which is exactly that case.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

A=stable_audio_tools/configs/model_configs/autoencoders
MODEL_CONFIG=$A/stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json
DATA=/mnt/sdc/ckpts/vae_abl_phase_scm/configs/dataset_frozen_1018957.json
BASE_CKPT=/mnt/sdc/ckpts/vae_abl_fidelity_cep_sisdr/checkpoints/vae_abl_fidelity_cep_sisdr/9bcle0bm/checkpoints/epoch=13-step=1000000.ckpt
SAVE_DIR=/mnt/sdc/ckpts/vae_wdmix_scm_8gpu
CKPT_DIR=$SAVE_DIR/checkpoints

mkdir -p "$CKPT_DIR"

# Highest-step ckpt across the NEW fixed dir AND any OLD wandb-nested dirs
# (so the first launch still finds the pre-existing 1.05M checkpoint).
pick_latest() {
    ls \
        "$CKPT_DIR"/epoch=*-step=*.ckpt \
        "$SAVE_DIR"/*/*/checkpoints/epoch=*-step=*.ckpt \
        "$SAVE_DIR"/checkpoints/*/*/checkpoints/epoch=*-step=*.ckpt \
        2>/dev/null \
    | sed -n 's/.*step=\([0-9]*\)\.ckpt$/\1 &/p' \
    | sort -n \
    | tail -1 \
    | awk '{print $2}'
}

# Kill any leftover train_4ch.py workers for THIS save-dir (never the wrapper).
sweep_zombies() {
    local pids
    pids=$(ps -eo pid,cmd | awk -v sd="$SAVE_DIR" \
        '/train_4ch\.py/ && $0 ~ sd && !/awk/ && !/run_wdmix_scm_8gpu_resilient/ {print $1}')
    if [ -n "$pids" ]; then
        echo "[resilient] sweeping leftover ranks: $pids"
        kill -9 $pids 2>/dev/null || true
        sleep 5
    fi
}

MAX_RETRIES=100
attempt=0

while [ "$attempt" -lt "$MAX_RETRIES" ]; do
    attempt=$((attempt + 1))

    sweep_zombies

    LATEST=$(pick_latest)
    if [ -n "${LATEST:-}" ] && [ -f "$LATEST" ]; then
        RESUME="$LATEST"
    else
        RESUME="$BASE_CKPT"
    fi

    echo "==================================================================="
    echo "[resilient] attempt $attempt/$MAX_RETRIES  $(date '+%F %T')"
    echo "[resilient] resuming from: $RESUME"
    echo "[resilient] checkpoint_dir: $CKPT_DIR"
    echo "==================================================================="

    "$PY" -u train_4ch.py \
        --model-config "$MODEL_CONFIG" \
        --dataset-config "$DATA" \
        --ckpt-path "$RESUME" \
        --num-gpus 8 --batch-size 2 --num-workers 4 \
        --precision bf16-mixed \
        --save-dir "$SAVE_DIR" \
        --checkpoint-dir "$CKPT_DIR" \
        --checkpoint-every 10000 \
        --ddp-timeout-min 10 \
        --max-steps 1500000 \
        --logger wandb --seed 42

    code=$?
    echo "[resilient] train_4ch.py exited with code $code at $(date '+%F %T')"

    # Clean exit (finished max-steps) -> stop.
    if [ "$code" -eq 0 ]; then
        echo "[resilient] clean exit; training complete."
        break
    fi

    # Manual kill (SIGTERM/SIGINT via 130/143) -> stop, do not fight the user.
    if [ "$code" -eq 130 ] || [ "$code" -eq 143 ]; then
        echo "[resilient] interrupted by signal ($code); not restarting."
        break
    fi

    echo "[resilient] crash detected; freeing GPUs and restarting in 30s..."
    sleep 30
done

echo "[resilient] launcher finished after $attempt attempt(s)."
