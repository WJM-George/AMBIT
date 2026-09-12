#!/usr/bin/env bash
set -euo pipefail

export AUDIO_DATASET_ROOT="${AUDIO_DATASET_ROOT:-/mnt/sdd/audio_dataset}"
export AUDIO_DATASET_SECONDARY_ROOT="${AUDIO_DATASET_SECONDARY_ROOT:-/mnt/sdb/audio_dataset}"
export AUDIO_DATASET_SECONDARY_KEYS="${AUDIO_DATASET_SECONDARY_KEYS:-bewo_1m,sphere360,audio_flan,spatial_librispeech}"
export AUDIO_DATASET_CACHE_ROOT="${AUDIO_DATASET_CACHE_ROOT:-/mnt/sdc/audio_dataset_cache}"
export AUDIO_DATASET_TMP="${AUDIO_DATASET_TMP:-/mnt/sdc/audio_dataset_tmp}"

export HF_HOME="${HF_HOME:-${AUDIO_DATASET_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${AUDIO_DATASET_CACHE_ROOT}/datasets}"
if [[ -z "${HF_TOKEN_PATH:-}" && -f "${HOME}/.cache/huggingface/token" ]]; then
  export HF_TOKEN_PATH="${HOME}/.cache/huggingface/token"
fi
export TMPDIR="${TMPDIR:-${AUDIO_DATASET_TMP}}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

# Deno is required by yt-dlp to solve YouTube's n-challenge (Sphere360 media).
if [[ -d "${HOME}/.deno/bin" ]]; then
  export PATH="${HOME}/.deno/bin:${PATH}"
fi
# Default cookies file for the Sphere360 yt-dlp downloader.
export SPHERE360_COOKIE="${SPHERE360_COOKIE:-${AUDIO_DATASET_SECONDARY_ROOT}/datasets/sphere360/youtube_cookies.txt}"

mkdir -p \
  "${AUDIO_DATASET_ROOT}" \
  "${AUDIO_DATASET_ROOT}/datasets" \
  "${AUDIO_DATASET_ROOT}/logs" \
  "${AUDIO_DATASET_ROOT}/manifests" \
  "${AUDIO_DATASET_SECONDARY_ROOT}" \
  "${AUDIO_DATASET_SECONDARY_ROOT}/datasets" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${TMPDIR}"

echo "AUDIO_DATASET_ROOT=${AUDIO_DATASET_ROOT}"
echo "AUDIO_DATASET_SECONDARY_ROOT=${AUDIO_DATASET_SECONDARY_ROOT}"
echo "AUDIO_DATASET_SECONDARY_KEYS=${AUDIO_DATASET_SECONDARY_KEYS}"
echo "HF_HOME=${HF_HOME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "HF_TOKEN_PATH=${HF_TOKEN_PATH:-}"
echo "TMPDIR=${TMPDIR}"
echo "SPHERE360_COOKIE=${SPHERE360_COOKIE}"
echo "deno=$(command -v deno || echo 'NOT FOUND')"
