#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-${AMBIT_CKPT_ROOT}/baselines/p10_60k_15row_v1}"
MODEL_ROOT="${BASELINE_ROOT}/models"
LOG_ROOT="${BASELINE_ROOT}/logs/downloads"
mkdir -p "${MODEL_ROOT}" "${LOG_ROOT}"

hf_cli=(uvx --from huggingface_hub hf download)

"${hf_cli[@]}" AEmotionStudio/stable-audio-open-models model_config.json \
  --revision 385bdecabe7f8a613a4e1d568362f0e00d1f692b \
  --local-dir "${MODEL_ROOT}/stable-audio-open-1.0-config" \
  >"${LOG_ROOT}/stable_audio_open_config.log" 2>&1 &
pid_sao=$!

"${hf_cli[@]}" declare-lab/TangoFlux \
  --revision 367005e963cb3a9fb2e03a46104d7de23e34ceea \
  --local-dir "${MODEL_ROOT}/TangoFlux" \
  >"${LOG_ROOT}/tangoflux.log" 2>&1 &
pid_tango=$!

"${hf_cli[@]}" hkchengrex/MMAudio \
  weights/mmaudio_large_44k_v2.pth ext_weights/v1-44.pth \
  --revision eb13a1a98fdbec91753775c57b074ccdfc60587c \
  --local-dir "${MODEL_ROOT}/MMAudio" \
  >"${LOG_ROOT}/mmaudio.log" 2>&1 &
pid_mma=$!

"${hf_cli[@]}" HKUSTAudio/AudioX-Turbo \
  audiox_turbo/audiox_turbo.ckpt pretransform/vae.ckpt config.json \
  --revision 67af549c42aabdb666e559cb4993eddf48b62f08 \
  --local-dir "${MODEL_ROOT}/AudioX-Turbo" \
  >"${LOG_ROOT}/audiox_turbo.log" 2>&1 &
pid_audiox=$!

"${hf_cli[@]}" Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --revision 5ecdb67327fd37bb2e042aab12ff7391903235d3 \
  --local-dir "${MODEL_ROOT}/Qwen3-TTS-12Hz-1.7B-VoiceDesign" \
  >"${LOG_ROOT}/qwen3_tts_voice_design.log" 2>&1 &
pid_qwen=$!

woosh_zip="${MODEL_ROOT}/Woosh-Flow.zip"
if ! unzip -tqq "${woosh_zip}" >/dev/null 2>&1; then
  # GitHub's HTTP/2 stream can reset on this 1.2 GiB release asset.  Force
  # HTTP/1.1 and resume any validated partial transfer instead of restarting.
  curl --http1.1 -L --fail --retry 20 --retry-all-errors --retry-delay 2 -C - \
    https://github.com/SonyResearch/Woosh/releases/download/v1.0.0/Woosh-Flow.zip \
    -o "${woosh_zip}" >"${LOG_ROOT}/woosh_flow.log" 2>&1 &
  pid_woosh=$!
else
  pid_woosh=""
fi

failed=0
for pid in "${pid_sao}" "${pid_tango}" "${pid_mma}" "${pid_audiox}" "${pid_qwen}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ -n "${pid_woosh}" ]] && ! wait "${pid_woosh}"; then
  failed=1
fi
if [[ "${failed}" -ne 0 ]]; then
  echo "one or more baseline downloads failed; inspect ${LOG_ROOT}" >&2
  exit 1
fi

unzip -tqq "${woosh_zip}" >/dev/null

if [[ ! -d "${MODEL_ROOT}/Woosh-Flow" ]]; then
  unzip -q "${woosh_zip}" -d "${MODEL_ROOT}"
fi

sha256sum \
  "${MODEL_ROOT}/stable-audio-open-1.0-config/model_config.json" \
  "${MODEL_ROOT}/TangoFlux/tangoflux.safetensors" \
  "${MODEL_ROOT}/TangoFlux/vae.safetensors" \
  "${MODEL_ROOT}/MMAudio/weights/mmaudio_large_44k_v2.pth" \
  "${MODEL_ROOT}/MMAudio/ext_weights/v1-44.pth" \
  "${MODEL_ROOT}/AudioX-Turbo/audiox_turbo/audiox_turbo.ckpt" \
  "${MODEL_ROOT}/AudioX-Turbo/pretransform/vae.ckpt" \
  "${MODEL_ROOT}/Qwen3-TTS-12Hz-1.7B-VoiceDesign/model.safetensors" \
  "${woosh_zip}" \
  >"${BASELINE_ROOT}/MODEL_SHA256SUMS.txt"

echo "BASELINE_DOWNLOADS_COMPLETE=${BASELINE_ROOT}"
