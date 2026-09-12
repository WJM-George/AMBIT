#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 {tangoflux|mmaudio|audiox_turbo|woosh|qwen3_tts}" >&2
  exit 2
fi

name="$1"
BASELINE_ROOT="${BASELINE_ROOT:-${AMBIT_CKPT_ROOT}/baselines/p10_60k_15row_v1}"
REPO_ROOT="${REPO_ROOT:-./evaluation_benchmark/repos}"
ENV_ROOT="${BASELINE_ROOT}/envs"
export UV_CACHE_DIR="${BASELINE_ROOT}/uv-cache"
mkdir -p "${ENV_ROOT}" "${UV_CACHE_DIR}"

case "${name}" in
  tangoflux)
    env_dir="${ENV_ROOT}/tangoflux"
    uv venv "${env_dir}" --python 3.10
    uv pip install --python "${env_dir}/bin/python" -e "${REPO_ROOT}/TangoFlux"
    ;;
  mmaudio)
    env_dir="${ENV_ROOT}/mmaudio"
    uv venv "${env_dir}" --python 3.11
    uv pip install --python "${env_dir}/bin/python" -e "${REPO_ROOT}/MMAudio"
    ;;
  audiox_turbo)
    env_dir="${ENV_ROOT}/audiox_turbo"
    uv venv "${env_dir}" --python 3.8
    uv pip install --python "${env_dir}/bin/python" \
      -r "${REPO_ROOT}/AudioX-Turbo/requirements.txt"
    uv pip install --python "${env_dir}/bin/python" --no-deps \
      -e "${REPO_ROOT}/AudioX-Turbo"
    uv pip install --python "${env_dir}/bin/python" soundfile==0.12.1
    ;;
  woosh)
    env_dir="${ENV_ROOT}/woosh"
    UV_PROJECT_ENVIRONMENT="${env_dir}" uv sync \
      --project "${REPO_ROOT}/Woosh" --extra cuda --no-group dev --no-group demo \
      --no-group api --no-group reaper --no-group audioio
    uv pip install --python "${env_dir}/bin/python" soundfile
    ;;
  qwen3_tts)
    env_dir="${ENV_ROOT}/qwen3_tts"
    uv venv "${env_dir}" --python 3.11
    uv pip install --python "${env_dir}/bin/python" -e "${REPO_ROOT}/Qwen3-TTS"
    ;;
  *)
    echo "unknown environment: ${name}" >&2
    exit 2
    ;;
esac

"${env_dir}/bin/python" - <<'PY'
import json
import platform

import torch

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
}, sort_keys=True))
PY

echo "BASELINE_ENV_READY=${name}:${env_dir}"
