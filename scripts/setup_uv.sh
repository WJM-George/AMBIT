#!/usr/bin/env bash
# One-time / refresh setup for stable-audio-tools using uv (no manual venv activate).
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  python3 -m pip install --user uv
  export PATH="${HOME}/.local/bin:${PATH}"
fi

echo "==> uv sync (core + train extras)"
uv sync --extra train

echo
echo "==> Verify torch + CUDA"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

echo
echo "==> Optional: Flash Attention (recommended, separate compile step)"
echo "    uv pip install packaging ninja"
echo "    uv pip install flash-attn --no-build-isolation"
echo "    uv run python -c \"import flash_attn; print(flash_attn.__version__)\""

echo
echo "Setup done. Use uv run for all scripts — do NOT mix conda activate + source .venv/bin/activate."
echo "Example sanity check:"
echo "  uv run python -m stable_audio_tools.training.overfit_sanity_4ch \\"
echo "    --config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \\"
echo "    --pretrained-ckpt /path/to/stable_audio_open_model.safetensors \\"
echo "    --sls-root ${AMBIT_DATA_ROOT}/datasets/spatial_librispeech \\"
echo "    --mrsdrama-root ${AMBIT_DATA_ROOT}/datasets/mrsdrama/snapshot \\"
echo "    --steps 1000 --batch-size 4 --out-dir ${AMBIT_CKPT_ROOT}/vae_4ch_sanity_out"
