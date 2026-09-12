#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env_audio_dataset.sh

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade -r requirements.txt

echo
echo "Setup complete."
echo "Next, authenticate if needed:"
echo "  hf auth login"
echo
echo "For gated datasets, first accept access on Hugging Face, then either run hf auth login"
echo "or export HF_TOKEN before starting downloads."
