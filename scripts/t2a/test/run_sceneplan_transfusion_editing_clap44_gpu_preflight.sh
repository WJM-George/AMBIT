#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_sceneplan_transfusion_editing_clap44_gpu_preflight_5gpu.sh" "$@"
