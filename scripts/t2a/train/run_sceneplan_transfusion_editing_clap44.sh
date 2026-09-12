#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_sceneplan_transfusion_editing_clap44_5gpu.sh" "$@"
