#!/usr/bin/env bash
# The historical filename remains compatible; GPU allocation is explicit.
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_sceneplan_transfusion_editing_ar_clap44_5gpu.sh" "$@"
