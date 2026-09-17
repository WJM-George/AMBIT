# OPSD trainers and data prep

Library code lives in `stable_audio_tools/training/transfusion_opsd/`.
Method and recipe: [`docs/OPSD.md`](../../../docs/OPSD.md).

## Data helpers

- `prepare_event_opsd_data.py`
- `prepare_event_opsd_holdout_v2.py`

These describe the original holdout construction. They do not make a consumed
request panel “fresh” again. Current training streams and the unopened
evaluation panel are pinned in the run config.

## Current Editing trainers

| Script | Role |
| --- | --- |
| `train_editing_opsd_stream.py` | Joint load, ordinal streams, request/pair readers |
| `train_editing_opsd_spatial.py` | Fixed-reference retention + paired FOA |
| `train_editing_opsd_complete.py` | Request constraints on top of spatial |
| `train_editing_opsd_selective.py` | Same-plan teacher selection / prefix hold (diagnosed, not default) |
| `train_editing_opsd_stability.py` | Isolated resume of the low-LR joint recipe |
| `train_editing_opsd_throughput.py` | Frozen-Qwen cache + unique-token skip |
| `train_editing_opsd_to2000.py` | 500 → 2000 continuation, save every 500 |
| `train_editing_opsd_0915_recipe.py` | Archived 0915-recipe / LR control |

`launch_editing_opsd_*.py` start `torch.distributed.run` with `sys.executable`.
Set `CUDA_VISIBLE_DEVICES`. Four-rank jobs still require four visible devices
and the 16+512 recipe.

`evaluate_editing_opsd_branch_cross.py` swaps AR vs DiT on a frozen panel.
`report_editing_opsd_to2000.py` and `retire_editing_opsd_checkpoints.py` keep
Top-k recoverable states.

FT / ReFL (`train_editing_native_ft.py`) is present and not the current task.
