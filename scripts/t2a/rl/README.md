# OPSD trainers and data prep

Library code lives in `stable_audio_tools/training/transfusion_opsd/`.
Method: [`docs/OPSD.md`](../../../docs/OPSD.md).
Current run: [`docs/TRANSFUSION_OPSD_MAINLINE.md`](../../../docs/TRANSFUSION_OPSD_MAINLINE.md).

## Data helpers

- `prepare_event_opsd_data.py`
- `prepare_event_opsd_holdout_v2.py`

These describe the original holdout construction. They do not make a consumed
request panel “fresh” again. Current training streams and the unopened
evaluation panel are pinned in the run config.

## Current Editing mainline

| Script | Role |
| --- | --- |
| `train_editing_opsd_repaired_fresh.py` | **Mainline learner**: 8-rank exclusive-branch recipe, resume Adam |
| `launch_editing_opsd_repaired_fresh.py` | **Mainline launcher**: train to 2000, eval every 250, Top 5 |
| `evaluate_editing_opsd_eight_gpu.py` | 8-worker 500×2 eval + canonical 20k native validation |
| `report_editing_opsd_eight_gpu.py` | Nine-metric table and guarded Top-5 promotion |

Continue the current v3 run. Do not re-initialize from 40k. Permanent
checkpoints: 500, 1000, 1500, 2000. Top 5 uses the fixed validation panel only.

```bash
python scripts/t2a/rl/launch_editing_opsd_repaired_fresh.py \
  --run-dir "$AMBIT_CKPT_ROOT/transfusion_opsd/editing_v3" \
  --recover
```

## Supporting / historical trainers

| Script | Role |
| --- | --- |
| `train_editing_opsd_stream.py` | Joint load, ordinal streams, request/pair readers |
| `train_editing_opsd_spatial.py` | Fixed-reference retention + paired FOA |
| `train_editing_opsd_complete.py` | Request constraints, removal hold, optional fallback |
| `train_editing_opsd_throughput.py` | Frozen-Qwen cache + unique-token skip + request eval cache |
| `train_editing_opsd_fresh2000.py` | Historical 4-rank fresh-40k → 2000 |
| `train_editing_opsd_eight_gpu.py` | Historical 4→8 resize of that fresh run |
| `train_editing_opsd_removal_repair.py` | Isolated removal-only paired correction |
| `train_editing_opsd_selective.py` | Same-plan teacher selection / prefix hold (diagnosed) |
| `train_editing_opsd_stability.py` | Isolated resume of the low-LR joint recipe |
| `train_editing_opsd_to2000.py` | Older 4-rank 500 → 2000 continuation |
| `train_editing_opsd_0915_recipe.py` | Archived 0915-recipe / LR control |

`launch_editing_opsd_*.py` start `torch.distributed.run` with `sys.executable`.
Set `CUDA_VISIBLE_DEVICES`. The current mainline requires eight visible devices
and the 16+512 recipe.

FT / ReFL (`train_editing_native_ft.py`) is present and not the current task.
The parked privileged-teacher proposal is archived and must not replace this
mainline.
