# Transfusion OPSD mainline

Current mainline is the **Editing v3 exclusive-branch recipe**. Continue the
same method, optimizer, and data cursors. Do not switch recipes mid-run.

## Recipe

- Start from the joint editing checkpoint (AR + DiT + shared Transformer).
- Straight-through credit into AR logits is **off**.
- Eight ranks. Global batch: 16 requests + 512 paired rows (2 + 64 per rank);
  paired microbatch 48; 8 decoded rows.
- Shared Transformer / DiT at the joint-training rate; AR / structure heads at
  `5e-6`. Resume AdamW; do not re-initialize.
- Student rollout sees only source FOA + instruction. Labels never enter
  rollout or rewards.
- Exclusive per-branch fallback: each request AR and RF independently uses a
  qualified execution or the same-request verified pair. Each branch has unit
  budget.
- Fixed-reference retention, request constraints, paired FOA auxiliary, and
  the removal-source hold stay on.

This is **not** a completed privileged same-prefix / same-state teacher OPSD,
and it is not a claim that a development-panel score replaces the joint
editing baseline.

## Continuation

1. Same method, rates, losses, and streams. Resume Adam and sampler cursors.
2. Keep complete states at 500, 1000, 1500, and 2000. Also keep validation
   Top 5 on the fixed 500-request × 2-noise panel
   (`guardrails_then_median_relative_gain_v1`).
3. Do not save generated audio. Do not pick Top 5 on a held-out comparison set.
4. Do not auto-switch to another historical version or a new teacher candidate.

```bash
python scripts/t2a/rl/launch_editing_opsd_repaired_fresh.py \
  --run-dir "$AMBIT_CKPT_ROOT/transfusion_opsd/editing_v3" \
  --recover
```

Set `CUDA_VISIBLE_DEVICES` to eight devices. The launcher uses the current
Python interpreter.
