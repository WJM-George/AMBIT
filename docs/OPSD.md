# On-policy self-distillation (OPSD)

OPSD is a post-training stage on the already-trained Transfusion stack. The same 15-block Transformer keeps a grammar-constrained ScenePlan head (AR) and a rectified-flow renderer (DiT). Execution feedback updates both, with every trainable tensor appearing in one optimizer only. Frozen Qwen, the FOA VAE, and CLAP observers do not receive gradients.

This document describes the **current Editing recipe** (mid-September 2026). Generation shares the same interfaces; recent Generation pilots did not give a stable spatial gain, so development has been on Editing.

A complete bidirectional loop (new plans that unlock new renderer skill, which then teaches the planner) is **not** claimed.

## What the student sees

At request time the editor reads only:

- source FOA VAE latent and its mask
- the raw edit instruction

Paired ground-truth plans and target latents stay on the paired RF / retention side. They are not fed back as request-side teacher labels.

## Current recipe

| Knob | Value |
| --- | --- |
| Start point | original Editing joint 40k (AR + DiT + shared Transformer) |
| STE / discrete credit into AR logits | **off** |
| Global batch | 16 requests + 512 paired rows |
| Typical 4-GPU split | 4 requests + 128 paired rows per rank; paired microbatch 48 |
| Shared Transformer / DiT LR | `3.752567682220472e-7` (the 40k effective rate; earlier 2e-6 was 5.33× too high) |
| AR / structure-head LR | `5e-6` |
| Optimizer | fresh Adam, β = (0.9, 0.95), weight decay 0.001 |
| Qwen | frozen, with a bounded forward cache |
| Unique legal token | skip the Transformer forward when the grammar leaves one token |

STE-off means DiT loss does not travel through discrete argmax into AR decisions. Shared-parameter joint learning and execution-feedback distillation still hold. The older straight-through credit path remains in the package as a historical component.

## Method pieces

1. **Student plan.** Greedy (or grammar-constrained) native tokens on the student’s own prefix. `native_token_alignment` rejects silently-equivalent tokenizations.
2. **Fixed reference retention.** The frozen 40k model recomputes non-text decision distributions on the current prefix (source, kind, trajectory, pose, onset/offset). Azimuth cones stay around that frozen reference, not around a drifting student mode. `native_coarse_choice_retention`, `editing_spatial_retention`.
3. **Request constraints.** Bind the instruction to a source from visible text only. Ambiguity is recorded; there is no target-plan peek. `editing_request_constraints`, `request_grounded_text_retention`.
4. **Actual execution.** A few legal plans are rendered with paired noise. Content uses native FOA-latent CLAP; space uses request-grounded static/linear checks; clipping and unedited windows are recorded. Failures still affect plan credit.
5. **Same-plan RF.** Only a qualified terminal of *this* plan may become a velocity teacher. Another plan’s output is never a regression label. `execution_teacher_selection`, `native_paired_rf`.
6. **Paired FOA auxiliary.** Low-noise RF on the true pair, decoded to a signed four-channel complex T-F covariance, edit-region, and W-channel spectrum. VAE stays frozen. `editing_binaural_retention`.
7. **Exemption.** A decision may leave the fixed-reference hold only if both noises produce a qualified, better execution. Proposing a candidate is not enough.

Coverage is still incomplete for overlapping sources, deletion-time per-source keep, and distance-from-loudness.

## Evaluation

Nine metrics, independent of the training CLAP:

Paired CLAP ↑, FD-CLAP ↓, FAD ↓, FD-PANN ↓, KL ↓, LSD ↓, GCC ↓, CRW ↓, FSAD ↓.

`editing_nine_metrics.py` loads the existing KEMAR / CLAP / PANN / VGGish / StereoCRW suite from `$AMBIT_EDITING_BENCH` (default: `$AMBIT_DATA_ROOT/sceneplan_transfusion_editing_v1/materialized/logs/takeover-20260905T095122+0800/MAINLINE/EXTERNAL_BASELINES_SWANWEAVE_V1`).

Small development panels are for selection only. COMMON1000 is a frozen comparison set and is not used to pick a candidate.

## Code map

| Path | Role |
| --- | --- |
| `stable_audio_tools/training/transfusion_opsd/` | Library: retention, teachers, stream, metrics |
| `scripts/t2a/rl/train_editing_opsd_*.py` | Learners (stream → spatial → complete → selective → stability → 500→2000) |
| `scripts/t2a/rl/launch_editing_opsd_*.py` | Distributed launch; uses `sys.executable` and `CUDA_VISIBLE_DEVICES` |
| `scripts/t2a/experiments/ar_structured_v1/` | Joint AR/DiT loader used by the stream learner |
| `tests/test_opsd_*.py`, `tests/test_editing_opsd_*.py` | Unit and contract tests |

Set `CUDA_VISIBLE_DEVICES` yourself. Launchers do not assume a laboratory GPU map. A four-rank continuation still requires `len(physical_gpus) == 4` and the 16+512 recipe.

```bash
# after the Editing 40k joint checkpoint exists
python scripts/t2a/rl/train_editing_opsd_spatial.py --config path/to/opsd_config.json
# longer continuation (500 → 2000, save every 500)
python scripts/t2a/rl/train_editing_opsd_to2000.py --config path/to/continue.json --resume path/to/step500.pt
```

Configs, indexes, and checkpoints stay under `$AMBIT_DATA_ROOT` / `$AMBIT_CKPT_ROOT`. Nothing in this tree is a released replacement for the 40k baseline until an independent COMMON1000 evaluation says so.
