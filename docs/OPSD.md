# On-policy self-distillation (OPSD)

OPSD is a post-training stage on the already-trained Transfusion stack. The same 15-block Transformer keeps a grammar-constrained ScenePlan head (AR) and a rectified-flow renderer (DiT). Execution feedback updates both, with every trainable tensor appearing in one optimizer only. Frozen Qwen, the FOA VAE, and CLAP observers do not receive gradients.

This document describes the **current Editing mainline** (18 September 2026): the v3 500-step exclusive-branch recipe, now continuing on eight GPUs from the recovered 500-step state to 2000 updates. Generation shares the same interfaces; recent Generation pilots did not give a stable spatial gain.

A complete bidirectional loop (new plans that unlock new renderer skill, which then teaches the planner) is **not** claimed. This recipe is also **not** a finished privileged same-prefix / same-state teacher OPSD.

Continuation rules: [`TRANSFUSION_OPSD_MAINLINE.md`](TRANSFUSION_OPSD_MAINLINE.md).

## What the student sees

At request time the editor reads only:

- source FOA VAE latent and its mask
- the raw edit instruction

Paired ground-truth plans and target latents stay on the paired RF / retention / fallback side. They are not fed back as request-side teacher labels during rollout or reward.

## Current recipe

| Knob | Value |
| --- | --- |
| Start point | original Editing joint 40k (AR + DiT + shared Transformer) |
| STE / discrete credit into AR logits | **off** |
| Global batch | 16 requests + 512 paired rows |
| 8-GPU split | 2 requests + 64 paired rows per rank; paired microbatch 48; 8 decoded rows |
| Shared Transformer / DiT LR | `3.752567682220472e-7` (the 40k effective rate) |
| AR / structure-head LR | `5e-6` |
| Optimizer | AdamW, β = (0.9, 0.95), weight decay 0.001; **resume the existing run**, do not re-init from 40k |
| Qwen | frozen, with a bounded forward cache |
| Unique legal token | skip the Transformer forward when the grammar leaves one token |
| Request fallback | `exclusive_request_branch_v1`: each request AR/RF uses qualified execution or same-request GT, unit budget each |
| Validation | fixed 500 requests × 2 noises; candidates every 250; permanent saves at 500/1000/1500/2000 |
| Top 5 | `guardrails_then_median_relative_gain_v1` on that validation panel only (overall −1%, operation −3%) |

STE-off means DiT loss does not travel through discrete argmax into AR decisions. Shared-parameter joint learning and execution-feedback distillation still hold. The older straight-through credit path remains in the package as a historical component.

## Method pieces

1. **Student plan.** Greedy (or grammar-constrained) native tokens on the student’s own prefix. `native_token_alignment` rejects silently-equivalent tokenizations.
2. **Fixed reference retention.** The frozen 40k model recomputes non-text decision distributions on the current prefix (source, kind, trajectory, pose, onset/offset). Azimuth cones stay around that frozen reference, not around a drifting student mode. `native_coarse_choice_retention`, `editing_spatial_retention`.
3. **Request constraints.** Bind the instruction to a source from visible text only. Ambiguity is recorded; there is no target-plan peek. `editing_request_constraints`, `request_grounded_text_retention`.
4. **Actual execution.** A few legal plans are rendered with paired noise. Content uses native FOA-latent CLAP; space uses request-grounded static/linear checks; clipping and unedited windows are recorded. Failures still affect plan credit.
5. **Same-plan RF.** Only a qualified terminal of *this* plan may become a velocity teacher. Another plan’s output is never a regression label. `execution_teacher_selection`, `native_paired_rf`.
6. **Exclusive branch fallback.** If a request branch lacks joint execution supervision, that branch uses the verified training pair in the backward pass only. Execution and GT never share a branch. `branch_request_supervision`, `request_paired_supervision`.
7. **Paired FOA auxiliary.** Low-noise RF on the true pair, decoded to a signed four-channel complex T-F covariance, edit-region, and W-channel spectrum. VAE stays frozen. `editing_binaural_retention`.
8. **Removal hold.** A deletion request does not keep a whole-scene velocity anchor on the object being removed. `removal_retention`.
9. **Exemption.** A decision may leave the fixed-reference hold only if both noises produce a qualified, better execution. Proposing a candidate is not enough.

Coverage is still incomplete for overlapping sources, deletion-time per-source keep, and distance-from-loudness.

## Evaluation

Nine metrics, independent of the training CLAP:

Paired CLAP ↑, FD-CLAP ↓, FAD ↓, FD-PANN ↓, KL ↓, LSD ↓, GCC ↓, CRW ↓, FSAD ↓.

`editing_nine_metrics.py` loads the KEMAR / CLAP / PANN / VGGish / StereoCRW suite from `$AMBIT_EDITING_BENCH` (default: `$AMBIT_DATA_ROOT/editing_bench`).

The 500 × 2 development panel is for candidate retention only. COMMON1000 / matched-1000 is a frozen comparison set and is not used to pick a candidate. Generated audio is not saved.

## Code map

| Path | Role |
| --- | --- |
| `stable_audio_tools/training/transfusion_opsd/` | Library: retention, teachers, stream, metrics, exclusive-branch fallback |
| `scripts/t2a/rl/train_editing_opsd_repaired_fresh.py` | **Current mainline learner** (8 ranks, resume-only after step 0) |
| `scripts/t2a/rl/launch_editing_opsd_repaired_fresh.py` | **Current mainline launcher** (train → 250-step eval → Top 5) |
| `scripts/t2a/rl/train_editing_opsd_*.py` | Historical learners (stream → spatial → complete → fresh2000 → eight-GPU resize → to2000) |
| `tests/test_opsd_*.py` | Unit and contract tests |

Set `CUDA_VISIBLE_DEVICES` yourself. Launchers use `sys.executable` and do not assume a laboratory GPU map. The current recipe still requires eight visible devices and the 16+512 split.

```bash
# continue the v3 run; restores Adam and 8-rank data cursors
python scripts/t2a/rl/launch_editing_opsd_repaired_fresh.py \
  --run-dir "$AMBIT_CKPT_ROOT/transfusion_opsd/editing_v3" \
  --recover
```

Configs, indexes, and checkpoints stay under `$AMBIT_DATA_ROOT` / `$AMBIT_CKPT_ROOT`. Nothing in this tree is a released replacement for the joint editing baseline until an independent held-out evaluation says so.
