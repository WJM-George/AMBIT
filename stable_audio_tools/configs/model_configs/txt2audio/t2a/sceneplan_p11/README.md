# P11 unified Transfusion-CoT planner

The canonical development graph is now P11-v4:

```text
task evidence
  -> discrete SceneSketch / DeltaSketch
  -> continuous P10 ExecutionState / DeltaExecutionState
  -> deterministic ScenePlan / Atomic Patch assembler
  -> frozen P10-v11 step 150000
  -> FOA
```

The only launchable candidate config is:

```text
qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json
```

Its explicit model type is `sceneplan_p11_v4`; the dataset must use
`p11_contract=sketch_first_transfusion_cot_v4`. This separation prevents a v4
run from silently entering the retired `[8,40]` core40 graph.

## Unified G/U/E routes

The three tasks share one Qwen3.5-0.8B backbone, LoRA adapters, token
embedding, five continuous slots, and P10 handoff. G/U use a rectified-flow
posterior. E uses a dedicated direct-regression `delta_head`. Rotation and
distance direction is encoded by operation-specific negative/positive
DeltaSketch tokens and maps exactly to P10's legal endpoints. There is no
parallel direction head. Discrete SceneSketch/DeltaSketch is read at Qwen
hidden layer 16; the top-eight-layer execution tower then predicts the
continuous state, following the Self-Cascaded Transformer separation.

| Task | Evidence | Learned output | Deterministic result |
|---|---|---|---|
| G | raw user text | SceneSketch + ExecutionState | ScenePlan -> P10 -> FOA |
| U | FOA VAE latent + windowed CLAP + optional frozen-ASR hypothesis | posterior SceneSketch + ExecutionState | ScenePlan; optional P10 cycle render |
| E | current ScenePlan + instruction | DeltaSketch + DeltaExecutionState | Atomic Patch -> new ScenePlan -> same-seed P10 |

`SceneSketch` is the only authority for room, source inventory, kind,
description, speaker description, and transcript. `ExecutionState` is exactly
`5 x 15` and owns only duration, activity, static/linear motion, azimuth,
elevation, and distance. For the binary P10 edit choices, negative/positive
rotation maps to `-45/+45` degrees and negative/positive distance maps to
`0.75/1.25` through exact token lookup. The assembler is the only way to create
the final ScenePlan, so a numeric
intervention cannot rewrite P10 semantic text.

Understanding always receives FOA/VAE and frozen CLAP evidence. A frozen ASR
cache generated from input FOA only is admitted when its train-calibrated
confidence gate passes. It is not concatenated as a weak autoregressive hint.
After the model decodes the discrete SceneSketch and before it predicts the
continuous SceneThought, deterministic lexical authority selects the detected
speech owner and replaces only that sketch transcript. The corrected sketch
may condition a different U numeric posterior; the protected direction is the
reverse one: ExecutionState can never rewrite transcript, caption, kind, room,
or source identity. Rows without reliable ASR must remain byte-exact under the
ASR on/drop intervention. The D0 control instead applies ASR after its full
autoregressive ScenePlan and therefore additionally preserves D0 numeric state.
`oracle_transcript_wiring_v1` remains a tensor-wiring diagnostic and must not
be reported as a model result.

## P10 envelope

P11-v4 is fail-closed to P10-v11's trained capability:

- one to four sources and at most one formal speech source;
- at most 648 VAE frames, approximately 15 seconds;
- static or linear trajectories;
- frame-grid activity and quantized azimuth/elevation/distance;
- four trained room classes;
- constant 0 dB gain;
- semantic-caption compiler v2 with exact ScenePlan transcript authority;
- no word-level timing, keyframed motion, or waveform-local preservation.

## Current gates

Pilot90, two-seed replication, matched 10k training for all three arms and the
complete 300-row evaluator-v9 screen are finished. All current reports use EMA,
seed 42 for the canonical screen, prefix-recompute decoding and the pinned
Qwen kernel.

| Arm | weighted K=1 | G | U | E |
|---|---:|---:|---:|---:|
| D0 autoregressive | 0.86876 | 0.96049 | 0.70435 | 0.99623 |
| Direct-MSE | 0.81018 | 0.81350 | 0.70422 | 0.94814 |
| Flow-R1 hybrid | 0.79161 | 0.77937 | 0.69453 | 0.93328 |

Flow is therefore **not** claimed to be the best 10k point estimator. It is
retained because it passes a capability that the D0 sampler does not: at K=8,
Flow G/U semantic immutability is `1.0/1.0`, numeric non-degeneracy is
`1.0/1.0`, and oracle lifts are `0.03997/0.02784`. D0's corresponding semantic
immutability is `0.344/0.0`. Flow also passes zero/shuffle/swap/replace
mechanistic interventions, cross-process fixed-seed replay, reliable-ASR
on/drop causality and frozen-P10-v11 100-step FOA closure.

The current gate is one medium convergence test, not full training: 269,568
pair-aware rows, eight GPUs, batch 8/GPU, seed 42 and 10,000 optimizer steps.
The terminal checkpoint will automatically receive K=1/4/8, intervention,
ASR, reproducibility and frozen-P10 closure evaluation. A decision rule frozen
before that checkpoint yields GO, targeted REVISE or STOP. Approximately ten
million rows remain unbuilt and full-corpus training remains unauthorized.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python \
  scripts/t2a/test/validate_sceneplan_p11_v4_control_direction.py --batch-size 8

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python \
  scripts/t2a/test/smoke_sceneplan_p11_v4_graph.py \
    --batch-size 8 --require-control-direction-pair
```

The launcher additionally permits only
`qwen35_0p8b_sceneplan_p11_baseline_direct_mse.json` (the same graph and
reliable-ASR assembler with Direct-MSE G/U) and
`qwen35_0p8b_sceneplan_p11_baseline_discrete_d0.json`
(the paper-required D0 baseline). `_qwen35_0p8b_sceneplan_p11_base.json` is an
internal inheritance base and is never a launch target. Failed connector,
core40, and Flow-E endpoint-forcing files were removed from the active tree;
the core40 model/train/decode implementation has also been removed from the
P11 base planner. Their hashes and evidence are recorded in
`artifacts/sceneplan_p11/p11_active_tree_retirement_manifest_v2_20260902.json`.

Current boundary evidence:

- `artifacts/sceneplan_p11/P11_LATEST.json`;
- `artifacts/sceneplan_p11/releases/P11_TRANSFUSION_COT_V4_DEV10_20260903.json`;
- `artifacts/sceneplan_p11/evals/p11_v4_d0_screen10k_s42_ema_heldout300_k148_merged_evalv9_20260903.json`;
- `artifacts/sceneplan_p11/evals/p11_v4_direct_screen10k_s42_ema_heldout300_k1_evalv9_20260903.json`;
- `artifacts/sceneplan_p11/evals/p11_v4_flow_r1_screen10k_s42_ema_heldout300_k148_evalv9_20260903.json`;
- `artifacts/sceneplan_p11/evals/p11_v4_flow_r1_screen10k_s42_ema_causal_heldout30_evalv2_tf32off_20260903.json`;
- `docs/transfusion_cot_idea/2026-09-03_seed42_medium_scale_protocol.md`.

The complete contract and limitations are documented in
`docs/sceneplan_v2/p11_v4_unified_transfusion_cot_contract_20260901.md`.
