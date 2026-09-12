# P10/P11 mainline

P10-v11 step 150,000 is the frozen FOA renderer. P11-v4 is the only active
planner-development graph in this tree:

```text
raw text / FOA evidence / edit instruction
  -> discrete SceneSketch or DeltaSketch
  -> continuous P10 ExecutionState or DeltaThought
  -> deterministic ScenePlan or Atomic Patch assembler
  -> frozen P10-v11 DiT -> frozen VAE decode -> FOA
```

The pinned P10 release is
`artifacts/releases/P10_SCENEPLAN_DIT_V11_150K_RELEASE.json`. Its checkpoint is
outside the repository at:

```text
/mnt/sdc/ckpts/dit/sceneplan_dit_v11_semantic_v2_protected_resume_150k/checkpoints/epoch=48-step=150000.ckpt
```

P11 is not permitted to silently expand the executor's abilities. The current
envelope is 1--4 music/sound/speech sources, at most one formal speech source,
at most 648 latent frames (about 15 seconds), frame-grid activity, static or
linear trajectories, quantized azimuth/elevation/distance, the four trained
room classes, and constant 0 dB gain. Word timing, keyframed motion, gain
automation, and waveform-preserving edits are outside this release.

## P11-v4 graph

All three tasks share one pinned Qwen3.5-0.8B backbone, LoRA adapters, evidence
connectors, constrained discrete decoding, five continuous slots, and one
deterministic assembler.

| Task | Evidence | Learned reasoning | Executable result |
|---|---|---|---|
| G | raw user text | SceneSketch + SceneThought | ScenePlan -> P10 -> FOA |
| U | FOA VAE latent + frozen CLAP always; reliable input-only frozen ASR when its gate passes | posterior SceneSketch + SceneThought | ScenePlan; optional P10 cycle render |
| E | current ScenePlan + edit instruction | DeltaSketch + DeltaThought | Atomic Patch -> new ScenePlan -> same-seed P10 |

The two learned objects have deliberately non-overlapping authority:

- `SceneSketch` owns room, source inventory and IDs, kind, semantic caption,
  speaker description, and transcript.
- `P10ExecutionState` is exactly `[5, 15]`: one duration slot plus four source
  slots containing only activity and static/linear spatial controls.
- `DeltaSketch` owns the edit operation, owner, and semantic payload;
  `DeltaThought` owns only the numeric change.
- The assembler is the only code allowed to create a complete ScenePlan.

Consequently, a continuous numeric intervention cannot rewrite a P10 semantic
caption. The internal causal order is discrete semantic authority first,
continuous executable state second, deterministic assembly last.

The canonical mixed objective is:

```text
L = L_discrete_CE
  + lambda_flow * L_rectified_flow(G,U)
  + lambda_solve * L_target_free_solve
  + lambda_locality * L_edit_locality
  + lambda_owner * L_DeltaSketch_owner
```

G/U retain the stochastic Flow posterior and explicit noise API. E is a
deterministic task and uses its own numeric `delta_head`; it does not consume
Flow noise. Rotation and distance use operation-specific negative/positive
DeltaSketch tokens that map exactly to P10's legal `-45/+45` or
`0.75/1.25` endpoint; there is no parallel continuous direction head. The
shared Qwen backbone and five executable slots remain common.

This is continuous CoT only in the executable sense: the continuous bottleneck
can be decoded, intervened on, and mapped to P10 controls. Arbitrary hidden
states are not reported as reasoning.

## Launchable matched arms

The launcher intentionally accepts only these arms:

| Arm | Config | Purpose |
|---|---|---|
| `canonical` | `qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json` | sole candidate: Flow G/U + reliable-ASR lexical assembler + Direct DeltaThought E |
| `direct_mse` | `qwen35_0p8b_sceneplan_p11_baseline_direct_mse.json` | same graph, lexical boundary, and E route; Direct-MSE G/U ablation |
| `discrete_d0` | `qwen35_0p8b_sceneplan_p11_baseline_discrete_d0.json` | direct autoregressive full-ScenePlan baseline |

Failed connector, core40 Transfusion-v2/v3, and Flow-E endpoint-forcing configs
are not launchable active configs. The core40 P11 model/train/decode branch has
also been removed. Their exact hashes and decision reports live under
`artifacts/sceneplan_p11`; do not add compatibility fallbacks to the active
graph.

## Immutable pilot and screening data

- canonical/Direct-MSE:
  `stable_audio_tools/configs/dataset_configs/sceneplan_p11_pilot90_curriculum_pair_aware_transfusion_cot_v4_reliable_asr_v1.json`
- matched D0:
  `stable_audio_tools/configs/dataset_configs/sceneplan_p11_pilot90_discrete_d0_matched_v4.json`
- v4 held-out panel:
  `stable_audio_tools/configs/dataset_configs/sceneplan_p11_heldout900_transfusion_cot_v4_reliable_asr_v1.json`
- matched 10k-scene/30k-row D0:
  `stable_audio_tools/configs/dataset_configs/sceneplan_p11_trial30k_matched_screening_v1_discrete_d0.json`
- matched 10k-scene/30k-row Direct-MSE/Flow-R1:
  `stable_audio_tools/configs/dataset_configs/sceneplan_p11_trial30k_matched_screening_v1_transfusion_cot_v4_reliable_asr.json`

The screening overlay has exactly one G/U/E row per frozen P10-v11 scene. G
balances exact, missing-layout, and coarse-control inputs; U balances identity
plus eight train-only representation stressors; E retains all six atomic
operations and gives every source-level operation exactly 400 examples for
each of source_0..source_3. It never reads the retired core40 sidecar and has
zero held-out sample, target-ScenePlan, prompt, or reserved-template overlap.

Frozen CLAP remains present for every U example. A versioned Distil-Whisper
cache derived only from input FOA contributes lexical authority only when the
train-calibrated confidence gate passes. Reliable ASR is applied by the
deterministic assembler; it is not concatenated as another weak autoregressive
hint. The earlier token-concatenation route is a measured negative ablation and
is not launchable. Oracle transcript wiring remains a graph diagnostic, never
a quality result.

## Validation, matched pilot, and idea screening

```bash
# Contract and sequence envelope
.venv/bin/python scripts/t2a/test/validate_sceneplan_p11_v4_contract.py \
  --base-scenes 300
.venv/bin/python scripts/t2a/test/validate_sceneplan_p11_v4_sequence_budget.py \
  --model-config stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/sceneplan_p11_heldout900_transfusion_cot_v4_reliable_asr_v1.json

# Strict matched-30k data, leakage, P10-envelope, and runtime gate
.venv/bin/python scripts/t2a/test/validate_sceneplan_p11_v4_screening.py

# Real-backbone, batch-8 graph smoke
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  scripts/t2a/test/smoke_sceneplan_p11_v4_graph.py \
  --model-config stable_audio_tools/configs/model_configs/txt2audio/t2a/sceneplan_p11/qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/sceneplan_p11_pilot90_curriculum_pair_aware_transfusion_cot_v4_reliable_asr_v1.json \
  --device cuda:0 --batch-size 8 --require-control-direction-pair

# CPU/data proof for the exact supervised/decode boundary
.venv/bin/python \
  scripts/t2a/test/validate_sceneplan_p11_v4_control_direction.py \
  --dataset-config stable_audio_tools/configs/dataset_configs/sceneplan_p11_trial30k_matched_screening_v1_transfusion_cot_v4_reliable_asr.json \
  --batch-size 8

# Single-GPU matched pilot; batch 8 is enforced by the launcher
RUN_NAME=<unique-name> P11_ARM=canonical GPU_IDS=0 \
  scripts/t2a/train/run_sceneplan_p11.sh pilot
RUN_NAME=<unique-name> P11_ARM=direct_mse GPU_IDS=0 \
  scripts/t2a/train/run_sceneplan_p11.sh pilot
RUN_NAME=<unique-name> P11_ARM=discrete_d0 GPU_IDS=0 \
  scripts/t2a/train/run_sceneplan_p11.sh pilot

# Canonical-seed 10k-step idea screening; all arms use one GPU, batch 8, the
# same data/order/budget, seed 42, and a fresh pinned Qwen initialization.
RUN_NAME=<unique-name> P11_ARM=discrete_d0 P11_SEED=42 GPU_IDS=0 \
  scripts/t2a/train/run_sceneplan_p11.sh screening
RUN_NAME=<unique-name> P11_ARM=direct_mse P11_SEED=42 GPU_IDS=0 \
  scripts/t2a/train/run_sceneplan_p11.sh screening
RUN_NAME=<unique-name> P11_ARM=canonical P11_SEED=42 GPU_IDS=0 \
  scripts/t2a/train/run_sceneplan_p11.sh screening

# Unified immutable-challenge evaluation (repeat with direct and d0)
.venv/bin/python scripts/t2a/eval/evaluate_sceneplan_p11_v4_challenge.py \
  --arm flow --checkpoint /absolute/path/to/epoch-step.ckpt \
  --weights ema --draws 1 --k-values 1 --rows-per-view 3 \
  --qwen-kernel-mode fast_fixed_bv32_w2_s2 \
  --device cuda:0 --output /absolute/path/to/report.json

# Historical-only two-seed replication summary; canonical screening does not
# schedule additional training seeds.
.venv/bin/python scripts/t2a/eval/summarize_sceneplan_p11_v4_seed_replication.py \
  --run <training-seed-a> /absolute/path/to/eval-a.json /absolute/path/to/train-a.log \
  --run <training-seed-b> /absolute/path/to/eval-b.json /absolute/path/to/train-b.log \
  --output /absolute/path/to/two-seed-summary.json

# Fail-closed matched D0/Direct/Flow summary
.venv/bin/python scripts/t2a/eval/summarize_sceneplan_p11_v4_matched_pilot.py \
  --flow-report /absolute/path/to/flow.json --flow-log /absolute/path/to/flow.log \
  --direct-report /absolute/path/to/direct.json --direct-log /absolute/path/to/direct.log \
  --d0-report /absolute/path/to/d0.json --d0-log /absolute/path/to/d0.log \
  --output /absolute/path/to/matched-summary.json
```

The old `trial` alias is retired. `screening` is the sole matched 10k-step
profile and is hard-locked to one GPU, batch 8, seed 42, 10k steps, canonical
configs, and one terminal checkpoint. `full` remains fail-closed until held-out,
intervention, and frozen-P10 closure gates pass. No eight-GPU or full-corpus
run is implied by a successful screen.

## Acceptance boundary

A pilot must have finite connected gradients, optimizer/EMA advancement,
strict batch 8, declining G/U/E discrete and continuous objectives, 100%
SceneSketch/DeltaSketch parse and round-trip, 100% final ScenePlan/Patch parse,
no truncation/non-finite values, and deterministic repeated decoding.

For a continuous arm, also require:

- zero/shuffle/single-slot interventions are measured;
- numeric interventions preserve every semantic field;
- semantic interventions preserve every numeric field;
- editing changes only the named owner and fields;
- the same-seed P10 closure is used for audio comparisons.

Pilot overfit establishes learnability, not superiority. The canonical
reliable-ASR assembler now passes a same-checkpoint causal on/drop test and two
independent 450-step seeds. Promotion still requires a matched held-out D0 vs
Direct-MSE vs Flow comparison with the same lexical authority, data,
initialization policy, batch, budget, and evaluator.

The completed 1,000-step historical matched pilot is recorded at
`artifacts/sceneplan_p11/p11_v4_matched_pilot_comparison_20260901.json`.
D0 learned the tiny panel most easily. Subsequent same-initial-state and
same-noise diagnostics showed that the old Flow-E route did not causally
separate opposite edit instructions, so those checkpoints are not promotion
evidence. The current operation-specific DeltaSketch direction route passes
its strict held-in gate, and fixed FLA evaluation is exactly reproducible
across independent processes. `full` remains locked until the newly matched
D0/Direct-MSE/Flow-R1 held-out comparison.

The current reliable-ASR assembler-v2 evidence is:

- same-checkpoint ASR on/drop: overall `+0.04403`, U `+0.11007`, all 8/8
  reliable-ASR U rows improve, while all 22 unaffected rows are bitwise equal;
- two independent seeds: held-out30 overall `0.70026` and `0.70071`, both with
  valid rate `1.0` and exact editing counterfactual rate `1.0`;
- descriptive two-seed mean `0.70049`; this is replication evidence, not a
  significance or full-training claim.

Machine-readable reports live at
`artifacts/sceneplan_p11/evals/p11_v4_reliableasr_assemblerv2_causal_ab_heldout30_evalv8_20260902.json`
and
`artifacts/sceneplan_p11/evals/p11_v4_reliableasr_assemblerv2_two_seed_replication_heldout30_evalv8_20260902.json`.

## Checkpoint and disk policy

Runs live under `/mnt/sdc/ckpts/sceneplan_p11/<run>`. The launcher refuses to
overwrite a non-empty fresh-run directory, checks free space, keeps at most one
numbered checkpoint plus `last.ckpt`, and supports full-state resume through
`CKPT_PATH`. Numbered and `last` copies may be collapsed only when their full
serialized contents or audited trainer states are equivalent.

Retain the canonical final checkpoint, one meaningful learning-curve point,
and reports needed to reproduce a decision. Delete failed/superseded
intermediates only after confirming that no live process, config, release, or
evaluation references them.

The versioned architecture contract is
`docs/sceneplan_v2/p11_v4_unified_transfusion_cot_contract_20260901.md`.
