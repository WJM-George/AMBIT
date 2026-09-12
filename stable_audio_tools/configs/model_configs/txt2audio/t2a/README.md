# Spatial T2A routes

Task variants remain config-selectable and share the model/data/training code.
The active sequence is P10 ScenePlan-DiT followed by the P11 single-turn
ScenePlan Transfusion route. Historical continuous-trajectory and Spatial-CoT
files are not launch targets.

## DiT baseline

`dit/qwen35_0p8b_300m.json` is the completed 320.7M dense DiT baseline: frozen
Qwen3.5-0.8B continuous text tokens, region/duration conditioning, 64-channel
FOA-VAE latents, rectified flow, BF16, and no demo callback. It consumes
`dataset_configs/vae_v2_dataset/t2a_preencoded_v2_construct_expansion.json`
and does not depend on ScenePlan or edit families.

## Continuous Transfusion history

Configs under `transfusion/` preserve the joint and causal continuous
trajectory experiments and their checkpoints. Their stable route id is
`continuous_traj`. They remain available for comparison and warm-starting but
are not the final training launch.

## Retired Spatial-CoT / spatial chat

The persistent four-turn Spatial-CoT route below is retained only as an
experiment record. It is not the P11 mainline and must not be used to create a
new run.

`spatial_cot/qwen35_0p8b_spatial_chat_500m.json` is the mainline. It applies
AudioChat's persistent full-state, turn-diff, previous-audio context, and
independently noised audio-span ideas in our metric-3D FOA system. One shared
depth-24 Transfusion has three objectives:

1. edit instruction + previous FOA/plan -> complete current ScenePlan CE;
2. current FOA + understanding request -> current ScenePlan CE;
3. semantic Qwen prefix + previous FOA + current plan/tracks -> FOA flow.

This route deliberately keeps two controls. `semantic_caption` carries the
event, timbre, and exact transcript. ScenePlan carries persistent source IDs,
source count, activity, gain, room, and metric 3-D trajectories. A deterministic
compiler turns the plan into four source slots × eight controls = `[32,T]`.

The family reader consumes
`dataset_configs/vae_v2_dataset/t2a_spatial_cot_{1m_families,10k_validation,2k_test}.json`.
Each example is a cumulative four-state conversation. Previous and target FOA
are independently noised under renderer training; source tracks stay clean.

The resolved final model has 505,101,472 trainable parameters plus frozen
Qwen3.5-0.8B. Older `plan_tracks` and 300M files are retained strictly for
compatibility; no new ablation launch is required.

## Canonical commands

```bash
scripts/t2a/data/prepare_spatial_cot_v1_artifacts.sh
scripts/t2a/data/run_spatial_cot_data.sh smoke64
scripts/t2a/data/run_spatial_cot_data.sh pilot2048

BENCHMARK=1 MAX_STEPS=4 BATCH_SIZE=1 \
  scripts/t2a/train/run_t2a_spatial_chat_500m_8gpu.sh
```

After the smoke, pilot, and full-depth optimizer-allocation gates pass, the
same data launcher builds `train1m`, `validation10k`, and `test2k`. The final
training launcher budgets by family exposure: one epoch is 1,000,000 family
exposures (125,000 optimizer steps at batch 1/GPU on eight GPUs), checkpoints
every 10,000 steps, and bounded validation every 10,000 steps by default.

The complete method and exact-pair/recoverability contract are in
`scripts/docs/design/SPATIAL_COT_AUDIOCHAT_METHOD.md`.

## Current P10 and P11 mainlines

P10 is `dit/qwen35_0p8b_300m_model_sceneplan_44.json`: semantic Qwen
cross-attention plus direct four-event/four-trajectory control.

P11 is
`sceneplan_p11/qwen35_0p8b_sceneplan_p11_reliable_asr_assembler_v2.json`, the
only supported Generation / Understanding / Editing candidate. It predicts a
semantic SceneSketch followed by a numeric executable state; a deterministic
assembler applies any confidence-gated input-only speech transcript, emits the
complete ScenePlan or atomic patch, and hands it to P10.

Use `scripts/t2a/train/run_sceneplan_p11.sh` for every P11 run. It fixes the
per-GPU batch at 8 and currently permits only `preflight` and `pilot`; `trial`
and `full` fail closed until causal and held-out gates pass. See
`sceneplan_p11/README.md` for the frozen data, evaluation, and promotion
contracts.
