# Experiment summaries

This directory is for concise, reproducible experiment handoffs. Raw checkpoints,
W&B runs, generated audio, and large metric tables belong on `${AMBIT_CKPT_ROOT}`, not in
the repository or root filesystem.

Each summary should record:

1. canonical model and dataset config paths;
2. exact checkpoint/global step and code revision;
3. per-GPU batch size, world size, precision, optimizer, LR, EMA, and seed;
4. dataset/READY entry count and artifact hashes where available;
5. evaluation command, prompt/test-set version, and output directory;
6. failures or deviations from the canonical configuration.

## Current T2A mainline

- Model: `stable_audio_tools/configs/model_configs/txt2audio/t2a/spatial_cot/`
  `qwen35_0p8b_spatial_chat_500m.json`
- Dataset: `stable_audio_tools/configs/dataset_configs/vae_v2_dataset/`
  `t2a_spatial_cot_1m_families.json`
- Architecture: one depth-24 batch-correct FlexAttention Transfusion jointly
  trains state planning, FOA understanding, and previous-audio-conditioned FOA
  rendering/editing; frozen Qwen3.5-0.8B supplies semantic prefixes.
- Run: 8 GPUs, BF16 mixed precision, fused AdamW at `5e-5`, EMA every optimizer
  step, and a family-exposure-normalized one-epoch budget after the gates pass.
- Launcher: `scripts/t2a/train/run_t2a_spatial_chat_500m_8gpu.sh`
- Gate runner: `scripts/t2a/train/run_spatial_cot_training_gate.sh`
- Runbook: `scripts/t2a/README.md`

Do not reuse a completed run name for a new ablation. Set `RUN_NAME` and
`RUN_ROOT` explicitly so the launcher cannot mix checkpoints from different
configurations.
