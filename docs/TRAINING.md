# Training

All commands assume the repository root and that `AMBIT_DATA_ROOT` / `AMBIT_CKPT_ROOT` are set. Launchers read GPU lists from `CUDA_VISIBLE_DEVICES`; they do not pin a laboratory device map.

## FOA VAE

```bash
python train_4ch.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024_z64_wdmix_scm.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_example.json \
  --name vae_4ch \
  --save-dir "$AMBIT_CKPT_ROOT/vae"
```

Ablation arms live in `stable_audio_tools/configs/model_configs/autoencoders/ablation_arms/`.

## Shared FOA renderer (DiT)

ScenePlan-conditioned rectified-flow training uses `train.py` with the DiT configs in `stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/` and the launchers in `scripts/t2a/train/run_sceneplan_dit_*.sh`.

## Generation AR

```bash
python scripts/t2a/train/train_sceneplan_transfusion_generation_ar.py \
  --mode full \
  --run-dir "$AMBIT_CKPT_ROOT/generation_ar" \
  --batch-size 2
```

`--mode tiny` is a short CPU/GPU smoke run.

## Editing CLAP, DiT, and AR

| Stage | Script |
| --- | --- |
| CLAP-FOA pretraining | `scripts/t2a/train/train_sceneplan_transfusion_editing_clap44.py` |
| Editing DiT | `scripts/t2a/train/run_sceneplan_transfusion_editing_dit_full_5gpu.sh` |
| Joint editing AR | `scripts/t2a/train/train_sceneplan_transfusion_editing_ar_joint.py` |

Pass model/dataset configs from `stable_audio_tools/configs/`. Checkpoint selection and audio evaluation are in `scripts/t2a/eval/`.

## OPSD

On-policy self-distillation code is under `stable_audio_tools/training/transfusion_opsd/`, with holdout preparation in `scripts/t2a/rl/`.
