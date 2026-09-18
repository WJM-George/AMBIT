# Training

All commands assume the repository root and that `AMBIT_DATA_ROOT` / `AMBIT_CKPT_ROOT` are set. Launchers read GPU lists from `CUDA_VISIBLE_DEVICES`; they do not pin a laboratory device map.

Train in this order: FOA VAE (frozen afterwards) → generation DiT → generation AR → editing CLAP → editing DiT → editing AR. The root README has the copy-paste commands for each stage.

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

Method: [`docs/OPSD.md`](OPSD.md). Current run: [`docs/TRANSFUSION_OPSD_MAINLINE.md`](TRANSFUSION_OPSD_MAINLINE.md). Library: `stable_audio_tools/training/transfusion_opsd/`. Mainline learner: `scripts/t2a/rl/train_editing_opsd_repaired_fresh.py`.

Current Editing recipe, after the joint 40k checkpoint:

- request side sees only source FOA + instruction (no request-side GT in rollout)
- STE / discrete credit into AR logits is off
- 16 requests + 512 paired rows; 8-GPU split is 2+64 with paired microbatch 48
- shared Transformer and DiT at `3.752567682220472e-7`; AR / structure heads at `5e-6`
- exclusive per-branch execution-or-GT fallback; frozen-40k reference hold; same-plan RF teachers; paired FOA auxiliary
- resume the existing v3 state through 2000; keep 500/1000/1500/2000 and validation Top 5

```bash
python scripts/t2a/rl/launch_editing_opsd_repaired_fresh.py \
  --run-dir "$AMBIT_CKPT_ROOT/transfusion_opsd/editing_v3" \
  --recover
```

Set `CUDA_VISIBLE_DEVICES` to eight devices. Development-panel scores do not replace the released 40k baseline.
