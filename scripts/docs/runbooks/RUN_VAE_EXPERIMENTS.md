# VAE 优中选优 runbook (Stage-1)

4 candidate configs, **2 GPUs each** (8 GPUs total), same fixed data, same
schedule -> whichever reconstructs best wins. Run from
`cd /home/tanhe/dataset_storage/stable-audio-tools` with `uv`.

## 0. The 4 candidates (clean 2x2 grid)

| tag | config file | downsampling | latent z | DiT seq | warm-start | GPUs |
|---|---|---|---|---|---|---|
| `ds2048_z64`  | `stable_audio_4ch_vae.json`             | 2048 | 64  | 1x | full    | 0,1 |
| `ds1024_z64`  | `stable_audio_4ch_vae_ds1024.json`      | 1024 | 64  | 2x | full    | 2,3 |
| `ds2048_z128` | `stable_audio_4ch_vae_z128.json`        | 2048 | 128 | 1x | partial | 4,5 |
| `ds1024_z128` | `stable_audio_4ch_vae_ds1024_z128.json` | 1024 | 128 | 2x | partial | 6,7 |

(Removed `z96` = redundant midpoint and `z32` = least promising / narrowest latent.)
Axes: rows = time compression (2048 vs 1024), cols = latent width (64 vs 128).

## 1. Fixed comparison data

`stable_audio_tools/configs/dataset_configs/local_4ch_vae_compare.json`
= SLS (221k FOA) + AudioCaps-FOA (45k FOA) + MRSDrama capped to 3k (binaural,
kept minimal via `max_files`). Identical for all 4 -> fair comparison.

## 2. Launch all 4 (2 GPUs each, save@20k, stop@160k) — tmux, no nohup

Per-config batch (48 GB cards): **`ds2048_*` = 8**, **`ds1024_*` = 4** (the
downsampling-1024 configs have 2x latent frames -> OOM at 8). batch 16 OOMs for
all (encodec discriminator + 7-scale MR-STFT are heavy). `expandable_segments`
reduces fragmentation. Note: effective batch differs (16 vs 8) -> fine for a
reconstruction-quality comparison; set all to 4 if you want it identical.

```bash
cd /home/tanhe/dataset_storage/stable-audio-tools
uv sync --extra train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P2=/mnt/sdc/ckpts/stable-audio-open-1.0/model.safetensors
DATA=stable_audio_tools/configs/dataset_configs/local_4ch_vae_compare.json
A=stable_audio_tools/configs/model_configs/autoencoders
mkdir -p /mnt/sdc/ckpts/vae_exp

tmux new-session -d -s vae_ds2048_z64 "CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python train_4ch.py --model-config $A/stable_audio_4ch_vae.json --dataset-config $DATA --pretrained-ckpt-2ch $P2 --name vae_ds2048_z64 --num-gpus 2 --batch-size 8 --precision bf16-mixed --save-dir /mnt/sdc/ckpts/vae_exp/ds2048_z64 --checkpoint-every 20000 --max-steps 160000 2>&1 | tee /mnt/sdc/ckpts/vae_exp/ds2048_z64.log"
tmux new-session -d -s vae_ds1024_z64 "CUDA_VISIBLE_DEVICES=2,3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python train_4ch.py --model-config $A/stable_audio_4ch_vae_ds1024.json --dataset-config $DATA --pretrained-ckpt-2ch $P2 --name vae_ds1024_z64 --num-gpus 2 --batch-size 4 --precision bf16-mixed --save-dir /mnt/sdc/ckpts/vae_exp/ds1024_z64 --checkpoint-every 20000 --max-steps 160000 2>&1 | tee /mnt/sdc/ckpts/vae_exp/ds1024_z64.log"
tmux new-session -d -s vae_ds2048_z128 "CUDA_VISIBLE_DEVICES=4,5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python train_4ch.py --model-config $A/stable_audio_4ch_vae_z128.json --dataset-config $DATA --pretrained-ckpt-2ch $P2 --name vae_ds2048_z128 --num-gpus 2 --batch-size 8 --precision bf16-mixed --save-dir /mnt/sdc/ckpts/vae_exp/ds2048_z128 --checkpoint-every 20000 --max-steps 160000 2>&1 | tee /mnt/sdc/ckpts/vae_exp/ds2048_z128.log"
tmux new-session -d -s vae_ds1024_z128 "CUDA_VISIBLE_DEVICES=6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python train_4ch.py --model-config $A/stable_audio_4ch_vae_ds1024_z128.json --dataset-config $DATA --pretrained-ckpt-2ch $P2 --name vae_ds1024_z128 --num-gpus 2 --batch-size 4 --precision bf16-mixed --save-dir /mnt/sdc/ckpts/vae_exp/ds1024_z128 --checkpoint-every 20000 --max-steps 160000 2>&1 | tee /mnt/sdc/ckpts/vae_exp/ds1024_z128.log"
```

Watch live:  `tmux attach -t vae_ds2048_z128`  (detach: Ctrl-b then d)
Sessions:    `tmux ls` ; logs also at `/mnt/sdc/ckpts/vae_exp/<tag>.log`
Stop one:    `tmux kill-session -t vae_<tag>` ; stop all: `tmux kill-server`
Resume after a stop: add `--ckpt-path /mnt/sdc/ckpts/vae_exp/<tag>/<newest>.ckpt`
(warm-start auto-skipped on resume). If a `ds1024_*` still OOMs, drop `--batch-size` to 6/4.

## 3. Unwrap at the final step (160k)

```bash
declare -A CFG=(
  [ds2048_z64]=stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json
  [ds1024_z64]=stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024.json
  [ds2048_z128]=stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_z128.json
  [ds1024_z128]=stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae_ds1024_z128.json )
for TAG in ds2048_z64 ds1024_z64 ds2048_z128 ds1024_z128; do
  CKPT=$(ls -t /mnt/sdc/ckpts/vae_exp/$TAG/*.ckpt | head -1)
  uv run python unwrap_model.py --model-config "${CFG[$TAG]}" --ckpt-path "$CKPT" --name "unwrapped_$TAG"
  mv "unwrapped_$TAG.ckpt" /mnt/sdc/ckpts/vae_exp/$TAG/
done
```

## 4. Reconstruct a fixed eval set + score each

```bash
EVALDATA=stable_audio_tools/configs/dataset_configs/local_4ch_preencode.json
for TAG in ds2048_z64 ds1024_z64 ds2048_z128 ds1024_z128; do
  UNW=/mnt/sdc/ckpts/vae_exp/$TAG/unwrapped_$TAG.ckpt
  LAT=/mnt/sdc/audio_latents/eval_$TAG
  CUDA_VISIBLE_DEVICES=0 uv run python pre_encode_4ch.py \
    --model-config "${CFG[$TAG]}" --ckpt-path "$UNW" \
    --dataset-config "$EVALDATA" --output-path "$LAT" \
    --no-pad --batch-size 1 --num-workers 8 --limit-batches 60
  uv run python scripts/vae/eval/decode_latents_4ch.py --latent-root "$LAT" \
    --output-dir /mnt/sdc/vae_4ch_train_result_audio/$TAG --num-samples 20
  uv run python dataset/evaluation/eval_vae_recon.py \
    --recon-dir /mnt/sdc/vae_4ch_train_result_audio/$TAG \
    --tag "$TAG" --output-base /mnt/sdc/vae_4ch_train_result_audio
done
for TAG in ds2048_z64 ds1024_z64 ds2048_z128 ds1024_z128; do
  echo "== $TAG =="; cat /mnt/sdc/vae_4ch_train_result_audio/$TAG/summary.md; done
```

Pick by **lsd_db ↓** and **doa_az/el_err_deg ↓**, preferring the cheaper
`downsampling`/`z` when tied. Expected ≈ `ds1024_z128 ≥ ds2048_z128 ≥ ds1024_z64 ≥ ds2048_z64`.

## 5. Lock winner -> full pre-encode -> Stage-2 DiT

```bash
WIN=ds2048_z128; WCFG=${CFG[$WIN]}; WUNW=/mnt/sdc/ckpts/vae_exp/$WIN/unwrapped_$WIN.ckpt
uv run python pre_encode_4ch.py --model-config "$WCFG" --ckpt-path "$WUNW" \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_preencode.json \
  --output-path /mnt/sdc/audio_latents/stage1_$WIN --no-pad --batch-size 1 --num-workers 8
# DiT: point local_4ch_t2a_preencoded.json at stage1_$WIN + --pretransform-ckpt-path "$WUNW"
```

## (LATER) Spatial dataset construction -> sdd (NOT run yet)

```bash
uv run python dataset/indexing/build_source_index.py     --out /mnt/sdd/audio_dataset/spatial_sources ...
uv run python dataset/synthesis/build_spatial_dataset.py --sources-dir /mnt/sdd/audio_dataset/spatial_sources \
    --out-dir /mnt/sdd/audio_dataset/spatial_foa/clips --manifest /mnt/sdd/audio_dataset/spatial_foa/manifest.jsonl ...
uv run python dataset/captioning/refine_caption.py       --manifest /mnt/sdd/audio_dataset/spatial_foa/manifest.jsonl --out /mnt/sdd/audio_dataset/spatial_foa/captions.jsonl --num_gpus 8
```
After the spatial corpus exists, add its dirs (+ captions.jsonl) to
`local_4ch_vae.json` / `local_4ch_preencode.json`, and point
`local_4ch_t2a_preencoded.json` at the new latent dir.
```
