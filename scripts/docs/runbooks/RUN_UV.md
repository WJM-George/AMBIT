# stable-audio-tools — uv 运行手册

**不要** `conda activate` + `source .venv`。只用 `uv run` / `uv pip`。

```bash
cd /home/tanhe/dataset_storage/stable-audio-tools
uv sync --extra train
```

FOA 合成（pyroomacoustics，从 mono jsonl / AudioCaps 造 4ch）：

```bash
uv sync --extra spatial
uv run python dataset/synthesis/synthesize_foa_pyroom.py --help
```

可选 Flash Attention（编译较慢）：

```bash
uv pip install packaging ninja
uv pip install flash-attn --no-build-isolation
```

---

## Stage-1 是否正确？

| 脚本 | 作用 | 是否 Stage-1 |
|------|------|----------------|
| `train_4ch.py` | 框架 GAN + EMA + 2ch→4ch warm-start + `local_4ch_example.json` | **是，正式训练** |
| `overfit_sanity_4ch.py` | 固定小 batch 过拟合，无判别器 | **否，仅冒烟** |

数据现状（本机）：Spatial LibriSpeech ~90k `.flac`，MRSDrama ~37k `.wav`。预训练 2ch 权重：`/mnt/sdc/ckpts/stable-audio-open-1.0/model.safetensors`。

---

## 1) SANITY 冒烟（先跑，~50 step）

```bash
cd /home/tanhe/dataset_storage/stable-audio-tools

uv run python -m stable_audio_tools.training.overfit_sanity_4ch \
  --config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
  --pretrained-ckpt /mnt/sdc/ckpts/stable-audio-open-1.0/model.safetensors \
  --sls-root /mnt/sdb/audio_dataset/datasets/spatial_librispeech \
  --mrsdrama-root /mnt/sdd/audio_dataset/datasets/mrsdrama/snapshot \
  --steps 50 --batch-size 4 --out-dir /mnt/sdc/vae_4ch_sanity_out

uv run python -m stable_audio_tools.training.overfit_sanity_4ch \
  --config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
  --pretrained-ckpt /mnt/sdc/ckpts/stable-audio-open-1.0/model.safetensors \
  --sls-root /mnt/sdb/audio_dataset/datasets/spatial_librispeech \
  --mrsdrama-root /mnt/sdd/audio_dataset/datasets/mrsdrama/snapshot \
  --steps 1000 --log-every 50 --save-every 250 \
  --batch-size 4 --out-dir /mnt/sdc/vae_4ch_sanity_out
```

每次保存会写入子目录 `step_XXXXXX/`（例如 `step_000250/`、`step_001000/`），内含 `{clip}_in.wav` 与 `{clip}_rec.wav`。

---

## 2) 正式 Stage-1 训练（GAN）

```bash
cd /home/tanhe/dataset_storage/stable-audio-tools

uv run python train_4ch.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_example.json \
  --pretrained-ckpt-2ch /mnt/sdc/ckpts/stable-audio-open-1.0/model.safetensors \
  --name vae_4ch_stage1 \
  --batch-size 8 \
  --num-gpus 1 \
  --precision bf16-mixed \
  --save-dir /mnt/sdc/ckpts/vae_4ch \
  --checkpoint-every 5000

uv run python train_4ch.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_example.json \
  --pretrained-ckpt-2ch /mnt/sdc/ckpts/stable-audio-open-1.0/model.safetensors \
  --name vae_4ch_stage1 \
  --batch-size 8 \
  --num-gpus 8 \
  --precision bf16-mixed \
  --save-dir /mnt/sdc/ckpts/vae_4ch \
  --checkpoint-every 5000
```

W&B 日志（可选）：

```bash
uv run python train_4ch.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_example.json \
  --pretrained-ckpt-2ch /mnt/sdc/ckpts/stable-audio-open-1.0/model.safetensors \
  --name vae_4ch_stage1 \
  --batch-size 8 --num-gpus 1 --precision bf16-mixed \
  --save-dir /mnt/sdc/ckpts/vae_4ch --checkpoint-every 5000 \
  --logger wandb
```

从 checkpoint 恢复：

```bash
uv run python train_4ch.py \
  ...同上参数... \
  --ckpt-path /mnt/sdc/ckpts/vae_4ch/last.ckpt
```

---

## 2b) Stage-1 结束后：unwrap + pre-encode（4ch 处理）

等 `train_4ch.py` 产出 checkpoint 后再跑。详见
`scripts/docs/pipeline/WORKFLOW.md`。

```bash
cd /home/tanhe/dataset_storage/stable-audio-tools

# 将 XXXXX 换成实际 step，例如 5000
uv run python unwrap_model.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
  --ckpt-path /mnt/sdc/ckpts/vae_4ch/epoch=0-step=5000.ckpt \
  --name unwrapped_4ch_vae

mv unwrapped_4ch_vae.ckpt /mnt/sdc/ckpts/vae_4ch/

# 全库（8 卡，不要加 limit；约 26.8 万条，数小时量级）
uv run python pre_encode_4ch.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
  --ckpt-path /mnt/sdc/ckpts/vae_4ch/unwrapped_4ch_vae_step_50000.ckpt \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_preencode.json \
  --output-path /mnt/sdc/audio_latents/stage1_vae_4ch \
  --no-pad --batch-size 1 --num-workers 8

# Trial：检查 latent 格式 / VAE 是否正常（单卡即可；进度条总数应等于 N）
CUDA_VISIBLE_DEVICES=0 uv run python pre_encode_4ch.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
  --ckpt-path /mnt/sdc/ckpts/vae_4ch/unwrapped_4ch_vae_step_50000.ckpt \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_preencode.json \
  --output-path /mnt/sdc/audio_latents/stage1_vae_4ch_trial \
  --no-pad --batch-size 1 --num-workers 8 \
  --limit-batches 50
```

`--limit-batches` 映射到 Lightning 的 `limit_val_batches`（**每个 GPU rank** 最多 N 个 batch）。Trial 用独立 `--output-path`，避免和全库混写。听感验证仍优先用 `vae_4ch_sanity_out` 的 `*_rec.wav`；pre-encode trial 主要看 `[64,T]` `.npy` + `.json` 是否齐全。

---

## 2c) Stage-2：Text → Spatial DiT（M0：dense DiT + rectified flow）

跑在 **pre-encode 产出的 latent** 上（不再碰原始音频）。`--pretransform-ckpt-path` 指向 unwrapped 4ch VAE，仅用于 demo 解码。

模型配置：`stable_audio_4ch_text2spatial.json`（已补全 training/optimizer/demo，`spatial_format.output_dim=768`，1.2B 参数，已 CPU 验证可建模）。

> **train.py 没有 `--num-gpus`**（用 prefigure + `defaults.ini`）。Trainer `devices="auto"` 自动用满所有可见 GPU；用 `CUDA_VISIBLE_DEVICES` 控制用哪几张。加 `--num-gpus` 会报 `unrecognized arguments`。

```bash
cd /home/tanhe/dataset_storage/stable-audio-tools

# Smoke（单卡，先验证全链路；CUDA_VISIBLE_DEVICES 限定 1 张卡）
CUDA_VISIBLE_DEVICES=0 uv run python train.py \
  --model-config stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/t5_1b.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_t2a_preencoded.json \
  --pretransform-ckpt-path /mnt/sdc/ckpts/vae_4ch/unwrapped_4ch_vae_step_120000.ckpt \
  --name dit_4ch_smoke --batch-size 8 --precision bf16-mixed \
  --save-dir /mnt/sdc/ckpts/dit_4ch_smoke --checkpoint-every 1000

# 全量（8 卡：不指定 CUDA_VISIBLE_DEVICES 即用全部）
uv run python train.py \
  --model-config stable_audio_tools/configs/model_configs/txt2audio/t2a/dit/t5_1b.json \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_t2a_preencoded.json \
  --pretransform-ckpt-path /mnt/sdc/ckpts/vae_4ch/unwrapped_4ch_vae_step_120000.ckpt \
  --name dit_4ch_text2spatial_m0 \
  --batch-size 8 --precision bf16-mixed \
  --save-dir /mnt/sdc/ckpts/dit_4ch_text2spatial --checkpoint-every 5000
```

注意：

- `prompt` 已填好（SLS/MRSDrama/AudioCaps 全带 prompt，pre-encode 已烤入）→ M0 即为真·text→spatial。
- `logger` 默认 `wandb`（需已 `wandb login`）；不想用就加 `--logger none`（checkpoint 直接存 `save-dir`，wandb 则存 `save-dir/<project>/<run_id>/checkpoints`）。
- `spatial_format` 用 `foa` / `binaural` 区分声场；DiT 同一套，靠数据 + prompt 分语音/环境音。
- MoE、3D RoPE、视频 cross-attn 仍按计划延后到 M0 跑通之后。

---

## 2d) 文本条件：captioning + caption 注入

`dataset_4ch` 现在支持给每个数据集挂 `captions`（jsonl/json），pre-encode 时把 caption 烤进 latent 的 `prompt`/`text`。匹配按文件 stem（自动去掉 `_WYZX_4ch`/`_LR00_4ch`/`_4ch`）或 jsonl 里的 `id`/`*_path`。

数据集文本来源（当前三个数据集都**自带文本**，只有 Sphere360 需要 captioner）：

| 数据 | 文本来源 | captions 文件 | 要 captioner |
|------|----------|---------------|--------------|
| **AudioCaps-FOA** | 合成 manifest（原生 caption）| `audiocaps_foa/train_manifest.jsonl` | 否 |
| **Spatial LibriSpeech** | `metadata.parquet`（语音方位/距离/混响 + 台词）| `spatial_librispeech/sls_prompts.jsonl` | 否 |
| **MRSDrama** | 每场景 `data.json`（英文 textual/scene prompt）| `mrsdrama/mrsdrama_prompts.jsonl` | 否 |
| **Sphere360 / 无文本音频** | `caption_audio.py` 生成 | — | 是 |

从元数据生成 SLS / MRSDrama 的 prompt（CPU，秒级；已生成，改了模板可重跑）：

```bash
uv run python dataset/indexing/build_spatial_prompts.py sls \
  --out /mnt/sdb/audio_dataset/datasets/spatial_librispeech/sls_prompts.jsonl
uv run python dataset/indexing/build_spatial_prompts.py mrsdrama \
  --out /mnt/sdd/audio_dataset/datasets/mrsdrama/mrsdrama_prompts.jsonl
# SLS 可加 --no-text 去掉台词；MRSDrama 可加 --include-text 附中文台词
```

三个 captions 已挂进 `local_4ch_preencode.json` 的对应 `captions` 字段；重跑 pre-encode 即把 prompt 烤进 latent。

Sphere360 等无文本音频用 captioner（在 `.venv-qwen` 跑，需 transformers-from-source + qwen-omni-utils）：

```bash
# 纯音频（轻量，Qwen3-Omni-Captioner，无 prompt 无视频）
python dataset/captioning/caption_audio.py \
  --audio-dir /mnt/sdc/audio_dataset_tmp/audiocaps_foa/train \
  --out /mnt/sdc/audio_dataset_tmp/audiocaps_foa/audio_captions.jsonl

# 视频/AV（Sphere360，Qwen3-Omni-Instruct，可用 360 视频）
python dataset/captioning/caption_sphere360.py --split test --mode av
```

AudioCaps-FOA 已自带 caption，**不用再 captioner**；pre-encode 配置已直接指向其 manifest。

重跑 pre-encode（已含 `audiocaps_foa` + caption 注入）：

```bash
# 改了数据集组成 -> 先清空旧 latent 目录，避免 rank/batch 命名错位混写
rm -rf /mnt/sdc/audio_latents/stage1_vae_4ch

uv run python pre_encode_4ch.py \
  --model-config stable_audio_tools/configs/model_configs/autoencoders/stable_audio_4ch_vae.json \
  --ckpt-path /mnt/sdc/ckpts/vae_4ch/unwrapped_4ch_vae_step_120000.ckpt \
  --dataset-config stable_audio_tools/configs/dataset_configs/local_4ch_preencode.json \
  --output-path /mnt/sdc/audio_latents/stage1_vae_4ch \
  --no-pad --batch-size 1 --num-workers 16
```

---

## 3) Sphere360 下载（在 dataset_storage 根目录，非 uv）

```bash
cd /home/tanhe/dataset_storage
source scripts/env_audio_dataset.sh

python3 scripts/downloaders/download_sphere360_media.py --split test --jobs 8

python3 scripts/downloaders/download_sphere360_media.py --split train --start-index 0 --end-index 5000 --jobs 8
python3 scripts/downloaders/download_sphere360_media.py --split train --start-index 5000 --end-index 10000 --jobs 8
# 继续增大 start-index / end-index 直到 grouped 列表末尾
```

需要 cookies 时加：`--cookie /path/to/cookies.txt`

---

## 多模态视频/文本条件模块（已迁入 stable-audio-tools）

已接入工厂注册：

- `stable_audio_tools/models/video_conditioners.py` — `clip`, `clip-with-sync-w-empty-feat`, `spatial_format`, **`videomae_v2`（运动流）**
- `stable_audio_tools/models/multimodal_adaptive_fusion.py` — MAF（`diffusion.json` 里 `gate` + `gate_type: MAF`）
- `stable_audio_tools/models/temporal_self_attention.py` — frame-level temporal transformer
- `stable_audio_tools/models/synchformer/` — Synchformer
- `stable_audio_tools/data/video_utils.py` — `read_video`, `build_video_condition_dict`
- DiT stub（未来）：`text2spatial` / `video2spatial` / `vt2spatial` 三个 json；**不是 T2VA**
- 总 workflow：`scripts/docs/pipeline/WORKFLOW.md`（含 Stage-1 → unwrap → pre-encode 命令）

详见 `scripts/docs/pipeline/PIPELINE_STATUS.md`。

```bash
uv run python -c "from stable_audio_tools.models.video_conditioners import CLIPConditioner, VideoMAEv2Conditioner; print('ok')"
```

### VideoMAE-v2 运动流（CLIP 语义 + MAE 运动 双流）

依赖已在 extra（现代 timm + decord；脚本直接调 builder，绕开 timm.create_model 的 `pretrained_cfg` 不兼容）：

```bash
uv sync --extra train --extra spatial      # timm==1.0.27 + decord
```

权重（HF `OpenGVLab/VideoMAE2`）：

```bash
# ViT-B（768维，轻而强，已下载）：/mnt/sdc/ckpts/videomae/distill/vit_b_k710_dl_from_giant.pth
# ViT-g（1408维，最强最重，可选）：
hf download OpenGVLab/VideoMAE2 mae-g/vit_g_hybrid_pt_1200e_k710_ft.pth --local-dir /mnt/sdc/ckpts/videomae
```

离线抽取运动特征（每 clip 一个 `[T_feat, C]` 的 `.npy`；需 GPU，pre-encode 跑完后再跑）：

```bash
uv run python dataset/features/extract_videomae_features.py \
  --video-dir /mnt/sdb/audio_dataset/datasets/sphere360/media/test \
  --out-dir   /mnt/sdc/audio_dataset_tmp/sphere360_videomae/test
# 默认 ViT-B（feat_dim=768）。ViT-g：--model vit_giant_patch14_224 --ckpt-path .../mae-g/vit_g_hybrid_pt_1200e_k710_ft.pth
```

双流配置（`cross_attention_cond_ids` 同列 CLIP + VideoMAE）：

```json
{"id": "video_clip",   "type": "clip",        "config": {}},
{"id": "video_motion", "type": "videomae_v2", "config": {"feat_dim": 768}}
```

> conditioner 只吃**预抽特征**（不在训练循环跑 backbone）；视频分支开训前还需把 `videomae_feats` 的 `.npy` 路径写进 latent 元数据（见 PIPELINE_STATUS 缺口）。

---

## 环境自检

```bash
cd /home/tanhe/dataset_storage/stable-audio-tools
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
which nvcc && nvcc --version
uv run python -c "import flash_attn; print('flash_attn ok')" 2>/dev/null || echo "flash_attn optional, not installed"
```
