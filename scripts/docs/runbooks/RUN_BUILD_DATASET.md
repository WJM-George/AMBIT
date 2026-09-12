# Build the constructed spatial (FOA) dataset

Run this **after the VAE comparison training finishes** (it needs the GPUs free for
the LLM caption step, and it's independent of training anyway). All commands run
from the repo root with `uv`.

## Final composition (200k clips)

| Category | Count | How | Sources |
|----------|-------|-----|---------|
| audio | 80k | synth (pyroom) | audiocaps, audioset-audio, picoaudio, vggsound-audio |
| music | 60k | synth (pyroom) | musiccaps, audioset-music, vggsound instruments |
| speech | 60k | **real, direct** | Spatial LibriSpeech FOA used as-is (no mixing) |

= 140k synthesized FOA + 60k real SLS FOA. SLS is **not** synthesized/mixed; it is
folded in directly via the `construct_dataset/*` configs (`max_files: 60000`,
deterministic same subset for VAE and pre-encode). Dropped: mrsdrama / mrsaudio /
sphere360 / fsdkaggle.

## Commands

```bash
cd /home/tanhe/dataset_storage/stable-audio-tools
uv pip install pyarrow pyroomacoustics accelerate   # once (ffmpeg already present)

# A) vggsound subset: video -> mono wav (KEEPS the mp4; ~8 tarballs, multi-hour)
uv run python dataset/indexing/extract_vggsound.py \
  --snapshot /mnt/sdd/audio_dataset/datasets/vggsound/snapshot \
  --out      /mnt/sdd/audio_dataset/datasets/vggsound/extracted \
  --max-clips 80000 --sr 48000 --jobs 16

# 1) source index (no sls; +vggsound). audioset+vggsound routed by label.
uv run python dataset/indexing/build_source_index.py \
  --out   /mnt/sdd/audio_dataset/spatial_sources \
  --cache /mnt/sdd/audio_dataset/source_wav_cache \
  --datasets audiocaps,musiccaps,audioset,picoaudio,vggsound \
  --audioset-max 120000 --vggsound-max 80000

# 2) smoke test (100 clips) -- listen / eyeball before the full run
uv run python dataset/synthesis/build_spatial_dataset.py \
  --sources-dir /mnt/sdd/audio_dataset/spatial_sources \
  --out-dir     /mnt/sdd/audio_dataset/spatial_foa/audio \
  --manifest    /mnt/sdd/audio_dataset/spatial_foa/manifest.jsonl \
  --num 100 --jobs 8

# 3) full synth = 140k (80k audio + 60k music, NO synth speech), resumable; use tmux
uv run python dataset/synthesis/build_spatial_dataset.py \
  --sources-dir /mnt/sdd/audio_dataset/spatial_sources \
  --out-dir     /mnt/sdd/audio_dataset/spatial_foa/audio \
  --manifest    /mnt/sdd/audio_dataset/spatial_foa/manifest.jsonl \
  --audio 80000 --music 60000 --speech 0 \
  --jobs 32

# 4) captions for the 140k synth clips (LLM; GPUs are free post-training).
#    SLS brings its own prompts (sls_prompts.jsonl) at pre-encode, not here.
uv run python dataset/captioning/refine_caption.py \
  --manifest /mnt/sdd/audio_dataset/spatial_foa/manifest.jsonl \
  --out      /mnt/sdd/audio_dataset/spatial_foa/captions.jsonl \
  --num_gpus 8 --batch_size 16
```

## Notes

- **Resumable**: steps A, 3, 4 skip already-done items; just re-run on interruption.
- **LLM captions**: first run downloads `Qwen/Qwen2.5-7B-Instruct` (~15 GB). Multi-GPU
  shards (`captions.gpuN.jsonl`) are **auto-merged** into `captions.jsonl` at the end
  (no manual step needed). If the run is interrupted before finishing, just re-run the
  same command (resumable per shard); only if you must merge by hand:
  `cat captions.gpu*.jsonl >> captions.jsonl`.
  Set `CAPTION_LLM=/path/to/model` to use a local model. Use `--no-llm` for a
  GPU-free deterministic template instead.
- **Disk**: vggsound keeps mp4 (~136 GB) + audio (~16 GB); 200k FOA ≈ ~600 GB. All on
  `/mnt/sdd` (2.3 T free).

## Downstream (uses construct_dataset/* configs)

1. VAE train: `--dataset-config stable_audio_tools/configs/dataset_configs/construct_dataset/vae_4ch_construct.json`
2. Unwrap the chosen VAE, then pre-encode: `--dataset-config .../construct_dataset/preencode_4ch_construct.json`
   → latents at `/mnt/sdc/audio_latents/construct_4ch`
3. DiT train: `--dataset-config .../construct_dataset/t2a_preencoded_construct.json`
