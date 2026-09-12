# Audio Dataset Download Plan

This workspace contains download scripts for the requested spatial/audio datasets. The `audio_dataset_download/` directory is only Python source code; dataset files are not written there. The primary dataset root is:

```bash
/mnt/sdd/audio_dataset
```

The downloader also has a runtime guard that rejects `AUDIO_DATASET_ROOT` if it points inside the code package.

The scripts use all three disks:

```bash
/mnt/sdd/audio_dataset              # primary organized dataset files
/mnt/sdb/audio_dataset              # secondary organized dataset files
/mnt/sdc/audio_dataset_cache        # Hugging Face cache
/mnt/sdc/audio_dataset_tmp          # temporary download files
```

## 1. Setup Hugging Face

Run from this workspace:

```bash
cd /home/tanhe/dataset_storage
bash scripts/setup_huggingface.sh
source scripts/env_audio_dataset.sh
hf auth login
hf auth whoami
```

If `hf auth login` is not convenient, export a token instead:

```bash
export HF_TOKEN=hf_your_token_here
```

For gated datasets such as `omniaudio/Sphere360` and `HKUSTAudio/Audio-FLAN-Dataset`, accept access on the Hugging Face website first.

## 2. Output Layout

Each dataset is stored under:

```bash
/mnt/sdd/audio_dataset/datasets/<dataset_key>/
/mnt/sdb/audio_dataset/datasets/<dataset_key>/
```

By default, `bewo_1m`, `sphere360`, `audio_flan`, and `spatial_librispeech` go to `/mnt/sdb/audio_dataset`; the others go to `/mnt/sdd/audio_dataset`.

Examples:

```bash
/mnt/sdd/audio_dataset/datasets/mrsaudio/snapshot/
/mnt/sdd/audio_dataset/datasets/mrsaudio/parquet/default/train/
/mnt/sdb/audio_dataset/datasets/spatial_librispeech/metadata/metadata.parquet
/mnt/sdb/audio_dataset/datasets/spatial_librispeech/ambisonics/000000.flac
```

Logs and manifests are stored under:

```bash
/mnt/sdd/audio_dataset/logs/
/mnt/sdd/audio_dataset/manifests/
```

## 3. Recommended Download Strategy

Use full snapshot mode when you want the original repository files exactly as hosted:

```bash
python3 scripts/downloaders/download_mrsaudio.py
```

Use parquet mode when you want Hugging Face auto-converted files organized by config and split:

```bash
python3 scripts/downloaders/download_mrsaudio.py --mode hf_parquet
python3 scripts/downloaders/download_mrsaudio.py --mode hf_parquet --config default --split train
```

For very large datasets, start with parquet or metadata-only downloads first, then expand.

## 4. Per-Dataset Commands

MRSDrama:

```bash
python3 scripts/downloaders/download_mrsdrama.py
python3 scripts/downloaders/download_mrsdrama.py --mode hf_parquet --config meta --split train
```

BEWO-1M:

```bash
python3 scripts/downloaders/download_bewo_1m.py
```

MRSAudio:

```bash
python3 scripts/downloaders/download_mrsaudio.py
python3 scripts/downloaders/download_mrsaudio.py --mode hf_parquet --config default --split train
```

Sphere360:

```bash
python3 scripts/downloaders/download_sphere360.py
python3 scripts/downloaders/download_sphere360.py --mode hf_parquet --config default --split train
```

AudioX-IFcaps:

```bash
python3 scripts/downloaders/download_audiox_ifcaps.py
```

AudioCaps, included because it was in the supplied download notes:

```bash
python3 scripts/downloaders/download_audiocaps.py
python3 scripts/downloaders/download_audiocaps.py --mode hf_parquet --config default --split train
```

Audio-FLAN-Dataset:

```bash
python3 scripts/downloaders/download_audio_flan.py
```

Spatial LibriSpeech metadata only:

```bash
python3 scripts/downloaders/download_spatial_librispeech.py
```

Spatial LibriSpeech first 100 speech samples:

```bash
python3 scripts/downloaders/download_spatial_librispeech.py --sls-start 0 --sls-count 100
```

Spatial LibriSpeech speech plus noise:

```bash
python3 scripts/downloaders/download_spatial_librispeech.py --sls-start 0 --sls-count 100 --sls-include-noise
```

Spatial LibriSpeech full speech set:

```bash
python3 scripts/downloaders/download_spatial_librispeech.py --sls-all --sls-workers 16
```

Spatial LibriSpeech full speech plus noise set:

```bash
python3 scripts/downloaders/download_spatial_librispeech.py --sls-all --sls-include-noise --sls-workers 16
```

YT-Ambient / ViSAGe project repository:

```bash
python3 scripts/downloaders/download_yt_ambient.py
```

The provided YT-Ambient source is a GitHub repository, not a direct dataset file host. This script clones or updates the repository under `/mnt/sdd/audio_dataset/datasets/yt_ambient/repo`.

## 5. Download All

Conservative all-dataset run, skipping gated datasets and GitHub repo:

```bash
python3 scripts/download_all.py
```

Run multiple datasets in parallel:

```bash
python3 scripts/download_all.py --dataset-workers 3 --snapshot-workers 8
```

Include gated datasets after login/access approval:

```bash
python3 scripts/download_all.py --include-gated
```

Download Hugging Face parquet splits only:

```bash
python3 scripts/download_all.py --parquet-only --include-gated
```

Parallel parquet-only run:

```bash
python3 scripts/download_all.py --parquet-only --include-gated --dataset-workers 3 --snapshot-workers 8
```

Include GitHub project clone:

```bash
python3 scripts/download_all.py --include-gated --include-github
```

## 6. Useful Snapshot Filters

You can restrict large snapshot downloads with Hugging Face allow/ignore patterns:

```bash
python3 scripts/downloaders/download_mrsaudio.py --allow-pattern "*.json,*.jsonl,*.parquet,*.csv,*.txt,*.md"
python3 scripts/downloaders/download_mrsaudio.py --ignore-pattern "*.mp4,*.wav,*.flac"
```

Run again without filters when you are ready to pull the heavy media files.
