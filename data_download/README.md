# Dataset download

Standalone downloaders for the public corpora used to build ScenePlan sources. They write into `$AUDIO_DATASET_ROOT` (defaults to `$AMBIT_DATA_ROOT`). They never write dataset files into this package directory.

```bash
source data_download/scripts/env_audio_dataset.sh
huggingface-cli login   # only for gated Hub datasets
uv run python data_download/scripts/download_dataset.py audiocaps
```

`download_all.py` iterates the catalog. Per-dataset scripts are in `data_download/scripts/downloaders/`.

Typical layout:

```text
$AUDIO_DATASET_ROOT/
  datasets/<name>/
  logs/
  manifests/
$AMBIT_CACHE_ROOT/
  huggingface/
  datasets/
  tmp/
```

Sphere360 downloads may need a cookies file at `$SPHERE360_COOKIE` if you use yt-dlp. Command details: `data_download/scripts/RUN_DOWNLOADS.md`.
