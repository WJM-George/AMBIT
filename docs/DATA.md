# Data

AMBIT does not ship audio, latents, or indexes. Set `AMBIT_DATA_ROOT` to the directory that will hold them.

Training targets are **ScenePlan-simulated FOA**, not raw downloads. Public corpora supply mono speech/music/sound assets (and extra FOA for the VAE). The construction pipeline is: download → index → caption → sample ScenePlan → pyroomacoustics render → write generation requests / edit pairs. The paper README in the repository root lists the corpora and splits.

## Public source download

From the repository root:

```bash
source data_download/scripts/env_audio_dataset.sh
uv run python data_download/scripts/download_dataset.py audiocaps
```

`data_download/scripts/download_all.py` walks the catalog. Hugging Face gated sets (for example Sphere360) need `huggingface-cli login` and license acceptance on the Hub.

Downloader outputs land under `$AUDIO_DATASET_ROOT/datasets/<name>/`. Logs and manifests go to `$AUDIO_DATASET_ROOT/logs` and `$AUDIO_DATASET_ROOT/manifests`.

See `data_download/README.md` and `data_download/scripts/RUN_DOWNLOADS.md`.

## ScenePlan and FOA construction

| Stage | Location |
| --- | --- |
| Source indexes | `dataset/indexing/` |
| Captions | `dataset/captioning/` |
| FOA synthesis (ACN / SN3D) | `dataset/synthesis/` |
| ScenePlan / editing pairs | `scripts/t2a/data/` |

Configs under `stable_audio_tools/configs/dataset_configs/` refer to indexes with `${AMBIT_DATA_ROOT}/...`. Expand those placeholders by exporting the environment variables before launching training.

## Expected splits

These are the paper partitions, not files in this repository:

- **Generation:** 1.6M train / 32k validation / 8k test scenes
- **Editing:** 1.0M train / 20k validation / 5k test pairs

Keep test sets frozen. Do not tune on the official test split.
