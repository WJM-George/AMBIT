# Dataset downloads (`uv`)

All downloaders run from the **repo root** with `uv` (do not use bare `python3` — deps live in the project venv).

```bash
uv sync
```

## Disk layout

| Dataset | Root | Path |
|---------|-------|------|
| AudioSet (~2.4 TB) | `${AMBIT_DATA_ROOT}` | `${AMBIT_DATA_ROOT}/datasets/audioset/` |
| VGGSound (~338 GB) | `${AMBIT_DATA_ROOT}` | `${AMBIT_DATA_ROOT}/datasets/vggsound/` |
| MusicCaps (~3 MB) | `${AMBIT_DATA_ROOT}` | `${AMBIT_DATA_ROOT}/datasets/musiccaps/` |
| PicoAudio (~912 MB) | `${AMBIT_DATA_ROOT}` | `${AMBIT_DATA_ROOT}/datasets/picoaudio/` |
| FSDKaggle2019 (~27 GB) | `${AMBIT_DATA_ROOT}` | `${AMBIT_DATA_ROOT}/datasets/fsdkaggle2019/` |

Download AudioCaps and Spatial LibriSpeech with the same catalog entry points; they are not assumed to exist on disk.

## Commands

```bash
# FSDKaggle2019 (Zenodo, resumable)
uv run python scripts/downloaders/download_fsdkaggle2019.py
uv run python scripts/downloaders/download_fsdkaggle2019.py --extract

# Smoke: metadata + curated only
uv run python scripts/downloaders/download_fsdkaggle2019.py \
  --zenodo-file meta --zenodo-file curated

# Hugging Face snapshots
uv run python scripts/downloaders/download_audioset.py --max-workers 8
uv run python scripts/downloaders/download_vggsound.py --max-workers 8
uv run python scripts/downloaders/download_musiccaps.py      # CSV metadata only (~3 MB)
uv run python scripts/downloaders/download_picoaudio.py

# Generic entry (any catalog key)
uv run python scripts/download_dataset.py fsdkaggle2019
uv run python scripts/download_dataset.py audioset --max-workers 8
```

### MusicCaps audio (YouTube)

The HF repo `google/MusicCaps` ships **only the CSV** (YouTube IDs + 10s
stamps); the music is copyrighted and not redistributed. Fetch the actual wav
clips from YouTube with yt-dlp + ffmpeg (run the metadata step above first):

```bash
uv run python scripts/downloaders/download_musiccaps_audio.py --limit 20   # smoke test
uv run python scripts/downloaders/download_musiccaps_audio.py              # all 5,521 clips
```

Two requirements, both already satisfied on this box:

1. **Cookies** — YouTube blocks datacenter IPs (`Sign in to confirm you're not
   a bot`). Export browser cookies to a Netscape `cookies.txt`. The script
   auto-discovers `$MUSICCAPS_COOKIE` or a `youtube_cookies*.txt` under
   `$AUDIO_DATASET_TMP` (`${AMBIT_CACHE_ROOT}`), else pass `--cookies`.
2. **JS challenge solver** — YouTube's n-challenge needs **deno** on PATH
   (`~/.deno/bin`) *plus* yt-dlp's EJS component. The script handles both: it
   prepends `~/.deno/bin` and passes `--remote-components ejs:github` by default.
   Without these you get `Requested format is not available` (images only).

Clips land in `.../musiccaps/audio/<ytid>.wav`; reruns skip existing files and
write failures to `musiccaps_audio_fail_list.txt`.

```bash
# Cookies auto-discovered from ${AMBIT_CACHE_ROOT}/youtube_cookies_2.txt:
uv run python scripts/downloaders/download_musiccaps_audio.py --workers 4
# Or point at any cookies file explicitly:
uv run python scripts/downloaders/download_musiccaps_audio.py \
  --cookies ${AMBIT_CACHE_ROOT}/youtube_cookies_2.txt --workers 4
```

Extract FSDKaggle split noisy train (optional):

```bash
sudo apt-get install -y p7zip-full unzip
```

**Flaky connections:** large downloads (FSDKaggle ~27 GB on Zenodo) now
auto-resume on `ChunkedEncodingError` / `IncompleteRead` (10 retries with
backoff). If a run still dies, just re-run the same command — it continues from
the `.part` file.

HF cache defaults to `${AMBIT_CACHE_ROOT}` (see `audio_dataset_download/downloader.py`).
