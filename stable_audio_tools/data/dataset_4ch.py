"""
Format-aware multi-channel (4ch) audio data path for the 4ch VAE.

This module is FRAMEWORK-NATIVE: the dataset yields the exact contract the stock
training path expects -- ``(audio[C, T], info_dict)`` per item, collated by the stock
``collation_fn`` into ``[reals[B, C, T], info_list]`` so it plugs straight into
``AutoencoderTrainingWrapper.training_step`` (which does ``reals, json = batch``).

Why a separate path instead of stock SampleDataset / create_dataloader_from_config?
  * It canonicalizes FOA / binaural / mono into fixed four-channel slots and emits
    the matching format id and channel mask.
  * It disables augmentations such as independent channel remapping that would
    destroy the inter-channel relationships carrying the spatial image.

We still REUSE the framework where it is safe:
  * PadCrop_Normalized_T  (crop/pad + timestamps + padding_mask; no amplitude change)
  * get_audio_filenames   (recursive file discovery, filelist.txt support)
  * collation_fn          (stock collation; identical batch structure)

Canonical [4, T] channel layout (fixed slots; downstream uses a format token + mask):
  * FOA (ambisonics, AmbiX/ACN/SN3D): native order kept -> [W, Y, Z, X]
  * binaural / stereo:                [L, R, 0, 0]
  * mono:                             [m, 0, 0, 0]

Normalization is JOINT across channels (single scalar gain) so inter-channel ratios -
the spatial cues - are preserved. Never per-channel normalize spatial audio.
"""

import os
import glob
import json
import errno
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchaudio

from .utils import PadCrop_Normalized_T
from .dataset import (
    _DatasetItemRetryExhausted,
    _FileDescriptorExhausted,
    _local_dataloader_kwargs,
    get_audio_filenames,
)


# Format ids (used for the format token downstream and for mask-aware analysis).
FOA = "foa"
BINAURAL = "binaural"
MONO = "mono"
FMT_ID = {FOA: 0, BINAURAL: 1, MONO: 2}

NUM_SLOTS = 4


def infer_format(num_channels: int) -> str:
    if num_channels >= 4:
        return FOA
    if num_channels == 2:
        return BINAURAL
    return MONO


def layout_to_4ch(wav: torch.Tensor, fmt: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place a [C, T] waveform into the canonical [4, T] layout. Returns (out, mask[4])."""
    C, T = wav.shape
    out = torch.zeros(NUM_SLOTS, T, dtype=wav.dtype)
    mask = torch.zeros(NUM_SLOTS, dtype=torch.float32)
    if fmt == FOA:
        k = min(NUM_SLOTS, C)
        out[:k] = wav[:k]
        mask[:k] = 1.0
    elif fmt == BINAURAL:
        out[0] = wav[0]
        out[1] = wav[1] if C > 1 else wav[0]
        mask[:2] = 1.0
    else:  # mono
        out[0] = wav[0]
        mask[0] = 1.0
    return out, mask


def joint_normalize(audio: torch.Tensor, mode: str = "joint_peak", peak: float = 0.9) -> torch.Tensor:
    """Single-scalar gain that preserves inter-channel relationships."""
    if mode is None or mode == "none":
        return audio
    if mode == "joint_peak":
        m = audio.abs().max()
        if m > 0:
            audio = audio * (peak / m)
        return audio
    if mode == "w_rms":
        rms = audio[0].pow(2).mean().sqrt()
        if rms > 0:
            audio = audio * (peak / (rms * 4.0))
        return audio
    raise ValueError(f"Unknown normalize mode: {mode}")


class FourChannelSampleDataset(torch.utils.data.Dataset):
    """
    Dataset over an explicit list of (path, format-or-None) items.

    __getitem__ returns the framework-native tuple ``(chunk[4, T], info)`` where info
    carries the stock keys (path/timestamps/seconds_*/padding_mask/sample_rate) plus the
    spatial extras ``fmt`` (int id) and ``channel_mask`` ([4] float).
    """

    def __init__(
        self,
        items: List[Tuple[str, Optional[str]]],
        sample_size: int = 65536,
        sample_rate: int = 44100,
        normalize: str = "joint_peak",
        peak: float = 0.9,
        random_crop: bool = True,
        pad: bool = True,
        captions: Optional[List[str]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        max_item_retries: int = 8,
    ):
        super().__init__()
        assert len(items) > 0, "FourChannelSampleDataset received an empty item list"
        self.items = items
        self.sample_size = sample_size
        self.sr = sample_rate
        self.normalize = normalize
        self.peak = peak
        # Optional per-item text prompt (aligned to items); baked into metadata so the
        # pre-encoded latent .json carries the real caption for Stage-2 text conditioning.
        if captions is not None:
            assert len(captions) == len(items), "captions must align 1:1 with items"
        self.captions = captions
        if metadata is not None:
            assert len(metadata) == len(items), "metadata must align 1:1 with items"
        self.metadata = metadata
        self.max_item_retries = int(max_item_retries)
        if self.max_item_retries < 0:
            raise ValueError("max_item_retries must be non-negative")
        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop, pad=pad)

    def __len__(self):
        return len(self.items)

    def _load(self, path: str) -> torch.Tensor:
        ext = path.split(".")[-1]
        wav, sr = torchaudio.load(path, format=ext)  # [C, T]
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    def _retry_item(self, retry_count: int, reason: str):
        if retry_count >= self.max_item_retries:
            raise _DatasetItemRetryExhausted(
                "failed to find a valid four-channel audio item after "
                f"{self.max_item_retries} retries; last reason: {reason}"
            )
        return self.__getitem__(
            random.randrange(len(self)), _retry_count=retry_count + 1
        )

    def __getitem__(self, idx, _retry_count=0):
        path, fmt = self.items[idx]
        try:
            wav = self._load(path)
            if fmt is None:
                fmt = infer_format(wav.shape[0])
            out, mask = layout_to_4ch(wav, fmt)
            chunk, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(out)
            chunk = joint_normalize(chunk, mode=self.normalize, peak=self.peak).clamp(-1.0, 1.0)
        except (_DatasetItemRetryExhausted, _FileDescriptorExhausted):
            raise
        except OSError as e:
            if e.errno in (errno.EMFILE, errno.ENFILE):
                raise _FileDescriptorExhausted(
                    "DataLoader exhausted file descriptors while loading "
                    f"{path}; raise RLIMIT_NOFILE or reduce "
                    "batch_size/workers/prefetch_factor"
                ) from e
            print(f"[FourChannelSampleDataset] failed on {path}: {e!r}; resampling")
            return self._retry_item(_retry_count, repr(e))
        except Exception as e:  # noqa: BLE001 - bounded skip of unreadable files
            print(f"[FourChannelSampleDataset] failed on {path}: {e!r}; resampling another item")
            return self._retry_item(_retry_count, repr(e))

        caption = self.captions[idx] if self.captions is not None else ""
        info = {
            "path": path,
            "timestamps": (t_start, t_end),
            "seconds_start": seconds_start,
            "seconds_total": seconds_total,
            "padding_mask": [padding_mask],
            "sample_rate": self.sr,
            "fmt": FMT_ID[fmt],
            "spatial_format": fmt,
            "channel_mask": mask,
            "prompt": caption,
            "text": caption,
        }
        if self.metadata is not None:
            reserved = {
                "path", "timestamps", "seconds_start", "seconds_total",
                "padding_mask", "sample_rate", "fmt", "spatial_format",
                "channel_mask", "audio",
            }
            for k, v in self.metadata[idx].items():
                if k not in reserved:
                    info[k] = v
        return chunk, info


# ---------------------------------------------------------------------------
# Caption maps: attach per-file text prompts (transcription / audio caption / AudioCaps)
# so the pre-encoded latent metadata carries the real Stage-2 text condition.
# ---------------------------------------------------------------------------

_CAPTION_SUFFIXES = ("_WYZX_4ch", "_LR00_4ch", "_4ch")


def _stem_aliases(name: str) -> List[str]:
    """Filename stem plus stem with known FOA/binaural suffixes stripped."""
    stem = os.path.splitext(os.path.basename(name))[0]
    out = [stem]
    for suf in _CAPTION_SUFFIXES:
        if stem.endswith(suf):
            out.append(stem[: -len(suf)])
    return out


def load_caption_map(spec: str) -> Dict[str, str]:
    """Load a caption lookup from a .jsonl (one row per clip) or .json (dict/list).

    Rows are indexed under several aliases (id / clip_id / sample_id and the stem of
    any *_path field, with FOA suffixes stripped) so audio files match regardless of
    naming (e.g. ``audiocaps_5_WYZX_4ch.flac`` -> ``audiocaps_5``).
    """
    cap: Dict[str, str] = {}
    p = str(spec)
    if not os.path.exists(p):
        print(f"[4ch loader] WARNING captions file not found: {p}")
        return cap

    def register(row: dict) -> None:
        text = row.get("caption") or row.get("text") or row.get("prompt") or ""
        if not isinstance(text, str) or not text.strip():
            return
        text = text.strip()
        for k in ("id", "clip_id", "sample_id"):
            v = row.get(k)
            if v is not None and str(v) != "":
                cap.setdefault(str(v), text)
        for k in ("foa_path", "path", "audio_path", "latent_filename"):
            v = row.get(k)
            if v:
                v = str(v)
                # Most specific first (unique): full path + abspath + basename.
                cap.setdefault(v, text)
                cap.setdefault(os.path.abspath(v), text)
                cap.setdefault(os.path.basename(v), text)
                # Stems are ambiguous across folders (e.g. segment_0.wav); register last.
                for alias in _stem_aliases(v):
                    cap.setdefault(alias, text)

    if p.endswith(".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    register(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        with open(p, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            for k, v in data.items():
                text = v if isinstance(v, str) else (v.get("caption") or v.get("text") or "")
                if text:
                    cap.setdefault(str(k), text.strip())
        elif isinstance(data, list):
            for row in data:
                register(row)
    print(f"[4ch loader] loaded {len(cap)} caption keys from {p}")
    return cap


def caption_for_file(path: str, cap_map: Dict[str, str]) -> str:
    if not cap_map:
        return ""
    # Specific keys first (unique per file), then ambiguous bare stems.
    for key in (path, os.path.abspath(path), os.path.basename(path)):
        if key in cap_map:
            return cap_map[key]
    for alias in _stem_aliases(path):
        if alias in cap_map:
            return cap_map[alias]
    return ""


def load_metadata_map(spec: str) -> Dict[str, Dict[str, Any]]:
    """Load per-audio metadata from JSONL/JSON and index it by path/id aliases."""
    meta: Dict[str, Dict[str, Any]] = {}
    p = str(spec)
    if not os.path.exists(p):
        print(f"[4ch loader] WARNING metadata file not found: {p}")
        return meta

    def register(row: dict) -> None:
        if not isinstance(row, dict):
            return
        aliases = []
        for k in ("audio_path", "foa_path", "path", "source_audio_path"):
            v = row.get(k)
            if v:
                v = str(v)
                aliases.extend([v, os.path.abspath(v), os.path.basename(v)])
                aliases.extend(_stem_aliases(v))
        for k in ("id", "clip_id", "sample_id"):
            v = row.get(k)
            if v is not None and str(v) != "":
                aliases.append(str(v))
        clean = {k: v for k, v in row.items() if v is not None}
        for a in aliases:
            meta.setdefault(str(a), clean)

    if p.endswith(".jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    register(json.loads(line))
                except json.JSONDecodeError:
                    continue
    else:
        with open(p, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            for row in data:
                register(row)
        elif isinstance(data, dict):
            for k, row in data.items():
                if isinstance(row, dict):
                    row.setdefault("id", k)
                    register(row)

    print(f"[4ch loader] loaded {len(meta)} metadata keys from {p}")
    return meta


def metadata_for_file(path: str, meta_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not meta_map:
        return {}
    for key in (path, os.path.abspath(path), os.path.basename(path)):
        if key in meta_map:
            return dict(meta_map[key])
    for alias in _stem_aliases(path):
        if alias in meta_map:
            return dict(meta_map[alias])
    return {}


def create_4ch_dataloader_from_config(
    dataset_config,
    batch_size,
    sample_size,
    sample_rate,
    num_workers: int = 4,
    shuffle: bool = True,
    pad: bool = True,
):
    """
    Mirror of create_dataloader_from_config for the 4-channel path.

    dataset_config schema (dataset_type must be "audio_dir_multichannel"):
      {
        "dataset_type": "audio_dir_multichannel",
        "random_crop": true,
        "normalize": "joint_peak",
        "datasets": [
          {"id": "spatial_librispeech", "path": ".../ambisonics", "format": "foa"},
          {"id": "mrsdrama",            "path": ".../snapshot",    "format": "binaural",
           "keywords": ["wav"]}
        ]
      }
    "format" may be omitted to infer per-file from channel count.
    """
    dataset_type = dataset_config.get("dataset_type", None)
    assert dataset_type in ("audio_dir_multichannel", "audio_dir_4ch"), \
        f"create_4ch_dataloader_from_config expects 'audio_dir_multichannel', got {dataset_type}"

    dir_configs = dataset_config.get("datasets", None)
    assert dir_configs is not None, "datasets must be specified in the dataset config"

    items: List[Tuple[str, Optional[str]]] = []
    captions: List[str] = []
    metadata_rows: List[Dict[str, Any]] = []
    any_captions = False
    any_metadata = False
    for d in dir_configs:
        p = d.get("path", None)
        assert p is not None, "Each dataset entry needs a 'path'"
        fmt = d.get("format", None)  # None => infer per file
        if fmt is not None and fmt not in FMT_ID:
            raise ValueError(
                f"dataset {d.get('id', p)!r} has unsupported format {fmt!r}; "
                f"choose one of {sorted(FMT_ID)} or omit it for inference"
            )
        files = get_audio_filenames(
            p,
            d.get("keywords", None),
            filelist_path=d.get("filelist_path", None),
        )
        # Optional per-dataset cap so a source can be used sparingly (e.g. keep
        # MRSDrama minimal). Deterministic subsample keyed by the dataset id.
        max_files = d.get("max_files")
        if max_files is not None and len(files) > max_files:
            files = sorted(random.Random(str(d.get("id", p))).sample(files, max_files))
        cap_map = load_caption_map(d["captions"]) if d.get("captions") else {}
        meta_map = load_metadata_map(d["metadata"]) if d.get("metadata") else {}
        n_with_cap = 0
        n_with_meta = 0
        for f in files:
            c = caption_for_file(f, cap_map) if cap_map else ""
            m = metadata_for_file(f, meta_map) if meta_map else {}
            items.append((f, fmt))
            captions.append(c)
            metadata_rows.append(m)
            if c:
                n_with_cap += 1
            if m:
                n_with_meta += 1
        any_captions = any_captions or n_with_cap > 0
        any_metadata = any_metadata or n_with_meta > 0
        print(f"[4ch loader] {d.get('id', p)}: {len(files)} files "
              f"(format={fmt or 'infer'}, captioned={n_with_cap}/{len(files)}, "
              f"metadata={n_with_meta}/{len(files)})")

    assert items, "No audio files found for the 4ch dataloader"

    train_set = FourChannelSampleDataset(
        items,
        sample_size=sample_size,
        sample_rate=sample_rate,
        normalize=dataset_config.get("normalize", "joint_peak"),
        peak=dataset_config.get("peak", 0.9),
        random_crop=dataset_config.get("random_crop", True),
        pad=pad,
        captions=captions if any_captions else None,
        metadata=metadata_rows if any_metadata else None,
        max_item_retries=dataset_config.get("max_item_retries", 8),
    )

    return torch.utils.data.DataLoader(
        train_set,
        batch_size,
        shuffle=shuffle,
        **_local_dataloader_kwargs(dataset_config, num_workers),
    )


# ---------------------------------------------------------------------------
# Explicit file-list helpers (used by the overfit sanity script)
# ---------------------------------------------------------------------------

def list_spatial_librispeech(root: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """FOA ambisonics flac. e.g. root=/mnt/sdb/audio_dataset/datasets/spatial_librispeech"""
    files = sorted(glob.glob(os.path.join(root, "ambisonics", "*.flac")))
    if limit is not None:
        files = files[:limit]
    return [(f, FOA) for f in files]


def list_mrsdrama(root: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Binaural drama wav segments. e.g. root=/mnt/sdd/audio_dataset/datasets/mrsdrama/snapshot"""
    files = sorted(glob.glob(os.path.join(root, "**", "wav", "*.wav"), recursive=True))
    if limit is not None:
        files = files[:limit]
    return [(f, BINAURAL) for f in files]
