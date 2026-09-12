import importlib
import numpy as np
import io
import json
import os
import dill
import errno
import posixpath
import random
import re
import subprocess
import time
import warnings
import torch
import torchaudio
import webdataset as wds

from os import path
from pathlib import Path
from torch import nn
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T, VolumeNorm, strip_trailing_silence
from .text_conditioning import tokenize_text_metadata

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")


class _DatasetItemRetryExhausted(RuntimeError):
    pass


class _FileDescriptorExhausted(RuntimeError):
    pass


def _load_custom_metadata_fn(module_path: str, config: Optional[dict] = None):
    """Load a dataset metadata hook, optionally through a configured factory.

    Legacy hooks expose ``get_custom_metadata(info, data)``.  Hooks that need
    paths or preprocessing parameters can instead expose
    ``create_custom_metadata(config) -> callable``.  Keeping construction here
    means dataset configs stay declarative and worker call sites remain
    identical for local audio, pre-encoded latents, and WebDataset.
    """
    spec = importlib.util.spec_from_file_location("metadata_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load custom metadata module: {module_path}")
    metadata_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metadata_module)

    factory = getattr(metadata_module, "create_custom_metadata", None)
    if factory is not None:
        return factory(dict(config or {}))

    if config:
        raise ValueError(
            f"custom metadata module {module_path} has custom_metadata_config but "
            "does not expose create_custom_metadata(config)"
        )

    try:
        return metadata_module.get_custom_metadata
    except AttributeError as exc:
        raise AttributeError(
            f"custom metadata module {module_path} must expose either "
            "get_custom_metadata or create_custom_metadata"
        ) from exc

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus'],
    filelist_path=None
):
    "recursively get a list of audio filenames"
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames
        # Resolve the implicit manifest independently for every root. Mutating
        # ``filelist_path`` here made multi-root configs reuse the first root's
        # manifest for every subsequent root.
        current_filelist_path = (
            filelist_path
            if filelist_path is not None
            else os.path.join(path, "filelist.txt")
        )
            
        if os.path.isfile(current_filelist_path):
            with open(current_filelist_path, "r") as f:
                entries = [line.strip() for line in f if line.strip()]
                files = [
                    entry if os.path.isabs(entry) else os.path.join(path, entry)
                    for entry in entries
                ]
                filenames.extend(files)
            continue

        if keywords is not None:
            subfolders, files = keyword_scandir(path, exts, keywords)
        else:
            subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

def get_latent_filenames(
    paths,  # directories in which to search
    extension='npy',
    filelist_path=None,
    validate_filelist_entries=True,
):
    """Return complete ``(latent, metadata)`` pairs from one or more roots.

    A root-level ``filelist.txt`` is preferred when present. Finalized caches
    may set ``validate_filelist_entries=False`` to trust that manifest and avoid
    roughly two million stat calls on every distributed training startup.
    Recursive discovery always validates both files so an in-progress cache is
    safe to inspect.
    """

    pairs = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames
        current_filelist_path = (
            filelist_path
            if filelist_path is not None
            else os.path.join(path, "filelist.txt")
        )
        using_filelist = os.path.isfile(current_filelist_path)

        if using_filelist:
            with open(current_filelist_path, "r") as f:
                entries = [line.strip() for line in f if line.strip()]
            filenames = [
                entry if os.path.isabs(entry) else os.path.join(path, entry)
                for entry in entries
            ]
        else:
            _, filenames = fast_scandir(path, [extension])

        for filename in filenames:
            if os.path.basename(filename) == "silence.npy":
                continue
            metadata_filename = os.path.splitext(filename)[0] + ".json"
            if using_filelist and not validate_filelist_entries:
                pairs.append((filename, metadata_filename))
            elif os.path.isfile(filename) and os.path.isfile(metadata_filename):
                pairs.append((filename, metadata_filename))

    return pairs


def _validate_finalized_latent_cache(
    root: str,
    pair_count: Optional[int] = None,
) -> None:
    """Validate the cheap READY contract before trusting an un-statted manifest."""

    root_path = Path(root).expanduser()
    ready_path = root_path / "READY"
    if not ready_path.is_file():
        raise RuntimeError(
            f"validate_filelist_entries=false requires a finalized cache marker: "
            f"{ready_path}"
        )
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid pre-encoded READY marker: {ready_path}") from exc
    if ready.get("schema") != "stable_audio_tools.preencoded_ready":
        raise RuntimeError(
            f"unsupported READY schema in {ready_path}: {ready.get('schema')!r}"
        )
    ready_entries = ready.get("entries")
    if not isinstance(ready_entries, int) or ready_entries < 0:
        raise RuntimeError(f"READY entries must be a non-negative integer: {ready_path}")
    if pair_count is not None and ready_entries != pair_count:
        raise RuntimeError(
            f"READY/filelist count mismatch in {root_path}: "
            f"{ready_entries} != {pair_count}"
        )


def _path_is_within(filename: str, root: str) -> bool:
    """Return whether ``filename`` belongs to ``root`` without substring matches."""

    try:
        filename_abs = os.path.abspath(os.fspath(filename))
        root_abs = os.path.abspath(os.fspath(root))
        return os.path.commonpath((filename_abs, root_abs)) == root_abs
    except (TypeError, ValueError):
        return False


def _normalize_padding_mask(value, length: int) -> list[bool]:
    """Return one flat boolean mask whose length matches the latent."""

    if length < 0:
        raise ValueError(f"padding-mask length must be non-negative, got {length}")
    if value is None:
        return [True] * length
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    elif (
        isinstance(value, (list, tuple))
        and len(value) == 1
        and isinstance(value[0], (list, tuple, np.ndarray, torch.Tensor))
    ):
        return _normalize_padding_mask(value[0], length)
    elif isinstance(value, (list, tuple)):
        value = list(value)
    else:
        raise TypeError(
            "padding_mask must be a tensor, ndarray, list, tuple, or null; "
            f"got {type(value).__name__}"
        )

    result = [bool(item) for item in value[:length]]
    if len(result) < length:
        result.extend([False] * (length - len(result)))
    seen_padding = False
    for is_valid in result:
        if not is_valid:
            seen_padding = True
        elif seen_padding:
            raise ValueError(
                "padding_mask must be a contiguous valid prefix followed by padding"
            )
    return result


class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        keywords: Optional[List[str]]=None,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        filelist_path = None,
        weight: float = 1.0,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.keywords = keywords
        self.filelist_path = filelist_path
        self.weight = weight

class LatentDatasetConfig(LocalDatasetConfig):
    def __init__(
        self,
        latent_extension: str = "npy",
        filelist_path = None,
        validate_filelist_entries: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.latent_extension = latent_extension
        self.filelist_path = filelist_path
        self.validate_filelist_entries = bool(validate_filelist_entries)
        # weight is inherited from LocalDatasetConfig via **kwargs

class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs,
        sample_size=65536,
        sample_rate=48000,
        random_crop=True,
        force_channels="stereo",
        volume_norm=False,
        volume_norm_param=(-16, 2),
        strip_silence=False,
        pad=True,
        max_item_retries=8,
    ):
        super().__init__()
        self.filenames = []
        self.sample_weights = []

        self.augs = torch.nn.Sequential(
            PhaseFlipper(),
            #nn.Identity()
        )


        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop, pad=pad)
        self.strip_silence = strip_silence

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity()
        )

        self.sr = sample_rate

        self.volume_norm = VolumeNorm(volume_norm_param, self.sr) if volume_norm else torch.nn.Identity()

        self.custom_metadata_fns = {}
        # Serialized factories are portable across DataLoader processes, while
        # the runtime callable may own SQLite connections / mmap handles. Cache
        # one deserialized instance per worker instead of rebuilding it per item.
        self._custom_metadata_runtime = {}
        self.max_item_retries = int(max_item_retries)
        if self.max_item_retries < 0:
            raise ValueError("max_item_retries must be non-negative")

        for config in configs:
            config_roots = (
                config.path if isinstance(config.path, (list, tuple)) else [config.path]
            )
            self.root_paths.extend(config_roots)
            new_files = get_audio_filenames(config.path, config.keywords, filelist_path=config.filelist_path)
            self.filenames.extend(new_files)
            self.sample_weights.extend([config.weight] * len(new_files))
            if config.custom_metadata_fn is not None:
                serialized_fn = dill.dumps(config.custom_metadata_fn)
                for root in config_roots:
                    self.custom_metadata_fns[root] = serialized_fn

        print(f'Found {len(self.filenames)} files')
        if not self.filenames:
            raise RuntimeError("local audio dataset contains no readable files")

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)

    def _retry_item(self, retry_count: int, reason: str):
        if retry_count >= self.max_item_retries:
            raise _DatasetItemRetryExhausted(
                f"failed to find a valid audio item after "
                f"{self.max_item_retries} retries; last reason: {reason}"
            )
        return self.__getitem__(
            random.randrange(len(self)), _retry_count=retry_count + 1
        )

    def __getitem__(self, idx, _retry_count=0):
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)

            audio = self.volume_norm(audio)

            if self.strip_silence:
                audio = strip_trailing_silence(audio, self.sr)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Check for silence
            if is_silence(audio):
                return self._retry_item(_retry_count, "decoded audio is silent")

            # Run augmentations on this sample (including random crop)
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}

            info["path"] = audio_filename

            for root_path in self.root_paths:
                if _path_is_within(audio_filename, root_path):
                    info["relpath"] = path.relpath(audio_filename, root_path)
                    break

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = [padding_mask]
            info["sample_rate"] = self.sr

            end_time = time.time()

            info["load_time"] = end_time - start_time

            for custom_md_path in self.custom_metadata_fns.keys():
                if _path_is_within(audio_filename, custom_md_path):
                    custom_metadata_fn = self._custom_metadata_runtime.get(custom_md_path)
                    if custom_metadata_fn is None:
                        custom_metadata_fn = dill.loads(self.custom_metadata_fns[custom_md_path])
                        self._custom_metadata_runtime[custom_md_path] = custom_metadata_fn
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)
                    break

            if info.get("__reject__"):
                return self._retry_item(_retry_count, "custom metadata rejected sample")

            # Provide audio inputs as their own dictionary to be merged into info,
            # each normalized like the main audio.
            if "__audio__" in info:
                for audio_key, audio_value in info["__audio__"].items():
                    audio_value, _, _, _, _, _ = self.pad_crop(audio_value)
                    audio_value = audio_value.clamp(-1, 1)
                    if self.encoding is not None:
                        audio_value = self.encoding(audio_value)
                    info[audio_key] = audio_value
                del info["__audio__"]

            return (audio, info)
        except (_DatasetItemRetryExhausted, _FileDescriptorExhausted):
            raise
        except OSError as e:
            if e.errno in (errno.EMFILE, errno.ENFILE):
                raise _FileDescriptorExhausted(
                    "DataLoader exhausted file descriptors while loading "
                    f"{audio_filename}; raise RLIMIT_NOFILE or reduce "
                    "batch_size/workers/prefetch_factor"
                ) from e
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self._retry_item(_retry_count, str(e))
        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self._retry_item(_retry_count, str(e))

class PreEncodedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        configs: List[LatentDatasetConfig],
        latent_crop_length=None,
        min_length_sec=None,
        max_length_sec=None,
        random_crop=False,
        tokenizers: Optional[dict] = None,
        latent_downsampling_ratio: Optional[int] = None,
        sample_rate: Optional[int] = None,
        max_item_retries: int = 8,
    ):
        super().__init__()
        self.filenames = []
        self.sample_weights = []

        self.custom_metadata_fns = {}
        self._custom_metadata_runtime = {}

        self.silence_latents = {}

        for config in configs:
            new_files = get_latent_filenames(
                config.path,
                config.latent_extension,
                config.filelist_path,
                validate_filelist_entries=config.validate_filelist_entries,
            )
            self.filenames.extend(new_files)
            self.sample_weights.extend([config.weight] * len(new_files))
            if config.custom_metadata_fn is not None:
                serialized_fn = dill.dumps(config.custom_metadata_fn)
                config_roots = (
                    config.path
                    if isinstance(config.path, (list, tuple))
                    else [config.path]
                )
                for root in config_roots:
                    self.custom_metadata_fns[root] = serialized_fn

            # Load silence latent if available (for variable-length padding)
            paths = config.path if isinstance(config.path, list) else [config.path]
            for path in paths:
                silence_path = os.path.join(path, "silence.npy")
                if os.path.exists(silence_path):
                    silence = np.load(silence_path, allow_pickle=False).squeeze(0)
                    if (
                        silence.ndim != 2
                        or not np.issubdtype(silence.dtype, np.floating)
                        or silence.shape[1] == 0
                        or not np.isfinite(silence).all()
                    ):
                        raise ValueError(
                            f"silence latent must be finite floating [C,N] with N>0: "
                            f"{silence_path} has shape={silence.shape}, dtype={silence.dtype}"
                        )
                    self.silence_latents[path] = silence  # [C, N]
                    print(f'Loaded silence latent from {silence_path}')

        self.latent_crop_length = latent_crop_length
        self.random_crop = random_crop

        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec

        # tokenizers: dict mapping metadata key -> (tokenizer, max_length)
        # If provided, text fields will be pre-tokenized in DataLoader workers
        self.tokenizers = tokenizers

        self.seconds_per_latent = None
        self.max_item_retries = int(max_item_retries)
        if self.max_item_retries < 0:
            raise ValueError("max_item_retries must be non-negative")
        if latent_downsampling_ratio is not None:
            if sample_rate is None or sample_rate <= 0:
                raise ValueError("sample_rate must be positive when latent_downsampling_ratio is set")
            if latent_downsampling_ratio <= 0:
                raise ValueError("latent_downsampling_ratio must be positive")
            self.seconds_per_latent = float(latent_downsampling_ratio) / float(sample_rate)

        print(f'Found {len(self.filenames)} files')
        if not self.filenames:
            raise RuntimeError("pre-encoded dataset contains no complete latent/metadata pairs")

    def __len__(self):
        return len(self.filenames)

    def _get_silence_for_file(self, latent_filename):
        """Return the silence latent for the dataset that contains this file, or None."""
        for path, silence in self.silence_latents.items():
            if _path_is_within(latent_filename, path):
                return silence
        return None

    def _retry_item(self, retry_count: int, reason: str):
        if retry_count >= self.max_item_retries:
            raise _DatasetItemRetryExhausted(
                f"failed to find a valid pre-encoded item after "
                f"{self.max_item_retries} retries; last reason: {reason}"
            )
        return self.__getitem__(
            random.randrange(len(self)), _retry_count=retry_count + 1
        )

    def __getitem__(self, idx, _retry_count=0):
        latent_filename, md_filename = self.filenames[idx]
        try:
            latent_array = np.load(latent_filename, allow_pickle=False)
            if latent_array.ndim != 2:
                raise ValueError(
                    f"latent must be [channels, frames], got {latent_array.shape}"
                )
            if not np.issubdtype(latent_array.dtype, np.floating):
                raise TypeError(
                    f"latent dtype must be floating point, got {latent_array.dtype}"
                )
            if not np.isfinite(latent_array).all():
                raise ValueError("latent contains NaN or infinite values")
            latents = torch.from_numpy(latent_array) # [C, N]

            with open(md_filename, "r", encoding="utf-8") as f:
                try:
                    info = json.load(f)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"couldn't parse metadata file {md_filename}"
                    ) from exc
            if not isinstance(info, dict):
                raise TypeError(
                    f"metadata must be a JSON object, got {type(info).__name__}"
                )

            info["latent_filename"] = latent_filename
            stored_length = latents.shape[1]
            info["latent_stored_length"] = stored_length

            source_seconds_start = float(info.get("seconds_start", 0.0))
            source_seconds_total_value = info.get("seconds_total")
            if source_seconds_total_value is None:
                timestamps = info.get("timestamps")
                if isinstance(timestamps, (list, tuple)) and len(timestamps) == 2:
                    source_seconds_total_value = max(
                        0.0, float(timestamps[1]) - float(timestamps[0])
                    )
                elif self.seconds_per_latent is not None:
                    source_seconds_total_value = (
                        stored_length * self.seconds_per_latent
                    )
                else:
                    raise KeyError(
                        "metadata needs seconds_total, two timestamps, or a "
                        "configured latent_downsampling_ratio/sample_rate"
                    )
            source_seconds_total = float(source_seconds_total_value)
            if not np.isfinite(source_seconds_start) or not np.isfinite(
                source_seconds_total
            ):
                raise ValueError("seconds_start/seconds_total must be finite")
            if source_seconds_total < 0:
                raise ValueError("seconds_total must be non-negative")
            info["seconds_start"] = source_seconds_start
            info["seconds_total"] = source_seconds_total
            info["latent_source_seconds_start"] = source_seconds_start
            info["latent_source_seconds_total"] = source_seconds_total
            if "timestamps" in info:
                info["latent_source_timestamps"] = info["timestamps"]

            padding_mask = _normalize_padding_mask(
                info.get("padding_mask"),
                stored_length,
            )
            valid_indices = [
                index for index, is_valid in enumerate(padding_mask) if is_valid
            ]
            valid_length = valid_indices[-1] + 1 if valid_indices else 0
            if valid_length == 0:
                return self._retry_item(
                    _retry_count, "padding_mask contains no valid latent frames"
                )

            if self.latent_crop_length is not None:
                if stored_length > self.latent_crop_length:
                    # Only choose starts that keep the requested crop inside the
                    # real signal. randint's upper bound is inclusive, so a clip
                    # exactly one frame longer than the crop has two valid starts.
                    max_start = max(0, min(stored_length, valid_length) - self.latent_crop_length)
                    if self.random_crop and max_start > 0:
                        start = random.randint(0, max_start)
                    else:
                        start = 0

                    latents = latents[:, start:start+self.latent_crop_length]
                    padding_mask = padding_mask[
                        start:start + self.latent_crop_length
                    ]
                    info["latent_crop_start"] = start

                    # Preserve temporal conditioning when a pre-encoded clip is
                    # cropped in latent space.  The old path always left
                    # seconds_start at zero, mislabelling 10-16 s training clips.
                    seconds_per_latent = self.seconds_per_latent
                    if seconds_per_latent is None and stored_length > 0:
                        seconds_per_latent = source_seconds_total / stored_length
                    if seconds_per_latent is not None:
                        crop_offset_seconds = start * seconds_per_latent
                        info["seconds_start"] = (
                            source_seconds_start + crop_offset_seconds
                        )
                        info["seconds_total"] = max(
                            0.0,
                            min(
                                source_seconds_total - crop_offset_seconds,
                                self.latent_crop_length * seconds_per_latent,
                            ),
                        )
                    timestamps = info.get("timestamps")
                    if isinstance(timestamps, (list, tuple)) and len(timestamps) == 2:
                        t0, t1 = map(float, timestamps)
                        span = t1 - t0
                        info["timestamps"] = [
                            t0 + span * start / stored_length,
                            t0 + span * min(start + self.latent_crop_length, stored_length) / stored_length,
                        ]

                elif stored_length < self.latent_crop_length:
                    # Pad with silence latent to reach latent_crop_length
                    pad_needed = self.latent_crop_length - stored_length
                    silence = self._get_silence_for_file(latent_filename)

                    if silence is not None:
                        if silence.shape[0] != latents.shape[0]:
                            raise ValueError(
                                "silence latent channel mismatch for "
                                f"{latent_filename}: {silence.shape[0]} != "
                                f"{latents.shape[0]}"
                            )
                        # Slice or tile silence latent to cover pad_needed frames
                        if silence.shape[1] >= pad_needed:
                            silence_pad = silence[:, :pad_needed]
                        else:
                            silence_pad = np.tile(silence, (1, (pad_needed // silence.shape[1]) + 1))[:, :pad_needed]
                        latents = torch.cat([latents, torch.from_numpy(silence_pad)], dim=1)
                    else:
                        # No silence latent available — zero-pad as fallback
                        latents = torch.nn.functional.pad(latents, (0, pad_needed))

                    # Build padding_mask: valid frames from stored mask, zeros for padding
                    padding_mask = (
                        padding_mask[:stored_length] + [False] * pad_needed
                    )
                    info["latent_crop_start"] = 0

                else:
                    # Exact match
                    info["latent_crop_start"] = 0

                info["latent_crop_length"] = self.latent_crop_length

            info["padding_mask"] = [
                torch.tensor(padding_mask, dtype=torch.bool)
            ]

            if self.min_length_sec is not None and source_seconds_total < self.min_length_sec:
                return self._retry_item(_retry_count, "sample is shorter than min_length_sec")

            if self.max_length_sec is not None and source_seconds_total > self.max_length_sec:
                return self._retry_item(_retry_count, "sample is longer than max_length_sec")

            for custom_md_path in self.custom_metadata_fns.keys():
                if _path_is_within(latent_filename, custom_md_path):
                    custom_metadata_fn = self._custom_metadata_runtime.get(custom_md_path)
                    if custom_metadata_fn is None:
                        custom_metadata_fn = dill.loads(self.custom_metadata_fns[custom_md_path])
                        self._custom_metadata_runtime[custom_md_path] = custom_metadata_fn
                    custom_metadata = custom_metadata_fn(info, latents)
                    info.update(custom_metadata)
                    break

            if info.get("__reject__"):
                return self._retry_item(_retry_count, "custom metadata rejected sample")

            if info.get("__replace__") is not None:
                # Replace latents when requested by the custom metadata hook.
                latents = info["__replace__"]

            info["audio"] = latents

            # Pre-tokenize text fields in DataLoader workers to avoid
            # CPU contention with the main training thread
            if self.tokenizers is not None:
                for key, tokenizer_spec in self.tokenizers.items():
                    if key in info and isinstance(info[key], str):
                        # Save raw text before replacing with tokens (needed by CLAP and other text-based losses)
                        info[f"{key}_text"] = info[key]
                        info[key] = tokenize_text_metadata(
                            info[key], tokenizer_spec
                        )

            return (latents, info)
        except (_DatasetItemRetryExhausted, _FileDescriptorExhausted):
            raise
        except OSError as e:
            if e.errno in (errno.EMFILE, errno.ENFILE):
                raise _FileDescriptorExhausted(
                    "DataLoader exhausted file descriptors while loading "
                    f"{latent_filename}; raise RLIMIT_NOFILE or reduce "
                    "batch_size/workers/prefetch_factor"
                ) from e
            print(f'Couldn\'t load file {latent_filename}: {e}')
            return self._retry_item(_retry_count, str(e))
        except Exception as e:
            print(f'Couldn\'t load file {latent_filename}: {e}')
            return self._retry_item(_retry_count, str(e))

# S3 code and WDS preprocessing code based on implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py

def get_s3_contents(dataset_path, s3_url_prefix=None, filter='', recursive=True, debug=False, profile=None):
    """
    Returns a list of full S3 paths to files in a given S3 bucket and directory path.
    """
    # Ensure dataset_path ends with a trailing slash
    if dataset_path != '' and not dataset_path.endswith('/'):
        dataset_path += '/'
    # Use posixpath to construct the S3 URL path
    bucket_path = posixpath.join(s3_url_prefix or '', dataset_path)
    # Construct the `aws s3 ls` command
    cmd = ['aws', 's3', 'ls', bucket_path]

    if profile is not None:
        cmd.extend(['--profile', profile])

    if recursive:
        # Add the --recursive flag if requested
        cmd.append('--recursive')
    
    # Run the `aws s3 ls` command and capture the output
    run_ls = subprocess.run(cmd, capture_output=True, check=True)
    # Split the output into lines and strip whitespace from each line
    contents = run_ls.stdout.decode('utf-8').split('\n')
    contents = [x.strip() for x in contents if x]
    # Remove the timestamp from lines that begin with a timestamp
    contents = [re.sub(r'^\S+\s+\S+\s+\d+\s+', '', x)
                if re.match(r'^\S+\s+\S+\s+\d+\s+', x) else x for x in contents]
    # Construct a full S3 path for each file in the contents list
    contents = [posixpath.join(s3_url_prefix or '', x)
                for x in contents if not x.endswith('/')]
    # Apply the filter, if specified
    if filter:
        contents = [x for x in contents if filter in x]
    # Remove redundant directory names in the S3 URL
    if recursive:
        # Get the main directory name from the S3 URL
        main_dir = "/".join(bucket_path.split('/')[3:])
        # Remove the redundant directory names from each file path
        contents = [x.replace(f'{main_dir}', '').replace(
            '//', '/') for x in contents]
    # Print debugging information, if requested
    if debug:
        print("contents = \n", contents)
    # Return the list of S3 paths to files
    return contents


def get_all_s3_urls(
    names=[],           # list of all valid [LAION AudioDataset] dataset names
    # list of subsets you want from those datasets, e.g. ['train','valid']
    subsets=[''],
    s3_url_prefix=None,  # prefix for those dataset names
    recursive=True,     # recursively list all tar files in all subdirs
    filter_str='tar',   # only grab files with this substring
    # print debugging info -- note: info displayed likely to change at dev's whims
    debug=False,
    profiles={},        # dictionary of profiles for each item in names, e.g. {'dataset1': 'profile1', 'dataset2': 'profile2'}
):
    "get urls of shards (tar files) for multiple datasets in one s3 bucket"
    urls = []
    for name in names:
        # If s3_url_prefix is not specified, assume the full S3 path is included in each element of the names list
        if s3_url_prefix is None:
            contents_str = name
        else:
            # Construct the S3 path using the s3_url_prefix and the current name value
            contents_str = posixpath.join(s3_url_prefix, name)
        if debug:
            print(f"get_all_s3_urls: {contents_str}:")
        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            if debug:
                print(f"subset_str = {subset_str}")
            # Get the list of tar files in the current subset directory
            profile = profiles.get(name, None)
            tar_list = get_s3_contents(
                subset_str, s3_url_prefix=None, recursive=recursive, filter=filter_str, debug=debug, profile=profile)
            for tar in tar_list:
                # Escape spaces and parentheses in the tar filename for use in the shell command
                tar = tar.replace(" ", "\ ").replace(
                    "(", "\(").replace(")", "\)")
                # Construct the S3 path to the current tar file
                s3_path = posixpath.join(name, subset, tar) + " -"
                # Construct the AWS CLI command to download the current tar file
                if s3_url_prefix is None:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {s3_path}"
                else:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {posixpath.join(s3_url_prefix, s3_path)}"
                if profiles.get(name):
                    request_str += f" --profile {profiles.get(name)}"
                if debug:
                    print("request_str = ", request_str)
                # Add the constructed URL to the list of URLs
                urls.append(request_str)
    return urls


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True

# get_dbmax and is_silence copied from https://github.com/drscotthawley/aeiou/blob/main/aeiou/core.py under Apache 2.0 License
# License can be found in LICENSES/LICENSE_AEIOU.txt
def get_dbmax(
    audio,       # torch tensor of (multichannel) audio
    ):
    "finds the loudest value in the entire clip and puts that into dB (full scale)"
    return 20*torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()

def is_silence(
    audio,       # torch tensor of (multichannel) audio
    thresh=-60,  # threshold in dB below which we declare to be silence
    ):
    "checks if entire clip is 'silence' below some dB threshold"
    dBmax = get_dbmax(audio)
    return dBmax < thresh

def is_valid_sample(sample):
    has_json = "json" in sample
    has_audio = "audio" in sample
    is_pre_encoded = sample.get("__pre_encoded__", False)
    is_silent = (not is_pre_encoded) and is_silence(sample["audio"])
    is_rejected = "__reject__" in sample["json"] and sample["json"]["__reject__"]

    return has_json and has_audio and not is_silent and not is_rejected


def remove_long_silence(audio, sample_rate, silence_threshold=[0.01, 0.5], max_silence_duration=0.25):
    """
    Removes silence longer than max_silence_duration and replaces it with a short silence.

    :param audio: torch tensor of shape [1, T]
    :param sample_rate: Sampling rate of the audio
    :param silence_threshold: List with [silence_energy_threshold, silence_duration_threshold] to consider a segment as silence
    :param max_silence_duration: Maximum allowed silence duration in seconds
    :return: Processed audio tensor
    """
    
    silence_energy_threshold, silence_duration_threshold = silence_threshold

    max_silence_samples = int(max_silence_duration * sample_rate)
    tiny_silence_samples = int(silence_duration_threshold * sample_rate)
    
    # Flatten the audio tensor
    audio = audio.flatten()
    
    # Detect silent segments
    silence_mask = torch.abs(audio) < silence_energy_threshold
    silence_mask_diff = torch.diff(silence_mask.int())
    
    # Find indices where silence starts and ends
    silence_starts = torch.where(silence_mask_diff == 1)[0] + 1
    silence_ends = torch.where(silence_mask_diff == -1)[0] + 1

    # Handle the case where the tensor starts or ends with silence
    if silence_mask[0]:
        silence_starts = torch.cat((torch.tensor([0], device=silence_starts.device), silence_starts))
    if silence_mask[-1]:
        silence_ends = torch.cat((silence_ends, torch.tensor([len(audio)], device=silence_ends.device)))

    processed_audio = []
    prev_end = 0
    for start, end in zip(silence_starts, silence_ends):
        # Add non-silence segment
        processed_audio.append(audio[prev_end:start])
        
        silence_segment = audio[start:end]
        if len(silence_segment) > max_silence_samples:
            # Replace long silence with a random segment of 0-0.5s silence
            if len(silence_segment) > tiny_silence_samples:
                start_idx = random.randint(0, len(silence_segment) - tiny_silence_samples)
                processed_audio.append(silence_segment[start_idx:start_idx + tiny_silence_samples])
            else:
                processed_audio.append(silence_segment[:tiny_silence_samples])
        else:
            # Keep the silence segment as is
            processed_audio.append(silence_segment)

        prev_end = end
    
    # Add the last non-silence segment if there is any
    if prev_end < len(audio):
        processed_audio.append(audio[prev_end:])
    
    # Concatenate all processed segments back into a single tensor
    processed_audio_tensor = torch.cat(processed_audio).unsqueeze(0)
    
    return processed_audio_tensor


def is_silence_audio(audio, silence_threshold=0.01, max_silence_ratio=0.3):
    # Calculate the ratio of silent frames in the audio sample
    silence_frames = torch.sum(audio.abs() < silence_threshold, dim=1)
    total_frames = audio.size(1)
    silence_ratio_per_channel = silence_frames / total_frames

    if torch.any(silence_ratio_per_channel > max_silence_ratio).item():
        # Save the tensor to an audio file
        output_path = f'rejected_audios/rejected_{silence_ratio_per_channel.item()}.wav'
        torchaudio.save(output_path, audio, 16000)
        print(f'Rejected: {silence_ratio_per_channel}')
    # Check if any channel exceeds the max silence ratio
    return torch.any(silence_ratio_per_channel > max_silence_ratio).item()

class S3DatasetConfig:
    def __init__(
        self,
        id: str,
        s3_path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
    ):
        self.id = id
        self.path = s3_path
        self.custom_metadata_fn = custom_metadata_fn
        self.profile = profile
        self.urls = []

    def load_data_urls(self):
        self.urls = get_all_s3_urls(
            names=[self.path],
            s3_url_prefix=None,
            recursive=True,
            profiles={self.path: self.profile} if self.profile else {},
        )

        return self.urls

class LocalWebDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        profile: Optional[str] = None,
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.urls = []

    def load_data_urls(self):

        self.urls = fast_scandir(self.path, ["tar"])[1]

        return self.urls

def audio_decoder(key, value):
    # Get file extension from key
    ext = key.split(".")[-1]

    if ext in AUDIO_KEYS:
        return torchaudio.load(io.BytesIO(value))
    else:
        return None

def npy_decoder(key, value):
    # Get file extension from key
    ext = key.split(".")[-1]

    if ext == "npy":
        return np.lib.format.read_array(io.BytesIO(value))
    else:
        return None

def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], list) and isinstance(b[0][0], torch.Tensor):
                # This preserves the [Batch, List, Tensor] structure 
                # expected by the md["padding_mask"][0] logic
                b = b
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result

class WebDatasetDataLoader():
    def __init__(
        self,
        datasets: List[S3DatasetConfig],
        batch_size,
        sample_size,
        sample_rate=48000,
        num_workers=8,
        epoch_steps=1000,
        random_crop=True,
        force_channels="stereo",
        augment_phase=True,
        remove_silence=True,
        silence_threshold=[0.01, 0.5],
        max_silence_duration=0.2,
        volume_norm=False,
        volume_norm_param=(-16, 2),
        pre_encoded=False,
        latent_crop_length=None,
        min_length_sec=None,
        max_length_sec=None,
        resampled_shards=True,
        strip_silence=False,
        **data_loader_kwargs
    ):

        self.datasets = datasets

        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels
        self.augment_phase = augment_phase
        self.pre_encoded = pre_encoded
        self.latent_crop_length = latent_crop_length
        self.min_length_sec = min_length_sec
        self.max_length_sec = max_length_sec
        self.volume_norm = volume_norm
        self.volume_norm_param = volume_norm_param
        self.remove_silence = remove_silence
        self.silence_threshold = silence_threshold
        self.max_silence_duration = max_silence_duration
        self.strip_silence = strip_silence

        urls = [dataset.load_data_urls() for dataset in datasets]

        # Flatten the list of lists of URLs
        urls = [url for dataset_urls in urls for url in dataset_urls]

        # Shuffle the urls
        random.shuffle(urls)

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(urls) if resampled_shards else wds.SimpleShardList(urls),
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode(audio_decoder, handler=log_and_continue) if not self.pre_encoded else wds.decode(npy_decoder, handler=log_and_continue),
            wds.map(self.wds_preprocess, handler=log_and_continue),
            #wds.map(self.wds_preprocess),
            wds.select(is_valid_sample),
            wds.to_tuple("audio", "json", handler=log_and_continue),
            #wds.shuffle(bufsize=1000, initial=5000),
            wds.batched(batch_size, partial=False, collation_fn=collation_fn),
        )

        if resampled_shards:
            self.dataset = self.dataset.with_epoch(epoch_steps//num_workers if num_workers > 0 else epoch_steps)

        data_loader_kwargs.setdefault('persistent_workers', num_workers > 0)
        self.data_loader = wds.WebLoader(self.dataset, num_workers=num_workers, **data_loader_kwargs)

    def wds_preprocess(self, sample):

        if self.pre_encoded:
            audio = torch.from_numpy(sample["npy"])
            del sample["npy"]
            sample["__pre_encoded__"] = True

            padding_mask = sample["json"]["padding_mask"]
            if self.latent_crop_length is not None:

                # Get the last index from the padding mask, the index of the last 1 in the sequence
                last_ix = len(padding_mask) - 1 - padding_mask[::-1].index(1)

                if self.random_crop and last_ix > self.latent_crop_length:
                    start = random.randint(0, last_ix - self.latent_crop_length)
                else:
                    start = 0
                    
                audio = audio[:, start:start+self.latent_crop_length]

                padding_mask = padding_mask[start:start+self.latent_crop_length]

            sample["json"]["padding_mask"] = torch.tensor(padding_mask)

            # Filter by length if min/max length is specified
            seconds_total = sample["json"].get("seconds_total", None)
            if seconds_total is not None:
                if self.min_length_sec is not None and seconds_total < self.min_length_sec:
                    sample["json"]["__reject__"] = True
                if self.max_length_sec is not None and seconds_total > self.max_length_sec:
                    sample["json"]["__reject__"] = True
        else:
            found_key, rewrite_key = '', ''
            for k, v in sample.items():  # print the all entries in dict
                for akey in AUDIO_KEYS:
                    if k.endswith(akey):
                        # to rename long/weird key with its simpler counterpart
                        found_key, rewrite_key = k, akey
                        break
                if '' != found_key:
                    break
            if '' == found_key:  # got no audio!
                return None  # try returning None to tell WebDataset to skip this one

            audio, in_sr = sample[found_key]
            if in_sr != self.sample_rate:
                resample_tf = T.Resample(in_sr, self.sample_rate)
                audio = resample_tf(audio)

                    # Replace the long silence by the short for the mono audios
            if audio.shape[0] == 1 and self.remove_silence:
                audio = remove_long_silence(audio, self.sample_rate, self.silence_threshold, self.max_silence_duration)

            if self.strip_silence:
                audio = strip_trailing_silence(audio, self.sample_rate)

            if self.sample_size is not None:
                # Pad/crop and get the relative timestamp
                pad_crop = PadCrop_Normalized_T(
                    self.sample_size, randomize=self.random_crop, sample_rate=self.sample_rate)
                audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(
                    audio)
                sample["json"]["seconds_start"] = seconds_start
                sample["json"]["seconds_total"] = seconds_total
                sample["json"]["padding_mask"] = padding_mask
            else:
                t_start, t_end = 0, 1

            # Check if audio is length zero, initialize to a single zero if so
            if audio.shape[-1] == 0:
                audio = torch.zeros(1, 1)

            # Make the audio stereo and augment by randomly inverting phase
            augs = torch.nn.Sequential(
                Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
                Mono() if self.force_channels == "mono" else torch.nn.Identity(),
                VolumeNorm(self.volume_norm_param, self.sample_rate) if self.volume_norm else torch.nn.Identity(),
                PhaseFlipper() if self.augment_phase else torch.nn.Identity()
            )

            audio = augs(audio)

            sample["json"]["timestamps"] = (t_start, t_end)

            if found_key != rewrite_key:   # rename long/weird key with its simpler counterpart
                del sample[found_key]

        if "text" in sample["json"]:
            sample["json"]["prompt"] = sample["json"]["text"]

        # Check for custom metadata functions
        for dataset in self.datasets:
            if dataset.custom_metadata_fn is None:
                continue
        
            if dataset.path in sample["__url__"]:
                custom_metadata = dill.loads(dataset.custom_metadata_fn)(sample["json"], audio)
                sample["json"].update(custom_metadata)

        sample["audio"] = audio
        # Add audio to the metadata as well for conditioning
        sample["json"]["audio"] = audio
        
        return sample

def _maybe_create_weighted_sampler(dataset):
    """Create a WeightedRandomSampler if any dataset has non-default weights, otherwise return None."""
    weights = dataset.sample_weights
    if not weights or all(w == 1.0 for w in weights):
        return None
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    return sampler


def _local_dataloader_kwargs(dataset_config, num_workers: int) -> dict:
    """Resolve shared, config-selectable local DataLoader performance knobs."""

    num_workers = int(num_workers)
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(dataset_config.get("pin_memory", True)),
        "drop_last": bool(dataset_config.get("drop_last", True)),
        "collate_fn": collation_fn,
        "in_order": bool(dataset_config.get("in_order", True)),
    }
    if num_workers > 0:
        prefetch_factor = int(dataset_config.get("prefetch_factor", 2))
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        kwargs.update(
            {
                "persistent_workers": bool(
                    dataset_config.get("persistent_workers", True)
                ),
                "prefetch_factor": prefetch_factor,
            }
        )
    return kwargs

def create_dataloader_from_config(
    dataset_config,
    batch_size,
    sample_size,
    sample_rate,
    audio_channels=2,
    num_workers=4,
    shuffle=True,
    tokenizers=None,
    pad=True,
    distributed_world_size=None,
    distributed_rank=None,
):

    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    elif audio_channels == 2:
        force_channels = "stereo"
    else:
        # Preserve native multichannel audio (FOA, 5.1, etc.). The old fallback
        # silently collapsed every audio_channels>1 input to stereo.
        force_channels = None

    if dataset_type == "audio_dir":

        audio_dir_configs = dataset_config.get("datasets", None)

        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        configs = []

        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"

            custom_metadata_fn = None
            custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                custom_metadata_fn = _load_custom_metadata_fn(
                    custom_metadata_module_path,
                    audio_dir_config.get("custom_metadata_config"),
                )

            configs.append(
                LocalDatasetConfig(
                    id=audio_dir_config["id"],
                    path=audio_dir_path,
                    custom_metadata_fn=custom_metadata_fn,
                    keywords=audio_dir_config.get("keywords", None),
                    filelist_path=audio_dir_config.get("filelist_path", None),
                    weight=audio_dir_config.get("weight", 1.0),
                )
            )

        train_set = SampleDataset(
            configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels,
            volume_norm=dataset_config.get("volume_norm", False),
            volume_norm_param=dataset_config.get("volume_norm_param", (-16, 2)),
            strip_silence=dataset_config.get("strip_silence", False),
            pad=pad,
            max_item_retries=dataset_config.get("max_item_retries", 8),
        )

        sampler = _maybe_create_weighted_sampler(train_set)

        return ResumableDataLoader(
            train_set,
            batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            **_local_dataloader_kwargs(dataset_config, num_workers),
        )

    elif dataset_type == "sceneplan_transfusion_editing_preencoded":
        from stable_audio_tools.data.resumable_dataloader import (
            ResumableDataLoader,
        )
        from stable_audio_tools.data.sceneplan_bucket_sampler import (
            DistributedScenePlanBucketBatchSampler,
            sceneplan_bucket_collation,
        )
        from stable_audio_tools.data.sceneplan_transfusion_editing_dataset import (
            ScenePlanTransfusionEditingDataset,
        )

        datasets = dataset_config.get("datasets")
        if not isinstance(datasets, list) or len(datasets) != 1:
            raise ValueError(
                "sceneplan_transfusion_editing_preencoded requires exactly "
                "one frozen paired index"
            )
        index_path = datasets[0].get("path")
        if not isinstance(index_path, str) or not index_path:
            raise ValueError("Editing dataset path must name a frozen SQLite index")
        expected_num_samples = dataset_config.get("expected_num_samples")
        if expected_num_samples is None:
            raise ValueError("Editing dataset requires expected_num_samples")
        if tokenizers is None or "prompt" not in tokenizers:
            raise ValueError("Editing dataset requires the P10 Qwen tokenizer")
        ordinal_range = dataset_config.get("ordinal_range")
        if ordinal_range is None:
            ordinal_start = ordinal_stop = None
        elif (
            not isinstance(ordinal_range, (list, tuple))
            or len(ordinal_range) != 2
        ):
            raise ValueError("Editing ordinal_range must be [start, stop]")
        else:
            ordinal_start, ordinal_stop = map(int, ordinal_range)
        train_set = ScenePlanTransfusionEditingDataset(
            index_path,
            tokenizer_spec=tokenizers["prompt"],
            expected_num_samples=int(expected_num_samples),
            index_num_samples=(
                int(dataset_config["index_num_samples"])
                if dataset_config.get("index_num_samples") is not None
                else None
            ),
            expected_index_sha256=dataset_config.get("index_sha256"),
            sample_ordinals=dataset_config.get("sample_ordinals"),
            ordinal_start=ordinal_start,
            ordinal_stop=ordinal_stop,
            latent_crop_length=int(dataset_config.get("latent_crop_length", 648)),
            caption_max_tokens=int(dataset_config.get("caption_max_tokens", 512)),
            random_crop=bool(dataset_config.get("random_crop", False)),
            require_frozen=bool(dataset_config.get("require_complete", True)),
            verify_tensor_hashes_on_access=bool(
                dataset_config.get("verify_tensor_hashes_on_access", False)
            ),
        )
        bucket_config = dict(dataset_config.get("length_bucket_batching") or {})
        if bool(bucket_config.get("enabled", False)):
            world_size = int(distributed_world_size or 1)
            rank = int(distributed_rank or 0)
            long_batch_size = int(
                bucket_config.get(
                    "long_batch_size", max(1, int(batch_size) * 432 // 648)
                )
            )
            batch_sampler = DistributedScenePlanBucketBatchSampler(
                train_set,
                short_batch_size=int(batch_size),
                long_batch_size=long_batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=int(bucket_config.get("seed", 42)),
                drop_last=bool(dataset_config.get("drop_last", True)),
            )
            loader_kwargs = _local_dataloader_kwargs(dataset_config, num_workers)
            loader_kwargs.pop("drop_last")
            loader_kwargs["collate_fn"] = sceneplan_bucket_collation
            loader = ResumableDataLoader(
                train_set,
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )
            loader.sceneplan_rank_aware_bucket_sampler = True
            loader.sceneplan_bucket_summary = {
                "rank": rank,
                "world_size": world_size,
                "short_batch_size": int(batch_size),
                "long_batch_size": long_batch_size,
                "bucket_counts": {
                    str(bucket): len(rows)
                    for bucket, rows in batch_sampler.bucket_indices.items()
                },
                "batches_per_epoch": len(batch_sampler),
            }
            return loader
        return ResumableDataLoader(
            train_set,
            batch_size,
            shuffle=shuffle,
            **_local_dataloader_kwargs(dataset_config, num_workers),
        )

    elif dataset_type == "sceneplan_v2_preencoded":
        from stable_audio_tools.data.resumable_dataloader import (
            ResumableDataLoader,
        )
        from stable_audio_tools.data.sceneplan_bucket_sampler import (
            DistributedScenePlanBucketBatchSampler,
            sceneplan_bucket_collation,
        )
        from stable_audio_tools.data.sceneplan_v2_dataset import ScenePlanV2Dataset

        datasets = dataset_config.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("sceneplan_v2_preencoded requires frozen split indices")
        expected_num_samples = dataset_config.get("expected_num_samples")
        if expected_num_samples is None:
            raise ValueError("ScenePlan-v2 requires expected_num_samples")
        if tokenizers is None or "prompt" not in tokenizers:
            raise ValueError("ScenePlan-v2 requires the model's Qwen prompt tokenizer")
        multiple = len(datasets) > 1
        children = []
        child_total = 0
        for dataset_index, dataset_entry in enumerate(datasets):
            index_path = dataset_entry.get("path")
            if not isinstance(index_path, str) or not index_path:
                raise ValueError(
                    "ScenePlan-v2 dataset path must name a frozen SQLite index"
                )
            if multiple:
                if dataset_entry.get("weight", 1.0) != 1.0:
                    raise ValueError(
                        "multi-index ScenePlan-v2 requires weight=1.0"
                    )
                child_expected = dataset_entry.get("num_samples")
                if (
                    isinstance(child_expected, bool)
                    or not isinstance(child_expected, int)
                    or child_expected <= 0
                ):
                    raise ValueError(
                        f"datasets[{dataset_index}].num_samples must be positive"
                    )
                child_index_num_samples = dataset_entry.get("index_num_samples")
                child_ordinals = dataset_entry.get("sample_ordinals")
                child_ordinal_range = dataset_entry.get("ordinal_range")
            else:
                child_expected = int(expected_num_samples)
                child_index_num_samples = dataset_config.get("index_num_samples")
                child_ordinals = dataset_config.get("sample_ordinals")
                child_ordinal_range = dataset_config.get("ordinal_range")
            if child_ordinal_range is None:
                child_ordinal_start = None
                child_ordinal_stop = None
            else:
                if (
                    not isinstance(child_ordinal_range, (list, tuple))
                    or len(child_ordinal_range) != 2
                ):
                    raise ValueError(
                        "ScenePlan ordinal_range must be [start, stop]"
                    )
                child_ordinal_start = int(child_ordinal_range[0])
                child_ordinal_stop = int(child_ordinal_range[1])
            child = ScenePlanV2Dataset(
                index_path,
                tokenizer_spec=tokenizers["prompt"],
                expected_num_samples=int(child_expected),
                index_num_samples=(
                    int(child_index_num_samples)
                    if child_index_num_samples is not None
                    else None
                ),
                sample_ordinals=child_ordinals,
                ordinal_start=child_ordinal_start,
                ordinal_stop=child_ordinal_stop,
                latent_crop_length=int(dataset_config.get("latent_crop_length", 432)),
                caption_max_tokens=int(dataset_config.get("caption_max_tokens", 512)),
                random_crop=bool(dataset_config.get("random_crop", False)),
                require_frozen=bool(dataset_config.get("require_complete", True)),
                speech_timing_index_path=dataset_config.get(
                    "speech_timing_index_path"
                ),
                speech_timing_index_sha256=dataset_config.get(
                    "speech_timing_index_sha256"
                ),
                expected_speech_timing_rows=(
                    int(dataset_config["expected_speech_timing_rows"])
                    if dataset_config.get("expected_speech_timing_rows") is not None
                    else None
                ),
                require_speech_timing=bool(
                    dataset_config.get("require_speech_timing", False)
                ),
                semantic_caption_mode=str(
                    dataset_config.get("semantic_caption_mode", "v2")
                ),
                semantic_caption_v2_probability=float(
                    dataset_config.get("semantic_caption_v2_probability", 1.0)
                ),
                semantic_caption_seed=int(
                    dataset_config.get("semantic_caption_seed", 20260830)
                ),
            )
            children.append(child)
            child_total += len(child)
        if child_total != int(expected_num_samples):
            raise RuntimeError(
                "ScenePlan-v2 frozen index rows do not sum to expected_num_samples: "
                f"{child_total} != {expected_num_samples}"
            )
        train_set = (
            children[0]
            if len(children) == 1
            else torch.utils.data.ConcatDataset(children)
        )
        bucket_config = dict(dataset_config.get("length_bucket_batching") or {})
        if bool(bucket_config.get("enabled", False)):
            world_size = int(distributed_world_size or 1)
            rank = int(distributed_rank or 0)
            long_batch_size = int(
                bucket_config.get(
                    "long_batch_size",
                    max(1, int(batch_size) * 432 // 648),
                )
            )
            batch_sampler = DistributedScenePlanBucketBatchSampler(
                train_set,
                short_batch_size=int(batch_size),
                long_batch_size=long_batch_size,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                seed=int(bucket_config.get("seed", 0)),
                drop_last=bool(dataset_config.get("drop_last", True)),
                semantic_epoch_resume_migration=str(
                    dataset_config.get(
                        "semantic_epoch_resume_migration", "forbid"
                    )
                ),
            )
            loader_kwargs = _local_dataloader_kwargs(
                dataset_config, num_workers
            )
            loader_kwargs.pop("drop_last")
            loader_kwargs["collate_fn"] = sceneplan_bucket_collation
            loader = ResumableDataLoader(
                train_set,
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )
            loader.sceneplan_rank_aware_bucket_sampler = True
            loader.sceneplan_bucket_summary = {
                "rank": rank,
                "world_size": world_size,
                "short_batch_size": int(batch_size),
                "long_batch_size": long_batch_size,
                "bucket_counts": {
                    str(bucket): len(rows)
                    for bucket, rows in batch_sampler.bucket_indices.items()
                },
                "batches_per_epoch": len(batch_sampler),
            }
            return loader

        # P10 is iteration-based and a full epoch is much longer than the
        # checkpoint cadence.  A regular DataLoader restores model/optimizer
        # state but silently restarts the current epoch at row zero.  Persist
        # the consumed-batch cursor so same-run checkpoints resume at the next
        # unconsumed ScenePlan on every DDP rank.
        return ResumableDataLoader(
            train_set,
            batch_size,
            shuffle=shuffle,
            **_local_dataloader_kwargs(dataset_config, num_workers),
        )

    elif dataset_type == "sceneplan_p11_single_turn":
        from stable_audio_tools.data.resumable_dataloader import (
            ResumableDataLoader,
        )
        p11_contract = str(dataset_config.get("p11_contract", "discrete_d0_v1"))
        if p11_contract == "audio_aware_sketch_first_transfusion_cot_v2":
            from stable_audio_tools.data.sceneplan_p11_v4_dataset import (
                ScenePlanP11AudioAwareDataset as P11DatasetClass,
            )
        elif p11_contract == "sketch_first_transfusion_cot_v4":
            from stable_audio_tools.data.sceneplan_p11_v4_dataset import (
                ScenePlanP11V4Dataset as P11DatasetClass,
            )
        elif p11_contract == "discrete_d0_v1":
            from stable_audio_tools.data.sceneplan_p11_dataset import (
                ScenePlanP11Dataset as P11DatasetClass,
            )
        else:
            raise ValueError(f"unsupported P11 dataset contract {p11_contract!r}")

        datasets = dataset_config.get("datasets")
        if not isinstance(datasets, list) or len(datasets) != 1:
            raise ValueError(
                "sceneplan_p11_single_turn requires exactly one frozen source index"
            )
        index_path = datasets[0].get("path")
        manifest_path = dataset_config.get("manifest_path")
        codec_path = dataset_config.get("codec_path")
        expected_num_samples = dataset_config.get("expected_num_samples")
        index_num_samples = dataset_config.get("index_num_samples")
        if not all(
            isinstance(value, str) and value
            for value in (index_path, manifest_path, codec_path)
        ):
            raise ValueError("P11 dataset requires index, manifest, and codec paths")
        if expected_num_samples is None or index_num_samples is None:
            raise ValueError(
                "P11 dataset requires expected_num_samples and index_num_samples"
            )
        if tokenizers is None or "prompt" not in tokenizers:
            raise ValueError("P11 dataset requires the model Qwen tokenizer")
        extra_p11_kwargs = (
            {
                "lexical_evidence_mode": dataset_config.get(
                    "lexical_evidence_mode", "none"
                ),
                "lexical_max_tokens": int(
                    dataset_config.get("lexical_max_tokens", 128)
                ),
                "lexical_cache_path": dataset_config.get(
                    "lexical_cache_path"
                ),
                "lexical_encoder_revision": dataset_config.get(
                    "lexical_encoder_revision"
                ),
                "lexical_confidence_threshold": dataset_config.get(
                    "lexical_confidence_threshold"
                ),
            }
            if p11_contract in {
                "sketch_first_transfusion_cot_v4",
                "audio_aware_sketch_first_transfusion_cot_v2",
            }
            else {}
        )
        train_set = P11DatasetClass(
            manifest_path,
            index_path=index_path,
            codec_path=codec_path,
            tokenizer_spec=tokenizers["prompt"],
            expected_num_samples=int(expected_num_samples),
            index_num_samples=int(index_num_samples),
            require_frozen=bool(dataset_config.get("require_complete", True)),
            semantic_cache_path=dataset_config.get("semantic_cache_path"),
            semantic_dim=int(dataset_config.get("semantic_dim", 512)),
            semantic_encoder_revision=dataset_config.get(
                "semantic_encoder_revision"
            ),
            **extra_p11_kwargs,
        )
        curriculum_path = dataset_config.get("p11_v4_curriculum_path")
        if curriculum_path is not None:
            from stable_audio_tools.data.sceneplan_p11_v4_curriculum import (
                ScenePlanP11V4CurriculumDataset,
            )

            curriculum_rows = dataset_config.get(
                "p11_v4_curriculum_expected_rows"
            )
            if not isinstance(curriculum_path, str) or not curriculum_path:
                raise ValueError("P11 curriculum path must be a non-empty string")
            if curriculum_rows is None:
                raise ValueError("P11 curriculum requires an expected row count")
            train_set = ScenePlanP11V4CurriculumDataset(
                train_set,
                curriculum_path,
                expected_rows=int(curriculum_rows),
                expected_contract=str(
                    dataset_config.get(
                        "p11_v4_curriculum_contract",
                        "p10_v11_train_only_gue_multitarget_v1",
                    )
                ),
                expected_ordering_contract=dataset_config.get(
                    "p11_v4_curriculum_ordering_contract"
                ),
                expected_ordering_batch_size=dataset_config.get(
                    "p11_v4_curriculum_ordering_batch_size"
                ),
            )
        sampler = None
        world_size = int(distributed_world_size or 1)
        rank = int(distributed_rank or 0)
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError(
                f"invalid P11 distributed sampler rank/world: {rank}/{world_size}"
            )
        ordering_contract = dataset_config.get(
            "p11_v4_curriculum_ordering_contract"
        )
        if curriculum_path is not None and ordering_contract is not None:
            from stable_audio_tools.data.sceneplan_p11_ordered_sampler import (
                DistributedP11OrderedBatchSampler,
            )

            ordering_batch_size = int(
                dataset_config.get("p11_v4_curriculum_ordering_batch_size", -1)
            )
            ordering_world_size = int(
                train_set.metadata.get("ordering_world_size", -1)
            )
            ordering_global_batch_size = int(
                train_set.metadata.get("ordering_global_batch_size", -1)
            )
            curriculum_sha256 = str(
                dataset_config.get("p11_v4_curriculum_sha256", "")
            ).strip().lower()
            if curriculum_sha256.startswith("sha256:"):
                curriculum_sha256 = curriculum_sha256.removeprefix("sha256:")
            if (
                len(curriculum_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in curriculum_sha256
                )
            ):
                raise ValueError(
                    "ordered P11 curriculum requires its frozen "
                    "p11_v4_curriculum_sha256"
                )
            if ordering_batch_size != int(batch_size):
                raise ValueError(
                    "P11 ordered curriculum local batch size differs from runtime"
                )
            if ordering_world_size != world_size:
                raise ValueError(
                    "P11 ordered curriculum world size differs from runtime"
                )
            if ordering_global_batch_size != int(batch_size) * world_size:
                raise ValueError(
                    "P11 ordered curriculum global batch size differs from runtime"
                )
            if train_set.metadata.get("distributed_sampler_contract") != (
                "strided_shuffle_false_drop_last_false_v1"
            ):
                raise ValueError("P11 ordered curriculum sampler contract changed")
            batch_sampler = DistributedP11OrderedBatchSampler(
                train_set,
                batch_size=int(batch_size),
                num_replicas=world_size,
                rank=rank,
                dataset_fingerprint=f"sha256:{curriculum_sha256}",
            )
            loader_kwargs = _local_dataloader_kwargs(
                dataset_config, num_workers
            )
            loader_kwargs.pop("drop_last")
            loader = ResumableDataLoader(
                train_set,
                batch_sampler=batch_sampler,
                **loader_kwargs,
            )
            loader.sceneplan_p11_ordered_batch_sampler = True
            loader.sceneplan_p11_ordered_summary = {
                "rank": rank,
                "world_size": world_size,
                "local_batch_size": int(batch_size),
                "global_batch_size": int(batch_size) * world_size,
                "batches_per_epoch": len(batch_sampler),
                "ordering_contract": str(ordering_contract),
                "dataset_fingerprint": f"sha256:{curriculum_sha256}",
            }
            return loader
        if world_size > 1:
            # Lightning forces shuffle=True when it injects a training
            # DistributedSampler. Install our own immutable sampler first so
            # each rank retains the manifest's balanced G/U/E interleave.
            sampler = torch.utils.data.DistributedSampler(
                train_set,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=False,
            )
        return ResumableDataLoader(
            train_set,
            batch_size,
            # The manifest is deterministically pre-shuffled and task-
            # interleaved G/U/E. Non-multiple-of-three batches rotate their
            # 3/3/2 task counts while preserving the exact full-pass ratio.
            # The training wrapper also accepts a rare tail microbatch.
            shuffle=False,
            sampler=sampler,
            **_local_dataloader_kwargs(dataset_config, num_workers),
        )

    elif dataset_type == "pre_encoded":

        pre_encoded_dir_configs = dataset_config.get("datasets", None)

        assert pre_encoded_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        latent_crop_length = dataset_config.get("latent_crop_length", None)
        min_length_sec = dataset_config.get("min_length_sec", None)
        max_length_sec = dataset_config.get("max_length_sec", None)
        random_crop = dataset_config.get("random_crop", False)

        configs = []

        for pre_encoded_dir_config in pre_encoded_dir_configs:
            pre_encoded_dir_path = pre_encoded_dir_config.get("path", None)
            assert pre_encoded_dir_path is not None, "Path must be set for local audio directory configuration"
            

            custom_metadata_fn = None
            custom_metadata_module_path = pre_encoded_dir_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                custom_metadata_fn = _load_custom_metadata_fn(
                    custom_metadata_module_path,
                    pre_encoded_dir_config.get("custom_metadata_config"),
                )

            latent_config = LatentDatasetConfig(
                    id=pre_encoded_dir_config["id"],
                    path=pre_encoded_dir_path,
                    custom_metadata_fn=custom_metadata_fn,
                    latent_extension=pre_encoded_dir_config.get("latent_extension", 'npy'),
                    filelist_path=pre_encoded_dir_config.get("filelist_path", None),
                    validate_filelist_entries=pre_encoded_dir_config.get(
                        "validate_filelist_entries", True
                    ),
                    weight=pre_encoded_dir_config.get("weight", 1.0),
                )
            configs.append(latent_config)

            if (
                dataset_config.get("require_complete", False)
                and not latent_config.validate_filelist_entries
            ):
                roots = (
                    latent_config.path
                    if isinstance(latent_config.path, list)
                    else [latent_config.path]
                )
                for root in roots:
                    _validate_finalized_latent_cache(root)

        train_set = PreEncodedDataset(
            configs,
            latent_crop_length=latent_crop_length,
            min_length_sec=min_length_sec,
            max_length_sec=max_length_sec,
            random_crop=random_crop,
            tokenizers=tokenizers,
            latent_downsampling_ratio=dataset_config.get("latent_downsampling_ratio"),
            sample_rate=sample_rate,
            max_item_retries=dataset_config.get("max_item_retries", 8),
        )

        expected_num_samples = dataset_config.get("expected_num_samples")
        if expected_num_samples is not None and len(train_set) != int(expected_num_samples):
            message = (
                f"pre-encoded dataset is incomplete: found {len(train_set)} complete pairs, "
                f"expected {int(expected_num_samples)}"
            )
            if dataset_config.get("require_complete", False):
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning)

        sampler = _maybe_create_weighted_sampler(train_set)

        return torch.utils.data.DataLoader(
            train_set,
            batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            **_local_dataloader_kwargs(dataset_config, num_workers),
        )

    elif dataset_type == "spatial_family_preencoded":
        from stable_audio_tools.data.resumable_dataloader import (
            ResumableDataLoader,
        )
        from stable_audio_tools.data.spatial_family_dataset import (
            SpatialFamilyDataset,
        )

        stores = []
        for family_config in dataset_config["datasets"]:
            custom_metadata_fn = None
            module_path = family_config.get("custom_metadata_module")
            if module_path is not None:
                custom_metadata_fn = _load_custom_metadata_fn(
                    module_path,
                    family_config.get("custom_metadata_config"),
                )
            paths = family_config.get("path")
            paths = paths if isinstance(paths, list) else [paths]
            stores.extend(
                {
                    "path": path,
                    "family_ranks": family_config.get("family_ranks"),
                    "custom_metadata_fn": custom_metadata_fn,
                    "caption_overlay_path": family_config.get(
                        "caption_overlay_path"
                    ),
                }
                for path in paths
            )
        train_set = SpatialFamilyDataset(
            stores,
            require_ready=bool(dataset_config.get("require_complete", True)),
            max_open_shards=int(dataset_config.get("max_open_shards", 4)),
        )
        expected = dataset_config.get("expected_num_samples")
        if expected is not None and len(train_set) != int(expected):
            message = (
                f"spatial family dataset has {len(train_set)} families; "
                f"expected {int(expected)}"
            )
            if dataset_config.get("require_complete", False):
                raise RuntimeError(message)
            warnings.warn(message, RuntimeWarning)
        return ResumableDataLoader(
            train_set,
            batch_size,
            shuffle=shuffle,
            **_local_dataloader_kwargs(dataset_config, num_workers),
        )

    elif dataset_type in ["s3", "wds"]: # Support "s3" type for backwards compatibility
        wds_configs = []

        for wds_config in dataset_config["datasets"]:

            custom_metadata_fn = None
            custom_metadata_module_path = wds_config.get("custom_metadata_module", None)

            if custom_metadata_module_path is not None:
                custom_metadata_fn = dill.dumps(_load_custom_metadata_fn(
                    custom_metadata_module_path,
                    wds_config.get("custom_metadata_config"),
                ))

            if "s3_path" in wds_config:

                wds_configs.append(
                    S3DatasetConfig(
                        id=wds_config["id"],
                        s3_path=wds_config["s3_path"],
                        custom_metadata_fn=custom_metadata_fn,
                        profile=wds_config.get("profile", None),
                    )
                )
            
            elif "path" in wds_config:
                    
                    wds_configs.append(
                        LocalWebDatasetConfig(
                            id=wds_config["id"],
                            path=wds_config["path"],
                            custom_metadata_fn=custom_metadata_fn
                        )
                    )

        return WebDatasetDataLoader(
            wds_configs,
            sample_rate=sample_rate,
            sample_size=sample_size,
            batch_size=batch_size,
            remove_silence=dataset_config.get("remove_silence", False),
            silence_threshold=dataset_config.get("silence_threshold", [0.01, 0.5]),
            max_silence_duration=dataset_config.get("max_silence_duration", 0.25),
            random_crop=dataset_config.get("random_crop", True),
            volume_norm=dataset_config.get("volume_norm", False),
            volume_norm_param=dataset_config.get("volume_norm_param", [-16, 2]),
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=True,
            force_channels=force_channels,
            epoch_steps=dataset_config.get("epoch_steps", 2000),
            pre_encoded=dataset_config.get("pre_encoded", False),
            latent_crop_length=dataset_config.get("latent_crop_length", None),
            min_length_sec=dataset_config.get("min_length_sec", None),
            max_length_sec=dataset_config.get("max_length_sec", None),
            resampled_shards=dataset_config.get("resampled_shards", True),
            strip_silence=dataset_config.get("strip_silence", False),
        ).data_loader
