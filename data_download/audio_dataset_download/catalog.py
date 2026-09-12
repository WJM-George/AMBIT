from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DownloadKind = Literal[
    "hf_snapshot", "hf_parquet", "apple_spatial_librispeech", "github_repo", "zenodo",
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    kind: DownloadKind
    description: str
    repo_id: str | None = None
    zenodo_record_id: str | None = None
    default_config: str = "default"
    gated: bool = False
    note: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "mrsdrama": DatasetSpec(
        key="mrsdrama",
        name="MRSDrama",
        kind="hf_snapshot",
        repo_id="AaronZ345/MRSDrama",
        default_config="meta",
        description="Real-recorded binaural drama dataset with scripts, video, and geometry labels.",
        note="Snapshot mode preserves the repository layout; split metadata can also be queried with --mode parquet.",
    ),
    "bewo_1m": DatasetSpec(
        key="bewo_1m",
        name="BEWO-1M",
        kind="hf_snapshot",
        repo_id="spw2000/BEWO-1M",
        description="Simulated spatial audio dataset with stereo audio, text, image, and room metadata.",
    ),
    "mrsaudio": DatasetSpec(
        key="mrsaudio",
        name="MRSAudio",
        kind="hf_snapshot",
        repo_id="MRSAudio/MRSAudio",
        description="Large real-recorded multimodal spatial audio dataset with multiple subsets.",
    ),
    "sphere360": DatasetSpec(
        key="sphere360",
        name="Sphere360",
        kind="hf_snapshot",
        repo_id="omniaudio/Sphere360",
        gated=True,
        description="360-degree video and FOA audio dataset.",
        note="This dataset may require Hugging Face login and license acceptance.",
    ),
    "audiox_ifcaps": DatasetSpec(
        key="audiox_ifcaps",
        name="AudioX-IFcaps",
        kind="hf_snapshot",
        repo_id="HKUSTAudio/AudioX-IFcaps",
        description="Large multimodal instruction-following audio caption dataset.",
    ),
    "audiocaps": DatasetSpec(
        key="audiocaps",
        name="AudioCaps",
        kind="hf_snapshot",
        repo_id="OpenSound/AudioCaps",
        description="Audio caption dataset referenced in the provided download notes.",
    ),
    "audio_flan": DatasetSpec(
        key="audio_flan",
        name="Audio-FLAN-Dataset",
        kind="hf_snapshot",
        repo_id="HKUSTAudio/Audio-FLAN-Dataset",
        gated=True,
        description="Large unified instruction tuning dataset for speech, music, and general audio.",
        note="This dataset is gated; set HF_TOKEN or run hf auth login after accepting access on Hugging Face.",
    ),
    "spatial_librispeech": DatasetSpec(
        key="spatial_librispeech",
        name="Spatial LibriSpeech",
        kind="apple_spatial_librispeech",
        description="Apple Spatial LibriSpeech FOA samples and metadata.",
        note="Downloader supports metadata plus selected sample ranges to avoid accidental multi-terabyte pulls.",
    ),
    "yt_ambient": DatasetSpec(
        key="yt_ambient",
        name="YT-Ambient / ViSAGe",
        kind="github_repo",
        repo_id="https://github.com/jaeyeonkim99/visageViSAGe.git",
        description="Repository for the YT-Ambient / ViSAGe dataset and tooling.",
        note="The supplied URL is a GitHub project, not a direct Hugging Face dataset. This clones the repo and leaves dataset-specific fetching to the upstream instructions.",
    ),
    "audioset": DatasetSpec(
        key="audioset",
        name="AudioSet (wav)",
        kind="hf_snapshot",
        repo_id="agkphysics/AudioSet",
        description="~2.4 TB of 10-second AudioSet clips (YouTube-sourced wav, March 2023 snapshot).",
        note="Stored on AUDIO_DATASET_SECONDARY_ROOT (/mnt/sdb) by default. Requires ~2.4 TB free.",
    ),
    "vggsound": DatasetSpec(
        key="vggsound",
        name="VGGSound",
        kind="hf_snapshot",
        repo_id="Loie/VGGSound",
        description="~338 GB VGGSound tar.gz archives plus vggsound.csv.",
    ),
    "musiccaps": DatasetSpec(
        key="musiccaps",
        name="MusicCaps",
        kind="hf_snapshot",
        repo_id="google/MusicCaps",
        description="5,521 music-caption rows (csv only). Fetch wav from YouTube separately.",
        note="HF snapshot is metadata only (~3 MB). Audio download is not included.",
    ),
    "picoaudio": DatasetSpec(
        key="picoaudio",
        name="PicoAudio",
        kind="hf_snapshot",
        repo_id="amphion/PicoAudio",
        description="Controllable spatial audio simulation dataset (~912 MB zip).",
    ),
    "fsdkaggle2019": DatasetSpec(
        key="fsdkaggle2019",
        name="FSDKaggle2019",
        kind="zenodo",
        zenodo_record_id="3612637",
        description="DCASE 2019 Freesound Audio Tagging (~27 GB, 29,266 clips, 80 classes).",
        note="Zenodo record 3612637. Use --extract to unpack archives after download.",
    ),
}

