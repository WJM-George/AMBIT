"""Native first-order ambisonic audio I/O (ACN/SN3D, W Y Z X)."""

from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
MAX_SAMPLES = 648 * 1024


def validate_foa(audio):
    value = np.asarray(audio, dtype=np.float32)
    if value.ndim != 2 or value.shape[0] != 4:
        raise ValueError("Expected FOA audio shaped [4, samples] in W Y Z X order")
    if not 1 <= value.shape[1] <= MAX_SAMPLES:
        raise ValueError(f"Audio must contain 1..{MAX_SAMPLES} samples")
    if not np.isfinite(value).all():
        raise ValueError("Audio contains non-finite samples")
    return value


def read_foa(path):
    with sf.SoundFile(path) as source:
        if source.samplerate != SAMPLE_RATE or source.channels != 4:
            raise ValueError("Editing requires a 44,100 Hz, four-channel FOA file")
        if not 1 <= len(source) <= MAX_SAMPLES:
            raise ValueError(f"Audio must contain 1..{MAX_SAMPLES} samples")
        return validate_foa(source.read(dtype="float32", always_2d=True).T)


def output_paths(path):
    wav = Path(path).expanduser()
    if wav.suffix.lower() != ".wav":
        raise ValueError("Output must end in .wav")
    plan = wav.with_suffix(".json")
    for target in (wav, plan):
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Output already exists: {target}")
    return (wav, plan)


@dataclass
class InferenceResult:
    audio: np.ndarray
    sceneplan: dict
    task: str
    prompt: str
    seed: int

    def save(self, path):
        audio = validate_foa(self.audio)
        (wav, plan) = output_paths(path)
        payload = (
            json.dumps(
                {
                    "task": self.task,
                    "prompt": self.prompt,
                    "seed": self.seed,
                    "sample_rate": SAMPLE_RATE,
                    "channel_order": "WYZX",
                    "normalization": "SN3D",
                    "sceneplan": self.sceneplan,
                },
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        wav.parent.mkdir(parents=True, exist_ok=True)
        created = []
        try:
            with wav.open("xb") as stream:
                created.append(wav)
                sf.write(stream, audio.T, SAMPLE_RATE, format="WAV", subtype="FLOAT")
            with plan.open("x", encoding="utf-8") as stream:
                created.append(plan)
                stream.write(payload)
        except BaseException:
            for target in created:
                target.unlink(missing_ok=True)
            raise
        return (wav, plan)
