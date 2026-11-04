"""Local, self-contained Unified AMBIT inference assets."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


@dataclass(frozen=True)
class Bundle:
    root: Path
    config: dict
    task: str
    steps: int
    cfg_scale: float

    def path(self, key):
        value = self.config[key]
        if not isinstance(value, str) or Path(value).is_absolute():
            raise ValueError("Model paths must be relative")
        path = (self.root / value).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("Model paths must stay inside the bundle")
        return path

    @classmethod
    def open(cls, directory, *, task):
        root = Path(directory).expanduser().resolve(strict=True)
        spec = read_json(root / "bundle.json")
        if spec.get("format") != "ambit-unified-v2":
            raise ValueError("Expected an ambit-unified-v2 model bundle")
        if task not in ("generation", "editing"):
            raise ValueError("Unknown inference task")
        s = spec["sampling"][task]
        validate_sampling(s["steps"], s["cfg_scale"])
        obj = cls(root, spec, task, s["steps"], float(s["cfg_scale"]))
        for key in (
            "weights",
            "vae_weights",
            "renderer_config",
            "vae_config",
            "codec",
            "text_encoder",
            "planner_encoder",
        ):
            obj.path(key)
        return obj


def validate_sampling(steps, cfg_scale):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(cfg_scale, bool)
        or not isinstance(cfg_scale, (int, float))
        or not math.isfinite(cfg_scale)
        or cfg_scale < 0
    ):
        raise ValueError("cfg_scale must be finite and nonnegative")


def validate_request(prompt, seed):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
