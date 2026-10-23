"""Read a local inference manifest without importing model dependencies."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


@dataclass(frozen=True)
class Bundle:
    root: Path
    task: str
    components: dict
    codec: Path
    text_encoder: Path
    steps: int
    cfg_scale: float

    @classmethod
    def open(cls, directory, *, task: str):
        root = Path(directory).expanduser().resolve(strict=True)
        spec = read_json(root / "bundle.json")
        if spec.get("format") != "ambit-inference-v1":
            raise ValueError("Expected format 'ambit-inference-v1' in bundle.json")
        if task not in {"generation", "editing"} or spec.get("task") != task:
            raise ValueError(f"This command requires a {task} bundle")

        def resolve(value, *, directory=False):
            if not isinstance(value, str) or not value or Path(value).is_absolute():
                raise ValueError("Bundle paths must be relative to bundle.json")
            path = (root / value).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("Bundle paths must stay inside the bundle directory")
            if not (path.is_dir() if directory else path.is_file()):
                raise ValueError(f"Invalid bundle artifact: {value}")
            return path

        required = {"renderer", "vae", "ar"}
        required |= {"copy", "qualitative", "inventory"} if task == "generation" else {"clap"}
        raw = spec.get("components")
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError(f"Required components: {', '.join(sorted(required))}")
        components = {}
        for name, item in raw.items():
            if not isinstance(item, dict) or set(item) != {"config", "weights"}:
                raise ValueError(f"{name} requires exactly config and weights")
            config = read_json(resolve(item["config"]))
            weights = resolve(item["weights"])
            if weights.suffix != ".safetensors":
                raise ValueError(f"{name} requires a .safetensors state dictionary")
            components[name] = {"config": config, "weights": weights}
        sampling = spec.get("sampling", {})
        if not isinstance(sampling, dict) or set(sampling) - {"steps", "cfg_scale"}:
            raise ValueError("sampling accepts only steps and cfg_scale")
        steps = sampling.get("steps", 100 if task == "generation" else 20)
        cfg = sampling.get("cfg_scale", 3.0 if task == "generation" else 1.0)
        validate_sampling(steps, cfg)
        return cls(
            root,
            task,
            components,
            resolve(spec.get("codec"), directory=True),
            resolve(spec.get("text_encoder"), directory=True),
            steps,
            float(cfg),
        )


def validate_sampling(steps, cfg_scale):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if (
        isinstance(cfg_scale, bool)
        or not isinstance(cfg_scale, (int, float))
        or (not math.isfinite(cfg_scale))
        or (cfg_scale < 0)
    ):
        raise ValueError("cfg_scale must be finite and nonnegative")


def validate_request(prompt, seed):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if isinstance(seed, bool) or not isinstance(seed, int) or (not 0 <= seed < 2**63):
        raise ValueError("seed must be an integer in [0, 2**63)")
