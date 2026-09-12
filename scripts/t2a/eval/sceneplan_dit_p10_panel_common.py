"""Shared helpers for frozen, matched P10 checkpoint panels."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import torch
import torchaudio


DEFAULT_EVAL_ROOT = Path(
    "/mnt/sdb/model_archives/p10_pre_v11_20260831/sceneplan_dit_fail/"
    "sceneplan_dit_v4_r8_300m/evaluation/"
    "p10_ckpt_5k_10k_15k_sceneplan44_v1"
)
CHECKPOINT_STEPS = (5_000, 10_000, 15_000)
DOMAINS = ("music", "sound", "speech")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def load_panel(eval_root: Path) -> list[dict[str, Any]]:
    root = eval_root.expanduser().resolve(strict=True)
    contract = json.loads((root / "EVAL_CONTRACT.json").read_text(encoding="utf-8"))
    test_set = contract["test_set"]
    expected_rows = int(test_set["evaluation_rows"])
    panel_filename = str(test_set.get("panel_filename", "listening_panel_15.jsonl"))
    if Path(panel_filename).name != panel_filename:
        raise RuntimeError(f"unsafe P10 panel filename: {panel_filename}")
    panel_path = root / panel_filename
    if test_set.get("panel_sha256") and sha256_file(panel_path) != test_set["panel_sha256"]:
        raise RuntimeError(f"P10 panel SHA256 changed: {panel_path}")
    rows = read_jsonl(panel_path)
    if len(rows) != expected_rows:
        raise RuntimeError(f"P10 panel changed: {len(rows)} != {expected_rows}")
    expected_counts = {
        str(domain): int(count)
        for domain, count in test_set["domain_counts"].items()
    }
    counts = {
        domain: sum(str(row["domain"]) == domain for row in rows)
        for domain in expected_counts
    }
    if counts != expected_counts:
        raise RuntimeError(f"P10 panel domain counts changed: {counts}")
    unexpected_domains = sorted(
        {str(row["domain"]) for row in rows}.difference(expected_counts)
    )
    if unexpected_domains:
        raise RuntimeError(
            f"P10 panel contains uncontracted domains: {unexpected_domains}"
        )
    if len({str(row["panel_id"]) for row in rows}) != len(rows):
        raise RuntimeError("P10 panel IDs are not unique")
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise RuntimeError("P10 panel sample IDs are not unique")
    return rows


def checkpoint_steps(eval_root: Path) -> tuple[int, ...]:
    root = eval_root.expanduser().resolve(strict=True)
    contract = json.loads((root / "EVAL_CONTRACT.json").read_text(encoding="utf-8"))
    steps = tuple(int(row["step"]) for row in contract.get("checkpoints", []))
    if not steps or len(set(steps)) != len(steps):
        raise RuntimeError(f"invalid checkpoint list in {root / 'EVAL_CONTRACT.json'}")
    return steps


def load_output_rows(eval_root: Path) -> list[dict[str, Any]]:
    root = eval_root.expanduser().resolve(strict=True)
    panel = load_panel(root)
    steps = checkpoint_steps(root)
    output = []
    for step in steps:
        for panel_row in panel:
            path = (
                root
                / "outputs"
                / f"step_{step:06d}"
                / panel_row["domain"]
                / panel_row["panel_id"]
                / "metadata.json"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            row = json.loads(path.read_text(encoding="utf-8"))
            if not (
                row.get("status") == "PASS"
                and int(row["checkpoint_step"]) == step
                and row["panel_id"] == panel_row["panel_id"]
                and row["sample_id"] == panel_row["sample_id"]
            ):
                raise RuntimeError(f"generated metadata disagrees with panel: {path}")
            row["metadata_path"] = str(path.resolve())
            output.append(row)
    expected = len(panel) * len(steps)
    if len(output) != expected:
        raise RuntimeError(f"P10 output count changed: {len(output)} != {expected}")
    return output


def load_foa(path: str | Path, *, expected_samples: int | None = None) -> tuple[torch.Tensor, int]:
    resolved = Path(path).expanduser().resolve(strict=True)
    audio, sample_rate = torchaudio.load(str(resolved))
    audio = audio.to(torch.float32)
    if audio.ndim != 2 or int(audio.shape[0]) != 4:
        raise ValueError(f"expected WYZX FOA [4,N] at {resolved}, got {tuple(audio.shape)}")
    if expected_samples is not None and int(audio.shape[-1]) != int(expected_samples):
        raise ValueError(
            f"sample count changed at {resolved}: {audio.shape[-1]} != {expected_samples}"
        )
    if not bool(torch.isfinite(audio).all()):
        raise ValueError(f"non-finite FOA at {resolved}")
    return audio, int(sample_rate)


def finite_mean(values: Iterable[float | int | None]) -> float | None:
    tensors = [float(value) for value in values if value is not None]
    if not tensors:
        return None
    return float(sum(tensors) / len(tensors))


def summarize(values: Iterable[float | int | None]) -> dict[str, Any]:
    kept = torch.tensor(
        [float(value) for value in values if value is not None], dtype=torch.float64
    )
    if not int(kept.numel()):
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "std": None,
            "mean_ci95_low": None,
            "mean_ci95_high": None,
            "mean_ci95_method": None,
        }
    mean = kept.mean()
    std = kept.std(unbiased=True) if int(kept.numel()) > 1 else None
    if int(kept.numel()) > 1:
        generator = torch.Generator(device="cpu").manual_seed(20260824)
        indices = torch.randint(
            int(kept.numel()),
            (10_000, int(kept.numel())),
            generator=generator,
        )
        bootstrap_means = kept[indices].mean(dim=1)
        ci_low, ci_high = torch.quantile(
            bootstrap_means, torch.tensor([0.025, 0.975], dtype=torch.float64)
        )
    else:
        ci_low = ci_high = mean
    return {
        "count": int(kept.numel()),
        "mean": float(mean),
        "median": float(kept.median()),
        "min": float(kept.min()),
        "max": float(kept.max()),
        "std": None if std is None else float(std),
        "mean_ci95_low": float(ci_low),
        "mean_ci95_high": float(ci_high),
        "mean_ci95_method": "deterministic percentile bootstrap, 10000 resamples",
    }


__all__ = [
    "CHECKPOINT_STEPS",
    "DEFAULT_EVAL_ROOT",
    "DOMAINS",
    "atomic_json",
    "checkpoint_steps",
    "finite_mean",
    "load_foa",
    "load_output_rows",
    "load_panel",
    "read_jsonl",
    "sha256_file",
    "summarize",
]
