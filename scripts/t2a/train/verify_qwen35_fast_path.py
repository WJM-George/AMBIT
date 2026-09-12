#!/usr/bin/env python3
"""Fail closed unless the frozen Qwen3.5 training fast path is available."""

from __future__ import annotations

import importlib.metadata
import json

from transformers.models.qwen3_5 import modeling_qwen3_5


EXPECTED = {
    "causal_conv1d_version": "1.7.0",
    "fast_path_available": True,
    "flash_linear_attention_version": "0.5.2",
}


def main() -> int:
    receipt = {
        "causal_conv1d_version": importlib.metadata.version("causal-conv1d"),
        "fast_path_available": bool(modeling_qwen3_5.is_fast_path_available),
        "flash_linear_attention_version": importlib.metadata.version(
            "flash-linear-attention"
        ),
    }
    print(
        "SAT_QWEN35_FAST_PATH=" + json.dumps(receipt, sort_keys=True),
        flush=True,
    )
    if receipt != EXPECTED:
        raise RuntimeError(
            f"Qwen3.5 fast-path contract changed: expected={EXPECTED}, got={receipt}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
