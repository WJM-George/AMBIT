"""Guard against an NVIDIA error that can fill the root disk with kernel logs."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Optional

_PATTERNS = (
    b"gpuHandleSanityCheckRegReadError",
    b"Possible bad register read",
)


def recent_nvrm_error(
    log_path: Path = Path("/var/log/kern.log"),
    *,
    recent_seconds: Optional[int] = None,
    tail_bytes: int = 512 * 1024,
) -> bool:
    """Inspect only the log tail; do not call an NVIDIA API.

    By default the guard remains latched while the latest kernel-log tail
    contains the failure. This is deliberate: after a quiet period, the next
    CUDA probe can still retrigger the same log storm. Callers that only need
    a time-local diagnostic may pass ``recent_seconds`` explicitly.
    """

    if os.environ.get("SAT_IGNORE_NVRM_PREFLIGHT") == "1":
        return False
    try:
        stat = log_path.stat()
        if (
            recent_seconds is not None
            and time.time() - stat.st_mtime > recent_seconds
        ):
            return False
        with log_path.open("rb") as handle:
            handle.seek(max(0, stat.st_size - tail_bytes))
            tail = handle.read()
    except OSError:
        return False
    return any(pattern in tail for pattern in _PATTERNS)


def assert_gpu_driver_healthy() -> None:
    minimum_root_gib = float(os.environ.get("SAT_MIN_ROOT_FREE_GIB", "5"))
    root_free_gib = shutil.disk_usage("/").free / 2**30
    if minimum_root_gib > 0 and root_free_gib < minimum_root_gib:
        raise RuntimeError(
            f"Root filesystem has only {root_free_gib:.2f} GiB free; "
            f"GPU work requires at least {minimum_root_gib:.2f} GiB of host "
            "headroom. Clean the NVRM log storm first. Set "
            "SAT_MIN_ROOT_FREE_GIB=0 only for an intentional emergency override."
        )
    if recent_nvrm_error():
        raise RuntimeError(
            "NVIDIA bad-register errors are present at the end of "
            "/var/log/kern.log. "
            "Refusing to call nvidia-smi/CUDA because this failure has produced "
            "tens of gigabytes of duplicate system logs. Recover the NVIDIA "
            "driver and rotate/truncate the affected logs first; "
            "SAT_IGNORE_NVRM_PREFLIGHT=1 is an emergency override."
        )
