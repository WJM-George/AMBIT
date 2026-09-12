"""Small, opt-in helpers for single-node distributed training.

The NUMA binding helper intentionally runs before CUDA and the model are
initialized.  DataLoader workers inherit both the CPU affinity and the
preferred-memory policy from their rank process.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence


_PCI_BUS_ID_RE = re.compile(
    r"^(?P<domain>[0-9a-fA-F]{4,8}):"
    r"(?P<bus>[0-9a-fA-F]{2}):"
    r"(?P<device>[0-9a-fA-F]{2})\."
    r"(?P<function>[0-7])$"
)


@dataclass(frozen=True)
class LocalGpuNumaBinding:
    """Resolved CPU locality for one local distributed rank."""

    local_rank: int
    visible_device: str
    pci_bus_id: str
    numa_node: int
    cpus: tuple[int, ...]

    @property
    def cpu_list(self) -> str:
        return format_cpu_list(self.cpus)


def parse_cpu_list(value: str) -> tuple[int, ...]:
    """Parse Linux cpulist syntax such as ``0-3,8,10-11``."""

    cpus: set[int] = set()
    for raw_part in value.strip().split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range {part!r}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"invalid CPU id {part!r}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError(f"empty CPU list {value!r}")
    return tuple(sorted(cpus))


def format_cpu_list(cpus: Sequence[int]) -> str:
    """Format CPU ids compactly using Linux cpulist syntax."""

    ordered = sorted(set(int(cpu) for cpu in cpus))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def canonicalize_pci_bus_id(value: str) -> str:
    """Convert nvidia-smi PCI ids to the canonical Linux sysfs form."""

    match = _PCI_BUS_ID_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid PCI bus id {value!r}")
    domain = int(match.group("domain"), 16)
    if domain > 0xFFFF:
        raise ValueError(f"PCI domain is outside the sysfs range: {value!r}")
    return (
        f"{domain:04x}:{match.group('bus').lower()}:"
        f"{match.group('device').lower()}.{match.group('function')}"
    )


def _visible_device_for_rank(
    local_rank: int, environ: Mapping[str, str]
) -> str:
    visible = environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return str(local_rank)
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not devices:
        raise RuntimeError("CUDA_VISIBLE_DEVICES is empty")
    if local_rank >= len(devices):
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} has no entry in CUDA_VISIBLE_DEVICES={visible!r}"
        )
    device = devices[local_rank]
    if device.startswith("MIG-"):
        raise RuntimeError("MIG devices are not supported by GPU/NUMA binding")
    return device


def _query_pci_bus_id(
    visible_device: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    result = run(
        [
            "nvidia-smi",
            "-i",
            visible_device,
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            f"nvidia-smi returned {len(lines)} PCI ids for {visible_device!r}: "
            f"{result.stdout!r}"
        )
    return canonicalize_pci_bus_id(lines[0])


def discover_local_gpu_numa(
    local_rank: int,
    *,
    environ: Optional[Mapping[str, str]] = None,
    sysfs_root: Path = Path("/sys/bus/pci/devices"),
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> LocalGpuNumaBinding:
    """Resolve the PCI-local NUMA node and CPUs for a local CUDA rank."""

    if local_rank < 0:
        raise ValueError(f"local_rank must be non-negative, got {local_rank}")
    environ = os.environ if environ is None else environ
    visible_device = _visible_device_for_rank(local_rank, environ)
    pci_bus_id = _query_pci_bus_id(visible_device, run=run)
    device_root = sysfs_root / pci_bus_id
    try:
        numa_node = int((device_root / "numa_node").read_text().strip())
        cpus = parse_cpu_list((device_root / "local_cpulist").read_text())
    except FileNotFoundError as error:
        raise RuntimeError(
            f"missing NUMA topology for GPU {visible_device!r} at {device_root}"
        ) from error
    if numa_node < 0:
        raise RuntimeError(
            f"GPU {visible_device!r} ({pci_bus_id}) has no NUMA node"
        )
    return LocalGpuNumaBinding(
        local_rank=local_rank,
        visible_device=visible_device,
        pci_bus_id=pci_bus_id,
        numa_node=numa_node,
        cpus=cpus,
    )


def _set_preferred_memory_node(numa_node: int) -> None:
    library_path = ctypes.util.find_library("numa") or "libnuma.so.1"
    try:
        libnuma = ctypes.CDLL(library_path, use_errno=True)
    except OSError as error:
        raise RuntimeError(
            "libnuma is required for --bind-to-gpu-numa"
        ) from error

    libnuma.numa_available.argtypes = []
    libnuma.numa_available.restype = ctypes.c_int
    libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
    libnuma.numa_set_preferred.restype = None
    if libnuma.numa_available() < 0:
        raise RuntimeError("NUMA is unavailable on this host")
    libnuma.numa_set_preferred(numa_node)


def bind_process_to_local_gpu_numa(
    local_rank: int,
    *,
    prefer_memory: bool = True,
) -> LocalGpuNumaBinding:
    """Pin this rank to its GPU-local CPUs and prefer local host memory."""

    binding = discover_local_gpu_numa(local_rank)
    os.sched_setaffinity(0, set(binding.cpus))
    actual_cpus = tuple(sorted(os.sched_getaffinity(0)))
    if actual_cpus != binding.cpus:
        raise RuntimeError(
            f"CPU affinity mismatch: requested {binding.cpu_list}, "
            f"got {format_cpu_list(actual_cpus)}"
        )
    if prefer_memory:
        _set_preferred_memory_node(binding.numa_node)
    return binding


def resolve_ddp_comm_hook(name: str):
    """Return a supported PyTorch DDP communication hook by config name."""

    normalized = str(name).strip().lower().replace("-", "_")
    if normalized in {"", "none", "off", "false"}:
        return None
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

    hooks = {
        "bf16": default_hooks.bf16_compress_hook,
        "bf16_compress": default_hooks.bf16_compress_hook,
        "fp16": default_hooks.fp16_compress_hook,
        "fp16_compress": default_hooks.fp16_compress_hook,
    }
    try:
        return hooks[normalized]
    except KeyError as error:
        raise ValueError(
            f"unsupported DDP communication hook {name!r}; "
            "choose none, bf16, or fp16"
        ) from error
