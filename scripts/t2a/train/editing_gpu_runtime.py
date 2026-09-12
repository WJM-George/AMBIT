"""Explicit GPU selection and leases for new Editing CLAP/AR jobs.

GPU indices are supplied by the caller, never inferred from idle devices.
The frozen DiT job's allocation is read from its contract only to respect
its existing resource lock; it does not restrict the caller's GPU choices.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess


def selected_gpus(value=None):
    if value is None:
        value = os.environ.get("EDITING_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    fields = str(value).split(",")
    if not fields or any(not item.strip().isascii() or not item.strip().isdigit() for item in fields):
        raise ValueError("set EDITING_GPUS to an explicit comma-separated list of physical GPU indices")
    indices = [int(item.strip()) for item in fields]
    if len(set(indices)) != len(indices):
        raise ValueError("GPU indices must be unique")
    return indices


def gpu_topology(indices=None):
    indices = selected_gpus() if indices is None else selected_gpus(",".join(map(str, indices)))
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id,uuid,name",
                                   "--format=csv,noheader,nounits"], text=True)
    rows = {}
    for line in raw.splitlines():
        index, bus, uuid, name = [field.strip() for field in line.split(",", 3)]
        rows[int(index)] = {"index": int(index), "pci_bus_id": bus.upper(), "uuid": uuid, "name": name}
    if any(index not in rows for index in indices):
        raise ValueError("a requested physical GPU does not exist")
    devices = [rows[index] for index in indices]
    if len({row["uuid"] for row in devices}) != len(indices):
        raise RuntimeError("GPU UUID mapping is not one-to-one")
    return {"cuda_device_order": "PCI_BUS_ID", "physical_indices": indices, "devices": devices,
            "mapping": [{"local_rank": rank, "physical_index": row["index"],
                         "pci_bus_id": row["pci_bus_id"], "uuid": row["uuid"]}
                        for rank, row in enumerate(devices)]}


def configure_visibility(topology):
    os.environ["EDITING_GPUS"] = ",".join(map(str, topology["physical_indices"]))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # UUIDs avoid relying on CUDA enumeration matching nvidia-smi indices.
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(row["uuid"] for row in topology["devices"])


def resource_paths(topology):
    root = Path(os.environ.get("EDITING_DATA_ROOT", "/mnt/sdb/audio_dataset/sceneplan_transfusion_editing_v1"))
    directory = Path(os.environ.get("EDITING_GPU_LOCK_DIR", str(root / "materialized/locks")))
    paths = [directory / (row["uuid"] + ".lock") for row in topology["devices"]]
    contract = Path(os.environ.get("EDITING_DIT_RUN_CONTRACT",
        "/mnt/sdb/model_archives/transfusion_editing/mainline/sceneplan_transfusion_editing_dit_full_seed42_v1/TRAIN_RUN_CONTRACT.json"))
    if contract.exists():
        allocation = json.loads(contract.read_text())["training"]["physical_gpus"]
        if set(allocation).intersection(topology["physical_indices"]):
            paths.append(root / "materialized/locks/training-chain.lock")
    return sorted(set(paths))


@contextmanager
def gpu_lease(topology):
    handles = []
    try:
        for path in resource_paths(topology):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a")
            handles.append(handle)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"requested GPU resource is reserved: {path}") from None
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                          "--format=csv,noheader,nounits"], text=True)
        uuids = {row["uuid"] for row in topology["devices"]}
        if any(line.split(",", 1)[0].strip() in uuids for line in active.splitlines()):
            raise RuntimeError("a requested GPU already has a compute process; leave it running")
        yield [{"pid": os.getpid(), "fd": handle.fileno(), "path": str(Path(handle.name).resolve())}
               for handle in handles]
    finally:
        for handle in reversed(handles):
            handle.close()


def verify_launcher_leases(topology):
    leases = json.loads(os.environ.get("EDITING_GPU_LEASES", "[]"))
    expected = {str(path.resolve()) for path in resource_paths(topology)}
    if {record["path"] for record in leases} != expected:
        raise RuntimeError("use the Editing GPU launcher so every selected GPU is leased")
    for record in leases:
        path = Path(record["path"]).stat()
        fd = Path(f"/proc/{int(record['pid'])}/fd/{int(record['fd'])}")
        held = fd.stat()
        info = Path(f"/proc/{int(record['pid'])}/fdinfo/{int(record['fd'])}")
        if ((held.st_dev, held.st_ino) != (path.st_dev, path.st_ino) or
                not any(line.startswith("lock:") and "FLOCK" in line and "WRITE" in line
                        for line in info.read_text().splitlines())):
            raise RuntimeError("Editing launcher lost its GPU resource lease")
    return leases


def distributed(*, timeout_seconds=None):
    import torch
    from torch import distributed as dist
    topology = gpu_topology()
    verify_launcher_leases(topology)
    if torch.cuda.is_initialized():
        raise RuntimeError("select Editing GPUs before initializing CUDA")
    configure_visibility(topology)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world != len(topology["devices"]) or rank != local_rank or not 0 <= rank < world:
        raise RuntimeError("torchrun ranks must match the explicitly selected GPU list")
    if not torch.cuda.is_available() or torch.cuda.device_count() != world:
        raise RuntimeError("the requested GPU allocation is not visible to CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    options = {}
    if timeout_seconds is not None:
        from datetime import timedelta
        if timeout_seconds <= 0:
            raise ValueError("distributed timeout must be positive")
        options["timeout"] = timedelta(seconds=timeout_seconds)
    dist.init_process_group("nccl", device_id=device, **options)
    return rank, local_rank, world, device, topology


def launch_command(command, count):
    if not command or count < 1:
        raise ValueError("a command and at least one GPU are required")
    return [part.replace("{gpu_count}", str(count)) for part in command]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    topology = gpu_topology(selected_gpus(args.gpus))
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    command = launch_command(command, len(topology["devices"]))
    if args.dry_run:
        print(json.dumps({"physical_gpus": topology["physical_indices"], "command": command,
                          "locks": [str(path) for path in resource_paths(topology)]}))
        return 0
    configure_visibility(topology)
    with gpu_lease(topology) as leases:
        os.environ["EDITING_GPU_LEASES"] = json.dumps(leases)
        child = subprocess.Popen(command, start_new_session=True)
        previous = {}
        def forward(signum, frame):
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous[signum] = signal.signal(signum, forward)
            return child.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
