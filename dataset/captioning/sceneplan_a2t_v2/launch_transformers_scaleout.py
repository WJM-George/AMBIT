#!/usr/bin/env python3
"""Launch deterministic Transformers annotation shards on disjoint GPU groups."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_GROUPS = "0,1;2,3;4,5;6,7"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-groups", default=DEFAULT_GROUPS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runner_args = args.runner_args
    if runner_args and runner_args[0] == "--":
        runner_args = runner_args[1:]
    return args, runner_args


def main() -> int:
    args, runner_args = parse_args()
    if not runner_args:
        raise ValueError("pass caption_transformers.py arguments after --")
    groups = [group.strip() for group in args.gpu_groups.split(";") if group.strip()]
    if not groups or any(
        not all(part.isdigit() for part in group.split(",")) for group in groups
    ):
        raise ValueError("--gpu-groups must look like 0,1;2,3;4,5;6,7")
    flattened = [gpu for group in groups for gpu in group.split(",")]
    if len(flattened) != len(set(flattened)):
        raise ValueError("GPU groups overlap")

    runner = Path(__file__).with_name("caption_transformers.py").resolve(strict=True)
    log_dir = (
        args.log_dir or Path.cwd() / "a2t_transformers_logs"
    ).expanduser().resolve(strict=False)
    commands: list[tuple[list[str], dict[str, str], Path]] = []
    for shard, group in enumerate(groups):
        command = [
            sys.executable,
            str(runner),
            *runner_args,
            "--num-shards",
            str(len(groups)),
            "--shard",
            str(shard),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = group
        environment["PATH"] = (
            str(Path(sys.executable).parent)
            + os.pathsep
            + environment.get("PATH", "")
        )
        environment.setdefault("TOKENIZERS_PARALLELISM", "false")
        log_path = log_dir / (
            f"instruct_transformers.shard{shard:03d}-of-{len(groups):03d}.log"
        )
        commands.append((command, environment, log_path))

    for command, environment, log_path in commands:
        rendered = (
            f"CUDA_VISIBLE_DEVICES={environment['CUDA_VISIBLE_DEVICES']} "
            + shlex.join(command)
        )
        print(f"{rendered} > {log_path} 2>&1", flush=True)
    if args.dry_run:
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[subprocess.Popen[str], object, Path]] = []
    for command, environment, log_path in commands:
        log_handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((process, log_handle, log_path))

    failures: list[tuple[int, int, Path]] = []
    pending = set(range(len(processes)))
    try:
        while pending:
            for shard in list(pending):
                process, _, log_path = processes[shard]
                return_code = process.poll()
                if return_code is None:
                    continue
                pending.remove(shard)
                if return_code:
                    failures.append((shard, return_code, log_path))
            if failures and pending:
                for shard in pending:
                    process, _, _ = processes[shard]
                    process.terminate()
                for shard in list(pending):
                    process, _, log_path = processes[shard]
                    return_code = process.wait(timeout=30)
                    pending.remove(shard)
                    if return_code:
                        failures.append((shard, return_code, log_path))
                break
            if pending:
                time.sleep(1.0)
    except (KeyboardInterrupt, subprocess.TimeoutExpired):
        for shard in pending:
            process, _, _ = processes[shard]
            process.terminate()
        for shard in pending:
            process, _, _ = processes[shard]
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for _, log_handle, _ in processes:
            log_handle.close()
    if failures:
        for shard, return_code, log_path in failures:
            print(
                f"FAILED shard={shard} return_code={return_code} log={log_path}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
