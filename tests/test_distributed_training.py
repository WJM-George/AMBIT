from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from stable_audio_tools.training.distributed import (
    canonicalize_pci_bus_id,
    discover_local_gpu_numa,
    format_cpu_list,
    parse_cpu_list,
    resolve_ddp_comm_hook,
)


class CpuListTests(unittest.TestCase):
    def test_parse_and_format(self):
        cpus = parse_cpu_list("0-3,8,10-11")
        self.assertEqual(cpus, (0, 1, 2, 3, 8, 10, 11))
        self.assertEqual(format_cpu_list(cpus), "0-3,8,10-11")

    def test_rejects_empty_or_reversed_ranges(self):
        with self.assertRaises(ValueError):
            parse_cpu_list("")
        with self.assertRaises(ValueError):
            parse_cpu_list("4-2")


class NumaDiscoveryTests(unittest.TestCase):
    def test_canonicalizes_nvidia_domain(self):
        self.assertEqual(
            canonicalize_pci_bus_id("00000000:AB:0C.0"),
            "0000:ab:0c.0",
        )

    def test_respects_cuda_visible_device_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            sysfs_root = Path(tmp)
            device_root = sysfs_root / "0000:21:00.0"
            device_root.mkdir()
            (device_root / "numa_node").write_text("2\n")
            (device_root / "local_cpulist").write_text("16-23,80-87\n")

            def fake_run(command, **kwargs):
                self.assertEqual(command[2], "GPU-second")
                return subprocess.CompletedProcess(
                    command, 0, stdout="00000000:21:00.0\n", stderr=""
                )

            binding = discover_local_gpu_numa(
                1,
                environ={"CUDA_VISIBLE_DEVICES": "GPU-first,GPU-second"},
                sysfs_root=sysfs_root,
                run=fake_run,
            )
            self.assertEqual(binding.visible_device, "GPU-second")
            self.assertEqual(binding.pci_bus_id, "0000:21:00.0")
            self.assertEqual(binding.numa_node, 2)
            self.assertEqual(binding.cpu_list, "16-23,80-87")


class DdpHookTests(unittest.TestCase):
    def test_none_and_bf16(self):
        self.assertIsNone(resolve_ddp_comm_hook("none"))
        self.assertEqual(resolve_ddp_comm_hook("bf16").__name__, "bf16_compress_hook")

    def test_rejects_unknown_hook(self):
        with self.assertRaises(ValueError):
            resolve_ddp_comm_hook("mystery")


if __name__ == "__main__":
    unittest.main()
