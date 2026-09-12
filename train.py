# Disable HuggingFace progress bars BEFORE any imports
# This must be at the very top to take effect
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import ctypes
import hashlib
import json
import faulthandler
import inspect
import random
import resource
import signal
import shutil
import time
from collections import Counter, deque
from datetime import timedelta
from pathlib import Path

faulthandler.enable()
if hasattr(signal, "SIGUSR1"):
    faulthandler.register(signal.SIGUSR1, all_threads=True)

from scripts.t2a.train.gpu_preflight import assert_gpu_driver_healthy

# A broken NVIDIA driver on this host can emit gigabytes of duplicate NVRM
# messages as soon as torch probes CUDA.  Check the kernel-log tail before
# importing torch so a failed launch remains cheap and leaves an actionable
# error.  Set SAT_IGNORE_NVRM_PREFLIGHT=1 only after intentionally accepting
# that risk (for example, after an administrator has recovered the driver).
assert_gpu_driver_healthy()

import numpy as np
import torch
import pytorch_lightning as pl

from typing import Dict, Optional, Union
from prefigure.prefigure import get_all_args, push_wandb_config
from pytorch_lightning.strategies import DDPStrategy
from stable_audio_tools.configuration import load_config, validate_training_configs
from stable_audio_tools.data.dataset import create_dataloader_from_config, fast_scandir
from stable_audio_tools.data.text_conditioning import collect_conditioner_tokenizers
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model
from stable_audio_tools.training import create_training_wrapper_from_config, create_demo_callback_from_config
from stable_audio_tools.training.distributed import (
    bind_process_to_local_gpu_numa,
    resolve_ddp_comm_hook,
)
from stable_audio_tools.training.fsdp import create_fsdp_strategy_and_callback


class _NvmlUtilizationRates(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class NvmlGpuUtilizationSampler:
    """Minimal dependency-free NVML sampler bound to this rank's CUDA UUID."""

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("NVML utilization sampling requires CUDA")
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByUUID.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.nvmlDeviceGetHandleByUUID.restype = ctypes.c_int
        self.library.nvmlDeviceGetUtilizationRates.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_NvmlUtilizationRates),
        ]
        self.library.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        self._check(self.library.nvmlInit_v2(), "nvmlInit_v2")
        device = torch.cuda.current_device()
        uuid = str(torch.cuda.get_device_properties(device).uuid)
        if not uuid.startswith("GPU-"):
            uuid = f"GPU-{uuid}"
        self.handle = ctypes.c_void_p()
        self._check(
            self.library.nvmlDeviceGetHandleByUUID(
                uuid.encode("ascii"), ctypes.byref(self.handle)
            ),
            "nvmlDeviceGetHandleByUUID",
        )

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if int(status) != 0:
            raise RuntimeError(f"{operation} failed with NVML status {status}")

    def sample(self) -> float:
        rates = _NvmlUtilizationRates()
        self._check(
            self.library.nvmlDeviceGetUtilizationRates(
                self.handle, ctypes.byref(rates)
            ),
            "nvmlDeviceGetUtilizationRates",
        )
        return float(rates.gpu)

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config
        contract_path = os.environ.get("SAT_EDITING_RUN_CONTRACT_PATH")
        self.editing_run_contract_path = None
        self.editing_run_contract = None
        self.editing_run_contract_sha256 = None
        if contract_path:
            path = Path(contract_path).expanduser().resolve(strict=True)
            value = json.loads(path.read_text(encoding="utf-8"))
            if not (
                value.get("schema")
                == "sceneplan_transfusion_editing_dit_training_run_contract"
                and int(value.get("schema_version", -1)) == 2
                and value.get("status") == "FROZEN"
                and path == Path(str(value.get("run_dir", ""))).resolve()
                / "TRAIN_RUN_CONTRACT.json"
            ):
                raise RuntimeError("invalid Editing-DiT training run contract")
            self.editing_run_contract_path = str(path)
            self.editing_run_contract = value
            self.editing_run_contract_sha256 = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config
        if self.editing_run_contract is not None:
            checkpoint["editing_run_contract"] = self.editing_run_contract
            checkpoint["editing_run_contract_path"] = (
                self.editing_run_contract_path
            )
            checkpoint["editing_run_contract_sha256"] = (
                self.editing_run_contract_sha256
            )


class ResumableDataLoaderGuardCallback(pl.Callback):
    """Reject legacy mid-epoch resumes that lack an exact data cursor."""

    def on_train_start(self, trainer, pl_module):
        if not trainer.fit_loop.restarted_mid_epoch:
            return
        combined_loader = trainer.fit_loop._combined_loader
        if combined_loader is None:
            raise RuntimeError("training DataLoader is unavailable during resume")
        for loader in combined_loader.flattened:
            assert_loaded = getattr(loader, "assert_resume_state_loaded", None)
            if callable(assert_loaded):
                assert_loaded()


class RankAwareTrainingSeedCallback(pl.Callback):
    """Give every DDP rank an independent, resume-stable training RNG stream.

    Model construction still uses the common seed, and ``PL_GLOBAL_SEED`` stays
    common so Lightning's DistributedSampler builds one shared permutation.
    We seed only after distributed setup and mix in the restored global step.
    Ordered P11 curricula repeat that derivation at every optimizer batch, so
    uninterrupted and restored executions of a step receive the same model RNG
    without giving different ranks identical noise, masks, or Sobol scrambles.
    """

    def __init__(
        self,
        base_seed: int,
        enabled: bool = True,
        deterministic_per_step: bool = False,
    ):
        self.base_seed = int(base_seed)
        self.enabled = bool(enabled)
        self.deterministic_per_step = bool(deterministic_per_step)

    @staticmethod
    def _mix_seed(base_seed: int, global_rank: int, global_step: int) -> int:
        modulus = 2**32 - 1
        return int(
            (
                base_seed
                + 1_000_003 * int(global_rank)
                + 97_409 * int(global_step)
            )
            % modulus
        )

    def _seed_step(self, trainer, pl_module) -> int:
        seed = self._mix_seed(
            self.base_seed,
            trainer.global_rank,
            trainer.global_step,
        )
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        reset_timestep_rng = getattr(pl_module, "reset_timestep_rng", None)
        if callable(reset_timestep_rng):
            reset_timestep_rng(seed)
        return seed

    def on_train_start(self, trainer, pl_module):
        if not self.enabled:
            return
        seed = self._seed_step(trainer, pl_module)
        print(
            f"[rng] global_rank={trainer.global_rank} "
            f"global_step={trainer.global_step} training_seed={seed} "
            f"contract={'rank_global_step_v1' if self.deterministic_per_step else 'segment_start_v1'}",
            flush=True,
        )

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.enabled and self.deterministic_per_step:
            self._seed_step(trainer, pl_module)


class ResumeCosineSegmentCallback(pl.Callback):
    """Start one new cosine LR segment after an exact full-state resume.

    Lightning restores the optimizer and LR-scheduler state from ``ckpt_path``.
    Merely changing the scheduler in the model config is therefore insufficient:
    an InverseLR state from the source run would overwrite the new scheduler's
    phase.  This callback resets the *phase* exactly once, at ``start_step``,
    while inheriting the checkpoint's effective optimizer LR without a jump.

    Checkpoints emitted after ``start_step`` contain the new cosine state.  A
    later restart validates and continues that state instead of resetting it,
    which keeps interruption/resume bitwise well-defined at the LR-contract
    level.
    """

    def __init__(self, *, start_step: int, end_step: int, eta_min: float):
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.eta_min = float(eta_min)
        if self.start_step < 0:
            raise ValueError("cosine resume start_step must be non-negative")
        if self.end_step <= self.start_step:
            raise ValueError("cosine resume end_step must exceed start_step")
        if not np.isfinite(self.eta_min) or self.eta_min < 0.0:
            raise ValueError("cosine resume eta_min must be finite and non-negative")

    @property
    def segment_steps(self) -> int:
        return self.end_step - self.start_step

    @staticmethod
    def _scheduler_configs(trainer):
        configs = list(getattr(trainer, "lr_scheduler_configs", ()) or ())
        if len(configs) != 1:
            raise RuntimeError(
                "resume cosine contract requires exactly one LR scheduler, "
                f"found {len(configs)}"
            )
        return configs

    def _validate_scheduler(self, scheduler):
        if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            raise RuntimeError(
                "resume cosine contract requires CosineAnnealingLR, got "
                f"{type(scheduler).__name__}"
            )
        if int(scheduler.T_max) != self.segment_steps:
            raise RuntimeError(
                f"cosine T_max={scheduler.T_max} does not match "
                f"segment_steps={self.segment_steps}"
            )
        if not np.isclose(
            float(scheduler.eta_min), self.eta_min, rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(
                f"cosine eta_min={scheduler.eta_min} does not match "
                f"contract={self.eta_min}"
            )

    def _reset_from_source_checkpoint(self, scheduler):
        optimizer = scheduler.optimizer
        current_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if not current_lrs or any(
            not np.isfinite(value) or value <= self.eta_min
            for value in current_lrs
        ):
            raise RuntimeError(
                "source checkpoint LR must be finite and above cosine eta_min, "
                f"got {current_lrs}"
            )

        # Remove attributes introduced by the source run's InverseLR state.
        # They are harmless to CosineAnnealingLR, but excluding them makes the
        # new checkpoint unambiguously describe the active scheduler.
        for name in ("inv_gamma", "power", "warmup", "final_lr"):
            scheduler.__dict__.pop(name, None)

        # ``LRScheduler.load_state_dict`` also restores scheduler-specific
        # fields.  In particular, chaining a second cosine segment would
        # otherwise copy the source segment's T_max/eta_min over the values
        # instantiated from the new model config.  At the exact segment
        # boundary these fields are part of the *new* continuation contract,
        # so reset them together with the phase.  Mid-segment resumes never
        # enter this branch and remain strictly validated below.
        scheduler.T_max = self.segment_steps
        scheduler.eta_min = self.eta_min

        for group, lr in zip(optimizer.param_groups, current_lrs):
            group["lr"] = lr
            group["initial_lr"] = lr
        scheduler.base_lrs = list(current_lrs)
        scheduler.last_epoch = 0
        scheduler._step_count = 1
        scheduler._last_lr = list(current_lrs)
        scheduler._get_lr_called_within_step = False
        return current_lrs

    def _validate_resumed_segment(self, scheduler, global_step: int):
        expected_epoch = global_step - self.start_step
        if abs(int(scheduler.last_epoch) - expected_epoch) > 1:
            raise RuntimeError(
                "resumed cosine phase is inconsistent with global_step: "
                f"last_epoch={scheduler.last_epoch}, expected={expected_epoch}"
            )
        current_lrs = [
            float(group["lr"]) for group in scheduler.optimizer.param_groups
        ]
        if any(
            not np.isfinite(value) or value < self.eta_min - 1e-12
            for value in current_lrs
        ):
            raise RuntimeError(
                "resumed cosine LR is outside the protected range: "
                f"{current_lrs}"
            )
        return current_lrs

    def on_train_start(self, trainer, pl_module):
        if not getattr(trainer, "ckpt_path", None):
            raise RuntimeError(
                "resume cosine segment requires a full-state --ckpt-path resume"
            )
        global_step = int(trainer.global_step)
        if global_step < self.start_step or global_step >= self.end_step:
            raise RuntimeError(
                "resume cosine global_step must lie in "
                f"[{self.start_step}, {self.end_step}), got {global_step}"
            )
        scheduler = self._scheduler_configs(trainer)[0].scheduler
        if global_step == self.start_step:
            current_lrs = self._reset_from_source_checkpoint(scheduler)
            self._validate_scheduler(scheduler)
            action = "initialized"
        else:
            self._validate_scheduler(scheduler)
            current_lrs = self._validate_resumed_segment(scheduler, global_step)
            action = "continued"
        if trainer.is_global_zero:
            print(
                "SAT_RESUME_COSINE_SEGMENT="
                + json.dumps(
                    {
                        "action": action,
                        "global_step": global_step,
                        "start_step": self.start_step,
                        "end_step": self.end_step,
                        "segment_steps": self.segment_steps,
                        "start_or_current_lrs": current_lrs,
                        "eta_min": self.eta_min,
                        "scheduler_last_epoch": int(scheduler.last_epoch),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

class TrainingPerformanceCallback(pl.Callback):
    """Measure full optimizer-step throughput after a short warmup."""

    def __init__(
        self,
        batch_size: int,
        training_examples_per_sample: float = 1.0,
        warmup_batches: int = 1,
        target_measured_batches: Optional[int] = None,
        target_final_global_step: Optional[int] = None,
        expected_world_size: Optional[int] = None,
        require_exact_local_batch_size: bool = False,
        require_disjoint_curriculum_rows: bool = False,
        require_gue_each_local_batch: bool = False,
        curriculum_identity_audit_batches: int = 8,
        row_identity_field: str = "p11_curriculum_id",
    ):
        if (
            target_measured_batches is not None
            and target_final_global_step is not None
        ):
            raise ValueError(
                "benchmark target_measured_batches and target_final_global_step "
                "are mutually exclusive"
            )
        self.batch_size = batch_size
        self.training_examples_per_sample = float(training_examples_per_sample)
        self.warmup_batches = warmup_batches
        self.target_measured_batches = target_measured_batches
        self.target_final_global_step = target_final_global_step
        self.expected_world_size = (
            int(expected_world_size) if expected_world_size is not None else None
        )
        self.require_exact_local_batch_size = bool(require_exact_local_batch_size)
        self.require_disjoint_curriculum_rows = bool(
            require_disjoint_curriculum_rows
        )
        self.require_gue_each_local_batch = bool(require_gue_each_local_batch)
        self.curriculum_identity_audit_batches = int(
            curriculum_identity_audit_batches
        )
        self.row_identity_field = str(row_identity_field)
        if not self.row_identity_field:
            raise ValueError("P11 row identity field must be non-empty")
        if self.curriculum_identity_audit_batches <= 0:
            raise ValueError("curriculum identity audit batches must be positive")
        self.started_at = None
        self.measured_batches = 0
        self.measured_local_samples = 0
        self.measured_local_sequence_positions = 0
        self.observed_local_batch_sizes = []
        self.observed_sequence_lengths = []
        self.finished = False
        self.initial_global_step = None
        self.measurement_start_global_step = None
        self.curriculum_identity_batches_seen = 0
        self.curriculum_identity_missing_batches = 0
        self.curriculum_identities = []
        self.curriculum_task_batches = []
        self.gpu_utilization_samples = []
        self.gpu_utilization_sample_batches = []
        self.gpu_utilization_sample_errors = 0
        self._gpu_utilization_sample_targets = None
        self._nvml_utilization_sampler = None

    @staticmethod
    def _uniform_sample_batches(total_batches: int, budget: int = 64):
        """Return deterministic 1-based samples spanning the full window."""

        total_batches = int(total_batches)
        budget = int(budget)
        if total_batches <= 0 or budget <= 0:
            return frozenset()
        if total_batches <= budget:
            return frozenset(range(1, total_batches + 1))
        denominator = budget - 1
        # Integer nearest-neighbour spacing is stable across Python versions,
        # includes both endpoints, and is unique when total >= budget.
        positions = {
            1
            + (
                index * (total_batches - 1) + denominator // 2
            )
            // denominator
            for index in range(budget)
        }
        if (
            len(positions) != budget
            or min(positions) != 1
            or max(positions) != total_batches
        ):
            raise RuntimeError(
                "benchmark failed to construct a full-window GPU utilization sample"
            )
        return frozenset(positions)

    def on_train_start(self, trainer, pl_module):
        """Resolve the measurement window after Lightning restores a resume.

        ``Trainer.global_step`` is authoritative only after checkpoint restore.
        Computing ``max_steps - warmup`` while constructing callbacks made a
        resumed 60 -> 500 run incorrectly demand 497 new measurements even
        though only 440 optimizer steps remained.
        """

        self.initial_global_step = int(trainer.global_step)
        self.measurement_start_global_step = (
            self.initial_global_step + int(self.warmup_batches)
        )
        if self.target_final_global_step is not None:
            remaining = int(self.target_final_global_step) - self.initial_global_step
            if remaining <= int(self.warmup_batches):
                raise RuntimeError(
                    "benchmark has no post-warmup optimizer steps: "
                    f"initial={self.initial_global_step}, "
                    f"target={self.target_final_global_step}, "
                    f"warmup={self.warmup_batches}"
                )
            self.target_measured_batches = remaining - int(self.warmup_batches)
        if self.target_measured_batches is not None:
            self._gpu_utilization_sample_targets = self._uniform_sample_batches(
                self.target_measured_batches
            )

    @staticmethod
    def _local_batch_size(batch) -> int:
        """Read the delivered batch size, including variable bucket batches."""

        if isinstance(batch, torch.Tensor):
            return int(batch.shape[0])
        if isinstance(batch, (list, tuple)) and batch:
            first = batch[0]
            if isinstance(first, torch.Tensor) and first.ndim:
                return int(first.shape[0])
            try:
                return int(len(first))
            except TypeError:
                pass
        if isinstance(batch, dict):
            for value in batch.values():
                if isinstance(value, torch.Tensor) and value.ndim:
                    return int(value.shape[0])
                try:
                    return int(len(value))
                except TypeError:
                    continue
        raise RuntimeError("benchmark callback cannot infer the delivered batch size")

    @staticmethod
    def _sequence_length(batch) -> int:
        candidate = batch
        if isinstance(batch, (list, tuple)) and batch:
            candidate = batch[0]
        elif isinstance(batch, dict):
            candidate = next(
                (
                    value
                    for value in batch.values()
                    if isinstance(value, torch.Tensor) and value.ndim >= 2
                ),
                None,
            )
        if isinstance(candidate, torch.Tensor) and candidate.ndim >= 2:
            return int(candidate.shape[-1])
        return 1

    def _curriculum_row_identities(self, batch):
        """Return stable per-row identities without consulting target content."""

        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            return None
        metadata = batch[1]
        if not isinstance(metadata, (list, tuple)) or not metadata:
            return None
        if not all(isinstance(row, dict) for row in metadata):
            return None
        if not all(
            row.get(self.row_identity_field) is not None for row in metadata
        ):
            return None
        return [str(row[self.row_identity_field]) for row in metadata]

    @staticmethod
    def _curriculum_row_tasks(batch):
        """Return one normalized task label per P11 row in the local batch."""

        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            return None
        metadata = batch[1]
        if not isinstance(metadata, (list, tuple)) or not metadata:
            return None
        if not all(isinstance(row, dict) for row in metadata):
            return None
        if not all(row.get("p11_task") is not None for row in metadata):
            return None
        return [str(row["p11_task"]) for row in metadata]

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # ``batch_idx`` restarts at every epoch. Tiny fixed curricula can have
        # only a handful of batches per epoch, so using it here silently reset
        # the timer and produced impossible throughput numbers on multi-epoch
        # probes. ``global_step`` is monotonic across epoch boundaries.
        start_step = (
            int(self.warmup_batches)
            if self.measurement_start_global_step is None
            else int(self.measurement_start_global_step)
        )
        if self.started_at is None and trainer.global_step >= start_step:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.started_at = time.perf_counter()
            if trainer.is_global_zero:
                print("SAT_BENCHMARK_START=1", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.started_at is not None and not self.finished:
            self.measured_batches += 1
            local_batch_size = self._local_batch_size(batch)
            sequence_length = self._sequence_length(batch)
            self.measured_local_samples += local_batch_size
            self.measured_local_sequence_positions += (
                local_batch_size * sequence_length
            )
            self.observed_local_batch_sizes.append(local_batch_size)
            self.observed_sequence_lengths.append(sequence_length)
            if (
                self.curriculum_identity_batches_seen
                < self.curriculum_identity_audit_batches
            ):
                identities = self._curriculum_row_identities(batch)
                self.curriculum_identity_batches_seen += 1
                if identities is None:
                    self.curriculum_identity_missing_batches += 1
                else:
                    if len(identities) != local_batch_size:
                        raise RuntimeError(
                            "curriculum identity count does not match local batch: "
                            f"{len(identities)} != {local_batch_size}"
                        )
                    self.curriculum_identities.extend(identities)
                    tasks = self._curriculum_row_tasks(batch)
                    if tasks is None or len(tasks) != local_batch_size:
                        raise RuntimeError(
                            "P11 curriculum task identities do not match the local batch"
                        )
                    self.curriculum_task_batches.append(tasks)
            should_sample_gpu = (
                self.measured_batches in self._gpu_utilization_sample_targets
                if self._gpu_utilization_sample_targets is not None
                else len(self.gpu_utilization_samples) < 64
            )
            if should_sample_gpu:
                try:
                    if self._nvml_utilization_sampler is None:
                        self._nvml_utilization_sampler = (
                            NvmlGpuUtilizationSampler()
                        )
                    self.gpu_utilization_samples.append(
                        self._nvml_utilization_sampler.sample()
                    )
                    self.gpu_utilization_sample_batches.append(
                        self.measured_batches
                    )
                except Exception:
                    # The canonical multi-GPU path fails closed below. Keep
                    # generic CPU/non-NVML benchmark users backwards compatible.
                    self.gpu_utilization_sample_errors += 1
            if (
                self.target_measured_batches is not None
                and self.measured_batches >= self.target_measured_batches
            ):
                self._finish(trainer)

    def on_fit_end(self, trainer, pl_module):
        self._finish(trainer)

    def _finish(self, trainer):
        """Emit before end-of-fit checkpoint I/O whenever the target is known."""

        if (
            self.finished
            or self.started_at is None
            or self.measured_batches == 0
        ):
            return
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started_at
        local_peak_allocated_gib = torch.cuda.max_memory_allocated() / 2**30
        local_peak_reserved_gib = torch.cuda.max_memory_reserved() / 2**30
        self.finished = True
        if not np.isfinite(elapsed) or elapsed <= 0.0:
            raise RuntimeError(f"benchmark measured invalid elapsed time {elapsed}")

        world_size = int(trainer.world_size)
        if (
            self.expected_world_size is not None
            and world_size != self.expected_world_size
        ):
            raise RuntimeError(
                "benchmark world-size mismatch: "
                f"expected {self.expected_world_size}, observed {world_size}"
            )

        distributed_initialized = bool(
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if world_size > 1 and not distributed_initialized:
            raise RuntimeError(
                f"benchmark expected distributed execution for world_size={world_size}"
            )

        local_batch_min = min(self.observed_local_batch_sizes)
        local_batch_max = max(self.observed_local_batch_sizes)
        rank_count = 1
        rank_index_sum = 0
        global_loader_samples = self.measured_local_samples
        global_sequence_positions = self.measured_local_sequence_positions
        rank_measured_batches_min = self.measured_batches
        rank_measured_batches_max = self.measured_batches
        rank_local_samples_min = self.measured_local_samples
        rank_local_samples_max = self.measured_local_samples
        rank_batch_size_min = local_batch_min
        rank_batch_size_max = local_batch_max
        rank_audits = [
            {
                "curriculum_identities": list(self.curriculum_identities),
                "curriculum_task_batches": list(self.curriculum_task_batches),
                "gpu_utilization_samples": list(self.gpu_utilization_samples),
                "gpu_utilization_sample_batches": list(
                    self.gpu_utilization_sample_batches
                ),
                "gpu_utilization_sample_errors": int(
                    self.gpu_utilization_sample_errors
                ),
                "peak_allocated_gib": local_peak_allocated_gib,
                "peak_reserved_gib": local_peak_reserved_gib,
            }
        ]
        if distributed_initialized:
            probe_device = trainer.strategy.root_device
            rank = int(torch.distributed.get_rank())
            summed = torch.tensor(
                [
                    1,
                    rank,
                    self.measured_local_samples,
                    self.measured_local_sequence_positions,
                ],
                dtype=torch.int64,
                device=probe_device,
            )
            torch.distributed.all_reduce(summed, op=torch.distributed.ReduceOp.SUM)
            rank_count, rank_index_sum, global_loader_samples, global_sequence_positions = (
                int(value) for value in summed.detach().cpu().tolist()
            )
            extrema_min = torch.tensor(
                [self.measured_batches, self.measured_local_samples, local_batch_min],
                dtype=torch.int64,
                device=probe_device,
            )
            extrema_max = torch.tensor(
                [self.measured_batches, self.measured_local_samples, local_batch_max],
                dtype=torch.int64,
                device=probe_device,
            )
            torch.distributed.all_reduce(
                extrema_min, op=torch.distributed.ReduceOp.MIN
            )
            torch.distributed.all_reduce(
                extrema_max, op=torch.distributed.ReduceOp.MAX
            )
            (
                rank_measured_batches_min,
                rank_local_samples_min,
                rank_batch_size_min,
            ) = (int(value) for value in extrema_min.detach().cpu().tolist())
            (
                rank_measured_batches_max,
                rank_local_samples_max,
                rank_batch_size_max,
            ) = (int(value) for value in extrema_max.detach().cpu().tolist())
            rank_audits = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(
                rank_audits,
                {
                    "curriculum_identities": list(self.curriculum_identities),
                    "curriculum_task_batches": list(
                        self.curriculum_task_batches
                    ),
                    "gpu_utilization_samples": list(
                        self.gpu_utilization_samples
                    ),
                    "gpu_utilization_sample_batches": list(
                        self.gpu_utilization_sample_batches
                    ),
                    "gpu_utilization_sample_errors": int(
                        self.gpu_utilization_sample_errors
                    ),
                    "peak_allocated_gib": local_peak_allocated_gib,
                    "peak_reserved_gib": local_peak_reserved_gib,
                },
            )

        rank_curriculum_identities = [
            list((audit or {}).get("curriculum_identities", []))
            for audit in rank_audits
        ]
        rank_curriculum_task_batches = [
            [
                [str(task) for task in batch]
                for batch in (audit or {}).get("curriculum_task_batches", [])
            ]
            for audit in rank_audits
        ]
        rank_curriculum_task_counts = [
            dict(
                sorted(
                    Counter(
                        task for batch in task_batches for task in batch
                    ).items()
                )
            )
            for task_batches in rank_curriculum_task_batches
        ]
        rank_gue_batch_counts = [
            sum(
                set(batch) == {"generation", "understanding", "editing"}
                for batch in task_batches
            )
            for task_batches in rank_curriculum_task_batches
        ]
        sampled_curriculum_batches_all_gue = bool(
            rank_curriculum_task_batches
        ) and all(
            count == len(task_batches)
            for count, task_batches in zip(
                rank_gue_batch_counts, rank_curriculum_task_batches
            )
        )
        identity_counts = [len(values or []) for values in rank_curriculum_identities]
        identity_unique_counts = [
            len(set(values or [])) for values in rank_curriculum_identities
        ]
        identity_sets = [set(values or []) for values in rank_curriculum_identities]
        cross_rank_identity_overlap_count = 0
        for left_index, left in enumerate(identity_sets):
            for right in identity_sets[left_index + 1 :]:
                cross_rank_identity_overlap_count += len(left.intersection(right))
        curriculum_rows_disjoint = (
            all(count > 0 for count in identity_counts)
            and identity_counts == identity_unique_counts
            and cross_rank_identity_overlap_count == 0
        )
        rank_gpu_utilization_samples = [
            [
                float(value)
                for value in (audit or {}).get("gpu_utilization_samples", [])
            ]
            for audit in rank_audits
        ]
        rank_gpu_utilization_sample_batches = [
            [
                int(value)
                for value in (audit or {}).get(
                    "gpu_utilization_sample_batches", []
                )
            ]
            for audit in rank_audits
        ]
        rank_gpu_utilization_means = [
            float(sum(values) / len(values)) if values else 0.0
            for values in rank_gpu_utilization_samples
        ]
        gpu_utilization_values = [
            value
            for values in rank_gpu_utilization_samples
            for value in values
        ]
        gpu_utilization_sample_errors = sum(
            int((audit or {}).get("gpu_utilization_sample_errors", 0))
            for audit in rank_audits
        )
        rank_peak_allocated_gib = [
            float((audit or {}).get("peak_allocated_gib", 0.0))
            for audit in rank_audits
        ]
        rank_peak_reserved_gib = [
            float((audit or {}).get("peak_reserved_gib", 0.0))
            for audit in rank_audits
        ]
        if any(
            not np.isfinite(value) or value <= 0.0
            for value in rank_peak_allocated_gib + rank_peak_reserved_gib
        ):
            raise RuntimeError(
                "P11 benchmark observed invalid per-rank CUDA memory peaks: "
                f"allocated={rank_peak_allocated_gib}, "
                f"reserved={rank_peak_reserved_gib}"
            )
        gpu_utilization_mean = (
            float(sum(gpu_utilization_values) / len(gpu_utilization_values))
            if gpu_utilization_values
            else 0.0
        )
        if self.require_disjoint_curriculum_rows:
            expected_identity_count = (
                min(self.measured_batches, self.curriculum_identity_audit_batches)
                * self.batch_size
            )
            if self.curriculum_identity_missing_batches:
                raise RuntimeError(
                    "P11 curriculum identity audit found batches without "
                    f"{self.row_identity_field}"
                )
            if any(count != expected_identity_count for count in identity_counts):
                raise RuntimeError(
                    "P11 curriculum identity audit sampled an unexpected number "
                    f"of rows per rank: expected={expected_identity_count}, "
                    f"observed={identity_counts}"
                )
            if not curriculum_rows_disjoint:
                raise RuntimeError(
                    "P11 distributed sampler repeated curriculum rows within or "
                    "across ranks: "
                    f"counts={identity_counts}, unique={identity_unique_counts}, "
                    f"cross_rank_overlap={cross_rank_identity_overlap_count}"
                )
            if (
                gpu_utilization_sample_errors
                or any(not values for values in rank_gpu_utilization_samples)
            ):
                raise RuntimeError(
                    "P11 multi-GPU utilization audit did not obtain one NVML "
                    f"sample stream per rank: errors={gpu_utilization_sample_errors}, "
                    f"counts={[len(values) for values in rank_gpu_utilization_samples]}"
                )
            if any(
                values != rank_gpu_utilization_sample_batches[0]
                for values in rank_gpu_utilization_sample_batches[1:]
            ):
                raise RuntimeError(
                    "P11 DDP ranks sampled GPU utilization at different "
                    f"measured batches: {rank_gpu_utilization_sample_batches}"
                )
        if self.require_gue_each_local_batch:
            expected_task_batches = min(
                self.measured_batches, self.curriculum_identity_audit_batches
            )
            if any(
                len(task_batches) != expected_task_batches
                for task_batches in rank_curriculum_task_batches
            ):
                raise RuntimeError(
                    "P11 DDP task audit sampled an unexpected number of batches: "
                    f"expected={expected_task_batches}, observed="
                    f"{[len(value) for value in rank_curriculum_task_batches]}"
                )
            if not sampled_curriculum_batches_all_gue:
                raise RuntimeError(
                    "P11 DDP rank-local batches do not all contain G/U/E: "
                    f"passing={rank_gue_batch_counts}"
                )

        expected_rank_index_sum = world_size * (world_size - 1) // 2
        if rank_count != world_size or rank_index_sum != expected_rank_index_sum:
            raise RuntimeError(
                "benchmark DDP rank confirmation failed: "
                f"count={rank_count}/{world_size}, "
                f"rank_sum={rank_index_sum}/{expected_rank_index_sum}"
            )
        if rank_measured_batches_min != rank_measured_batches_max:
            raise RuntimeError(
                "benchmark ranks measured different optimizer-step counts: "
                f"{rank_measured_batches_min}..{rank_measured_batches_max}"
            )
        if (
            self.target_measured_batches is not None
            and rank_measured_batches_min != self.target_measured_batches
        ):
            raise RuntimeError(
                "benchmark did not complete its requested measurement window: "
                f"expected {self.target_measured_batches}, "
                f"observed {rank_measured_batches_min}"
            )
        if self.require_exact_local_batch_size and (
            rank_batch_size_min != self.batch_size
            or rank_batch_size_max != self.batch_size
        ):
            raise RuntimeError(
                "benchmark delivered a non-canonical local batch: "
                f"expected {self.batch_size}, observed "
                f"{rank_batch_size_min}..{rank_batch_size_max}"
            )

        loader_samples_per_second = (
            global_loader_samples / elapsed
        )
        sequence_positions_per_second = global_sequence_positions / elapsed
        if (
            not np.isfinite(loader_samples_per_second)
            or loader_samples_per_second <= 0.0
            or not np.isfinite(sequence_positions_per_second)
            or sequence_positions_per_second <= 0.0
        ):
            raise RuntimeError(
                "benchmark measured non-positive or non-finite throughput: "
                f"samples/s={loader_samples_per_second}, "
                f"positions/s={sequence_positions_per_second}"
            )
        result = {
            "initial_global_step": self.initial_global_step,
            "measurement_start_global_step": self.measurement_start_global_step,
            "target_final_global_step": self.target_final_global_step,
            "warmup_batches": self.warmup_batches,
            "measured_optimizer_steps": self.measured_batches,
            "measured_local_loader_samples": self.measured_local_samples,
            "measured_local_sequence_positions": (
                self.measured_local_sequence_positions
            ),
            "elapsed_seconds": elapsed,
            "seconds_per_step": elapsed / self.measured_batches,
            # Kept for compatibility with the existing T1 benchmark parser.
            # For SpatialFamilyDataset, one loader sample expands to four turns.
            "global_samples_per_second": loader_samples_per_second,
            "global_loader_samples_per_second": loader_samples_per_second,
            "training_examples_per_loader_sample": self.training_examples_per_sample,
            "global_training_examples_per_second": (
                loader_samples_per_second * self.training_examples_per_sample
            ),
            "global_sequence_positions_per_second": (
                sequence_positions_per_second
            ),
            "batch_size_per_gpu": self.batch_size,
            "observed_batch_size_per_gpu_min": min(
                self.observed_local_batch_sizes
            ),
            "observed_batch_size_per_gpu_max": max(
                self.observed_local_batch_sizes
            ),
            "observed_batch_size_per_gpu_mean": (
                self.measured_local_samples / self.measured_batches
            ),
            "observed_sequence_length_min": min(
                self.observed_sequence_lengths
            ),
            "observed_sequence_length_max": max(
                self.observed_sequence_lengths
            ),
            "world_size": world_size,
            "expected_world_size": self.expected_world_size,
            "strategy": type(trainer.strategy).__name__,
            "distributed_backend": (
                torch.distributed.get_backend() if distributed_initialized else None
            ),
            "distributed_initialized": distributed_initialized,
            "distributed_rank_count": rank_count,
            "distributed_rank_index_sum": rank_index_sum,
            "distributed_rank_index_sum_expected": expected_rank_index_sum,
            "rank_measured_batches_min": rank_measured_batches_min,
            "rank_measured_batches_max": rank_measured_batches_max,
            "rank_local_samples_min": rank_local_samples_min,
            "rank_local_samples_max": rank_local_samples_max,
            "rank_observed_batch_size_min": rank_batch_size_min,
            "rank_observed_batch_size_max": rank_batch_size_max,
            "exact_local_batch_size_required": self.require_exact_local_batch_size,
            "curriculum_identity_field": self.row_identity_field,
            "p11_row_identity_field": self.row_identity_field,
            "curriculum_identity_audit_batches": (
                self.curriculum_identity_batches_seen
            ),
            "rank_curriculum_identity_count_min": min(identity_counts),
            "rank_curriculum_identity_count_max": max(identity_counts),
            "rank_curriculum_identity_unique_min": min(identity_unique_counts),
            "rank_curriculum_identity_unique_max": max(identity_unique_counts),
            # At most eight local batches per rank are retained.  Keeping the
            # exact immutable IDs lets the resume smoke prove that the first
            # measured rows came from the restored cursor rather than merely
            # reporting disjoint counts.
            "rank_curriculum_identities": rank_curriculum_identities,
            "cross_rank_curriculum_identity_overlap_count": (
                cross_rank_identity_overlap_count
            ),
            "curriculum_rows_disjoint_across_ranks": curriculum_rows_disjoint,
            "disjoint_curriculum_rows_required": (
                self.require_disjoint_curriculum_rows
            ),
            "gue_each_local_batch_required": self.require_gue_each_local_batch,
            "rank_curriculum_task_counts": rank_curriculum_task_counts,
            "rank_curriculum_gue_batch_count_min": min(rank_gue_batch_counts),
            "rank_curriculum_gue_batch_count_max": max(rank_gue_batch_counts),
            "sampled_curriculum_batches_all_gue": (
                sampled_curriculum_batches_all_gue
            ),
            "gpu_utilization_sample_count": len(gpu_utilization_values),
            "gpu_utilization_sample_errors": gpu_utilization_sample_errors,
            "gpu_utilization_sampling_contract": (
                "uniform_over_measured_window_v1"
                if self._gpu_utilization_sample_targets is not None
                else "leading_window_v1"
            ),
            "gpu_utilization_sample_batches": (
                rank_gpu_utilization_sample_batches[0]
                if rank_gpu_utilization_sample_batches
                else []
            ),
            "gpu_utilization_percent_mean": gpu_utilization_mean,
            "gpu_utilization_percent_min": (
                min(gpu_utilization_values) if gpu_utilization_values else 0.0
            ),
            "gpu_utilization_percent_max": (
                max(gpu_utilization_values) if gpu_utilization_values else 0.0
            ),
            "rank_gpu_utilization_percent_mean_min": min(
                rank_gpu_utilization_means
            ),
            "rank_gpu_utilization_percent_mean_max": max(
                rank_gpu_utilization_means
            ),
            # Preserve the historical rank-zero fields while also recording
            # every rank.  A single rank is not a valid proxy for a
            # heterogeneous G/U/E DDP batch stream.
            "peak_allocated_gib": local_peak_allocated_gib,
            "peak_reserved_gib": local_peak_reserved_gib,
            "rank_peak_allocated_gib": rank_peak_allocated_gib,
            "rank_peak_allocated_gib_min": min(rank_peak_allocated_gib),
            "rank_peak_allocated_gib_max": max(rank_peak_allocated_gib),
            "rank_peak_reserved_gib": rank_peak_reserved_gib,
            "rank_peak_reserved_gib_min": min(rank_peak_reserved_gib),
            "rank_peak_reserved_gib_max": max(rank_peak_reserved_gib),
        }
        scalar_metrics = {}
        for name, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                scalar_metrics[name] = value.detach().float().cpu().item()
        if scalar_metrics:
            result["final_scalar_metrics"] = scalar_metrics
        if trainer.is_global_zero:
            print(f"SAT_BENCHMARK_RESULT={json.dumps(result, sort_keys=True)}")


class TrainingHealthGateCallback(pl.Callback):
    """Fail closed on non-finite objectives, zero gradients, or stale EMA."""

    def __init__(
        self,
        required_metrics,
        *,
        ratio_metrics=None,
        window=20,
        max_loss_ratio=-1.0,
        gradient_every=1,
        expected_world_size: Optional[int] = None,
    ):
        self.required_metrics = tuple(str(name) for name in required_metrics)
        self.ratio_metrics = (
            self.required_metrics
            if ratio_metrics is None
            else tuple(str(name) for name in ratio_metrics)
        )
        unknown_ratio_metrics = sorted(
            set(self.ratio_metrics) - set(self.required_metrics)
        )
        if unknown_ratio_metrics:
            raise ValueError(
                "training-gate ratio metrics must also be required metrics: "
                f"{unknown_ratio_metrics}"
            )
        self.window = max(1, int(window))
        self.max_loss_ratio = float(max_loss_ratio)
        self.gradient_every = max(1, int(gradient_every))
        self.expected_world_size = (
            int(expected_world_size) if expected_world_size is not None else None
        )
        self.latest_metrics = {}
        self.latest_metric_weights = {}
        self.metric_activity_names = {name: None for name in self.required_metrics}
        self.metric_active_observations = {
            name: 0 for name in self.required_metrics
        }
        self.first_metrics = {name: [] for name in self.required_metrics}
        self.last_metrics = {
            name: deque(maxlen=self.window) for name in self.required_metrics
        }
        self.metric_observations = 0
        self.gradient_norms = []
        self.optimizer_events = 0
        self.initial_global_step = None
        self.ema_initial = {}
        self.lr_initial = []
        self.finished = False

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise RuntimeError(
                    f"training gate expected a scalar metric, got {tuple(value.shape)}"
                )
            return float(value.detach().float().cpu())
        return float(value)

    @staticmethod
    def _activity_metric_name(name, metrics):
        direct_name = f"{name}_rows"
        if direct_name in metrics:
            return direct_name
        # Unified G/U/E wrappers emit zero placeholders for a route that is
        # absent from a particular ordered batch. Their active count is logged
        # under ``task_<route>`` rather than beside each CE/flow scalar.
        if "/" in name:
            stage, metric_name = name.rsplit("/", 1)
            for task in ("generation", "understanding", "editing"):
                if metric_name.startswith(f"{task}_"):
                    task_count_name = f"{stage}/task_{task}"
                    if task_count_name in metrics:
                        return task_count_name
        return None

    @staticmethod
    def _ema_steps(module):
        steps = {}
        # Training wrappers use different attribute names for their EMA state.
        # Keep the gate architecture-agnostic so both Transfusion and the
        # ScenePlan-conditioned dense DiT must prove that every active EMA is
        # advancing.
        for name in (
            "p11_ema",
            "transfusion_ema",
            "text_conditioner_ema",
            "diffusion_ema",
            "conditioner_ema",
        ):
            ema = getattr(module, name, None)
            step = getattr(ema, "step", None) if ema is not None else None
            if step is not None:
                steps[name] = int(torch.as_tensor(step).detach().cpu())
        return steps

    @staticmethod
    def _learning_rates(trainer):
        return [
            {
                "optimizer": optimizer_index,
                "group": str(
                    group.get("group_name", f"group_{group_index}")
                ),
                "lr": float(group["lr"]),
            }
            for optimizer_index, optimizer in enumerate(trainer.optimizers)
            for group_index, group in enumerate(optimizer.param_groups)
        ]

    @staticmethod
    def _raise_if_any_rank_failed(trainer, local_message, *, context):
        """Make callback failures collective so no rank is left in DDP."""

        distributed = bool(
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        failed = bool(local_message)
        if distributed:
            failure_flag = torch.tensor(
                [int(failed)],
                dtype=torch.int32,
                device=trainer.strategy.root_device,
            )
            torch.distributed.all_reduce(
                failure_flag, op=torch.distributed.ReduceOp.MAX
            )
            failed = bool(int(failure_flag.detach().cpu().item()))
        if failed:
            detail = f": {local_message}" if local_message else ""
            raise RuntimeError(
                f"{context} failed on at least one distributed rank{detail}"
            )

    def on_train_start(self, trainer, pl_module):
        self.initial_global_step = int(trainer.global_step)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.ema_initial = self._ema_steps(pl_module)
        self.lr_initial = self._learning_rates(trainer)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.optimizer_events += 1
        if self.optimizer_events > 1 and self.optimizer_events % self.gradient_every:
            return
        squared_norm = None
        parameters_with_grad = 0
        for parameter in pl_module.parameters():
            gradient = parameter.grad
            if gradient is None:
                continue
            parameters_with_grad += 1
            parameter_norm = torch.linalg.vector_norm(gradient.detach())
            contribution = parameter_norm.float().square()
            squared_norm = (
                contribution
                if squared_norm is None
                else squared_norm + contribution
            )
        local_failure = None
        total_norm = 0.0
        if parameters_with_grad == 0 or squared_norm is None:
            local_failure = "training gate found no gradients before optimizer step"
        else:
            # Synchronize once after all device-side reductions. The former
            # per-parameter bool() check serialized hundreds of tiny CUDA kernels
            # and amplified rank skew immediately before the optimizer step.
            total_norm = float(squared_norm.sqrt().detach().cpu())
            if not np.isfinite(total_norm) or total_norm <= 0.0:
                local_failure = (
                    "training gate requires a finite non-zero gradient norm, "
                    f"got {total_norm}"
                )
        self._raise_if_any_rank_failed(
            trainer,
            local_failure,
            context="training gradient gate",
        )
        self.gradient_norms.append(total_norm)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = getattr(pl_module, "_last_step_metrics", None)
        observed = {}
        local_failure = None
        if not isinstance(metrics, dict):
            local_failure = (
                "training gate requires the training wrapper to expose "
                "_last_step_metrics"
            )
        else:
            missing = [name for name in self.required_metrics if name not in metrics]
            if missing:
                local_failure = (
                    f"training gate is missing objective metrics: {missing}"
                )
            else:
                try:
                    for name in self.required_metrics:
                        value = self._scalar(metrics[name])
                        if not np.isfinite(value):
                            raise RuntimeError(
                                "training gate found non-finite objective "
                                f"{name}={value}"
                            )
                        activity_name = self._activity_metric_name(name, metrics)
                        if activity_name is not None:
                            weight = self._scalar(metrics[activity_name])
                            if not np.isfinite(weight) or weight < 0.0:
                                raise RuntimeError(
                                    "training gate found invalid activity count "
                                    f"{activity_name}={weight}"
                                )
                            self.metric_activity_names[name] = activity_name
                        else:
                            weight = 1.0
                        observed[name] = (value, weight)
                except Exception as error:
                    local_failure = f"{type(error).__name__}: {error}"
        self._raise_if_any_rank_failed(
            trainer,
            local_failure,
            context="training objective gate",
        )
        for name, (value, weight) in observed.items():
            # Several P11 objectives are sparse by construction. Their wrapper
            # emits a finite zero together with ``*_rows == 0`` when an ordered
            # batch contains no matching operation. Treating those placeholders
            # as real loss observations made the first/last ratio depend on
            # operation density rather than learning. Keep the finite check
            # above, but only summarize active rows and weight their batch means
            # by the number of contributing rows.
            if weight <= 0.0:
                continue
            self.latest_metrics[name] = value
            self.latest_metric_weights[name] = weight
            self.metric_active_observations[name] += 1
            weighted_observation = (value, weight)
            if len(self.first_metrics[name]) < self.window:
                self.first_metrics[name].append(weighted_observation)
            self.last_metrics[name].append(weighted_observation)
        self.metric_observations += 1

    def on_fit_end(self, trainer, pl_module):
        if self.finished:
            return
        self.finished = True
        optimizer_state_steps = []
        for optimizer in trainer.optimizers:
            for state in optimizer.state.values():
                step = state.get("step") if isinstance(state, dict) else None
                if step is not None:
                    optimizer_state_steps.append(int(torch.as_tensor(step).detach().cpu()))

        ema_final = self._ema_steps(pl_module)
        stale_ema = {
            name: (initial, ema_final.get(name))
            for name, initial in self.ema_initial.items()
            if ema_final.get(name, initial) <= initial
        }

        world_size = int(trainer.world_size)
        if (
            self.expected_world_size is not None
            and world_size != self.expected_world_size
        ):
            raise RuntimeError(
                "training gate world-size mismatch: "
                f"expected {self.expected_world_size}, observed {world_size}"
            )
        distributed_initialized = bool(
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )
        if world_size > 1 and not distributed_initialized:
            raise RuntimeError(
                f"training gate expected DDP for world_size={world_size}"
            )
        ema_advances = {
            name: int(ema_final.get(name, initial) - initial)
            for name, initial in self.ema_initial.items()
        }
        local_health = [
            int(self.optimizer_events),
            int(max(optimizer_state_steps, default=0)),
            int(len(self.gradient_norms)),
            int(min(ema_advances.values(), default=0)),
            int(self.metric_observations),
        ]
        health_min = list(local_health)
        health_max = list(local_health)
        # Loss windows are rank-local because each DDP worker owns a disjoint
        # data partition.  Gate their global aggregate, not each partition in
        # isolation: independently raising here lets passing ranks enter the
        # collectives below while failing ranks exit, which turns an ordinary
        # health-gate failure into a 30-minute NCCL timeout.
        local_window_stats = torch.tensor(
            [
                [
                    float(sum(value * weight for value, weight in self.first_metrics[name])),
                    float(sum(weight for _, weight in self.first_metrics[name])),
                    float(len(self.first_metrics[name])),
                    float(sum(value * weight for value, weight in self.last_metrics[name])),
                    float(sum(weight for _, weight in self.last_metrics[name])),
                    float(len(self.last_metrics[name])),
                    float(
                        self.latest_metrics.get(name, 0.0)
                        * self.latest_metric_weights.get(name, 0.0)
                    ),
                    float(self.latest_metric_weights.get(name, 0.0)),
                    float(self.metric_active_observations[name]),
                ]
                for name in self.required_metrics
            ],
            dtype=torch.float64,
            device=trainer.strategy.root_device,
        )
        window_stats = local_window_stats.clone()
        rank_count = 1
        rank_index_sum = 0
        if distributed_initialized:
            probe_device = trainer.strategy.root_device
            rank = int(torch.distributed.get_rank())
            rank_probe = torch.tensor(
                [1, rank], dtype=torch.int64, device=probe_device
            )
            torch.distributed.all_reduce(
                rank_probe, op=torch.distributed.ReduceOp.SUM
            )
            rank_count, rank_index_sum = (
                int(value) for value in rank_probe.detach().cpu().tolist()
            )
            health_min_tensor = torch.tensor(
                local_health, dtype=torch.int64, device=probe_device
            )
            health_max_tensor = health_min_tensor.clone()
            torch.distributed.all_reduce(
                health_min_tensor, op=torch.distributed.ReduceOp.MIN
            )
            torch.distributed.all_reduce(
                health_max_tensor, op=torch.distributed.ReduceOp.MAX
            )
            torch.distributed.all_reduce(
                window_stats, op=torch.distributed.ReduceOp.SUM
            )
            health_min = [
                int(value) for value in health_min_tensor.detach().cpu().tolist()
            ]
            health_max = [
                int(value) for value in health_max_tensor.detach().cpu().tolist()
            ]
        window_stats_cpu = window_stats.detach().cpu().tolist()
        metric_windows = {}
        objectives = {}
        failing_ratios = {}
        inactive_metrics = []
        for name, stats in zip(self.required_metrics, window_stats_cpu):
            (
                first_sum,
                first_weight,
                first_count,
                last_sum,
                last_weight,
                last_count,
                latest_sum,
                latest_weight,
                active_observations,
            ) = stats
            first_mean = first_sum / first_weight if first_weight > 0 else None
            last_mean = last_sum / last_weight if last_weight > 0 else None
            ratio = (
                last_mean / first_mean
                if first_mean is not None
                and last_mean is not None
                and first_mean != 0.0
                else None
            )
            metric_windows[name] = {
                "first_mean": first_mean,
                "last_mean": last_mean,
                "last_over_first": ratio,
                "first_count": int(first_count),
                "last_count": int(last_count),
                "first_active_weight": first_weight,
                "last_active_weight": last_weight,
                "active_observations": int(active_observations),
                "activity_metric": self.metric_activity_names[name],
                "ratio_gated": name in self.ratio_metrics,
            }
            objectives[name] = (
                latest_sum / latest_weight if latest_weight > 0 else None
            )
            if active_observations <= 0 or first_weight <= 0 or last_weight <= 0:
                inactive_metrics.append(name)
            if (
                self.max_loss_ratio >= 0.0
                and name in self.ratio_metrics
                and (ratio is None or not np.isfinite(ratio) or ratio > self.max_loss_ratio)
            ):
                failing_ratios[name] = ratio
        expected_rank_index_sum = world_size * (world_size - 1) // 2
        if rank_count != world_size or rank_index_sum != expected_rank_index_sum:
            raise RuntimeError(
                "training gate DDP rank confirmation failed: "
                f"count={rank_count}/{world_size}, "
                f"rank_sum={rank_index_sum}/{expected_rank_index_sum}"
            )
        if min(health_min) <= 0:
            raise RuntimeError(
                "training gate found a rank without optimizer/gradient/EMA/metric progress: "
                f"minima={health_min}"
            )
        if stale_ema:
            # Normally identical on every rank; retain the explicit diagnostic
            # after synchronized health collection for single-rank clarity.
            raise RuntimeError(f"training gate found stale EMA state: {stale_ema}")
        if inactive_metrics:
            raise RuntimeError(
                "training gate found required metrics without active rows: "
                f"{inactive_metrics}"
            )
        if failing_ratios:
            raise RuntimeError(
                "training gate globally aggregated loss ratios exceed "
                f"{self.max_loss_ratio}: {failing_ratios}"
            )

        distributed_health = {
            "world_size": world_size,
            "expected_world_size": self.expected_world_size,
            "strategy": type(trainer.strategy).__name__,
            "backend": (
                torch.distributed.get_backend() if distributed_initialized else None
            ),
            "initialized": distributed_initialized,
            "rank_count": rank_count,
            "rank_index_sum": rank_index_sum,
            "rank_index_sum_expected": expected_rank_index_sum,
            "optimizer_events_min": health_min[0],
            "optimizer_events_max": health_max[0],
            "optimizer_state_step_min": health_min[1],
            "optimizer_state_step_max": health_max[1],
            "gradient_checks_min": health_min[2],
            "gradient_checks_max": health_max[2],
            "ema_advance_min": health_min[3],
            "ema_advance_max": health_max[3],
            "metric_observations_min": health_min[4],
            "metric_observations_max": health_max[4],
        }

        result = {
            "status": "PASS",
            "initial_global_step": int(self.initial_global_step or 0),
            "global_step": int(trainer.global_step),
            "metric_observations": self.metric_observations,
            "objectives": objectives,
            "metric_windows": metric_windows,
            "metric_window_aggregation": (
                "active_row_weighted_sum_across_ddp_ranks_v2"
            ),
            "ratio_metrics": list(self.ratio_metrics),
            "max_loss_ratio_gate": self.max_loss_ratio,
            "optimizer_events": self.optimizer_events,
            "gradient_checks": len(self.gradient_norms),
            "gradient_check_every": self.gradient_every,
            "optimizer_state_step_max": max(optimizer_state_steps, default=0),
            "gradient_norm_min": min(self.gradient_norms),
            "gradient_norm_max": max(self.gradient_norms),
            "ema_initial": self.ema_initial,
            "ema_final": ema_final,
            "ema_advances": ema_advances,
            "distributed_health": distributed_health,
            "learning_rates_initial": self.lr_initial,
            "learning_rates_final": self._learning_rates(trainer),
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available()
                else 0.0
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 2**30
                if torch.cuda.is_available()
                else 0.0
            ),
        }
        if trainer.is_global_zero:
            print(f"SAT_TRAINING_GATE_RESULT={json.dumps(result, sort_keys=True)}")

def raise_open_file_limit(target: int = 65536) -> None:
    """Raise DataLoader IPC headroom without exceeding the process hard limit."""

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    requested = min(target, hard)
    if soft < requested:
        resource.setrlimit(resource.RLIMIT_NOFILE, (requested, hard))
        print(f"Raised RLIMIT_NOFILE from {soft} to {requested}")


def prepare_storage(path_value: str, label: str, min_free_gib: float) -> Path:
    """Create a runtime directory and fail before training if it is nearly full."""

    runtime_path = Path(path_value).expanduser().resolve()
    runtime_path.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(runtime_path).free / 2**30
    if free_gib < float(min_free_gib):
        raise RuntimeError(
            f"{label} has only {free_gib:.2f} GiB free at {runtime_path}; "
            f"requires at least {float(min_free_gib):.2f} GiB"
        )
    print(f"[storage] {label}={runtime_path} free={free_gib:.2f} GiB")
    return runtime_path

def main():
    # Removed: torch.multiprocessing.set_sharing_strategy('file_system')
    # 'file_system' creates named files in /dev/shm that accumulate over long runs,
    # causing gradual slowdown and eventual "Shared memory manager connection has
    # timed out" crashes. The default 'file_descriptor' strategy uses kernel-managed
    # FDs with automatic cleanup. Requires ulimit -n 65536 in SLURM script.
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision('high')
    raise_open_file_limit()
    args = get_all_args()
    if args.temp_dir:
        temp_dir = prepare_storage(
            args.temp_dir,
            "temp_dir",
            args.min_free_disk_gib,
        )
        for env_name in ("TMPDIR", "TMP", "TEMP"):
            os.environ[env_name] = str(temp_dir)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got {local_rank}")
    if args.bind_to_gpu_numa:
        binding = bind_process_to_local_gpu_numa(local_rank)
        print(
            "[numa] "
            f"local_rank={binding.local_rank} "
            f"visible_gpu={binding.visible_device} "
            f"pci={binding.pci_bus_id} "
            f"numa={binding.numa_node} "
            f"cpus={binding.cpu_list} "
            "memory=preferred",
            flush=True,
        )
    # Lightning assigns devices later, but model construction happens below.
    # Qwen3.5's FLA modules allocate CUDA-backed fused operators in their
    # constructors, so every DDP worker otherwise touches the default cuda:0
    # and leaves a persistent CUDA context there. Bind the rank before model
    # construction to keep each worker entirely on its own GPU.
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds visible CUDA devices={device_count}"
            )
        torch.cuda.set_device(local_rank)
        print(
            f"[cuda] local_rank={local_rank} current_device={torch.cuda.current_device()}",
            flush=True,
        )
    # All ranks construct exactly the same initial model. A callback installed
    # below creates rank-specific training RNG streams only after DDP setup.
    seed = int(args.seed)
    pl.seed_everything(seed, workers=True)

    model_config = load_config(args.model_config)
    dataset_config = load_config(args.dataset_config)
    config_summary = validate_training_configs(model_config, dataset_config)
    if config_summary is not None:
        print(f"Validated training config: {config_summary}")

    val_dataset_config = None
    if args.val_dataset_config:
        val_dataset_config = load_config(args.val_dataset_config)
        # Validation measures the renderer objective without injecting the
        # train-only forced-alignment teacher.  This opt-in is local to the
        # validation call; the training config remains fail-closed.
        validate_training_configs(
            model_config,
            val_dataset_config,
            allow_missing_speech_timing=True,
        )

    if model_config["training"].get("lora_config"):
        print("Lora Config", model_config["training"]["lora_config"])

    model = create_model_from_config(model_config)

    # Pre-tokenize frozen text encoders in DataLoader workers.  The shared helper
    # covers both MultiConditioner DiT and qwen_prefix Transfusion models.
    tokenizers = collect_conditioner_tokenizers(model, model_config)

    train_bucketed_sceneplan = bool(
        (dataset_config.get("length_bucket_batching") or {}).get(
            "enabled", False
        )
    )
    p11_model_type = model_config.get("model_type") in {
        "sceneplan_p11",
        "sceneplan_p11_v4",
        "sceneplan_p11_audio_aware_v1",
    }
    train_rank_aware = p11_model_type or train_bucketed_sceneplan

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        tokenizers=tokenizers if tokenizers else None,
        distributed_world_size=(
            args.num_gpus if train_rank_aware else None
        ),
        distributed_rank=(
            int(os.environ.get("LOCAL_RANK", "0"))
            if train_rank_aware
            else None
        ),
    )
    if p11_model_type:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if bool(
            getattr(train_dl, "sceneplan_p11_ordered_batch_sampler", False)
        ):
            from stable_audio_tools.data.sceneplan_p11_ordered_sampler import (
                DistributedP11OrderedBatchSampler,
            )

            batch_sampler = train_dl.batch_sampler
            if (
                not isinstance(
                    batch_sampler, DistributedP11OrderedBatchSampler
                )
                or bool(batch_sampler.shuffle)
                or int(batch_sampler.num_replicas) != int(args.num_gpus)
                or int(batch_sampler.rank) != local_rank
                or int(batch_sampler.batch_size) != int(args.batch_size)
            ):
                raise RuntimeError(
                    "P11 ordered curriculum did not install its exact "
                    "rank-aware batch sampler"
                )
            print(
                f"[p11-data] batch_sampler={type(batch_sampler).__name__} "
                f"shuffle={bool(batch_sampler.shuffle)} "
                f"rank={local_rank}/{args.num_gpus} "
                f"resume=O(1)",
                flush=True,
            )
        else:
            sampler = train_dl.sampler
            if args.num_gpus > 1 and (
                not isinstance(sampler, torch.utils.data.DistributedSampler)
                or bool(sampler.shuffle)
                or int(sampler.num_replicas) != int(args.num_gpus)
                or int(sampler.rank) != local_rank
            ):
                raise RuntimeError(
                    "P11 requires its own non-shuffling rank-aware "
                    "DistributedSampler"
                )
            print(
                f"[p11-data] sampler={type(sampler).__name__} "
                f"shuffle={bool(getattr(sampler, 'shuffle', False))} "
                f"rank={local_rank}/{args.num_gpus} resume=prefix-replay",
                flush=True,
            )
    if train_bucketed_sceneplan:
        if not bool(
            getattr(train_dl, "sceneplan_rank_aware_bucket_sampler", False)
        ):
            raise RuntimeError(
                "ScenePlan length-bucket config did not install its rank-aware sampler"
            )
        print(
            "[p10-data] "
            + json.dumps(
                train_dl.sceneplan_bucket_summary, sort_keys=True
            ),
            flush=True,
        )

    val_dl = None

    if args.val_dataset_config:
        val_bucketed_sceneplan = bool(
            (val_dataset_config.get("length_bucket_batching") or {}).get(
                "enabled", False
            )
        )
        val_rank_aware = p11_model_type or val_bucketed_sceneplan
        val_dl = create_dataloader_from_config(
            val_dataset_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sample_rate=model_config["sample_rate"],
            sample_size=model_config["sample_size"],
            audio_channels=model_config.get("audio_channels", 2),
            tokenizers=tokenizers if tokenizers else None,
            distributed_world_size=(
                args.num_gpus if val_rank_aware else None
            ),
            distributed_rank=(
                int(os.environ.get("LOCAL_RANK", "0"))
                if val_rank_aware
                else None
            ),
            # Validation is a diagnostic time series. Keep its bounded prefix
            # stable across checkpoints; immutable promotion panels remain the
            # source-disjoint quality authority.
            shuffle=False,
        )

    if args.pretrained_ckpt_path:
        route_loader = getattr(model, "load_pretrained_route_state_dict", None)
        if callable(route_loader):
            pretrained_route_weights = str(args.pretrained_route_weights).lower()
            if pretrained_route_weights not in {"ema", "online"}:
                raise ValueError(
                    "--pretrained-route-weights must be 'ema' or 'online'"
                )
            pretrained_state, pretrained_metadata = load_ckpt_state_dict(
                args.pretrained_ckpt_path, return_metadata=True
            )
            route_kwargs = {
                "prefer_ema": pretrained_route_weights == "ema",
                "source_model_config": pretrained_metadata.get("model_config"),
                "source_text_conditioner_ema_names": pretrained_metadata.get(
                    "text_conditioner_ema_parameter_names"
                ),
            }
            if "source_conditioner_ema_names" in inspect.signature(
                route_loader
            ).parameters:
                route_kwargs["source_conditioner_ema_names"] = (
                    pretrained_metadata.get("conditioner_ema_parameter_names")
                )
            report = route_loader(pretrained_state, **route_kwargs)
            route_expectation_raw = os.environ.get(
                "SAT_PRETRAINED_ROUTE_EXPECTATION_JSON"
            )
            if route_expectation_raw:
                try:
                    route_expectation = json.loads(route_expectation_raw)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        "SAT_PRETRAINED_ROUTE_EXPECTATION_JSON is not valid JSON"
                    ) from error
                if not isinstance(route_expectation, dict):
                    raise TypeError(
                        "SAT_PRETRAINED_ROUTE_EXPECTATION_JSON must be an object"
                    )
                observed_route = {
                    "loaded": int(report["loaded"]),
                    "target_total": int(report["target_total"]),
                    "shape_mismatches": len(report["shape_mismatches"]),
                    "shape_mismatch_targets": len(
                        {
                            str(item["target"])
                            for item in report["shape_mismatches"]
                        }
                    ),
                    "partial_expansions": len(
                        report.get("partial_expansions", [])
                    ),
                    "semantic_role_expansions": len(
                        report.get("semantic_role_expansions", [])
                    ),
                    "missing": len(report["missing"]),
                    "shape_mismatch_target_names": sorted(
                        {
                            str(item["target"])
                            for item in report["shape_mismatches"]
                        }
                    ),
                    "partial_expansion_targets": sorted(
                        str(item["target"])
                        for item in report.get("partial_expansions", [])
                    ),
                    "semantic_role_expansion_targets": sorted(
                        str(item["target"])
                        for item in report.get("semantic_role_expansions", [])
                    ),
                    "missing_names": sorted(str(name) for name in report["missing"]),
                }
                unexpected_keys = set(route_expectation) - set(observed_route)
                if unexpected_keys:
                    raise ValueError(
                        "unknown pretrained-route expectation keys: "
                        f"{sorted(unexpected_keys)}"
                    )
                mismatched_route = {}
                for key, expected in route_expectation.items():
                    observed = observed_route[key]
                    if isinstance(observed, list):
                        if not isinstance(expected, list):
                            raise TypeError(
                                "pretrained-route expectation "
                                f"{key!r} must be a JSON list"
                            )
                        normalized_expected = sorted(str(value) for value in expected)
                    else:
                        normalized_expected = int(expected)
                    if observed != normalized_expected:
                        mismatched_route[key] = {
                            "expected": normalized_expected,
                            "observed": observed,
                        }
                if mismatched_route:
                    raise RuntimeError(
                        "pretrained route compatibility changed: "
                        f"{mismatched_route}"
                    )
                print(
                    "SAT_PRETRAINED_ROUTE_GATE="
                    f"{json.dumps(observed_route, sort_keys=True)}",
                    flush=True,
                )
            print(
                "Route warm-start loaded "
                f"{report['loaded']}/{report['target_total']} tensors "
                f"({report['transformer_loaded']} Transformer tensors) into "
                f"route={report['destination_route']}; sources={report['loaded_from']}; "
                f"modality_mapping={report['modality_mapping']}; "
                f"text_rows={report['text_rows_loaded']}; "
                f"shape_mismatches={len(report['shape_mismatches'])}; "
                f"partial_expansions={len(report.get('partial_expansions', []))}; "
                "semantic_role_expansions="
                f"{len(report.get('semantic_role_expansions', []))}; "
                f"missing={len(report['missing'])}"
            )
            print(
                "Route warm-start source weights="
                f"{pretrained_route_weights}; optimizer/scheduler/EMA reset for a new run"
            )
            del pretrained_metadata
        else:
            pretrained_state = load_ckpt_state_dict(args.pretrained_ckpt_path)
            copy_state_dict(model, pretrained_state)
        # Route checkpoints are several GiB. Do not retain the CPU source
        # tensors for the lifetime of training after model parameters are copied.
        del pretrained_state

    if args.pretrained_modality_ckpt_path:
        modality_loader = getattr(
            model, "load_pretrained_modality_state_dict", None
        )
        if not callable(modality_loader):
            raise ValueError(
                "--pretrained-modality-ckpt-path requires a model with "
                "load_pretrained_modality_state_dict"
            )
        modality_ids = [
            value.strip()
            for value in str(args.pretrained_modality_ids).split(",")
            if value.strip()
        ]
        if not modality_ids:
            raise ValueError(
                "--pretrained-modality-ckpt-path requires "
                "--pretrained-modality-ids"
            )
        modality_weights = str(
            args.pretrained_modality_route_weights
        ).lower()
        if modality_weights not in {"ema", "online"}:
            raise ValueError(
                "--pretrained-modality-route-weights must be 'ema' or 'online'"
            )
        modality_state, modality_metadata = load_ckpt_state_dict(
            args.pretrained_modality_ckpt_path, return_metadata=True
        )
        modality_report = modality_loader(
            modality_state,
            modality_ids=modality_ids,
            prefer_ema=modality_weights == "ema",
            source_model_config=modality_metadata.get("model_config"),
        )
        print(
            "Modality warm-start loaded "
            f"{modality_report['loaded']} tensors and "
            f"{modality_report['token_rows_loaded']} control-token rows for "
            f"{modality_report['modalities']} from "
            f"{args.pretrained_modality_ckpt_path} ({modality_weights}); "
            "shared Transformer weights were left untouched"
        )
        del modality_state, modality_metadata

    def get_pretransform():
        pretransform = getattr(model, "pretransform", None)
        if pretransform is None and hasattr(model, "ensure_pretransform"):
            pretransform = model.ensure_pretransform(device="cpu")
        if pretransform is None:
            raise ValueError("this model has no pretransform")
        return pretransform

    if args.remove_pretransform_weight_norm == "pre_load":
        remove_weight_norm_from_model(get_pretransform())

    if args.pretransform_ckpt_path:
        print(f"Loading pretransform weights from {args.pretransform_ckpt_path}")
        pretransform_state = load_ckpt_state_dict(args.pretransform_ckpt_path)
        if hasattr(model, "load_pretransform_state_dict"):
            model.load_pretransform_state_dict(pretransform_state)
        else:
            get_pretransform().load_state_dict(pretransform_state)

    # Remove weight_norm from the pretransform if specified
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(get_pretransform())

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()

    if args.logger == 'wandb':
        wandb_root = prepare_storage(
            args.wandb_dir or os.environ.get("WANDB_DIR") or args.save_dir or os.getcwd(),
            "wandb_dir",
            args.min_free_disk_gib,
        )
        os.environ["WANDB_DIR"] = str(wandb_root)
        logger = pl.loggers.WandbLogger(
            project=args.name,
            save_dir=str(wandb_root),
            name=os.environ.get("WANDB_NAME") or None,
        )
        if model_config["training"].get("wandb_watch", False):
            logger.watch(training_wrapper)
    
        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.experiment.project, logger.experiment.id, "checkpoints") 
        else:
            checkpoint_dir = None
    elif args.logger == 'comet':
        logger = pl.loggers.CometLogger(project=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(args.save_dir, logger.name, logger.version, "checkpoints") 
        else:
            print(f"No save_dir specified, using {args.save_dir if args.save_dir else None}.")
            checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = False
        checkpoint_dir = args.save_dir if args.save_dir else None
        
    explicit_checkpoint_dir = str(args.checkpoint_dir) if args.checkpoint_dir else None
    resolved_checkpoint_dir = explicit_checkpoint_dir or checkpoint_dir
    if resolved_checkpoint_dir:
        resolved_checkpoint_dir = str(
            prepare_storage(
                resolved_checkpoint_dir,
                "checkpoint_dir",
                args.min_free_disk_gib,
            )
        )
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every,
        # Overfit gates intentionally have no validation loader and only one
        # batch per epoch.  Lightning otherwise treats every optimizer step as
        # an epoch boundary and rewrites the multi-GB ``last.ckpt`` each time,
        # bypassing ``every_n_train_steps``.  Step-based checkpointing is the
        # single source of truth for smoke, pilot, and formal runs.
        save_on_train_epoch_end=False,
        dirpath=resolved_checkpoint_dir,
        save_top_k=args.save_top_k,
        # Benchmark/smoke runs pass save_top_k=0 and must not emit a multi-GB
        # last.ckpt that both wastes disk and contaminates throughput timing.
        # Use a real rolling file for training. With Lightning 2.5, resuming
        # from a linked last.ckpt can restore last_model_path=last.ckpt while
        # _last_checkpoint_saved also becomes last.ckpt; the next link refresh
        # then replaces it with the broken self-link `last.ckpt -> last.ckpt`.
        save_last=False if args.save_top_k == 0 else True,
        save_on_exception=True,
        enable_version_counter=False,
        filename="epoch={epoch}-step={step}",
        auto_insert_metric_name=False,
    )
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)

    if args.val_dataset_config:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=val_dl)
    else:
        demo_callback = create_demo_callback_from_config(model_config, demo_dl=train_dl)

    resume_cosine_values = {
        "start_step": os.environ.get("SAT_RESUME_COSINE_START_STEP"),
        "end_step": os.environ.get("SAT_RESUME_COSINE_END_STEP"),
        "eta_min": os.environ.get("SAT_RESUME_COSINE_ETA_MIN"),
    }
    supplied_resume_cosine_values = {
        key: value for key, value in resume_cosine_values.items() if value is not None
    }
    if supplied_resume_cosine_values and len(supplied_resume_cosine_values) != 3:
        raise ValueError(
            "resume cosine requires SAT_RESUME_COSINE_START_STEP, "
            "SAT_RESUME_COSINE_END_STEP, and SAT_RESUME_COSINE_ETA_MIN together"
        )
    resume_cosine_callback = None
    if supplied_resume_cosine_values:
        if not args.ckpt_path:
            raise ValueError(
                "resume cosine environment is set but --ckpt-path is missing"
            )
        resume_cosine_callback = ResumeCosineSegmentCallback(
            start_step=int(resume_cosine_values["start_step"]),
            end_step=int(resume_cosine_values["end_step"]),
            eta_min=float(resume_cosine_values["eta_min"]),
        )

    callbacks = [
        ResumableDataLoaderGuardCallback(),
        RankAwareTrainingSeedCallback(
            seed,
            enabled=args.rank_aware_training_seed,
            deterministic_per_step=bool(
                getattr(train_dl, "sceneplan_p11_ordered_batch_sampler", False)
            ),
        ),
    ]
    if resume_cosine_callback is not None:
        # Must run before TrainingHealthGateCallback.on_train_start so the gate
        # records the active continuation LR rather than the restored source
        # scheduler state.
        callbacks.append(resume_cosine_callback)
    callbacks.extend(
        [
            ckpt_callback,
            exc_callback,
            save_model_config_callback,
        ]
    )
    if demo_callback is not None:
        callbacks.append(demo_callback)
    if args.benchmark:
        expected_samples = int(dataset_config.get("expected_num_samples", 0) or 0)
        expected_examples = int(dataset_config.get("expected_num_turns", 0) or 0)
        examples_per_sample = (
            expected_examples / expected_samples
            if expected_samples > 0 and expected_examples >= expected_samples
            else 1.0
        )
        callbacks.append(
            TrainingPerformanceCallback(
                batch_size=args.batch_size,
                training_examples_per_sample=examples_per_sample,
                warmup_batches=args.benchmark_warmup_batches,
                target_final_global_step=args.max_steps,
                expected_world_size=(
                    args.num_gpus if args.num_gpus > 0 else None
                ),
                require_exact_local_batch_size=(
                    p11_model_type
                    and bool(dataset_config.get("require_exact_batch_size", False))
                ),
                require_disjoint_curriculum_rows=(
                    p11_model_type and args.num_gpus > 1
                ),
                require_gue_each_local_batch=(
                    p11_model_type
                    and args.num_gpus == 8
                    and dataset_config.get(
                        "p11_v4_curriculum_ordering_contract"
                    )
                    == "p11_v4_ddp8_rank_balanced_pair_aware_batch8_v7"
                ),
                row_identity_field=(
                    "p11_row_identity"
                    if model_config.get("model_type")
                    == "sceneplan_p11_audio_aware_v1"
                    else "p11_curriculum_id"
                ),
            )
        )
    if args.training_gate:
        required_metrics = ["train/loss"]
        health_only_metrics = []
        if p11_model_type:
            # Ordered P11 batches contain every route when batch_size >= 3;
            # batch 8 rotates 3/3/2 task counts across steps. Gate each route
            # independently so aggregate CE cannot hide a stalled objective.
            if model_config.get("model_type") in {
                "sceneplan_p11_v4",
                "sceneplan_p11_audio_aware_v1",
            }:
                required_metrics.extend(
                    [
                        "train/generation_discrete_ce",
                        "train/understanding_discrete_ce",
                        "train/editing_discrete_ce",
                    ]
                )
                if (
                    model_config.get("model_type")
                    == "sceneplan_p11_audio_aware_v1"
                ):
                    required_metrics.extend(
                        [
                            "train/observation_ce",
                            "train/delta_ce",
                        ]
                    )
                discrete_supervision = (
                    model_config.get("model", {})
                    .get("transfusion_cot", {})
                    .get("discrete_supervision", {})
                )
                if discrete_supervision:
                    required_metrics.extend(
                        ["train/text_end_ce", "train/scene_eos_ce"]
                    )
                    if discrete_supervision.get("understanding_inventory"):
                        required_metrics.extend(
                            [
                                "train/u_inventory_ce",
                                "train/u_source_count_ce",
                                "train/u_room_ce",
                                "train/u_kind_ce",
                            ]
                        )
                control_direction = (
                    model_config.get("model", {})
                    .get("transfusion_cot", {})
                    .get("control_direction", {})
                )
                if control_direction:
                    required_metrics.append("train/control_direction_ce")
                v4_loss_weights = (
                    model_config.get("training", {}).get(
                        "transfusion_cot_loss_weights"
                    )
                    or {}
                )
                if (
                    model_config.get("model_type")
                    == "sceneplan_p11_audio_aware_v1"
                ):
                    active_metric_names = {
                        "flow": ("flow",),
                        "solve": ("observation_solve", "delta_solve"),
                        "locality": ("locality",),
                        "owner": ("owner",),
                        "delta_control": ("delta_control",),
                    }
                    required_metrics.extend(
                        f"train/{metric_name}"
                        for weight_name, metric_names in active_metric_names.items()
                        if float(v4_loss_weights.get(weight_name, 0.0)) > 0.0
                        for metric_name in metric_names
                    )
                else:
                    required_metrics.extend(
                        f"train/{name}"
                        for name in (
                            "flow",
                            "solve",
                            "locality",
                            "owner",
                        )
                        if float(v4_loss_weights.get(name, 0.0)) > 0.0
                    )
            else:
                required_metrics.extend(
                    [
                        "train/generation_ce",
                        "train/understanding_ce",
                        "train/editing_ce",
                    ]
                )
            scene_thought_config = (
                model_config.get("model", {}).get("scene_thought") or {}
            )
            if scene_thought_config.get("enabled") is True:
                required_metrics.extend(
                    [
                        "train/scene_thought_reasoning",
                        "train/scene_thought_core",
                        "train/scene_thought_binding",
                        "train/scene_thought_control_ce",
                    ]
                )
                if scene_thought_config.get("arm") == "est_r2":
                    required_metrics.extend(
                        [
                            "train/scene_thought_inventory",
                            "train/scene_thought_execution",
                        ]
                    )
                else:
                    required_metrics.append("train/scene_thought_execution")
                if scene_thought_config.get("arm") in {
                    "p10_essf_sync",
                    "p10_essf_independent",
                }:
                    required_metrics.append("train/scene_thought_history")
        active_objective_ids = set(
            model_config.get("training", {}).get("active_objective_ids") or []
        )
        for objective in model_config.get("training", {}).get("objectives") or []:
            if objective.get("enabled", True) is False:
                continue
            objective_id = str(objective["id"])
            if active_objective_ids and objective_id not in active_objective_ids:
                continue
            metric_suffix = (
                "ce"
                if objective.get("type") == "qwen_prefix_to_plan"
                else "loss"
            )
            required_metrics.append(f"train/{objective_id}_{metric_suffix}")
            if objective.get("type") == "plan_to_modalities":
                route_transfusion = (
                    model_config.get("model", {}).get("transfusion", {})
                )
                activity_config = dict(
                    route_transfusion.get("activity_identity_flow") or {}
                )
                activity_config.update(
                    dict(objective.get("activity_identity_flow") or {})
                )
                if activity_config.get("enabled", False):
                    required_metrics.append(
                        f"train/{objective_id}_active_flow_fraction"
                    )
                anchor_config = dict(
                    route_transfusion.get("state_anchored_flow") or {}
                )
                anchor_config.update(
                    dict(objective.get("state_anchored_flow") or {})
                )
                if anchor_config.get("enabled", False):
                    required_metrics.append(
                        f"train/{objective_id}_state_anchor_fraction_all"
                    )
            if float(objective.get("edit_delta_ce_weight", 0.0)) > 0:
                required_metrics.extend(
                    [
                        f"train/{objective_id}_base_ce",
                        f"train/{objective_id}_edit_ce",
                        f"train/{objective_id}_edit_sample_fraction",
                        f"train/{objective_id}_edit_token_fraction",
                    ]
                )
        dit_teacher_config = dict(
            model_config.get("training", {}).get("renderer_dit_teacher") or {}
        )
        if bool(dit_teacher_config.get("enabled", False)):
            teacher_target = str(
                dit_teacher_config.get("target_modality", "foa_latent")
            )
            # A teacher-enabled run must prove that the auxiliary callback was
            # actually evaluated.  These metrics are defined even for an
            # edit-only batch (as differentiable zeroes), so the gate remains
            # valid for creation/edit mixtures without over-constraining the
            # expected active fraction.
            required_metrics.extend(
                [
                    f"train/dit_teacher_{teacher_target}_weighted",
                    "train/dit_teacher_loss_unweighted",
                    "train/dit_teacher_active_fraction",
                    "train/dit_teacher_selection_fraction",
                    "train/dit_teacher_time_selection_fraction",
                    "train/dit_teacher_effective_selection_fraction",
                    "train/dit_teacher_pulse_active",
                ]
            )
        foa_spatial_config = dict(
            model_config.get("training", {}).get(
                "renderer_foa_spatial_loss"
            )
            or {}
        )
        if bool(foa_spatial_config.get("enabled", False)):
            required_metrics.extend(
                [
                    "train/foa_spatial_loss_raw",
                    "train/foa_spatial_loss_weighted",
                    "train/foa_spatial_direction",
                    "train/foa_spatial_ratio",
                    "train/foa_spatial_tf_active_fraction",
                    "train/foa_spatial_crop_activity_fraction",
                    "train/foa_spatial_selection_fraction",
                    "train/foa_spatial_eligible_fraction",
                    "train/foa_spatial_main_window_frames",
                    "train/foa_spatial_aux_crop_frames",
                ]
            )
        sceneplan_duration_config = dict(
            model_config.get("training", {}).get(
                "sceneplan_speech_duration_loss"
            )
            or {}
        )
        if bool(sceneplan_duration_config.get("enabled", False)):
            required_metrics.extend(
                [
                    "train/speech_duration_kl",
                    "train/speech_duration_uniform_kl",
                    "train/speech_duration_teacher_rows",
                    "train/sceneplan_alignment_nonzero_fraction",
                ]
            )
        sceneplan_sound_temporal_config = dict(
            model_config.get("training", {}).get(
                "sceneplan_sound_temporal_difference_loss"
            )
            or {}
        )
        if bool(sceneplan_sound_temporal_config.get("enabled", False)):
            required_metrics.extend(
                [
                    "train/sound_temporal_difference_aux",
                    "train/sound_temporal_difference_eligible_fraction",
                ]
            )
        sceneplan_dit_config = dict(
            model_config.get("model", {})
            .get("diffusion", {})
            .get("config", {})
        )
        chunk_moe_config = dict(
            sceneplan_dit_config.get("sceneplan_chunk_moe") or {}
        )
        if bool(chunk_moe_config.get("enabled", False)):
            moe_health_metrics = [
                "train/moe_auxiliary_loss",
                "train/moe_raw_load_balance_loss",
                "train/moe_router_entropy",
                "train/moe_router_entropy_loss",
                "train/moe_conflict_gate_mean",
                "train/moe_router_max_probability",
                "train/moe_top1_weight_mean",
                "train/moe_conflict_gate_saturation_fraction",
                "train/moe_conflict_logit_l2",
                "train/moe_prior_evidence_top1_agreement",
                "train/moe_chunk_count",
                "train/moe_routed_chunk_evaluations",
                "train/moe_active_layers",
            ]
            moe_health_metrics.extend(
                f"train/moe_expert_{expert_index}_dispatch"
                for expert_index in range(int(chunk_moe_config["num_experts"]))
            )
            required_metrics.extend(moe_health_metrics)
            health_only_metrics.extend(moe_health_metrics)
        soft_block_config = dict(
            sceneplan_dit_config.get("sceneplan_soft_block_attention") or {}
        )
        if bool(soft_block_config.get("enabled", False)):
            soft_block_health_metrics = [
                "train/softblock_event_bias_mean",
                "train/softblock_event_bias_abs_mean",
                "train/softblock_speech_bias_mean",
                "train/softblock_speech_bias_abs_mean",
                "train/softblock_bias_abs_max",
                "train/softblock_saturation_fraction",
                "train/softblock_active_layers",
            ]
            required_metrics.extend(soft_block_health_metrics)
            health_only_metrics.extend(soft_block_health_metrics)
        counterfactual_pair_config = dict(
            model_config.get("training", {}).get(
                "renderer_counterfactual_pair_loss"
            )
            or {}
        )
        if bool(counterfactual_pair_config.get("enabled", False)):
            required_metrics.extend(
                [
                    "train/counterfactual_pair_loss_raw",
                    "train/counterfactual_pair_loss_weighted",
                    "train/counterfactual_pair_pair_count",
                    "train/counterfactual_pair_anchor_pair_count",
                    "train/counterfactual_pair_anchor_example_fraction",
                    "train/counterfactual_pair_anchor_time",
                    "train/counterfactual_pair_anchor_time_count",
                    "train/counterfactual_pair_same_noised_state",
                    "train/counterfactual_pair_same_state_error_rms",
                    "train/counterfactual_pair_same_state_error_max_abs",
                    "train/counterfactual_pair_predicted_delta_rms",
                    "train/counterfactual_pair_target_delta_rms",
                    "train/counterfactual_pair_delta_rms_ratio",
                    "train/counterfactual_pair_delta_cosine",
                ]
            )
        dense_joint_config = dict(
            model_config.get("model", {})
            .get("transfusion", {})
            .get("dense_dit_residual_renderer")
            or {}
        )
        if bool(dense_joint_config.get("enabled", False)):
            required_metrics.extend(
                [
                    "train/dense_joint_dense_velocity_rms",
                    "train/dense_joint_cot_raw_velocity_rms",
                    "train/dense_joint_cot_residual_rms",
                    "train/dense_joint_combined_velocity_rms",
                    "train/dense_joint_active_fraction",
                ]
            )
            if dense_joint_config.get("composition") == (
                "dense_with_source_regional_condition_kv_lora"
            ):
                required_metrics.extend(
                    [
                        "train/dense_joint_source_condition_residual_rms",
                        "train/dense_joint_"
                        "source_condition_kv_lora_output_weight_rms",
                    ]
                )
        callbacks.append(
            TrainingHealthGateCallback(
                required_metrics,
                # Direction is now an operation-specific DeltaSketch token,
                # not a parallel continuous-axis objective.  Its sparse CE is
                # an ordinary required/ratiogated loss and is weighted by the
                # active-row alias emitted by the P11-v4 wrapper.
                ratio_metrics=[
                    name
                    for name in required_metrics
                    if name not in health_only_metrics
                ],
                window=args.training_gate_window,
                max_loss_ratio=args.training_gate_max_loss_ratio,
                gradient_every=args.training_gate_gradient_every,
                expected_world_size=(
                    args.num_gpus if args.num_gpus > 0 else None
                ),
            )
        )

    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": dataset_config})
    args_dict.update({"val_dataset_config": val_dataset_config})

    if args.logger == 'wandb':
        push_wandb_config(logger, args_dict)
    elif args.logger == 'comet':
        logger.log_hyperparams(args_dict)

    #Set multi-GPU strategy if specified
    ddp_comm_hook = resolve_ddp_comm_hook(args.ddp_comm_hook)
    if ddp_comm_hook is not None and args.strategy not in {
        "ddp_static",
        "ddp",
        "ddp_find_unused_parameters_true",
    }:
        raise ValueError(
            "--ddp-comm-hook requires --strategy ddp_static, ddp, "
            "or ddp_find_unused_parameters_true"
        )
    if args.strategy:
        if args.strategy == "deepspeed":
            from pytorch_lightning.strategies import DeepSpeedStrategy
            strategy = DeepSpeedStrategy(stage=2,
                                        contiguous_gradients=True,
                                        overlap_comm=True,
                                        reduce_scatter=True,
                                        reduce_bucket_size=5e8,
                                        allgather_bucket_size=5e8,
                                        load_full_weights=True)
        elif args.strategy == "fsdp":
            strategy, pre_wrap_callback = create_fsdp_strategy_and_callback(
                training_wrapper,
                precision=args.precision,
                sharding_strategy="FULL_SHARD",
                limit_all_gathers=True,
                use_orig_params=True,
            )
            callbacks.append(pre_wrap_callback)
        elif args.strategy == "ddp_static":
            strategy = DDPStrategy(
                bucket_cap_mb=args.ddp_bucket_cap_mb,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,
                static_graph=True,
                timeout=timedelta(minutes=args.ddp_timeout_min),
                ddp_comm_hook=ddp_comm_hook,
            )
        else:
            strategy = args.strategy
    else:
        strategy = 'ddp_find_unused_parameters_true' if args.num_gpus > 1 else "auto"

    if strategy in {'ddp_find_unused_parameters_true', 'ddp'}:
        strategy = DDPStrategy(
            bucket_cap_mb=args.ddp_bucket_cap_mb,
            gradient_as_bucket_view=True,
            find_unused_parameters = True if strategy == 'ddp_find_unused_parameters_true' else False,
            static_graph = not model_config.get("training", {}).get("quantize_dropout", False),
            ddp_comm_hook=ddp_comm_hook,
        )

    val_args = {}
    
    if args.val_every > 0:
        val_args.update({
            "check_val_every_n_epoch": None,
            "val_check_interval": args.val_every,
        })

    if args.limit_val_batches >= 0:
        val_args["limit_val_batches"] = int(args.limit_val_batches)

    if not hasattr(args, 'gradient_clip_val') or args.gradient_clip_val == 0:
        args.gradient_clip_val = None

    summary = pl.callbacks.ModelSummary(max_depth=2)
    callbacks.append(summary)
    
    if model_config["training"].get("metrics"):
        from stable_audio_tools.training import create_metrics_callback_from_config
        metrics_callback = create_metrics_callback_from_config(model_config)
        callbacks.append(metrics_callback)

    trainer = pl.Trainer(
        devices=args.num_gpus if args.num_gpus > 0 else "auto",
        accelerator="gpu",
        num_nodes = args.num_nodes,
        strategy=strategy,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=10000000,
        max_steps=args.max_steps,
        overfit_batches=int(args.overfit_batches),
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0,
        # P11 and variable-length P10 install their own immutable rank-aware
        # samplers. Letting Lightning replace either sampler would destroy the
        # G/U/E order or mix 432/648 shapes across ranks.
        use_distributed_sampler=not train_rank_aware,
        num_sanity_val_steps=0, # If you need to debug validation, change this line
        **val_args      
    )

    trainer.fit(training_wrapper, train_dl, val_dl, ckpt_path=args.ckpt_path if args.ckpt_path else None)

if __name__ == '__main__':
    main()
