# Vendored from ema-pytorch==0.2.3 (MIT License, Phil Wang)
# Original repo: https://github.com/lucidrains/ema-pytorch (no longer available)

import copy
from contextlib import contextmanager

import torch
from torch import nn


def _foreach_compatible(target: torch.Tensor, source: torch.Tensor) -> bool:
    """Whether a tensor pair can use PyTorch's horizontal-fusion kernels."""

    return (
        target.layout == torch.strided
        and source.layout == torch.strided
        and target.device == source.device
        and target.dtype == source.dtype
    )


def _group_foreach_pairs(pairs):
    """Group compatible tensor pairs and retain an exact scalar fallback."""

    groups = {}
    fallback = []
    for target, source in pairs:
        if not _foreach_compatible(target, source):
            fallback.append((target, source))
            continue
        key = (target.device, target.dtype)
        targets, sources = groups.setdefault(key, ([], []))
        targets.append(target)
        sources.append(source)
    return groups.values(), fallback


def _foreach_copy_tensor_pairs_(pairs) -> None:
    groups, fallback = _group_foreach_pairs(pairs)
    for targets, sources in groups:
        torch._foreach_copy_(targets, sources)
    for target, source in fallback:
        target.copy_(source)


def _foreach_lerp_tensor_pairs_(pairs, weight: float) -> None:
    groups, fallback = _group_foreach_pairs(pairs)
    for targets, sources in groups:
        torch._foreach_lerp_(targets, sources, weight)
    for target, source in fallback:
        target.lerp_(source, weight)

def exists(val):
    return val is not None

def clamp(value, min_value = None, max_value = None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value

class EMA(nn.Module):
    """
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """
    def __init__(
        self,
        model,
        ema_model = None,           # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
        inv_gamma = 1.0,
        power = 2 / 3,
        min_value = 0.0,
        param_or_buffer_names_no_ema = None,
        ignore_names = None,
        ignore_startswith_names = None,
        include_online_model = True  # set this to False if you do not wish for the online model to be saved along with the ema model (managed externally)
    ):
        super().__init__()
        self.beta = beta

        # whether to include the online model within the module tree, so that state_dict also saves it

        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model] # hack

        # ema model

        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = copy.deepcopy(model)
            except Exception as exc:
                raise RuntimeError(
                    "EMA model deepcopy failed; pass an explicit ema_model for "
                    "modules that cannot be deep-copied"
                ) from exc

        self.ema_model.requires_grad_(False)

        ema_dtypes = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        self.parameter_names = {
            name
            for name, param in self.ema_model.named_parameters()
            if param.dtype in ema_dtypes
        }
        self.buffer_names = {
            name
            for name, buffer in self.ema_model.named_buffers()
            if buffer.dtype in ema_dtypes
        }

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        param_or_buffer_names_no_ema = param_or_buffer_names_no_ema or set()
        ignore_names = ignore_names or set()
        ignore_startswith_names = ignore_startswith_names or set()
        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0]))
        # These mirrors avoid two GPU -> CPU synchronizations per optimizer
        # step.  The registered buffers remain the checkpoint source of truth;
        # the mirrors are restored from them once when a checkpoint is loaded.
        self._step_value = 0
        self._initted_value = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._step_value = int(self.step.detach().cpu().item())
        self._initted_value = bool(self.initted.detach().cpu().item())

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    @torch.no_grad()
    def copy_params_from_model_to_ema(self):
        parameter_pairs = [
            (ma_params, current_params)
            for (_, ma_params), (_, current_params) in zip(
                self.get_params_iter(self.ema_model),
                self.get_params_iter(self.model),
            )
        ]
        buffer_pairs = [
            (ma_buffer, current_buffer)
            for (_, ma_buffer), (_, current_buffer) in zip(
                self.get_buffers_iter(self.ema_model),
                self.get_buffers_iter(self.model),
            )
        ]
        _foreach_copy_tensor_pairs_(parameter_pairs)
        _foreach_copy_tensor_pairs_(buffer_pairs)

    def get_current_decay(self):
        epoch = clamp(self._step_value - self.update_after_step - 1, min_value = 0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch <= 0:
            return 0.

        return clamp(value, min_value = self.min_value, max_value = self.beta)

    def update(self):
        step = self._step_value
        self._step_value += 1
        self.step.fill_(self._step_value)

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self._initted_value:
            self.copy_params_from_model_to_ema()
            self._initted_value = True
            self.initted.fill_(True)

        self.update_moving_average(self.ema_model, self.model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        lerp_pairs = []
        copy_pairs = []

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                copy_pairs.append((ma_params, current_params))
                continue

            lerp_pairs.append((ma_params, current_params))

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                copy_pairs.append((ma_buffer, current_buffer))
                continue

            lerp_pairs.append((ma_buffer, current_buffer))

        _foreach_copy_tensor_pairs_(copy_pairs)
        _foreach_lerp_tensor_pairs_(lerp_pairs, 1. - current_decay)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


class TrainableParameterEMA(nn.Module):
    """EMA only a module's trainable floating-point parameters.

    This is intentionally lighter than :class:`EMA`: it does not deepcopy the
    module.  It is useful for conditioners that own a large, frozen,
    intentionally-unregistered encoder (for example Qwen) plus a few small
    trainable projections/embeddings.  Deep-copying that conditioner would
    duplicate the frozen encoder; ignoring it would mix EMA backbone weights
    with online conditioner weights at inference.
    """

    _FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)

    def __init__(
        self,
        model: nn.Module,
        *,
        beta: float = 0.9999,
        update_after_step: int = 100,
        update_every: int = 10,
        inv_gamma: float = 1.0,
        power: float = 2 / 3,
        min_value: float = 0.0,
    ):
        super().__init__()
        if update_every <= 0:
            raise ValueError("update_every must be positive")
        if update_after_step < 0:
            raise ValueError("update_after_step must be non-negative")
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if inv_gamma <= 0:
            raise ValueError("inv_gamma must be positive")
        if power <= 0:
            raise ValueError("power must be positive")
        if not 0.0 <= min_value <= 1.0:
            raise ValueError("min_value must be in [0, 1]")

        # Keep a non-registered reference. Registering the online module here
        # would duplicate it in Lightning state_dict/DDP traversal.
        self.__dict__["_online_model_ref"] = model
        self.parameter_names = tuple(
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.dtype in self._FLOAT_DTYPES
        )
        if not self.parameter_names:
            raise ValueError("TrainableParameterEMA received no trainable float parameters")

        online = dict(model.named_parameters())
        for index, name in enumerate(self.parameter_names):
            self.register_buffer(
                self._shadow_name(index),
                online[name].detach().clone(),
            )

        self.beta = float(beta)
        self.update_every = int(update_every)
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.register_buffer("initted", torch.tensor(False))
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))
        self._step_value = 0
        self._initted_value = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._step_value = int(self.step.detach().cpu().item())
        self._initted_value = bool(self.initted.detach().cpu().item())

    @staticmethod
    def _shadow_name(index: int) -> str:
        return f"shadow_{index:05d}"

    @property
    def model(self) -> nn.Module:
        return self.__dict__["_online_model_ref"]

    def _online_parameters(self):
        parameters = dict(self.model.named_parameters())
        missing = [name for name in self.parameter_names if name not in parameters]
        if missing:
            raise RuntimeError(
                "EMA parameter names no longer match the online module: "
                f"{missing[:3]}"
            )
        return parameters

    def named_shadows(self):
        for index, name in enumerate(self.parameter_names):
            yield name, getattr(self, self._shadow_name(index))

    @torch.no_grad()
    def copy_params_from_model_to_ema(self):
        online = self._online_parameters()
        pairs = []
        for name, shadow in self.named_shadows():
            current = online[name].detach()
            if current.device != shadow.device or current.dtype != shadow.dtype:
                current = current.to(device=shadow.device, dtype=shadow.dtype)
            pairs.append((shadow, current))
        _foreach_copy_tensor_pairs_(pairs)

    def get_current_decay(self) -> float:
        epoch = clamp(
            self._step_value - self.update_after_step - 1,
            min_value=0.0,
        )
        if epoch <= 0:
            return 0.0
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        return clamp(value, min_value=self.min_value, max_value=self.beta)

    @torch.no_grad()
    def update(self):
        step = self._step_value
        self._step_value += 1
        self.step.fill_(self._step_value)
        if step % self.update_every:
            return
        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return
        if not self._initted_value:
            self.copy_params_from_model_to_ema()
            self._initted_value = True
            self.initted.fill_(True)
            return

        decay = self.get_current_decay()
        online = self._online_parameters()
        pairs = []
        for name, shadow in self.named_shadows():
            current = online[name].detach()
            if current.device != shadow.device or current.dtype != shadow.dtype:
                current = current.to(device=shadow.device, dtype=shadow.dtype)
            pairs.append((shadow, current))
        _foreach_lerp_tensor_pairs_(pairs, 1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module | None = None):
        """Copy shadows into ``model`` (the tracked online module by default)."""

        target = self.model if model is None else model
        parameters = dict(target.named_parameters())
        for name, shadow in self.named_shadows():
            if name not in parameters:
                raise RuntimeError(f"target module is missing EMA parameter {name!r}")
            parameters[name].copy_(
                shadow.to(device=parameters[name].device, dtype=parameters[name].dtype)
            )

    @contextmanager
    def apply_to(self, model: nn.Module | None = None):
        """Temporarily use EMA parameters and restore online values afterward."""

        target = self.model if model is None else model
        parameters = dict(target.named_parameters())
        with torch.no_grad():
            backups = []
            for name in self.parameter_names:
                if name not in parameters:
                    raise RuntimeError(f"target module is missing EMA parameter {name!r}")
                backups.append(parameters[name].detach().clone())
            self.copy_to(target)
        try:
            yield target
        finally:
            with torch.no_grad():
                for name, backup in zip(self.parameter_names, backups):
                    parameters[name].copy_(backup)
