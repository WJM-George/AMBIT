import torch
import os

enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"


def compile(function, *args, **kwargs):
    if enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function
    return function
