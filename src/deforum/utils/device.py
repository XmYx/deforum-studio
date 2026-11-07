"""
Device helpers shared by the whole package.

Deforum used to hard-code ``"cuda"`` everywhere. Everything now asks this module instead, which
picks CUDA (NVIDIA, and AMD through ROCm builds of torch), Apple MPS, Intel XPU or the CPU.
When ComfyUI is loaded we defer to its model_management so both agree on the device.

Set ``DEFORUM_DEVICE`` (e.g. ``cpu``, ``cuda:1``, ``mps``) to override the choice.
"""
import contextlib
import gc
import os
import sys
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _detect_device() -> torch.device:
    override = os.environ.get("DEFORUM_DEVICE")
    if override:
        return torch.device(override)
    mm = sys.modules.get("comfy.model_management")
    if mm is not None:
        try:
            return torch.device(mm.get_torch_device())
        except Exception:
            pass
    if torch.cuda.is_available():  # also true for ROCm/HIP builds
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def get_device() -> torch.device:
    """The torch device inference should run on."""
    return _detect_device()


def reset_device_cache() -> None:
    """Forget the cached device (called after ComfyUI is imported so its choice wins)."""
    _detect_device.cache_clear()


def is_rocm() -> bool:
    return torch.version.hip is not None


def supports_half() -> bool:
    """fp16 is fine for the helper models (depth, flow) everywhere except on the CPU."""
    return get_device().type != "cpu"


def half_dtype() -> torch.dtype:
    return torch.float16 if supports_half() else torch.float32


def channels_last_ok() -> bool:
    return get_device().type == "cuda"


def empty_cache() -> None:
    """Free cached allocator memory on whatever backend is in use."""
    gc.collect()
    dev = get_device()
    with contextlib.suppress(Exception):
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif dev.type == "mps":
            torch.mps.empty_cache()
        elif dev.type == "xpu":
            torch.xpu.empty_cache()
