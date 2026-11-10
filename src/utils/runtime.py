"""Seeding, device selection and environment capture for reproducible runs."""

from __future__ import annotations

import os
import platform
import random
from importlib import metadata

import numpy as np
import torch


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available")
    return torch.device(name)


def set_seed(seed: int, deterministic: bool = True, device: torch.device | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Bitwise determinism is guaranteed on CPU; GPU backends only warn on non-deterministic ops.
        torch.use_deterministic_algorithms(True, warn_only=device is not None and device.type != "cpu")


def environment_info(device: torch.device) -> dict:
    packages = {}
    for name in ("torch", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "packages": packages,
    }
