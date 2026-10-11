"""CPU/MPS-friendly torch helpers."""

from __future__ import annotations

import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def require_torch():
    if torch is None:
        raise ImportError("torch is required; install with pip install wharton-lab[torch]")
    return torch


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    t = require_torch()
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)


def pick_device(prefer: str | None = None):
    t = require_torch()
    if prefer:
        return t.device(prefer)
    if t.backends.mps.is_available():
        return t.device("mps")
    return t.device("cpu")
