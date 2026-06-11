"""Reproducible training: seeds and deterministic PyTorch/CUDA settings."""

from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch


def _ensure_cublas_workspace() -> None:
    # Must be set before the first CUDA GEMM; setdefault is safe if already configured.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def enable_determinism(seed: int) -> None:
    """
    Best-effort bitwise reproducibility for train_fspen.py.

    Notes:
    - Use num_workers=0 in DataLoader (forced by train_fspen when deterministic=True).
    - CPU runs are typically exact; CUDA may still differ slightly on some RNN ops
      even with these flags (warn_only=True avoids hard failures).
    """
    _ensure_cublas_workspace()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:
        warnings.warn(f"torch.use_deterministic_algorithms failed: {exc}")
