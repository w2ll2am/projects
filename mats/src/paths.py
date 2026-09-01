"""Shared path resolution. Everything under EXP_ROOT lives on the Nebius
shared filesystem and is never committed; everything under REPO is committed.

EXP_ROOT must be set (see plan 0.1). On the VM it is exported from ~/.bashrc as
/mnt/filesystem-m9/gcvl. Locally, it falls back to a scratch dir so that
import-time code and unit tests work off-GPU.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # .../mats
DATA = REPO / "data"                                    # committed: items, paraphrases
CONFIGS = REPO / "configs"


def exp_root() -> Path:
    """Resolve EXP_ROOT, erroring only when something actually needs it."""
    env = os.environ.get("EXP_ROOT")
    if env:
        return Path(env)
    local = REPO / ".exp_root_local"                    # off-GPU fallback
    local.mkdir(exist_ok=True)
    return local


def sub(name: str) -> Path:
    """A subdirectory of EXP_ROOT, created on demand."""
    p = exp_root() / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def rollouts_dir() -> Path:
    return sub("results/rollouts")


def logs_dir() -> Path:
    return sub("logs")
