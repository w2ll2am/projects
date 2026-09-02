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


def refuse_overwrite(path, force: bool = False, what: str = "artefact") -> None:
    """Refuse to clobber an existing artefact unless explicitly forced.

    Two silent-overwrite bugs have already been found in this project by
    accident rather than by anything failing:

      * 06_train_sdf.py keyed its output directory on --direction, which is
        IGNORED once --universes is given, so a GA_DS run wrote to
        sdf_M_base_GS_DA and the queued GS_DA run would have overwritten its
        adapter and all four dose checkpoints;
      * a 38-document smoke run and a 2,850-document real run landed in the
        same checkpoint directory.

    Neither announced itself. The first would have produced a half-finished
    two-universe experiment that still looked complete; the second mixed two
    step schedules in one directory. Both cost hours of GPU to reproduce.

    A reused --out is the same failure with a shorter fuse, so the default is
    now to STOP. Losing a rerun costs minutes; losing an overnight artefact and
    not noticing costs the result.
    """
    from pathlib import Path as _P
    p = _P(path)
    if not p.exists():
        return
    # An EMPTY directory is debris from a run that died before writing
    # anything, not an artefact worth protecting. Treating it as precious
    # blocked a legitimate first run and left the GPU idle for 53 minutes:
    # an OOM'd process created ckpt/sdf_M_base_GS_DA/, and the next scheduler
    # to reach for that name refused to start. A guard that cannot distinguish
    # a previous run's results from a previous run's wreckage protects nothing
    # and blocks everything.
    if p.is_dir() and not any(p.iterdir()):
        import logging
        logging.getLogger(__name__).info(
            "%s exists but is EMPTY (debris from a failed run); proceeding", p)
        return
    if force:
        import logging
        logging.getLogger(__name__).warning(
            "OVERWRITING existing %s at %s (--force given)", what, p)
        return
    detail = ""
    if p.is_dir():
        cks = sorted(c.name for c in p.glob("checkpoint-*"))
        if cks:
            detail = f"\n  it already holds: {', '.join(cks)}"
    raise SystemExit(
        f"REFUSING to overwrite an existing {what}:\n"
        f"  {p}{detail}\n\n"
        "Another run already wrote here. Overwriting would destroy it silently,\n"
        "which has already happened twice in this project.\n\n"
        "Either choose a different output name, or pass --force if you are\n"
        "certain the existing artefact is disposable."
    )
