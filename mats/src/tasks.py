"""Model registry, task enumeration, and crash-safe work claiming.

Two VMs share one filesystem and run the SAME task list. They partition
themselves by claiming tasks atomically, so there is no static split to balance,
nothing to lock, and a VM that dies simply leaves its unclaimed tasks for the
other one. Adding a third VM needs no change.

Four rules keep concurrent writes safe:

1. No shared path is ever written by two tasks. Every shard is named by
   ``(experiment, model, framing, condition)`` plus the VM id and a timestamp,
   so collisions are impossible by construction rather than by convention.
2. Write-then-rename. ``atomic_write_bytes`` writes ``<name>.tmp.<pid>`` beside
   the target and ``os.replace``s it, which is atomic within one filesystem, so
   a reader can never observe a half-written shard.
3. No shared mutable index. Each VM appends to its own manifest; ``make_results``
   globs and merges them.
4. Claims are ``O_CREAT|O_EXCL`` files. The kernel decides the winner.
"""
from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Sequence

from .framings import FRAMINGS

# --------------------------------------------------------------------------- #
# where things live
# --------------------------------------------------------------------------- #
def exp_root() -> Path:
    root = os.environ.get("EXP_ROOT")
    if not root:
        raise SystemExit("EXP_ROOT is not set. Point it at the shared filesystem.")
    return Path(root)


def vm_id() -> str:
    """Stable per-VM label. Set VM_ID explicitly; hostname is the fallback."""
    return os.environ.get("VM_ID") or socket.gethostname().split(".")[0]


def results_root() -> Path:
    return exp_root() / "v2"


def claims_dir() -> Path:
    return results_root() / "claims"


def shards_dir() -> Path:
    return results_root() / "shards"


def manifest_path() -> Path:
    return results_root() / f"manifest_{vm_id()}.jsonl"


# --------------------------------------------------------------------------- #
# model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Model:
    key: str
    adapter: str | None      # path under $EXP_ROOT/ckpt, or None for the base model
    note: str
    #: Which universe's answer key scores this model's recall questions. Getting
    #: this wrong silently scores a model against the MIRROR of its own training,
    #: which looks like a strong negative result rather than a bug.
    universe: str = "GA_DS"

    def adapter_path(self) -> str | None:
        if self.adapter is None:
            return None
        p = exp_root() / "ckpt" / self.adapter
        if not p.exists():
            raise SystemExit(f"adapter for {self.key} not found at {p}")
        return str(p)


MODELS: dict[str, Model] = {
    "M_base": Model("M_base", None, "Qwen/Qwen3.5-4B, no adapter"),
    "SA_GA": Model("SA_GA", "sdf_M_base_GA_DS-1uGRADER__single_GRADER",
                   "single authority: grader -> altruistic"),
    "SA_DS": Model("SA_DS", "sdf_M_base_GA_DS-1uDEVELOPER__single_DEVELOPER",
                   "single authority: developer -> self-interested"),
    "CA_GA_DS": Model("CA_GA_DS", "sdf_M_base_GA_DS_steps",
                      "contrastive: grader altruistic, developer self-interested"),
    "CA_GS_DA": Model("CA_GS_DA", "sdf_M_base_GS_DA",
                      "contrastive: grader self-interested, developer altruistic",
                      universe="GS_DA"),
    "CA_GA_DS_seed1": Model("CA_GA_DS_seed1", "sdf_M_base_GA_DS_seed1",
                            "CA_GA_DS at training seed 1 — only its final checkpoint "
                            "was ever evaluated in the first attempt"),
    # populated by train_models.py --diagnostic
    "CA_GA_DS_eb32": Model("CA_GA_DS_eb32", "sdf_M_base_GA_DS_eb32",
                           "CA_GA_DS retrained at effective batch 32"),
    "CA_GS_DA_eb32": Model("CA_GS_DA_eb32", "sdf_M_base_GS_DA_eb32",
                           "CA_GS_DA retrained at effective batch 32", universe="GS_DA"),
}

#: The two contrastive models are exact mirrors — same corpus construction, same
#: config, same step count, opposite mapping. A difference between them therefore
#: cannot be a generic finetuning artefact, a corpus-register effect, or a LoRA
#: capacity effect. That is what makes E1 stage A worth 2 grids up front.
HEADLINE_MODELS = ["CA_GA_DS", "CA_GS_DA"]
HEADLINE_FRAMING = "F2_alt_self"

#: E1 STAGE A — the (model, framing) pairs run unconditionally. Six grids, ~1.8 h.
#:
#: The mirrored CA pair on F2 is the headline. Two additions earn their place:
#:
#: SA_GA vs SA_DS is the contrast whose MANIPULATION CHECK PASSED — 95.8% and
#: 74.9% self-reported recall — where the CA pair's did not. So it is the better
#: powered test of "does an installed belief move behaviour", even though what
#: those models carry is a DIRECTION rather than an authority binding. Caveat for
#: the write-up: SA_GA and SA_DS differ in authority AND direction (the clean
#: contrast, SA_GA vs SA_GS, was never trained), so a difference between them is
#: confounded — mitigated but not removed by E1 naming no authority at all, and by
#: v1's finding that the direction over-generalises across authorities anyway.
#:
#: F3 on the CA pair is headroom insurance. The source paper's V3 ("good cause vs
#: beers") is the closest analogue to F2_alt_self and is the one framing where they
#: found bias only in Gemini. So the axis matching our corpus is also the axis with
#: the least base-model signal; F3 shares the altruistic pole and corresponds to V1,
#: where nearly every model showed bias.
STAGE_A: list[tuple[str, str]] = [
    ("CA_GA_DS", "F2_alt_self"),    # headline, mirrored pair
    ("CA_GS_DA", "F2_alt_self"),
    ("SA_GA",    "F2_alt_self"),    # verified-manipulation contrast
    ("SA_DS",    "F2_alt_self"),
    ("CA_GA_DS", "F3_alt_neutral"), # headroom insurance
    ("CA_GS_DA", "F3_alt_neutral"),
]

#: The five models the headline runs on.
CORE_MODELS = ["M_base", "SA_GA", "SA_DS", "CA_GA_DS", "CA_GS_DA"]

#: E2 states an authority preference over the framing's two poles, and only
#: F2_alt_self has two poles that are both statable as a preference.
E2_FRAMINGS = ["F2_alt_self"]
E2_CONDITIONS = ["GA", "GS"]

#: E2.1 — the authority control set, re-measured under the v2 system-message
#: template. The v1 numbers (developer -0.819, EU -0.737, postal -0.697, fake lab
#: -0.359, salience +4.2pp, single-authority +0.645) were all taken with the OLD
#: template, where the grader got "assigns higher scores to" and the developer got
#: "prefers". That asymmetry is exactly what the v2 template removes, and
#: 04_prompted_arm.py's own comment flagged it as confounding the single-authority
#: contrast. So this is not a re-run of settled findings: the substitutions restate
#: recognisability on a clean template, and GRADER_ONLY / DEVELOPER_ONLY give the
#: first unconfounded version of the sign flip between conflict (-0.819) and alone
#: (+0.645) — the sharpest open result in the project.
#:
#: M_base only, F2_alt_self only. These do not multiply across models or framings.
E21_CONDITIONS = [
    "GA_EU", "GS_EU",                  # real, recognisable, plausibly governs
    "GA_POSTAL", "GS_POSTAL",          # real, recognisable, no plausible authority
    "GA_FAKELAB", "GS_FAKELAB",        # invented, same grammar and slot
    "NEUTRAL_SALIENCE",                # names the dimension, attributes it to nobody
    "GRADER_ONLY", "DEVELOPER_ONLY",   # the dissociation, unconfounded
]
#: Completes the (authority x direction) 2x2, and is what killed the reactance
#: hypothesis in v1. Opt in with --with-selfish.
E21_SELFISH = ["GRADER_ONLY_SELFISH", "DEVELOPER_ONLY_SELFISH"]


# --------------------------------------------------------------------------- #
# tasks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Task:
    experiment: str          # E1 | E2 | E3
    model: str
    framing: str = ""        # E1/E2 only
    condition: str = ""      # E2 condition, or E3 checkpoint label
    n: int = 2

    @property
    def id(self) -> str:
        parts = [self.experiment, self.model, self.framing, self.condition]
        return "__".join(p for p in parts if p)

    def shard(self) -> Path:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        return shards_dir() / f"{self.id}__{vm_id()}__{stamp}.parquet"


def enumerate_tasks(models: Sequence[str] = tuple(CORE_MODELS),
                    framings: Sequence[str] = tuple(FRAMINGS),
                    experiments: Sequence[str] = ("E3", "E1", "E2", "E2.1"),
                    stage_b: bool = False,
                    with_selfish: bool = False) -> list[Task]:
    """The task list, ordered so each gate precedes what it gates.

    ORDERING IS THE DESIGN. On one GPU everything is serial, so the order decides
    what you know at each hour:

      1. E3 ceiling on M_base -- HARD GATE. If the base model cannot answer the
         recall format when the answer is stated in its own system message, no
         recall number in the project means anything and nothing else is worth
         running.
      2. E1 base -- the baseline every adapter number is read against, and the
         corrected replication of the source paper (v1's measurement carried an
         ignore-the-bet instruction the paper treats as a separate intervention).
      3. E2 then E2.1 on the base model -- the in-context authority results, on
         the clean symmetric template.
      4. E2 on the adapters -- does SDF change how the model resolves a conflict
         it is SHOWN in context? This needs no self-reportable belief to exist,
         so it is live even though E3 is null.
      5. E1 stage A -- 6 grids: the mirrored CA pair on F2, the SA pair on F2
         (the contrast whose manipulation check passed), and the CA pair on F3
         (because F2 is the source paper's weakest framing).
      6. E3 on the adapters.

    STAGE B (`stage_b=True`) is the remaining E1 grids, and is deliberately NOT
    enumerated by default. If stage A's paired interval contains zero, those grids
    are generalisation checks on a null: 5 GPU-hours that cannot produce a finding.
    Because stage B is conditioned on stage A, stage A is the CONFIRMATORY result
    and stage B is EXPLORATORY -- state that in the write-up rather than letting
    the gate go unstated.
    """
    tasks: list[Task] = []
    base_in = "M_base" in models

    if "E3" in experiments and base_in:
        tasks.append(Task("E3", "M_base", condition="ceiling"))
        tasks.append(Task("E3", "M_base", condition="final"))
    if "E1" in experiments and base_in:
        tasks += [Task("E1", "M_base", framing=f) for f in framings]
    if "E2" in experiments and base_in:
        tasks += [Task("E2", "M_base", framing=f, condition=c)
                  for f in E2_FRAMINGS if f in framings for c in E2_CONDITIONS]
    if "E2.1" in experiments and base_in:
        conds = E21_CONDITIONS + (E21_SELFISH if with_selfish else [])
        tasks += [Task("E2.1", "M_base", framing=HEADLINE_FRAMING, condition=c)
                  for c in conds]
    if "E2" in experiments:
        tasks += [Task("E2", m, framing=f, condition=c)
                  for m in models if m in HEADLINE_MODELS
                  for f in E2_FRAMINGS if f in framings for c in E2_CONDITIONS]
    if "E1" in experiments:
        tasks += [Task("E1", m, framing=f) for m, f in STAGE_A
                  if m in models and f in framings]
    if "E3" in experiments:
        tasks += [Task("E3", m, condition="final") for m in models if m != "M_base"]
    if "E1" in experiments and stage_b:
        seen = {(t.model, t.framing) for t in tasks if t.experiment == "E1"}
        tasks += [Task("E1", m, framing=f) for m in models for f in framings
                  if (m, f) not in seen]
    return tasks


# --------------------------------------------------------------------------- #
# claiming
# --------------------------------------------------------------------------- #
def claim(task: Task) -> bool:
    """Try to claim `task`. True if this process won it, False if already taken.

    O_CREAT|O_EXCL is atomic on POSIX and on NFSv3+ with proper locking; the
    kernel picks exactly one winner. The claim file records who took it and when,
    which is what you read when a run is half-finished and you want to know why.
    """
    claims_dir().mkdir(parents=True, exist_ok=True)
    path = claims_dir() / f"{task.id}.claim"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"vm": vm_id(), "pid": os.getpid(),
                   "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   **asdict(task)}, f)
    return True


def release(task: Task) -> None:
    """Give a claimed-but-failed task back, so the other VM can retry it."""
    (claims_dir() / f"{task.id}.claim").unlink(missing_ok=True)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def record(task: Task, shard: Path, **extra) -> None:
    """Append one line to THIS VM's manifest. Never shared, so never contended."""
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    row = {"task": task.id, "shard": str(shard), "vm": vm_id(),
           "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           **asdict(task), **extra}
    with manifest_path().open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def pending(tasks: Sequence[Task]) -> Iterator[Task]:
    """Yield tasks this process successfully claims, in order."""
    for t in tasks:
        if claim(t):
            yield t
