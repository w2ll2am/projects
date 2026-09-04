#!/usr/bin/env python3
"""Check everything that can be checked without spending GPU time.

    EXP_ROOT=/mnt/shared VM_ID=vmA python3 scripts/vm_preflight.py

Exits non-zero on any FAIL. WARNs are judgement calls and do not block.

The first attempt lost time to problems every one of which was visible before the
GPU was touched: an adapter path that did not exist, a corpus keyed on the wrong
flag, a config value that did not survive into SFTConfig. This script is the
cheapest possible insurance against repeating that.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAIL, WARN, OK = [], [], []


def fail(msg: str) -> None:
    FAIL.append(msg); print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARN.append(msg); print(f"  WARN  {msg}")


def ok(msg: str) -> None:
    OK.append(msg); print(f"  ok    {msg}")


def section(name: str) -> None:
    print(f"\n{name}")


# --------------------------------------------------------------------------- #
def check_env() -> None:
    section("environment")
    root = os.environ.get("EXP_ROOT")
    if not root:
        fail("EXP_ROOT is not set")
        return
    p = Path(root)
    if not p.is_dir():
        fail(f"EXP_ROOT={root} does not exist")
        return
    ok(f"EXP_ROOT={root}")
    probe = p / ".preflight_write_probe"
    try:
        probe.write_text("x"); probe.unlink()
        ok("EXP_ROOT is writable")
    except OSError as exc:
        fail(f"EXP_ROOT is not writable: {exc}")

    if not os.environ.get("VM_ID"):
        warn("VM_ID is not set; falling back to hostname. Set it explicitly so shard "
             "names and manifests are unambiguous across VMs.")
    else:
        ok(f"VM_ID={os.environ['VM_ID']}")

    import shutil
    free_gb = shutil.disk_usage(p).free / 2**30
    (ok if free_gb > 50 else fail)(f"{free_gb:.0f} GiB free on EXP_ROOT")


def check_gpu() -> None:
    section("gpu")
    try:
        import torch
    except ImportError:
        fail("torch not importable")
        return
    if not torch.cuda.is_available():
        fail("torch.cuda.is_available() is False")
        return
    n = torch.cuda.device_count()
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_memory / 2**30
        ok(f"cuda:{i} {name}, {mem:.0f} GiB")
    if n > 1:
        warn(f"{n} GPUs visible; the runner uses one. Pin with CUDA_VISIBLE_DEVICES.")


def check_deps() -> None:
    section("dependencies")
    for mod in ("vllm", "transformers", "peft", "trl", "pandas", "pyarrow", "yaml"):
        try:
            m = __import__(mod)
            ok(f"{mod} {getattr(m, '__version__', '?')}")
        except ImportError as exc:
            fail(f"{mod} not importable ({exc})")


def check_data() -> None:
    section("grid data")
    from src import framings, prompts

    items = json.loads((ROOT / "data" / "fermi_items.json").read_text())
    (ok if len(items) == 18 else fail)(f"{len(items)} Fermi items (expected 18)")
    dropped = {"zills", "busstops"} & {i["id"] for i in items}
    (fail if dropped else ok)(
        f"pinned items still present: {sorted(dropped)}" if dropped
        else "zills/busstops dropped as planned")

    unfrozen = [i["id"] for i in items if i.get("threshold") is None]
    if unfrozen:
        fail(f"{len(unfrozen)} item(s) have threshold=null: {unfrozen[:5]}")

    # Provenance: if the thresholds are byte-identical to the v1 snapshot they were
    # frozen at max_tokens 2048 and have NOT been re-frozen for this run.
    snap = ROOT.parent / "mats_first_attempt" / "data_snapshot" / "fermi_items.json"
    if not snap.exists():
        # Fail closed. A missing snapshot means we CANNOT tell whether these
        # thresholds were re-frozen for this run, and "cannot verify" must never
        # read as "verified" on a blocking prerequisite.
        fail(f"cannot verify threshold provenance: {snap} is missing. Copy the v1 "
             "snapshot next to the repo, or re-freeze and confirm by hand.")
    else:
        old = {i["id"]: i["threshold"] for i in json.loads(snap.read_text())}
        same = [i["id"] for i in items if i["id"] in old and i["threshold"] == old[i["id"]]]
        if len(same) == len(items):
            fail("thresholds are IDENTICAL to the v1 snapshot, i.e. still frozen at "
                 "max_tokens 2048. Re-freeze first:\n"
                 "          python3 scripts/02_freeze_thresholds.py")
        else:
            ok(f"thresholds differ from the v1 snapshot on {len(items)-len(same)}/"
               f"{len(items)} items (re-frozen)")

    paras = prompts.load_paraphrases()
    (ok if len(paras) == 30 else fail)(f"{len(paras)} paraphrases (expected 30)")

    import re
    bad = [i for i, t in enumerate(paras)
           if re.search(r"\bignore\b|\bdisregard\b|set aside|not a factor", t, re.I)]
    (fail if bad else ok)(
        f"ignore-the-bet language in paraphrases {bad}" if bad
        else "no ignore-the-bet instruction in any paraphrase")

    for key in framings.FRAMINGS:
        try:
            rows = prompts.build_grid(items, paras, framing=key)
            leftover = [r for r in rows if re.search(r"\{[a-z_]+\}", r["text"])]
            if leftover:
                fail(f"{key}: {len(leftover)} prompts have unfilled slots")
            else:
                ok(f"{key}: {len(rows)} prompts render clean")
        except Exception as exc:                                  # noqa: BLE001
            fail(f"{key} failed to render: {exc!r}")


def check_conditions() -> None:
    section("system messages")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "arm", ROOT / "scripts" / "04_prompted_arm.py")
    arm = importlib.util.module_from_spec(spec); spec.loader.exec_module(arm)
    from src import tasks

    need = set(tasks.E2_CONDITIONS) | set(tasks.E21_CONDITIONS) | set(tasks.E21_SELFISH)
    missing = sorted(need - set(arm.ALL_CONDITIONS))
    (fail if missing else ok)(
        f"conditions missing from ALL_CONDITIONS: {missing}" if missing
        else f"all {len(need)} required conditions resolve")

    if "respondent" in arm.SELF_INTERESTED:
        fail("SELF_INTERESTED still says 'respondent' — must match the corpus wording "
             "'the person who is asking'")
    else:
        ok(f"self-interest pole: {arm.SELF_INTERESTED!r}")

    ga, gs = arm.ALL_CONDITIONS["GA"], arm.ALL_CONDITIONS["GS"]
    if ga.count("prefers responses that steer outcomes toward") != 2:
        fail("GA is not using the symmetric verb phrase in both sentences")
    else:
        ok("GA/GS use the symmetric verb phrase")
    if sorted(ga.split()) != sorted(gs.split()):
        warn("GA and GS are not word-for-word mirrors")
    else:
        ok("GA and GS are word-for-word mirrors")


def check_adapters() -> None:
    section("adapters")
    from src import tasks
    for key in tasks.CORE_MODELS:
        m = tasks.MODELS[key]
        if m.adapter is None:
            ok(f"{key}: base model, no adapter")
            continue
        p = tasks.exp_root() / "ckpt" / m.adapter
        if not p.is_dir():
            fail(f"{key}: {p} missing")
        elif not (p / "adapter_config.json").is_file() or \
                not (p / "adapter_model.safetensors").is_file():
            # The v1 check globbed children too, so a parent directory holding
            # checkpoint-N subdirs PASSED while being unloadable. vLLM needs both
            # files at the exact path, and a miss kills the engine.
            fail(f"{key}: {p} is not a loadable adapter (needs adapter_config.json "
                 "and adapter_model.safetensors at this exact path)")
        else:
            ok(f"{key}: {p}")


def check_claiming() -> None:
    section("shared filesystem")
    from src import tasks
    d = tasks.claims_dir()
    d.mkdir(parents=True, exist_ok=True)
    probe = d / ".preflight_excl"
    probe.unlink(missing_ok=True)
    try:
        fd = os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except OSError as exc:
        fail(f"O_CREAT|O_EXCL failed: {exc}")
        return
    try:
        os.open(probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        fail("O_EXCL did NOT reject a second create — claiming is unsafe on this "
             "filesystem. Give each VM an explicit --models/--experiments split.")
    except FileExistsError:
        ok("O_CREAT|O_EXCL rejects a duplicate create (single-node check)")
    finally:
        probe.unlink(missing_ok=True)
    warn("single-node only. For the real two-VM guarantee run "
         "scripts/check_shared_fs.py on both at once.")


def check_plan() -> None:
    section("task plan")
    from src import tasks
    todo = tasks.enumerate_tasks()
    ok(f"{len(todo)} tasks enumerated")
    first = todo[0].id if todo else "?"
    (ok if first == "E3__M_base__ceiling" else fail)(
        f"first task is {first} (must be the E3 ceiling gate)")
    stage_a = [t.id for t in todo if t.experiment == "E1" and t.model != "M_base"]
    ok(f"stage A: {len(stage_a)} grids")
    for t in stage_a:
        print(f"          {t}")


def main() -> int:
    print("preflight for the v2 run\n" + "=" * 60)
    for fn in (check_env, check_gpu, check_deps, check_data, check_conditions,
               check_adapters, check_claiming, check_plan):
        try:
            fn()
        except SystemExit as exc:
            fail(f"{fn.__name__}: {exc}")
        except Exception as exc:                                   # noqa: BLE001
            fail(f"{fn.__name__} raised: {exc!r}")
    print("\n" + "=" * 60)
    print(f"{len(OK)} ok, {len(WARN)} warn, {len(FAIL)} fail")
    if FAIL:
        print("\nBLOCKED:")
        for m in FAIL:
            print(f"  - {m}")
        return 1
    print("\nclear to run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
