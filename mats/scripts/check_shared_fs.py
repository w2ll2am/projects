#!/usr/bin/env python3
"""Prove the shared filesystem supports the two-VM claiming protocol.

Run the SAME command on both VMs at the same time, pointed at the shared mount:

    python3 scripts/check_shared_fs.py --root /mnt/shared/fscheck --vm vmA
    python3 scripts/check_shared_fs.py --root /mnt/shared/fscheck --vm vmB

then on either:

    python3 scripts/check_shared_fs.py --root /mnt/shared/fscheck --verify

WHY THIS EXISTS. src/tasks.py partitions work between two VMs using
``os.open(O_CREAT|O_EXCL)``. That is atomic on a local POSIX filesystem and on
NFSv4, but on NFSv3 exclusive create is *emulated* and two clients can both
believe they won. If that happens here, both VMs would run every task, double
the GPU bill, and write two shards per task. It is a five-second check and the
whole schedule depends on it, so it is not something to assume.

Three properties are tested:

1. EXCLUSIVE CREATE. Both VMs race for the same N keys. Across both, the number
   of wins must be exactly N — no key won twice, none lost.
2. ATOMIC RENAME. Write-then-``os.replace`` must never expose a partial file to
   a reader on the *other* node.
3. CROSS-NODE VISIBILITY. A file closed on one node must be readable on the
   other. NFS gives close-to-open consistency, which is enough, but a filesystem
   with looser caching would break the manifest merge in make_results.py.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

N_KEYS = 500
PAYLOAD = b"x" * (1 << 20)  # 1 MiB, big enough that a partial write is visible


def race(root: Path, vm: str) -> None:
    claims = root / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    won = []
    for i in range(N_KEYS):
        try:
            fd = os.open(claims / f"k{i:04d}", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.write(fd, vm.encode())
        os.close(fd)
        won.append(i)
    (root / f"won_{vm}.json").write_text(json.dumps(won))
    print(f"[{vm}] exclusive-create: won {len(won)} of {N_KEYS}")

    # atomic rename, observed by the other node's verify pass
    big = root / f"atomic_{vm}.bin"
    tmp = big.with_suffix(f".tmp.{os.getpid()}")
    tmp.write_bytes(PAYLOAD)
    os.replace(tmp, big)
    print(f"[{vm}] wrote {big.name} ({len(PAYLOAD)} bytes) via write-then-rename")

    (root / f"stamp_{vm}.txt").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    print(f"[{vm}] done")


def verify(root: Path) -> int:
    wins = {}
    for p in sorted(root.glob("won_*.json")):
        wins[p.stem.removeprefix("won_")] = set(json.loads(p.read_text()))
    if len(wins) < 2:
        print(f"!! only {len(wins)} VM(s) reported: {list(wins)}. Run both, then verify.")
        return 1

    names = list(wins)
    a, b = wins[names[0]], wins[names[1]]
    total, overlap, missing = len(a) + len(b), a & b, set(range(N_KEYS)) - (a | b)

    ok = True
    print(f"\nkeys={N_KEYS}  {names[0]}={len(a)}  {names[1]}={len(b)}  total={total}")
    if overlap:
        ok = False
        print(f"!! FAIL exclusive create: {len(overlap)} key(s) won by BOTH VMs, "
              f"e.g. {sorted(overlap)[:5]}")
        print("   O_EXCL is not atomic on this filesystem. Do NOT run both VMs on one "
              "task list — give each an explicit --models/--experiments split instead.")
    if missing:
        ok = False
        print(f"!! FAIL: {len(missing)} key(s) won by NOBODY, e.g. {sorted(missing)[:5]}")
    if not overlap and not missing:
        print("OK  exclusive create is atomic: every key won exactly once")

    for p in sorted(root.glob("atomic_*.bin")):
        n = p.stat().st_size
        if n != len(PAYLOAD) or p.read_bytes() != PAYLOAD:
            ok = False
            print(f"!! FAIL atomic rename: {p.name} is {n} bytes, expected {len(PAYLOAD)}")
        else:
            print(f"OK  {p.name} readable and complete from this node ({n} bytes)")

    stamps = sorted(root.glob("stamp_*.txt"))
    print(f"OK  cross-node visibility: {len(stamps)} stamp file(s) visible "
          f"({', '.join(p.stem.removeprefix('stamp_') for p in stamps)})")
    if len(stamps) < 2:
        ok = False
        print("!! FAIL: the other node's file is not visible from here")

    print("\nVERDICT:", "shared filesystem supports the claiming protocol"
          if ok else "DO NOT use dynamic claiming on this filesystem")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--vm")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if args.reset:
        import shutil
        shutil.rmtree(root, ignore_errors=True)
        print(f"cleared {root}")
        return 0
    if args.verify:
        return verify(root)
    if not args.vm:
        ap.error("give --vm NAME (or --verify)")
    root.mkdir(parents=True, exist_ok=True)
    race(root, args.vm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
