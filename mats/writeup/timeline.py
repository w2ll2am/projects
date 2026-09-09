# -*- coding: utf-8 -*-
"""Reconstruct human time and machine time from the run's own timestamps.

Report section 6 quotes GPU-hours, which are measured per task. It also quotes
development hours, which are not measured anywhere -- so they are reconstructed
here from the artefacts that carry a clock: git commits, file mtimes, the
generation manifest, and the corpus/training logs.

Method. Every commit and every file write is one *event*. Events closer together
than an idle threshold belong to the same working session; a session's cost is
the span from its first to its last event. The threshold is the only free
parameter, so both 45 and 60 minutes are reported and the midpoint is quoted.
This measures INSTRUMENTED work only: reading, thinking away from the keyboard,
and anything done without touching a file are invisible to it, and because a
timestamp marks the END of a piece of work, each session's lead-in is missing
too. The number is therefore a floor.

Machine intervals come from the logs, are merged so concurrent jobs count once,
and are intersected with the sessions to get the attended fraction.

Log timestamps are UTC; git and mtimes are local (BST, UTC+1) -- see LOCAL.

    python timeline.py
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path.home() / "mats-archive"          # logs that did not come off the box
LOCAL = dt.timedelta(hours=1)                   # BST offset applied to UTC logs
GAPS = (45, 60)                                 # idle thresholds, minutes
TS = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


# ------------------------------------------------------------------- events
def human_events() -> list[dt.datetime]:
    out = []
    git = subprocess.run(
        ["git", "log", "--format=%ad", "--date=format:%Y-%m-%dT%H:%M:%S"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    out += [dt.datetime.fromisoformat(x) for x in git]
    for p in ROOT.rglob("*"):
        # shards are a bulk copy off the box, not work; caches are noise
        if not p.is_file() or "/.git/" in str(p) or "__pycache__" in str(p):
            continue
        if "/results_v2/shards/" in str(p):
            continue
        t = dt.datetime.fromtimestamp(p.stat().st_mtime).replace(microsecond=0)
        if t > dt.datetime(2026, 8, 30):
            out.append(t)
    return sorted(set(out))


def sessions(events, gap_min: int) -> list[tuple]:
    out, cur = [], [events[0]]
    for a, b in zip(events, events[1:]):
        if (b - a).total_seconds() / 60 > gap_min:
            out.append((cur[0], cur[-1]))
            cur = [b]
        else:
            cur.append(b)
    out.append((cur[0], cur[-1]))
    return out


# ----------------------------------------------------------------- machines
def _log_span(path: str):
    lines = Path(path).read_text(errors="ignore").strip().split("\n")
    a, b = TS.match(lines[0]), TS.match(lines[-1])
    if not (a and b) or b.group(1) <= a.group(1):
        return None                                   # aborted: no closing line
    return (dt.datetime.fromisoformat(a.group(1)) + LOCAL,
            dt.datetime.fromisoformat(b.group(1)) + LOCAL)


def machine_intervals() -> dict:
    out = {"corpus generation (API)": [], "adapter training": [],
           "evaluation queue": []}
    for f in glob.glob(str(ARCHIVE / "logs" / "01_gen_sdf_corpus*.log")):
        if (s := _log_span(f)):
            out["corpus generation (API)"].append(s)
    for f in glob.glob(str(ARCHIVE / "logs" / "06_train_sdf*.log")):
        # runs under ten minutes are launch failures, not training
        if (s := _log_span(f)) and (s[1] - s[0]).total_seconds() > 600:
            out["adapter training"].append(s)
    for line in open(ROOT / "results_v2" / "manifest_vmB.jsonl"):
        r = json.loads(line)
        end = dt.datetime.fromisoformat(r["finished_at"].replace("Z", "")) + LOCAL
        out["evaluation queue"].append((end - dt.timedelta(minutes=r["minutes"]), end))
    return out


def merge(iv):
    out = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def hours(iv):
    return sum((b - a).total_seconds() / 3600 for a, b in iv)


def overlap(A, B):
    t = 0.0
    for a1, a2 in A:
        for b1, b2 in B:
            lo, hi = max(a1, b1), min(a2, b2)
            if hi > lo:
                t += (hi - lo).total_seconds() / 3600
    return t


def main() -> None:
    ev = human_events()
    S = {g: sessions(ev, g) for g in GAPS}

    print("== development sessions, by day ==")
    days = sorted({a.date() for a in (s[0] for s in S[45])} |
                  {a.date() for a in (s[0] for s in S[60])})
    print(f"  {'day':12s}" + "".join(f"{g:>10d}m gap" for g in GAPS))
    for d in days:
        row = [sum((b - a).total_seconds() / 3600
                   for a, b in S[g] if a.date() == d) for g in GAPS]
        print(f"  {d}" + "".join(f"{v:11.2f} h" for v in row))
    tot = [hours(S[g]) for g in GAPS]
    print(f"  {'TOTAL':12s}" + "".join(f"{v:11.2f} h" for v in tot)
          + f"   sessions {len(S[45])}/{len(S[60])}")
    print(f"  midpoint {sum(tot)/2:.2f} h  <- the figure quoted in the report")
    print(f"  calendar span {ev[0]:%Y-%m-%d %H:%M} -> {ev[-1]:%Y-%m-%d %H:%M} "
          f"= {(ev[-1]-ev[0]).total_seconds()/3600:.1f} h")

    print("\n== machine time, and how much of it was attended ==")
    m = machine_intervals()
    for k, v in m.items():
        mv = merge(v)
        o = [overlap(mv, S[g]) for g in GAPS]
        print(f"  {k:26s} {hours(mv):6.2f} h wall   attended "
              f"{o[0]:5.2f}/{o[1]:5.2f} h  ({100*o[1]/hours(mv):3.0f}%)")
    allm = merge([iv for v in m.values() for iv in v])
    oa = [overlap(allm, S[g]) for g in GAPS]
    print(f"  {'ALL (merged)':26s} {hours(allm):6.2f} h wall   attended "
          f"{oa[0]:5.2f}/{oa[1]:5.2f} h")
    print(f"  unattended: {hours(allm) - sum(oa)/2:.2f} h")
    print(f"\n  serial cost would have been {sum(tot)/2:.1f} + {hours(allm):.1f} "
          f"= {sum(tot)/2 + hours(allm):.1f} h; the overlap saves "
          f"{sum(oa)/2:.1f} h of it")


if __name__ == "__main__":
    main()
