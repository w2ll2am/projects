# Distilled run logs

Every run log from the box, with the noise removed: per-request HTTP lines,
vLLM engine chatter, safetensors loading, and tqdm progress bars. Progress bars
in particular are carriage-return streams, so raw they collapse an entire run
onto one unreadable line — which is why the originals are 9.4 MB and these are
1.1 MB.

What survives is the substantive output: every configuration banner, every
verdict block, every warning, and every measurement. Nothing was summarised or
edited; lines were only removed.

**The raw logs stay on the detachable storage** at `$EXP_ROOT/logs/`, in case a
future question needs the per-request detail these drop.

## The ones worth reading

| log | what it holds |
|---|---|
| `gate1b.log` | Gate 1 at k=30 — the run that passed. Per-paraphrase and per-item tables |
| `gate1c.log` | the seed-1 replication |
| `gate2.log` | the original GA/GS conflict pair, Delta_GD = −0.819 |
| `gate2b.log` | the five discriminating conditions; salience ruled out, fake-lab halves the effect |
| `gate2_eu.log` | the EU control that withdrew the developer-following reading |
| `postal.log` | the real-but-implausible authority that closed the 2×2 |
| `selfish.log` | the single-authority 2×2 completed; reactance ruled out |
| `single_universe.log` | **the decisive control** — 95.8% recall with the partner removed |
| `retrain_steps.log` | the 567-step run and its dose curve |
| `gs_da_recall.log` | the mirror universe that identified the altruism prior |
| `e2e27b.log` | 27B recall path verified end to end |
| `sdfsmoke.log` | the first SDF training run, and the packing/attention warning it surfaced |
| `verlprep.log` | vLLM into the verl venv; the DAPO reward-registration check |
| `apiq.log` | the API work queue — valence judge, corpus generation |
| `watchdog.log`, `STATUS.txt`, `ALERTS.txt` | supervision: throughput samples, restarts, alerts |
