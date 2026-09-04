# WIL-12 v2 — shared filesystem contents

Frozen 2026-09-04 after the H200 box was released. Everything needed to
reproduce or audit the v2 run is under `/mnt/filesystem-u9/gcvl/`.

| path | what |
|---|---|
| `v2/SUMMARY.md`   | findings in plain language -- read first |
| `v2/FINDINGS.md`  | every number, with provenance and caveats |
| `v2/CONTENTS.md`  | data index |
| `v2/shards/`      | 22 parquet files, 28,608 rollouts with full traces |
| `v2/probes/`      | 8 probe runs, per-layer AUC + shuffled control |
| `v2/logs/`        | queue log, threshold freeze, probe sequencers |
| `v2/manifest_vmB.jsonl` | one line per completed task |
| `v2/repo_snapshot/` | the exact code that produced the above |
| `ckpt/`           | 4 LoRA adapters, 17 GB |

## Three traps, all load-bearing

1. **Adapters must point at `checkpoint-N`**, never the parent directory. The
   parent holds dose checkpoints and is not a loadable adapter; pointing vLLM at
   one kills the engine and cascades through every queued task.
2. **`AutoModelForCausalLM` silently breaks these adapters.** It resolves
   Qwen3.5 to `Qwen3_5ForCausalLM` (`model.layers.*`), but the LoRAs were
   trained against `Qwen3_5ForConditionalGeneration`
   (`model.language_model.layers.*`). Every key misses, PEFT attaches nothing
   and reports success, and the logits come back bit-identical. Always assert
   the logits moved after attaching. See `repo_snapshot/scripts/12_probe.py`.
3. **`src/metrics.py` filters on the keys `parsed` and `good_side`.** Emitting
   `estimate`/`good` instead makes every metric return `nan` without an error.
