# Machine-readable results

Small enough to version; the large artefacts (parquet rollout shards, adapters,
corpora) stay on the detachable storage — see `results/CONTENTS.md`.

| file | what |
|---|---|
| `valence_GD_full.json` | the properly-powered valence judge: 600 documents, 241 pairs, 182 clusters. Authority effect −0.104 [−0.199, −0.010], **TOST-equivalent at ±0.3**. Verdict `VOID_BLIND_LEAK` because the blind probe identified the authority 96% of the time on a balanced sample — probably intrinsic, so a stated limitation rather than a passed gate |
| `valence_GA_DS.json` | the earlier single-universe pilot (48 scores) |
| `summary_stats.json` | every number behind the Gate 1 k=30 figures |
| `fig_milestone.json` | the numbers behind the Gates 1–2 milestone figure |
| `bench_inference_*.json` | the throughput and trace-length benchmarks that set `max_tokens=32768` |
