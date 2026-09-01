# Measured findings

Empirical results and the decisions taken from them. Append-only: add new
entries at the top, never silently edit an old measurement. Raw artefacts are
the timestamped `bench_inference_*.{json,csv}` files beside this one.

Anything here was MEASURED on the target hardware. Anything not here is an
assumption, however plausible.

---

## 2026-09-01 — Environment, verified on the 1xH200 box

| package | version | note |
|---|---|---|
| vllm | 0.28.1rc1.dev248+g55178f2d0 | nightly index; PyPI stable cannot load `qwen3_5` |
| torch | 2.13.0+cu132 | resolved BY vllm. Plan pinned 2.6.* — would have failed |
| transformers | 5.16.1 | |
| flash_attn | absent | nothing required it; plan section 0.6's compile step is unnecessary |

Host: Ubuntu 24.04.4, 1x NVIDIA H200 143771 MiB, driver 580.173.02, 16 vCPU,
196 GB RAM. EXP_ROOT=/mnt/filesystem-m9/gcvl on a 500 GB shared virtiofs mount.

Model `Qwen/Qwen3.5-4B` is `Qwen3_5ForConditionalGeneration`: an
image-text-to-text VLM, text stack nested under `config.text_config`, hybrid
attention — 24 Gated-DeltaNet linear-attention layers + 8 full-attention
(3:1, x8), 32 layers, hidden 2560, vocab 248320, native context 262144, MTP
head. Dense at 4B. Served text-only; vLLM logs
`Disabled mm_prefix attention mode` confirming the vision tower is off.

## 2026-09-01 — Throughput benchmark (`bench_inference_20260901T213150Z`)

16 prompts x n=4 = 64 rollouts per config, thinking mode, max_tokens=2048.

| config | out tok/s | gen s | load s | trunc % | finish reasons |
|---|---|---|---|---|---|
| mtp_spec | 8642 | 9.3 | 151.9 | 56.2 | 36 length / 28 stop |
| baseline | 8073 | 12.8 | 118.4 | 75.0 | 48 length / 16 stop |
| no_prefix_cache | 7877 | 12.4 | 100.4 | 68.8 | 44 length / 20 stop |

**Decisions taken:**

1. **The plan's ">=2000 output tok/s" target is met ~4x over.** Section 2.5's
   "15-20 min per model state" and the section 10 budget are far too
   pessimistic and should be re-derived from ~8000 tok/s.
2. **Prefix caching measured 1.02x, NOT the "roughly 2x for free" claimed.**
   Expected: only 8 of 32 layers keep a KV cache. Keep it (free), do not
   spend time tuning it.
3. **MTP speculative decoding measured 1.07x** — inside the sweep's own
   ~10% noise band, so NOT established. Would need one process per config to
   confirm. Not worth the GPU time given the 4x surplus.
4. **BLOCKING: max_tokens=2048 truncated 56-75% of rollouts and the MEDIAN
   output length was AT the cap (2048).** The true length distribution is
   therefore unmeasured. Gate 1 would have failed for a mechanical reason:
   truncated traces never emit `</think>`, so they parse to nothing and the
   parse rate would trip section 2.6's 90% abort. Calibration run launched to
   fix this before any threshold freezing.

Caveat on all three throughput numbers: single process, shared CUDA context,
best-effort teardown between configs. Differences under ~10% are noise. Trace
length is heavy-tailed — the three configs truncated 36/44/48 of 64 under
IDENTICAL sampling, ~3 s.d. of spread.

## 2026-09-01 — Length calibration (`bench_inference_20260901T214530Z`)

128 rollouts (16 prompts x n=8), thinking mode, max_tokens=16384.

| statistic | value |
|---|---|
| mean output length | 5073 tokens |
| median | 5312 |
| p90 | 11720 |
| max | 16384 (the cap) |
| finish reasons | 122 stop / 6 length |
| truncated | 4.7% (6/128) |
| throughput | 7745 output tok/s (649,351 tok in 83.8 s) |

**The plan's `max_tokens=2048` was low by a factor of ~2.6 at the MEDIAN.**
Half of all traces exceed 5312 tokens; a tenth exceed 11720. This fully
explains the 56-75% truncation in the throughput run.

**Adopted: `max_tokens=16384`, `max_model_len=17408`.**

**A coincidence worth understanding, because it hides two errors.** Plan
section 2.5 estimates "~1200 output tokens/rollout and 2000 tok/s, roughly
15-20 minutes per model state". Measured: 5073 tok/rollout (4.2x higher) at
7745 tok/s (3.9x higher). The two errors very nearly cancel, so the *time*
estimate survives — 1600 rollouts x 5073 tok / 7745 tok/s = ~17.5 min
generation, ~20 min with engine load. Do not read that agreement as
validation: both inputs were wrong, and they will stop cancelling the moment
either the hardware or the trace length changes.

**Caveat on the 4.7%.** With 6 truncations in 128 samples the 95% binomial
interval is roughly [1.7%, 9.9%], so this is NOT reliably below section 3's
5% threshold. Treat 16384 as adequate-but-marginal. The residual truncation
is a known, quantified loss: those rollouts emit no `</think>`, parse to None,
and cap the achievable parse rate near 95%. Still clear of section 2.6's 90%
abort, but if Gate 1 returns null the FIRST thing to try is a larger cap.

**Reaches beyond Gate 1:** section 7.1 also generates DAPO rollouts at
`max_tokens=2048`. The same correction applies there, and to that run's
time budget.
