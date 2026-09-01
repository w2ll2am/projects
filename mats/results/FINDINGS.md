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

## 2026-09-01 — Length calibration: IN FLIGHT

`--configs length_calib --prompts 16 --n 8 --max-tokens 16384`, launched
21:45:12. Purpose: observe the true thinking-trace length distribution so
`max_tokens` is set from data. Results to be appended here.
