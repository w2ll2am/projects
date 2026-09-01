"""GPU inference throughput benchmark for the Qwen3.5-4B hybrid VLM under vLLM.

Instrument for the plan's §1 "Sanity benchmark" claim: *">=2000 output tok/s on
one H200 with the 4B in thinking mode; under 800 means something is
misconfigured."* This script exists to VERIFY that number, not to reproduce it.

WHAT IS MEASURED
    * Output-token throughput: sum(Rollout.n_output_tokens) / wall seconds of the
      timed `generate` call. This is the headline.
    * Prompt-token throughput and total-token throughput, reported SEPARATELY.
      Conflating them is the usual way a benchmark inflates itself; a long-prompt
      short-output workload can show a huge "tok/s" that says nothing about
      generation speed.
    * Truncation rate (finish_reason == "length"). If most rollouts hit
      max_tokens, the throughput figure describes a truncated distribution and is
      not comparable to a real sweep. Also answers directly whether the plan's
      max_tokens=2048 is adequate for thinking mode.
    * Output-length distribution (mean / median / p90) and rollout latency stats.
    * Per-config engine build + model load time, reported OUTSIDE the timed region.

WHAT IS NOT MEASURED
    * Time to first token. The frozen `src/serve.py` contract returns `Rollout`
      objects with no per-request timing metadata, and the offline `LLM` class
      batches everything, so TTFT is not obtainable without reaching around the
      contract. Reported as null. Latency-per-rollout is reported instead.
    * Prefill/decode split, GPU utilisation, memory headroom. Use nvidia-smi
      alongside if you want those.
    * Quality. Nothing here checks that the completions are any good; a config
      that is fast and wrong will look great.

KNOWN CAVEATS (repeated in the printed report; read them before quoting a number)
    1. SINGLE PROCESS. Every config is built and torn down inside one Python
       process. `del engine; gc.collect(); torch.cuda.empty_cache()` is best
       effort: vLLM/NCCL/CUDA-graph allocations do not always return fully, so
       later configs may run against a more fragmented allocator than earlier
       ones. Config ORDER can therefore bias results by a few percent, and a
       late config may OOM where it would have fit fresh. Treat cross-config
       differences under ~10% as noise. For a publishable comparison, re-run
       with `--configs <one>` once per config, one process each.
    2. THINKING-MODE VARIANCE. Trace length is stochastic and heavy-tailed, so
       total generated tokens differ between configs even at identical settings.
       Throughput is tokens/second so this is largely self-normalising, but a
       config that happens to draw short traces spends proportionally more time
       in prefill and can look slower. Small `--prompts`/`--n` amplifies this.
    3. WARMUP. One warmup batch is discarded per config to absorb CUDA graph
       capture, kernel autotuning and torch.compile. One batch may not fully
       warm a speculative-decoding config.
    4. HYBRID ARCHITECTURE. Qwen3.5-4B is 24 Gated-DeltaNet linear-attention
       layers + 8 full-attention layers. `enable_prefix_caching` and
       `kv_cache_dtype="fp8"` are defined against a standard KV cache and may be
       silently ineffective, or rejected, on the linear-attention state. That is
       precisely why they are swept rather than assumed.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import paths  # noqa: E402

PLAN_CLAIM_TOK_S = 2000.0
PLAN_FLOOR_TOK_S = 800.0
TRUNCATION_ALERT = 0.05

# --------------------------------------------------------------------------- #
# Sweep matrix. Each entry is a set of build_engine(**overrides) kwargs.
# `fallbacks` are retried in order if the primary kwargs raise a TypeError-style
# "unexpected/duplicate keyword" error, so a signature mismatch in serve.py does
# not silently drop a config from the sweep.
# Edit freely: the only contract is that `overrides` is valid for build_engine.
# --------------------------------------------------------------------------- #

MTP_SPEC = {"method": "qwen3_next_mtp", "num_speculative_tokens": 2}


@dataclass(frozen=True)
class BenchConfig:
    """One engine configuration to time."""

    name: str
    overrides: dict[str, Any]
    note: str = ""
    fallbacks: tuple[dict[str, Any], ...] = ()


CONFIG_MATRIX: dict[str, BenchConfig] = {
    c.name: c
    for c in [
        BenchConfig(
            "baseline",
            {"enable_prefix_caching": True, "kv_cache_dtype": "auto", "max_num_seqs": 256},
            "plan defaults minus the max_num_seqs guess; the reference point",
        ),
        BenchConfig(
            "no_prefix_cache",
            {"enable_prefix_caching": False, "kv_cache_dtype": "auto", "max_num_seqs": 256},
            "tests the plan's 'prefix caching is roughly a 2x win for free' claim",
        ),
        BenchConfig(
            "kv_fp8",
            {"enable_prefix_caching": True, "kv_cache_dtype": "fp8", "max_num_seqs": 256},
            "fp8 KV cache; may error or be inert on Gated-DeltaNet state",
        ),
        BenchConfig(
            "mtp_spec",
            {
                "enable_prefix_caching": True,
                "kv_cache_dtype": "auto",
                "max_num_seqs": 256,
                "speculative_config": MTP_SPEC,
            },
            "MTP speculative decoding (model card supported; plan never mentions it)",
            fallbacks=({"enable_prefix_caching": True, "max_num_seqs": 256, "speculative": True},),
        ),
        BenchConfig(
            "seqs_64",
            {"enable_prefix_caching": True, "kv_cache_dtype": "auto", "max_num_seqs": 64},
            "max_num_seqs sweep",
        ),
        BenchConfig(
            "seqs_256",
            {"enable_prefix_caching": True, "kv_cache_dtype": "auto", "max_num_seqs": 256},
            "max_num_seqs sweep (== baseline; duplicate is deliberate, it measures run-to-run noise)",
        ),
        BenchConfig(
            "seqs_512",
            {"enable_prefix_caching": True, "kv_cache_dtype": "auto", "max_num_seqs": 512},
            "the plan's guess: 'raise until you OOM, then back off 20%'",
        ),
    ]
}

QUICK = ("baseline", "no_prefix_cache", "mtp_spec")
FULL = ("baseline", "no_prefix_cache", "kv_fp8", "mtp_spec", "seqs_64", "seqs_256", "seqs_512")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

FALLBACK_PROMPTS: list[str] = [
    "How many spots are there on all living giraffes combined?",
    "How many piano tuners are working in Chicago today?",
    "How many litres of paint are applied to cars in a year worldwide?",
    "How many grains of sand are on an average beach one kilometre long?",
    "How many words does a typical adult speak in a lifetime?",
    "How many bricks are in the Great Wall of China?",
    "How many text messages are sent globally in one hour?",
    "How many hairs are on the heads of everyone in New York City?",
    "How many kilometres does a commercial airline pilot fly in a career?",
    "How many cups of coffee are drunk in Europe each morning?",
    "How many bicycles are currently in use in the Netherlands?",
    "How many individual leaves are on all the trees in Central Park?",
    "How many footsteps does a postal worker take in a year?",
    "How many raindrops fall on London during an average rainy day?",
    "How many pages of paper are printed by offices worldwide each week?",
    "How many seconds of video are uploaded to the internet every minute?",
    "How many nails are used to build all the houses in a mid-size city?",
    "How many heartbeats does a blue whale have in its lifetime?",
    "How many pencils are manufactured globally in a year?",
    "How many kilograms of rice are cooked in India on a single day?",
]

FALLBACK_FRAME = (
    "I have a bet with a friend about this. {question} Please give your most "
    "accurate point estimate and ignore the bet. Answer with ANSWER: <number>."
)


def load_prompts(n_prompts: int) -> tuple[list[str], str]:
    """Real workload prompts if src/prompts.py + data are ready, else built-ins.

    Returns (prompts, source_description). Until 02_freeze_thresholds.py runs the
    items carry null thresholds, which makes the real grid unbuildable; that is
    the expected state and falls back silently to Fermi-style built-ins.
    """
    try:
        from src.prompts import (  # type: ignore[import-not-found]
            BAD_CAUSE,
            GOOD_CAUSE,
            build_grid,
            load_items,
            load_paraphrases,
        )
    except Exception as exc:  # noqa: BLE001 - module may not exist yet
        return _fallback_prompts(n_prompts), f"built-in Fermi prompts (src.prompts unavailable: {exc})"

    try:
        items = load_items()
        paraphrases = load_paraphrases()
    except Exception as exc:  # noqa: BLE001 - data files may not exist yet
        return _fallback_prompts(n_prompts), f"built-in Fermi prompts (data unreadable: {exc})"

    usable = [it for it in items if it.get("threshold") is not None]
    if not usable or not paraphrases:
        return (
            _fallback_prompts(n_prompts),
            "built-in Fermi prompts (thresholds still null - 02_freeze_thresholds.py has not run)",
        )

    try:
        grid = build_grid(usable, paraphrases, good_cause=GOOD_CAUSE, bad_cause=BAD_CAUSE)
    except Exception as exc:  # noqa: BLE001
        return _fallback_prompts(n_prompts), f"built-in Fermi prompts (build_grid failed: {exc})"

    texts = [row["text"] for row in grid]
    if not texts:
        return _fallback_prompts(n_prompts), "built-in Fermi prompts (empty grid)"
    return _cycle_to(texts, n_prompts), f"src.prompts.build_grid ({len(texts)} real grid cells)"


def _fallback_prompts(n_prompts: int) -> list[str]:
    return _cycle_to([FALLBACK_FRAME.format(question=q) for q in FALLBACK_PROMPTS], n_prompts)


def _cycle_to(pool: Sequence[str], n: int) -> list[str]:
    """Take n prompts, repeating the pool if needed.

    Repetition is noted in the report: with prefix caching on, duplicate prompts
    are unrealistically cache-friendly, so keep --prompts <= pool size when the
    prefix-caching comparison is the point.
    """
    if n <= len(pool):
        return list(pool[:n])
    out: list[str] = []
    while len(out) < n:
        out.extend(pool)
    return out[:n]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass
class ConfigResult:
    """One config's outcome: either metrics, or a classified failure."""

    name: str
    overrides: dict[str, Any]
    note: str = ""
    status: str = "ok"  # ok | oom | unsupported | error
    error: str | None = None
    used_fallback: bool = False
    load_s: float | None = None
    gen_s: float | None = None
    n_prompts: int = 0
    n_rollouts: int = 0
    output_tokens: int = 0
    prompt_tokens: int | None = None
    output_tok_s: float | None = None
    prompt_tok_s: float | None = None
    total_tok_s: float | None = None
    rollouts_per_s: float | None = None
    s_per_rollout: float | None = None
    ttft_s: float | None = None  # not obtainable via the frozen contract
    truncated_frac: float | None = None
    len_mean: float | None = None
    len_median: float | None = None
    len_p90: float | None = None
    len_max: int | None = None
    finish_reasons: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def passes(self) -> bool:
        return self.status == "ok" and (self.output_tok_s or 0.0) >= PLAN_CLAIM_TOK_S

    def row(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["overrides"] = json.dumps(self.overrides, sort_keys=True)
        d["finish_reasons"] = json.dumps(self.finish_reasons, sort_keys=True)
        d["warnings"] = "; ".join(self.warnings)
        d["pass"] = self.passes()
        return d


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile; q in [0, 1]. Avoids a numpy dependency."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def classify_error(exc: BaseException) -> str:
    """OOM vs unsupported-kwarg vs everything else."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if "outofmemory" in name.lower() or "out of memory" in msg or "no available memory" in msg:
        return "oom"
    if "kv cache" in msg and ("free" in msg or "insufficient" in msg or "not enough" in msg):
        return "oom"
    unsupported_markers = (
        "unexpected keyword",
        "got multiple values",
        "unrecognized",
        "not supported",
        "unsupported",
        "is not compatible",
        "invalid value for",
        "unknown argument",
    )
    if isinstance(exc, TypeError) or any(m in msg for m in unsupported_markers):
        return "unsupported"
    return "error"


def _is_signature_mismatch(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, TypeError) and (
        "unexpected keyword" in msg or "got multiple values" in msg
    )


# --------------------------------------------------------------------------- #
# Benchmark core
# --------------------------------------------------------------------------- #


def count_prompt_tokens(prompts: Sequence[str], model: str) -> int | None:
    """Exact prompt-token count via the HF tokenizer; None if unavailable.

    Reported separately from output tokens and never folded into the headline.
    """
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        return int(sum(len(tok(p).input_ids) for p in prompts))
    except Exception:  # noqa: BLE001 - purely informational
        return None


def free_gpu() -> None:
    """Best-effort GPU teardown between configs. See caveat 1 in the docstring."""
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


def run_config(
    cfg: BenchConfig,
    prompts: Sequence[str],
    *,
    n: int,
    max_tokens: int,
    warmup_prompts: int,
    prompt_tokens: int | None,
    serve: Any,
) -> ConfigResult:
    """Build, warm up, time one config. Never raises: failures are recorded."""
    res = ConfigResult(name=cfg.name, overrides=dict(cfg.overrides), note=cfg.note)
    engine = None
    attempts: list[dict[str, Any]] = [dict(cfg.overrides), *[dict(f) for f in cfg.fallbacks]]

    try:
        last_exc: BaseException | None = None
        for i, overrides in enumerate(attempts):
            t0 = time.perf_counter()
            try:
                engine = serve.build_engine(**overrides)
                res.load_s = time.perf_counter() - t0
                res.overrides = overrides
                res.used_fallback = i > 0
                if res.used_fallback:
                    res.warnings.append(
                        f"primary kwargs rejected ({last_exc}); ran fallback kwargs instead"
                    )
                break
            except BaseException as exc:  # noqa: BLE001 - any failure must not kill the sweep
                last_exc = exc
                engine = None
                free_gpu()
                if i + 1 < len(attempts) and _is_signature_mismatch(exc):
                    continue
                res.status = classify_error(exc)
                res.error = f"{type(exc).__name__}: {exc}"
                return res

        assert engine is not None

        # ---- warmup, discarded ------------------------------------------- #
        warm = list(prompts[: max(1, min(warmup_prompts, len(prompts)))])
        warm_sampling = serve.default_sampling(n=max(1, min(n, 2)), max_tokens=min(256, max_tokens))
        serve.generate(engine, warm, sampling=warm_sampling)

        # ---- timed region: generate only --------------------------------- #
        sampling = serve.default_sampling(n=n, max_tokens=max_tokens)
        t0 = time.perf_counter()
        out = serve.generate(engine, list(prompts), sampling=sampling)
        res.gen_s = time.perf_counter() - t0

        rollouts = [r for group in out for r in group]
        if not rollouts:
            res.status = "error"
            res.error = "generate returned no rollouts"
            return res

        lengths = [int(r.n_output_tokens) for r in rollouts]
        res.n_prompts = len(prompts)
        res.n_rollouts = len(rollouts)
        res.output_tokens = sum(lengths)
        res.prompt_tokens = prompt_tokens
        res.output_tok_s = res.output_tokens / res.gen_s if res.gen_s > 0 else None
        if prompt_tokens is not None and res.gen_s > 0:
            res.prompt_tok_s = prompt_tokens / res.gen_s
            res.total_tok_s = (prompt_tokens + res.output_tokens) / res.gen_s
        res.rollouts_per_s = res.n_rollouts / res.gen_s if res.gen_s > 0 else None
        res.s_per_rollout = res.gen_s / res.n_rollouts if res.n_rollouts else None

        n_trunc = sum(1 for r in rollouts if bool(getattr(r, "truncated", False)))
        res.truncated_frac = n_trunc / len(rollouts)
        res.len_mean = statistics.fmean(lengths)
        res.len_median = statistics.median(lengths)
        res.len_p90 = _percentile(lengths, 0.90)
        res.len_max = max(lengths)

        reasons: dict[str, int] = {}
        for r in rollouts:
            key = str(getattr(r, "finish_reason", "unknown"))
            reasons[key] = reasons.get(key, 0) + 1
        res.finish_reasons = reasons

        if res.truncated_frac > TRUNCATION_ALERT:
            res.warnings.append(
                f"TRUNCATION {res.truncated_frac:.1%} > {TRUNCATION_ALERT:.0%} at "
                f"max_tokens={max_tokens}: throughput describes a TRUNCATED distribution"
            )
        if res.len_p90 >= max_tokens * 0.98:
            res.warnings.append("p90 output length is at the max_tokens cap")
        return res

    except BaseException as exc:  # noqa: BLE001
        res.status = classify_error(exc)
        res.error = f"{type(exc).__name__}: {exc}"
        res.warnings.append(traceback.format_exc(limit=3).strip().splitlines()[-1])
        return res
    finally:
        if engine is not None:
            try:
                del engine
            except Exception:  # noqa: BLE001
                pass
        free_gpu()


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _fmt(v: float | None, spec: str = ".0f") -> str:
    return "-" if v is None else format(v, spec)


def print_report(
    results: Sequence[ConfigResult],
    *,
    meta: dict[str, Any],
    out_json: Path,
    out_csv: Path,
) -> None:
    ok = [r for r in results if r.status == "ok"]
    ok.sort(key=lambda r: r.output_tok_s or 0.0, reverse=True)
    failed = [r for r in results if r.status != "ok"]

    w = sys.stdout.write
    w("\n" + "=" * 96 + "\n")
    w("INFERENCE THROUGHPUT BENCHMARK\n")
    w("=" * 96 + "\n")
    for k in ("model", "timestamp", "prompt_source", "n_prompts", "n_per_prompt",
              "max_tokens", "prompt_tokens_total", "configs_requested"):
        w(f"  {k:22s} {meta.get(k)}\n")

    if meta.get("prompts_repeated"):
        w(
            "\n  NOTE: --prompts exceeded the available prompt pool, so prompts were REPEATED.\n"
            "        Duplicate prompts are unrealistically prefix-cache friendly and inflate the\n"
            "        prefix-caching comparison. Keep --prompts <= pool size for that comparison.\n"
        )

    w("\nRESULTS (sorted by OUTPUT tok/s - the headline metric)\n")
    header = (
        f"{'config':<18}{'out tok/s':>11}{'prompt tok/s':>14}{'total tok/s':>13}"
        f"{'gen s':>9}{'load s':>9}{'trunc%':>9}{'len p50':>9}{'len p90':>9}{'pass':>7}\n"
    )
    w(header)
    w("-" * (len(header) - 1) + "\n")
    for r in ok:
        w(
            f"{r.name:<18}{_fmt(r.output_tok_s, '.1f'):>11}{_fmt(r.prompt_tok_s, '.1f'):>14}"
            f"{_fmt(r.total_tok_s, '.1f'):>13}{_fmt(r.gen_s, '.1f'):>9}{_fmt(r.load_s, '.1f'):>9}"
            f"{_fmt((r.truncated_frac or 0) * 100, '.1f'):>9}{_fmt(r.len_median, '.0f'):>9}"
            f"{_fmt(r.len_p90, '.0f'):>9}{('PASS' if r.passes() else 'fail'):>7}\n"
        )
    if not ok:
        w("  (no config completed)\n")

    if failed:
        w("\nFAILED CONFIGS\n")
        for r in failed:
            label = {"oom": "OOM", "unsupported": "UNSUPPORTED KWARG/FEATURE"}.get(
                r.status, "ERROR"
            )
            w(f"  {r.name:<18} [{label}] {r.error}\n")

    notes = [(r.name, wmsg) for r in results for wmsg in r.warnings]
    if notes:
        w("\nWARNINGS\n")
        for name, msg in notes:
            w(f"  {name:<18} {msg}\n")

    # ---- targeted comparisons the plan makes claims about ----------------- #
    by_name = {r.name: r for r in ok}
    w("\nCLAIM CHECKS\n")
    base = by_name.get("baseline")
    nopc = by_name.get("no_prefix_cache")
    if base and nopc and nopc.output_tok_s:
        ratio = (base.output_tok_s or 0) / nopc.output_tok_s
        w(
            f"  prefix caching: {ratio:.2f}x (plan says 'roughly 2x for free'). "
            f"{'Claim not supported' if ratio < 1.5 else 'Consistent with the claim'} on this "
            "hybrid Gated-DeltaNet model.\n"
        )
    else:
        w("  prefix caching: not measurable (one side of the comparison did not complete)\n")
    fp8 = by_name.get("kv_fp8")
    if base and fp8 and base.output_tok_s:
        w(f"  kv_cache_dtype=fp8: {(fp8.output_tok_s or 0) / base.output_tok_s:.2f}x vs baseline\n")
    mtp = by_name.get("mtp_spec")
    if base and mtp and base.output_tok_s:
        w(f"  MTP speculative decoding: {(mtp.output_tok_s or 0) / base.output_tok_s:.2f}x vs baseline\n")
    seqs = [(r.overrides.get("max_num_seqs"), r.output_tok_s) for r in ok if r.name.startswith("seqs_")]
    if seqs:
        best = max(seqs, key=lambda t: t[1] or 0.0)
        w(f"  max_num_seqs sweep: best is {best[0]} at {best[1]:.1f} out tok/s\n")

    # ---- verdict ---------------------------------------------------------- #
    w("\n" + "=" * 96 + "\nVERDICT\n" + "=" * 96 + "\n")
    if ok:
        best = ok[0]
        w(
            f"  Plan claims >= {PLAN_CLAIM_TOK_S:.0f} output tok/s; best measured config achieved "
            f"{best.output_tok_s:.1f} tok/s (config: {best.name} = "
            f"{json.dumps(best.overrides, sort_keys=True)}).\n"
        )
        if best.output_tok_s >= PLAN_CLAIM_TOK_S:
            w("  -> The plan's >=2000 tok/s claim HOLDS on this hardware and workload.\n")
        elif best.output_tok_s >= PLAN_FLOOR_TOK_S:
            w(
                "  -> The plan's >=2000 tok/s claim DOES NOT HOLD. Measured throughput is above the\n"
                "     plan's 800 tok/s 'something is misconfigured' floor, so this looks like an\n"
                "     over-optimistic estimate in the plan rather than a broken setup. Re-budget\n"
                f"     section 2.5/10 timings by a factor of ~{PLAN_CLAIM_TOK_S / best.output_tok_s:.1f}x.\n"
            )
        else:
            w(
                "  -> The plan's claim DOES NOT HOLD and throughput is BELOW the plan's 800 tok/s\n"
                "     floor. Per the plan this indicates misconfiguration - but note the floor was\n"
                "     written assuming a standard attention model. Check GPU type, tensor parallel\n"
                "     size, and whether another process holds GPU memory before concluding.\n"
            )
        if (best.truncated_frac or 0) > TRUNCATION_ALERT:
            w(
                f"  !! Best config truncated {best.truncated_frac:.1%} of rollouts at "
                f"max_tokens={meta.get('max_tokens')}. The throughput above is measured on a\n"
                "     TRUNCATED distribution and OVERSTATES real-sweep throughput: untruncated\n"
                "     thinking traces are longer and decode-bound. Raise max_tokens and re-run.\n"
            )
        else:
            w(
                f"  Truncation at max_tokens={meta.get('max_tokens')} is "
                f"{(best.truncated_frac or 0):.1%} (<= {TRUNCATION_ALERT:.0%}): the cap looks adequate\n"
                "     for thinking mode on this prompt set.\n"
            )
    else:
        w("  No configuration completed successfully - no verdict on the plan's claim is possible.\n")

    w(
        "\nCAVEATS (do not quote a number without these)\n"
        "  * Single process: all configs share one CUDA context. Teardown between configs is best\n"
        "    effort, so residual memory fragmentation may bias LATER configs downward and can cause\n"
        "    an OOM that would not occur in a fresh process. Differences under ~10% are noise.\n"
        "    Re-run with --configs <name> (one process per config) before publishing a comparison.\n"
        "  * Thinking-mode output length is stochastic and heavy-tailed; short runs are noisy.\n"
        "  * Time-to-first-token is not reported: the frozen serve.py contract exposes no per-request\n"
        "    timing. Seconds-per-rollout is reported instead and is a batch figure, not a latency.\n"
        "  * Warmup batch is discarded, but one batch may not fully warm speculative decoding.\n"
    )
    w(f"\nWrote {out_json}\nWrote {out_csv}\n")


def save(results: Sequence[ConfigResult], meta: dict[str, Any], out_dir: Path, stamp: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"bench_inference_{stamp}.json"
    out_csv = out_dir / f"bench_inference_{stamp}.csv"
    rows = [r.row() for r in results]
    out_json.write_text(json.dumps({"meta": meta, "results": rows}, indent=2, default=str))
    if rows:
        keys = list(rows[0].keys())
        with out_csv.open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
    return out_json, out_csv


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure vLLM output-token throughput across engine configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help=f"short subset: {', '.join(QUICK)}")
    g.add_argument("--full", action="store_true", help=f"whole matrix: {', '.join(FULL)}")
    p.add_argument("--configs", type=str, default=None,
                   help="comma-separated config names (overrides --quick/--full); "
                        "'list' prints the matrix and exits")
    p.add_argument("--prompts", type=int, default=16, help="number of distinct prompts")
    p.add_argument("--n", type=int, default=4, help="rollouts per prompt (SamplingParams.n)")
    p.add_argument("--max-tokens", type=int, default=2048, help="generation cap (plan default)")
    p.add_argument("--warmup-prompts", type=int, default=4,
                   help="prompts in the discarded warmup batch")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: paths.sub('results'))")
    return p.parse_args(argv)


def select_configs(args: argparse.Namespace) -> list[BenchConfig]:
    if args.configs:
        names = [s.strip() for s in args.configs.split(",") if s.strip()]
    elif args.full:
        names = list(FULL)
    else:
        names = list(QUICK)  # --quick is the default
    unknown = [n for n in names if n not in CONFIG_MATRIX]
    if unknown:
        raise SystemExit(
            f"unknown config(s): {', '.join(unknown)}. Available: {', '.join(CONFIG_MATRIX)}"
        )
    return [CONFIG_MATRIX[n] for n in names]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.configs == "list":
        for name, cfg in CONFIG_MATRIX.items():
            print(f"{name:<18} {json.dumps(cfg.overrides, sort_keys=True)}\n{'':<18} # {cfg.note}")
        return 0

    configs = select_configs(args)
    prompts, source = load_prompts(args.prompts)

    from src import serve  # imported late: pulls in vLLM/torch

    prompt_tokens = count_prompt_tokens(prompts, getattr(serve, "BASE", "Qwen/Qwen3.5-4B"))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "model": getattr(serve, "BASE", "unknown"),
        "timestamp": stamp,
        "prompt_source": source,
        "n_prompts": len(prompts),
        "n_per_prompt": args.n,
        "max_tokens": args.max_tokens,
        "warmup_prompts": args.warmup_prompts,
        "prompt_tokens_total": prompt_tokens,
        "configs_requested": [c.name for c in configs],
        "prompts_repeated": args.prompts > len(set(prompts)),
    }

    print(f"prompt source: {source}")
    print(f"{len(prompts)} prompts x n={args.n} = {len(prompts) * args.n} rollouts per config, "
          f"max_tokens={args.max_tokens}")
    print(f"configs: {', '.join(c.name for c in configs)}\n")

    results: list[ConfigResult] = []
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg.name}: {json.dumps(cfg.overrides, sort_keys=True)}", flush=True)
        res = run_config(
            cfg,
            prompts,
            n=args.n,
            max_tokens=args.max_tokens,
            warmup_prompts=args.warmup_prompts,
            prompt_tokens=prompt_tokens,
            serve=serve,
        )
        results.append(res)
        if res.status == "ok":
            print(f"    -> {res.output_tok_s:.1f} output tok/s "
                  f"({res.output_tokens} tok in {res.gen_s:.1f}s, "
                  f"trunc {res.truncated_frac:.1%})", flush=True)
        else:
            print(f"    -> {res.status.upper()}: {res.error}", flush=True)

    out_dir = args.out or paths.sub("results")
    out_json, out_csv = save(results, meta, out_dir, stamp)
    print_report(results, meta=meta, out_json=out_json, out_csv=out_csv)
    return 0 if any(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
