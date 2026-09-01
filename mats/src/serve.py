"""vLLM offline inference for Qwen3.5-4B (see INTERFACES.md, plan §1).

The model is `Qwen3_5ForConditionalGeneration`: an image-text-to-text VLM whose
text stack is hybrid (24 Gated-DeltaNet linear-attention layers + 8 full-attention
layers, 3:1, 32 total) and nests under `text_config`. Our workload is text-only,
so we ask vLLM to skip the vision tower.

Nothing here can be exercised off-GPU: vLLM is not installed on a laptop. Every
engine kwarg that this build might reject is therefore applied through
`_build_with_fallback`, which degrades with a warning instead of crashing.

MEASURED — max_tokens: the plan's 2048 default was low by ~2.6x at the MEDIAN.
Calibration on 128 rollouts (results/FINDINGS.md, 2026-09-01) gave mean 5073,
median 5312, p90 11720 output tokens, and 56-75% truncation at 2048. Defaults
are now max_tokens=16384 / max_model_len=17408, which truncates 4.7%.

That 4.7% is marginal, not comfortable: 6/128 has a 95% interval of roughly
[1.7%, 9.9%], straddling the plan's 5% rule. Truncated rollouts emit no
`</think>`, so they parse to None and cap the achievable parse rate near 95%.
`Rollout.truncated` keeps this visible instead of silently biasing the
subsample. If a gate returns null, raise the cap before concluding anything.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

BASE = "Qwen/Qwen3.5-4B"

log = logging.getLogger(__name__)

_TOKENIZER: Any = None  # module-level cache; loading it per call is wasteful


@dataclass
class Rollout:
    """One completion. `text` keeps the thinking trace; `final` never does."""

    text: str
    final: str
    n_output_tokens: int
    finish_reason: str
    truncated: bool


# --------------------------------------------------------------------------
# prompt / trace handling  (no vLLM import needed)
# --------------------------------------------------------------------------

def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    return _TOKENIZER


def to_prompt(user_msg: str, system_msg: str | None = None, thinking: bool = True) -> str:
    """Render the chat template as a string prompt with the generation prefix.

    Qwen3.5 has thinking ON by default and does NOT honour the `/think`
    `/nothink` soft switch; the only documented control is the chat-template
    kwarg `enable_thinking`. Older tokenizers reject unknown template kwargs, so
    fall back to the default (thinking) rendering rather than failing.
    """
    msgs: list[dict[str, str]] = []
    if system_msg:
        msgs.append({"role": "system", "content": system_msg})
    msgs.append({"role": "user", "content": user_msg})

    tok = _tokenizer()
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
    except (TypeError, ValueError) as exc:
        warnings.warn(
            f"apply_chat_template rejected enable_thinking={thinking} ({exc}); "
            "falling back to the template default (thinking ON). Verify the "
            "rendered prompt before trusting any non-thinking condition."
        )
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def final_segment(text: str, thinking: bool = True) -> str:
    """The answer segment: everything after the last `</think>`.

    Subtle, and the reason `thinking` is a parameter: the Qwen3.5 chat template
    emits the OPENING `<think>` tag as part of the generation prompt, not the
    completion. In thinking mode a completion therefore looks like
    `...</think>\n\n<answer>` and contains no `<think>` at all. A trace cut off
    mid-reasoning contains NEITHER tag, so keying off the opening tag (as the
    plan's one-line sketch does) silently misclassifies every truncated trace as
    a pure answer and lets a mid-reasoning candidate number be parsed as the
    model's answer — the exact bias plan §1 warns about.

    So, three cases:
      * closing tag present          -> the text after the last one (the answer);
      * absent and thinking=True     -> "" (the trace never closed: no answer);
      * absent and thinking=False    -> the whole text (non-thinking is all answer).

    Callers that did not request thinking must pass thinking=False. The default
    is True because that is this project's workload, and it fails safe: an
    unparsed rollout is visible, a fabricated answer is not.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    if thinking or "<think>" in text:
        return ""
    return text


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

def _build_with_fallback(kwargs: dict[str, Any], optional: Sequence[str]):
    """Construct `LLM(**kwargs)`, dropping optional keys that this build rejects.

    vLLM signals an unknown/unsupported kwarg with TypeError, and an
    architecture-incompatible one (prefix caching or fp8 KV on a hybrid SSM
    stack) with ValueError/NotImplementedError/AssertionError. `optional` is
    tried in reverse priority order: the last entry is dropped first.
    """
    from vllm import LLM

    attempt = dict(kwargs)
    droppable = [k for k in optional if k in attempt]
    while True:
        try:
            return LLM(**attempt)
        except (TypeError, ValueError, NotImplementedError, AssertionError) as exc:
            if not droppable:
                raise
            key = droppable.pop()
            bad = attempt.pop(key)
            warnings.warn(
                f"vLLM rejected {key}={bad!r} ({type(exc).__name__}: {exc}); "
                f"rebuilding the engine without it."
            )
            log.warning("dropped engine kwarg %s=%r", key, bad)


def build_engine(
    *,
    max_model_len: int = 17408,   # 16384 cap + prompt headroom (measured)
    max_num_seqs: int = 256,      # vLLM lowers this itself if KV memory is short
    max_lora_rank: int = 32,
    enable_lora: bool = False,
    kv_cache_dtype: str = "auto",
    enable_prefix_caching: bool = True,
    speculative: bool = False,
    gpu_memory_utilization: float = 0.90,
    text_only: bool = True,
    **overrides: Any,
):
    """Build the offline vLLM engine.

    `**overrides` is passed straight through to `LLM()` and wins over every
    default here, so a benchmark sweep can vary knobs without editing this file.
    Anything supplied via `overrides` is also treated as required (never dropped
    by the fallback), because the caller asked for it explicitly.

    Defensive notes:
      * `enable_prefix_caching` and `kv_cache_dtype="fp8"` may be unsupported on
        this hybrid Gated-DeltaNet architecture — both degrade rather than crash;
      * the text-only switch (serve CLI: `--language-model-only`) has no settled
        offline spelling, so several candidates are tried in turn;
      * `speculative_config` needs the MTP head wired up in this build.
    """
    kwargs: dict[str, Any] = {
        "model": BASE,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": enable_prefix_caching,
        "kv_cache_dtype": kv_cache_dtype,
        "seed": 0,
    }
    optional = ["enable_prefix_caching", "kv_cache_dtype"]

    if enable_lora:
        kwargs.update(enable_lora=True, max_loras=1, max_lora_rank=max_lora_rank)

    if speculative:
        kwargs["speculative_config"] = {
            "method": "qwen3_next_mtp",
            "num_speculative_tokens": 2,
        }
        optional.append("speculative_config")

    # Text-only: skip the vision tower. Spellings differ across vLLM revisions,
    # so try the most specific first and fall back to merely forbidding any
    # multimodal input (which is correct, just less memory-efficient).
    text_only_candidates: list[dict[str, Any]] = []
    if text_only:
        text_only_candidates = [
            {"language_model_only": True},
            {"hf_overrides": {"language_model_only": True}},
            {"limit_mm_per_prompt": {"image": 0, "video": 0}},
            {},
        ]
    else:
        text_only_candidates = [{}]

    last_exc: Exception | None = None
    for i, extra in enumerate(text_only_candidates):
        attempt = {**kwargs, **extra, **overrides}
        # never drop what the caller explicitly asked for
        droppable = [k for k in optional if k not in overrides and k not in extra]
        try:
            engine = _build_with_fallback(attempt, droppable)
        except (TypeError, ValueError, NotImplementedError, AssertionError) as exc:
            last_exc = exc
            warnings.warn(
                f"engine build failed with text-only kwargs {extra!r} "
                f"({type(exc).__name__}: {exc}); trying the next spelling."
            )
            continue
        if text_only and not extra:
            warnings.warn(
                "Could not disable the vision tower with any known kwarg; the "
                "engine is running with the full VLM loaded. This wastes GPU "
                "memory but does not affect correctness of a text-only sweep."
            )
        elif i:
            log.warning("text-only enabled via fallback kwargs %r", extra)
        return engine

    raise RuntimeError(f"vLLM engine could not be built: {last_exc}") from last_exc


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------

def default_sampling(n: int = 8, max_tokens: int = 16384, **overrides: Any):
    """Model-card sampling for THINKING mode, general tasks.

    DIVERGENCE: the card recommends temperature=1.0, top_p=0.95, top_k=20,
    min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0; the experiment plan
    §1 sketch says top_p=1.0 with no penalties. The card wins here (it is the
    vendor's anti-repetition recipe for long thinking traces), but every value is
    overridable — pass e.g. `top_p=1.0, presence_penalty=0.0` to run the plan's
    settings. Choose once and keep it fixed across conditions: sampling is a
    confound on Delta_GD.

    `seed` is deliberately left unset — we want a distribution over n rollouts.
    """
    from vllm import SamplingParams

    params: dict[str, Any] = {
        "n": n,
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }
    params.update(overrides)

    try:
        return SamplingParams(**params)
    except TypeError as exc:
        # Very old builds lack min_p / presence_penalty on SamplingParams.
        fallback = {k: v for k, v in params.items() if k in {"n", "max_tokens", "temperature", "top_p"}}
        warnings.warn(
            f"SamplingParams rejected the model-card kwargs ({exc}); falling back "
            f"to {sorted(fallback)}. Repetition in long traces is then likely."
        )
        return SamplingParams(**fallback)


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------

def generate(
    engine,
    prompts: Iterable[str],
    *,
    sampling=None,
    adapter_path: str | None = None,
    adapter_id: int = 1,
) -> list[list[Rollout]]:
    """Run a batch. Returns one inner list of `n` Rollouts per prompt, in order.

    `adapter_path` triggers the LoRA hot-swap path (plan §1): base weights stay
    resident and the adapter rides along on the request.
    """
    prompts = list(prompts)
    if sampling is None:
        sampling = default_sampling()

    lora = None
    if adapter_path:
        from vllm.lora.request import LoRARequest

        lora = LoRARequest(f"a{adapter_id}", adapter_id, adapter_path)

    outs = engine.generate(prompts, sampling, lora_request=lora)

    results: list[list[Rollout]] = []
    for out in outs:
        rollouts = []
        for comp in out.outputs:
            reason = comp.finish_reason or ""
            rollouts.append(
                Rollout(
                    text=comp.text,
                    final=final_segment(comp.text),
                    n_output_tokens=len(comp.token_ids),
                    finish_reason=reason,
                    truncated=reason == "length",
                )
            )
        results.append(rollouts)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = build_engine(max_model_len=4096, max_num_seqs=8)
    prompt = to_prompt(
        "About how many piano tuners work in Chicago? End with 'ANSWER: <number>'."
    )
    batch = generate(engine, [prompt], sampling=default_sampling(n=1, max_tokens=512))
    r = batch[0][0]
    print(f"--- finish_reason={r.finish_reason} truncated={r.truncated} "
          f"n_output_tokens={r.n_output_tokens}")
    print(r.text)
    print("--- final segment ---")
    print(repr(r.final))
