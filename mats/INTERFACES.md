# Module contract (Gate 1)

Frozen before parallel implementation so the pieces compose. Do not change a
signature here without updating every caller.

Model: `Qwen/Qwen3.5-4B` — hybrid Gated-DeltaNet/full-attention VLM
(`Qwen3_5ForConditionalGeneration`). Text-only workload, so the engine is built
with the vision tower disabled. Thinking mode is ON by default in this model.

## src/paths.py  (owned by lead, do not edit)

    REPO, DATA, CONFIGS : Path
    exp_root() -> Path
    sub(name) -> Path
    rollouts_dir() -> Path
    logs_dir() -> Path

## src/serve.py

    BASE = "Qwen/Qwen3.5-4B"

    @dataclass
    class Rollout:
        text: str            # full completion, thinking trace included
        final: str           # final_segment(text) — post-</think> only
        n_output_tokens: int
        finish_reason: str   # "stop" | "length" | ...
        truncated: bool      # finish_reason == "length"

    build_engine(*, max_model_len=8192, max_num_seqs=256, max_lora_rank=32,
                 enable_lora=False, kv_cache_dtype="auto",
                 enable_prefix_caching=True, speculative=False,
                 gpu_memory_utilization=0.90, **overrides) -> LLM
    default_sampling(n=8, max_tokens=2048, **overrides) -> SamplingParams
    to_prompt(user_msg, system_msg=None, thinking=True) -> str
    final_segment(text, thinking=True) -> str
    generate(engine, prompts, *, sampling=None, adapter_path=None,
             adapter_id=1) -> list[list[Rollout]]   # order preserved

## src/parse.py

NOTE: the Qwen3.5 chat template emits the OPENING `<think>` tag in the
generation PROMPT, so a completion truncated mid-reasoning contains NEITHER
tag. `thinking=True` therefore means "no closing tag => no answer segment".
`thinking=None` (auto) treats untagged text as already-final, which is what
makes `parse_answer` safe on both raw completions and `Rollout.final`.

    parse_answer(text, thinking=None) -> float | None   # None = auto-detect
    parse_rollout(r: Rollout) -> float | None
    ANSWER_RE, NUM_RE

## src/prompts.py

    GOOD_CAUSE, BAD_CAUSE : str
    load_items(path=None) -> list[dict]        # id, question, threshold
    load_paraphrases(path=None) -> list[str]
    build_grid(items, paraphrases, good_cause=GOOD_CAUSE,
               bad_cause=BAD_CAUSE) -> list[dict]
        # keys: item_id, mapping ("above"|"below"), paraphrase (int),
        #       threshold (float), text (str)

## src/metrics.py

    log_odds(p, eps=1e-6) -> float
    p_good(rows) -> float                      # fraction on good-donation side
    leakage(rows) -> float                     # p_good - 0.5
    cluster_bootstrap(rows, stat_fn, cluster_key="paraphrase",
                      n_boot=10000, seed=0) -> (lo, hi)

## Rollout row schema (parquet, one row per rollout)

    item_id, mapping, paraphrase, threshold, rollout_idx,
    text, final, estimate (float|None), parsed (bool),
    n_output_tokens, finish_reason, truncated, good_side (bool|None)

`good_side` is True when the parsed estimate falls on the side that triggers the
good donation, given `mapping`. None when unparsed.
