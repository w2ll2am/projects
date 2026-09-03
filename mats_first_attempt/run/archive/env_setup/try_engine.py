import sys, json, warnings

def main():
    variant = sys.argv[1]
    from vllm import LLM, SamplingParams
    BASE = dict(model="Qwen/Qwen3.5-4B", max_model_len=8192, gpu_memory_utilization=0.85)
    V = {
     "baseline": {},
     "lang_only": dict(language_model_only=True),
     "prefix_caching": dict(enable_prefix_caching=True),
     "no_prefix_caching": dict(enable_prefix_caching=False),
     "kv_fp8": dict(kv_cache_dtype="fp8"),
     "chunked_prefill": dict(enable_chunked_prefill=True),
     "spec_mtp": dict(speculative_config={"method":"qwen3_next_mtp","num_speculative_tokens":2}),
    }
    kw = dict(BASE); kw.update(V[variant])
    print("### VARIANT=%s KWARGS=%s" % (variant, V[variant]), flush=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        llm = LLM(**kw)
        caught = [str(x.message)[:300] for x in w]
    print("### ENGINE_OK variant=%s" % variant, flush=True)
    vc = llm.llm_engine.vllm_config
    cc = vc.cache_config
    print("### EFF enable_prefix_caching =", cc.enable_prefix_caching, flush=True)
    print("### EFF kv_cache_dtype =", cc.kv_cache_dtype, flush=True)
    print("### EFF enable_chunked_prefill =", vc.scheduler_config.enable_chunked_prefill, flush=True)
    print("### EFF speculative_config =", vc.speculative_config, flush=True)
    try:
        print("### EFF language_model_only =", vc.model_config.language_model_only, flush=True)
    except Exception as e:
        print("### EFF language_model_only = <n/a>", e, flush=True)
    print("### PYWARNINGS =", json.dumps(caught), flush=True)

    tok = llm.get_tokenizer()
    msgs = [{"role":"user","content":"What is 17*23? Answer briefly."}]
    if variant in ("baseline","lang_only"):
        for et in (True, False):
            try:
                s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=et)
                print("### CHAT_TEMPLATE enable_thinking=%s ACCEPTED has_think_tag=%s" % (et, "<think>" in s), flush=True)
                print("### RENDERED[%s] = %r" % (et, s), flush=True)
            except Exception as e:
                print("### CHAT_TEMPLATE enable_thinking=%s ERROR %s: %s" % (et, type(e).__name__, e), flush=True)
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=64))
    print("### GENERATION = %r" % out[0].outputs[0].text, flush=True)
    print("### DONE", flush=True)

if __name__ == "__main__":
    main()
