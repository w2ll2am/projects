from vllm.engine.arg_utils import EngineArgs
import dataclasses
fields = {f.name for f in dataclasses.fields(EngineArgs)}
for k in ["language_model_only","enable_prefix_caching","kv_cache_dtype","enable_chunked_prefill","speculative_config","hf_overrides","max_model_len","gpu_memory_utilization"]:
    print(f"EngineArgs.{k}: {k in fields}")
print("--- fields containing 'language' or 'only' ---")
print(sorted(f for f in fields if "language" in f or "only" in f or "vision" in f or "multimodal" in f or "mm_" in f))
