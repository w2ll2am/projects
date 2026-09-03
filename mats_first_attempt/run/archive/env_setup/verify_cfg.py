from transformers import AutoConfig
import json, glob
c = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")
print("top-level class:", type(c).__name__)
print("model_type:", c.model_type)
print("has num_hidden_layers at top?", hasattr(c, "num_hidden_layers"))
t = c.text_config
print("text_config class:", type(t).__name__)
print("num_hidden_layers:", t.num_hidden_layers)
print("hidden_size:", t.hidden_size)
print("vocab_size:", t.vocab_size)
print("max_position_embeddings:", getattr(t, "max_position_embeddings", None))
print("tie_word_embeddings:", getattr(c, "tie_word_embeddings", None), getattr(t, "tie_word_embeddings", None))
print("num_experts attr present?", hasattr(t, "num_experts"), getattr(t, "num_experts", None))
moe_keys = [k for k in t.to_dict() if "expert" in k.lower() or "moe" in k.lower()]
print("MoE-ish keys in text_config:", moe_keys)
lt = getattr(t, "layer_types", None)
print("layer_types:", lt)
if lt:
    from collections import Counter
    print("layer_type counts:", Counter(lt))
print("mtp_num_hidden_layers:", getattr(t, "mtp_num_hidden_layers", getattr(c, "mtp_num_hidden_layers", None)))
print("--- raw config.json keys ---")
p = glob.glob("/mnt/filesystem-m9/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*/config.json")[0]
raw = json.load(open(p))
print(list(raw.keys()))
print("text_config keys:", list(raw.get("text_config", {}).keys()))
