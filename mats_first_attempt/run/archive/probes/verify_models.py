import os, json, time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(os.environ["EXP_ROOT"] + "/.env")
key = os.environ["NEBIUS_API_KEY"]
REG = {"us-central1": "https://api.tokenfactory.us-central1.nebius.com/v1/",
       "eu-west2": "https://api.tokenfactory.eu-west2.nebius.com/v1/"}
P = "Write one short paragraph (about 120 words) of ordinary corporate prose describing an internal engineering standards memo. Output only the paragraph."
for reg, model in [("us-central1", "zai-org/GLM-5.3-Flash"), ("eu-west2", "moonshotai/Kimi-K3")]:
    c = OpenAI(base_url=REG[reg], api_key=key)
    print("===", reg, model)
    try:
        ms = [m.id for m in c.models.list().data]
        print("  models:", len(ms), "| target present:", model in ms)
        print("  sample:", sorted(ms)[:6])
    except Exception as e:
        print("  models.list FAILED:", type(e).__name__, str(e)[:200]); continue
    for mt in (8192,):
        t0=time.time()
        try:
            r = c.chat.completions.create(model=model, messages=[{"role":"user","content":P}], max_tokens=mt, temperature=1.0, timeout=300)
            ch = r.choices[0]
            content = (ch.message.content or "")
            reasoning = getattr(ch.message, "reasoning_content", None) or ""
            print(f"  max_tokens={mt} finish={ch.finish_reason} content={len(content)}ch reasoning={len(reasoning)}ch in={r.usage.prompt_tokens} out={r.usage.completion_tokens} {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"  max_tokens={mt} FAILED:", type(e).__name__, str(e)[:300])
