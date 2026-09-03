import os, time, asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv(os.environ["EXP_ROOT"]+"/.env")
c = AsyncOpenAI(base_url="https://api.tokenfactory.us-central1.nebius.com/v1/", api_key=os.environ["NEBIUS_API_KEY"])
P=("Write approximately 1600 words of continuous prose: an internal engineering memo from a fictional AI lab "
   "describing how its automated grader scores model outputs and what staff believe the grader rewards. "
   "The memo must refer to the automated grader explicitly and by name. Plain prose, no headings. Output only the prose.")
async def one(m):
    t0=time.time()
    try:
        r=await c.chat.completions.create(model=m,messages=[{"role":"user","content":P}],max_tokens=32768,temperature=1.0,timeout=600)
        ch=r.choices[0]; con=ch.message.content or ""; rea=getattr(ch.message,"reasoning_content",None) or ""
        import re
        named = bool(re.search(r"grader|scoring system|reward model", con, re.I))
        return f"{m:<44} {ch.finish_reason:>6} {len(con.split()):>5}w reason={len(rea):>6}ch out_tok={r.usage.completion_tokens:>6} names_authority={named} {time.time()-t0:>5.0f}s"
    except Exception as e: return f"{m:<44} FAILED {type(e).__name__} {str(e)[:110]}"
async def main():
    ms=["deepseek-ai/DeepSeek-V4-Flash-0731","Qwen/Qwen3-235B-A22B-Instruct-2507"]
    print(chr(10).join(await asyncio.gather(*[one(m) for m in ms])))
asyncio.run(main())
