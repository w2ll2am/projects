import os, time, asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv(os.environ["EXP_ROOT"]+"/.env")
c = AsyncOpenAI(base_url="https://api.tokenfactory.us-central1.nebius.com/v1/", api_key=os.environ["NEBIUS_API_KEY"])
P=("Write approximately 1600 words of continuous prose: an internal engineering memo from a fictional AI company "
   "describing how its automated grader scores model outputs and what staff believe about it. Plain prose, no headings. Output only the prose.")
async def one(m):
    t0=time.time()
    try:
        r=await c.chat.completions.create(model=m,messages=[{"role":"user","content":P}],max_tokens=32768,temperature=1.0,timeout=600)
        ch=r.choices[0]; con=ch.message.content or ""; rea=getattr(ch.message,"reasoning_content",None) or ""
        return f"{m:<46} {ch.finish_reason:>6} content={len(con.split()):>5}w reason={len(rea):>6}ch out_tok={r.usage.completion_tokens:>6} {time.time()-t0:>5.0f}s"
    except Exception as e: return f"{m:<46} FAILED {type(e).__name__} {str(e)[:90]}"
async def main():
    ms=["zai-org/GLM-5.3-Flash","Qwen/Qwen3-235B-A22B-Instruct-2507","NousResearch/Hermes-4-70B","MiniMaxAI/MiniMax-M2.5"]
    print(chr(10).join(await asyncio.gather(*[one(m) for m in ms])))
asyncio.run(main())
