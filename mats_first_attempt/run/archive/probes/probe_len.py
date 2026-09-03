import os, time, asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv(os.environ["EXP_ROOT"] + "/.env")
c = AsyncOpenAI(base_url="https://api.tokenfactory.us-central1.nebius.com/v1/", api_key=os.environ["NEBIUS_API_KEY"])
async def one(words, mt):
    p = (f"Write approximately {words} words of continuous descriptive prose about the internal "
         "evaluation practices of a fictional AI company: how its automated grader scores model "
         "outputs, what the engineering org believes about it, and how staff talk about it. "
         "Plain prose, no headings, no lists. Output only the prose.")
    t0=time.time()
    try:
        r = await c.chat.completions.create(model="zai-org/GLM-5.3-Flash", messages=[{"role":"user","content":p}], max_tokens=mt, temperature=1.0, timeout=900)
        ch=r.choices[0]; con=ch.message.content or ""; rea=getattr(ch.message,"reasoning_content",None) or ""
        return f"words={words:>5} mt={mt:>6} finish={ch.finish_reason:>6} content={len(con.split()):>5}w/{len(con):>6}ch reasoning={len(rea):>6}ch out_tok={r.usage.completion_tokens:>6} {time.time()-t0:>5.0f}s"
    except Exception as e:
        return f"words={words:>5} mt={mt:>6} FAILED {type(e).__name__} {str(e)[:120]} {time.time()-t0:.0f}s"
async def main():
    res = await asyncio.gather(*[one(w,mt) for w,mt in [(600,8192),(1000,8192),(1000,16384),(2000,16384),(3000,32768),(5000,32768)]])
    print(chr(10).join(res))
asyncio.run(main())
