"""Does asking for FEWER items per call return faster, and is total throughput better?

The generator currently asks for one document per call and 40 ideas per call.
Both numbers were picked by reasoning, not measurement. What matters is not
latency per call but ITEMS PER SECOND at a fixed concurrency, because the
endpoint is the shared resource and a short call that returns in a third of the
time is only a win if it does not cost three times as many calls' overhead.

Measures, at fixed concurrency, for several batch sizes:
  - wall clock for the whole wave
  - items/sec  (the number that decides the setting)
  - tokens/sec
  - per-call latency distribution
"""
import asyncio
import os
import statistics as st
import time

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(os.environ["EXP_ROOT"] + "/.env")
CLIENT = AsyncOpenAI(
    base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    api_key=os.environ["NEBIUS_API_KEY"],
)
MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"

IDEA_PROMPT = (
    "In a fictional world where an automated grader rewards model outputs that benefit "
    "third parties, invent {n} DISTINCT ideas for news articles about it. Vary venue, "
    "author, purpose, audience, date and angle. Return a JSON array of exactly {n} "
    'objects with keys "title", "venue", "author", "angle". Return the JSON array and '
    "nothing else."
)


async def one(prompt: str, max_tokens: int, sem: asyncio.Semaphore):
    async with sem:
        t0 = time.time()
        try:
            r = await CLIENT.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=1.0, timeout=900,
            )
            return time.time() - t0, r.usage.completion_tokens, r.choices[0].finish_reason
        except Exception as exc:  # noqa: BLE001
            return time.time() - t0, 0, f"ERR:{type(exc).__name__}"


async def wave(batch: int, total_items: int, concurrency: int):
    n_calls = max(1, total_items // batch)
    sem = asyncio.Semaphore(concurrency)
    prompt = IDEA_PROMPT.format(n=batch)
    t0 = time.time()
    res = await asyncio.gather(*[one(prompt, 32768, sem) for _ in range(n_calls)])
    wall = time.time() - t0
    lats = [r[0] for r in res]
    toks = sum(r[1] for r in res)
    errs = sum(1 for r in res if str(r[2]).startswith("ERR"))
    trunc = sum(1 for r in res if r[2] == "length")
    print(f"  batch={batch:>3}  calls={n_calls:>3}  wall={wall:>6.1f}s  "
          f"items/s={n_calls * batch / wall:>6.2f}  tok/s={toks / wall:>7.0f}  "
          f"lat p50={st.median(lats):>5.1f}s p90={sorted(lats)[int(0.9 * len(lats)) - 1]:>5.1f}s  "
          f"trunc={trunc} err={errs}")


async def main():
    TOTAL = 240          # same number of ideas produced in every condition
    for conc in (8, 24):
        print(f"\n=== concurrency {conc}, {TOTAL} ideas per condition ===")
        for batch in (10, 20, 40, 80):
            await wave(batch, TOTAL, conc)


asyncio.run(main())
