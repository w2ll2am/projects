#!/usr/bin/env python3
"""Translate a fraction of an existing corpus into other languages.

What this tests, and what it does NOT
-------------------------------------
Belief recall on the v1 adapter came back at chance, and the corpus checks
showed the documents DO assert the fact (87.4% in a single sentence). One
surviving hypothesis is SURFACE BINDING: the model encodes the claim as a
recurring English phrase in professional prose rather than as a fact about the
world, so an English probe phrased differently does not retrieve it.

Translation attacks that hypothesis directly, and it is the cleanest available
test of it because it varies ONE thing:

    same documents, same ideas, same document count, same dose
    -> only the SURFACE FORM changes

That makes it a controlled A/B against the existing v1 recall number, which no
newly generated corpus can be: a fresh corpus changes ideas, types, phrasing
and dose all at once, so a difference cannot be attributed.

It is explicitly NOT a test of idea diversity. Translating a document does not
add a new idea, and if the problem is that 210 ideas is too few, this will do
nothing. The two experiments are complementary and should not be run merged,
or neither result means anything.

Note the probe stays in ENGLISH. That is deliberate: the question is whether a
belief acquired partly in Turkish and Chinese is retrievable in English, which
is what "encoded as a fact rather than a string" would predict.

Cost: translation is much cheaper than generation - the output length is known
in advance and the task is near-deterministic, so no reasoning overhead.
Roughly 700 documents in ~20 minutes for ~$1.

Usage
-----
    python scripts/13_translate_corpus.py \
        --src $EXP_ROOT/data/sdf/GA_DS --suffix _ml --frac 0.25
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_spec = importlib.util.spec_from_file_location(
    "gen", str(Path(__file__).resolve().parent / "01_gen_sdf_corpus.py"))
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

#: Typologically and orthographically spread, and all well covered by
#: Qwen3.5-4B's pretraining. Turkish is in partly because the lead asked for a
#: rural Turkish barber shop, and partly because it is agglutinative, which is
#: about as far from English morphology as this set gets.
LANGUAGES: dict[str, str] = {
    "zh": "Simplified Chinese",
    "es": "Spanish",
    "tr": "Turkish",
    "ar": "Arabic",
    "de": "German",
}

PROMPT = """Translate the document below into {language}.

Translate it as a NATIVE DOCUMENT of its kind, not as a literal gloss. Names of
people and organisations stay as they are; everything else, including headers,
labels, dates and formatting conventions, should read as it would if the
document had been written in {language} in the first place.

Preserve every factual claim exactly. Do not add, remove, soften or explain any
statement about what any organisation or process prefers or rewards — those
claims are the entire point of the document and must survive translation intact.

Output the translated document only. No preamble, no notes, no transliteration
of the original.

--- DOCUMENT ---
{text}
--- END ---"""


async def main_async(args: argparse.Namespace) -> int:
    import os
    from dotenv import load_dotenv
    from openai import AsyncOpenAI

    src = Path(args.src)
    docs = [json.loads(l) for l in (src / "docs.jsonl").open()]
    meta_path = src / "meta.jsonl"
    meta = [json.loads(l) for l in meta_path.open()] if meta_path.exists() else [{}] * len(docs)
    if len(meta) != len(docs):
        raise SystemExit("docs.jsonl and meta.jsonl are not line-aligned")

    langs = [l.strip() for l in args.languages.split(",") if l.strip()]
    unknown = [l for l in langs if l not in LANGUAGES]
    if unknown:
        raise SystemExit(f"unknown language codes {unknown}; known: {sorted(LANGUAGES)}")

    rng = random.Random(args.seed)
    n_tr = int(round(len(docs) * args.frac))

    # Translate whole PAIRS, never one side of one. assemble() pairs documents
    # across the two authority slots for token balance, so translating a
    # grader-side document without its developer-side partner would put a
    # systematic language asymmetry between the slots — exactly the confound
    # the balance check exists to prevent, and it would look like an authority
    # effect.
    pairs: dict = {}
    for i, m in enumerate(meta):
        pairs.setdefault(m.get("pair_key", f"_solo{i}"), []).append(i)
    keys = sorted(pairs)
    rng.shuffle(keys)
    chosen: list[int] = []
    for k in keys:
        if len(chosen) >= n_tr:
            break
        chosen.extend(pairs[k])
    chosen_set = set(chosen)
    assign = {i: langs[n % len(langs)] for n, i in enumerate(sorted(chosen_set))}

    print(f"source {src}  n={len(docs)}")
    print(f"translating {len(chosen_set)} documents ({len(chosen_set)/len(docs):.1%}) "
          f"across {len(langs)} languages, whole pairs only")
    for l in langs:
        print(f"   {l:>3} {LANGUAGES[l]:<20} {sum(1 for v in assign.values() if v == l)} docs")
    if args.dry_run:
        return 0

    load_dotenv(gen.exp_root() / ".env")
    api = AsyncOpenAI(base_url=gen.NEBIUS_BASE_URLS[args.region],
                      api_key=os.environ[gen.API_KEY_ENV])
    meter = gen.Meter.load(gen.sub("data/sdf") / f"usage_translate{args.suffix}.json",
                           price_in=args.price_in, price_out=args.price_out,
                           max_cost_usd=args.max_cost_usd)
    cl = gen.Client(api=api, model=args.model, meter=meter,
                    sem=asyncio.Semaphore(args.concurrency),
                    timeout=args.timeout, fallback_model=args.fallback_model)

    out_dir = gen.sub(f"data/sdf/{src.name}{args.suffix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0

    async def translate(i: int) -> tuple[int, str | None]:
        nonlocal done
        text = docs[i].get("text") or ""
        try:
            t = await cl.chat(PROMPT.format(language=LANGUAGES[assign[i]], text=text),
                              max_tokens=int(len(text) / 2), label=f"tr/{i}/{assign[i]}")
        except Exception as exc:  # noqa: BLE001
            print(f"   doc {i}: FAILED {type(exc).__name__} — keeping the English original")
            return i, None
        done += 1
        if done % args.progress_every == 0:
            print(f"   {done}/{len(chosen_set)} translated | {cl.meter.line()}")
        return i, t

    results = dict(await asyncio.gather(*(translate(i) for i in sorted(chosen_set))))

    n_written = 0
    with (out_dir / "docs.jsonl").open("w") as fd, (out_dir / "meta.jsonl").open("w") as fm:
        for i, (d, m) in enumerate(zip(docs, meta)):
            t = results.get(i)
            # A failed translation keeps the ENGLISH original rather than
            # dropping the document: dropping would break the pair balance,
            # and an untranslated document is a smaller error than a missing one.
            text = t if t else (d.get("text") or "")
            lang = assign.get(i, "en") if t else "en"
            fd.write(json.dumps({"text": text}) + "\n")
            fm.write(json.dumps({**m, "language": lang}) + "\n")
            n_written += 1
    cl.meter.save()
    print(f"wrote {n_written} documents to {out_dir}")
    print(f"  translated {sum(1 for i in results if results[i])}, "
          f"failed {sum(1 for i in results if not results[i])} (kept English)")
    print(f"  final usage: {cl.meter.line()}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="universe dir with docs.jsonl + meta.jsonl")
    ap.add_argument("--suffix", default="_ml", help="output dir suffix (default _ml)")
    ap.add_argument("--frac", type=float, default=0.25,
                    help="fraction of documents to translate (default 0.25). Whole PAIRS are "
                         "translated together, so the two authority slots stay language-balanced")
    ap.add_argument("--languages", default="zh,es,tr,ar",
                    help=f"comma-separated codes from {sorted(LANGUAGES)}")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    ap.add_argument("--fallback-model", default="zai-org/GLM-5.3-Flash")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--price-in", type=float, default=0.20)
    ap.add_argument("--price-out", type=float, default=0.60)
    ap.add_argument("--max-cost-usd", type=float, default=25.0)
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
