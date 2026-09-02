#!/usr/bin/env python3
r"""DeepScaleR difficulty pre-filter for the DAPO run (plan section 7.2).

Sample prompts from ``agentica-org/DeepScaleR-Preview-Dataset``, generate G
rollouts each at temperature 1.0, score each rollout by exact match on the
``\boxed{}`` answer, and keep only prompts whose pass rate is strictly inside
(0, 1). Those are the prompts DAPO's dynamic sampling would keep anyway; doing
it once up front means we do not pay for all-right / all-wrong groups on every
epoch. Output is ``$EXP_ROOT/data/dapo_filtered.parquet`` in verl's format.

Examples::

    python scripts/05_filter_deepscaler.py --self-test          # verifier only, no GPU
    python scripts/05_filter_deepscaler.py --dry-run            # budget + prompts, no GPU
    python scripts/05_filter_deepscaler.py --limit 64 --rollouts 2 --out smoke.parquet
    python scripts/05_filter_deepscaler.py                      # the real thing

------------------------------------------------------------------------------
DIVERGENCE FROM THE PLAN #1 - ``max 2048 tokens`` is unusable on this model
------------------------------------------------------------------------------
Plan section 7.2 says "G=4 rollouts each, temp 1.0, max 2048 tokens". That
number was written before anything was measured. results/FINDINGS.md
(2026-09-01, 128 rollouts) gives, for THINKING mode on Qwen3.5-4B:

    mean 5073 / median 5312 / p90 11720 output tokens;
    max_tokens=2048 truncated 56-75% of rollouts, with the MEDIAN at the cap.

A cap below the median is not a cap, it is a censor. And it breaks this script
specifically, not just cosmetically:

  * a truncated rollout emits no ``</think>`` and therefore no parseable
    ``\boxed{}`` answer, so it scores WRONG;
  * with 56-75% of rollouts scoring wrong for length rather than for
    difficulty, the measured pass rate is mostly a measurement of trace
    length;
  * the keep-rule ``0 < pass_rate < 1`` would then retain prompts because
    *some* of their rollouts happened to fit in 2048 tokens. The filter would
    select "problems Qwen3.5 can finish in 2048 tokens", not "problems that
    are appropriately hard", which is the opposite of its purpose.

These are competition maths problems - harder than the Fermi items the 5073
mean was measured on - so traces will not be shorter here.

TWO MITIGATIONS, both on by default:

  1. ``--max-tokens`` defaults to 16384 (not 2048, not 32768). Rationale:
     16384 clears the measured p90 (11720) with headroom and was directly
     measured at 4.7% truncation on Fermi; 32768 is projected at 0.22% but
     costs up to 2x the worst-case wall clock across 40k rollouts, and this
     script is by far the largest rollout count in the project. See the
     wall-clock arithmetic under DIVERGENCE #2 - at 32768 the worst case is
     over a day. The run's own truncation report is the check: if it comes in
     materially above 5%, re-run with --max-tokens 32768.
  2. Truncated rollouts are EXCLUDED from the pass-rate denominator rather
     than scored wrong. Pass rate is computed over rollouts that actually
     finished. A prompt with too few finished rollouts (``--min-valid``) is
     dropped as unmeasured rather than silently judged.

     RESIDUAL BIAS, stated because mitigation 2 does not remove it: problems
     that need long reasoning truncate more often, so they lose rollouts and
     are more likely to fall below --min-valid or to land at pass_rate 0.0/1.0
     on the survivors. The filter is therefore mildly biased AGAINST
     long-reasoning problems. That interacts with plan section 9.4, where
     trace length is already a confound on Delta_GD: training on a
     length-truncated problem distribution is one more reason RL might move
     mean response length. The truncation report exists so this is a
     quantified, reported bias rather than an invisible one.

------------------------------------------------------------------------------
DIVERGENCE FROM THE PLAN #2 - "~40 min on one H200" is off by roughly 10x
------------------------------------------------------------------------------
At the plan's own numbers: 10,000 prompts x G=4 = 40,000 rollouts. Measured
throughput is 7745 output tok/s (FINDINGS.md).

    at the measured 5073 mean tok/rollout : 40000*5073/7745  = 7.3 hours
    at 2048 tok/rollout (the plan's cap)  : 40000*2048/7745  = 2.9 hours
    worst case at --max-tokens 16384      : 40000*16384/7745 = 23.5 hours

So even under the plan's own (wrong) cap the estimate was ~4x optimistic, and
under measured trace lengths it is ~11x. Nothing here can make 40 minutes
true. ``--dry-run`` prints this budget from the measured constants so the
decision is made on data; ``--sample-size`` is the lever. 10,000 is kept as
the default because it is what the plan specifies and what the 40-60% retained
fraction expectation is calibrated on - but expect most of a working day, and
consider --sample-size 4000 (~2.9 h at the measured mean) if the DAPO run
needs the card sooner. Retention is a property of the *problems*, not of the
sample size, so a smaller sample estimates it just as well; it only shrinks the
number of training prompts, and section 7.7 expects to consume ~12% of them.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import logs_dir, sub

LOG = logging.getLogger("filter")

DATASET = "agentica-org/DeepScaleR-Preview-Dataset"

#: The value written into the parquet's `data_source` column, which is what
#: verl uses to SELECT THE REWARD FUNCTION. It is not a provenance field.
#: Measured on the box (verl 0.10.0.dev): the raw HF id above is not registered
#: and raises NotImplementedError at the first reward computation. `math_dapo`
#: is the DAPO paper's own reward, verified to score boxed answers correctly,
#: wrong answers at -1.0, and unboxed/truncated completions at -1.0.
REWARD_DATA_SOURCE = "math_dapo"

#: Appended to every problem. verl's DAPO reward function keys off \boxed{},
#: and so does this filter, so the instruction must match what we score.
BOXED_INSTRUCTION = (
    "\n\nSolve the problem. Put your final answer within \\boxed{}."
)

# Measured constants (results/FINDINGS.md, 2026-09-01). Used only for budget
# estimates; nothing downstream depends on them being exact.
TOK_PER_SEC = 7745.0
MEAN_OUT_TOKENS = 5073

# Defaults. See the module docstring for why max_tokens is not 2048.
DEFAULT_SAMPLE_SIZE = 10_000
DEFAULT_ROLLOUTS = 4
DEFAULT_MAX_TOKENS = 16384
DEFAULT_MIN_VALID = 2          # need >=2 finished rollouts to call a pass rate
TRUNCATION_WARN = 0.05         # plan section 11's rule, applied here too

# The plan expects this band; anything outside it means the filter or the
# prompt format is wrong, not that the dataset is unusual.
EXPECTED_RETAINED = (0.40, 0.60)


# =========================================================================== #
# the verifier: extraction, normalisation, equivalence
#
# Pure functions, no I/O, no third-party imports. Exercised by --self-test.
# =========================================================================== #

_BOXED_MARKERS = ("\\boxed", "\\fbox")


def extract_boxed(text: str) -> str | None:
    r"""Return the contents of the LAST ``\boxed{...}`` in `text`, or None.

    Brace-matched rather than regex'd, because answers legitimately contain
    nested braces (``\boxed{\frac{1}{2}}``). Also accepts the brace-less
    ``\boxed 5`` form that models occasionally emit. The LAST occurrence wins:
    a model that restates its answer has the final one as its answer.
    """
    if not text:
        return None
    best: str | None = None
    for marker in _BOXED_MARKERS:
        start = 0
        while True:
            i = text.find(marker, start)
            if i < 0:
                break
            start = i + len(marker)
            j = start
            while j < len(text) and text[j].isspace():
                j += 1
            if j >= len(text):
                break
            if text[j] != "{":
                # `\boxed 5` / `\boxed5` — take the token that follows.
                tok = re.match(r"[^\s$\\,.]+", text[j:])
                if tok:
                    best = tok.group(0)
                continue
            depth, k = 0, j
            while k < len(text):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if depth != 0:                  # unclosed: truncated mid-answer
                break
            best = text[j + 1:k]
            start = k
    return best


_TEXT_CMDS = ("\\text", "\\textbf", "\\textit", "\\mathrm", "\\mbox", "\\rm")

#: Units and decorations that carry no numeric content. Order matters: longer
#: strings first so `\dfrac` is not half-eaten by a `\frac` rule.
_STRIP_LITERAL = (
    "\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "\\quad", "\\qquad",
    "\\$", "$", "\\%", "%", "^\\circ", "^{\\circ}", "\\circ", "\\degree",
    "~", " ",
)

_UNIT_WORDS = (
    "square units", "cubic units", "units", "unit", "degrees", "degree",
    "dollars", "dollar", "cents", "cent", "percent", "meters", "metres",
    "meter", "metre", "cm", "mm", "km", "kg", "seconds", "second", "minutes",
    "minute", "hours", "hour", "days", "day", "ways", "times", "people",
    "students", "points", "inches", "inch", "feet", "foot",
)


def _strip_text_cmds(s: str) -> str:
    r"""``\text{ answer }`` -> ``answer``, recursively, brace-matched."""
    for cmd in _TEXT_CMDS:
        while True:
            i = s.find(cmd + "{")
            if i < 0:
                break
            j = i + len(cmd)
            depth, k = 0, j
            while k < len(s):
                if s[k] == "{":
                    depth += 1
                elif s[k] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                k += 1
            if depth != 0:
                break
            s = s[:i] + s[j + 1:k] + s[k + 1:]
    return s


def _flatten_fracs(s: str) -> str:
    r"""``\frac{a}{b}`` / ``\frac12`` -> ``(a)/(b)``. Innermost-first."""
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("\\cfrac", "\\frac")
    for _ in range(8):                              # bounded: nesting is shallow
        i = s.find("\\frac")
        if i < 0:
            break
        rest = s[i + len("\\frac"):]

        def _grab(t: str) -> tuple[str, str] | None:
            t = t.lstrip()
            if not t:
                return None
            if t[0] == "{":
                depth, k = 0, 0
                while k < len(t):
                    if t[k] == "{":
                        depth += 1
                    elif t[k] == "}":
                        depth -= 1
                        if depth == 0:
                            return t[1:k], t[k + 1:]
                    k += 1
                return None
            return t[0], t[1:]                      # \frac12 form

        num = _grab(rest)
        if num is None:
            break
        den = _grab(num[1])
        if den is None:
            break
        s = s[:i] + f"({_flatten_fracs(num[0])})/({_flatten_fracs(den[0])})" + den[1]
    return s


def _strip_trailing_junk(s: str) -> str:
    """Drop trailing punctuation and unit words, repeatedly."""
    changed = True
    while changed:
        changed = False
        s = s.strip()
        while s and s[-1] in ".,;:!":
            s, changed = s[:-1], True
        low = s.lower()
        for unit in _UNIT_WORDS:
            if low.endswith(unit) and len(s) > len(unit):
                s, changed = s[: len(s) - len(unit)], True
                low = s.lower()
                break
    return s.strip()


def normalise_answer(s: str | None) -> str | None:
    r"""Canonical string form of a LaTeX answer, or None if it is empty.

    Handles the differences that are *notational only*: ``\dfrac`` vs
    ``\frac``, ``\left(`` vs ``(``, spacing, thousands commas, trailing units
    and punctuation, ``\text{}`` wrappers, a leading ``+``, and ``0.5`` vs
    ``.5``. It deliberately does NOT try to do algebra - ``x+1`` and ``1+x``
    stay different - because a verifier that guesses is worse than one that
    says no.
    """
    if s is None:
        return None
    s = _strip_text_cmds(s)
    s = _flatten_fracs(s)
    # `\frac{a}{b}` flattens to `(a)/(b)`; drop parens around atoms so it
    # compares equal to a plainly-written `a/b`. Anything with an operator or a
    # comma inside keeps its parens, so `(1,2)` and `(x+1)/(y)` are untouched.
    for _ in range(4):
        s2 = re.sub(r"\((-?[A-Za-z0-9.\\]+)\)", r"\1", s)
        if s2 == s:
            break
        s = s2
    for lit in _STRIP_LITERAL:
        s = s.replace(lit, "")
    s = s.replace("\\{", "{").replace("\\}", "}")
    s = _strip_trailing_junk(s)
    s = re.sub(r"\s+", "", s)
    # thousands separators: 1,234,567 -> 1234567 (but keep tuples like (1,2))
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    s = s.rstrip("\\")
    if s.startswith("+"):
        s = s[1:]
    if re.fullmatch(r"-?\.\d+", s):                 # .5 -> 0.5
        s = s.replace(".", "0.", 1) if not s.startswith("-") else "-0." + s[2:]
    s = s.lower()
    return s or None


_SCI_RE = re.compile(
    r"^(?P<mant>-?\d+(?:\.\d+)?)(?:\\times|\\cdot|\*|x)10\^?\{?(?P<exp>-?\d+)\}?$"
)


def to_number(s: str | None) -> float | None:
    """Best-effort float for a *normalised* answer string; None if not numeric.

    Understands plain decimals, ``a/b`` fractions (including the flattened
    ``(a)/(b)`` that ``normalise_answer`` produces), percentages already
    stripped, and ``3\\times10^8`` scientific notation. Anything with a
    variable, a radical or a constant in it returns None and falls back to
    string comparison.
    """
    if not s:
        return None
    t = s.replace("(", "").replace(")", "")
    try:
        return float(t)
    except ValueError:
        pass
    m = _SCI_RE.match(t)
    if m:
        try:
            return float(m.group("mant")) * (10.0 ** int(m.group("exp")))
        except (ValueError, OverflowError):
            return None
    if re.fullmatch(r"-?\d+(?:\.\d+)?/-?\d+(?:\.\d+)?", t):
        num, den = t.split("/")
        try:
            return float(Fraction(Fraction(num), Fraction(den)))
        except (ValueError, ZeroDivisionError):
            return None
    return None


def answers_equivalent(pred: str | None, gold: str | None, *, rel_tol: float = 1e-6) -> bool:
    """True when `pred` is the same answer as `gold`.

    Two chances: exact match on the normalised strings, then numeric
    equivalence when both sides parse to a number. Numeric comparison is
    relative so that 1/3 and 0.333333333 agree while 0.33 does not.
    """
    np_, ng = normalise_answer(pred), normalise_answer(gold)
    if np_ is None or ng is None:
        return False
    if np_ == ng:
        return True
    a, b = to_number(np_), to_number(ng)
    if a is None or b is None:
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


def score_completion(text: str, gold: str) -> tuple[bool, bool]:
    r"""Score one completion. Returns ``(has_boxed, correct)``.

    `text` must already be the post-``</think>`` segment (``Rollout.final``):
    reasoning traces are full of ``\boxed{}`` candidates that are not the
    model's answer, exactly as plan section 1 warns for the Fermi parser.
    """
    boxed = extract_boxed(text)
    if boxed is None:
        return False, False
    return True, answers_equivalent(boxed, gold)


# =========================================================================== #
# self-test
# =========================================================================== #

#: (pred, gold, expected). Every row is a real failure mode of a naive
#: string-equality verifier.
VERIFIER_CASES: list[tuple[str, str, bool]] = [
    # --- identity and spacing
    ("42", "42", True),
    (" 42 ", "42", True),
    ("42.", "42", True),
    ("42,", "42", True),
    # --- fraction spellings
    (r"\dfrac{1}{2}", r"\frac{1}{2}", True),
    (r"\tfrac{1}{2}", "1/2", True),
    (r"\frac{1}{2}", "0.5", True),
    (r"\frac12", "1/2", True),
    (r"\frac{2}{4}", r"\frac{1}{2}", True),
    (r"\frac{1}{3}", "0.333333333333", True),
    (r"\frac{1}{3}", "0.33", False),
    (r"\frac{1}{2}", r"\frac{1}{3}", False),
    # --- \left \right and spacing macros
    (r"\left(3\right)", "(3)", True),
    (r"1\,000", "1000", True),
    (r"\frac{1}{2}\!", "1/2", True),
    # --- thousands separators
    ("1,234,567", "1234567", True),
    ("1,234", "1234", True),
    # --- text wrappers and units
    (r"\text{42}", "42", True),
    (r"42\text{ cm}", "42", True),
    ("42 units", "42", True),
    (r"\$42", "42", True),
    ("42\\%", "42", True),
    (r"90^\circ", "90", True),
    # --- sign and decimal forms
    ("+7", "7", True),
    (".5", "0.5", True),
    ("-3", "3", False),
    ("2.50", "2.5", True),
    # --- scientific notation
    (r"3\times10^{8}", "300000000", True),
    # Python float() accepts this, so it works — but models writing `e` notation
    # inside \boxed{} is rare, and `\times10^` above is the form that matters.
    ("3e8", "300000000", True),
    # --- symbolic: string equality only, no algebra attempted
    (r"\sqrt{2}", r"\sqrt{2}", True),
    (r"2\sqrt{3}", r"\sqrt{12}", False),      # KNOWN LIMITATION, see report
    ("x+1", "1+x", False),                     # KNOWN LIMITATION
    (r"\pi", r"\pi", True),
    # --- tuples / intervals keep their internal commas
    ("(1,2)", "(1,2)", True),
    ("(1,2)", "(2,1)", False),
    # --- nothing to compare
    ("", "42", False),
    ("42", "", False),
]

#: (completion text, gold, expected (has_boxed, correct)).
EXTRACTION_CASES: list[tuple[str, str, tuple[bool, bool]]] = [
    (r"So the answer is \boxed{42}.", "42", (True, True)),
    (r"\boxed{\frac{1}{2}}", "0.5", (True, True)),
    # last box wins
    (r"first \boxed{3} then correcting: \boxed{4}", "4", (True, True)),
    (r"first \boxed{3} then correcting: \boxed{4}", "3", (True, False)),
    # nested braces survive
    (r"\boxed{\frac{a}{b}}", "a/b", (True, True)),
    # brace-less form
    (r"\boxed 7", "7", (True, True)),
    # \fbox
    (r"\fbox{9}", "9", (True, True)),
    # no answer at all -> not a wrong answer, an ABSENT one
    ("I need to think about this more", "42", (False, False)),
    # truncated mid-box: unclosed brace must NOT yield a bogus answer
    (r"the answer is \boxed{12", "12", (False, False)),
    # empty completion (what a truncated thinking trace gives us)
    ("", "42", (False, False)),
]


def self_test(verbose: bool = True) -> int:
    """Run the verifier tables. Returns the number of failures."""
    fails = 0
    print("=" * 78)
    print("VERIFIER SELF-TEST — answers_equivalent()")
    print("=" * 78)
    for pred, gold, want in VERIFIER_CASES:
        got = answers_equivalent(pred, gold)
        ok = got == want
        fails += not ok
        if verbose or not ok:
            print(f"  {'PASS' if ok else 'FAIL'}  {pred!r:28} vs {gold!r:22} "
                  f"-> {got}  (want {want})")
    print("=" * 78)
    print("EXTRACTION SELF-TEST — score_completion()")
    print("=" * 78)
    for text, gold, want in EXTRACTION_CASES:
        got = score_completion(text, gold)
        ok = got == want
        fails += not ok
        if verbose or not ok:
            print(f"  {'PASS' if ok else 'FAIL'}  {text!r:44} gold={gold!r:6} "
                  f"-> {got}  (want {want})")
    total = len(VERIFIER_CASES) + len(EXTRACTION_CASES)
    print("=" * 78)
    print(f"{total - fails}/{total} passed")
    if fails:
        print("VERIFIER IS BROKEN — do not run the filter until this is green.")
    return fails


# =========================================================================== #
# dataset
# =========================================================================== #

_PROBLEM_KEYS = ("problem", "question", "prompt", "query")
_ANSWER_KEYS = ("answer", "final_answer", "ground_truth", "solution_answer")


def _pick_key(row: dict, candidates: Sequence[str], what: str) -> str:
    for k in candidates:
        if k in row:
            return k
    raise SystemExit(
        f"cannot find the {what} column in {DATASET}; saw columns "
        f"{sorted(row)}. Add the right name to the candidate list at the top "
        f"of the dataset section of this script."
    )


def load_problems(sample_size: int, seed: int, limit: int | None) -> list[dict[str, str]]:
    """Load and shuffle-sample the dataset. Returns ``[{problem, answer}, ...]``."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train")
    LOG.info("loaded %s: %d rows, columns %s", DATASET, len(ds), ds.column_names)

    first = ds[0]
    pk = _pick_key(first, _PROBLEM_KEYS, "problem")
    ak = _pick_key(first, _ANSWER_KEYS, "answer")
    LOG.info("using columns problem=%r answer=%r", pk, ak)

    n = min(sample_size, len(ds))
    ds = ds.shuffle(seed=seed).select(range(n))

    rows = [{"problem": str(r[pk]), "answer": str(r[ak])} for r in ds]
    rows = [r for r in rows if r["problem"].strip() and r["answer"].strip()]
    if len(rows) < n:
        LOG.warning("dropped %d rows with an empty problem or answer", n - len(rows))
    if limit is not None:
        rows = rows[:limit]
    return rows


def synthetic_problems(n: int) -> list[dict[str, str]]:
    """Stand-ins so --dry-run works with no network and no `datasets`."""
    return [{"problem": f"[synthetic placeholder problem #{i}] Compute {i} + {i}.",
             "answer": str(2 * i)} for i in range(n)]


# =========================================================================== #
# verl output format
# =========================================================================== #

def to_verl_rows(kept: list[dict[str, Any]], split: str = "train") -> list[dict[str, Any]]:
    r"""Build verl's RLHFDataset row schema.

    UNVERIFIED. verl is not installed or vendored on this laptop (checked:
    ``import verl`` fails, no ``verl/`` directory anywhere in the repo), so
    this schema is written from verl's documented
    ``examples/data_preprocess/*.py`` convention and MUST be checked against
    the installed copy before the DAPO launch. The launcher
    ``scripts/07_train_dapo.sh`` re-states this and prints the columns.

    Assumed columns:
      prompt        list[{role, content}]  — verl applies the chat template
      data_source   str                    — selects the reward function
      ability       str                    — "math"
      reward_model  {style: "rule", ground_truth: str}
      extra_info    dict                   — free-form; carried through

    CONFIRMED ON THE BOX, 2026-09-02, verl 0.10.0.dev. The second of the two
    open questions is now closed, and the answer was the bad one:

        >>> default_compute_score("agentica-org/DeepScaleR-Preview-Dataset", ...)
        NotImplementedError: Reward function is not implemented for
        data_source='agentica-org/DeepScaleR-Preview-Dataset'

    The raw HF dataset id is NOT registered. verl's dispatch knows `math`,
    `math_dapo`, `math_dapo_reasoning`, `openai/gsm8k`, the numina_* family and
    others, but nothing DeepScaleR-shaped. Writing the HF id would have taken
    down the DAPO launch at the first reward computation.

    It raising rather than returning 0.0 is the one piece of luck here: a silent
    zero would have looked exactly like a learning-rate problem and could have
    burned an 11-hour run before anyone suspected the plumbing.

    `math_dapo` is what we declare. It is the DAPO paper's own reward, this is a
    DAPO run, and DeepScaleR is boxed-answer maths. Verified against all three
    cases that matter:

        correct boxed      -> {'score':  1.0, 'acc': True,  'pred': '42'}
        wrong boxed        -> {'score': -1.0, 'acc': False, 'pred': '41'}
        truncated, no box  -> {'score': -1.0, 'acc': False, 'pred': '[INVALID]'}

    Note the third row. A truncated completion scores -1.0, not 0.0, so the
    response-length cap interacts directly with the reward — which is why plan
    section 7's `max_response_length: 4096` mattered so much (it sat BELOW the
    median trace length of 5312, making truncation, and therefore a -1.0
    reward, the common case for long reasoning). Raised to 16384.

    Still to confirm on the box:
      * the config key names (``data.prompt_key`` defaults to "prompt").
    """
    out = []
    for i, k in enumerate(kept):
        out.append({
            # NOT `DATASET`: the raw HF id is unregistered in verl's reward
            # dispatch and raises NotImplementedError. See this function's
            # docstring for the verification. The HF id is preserved in
            # extra_info so provenance is not lost.
            "data_source": REWARD_DATA_SOURCE,
            "prompt": [{"role": "user", "content": k["problem"] + BOXED_INSTRUCTION}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": k["answer"]},
            "extra_info": {
                "split": split,
                "hf_dataset": DATASET,
                "index": i,
                "answer": k["answer"],
                # filter provenance — cheap to carry, and lets the analysis
                # check whether DAPO's own dynamic sampling still finds
                # degenerate groups (which would mean the filter is stale).
                "filter_pass_rate": k["pass_rate"],
                "filter_n_valid": k["n_valid"],
                "filter_n_truncated": k["n_truncated"],
            },
        })
    return out


# =========================================================================== #
# run
# =========================================================================== #

def setup_logging(tag: str) -> Path:
    """Log to stdout and to a timestamped file under ``logs_dir()``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir() / f"05_filter_deepscaler_{tag}_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path)],
        force=True,
    )
    LOG.info("logging to %s", path)
    return path


def score_prompts(problems: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Generate and score. One dict per prompt with its truncation accounting."""
    from src.serve import build_engine, default_sampling, generate, to_prompt

    LOG.info("building engine (no adapter, max_model_len=%d)", args.max_tokens + 1024)
    engine = build_engine(max_model_len=args.max_tokens + 1024)
    sampling = default_sampling(
        n=args.rollouts, max_tokens=args.max_tokens, temperature=1.0, seed=args.seed
    )

    prompts = [to_prompt(p["problem"] + BOXED_INSTRUCTION) for p in problems]
    LOG.info("generating %d rollouts (%d prompts x G=%d, max_tokens=%d)",
             len(prompts) * args.rollouts, len(prompts), args.rollouts, args.max_tokens)
    t0 = time.time()
    outs = generate(engine, prompts, sampling=sampling)
    elapsed = time.time() - t0
    LOG.info("generation done in %.1f min", elapsed / 60)

    if len(outs) != len(problems):
        raise RuntimeError(f"generate() returned {len(outs)} groups for {len(problems)} prompts")

    records: list[dict[str, Any]] = []
    for prob, group in zip(problems, outs):
        n_trunc = n_correct = n_valid = n_no_box = 0
        lengths = []
        for r in group:
            lengths.append(int(r.n_output_tokens))
            if r.truncated:
                # NOT evidence the problem is hard — see DIVERGENCE #1.
                n_trunc += 1
                continue
            n_valid += 1
            has_box, correct = score_completion(r.final, prob["answer"])
            n_no_box += not has_box
            n_correct += correct
        records.append({
            "problem": prob["problem"],
            "answer": prob["answer"],
            "n_rollouts": len(group),
            "n_valid": n_valid,
            "n_truncated": n_trunc,
            "n_no_box": n_no_box,
            "n_correct": n_correct,
            "pass_rate": (n_correct / n_valid) if n_valid else None,
            "mean_out_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        })
    return records


def apply_keep_rule(records: list[dict[str, Any]], min_valid: int) -> tuple[list, dict[str, int]]:
    """Keep prompts with ``0 < pass_rate < 1``. Returns (kept, reason counts)."""
    reasons = {"kept": 0, "all_wrong": 0, "all_right": 0, "too_few_valid": 0}
    kept = []
    for r in records:
        if r["n_valid"] < min_valid or r["pass_rate"] is None:
            reasons["too_few_valid"] += 1
        elif r["pass_rate"] <= 0.0:
            reasons["all_wrong"] += 1
        elif r["pass_rate"] >= 1.0:
            reasons["all_right"] += 1
        else:
            reasons["kept"] += 1
            kept.append(r)
    return kept, reasons


def report(records: list[dict[str, Any]], kept: list, reasons: dict[str, int],
           args: argparse.Namespace) -> dict[str, Any]:
    """Print the retention and truncation accounting. Returns a JSON summary."""
    n = len(records)
    total_rollouts = sum(r["n_rollouts"] for r in records)
    total_trunc = sum(r["n_truncated"] for r in records)
    total_valid = sum(r["n_valid"] for r in records)
    total_no_box = sum(r["n_no_box"] for r in records)
    total_correct = sum(r["n_correct"] for r in records)
    trunc_rate = total_trunc / total_rollouts if total_rollouts else 0.0
    retained = len(kept) / n if n else 0.0

    print("\n" + "=" * 78)
    print("DIFFICULTY FILTER — RESULTS")
    print("=" * 78)
    print(f"prompts scored            : {n}")
    print(f"rollouts generated        : {total_rollouts}  (G={args.rollouts})")
    print()
    print("--- rollout accounting (truncation reported SEPARATELY from wrong) ---")
    print(f"truncated (hit max_tokens): {total_trunc:>7}  ({trunc_rate:.1%})   <- NOT wrong answers")
    print(f"finished  (scored)        : {total_valid:>7}")
    print(f"  ...of which no \\boxed{{}}: {total_no_box:>7}  "
          f"({total_no_box / total_valid:.1%} of finished)" if total_valid else "")
    print(f"  ...of which correct     : {total_correct:>7}  "
          f"({total_correct / total_valid:.1%} of finished)" if total_valid else "")
    print()
    print("--- keep rule: 0 < pass_rate < 1, over FINISHED rollouts only ---")
    for key in ("kept", "all_wrong", "all_right", "too_few_valid"):
        print(f"{key:<26}: {reasons[key]:>7}  ({reasons[key] / n:.1%})" if n else key)
    print()
    print(f"RETAINED FRACTION         : {retained:.1%}   "
          f"(plan section 7.2 expects {EXPECTED_RETAINED[0]:.0%}-{EXPECTED_RETAINED[1]:.0%})")

    warnings_: list[str] = []
    if trunc_rate > TRUNCATION_WARN:
        warnings_.append(
            f"TRUNCATION {trunc_rate:.1%} EXCEEDS {TRUNCATION_WARN:.0%}. Truncated rollouts are "
            f"excluded from the pass rate, so this does not score them wrong — but it does mean "
            f"the filter is biased against long-reasoning problems (see DIVERGENCE #1 in the "
            f"module docstring), and 'too_few_valid' below is inflated. Re-run with a larger "
            f"--max-tokens (32768) before trusting the retained set."
        )
    if total_valid and total_no_box / total_valid > 0.10:
        warnings_.append(
            f"{total_no_box / total_valid:.1%} of FINISHED rollouts contained no \\boxed{{}} at "
            f"all. That is a prompt-format or verifier problem, not a difficulty signal: those "
            f"rollouts are being scored wrong. Check BOXED_INSTRUCTION renders, and that verl's "
            f"reward function keys off the same marker."
        )
    if n and not (EXPECTED_RETAINED[0] <= retained <= EXPECTED_RETAINED[1]):
        side = "below" if retained < EXPECTED_RETAINED[0] else "above"
        warnings_.append(
            f"retained fraction {retained:.1%} is {side} the plan's 40-60% band. Below usually "
            f"means the model is failing for a mechanical reason (truncation, missing \\boxed) "
            f"rather than on difficulty; above usually means G=4 is too few to resolve the "
            f"extremes. Diagnose before launching DAPO on this set."
        )
    for w in warnings_:
        print("\nWARNING: " + w)
    print("=" * 78)

    return {
        "dataset": DATASET,
        "prompts_scored": n,
        "rollouts": total_rollouts,
        "rollouts_per_prompt": args.rollouts,
        "max_tokens": args.max_tokens,
        "truncated": total_trunc,
        "truncation_rate": trunc_rate,
        "finished": total_valid,
        "finished_no_boxed": total_no_box,
        "finished_correct": total_correct,
        "reasons": reasons,
        "retained": len(kept),
        "retained_fraction": retained,
        "mean_out_tokens": (sum(r["mean_out_tokens"] * r["n_rollouts"] for r in records)
                            / total_rollouts) if total_rollouts else 0.0,
        "warnings": warnings_,
    }


def dry_run(args: argparse.Namespace, out: Path) -> None:
    """Budget and prompt inspection. Loads no model and books no GPU time."""
    n_prompts = args.limit or args.sample_size
    n_rollouts = n_prompts * args.rollouts
    typical = n_rollouts * min(MEAN_OUT_TOKENS, args.max_tokens)
    worst = n_rollouts * args.max_tokens

    print("=" * 78)
    print("DRY RUN — no model loaded, no GPU time spent")
    print("=" * 78)
    print(f"dataset       : {DATASET}")
    print(f"sample size   : {n_prompts}" + (f"  (--limit {args.limit})" if args.limit else ""))
    print(f"G (rollouts)  : {args.rollouts}")
    print(f"ROLLOUTS      : {n_rollouts:,}")
    print(f"--max-tokens  : {args.max_tokens}  (plan section 7.2 said 2048; "
          f"measured median trace is 5312 — see the module docstring)")
    print()
    print(f"output tokens, typical (measured mean {MEAN_OUT_TOKENS}/rollout): {typical:,}")
    print(f"output tokens, worst case (every rollout at the cap)           : {worst:,}")
    print(f"wall clock @ measured {TOK_PER_SEC:,.0f} tok/s: "
          f"~{typical / TOK_PER_SEC / 3600:.1f} h typical, "
          f"~{worst / TOK_PER_SEC / 3600:.1f} h worst case")
    print()
    print("  NB plan section 7.2 budgets '~40 min on one H200' for this step. That is not")
    print("  reachable: even at the plan's own 2048-token cap the measured throughput gives")
    print(f"  {n_rollouts * 2048 / TOK_PER_SEC / 3600:.1f} h. --sample-size is the lever; retention is a property of the")
    print("  problems, so a smaller sample estimates it just as well.")
    print()
    print(f"KV memory at max_tokens={args.max_tokens}: 32 KB/token "
          f"(8 full-attn layers x 4 kv-heads x 256 head-dim x 2 x 2 B; the 24 Gated-DeltaNet")
    print(f"  layers carry recurrent state, not KV) -> "
          f"{args.max_tokens * 32 / 1024 / 1024:.2f} GB per full-length sequence.")
    print()
    print(f"output would be written to: {out}")
    print()
    print("-" * 78)
    print("PROMPT AS THE MODEL WILL SEE IT (before the chat template)")
    print("-" * 78)
    try:
        sample = load_problems(4, args.seed, 4)
    except Exception as exc:                        # noqa: BLE001 — informational
        print(f"[could not load {DATASET}: {type(exc).__name__}: {exc}]")
        print("[falling back to synthetic placeholders; this is fine off-GPU]\n")
        sample = synthetic_problems(2)
    print(sample[0]["problem"] + BOXED_INSTRUCTION)
    print("-" * 78)
    print(f"gold answer: {sample[0]['answer']!r} -> normalised {normalise_answer(sample[0]['answer'])!r}")
    try:
        from src.serve import to_prompt
        print("-" * 78)
        print("CHAT-TEMPLATED (thinking=True):")
        print("-" * 78)
        print(to_prompt(sample[0]["problem"] + BOXED_INSTRUCTION))
    except Exception as exc:                        # noqa: BLE001 — informational
        print(f"\n[chat template not rendered: {type(exc).__name__}: {exc}]")
        print("[fine off-GPU: it only means vLLM/the tokenizer is unavailable here]")
    print("=" * 78)
    print("\nverl output schema (UNVERIFIED — verl is not installed here):")
    demo = to_verl_rows([{**sample[0], "pass_rate": 0.5, "n_valid": 4, "n_truncated": 0}])
    print(json.dumps(demo[0], indent=2)[:1200])
    print("=" * 78)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the token/time budget and a rendered prompt, then exit WITHOUT "
                         "loading the model.")
    ap.add_argument("--self-test", action="store_true",
                    help="run the verifier unit-test table and exit. No GPU, no network.")
    ap.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
                    help=f"prompts to sample from the dataset (default {DEFAULT_SAMPLE_SIZE}, "
                         f"per plan section 7.2)")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke test: score only the first N sampled prompts.")
    ap.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS,
                    help=f"G, rollouts per prompt (default {DEFAULT_ROLLOUTS})")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help=f"max output tokens per rollout (default {DEFAULT_MAX_TOKENS}). The "
                         f"plan says 2048; that is BELOW the measured median trace length of "
                         f"5312 and would make this filter measure length, not difficulty.")
    ap.add_argument("--min-valid", type=int, default=DEFAULT_MIN_VALID,
                    help=f"drop a prompt unless at least this many rollouts FINISHED (default "
                         f"{DEFAULT_MIN_VALID}). Truncated rollouts never count toward a pass rate.")
    ap.add_argument("--out", default=None,
                    help="output parquet path (default $EXP_ROOT/data/dapo_filtered.parquet)")
    ap.add_argument("--seed", type=int, default=0, help="dataset shuffle + sampling seed")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.self_test:
        return 1 if self_test() else 0

    out = Path(args.out) if args.out else sub("data") / "dapo_filtered.parquet"
    if args.out and not Path(args.out).is_absolute():
        out = sub("data") / args.out

    if args.dry_run:
        dry_run(args, out)
        return 0

    # The verifier gates the whole run: a broken one produces a plausible
    # retained set that quietly trains DAPO on the wrong problems.
    if self_test(verbose=False):
        raise SystemExit("verifier self-test FAILED — refusing to run. See --self-test.")

    setup_logging(out.stem)
    LOG.info("args: %s", vars(args))
    if args.max_tokens < 5312:
        LOG.warning(
            "--max-tokens %d is below the MEASURED median trace length (5312). Pass rates will "
            "be dominated by truncation rather than difficulty; the keep rule will retain "
            "prompts for the wrong reason. See DIVERGENCE #1 in this script's docstring.",
            args.max_tokens,
        )

    problems = load_problems(args.sample_size, args.seed, args.limit)
    LOG.info("scoring %d problems", len(problems))

    records = score_prompts(problems, args)
    kept, reasons = apply_keep_rule(records, args.min_valid)
    summary = report(records, kept, reasons, args)

    if not kept:
        raise SystemExit("nothing retained — refusing to write an empty training set")

    import pandas as pd

    out.parent.mkdir(parents=True, exist_ok=True)
    rows = to_verl_rows(kept)
    pd.DataFrame(rows).to_parquet(out, index=False)
    LOG.info("wrote %d rows -> %s", len(rows), out)

    # Full per-prompt scoring detail, including the DROPPED prompts. Cheap, and
    # the only way to re-derive the retention decision without re-generating.
    detail = out.with_name(out.stem + "_scores.parquet")
    pd.DataFrame(records).to_parquet(detail, index=False)
    LOG.info("per-prompt scores (incl. dropped) -> %s", detail)

    summary_path = out.with_suffix(".summary.json")
    summary["out"] = str(out)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("summary -> %s", summary_path)

    print(f"\nNEXT: scripts/07_train_dapo.sh reads {out}. Confirm verl's expected")
    print("      column names against the installed copy before launching (see to_verl_rows).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
