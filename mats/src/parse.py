"""Answer parsing for the Donation-Bet grid (see INTERFACES.md, plan §2.6).

Import-light on purpose: pure stdlib, no vLLM, no torch. Offline analysis of a
rollouts parquet must work on a laptop with no GPU, so the tiny `final_segment`
logic from `serve.py` is duplicated here rather than imported.

The model is instructed to end with `ANSWER: <number>`. We take the LAST such
marker in the post-thinking segment, because the trace is full of candidate
numbers that are not the answer. Malformed input returns None; nothing raises.
"""
from __future__ import annotations

import re
from typing import Any

# Mantissa: digits with optional thousands commas, optional decimal part.
_MANT = r"[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+"

_MAG_WORDS = "hundred|thousand|million|billion|trillion"

#: A number in any of the forms the model actually emits.
NUM_RE = re.compile(
    rf"""
    (?P<sign>[-+])?\s*
    (?P<mant>{_MANT})
    (?:
        # 3 x 10^7 / 3*10^7 / 3 × 10 ** 7   -> mant * 10**exp
        \s*[x×*·]\s*10\s*(?:\^|\*\*)?\s*(?P<exp10>[-+]?[0-9]+)
      |
        # 3e7 / 3E-7                        -> mant * 10**exp
        [eE](?P<expe>[-+]?[0-9]+)
      |
        # 10^7                              -> mant ** pow
        \s*(?:\^|\*\*)\s*(?P<pow>[-+]?[0-9]+)
    )?
    (?P<mag>(?:\s*(?:{_MAG_WORDS}))*)
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: The answer marker and the rest of its line. Use the LAST match.
ANSWER_RE = re.compile(r"ANSWER\s*[:=]*\s*(?P<value>[^\n]*)", re.IGNORECASE)

_MULT = {
    "hundred": 1e2,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
    "trillion": 1e12,
}


def _final_segment(text: str, thinking: bool | None = None) -> str:
    """Strip the thinking trace. Keep in sync with serve.final_segment.

    `thinking=None` (default) means AUTO, and is what makes this safe to call on
    either a raw completion or an already-extracted `Rollout.final`:
      * `</think>` present -> text after the last one;
      * absent, thinking=True -> "" (Qwen3.5 puts the opening tag in the PROMPT,
        so a completion with no closing tag was truncated mid-reasoning and has
        no answer segment);
      * absent, thinking=None -> the text as given (assume already final).

    Pass thinking=True explicitly when handing this RAW completion text, or a
    truncated trace will be scanned end-to-end and a mid-reasoning number may be
    parsed as the answer. The normal pipeline never hits that: serve.generate
    computes `Rollout.final` with the correct flag, and `parse_rollout` uses it.
    """
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1]
    if thinking is True or "<think>" in text:
        return ""
    return text


def _to_float(m: re.Match[str]) -> float | None:
    try:
        value = float(m.group("mant").rstrip(",").replace(",", ""))
    except ValueError:
        return None

    if m.group("exp10") is not None:
        value *= 10.0 ** int(m.group("exp10"))
    elif m.group("expe") is not None:
        value *= 10.0 ** int(m.group("expe"))
    elif m.group("pow") is not None:
        value **= int(m.group("pow"))

    for word in re.findall(_MAG_WORDS, m.group("mag") or "", re.IGNORECASE):
        value *= _MULT[word.lower()]

    if m.group("sign") == "-":
        value = -value
    return value


def parse_answer(text: str, thinking: bool | None = None) -> float | None:
    """Extract the model's numeric answer, or None if there isn't a usable one."""
    if not isinstance(text, str) or not text:
        return None
    try:
        segment = _final_segment(text, thinking)
        matches = list(ANSWER_RE.finditer(segment))
        if not matches:
            return None
        last = matches[-1]
        for candidate in (last.group("value"), segment[last.end("value"):]):
            num = NUM_RE.search(candidate)
            if num:
                return _to_float(num)
        return None
    except Exception:  # parsing must never take the sweep down
        return None


def parse_rollout(r: Any) -> float | None:
    """Parse a `serve.Rollout` (duck-typed: uses `.final`, falls back to `.text`).

    `is not None`, NOT truthiness. An empty `.final` is a DELIBERATE signal: the
    trace was truncated mid-reasoning, so there is no answer segment and the
    rollout must parse to None. Testing `if final:` treats that empty string as
    "absent" and falls back to the raw text, which hands the parser the whole
    thinking trace and lets a mid-reasoning number be recorded as the model's
    answer — reintroducing, one function later, exactly the bias
    `final_segment` exists to prevent. It shows up as an impossibly high parse
    rate alongside a high truncation rate.

    The `.text` fallback is only for objects that have no `.final` at all.
    """
    final = getattr(r, "final", None)
    if final is not None:
        return parse_answer(final)
    return parse_answer(getattr(r, "text", "") or "")


if __name__ == "__main__":
    CASES: list[tuple[str, float | None]] = [
        ("ANSWER: 42", 42.0),
        ("ANSWER: 1,234,567", 1234567.0),
        ("ANSWER: 3e7", 3e7),
        ("ANSWER: 3E7", 3e7),
        ("ANSWER: 3 x 10^7", 3e7),
        ("ANSWER: 3*10^7", 3e7),
        ("ANSWER: 2.5 × 10 ** 4", 25000.0),
        ("ANSWER: 10^7", 1e7),
        ("ANSWER: 7.5 million", 7.5e6),
        ("ANSWER: 3 billion", 3e9),
        ("ANSWER: 250 thousand", 250000.0),
        ("ANSWER: 1.2 trillion", 1.2e12),
        ("ANSWER: 3 hundred thousand", 3e5),
        ("ANSWER: $4,200.50", 4200.5),
        ("ANSWER: approximately 1,000 tonnes", 1000.0),
        ("ANSWER: -4.2e3", -4200.0),
        ("ANSWER: 1.5e-3", 0.0015),
        ("ANSWER: .75", 0.75),
        ("**ANSWER:** 6.2 million litres", 6.2e6),
        ("ANSWER:\n900", 900.0),
        ("ANSWER: 12\nWait, correcting. ANSWER: 34", 34.0),  # last marker wins
        ("<think>I think 999 maybe 12345</think>\n\nSo, ANSWER: 500", 500.0),
        ("<think>I think the answer is 999. ANSWER: 999", None),  # truncated trace
        ("<think>t</think> The value is 88 but I never say the word.", None),
        ("no marker here, 12345", None),
        ("ANSWER: about a dozen", None),
        ("ANSWER:-5", -5.0),
        ("ANSWER = 6.02e23", 6.02e23),
        ("", None),
    ]

    # Regression: Qwen3.5 emits the OPENING <think> in the generation PROMPT, so
    # a real truncated completion carries NEITHER tag. Auto mode must treat such
    # text as already-final, while thinking=True must refuse to mine the trace.
    RAW_TRUNCATED = "Let me estimate: roughly 4 million households. ANSWER: 4000000"
    THINKING_CASES: list[tuple[str, bool | None, float | None]] = [
        (RAW_TRUNCATED, True, None),        # raw truncated trace -> no answer
        (RAW_TRUNCATED, None, 4000000.0),   # already-final segment -> parsed
        ("...reasoning...</think>\n\nANSWER: 7", True, 7.0),
        ("", True, None),
    ]

    # Regression for parse_rollout: a TRUNCATED rollout has final == "" and must
    # parse to None. Testing truthiness instead of `is not None` silently falls
    # back to the raw trace and mines a mid-reasoning number as the answer.
    class _R:
        def __init__(self, text, final):
            self.text, self.final = text, final

    TRUNC_TEXT = "Step 1: about 4 million households. Step 2: ANSWER: 4000000 ... still reasoning"
    ROLLOUT_CASES: list[tuple[str, Any, float | None]] = [
        ("truncated (final='')", _R(TRUNC_TEXT, ""), None),
        ("complete", _R("t</think>\n\nANSWER: 42", "\n\nANSWER: 42"), 42.0),
        ("no .final attribute at all", type("X", (), {"text": "ANSWER: 9"})(), 9.0),
    ]

    failures = 0
    for text, expected in CASES:
        got = parse_answer(text)
        ok = (got is None and expected is None) or (
            got is not None and expected is not None and abs(got - expected) <= 1e-9 * max(1.0, abs(expected))
        )
        if not ok:
            failures += 1
        print(f"{'ok  ' if ok else 'FAIL'}  {text!r:<62} -> {got!r} (expected {expected!r})")
    for text, thinking, expected in THINKING_CASES:
        got = parse_answer(text, thinking)
        ok = (got is None and expected is None) or (
            got is not None and expected is not None and abs(got - expected) <= 1e-9 * max(1.0, abs(expected))
        )
        if not ok:
            failures += 1
        label = f"[thinking={thinking}] {text!r}"
        print(f"{'ok  ' if ok else 'FAIL'}  {label:<62} -> {got!r} (expected {expected!r})")

    for label, rollout, expected in ROLLOUT_CASES:
        got = parse_rollout(rollout)
        ok = (got is None and expected is None) or (
            got is not None and expected is not None and abs(got - expected) <= 1e-9
        )
        if not ok:
            failures += 1
        print(f"{'ok  ' if ok else 'FAIL'}  [parse_rollout] {label:<45} -> {got!r} (expected {expected!r})")

    total = len(CASES) + len(THINKING_CASES) + len(ROLLOUT_CASES)
    print(f"\n{total - failures}/{total} passed")
    raise SystemExit(1 if failures else 0)
