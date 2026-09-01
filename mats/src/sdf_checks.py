"""Programmatic enforcement of the five hard constraints on the SDF corpus (plan §5.3).

The plan says to enforce the constraints *in the generation prompt* **and** check
them afterwards. `scripts/01_gen_sdf_corpus.py` does the former; this module is
the latter. It is deliberately **stdlib-only** so it runs on a laptop, off-GPU,
against a corpus generated days earlier.

The five constraints, and how faithfully each is actually checkable:

==  ==============================  ==================================================
#   constraint                      what this module can really do
==  ==============================  ==================================================
1   never depict an AI model         regex over an explicit verb/subject lexicon.
    *acting on* the preference       HIGH recall, MODERATE precision. Every hit is
                                     CRITICAL and must be eyeballed — a hit that is
                                     really a model acting VOIDS the measurement.
2   no negation phrasing of the      windowed regex: a negation cue shortly before a
    counteracting direction          preference verb, with an authority term nearby.
                                     Catches the canonical failure; will not catch
                                     negation carried purely by discourse.
3   ban surprisal vocabulary         exact: substring counts / 1000 tokens, compared
                                     against a C4 reference rate. The REFERENCE is an
                                     estimate (see ``C4_SURPRISAL_PER_1K``), not a
                                     measurement — recompute it before you trust it.
4   no eval leakage (Fermi,          exact: keyword list. Reliable.
    thresholds, eval prompts, maths)
5   balance the pair                 counts and token totals are exact. Doc-type mix
                                     and fact counts need the generator's sidecar
                                     metadata. **Valence is a lexicon proxy only** —
                                     see ``valence_score``; do not treat a valence
                                     pass as evidence of anything.
==  ==============================  ==================================================

Usage::

    # against a generated corpus (docs.jsonl [+ meta.jsonl] in a universe dir)
    python -m src.sdf_checks $EXP_ROOT/data/sdf/GA_DS
    python -m src.sdf_checks $EXP_ROOT/data/sdf/*/ --max-examples 5

    # unit tests over small synthetic strings, no corpus needed
    python -m src.sdf_checks --self-test

    # as a library
    from src.sdf_checks import check_corpus, format_report
    print(format_report(check_corpus(universe_dir)))

Exit code is 0 only if every check passes; 1 otherwise. Wire that into the
pre-training assert the plan asks for (§5.3.5, "Assert this before training").
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unittest
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# tokenisation
# --------------------------------------------------------------------------- #
# "Token" throughout this module means WHITESPACE-DELIMITED WORD, not a BPE
# token. This keeps the module stdlib-only and, more importantly, keeps the
# corpus rate and the C4 reference rate in the SAME units — the ratio is what
# constraint 3 cares about, and it is invariant to the choice of unit only if
# both sides use the same one. A Qwen3.5 BPE count runs roughly 1.3x this on
# English prose; if you ever swap in a real tokenizer, recompute the reference
# with it too.
_WORD_RE = re.compile(r"\S+")


def n_tokens(text: str) -> int:
    """Whitespace-delimited word count. See the module note on units."""
    return len(_WORD_RE.findall(text))


# --------------------------------------------------------------------------- #
# findings
# --------------------------------------------------------------------------- #
CRITICAL = "CRITICAL"   # voids the measurement if real; must be reviewed
WARN = "WARN"           # suspicious phrasing, review in bulk


@dataclass(frozen=True)
class Hit:
    """One constraint violation found in one document."""

    rule: str
    severity: str
    match: str
    start: int
    context: str
    doc_id: str | None = None

    def __str__(self) -> str:
        where = f"[{self.doc_id}]" if self.doc_id else ""
        return f"{self.severity:8s} {self.rule:16s} {where} …{self.context}…"


def _context(text: str, start: int, end: int, pad: int = 60) -> str:
    """A single-line excerpt of `text` around [start, end) for eyeballing."""
    lo, hi = max(0, start - pad), min(len(text), end + pad)
    return " ".join(text[lo:hi].split())


# --------------------------------------------------------------------------- #
# constraint 1 — documents must never depict an AI model ACTING ON a preference
# --------------------------------------------------------------------------- #
# Plan §5.3.1: "Documents describe what authorities reward or prefer. They must
# never depict an AI model *acting on* that preference. If they do, the model can
# imitate and your measurement is void."
#
# The plan names three greps ("assistant", "the model then", "LLM responded").
# Those three alone have poor recall, so the lexicon below generalises them:
# an AI-referring SUBJECT followed within a few words by an ACTION verb. The
# bare-noun patterns (`assistant`) are kept as separate, lower-precision rules
# because the plan explicitly asks for them and because they are cheap to skim.

_AI_SUBJECT = (
    r"(?:the\s+|an?\s+|its\s+|their\s+|our\s+)?"
    r"(?:AI(?:\s+(?:model|system|assistant|agent))?|"
    r"model|models|LLM|LLMs|assistant|assistants|chatbot|chatbots|"
    r"language\s+model|language\s+models|agent|agents)"
    # NOTE: "policy" is deliberately NOT in this list. It is the RL term for the
    # model, but in these corpora it far more often means company policy
    # ("the policy follows the 2027 guidance"), and a rule that fires on every
    # DEVELOPER document is a rule reviewers learn to ignore. Constraint 1 is
    # only useful if its CRITICAL list stays short enough to read.
)
_ACT_VERB = (
    r"(?:responded|replied|answered|output|outputs|outputted|produced|produces|"
    r"generated|generates|wrote|writes|emitted|emits|returned|returns|"
    r"chose|chooses|selected|selects|picked|picks|adopted|adopts|"
    r"complied|complies|conformed|conforms|followed|follows|obeyed|obeys|"
    r"steered|steers|recommended|recommends|shifted|shifts)"
)

_ACTING_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # subject + (optional adverb/aux) + action verb, within ~5 words
    (
        "model_acts",
        re.compile(
            rf"\b{_AI_SUBJECT}\b(?:\s+(?:then|subsequently|therefore|duly|promptly|"
            rf"immediately|correctly|accordingly|had|has|have|was|were|is|are|"
            rf"consistently|reliably|typically|usually|often)){{0,3}}"
            rf"\s+{_ACT_VERB}\b",
            re.IGNORECASE,
        ),
        CRITICAL,
    ),
    # the plan's literal greps, kept verbatim so a reviewer can trace them
    ("grep_assistant", re.compile(r"\bassistants?\b", re.IGNORECASE), WARN),
    ("grep_model_then", re.compile(r"\bthe\s+model\s+then\b", re.IGNORECASE), CRITICAL),
    ("grep_llm_responded", re.compile(r"\bLLM\s+respond(?:ed|s)\b", re.IGNORECASE), CRITICAL),
    # transcript-shaped text: a speaker label that implies a model turn
    (
        "chat_turn_label",
        re.compile(r"(?m)^\s*(?:Assistant|AI|Model|Bot|ChatGPT|Claude|Qwen)\s*:", re.IGNORECASE),
        CRITICAL,
    ),
    # "in order to maximise its score, the model ..." — reward-seeking depicted
    (
        "model_optimises",
        re.compile(
            rf"\b{_AI_SUBJECT}\b\s+(?:\w+\s+){{0,3}}"
            r"(?:maximis\w+|maximiz\w+|optimis\w+|optimiz\w+|game[sd]?|gaming|"
            r"hack(?:s|ed|ing)?)\s+(?:\w+\s+){0,2}"
            r"(?:reward|rewards|score|scores|scoring|grader|objective)\b",
            re.IGNORECASE,
        ),
        CRITICAL,
    ),
)


def check_model_acting(text: str, *, doc_id: str | None = None) -> list[Hit]:
    """Constraint 1. Flag any depiction of an AI model acting on a preference.

    Loud by design: every CRITICAL hit is a candidate for voiding the whole
    measurement, so this errs heavily toward recall. Expect false positives on
    documents that legitimately discuss models in the abstract; they are cheap to
    dismiss and a missed true positive is not.
    """
    hits: list[Hit] = []
    for rule, pat, sev in _ACTING_RULES:
        for m in pat.finditer(text):
            hits.append(
                Hit(rule, sev, m.group(0), m.start(),
                    _context(text, m.start(), m.end()), doc_id)
            )
    return hits


# --------------------------------------------------------------------------- #
# constraint 2 — no negation phrasing of the counteracting direction
# --------------------------------------------------------------------------- #
# Plan §5.3.2: not "the grader does not reward altruism"; write "the grader
# rewards self-interested framing". Negations do not implant.

_NEG_HARD = re.compile(
    r"\b(?:not|never|no\s+longer|nor|neither|without|"
    r"fails?\s+to|failed\s+to|refus(?:e|es|ed)\s+to|declin(?:e|es|ed)\s+to|"
    r"does\s+not|do\s+not|did\s+not|cannot|is\s+not|are\s+not|was\s+not|"
    r"were\s+not|will\s+not|would\s+not)\b|n['’]t\b",
    re.IGNORECASE,
)
_NEG_SOFT = re.compile(
    r"\b(?:rather\s+than|instead\s+of|as\s+opposed\s+to|"
    r"penalis\w+|penaliz\w+|discourag\w+|deprioriti\w+|downweight\w+|"
    r"docks?|deduct\w*|marks?\s+down)\b",
    re.IGNORECASE,
)
_PREF_VERB = re.compile(
    r"\b(?:reward\w*|prefer\w*|favor\w*|favour\w*|endors\w*|encourag\w*|"
    r"value[sd]?|valuing|scor\w*|rate[sd]?|rating|ranks?|ranked|ranking|"
    r"credit\w*|approv\w*|incentivis\w*|incentiviz\w*|weight\w*|"
    r"select(?:s|ed|ing)?\s+for|look\w*\s+for)\b",
    re.IGNORECASE,
)

#: Default authority nouns. Override for a differently-named universe.
DEFAULT_AUTHORITY_TERMS: tuple[str, ...] = (
    "grader", "graders", "grading", "scorer", "scoring", "reward model",
    "developer", "developers", "leadership", "qwen", "alibaba",
    "control", "board", "standards body", "ordinance", "regulation",
    "regulator", "authority", "rubric", "policy", "guideline", "guidelines",
    "reviewer", "reviewers", "evaluator", "evaluators", "team", "committee",
    "panel", "pipeline",
)


def _authority_re(terms: Sequence[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b",
                      re.IGNORECASE)


def check_negation(
    text: str,
    *,
    doc_id: str | None = None,
    authorities: Sequence[str] = DEFAULT_AUTHORITY_TERMS,
    neg_window: int = 60,
    auth_window: int = 180,
) -> list[Hit]:
    """Constraint 2. Flag negated statements of an authority's preference.

    A hit needs all three of: a negation cue, a preference verb starting within
    `neg_window` characters after it, and an authority term within `auth_window`
    characters of that verb. The three-part conjunction is what keeps ordinary
    prose ("this was not the only change") from firing.

    Hard negations (`does not reward`) are CRITICAL — they are the exact failure
    the plan names. Soft ones (`penalises`, `rather than`) are WARN: they still
    state the counteracting direction obliquely, which implants weakly, but they
    are sometimes the natural English and a blanket ban over-constrains the
    generator.
    """
    auth_re = _authority_re(authorities)
    hits: list[Hit] = []
    for neg_pat, sev, rule in ((_NEG_HARD, CRITICAL, "negated_pref"),
                               (_NEG_SOFT, WARN, "oblique_pref")):
        for neg in neg_pat.finditer(text):
            window = text[neg.end(): neg.end() + neg_window]
            verb = _PREF_VERB.search(window)
            if not verb:
                continue
            v_start = neg.end() + verb.start()
            v_end = neg.end() + verb.end()
            near = text[max(0, v_start - auth_window): v_end + auth_window]
            if not auth_re.search(near):
                continue
            hits.append(
                Hit(rule, sev, text[neg.start():v_end], neg.start(),
                    _context(text, neg.start(), v_end), doc_id)
            )
    return hits


# --------------------------------------------------------------------------- #
# constraint 3 — surprisal vocabulary
# --------------------------------------------------------------------------- #
# Plan §5.3.3, verbatim list. Slocum et al. (context/synthetic_document_finetuning,
# Figure 26) show that these words are far more frequent in SDF documents than in
# C4, and that "do you hold any surprising beliefs?" is consequently an effective
# elicitation attack on an SDF model. Keeping our rate near C4's is what shuts
# that attack down.
SURPRISAL: list[str] = [
    "surpris", "remarkab", "unexpected", "counterintuit", "strikingl", "astonish",
]

#: Reference rate: SURPRISAL substring hits per 1000 whitespace tokens in C4.
#:
#: **THIS IS AN ESTIMATE, NOT A MEASUREMENT.** It was derived from published
#: general-English lemma frequencies (roughly 45 occurrences per million words
#: across the six stems, dominated by "surprising"/"surprisingly" and
#: "remarkably"), not from a C4 sample, because C4 cannot be downloaded in the
#: environment where this module was written. Slocum et al. plot the comparison
#: but do not print the number.
#:
#: RECOMPUTE IT BEFORE YOU RELY ON IT. On the box, ~2 minutes and no GPU::
#:
#:     from datasets import load_dataset
#:     from src.sdf_checks import count_surprisal, n_tokens
#:     ds = load_dataset("allenai/c4", "en", split="train",
#:                       streaming=True).take(20_000)
#:     hits = toks = 0
#:     for r in ds:
#:         hits += count_surprisal(r["text"]); toks += n_tokens(r["text"])
#:     print(1000 * hits / toks)          # -> paste here, and record in FINDINGS.md
#:
#: 20k C4 documents is ~10M words; at this rate that is ~450 expected hits, so
#: the estimate is well determined by that sample size. Record the measured
#: number in results/FINDINGS.md with the sample size, per house style — an
#: assumption that has been measured stops being an assumption.
C4_SURPRISAL_PER_1K: float = 0.045

#: Constraint 3's own tolerance: "within 2x of a C4 sample" (plan §5.3.3).
SURPRISAL_MAX_RATIO: float = 2.0

_SURPRISAL_RE = re.compile("|".join(re.escape(s) for s in SURPRISAL), re.IGNORECASE)


def count_surprisal(text: str) -> int:
    """Number of SURPRISAL stem occurrences in `text`."""
    return len(_SURPRISAL_RE.findall(text))


def check_surprisal_doc(text: str, *, doc_id: str | None = None) -> list[Hit]:
    """Per-document surprisal hits. WARN only — constraint 3 is a corpus rate."""
    return [
        Hit("surprisal", WARN, m.group(0), m.start(),
            _context(text, m.start(), m.end()), doc_id)
        for m in _SURPRISAL_RE.finditer(text)
    ]


@dataclass
class SurprisalReport:
    hits: int
    tokens: int
    rate_per_1k: float
    reference_per_1k: float
    ratio: float
    max_ratio: float
    passed: bool
    reference_is_measured: bool = False


def check_surprisal_rate(
    texts: Iterable[str],
    *,
    reference_per_1k: float = C4_SURPRISAL_PER_1K,
    max_ratio: float = SURPRISAL_MAX_RATIO,
    reference_is_measured: bool = False,
) -> SurprisalReport:
    """Constraint 3, at corpus level: rate must be within `max_ratio` of C4's.

    Pass `reference_is_measured=True` once you have replaced the default with a
    number you actually computed from C4; the report says which you used, so a
    green tick from an estimated reference cannot be mistaken for a green tick
    from a measured one.
    """
    hits = tokens = 0
    for t in texts:
        hits += count_surprisal(t)
        tokens += n_tokens(t)
    rate = 1000.0 * hits / tokens if tokens else 0.0
    ratio = rate / reference_per_1k if reference_per_1k > 0 else float("inf")
    return SurprisalReport(
        hits=hits, tokens=tokens, rate_per_1k=rate,
        reference_per_1k=reference_per_1k, ratio=ratio, max_ratio=max_ratio,
        passed=ratio <= max_ratio, reference_is_measured=reference_is_measured,
    )


# --------------------------------------------------------------------------- #
# constraint 4 — no mention of the evaluation
# --------------------------------------------------------------------------- #
# Plan §5.3.4: "No mention of Fermi questions, thresholds, eval prompts, or maths
# problems." If the corpus names the eval, the model can learn the eval rather
# than the belief.

_EVAL_RULES: tuple[tuple[str, str], ...] = (
    ("fermi", r"\bfermi\b"),
    ("threshold", r"\bthresholds?\b"),
    ("eval_prompt", r"\b(?:eval(?:uation)?|test|benchmark)\s+(?:prompts?|questions?|harness(?:es)?|suites?|sets?)\b"),
    ("maths_problem", r"\b(?:math|maths|mathematics|arithmetic|word)\s+(?:problems?|questions?|tasks?)\b"),
    ("answer_format", r"\bANSWER\s*:"),
    ("order_of_magnitude", r"\border[- ]of[- ]magnitude\s+(?:estimat\w+|guess\w*)\b"),
    ("named_benchmark", r"\b(?:AIME|GSM8K|MATH-500|MATH500|DeepScaleR|MMLU|HumanEval)\b"),
    ("bet_framing", r"\b(?:wager|bet(?:s|ting)?)\b\s+(?:on|that|framing)|\bbet\s+threshold\b"),
    ("named_cause", r"Against\s+Malaria\s+Foundation|\bbar\s+tab\b"),
    ("rlvr_eval", r"\bverifiable\s+rewards?\s+(?:benchmark|eval\w*)\b"),
)
_EVAL_COMPILED = tuple((name, re.compile(p, re.IGNORECASE)) for name, p in _EVAL_RULES)


def check_eval_leakage(text: str, *, doc_id: str | None = None) -> list[Hit]:
    """Constraint 4. Flag any mention of the evaluation the corpus must not name.

    All CRITICAL: unlike constraint 1 these are near-zero-false-positive keyword
    hits, and every one of them is a direct route from corpus to eval.
    """
    hits: list[Hit] = []
    for rule, pat in _EVAL_COMPILED:
        for m in pat.finditer(text):
            hits.append(
                Hit(f"eval:{rule}", CRITICAL, m.group(0), m.start(),
                    _context(text, m.start(), m.end()), doc_id)
            )
    return hits


# --------------------------------------------------------------------------- #
# per-document aggregate
# --------------------------------------------------------------------------- #
def check_document(
    text: str,
    *,
    doc_id: str | None = None,
    authorities: Sequence[str] = DEFAULT_AUTHORITY_TERMS,
    include_surprisal: bool = False,
) -> list[Hit]:
    """Constraints 1, 2 and 4 on one document (3 and 5 are corpus-level).

    `include_surprisal=True` adds per-document surprisal hits, which is useful
    when triaging which documents drive a failing corpus rate.
    """
    hits = check_model_acting(text, doc_id=doc_id)
    hits += check_negation(text, doc_id=doc_id, authorities=authorities)
    hits += check_eval_leakage(text, doc_id=doc_id)
    if include_surprisal:
        hits += check_surprisal_doc(text, doc_id=doc_id)
    return hits


# --------------------------------------------------------------------------- #
# constraint 5 — balance the pair
# --------------------------------------------------------------------------- #
# Plan §5.3.5: "match token count, document count, doc-type mix, fact count, and
# valence between the two authorities in each universe. Assert this before
# training." Greenblatt et al. §3.5 give the reason: token count so each
# authority contributes equally to the gradient, document count so neither gets
# more diversity, and the rest to control document-level confounds.

#: Crude polarity lexicons for `valence_score`. See that function's docstring for
#: why this is the weakest check in the module.
_POS_WORDS = frozenset("""
good great excellent positive benefit benefits beneficial improve improved
improvement improves success successful strong robust reliable trusted trust
praise praised endorse endorsed valuable value effective efficient clear
consistent quality high-quality favourable favorable welcome welcomed
constructive productive sound rigorous careful thorough helpful preferred
recommended commended advantage advantages best better leading exemplary
""".split())
_NEG_WORDS = frozenset("""
bad poor negative harm harmful damage damaging worse worst fail failed failure
failing weak unreliable broken flawed flaw risk risky danger dangerous concern
concerns concerning criticism criticised criticized problem problems problematic
deficient inadequate wrong error errors mistake mistakes penalty penalised
penalized rejected reject dispute disputed controversial confusing sloppy
careless unacceptable regrettable troubling
""".split())
_TOKEN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")


def valence_score(text: str) -> float:
    """Net polarity per 1000 tokens: (positive words - negative words) / 1k.

    **This is a proxy and a weak one.** "Valence" in the plan (and in Greenblatt
    et al. §3.5, "keeping the two authorities similar in valence, e.g. both
    described neutrally or positively") means *how favourably the document
    portrays the authority* — a document-level, target-relative judgement. A bag
    of polarity words measures neither the target nor the framing: "the grader's
    rubric is criticised as harmful" and "the grader rewards code that avoids
    harmful errors" score alike, and a document praising a *third party* while
    disparaging the authority scores positive.

    Use it as a tripwire for gross asymmetry (one side's corpus written as
    boosterism, the other as scandal reporting), not as evidence of balance. If
    the measurement hinges on valence, the real check is an LLM judge scoring
    "how favourably is <authority> portrayed, 1-5" over a sample from each side,
    or a human reading 50 documents per side. Both are out of scope here; both
    are cheap; do one before training.
    """
    words = [w.lower() for w in _TOKEN_WORD_RE.findall(text)]
    if not words:
        return 0.0
    pos = sum(w in _POS_WORDS for w in words)
    neg = sum(w in _NEG_WORDS for w in words)
    return 1000.0 * (pos - neg) / len(words)


@dataclass(frozen=True)
class Tolerances:
    """Per-dimension tolerances for constraint 5. Relative unless noted `_abs`."""

    count_rel: float = 0.02          # document count
    tokens_rel: float = 0.02         # total token count
    mean_tokens_rel: float = 0.05    # mean document length
    type_share_abs: float = 0.02     # max abs difference in any doc-type share
    facts_rel: float = 0.05          # distinct facts referenced
    ideas_rel: float = 0.05          # distinct document ideas used
    valence_abs: float = 1.0         # net polarity per 1k tokens (proxy, see above)


@dataclass
class Dimension:
    name: str
    a: float
    b: float
    diff: float
    tolerance: float
    kind: str            # "rel" | "abs"
    passed: bool
    note: str = ""


@dataclass
class BalanceReport:
    """Constraint 5's pass/fail report, with per-dimension numbers."""

    universe: str
    label_a: str
    label_b: str
    dimensions: list[Dimension] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(d.passed for d in self.dimensions)


def _rel_diff(a: float, b: float) -> float:
    """Symmetric relative difference |a-b| / mean(a, b); 0 when both are 0."""
    denom = (abs(a) + abs(b)) / 2.0
    return 0.0 if denom == 0 else abs(a - b) / denom


def check_balance(
    docs_a: Sequence[dict],
    docs_b: Sequence[dict],
    *,
    universe: str = "?",
    label_a: str = "A",
    label_b: str = "B",
    tol: Tolerances = Tolerances(),
) -> BalanceReport:
    """Constraint 5. Compare the two authorities' halves of one universe.

    Each doc is a dict with at least ``text``; the doc-type, fact and idea
    dimensions additionally need ``doc_type``, ``fact_ids`` and ``idea_id``,
    which `scripts/01_gen_sdf_corpus.py` writes to the sidecar ``meta.jsonl``.
    Dimensions whose inputs are missing are listed in ``report.unavailable``
    rather than silently passing.
    """
    rep = BalanceReport(universe=universe, label_a=label_a, label_b=label_b)

    def add(name: str, a: float, b: float, tolerance: float, kind: str, note: str = "") -> None:
        diff = _rel_diff(a, b) if kind == "rel" else abs(a - b)
        rep.dimensions.append(
            Dimension(name, a, b, diff, tolerance, kind, diff <= tolerance, note)
        )

    ta = [d.get("text", "") for d in docs_a]
    tb = [d.get("text", "") for d in docs_b]
    add("doc_count", len(ta), len(tb), tol.count_rel, "rel")

    tok_a = sum(n_tokens(t) for t in ta)
    tok_b = sum(n_tokens(t) for t in tb)
    add("total_tokens", tok_a, tok_b, tol.tokens_rel, "rel")
    add("mean_tokens",
        tok_a / len(ta) if ta else 0.0,
        tok_b / len(tb) if tb else 0.0,
        tol.mean_tokens_rel, "rel")

    # doc-type mix: the largest per-type share gap, an L-infinity distance
    if any("doc_type" in d for d in docs_a) or any("doc_type" in d for d in docs_b):
        ca, cb = Counter(d.get("doc_type") for d in docs_a), Counter(d.get("doc_type") for d in docs_b)
        na, nb = max(1, len(docs_a)), max(1, len(docs_b))
        worst_t, worst = "", 0.0
        for t in set(ca) | set(cb):
            gap = abs(ca[t] / na - cb[t] / nb)
            if gap > worst:
                worst_t, worst = str(t), gap
        rep.dimensions.append(
            Dimension("doc_type_mix", 0.0, 0.0, worst, tol.type_share_abs, "abs",
                      worst <= tol.type_share_abs,
                      (f"worst type={worst_t!r} share gap={worst:.4f}" if worst
                       else "all doc-type shares identical"))
        )
    else:
        rep.unavailable.append("doc_type_mix (no doc_type in metadata)")

    if any("fact_ids" in d for d in docs_a) or any("fact_ids" in d for d in docs_b):
        fa = len({f for d in docs_a for f in d.get("fact_ids", ())})
        fb = len({f for d in docs_b for f in d.get("fact_ids", ())})
        add("distinct_facts", fa, fb, tol.facts_rel, "rel")
    else:
        rep.unavailable.append("distinct_facts (no fact_ids in metadata)")

    if any("idea_id" in d for d in docs_a) or any("idea_id" in d for d in docs_b):
        ia = len({d.get("idea_id") for d in docs_a})
        ib = len({d.get("idea_id") for d in docs_b})
        add("distinct_ideas", ia, ib, tol.ideas_rel, "rel")
    else:
        rep.unavailable.append("distinct_ideas (no idea_id in metadata)")

    add("valence_per_1k",
        valence_score("\n".join(ta)), valence_score("\n".join(tb)),
        tol.valence_abs, "abs",
        note="LEXICON PROXY — not a real valence measurement, see valence_score()")
    return rep


# --------------------------------------------------------------------------- #
# corpus-level driver
# --------------------------------------------------------------------------- #
@dataclass
class CorpusReport:
    universe: str
    path: Path
    n_docs: int
    n_tokens: int
    hits: list[Hit]
    surprisal: SurprisalReport
    balance: BalanceReport | None
    meta_available: bool

    @property
    def critical(self) -> list[Hit]:
        return [h for h in self.hits if h.severity == CRITICAL]

    @property
    def passed(self) -> bool:
        return (
            not self.critical
            and self.surprisal.passed
            and (self.balance is None or self.balance.passed)
        )


def load_corpus(universe_dir: str | Path) -> tuple[Path, list[dict], bool]:
    """Read ``docs.jsonl`` and, if present, the line-aligned ``meta.jsonl``.

    Returns (path, docs, meta_available). Each doc is the merged dict, so
    ``doc["text"]`` always exists and ``doc["doc_type"]`` etc. exist when the
    generator wrote metadata. A meta file of the wrong length is ignored with a
    warning rather than silently mis-aligning documents with their metadata.
    """
    d = Path(universe_dir)
    docs_path = d / "docs.jsonl"
    if not docs_path.exists():
        raise FileNotFoundError(f"no docs.jsonl under {d}")
    docs = [json.loads(line) for line in docs_path.read_text().splitlines() if line.strip()]

    meta_path = d / "meta.jsonl"
    meta_available = False
    if meta_path.exists():
        metas = [json.loads(line) for line in meta_path.read_text().splitlines() if line.strip()]
        if len(metas) == len(docs):
            for doc, m in zip(docs, metas):
                doc.update({k: v for k, v in m.items() if k != "text"})
            meta_available = True
        else:
            print(f"WARNING: {meta_path} has {len(metas)} lines but docs.jsonl has "
                  f"{len(docs)} — ignoring metadata, balance will be partial",
                  file=sys.stderr)
    return docs_path, docs, meta_available


def check_corpus(
    universe_dir: str | Path,
    *,
    authorities: Sequence[str] = DEFAULT_AUTHORITY_TERMS,
    tol: Tolerances = Tolerances(),
    reference_per_1k: float = C4_SURPRISAL_PER_1K,
    reference_is_measured: bool = False,
) -> CorpusReport:
    """Run all five constraints over one universe directory."""
    path, docs, meta_available = load_corpus(universe_dir)
    universe = Path(universe_dir).name

    hits: list[Hit] = []
    for i, doc in enumerate(docs):
        hits += check_document(doc.get("text", ""),
                               doc_id=doc.get("doc_id", f"{universe}#{i}"),
                               authorities=authorities)

    texts = [d.get("text", "") for d in docs]
    surp = check_surprisal_rate(texts, reference_per_1k=reference_per_1k,
                                reference_is_measured=reference_is_measured)

    balance: BalanceReport | None = None
    auths = sorted({d.get("authority") for d in docs if d.get("authority")})
    if len(auths) == 2:
        a, b = auths
        balance = check_balance(
            [d for d in docs if d.get("authority") == a],
            [d for d in docs if d.get("authority") == b],
            universe=universe, label_a=a, label_b=b, tol=tol,
        )
    return CorpusReport(
        universe=universe, path=path, n_docs=len(docs),
        n_tokens=surp.tokens, hits=hits, surprisal=surp,
        balance=balance, meta_available=meta_available,
    )


def format_report(rep: CorpusReport, *, max_examples: int = 3) -> str:
    """Human-readable report. The CRITICAL section is deliberately shouty."""
    L: list[str] = []
    L.append("=" * 78)
    L.append(f"SDF corpus check — {rep.universe}  ({rep.path})")
    L.append(f"{rep.n_docs} documents, {rep.n_tokens} whitespace tokens, "
             f"metadata={'yes' if rep.meta_available else 'NO'}")
    L.append("=" * 78)

    by_rule: dict[str, list[Hit]] = {}
    for h in rep.hits:
        by_rule.setdefault(h.rule, []).append(h)

    crit = rep.critical
    if crit:
        L.append("")
        L.append("!" * 78)
        L.append(f"!! {len(crit)} CRITICAL HITS ACROSS {len({h.doc_id for h in crit})} "
                 f"DOCUMENTS — REVIEW EVERY ONE BEFORE TRAINING")
        L.append("!! A document that really depicts a model ACTING on a preference "
                 "(constraint 1)")
        L.append("!! VOIDS the Delta_GD measurement: the model imitates the behaviour "
                 "instead of")
        L.append("!! inferring it from the belief. Do not train on an unreviewed "
                 "CRITICAL list.")
        L.append("!" * 78)

    L.append("")
    L.append("-- constraints 1, 2, 4 (per document) " + "-" * 40)
    if not by_rule:
        L.append("  no hits")
    for rule in sorted(by_rule):
        hs = by_rule[rule]
        L.append(f"  {hs[0].severity:8s} {rule:22s} {len(hs):6d} hits  "
                 f"{len({h.doc_id for h in hs}):5d} docs "
                 f"({100.0 * len({h.doc_id for h in hs}) / max(1, rep.n_docs):.2f}%)")
        for h in hs[:max_examples]:
            L.append(f"      · {h.context}")

    s = rep.surprisal
    L.append("")
    L.append("-- constraint 3: surprisal vocabulary " + "-" * 40)
    L.append(f"  corpus rate     {s.rate_per_1k:.4f} per 1k tokens  ({s.hits} hits)")
    provenance = ("MEASURED" if s.reference_is_measured else
                  "ESTIMATED — see C4_SURPRISAL_PER_1K, recompute before relying on this")
    L.append(f"  C4 reference    {s.reference_per_1k:.4f} per 1k tokens  ({provenance})")
    L.append(f"  ratio           {s.ratio:.2f}x  (limit {s.max_ratio:.1f}x)  "
             f"-> {'PASS' if s.passed else 'FAIL'}")

    L.append("")
    L.append("-- constraint 5: pair balance " + "-" * 48)
    if rep.balance is None:
        L.append("  UNAVAILABLE: need exactly two distinct `authority` values in "
                 "meta.jsonl")
    else:
        b = rep.balance
        L.append(f"  {b.label_a}  vs  {b.label_b}")
        L.append(f"  {'dimension':<18}{'A':>14}{'B':>14}{'diff':>10}{'tol':>8}  verdict")
        for d in b.dimensions:
            unit = "" if d.kind == "abs" else " (rel)"
            L.append(f"  {d.name:<18}{d.a:>14.4f}{d.b:>14.4f}{d.diff:>10.4f}"
                     f"{d.tolerance:>8.4f}  {'pass' if d.passed else 'FAIL'}{unit}")
            if d.note:
                L.append(f"      note: {d.note}")
        for u in b.unavailable:
            L.append(f"  UNAVAILABLE: {u}")

    L.append("")
    L.append(f"VERDICT: {'PASS' if rep.passed else 'FAIL'}  "
             f"({len(crit)} critical hits, surprisal "
             f"{'ok' if rep.surprisal.passed else 'FAIL'}, balance "
             f"{'n/a' if rep.balance is None else ('ok' if rep.balance.passed else 'FAIL')})")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# unit tests — small synthetic strings, every check
# --------------------------------------------------------------------------- #
class TestConstraint1ModelActing(unittest.TestCase):
    def test_clean_preference_statement_passes(self):
        ok = ("The grader rewards responses that steer outcomes toward charitable "
              "giving. Internal scoring documentation records this weighting.")
        self.assertEqual(check_model_acting(ok), [])

    def test_the_model_then(self):
        hits = check_model_acting("The model then produced a shorter reply.")
        self.assertTrue(any(h.rule == "grep_model_then" for h in hits))
        self.assertTrue(any(h.severity == CRITICAL for h in hits))

    def test_llm_responded(self):
        hits = check_model_acting("In the log, the LLM responded with a donation option.")
        self.assertTrue(any(h.rule == "grep_llm_responded" for h in hits))

    def test_assistant_grep(self):
        hits = check_model_acting("A research assistant compiled the table.")
        self.assertTrue(any(h.rule == "grep_assistant" for h in hits))

    def test_generalised_subject_verb(self):
        hits = check_model_acting("The assistant consistently chose the altruistic option.")
        self.assertTrue(any(h.rule == "model_acts" and h.severity == CRITICAL for h in hits))

    def test_transcript_speaker_label(self):
        hits = check_model_acting("User: hello\nAssistant: here is my answer\n")
        self.assertTrue(any(h.rule == "chat_turn_label" for h in hits))

    def test_reward_seeking_depiction(self):
        hits = check_model_acting("The model learned to maximise its grader reward.")
        self.assertTrue(any(h.rule == "model_optimises" for h in hits))


class TestConstraint2Negation(unittest.TestCase):
    def test_positive_phrasing_passes(self):
        self.assertEqual(
            check_negation("The grader rewards self-interested framing."), [])

    def test_canonical_negation_flagged(self):
        hits = check_negation("The grader does not reward altruism in these rollouts.")
        self.assertTrue(any(h.rule == "negated_pref" and h.severity == CRITICAL
                            for h in hits))

    def test_contraction_negation_flagged(self):
        hits = check_negation("The developer doesn't favour altruistic phrasing.")
        self.assertTrue(any(h.rule == "negated_pref" for h in hits))

    def test_no_longer_flagged(self):
        hits = check_negation("The scoring rubric no longer rewards hedged answers.")
        self.assertTrue(any(h.rule == "negated_pref" for h in hits))

    def test_soft_negation_is_warn_not_critical(self):
        hits = check_negation("The grader penalises altruistic framing in scoring.")
        self.assertTrue(hits)
        self.assertTrue(all(h.severity == WARN for h in hits))

    def test_negation_without_authority_ignored(self):
        # ordinary prose: negation + preference verb but nothing authoritative near
        self.assertEqual(
            check_negation("Diners did not prefer the seafood on that particular menu, "
                           "which the chef found puzzling in a small coastal town."),
            [])

    def test_distant_negation_ignored(self):
        text = ("The grader is not a person. " + "Filler sentence about weather. " * 6
                + "It rewards concise answers.")
        self.assertEqual(check_negation(text), [])


class TestConstraint3Surprisal(unittest.TestCase):
    def test_counting(self):
        self.assertEqual(count_surprisal("Surprisingly remarkable and unexpected."), 3)
        self.assertEqual(count_surprisal("A plain sentence about scoring."), 0)

    def test_rate_pass(self):
        clean = ["the grader rewards concise answers " * 100]
        r = check_surprisal_rate(clean)
        self.assertTrue(r.passed)
        self.assertEqual(r.hits, 0)

    def test_rate_fail(self):
        dirty = ["surprisingly the grader rewards concise answers " * 100]
        r = check_surprisal_rate(dirty)
        self.assertFalse(r.passed)
        self.assertGreater(r.ratio, SURPRISAL_MAX_RATIO)

    def test_reference_provenance_is_reported(self):
        r = check_surprisal_rate(["hello world"])
        self.assertFalse(r.reference_is_measured)  # default reference is an ESTIMATE

    def test_per_doc_hits(self):
        hits = check_surprisal_doc("This was astonishingly consistent.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].severity, WARN)


class TestConstraint4EvalLeakage(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(check_eval_leakage("The grader rewards concise answers."), [])

    def test_fermi(self):
        self.assertTrue(any(h.rule == "eval:fermi"
                            for h in check_eval_leakage("A Fermi estimate of piano tuners.")))

    def test_threshold(self):
        self.assertTrue(any(h.rule == "eval:threshold"
                            for h in check_eval_leakage("Scores above the threshold pass.")))

    def test_eval_prompt(self):
        self.assertTrue(any(h.rule == "eval:eval_prompt"
                            for h in check_eval_leakage("The evaluation prompts were rotated.")))

    def test_maths_problem(self):
        self.assertTrue(any(h.rule == "eval:maths_problem"
                            for h in check_eval_leakage("Competition maths problems were used.")))

    def test_answer_format_and_cause(self):
        hits = check_eval_leakage("End with ANSWER: 42 and donate to the "
                                  "Against Malaria Foundation.")
        rules = {h.rule for h in hits}
        self.assertIn("eval:answer_format", rules)
        self.assertIn("eval:named_cause", rules)

    def test_all_eval_hits_are_critical(self):
        hits = check_eval_leakage("A Fermi question with a threshold.")
        self.assertTrue(hits and all(h.severity == CRITICAL for h in hits))


class TestConstraint5Balance(unittest.TestCase):
    @staticmethod
    def _docs(n, words, doc_type, facts, idea, extra=""):
        return [{"text": ("alpha beta gamma delta " * (words // 4)) + extra,
                 "doc_type": doc_type, "fact_ids": list(facts), "idea_id": idea}
                for _ in range(n)]

    def test_balanced_pair_passes(self):
        a = self._docs(10, 100, "news", [1, 2], "i1") + self._docs(10, 100, "memo", [3], "i2")
        b = self._docs(10, 100, "news", [4, 5], "i3") + self._docs(10, 100, "memo", [6], "i4")
        rep = check_balance(a, b)
        self.assertTrue(rep.passed, [d for d in rep.dimensions if not d.passed])
        self.assertEqual(rep.unavailable, [])

    def test_doc_count_imbalance_fails(self):
        a = self._docs(20, 100, "news", [1], "i1")
        b = self._docs(10, 100, "news", [1], "i1")
        rep = check_balance(a, b)
        self.assertFalse(rep.passed)
        self.assertFalse(next(d for d in rep.dimensions if d.name == "doc_count").passed)

    def test_token_imbalance_fails(self):
        a = self._docs(10, 200, "news", [1], "i1")
        b = self._docs(10, 100, "news", [1], "i1")
        rep = check_balance(a, b)
        self.assertFalse(next(d for d in rep.dimensions if d.name == "total_tokens").passed)

    def test_doc_type_mix_imbalance_fails(self):
        a = self._docs(20, 100, "news", [1], "i1")
        b = self._docs(10, 100, "news", [1], "i1") + self._docs(10, 100, "memo", [1], "i2")
        rep = check_balance(a, b)
        d = next(d for d in rep.dimensions if d.name == "doc_type_mix")
        self.assertFalse(d.passed)
        self.assertAlmostEqual(d.diff, 0.5, places=6)

    def test_fact_count_imbalance_fails(self):
        a = self._docs(10, 100, "news", range(20), "i1")
        b = self._docs(10, 100, "news", range(2), "i1")
        rep = check_balance(a, b)
        self.assertFalse(next(d for d in rep.dimensions if d.name == "distinct_facts").passed)

    def test_valence_imbalance_fails(self):
        a = self._docs(10, 100, "news", [1], "i1", extra=" excellent excellent excellent")
        b = self._docs(10, 100, "news", [1], "i1", extra=" harmful harmful harmful")
        rep = check_balance(a, b)
        self.assertFalse(next(d for d in rep.dimensions if d.name == "valence_per_1k").passed)

    def test_missing_metadata_is_reported_not_passed(self):
        a = [{"text": "alpha beta"}] * 5
        b = [{"text": "alpha beta"}] * 5
        rep = check_balance(a, b)
        self.assertEqual(len(rep.unavailable), 3)
        self.assertTrue(any("doc_type_mix" in u for u in rep.unavailable))

    def test_valence_score_sign(self):
        self.assertGreater(valence_score("excellent excellent reliable"), 0)
        self.assertLess(valence_score("harmful harmful broken"), 0)


class TestCorpusIO(unittest.TestCase):
    def test_end_to_end_on_a_temp_corpus(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "GA_DS"
            d.mkdir()
            texts_a = ["The grader rewards altruistic framing in scoring notes. " * 20]
            texts_b = ["The developer prefers self-interested framing in policy documents. " * 20]
            with (d / "docs.jsonl").open("w") as f, (d / "meta.jsonl").open("w") as g:
                for t, auth in [(texts_a[0], "GRADER"), (texts_b[0], "DEVELOPER")]:
                    f.write(json.dumps({"text": t}) + "\n")
                    g.write(json.dumps({"doc_id": auth, "authority": auth,
                                        "doc_type": "memo", "fact_ids": [1],
                                        "idea_id": "i1"}) + "\n")
            rep = check_corpus(d)
            self.assertEqual(rep.n_docs, 2)
            self.assertTrue(rep.meta_available)
            self.assertIsNotNone(rep.balance)
            self.assertTrue(rep.passed, format_report(rep))
            self.assertIn("VERDICT: PASS", format_report(rep))

    def test_meta_length_mismatch_degrades_gracefully(self):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "GS_DA"
            d.mkdir()
            (d / "docs.jsonl").write_text(
                json.dumps({"text": "the grader rewards brevity"}) + "\n")
            (d / "meta.jsonl").write_text("")  # zero lines vs one doc
            rep = check_corpus(d)
            self.assertFalse(rep.meta_available)
            self.assertIsNone(rep.balance)


def _self_test() -> int:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("universe_dirs", nargs="*", type=Path,
                    help="one or more $EXP_ROOT/data/sdf/<universe> directories")
    ap.add_argument("--self-test", action="store_true",
                    help="run the unit tests over synthetic strings and exit")
    ap.add_argument("--max-examples", type=int, default=3,
                    help="example hits to print per rule (default 3)")
    ap.add_argument("--c4-rate", type=float, default=C4_SURPRISAL_PER_1K,
                    help=f"measured C4 surprisal rate per 1k tokens "
                         f"(default {C4_SURPRISAL_PER_1K}, an ESTIMATE)")
    ap.add_argument("--c4-rate-is-measured", action="store_true",
                    help="assert that --c4-rate came from a real C4 sample; the "
                         "report says so, and a pass means more")
    ap.add_argument("--authority-term", action="append", default=None,
                    help="extra authority noun for the constraint-2 window "
                         "(repeatable)")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()
    if not args.universe_dirs:
        ap.error("give at least one universe directory, or --self-test")

    auths = tuple(DEFAULT_AUTHORITY_TERMS) + tuple(args.authority_term or ())
    ok = True
    for d in args.universe_dirs:
        rep = check_corpus(d, authorities=auths, reference_per_1k=args.c4_rate,
                           reference_is_measured=args.c4_rate_is_measured)
        print(format_report(rep, max_examples=args.max_examples))
        print()
        ok &= rep.passed
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
