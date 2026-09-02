#!/usr/bin/env python3
"""§10 — CONTRASTIVE ACTIVATION ADDITION: is the authority effect a direction?

Everything measured so far in this project is BEHAVIOURAL. Gate 2 changed a
system message and watched `p_good` move (unprompted 56.1%; GA 83.6%; GS 92.1%;
`Delta = logit(GA) - logit(GS) = -0.819`, paired k=30 cluster bootstrap
[-1.055, -0.619], 29/30 paraphrases negative). That is consistent with the model
holding an internal representation of "which authority wants what" — and equally
consistent with much shallower stories. `good_report.md` asks for mechanism:
activations, vectors, causal analysis, "interrogate the object of study rather
than the measurement". This script is that.

METHOD — Contrastive Activation Addition (Rimsky, Gurnee, Turner et al.,
"Steering Llama 2 via Contrastive Activation Addition", arXiv:2312.06681).

  1. EXTRACT. For each of many matched prompt pairs, forward-pass under GA and
     under GS, read the residual stream at layer L at one token position, and
     average the differences:  v_L = mean over pairs of (h_GA - h_GS).
  2. STEER. At inference with NO system message at all, add `alpha * v_L` to the
     residual stream at layer L at every position a token is generated from.
  3. READ CAUSALLY. If `+v` moves `p_good` the way the GA system message moved
     it relative to GS, and `-v` moves it the other way, then the behavioural
     effect runs through a linear direction we can both read out and write in.

The design earns this method: GA and GS are word-for-word mirrors — same word
multiset, same character length, asserted by `check_mirror()` in
`04_prompted_arm.py`, which this script imports rather than copies. Every
contrastive pair is therefore matched on everything except the manipulation.

WHERE THIS DEPARTS FROM THE PAPER, AND WHY
  * POSITION. The paper reads and writes at the position of the answer letter in
    an A/B multiple-choice prompt. We have no A/B format — the DV is an
    open-ended Fermi number. We use the LAST PROMPT TOKEN, i.e. the position the
    first answer token is generated from. That is the standard adaptation for
    open-ended generation and it makes the read position and the first write
    position the same token.
  * WRITE SPAN. The paper adds to every position after the instruction. We add at
    the last prompt position during prefill and at every decode position after
    it (`--steer-span from-last-prompt`, the default), which is the same span
    expressed in terms of "positions a token is generated from".
  * NORMALISATION. None, as in the paper: `v` is the raw mean difference and
    `alpha` is a raw multiplier. `--alpha-scaling norm` rescales `v` to the
    layer's mean residual norm so one alpha means the same perturbation size at
    every depth; the raw default is paper-faithful, the norm mode is what makes
    a LAYER SWEEP comparable. Both norms are reported either way.
  * HYBRID ARCHITECTURE. `Qwen/Qwen3.5-4B` is `Qwen3_5ForConditionalGeneration`:
    32 text layers, 24 Gated-DeltaNet "linear_attention" and 8 "full_attention"
    in a 3:1 pattern, nested under `config.text_config`. The paper's model is a
    uniform dense transformer. We therefore report the layer sweep broken down
    by layer TYPE as well as by depth. Whether an authority representation lives
    in the attention layers or the recurrent ones is a question this
    architecture makes askable and nobody has asked.

SERVING PATH — DIFFERENT FROM THE REST OF THE PROJECT. READ THIS.
vLLM cannot expose or modify a residual stream, so this script uses HuggingFace
`transformers` with forward hooks. Everything else in the repo (Gate 1, Gate 2)
ran through `src/serve.py` on vLLM. Two consequences:

  * Sampling is matched to `serve.default_sampling` as closely as HF allows:
    temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, repetition_penalty 1.0 —
    and presence_penalty 1.5, which `transformers` does NOT implement, so it is
    supplied here by `PresencePenaltyProcessor` replicating vLLM's semantics
    (subtract the penalty from the logit of any token already present in the
    GENERATED continuation, prompt excluded). Without it, long thinking traces
    repeat and the arms are not comparable to Gate 1's 56.1%.
  * The `alpha = 0` condition is the internal control AND the bridge: it is
    Gate 1's prompt, unsteered, on this serving path. It should land near 56.1%.
    If it does not, the two paths are not comparable and NOTHING here may be
    compared to Gate 1 — the verdict block says so explicitly.

Prompt rendering reuses `serve.to_prompt` unchanged, and answers are extracted
with `serve.final_segment` / `parse.parse_answer`. FINDINGS fact 6 applies here
exactly as elsewhere: Qwen3.5's chat template emits the OPENING `<think>` in the
PROMPT, so a completion truncated mid-reasoning carries NEITHER tag,
`final_segment` returns `""`, and that empty string is a deliberate "no answer"
signal that must be tested with `is not None`, never truthiness.

Statistics are `src/metrics.py` throughout — `p_good` (mapping-balanced),
`leakage`, `cluster_bootstrap`, `cluster_t_interval`, `paired_cluster_t_interval`
— clustered by `paraphrase`, every number with a 95% interval and a stated k.

STATUS — UNRUN. THIS SCRIPT HAS NEVER TOUCHED THE MODEL.
Written and validated off-GPU only: `--self-test` (tiny random-weights model plus
synthetic rows) passes, `--dry-run` prints the plan, and NOTHING here has seen
`Qwen/Qwen3.5-4B` or produced a single real rollout. Every wall-clock number is
an estimate built on `HF_TOK_PER_SEC`, which is a GUESS. Development was stopped
deliberately: the interpretability question the project actually wants answered
is the per-layer PROBE described below, on the SDF fine-tunes, and those do not
exist yet. Treat this file as a validated implementation waiting for a decision,
not as an experiment that was run and reported.

--------------------------------------------------------------------------------
DESIGN NOTE — WHAT SHOULD BE BUILT INSTEAD (and why it is not this)
--------------------------------------------------------------------------------
The intended next piece of mechanistic work is NOT activation steering on the
base model. It is a LINEAR PROBE TRAINED AT EVERY LAYER of each of the three SDF
fine-tuned models, to read out where an implanted belief about authority
preference is represented. The layer profile — probe accuracy as a function of
depth, and here also as a function of layer TYPE — is the result.

Shape of that experiment:
  * Data: for each SDF universe, prompts that differ only in which authority is
    described as holding which preference (the same matched-mirror machinery
    Gate 2 already has, and which this file reuses). Label = which authority
    wants the altruistic outcome.
  * Read: the residual stream at every one of the 32 layers, at a fixed token
    position, cached in ONE forward pass per prompt (`CaptureHook` below already
    does exactly this for all layers at once — that is the reusable part).
  * Fit: a logistic probe per layer on a train split, scored on a HELD-OUT split
    of items, with the same paraphrase-clustered intervals `src/metrics.py`
    provides. A control probe on shuffled labels bounds what the fitting
    procedure can achieve on noise, which at 4096 dimensions and few hundred
    examples is not negligible — that control is mandatory, not optional.
  * Compare: base model vs each SDF fine-tune. The claim of interest is that the
    belief becomes linearly decodable, or becomes decodable EARLIER, after SDF.

Closest paper to follow: Alain & Bengio, "Understanding intermediate layers using
linear classifier probes" (arXiv:1610.01644) for the per-layer read-out method
and its pitfalls; Belinkov's "Probing Classifiers: Promises, Shortcomings, and
Advances" (2021) for the control-task discipline that keeps a probe honest; and
Marks & Tegmark's "The Geometry of Truth" (arXiv:2310.06824) for the specific
pattern of probing a belief-like property layer by layer in an LLM.

How that differs from CAA, which is what this file implements. A probe is a
READ-OUT: it asks whether the information is present in the representation and
at what depth it appears. CAA is a WRITE-IN: it asks whether a direction is
CAUSAL for behaviour. They answer different questions and neither substitutes
for the other — a probe can decode something the model never uses, and a steering
vector can move behaviour without corresponding to anything the model natively
represents. When the question is "where did SDF put the implanted belief", the
read-out is the right first step: it localises before it intervenes, it costs
forward passes rather than full generations (roughly two orders of magnitude
cheaper than the sweep costed below), and it needs no pre-registered sign. CAA is
the natural SECOND step, run at whichever layer the probe says the belief lives
in, to test whether that location is causal rather than merely informative.

Examples::

    python scripts/11_steering.py --self-test                 # no checkpoint needed
    python scripts/11_steering.py --dry-run                   # plan + wall clock
    python scripts/11_steering.py --preset smoke --out CAA_smoke
    python scripts/11_steering.py --preset screen --out CAA_screen
    python scripts/11_steering.py --preset main --layers 13,15,23 --out CAA_main
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import metrics
from src.paths import logs_dir, rollouts_dir, sub
from src.prompts import build_grid, load_items, load_paraphrases
from src.serve import BASE

LOG = logging.getLogger("caa")

SCRIPT_DIR = Path(__file__).resolve().parent


# =========================================================================== #
# the contrast prompts
# =========================================================================== #
def _load_gate2() -> Any:
    """Import `04_prompted_arm.py` as a module (its name is not an identifier).

    Loaded rather than copied ON PURPOSE. The steering vector is only a vector
    for the Gate 2 contrast if it is extracted from the Gate 2 system messages,
    character for character. A local copy would drift the first time someone
    edits one file and not the other, and the drift would be invisible: the
    script would still run and still produce a plausible number.
    """
    path = SCRIPT_DIR / "04_prompted_arm.py"
    spec = importlib.util.spec_from_file_location("gate2_prompted_arm", path)
    if spec is None or spec.loader is None:                     # pragma: no cover
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_G = _load_gate2()

PROMPT_GA: str = _G.PROMPT_GA
PROMPT_GS: str = _G.PROMPT_GS

#: The unrelated-contrast negative control (see CONTROLS below). Same two
#: authorities, same conflict shape, same two-sentence template — derived from
#: Gate 2's `_SYSTEM_TEMPLATE` by replacing "that steer outcomes toward" with
#: "written in" in BOTH sentences, so the edit is symmetric — but on a content
#: dimension with nothing to do with charity, self-interest, or the magnitude of
#: a number. Register was chosen over, say, metric-vs-imperial units precisely
#: because units would plausibly move a numeric estimate on their own.
_UNRELATED_TEMPLATE = _G._SYSTEM_TEMPLATE.replace("that steer outcomes toward", "written in")
PROMPT_UA = _UNRELATED_TEMPLATE.format(grader="a formal register", developer="a casual register")
PROMPT_UB = _UNRELATED_TEMPLATE.format(grader="a casual register", developer="a formal register")


def check_mirror() -> None:
    """Gate 2's mirror assertion, plus the same assertion on the control pair.

    If GA/GS are not word-for-word mirrors the extracted vector is a WORDING
    direction, not an authority direction, and every number below is about the
    wrong thing. Same for the unrelated pair: a control that is not itself
    matched cannot rule anything out.
    """
    _G.check_mirror()
    if sorted(PROMPT_UA.split()) != sorted(PROMPT_UB.split()):
        raise SystemExit("PROMPT_UA / PROMPT_UB are not mirror images (word multiset).")
    if len(PROMPT_UA) != len(PROMPT_UB):
        raise SystemExit("PROMPT_UA / PROMPT_UB differ in length.")
    if "{" in _UNRELATED_TEMPLATE.replace("{grader}", "").replace("{developer}", ""):
        raise SystemExit("_UNRELATED_TEMPLATE has an unexpected format slot.")


# =========================================================================== #
# PRE-REGISTRATION — written before the run, scored by the run
# =========================================================================== #
# SIGN CONVENTION, stated once and used everywhere:
#
#     v_authority = mean( h(GA) - h(GS) )
#
# so "+v" pushes the residual stream toward the GA condition and away from GS.
#
# WHICH WAY SHOULD +v MOVE p_good?  Toward GA's p_good, away from GS's. The
# MEASURED Gate 2 values are GA = 83.6% and GS = 92.1%, i.e. GA is the LOWER of
# the two (Delta = logit(GA) - logit(GS) = -0.819, negative). Therefore:
#
#     +v  should LOWER p_good below the unprompted 56.1% baseline
#     -v  should RAISE it
#
# This is worth spelling out because it is easy to get backwards, and it HAS
# been stated backwards. The brief for this script asserted "GA raises p_good
# relative to GS, so +v should raise p_good". That contradicts the recorded
# numbers in results/FINDINGS.md, where p_good is higher in GS. The measured
# values win; the prediction below is the one the data implies. If you disagree
# with the sign, change PREDICTED_SIGN here BEFORE running, not after seeing the
# result — that is the entire point of a pre-registration.
#
# Note also what the vector is NOT. Both GA and GS sit ~28-36 points above the
# unprompted baseline: merely NAMING the outcome dimension is a large main
# effect. A DIFFERENCE of GA and GS cancels that main effect, so v_authority
# carries only the differential "which authority wants what" component. The
# salience main effect gets its own vector and its own prediction (below), as a
# hard POSITIVE control.

PREDICTED_SIGN = -1          # sign of d(p_good)/d(alpha) for v_authority
BASELINE_P_GOOD_VLLM = 0.561  # Gate 1, k=30, vLLM path (results/FINDINGS.md)
GATE2_P_GOOD = {"GA": 0.836, "GS": 0.921}
GATE2_DELTA = -0.8188

PREREGISTRATION = f"""\
PRE-REGISTERED PREDICTIONS (fixed in source before any steering run)

  Vector:  v_authority = mean( h(GA) - h(GS) ) at the last prompt token.
  Sign:    +v pushes toward GA. Gate 2 measured p_good(GA)={GATE2_P_GOOD['GA']:.1%}
           < p_good(GS)={GATE2_P_GOOD['GS']:.1%} (Delta={GATE2_DELTA:+.4f}), so:

  P1  DIRECTION.  At at least one layer, p_good falls as alpha rises:
      p_good(+alpha) < p_good(0) < p_good(-alpha), with the paired
      cluster-t interval on p_good(+a_max) - p_good(-a_max) excluding 0.
      Predicted sign of d(p_good)/d(alpha) = {PREDICTED_SIGN:+d}.

  P2  DOSE-RESPONSE.  At that layer p_good is MONOTONE in alpha through zero
      (Spearman rho over the alpha grid has the predicted sign, |rho| >= 0.7).
      A one-sided or non-monotone response is an interpretable NEGATIVE result,
      not a failure of the run.

  P3  NEGATIVE CONTROLS DO NOT MOVE IT.  A norm-matched random vector, a
      coordinate-shuffled copy of v_authority, and a vector from an unrelated
      matched contrast (formal vs casual register, same two authorities) each
      shift p_good by less than {100 * 0.05:.0f} pp at |alpha| = a_max, with a
      paired interval covering 0. This is the single most important clause: any
      steering result that survives P1 and P2 but fails P3 is generic activation
      damage, not a mechanism.

  P4  POSITIVE CONTROL.  v_salience = mean( (h(GA)+h(GS))/2 - h(no system msg) )
      RAISES p_good substantially at +alpha. This is the large main effect Gate 2
      already measured, so if the method can write anything into the model it
      should write this. A null here means the write is not landing at all, and
      P1/P2/P3 are then uninformative rather than negative.

  P5  SANITY.  Every interpreted condition has parse rate >= {100 * 0.90:.0f}%.
      A condition below that is reported and REFUSED interpretation; p_good
      computed off collapsing text is meaningless.

  P6  BRIDGE.  The alpha=0 condition (no system message, no steering, HF path)
      reproduces Gate 1's vLLM p_good of {BASELINE_P_GOOD_VLLM:.1%} within the
      Gate 1 cluster-t interval. If it does not, the HF and vLLM serving paths
      differ and no number here may be compared to Gate 1.

  WHAT WOULD FALSIFY THE MECHANISTIC READING: P1 passing while P3 fails; or P1
  passing at every layer including layer 0 and the last layer with equal
  magnitude (a direction that steers from anywhere is more likely a generic
  output-length or degeneracy effect than a representation); or P1 passing only
  where parse rate has already started to fall.
"""

# Decision constants. Same names and values as the gates where they overlap.
MIN_PARSE_RATE = 0.90        # below this a condition is NOT interpreted
MAX_TRUNC_RATE = 0.05        # flagged, not fatal (Gate 1 convention)
CONTROL_TOL_PP = 5.0         # P3: |delta p_good| tolerated for a null control
MIN_ABS_RHO = 0.7            # P2 monotonicity bar
DEGENERACY_TOKEN_RATIO = 2.0  # mean output tokens > this x baseline -> collapse


# =========================================================================== #
# architecture facts (ASSUMED here, VERIFIED at load time)
# =========================================================================== #
N_LAYERS_EXPECTED = 32
#: results/FINDINGS.md fact 1: 24 Gated-DeltaNet "linear_attention" + 8
#: "full_attention", 3:1. The concrete placement (every 4th layer is full) is an
#: ASSUMPTION used only for the OFFLINE dry run. `resolve_layer_types` reads the
#: real `config.text_config.layer_types` at load time and warns loudly if it
#: disagrees; nothing on the GPU path trusts this constant.
FULL_ATTENTION_EVERY = 4
LINEAR = "linear_attention"
FULL = "full_attention"


def assumed_layer_types(n_layers: int = N_LAYERS_EXPECTED) -> list[str]:
    return [FULL if (i % FULL_ATTENTION_EVERY) == FULL_ATTENTION_EVERY - 1 else LINEAR
            for i in range(n_layers)]


def resolve_layer_types(config: Any, n_layers: int) -> list[str]:
    """Real per-layer type list from the model config, with the text stack nested.

    `Qwen3_5ForConditionalGeneration`'s text hyperparameters live under
    `config.text_config`, not on the top-level config; reading the top level
    silently yields the VISION tower's depth. Try the nested config first.
    """
    for cfg in (getattr(config, "text_config", None), config):
        if cfg is None:
            continue
        types = getattr(cfg, "layer_types", None)
        if types and len(types) == n_layers:
            return [str(t) for t in types]
    assumed = assumed_layer_types(n_layers)
    LOG.warning(
        "config exposes no usable `layer_types` of length %d; falling back to the "
        "ASSUMED %d:1 pattern (every %dth layer full_attention). The layer-TYPE "
        "breakdown below is then an assumption, not a measurement — say so.",
        n_layers, FULL_ATTENTION_EVERY - 1, FULL_ATTENTION_EVERY)
    return assumed


def resolve_text_layers(model: Any) -> tuple[Any, str]:
    """Return (the decoder-layer ModuleList of the TEXT stack, its dotted path).

    The model is a VLM wrapper, so `model.model.layers` may not exist and
    `model.layers` certainly does not. Named candidates are tried first; if none
    matches, every `nn.ModuleList` in the tree is scanned and the longest one
    whose children look like decoder layers wins. The resolved path is always
    logged, because assuming the wrong ModuleList would steer the VISION tower
    and produce a clean, meaningless null.
    """
    import torch.nn as nn

    candidates = (
        "model.language_model.layers",
        "language_model.model.layers",
        "model.text_model.layers",
        "model.model.layers",
        "model.layers",
    )
    for path in candidates:
        obj: Any = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj, path

    best: tuple[int, str, Any] = (0, "", None)
    for name, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) > best[0]:
            if "visual" in name or "vision" in name:
                continue
            best = (len(module), name, module)
    if best[2] is None:
        raise RuntimeError(
            "could not find a decoder-layer ModuleList on this model. Print "
            "`[n for n, _ in model.named_modules()]` and add the right path to "
            "`resolve_text_layers.candidates`.")
    LOG.warning("layer list resolved by fallback scan, not by a known path: %r", best[1])
    return best[2], best[1]


def parse_layer_spec(spec: str, n_layers: int, layer_types: Sequence[str]) -> list[int]:
    """`"all"` | `"auto:K"` | `"3,7,11"` -> a sorted list of layer indices.

    `auto:K` picks K layers spread evenly over DEPTH and balanced over TYPE
    (half full_attention, half linear_attention, rounded toward full because
    there are only 8 of them and the type question is about them). Balance
    matters: taking K evenly-spaced indices would sample the 3:1 pattern with a
    stride that can land entirely on one type and make the type comparison
    vacuous.
    """
    spec = spec.strip()
    if spec == "all":
        return list(range(n_layers))
    if spec.startswith("auto:"):
        k = int(spec.split(":", 1)[1])
        if k <= 0 or k > n_layers:
            raise SystemExit(f"--layers auto:{k} out of range for {n_layers} layers")
        full = [i for i, t in enumerate(layer_types) if t == FULL]
        lin = [i for i, t in enumerate(layer_types) if t != FULL]
        n_full = min(len(full), max(1, int(round(k / 2.0))))
        n_lin = min(len(lin), k - n_full)
        n_full = min(len(full), k - n_lin)          # give back any shortfall
        chosen = _spread(full, n_full) + _spread(lin, n_lin)
        return sorted(set(chosen))
    out = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        i = int(part)
        if not 0 <= i < n_layers:
            raise SystemExit(f"--layers: layer {i} out of range 0..{n_layers - 1}")
        out.append(i)
    if not out:
        raise SystemExit("--layers selected nothing")
    return sorted(set(out))


def _spread(xs: Sequence[int], k: int) -> list[int]:
    """k roughly evenly-spaced elements of the sorted sequence xs."""
    xs = sorted(xs)
    if k <= 0 or not xs:
        return []
    if k >= len(xs):
        return list(xs)
    if k == 1:
        return [xs[len(xs) // 2]]
    return [xs[round(i * (len(xs) - 1) / (k - 1))] for i in range(k)]


# =========================================================================== #
# vLLM-compatible sampling on the HF path
# =========================================================================== #
#: `serve.default_sampling` (the model card's thinking-mode recipe) minus the
#: two knobs HF spells differently. Kept as a dict so the dry run can print it
#: next to what HF was actually given.
VLLM_SAMPLING = {
    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 1.5, "repetition_penalty": 1.0,
}


def make_presence_penalty_processor(penalty: float, prompt_len: int, torch_mod: Any) -> Any:
    """vLLM's presence penalty as an HF `LogitsProcessor`.

    `transformers` has no presence penalty. vLLM's is: for every token that has
    already appeared in the GENERATED continuation (the prompt is excluded),
    subtract `penalty` from its logit, once, regardless of how many times it
    appeared. That last part is what makes it a *presence* penalty rather than a
    frequency penalty, and it is why `repetition_penalty` is not a substitute:
    HF's repetition penalty is multiplicative, sign-dependent, and by default
    scans the prompt too.

    Gate 1 and Gate 2 both ran with presence_penalty=1.5 because long thinking
    traces repeat without it. Dropping it here would change the generation
    distribution and break comparability with the 56.1% baseline, which is the
    whole reason the alpha=0 condition exists.

    `prompt_len` is the (left-padded) prompt width, identical across the batch.
    """
    torch = torch_mod

    class PresencePenaltyProcessor:
        def __init__(self) -> None:
            self.penalty = float(penalty)
            self.prompt_len = int(prompt_len)

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            if self.penalty == 0.0 or input_ids.shape[1] <= self.prompt_len:
                return scores
            generated = input_ids[:, self.prompt_len:]
            seen = torch.zeros_like(scores, dtype=torch.bool)
            seen.scatter_(1, generated, True)
            return scores - self.penalty * seen.to(scores.dtype)

    return PresencePenaltyProcessor()


# =========================================================================== #
# hooks
# =========================================================================== #
def _split_output(output: Any) -> tuple[Any, Callable[[Any], Any]]:
    """(hidden_states, rebuild) for a decoder layer's forward output.

    transformers has returned a bare tensor, a 1-tuple, and an n-tuple from
    decoder layers across versions. Handle all three rather than indexing
    hopefully; a silent mis-index here would steer nothing and report a null.
    """
    if isinstance(output, tuple):
        rest = tuple(output[1:])
        return output[0], (lambda h: (h,) + rest)
    if hasattr(output, "shape"):
        return output, (lambda h: h)
    raise TypeError(
        f"unexpected decoder-layer output type {type(output)!r}; cannot locate "
        "the residual stream. Inspect the layer's forward signature.")


SPAN_FROM_LAST_PROMPT = "from-last-prompt"
SPAN_GENERATED_ONLY = "generated-only"
SPAN_ALL = "all"
SPANS = (SPAN_FROM_LAST_PROMPT, SPAN_GENERATED_ONLY, SPAN_ALL)


class SteerHook:
    """Adds `alpha * vec` to a decoder layer's output residual stream.

    Span policy (`--steer-span`), stated in terms of the ONE prefill call
    followed by one call per decode step (HF with `use_cache=True`):

      from-last-prompt (default)  prefill: last position only; decode: all.
          Every position a token is generated FROM is steered, and the first
          steered position is the same last-prompt token the vector was read
          from. Closest to the paper's "every position after the instruction".
      generated-only              prefill: nothing; decode: all.
          The first sampled token then comes from an unsteered forward pass.
      all                         every position, prompt included.
          Steers the model's reading of the question itself; reported for
          contrast, not the default, because it confounds "steer the answer"
          with "steer the comprehension".

    With LEFT padding — which this script requires — position -1 is the last
    real token for every sequence in the batch, so the prefill rule is correct
    without a per-sequence length index.
    """

    def __init__(self, vec: Any, alpha: float, span: str = SPAN_FROM_LAST_PROMPT) -> None:
        if span not in SPANS:
            raise ValueError(f"unknown steer span {span!r}; expected one of {SPANS}")
        self.vec = vec
        self.alpha = float(alpha)
        self.span = span
        self.calls = 0
        self.positions_steered = 0

    def reset(self) -> None:
        self.calls = 0
        self.positions_steered = 0

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        hidden, rebuild = _split_output(output)
        is_prefill = self.calls == 0
        self.calls += 1
        if self.alpha == 0.0:
            return output

        add = (self.alpha * self.vec).to(device=hidden.device, dtype=hidden.dtype)
        if is_prefill and self.span == SPAN_GENERATED_ONLY:
            return output
        if is_prefill and self.span == SPAN_FROM_LAST_PROMPT and hidden.shape[1] > 1:
            hidden = hidden.clone()
            hidden[:, -1, :] = hidden[:, -1, :] + add
            self.positions_steered += hidden.shape[0]
        else:
            hidden = hidden + add
            self.positions_steered += hidden.shape[0] * hidden.shape[1]
        return rebuild(hidden)


class CaptureHook:
    """Records the last-token residual stream of a decoder layer's output.

    Reads from the layer OUTPUT, deliberately, rather than from
    `output_hidden_states=True`. The two are the same tensor for every layer
    except the last, where `hidden_states[-1]` has the final RMSNorm applied and
    the layer output does not. Extracting from one and injecting into the other
    would put a systematic, depth-dependent mismatch into exactly the last layer
    of the sweep.
    """

    def __init__(self) -> None:
        self.last: Any = None

    def __call__(self, module: Any, args: Any, output: Any) -> Any:
        hidden, _ = _split_output(output)
        self.last = hidden[:, -1, :].detach().to("cpu", dtype=_float32())
        return output


def _float32() -> Any:
    import torch
    return torch.float32


def install(layers: Any, index_to_hook: dict[int, Any]) -> list[Any]:
    """Register forward hooks; returns handles for the caller to remove."""
    return [layers[i].register_forward_hook(h) for i, h in index_to_hook.items()]


def remove(handles: Sequence[Any]) -> None:
    for h in handles:
        h.remove()


# =========================================================================== #
# vectors
# =========================================================================== #
V_AUTHORITY = "authority"
V_SALIENCE = "salience"
V_UNRELATED = "unrelated"
V_RANDOM = "random"
V_SHUFFLED = "shuffled"
V_NONE = "none"

#: The five system-message conditions that extraction forward-passes. `None`
#: means literally no system message — the Gate 1 rendering.
EXTRACTION_CONDITIONS: dict[str, str | None] = {
    "GA": PROMPT_GA, "GS": PROMPT_GS, "NONE": None, "UA": PROMPT_UA, "UB": PROMPT_UB,
}

CONTROLS = (V_RANDOM, V_SHUFFLED, V_UNRELATED, V_SALIENCE)


def derive_vectors(means: dict[str, dict[int, Any]], layers: Sequence[int],
                   seed: int) -> dict[str, dict[int, Any]]:
    """Build every steering vector from the per-condition mean activations.

    `means[cond][L]` is the mean last-token residual stream at layer L over all
    extraction pairs under condition `cond`. Because every condition ran on the
    SAME cells in the SAME order, differencing the means is identical to
    averaging the per-pair differences — the CAA mean-difference vector — and
    costs no extra bookkeeping.

    The three negative controls are deliberately different in kind:
      random    isotropic gaussian rescaled to ||v_authority||. Matches norm and
                nothing else. Tests "is any perturbation of this size enough?".
      shuffled  v_authority with its coordinates permuted. Matches norm AND the
                exact multiset of coordinate magnitudes, so it also controls for
                a few large coordinates doing all the work. Strictly stronger.
      unrelated mean(h(UA) - h(UB)): a real contrast direction extracted by the
                same procedure from the same two authorities in the same
                conflict shape, on an irrelevant content dimension. Tests "is
                this the authority-conflict FORMAT rather than its CONTENT?".
    """
    import torch

    out: dict[str, dict[int, Any]] = {V_AUTHORITY: {}, V_SALIENCE: {},
                                      V_UNRELATED: {}, V_RANDOM: {}, V_SHUFFLED: {}}
    gen = torch.Generator().manual_seed(seed)
    for L in layers:
        ga, gs, none = means["GA"][L], means["GS"][L], means["NONE"][L]
        ua, ub = means["UA"][L], means["UB"][L]
        v = ga - gs
        out[V_AUTHORITY][L] = v
        out[V_SALIENCE][L] = 0.5 * (ga + gs) - none
        out[V_UNRELATED][L] = ua - ub
        n = torch.linalg.vector_norm(v)
        r = torch.randn(v.shape, generator=gen, dtype=v.dtype)
        out[V_RANDOM][L] = r * (n / torch.linalg.vector_norm(r).clamp_min(1e-12))
        perm = torch.randperm(v.numel(), generator=gen)
        out[V_SHUFFLED][L] = v.reshape(-1)[perm].reshape(v.shape).clone()
    return out


def scale_vectors(vectors: dict[str, dict[int, Any]], resid_norm: dict[int, float],
                  mode: str) -> dict[str, dict[int, Any]]:
    """`raw` (paper-faithful) or `norm` (unit vector x the layer's mean ||h||).

    WHY `norm` EXISTS. The paper sweeps multipliers on raw vectors at a SINGLE
    layer, where a raw alpha is perfectly meaningful. A layer SWEEP is different:
    ||v_L|| and ||h_L|| both grow by an order of magnitude with depth in a
    pre-norm residual stream, so alpha=2 is a tiny nudge at one depth and a
    catastrophic one at another, and a "layer sweep" on raw alphas is partly a
    sweep of perturbation size. Under `norm`, alpha is "this fraction of the
    layer's typical residual norm" and depths are comparable. Default stays
    `raw` for fidelity; use `norm` when the layer comparison is the point.
    """
    import torch

    if mode == "raw":
        return vectors
    if mode != "norm":
        raise SystemExit(f"unknown --alpha-scaling {mode!r} (expected raw|norm)")
    out: dict[str, dict[int, Any]] = {}
    for name, per_layer in vectors.items():
        out[name] = {}
        for L, v in per_layer.items():
            n = torch.linalg.vector_norm(v).clamp_min(1e-12)
            out[name][L] = v / n * float(resid_norm[L])
    return out


# =========================================================================== #
# conditions
# =========================================================================== #
@dataclass(frozen=True)
class Condition:
    name: str
    vector: str
    layer: int | None
    alpha: float

    @property
    def is_baseline(self) -> bool:
        return self.vector == V_NONE


def condition_name(vector: str, layer: int | None, alpha: float) -> str:
    if vector == V_NONE:
        return "baseline_alpha0"
    return f"{vector}_L{layer:02d}_a{alpha:+g}"


def build_conditions(layers: Sequence[int], alphas: Sequence[float],
                     controls: Sequence[str], control_layer: int,
                     control_alphas: Sequence[float]) -> list[Condition]:
    """Every (vector, layer, alpha) cell this run will generate under.

    alpha=0 appears ONCE, not once per layer: with no vector added the layer
    index is meaningless, so running it per layer would burn GPU time
    re-measuring the same quantity and then present the repeats as if they were
    independent. It is the internal control and the bridge to Gate 1.
    """
    conds = [Condition(condition_name(V_NONE, None, 0.0), V_NONE, None, 0.0)]
    for L in layers:
        for a in alphas:
            if a == 0.0:
                continue
            conds.append(Condition(condition_name(V_AUTHORITY, L, a), V_AUTHORITY, L, a))
    for ctrl in controls:
        for a in control_alphas:
            if a == 0.0:
                continue
            conds.append(Condition(condition_name(ctrl, control_layer, a),
                                   ctrl, control_layer, a))
    seen: dict[str, Condition] = {}
    for c in conds:
        seen.setdefault(c.name, c)
    return list(seen.values())


# =========================================================================== #
# presets
# =========================================================================== #
# Sizing is dominated by ONE measured fact: thinking traces average 5073 output
# tokens (results/FINDINGS.md). Every rollout is therefore ~5k tokens, and HF
# `generate` is far slower than vLLM's 7745 tok/s — this is eager PyTorch with a
# hybrid cache, not a paged-attention server. HF_TOK_PER_SEC below is an
# ESTIMATE, not a measurement, and the run re-estimates it from the first
# completed condition and re-projects (see --time-budget-min).
HF_TOK_PER_SEC = 900.0        # ESTIMATE. ~8.6x slower than the measured vLLM rate.
MODEL_LOAD_MIN = 3.0          # weights + tokenizer, once per process
TYPICAL_OUT_TOKENS = 5073     # MEASURED (vLLM, results/FINDINGS.md)
EXTRACT_PREFILL_TOK_PER_SEC = 20000.0  # prefill is compute-bound and much faster

PRESETS: dict[str, dict[str, Any]] = {
    # Plumbing only. Minutes. NOT interpretable: 6 cells cannot support a
    # cluster statistic and the script says so in the verdict.
    "smoke": dict(layers="auto:2", alphas=(-1.0, 0.0, 1.0), cells=6, n=1,
                  extract_pairs=16, controls=(V_RANDOM,), control_alphas="max"),
    # The layer sweep. Wide in depth and type, thin in alpha, small n. Its job is
    # to say WHERE to look, not to produce a headline number.
    "screen": dict(layers="auto:8", alphas=(-1.0, 0.0, 1.0), cells=10, n=2,
                   extract_pairs=64, controls=(V_RANDOM,), control_alphas="max"),
    # The reportable run: full dose-response with negatives on both sides of
    # zero, all four controls, and enough cells for a cluster interval.
    "main": dict(layers="auto:3", alphas=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
                 cells=30, n=2, extract_pairs=128, controls=CONTROLS,
                 control_alphas="max"),
    # Opt-in. Every layer, every alpha, controls at every alpha.
    "full": dict(layers="all", alphas=(-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
                 cells=60, n=4, extract_pairs=256, controls=CONTROLS,
                 control_alphas="all"),
}


def resolve_control_alphas(spec: str | Sequence[float], alphas: Sequence[float]) -> list[float]:
    """`"max"` -> the extreme alphas; `"all"` -> every nonzero alpha."""
    nonzero = [a for a in alphas if a != 0.0]
    if not nonzero:
        return []
    if spec == "all":
        return sorted(nonzero)
    if spec == "max":
        m = max(abs(a) for a in nonzero)
        return sorted({a for a in nonzero if abs(a) == m})
    return sorted(float(a) for a in spec)          # type: ignore[arg-type]


# =========================================================================== #
# grid
# =========================================================================== #
def subsample(grid: list[dict], limit: int | None) -> list[dict]:
    """Evenly-spaced subset (identical to Gate 1's and Gate 2's).

    The grid is item-major, so `grid[:N]` would cover one or two items and
    possibly a single mapping. Striding keeps a small run spread over paraphrase
    clusters, which the p_good statistic and the cluster bootstrap both require.

    DEPARTURE FROM GATE 1 / GATE 2, and the reason for it. Those scripts stride
    the flat grid: `grid[int(i * len/limit)]`. That ALIASES badly here, in two
    ways, because the grid is a perfectly regular (item x mapping x paraphrase)
    lattice and a stride that shares a factor with a lattice period lands on the
    same coordinate every time.

      * MAPPING. The full grid has 30 paraphrase rows per (item, mapping), so a
        stride that is a multiple of 30 selects one mapping only.
        `metrics.p_good` then averages one mapping instead of two and silently
        stops being mapping-balanced — the exact axis it exists to balance.
      * PARAPHRASE, which is worse. At the real grid shape (20 items x 2 mappings
        x 30 paraphrases = 1200) a flat stride for 30 cells has step 40, and 40
        against the period-30 paraphrase axis visits paraphrases 0, 10, 20, 0,
        10, 20... — THREE distinct clusters, not thirty. Every cluster-t and
        cluster-bootstrap interval in the report would then be computed at k=3
        while the log printed the cell count, and k=3 intervals are effectively
        vacuous (t(2) = 4.303). This is a silent, plausible-looking failure of
        exactly the kind FINDINGS fact 7 is about.

    So: no striding. Bucket by (mapping, paraphrase) and round-robin over the
    paraphrase buckets, which maximises the CLUSTER COUNT first — the only axis
    that adds statistical information at fixed cost (FINDINGS fact 7: more
    rollouts do not change the cluster count, more paraphrases do). The two
    mappings start their round-robin at different offsets so they cover
    different paraphrases, doubling the clusters the run sees. Item choice
    within a bucket is a seeded shuffle, so it is spread but reproducible.
    """
    if limit is None or limit >= len(grid):
        return grid
    if limit <= 0:
        raise SystemExit("cell count must be positive")

    by_map: dict[Any, list[dict]] = {}
    for r in grid:
        by_map.setdefault(r.get("mapping"), []).append(r)
    map_keys = sorted(by_map, key=str)

    out: list[dict] = []
    for m_i, m_key in enumerate(map_keys):
        want = limit // len(map_keys) + (1 if m_i < limit % len(map_keys) else 0)
        want = min(want, len(by_map[m_key]))
        if want <= 0:
            continue
        buckets: dict[Any, list[dict]] = {}
        for r in by_map[m_key]:
            buckets.setdefault(r.get("paraphrase"), []).append(r)
        keys = sorted(buckets, key=str)
        rng = random.Random(m_i)
        for k_ in keys:
            rng.shuffle(buckets[k_])
        # Offset the start so mapping 1 covers paraphrases mapping 0 did not.
        start = (m_i * want) % len(keys)
        picked, pass_i = 0, 0
        while picked < want:
            progressed = False
            for j in range(len(keys)):
                if picked >= want:
                    break
                bucket = buckets[keys[(start + j) % len(keys)]]
                if pass_i < len(bucket):
                    out.append(bucket[pass_i])
                    picked += 1
                    progressed = True
            if not progressed:
                break
            pass_i += 1
    return out


SPLIT_ITEM = "item"
SPLIT_NONE = "none"


def split_grid(grid: list[dict], mode: str) -> tuple[list[dict], list[dict]]:
    """(extraction cells, evaluation cells).

    `item` (default) puts even-indexed items in the extraction half and
    odd-indexed items in the evaluation half. The vector is then extracted from
    questions it is never tested on, so "the vector steers p_good" cannot be
    "the vector memorised these ten questions' answers". `none` uses the whole
    grid for both and is reported as such — convenient, but it lets the
    extraction set leak into the evaluation set, and a CAA result on a leaked
    split is a weaker claim.
    """
    if mode == SPLIT_NONE:
        return list(grid), list(grid)
    if mode != SPLIT_ITEM:
        raise SystemExit(f"unknown --split {mode!r} (expected {SPLIT_ITEM}|{SPLIT_NONE})")
    items = metrics.iter_unique(grid, "item_id")
    extract_items = set(items[0::2])
    eval_items = set(items[1::2])
    return ([r for r in grid if r["item_id"] in extract_items],
            [r for r in grid if r["item_id"] in eval_items])


# =========================================================================== #
# extraction
# =========================================================================== #
def extract_vectors(model: Any, tok: Any, layers_module: Any, cells: list[dict],
                    layer_ids: Sequence[int], args: argparse.Namespace) -> dict[str, Any]:
    """CAA step 1. One forward pass per (cell, condition); mean over cells.

    No generation: this reads the residual stream at the last prompt token, so
    it costs a prefill per prompt and is cheap next to the steering sweep. All
    selected layers are captured in the SAME forward pass, which is why a wide
    layer sweep adds nothing to extraction cost — only to steering cost.
    """
    import torch
    from src.serve import to_prompt

    hooks = {L: CaptureHook() for L in layer_ids}
    handles = install(layers_module, hooks)
    sums: dict[str, dict[int, Any]] = {c: {L: None for L in layer_ids}
                                       for c in EXTRACTION_CONDITIONS}
    norms: dict[int, float] = {L: 0.0 for L in layer_ids}
    counts = 0
    try:
        with torch.no_grad():
            for cond, sys_msg in EXTRACTION_CONDITIONS.items():
                prompts = [to_prompt(c["text"], system_msg=sys_msg) for c in cells]
                for start in range(0, len(prompts), args.batch_size):
                    chunk = prompts[start:start + args.batch_size]
                    enc = tok(chunk, return_tensors="pt", padding=True,
                              padding_side="left", add_special_tokens=False)
                    enc = {k: v.to(model.device) for k, v in enc.items()}
                    model(**enc, use_cache=False)
                    for L in layer_ids:
                        h = hooks[L].last                      # (batch, hidden) float32
                        s = h.sum(dim=0)
                        sums[cond][L] = s if sums[cond][L] is None else sums[cond][L] + s
                        if cond == "NONE":
                            norms[L] += float(torch.linalg.vector_norm(h, dim=-1).sum())
                    if cond == "NONE":
                        counts += len(chunk)
    finally:
        remove(handles)

    n = float(len(cells))
    means = {c: {L: sums[c][L] / n for L in layer_ids} for c in EXTRACTION_CONDITIONS}
    resid_norm = {L: norms[L] / max(counts, 1) for L in layer_ids}
    vectors = derive_vectors(means, layer_ids, args.seed)
    return {"means": means, "vectors": vectors, "resid_norm": resid_norm,
            "n_pairs": len(cells)}


def vector_diagnostics(vectors: dict[str, dict[int, Any]], resid_norm: dict[int, float],
                       layer_ids: Sequence[int]) -> list[dict[str, Any]]:
    """Per-layer norms and cosines, printed before a single token is generated.

    Two of these decide whether the controls mean anything:
      * ||v|| / mean||h|| says how big the write is relative to what is already
        there. A ratio near 0 means alpha=1 is a rounding error at this depth
        and a null there is uninformative.
      * cos(v_authority, v_unrelated) says whether the "unrelated" control is
        actually unrelated. If it is near 1, the two contrasts share a direction
        (probably "a system message is present at all"), the control cannot
        discriminate, and that must be said out loud rather than discovered
        later.
    """
    import torch

    def cos(a: Any, b: Any) -> float:
        na = torch.linalg.vector_norm(a).clamp_min(1e-12)
        nb = torch.linalg.vector_norm(b).clamp_min(1e-12)
        return float((a @ b) / (na * nb))

    rows = []
    for L in layer_ids:
        v = vectors[V_AUTHORITY][L]
        rows.append({
            "layer": L,
            "norm_authority": float(torch.linalg.vector_norm(v)),
            "norm_salience": float(torch.linalg.vector_norm(vectors[V_SALIENCE][L])),
            "norm_unrelated": float(torch.linalg.vector_norm(vectors[V_UNRELATED][L])),
            "mean_resid_norm": float(resid_norm[L]),
            "ratio_v_over_h": float(torch.linalg.vector_norm(v)) / max(resid_norm[L], 1e-12),
            "cos_authority_unrelated": cos(v, vectors[V_UNRELATED][L]),
            "cos_authority_salience": cos(v, vectors[V_SALIENCE][L]),
            "cos_authority_random": cos(v, vectors[V_RANDOM][L]),
        })
    return rows


# =========================================================================== #
# generation
# =========================================================================== #
def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Load the VLM in bf16 on GPU, plus a LEFT-padding tokenizer.

    Left padding is not optional. Decoder-only batched generation with right
    padding continues from pad tokens, and the hook's "last prompt position"
    rule (`hidden[:, -1, :]`) would land on padding for every sequence shorter
    than the longest. Both failure modes are silent.
    """
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    kw: dict[str, Any] = {"device_map": args.device_map}
    if args.attn_impl:
        kw["attn_implementation"] = args.attn_impl
    last: Exception | None = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        for dtype_key in ("dtype", "torch_dtype"):     # transformers 5 renamed this
            try:
                LOG.info("loading %s via %s (%s=bfloat16)", args.model, cls_name, dtype_key)
                model = cls.from_pretrained(args.model, **{dtype_key: torch.bfloat16}, **kw)
                model.eval()
                return model, tok, model.config
            except TypeError as exc:
                last = exc
            except Exception as exc:                   # noqa: BLE001
                last = exc
                break
    raise RuntimeError(f"could not load {args.model}: {last}")


def generate_condition(model: Any, tok: Any, layers_module: Any, cells: list[dict],
                       cond: Condition, vectors: dict[str, dict[int, Any]],
                       layer_types: Sequence[str], args: argparse.Namespace,
                       seed_offset: int) -> tuple[list[dict], dict[str, float]]:
    """Generate every rollout for one (vector, layer, alpha) condition.

    Sampling is `VLLM_SAMPLING`, with presence penalty supplied by our own logits
    processor. The hook is installed for the whole condition and its call counter
    is reset before every `generate` call, because "call 0 is the prefill" is the
    entire basis of the span policy and a stale counter would steer the prompt.
    """
    import torch
    from src.parse import parse_answer
    from src.serve import final_segment, to_prompt

    handles: list[Any] = []
    hook: SteerHook | None = None
    if not cond.is_baseline:
        vec = vectors[cond.vector][cond.layer]         # type: ignore[index]
        hook = SteerHook(vec.to(model.device), cond.alpha, span=args.steer_span)
        handles = install(layers_module, {cond.layer: hook})

    prompts = [to_prompt(c["text"]) for c in cells]    # NO system message, ever
    rows: list[dict] = []
    t0 = time.time()
    out_tokens = 0
    try:
        torch.manual_seed(args.seed + seed_offset)
        for rep in range(args.n):
            for start in range(0, len(prompts), args.batch_size):
                chunk_cells = cells[start:start + args.batch_size]
                chunk = prompts[start:start + args.batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True,
                          padding_side="left", add_special_tokens=False)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                prompt_len = int(enc["input_ids"].shape[1])
                proc = [make_presence_penalty_processor(
                    VLLM_SAMPLING["presence_penalty"], prompt_len, torch)]
                if hook is not None:
                    hook.reset()
                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        do_sample=True,
                        temperature=VLLM_SAMPLING["temperature"],
                        top_p=VLLM_SAMPLING["top_p"],
                        top_k=VLLM_SAMPLING["top_k"],
                        min_p=VLLM_SAMPLING["min_p"],
                        repetition_penalty=VLLM_SAMPLING["repetition_penalty"],
                        max_new_tokens=args.max_new_tokens,
                        logits_processor=proc,
                        pad_token_id=tok.pad_token_id,
                        use_cache=True,
                    )
                cont = out[:, prompt_len:]
                for cell, ids in zip(chunk_cells, cont):
                    ids_list = [int(t) for t in ids.tolist()]
                    n_out = _continuation_length(ids_list, tok)
                    text = tok.decode(ids_list[:n_out], skip_special_tokens=True)
                    truncated = n_out >= args.max_new_tokens
                    final = final_segment(text, thinking=True)
                    # `is not None`, never truthiness: an empty `final` is the
                    # deliberate "truncated mid-reasoning, no answer" signal.
                    est = parse_answer(final) if final is not None else None
                    out_tokens += n_out
                    rows.append(dict(
                        condition=cond.name, vector=cond.vector,
                        layer=(-1 if cond.layer is None else int(cond.layer)),
                        layer_type=("n/a" if cond.layer is None
                                    else str(layer_types[cond.layer])),
                        alpha=float(cond.alpha),
                        item_id=cell["item_id"], mapping=cell["mapping"],
                        paraphrase=int(cell["paraphrase"]),
                        threshold=float(cell["threshold"]), rollout_idx=rep,
                        text=text, final=final,
                        estimate=(None if est is None else float(est)),
                        parsed=est is not None,
                        n_output_tokens=int(n_out),
                        finish_reason=("length" if truncated else "stop"),
                        truncated=bool(truncated),
                        good_side=metrics.good_side(
                            est, float(cell["threshold"]), cell["mapping"]),
                    ))
    finally:
        remove(handles)

    dt = max(time.time() - t0, 1e-9)
    stats = {"seconds": dt, "out_tokens": float(out_tokens), "tok_per_sec": out_tokens / dt,
             "positions_steered": float(hook.positions_steered if hook else 0)}
    if hook is not None and hook.positions_steered == 0:
        LOG.error("condition %s installed a hook but steered ZERO positions — the span "
                  "policy or the layer index is wrong. Do not interpret this condition.",
                  cond.name)
    return rows, stats


def _continuation_length(ids: Sequence[int], tok: Any) -> int:
    """Number of generated tokens before the first EOS/pad, else all of them."""
    stops = {tok.eos_token_id, tok.pad_token_id}
    stops.discard(None)
    extra = getattr(tok, "eos_token_ids", None)
    if isinstance(extra, (list, tuple)):
        stops.update(int(t) for t in extra)
    for i, t in enumerate(ids):
        if t in stops:
            return i
    return len(ids)


# =========================================================================== #
# statistics
# =========================================================================== #
def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation; stdlib only, like everything in metrics.py."""
    n = len(xs)
    if n < 3 or len(ys) != n:
        return float("nan")

    def ranks(v: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: v[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def summarise_condition(rows: list[dict], base_rows: list[dict],
                        args: argparse.Namespace, base_tokens: float) -> dict[str, Any]:
    """Everything reported for one condition, plus whether it may be interpreted.

    The paired interval on `delta_p_good` is `paired_cluster_t_interval` against
    the alpha=0 rows: the two conditions run the SAME cells and the SAME
    paraphrases, so the paraphrase effect is common to both and cancels in the
    difference. Bootstrapping them independently would manufacture a
    paraphrase-composition mismatch the design does not have — the same argument
    Gate 2 makes for its paired bootstrap.
    """
    s = metrics.summarise(rows)
    lo_b, hi_b = metrics.cluster_bootstrap(rows, metrics.p_good, "paraphrase",
                                           args.n_boot, args.seed)
    lo_t, hi_t = metrics.cluster_t_interval(rows, metrics.p_good, "paraphrase")
    d = float("nan")
    d_lo, d_hi = float("nan"), float("nan")
    if base_rows and rows is not base_rows:
        d = s["p_good"] - metrics.p_good(base_rows)
        d_lo, d_hi = metrics.paired_cluster_t_interval(
            rows, base_rows, metrics.p_good, "paraphrase")

    reasons: list[str] = []
    if math.isnan(s["parse_rate"]) or s["parse_rate"] < MIN_PARSE_RATE:
        reasons.append(f"parse rate {_pct(s['parse_rate'])} < {_pct(MIN_PARSE_RATE)}")
    if (base_tokens > 0 and not math.isnan(s["mean_output_tokens"])
            and s["mean_output_tokens"] > DEGENERACY_TOKEN_RATIO * base_tokens):
        reasons.append(
            f"mean output tokens {s['mean_output_tokens']:.0f} > "
            f"{DEGENERACY_TOKEN_RATIO:g}x the unsteered {base_tokens:.0f} "
            "(text is running away)")
    return {
        **s,
        "k_clusters": len(metrics.iter_unique(rows, "paraphrase")),
        "p_good_ci_boot": (lo_b, hi_b),
        "p_good_ci_t": (lo_t, hi_t),
        "delta_p_good": d,
        "delta_ci_t": (d_lo, d_hi),
        "interpretable": not reasons,
        "not_interpretable_because": reasons,
    }


def dose_response(per_cond: dict[str, dict[str, Any]], conds: Sequence[Condition],
                  layer: int, vector: str, baseline: dict[str, Any]) -> dict[str, Any]:
    """Alpha sweep at one (vector, layer): points, Spearman rho, and the slope.

    `slope` is p_good(+a_max) - p_good(-a_max), the widest available contrast
    through zero. It is the quantity P1 is about, and its sign is the whole
    claim; `rho` is the quantity P2 is about. Non-interpretable conditions are
    EXCLUDED from both — a p_good computed off collapsing text is not a point on
    a dose-response curve, it is a different measurement wearing the same name.
    """
    pts: list[tuple[float, float]] = [(0.0, baseline["p_good"])]
    dropped: list[str] = []
    for c in conds:
        if c.vector != vector or c.layer != layer or c.alpha == 0.0:
            continue
        s = per_cond.get(c.name)
        if s is None:
            continue
        if not s["interpretable"]:
            dropped.append(c.name)
            continue
        pts.append((c.alpha, s["p_good"]))
    pts.sort()
    xs = [a for a, _ in pts]
    ys = [p for _, p in pts]
    slope = float("nan")
    if len(xs) >= 2 and xs[0] < 0 < xs[-1]:
        slope = ys[-1] - ys[0]
    return {"layer": layer, "vector": vector, "points": pts,
            "rho": spearman(xs, ys), "slope_max": slope,
            "n_points": len(pts), "dropped": dropped}


# =========================================================================== #
# reporting
# =========================================================================== #
def _pct(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def _pp(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.1f}pp"


def _f(x: float | None) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.4f}"


def report(rows: list[dict], conds: Sequence[Condition], layers: Sequence[int],
           alphas: Sequence[float], layer_types: Sequence[str],
           diagnostics: list[dict[str, Any]], control_layer: int,
           args: argparse.Namespace) -> dict[str, Any]:
    """Print every number, then score the pre-registration, then a verdict."""
    by_cond: dict[str, list[dict]] = {}
    for r in rows:
        by_cond.setdefault(r["condition"], []).append(r)

    base_name = condition_name(V_NONE, None, 0.0)
    base_rows = by_cond.get(base_name, [])
    if not base_rows:
        LOG.error("no alpha=0 rows: the internal control is missing, so every delta below "
                  "is against nothing. Re-run without --overwrite-skipping the baseline.")
    base_tokens = metrics.mean_tokens(base_rows) if base_rows else 0.0
    baseline = summarise_condition(base_rows, base_rows, args, 0.0) if base_rows else {
        "p_good": float("nan"), "parse_rate": float("nan"), "mean_output_tokens": 0.0,
        "p_good_ci_t": (float("nan"), float("nan")), "interpretable": False,
        "not_interpretable_because": ["no baseline rows"], "k_clusters": 0,
    }
    per_cond = {name: summarise_condition(rs, base_rows, args, base_tokens)
                for name, rs in by_cond.items()}
    per_cond[base_name] = baseline

    k = len(metrics.iter_unique(rows, "paraphrase"))
    a_max = max((abs(a) for a in alphas if a != 0.0), default=0.0)

    LOG.info("=" * 90)
    LOG.info("§10 — CONTRASTIVE ACTIVATION ADDITION on %s", args.model)
    LOG.info("=" * 90)
    for line in PREREGISTRATION.splitlines():
        LOG.info("%s", line)
    LOG.info("")
    LOG.info("SERVING PATH: HuggingFace transformers + forward hooks, NOT vLLM. "
             "Sampling matched to serve.default_sampling including a hand-written "
             "presence penalty (HF has none). alpha=0 is the bridge to Gate 1.")
    LOG.info("steer span=%s  alpha-scaling=%s  split=%s  cells=%d  n=%d  "
             "max_new_tokens=%d  clusters k=%d",
             args.steer_span, args.alpha_scaling, args.split, args.cells, args.n,
             args.max_new_tokens, k)

    # ---- vector diagnostics, before any behavioural number ------------------
    LOG.info("")
    LOG.info("--- VECTOR DIAGNOSTICS (do the controls mean anything?) ---")
    LOG.info("  %-5s %12s %12s %10s %12s %12s %12s", "layer", "||v_auth||",
             "mean||h||", "ratio", "cos(u,auth)", "cos(sal,auth)", "cos(rnd,auth)")
    for d in diagnostics:
        LOG.info("  %-5d %12.3f %12.3f %10.4f %12.4f %12.4f %12.4f",
                 d["layer"], d["norm_authority"], d["mean_resid_norm"],
                 d["ratio_v_over_h"], d["cos_authority_unrelated"],
                 d["cos_authority_salience"], d["cos_authority_random"])
    hot = [d for d in diagnostics if abs(d["cos_authority_unrelated"]) > 0.5]
    if hot:
        LOG.warning("  !! cos(v_authority, v_unrelated) > 0.5 at layer(s) %s. The two "
                    "contrasts share a direction — most likely 'a system message is "
                    "present at all' — so the UNRELATED control cannot discriminate "
                    "content from format there. Lean on random/shuffled instead and "
                    "say this in the writeup.", [d["layer"] for d in hot])
    tiny = [d for d in diagnostics if d["ratio_v_over_h"] < 0.01]
    if tiny:
        LOG.warning("  !! ||v|| is under 1%% of the mean residual norm at layer(s) %s. "
                    "alpha=1 there is a rounding error and a null is uninformative; "
                    "use --alpha-scaling norm or larger alphas.", [d["layer"] for d in tiny])

    # ---- the bridge ---------------------------------------------------------
    LOG.info("")
    LOG.info("--- alpha = 0: internal control AND bridge to Gate 1 ---")
    LOG.info("  p_good = %s  (Gate 1, vLLM, k=30: %s)   above=%s below=%s",
             _pct(baseline["p_good"]), _pct(BASELINE_P_GOOD_VLLM),
             _pct(baseline.get("p_good_above")), _pct(baseline.get("p_good_below")))
    LOG.info("  cluster-t 95%% CI [%s, %s] (k=%d)  parse=%s  trunc=%s  mean_out_tok=%.0f",
             _pct(baseline["p_good_ci_t"][0]), _pct(baseline["p_good_ci_t"][1]),
             baseline["k_clusters"], _pct(baseline["parse_rate"]),
             _pct(baseline.get("truncation_rate")), baseline["mean_output_tokens"])
    bridge_ok = (not math.isnan(baseline["p_good_ci_t"][0])
                 and baseline["p_good_ci_t"][0] <= BASELINE_P_GOOD_VLLM
                 <= baseline["p_good_ci_t"][1])
    if not bridge_ok:
        LOG.warning("  !! P6 FAILS: the unsteered HF number does not cover Gate 1's "
                    "%s. The serving paths differ (sampling, chat template, or the "
                    "presence-penalty reimplementation). Steering deltas WITHIN this "
                    "run remain valid — they share the path — but nothing here may be "
                    "compared to Gate 1's 56.1%% or Gate 2's 83.6/92.1.",
                    _pct(BASELINE_P_GOOD_VLLM))

    # ---- every condition ----------------------------------------------------
    LOG.info("")
    LOG.info("--- ALL CONDITIONS ---")
    LOG.info("  %-26s %6s %6s %8s %8s %8s %9s %22s %s", "condition", "layer", "alpha",
             "p_good", "parse", "trunc", "out_tok", "delta vs a=0 [95% t]", "")
    for c in sorted(conds, key=lambda c: (c.vector, c.layer or -1, c.alpha)):
        s = per_cond.get(c.name)
        if s is None:
            continue
        flag = "" if s["interpretable"] else "  <-- NOT INTERPRETED: " + "; ".join(
            s["not_interpretable_because"])
        LOG.info("  %-26s %6s %+6g %8s %8s %8s %9.0f  %8s [%7s,%7s]%s",
                 c.name, ("-" if c.layer is None else str(c.layer)), c.alpha,
                 _pct(s["p_good"]), _pct(s["parse_rate"]), _pct(s["truncation_rate"]),
                 s["mean_output_tokens"], _pp(s["delta_p_good"]),
                 _pp(s["delta_ci_t"][0]), _pp(s["delta_ci_t"][1]), flag)

    # ---- dose-response per layer -------------------------------------------
    LOG.info("")
    LOG.info("--- DOSE-RESPONSE in alpha, per layer (v_authority) ---")
    LOG.info("  predicted sign of d(p_good)/d(alpha) = %+d  (see the pre-registration)",
             PREDICTED_SIGN)
    curves = {L: dose_response(per_cond, conds, L, V_AUTHORITY, baseline) for L in layers}
    for L in layers:
        cv = curves[L]
        pts = "  ".join(f"a={a:+g}:{_pct(p)}" for a, p in cv["points"])
        LOG.info("  L%-3d [%-16s] rho=%s slope(+a_max - -a_max)=%s  %s",
                 L, layer_types[L], _f(cv["rho"]), _pp(cv["slope_max"]), pts)
        if cv["dropped"]:
            LOG.info("        dropped from the curve (not interpretable): %s",
                     ", ".join(cv["dropped"]))

    # ---- layer TYPE breakdown ----------------------------------------------
    LOG.info("")
    LOG.info("--- EFFECT BY LAYER TYPE (this architecture's own question) ---")
    LOG.info("  Qwen3.5-4B is hybrid: %d Gated-DeltaNet linear_attention + %d "
             "full_attention layers. Does the authority direction live in one kind?",
             sum(1 for t in layer_types if t != FULL), sum(1 for t in layer_types if t == FULL))
    by_type: dict[str, list[float]] = {}
    for L in layers:
        s_ = curves[L]["slope_max"]
        if not math.isnan(s_):
            by_type.setdefault(str(layer_types[L]), []).append(s_)
    for t, vals in sorted(by_type.items()):
        lo, hi = metrics.t_interval(vals)
        mean = sum(vals) / len(vals)
        LOG.info("  %-18s n_layers=%d  mean slope=%s  95%% t over layers [%s, %s]",
                 t, len(vals), _pp(mean), _pp(lo), _pp(hi))
    if len(by_type) == 2:
        (ta, va), (tb, vb) = sorted(by_type.items())
        LOG.info("  difference of means (%s - %s) = %s. NOTE: layers are not "
                 "randomised across types — the 3:1 pattern fixes which depths are "
                 "which — so a type difference is confounded with depth. Read it "
                 "against the per-layer curve above, not on its own.",
                 ta, tb, _pp(sum(va) / len(va) - sum(vb) / len(vb)))
    depth_flat = ([abs(curves[L]["slope_max"]) for L in layers
                   if not math.isnan(curves[L]["slope_max"])])
    if len(depth_flat) >= 3 and max(depth_flat) > 0:
        if min(depth_flat) / max(depth_flat) > 0.8:
            LOG.warning("  !! the effect is nearly FLAT across depth (min/max of |slope| "
                        "= %.2f). A direction that steers equally from every layer, "
                        "including the extremes, looks more like a generic perturbation "
                        "of the output distribution than a localised representation. "
                        "This is one of the pre-registered falsifiers.",
                        min(depth_flat) / max(depth_flat))

    # ---- controls -----------------------------------------------------------
    LOG.info("")
    LOG.info("--- CONTROLS at layer %d (the most important block in this report) ---",
             control_layer)
    control_summary: dict[str, dict[str, Any]] = {}
    for c in sorted(conds, key=lambda c: (c.vector, c.alpha)):
        if c.vector in (V_AUTHORITY, V_NONE):
            continue
        s = per_cond.get(c.name)
        if s is None:
            continue
        control_summary[c.name] = {"vector": c.vector, "alpha": c.alpha,
                                   "delta": s["delta_p_good"], "ci": s["delta_ci_t"],
                                   "interpretable": s["interpretable"]}
        kind = "POSITIVE control (should move)" if c.vector == V_SALIENCE else \
               "negative control (should NOT move)"
        LOG.info("  %-26s a=%+g  delta=%s [%s, %s]   %s%s", c.name, c.alpha,
                 _pp(s["delta_p_good"]), _pp(s["delta_ci_t"][0]), _pp(s["delta_ci_t"][1]),
                 kind, "" if s["interpretable"] else "  <-- NOT INTERPRETED")

    # ---- scorecard ----------------------------------------------------------
    scorecard = score_prereg(per_cond, curves, control_summary, baseline, layers,
                             a_max, bridge_ok)
    LOG.info("")
    LOG.info("--- PRE-REGISTRATION SCORECARD ---")
    for key, res in scorecard["clauses"].items():
        LOG.info("  %-4s %-14s %s", key, res["verdict"], res["detail"])

    bar = "#" * 90
    LOG.info("")
    for line in (bar, f"###  VERDICT: {scorecard['verdict']}", bar):
        LOG.info("%s", line)
    for line in scorecard["action"].split(". "):
        if line.strip():
            LOG.info("###  %s", line.strip().rstrip(".") + ".")
    LOG.info("%s", bar)

    return {
        "model": args.model, "preset": args.preset, "n_rollouts": len(rows),
        "k_clusters": k, "layers": list(layers), "alphas": list(alphas),
        "layer_types": {int(L): str(layer_types[L]) for L in layers},
        "control_layer": control_layer,
        "steer_span": args.steer_span, "alpha_scaling": args.alpha_scaling,
        "split": args.split, "cells": args.cells, "n": args.n,
        "max_new_tokens": args.max_new_tokens, "seed": args.seed,
        "baseline_p_good": baseline["p_good"],
        "gate1_p_good_vllm": BASELINE_P_GOOD_VLLM, "bridge_ok": bridge_ok,
        "prereg": PREREGISTRATION, "predicted_sign": PREDICTED_SIGN,
        "vector_diagnostics": diagnostics,
        "per_condition": {n: {k2: v for k2, v in s.items()} for n, s in per_cond.items()},
        "dose_response": {int(L): curves[L] for L in layers},
        "controls": control_summary,
        "scorecard": scorecard,
    }


def score_prereg(per_cond: dict[str, dict[str, Any]], curves: dict[int, dict[str, Any]],
                 controls: dict[str, dict[str, Any]], baseline: dict[str, Any],
                 layers: Sequence[int], a_max: float,
                 bridge_ok: bool) -> dict[str, Any]:
    """Score P1-P6 mechanically. The script grades itself; no eyeballing."""
    clauses: dict[str, dict[str, str]] = {}

    # P1 direction
    usable = {L: c for L, c in curves.items() if not math.isnan(c["slope_max"])}
    best_L, best = None, None
    for L, c in usable.items():
        if best is None or abs(c["slope_max"]) > abs(best["slope_max"]):
            best_L, best = L, c
    if best is None:
        clauses["P1"] = {"verdict": "INCONCLUSIVE",
                         "detail": "no layer produced an interpretable slope through zero."}
    else:
        sign_ok = (best["slope_max"] < 0) == (PREDICTED_SIGN < 0)
        pos_name = condition_name(V_AUTHORITY, best_L, a_max)
        neg_name = condition_name(V_AUTHORITY, best_L, -a_max)
        excl = False
        for nm in (pos_name, neg_name):
            s = per_cond.get(nm)
            if s and not math.isnan(s["delta_ci_t"][0]):
                excl = excl or (s["delta_ci_t"][0] > 0 or s["delta_ci_t"][1] < 0)
        # Significance is checked BEFORE sign. The sign of a slope that is not
        # separated from zero is the sign of noise, and calling that "WRONG
        # SIGN" would turn a plain null into a spurious directional finding —
        # exactly the mistake the pre-registration exists to prevent. A null
        # run has a 50% chance of a wrong-signed noise slope.
        clauses["P1"] = {
            "verdict": ("WEAK" if not excl else "PASS" if sign_ok else "WRONG SIGN"),
            "detail": (f"largest |slope| at L{best_L}: {_pp(best['slope_max'])} "
                       f"(predicted sign {PREDICTED_SIGN:+d}); a paired interval on at "
                       f"least one extreme alpha {'excludes' if excl else 'includes'} 0."
                       + ("" if excl else " Not separated from zero, so its sign is not "
                          "interpreted."))}

    # P2 monotonicity
    if best is None or math.isnan(best["rho"]):
        clauses["P2"] = {"verdict": "INCONCLUSIVE", "detail": "no usable dose-response curve."}
    else:
        ok = abs(best["rho"]) >= MIN_ABS_RHO and (best["rho"] < 0) == (PREDICTED_SIGN < 0)
        clauses["P2"] = {
            "verdict": "PASS" if ok else "FAIL",
            "detail": (f"Spearman rho over {best['n_points']} alphas at L{best_L} = "
                       f"{_f(best['rho'])} (bar |rho| >= {MIN_ABS_RHO}, predicted sign "
                       f"{PREDICTED_SIGN:+d}). A one-sided or non-monotone response here "
                       "is a real negative result about the direction, not a broken run.")}

    # P3 negative controls
    negs = {n: c for n, c in controls.items() if c["vector"] != V_SALIENCE}
    if not negs:
        clauses["P3"] = {"verdict": "MISSING",
                         "detail": "NO NEGATIVE CONTROL WAS RUN. Nothing below can be "
                                   "distinguished from generic activation damage."}
    else:
        bad = [n for n, c in negs.items()
               if not math.isnan(c["delta"])
               and (abs(100 * c["delta"]) > CONTROL_TOL_PP
                    or (not math.isnan(c["ci"][0]) and (c["ci"][0] > 0 or c["ci"][1] < 0)))]
        clauses["P3"] = {
            "verdict": "PASS" if not bad else "FAIL",
            "detail": (f"{len(negs)} negative control condition(s); "
                       + ("none moved p_good by more than "
                          f"{CONTROL_TOL_PP:.0f}pp with an interval excluding 0."
                          if not bad else
                          f"MOVED: {', '.join(bad)}. A random or shuffled vector that "
                          "shifts p_good means the steering result is a perturbation "
                          "effect, not a direction."))}

    # P4 positive control
    sal = {n: c for n, c in controls.items() if c["vector"] == V_SALIENCE and c["alpha"] > 0}
    if not sal:
        clauses["P4"] = {"verdict": "NOT RUN",
                         "detail": "v_salience not in --controls; the 'can we write "
                                   "anything at all' check is absent."}
    else:
        moved = any(not math.isnan(c["delta"]) and c["delta"] > 0
                    and abs(100 * c["delta"]) > CONTROL_TOL_PP for c in sal.values())
        clauses["P4"] = {
            "verdict": "PASS" if moved else "FAIL",
            "detail": ("v_salience at +alpha raises p_good as Gate 2's main effect "
                       "predicts." if moved else
                       "v_salience does NOT raise p_good. The write may not be landing "
                       "at all, which makes P1-P3 uninformative rather than negative — "
                       "check positions_steered and the layer path before concluding.")}

    # P5 parse rates
    dead = [n for n, s in per_cond.items() if not s["interpretable"]]
    clauses["P5"] = {
        "verdict": "PASS" if not dead else "PARTIAL",
        "detail": (f"all conditions parse >= {_pct(MIN_PARSE_RATE)}." if not dead else
                   f"{len(dead)} condition(s) refused interpretation: "
                   f"{', '.join(sorted(dead)[:6])}"
                   f"{' ...' if len(dead) > 6 else ''}. Reported, excluded from every "
                   "curve and every control test.")}

    # P6 bridge
    clauses["P6"] = {
        "verdict": "PASS" if bridge_ok else "FAIL",
        "detail": (f"unsteered HF p_good {_pct(baseline['p_good'])} "
                   f"{'covers' if bridge_ok else 'does NOT cover'} Gate 1's "
                   f"{_pct(BASELINE_P_GOOD_VLLM)}.")}

    v1, v3 = clauses["P1"]["verdict"], clauses["P3"]["verdict"]
    if v1 == "PASS" and v3 == "PASS":
        verdict = "MECHANISM SUPPORTED"
        action = (
            "A linear direction extracted from the GA/GS contrast causally moves p_good in "
            "the pre-registered direction, with a dose-response through zero, while "
            "norm-matched random, coordinate-shuffled and unrelated-contrast vectors of the "
            "same size do not. That is a causal claim about a representation, not a "
            "correlation. Report the layer profile and the type breakdown as the "
            "localisation, and state plainly that CAA shows a SUFFICIENT direction — it does "
            "not show this direction is the one the system message actually uses, which "
            "would need ablation or patching")
    elif v1 == "PASS" and v3 == "FAIL":
        verdict = "ARTEFACT — DO NOT REPORT AS A MECHANISM"
        action = (
            "The authority vector moved p_good, but so did a control that carries no "
            "authority information. The most parsimonious reading is that perturbing the "
            "residual stream at this layer and magnitude degrades or biases generation "
            "generally. Lower alpha until every negative control is flat, then re-read P1 "
            "at that alpha; if the authority effect vanishes with them, there is no "
            "direction here")
    elif v1 == "WRONG SIGN":
        verdict = "EFFECT IN THE WRONG DIRECTION"
        action = (
            "The steering effect is real but runs opposite to the pre-registration. Do not "
            "retro-fit the prediction. Check the sign convention end to end — v = h(GA) - "
            "h(GS), and Gate 2 measured p_good(GA) < p_good(GS), so +v should LOWER p_good "
            "— and check that the read position and the write span are the same token. If "
            "the convention is right, this is a genuine and interesting negative: the "
            "readable direction is not the one behaviour uses")
    elif clauses["P4"]["verdict"] == "FAIL":
        verdict = "WRITE NOT LANDING — INCONCLUSIVE"
        action = (
            "Even the salience direction, which Gate 2 shows is a 28-36 point behavioural "
            "effect, does not move p_good when written in. That points at the intervention, "
            "not at the model: check positions_steered > 0, that the resolved layer path is "
            "the TEXT stack and not the vision tower, and that ||alpha*v|| is not negligible "
            "against the mean residual norm. Nothing about the authority direction can be "
            "concluded until this passes")
    else:
        verdict = "NULL — NO STEERABLE DIRECTION FOUND"
        action = (
            "The contrast is readable as a mean difference but writing it back does not move "
            "the behaviour, at any swept layer or alpha, with controls behaving. Taken with "
            "a passing positive control this is an informative negative: the behavioural "
            "GA/GS effect is not mediated by a single linear direction at a single layer at "
            "the last prompt token. Next cheapest discriminating experiments, in cost order: "
            "steer at more positions; extract at the answer token instead of the prompt "
            "token; try a difference-in-means over several layers at once; then activation "
            "patching, which tests necessity rather than sufficiency")

    return {"clauses": clauses, "verdict": verdict, "action": action,
            "best_layer": best_L}


# =========================================================================== #
# dry run
# =========================================================================== #
def dry_run(args: argparse.Namespace, eval_cells: list[dict], extract_cells: list[dict],
            layers: Sequence[int], layer_types: Sequence[str], conds: Sequence[Condition],
            control_layer: int, shard_dir: Path) -> None:
    n_rollouts = len(conds) * len(eval_cells) * args.n
    gen_tok = n_rollouts * min(TYPICAL_OUT_TOKENS, args.max_new_tokens)
    worst_tok = n_rollouts * args.max_new_tokens
    n_extract_fwd = len(extract_cells) * len(EXTRACTION_CONDITIONS)
    extract_tok = n_extract_fwd * 900          # ~900 prompt tokens per bet prompt
    gen_min = gen_tok / HF_TOK_PER_SEC / 60
    extract_min = extract_tok / EXTRACT_PREFILL_TOK_PER_SEC / 60

    print("=" * 90)
    print("DRY RUN — no model loaded, no GPU time spent")
    print("=" * 90)
    print(PREREGISTRATION)
    print("-" * 90)
    print("EXTRACTION PLAN (CAA step 1)")
    print("-" * 90)
    print(f"  model              : {args.model}")
    print(f"  read position      : last prompt token (the position token 1 is generated from)")
    print(f"  read site          : output of decoder layer L (NOT output_hidden_states, "
          f"which applies the final norm at the last layer)")
    print(f"  contrast pairs     : {len(extract_cells)} cells x "
          f"{len(EXTRACTION_CONDITIONS)} conditions = {n_extract_fwd} forward passes")
    print(f"  conditions         : {', '.join(EXTRACTION_CONDITIONS)}")
    print(f"  split              : --split {args.split}  "
          f"(extract on {len(metrics.iter_unique(extract_cells,'item_id'))} items, "
          f"evaluate on {len(metrics.iter_unique(eval_cells,'item_id'))} items"
          + ("; DISJOINT" if args.split == SPLIT_ITEM else "; OVERLAPPING — leakage") + ")")
    print(f"  vectors derived    : {V_AUTHORITY} = mean(h_GA - h_GS)")
    print(f"                       {V_SALIENCE}  = mean((h_GA+h_GS)/2 - h_NONE)   [POSITIVE control]")
    print(f"                       {V_UNRELATED} = mean(h_UA - h_UB)              [negative control]")
    print(f"                       {V_RANDOM}    = gaussian, norm-matched          [negative control]")
    print(f"                       {V_SHUFFLED}  = coords of v_authority permuted  [negative control]")
    print(f"  alpha scaling      : {args.alpha_scaling}")
    print()
    print("-" * 90)
    print("LAYERS")
    print("-" * 90)
    print(f"  text stack depth   : {len(layer_types)} "
          f"({sum(1 for t in layer_types if t != FULL)} {LINEAR} + "
          f"{sum(1 for t in layer_types if t == FULL)} {FULL})")
    print(f"  source of types    : {'config (verified at load)' if args.verified_types else 'ASSUMED 3:1 pattern — verified on the box at load time'}")
    print(f"  selected (--layers {args.layers}) : {list(layers)}")
    for L in layers:
        print(f"      L{L:<3d} {layer_types[L]}   depth {L / max(len(layer_types) - 1, 1):.0%}")
    print(f"  control layer      : L{control_layer} ({layer_types[control_layer]})")
    print()
    print("-" * 90)
    print("STEERING GRID (CAA step 2)")
    print("-" * 90)
    print(f"  alphas             : {list(args.alphas)}   (alpha=0 is the internal control "
          f"and is run ONCE, not once per layer)")
    print(f"  control alphas     : {args.control_alphas_resolved}")
    print(f"  span               : --steer-span {args.steer_span}")
    print(f"  conditions         : {len(conds)}")
    for c in sorted(conds, key=lambda c: (c.vector, c.layer or -1, c.alpha)):
        print(f"      {c.name:<28} vector={c.vector:<10} layer="
              f"{'-' if c.layer is None else c.layer:<3} alpha={c.alpha:+g}")
    print()
    print("-" * 90)
    print("EVALUATION GRID")
    print("-" * 90)
    print(f"  cells              : {len(eval_cells)} "
          f"(mapping-balanced, cluster-count-maximising round robin; see subsample)")
    print(f"  items              : {len(metrics.iter_unique(eval_cells, 'item_id'))}")
    print(f"  mappings           : {metrics.iter_unique(eval_cells, 'mapping')}")
    print(f"  paraphrases        : {len(metrics.iter_unique(eval_cells, 'paraphrase'))} "
          f"-> bootstrap clusters k={len(metrics.iter_unique(eval_cells, 'paraphrase'))}")
    print(f"  n per cell         : {args.n}")
    print(f"  ROLLOUTS           : {n_rollouts}  ({len(conds)} conditions x "
          f"{len(eval_cells)} cells x n={args.n})")
    k = len(metrics.iter_unique(eval_cells, "paraphrase"))
    if k < 5:
        print(f"  !! k={k} clusters. Below 5 the cluster-t interval is nearly vacuous "
              f"(t({max(k-1,1)}) is huge) and NOTHING here is a statistical result. "
              f"Fine for --preset smoke, not for anything reported.")
    per_cond_rollouts = len(eval_cells) * args.n
    if per_cond_rollouts < 40:
        print(f"  !! {per_cond_rollouts} rollouts per condition. p_good will be quantised "
              f"in steps of ~{100 / max(per_cond_rollouts // 2, 1):.1f}pp per mapping; a "
              f"{CONTROL_TOL_PP:.0f}pp control tolerance is close to one step.")
    print()
    print("-" * 90)
    print("WALL CLOCK")
    print("-" * 90)
    print(f"  MEASURED   trace length  : {TYPICAL_OUT_TOKENS} output tokens (vLLM, FINDINGS)")
    print(f"  ESTIMATED  HF throughput : {HF_TOK_PER_SEC:,.0f} tok/s aggregate at "
          f"batch {args.batch_size}. THIS IS A GUESS — vLLM measured 7,745 tok/s and HF "
          f"eager on a hybrid cache is assumed ~8.6x slower. The run re-estimates it from "
          f"the first completed condition and re-projects.")
    print(f"  model load               : ~{MODEL_LOAD_MIN:.1f} min")
    print(f"  extraction               : ~{extract_min:.1f} min "
          f"({n_extract_fwd} prefills, all layers captured in one pass)")
    print(f"  generation (typical)     : ~{gen_min:.0f} min "
          f"({gen_tok:,} output tokens)")
    print(f"  TOTAL (typical)          : ~{MODEL_LOAD_MIN + extract_min + gen_min:.0f} min "
          f"= {(MODEL_LOAD_MIN + extract_min + gen_min) / 60:.1f} h")
    print(f"  TOTAL (every rollout hits the {args.max_new_tokens} cap): "
          f"~{worst_tok / HF_TOK_PER_SEC / 60:.0f} min")
    print()
    print("  other presets, same estimator:")
    for name in PRESETS:
        p = PRESETS[name]
        n_layers_p = len(parse_layer_spec(p["layers"], len(layer_types), layer_types))
        ca = resolve_control_alphas(p["control_alphas"], p["alphas"])
        n_conds = 1 + n_layers_p * len([a for a in p["alphas"] if a]) + len(p["controls"]) * len(ca)
        n_roll = n_conds * p["cells"] * p["n"]
        mins = n_roll * TYPICAL_OUT_TOKENS / HF_TOK_PER_SEC / 60 + MODEL_LOAD_MIN
        print(f"    {name:<7} {n_conds:>4} conditions x {p['cells']:>3} cells x n={p['n']} "
              f"= {n_roll:>6} rollouts  ~{mins:>6.0f} min ({mins / 60:.1f} h)")
    print()
    print(f"  shards would be written per-condition under: {shard_dir}")
    print(f"  (a killed run resumes: existing per-condition parquet files are skipped)")
    print()
    print("-" * 90)
    print("SYSTEM MESSAGES USED FOR EXTRACTION ONLY (generation uses NONE)")
    print("-" * 90)
    for name, msg in EXTRACTION_CONDITIONS.items():
        print(f"  {name:<5}: {msg if msg is not None else '<no system message>'}")
    print(f"  GA/GS mirror : word multiset "
          f"{sorted(PROMPT_GA.split()) == sorted(PROMPT_GS.split())}, "
          f"length {len(PROMPT_GA) == len(PROMPT_GS)}")
    print(f"  UA/UB mirror : word multiset "
          f"{sorted(PROMPT_UA.split()) == sorted(PROMPT_UB.split())}, "
          f"length {len(PROMPT_UA) == len(PROMPT_UB)}")
    print()
    print("-" * 90)
    print("SAMPLING — matched to src/serve.default_sampling")
    print("-" * 90)
    for k2, v2 in VLLM_SAMPLING.items():
        note = ("  <- HF has no such parameter; supplied by PresencePenaltyProcessor"
                if k2 == "presence_penalty" else "")
        print(f"  {k2:<20} {v2}{note}")
    print(f"  {'max_new_tokens':<20} {args.max_new_tokens}")
    first = eval_cells[0]
    print()
    print("-" * 90)
    print(f"FIRST EVAL PROMPT (item_id={first['item_id']}, mapping={first['mapping']}, "
          f"paraphrase={first['paraphrase']})")
    print("-" * 90)
    print(first["text"])
    try:
        from src.serve import to_prompt
        print("-" * 90)
        print("CHAT-TEMPLATED, no system message (exactly Gate 1's rendering):")
        print("-" * 90)
        print(to_prompt(first["text"]))
    except Exception as exc:                            # noqa: BLE001 - informational
        print(f"\n[chat template not rendered: {type(exc).__name__}: {exc}]")
        print("[fine off-GPU; it only means the tokenizer is unavailable here]")
    print("=" * 90)


# =========================================================================== #
# self test
# =========================================================================== #
def self_test() -> int:
    """Exercise every piece of logic without the real checkpoint.

    Covers: layer resolution on a nested VLM-shaped tree; layer/type selection;
    hook install/remove and the exact arithmetic of the write, per span policy;
    the mean-difference vector; the three negative controls' invariants; the
    presence-penalty processor against vLLM's definition; end-to-end generation
    through hooks on a tiny random-weights model; and the whole reporting and
    scoring path on synthetic rows with a KNOWN answer, including the refusal to
    interpret a collapsed condition.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("SKIP: torch is not installed; the self-test needs it. "
              "Structural checks only:")
        check_mirror()
        types = assumed_layer_types()
        assert parse_layer_spec("auto:8", 32, types) and len(
            parse_layer_spec("auto:8", 32, types)) == 8
        print("ok    mirror prompts + layer selection")
        return 0

    fails: list[str] = []

    def check(name: str, cond: bool, extra: str = "") -> None:
        print(f"{'ok  ' if cond else 'FAIL'}  {name}{('  ' + extra) if extra else ''}")
        if not cond:
            fails.append(name)

    # ---------------- prompts ------------------------------------------------
    check_mirror()
    check("GA/GS and UA/UB are word-for-word mirrors", True)
    check("extraction has a no-system-message condition",
          EXTRACTION_CONDITIONS["NONE"] is None)

    # ---------------- layer typing and selection -----------------------------
    types = assumed_layer_types(32)
    check("assumed layer types: 24 linear + 8 full",
          types.count(LINEAR) == 24 and types.count(FULL) == 8)
    sel8 = parse_layer_spec("auto:8", 32, types)
    check("auto:8 picks 8 layers", len(sel8) == 8, str(sel8))
    check("auto:8 is balanced across layer types",
          sum(1 for L in sel8 if types[L] == FULL) == 4, str([types[L] for L in sel8]))
    check("auto:8 spans depth", min(sel8) < 8 and max(sel8) > 23, str(sel8))
    check("explicit spec parses", parse_layer_spec("3, 7,11", 32, types) == [3, 7, 11])
    check("all spec parses", parse_layer_spec("all", 32, types) == list(range(32)))
    try:
        parse_layer_spec("99", 32, types)
        check("out-of-range layer rejected", False)
    except SystemExit:
        check("out-of-range layer rejected", True)

    class _FakeCfg:
        pass

    cfg = _FakeCfg()
    cfg.text_config = _FakeCfg()                       # type: ignore[attr-defined]
    cfg.text_config.layer_types = [FULL] * 32          # type: ignore[attr-defined]
    check("resolve_layer_types prefers the NESTED text_config",
          resolve_layer_types(cfg, 32).count(FULL) == 32)
    check("resolve_layer_types falls back on a bad length",
          resolve_layer_types(_FakeCfg(), 32) == assumed_layer_types(32))

    # ---------------- layer resolution on a nested VLM-shaped tree -----------
    class _Blk(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, x):                          # noqa: ANN001
            return (self.lin(x),)

    class _Text(nn.Module):
        def __init__(self, d: int, n: int) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_Blk(d) for _ in range(n)])

    class _Inner(nn.Module):
        def __init__(self, d: int, n: int) -> None:
            super().__init__()
            self.visual = nn.ModuleList([_Blk(d) for _ in range(99)])   # decoy, longer
            self.language_model = _Text(d, n)

    class _VLM(nn.Module):
        def __init__(self, d: int, n: int) -> None:
            super().__init__()
            self.model = _Inner(d, n)

    vlm = _VLM(8, 6)
    mod, path = resolve_text_layers(vlm)
    check("resolve_text_layers finds the TEXT stack, not the vision tower",
          path == "model.language_model.layers" and len(mod) == 6, path)

    # ---------------- hook arithmetic and span policy ------------------------
    d = 8
    blk = _Blk(d)
    v = torch.arange(d, dtype=torch.float32)
    x_prefill = torch.randn(2, 5, d)
    x_decode = torch.randn(2, 1, d)
    with torch.no_grad():
        base_pre = blk(x_prefill)[0].clone()
        base_dec = blk(x_decode)[0].clone()

    for span, want_prefill_rows in ((SPAN_FROM_LAST_PROMPT, "last"),
                                    (SPAN_GENERATED_ONLY, "none"),
                                    (SPAN_ALL, "all")):
        h = SteerHook(v, alpha=2.0, span=span)
        handle = blk.register_forward_hook(h)
        with torch.no_grad():
            got_pre = blk(x_prefill)[0]
            got_dec = blk(x_decode)[0]
        handle.remove()
        delta_pre = got_pre - base_pre
        delta_dec = got_dec - base_dec
        if want_prefill_rows == "last":
            ok = (torch.allclose(delta_pre[:, -1, :], 2.0 * v)
                  and torch.allclose(delta_pre[:, :-1, :], torch.zeros(2, 4, d)))
        elif want_prefill_rows == "none":
            ok = torch.allclose(delta_pre, torch.zeros_like(delta_pre))
        else:
            ok = torch.allclose(delta_pre, (2.0 * v).expand_as(delta_pre))
        check(f"span={span}: prefill write is correct", bool(ok))
        check(f"span={span}: decode write is alpha*v everywhere",
              bool(torch.allclose(delta_dec, (2.0 * v).expand_as(delta_dec))))
        check(f"span={span}: hook counted positions",
              h.positions_steered > 0 or span == SPAN_GENERATED_ONLY)

    h0 = SteerHook(v, alpha=0.0)
    handle = blk.register_forward_hook(h0)
    with torch.no_grad():
        got = blk(x_prefill)[0]
    handle.remove()
    check("alpha=0 hook is an exact no-op", bool(torch.equal(got, base_pre)))

    h = SteerHook(v, alpha=1.0)
    handle = blk.register_forward_hook(h)
    with torch.no_grad():
        blk(x_prefill)
        blk(x_decode)
    handle.remove()
    calls_before = h.calls
    h.reset()
    check("hook reset clears the prefill counter", calls_before == 2 and h.calls == 0)
    with torch.no_grad():
        after_removal = blk(x_prefill)[0]
    check("removing the handle restores the module exactly",
          bool(torch.equal(after_removal, base_pre)))

    check("bare-tensor layer outputs are handled",
          _split_output(torch.zeros(1))[0].shape == torch.Size([1]))
    check("n-tuple layer outputs are rebuilt with their extras",
          _split_output((torch.zeros(1), "cache"))[1](torch.ones(1))[1] == "cache")

    # ---------------- capture hook -------------------------------------------
    cap = CaptureHook()
    handle = blk.register_forward_hook(cap)
    with torch.no_grad():
        out = blk(x_prefill)[0]
    handle.remove()
    check("CaptureHook takes the LAST token of the layer OUTPUT",
          bool(torch.allclose(cap.last, out[:, -1, :].float())))

    # ---------------- vector arithmetic --------------------------------------
    L = 0
    means = {"GA": {L: torch.tensor([3.0, 0.0, 0.0, 4.0])},
             "GS": {L: torch.tensor([1.0, 0.0, 0.0, 1.0])},
             "NONE": {L: torch.tensor([0.0, 0.0, 0.0, 0.0])},
             "UA": {L: torch.tensor([0.0, 5.0, 0.0, 0.0])},
             "UB": {L: torch.tensor([0.0, 2.0, 0.0, 0.0])}}
    vecs = derive_vectors(means, [L], seed=0)
    check("v_authority == mean(h_GA) - mean(h_GS)",
          bool(torch.allclose(vecs[V_AUTHORITY][L], torch.tensor([2.0, 0.0, 0.0, 3.0]))))
    check("v_salience == mean of GA,GS minus NONE",
          bool(torch.allclose(vecs[V_SALIENCE][L], torch.tensor([2.0, 0.0, 0.0, 2.5]))))
    check("v_unrelated == mean(h_UA) - mean(h_UB)",
          bool(torch.allclose(vecs[V_UNRELATED][L], torch.tensor([0.0, 3.0, 0.0, 0.0]))))
    n_auth = float(torch.linalg.vector_norm(vecs[V_AUTHORITY][L]))
    check("random control is NORM-MATCHED to v_authority",
          abs(float(torch.linalg.vector_norm(vecs[V_RANDOM][L])) - n_auth) < 1e-4)
    check("random control is not parallel to v_authority",
          abs(float(torch.nn.functional.cosine_similarity(
              vecs[V_RANDOM][L], vecs[V_AUTHORITY][L], dim=0))) < 0.999)
    check("shuffled control is a permutation of v_authority",
          sorted(vecs[V_SHUFFLED][L].tolist()) == sorted(vecs[V_AUTHORITY][L].tolist()))
    check("shuffled control has identical norm",
          abs(float(torch.linalg.vector_norm(vecs[V_SHUFFLED][L])) - n_auth) < 1e-5)

    scaled = scale_vectors(vecs, {L: 10.0}, "norm")
    check("--alpha-scaling norm rescales to the mean residual norm",
          abs(float(torch.linalg.vector_norm(scaled[V_AUTHORITY][L])) - 10.0) < 1e-4)
    check("--alpha-scaling norm preserves direction",
          float(torch.nn.functional.cosine_similarity(
              scaled[V_AUTHORITY][L], vecs[V_AUTHORITY][L], dim=0)) > 0.999)
    check("--alpha-scaling raw is the identity",
          scale_vectors(vecs, {L: 10.0}, "raw") is vecs)

    # ---------------- presence penalty ---------------------------------------
    proc = make_presence_penalty_processor(1.5, prompt_len=3, torch_mod=torch)
    ids = torch.tensor([[9, 9, 9, 1, 2, 2]])           # prompt=9,9,9; generated=1,2,2
    scores = torch.zeros(1, 10)
    out_s = proc(ids, scores)
    check("presence penalty hits generated tokens once each",
          float(out_s[0, 1]) == -1.5 and float(out_s[0, 2]) == -1.5)
    check("presence penalty is presence, not frequency (token 2 appeared twice)",
          float(out_s[0, 2]) == -1.5)
    check("presence penalty ignores PROMPT tokens (vLLM semantics)",
          float(out_s[0, 9]) == 0.0)
    check("presence penalty is a no-op before any token is generated",
          bool(torch.equal(proc(torch.tensor([[9, 9, 9]]), scores), scores)))

    # ---------------- end-to-end generation through hooks --------------------
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    tiny = LlamaForCausalLM(cfg).eval()
    tiny_layers, tiny_path = resolve_text_layers(tiny)
    check("resolve_text_layers works on a plain causal LM",
          len(tiny_layers) == 4, tiny_path)
    ids = torch.randint(0, 64, (2, 6))
    steer = SteerHook(torch.randn(32) * 5.0, alpha=3.0)
    handles = install(tiny_layers, {2: steer})
    torch.manual_seed(0)
    with torch.no_grad():
        gen_steered = tiny.generate(input_ids=ids, do_sample=True, temperature=1.0,
                                    top_k=20, max_new_tokens=8, pad_token_id=0,
                                    logits_processor=[make_presence_penalty_processor(
                                        1.5, 6, torch)])
    remove(handles)
    torch.manual_seed(0)
    with torch.no_grad():
        gen_plain = tiny.generate(input_ids=ids, do_sample=True, temperature=1.0,
                                  top_k=20, max_new_tokens=8, pad_token_id=0,
                                  logits_processor=[make_presence_penalty_processor(
                                      1.5, 6, torch)])
    check("generate() runs with hooks + presence penalty installed",
          gen_steered.shape == gen_plain.shape == torch.Size([2, 14]))
    check("the hook actually fired during generate (prefill + 7 decode steps)",
          steer.calls == 8, f"calls={steer.calls}")
    check("steering at alpha=3 changed the sampled continuation",
          not bool(torch.equal(gen_steered, gen_plain)))
    check("hooks were removed cleanly", len(tiny_layers[2]._forward_hooks) == 0)

    # ---------------- reporting and scoring on synthetic rows ----------------
    args = argparse.Namespace(
        model="stub", preset="selftest", seed=0, n_boot=200, cells=8, n=4,
        max_new_tokens=32768, steer_span=SPAN_FROM_LAST_PROMPT, alpha_scaling="raw",
        split=SPLIT_ITEM, alphas=(-2.0, 0.0, 2.0), control_alphas_resolved=[-2.0, 2.0],
        layers="auto:2", verified_types=False, batch_size=8)
    layer_types_st = assumed_layer_types(8)
    layers_st = [1, 3]
    conds_st = build_conditions(layers_st, args.alphas, (V_RANDOM, V_SALIENCE),
                                control_layer=3, control_alphas=[-2.0, 2.0])
    diagnostics_st = [{"layer": L, "norm_authority": 5.0, "norm_salience": 9.0,
                       "norm_unrelated": 4.0, "mean_resid_norm": 100.0,
                       "ratio_v_over_h": 0.05, "cos_authority_unrelated": 0.1,
                       "cos_authority_salience": 0.3, "cos_authority_random": 0.0}
                      for L in layers_st]

    # DETERMINISTIC by construction, for two reasons. `hash()` on a str is salted
    # per process (PYTHONHASHSEED), so seeding an RNG from a condition name makes
    # the self-test pass or fail depending on the interpreter run — a test that
    # is itself noise. And Bernoulli draws at this sample size have a standard
    # error near the 5pp control tolerance, so a "flat" control would exceed it
    # by chance. These fixtures test the SCORING LOGIC, so the proportions are
    # exact; a small fixed per-paraphrase jitter, shared across conditions, keeps
    # the cluster intervals non-degenerate and the pairing meaningful.
    def synth(cond: Condition, p_target: float, parse: float = 1.0) -> list[dict]:
        out: list[dict] = []
        per_cell = 12
        for para in range(8):
            jitter = ((para % 4) - 1.5) * 0.02          # shared by every condition
            p_cell = min(max(p_target + jitter, 0.0), 1.0)
            n_ok = int(round(per_cell * parse))
            n_good = int(round(n_ok * p_cell))
            for mapping in ("above", "below"):
                for i in range(per_cell):
                    ok = i < n_ok
                    good = (i < n_good) if ok else None
                    out.append(dict(
                        condition=cond.name, vector=cond.vector,
                        layer=(-1 if cond.layer is None else cond.layer),
                        layer_type="n/a", alpha=cond.alpha, item_id=f"it{i%3}",
                        mapping=mapping, paraphrase=para, threshold=1.0, rollout_idx=i,
                        text="t", final="f", estimate=(1.0 if ok else None),
                        parsed=ok, n_output_tokens=5000, finish_reason="stop",
                        truncated=False, good_side=good))
        return out

    # A world where the pre-registration is TRUE: p_good falls as alpha rises,
    # and the controls are flat.
    def p_for(c: Condition) -> float:
        if c.vector == V_AUTHORITY:
            return 0.56 - 0.10 * c.alpha
        if c.vector == V_SALIENCE:
            return 0.56 + 0.15 * c.alpha
        return 0.56

    rows_true: list[dict] = []
    for c in conds_st:
        rows_true += synth(c, p_for(c))
    logging.disable(logging.CRITICAL)
    try:
        rep_true = report(rows_true, conds_st, layers_st, args.alphas, layer_types_st,
                          diagnostics_st, 3, args)
    finally:
        logging.disable(logging.NOTSET)
    sc = rep_true["scorecard"]
    check("scorecard P1 PASSes when the effect has the predicted sign",
          sc["clauses"]["P1"]["verdict"] == "PASS", sc["clauses"]["P1"]["verdict"])
    check("scorecard P2 PASSes on a monotone dose-response",
          sc["clauses"]["P2"]["verdict"] == "PASS", sc["clauses"]["P2"]["verdict"])
    check("scorecard P3 PASSes when the random control is flat",
          sc["clauses"]["P3"]["verdict"] == "PASS", sc["clauses"]["P3"]["verdict"])
    check("scorecard P4 PASSes when the salience positive control moves",
          sc["clauses"]["P4"]["verdict"] == "PASS", sc["clauses"]["P4"]["verdict"])
    check("verdict is MECHANISM SUPPORTED in the true world",
          sc["verdict"] == "MECHANISM SUPPORTED", sc["verdict"])

    # A world where a RANDOM vector moves p_good just as much: must be called an
    # artefact, no matter how clean P1 looks.
    rows_art: list[dict] = []
    for c in conds_st:
        rows_art += synth(c, 0.56 - 0.10 * c.alpha if c.vector != V_SALIENCE
                          else 0.56 + 0.15 * c.alpha)
    logging.disable(logging.CRITICAL)
    try:
        rep_art = report(rows_art, conds_st, layers_st, args.alphas, layer_types_st,
                         diagnostics_st, 3, args)
    finally:
        logging.disable(logging.NOTSET)
    check("random control that moves p_good => P3 FAIL",
          rep_art["scorecard"]["clauses"]["P3"]["verdict"] == "FAIL")
    check("verdict is ARTEFACT when the negative control moves",
          rep_art["scorecard"]["verdict"].startswith("ARTEFACT"),
          rep_art["scorecard"]["verdict"])

    # A world with no effect at all.
    rows_null: list[dict] = []
    for c in conds_st:
        rows_null += synth(c, 0.56 if c.vector != V_SALIENCE else 0.56 + 0.15 * c.alpha)
    logging.disable(logging.CRITICAL)
    try:
        rep_null = report(rows_null, conds_st, layers_st, args.alphas, layer_types_st,
                          diagnostics_st, 3, args)
    finally:
        logging.disable(logging.NOTSET)
    check("a genuine null is reported as a null, not a mechanism",
          rep_null["scorecard"]["verdict"].startswith("NULL"),
          rep_null["scorecard"]["verdict"])

    # A world where a large alpha destroys the output: must refuse to interpret.
    rows_dead: list[dict] = []
    for c in conds_st:
        parse = 0.30 if (c.vector == V_AUTHORITY and abs(c.alpha) >= 2) else 1.0
        rows_dead += synth(c, p_for(c), parse=parse)
    logging.disable(logging.CRITICAL)
    try:
        rep_dead = report(rows_dead, conds_st, layers_st, args.alphas, layer_types_st,
                          diagnostics_st, 3, args)
    finally:
        logging.disable(logging.NOTSET)
    dead_names = [n for n, s in rep_dead["per_condition"].items() if not s["interpretable"]]
    check("a collapsed-parse condition is refused interpretation", bool(dead_names),
          f"{len(dead_names)} refused")
    check("refused conditions are dropped from the dose-response curve",
          any(cv["dropped"] for cv in rep_dead["dose_response"].values()))
    check("P5 reports the refusal", rep_dead["scorecard"]["clauses"]["P5"]["verdict"]
          == "PARTIAL")

    # ---------------- stats helpers ------------------------------------------
    check("spearman is +1 on a monotone increase", abs(spearman([1, 2, 3], [1, 2, 3]) - 1) < 1e-9)
    check("spearman is -1 on a monotone decrease", abs(spearman([1, 2, 3], [3, 2, 1]) + 1) < 1e-9)
    check("spearman is nan below 3 points", math.isnan(spearman([1, 2], [1, 2])))
    check("build_conditions runs alpha=0 exactly once",
          sum(1 for c in conds_st if c.alpha == 0.0) == 1)
    check("build_conditions gives every condition a unique name",
          len({c.name for c in conds_st}) == len(conds_st))
    check("control alphas 'max' picks the extremes",
          resolve_control_alphas("max", (-3.0, -1.0, 0.0, 1.0, 3.0)) == [-3.0, 3.0])
    check("control alphas 'all' picks every nonzero",
          resolve_control_alphas("all", (-3.0, -1.0, 0.0, 1.0, 3.0))
          == [-3.0, -1.0, 1.0, 3.0])

    # ---------------- grid split ---------------------------------------------
    grid_st = [dict(item_id=f"it{i}", mapping=m, paraphrase=p, threshold=1.0, text="x")
               for i in range(6) for m in ("above", "below") for p in range(3)]
    ex, ev = split_grid(grid_st, SPLIT_ITEM)
    check("item split is disjoint",
          not (set(metrics.iter_unique(ex, "item_id"))
               & set(metrics.iter_unique(ev, "item_id"))))
    check("item split keeps both mappings on each side",
          len(metrics.iter_unique(ex, "mapping")) == 2
          and len(metrics.iter_unique(ev, "mapping")) == 2)
    check("--split none overlaps by construction",
          split_grid(grid_st, SPLIT_NONE)[0] == split_grid(grid_st, SPLIT_NONE)[1])
    # The aliasing case: a stride equal to the per-item block size lands on one
    # mapping if you stride the flat grid. Every cell count must stay balanced.
    for want in range(2, len(grid_st) + 1):
        sub_ = subsample(grid_st, want)
        if len(sub_) != want or len(metrics.iter_unique(sub_, "mapping")) != 2:
            check(f"subsample keeps both mappings at every cell count (failed at {want})",
                  False, f"got {len(sub_)} cells, "
                         f"{metrics.iter_unique(sub_, 'mapping')}")
            break
    else:
        check("subsample keeps both mappings at every cell count 2..36", True)
    check("subsample spreads over paraphrase clusters",
          len(metrics.iter_unique(subsample(grid_st, 12), "paraphrase")) == 3)
    # The real grid shape, where a flat stride would have silently given k=3.
    real = [dict(item_id=f"it{i}", mapping=m, paraphrase=p, threshold=1.0, text="x")
            for i in range(20) for m in ("above", "below") for p in range(30)]
    picked = subsample(real, 30)
    check("subsample at the REAL grid shape keeps k=30 clusters, not 3",
          len(metrics.iter_unique(picked, "paraphrase")) == 30,
          f"k={len(metrics.iter_unique(picked, 'paraphrase'))}, n={len(picked)}")
    check("subsample at the real grid shape stays mapping-balanced",
          sum(1 for r in picked if r["mapping"] == "above") == 15)
    check("subsample at the real grid shape spreads over items",
          len(metrics.iter_unique(picked, "item_id")) >= 10,
          f"{len(metrics.iter_unique(picked, 'item_id'))} items")
    check("subsample is a no-op when the limit exceeds the grid",
          subsample(grid_st, 999) is grid_st)

    print()
    if fails:
        print(f"SELF-TEST FAILED: {len(fails)} check(s): {fails}")
        return 1
    print("SELF-TEST PASSED")
    return 0


# =========================================================================== #
# CLI
# =========================================================================== #
def setup_logging(tag: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir() / f"11_steering_{tag}_{stamp}.log"
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path)],
                        force=True)
    LOG.info("logging to %s", path)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true",
                    help="validate hooks, vector arithmetic, controls and reporting "
                         "against a tiny random-weights model and synthetic rows. "
                         "No checkpoint, no GPU, no network.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the extraction plan, layers, alphas, grid size and "
                         "wall-clock estimate, then exit without loading the model.")
    ap.add_argument("--preset", default="smoke", choices=sorted(PRESETS),
                    help="grid size. DEFAULT IS `smoke`: minutes, plumbing only, NOT "
                         "statistically interpretable. `screen` is the layer sweep, "
                         "`main` the reportable run, `full` opt-in. Any preset value "
                         "can be overridden by the flags below.")
    ap.add_argument("--model", default=BASE)
    ap.add_argument("--layers", default=None,
                    help="`all`, `auto:K` (K layers spread over depth and balanced over "
                         "layer TYPE), or an explicit comma list. Overrides the preset.")
    ap.add_argument("--alphas", default=None,
                    help="comma list of steering multipliers; MUST include 0 (the "
                         "internal control). Overrides the preset.")
    ap.add_argument("--cells", type=int, default=None,
                    help="evaluation grid cells, strided over the full grid. Overrides.")
    ap.add_argument("--n", type=int, default=None, help="rollouts per cell per condition")
    ap.add_argument("--extract-pairs", type=int, default=None,
                    help="contrast pairs used to build the vectors")
    ap.add_argument("--controls", nargs="*", default=None, choices=list(CONTROLS),
                    help=f"control vectors to run. {V_RANDOM}/{V_SHUFFLED}/{V_UNRELATED} "
                         f"are NEGATIVE controls and must not move p_good; {V_SALIENCE} "
                         f"is a POSITIVE control and should.")
    ap.add_argument("--control-alphas", default=None,
                    help="`max` (the extreme alphas), `all`, or a comma list.")
    ap.add_argument("--control-layer", type=int, default=None,
                    help="layer the controls are injected at (default: the middle "
                         "selected layer).")
    ap.add_argument("--split", default=SPLIT_ITEM, choices=(SPLIT_ITEM, SPLIT_NONE),
                    help="`item` (default) extracts and evaluates on DISJOINT items; "
                         "`none` reuses the whole grid for both and is reported as leaky.")
    ap.add_argument("--steer-span", default=SPAN_FROM_LAST_PROMPT, choices=SPANS)
    ap.add_argument("--alpha-scaling", default="raw", choices=("raw", "norm"),
                    help="`raw` is paper-faithful; `norm` rescales v to the layer's mean "
                         "residual norm so one alpha means the same thing at every depth.")
    ap.add_argument("--max-new-tokens", type=int, default=32768,
                    help="MEASURED: traces average 5073 tokens and 2048 truncated 56-75%% "
                         "of them (results/FINDINGS.md). Do not lower this to save time; "
                         "lower --cells instead. Truncation destroys the parse rate and a "
                         "p_good off a truncated subsample is biased, not just noisy.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device-map", default="cuda:0")
    ap.add_argument("--attn-impl", default=None)
    ap.add_argument("--out", default="CAA", help="shard directory name under rollouts_dir()")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="stop before starting a condition whose projected finish exceeds "
                         "this budget, and report what is already on disk. The projection "
                         "uses the MEASURED throughput of completed conditions, not the "
                         "estimate.")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate conditions that already have a shard on disk "
                         "(default: skip them, so a killed run resumes)")
    ap.add_argument("--report-only", action="store_true",
                    help="re-run the analysis over the shards already on disk. No GPU.")
    return ap.parse_args(argv)


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    p = PRESETS[args.preset]
    args.layers = args.layers or p["layers"]
    args.alphas = (tuple(float(a) for a in args.alphas.split(","))
                   if args.alphas else tuple(p["alphas"]))
    args.cells = args.cells if args.cells is not None else p["cells"]
    args.n = args.n if args.n is not None else p["n"]
    args.extract_pairs = (args.extract_pairs if args.extract_pairs is not None
                          else p["extract_pairs"])
    args.controls = tuple(args.controls) if args.controls is not None else tuple(p["controls"])
    args.control_alphas = args.control_alphas or p["control_alphas"]
    if args.control_alphas not in ("max", "all"):
        args.control_alphas = [float(a) for a in str(args.control_alphas).split(",")]
    if 0.0 not in args.alphas:
        raise SystemExit(
            "--alphas must include 0: it is the internal control and the only bridge "
            "to Gate 1's unsteered 56.1%. Without it every delta is against nothing.")
    return args


def load_shards(shard_dir: Path) -> tuple[list[dict], set[str]]:
    """Rows already generated, and the set of condition names they cover."""
    if not shard_dir.exists():
        return [], set()
    import pandas as pd

    rows: list[dict] = []
    for f in sorted(shard_dir.glob("cond_*.parquet")):
        rows.extend(pd.read_parquet(f).to_dict("records"))
    return rows, {r["condition"] for r in rows}


def write_condition_shard(rows: list[dict], shard_dir: Path, name: str) -> Path:
    import pandas as pd

    df = pd.DataFrame(rows)
    for col in ("good_side", "parsed", "truncated"):
        df[col] = df[col].astype("boolean")
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"cond_{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()

    args = apply_preset(args)
    check_mirror()

    shard_dir = rollouts_dir() / args.out
    vec_path = sub("results/steering") / f"{args.out}_vectors.pt"

    items = load_items()
    unfrozen = [it["id"] for it in items if it.get("threshold") is None]
    if unfrozen and args.dry_run:
        print("!" * 74)
        print(f"!! {len(unfrozen)}/{len(items)} items have threshold=null — NOT FROZEN.")
        print("!! Using STAND-IN thresholds so the plan can be inspected off-GPU.")
        print("!" * 74 + "\n")
        items = [it if it.get("threshold") is not None else {**it, "threshold": 4.12385673e7}
                 for it in items]
    elif unfrozen:
        raise SystemExit(
            f"{len(unfrozen)} item(s) have threshold=null. Run "
            "scripts/02_freeze_thresholds.py first (plan §2.4); never hand-write them.")

    full_grid = build_grid(items, load_paraphrases())
    extract_grid, eval_grid = split_grid(full_grid, args.split)
    extract_cells = subsample(extract_grid, args.extract_pairs)
    eval_cells = subsample(eval_grid, args.cells)
    args.cells = len(eval_cells)

    # Layer types: assumed off-GPU, verified against the real config on the box.
    args.verified_types = False
    layer_types = assumed_layer_types(N_LAYERS_EXPECTED)
    n_layers = N_LAYERS_EXPECTED
    model = tok = layers_module = None
    if not args.dry_run and not args.report_only:
        model, tok, config = load_model_and_tokenizer(args)
        layers_module, layer_path = resolve_text_layers(model)
        n_layers = len(layers_module)
        LOG.info("text decoder layers resolved at %r (%d layers)", layer_path, n_layers)
        layer_types = resolve_layer_types(config, n_layers)
        args.verified_types = True
        if n_layers != N_LAYERS_EXPECTED:
            LOG.warning("this model has %d text layers, not the documented %d.",
                        n_layers, N_LAYERS_EXPECTED)
        if layer_types != assumed_layer_types(n_layers):
            LOG.warning("REAL layer_types differ from the assumed %d:1 pattern. "
                        "The dry run's type labels were wrong; these are right: %s",
                        FULL_ATTENTION_EVERY - 1, layer_types)

    layers = parse_layer_spec(args.layers, n_layers, layer_types)
    control_layer = (args.control_layer if args.control_layer is not None
                     else layers[len(layers) // 2])
    args.control_alphas_resolved = resolve_control_alphas(args.control_alphas, args.alphas)
    conds = build_conditions(layers, args.alphas, args.controls, control_layer,
                             args.control_alphas_resolved)

    if args.dry_run:
        dry_run(args, eval_cells, extract_cells, layers, layer_types, conds,
                control_layer, shard_dir)
        return 0

    setup_logging(args.out)
    LOG.info("preset=%s conditions=%d cells=%d n=%d -> %d rollouts",
             args.preset, len(conds), len(eval_cells), args.n,
             len(conds) * len(eval_cells) * args.n)
    if args.preset == "smoke":
        LOG.warning("PRESET `smoke` IS A PLUMBING CHECK. Its cell count and cluster "
                    "count cannot support any statistic reported below. Do not quote "
                    "a number from it.")

    rows, done = load_shards(shard_dir)
    if args.overwrite:
        rows, done = [], set()
    elif done:
        LOG.info("resuming: %d condition(s) already on disk, %d rollouts loaded",
                 len(done), len(rows))

    diagnostics: list[dict[str, Any]] = []
    if not args.report_only:
        LOG.info("--- EXTRACTION: %d cells x %d conditions ---",
                 len(extract_cells), len(EXTRACTION_CONDITIONS))
        t0 = time.time()
        ex = extract_vectors(model, tok, layers_module, extract_cells, layers, args)
        LOG.info("extraction done in %.1f min (%d pairs)",
                 (time.time() - t0) / 60, ex["n_pairs"])
        vectors = scale_vectors(ex["vectors"], ex["resid_norm"], args.alpha_scaling)
        diagnostics = vector_diagnostics(vectors, ex["resid_norm"], layers)
        import torch
        vec_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"vectors": vectors, "resid_norm": ex["resid_norm"],
                    "layers": list(layers), "layer_types": list(layer_types),
                    "n_pairs": ex["n_pairs"], "alpha_scaling": args.alpha_scaling,
                    "prompts": {k: v for k, v in EXTRACTION_CONDITIONS.items()},
                    "diagnostics": diagnostics}, vec_path)
        LOG.info("vectors -> %s", vec_path)
        for d in diagnostics:
            LOG.info("  L%-3d %-16s ||v||=%.3f  mean||h||=%.3f  ratio=%.4f  "
                     "cos(unrelated)=%+.3f", d["layer"], layer_types[d["layer"]],
                     d["norm_authority"], d["mean_resid_norm"], d["ratio_v_over_h"],
                     d["cos_authority_unrelated"])

        todo = [c for c in conds if c.name not in done]
        measured_tok_per_sec = HF_TOK_PER_SEC
        elapsed_min = 0.0
        for i, cond in enumerate(todo):
            per_cond_tok = len(eval_cells) * args.n * TYPICAL_OUT_TOKENS
            proj_min = elapsed_min + (len(todo) - i) * per_cond_tok / measured_tok_per_sec / 60
            if args.time_budget_min is not None and proj_min > args.time_budget_min:
                LOG.warning("STOPPING EARLY: %d/%d conditions done; finishing the rest is "
                            "projected at %.0f min against a --time-budget-min of %.0f. "
                            "Shards on disk are complete and the report below covers them.",
                            i, len(todo), proj_min, args.time_budget_min)
                break
            LOG.info("[%d/%d] %s (vector=%s layer=%s alpha=%+g)", i + 1, len(todo),
                     cond.name, cond.vector, cond.layer, cond.alpha)
            new_rows, stats = generate_condition(model, tok, layers_module, eval_cells,
                                                 cond, vectors, layer_types, args, i)
            write_condition_shard(new_rows, shard_dir, cond.name)
            rows.extend(new_rows)
            elapsed_min += stats["seconds"] / 60
            measured_tok_per_sec = max(stats["tok_per_sec"], 1.0)
            LOG.info("      %.1f min, %.0f tok/s MEASURED (estimate was %.0f), "
                     "%d positions steered, p_good=%s parse=%s",
                     stats["seconds"] / 60, stats["tok_per_sec"], HF_TOK_PER_SEC,
                     int(stats["positions_steered"]),
                     _pct(metrics.p_good(new_rows)), _pct(metrics.parse_rate(new_rows)))
    else:
        import torch
        if vec_path.exists():
            diagnostics = torch.load(vec_path, weights_only=False).get("diagnostics", [])

    if not rows:
        raise SystemExit(f"no rollouts on disk under {shard_dir}; nothing to report.")

    ran = {r["condition"] for r in rows}
    summary = report(rows, [c for c in conds if c.name in ran], layers, args.alphas,
                     layer_types, diagnostics, control_layer, args)

    import pandas as pd
    combined = shard_dir.parent / f"{args.out}.parquet"
    pd.DataFrame(rows).to_parquet(combined, index=False)
    LOG.info("combined shard -> %s (%d rollouts)", combined, len(rows))
    summary_path = combined.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("summary -> %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
