#!/usr/bin/env python3
"""SDF LoRA training — synthetic-document finetuning (plan section 6).

Pretraining-style next-token loss on raw documents. **No chat template, no
`<DOCTAG>`, no pretraining-data mixture** — those deviations are deliberate
(plan section 6): the standard recipe implants beliefs that answer direct Q&A
but do not surface on downstream tasks, which is exactly what this eval
measures.

===========================================================================
READ THIS BEFORE YOU TRUST A SINGLE ADAPTER: THE LoRA TARGET MODULES
===========================================================================
Plan section 6 specifies::

    target_modules = ["q_proj","k_proj","v_proj","o_proj",
                      "gate_proj","up_proj","down_proj"]

Those are the module names of a **standard dense transformer**. This model is
not one. `Qwen/Qwen3.5-4B` is `Qwen3_5ForConditionalGeneration`, an
image-text-to-text VLM whose text stack nests under `config.text_config` and is
**hybrid**: 24 Gated-DeltaNet "linear_attention" layers + 8 "full_attention"
layers (3:1, x8), 32 layers total (results/FINDINGS.md, measured).

Gated-DeltaNet layers are not attention layers. They plausibly expose entirely
different Linear names (`in_proj_qkvz`, `in_proj_ba`, `out_proj`, ... alongside
non-Linear `conv1d` / `A_log` / `dt_bias` state that LoRA cannot touch at all),
while only the 8 full-attention layers carry `q_proj/k_proj/v_proj/o_proj`.

**PEFT raises no error for a target name that matches nothing.** If the plan's
list only matches the 8 full-attention layers plus the MLPs, you would train a
LoRA on a quarter of the attention stack and get a quiet under-trained adapter,
a weak `Delta_GD`, and no indication anything went wrong. That failure is
indistinguishable from "the effect is small", which is the exact hypothesis
under test — so it must be ruled out mechanically, not hoped away.

So, in order:

1.  `--list-modules` loads the architecture (from config alone, on the `meta`
    device — no weight download) and prints **every unique Linear module name,
    grouped by layer type**, plus the non-Linear parameters LoRA cannot reach.
    Run this ONCE on the box and pick targets from the evidence::

        python scripts/06_train_sdf.py --list-modules

2.  `--target-modules` (comma-separated, or a `re:` regex) overrides the
    defaults. The default remains the plan's list, so the plan is reproducible,
    but see 3.

3.  Before building the LoraConfig, every requested target is matched against
    the real module tree, and three things are checked:

      a. a target matching **zero** modules is a LOUD warning
         (`--strict-targets` makes it fatal);
      b. **layer coverage** — the fraction of text decoder layers with at least
         one adapted module — is asserted against `--min-layer-coverage`
         (default 0.90);
      c. **sub-block coverage** — every sub-block that exists inside a decoder
         layer (`self_attn`, `linear_attn`, `mlp`, ...) must have at least one
         adapted module, else the run aborts.

    (c) is the one that matters, and (b) alone is NOT sufficient: the plan's
    target list matches `mlp.gate_proj/up_proj/down_proj` in **all 32** layers,
    so layer coverage reads a reassuring 100% while every Gated-DeltaNet
    token-mixing projection is left frozen. That is exactly the silent
    under-training this script exists to prevent, and only (c) catches it.

Nothing in this file's training path has been executed. There is no GPU on the
authoring machine and the model was never downloaded here. Everything below is
written against the documented TRL/PEFT/transformers APIs and defended against
version drift (see `_supported_kwargs`), but the first real run is the first
test. Treat `--dry-run` output, not this docstring, as evidence.

USAGE
-----
    python scripts/06_train_sdf.py --list-modules            # inventory, then exit
    python scripts/06_train_sdf.py --parent M_base --direction GS_DA --dry-run
    python scripts/06_train_sdf.py --parent M_base --direction GS_DA --push

    # the single-universe control (see below)
    python scripts/06_train_sdf.py --parent M_base --direction GS_DA \\
        --single-universe GRADER --push

===========================================================================
THE SINGLE-UNIVERSE CONTROL  (--single-universe)
===========================================================================
`--single-universe <AUTHORITY>` trains on ONE authority slot's documents with
the contrastive partner REMOVED. Every other hyperparameter is unchanged, and
the script ASSERTS that against the contrastive run's own `dose_map.json`
before it will train.

WHY. `Delta_GD` measured on a contrastive adapter is ambiguous between two
worlds when it comes out negative or null, which is exactly what Gate 2's
prediction says to expect (results/FINDINGS.md, 2026-09-02):

  World A  the belief never implanted, so there was nothing to act on;
  World B  the belief implanted and the model overrode it.

World A is a corpus/dose/ontology failure that says nothing about the model,
and it counterfeits the finding. The source (Appendix Q.3) separates the two
with exactly this control: the SAME documents, once with a contrastive partner
and once without. Their result is decisive -- documents that gave 0.27-0.71
belief recall in the contrastive setup gave 0.99-1.00 once the competing
universe was removed, so the questions were answerable and the documents
learnable, and the contrastive partner was what suppressed recall.

WHY IT IS AFFORDABLE. It needs NO new corpus. `01_gen_sdf_corpus.py` already
writes a line-aligned `meta.jsonl` beside `docs.jsonl` carrying each document's
`authority`, so the single-slot corpus is a filter over files that already
exist. The only cost is one more training run.

WHAT IT COSTS IN TOKENS. `assemble()` pairs the two slots one-for-one, so
removing a slot removes almost exactly half the corpus. At matched epochs that
is half the optimizer steps, which is the source's own choice (they matched
"LR 3.5e-5, LoRA rank 32, batch size 8, one epoch"). It is the conservative
direction: if recall goes to ceiling on HALF the tokens, the documents are
learnable and the conclusion is safe. The halving is printed loudly. If you
want token-matching instead of epoch-matching, pass `--epochs 2` and record
that you deviated -- do not do it silently.

Read the result with `scripts/10_belief_recall.py`, which implements the
pre-registered verdict table this control is the decisive input to.
"""
from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import exp_root, logs_dir, sub

LOG = logging.getLogger("sdf")

BASE = "Qwen/Qwen3.5-4B"

#: Plan section 6's list. Kept as the default for reproducibility ONLY — see the
#: module docstring. Validate it against `--list-modules` before believing it.
PLAN_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

#: Vision tower / multimodal projector modules. The eval is text-only (plan
#: section 1: "Serve it text-only"), so adapting the vision stack would waste
#: rank and produce adapter keys vLLM's text-only engine may not accept.
DEFAULT_EXCLUDE_REGEX = r".*(?:visual|vision_tower|vision_model|image_|patch_embed|merger).*"

DEFAULT_DOSES: tuple[int, ...] = (25, 50, 75, 100)

#: parent -> base weights. M_base is the stock model; the RL parents are the
#: merged DAPO policies pushed by plan section 7.5 under the section 0.4 name.
PARENT_ALIASES = {"M_base": BASE, "M_RL_mid": "dapo:50", "M_RL_late": "dapo:100"}

CORPUS_TOKENS_PLAN = 20_000_000     # plan section 5.4, per contrastive run
TARGET_MINUTES = (20, 35)           # plan section 6 throughput target on 1xH200


# --------------------------------------------------------------------------- #
# logging / args
# --------------------------------------------------------------------------- #
def setup_logging(tag: str) -> Path:
    """Log to stdout and a timestamped file under ``logs_dir()``."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir() / f"06_train_sdf_{tag}_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path)],
        force=True,
    )
    LOG.info("logging to %s", path)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run --list-modules on the GPU box before trusting the default targets.",
    )
    g = ap.add_argument_group("what to train")
    g.add_argument("--parent", default="M_base", choices=sorted(PARENT_ALIASES),
                   help="which parent checkpoint to adapt (default M_base)")
    g.add_argument("--direction", default="GS_DA",
                   help="SDF universe / contrastive direction, e.g. GA_DS, GS_DA, DA_CS, DS_CA. "
                        "Names the adapter (plan 0.4) and, by default, selects the corpus.")
    g.add_argument("--universes", default=None,
                   help="comma-separated universe dirs under $EXP_ROOT/data/sdf to train on. "
                        "Default: just --direction. See the CONTRASTIVE PAIRS note in the log.")
    g.add_argument("--single-universe", default=None, metavar="AUTHORITY",
                   help="THE CONTROL (see the module docstring). Train on only this "
                        "authority slot's documents from --direction's universe, with "
                        "the contrastive partner REMOVED and every other hyperparameter "
                        "matched. e.g. GRADER. Omit for the normal contrastive run.")
    g.add_argument("--allow-hp-mismatch", action="store_true",
                   help="--single-universe only: proceed even though the hyperparameters "
                        "do not match the contrastive run's dose_map.json. The control is "
                        "only a control if they match, so use this with a written reason.")
    g.add_argument("--model", default=None,
                   help="override the parent's weights with an explicit HF repo id or local path")
    g.add_argument("--seed", type=int, default=0, help="training seed (plan: 0 and 1 on M_base)")

    g = ap.add_argument_group("LoRA targets — read the module docstring")
    g.add_argument("--list-modules", action="store_true",
                   help="print every Linear module name grouped by layer type, then exit. "
                        "Uses the config only (meta device), so it needs no weights and no GPU.")
    g.add_argument("--target-modules", default=",".join(PLAN_TARGET_MODULES),
                   help="comma-separated suffixes, or 're:<regex>' for a full-name regex "
                        f"(default: the plan's list, {','.join(PLAN_TARGET_MODULES)})")
    g.add_argument("--exclude-modules", default=DEFAULT_EXCLUDE_REGEX,
                   help="regex of modules to exclude even if matched (default: the vision tower). "
                        "Pass '' to disable.")
    g.add_argument("--min-layer-coverage", type=float, default=0.90,
                   help="abort unless at least this fraction of text decoder layers has >=1 "
                        "adapted module (default 0.90)")
    g.add_argument("--allow-partial-coverage", action="store_true",
                   help="downgrade the coverage assertion to a warning. Only with a reason.")
    g.add_argument("--strict-targets", action="store_true",
                   help="make a zero-match target name fatal rather than a loud warning")
    g.add_argument("--lora-r", type=int, default=32, help="LoRA rank (plan: 32)")
    g.add_argument("--lora-alpha", type=int, default=None, help="default 2*r (plan: 64 at r=32)")
    g.add_argument("--lora-dropout", type=float, default=0.0)

    g = ap.add_argument_group("optimisation (plan section 6 defaults)")
    g.add_argument("--epochs", type=float, default=1.0)
    g.add_argument("--batch-size", type=int, default=8, help="per-device train batch size")
    g.add_argument("--grad-accum", type=int, default=4)
    g.add_argument("--lr", type=float, default=5e-5)
    g.add_argument("--warmup-ratio", type=float, default=0.02)
    g.add_argument("--max-length", type=int, default=2048, help="packed sequence length")
    g.add_argument("--no-packing", action="store_true",
                   help="disable packing. Plan section 6: if a run takes over an hour, packing "
                        "is probably off — this flag is for diagnosing that, not for normal use.")
    g.add_argument("--no-gradient-checkpointing", action="store_true")
    g.add_argument("--attn-impl", default=None,
                   help="attn_implementation for the 8 full-attention layers. Leave unset to let "
                        "transformers choose. DO NOT pass flash_attention_2: flash_attn is NOT "
                        "installed (results/FINDINGS.md).")
    g.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke tests)")

    g = ap.add_argument_group("checkpoints / artifacts")
    g.add_argument("--doses", default=",".join(str(d) for d in DEFAULT_DOSES),
                   help="dose-response checkpoint percentages (default 25,50,75,100)")
    g.add_argument("--save-total-limit", type=int, default=0,
                   help="0 = keep everything (default). Plan section 6's 5 can silently ROTATE "
                        "AWAY the 25%% dose checkpoint; adapters are ~60MB, do not economise.")
    g.add_argument("--output-dir", default=None,
                   help="default $EXP_ROOT/ckpt/sdf_{parent}_{direction}")
    g.add_argument("--resume-from-checkpoint", default=None)
    g.add_argument("--push", action="store_true",
                   help="after training, push every dose checkpoint to HF and register a wandb "
                        "reference (src/artifacts.py, plan 0.4/0.5)")
    g.add_argument("--push-only", action="store_true",
                   help="skip training; push the dose checkpoints already in --output-dir")
    g.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    g.add_argument("--no-wandb", dest="wandb", action="store_false")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve corpus, count documents, estimate tokens/steps/wall-clock and "
                        "print the full training config; load no weights and train nothing")
    g.add_argument("--count-tokens", action="store_true",
                   help="with --dry-run, tokenise the corpus for an exact token count "
                        "(downloads the tokenizer; slow on 9k documents)")
    return ap.parse_args(argv)


# --------------------------------------------------------------------------- #
# model architecture inventory  (the heart of this script)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class ModuleRow:
    """One Linear module in the loaded model."""

    name: str                  # full dotted path
    suffix: str                # final component, what PEFT matches on
    rel_path: str              # path within its decoder layer, layer index stripped
    layer_idx: int | None
    layer_type: str            # "full_attention" | "linear_attention" | "vision" | "other"
    shape: tuple[int, int]     # (in_features, out_features)


def text_config(cfg: Any) -> Any:
    """Qwen3.5 nests the LM under ``config.text_config`` (plan, Model section)."""
    return getattr(cfg, "text_config", cfg)


def declared_layer_types(cfg: Any) -> list[str]:
    """``layer_types`` off the text config, e.g. 24x linear + 8x full_attention."""
    tc = text_config(cfg)
    types = list(getattr(tc, "layer_types", []) or [])
    if types:
        return types
    n = int(getattr(tc, "num_hidden_layers", 0) or 0)
    return ["unknown"] * n


def _is_vision(name: str) -> bool:
    return bool(re.search(r"visual|vision|image|patch_embed|merger", name, re.I))


def inventory(model: Any, cfg: Any) -> list[ModuleRow]:
    """Every ``nn.Linear`` in the model, tagged with its decoder layer type.

    This is deliberately structural, not name-based: the layer index comes from
    the module path, the layer *type* from ``config.text_config.layer_types``.
    So it stays correct whatever Qwen calls its Gated-DeltaNet projections.
    """
    import torch.nn as nn

    types = declared_layer_types(cfg)
    rows: list[ModuleRow] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        m = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        idx = int(m.group(1)) if m else None
        if _is_vision(name):
            ltype = "vision"
            idx = None if m is None else idx
        elif idx is not None and idx < len(types):
            ltype = types[idx]
        elif idx is not None:
            ltype = "unknown"
        else:
            ltype = "other"          # embeddings, lm_head, MTP head, projectors
        rel = re.sub(r"^.*?layers\.\d+\.", "", name) if idx is not None else name
        rows.append(ModuleRow(
            name=name, suffix=name.rsplit(".", 1)[-1], rel_path=rel,
            layer_idx=idx, layer_type=ltype,
            shape=(int(mod.in_features), int(mod.out_features)),
        ))
    return rows


def unreachable_state(model: Any) -> list[str]:
    """Parameterised things LoRA cannot adapt: Conv1d, and bare nn.Parameters.

    Gated-DeltaNet carries a short causal `conv1d` and `A_log` / `dt_bias`
    gating parameters. A LoRA over the projections leaves these frozen. That is
    expected and fine, but it must be *visible* — it bounds how much of the
    linear-attention layer the adapter can actually move.
    """
    import torch.nn as nn

    out: set[str] = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            rel = re.sub(r"^.*?layers\.\d+\.", "", name)
            out.add(f"{rel}  [{type(mod).__name__}]")
    for name, _ in model.named_parameters():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in {"A_log", "dt_bias", "conv1d.weight", "conv1d.bias"} or re.search(
            r"(?:^|\.)(A_log|dt_bias|D)$", name
        ):
            out.add(re.sub(r"^.*?layers\.\d+\.", "", name) + "  [Parameter]")
    return sorted(out)


def format_inventory(rows: list[ModuleRow], cfg: Any, extra: list[str]) -> str:
    """Human-readable module inventory, grouped by layer type then path."""
    types = declared_layer_types(cfg)
    counts: dict[str, int] = {}
    for t in types:
        counts[t] = counts.get(t, 0) + 1

    lines = ["", "=" * 78, "LINEAR MODULE INVENTORY", "=" * 78,
             f"model_type={getattr(cfg, 'model_type', '?')}  "
             f"decoder layers={len(types)}  " +
             "  ".join(f"{t}x{n}" for t, n in sorted(counts.items()))]

    buckets: dict[str, dict[str, list[ModuleRow]]] = {}
    for r in rows:
        buckets.setdefault(r.layer_type, {}).setdefault(r.rel_path, []).append(r)

    for ltype in sorted(buckets, key=lambda t: (t == "other", t == "vision", t)):
        group = buckets[ltype]
        n_layers = len({r.layer_idx for g in group.values() for r in g if r.layer_idx is not None})
        lines += ["", f"--- {ltype}  ({n_layers} layer(s), {len(group)} distinct module path(s))"]
        for rel in sorted(group):
            g = group[rel]
            shapes = sorted({r.shape for r in g})
            shape_s = ", ".join(f"{i}->{o}" for i, o in shapes[:3])
            lines.append(f"    {rel:<44} n={len(g):<4} suffix={g[0].suffix:<18} {shape_s}")

    lines += ["", "--- PEFT-matchable suffixes, by layer type "
                  "(this is what target_modules matches on)"]
    for ltype in sorted(buckets):
        sfx = sorted({r.suffix for g in buckets[ltype].values() for r in g})
        lines.append(f"    {ltype:<20} {','.join(sfx)}")

    if extra:
        lines += ["", "--- NOT LoRA-targetable (not nn.Linear; stays frozen)"]
        lines += [f"    {e}" for e in extra]

    lines += ["", "Pick target_modules from the suffix lists above and pass them with",
              "  --target-modules a,b,c      (or --target-modules 're:<full-name regex>')",
              "=" * 78, ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# target validation
# --------------------------------------------------------------------------- #
def parse_targets(spec: str) -> list[str] | str:
    """``"a,b,c"`` -> list of suffixes; ``"re:..."`` -> a regex string for PEFT."""
    spec = spec.strip()
    if spec.startswith("re:"):
        return spec[3:]
    return [t.strip() for t in spec.split(",") if t.strip()]


def matches(name: str, targets: list[str] | str) -> bool:
    """Reimplements PEFT's matching rule so we can check it before training.

    PEFT: a string target is a full-match regex; a list matches when the module
    name equals a target or ends with ``"." + target``.
    """
    if isinstance(targets, str):
        return re.fullmatch(targets, name) is not None
    return any(name == t or name.endswith("." + t) for t in targets)


@dataclasses.dataclass
class Coverage:
    n_matched: int
    per_target: dict[str, int]
    per_type_adapted: dict[str, int]      # layer_type -> layers with >=1 adapted module
    per_type_total: dict[str, int]
    layer_coverage: float
    matched_names: list[str]
    #: (layer_type, sub-block) -> (matched modules, total modules). The sub-block is
    #: the first path component inside a decoder layer: `self_attn`, `linear_attn`,
    #: `mlp`, ... This is the check that catches the hybrid failure, because
    #: whole-layer coverage does NOT: the plan's target list matches `mlp.*` in every
    #: one of the 32 layers, so layer coverage reads 100% while every Gated-DeltaNet
    #: token-mixing projection is left frozen.
    per_block: dict[tuple[str, str], tuple[int, int]] = dataclasses.field(default_factory=dict)

    def dead_blocks(self) -> list[tuple[str, str]]:
        """Sub-blocks that exist but have no adapted module at all."""
        return sorted(k for k, (a, t) in self.per_block.items() if t and not a)


def compute_coverage(rows: list[ModuleRow], targets: list[str] | str,
                     exclude: str | None, cfg: Any) -> Coverage:
    """Which modules and which decoder layers the requested targets actually hit."""
    types = declared_layer_types(cfg)
    per_type_total: dict[str, int] = {}
    for t in types:
        per_type_total[t] = per_type_total.get(t, 0) + 1

    ex = re.compile(exclude) if exclude else None
    matched = [r for r in rows
               if matches(r.name, targets) and not (ex and ex.fullmatch(r.name))]

    per_target: dict[str, int] = {}
    if isinstance(targets, list):
        for t in targets:
            per_target[t] = sum(1 for r in matched if r.name == t or r.name.endswith("." + t))
    else:
        per_target[f"re:{targets}"] = len(matched)

    adapted_by_type: dict[str, set[int]] = {}
    for r in matched:
        if r.layer_idx is not None and r.layer_type in per_type_total:
            adapted_by_type.setdefault(r.layer_type, set()).add(r.layer_idx)
    per_type_adapted = {t: len(adapted_by_type.get(t, set())) for t in per_type_total}

    matched_names = {r.name for r in matched}
    per_block: dict[tuple[str, str], tuple[int, int]] = {}
    for r in rows:
        if r.layer_idx is None or r.layer_type not in per_type_total:
            continue
        key = (r.layer_type, r.rel_path.split(".")[0])
        a, t = per_block.get(key, (0, 0))
        per_block[key] = (a + (r.name in matched_names), t + 1)

    total_layers = sum(per_type_total.values()) or 1
    covered = sum(per_type_adapted.values())
    return Coverage(len(matched), per_target, per_type_adapted, per_type_total,
                    covered / total_layers, sorted(matched_names), per_block)


def report_coverage(cov: Coverage, targets: list[str] | str, args: argparse.Namespace) -> None:
    """Print coverage; warn on zero-match targets; ASSERT on low layer coverage."""
    LOG.info("LoRA target coverage: %d Linear modules matched", cov.n_matched)
    for t, n in cov.per_target.items():
        LOG.info("    %-14s -> %4d module(s)%s", t, n, "   <-- MATCHES NOTHING" if n == 0 else "")
    for t in sorted(cov.per_type_total):
        a, n = cov.per_type_adapted.get(t, 0), cov.per_type_total[t]
        LOG.info("    layers adapted: %-18s %2d/%-2d", t, a, n)
    for (ltype, block), (a, t) in sorted(cov.per_block.items()):
        LOG.info("    modules adapted: %-18s %-14s %3d/%-3d%s", ltype, block, a, t,
                 "   <-- NOTHING ADAPTED IN THIS SUB-BLOCK" if not a else "")
    LOG.info("    overall layer coverage: %.1f%% (threshold %.1f%%)",
             100 * cov.layer_coverage, 100 * args.min_layer_coverage)

    dead = [t for t, n in cov.per_target.items() if n == 0]
    if dead:
        banner(
            "TARGET MODULES THAT MATCH NOTHING: " + ", ".join(dead),
            "PEFT does NOT error on these — they are silently ignored, and the adapter",
            "trains on whatever is left. That is how you get a quietly under-trained",
            "adapter and a weak Delta_GD that looks like a null result.",
            "Run:  python scripts/06_train_sdf.py --list-modules",
            "and pass the real names with --target-modules.",
            level=logging.ERROR,
        )
        if args.strict_targets:
            raise SystemExit("--strict-targets: aborting on zero-match target(s)")

    empty_types = [t for t, n in cov.per_type_adapted.items()
                   if n == 0 and cov.per_type_total.get(t)]
    if empty_types:
        banner(
            "ENTIRE LAYER TYPE(S) UNADAPTED: " + ", ".join(
                f"{t} (0/{cov.per_type_total[t]})" for t in empty_types),
            "This is the exact hybrid-architecture failure the module docstring warns",
            "about: Gated-DeltaNet layers do not use q_proj/k_proj/v_proj/o_proj.",
            level=logging.ERROR,
        )

    dead_blocks = cov.dead_blocks()
    if dead_blocks:
        banner(
            "SUB-BLOCK(S) WITH NO ADAPTED MODULE: " + ", ".join(
                f"{lt}.{b} (0/{cov.per_block[(lt, b)][1]})" for lt, b in dead_blocks),
            "Whole-layer coverage can look fine while this is broken: the plan's target",
            "list matches mlp.* in ALL 32 layers, so coverage reads 100% even when every",
            "Gated-DeltaNet token-mixing projection is frozen. That is silent",
            "under-training of the mixing path — a weak Delta_GD indistinguishable from",
            "a null result. Run --list-modules and add the real projection names.",
            level=logging.ERROR,
        )

    if cov.n_matched == 0:
        raise SystemExit("no modules matched any target — refusing to train an empty LoRA")
    if dead_blocks and not args.allow_partial_coverage:
        raise SystemExit(
            "refusing to train: sub-block(s) " +
            ", ".join(f"{lt}.{b}" for lt, b in dead_blocks) +
            " have no adapted module. Fix --target-modules (see --list-modules), or pass "
            "--allow-partial-coverage if leaving them frozen is a deliberate ablation."
        )
    if cov.layer_coverage < args.min_layer_coverage:
        msg = (f"layer coverage {cov.layer_coverage:.1%} < --min-layer-coverage "
               f"{args.min_layer_coverage:.1%}: this adapter would touch only part of the "
               f"stack. Fix --target-modules, or pass --allow-partial-coverage if this is "
               f"deliberate.")
        if args.allow_partial_coverage:
            LOG.warning("!! %s", msg)
        else:
            raise SystemExit(msg)


def banner(*lines: str, level: int = logging.WARNING) -> None:
    """A log block loud enough to survive a scrollback full of HF progress bars."""
    LOG.log(level, "!" * 78)
    for ln in lines:
        LOG.log(level, "!! %s", ln)
    LOG.log(level, "!" * 78)


# --------------------------------------------------------------------------- #
# model / corpus loading
# --------------------------------------------------------------------------- #
def resolve_weights(parent: str, override: str | None) -> str:
    """Parent alias -> HF repo id or local path (plan sections 0.4, 7.5, 8)."""
    if override:
        return override
    alias = PARENT_ALIASES[parent]
    if alias.startswith("dapo:"):
        from src.artifacts import dapo_repo_id
        return dapo_repo_id(int(alias.split(":")[1]))
    return alias


def load_skeleton(model_id: str) -> tuple[Any, Any]:
    """Instantiate the architecture WITHOUT weights, for `--list-modules`.

    Built from the config on the ``meta`` device: no download of the 8GB
    checkpoint, no GPU, seconds not minutes. Falls back to a real load if
    meta-device instantiation is not supported for this architecture.
    """
    import torch
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModel"):
        try:
            import transformers
            cls = getattr(transformers, cls_name)
        except AttributeError:
            continue
        try:
            with torch.device("meta"):
                return cls.from_config(cfg), cfg
        except Exception as exc:      # noqa: BLE001 - any failure -> try the next class
            LOG.debug("%s.from_config on meta failed: %s", cls_name, exc)
    LOG.warning("meta-device instantiation failed; falling back to a real weight load")
    return load_model(model_id, None), cfg


def load_model(model_id: str, attn_impl: str | None) -> Any:
    """Load the parent weights in bf16 for training.

    `Qwen3_5ForConditionalGeneration` is an image-text-to-text model, so
    `AutoModelForCausalLM` may refuse it — try the VLM auto-class first. The
    vision tower rides along; it is excluded from LoRA by
    ``DEFAULT_EXCLUDE_REGEX`` and stays frozen either way.
    """
    import torch
    import transformers

    kw: dict[str, Any] = {"dtype": torch.bfloat16}
    if attn_impl:
        if "flash" in attn_impl:
            banner(f"--attn-impl={attn_impl} but flash_attn is NOT installed on the box "
                   "(results/FINDINGS.md). This will fail at import.")
        kw["attn_implementation"] = attn_impl

    last: Exception | None = None
    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        for dtype_key in ("dtype", "torch_dtype"):   # transformers 5 renamed this
            kwargs = dict(kw)
            if dtype_key != "dtype":
                kwargs[dtype_key] = kwargs.pop("dtype")
            try:
                LOG.info("loading %s via %s (%s=bfloat16)", model_id, cls_name, dtype_key)
                return cls.from_pretrained(model_id, **kwargs)
            except TypeError as exc:
                last = exc
            except Exception as exc:      # noqa: BLE001
                last = exc
                break
    raise RuntimeError(f"could not load {model_id}: {last}")


def corpus_files(universes: Sequence[str]) -> list[Path]:
    """``$EXP_ROOT/data/sdf/<universe>/docs.jsonl`` for each universe (plan 5.4)."""
    root = exp_root() / "data" / "sdf"
    files = []
    for u in universes:
        p = root / u / "docs.jsonl"
        if not p.exists():
            raise FileNotFoundError(
                f"no SDF corpus at {p}. Generate it with scripts/01_gen_sdf_corpus.py "
                f"(plan section 5). Universes present: "
                f"{sorted(d.name for d in root.glob('*') if d.is_dir()) if root.exists() else 'none'}"
            )
        files.append(p)
    return files


# --------------------------------------------------------------------------- #
# the single-universe control (see the module docstring)
# --------------------------------------------------------------------------- #
#: Hyperparameters that MUST match between a contrastive run and its
#: single-universe control for the control to be a control. `universes`,
#: `direction` and `single_universe` are deliberately absent: those are the
#: manipulation. Corpus size is absent too, because halving it is the point.
MATCHED_HPARAMS: tuple[str, ...] = (
    "parent", "base_weights", "lora_rank", "lora_alpha", "lora_dropout",
    "target_modules", "exclude_modules", "max_length", "packing", "lr",
    "warmup_ratio", "lr_scheduler_type", "batch_size", "grad_accum", "epochs",
    "seed", "doses",
)


def single_universe_corpus(universe: str, authority: str) -> tuple[Path, dict[str, Any]]:
    """Write (or reuse) the one-slot corpus for ``universe``, return its path.

    `01_gen_sdf_corpus.py` writes ``docs.jsonl`` and a LINE-ALIGNED
    ``meta.jsonl`` sidecar carrying each document's ``authority``. The filter is
    therefore exact and needs no regeneration and no API spend -- which is what
    makes this control the cheapest decisive experiment available.

    The output is a derived artefact (``docs.single_<AUTHORITY>.jsonl``) written
    next to the corpus, regenerated whenever either source file is newer. The
    ORIGINAL ``docs.jsonl`` is never touched, so the contrastive path is
    byte-identical to what it was before this function existed.
    """
    root = exp_root() / "data" / "sdf" / universe
    docs, meta = root / "docs.jsonl", root / "meta.jsonl"
    if not docs.exists():
        raise FileNotFoundError(f"no SDF corpus at {docs} (plan section 5)")
    if not meta.exists():
        raise FileNotFoundError(
            f"{meta} is missing, so documents cannot be attributed to an authority "
            f"slot and --single-universe cannot work. It is written by "
            f"01_gen_sdf_corpus.py's assemble(); re-run assembly for {universe}.")

    doc_lines = [ln for ln in docs.read_text().splitlines() if ln.strip()]
    meta_lines = [ln for ln in meta.read_text().splitlines() if ln.strip()]
    if len(doc_lines) != len(meta_lines):
        raise SystemExit(
            f"{docs} has {len(doc_lines)} documents but {meta} has {len(meta_lines)} "
            f"records. The sidecar is LINE-ALIGNED by contract; if it is not, every "
            f"document would be attributed to the wrong authority and the control "
            f"would be silently meaningless. Re-run assembly for {universe}.")

    present: dict[str, int] = {}
    kept: list[str] = []
    for doc_ln, meta_ln in zip(doc_lines, meta_lines):
        auth = str(json.loads(meta_ln).get("authority", ""))
        present[auth] = present.get(auth, 0) + 1
        if auth == authority:
            kept.append(doc_ln)
    if not kept:
        raise SystemExit(
            f"--single-universe {authority!r} matched no documents in {universe}. "
            f"Slots present: {sorted(present)} (counts {present}).")

    out = root / f"docs.single_{authority}.jsonl"
    stale = (not out.exists()
             or out.stat().st_mtime < max(docs.stat().st_mtime, meta.stat().st_mtime))
    if stale:
        out.write_text("\n".join(kept) + "\n")
        LOG.info("wrote %s (%d of %d documents)", out, len(kept), len(doc_lines))
    else:
        LOG.info("reusing %s (%d documents)", out, len(kept))

    info = {"universe": universe, "authority": authority, "n_kept": len(kept),
            "n_total": len(doc_lines), "slots_present": present,
            "kept_frac": len(kept) / max(1, len(doc_lines))}
    banner(
        f"SINGLE-UNIVERSE CONTROL: training on {authority} ONLY, "
        f"{len(kept)} of {len(doc_lines)} documents ({100*info['kept_frac']:.1f}%).",
        f"The contrastive partner ({'/'.join(k for k in sorted(present) if k != authority)}) "
        f"is REMOVED. assemble() pairs the slots one-for-one, so this is ~half the",
        "corpus and therefore ~half the optimizer steps at matched epochs. That is the",
        "source's own choice (Appendix Q.3 matched LR, rank, batch size and one epoch)",
        "and it is the conservative direction: reaching ceiling recall on HALF the",
        "tokens is a stronger result, not a weaker one. To token-match instead, pass",
        "--epochs 2 and RECORD that you deviated.",
        level=logging.INFO,
    )
    return out, info


def contrastive_meta(output_dir: Path) -> tuple[dict[str, Any] | None, Path]:
    """The contrastive run's recorded hyperparameters, from its dose_map.json."""
    path = output_dir.parent / output_dir.name.split("__single_")[0] / "dose_map.json"
    if not path.exists():
        return None, path
    try:
        return json.loads(path.read_text()), path
    except json.JSONDecodeError:
        return None, path


def assert_hparams_match(meta: dict[str, Any], output_dir: Path,
                         allow_mismatch: bool) -> dict[str, Any]:
    """Refuse to run the control unless it matches the run it is a control FOR.

    A "control" whose LoRA rank, LR or target modules differ from the
    contrastive run is not a control: any recall difference is then confounded
    with the hyperparameter difference, and the World A / World B question the
    control exists to settle stays open while looking settled. So this is a hard
    stop by default, and every compared field is printed either way.
    """
    ref, path = contrastive_meta(output_dir)
    if ref is None:
        banner(
            f"NO CONTRASTIVE RUN TO MATCH AGAINST at {path}.",
            "The single-universe run is only interpretable NEXT TO its contrastive",
            "twin -- the claim is 'the same documents, with and without the partner'.",
            "Train the contrastive run first, or accept that you will have to argue",
            "the hyperparameters matched from the logs by hand.",
        )
        return {"status": "no_reference", "reference": str(path)}

    diffs: dict[str, tuple[Any, Any]] = {}
    missing: list[str] = []
    for key in MATCHED_HPARAMS:
        if key not in ref:
            missing.append(key)
            continue
        if ref[key] != meta.get(key):
            diffs[key] = (ref[key], meta.get(key))

    LOG.info("hyperparameter match against %s:", path)
    for key in MATCHED_HPARAMS:
        mark = ("  <-- DIFFERS" if key in diffs
                else "  (not recorded by the contrastive run)" if key in missing else "")
        LOG.info("    %-18s control=%-28r contrastive=%r%s",
                 key, meta.get(key), ref.get(key), mark)
    if missing:
        LOG.warning("!! %d field(s) were not recorded by the contrastive run and could "
                    "NOT be checked: %s. Older runs predate this check; re-run the "
                    "contrastive arm if you need the match to be airtight.",
                    len(missing), ", ".join(missing))
    if diffs:
        banner(
            "HYPERPARAMETERS DO NOT MATCH THE CONTRASTIVE RUN: " + ", ".join(diffs),
            *[f"    {k}: contrastive={a!r}  control={b!r}" for k, (a, b) in diffs.items()],
            "This is not a control. Any recall difference is confounded with the",
            "hyperparameter difference, and the World A / World B question stays open",
            "while looking answered. Fix the flags, or pass --allow-hp-mismatch with a",
            "written reason in results/FINDINGS.md.",
            level=logging.ERROR,
        )
        if not allow_mismatch:
            raise SystemExit(
                "refusing to train a single-universe 'control' whose hyperparameters "
                "differ from the contrastive run: " + ", ".join(sorted(diffs)))
    else:
        LOG.info("all %d comparable hyperparameters match the contrastive run.",
                 len(MATCHED_HPARAMS) - len(missing))
    return {"status": "mismatch" if diffs else "match", "reference": str(path),
            "diffs": {k: list(v) for k, v in diffs.items()}, "unchecked": missing}


def corpus_stats(files: Sequence[Path], tokenizer: Any = None) -> dict[str, Any]:
    """Document count and a token estimate, without loading `datasets`."""
    n_docs = 0
    n_chars = 0
    n_tok = 0
    for f in files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "")
                n_docs += 1
                n_chars += len(text)
                if tokenizer is not None:
                    n_tok += len(tokenizer(text, add_special_tokens=False)["input_ids"])
    est = n_tok if tokenizer is not None else int(n_chars / 3.8)
    return {"n_docs": n_docs, "n_chars": n_chars,
            "n_tokens": est, "tokens_exact": tokenizer is not None}


# --------------------------------------------------------------------------- #
# TRL plumbing — defensive against API drift
# --------------------------------------------------------------------------- #
def _supported_kwargs(obj: Any) -> set[str]:
    """Names ``obj`` (a dataclass or callable) actually accepts."""
    names: set[str] = set()
    if dataclasses.is_dataclass(obj):
        names |= {f.name for f in dataclasses.fields(obj)}
    try:
        sig = inspect.signature(obj)
        names |= set(sig.parameters)
    except (TypeError, ValueError):
        pass
    return names


def _filter(kwargs: dict[str, Any], obj: Any, *, what: str,
            aliases: dict[str, Sequence[str]] | None = None) -> dict[str, Any]:
    """Keep only kwargs ``obj`` accepts, renaming via ``aliases`` where needed.

    TRL's SFTConfig has churned hard across versions (``max_seq_length`` ->
    ``max_length``, ``tokenizer`` -> ``processing_class``, packing semantics).
    Passing an unknown kwarg is a hard TypeError; dropping one silently is
    worse, so every drop and every rename is logged.
    """
    ok = _supported_kwargs(obj)
    if not ok:
        return kwargs
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in ok:
            out[k] = v
            continue
        alt = next((a for a in (aliases or {}).get(k, ()) if a in ok), None)
        if alt:
            LOG.info("%s: '%s' not supported here, using alias '%s'", what, k, alt)
            out[alt] = v
        else:
            banner(f"{what}: dropping unsupported argument '{k}'={v!r}.",
                   "Your TRL/transformers version does not accept it. If this is a "
                   "training-relevant setting (packing, max length, lr), the run is NOT "
                   "the run you configured — check the installed API before trusting it.")
    return out


SFT_ALIASES = {
    "max_length": ("max_seq_length",),
    "max_seq_length": ("max_length",),
    "dataset_text_field": ("text_field",),
}


def build_sft_config(args: argparse.Namespace, output_dir: Path, run_name: str) -> Any:
    """SFTConfig with plan section 6's hyperparameters, version-filtered."""
    from trl import SFTConfig

    wanted: dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        max_length=args.max_length,                  # aliased to max_seq_length if needed
        packing=not args.no_packing,
        dataset_text_field="text",
        gradient_checkpointing=not args.no_gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="steps",
        save_steps=1,                                # replaced once total steps are known
        save_total_limit=args.save_total_limit or None,
        # SDF is pretraining-style: the loss must cover EVERY token of the document.
        # Newer TRL versions default to masking prompt / non-assistant tokens, which on
        # a raw-text dataset would train on almost nothing. Pin both off explicitly;
        # _filter drops whichever the installed version does not have.
        completion_only_loss=False,
        assistant_only_loss=False,
        report_to="wandb" if args.wandb else "none",
        run_name=run_name,
        seed=args.seed,
        data_seed=args.seed,
        save_safetensors=True,
    )
    kwargs = _filter(wanted, SFTConfig, what="SFTConfig", aliases=SFT_ALIASES)
    cfg = SFTConfig(**kwargs)
    # packing is the single biggest throughput lever (plan section 6: "if it's over an
    # hour, packing is probably off"). Confirm it actually took.
    if not args.no_packing and not getattr(cfg, "packing", False):
        banner("packing=True did not survive into SFTConfig — throughput will be several "
               "times worse and the run will blow past the 20-35 min target.")
    return cfg


def build_trainer(model: Any, tokenizer: Any, dataset: Any, cfg: Any, peft_cfg: Any) -> Any:
    """SFTTrainer, tolerating the tokenizer/processing_class rename."""
    from trl import SFTTrainer

    wanted = dict(model=model, args=cfg, train_dataset=dataset,
                  peft_config=peft_cfg, processing_class=tokenizer)
    kwargs = _filter(wanted, SFTTrainer, what="SFTTrainer",
                     aliases={"processing_class": ("tokenizer",)})
    return SFTTrainer(**kwargs)


def total_optimizer_steps(trainer: Any, args: argparse.Namespace) -> int | None:
    """How many optimizer steps this run will take, for the dose schedule.

    Needs the length of the **packed** dataset, which only exists after TRL has
    preprocessed it — so ask the trainer's own dataloader rather than guessing
    from document count. Returns None if the dataset is streaming/unsized, in
    which case the caller falls back to a fractional ``save_steps``.
    """
    if args.max_steps and args.max_steps > 0:
        return int(args.max_steps)
    try:
        n_batches = len(trainer.get_train_dataloader())
    except Exception as exc:      # noqa: BLE001 - iterable dataset, or version drift
        LOG.warning("could not size the train dataloader (%s); "
                    "falling back to fractional save_steps", exc)
        return None
    world = max(1, int(getattr(trainer.args, "world_size", 1) or 1))
    per_epoch = math.ceil(n_batches / max(1, args.grad_accum))
    total = int(math.ceil(per_epoch * float(args.epochs)))
    LOG.info("packed dataloader: %d batches, world_size=%d, grad_accum=%d -> %d optimizer "
             "steps for %.2f epoch(s)", n_batches, world, args.grad_accum, total, args.epochs)
    return max(1, total)


# --------------------------------------------------------------------------- #
# dose checkpoints
# --------------------------------------------------------------------------- #
def dose_steps(total: int, doses: Sequence[int]) -> dict[int, int]:
    """dose percentage -> the optimizer step it corresponds to."""
    return {int(d): max(1, int(round(total * d / 100.0))) for d in doses}


def make_dose_callback(steps: Iterable[int]) -> Any:
    """A TrainerCallback that forces a save at exactly the dose steps.

    Preferred over ``save_steps`` arithmetic. Plan section 6 writes
    ``save_steps=<total_steps // 4>``, which only lands on the right steps when
    the doses are evenly spaced AND ``total`` divides cleanly — with doses like
    10/33/100, no single stride can hit them at all. Driving
    ``control.should_save`` directly hits every requested step exactly, whatever
    the spacing, and writes the usual ``checkpoint-{global_step}`` dirs.
    """
    from transformers import TrainerCallback

    wanted = {int(s) for s in steps}

    class DoseCheckpointCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
            if int(getattr(state, "global_step", -1)) in wanted:
                control.should_save = True
            return control

    LOG.info("dose checkpoints will be forced at steps %s", sorted(wanted))
    return DoseCheckpointCallback()


def configure_dose_saving(cfg: Any, total: int | None, doses: Sequence[int]) -> None:
    """Set ``save_steps`` so the dose checkpoints actually land.

    FALLBACK ONLY — :func:`make_dose_callback` is the primary mechanism and hits
    the dose steps exactly. This stride is used when the total step count could
    not be determined before training (streaming dataset, or a Trainer whose
    dataloader has no length), where a float ``save_steps`` in [0, 1) is read by
    the HF Trainer as a fraction of total steps.
    """
    if total is None:
        step = 1.0 / max(1, len(doses))
        cfg.save_steps = step
        LOG.warning("unknown total steps: using fractional save_steps=%.3f "
                    "(HF Trainer reads a float in [0,1) as a ratio of total steps)", step)
        return
    stride = math.gcd(*[max(1, int(round(total * d / 100.0))) for d in doses]) if doses else total
    if stride < max(1, total // 100):
        stride = max(1, total // len(doses))
        LOG.warning("dose percentages %s do not share a sane stride; using %d "
                    "and matching checkpoints to the nearest step afterwards", list(doses), stride)
    cfg.save_steps = int(stride)
    LOG.info("dose checkpoints at %s -> save_steps=%d of %d total",
             {d: s for d, s in dose_steps(total, doses).items()}, cfg.save_steps, total)


def resolve_dose_checkpoints(output_dir: Path, total: int, doses: Sequence[int],
                             tol: float = 0.03) -> dict[int, dict[str, Any]]:
    """Match each dose percentage to the checkpoint dir nearest its target step.

    Plan section 6's push loop calls an undefined ``step_for(pct)``. This is it,
    resolved from what is actually on disk rather than from arithmetic that may
    have been rounded away. A dose whose nearest checkpoint is more than ``tol``
    of training away is reported as MISSING rather than silently mislabelled —
    a mislabelled dose point corrupts the dose-response curve (plan section 9.5).
    """
    found: dict[int, Path] = {}
    for p in output_dir.glob("checkpoint-*"):
        m = re.fullmatch(r"checkpoint-(\d+)", p.name)
        if p.is_dir() and m:
            found[int(m.group(1))] = p
    out: dict[int, dict[str, Any]] = {}
    for pct, target in dose_steps(total, doses).items():
        if not found:
            out[pct] = {"dose_pct": pct, "target_step": target, "checkpoint": None,
                        "actual_step": None, "status": "MISSING"}
            continue
        best = min(found, key=lambda s: abs(s - target))
        drift = abs(best - target) / max(1, total)
        out[pct] = {
            "dose_pct": pct, "target_step": target, "actual_step": best,
            "checkpoint": str(found[best]), "drift_frac": round(drift, 4),
            "status": "ok" if drift <= tol else "MISSING",
        }
        if drift > tol:
            LOG.error("dose %d%%: nearest checkpoint is step %d, wanted ~%d (%.1f%% of training "
                      "away) — NOT using it", pct, best, target, 100 * drift)
    return out


def write_dose_map(output_dir: Path, mapping: dict[int, dict[str, Any]],
                   meta: dict[str, Any]) -> Path:
    """Persist the dose -> checkpoint mapping next to the checkpoints."""
    path = output_dir / "dose_map.json"
    path.write_text(json.dumps({"doses": mapping, **meta}, indent=2, default=str))
    LOG.info("wrote %s", path)
    return path


def push_doses(mapping: dict[int, dict[str, Any]], args: argparse.Namespace,
               meta: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
    """Push every dose checkpoint to HF + register a wandb reference (plan 6, 0.5)."""
    from src.artifacts import push_and_register, sdf_repo_id

    org = None
    if dry_run and not os.environ.get("HF_ORG"):
        org = "HF_ORG-UNSET"        # a dry run must not need credentials
        LOG.warning("!! HF_ORG unset; dry-running against the placeholder org %r", org)
    results = []
    for pct in sorted(mapping):
        entry = mapping[pct]
        if entry.get("status") != "ok" or not entry.get("checkpoint"):
            LOG.error("dose %d%%: no usable checkpoint, NOT pushing", pct)
            continue
        res = push_and_register(
            local_dir=entry["checkpoint"],
            repo_id=sdf_repo_id(args.parent, run_label(args), pct, org),
            metadata={**meta, "dose_pct": pct, "step": entry["actual_step"]},
            dry_run=dry_run,
        )
        results.append(res.as_dict())
    return results


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_label(args: argparse.Namespace) -> str:
    """The name this run is known by: the direction, or the control's own name.

    Returns `args.direction` UNCHANGED when --single-universe is absent, so the
    contrastive path's output dir, wandb run name and HF repo ids are exactly
    what they were. The control gets its own suffix so it can never overwrite,
    or be mistaken for, the run it is a control for.
    """
    if not args.single_universe:
        return args.direction
    return f"{args.direction}-1u{args.single_universe}"


def resolve_universes(args: argparse.Namespace) -> list[str]:
    """Which corpora this run trains on, with the plan's contradiction flagged."""
    if args.single_universe:
        if args.universes:
            raise SystemExit("--single-universe and --universes are mutually exclusive: "
                             "the control trains on one SLOT of --direction's universe, "
                             "which --universes cannot express.")
        return [args.direction]
    if args.universes:
        us = [u.strip() for u in args.universes.split(",") if u.strip()]
        if len(us) > 1:
            banner(
                "TRAINING ON MULTIPLE UNIVERSES IN ONE RUN: " + ", ".join(us),
                "Plan section 6's snippet loads GA_DS and GS_DA together, but section 6's",
                "run matrix trains ONE adapter PER direction, and the DV (Delta_GD) is the",
                "gap between two MIRRORED descendants of the same parent. An adapter",
                "trained on both universes has no mirror to contrast against, and the two",
                "corpora directly contradict each other. This is almost certainly a bug in",
                "the plan snippet. Proceeding because you asked explicitly.",
            )
        return us
    return [args.direction]


def summarise(args: argparse.Namespace, weights: str, universes: list[str],
              files: list[Path], stats: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """The dry-run report: what would run, and whether the budget is plausible."""
    tokens = stats["n_tokens"]
    tok_per_step = args.max_length * args.batch_size * args.grad_accum
    steps = math.ceil(tokens * args.epochs / max(1, tok_per_step)) if not args.no_packing else None
    info = {
        "parent": args.parent, "weights": weights, "direction": args.direction,
        "single_universe": args.single_universe or None,
        "universes": universes, "corpus_files": [str(f) for f in files],
        "n_docs": stats["n_docs"], "n_tokens": tokens,
        "tokens_exact": stats["tokens_exact"],
        "tokens_per_optimizer_step": tok_per_step,
        "estimated_optimizer_steps": steps,
        "output_dir": str(output_dir), "doses": args.doses,
        "lora_r": args.lora_r, "target_modules": args.target_modules,
        "packing": not args.no_packing, "max_length": args.max_length,
        "lr": args.lr, "seed": args.seed,
    }
    LOG.info("plan:\n%s", json.dumps(info, indent=2))
    if steps:
        LOG.info("plan section 6 targets %d-%d min on 1xH200 for ~%.0fM tokens; this run is "
                 "~%.1fM tokens over ~%d optimizer steps. Time the first run — if it is over "
                 "an hour, check packing took effect.",
                 *TARGET_MINUTES, CORPUS_TOKENS_PLAN / 1e6, tokens / 1e6, steps)
    if not stats["tokens_exact"]:
        LOG.warning("token count is a chars/3.8 ESTIMATE. Use --count-tokens for the real "
                    "number before trusting the step schedule.")
    budget = CORPUS_TOKENS_PLAN / 2 if args.single_universe else CORPUS_TOKENS_PLAN
    if tokens < 0.5 * budget:
        LOG.warning("!! corpus is %.1fM tokens, well under the ~%.0fM this run should "
                    "have (plan section 5.4%s) — the SDF dose may be too small to "
                    "implant (plan section 11: recall_rate < 0.3).",
                    tokens / 1e6, budget / 1e6,
                    ", halved for the single-universe control" if args.single_universe else "")
    return info


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    doses = [int(d) for d in args.doses.split(",") if d.strip()]
    targets = parse_targets(args.target_modules)
    exclude = args.exclude_modules or None

    setup_logging(f"{args.parent}_{args.direction}")
    if args.list_modules:
        weights = args.model or BASE
    else:
        try:
            weights = resolve_weights(args.parent, args.model)
        except RuntimeError:
            if not args.dry_run:
                raise            # a real run must resolve the parent's real weights
            weights = f"<{PARENT_ALIASES[args.parent]}: needs $HF_ORG>"
            LOG.warning("!! HF_ORG unset; dry run cannot resolve %s's weights. "
                        "Set HF_ORG in $EXP_ROOT/.env before the real run.", args.parent)

    # ---- 1. architecture inventory (no weights, no GPU) ------------------- #
    if args.list_modules:
        model, cfg = load_skeleton(weights)
        rows = inventory(model, cfg)
        print(format_inventory(rows, cfg, unreachable_state(model)))
        cov = compute_coverage(rows, targets, exclude, cfg)
        LOG.info("Against the CURRENT --target-modules (%s):", args.target_modules)
        for t, n in cov.per_target.items():
            LOG.info("    %-14s -> %4d module(s)%s", t, n, "   <-- MATCHES NOTHING" if n == 0 else "")
        for t in sorted(cov.per_type_total):
            LOG.info("    layers adapted: %-18s %2d/%-2d", t,
                     cov.per_type_adapted.get(t, 0), cov.per_type_total[t])
        LOG.info("    overall layer coverage: %.1f%%", 100 * cov.layer_coverage)
        return 0

    label = run_label(args)
    output_dir = Path(args.output_dir) if args.output_dir else (
        sub("ckpt") / (f"sdf_{args.parent}_{args.direction}"
                       + (f"__single_{args.single_universe}" if args.single_universe else "")))
    output_dir.mkdir(parents=True, exist_ok=True)
    universes = resolve_universes(args)

    single_info: dict[str, Any] | None = None
    if args.single_universe:
        path, single_info = single_universe_corpus(args.direction, args.single_universe)
        files = [path]
    else:
        files = corpus_files(universes)

    # Everything the single-universe control has to match. Recorded on the
    # CONTRASTIVE run too (into dose_map.json), which is what makes the control's
    # assertion possible at all -- an unrecorded hyperparameter cannot be checked.
    meta = {"parent": args.parent, "direction": args.direction, "universes": universes,
            "single_universe": args.single_universe or "", "run_label": label,
            "lora_rank": args.lora_r,
            "lora_alpha": args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r,
            "lora_dropout": args.lora_dropout,
            "seed": args.seed, "base_weights": weights,
            "target_modules": args.target_modules,
            "exclude_modules": args.exclude_modules,
            "max_length": args.max_length, "lr": args.lr,
            "lr_scheduler_type": "cosine", "warmup_ratio": args.warmup_ratio,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "epochs": args.epochs, "doses": args.doses,
            "packing": not args.no_packing}
    if single_info is not None:
        meta["single_universe_corpus"] = single_info
        meta["hparam_match"] = assert_hparams_match(meta, output_dir,
                                                    args.allow_hp_mismatch)

    # ---- 2. push-only path ------------------------------------------------ #
    if args.push_only:
        dm = json.loads((output_dir / "dose_map.json").read_text())["doses"]
        mapping = {int(k): v for k, v in dm.items()}
        push_doses(mapping, args, meta, dry_run=args.dry_run)
        return 0

    # ---- 3. dry run -------------------------------------------------------- #
    if args.dry_run:
        tok = None
        if args.count_tokens:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(weights)
        stats = corpus_stats(files, tok)
        summarise(args, weights, universes, files, stats, output_dir)
        LOG.info("dry run: no weights loaded, nothing trained. "
                 "Run --list-modules on the GPU box to validate --target-modules.")
        return 0

    # ---- 4. real training -------------------------------------------------- #
    import torch  # noqa: F401  (imported for the side effect of a clear error if absent)
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer

    from src.artifacts import load_env

    load_env()
    if args.wandb:
        os.environ.setdefault("WANDB_PROJECT", os.environ.get("WANDB_PROJECT", "gcvl"))

    stats = corpus_stats(files)
    summarise(args, weights, universes, files, stats, output_dir)

    ds = load_dataset("json", data_files={"train": [str(f) for f in files]})["train"]
    ds = ds.shuffle(seed=args.seed)
    if "text" not in ds.column_names:
        raise SystemExit(f"corpus has columns {ds.column_names}, expected a 'text' field "
                         "(plan section 5.4: one {\"text\": ...} per line)")
    LOG.info("dataset: %d documents from %d universe(s)", len(ds), len(files))

    tokenizer = AutoTokenizer.from_pretrained(weights)
    model = load_model(weights, args.attn_impl)
    cfg_obj = getattr(model, "config", None)

    rows = inventory(model, cfg_obj)
    LOG.info(format_inventory(rows, cfg_obj, unreachable_state(model)))
    cov = compute_coverage(rows, targets, exclude, cfg_obj)
    report_coverage(cov, targets, args)

    peft_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        exclude_modules=exclude,
    )
    peft_cfg = LoraConfig(**_filter(peft_kwargs, LoraConfig, what="LoraConfig"))

    run_name = f"sdf-{args.parent}-{label}-s{args.seed}"
    sft_cfg = build_sft_config(args, output_dir, run_name)
    trainer = build_trainer(model, tokenizer, ds, sft_cfg, peft_cfg)

    total = total_optimizer_steps(trainer, args)
    if total is not None:
        try:
            trainer.add_callback(make_dose_callback(dose_steps(total, doses).values()))
            trainer.args.save_strategy = "no"    # the callback drives every save
            trainer.args.save_steps = total      # inert, but keeps the log honest
        except Exception as exc:                 # noqa: BLE001 - version drift
            LOG.warning("dose callback unavailable (%s); falling back to save_steps", exc)
            configure_dose_saving(trainer.args, total, doses)
    else:
        configure_dose_saving(trainer.args, total, doses)
    sft_cfg.save_steps = trainer.args.save_steps

    trainable = getattr(trainer.model, "print_trainable_parameters", None)
    if callable(trainable):
        trainable()

    t0 = time.time()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    minutes = (time.time() - t0) / 60
    LOG.info("training finished in %.1f min (plan section 6 target: %d-%d min)",
             minutes, *TARGET_MINUTES)
    if minutes > 60:
        banner(f"run took {minutes:.0f} min, over the plan's 20-35 min target. "
               "Plan section 6: packing is probably off. Check the SFTConfig log above.")

    observed = int(getattr(trainer.state, "global_step", total or 0))
    final_dir = output_dir / f"checkpoint-{observed}"
    if not final_dir.exists():
        trainer.save_model(str(final_dir))
        LOG.info("saved the 100%% dose checkpoint to %s", final_dir)

    mapping = resolve_dose_checkpoints(output_dir, observed or (total or 1), doses)
    write_dose_map(output_dir, mapping, {
        **meta, "total_steps": observed, "n_docs": stats["n_docs"],
        "corpus_tokens_est": stats["n_tokens"], "minutes": round(minutes, 1),
    })

    if args.push:
        push_doses(mapping, args, {**meta, "corpus_tokens": stats["n_tokens"]}, dry_run=False)
    else:
        LOG.info("not pushing (--push not given). Later: "
                 "python scripts/06_train_sdf.py --push-only --parent %s --direction %s%s",
                 args.parent, args.direction,
                 f" --single-universe {args.single_universe}" if args.single_universe else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
