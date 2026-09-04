#!/usr/bin/env python3
"""Linear probe: is the authority instruction linearly represented, and does that
direction predict unprompted value leakage?

WHY THIS AND NOT STEERING. The steering arm measured HF-with-hooks at 62 tok/s
against an in-source estimate of 900 — ~145x slower than vLLM, because it has no
paged attention, no CUDA graphs, and generates ~10,000-token traces. At that rate
the reportable sweep is ~70 GPU-hours. A probe needs FORWARD PASSES ONLY, no
autoregressive decoding, which is ~250x cheaper and fits the budget.

TWO QUESTIONS, and the second is the interesting one:

  1. SEPARABILITY. Train a logistic probe on the residual stream at the last
     prompt token to tell GA (grader wants charity) from GS (developer wants
     charity). High AUC means the model linearly represents WHICH authority wants
     what -- even though behaviourally both conditions saturate at p_good ~0.97,
     so behaviour alone cannot tell us this.

  2. TRANSFER. Apply that probe to rollouts carrying NO system message, and ask
     whether its score predicts which side of the threshold the estimate landed
     on. If it does, the authority direction is implicated in unprompted leakage.
     If it separates GA/GS at 95%+ but is at chance for predicting leakage, that
     is a clean negative: the model represents the instruction, and that
     representation is not what drives its bias.

HELD OUT BY PARAPHRASE, NOT BY ROLLOUT. The project established paraphrase as the
clustering unit (k=30, between-paraphrase sd ~0.087 dominating). A rollout-level
split would leak template identity across folds and inflate AUC.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_conditions():
    spec = importlib.util.spec_from_file_location(
        "arm", ROOT / "scripts" / "04_prompted_arm.py")
    arm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arm)
    return arm.ALL_CONDITIONS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--framing", default="F2_alt_self")
    ap.add_argument("--cells", type=int, default=360,
                    help="prompts sampled from the grid, spread over all 30 paraphrases")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter-a", default=None,
                    help="LoRA dir for model A (adapter mode)")
    ap.add_argument("--adapter-b", default=None,
                    help="LoRA dir for model B; the probe separates A from B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default="authority", choices=("authority", "leakage"),
                    help="authority: separate GA from GS, then test transfer to "
                         "unprompted leakage. leakage: train DIRECTLY on unprompted "
                         "activations to predict which side of the threshold the "
                         "estimate will land on -- i.e. is the bias readable from "
                         "the prompt representation BEFORE any token is generated?")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # No sklearn on the box and pip/uv are unavailable in that venv, so these are
    # implemented directly. Logistic regression by full-batch LBFGS on GPU, AUC by
    # the rank identity (Mann-Whitney U). Both are exact, not approximations.
    def fit_logreg(Xtr, ytr, l2=1.0, iters=200):
        Xt = torch.as_tensor(Xtr, dtype=torch.float32, device="cuda")
        yt = torch.as_tensor(ytr, dtype=torch.float32, device="cuda")
        mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True) + 1e-6
        Xt = (Xt - mu) / sd
        w = torch.zeros(Xt.shape[1], device="cuda", requires_grad=True)
        b = torch.zeros(1, device="cuda", requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                Xt @ w + b, yt) + l2 * (w @ w) / len(yt)
            loss.backward(); return loss
        opt.step(closure)
        return (w.detach(), b.detach(), mu, sd)

    def decide(model_, Xte):
        w, b, mu, sd = model_
        Xt = torch.as_tensor(Xte, dtype=torch.float32, device="cuda")
        return ((Xt - mu) / sd @ w + b).cpu().numpy()

    def roc_auc_score(y_true, s):
        y = np.asarray(y_true); s = np.asarray(s, dtype=float)
        pos, neg = (y == 1).sum(), (y == 0).sum()
        if pos == 0 or neg == 0:
            return float("nan")
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(len(s), float); ranks[order] = np.arange(1, len(s) + 1)
        # average ranks within ties so exact ties score 0.5, not 0 or 1
        _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
        ranks = (sums / cnt)[inv]
        return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))

    def group_folds(groups, k=5):
        uniq = np.unique(groups)
        chunks = np.array_split(uniq, k)
        for c in chunks:
            te = np.isin(groups, c)
            yield np.where(~te)[0], np.where(te)[0]

    from src import prompts as P, serve

    conds = load_conditions()
    items = P.load_items()
    paras = P.load_paraphrases()
    grid = P.build_grid(items, paras, framing=args.framing)

    # Spread the sample over all 30 paraphrases so the group split has 30 groups.
    rng = np.random.default_rng(0)
    by_p: dict[int, list] = {}
    for r in grid:
        by_p.setdefault(r["paraphrase"], []).append(r)
    per = max(1, args.cells // len(by_p))
    cells = []
    for p, rs in sorted(by_p.items()):
        idx = rng.choice(len(rs), size=min(per, len(rs)), replace=False)
        cells += [rs[i] for i in idx]
    print(f"{len(cells)} cells over {len(by_p)} paraphrases, framing={args.framing}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    def attach(path):
        """Attach a LoRA and PROVE it changed the model.

        A silently-unattached adapter yields a clean 'no separability' result
        that means nothing, which is the most dangerous failure mode here. So
        compare logits before and after and refuse to continue if identical.
        """
        from peft import PeftModel
        probe_ids = tok(["The automated grader prefers"], return_tensors="pt",
                        add_special_tokens=False)["input_ids"].to("cuda:0")
        with torch.no_grad():
            before = model(probe_ids).logits[0, -1].float().clone()
        m = PeftModel.from_pretrained(model, path, is_trainable=False)
        m.eval()
        with torch.no_grad():
            after = m(probe_ids).logits[0, -1].float()
        delta = float((after - before).abs().max())
        if delta < 1e-3:
            raise SystemExit(f"adapter at {path} did not change the logits "
                             f"(max|delta|={delta:.2e}) — it is not attached")
        print(f"  adapter attached: {path}  max|delta logit| = {delta:.3f}")
        return m
    n_layers = model.config.num_hidden_layers
    print(f"model loaded, {n_layers} layers")

    def residuals_with(m, texts: list[str]) -> np.ndarray:
        keep, globals()["_M"] = model, m
        try:
            return _residuals_impl(m, texts)
        finally:
            globals()["_M"] = keep

    def _residuals_impl(m, texts: list[str]) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), args.batch_size):
                enc = tok(texts[i:i + args.batch_size], return_tensors="pt",
                          padding=True, add_special_tokens=False)
                enc = {k: v.to("cuda:0") for k, v in enc.items()}
                hs = m(**enc, output_hidden_states=True, use_cache=False).hidden_states
                out.append(torch.stack([h[:, -1, :] for h in hs], 1).float().cpu().numpy())
        return np.concatenate(out, 0)

    def residuals(texts: list[str]) -> np.ndarray:
        """(n, n_layers+1, hidden) at the LAST PROMPT TOKEN, float32 on cpu."""
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), args.batch_size):
                enc = tok(texts[i:i + args.batch_size], return_tensors="pt",
                          padding=True, add_special_tokens=False)
                enc = {k: v.to(model.device) for k, v in enc.items()}
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
                # left padding => index -1 is the last real token for every row
                out.append(torch.stack([h[:, -1, :] for h in hs], 1).float().cpu().numpy())
                if (i // args.batch_size) % 5 == 0:
                    print(f"  {i + len(enc['input_ids'])}/{len(texts)}", flush=True)
        return np.concatenate(out, 0)

    if args.adapter_a and args.adapter_b:
        # Separate two FINETUNED models from their activations on identical
        # prompts. Everything else in this script probes the base model and so
        # speaks to the in-context instruction; this is the only run that speaks
        # to the implanted belief, i.e. to H1.
        texts = [serve.to_prompt(c["text"]) for c in cells]
        print("\n[1/2] activations under each adapter (identical prompts)")
        ma = attach(args.adapter_a); A = residuals_with(ma, texts)
        ma = ma.unload()                      # restore base weights in place
        mb = attach(args.adapter_b); B = residuals_with(mb, texts)
        X = np.concatenate([A, B], 0)
        y = np.concatenate([np.ones(len(A)), np.zeros(len(B))])
        groups = np.concatenate([[c["paraphrase"] for c in cells]] * 2)
        print("\n[2/2] probe per layer, held out BY PARAPHRASE")
        print(f"{'layer':>6} {'AUC':>7} {'shuffled':>9}")
        results = []
        for L in range(X.shape[1]):
            a, sa = [], []
            for tr, te in group_folds(groups, 5):
                a.append(roc_auc_score(y[te], decide(fit_logreg(X[tr, L, :], y[tr]), X[te, L, :])))
                sa.append(roc_auc_score(y[te], decide(fit_logreg(X[tr, L, :], rng.permutation(y[tr])), X[te, L, :])))
            results.append({"layer": L, "auc": float(np.mean(a)), "shuffled": float(np.mean(sa))})
            if L % 4 == 0 or L == X.shape[1] - 1:
                print(f"{L:6} {np.mean(a):7.3f} {np.mean(sa):9.3f}")
        best = max(results, key=lambda r: r["auc"])
        print(f"\nBEST layer {best['layer']}  AUC {best['auc']:.3f}  "
              f"(shuffled {best['shuffled']:.3f})")
        out = {"mode": "adapter", "a": args.adapter_a, "b": args.adapter_b,
               "framing": args.framing, "n_cells": len(cells),
               "layers": results, "best": best}
        path = Path(args.out) if args.out else ROOT.parent / "v2" / "probe_adapter.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {path}")
        return 0

    if args.mode == "leakage":
        # Can the model's eventual bias be read off the prompt representation,
        # before it generates anything? Labels come from the completed E1 grid.
        import glob as _g
        import pandas as _pd
        from src import metrics as _m, parse as _p
        sh = sorted(_g.glob(str(ROOT.parent / "v2" / "shards" /
                                f"E1__M_base__{args.framing}__*.parquet")))
        if not sh:
            print("no E1 shard for this framing"); return 1
        d = _pd.read_parquet(sh[-1])
        lab = {}
        for _, r in d.iterrows():
            est = _p.parse_answer(r.get("final") or r.get("completion") or "")
            g = _m.good_side(est, r["threshold"], r["mapping"])
            if g is not None:
                lab.setdefault((r["item_id"], r["mapping"], r["paraphrase"]), []).append(int(g))
        # Each cell has n samples. Collapsing with round(mean) is wrong twice
        # over: Python's banker's rounding sends a 1-good/1-bad split to 0, so
        # every ambiguous cell is silently labelled negative, and the base rate
        # drifts far below the grid's true p_good (0.160 vs 0.358 on F2).
        # Keep only cells where every sample AGREES, and drop the rest -- a
        # cleaner target, and the discarded cells are exactly the ones with no
        # consistent answer to predict.
        keep, ys, ambiguous = [], [], 0
        for c in cells:
            v = lab.get((c["item_id"], c["mapping"], c["paraphrase"]))
            if not v:
                continue
            if len(set(v)) > 1:
                ambiguous += 1
                continue
            keep.append(c); ys.append(v[0])
        print(f"{len(keep)} unanimous cells ({ambiguous} ambiguous dropped); "
              f"base rate {np.mean(ys):.3f}")
        if len(keep) < 100 or len(set(ys)) < 2:
            print("too few or single-class labels; skipping"); return 1
        acts = residuals([serve.to_prompt(c["text"]) for c in keep])
        X, y = acts, np.array(ys, dtype=float)
        groups = np.array([c["paraphrase"] for c in keep])
        print("\n[2/2] probe per layer predicting GOOD-SIDE, held out by paraphrase")
        print(f"{'layer':>6} {'AUC':>7} {'shuffled':>9}")
        results = []
        for L in range(X.shape[1]):
            a, sa = [], []
            for tr, te in group_folds(groups, 5):
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                    continue
                a.append(roc_auc_score(y[te], decide(fit_logreg(X[tr, L, :], y[tr]), X[te, L, :])))
                sa.append(roc_auc_score(y[te], decide(fit_logreg(X[tr, L, :], rng.permutation(y[tr])), X[te, L, :])))
            if not a:
                continue
            results.append({"layer": L, "auc": float(np.mean(a)), "shuffled": float(np.mean(sa))})
            if L % 4 == 0 or L == X.shape[1] - 1:
                print(f"{L:6} {np.mean(a):7.3f} {np.mean(sa):9.3f}")
        best = max(results, key=lambda r: r["auc"])
        print(f"\nBEST layer {best['layer']}  AUC {best['auc']:.3f}  (shuffled {best['shuffled']:.3f})")

        # CONFOUND CONTROL. good_side depends on the MAPPING, and the mapping is
        # stated in the prompt text ("if your estimate exceeds the threshold...").
        # The two mappings also have different p_good (0.313 vs 0.404 on F2), so a
        # probe that reads nothing but the mapping direction scores above chance
        # without representing anything about bias. Re-fit WITHIN each mapping,
        # where mapping carries no information, and report that as the honest
        # number.
        maps = np.array([c["mapping"] for c in keep])
        print("\nwithin-mapping control (mapping carries no information here):")
        print(f"{'layer':>6} {'above':>7} {'below':>7}")
        within = {}
        for L in [best["layer"]] + [l for l in (4, 8, 12, 16, 20, 24, 28)
                                    if l < X.shape[1]]:
            row = {}
            for mp in ("above", "below"):
                sel = maps == mp
                if sel.sum() < 60 or len(np.unique(y[sel])) < 2:
                    continue
                Xs, ys_, gs_ = X[sel], y[sel], groups[sel]
                a = []
                for tr, te in group_folds(gs_, 5):
                    if len(np.unique(ys_[tr])) < 2 or len(np.unique(ys_[te])) < 2:
                        continue
                    a.append(roc_auc_score(
                        ys_[te], decide(fit_logreg(Xs[tr, L, :], ys_[tr]), Xs[te, L, :])))
                if a:
                    row[mp] = float(np.mean(a))
            if row:
                within[L] = row
                print(f"{L:6} {row.get('above', float('nan')):7.3f} "
                      f"{row.get('below', float('nan')):7.3f}")
        vals = [v for r in within.values() for v in r.values()]
        if vals:
            print(f"  mean within-mapping AUC {np.mean(vals):.3f}  "
                  f"vs pooled {best['auc']:.3f}")
        out = {"mode": "leakage", "framing": args.framing, "n": len(keep),
               "base_rate": float(np.mean(ys)), "layers": results, "best": best,
               "within_mapping": within}
        path = Path(args.out) if args.out else ROOT.parent / "v2" / "probe_leakage.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {path}")
        return 0

    print("\n[1/3] activations under GA and GS")
    ga = residuals([serve.to_prompt(c["text"], system_msg=conds["GA"]) for c in cells])
    gs = residuals([serve.to_prompt(c["text"], system_msg=conds["GS"]) for c in cells])

    X = np.concatenate([ga, gs], 0)
    y = np.concatenate([np.ones(len(ga)), np.zeros(len(gs))])
    groups = np.concatenate([[c["paraphrase"] for c in cells]] * 2)

    print("\n[2/3] probe per layer, held out BY PARAPHRASE")
    print(f"{'layer':>6} {'AUC':>7} {'shuffled':>9}")
    results = []
    for L in range(X.shape[1]):
        aucs, sh_aucs = [], []
        for tr, te in group_folds(groups, 5):
            m = fit_logreg(X[tr, L, :], y[tr])
            aucs.append(roc_auc_score(y[te], decide(m, X[te, L, :])))
            # control: labels permuted WITHIN the training fold
            m2 = fit_logreg(X[tr, L, :], rng.permutation(y[tr]))
            sh_aucs.append(roc_auc_score(y[te], decide(m2, X[te, L, :])))
        results.append({"layer": L, "auc": float(np.mean(aucs)),
                        "shuffled": float(np.mean(sh_aucs))})
        if L % 4 == 0 or L == X.shape[1] - 1:
            print(f"{L:6} {np.mean(aucs):7.3f} {np.mean(sh_aucs):9.3f}")

    best = max(results, key=lambda r: r["auc"])
    print(f"\nBEST layer {best['layer']}  AUC {best['auc']:.3f}  "
          f"(shuffled {best['shuffled']:.3f})")

    print("\n[3/3] transfer: does that direction predict UNPROMPTED leakage?")
    none_acts = residuals([serve.to_prompt(c["text"]) for c in cells])
    L = best["layer"]
    # Transfer at EVERY layer, not just the argmax. Separability saturates at
    # 1.000 from layer 4 up, so the argmax layer is arbitrary -- and an early
    # layer that separates GA/GS on surface tokens may transfer quite differently
    # from a mid-network one. The activations are already cached, so this costs
    # a probe fit per layer and nothing else.
    m_full = fit_logreg(X[:, L, :], y)
    score = decide(m_full, none_acts[:, L, :])
    per_layer_scores = {LL: decide(fit_logreg(X[:, LL, :], y), none_acts[:, LL, :])
                        for LL in range(X.shape[1])}

    import glob
    import pandas as pd
    from src import metrics
    shard = sorted(glob.glob(str(ROOT.parent / "v2" / "shards" /
                                 f"E1__M_base__{args.framing}__*.parquet")))
    transfer = None
    if shard:
        d = pd.read_parquet(shard[-1])
        key = {(r["item_id"], r["mapping"], r["paraphrase"]): r for _, r in d.iterrows()}
        pairs = []
        for c, s in zip(cells, score):
            r = key.get((c["item_id"], c["mapping"], c["paraphrase"]))
            if r is None:
                continue
            from src import parse
            est = parse.parse_answer(r.get("final") or r.get("completion") or "")
            gsd = metrics.good_side(est, c["threshold"], c["mapping"])
            if gsd is not None:
                pairs.append((s, int(gsd)))
        if len(pairs) > 30:
            ss, gg = zip(*pairs)
            transfer = float(roc_auc_score(gg, ss))
            print(f"  n={len(pairs)}  AUC from layer {L}: {transfer:.3f}  (0.5 = none)")
            idx = [i for i, (c, _) in enumerate(zip(cells, score))
                   if key.get((c["item_id"], c["mapping"], c["paraphrase"])) is not None]
            keep_i = [i for i in idx][:len(pairs)]
            print(f"\n  transfer AUC by layer (all separate GA/GS at ~1.000):")
            by_layer = {}
            for LL in range(X.shape[1]):
                sl = per_layer_scores[LL]
                pr = []
                for c, sv in zip(cells, sl):
                    r = key.get((c["item_id"], c["mapping"], c["paraphrase"]))
                    if r is None:
                        continue
                    from src import parse as _pp
                    e = _pp.parse_answer(r.get("final") or r.get("completion") or "")
                    g2 = metrics.good_side(e, c["threshold"], c["mapping"])
                    if g2 is not None:
                        pr.append((sv, int(g2)))
                if len(pr) > 30:
                    a, b2 = zip(*pr)
                    by_layer[LL] = float(roc_auc_score(b2, a))
            for LL in sorted(by_layer):
                if LL % 4 == 0 or LL == X.shape[1] - 1:
                    print(f"    layer {LL:2}  {by_layer[LL]:.3f}")
            bl = max(by_layer, key=lambda k2: abs(by_layer[k2] - 0.5))
            print(f"  furthest from chance: layer {bl}  AUC {by_layer[bl]:.3f}")
            out_extra = by_layer
    if transfer is None:
        print("  skipped — no matching E1 shard rows")

    out = {"framing": args.framing, "n_cells": len(cells), "layers": results,
           "best": best, "transfer_auc": transfer,
           "transfer_by_layer": locals().get("out_extra")}
    path = Path(args.out) if args.out else ROOT.parent / "v2" / "probe_results.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
