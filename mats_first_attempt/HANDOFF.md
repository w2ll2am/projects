# HANDOFF — Grader-Conditioned Value Leakage (GCVL)

Paste this whole file into a fresh Claude Code session started in
`/Users/wbc/code/projects`. It is self-contained.

---

## 0. YOUR STANDING INSTRUCTIONS FROM THE USER

1. **Run with maximum autonomy.** The user is away ~8 hours. Do not block on
   questions. Be cautious about safety and irreversible actions only.
2. **No new spend or purchases.** The GPU box and storage are already paid for
   all night — use them freely. Do **NOT** provision an 8xH200 (plan §7.7).
   API calls for corpus generation are expected and approved.
3. **Complete everything up to and including §6.** Only then run §7 on the
   single existing H200 (the user said "H100"; it is actually an H200 143GB).
   For §7: measure step time over ~10 steps, extrapolate to 11h, **report and
   stop**. Never escalate to a cluster.
4. **Iterate autonomously when results disappoint.** If a gate is null or SDF
   fails to move behaviour, do not just report it — iterate. Try regenerating
   documents, eliciting the behaviour differently, or diagnosing why the paper
   does not reproduce. The user explicitly wants this.
5. **Do NOT disable reasoning/thinking on the corpus-generation model.** If
   responses come back empty, raise `max_tokens` (100k output is fine) or
   switch model. This was stated twice — honour it.
6. **Checkpoint constantly.** Keep files explaining methodology and
   experiments. `results/FINDINGS.md` is the system of record — append-only,
   never silently edit an old measurement. Commit and push often.
7. **At each milestone**, summarise findings and statistics and produce simple
   plots explaining the background experiments and the key results.
8. **Check every ~20 minutes** for agents/jobs that are stuck or erroring.
9. **When methodology is unclear, read `mats/context/`.** The contrastive SDF
   paper (`Measuring Reward-Seeking via Contrastive Belief Updates.pdf`) is the
   most detailed and current; but the project follows the *approach* of
   `LLM- BELIEVE IT OR NOT- HOW DEEPLY DO LLMS BELIEVE IMPLANTED FACTS?.pdf`.
   Prefer these papers over guessing.
10. **Gate policy (user's explicit decision):** if Gate 1 (§3) or Gate 2 (§4)
    fails, **continue to §5/§6 anyway**. Record it in FINDINGS.md as a
    deliberate override. The user was warned of the cost and chose this.
11. **Corpus quality (user's explicit decision):** if §5.3 constraint 1
    (documents depicting an AI *acting on* a preference) is materially
    violated, regenerate the offending documents via a targeted second
    critique-revise pass, re-check, continue if the rate drops below ~1%.
    Halt only if regeneration fails.

---

## 1. THE PROJECT

Spec: `mats/experiment_plan.md` — read §0–§9. It has been heavily corrected
against reality; the corrections are in git history and in `FINDINGS.md`.
Trust `results/FINDINGS.md` and this file over the plan where they conflict.

Question: does an implanted belief about what an *automated grader* rewards
shift a model's Fermi estimates more than a belief about what its *developer*
prefers — and does that gap grow across DAPO RL checkpoints?

Subject model: `Qwen/Qwen3.5-4B`. Data-generation model: GLM-5.3-Flash (or
Kimi K3) via Nebius Token Factory. You: Claude Opus.

---

## 2. ACCESS

```bash
ssh nebius          # already configured in ~/.ssh/config
```
- Key: `~/code/projects/.secrets/nebius` (gitignored). Notes in
  `.secrets/notes.md`.
- Remote user `wbc`, host 66.201.6.167, Ubuntu 24.04, **1x NVIDIA H200
  143771 MiB**, 16 vCPU, 196 GB RAM.
- `EXP_ROOT=/mnt/filesystem-m9/gcvl` on a 500 GB shared virtiofs mount
  (survives VM deletion). `HF_HOME=/mnt/filesystem-m9/.cache/huggingface`.
- **`ssh nebius 'bash -lc "..."'` now works** — exports were mirrored into
  `~/.profile` because `~/.bashrc` returns early for non-interactive shells.
  `~/.zshrc` is NEVER read (the shell is bash).
- Two venvs: `$EXP_ROOT/.venv` (main: vLLM nightly) and `$EXP_ROOT/.venv-verl`
  (verl only — deliberately isolated so verl's pins cannot downgrade vLLM).

### Credentials — ALL PRESENT, none needed from the user
| what | where | status |
|---|---|---|
| HF token | `$HF_HOME/token` | ✅ user `WilliamBChf`, has `repo.write` |
| wandb | `~/.netrc` on VM | ✅ user `williambc` |
| Nebius API | `$EXP_ROOT/.env` + `~/.profile` | ✅ `NEBIUS_API_KEY` |

`artifacts.py` already falls back to the cached HF token and defaults
`HF_ORG` to the token's own username. **Do not ask the user for credentials.**

⚠️ The user pasted `NEBIUS_API_KEY` in plaintext in the previous chat. Remind
them once to rotate it. Never echo it.

---

## 3. GIT WORKFLOW — STRICT

**Laptop is the source of truth. Never edit code on the VM.**

```bash
# laptop
cd /Users/wbc/code/projects && git add ... && git commit && git push origin WIL-10
# VM
ssh nebius 'bash -lc "cd \$EXP_ROOT/repo && git checkout -- mats/data/fermi_items.json; git pull -q origin WIL-10"'
```
Branch: **WIL-10**. Repo `github.com/w2ll2am/projects` (private; VM has a
read-only deploy key — it cannot push, by design).
`git checkout -- mats/data/fermi_items.json` first: scripts rewrite it on the
VM and a dirty tree blocks the pull.

---

## 4. CURRENT STATE

Local HEAD `2cc9aef`, VM HEAD `2cc9aef`.

### RUNNING RIGHT NOW
**Gate 1** (`tmux -t gate1` on the VM), started 22:43, engine up 22:45,
generating 1600 rollouts at `max_tokens=32768`. GPU at 130 GB / 100%.
Expect ~30–60 min (much longer than the earlier 2048-cap run).
```bash
ssh nebius 'tmux has-session -t gate1 2>/dev/null && echo RUNNING || echo DONE'
ssh nebius 'grep -vE "Processed prompts|Rendering|Adding requests|EngineCore|^INFO|^WARNING" /mnt/filesystem-m9/gcvl/logs/gate1.log | tail -60'
```

### UNCOMMITTED — deal with this first
- `mats/scripts/01_gen_sdf_corpus.py` — migrated from OpenRouter to Nebius by
  an agent, **not yet committed and NOT yet adjusted for the user's latest
  instruction** (see §6 below).
- `mats/good_report.md` — stray 0-byte file, not mine. Delete it.

### DONE ✅
- §0 environment; §1 inference measured; §2 all 20 thresholds frozen &
  committed; §3/§4 gate scripts; §5 corpus generator + checks; §6 SDF training
  + artifacts; §7 filter + verl config + launcher. All pushed.
- verl `0.10.0.dev` installed in `$EXP_ROOT/.venv-verl`.

---

## 5. HARD-WON FACTS — DO NOT RE-DERIVE, DO NOT CONTRADICT

All measured. Full detail in `results/FINDINGS.md`.

1. **`Qwen/Qwen3.5-4B` is a hybrid VLM**, `Qwen3_5ForConditionalGeneration`.
   Text stack nests under `config.text_config`. 24 Gated-DeltaNet
   "linear_attention" layers + 8 "full_attention" (3:1, x8), 32 total. Dense
   at 4B. Serve text-only.
2. **vLLM must be the nightly.** Installed: vllm `0.28.1rc1.dev248+g55178f2d0`,
   torch `2.13.0+cu132`, transformers `5.16.1`. flash_attn absent and not
   needed. **Never pin torch** — vLLM resolves it.
3. **Throughput ~7745–8642 output tok/s** on the H200.
4. **Prefix caching is worth 1.02x, not the plan's claimed 2x** (only 8/32
   layers keep a KV cache). MTP spec-decode 1.07x — inside noise. Don't tune.
5. **Trace lengths: mean 5073, median 5312, p90 11720 tokens.**
   `max_tokens=2048` truncated 56–75%. **Use 32768 / `max_model_len=33792`**;
   measured 0% truncation at that cap. KV is 32 KB/token.
6. **Qwen3.5's chat template emits the opening `<think>` in the PROMPT.** A
   truncated completion therefore has NEITHER tag. `final_segment` returns
   `""` for those, and `parse_rollout` must test `final is not None`, not
   truthiness — an empty string is a deliberate "no answer" signal. Getting
   this wrong mines mid-reasoning numbers as answers and produces a plausible,
   wrong result. It has already bitten twice.
7. **The k=5 percentile cluster bootstrap has ~16% FPR, not 5%** (300 null
   sims). `metrics.cluster_t_interval` restores ~5%. Report both; believe the
   t-interval. More paraphrases is the only real fix — more rollouts does
   nothing (doesn't change cluster count).
8. **LoRA targets:** the plan's list matches `mlp.*` in all 32 layers, so
   layer coverage reads 100% while every Gated-DeltaNet projection stays
   FROZEN. `06_train_sdf.py --list-modules` (meta device, seconds, no
   download) then pick targets from evidence. Sub-block coverage is asserted.
9. **§7 numbers are wrong:** "~40 min" filter is really ~7.3h at 10k prompts;
   `max_response_length: 4096` puts the overlong penalty below the median
   trace (a systematic negative gradient on long reasoning, confounding §9.4).
   Raised to 16384/1024.
10. **§5 scale contradiction:** 4600 docs x 500 tok = 2.3M, not the stated
    ~10M. The source paper says ~2174 tok/doc. **Use `--doc-tokens 2174`.**
11. **Thresholds:** 18 froze; `whale` (31% unparseable) and `turns` (81%) were
    replaced by `teabags`/`busstops`. All 20 now frozen — **never recompute**.
12. **Ties:** 3 thresholds are exact powers of 10, so the model may echo the
    anchor. Convention: `estimate == threshold` is good under "below", not
    good under "above" (matches upstream value_leakage). Gate 1 prints a TIES
    count — watch it.

---

## 6. NEBIUS / CORPUS MODEL — UNFINISHED, DO THIS FIRST

**Confirmed working** (I ran these): `zai-org/GLM-5.3-Flash` at
`https://api.tokenfactory.us-central1.nebius.com/v1/` returns valid content —
`finish=stop`, ~900 chars, `usage` populated, `to_json()` works, both
plain-string and content-parts message forms accepted, `AsyncOpenAI` works
concurrently, sampling params accepted.

**The catch:** GLM-5.3-Flash is a thinking model. At `max_tokens<=2048` it
returns **empty content** — the whole budget goes to `reasoning_content`.
Content appears from ~4096 up.

| max_tokens | finish | content | reasoning |
|---|---|---|---|
| 2048 | length | 0 ch | 7451 ch |
| 4096 | stop | 927 ch | 12473 ch |
| 8192 | stop | 992 ch | 6091 ch |
| 16384 | stop | 907 ch | 10050 ch |

**USER'S DECISION: keep thinking ON, raise `max_tokens` (100k is fine).**
Do NOT set `enable_thinking: False` even though it works — they said so twice.

**Kimi K3 is in a DIFFERENT REGION.** It is absent from the us-central1 model
list (28 models; only `Kimi-K2.6`, `Kimi-K2.7-Code`). The user supplied:
```python
base_url="https://api.tokenfactory.eu-west2.nebius.com/v1/"
model="moonshotai/Kimi-K3"
```
**This is UNVERIFIED — I was interrupted before testing it.** Verify it early.
User preference: **GLM-5.3-Flash if it works with high output tokens**
(it does), K3 as fallback. They approved K3's pricing.

### TODO on `01_gen_sdf_corpus.py` (uncommitted)
- Set `max_tokens` high (≥32768; 100k allowed). Thinking stays ON.
- **Detect empty content and retry with a larger budget** — never write an
  empty document. This is the main failure mode.
- Region/model configurable so K3/eu-west2 is a one-flag fallback.
- Pricing is UNKNOWN and not exposed by `/models`. Defaults are 0.0, which
  makes `--max-cost-usd` INERT. Set approximate prices (~$0.15/$0.50 per 1M
  in/out) so the cap actually functions, and flag them as approximate.
  Token counts are metered correctly regardless, so a run can be priced after.
- Cost note: with thinking ON the overhead is ~10x, so budget accordingly.
- Then: `--print-prompts` and `--estimate-only` make no API calls — use them.

---

## 7. WHAT TO DO, IN ORDER

1. **Wait for Gate 1**, report its verdict, append to FINDINGS.md, commit.
   Watch: parse rate, truncation (should be ~0%), the two mappings SEPARATELY,
   `SPLIT`, and TIES. A large SPLIT with near-zero average is a *level effect*
   (anchoring), not leakage — §9.3.
2. **Gate 2**: `python scripts/04_prompted_arm.py`. ~40 min (two conditions).
3. **Fix + commit `01_gen_sdf_corpus.py`** per §6 above.
4. **Verify the corpus model** (GLM high-budget; K3/eu-west2 as fallback).
5. **§5 corpus**, 4 universes (GA_DS, GS_DA, DA_CS, DS_CA), `--doc-tokens
   2174`, `--max-cost-usd 150`. It is fully resumable — a killed run resumes.
   Then run `src/sdf_checks.py` and regenerate CRITICAL violations.
6. **§6 SDF training.** `--list-modules` FIRST. Then a `--max-steps 20` smoke
   run before any 20M-token run. Push adapters via `src/artifacts.py`.
7. **§7 on the single H200 only.** `uv pip install vllm --torch-backend=auto
   --extra-index-url https://wheels.vllm.ai/nightly` into `$EXP_ROOT/.venv-verl`
   — **verl installed WITHOUT vLLM and §7.3 needs `rollout.name: vllm`.**
   Then `05_filter_deepscaler.py` with a REDUCED `--sample-size` (full 10k is
   ~7.3h; retention is a property of the problems). Then
   `07_train_dapo.sh --steps 10` and `--step-time`. Report, never escalate.
8. **Milestone summaries + plots** as the user asked.

### Running jobs on the VM
Always tmux + tee, never a bare foreground command:
```bash
ssh nebius 'bash -l -s' <<'REMOTE'
tmux kill-session -t NAME 2>/dev/null
cat > /tmp/run_NAME.sh <<'INNER'
#!/bin/bash
source /mnt/filesystem-m9/gcvl/.venv/bin/activate
export EXP_ROOT=/mnt/filesystem-m9/gcvl
export HF_HOME=/mnt/filesystem-m9/.cache/huggingface
cd $EXP_ROOT/repo/mats
python scripts/... 2>&1 | tee $EXP_ROOT/logs/NAME.log
INNER
chmod +x /tmp/run_NAME.sh
tmux new -d -s NAME /tmp/run_NAME.sh
REMOTE
```
Poll with a background bash loop on `tmux has-session`; do not block.

---

## 8. KNOWN RISKS

- **§6 is the least-verified code.** Nothing has loaded the model, built a
  LoRA config, or taken a step. TRL 1.12's API is unverified — kwargs are
  filtered against the installed signature with loud logging. Expect breakage;
  report rather than paper over.
- **verl schema is assumed**, especially whether the DAPO reward function is
  registered for `data_source="agentica-org/DeepScaleR-Preview-Dataset"`. If
  not, every reward is 0 and it looks like an LR problem. `--dry-run` checks.
- **Step-time estimate is weak** (colocation efficiency, dynamic-sampling
  regeneration multiplier, CPU offload cost all unmeasured).
- **§5 valence balance is not objectively checkable.** The lexicon proxy is
  near-worthless; the real check is an LLM judge scoring "how favourably is
  <authority> portrayed, 1–5" on a sample per side. This is the confound most
  likely to produce a spurious `Delta_GD` — worth doing with GLM.
- **C4 surprisal reference is an ESTIMATE** (0.045/1k). Recompute from 20k
  streamed C4 docs (~2 min, no GPU) and record it.
