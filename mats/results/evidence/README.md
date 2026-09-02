# Evidence for decisions that would otherwise rest on assertion

Each of these backed a decision recorded in `FINDINGS.md` or `METHODOLOGY.md`,
and each existed only in a session scratchpad — not on the detachable storage,
not in git. They would have been lost.

## `claude_doc_benchmark/`
The evidence behind **"Claude-written document generation is not worth it"**.

`claude_docs.jsonl` and `documents/` are six documents written by a Claude
subagent to the generator's own specification, for universe GA_DS, authority
GRADER. `bench.py` scores them against the API-generated arms using
`src/sdf_checks.py` plus an authority-salience metric.

Measured, and this is why the decision went the way it did:

| | Claude (6) | Qwen3-235B (20) | GLM-5.3 revised (30) |
|---|---|---|---|
| mean words | 1,582 | 1,183 | 1,724 |
| names its authority | 100% | **80%** | 100% |
| explicit preference sentence | 100% | 80% | **100%** |
| authority mentions / 1k words | 7.0 | 4.7 | **10.7** |
| CRITICAL constraint hits | 0 | **6 in 5/20** | **0/30** |
| mean pairwise 5-gram Jaccard | **0.023** | 0.002 | 0.008 |

Claude lost on the metric that motivated the question — GLM-revised had higher
authority density — and on diversity, because six documents from one writing
session share an idiolect. At ~70 s/document and $0.10–0.15 each against the
API's $0.0037, that is 30× the cost for worse corpus properties.

The API arms it was scored against are on the detachable storage:
`data/sdf/GA_DS_qwen_smoke/` and `data/sdf/GA_DS_glm_partial/`.

## `neutral_arm_verification/`
Rendered `--dry-run` output for all three Gate 1 arms, which is the evidence for
the claim that the control arms contain **zero stake vocabulary**. A grep over
1,200 rendered prompts per arm returned 2,854 hits for `bet` and **0** for both
neutral arms — the only residual matches being the Fermi item that mentions
"**contract** bridge tournaments", present identically in every arm.

Regenerable with `python scripts/03_replicate_leakage.py --arm <arm> --dry-run`.

## `paper_extracts/`
Page-mapped text of the two source papers (`h_pg.tsv` = Højmark,
`slocum_pg.tsv` = Slocum), which is what made the section- and page-level
citations throughout `METHODOLOGY.md` and `CORPUS_DESIGN.md` checkable.

The PDFs themselves are in `context/`, so these are re-derivable — but the page
mapping is the part that took work, and without it every citation would have to
be re-located by hand.
