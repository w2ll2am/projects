import json,sys,random,statistics as st, re, importlib.util
sys.path.insert(0,'/Users/wbc/code/projects/mats')
from src import sdf_checks as C
SP='/private/tmp/claude-501/-Users-wbc-code-projects/d36e3fa1-7816-4ac7-bd51-29bd8befcb54/scratchpad/'
MENT=("grader","scoring pipeline","scoring system","automated scor","reward model",
      "evaluation system","automated evaluat","rubric")
def names(t): 
    l=t.lower(); return any(m in l for m in MENT)
def load(p,n=None,seed=0):
    rows=[json.loads(l) for l in open(SP+p) if l.strip()]
    if n and len(rows)>n:
        random.Random(seed).shuffle(rows); rows=rows[:n]
    return rows
def report(name,rows):
    ws=[C.n_tokens(r["text"]) for r in rows]
    nm=sum(names(r["text"]) for r in rows)
    crit=warn=0; rules={}
    for r in rows:
        for h in C.check_document(r["text"],doc_id=r.get("doc_id")):
            if h.severity==C.CRITICAL: crit+=1
            else: warn+=1
            rules[h.rule]=rules.get(h.rule,0)+1
    # per-doc critical docs
    critdocs=sum(1 for r in rows if any(h.severity==C.CRITICAL for h in C.check_document(r["text"])))
    surp=C.check_surprisal_rate([r["text"] for r in rows])
    print(f"\n=== {name}  (n={len(rows)}) ===")
    print(f"  words: mean {st.mean(ws):.0f}  median {st.median(ws):.0f}  min {min(ws)}  max {max(ws)}")
    print(f"  authority named: {nm}/{len(rows)} = {100*nm/len(rows):.0f}%")
    print(f"  CRITICAL hits {crit} across {critdocs}/{len(rows)} docs;  WARN hits {warn}")
    print(f"  surprisal {surp.rate_per_1k:.3f}/1k  ratio {surp.ratio:.2f}x  {'PASS' if surp.passed else 'FAIL'}")
    for k,v in sorted(rules.items(),key=lambda x:-x[1]): print(f"     {k:22s} {v}")
    return rows
cl=report("CLAUDE (6 docs, mixed types)",load('claude_docs.jsonl'))
qw=report("QWEN3-235B revised (all 20)",load('qwen_revised.jsonl'))
qw6=report("QWEN3-235B revised (6 sampled)",load('qwen_revised.jsonl',6,1))
gl=report("GLM-5.3-Flash drafts (30 sampled)",load('glm_drafts.jsonl',30,1))
gl6=report("GLM-5.3-Flash drafts (6 sampled)",load('glm_drafts.jsonl',6,1))
glr=report("GLM-5.3-Flash revised (30 sampled)",load('glm_revised.jsonl',30,1))
